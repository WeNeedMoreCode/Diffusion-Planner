import os

import torch
import torch.nn as nn
from timm.models.layers import Mlp
from timm.layers import DropPath

from diffusion_planner.model.module.mixer import MixerBlock

# DP_TORCHAIR=1: encoder runs as the static-shape StaticEncoderBody, compiled
# into a torchair GE graph (same switch and lazy-build pattern as the DiT
# sampler body in decoder.py). Default 0 keeps the upstream forward.
_TORCHAIR = os.environ.get("DP_TORCHAIR", "0") == "1"


def _agent_pos(x):
    """Neighbor token positions, eager / outside the graph (see StaticEncoderBody)."""
    x = x[..., :8]
    pos = x[:, :, -1, :7].clone()  # x, y, cos, sin
    pos[..., -3:] = 0.0
    pos[..., -3] = 1.0  # neighbor: [1,0,0]
    return pos


def _static_pos(x):
    """Static-object token positions, eager / outside the graph."""
    pos = x[:, :, :7].clone()  # x, y, cos, sin
    pos[..., -3:] = 0.0
    pos[..., -2] = 1.0  # static: [0,1,0]
    return pos


def _lane_pos(x, lane_len):
    """Lane token positions, eager / outside the graph.

    Contains atan2/cos/sin for the heading rewrite; torchair's GE backend has
    no aten.atan2 converter, so this whole pos-extraction step stays out of
    the compiled unit and runs eagerly (same hoisting pattern as RouteEncoder
    in decoder.py, there for dynamic shapes instead).
    """
    x = x[..., :8]
    pos = x[:, :, int(lane_len / 2), :7].clone()  # x, y, x'-x, y'-y
    heading = torch.atan2(pos[..., 3], pos[..., 2])
    pos[..., 2] = torch.cos(heading)
    pos[..., 3] = torch.sin(heading)
    pos[..., -3:] = 0.0
    pos[..., -1] = 1.0  # lane: [0,0,1]
    return pos


class StaticEncoderBody(nn.Module):
    """Static-shape, graph-friendly rewrite of Encoder.forward.

    The upstream forward scatters rows through boolean masks (x[valid_indices]
    and x_result[valid_indices] = x), which the torchair GE backend rejects
    (dynamic shapes, same blocker as RouteEncoder in decoder.py). Here every
    row is computed and invalid rows are zeroed at the output instead of being
    skipped:

      - the three fusion encoders are row-independent (LayerNorm and Mlp act
        per row, MixerBlock mixes tokens within a row), so valid rows go
        through the identical op sequence as upstream -> bit-exact;
      - invalid rows carry all-zero inputs, produce finite garbage through the
        MLPs (GELU(bias) is finite, LayerNorm's eps guards the zero-variance
        case), and are multiplied by 0 at the end -- reproducing the upstream
        zero-initialized scatter;
      - the exact-zero output matters: DiT's cross-attention has no context
        mask, so invalid context tokens participate in softmax as zero
        vectors. That is the trained semantics, not something to optimize
        away.

    Three more upstream behaviors are translated for graph friendliness, all
    semantically identical: the pos extraction runs outside the compiled unit
    (atan2 has no GE converter), the lane speed-limit masked-fill branches
    become a torch.where (both sides computed, selected per element), and the
    fusion blocks' bool key_padding_mask becomes the float additive mask with
    the ego row un-masked out-of-place (upstream does mask[:, 0] = False in
    place; see decisions/003 for the float-mask rationale).
    """

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder  # weight alias; the holder list keeps this module out of any state_dict

    def _encode_neighbors(self, ne, x):
        """Static AgentFusionEncoder: [B,P,V,D] -> ([B,P,H], mask [B,P])."""
        neighbor_type = x[:, :, -1, 8:]
        x = x[..., :8]

        B, P, V, _ = x.shape
        mask_v = torch.sum(torch.ne(x[..., :8], 0), dim=-1).to(x.device) == 0
        mask_p = torch.sum(~mask_v, dim=-1) == 0
        x = torch.cat([x, (~mask_v).float().unsqueeze(-1)], dim=-1)
        x = x.view(B * P, V, -1)
        valid = (~mask_p).view(B * P, 1).to(x.dtype)  # zero rows out at the end

        x = ne.channel_pre_project(x)
        x = x.permute(0, 2, 1)
        x = ne.token_pre_project(x)
        x = x.permute(0, 2, 1)
        for block in ne.blocks:
            x = block(x)

        x = torch.mean(x, dim=1)

        type_embedding = ne.type_emb(neighbor_type.view(B * P, -1))
        x = x + type_embedding

        x = ne.emb_project(ne.norm(x))
        x = x * valid

        return x.view(B, P, -1), mask_p.reshape(B, -1)

    def _encode_static(self, se, x):
        """Static StaticFusionEncoder: [B,P,D] -> ([B,P,H], mask [B,P])."""
        B, P, _ = x.shape

        mask_p = torch.sum(torch.ne(x[..., :10], 0), dim=-1).to(x.device) == 0
        valid = (~mask_p).view(B * P, 1).to(x.dtype)

        x = se.projection(x.view(B * P, -1))
        x = x * valid

        return x.view(B, P, -1), mask_p.view(B, P)

    def _encode_lanes(self, le, x, speed_limit, has_speed_limit):
        """Static LaneFusionEncoder: [B,P,V,D] -> ([B,P,H], mask [B,P])."""
        traffic = x[:, :, 0, 8:]
        x = x[..., :8]

        B, P, V, _ = x.shape
        mask_v = torch.sum(torch.ne(x[..., :8], 0), dim=-1).to(x.device) == 0
        mask_p = torch.sum(~mask_v, dim=-1) == 0
        x = x.view(B * P, V, -1)
        valid = (~mask_p).view(B * P, 1).to(x.dtype)

        x = le.channel_pre_project(x)
        x = x.permute(0, 2, 1)
        x = le.token_pre_project(x)
        x = x.permute(0, 2, 1)
        for block in le.blocks:
            x = block(x)

        x = torch.mean(x, dim=1)

        # static where instead of the upstream masked-fill branches; the
        # invalid-lane rows compute garbage on both sides and are zeroed by
        # `valid` below, matching the upstream zero scatter
        sl = speed_limit.view(B * P, 1)
        has = has_speed_limit.view(B * P, 1)
        speed_limit_embedding = torch.where(
            has, le.speed_limit_emb(sl), le.unknown_speed_emb.weight.expand(B * P, -1)
        )
        traffic_light_embedding = le.traffic_emb(traffic.view(B * P, -1))

        x = x + speed_limit_embedding + traffic_light_embedding
        x = le.emb_project(le.norm(x))
        x = x * valid

        return x.view(B, P, -1), mask_p.reshape(B, -1)

    def forward(self, neighbors, static, lanes, lanes_speed_limit, lanes_has_speed_limit,
                neighbor_pos, static_pos, lane_pos):
        enc = self.encoder
        B = neighbors.shape[0]

        encoding_neighbors, neighbors_mask = self._encode_neighbors(enc.neighbor_encoder, neighbors)
        encoding_static, static_mask = self._encode_static(enc.static_encoder, static)
        encoding_lanes, lanes_mask = self._encode_lanes(enc.lane_encoder, lanes, lanes_speed_limit, lanes_has_speed_limit)

        encoding_input = torch.cat([encoding_neighbors, encoding_static, encoding_lanes], dim=1)

        encoding_pos = self.encoder.pos_emb(
            torch.cat([neighbor_pos, static_pos, lane_pos], dim=1).view(B * enc.token_num, -1)
        )
        encoding_mask = torch.cat([neighbors_mask, static_mask, lanes_mask], dim=1)  # [B, token_num]
        # upstream zeroes invalid rows via a zero-initialized scatter; multiplying
        # by the valid indicator is exact for finite garbage
        valid_flat = (~encoding_mask).to(encoding_pos.dtype).view(B * enc.token_num, 1)
        encoding_input = encoding_input + (encoding_pos * valid_flat).view(B, enc.token_num, -1)

        # ego token always attends (upstream mask[:, 0] = False), out-of-place
        mask_no_ego = torch.cat([torch.zeros_like(encoding_mask[:, :1]), encoding_mask[:, 1:]], dim=1)
        mask_f = torch.zeros(mask_no_ego.shape, dtype=encoding_input.dtype, device=encoding_input.device).masked_fill(mask_no_ego, float("-inf"))

        return enc.fusion(encoding_input, mask_f)


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.hidden_dim = config.hidden_dim

        self.token_num = config.agent_num + config.static_objects_num + config.lane_num

        self.neighbor_encoder = AgentFusionEncoder(config.time_len, drop_path_rate=config.encoder_drop_path_rate, hidden_dim=config.hidden_dim, depth=config.encoder_depth)
        self.static_encoder = StaticFusionEncoder(config.static_objects_state_dim, drop_path_rate=config.encoder_drop_path_rate, hidden_dim=config.hidden_dim)
        self.lane_encoder = LaneFusionEncoder(config.lane_len, drop_path_rate=config.encoder_drop_path_rate, hidden_dim=config.hidden_dim, depth=config.encoder_depth)

        self.fusion = FusionEncoder(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            drop_path_rate=config.encoder_drop_path_rate,
            depth=config.encoder_depth,
            device=config.device
        )

        # position embedding encode x, y, cos, sin, type
        self.pos_emb = nn.Linear(7, config.hidden_dim)

        # static graph-friendly view, built lazily on first use (weights must
        # be loaded first); held in a list so nn.Module.__setattr__ does not
        # register it into state_dict (same pattern as Decoder._sampler_holder)
        self._static_holder = [None]

    def _get_static_body(self):
        if self._static_holder[0] is None:
            body = StaticEncoderBody(self).eval()
            if _TORCHAIR:
                import torchair  # top-level import would break CUDA-only environments

                config = torchair.CompilerConfig()
                # persist compiled graph, same rationale and stale-cache
                # caveat as decoder.py (cache key has no source hash)
                self._static_holder[0] = torchair.inference.cache_compile(
                    body.forward,
                    config=config,
                    dynamic=False,
                    cache_dir=os.environ.get("DP_TORCHAIR_CACHE", "/data/syx_dp/torchair_cache"),
                )
            else:
                self._static_holder[0] = body
        return self._static_holder[0]

    def forward(self, inputs):

        if _TORCHAIR:
            body = self._get_static_body()
            # pos extraction stays eager: lane heading needs atan2, which has
            # no torchair GE converter (see _lane_pos)
            encoding = body(
                inputs['neighbor_agents_past'],
                inputs['static_objects'],
                inputs['lanes'],
                inputs['lanes_speed_limit'],
                inputs['lanes_has_speed_limit'],
                _agent_pos(inputs['neighbor_agents_past']),
                _static_pos(inputs['static_objects']),
                _lane_pos(inputs['lanes'], self.lane_encoder._lane_len),
            )
            return {"encoding": encoding}

        encoder_outputs = {}

        # agents
        neighbors = inputs['neighbor_agents_past']

        # static objects
        static = inputs['static_objects']

        # vector maps
        lanes = inputs['lanes']
        lanes_speed_limit = inputs['lanes_speed_limit']
        lanes_has_speed_limit = inputs['lanes_has_speed_limit']

        B = neighbors.shape[0]

        encoding_neighbors, neighbors_mask, neighbor_pos = self.neighbor_encoder(neighbors)
        encoding_static, static_mask, static_pos = self.static_encoder(static)
        encoding_lanes, lanes_mask, lane_pos = self.lane_encoder(lanes, lanes_speed_limit, lanes_has_speed_limit)

        encoding_input = torch.cat([encoding_neighbors, encoding_static, encoding_lanes], dim=1)

        encoding_pos = torch.cat([neighbor_pos, static_pos, lane_pos], dim=1).view(B * self.token_num, -1)
        encoding_mask = torch.cat([neighbors_mask, static_mask, lanes_mask], dim=1).view(-1)
        encoding_pos = self.pos_emb(encoding_pos[~encoding_mask])
        encoding_pos_result = torch.zeros((B * self.token_num, self.hidden_dim), device=encoding_pos.device)
        encoding_pos_result[~encoding_mask] = encoding_pos  # Fill in valid parts

        encoding_input = encoding_input + encoding_pos_result.view(B, self.token_num, -1)

        encoder_outputs['encoding'] = self.fusion(encoding_input, encoding_mask.view(B, self.token_num))

        return encoder_outputs


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim=192, heads=6, dropout=0.1, mlp_ratio=4.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)

        self.drop_path = DropPath(dropout) if dropout > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=dropout)

    def forward(self, x, mask):
        x = x + self.drop_path(self.attn(self.norm1(x), x, x, key_padding_mask=mask)[0])
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class AgentFusionEncoder(nn.Module):
    def __init__(self, time_len, drop_path_rate=0.3, hidden_dim=192, depth=3, tokens_mlp_dim=64, channels_mlp_dim=128):
        super().__init__()

        self._hidden_dim = hidden_dim
        self._channel = channels_mlp_dim

        self.type_emb = nn.Linear(3, channels_mlp_dim)

        self.channel_pre_project = Mlp(in_features=8+1, hidden_features=channels_mlp_dim, out_features=channels_mlp_dim, act_layer=nn.GELU, drop=0.)
        self.token_pre_project = Mlp(in_features=time_len, hidden_features=tokens_mlp_dim, out_features=tokens_mlp_dim, act_layer=nn.GELU, drop=0.)

        self.blocks = nn.ModuleList([MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate) for i in range(depth)])

        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(in_features=channels_mlp_dim, hidden_features=hidden_dim, out_features=hidden_dim, act_layer=nn.GELU, drop=drop_path_rate)


    def forward(self, x):
        '''
        x: B, P, V, D (x, y, cos, sin, vx, vy, w, l, type(3))
        '''
        neighbor_type = x[:, :, -1, 8:]
        x = x[..., :8]

        pos = x[:, :, -1, :7].clone() # x, y, cos, sin
        # neighbor: [1,0,0]
        pos[..., -3:] = 0.0
        pos[..., -3] = 1.0
        
        B, P, V, _ = x.shape
        mask_v = torch.sum(torch.ne(x[..., :8], 0), dim=-1).to(x.device) == 0
        mask_p = torch.sum(~mask_v, dim=-1) == 0
        x = torch.cat([x, (~mask_v).float().unsqueeze(-1)], dim=-1)
        x = x.view(B * P, V, -1)

        valid_indices = ~mask_p.view(-1) 
        x = x[valid_indices] 

        x = self.channel_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.token_pre_project(x)
        x = x.permute(0, 2, 1)
        for block in self.blocks:
            x = block(x)  

        # pooling
        x = torch.mean(x, dim=1)

        neighbor_type = neighbor_type.view(B * P, -1)
        neighbor_type = neighbor_type[valid_indices]
        type_embedding = self.type_emb(neighbor_type)  # Type embedding for valid data
        x = x + type_embedding

        x = self.emb_project(self.norm(x))

        x_result = torch.zeros((B * P, x.shape[-1]), device=x.device)
        x_result[valid_indices] = x  # Fill in valid parts
        
        return x_result.view(B, P, -1) , mask_p.reshape(B, -1), pos.view(B, P, -1)

    
class StaticFusionEncoder(nn.Module):
    def __init__(self, dim, drop_path_rate=0.3, hidden_dim=192, device='cuda'):
        super().__init__()

        self._hidden_dim = hidden_dim

        self.projection = Mlp(in_features=dim, hidden_features=hidden_dim, out_features=hidden_dim, act_layer=nn.GELU, drop=drop_path_rate)

    def forward(self, x):
        '''
        x: B, P, D (x, y, cos, sin, w, l, type(4))
        ''' 
        B, P, _ = x.shape

        pos = x[:, :, :7].clone() # x, y, cos, sin
        # static: [0,1,0]
        pos[..., -3:] = 0.0
        pos[..., -2] = 1.0

        x_result = torch.zeros((B * P, self._hidden_dim), device=x.device)

        mask_p = torch.sum(torch.ne(x[..., :10], 0), dim=-1).to(x.device) == 0

        valid_indices = ~mask_p.view(-1) 

        if valid_indices.sum() > 0:
            x = x.view(B * P, -1)
            x = x[valid_indices]
            x = self.projection(x)
            x_result[valid_indices] = x

        return x_result.view(B, P, -1), mask_p.view(B, P), pos.view(B, P, -1)
    

class LaneFusionEncoder(nn.Module):
    def __init__(self, lane_len, drop_path_rate=0.3, hidden_dim=192, depth=3, tokens_mlp_dim=64, channels_mlp_dim=128):
        super().__init__()

        self._lane_len = lane_len
        self._channel = channels_mlp_dim

        self.speed_limit_emb = nn.Linear(1, channels_mlp_dim)
        self.unknown_speed_emb = nn.Embedding(1, channels_mlp_dim)
        self.traffic_emb = nn.Linear(4, channels_mlp_dim)

        self.channel_pre_project = Mlp(in_features=8, hidden_features=channels_mlp_dim, out_features=channels_mlp_dim, act_layer=nn.GELU, drop=0.)
        self.token_pre_project = Mlp(in_features=lane_len, hidden_features=tokens_mlp_dim, out_features=tokens_mlp_dim, act_layer=nn.GELU, drop=0.)

        self.blocks = nn.ModuleList([MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate) for i in range(depth)])

        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(in_features=channels_mlp_dim, hidden_features=hidden_dim, out_features=hidden_dim, act_layer=nn.GELU, drop=drop_path_rate)

    def forward(self, x, speed_limit, has_speed_limit):
        '''
        x: B, P, V, D (x, y, x'-x, y'-y, x_left-x, y_left-y, x_right-x, y_right-y, traffic(4))
        speed_limit: B, P, 1
        has_speed_limit: B, P, 1
        '''
        traffic = x[:, :, 0, 8:]
        x = x[..., :8]

        pos = x[:, :, int(self._lane_len / 2), :7].clone() # x, y, x'-x, y'-y
        heading = torch.atan2(pos[..., 3], pos[..., 2])
        pos[..., 2] = torch.cos(heading)
        pos[..., 3] = torch.sin(heading)
        # lane: [0,0,1]
        pos[..., -3:] = 0.0
        pos[..., -1] = 1.0

        B, P, V, _ = x.shape
        mask_v = torch.sum(torch.ne(x[..., :8], 0), dim=-1).to(x.device) == 0
        mask_p = torch.sum(~mask_v, dim=-1) == 0
        x = x.view(B * P, V, -1)

        valid_indices = ~mask_p.view(-1) 
        x = x[valid_indices] 

        x = self.channel_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.token_pre_project(x)
        x = x.permute(0, 2, 1)
        for block in self.blocks:
            x = block(x)  

        x = torch.mean(x, dim=1)

        # Reshape speed_limit and traffic to match flattened dimensions
        speed_limit = speed_limit.view(B * P, 1)
        has_speed_limit = has_speed_limit.view(B * P, 1)
        traffic = traffic.view(B * P, -1)

        # Apply embedding directly to valid speed limit data
        has_speed_limit = has_speed_limit[valid_indices].squeeze(-1)
        speed_limit = speed_limit[valid_indices].squeeze(-1)
        speed_limit_embedding = torch.zeros((speed_limit.shape[0], self._channel), device=x.device)

        if has_speed_limit.sum() > 0:
            speed_limit_with_limit = self.speed_limit_emb(speed_limit[has_speed_limit].unsqueeze(-1))
            speed_limit_embedding[has_speed_limit] = speed_limit_with_limit

        if (~has_speed_limit).sum() > 0:
            speed_limit_no_limit = self.unknown_speed_emb.weight.expand(
                (~has_speed_limit).sum().item(), -1
            )
            speed_limit_embedding[~has_speed_limit] = speed_limit_no_limit

        # Process traffic lights directly for valid positions
        traffic = traffic[valid_indices]
        traffic_light_embedding = self.traffic_emb(traffic)  # Traffic light embedding for valid data


        x = x + speed_limit_embedding + traffic_light_embedding
        x = self.emb_project(self.norm(x))

        x_result = torch.zeros((B * P, x.shape[-1]), device=x.device)
        x_result[valid_indices] = x  # Fill in valid parts
        
        return x_result.view(B, P, -1) , mask_p.reshape(B, -1), pos.view(B, P, -1)


class FusionEncoder(nn.Module):
    def __init__(self, hidden_dim=192, num_heads=6, drop_path_rate=0.3, depth=3, device='cuda'):
        super().__init__()

        dpr = drop_path_rate

        self.blocks = nn.ModuleList(
            [SelfAttentionBlock(hidden_dim, num_heads, dropout=dpr) for i in range(depth)]
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, mask):

        mask[:, 0] = False

        for b in self.blocks:
            x = b(x, mask)

        return self.norm(x)