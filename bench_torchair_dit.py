"""TorchAir graph-mode benchmark for the DiT sampling-step body.

Loads the real model + captured inputs, builds one DiT forward call, then
compares eager vs torch.compile(backend=torchair NPU backend):
  - graph build (compile) wall time for the first call
  - numeric max-abs-diff of one DiT forward
  - latency of repeated forwards (the DPM-solver loop body)

The first run compiles the WHOLE DiT including RouteEncoder (which contains
boolean-mask dynamic indexing) — whether that traces is itself the experiment.
"""
import argparse
import time

import torch
import torch.nn as nn
import torch_npu  # noqa: F401  registers the npu backend
import torchair

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config


def load_model(args_file: str, ckpt_path: str) -> Diffusion_Planner:
    config = Config(args_file, None)  # guidance_fn=None matches production yaml
    model = Diffusion_Planner(config)
    state_dict = torch.load(ckpt_path, map_location="npu")["ema_state_dict"]
    model_state_dict = {k[len("module."):]: v for k, v in state_dict.items() if k.startswith("module.")}
    if not model_state_dict:
        model_state_dict = state_dict
    model.load_state_dict(model_state_dict)
    model.eval()
    return model.to("npu")


def timed(fn, iters):
    ts = []
    for _ in range(iters):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize()
        ts.append(time.perf_counter() - t0)
    ts = sorted(1000 * t for t in ts)
    return ts[len(ts) // 2], ts[0]


class DiTBody(nn.Module):
    """DiT.forward with RouteEncoder hoisted out.

    The graph backend rejects RouteEncoder's boolean-mask scatter
    (x_result[valid_indices] = x, dynamic shape). route_lanes is constant for
    the whole sampling step, so its encoding is computed once outside and fed
    in; the body is static-shape only. Mathematically identical to DiT.forward.
    """

    def __init__(self, dit):
        super().__init__()
        self.dit = dit

    def forward(self, x, t, cross_c, route_encoding, attn_mask):
        d = self.dit
        B, P, _ = x.shape
        x = d.preproj(x)
        x_embedding = torch.cat(
            [d.agent_embedding.weight[0][None, :], d.agent_embedding.weight[1][None, :].expand(P - 1, -1)], dim=0
        )
        x = x + x_embedding[None, :, :].expand(B, -1, -1)
        y = route_encoding + d.t_embedder(t)
        for block in d.blocks:
            x = block(x, cross_c, y, attn_mask)
        x = d.final_layer(x, y)
        return x  # model_type == "x_start" returns x directly (asserted below)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_file")
    parser.add_argument("--ckpt", default="/data/syx_dp/checkpoints/model.pth")
    parser.add_argument("--args-file", default="/data/syx_dp/checkpoints/args.json")
    parser.add_argument("--iters", type=int, default=30)
    opt = parser.parse_args()

    assert torch.npu.is_available(), "npu is not available"
    model = load_model(opt.args_file, opt.ckpt)
    inputs = torch.load(opt.capture_file)
    inputs = {k: v.to("npu") for k, v in inputs.items()}

    # model.decoder is the Diffusion_Planner_Decoder shell; the real Decoder
    # (with dit / _predicted_neighbor_num / _future_len) is model.decoder.decoder
    dec = model.decoder.decoder
    ego_current = inputs["ego_current_state"][:, None, :4]
    neighbors_current = inputs["neighbor_agents_past"][:, : dec._predicted_neighbor_num, -1, :4]
    neighbor_mask = torch.sum(torch.ne(neighbors_current[..., :4], 0), dim=-1) == 0  # [B, N]
    B, P, _ = torch.cat([ego_current, neighbors_current], dim=1).shape
    # DiT.forward pads the ego row internally; DiTBody takes the padded [B, P] mask.
    full_mask = torch.zeros((B, P), dtype=torch.bool, device=neighbor_mask.device)
    full_mask[:, 1:] = neighbor_mask
    out_dim = (dec._future_len + 1) * 4

    with torch.no_grad():
        cross_c = model.encoder(inputs)["encoding"]
        route_lanes = inputs["route_lanes"]
        x = torch.randn(B, P, out_dim, device="npu", dtype=torch.float32) * 0.5
        t = torch.full((B,), 0.5, device="npu", dtype=torch.float32)

        dit = dec.dit
        assert dit.model_type == "x_start", "DiTBody assumes x_start (no marginal_prob_std scaling)"

        # group 1: production eager DiT (recomputes RouteEncoder every call)
        for _ in range(3):
            ref = dit(x, t, cross_c, route_lanes, neighbor_mask)
        torch.npu.synchronize()
        med1, min1 = timed(lambda: dit(x, t, cross_c, route_lanes, neighbor_mask), opt.iters)

        # group 2: eager body with route encoding hoisted out (same math, computed once)
        body = DiTBody(dit).eval()
        route_encoding = dit.route_encoder(route_lanes)
        for _ in range(3):
            ref2 = body(x, t, cross_c, route_encoding, full_mask)
        torch.npu.synchronize()
        med2, min2 = timed(lambda: body(x, t, cross_c, route_encoding, full_mask), opt.iters)

        # group 3: graph-compiled body (torchair backend)
        config = torchair.CompilerConfig()
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        compiled = torch.compile(body, backend=npu_backend, dynamic=False)

        torch.npu.synchronize()
        t0 = time.perf_counter()
        out = compiled(x, t, cross_c, route_encoding, full_mask)
        torch.npu.synchronize()
        build_s = time.perf_counter() - t0
        print(f"graph build (first call): {build_s:.1f}s", flush=True)

        for _ in range(3):
            compiled(x, t, cross_c, route_encoding, full_mask)
        torch.npu.synchronize()
        med3, min3 = timed(lambda: compiled(x, t, cross_c, route_encoding, full_mask), opt.iters)

        diff2 = (ref - ref2).abs().max().item()
        diff3 = (ref - out).abs().max().item()
        scale = ref.abs().max().item()
        print(f"1 eager-DiT     p50={med1:7.2f} min={min1:7.2f} ms   (production, route recomputed)", flush=True)
        print(f"2 eager-body    p50={med2:7.2f} min={min2:7.2f} ms   (route hoisted; 1->2 saves {med1-med2:.2f} ms)", flush=True)
        print(f"3 graph-body    p50={med3:7.2f} min={min3:7.2f} ms   (vs 2 speedup {med2/max(med3,1e-9):.2f}x, vs 1 speedup {med1/max(med3,1e-9):.2f}x)", flush=True)
        print(f"numeric: body-vs-ref max_abs_diff={diff2:.3e}, graph-vs-ref max_abs_diff={diff3:.3e}, ref_scale={scale:.3e}", flush=True)


if __name__ == "__main__":
    main()
