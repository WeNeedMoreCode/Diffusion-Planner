"""Fixed-seed equivalence check: upstream dpm_sampler vs fast_dpm_sample.

Loads the real checkpoint plus a captured forward input (DP_CAPTURE_DIR
artifact from planner.py), replicates the Decoder.forward inference preamble
(xT construction, constraint fn, begin_step), then runs both samplers on an
identical xT (same torch.manual_seed) and reports the final x0 difference.

Expected: float32-ulp-level divergence -- the fast path computes the schedule
coefficients once in a different association order and feeds the body output
directly (the noise_pred/data_pred wrapper cancels), so ~1e-6 relative diffs
are normal; anything above ~1e-3 relative means a formula mismatch.

Run inside the container (eager body, no torchair, so graph compilation noise
is out of the picture):
  ASCEND_RT_VISIBLE_DEVICES=5 python test_fast_dpm_equiv.py /data/syx_dp/capture \
      --ckpt /data/syx_dp/checkpoints/model.pth --args-file /data/syx_dp/checkpoints/args.json
"""

import argparse
import os

import torch
import torch_npu  # Ascend NPU backend, hard dependency (project rule: no try/except)

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.model.diffusion_utils import fast_dpm_sampler
from diffusion_planner.model.diffusion_utils.sampling import dpm_sampler
from diffusion_planner.utils.config import Config

SEED = 42


def load_model(args_file, ckpt_path):
    config = Config(args_file, None)
    model = Diffusion_Planner(config)
    state_dict = torch.load(ckpt_path, map_location="npu")["ema_state_dict"]
    model_state_dict = {k[len("module."):]: v for k, v in state_dict.items() if k.startswith("module.")}
    if not model_state_dict:
        model_state_dict = state_dict
    model.load_state_dict(model_state_dict)
    model.eval()
    return model.to("npu")


def make_xt(decoder, inputs, current_states):
    """Replicates the Decoder.forward inference-branch xT construction."""
    B, P, _ = current_states.shape
    torch.manual_seed(SEED)
    return torch.cat(
        [current_states[:, :, None],
         torch.randn(B, P, decoder._future_len, 4, device=current_states.device, dtype=torch.float32) * 0.5],
        dim=2,
    ).reshape(B, P, -1)


def run_pair(model, inputs):
    decoder_wrapper = model.decoder  # Diffusion_Planner_Decoder
    decoder = decoder_wrapper.decoder  # Decoder

    ego_current = inputs['ego_current_state'][:, None, :4]
    neighbors_current = inputs["neighbor_agents_past"][:, :decoder._predicted_neighbor_num, -1, :4]
    neighbor_current_mask = torch.sum(torch.ne(neighbors_current[..., :4], 0), dim=-1) == 0
    current_states = torch.cat([ego_current, neighbors_current], dim=1)
    B, P, _ = current_states.shape

    encoder_outputs = model.encoder(inputs)
    ego_neighbor_encoding = encoder_outputs['encoding']
    route_lanes = inputs['route_lanes']

    def constrain(xt, t, step):
        xt = xt.reshape(B, P, -1, 4)
        xt[:, :, 0, :] = current_states
        return xt.reshape(B, P, -1)

    sampler = decoder._sampler_holder[0]
    sampler.begin_step(route_lanes, neighbor_current_mask)

    # upstream path (correcting_xt_fn mutates xT in place, so each path gets a
    # freshly seeded xT)
    x0_old = dpm_sampler(
        sampler,
        make_xt(decoder, inputs, current_states),
        other_model_params={
            "cross_c": ego_neighbor_encoding,
            "route_lanes": route_lanes,
            "neighbor_current_mask": neighbor_current_mask,
        },
        dpm_solver_params={"correcting_xt_fn": constrain},
        model_wrapper_params={
            "classifier_fn": decoder._guidance_fn,
            "classifier_kwargs": {
                "model": decoder.dit,
                "model_condition": {
                    "cross_c": ego_neighbor_encoding,
                    "route_lanes": route_lanes,
                    "neighbor_current_mask": neighbor_current_mask,
                },
                "inputs": inputs,
                "observation_normalizer": decoder._observation_normalizer,
                "state_normalizer": decoder._state_normalizer,
            },
            "guidance_scale": 0.5,
            "guidance_type": "classifier" if decoder._guidance_fn is not None else "uncond",
        },
    )

    # fast path
    x0_new = fast_dpm_sampler.fast_dpm_sample(
        lambda x, t: sampler(x, t, ego_neighbor_encoding),
        make_xt(decoder, inputs, current_states),
        correcting_xt_fn=constrain,
    )
    return x0_old, x0_new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_dir")
    parser.add_argument("--ckpt", default="/data/syx_dp/checkpoints/model.pth")
    parser.add_argument("--args-file", default="/data/syx_dp/checkpoints/args.json")
    opt = parser.parse_args()

    assert torch.npu.is_available(), "npu is not available"
    assert os.environ.get("DP_TORCHAIR", "0") != "1", "run with eager body (DP_TORCHAIR unset)"

    model = load_model(opt.args_file, opt.ckpt)

    files = sorted(f for f in os.listdir(opt.capture_dir) if f.startswith("inputs_pid") and f.endswith(".pt"))
    assert files, f"no captured inputs under {opt.capture_dir}"
    for f in files:
        inputs = {k: v.to("npu") for k, v in torch.load(os.path.join(opt.capture_dir, f)).items()}
        x0_old, x0_new = run_pair(model, inputs)
        diff = (x0_old - x0_new).abs()
        scale = x0_old.abs().max().item()
        print(
            f"{f}: shape={tuple(x0_old.shape)} max_abs_diff={diff.max().item():.3e} "
            f"mean_abs_diff={diff.mean().item():.3e} (|x0|_max={scale:.3f}, "
            f"rel={diff.max().item() / max(scale, 1e-12):.3e})",
            flush=True,
        )


if __name__ == "__main__":
    main()
