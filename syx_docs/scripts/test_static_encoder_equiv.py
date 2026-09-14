"""Equivalence check: upstream Encoder.forward vs StaticEncoderBody (eager).

Loads the real checkpoint plus a captured forward input (DP_CAPTURE_DIR
artifact), runs both encoder paths on the same input and reports the encoding
difference. Expected: exact zero or float32-ulp level -- valid rows go through
the identical op sequence (row-independent modules), invalid rows are zeroed
by an exact multiply-by-zero, and the only reassociated math is the bool ->
float -inf key_padding_mask conversion in the fusion blocks (decomposed MHA
fills -inf for bool masks internally, so even that should match bit-for-bit).

Run inside the container with the eager body (DP_TORCHAIR unset):
  ASCEND_RT_VISIBLE_DEVICES=5 python test_static_encoder_equiv.py /data/syx_dp/capture
"""

import argparse
import os

import torch
import torch_npu  # Ascend NPU backend, hard dependency (project rule: no try/except)

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.model.module.encoder import StaticEncoderBody, _agent_pos, _static_pos, _lane_pos
from diffusion_planner.utils.config import Config


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_dir")
    parser.add_argument("--ckpt", default="/data/syx_dp/checkpoints/model.pth")
    parser.add_argument("--args-file", default="/data/syx_dp/checkpoints/args.json")
    opt = parser.parse_args()

    assert torch.npu.is_available(), "npu is not available"
    assert os.environ.get("DP_TORCHAIR", "0") != "1", "run with eager body (DP_TORCHAIR unset)"

    model = load_model(opt.args_file, opt.ckpt)
    upstream = model.encoder.encoder  # Encoder (upstream forward)
    static = StaticEncoderBody(upstream).eval()

    files = sorted(f for f in os.listdir(opt.capture_dir) if f.startswith("inputs_pid") and f.endswith(".pt"))
    assert files, f"no captured inputs under {opt.capture_dir}"
    for f in files:
        inputs = {k: v.to("npu") for k, v in torch.load(os.path.join(opt.capture_dir, f)).items()}

        with torch.no_grad():
            out_old = upstream(inputs)["encoding"]
            out_new = static(
                inputs["neighbor_agents_past"],
                inputs["static_objects"],
                inputs["lanes"],
                inputs["lanes_speed_limit"],
                inputs["lanes_has_speed_limit"],
                _agent_pos(inputs["neighbor_agents_past"]),
                _static_pos(inputs["static_objects"]),
                _lane_pos(inputs["lanes"], upstream.lane_encoder._lane_len),
            )

        diff = (out_old - out_new).abs()
        n_zero = int((diff == 0).sum().item())
        n_total = diff.numel()
        print(
            f"{f}: shape={tuple(out_old.shape)} bit-exact={n_zero}/{n_total} "
            f"max_abs_diff={diff.max().item():.3e} mean_abs_diff={diff.mean().item():.3e} "
            f"(|enc|_max={out_old.abs().max().item():.3f})",
            flush=True,
        )


if __name__ == "__main__":
    main()
