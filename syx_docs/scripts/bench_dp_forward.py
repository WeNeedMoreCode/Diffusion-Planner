"""Forward-only latency benchmark for Diffusion-Planner on Ascend NPU.

Replays inputs captured by planner.py (DP_CAPTURE_DIR=..., saved post-normalizer)
so no nuplan-devkit / map / observation machinery is needed here. Reports:
  - full forward latency (encoder + decoder)
  - encoder / decoder split (decoder = diffusion sampling, expected dominant)

Usage (inside container, conda env on PATH):
  ASCEND_RT_VISIBLE_DEVICES=7 python bench_dp_forward.py /data/syx_dp/capture \
      --ckpt /data/syx_dp/checkpoints/model.pth --args-file /data/syx_dp/checkpoints/args.json \
      [--iters 30] [--warmup 3]

Env: DP_JIT_WINDOW=1 (default) runs the forward under jit_compile=True like the
615ms baseline; DP_JIT_WINDOW=0 runs pure precompiled kernels.
"""

import argparse
import os
import time

import torch
import torch_npu  # Ascend NPU backend, hard dependency (project rule: no try/except)

torch.npu.set_compile_mode(jit_compile=os.environ.get("DP_JIT_WINDOW", "1") == "1")

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config


def load_model(args_file: str, ckpt_path: str) -> Diffusion_Planner:
    # guidance_fn=None matches the production planner yaml (guidance_fn: null -> uncond)
    config = Config(args_file, None)
    model = Diffusion_Planner(config)
    state_dict = torch.load(ckpt_path, map_location="npu")["ema_state_dict"]
    # strip DDP "module." prefix if present (same logic as planner.py initialize)
    model_state_dict = {k[len("module."):]: v for k, v in state_dict.items() if k.startswith("module.")}
    if not model_state_dict:
        model_state_dict = state_dict
    model.load_state_dict(model_state_dict)
    model.eval()
    return model.to("npu")


def bench(model: Diffusion_Planner, inputs: dict, warmup: int, iters: int) -> None:
    inputs = {k: v.to("npu") for k, v in inputs.items()}

    for _ in range(warmup):
        model(inputs)
    torch.npu.synchronize()

    full, enc, dec = [], [], []
    for _ in range(iters):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        encoder_outputs = model.encoder(inputs)
        torch.npu.synchronize()
        t1 = time.perf_counter()
        model.decoder(encoder_outputs, inputs)
        torch.npu.synchronize()
        t2 = time.perf_counter()
        full.append(t2 - t0)
        enc.append(t1 - t0)
        dec.append(t2 - t1)

    def stats(xs):
        xs = sorted(1000 * x for x in xs)
        n = len(xs)
        return xs[0], xs[n // 2], xs[int(n * 0.9)]

    for name, xs in (("full", full), ("encoder", enc), ("decoder", dec)):
        lo, mid, hi = stats(xs)
        print(f"  {name:8s} min={lo:7.1f} p50={mid:7.1f} p90={hi:7.1f} ms", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_dir", help="dir with inputs_pid*.pt saved by planner.py")
    parser.add_argument("--ckpt", default="/data/syx_dp/checkpoints/model.pth")
    parser.add_argument("--args-file", default="/data/syx_dp/checkpoints/args.json")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=30)
    opt = parser.parse_args()

    assert torch.npu.is_available(), "npu is not available"
    print(f"device: {torch.npu.get_device_name(0)}  jit_window={os.environ.get('DP_JIT_WINDOW', '1')}", flush=True)

    model = load_model(opt.args_file, opt.ckpt)

    files = sorted(f for f in os.listdir(opt.capture_dir) if f.startswith("inputs_pid") and f.endswith(".pt"))
    assert files, f"no captured inputs under {opt.capture_dir}"
    for f in files:
        print(f"== {f}", flush=True)
        inputs = torch.load(os.path.join(opt.capture_dir, f))
        bench(model, inputs, opt.warmup, opt.iters)


if __name__ == "__main__":
    main()
