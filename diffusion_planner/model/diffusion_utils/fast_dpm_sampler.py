"""Precomputed-coefficient DPM-Solver++ sampler for the fixed inference config.

The generic dpm_solver_pytorch path recomputes noise-schedule scalars on the
NPU at every step: marginal_lambda / marginal_std / expm1 chains around each
solver update plus a noise_pred -> data_pred wrapper around every model call.
On the 310P each of those is a host-launched small-tensor op (~30 per NFE,
~550 over the 11 NFEs of a 10-step sample), which py-spy measured at ~8% of
the whole planning step. Every one of those scalars depends only on the noise
schedule constants and the step index -- never on the input data -- so they
are precomputed once per process on CPU (via the upstream NoiseScheduleVP
class, so the formulas are literally the upstream ones) and the runtime loop
reduces to one linear combination per step.

For model_type == "x_start" with algorithm_type == "dpmsolver++" the wrapper
math cancels exactly: model_fn returns
    (x - sigma_t * (x - alpha_t * body_out) / sigma_t) / alpha_t == body_out,
so the fast path feeds the body output directly into the solver update, and
the final denoise_to_zero step is simply the body output itself.

Only this project's exact sampling config is supported (multistep, order=2,
skip_type='logSNR', linear schedule, denoise_to_zero=True, uncond guidance).
Anything else must keep using the upstream sampler. Equivalence is checked by
fixed-seed replay (test_fast_dpm_equiv.py).
"""

from collections import namedtuple
from typing import Callable, Optional

import torch

import diffusion_planner.model.diffusion_utils.dpm_solver_pytorch as dpm

# a[k]: coefficient of x in the update ending at timestep k
# b[1]: coefficient of m0 in the first (order-1) update
# c0[k]/c1[k]: coefficients of m0/m1 in the second-order update ending at k
_Coeffs = namedtuple("_Coeffs", "a b c0 c1 timesteps t_zero")

_COEFF_CACHE = {}  # steps -> _Coeffs
_TS_CACHE = {}  # (steps, device) -> (timesteps tensor, t_zero tensor) on device


def _precompute(steps: int) -> _Coeffs:
    """Replicates the upstream scalar chain once, on CPU float32.

    Formulas mirror DPM_Solver.get_time_steps('logSNR'), dpm_solver_first_update
    and multistep_dpm_solver_second_update (dpmsolver++ / solver_type
    'dpmsolver'), evaluated on the upstream NoiseScheduleVP so the numbers come
    from the same code that produced the baseline.
    """
    ns = dpm.NoiseScheduleVP(schedule='linear')
    t_T = ns.T
    t_0 = 1.0 / ns.total_N

    # --- timesteps: upstream get_time_steps, skip_type='logSNR' branch ---
    lambda_T = ns.marginal_lambda(torch.tensor(t_T))
    lambda_0 = ns.marginal_lambda(torch.tensor(t_0))
    ts = ns.inverse_lambda(torch.linspace(lambda_T.item(), lambda_0.item(), steps + 1))

    sigma = [ns.marginal_std(ts[k]) for k in range(steps + 1)]
    lam = [ns.marginal_lambda(ts[k]) for k in range(steps + 1)]
    alpha = [torch.exp(ns.marginal_log_mean_coeff(ts[k])) for k in range(steps + 1)]

    a = [None] * (steps + 1)
    b = [None] * (steps + 1)
    c0 = [None] * (steps + 1)
    c1 = [None] * (steps + 1)

    # step 1: dpm_solver_first_update (dpmsolver++):
    #   x_t = sigma_t/sigma_s * x - alpha_t * expm1(-h) * model_s
    h = lam[1] - lam[0]
    a[1] = (sigma[1] / sigma[0]).item()
    b[1] = (alpha[1] * torch.expm1(-h)).item()

    # steps 2..steps: multistep_dpm_solver_second_update (dpmsolver++):
    #   x_t = sigma_t/sigma_s * x - coef*m0 - 0.5*coef*(1/r0)*(m0 - m1)
    #       = sigma_t/sigma_s * x - coef*(1 + 0.5/r0)*m0 + coef*(0.5/r0)*m1
    for k in range(2, steps + 1):
        h_0 = lam[k - 1] - lam[k - 2]
        h = lam[k] - lam[k - 1]
        r0 = h_0 / h
        coef = alpha[k] * torch.expm1(-h)
        a[k] = (sigma[k] / sigma[k - 1]).item()
        c0[k] = (-coef * (1.0 + 0.5 / r0)).item()
        c1[k] = (coef * (0.5 / r0)).item()

    return _Coeffs(
        a=a, b=b, c0=c0, c1=c1,
        timesteps=[ts[k].item() for k in range(steps + 1)],
        t_zero=t_0,
    )


def fast_dpm_sample(
    model_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    correcting_xt_fn: Optional[Callable] = None,
    steps: int = 10,
) -> torch.Tensor:
    """10-step multistep DPM-Solver++ sample with precomputed coefficients.

    model_fn(x, t) must return the x_start (data) prediction body output
    directly -- the noise_pred/data_pred wrapper cancels for x_start +
    dpmsolver++ (see module docstring). correcting_xt_fn keeps the upstream
    signature (xt, t, step).
    """
    coeffs = _COEFF_CACHE.get(steps)
    if coeffs is None:
        coeffs = _precompute(steps)
        _COEFF_CACHE[steps] = coeffs

    cached = _TS_CACHE.get((steps, x.device))
    if cached is None:
        cached = (
            torch.tensor(coeffs.timesteps, dtype=torch.float32, device=x.device),
            torch.tensor([coeffs.t_zero], dtype=torch.float32, device=x.device),
        )
        _TS_CACHE[(steps, x.device)] = cached
    ts, t_zero = cached
    a, b, c0, c1 = coeffs.a, coeffs.b, coeffs.c0, coeffs.c1

    with torch.no_grad():
        # step 0: model at timesteps[0], then constrain (upstream applies
        # correcting_xt_fn to x_T before any update)
        m0 = model_fn(x, ts[0:1])
        if correcting_xt_fn is not None:
            x = correcting_xt_fn(x, ts[0:1], 0)

        # step 1: first-order init update, then model at timesteps[1]
        x = a[1] * x - b[1] * m0
        if correcting_xt_fn is not None:
            x = correcting_xt_fn(x, ts[1:2], 1)
        m1 = m0
        m0 = model_fn(x, ts[1:2])

        # steps 2..steps: second-order multistep updates
        for k in range(2, steps + 1):
            x = a[k] * x + c0[k] * m0 + c1[k] * m1
            if correcting_xt_fn is not None:
                x = correcting_xt_fn(x, ts[k:k + 1], k)
            m1 = m0
            if k < steps:
                m0 = model_fn(x, ts[k:k + 1])

        # denoise_to_zero at t_0: data_prediction_fn(x, t_0) == body output
        x = model_fn(x, t_zero)
        if correcting_xt_fn is not None:
            x = correcting_xt_fn(x, t_zero, steps + 1)

    return x
