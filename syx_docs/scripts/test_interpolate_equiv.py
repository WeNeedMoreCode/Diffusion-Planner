"""Equivalence check: old shapely _interpolate_points vs vectorized numpy version.

Run inside the container env (shapely available). Old implementation is inlined
here because map_process.py no longer contains it.
"""
import numpy as np
from shapely.geometry import LineString

from diffusion_planner.data_process.map_process import _interpolate_points


def old_impl(line, num_point):
    line = LineString(line)
    return np.concatenate(
        [line.interpolate(d).coords._coords for d in np.linspace(0, line.length, num_point)]
    )


def main():
    rng = np.random.default_rng(0)
    worst = 0.0
    cases = 0
    for trial in range(300):
        n = int(rng.integers(2, 60))
        pts = np.cumsum(rng.normal(size=(n, 2)), axis=0)  # random-walk polyline
        if trial % 5 == 0 and n > 4:
            pts[3] = pts[2]  # inject duplicate points -> zero-length segment
        if trial % 7 == 0:
            pts[1] = pts[0]
        for num_point in (10, 20, 40):
            a = old_impl(pts, num_point)
            b = _interpolate_points(pts, num_point)
            assert a.shape == b.shape == (num_point, 2), (a.shape, b.shape)
            worst = max(worst, float(np.abs(a - b).max()))
            cases += 1
    print(f"cases={cases} worst_abs_diff={worst:.3e}")
    assert worst < 1e-9, f"NOT equivalent: worst diff {worst}"
    print("EQUIVALENT_OK")


if __name__ == "__main__":
    main()
