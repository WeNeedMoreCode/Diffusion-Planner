"""Parse [dp-stage] / [dp-adapt] timing lines from a simulation log and report distributions."""
import re
import sys
import statistics

PAT_STAGE = re.compile(
    r"\[dp-stage\] adapt=\s*([\d.]+) norm=\s*([\d.]+) fwd=\s*([\d.]+) post=\s*([\d.]+) total=\s*([\d.]+)"
)
PAT_ADAPT = re.compile(
    r"\[dp-adapt\] ego=\s*([\d.]+) agents=\s*([\d.]+) route=\s*([\d.]+) "
    r"mapquery=\s*([\d.]+) mapproc=\s*([\d.]+) to_tensor=\s*([\d.]+)"
)

def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))]

def report(rows, names, label, warm_cut=None):
    if not rows:
        print(f"\n== {label}: no lines found")
        return
    if warm_cut is not None:
        # total is the last column; drop rows whose total exceeds the cutoff
        rows = [r for r in rows if r[-1] < warm_cut]
        label += f" (warm <{warm_cut}ms)"
    print(f"\n== {label} (n={len(rows)})")
    print(f"{'stage':10s} {'p10':>7s} {'med':>7s} {'p90':>7s} {'max':>8s} {'med%':>6s}")
    tot = statistics.median(r[-1] for r in rows)
    for i, name in enumerate(names):
        xs = [r[i] for r in rows]
        med = statistics.median(xs)
        share = f"{100*med/tot:5.1f}%" if i < len(names) - 1 else "    - "
        print(f"{name:10s} {q(xs,0.1):7.1f} {med:7.1f} {q(xs,0.9):7.1f} {max(xs):8.1f} {share:>6s}")

def main():
    stages, adapts = [], []
    for line in open(sys.argv[1], errors="ignore"):
        m = PAT_STAGE.search(line)
        if m:
            stages.append(tuple(float(x) for x in m.groups()))
            continue
        m = PAT_ADAPT.search(line)
        if m:
            adapts.append(tuple(float(x) for x in m.groups()))
    assert stages or adapts, "no timing lines found"
    if stages:
        report(stages, ["adapt", "norm", "fwd", "post", "total"], "all steps")
        report(stages, ["adapt", "norm", "fwd", "post", "total"], "warm", warm_cut=2000)
    if adapts:
        names = ["ego", "agents", "route", "mapquery", "mapproc", "to_tensor", "sum"]
        rows = [r + (sum(r),) for r in adapts]
        report(rows, names, "adapt sub-stages all")
        report(rows, names, "adapt sub-stages warm", warm_cut=2000)

if __name__ == "__main__":
    main()
