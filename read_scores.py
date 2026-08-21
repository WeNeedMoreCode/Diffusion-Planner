"""Print metric aggregator scores from a nuPlan experiment dir."""
import glob
import sys

import pandas as pd

d = sys.argv[1]
files = glob.glob(f"{d}/**/aggregator_metric/**/*.parquet", recursive=True)
assert files, f"no aggregator parquet under {d}"
df = pd.concat(pd.read_parquet(f) for f in files)
key = ["scenario_type", "num_scenarios", "score", "no_ego_at_fault_collisions",
       "time_to_collision_within_bound", "ego_progress_along_expert_route",
       "drivable_area_compliance", "speed_limit_compliance"]
cols = [c for c in key if c in df.columns]
print(df[cols].to_string(index=False))
tot = df[df.num_scenarios == df.num_scenarios.max()]
if len(tot) and "score" in df.columns:
    print(f"\noverall final_score={float(tot.score.iloc[0]):.4f} (n={int(tot.num_scenarios.iloc[0])})")
