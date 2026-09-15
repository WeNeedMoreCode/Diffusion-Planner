"""Aggregate a py-spy speedscope JSON into per-function self/total share.

Usage: python agg_speedscope.py <file.json> [filter_substring]

Prints top functions by self time and, separately, rows matching a substring
(e.g. "map_process" or "agent") with their self/total share of all samples.
"""
import json
import sys
from collections import Counter

path = sys.argv[1]
needle = sys.argv[2] if len(sys.argv) > 2 else None

with open(path) as f:
    data = json.load(f)

frames = data["shared"]["frames"]
names = [f"{fr['name']} ({fr.get('file', '?').rsplit('/', 1)[-1]}:{fr.get('line', '?')})"
         for fr in frames]

self_cnt = Counter()
total_cnt = Counter()
n_samples = 0

for prof in data["profiles"]:
    for stack in prof["samples"]:
        n_samples += 1
        if stack:
            self_cnt[stack[-1]] += 1
        for fi in set(stack):
            total_cnt[fi] += 1

print(f"samples={n_samples} ({n_samples / 50.0:.1f}s @50Hz)\n")

print("=== TOP 40 by SELF ===")
for fi, c in self_cnt.most_common(40):
    print(f"{100.0 * c / n_samples:6.2f}% self  {100.0 * total_cnt[fi] / n_samples:6.2f}% total  {names[fi]}")

if needle:
    print(f"\n=== rows matching '{needle}' (by total) ===")
    rows = [(fi, c) for fi, c in total_cnt.items() if needle in names[fi]]
    for fi, c in sorted(rows, key=lambda x: -x[1]):
        print(f"{100.0 * c / n_samples:6.2f}% total  {100.0 * self_cnt.get(fi, 0) / n_samples:6.2f}% self  {names[fi]}")
