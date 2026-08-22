#!/usr/bin/env python3
"""Dump every criterion case in a run directory as TSV: name, mean_ns, lo_ns, hi_ns."""
import json
import os
import sys

root = sys.argv[1] if len(sys.argv) > 1 else "target/criterion"
rows = []
for dirpath, dirnames, filenames in os.walk(root):
    if os.path.basename(dirpath) != "new" or "estimates.json" not in filenames:
        continue
    name = os.path.relpath(os.path.dirname(dirpath), root)
    with open(os.path.join(dirpath, "estimates.json")) as f:
        est = json.load(f)["mean"]
    rows.append((name, est["point_estimate"], est["confidence_interval"]["lower_bound"], est["confidence_interval"]["upper_bound"]))

rows.sort()
for name, mean, lo, hi in rows:
    print(f"{name}\t{mean:.1f}\t{lo:.1f}\t{hi:.1f}")
