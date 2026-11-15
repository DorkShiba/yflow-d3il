"""Summarize a freshly executed paired Stacking comparison (including short smoke runs)."""

import argparse
import json
from pathlib import Path

import numpy as np

from .recompute_comparison import paired_stats


def summarize(root):
    root = Path(root)
    complete = json.loads((root / "COMPLETE.json").read_text())
    arms = complete["arms"]
    combo = next(arm for arm in arms if arm != "baseline")
    records = []
    expected = complete["contexts"] * complete["repeats"]
    for arm in arms:
        result = json.loads((root / arm / "results.json").read_text())
        rows = result["episodes"]
        if len(rows) != expected:
            raise ValueError("Incomplete arm")
        records.extend(dict(row, arm=arm) for row in rows)
    keys = [
        "success",
        "hand_xyz_jerk_rms",
        "boundary_joint_velocity_jump_rms",
        "plan_ms_mean",
        "filter_ms_sum",
    ]
    summary = dict(
        arms={},
        paired_differences={},
        scope="new ID simulation; one checkpoint; zero simulated delay; short smoke runs are not performance estimates",
        timing="observed wall-clock plan time, including any GPU contention; filter_ms_sum is per rollout",
        motion="finite per-rollout values; short/failed rollouts can have missing jerk or boundary metrics",
    )
    for arm in arms:
        rows = [r for r in records if r["arm"] == arm]
        values = dict(
            rollouts=len(rows), failures=sum(r["failure"] is not None for r in rows)
        )
        for key in keys:
            finite = [
                float(r[key])
                for r in rows
                if r[key] is not None and np.isfinite(r[key])
            ]
            values[key] = dict(
                mean=float(np.mean(finite)) if finite else None, valid=len(finite)
            )
        summary["arms"][arm] = values
    for key in keys:
        summary["paired_differences"][key] = paired_stats(records, combo, key)
    (root / "rollout_metrics.json").write_text(
        json.dumps(records, indent=2, allow_nan=False)
    )
    (root / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    args = p.parse_args()
    print(json.dumps(summarize(args.input), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
