"""Recompute paired statistics from published per-rollout metrics, not raw motion."""

import argparse
import json
from pathlib import Path

import numpy as np


def paired_stats(records, arm, metric, draws=10000, seed=20261115):
    values = {}
    for row in records:
        context = row.get("context_id", row.get("context"))
        key = (context, row["repeat"], row["arm"])
        if key in values:
            raise ValueError("Duplicate context/repeat/arm")
        values[key] = row[metric]
    contexts = sorted({key[0] for key in values})
    repeats = sorted({key[1] for key in values})
    delta = []
    for context in contexts:
        differences = []
        for repeat in repeats:
            a, b = values[context, repeat, "baseline"], values[context, repeat, arm]
            differences.append(
                float(b) - float(a) if a is not None and b is not None else np.nan
            )
        delta.append(differences)
    delta = np.asarray(delta)
    count = np.isfinite(delta).sum(1)
    total = np.nansum(delta, axis=1)
    indices = np.random.default_rng(seed).integers(
        0, len(contexts), size=(draws, len(contexts))
    )
    denominators = count[indices].sum(1)
    bootstrap = total[indices].sum(1)[denominators > 0] / denominators[denominators > 0]
    return dict(
        combo_minus_baseline=float(np.nanmean(delta)) if count.sum() else None,
        ci95=(
            np.quantile(bootstrap, [0.025, 0.975]).tolist() if len(bootstrap) else None
        ),
        valid_pairs=int(count.sum()),
        valid_contexts=int((count > 0).sum()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.report_dir)
    records = json.loads((root / "rollout_metrics.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    arms = sorted({r["arm"] for r in records})
    arm = next(a for a in arms if a != "baseline")
    if len(arms) != 2 or len(records) != 800:
        raise ValueError("Expected paired baseline/combo 800-rollout release")
    output = {
        metric: paired_stats(records, arm, metric)
        for metric in summary["paired_differences"]
    }
    for metric, result in output.items():
        stored = summary["paired_differences"][metric]
        np.testing.assert_allclose(
            result["combo_minus_baseline"], stored["combo_minus_baseline"], atol=1e-10
        )
        np.testing.assert_allclose(result["ci95"], stored["ci95"], atol=1e-10)
        assert result["valid_pairs"] == stored["valid_pairs"]
    Path(args.output).write_text(
        json.dumps(
            dict(
                passed=True, raw_time_series_rechecked=False, paired_differences=output
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
