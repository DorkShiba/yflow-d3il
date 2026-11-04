"""Paired context bootstrap; never compare different experimental protocols."""
import argparse
import json
from pathlib import Path
import numpy as np


def compare(baseline, rtc, n_boot=10000):
    if baseline["method"] != "baseline" or rtc["method"] != "rtc":
        raise ValueError("Expected baseline and rtc results")
    if baseline["protocol"] != rtc["protocol"]:
        raise ValueError("Protocols differ: checkpoint, contexts, code and settings must match")
    key = lambda row: (row["context"], row["trajectory"])
    a, b = sorted(baseline["episodes"], key=key), sorted(rtc["episodes"], key=key)
    if [key(r) for r in a] != [key(r) for r in b]:
        raise ValueError("Episode identities differ")
    if len({key(r) for r in a}) != len(a):
        raise ValueError("Duplicate episodes")
    expected = baseline["protocol"]["n_contexts"] * baseline["protocol"]["n_traj"]
    if len(a) != expected:
        raise ValueError("Incomplete evaluation")
    output = {}
    metrics = ["success", "mean_distance", "steps", "plan_ms", "desired_jerk_rms_m_s3",
               "actual_jerk_rms_m_s3", "desired_seam_jerk_rms_m_s3", "actual_seam_jerk_rms_m_s3",
               "tracking_gap_mean_m", "duration_s", "actual_path_length_m"]
    for metric in metrics:
        pairs = [(x, y) for x, y in zip(a, b) if x[metric] is not None and y[metric] is not None
                 and np.isfinite(x[metric]) and np.isfinite(y[metric])]
        if not pairs:
            output[metric] = None
            continue
        contexts = sorted({x["context"] for x, _ in pairs})
        delta = np.array([np.mean([y[metric]-x[metric] for x, y in pairs if x["context"] == c]) for c in contexts])
        ci = None
        if len(contexts) > 1:
            rng = np.random.default_rng(20261104)
            boots = delta[rng.integers(0, len(delta), size=(n_boot, len(delta)))].mean(axis=1)
            ci = np.quantile(boots, [0.025, 0.975]).tolist()
        output[metric] = dict(baseline=float(np.mean([x[metric] for x, _ in pairs])),
                              rtc=float(np.mean([y[metric] for _, y in pairs])),
                              mean_context_difference=float(delta.mean()), ci95=ci,
                              valid_episodes=len(pairs), valid_contexts=len(contexts))
    return dict(protocol=baseline["protocol"], difference="rtc minus baseline", metrics=output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", required=True)
    p.add_argument("--rtc", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    result = compare(json.loads(Path(args.baseline).read_text()), json.loads(Path(args.rtc).read_text()))
    with Path(args.output).open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
