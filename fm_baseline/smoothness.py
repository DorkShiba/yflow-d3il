"""Diagnostics on executed XY displacement commands, not torque or jerk."""

import numpy as np


def rollout_metrics(actions, desired, actual, plan_starts):
    actions = np.asarray(actions, dtype=np.float64)
    desired = np.asarray(desired, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    starts = np.asarray(plan_starts, dtype=bool)
    if actions.ndim != 2 or actions.shape[1] != 2 or len(actions) == 0:
        raise ValueError("Expected nonempty (T,2) executed actions")
    if desired.shape != actions.shape or actual.shape != actions.shape or starts.shape != (len(actions),):
        raise ValueError("Rollout logs must have aligned lengths and XY dimensions")
    if not all(np.isfinite(v).all() for v in (actions, desired, actual)):
        raise ValueError("Nonfinite rollout diagnostic input")
    delta = np.linalg.norm(np.diff(actions, axis=0), axis=1)
    boundary = starts[1:]
    gap = np.linalg.norm(desired - actual, axis=1)

    def mean_or_none(values):
        return float(np.mean(values)) if len(values) else None

    # A full 300-transition tail is required; short successful episodes are
    # not classified as stationary. This threshold is diagnostic, not safety.
    tail_path = float(np.linalg.norm(np.diff(actual[-301:], axis=0), axis=1).sum()) if len(actual) >= 301 else None
    return {
        "action_delta_mean_m": mean_or_none(delta),
        "action_delta_max_m": float(delta.max()) if len(delta) else None,
        "within_plan_delta_mean_m": mean_or_none(delta[~boundary]),
        "replan_delta_mean_m": mean_or_none(delta[boundary]),
        "action_norm_mean_m": float(np.linalg.norm(actions, axis=1).mean()),
        "tracking_gap_mean_m": float(gap.mean()),
        "tracking_gap_max_m": float(gap.max()),
        "tracking_gap_over_0_1m_fraction": float(np.mean(gap > 0.1)),
        "tail_300_path_m": tail_path,
        "tail_300_stationary": None if tail_path is None else bool(tail_path < 0.05),
    }


def summarize_metrics(per_rollout):
    """Equal-episode means; counts expose unavailable short-rollout metrics."""
    if not per_rollout:
        return {}
    result = {}
    for key in next(iter(per_rollout.values())):
        values = [row[key] for row in per_rollout.values() if row[key] is not None]
        result[key] = {"mean": float(np.mean(values)) if values else None, "n": len(values)}
    return result
