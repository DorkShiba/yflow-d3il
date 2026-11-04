"""Finite-difference XY metrics at the environment action rate, in SI units."""
import numpy as np


def rms_vectors(values):
    return float(np.sqrt(np.mean(np.sum(values ** 2, axis=-1)))) if len(values) else None


def rollout_metrics(actions, desired, actual, plan_starts, dt, initial_desired, initial_actual):
    actions, desired, actual = [np.asarray(x, dtype=np.float64) for x in (actions, desired, actual)]
    starts = np.asarray(plan_starts, dtype=bool)
    if actions.ndim != 2 or actions.shape[1] != 2 or not len(actions):
        raise ValueError("Expected nonempty (T,2) actions")
    if desired.shape != actions.shape or actual.shape != actions.shape or starts.shape != (len(actions),):
        raise ValueError("Logs must have aligned XY shapes")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive seconds")
    initial_desired, initial_actual = [np.asarray(x, dtype=np.float64) for x in (initial_desired, initial_actual)]
    if initial_desired.shape != (2,) or initial_actual.shape != (2,):
        raise ValueError("Initial states must be XY vectors")
    if not all(np.isfinite(x).all() for x in (actions, desired, actual, initial_desired, initial_actual)):
        raise ValueError("Nonfinite rollout")
    delta = np.diff(actions, axis=0)
    seam = starts[1:]
    gap = np.linalg.norm(desired - actual, axis=1)
    out = {"action_delta_rms_m": rms_vectors(delta),
           "seam_velocity_jump_rms_m_s": rms_vectors(delta[seam] / dt),
           "within_chunk_velocity_change_rms_m_s": rms_vectors(delta[~seam] / dt),
           "seam_count": int(seam.sum()),
           "tracking_gap_mean_m": float(gap.mean()), "tracking_gap_max_m": float(gap.max()),
           "duration_s": len(actions) * dt}
    # A jerk stencil crosses a seam k when it includes velocity intervals
    # on both sides of k. Stencils ending at k and k+1 both qualify.
    jerk_seam = starts[1:-1] | starts[2:]
    for label, values, initial in (("desired", desired, initial_desired), ("actual", actual, initial_actual)):
        pos = np.vstack((initial, values))
        vel = np.diff(pos, axis=0) / dt
        acc = np.diff(pos, n=2, axis=0) / dt ** 2
        jerk = np.diff(pos, n=3, axis=0) / dt ** 3
        out.update({f"{label}_velocity_rms_m_s": rms_vectors(vel),
                    f"{label}_acceleration_rms_m_s2": rms_vectors(acc),
                    f"{label}_jerk_rms_m_s3": rms_vectors(jerk),
                    f"{label}_seam_jerk_rms_m_s3": rms_vectors(jerk[jerk_seam]),
                    f"{label}_jerk_energy_m2_s5": float(np.sum(jerk ** 2) * dt) if len(jerk) else None,
                    f"{label}_path_length_m": float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum())})
    return out


def summarize_metrics(rows):
    if not rows:
        return {}
    return {key: {"mean": float(np.mean(v)) if v else None, "n": len(v)}
            for key in next(iter(rows.values()))
            for v in [[r[key] for r in rows.values() if r[key] is not None]]}
