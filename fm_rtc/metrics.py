"""SI-unit finite differences on controller-rate XY position samples."""
import numpy as np


def rms(v):
    return float(np.sqrt(np.mean(np.sum(v * v, axis=-1)))) if len(v) else None


def trajectory_metrics(desired, actual, initial_desired, initial_actual, seams, dt):
    out = {}
    for label, points, initial in [("desired", desired, initial_desired), ("actual", actual, initial_actual)]:
        points = np.vstack((initial, np.asarray(points)[:, :2]))
        jerk = np.diff(points, n=3, axis=0) / dt ** 3
        # seam action k starts at position p[k]; include only stencils spanning
        # old and new actions (k-1 and k), not those entirely inside the new chunk.
        selected = np.zeros(len(jerk), dtype=bool)
        for k in np.flatnonzero(seams):
            for j in (k-2, k-1):
                if 0 <= j < len(jerk):
                    selected[j] = True
        out[label + "_jerk_rms_m_s3"] = rms(jerk)
        out[label + "_seam_jerk_rms_m_s3"] = rms(jerk[selected])
        out[label + "_path_length_m"] = float(np.linalg.norm(np.diff(points, axis=0), axis=-1).sum())
    out["tracking_gap_mean_m"] = float(np.linalg.norm(np.asarray(desired) - np.asarray(actual)[:, :2], axis=-1).mean())
    out["duration_s"] = len(desired) * dt
    return out
