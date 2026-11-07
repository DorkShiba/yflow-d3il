"""Discrete XY adaptation of LiPo (arXiv:2506.05165, equations 4–6).

Synchronous, zero-delay chunks. Bounds constrain the correction to the BLENDED
reference, not deviation from the raw policy. No dynamics/safety constraints.
"""
from dataclasses import dataclass
from time import perf_counter
import numpy as np


@dataclass(frozen=True)
class LipoConfig:
    blend_steps: int = 4
    blend_bound_m: float = 0.002
    path_bound_m: float = 0.0003

    def __post_init__(self):
        if isinstance(self.blend_steps, bool) or not isinstance(self.blend_steps, int) or self.blend_steps < 0:
            raise ValueError("blend_steps must be a nonnegative integer")
        if not all(np.isfinite(v) and v >= 0 for v in (self.blend_bound_m, self.path_bound_m)):
            raise ValueError("LiPo bounds must be finite nonnegative meters")


def xy(value, name):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 2 or not len(value) or not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite nonempty (T, 2)")
    return value


def minimum_jerk(reference, history, bounds):
    """Box QP on future positions; up to 3 executed samples remain fixed.

    Minimize ||D3 [history; reference + epsilon]||². Uniform dt^-6 is
    constant and omitted. Solve in millimeters to avoid tiny objective scales.
    Raise on solver failure rather than silently mixing failed trials with LiPo.
    """
    started = perf_counter()
    import clarabel
    from scipy import sparse

    reference, history = xy(reference, "reference"), xy(history, "history")[-3:]
    bounds = np.asarray(bounds, dtype=np.float64)
    n, h = len(reference), len(history)
    if bounds.shape != (n,) or not np.isfinite(bounds).all() or np.any(bounds < 0):
        raise ValueError("bounds must have shape (horizon,) and be finite/nonnegative")
    points = np.concatenate((history, reference))
    d3 = np.diff(np.eye(h + n), n=3, axis=0)
    a = d3[:, h:]
    unit = 0.001
    residual = d3 @ ((points - history[-1]) / unit)
    # Flatten axis-major so X and Y are independent blocks in one QP.
    design = sparse.kron(sparse.eye(2), sparse.csc_matrix(a), format="csc")
    p = sparse.triu(2 * (design.T @ design), format="csc")
    q = 2 * np.asarray(design.T @ residual.T.reshape(-1)).reshape(-1)
    eye = sparse.eye(2 * n, format="csc")
    constraints = sparse.vstack([eye, -eye], format="csc")
    limit = np.tile(bounds / unit, 2)
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.tol_gap_abs = settings.tol_gap_rel = settings.tol_feas = 1e-9
    solver = clarabel.DefaultSolver(p, q, constraints, np.tile(limit, 2),
                                   [clarabel.NonnegativeConeT(4 * n)], settings)
    sol = solver.solve()
    if str(sol.status) != "Solved":
        raise RuntimeError(f"LiPo QP failed: {sol.status}")
    correction = np.asarray(sol.x).reshape(2, n).T * unit
    if not np.isfinite(correction).all() or np.any(np.abs(correction) > bounds[:, None] + 1e-8):
        raise RuntimeError("LiPo QP violated perturbation bounds")
    # Eliminate sub-tolerance numerical box violations, including zero bounds.
    correction = np.clip(correction, -bounds[:, None], bounds[:, None])
    optimized = reference + correction
    before = float(np.sum(np.diff(points, n=3, axis=0) ** 2))
    after = float(np.sum(np.diff(np.concatenate((history, optimized)), n=3, axis=0) ** 2))
    if after > before + max(1e-16, before * 1e-6):
        raise RuntimeError("LiPo QP increased the discrete jerk objective")
    return optimized, {"solver_status": str(sol.status), "qp_ms": 1000 * (perf_counter() - started), "jerk_cost_before_m2": before,
                       "jerk_cost_after_m2": after, "qp_correction_max_m": float(np.abs(correction).max())}


class ChunkPostprocessor:
    """One instance per episode; previous_tail is in absolute desired XY."""
    def __init__(self, mode="none", execute_steps=8, config=None):
        if mode not in ("none", "blend", "lipo") or execute_steps < 1:
            raise ValueError("Invalid postprocessor mode or execute_steps")
        self.mode, self.execute_steps = mode, execute_steps
        self.config = config or LipoConfig()
        self.previous_tail = None
        self.diagnostics = []

    def process(self, actions, history):
        actions, history = xy(actions, "actions"), xy(history, "history")[-3:]
        if self.execute_steps > len(actions):
            raise ValueError("execute_steps exceeds chunk horizon")
        if self.mode != "none" and self.config.blend_steps > len(actions) - self.execute_steps:
            raise ValueError("blend_steps exceeds unexecuted overlap (horizon - execute_steps)")
        start = perf_counter()
        raw = history[-1] + np.cumsum(actions, axis=0)
        ref = raw.copy()
        blend = 0
        if self.mode != "none" and self.previous_tail is not None:
            blend = self.config.blend_steps
            if blend:
                alpha = (np.arange(1, blend + 1) / blend)[:, None]
                ref[:blend] = (1 - alpha) * self.previous_tail[:blend] + alpha * raw[:blend]
        diag = {"solver_status": "not_run", "qp_correction_max_m": 0.0}
        if self.mode == "lipo":
            bounds = np.full(len(ref), self.config.path_bound_m)
            bounds[:blend] = self.config.blend_bound_m
            optimized, diag = minimum_jerk(ref, history, bounds)
        else:
            optimized = ref
        self.previous_tail = optimized[self.execute_steps:].copy()
        # Preserve baseline increments exactly, avoiding a cumsum/diff roundtrip.
        output = actions.copy() if self.mode == "none" else np.diff(np.vstack((history[-1], optimized)), axis=0)
        diag.update(postprocess_ms=1000 * (perf_counter() - start),
                    raw_deviation_max_m=float(np.abs(optimized - raw).max()),
                    blend_deviation_max_m=float(np.abs(ref - raw).max()), blend_steps=blend)
        self.diagnostics.append(diag)
        return output
