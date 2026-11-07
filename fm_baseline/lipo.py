"""⑤ LiPo for the baseline evaluators (``--postprocess lipo``).

The method itself lives in ``fm_constraint/lipo.py`` (fixed-bound minimum-jerk QP,
the codex ``mode="qp"`` definition chosen on 2026-11-07). This wrapper applies it
to each new chunk of the unconstrained baseline: no blending with the previous
chunk, no projection constraints, synchronous zero-delay chunks.
"""
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from fm_constraint.lipo import jerk_qp

SOLVER = {'max_iter': 20000, 'feasibility_tolerance_m': 1e-7}


@dataclass(frozen=True)
class LipoConfig:
    path_bound_m: float = 0.0003

    def __post_init__(self):
        if not np.isfinite(self.path_bound_m) or self.path_bound_m <= 0:
            raise ValueError("LiPo path_bound_m must be finite and positive")


def xy(value, name):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 2 or not len(value) or not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite nonempty (T, 2)")
    return value


class ChunkPostprocessor:
    """One instance per episode. history = executed desired XY positions (last = current)."""
    def __init__(self, mode="none", execute_steps=8, config=None):
        if mode not in ("none", "lipo") or execute_steps < 1:
            raise ValueError("Invalid postprocessor mode or execute_steps")
        self.mode, self.execute_steps = mode, execute_steps
        self.config = config or LipoConfig()
        self.diagnostics = []

    def process(self, actions, history):
        actions, history = xy(actions, "actions"), xy(history, "history")[-3:]
        if self.execute_steps > len(actions):
            raise ValueError("execute_steps exceeds chunk horizon")
        start = perf_counter()
        if self.mode == "none":
            # Preserve baseline increments exactly.
            self.diagnostics.append({"solver_status": "not_run", "qp_correction_max_m": 0.0,
                                     "postprocess_ms": 1000 * (perf_counter() - start)})
            return actions.copy()
        solved = jerk_qp(actions, history, self.config.path_bound_m, SOLVER)
        if solved["status"] != "ok":
            # Never fall back silently to the raw chunk.
            raise RuntimeError(f"LiPo QP failed: {solved['status']} ({solved['solver_status']})")
        self.diagnostics.append({"solver_status": solved["solver_status"],
                                 "qp_ms": solved["latency_ms"],
                                 "jerk_cost_before_m2": solved["jerk_cost_before_m2"],
                                 "jerk_cost_after_m2": solved["jerk_cost_after_m2"],
                                 "qp_correction_max_m": solved["correction_max_m"],
                                 "postprocess_ms": 1000 * (perf_counter() - start)})
        return solved["value"]
