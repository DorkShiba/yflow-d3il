"""Feasible sets C and their Euclidean projections Π_C for D3IL plans.

Every set works on a *raw-unit* plan tensor ``x`` of shape (B, H, 2 + obs_dim):
``x[:, :, :2]`` are the actions (delta of the desired hand XY, metres per step),
``x[:, 0, 2:]`` is the conditioned current observation (desired hand XY, actual
hand XY, boxes ...).  Only the action channels are modified: those are what the
evaluation loop executes.  Observation channels are left untouched.

    project(x)   -> x_proj     Euclidean projection of the action block onto C
    violation(x) -> dict       max / fraction of violating steps (0 == feasible)

Box-type sets are all handled by ONE exact QP (``BoxStack`` -> ``qp.BoxQP``):
  * ActionBox      alo <= a_t <= ahi                       (on the actions)
  * WorkspaceBox   lo <= des_t <= hi                       (on des_t = des_0 + sum_{k<=t} a_k)
  * TrackingBox    |des_t - hand_0|_inf <= d               (same, relative to the actual hand)
Their intersection is again "box on a" x "box on L a", so the projection onto
the intersection is exact - no alternating projections.  Only the non-box
StepNorm (||a_t||_2 <= r) is combined by Dykstra.

When ActionBox is present, the desired-hand bounds are first clipped into the
range reachable from des_0 under the action bound (cumsum(alo), cumsum(ahi)).
If des_0 already lies outside a box (e.g. the target ran away before this
plan), the constraint therefore becomes "move back as fast as allowed" instead
of an infeasible QP.
"""
from __future__ import annotations

import numpy as np
import torch

from .qp import BoxQP

ACT = slice(0, 2)       # action channels in the plan
DES = slice(2, 4)       # desired hand XY (observation channel 0:2)
HAND = slice(4, 6)      # actual hand XY (observation channel 2:4)


def cumulative_des(x):
    """Desired hand XY after executing action t: des_0 + sum_{k<=t} a_k.  (B, H, 2)"""
    return x[:, 0:1, DES] + torch.cumsum(x[:, :, ACT], dim=1)


def _np(x):
    return x.detach().to("cpu", torch.float64).numpy()


def _stats(v):
    return {"max": float(v.max()), "frac_steps": float((v > 1e-9).mean())}


class ConstraintSet:
    name = "base"
    kind = None          # "a": box on actions, "s": box on desired hand, None: generic

    def project(self, x):
        raise NotImplementedError

    def violation(self, x):
        raise NotImplementedError

    def describe(self):
        return {"name": self.name}


class ActionBox(ConstraintSet):
    """|a_{t,i}| <= a_max per axis.  The data only contains {-a_max, 0, +a_max}."""

    name, kind = "action_box", "a"

    def __init__(self, a_max):
        self.a_max = float(a_max)

    def a_bounds(self, xn):
        B, H = xn.shape[:2]
        return np.full((B, H, 2), -self.a_max), np.full((B, H, 2), self.a_max)

    def project(self, x):
        out = x.clone()
        out[:, :, ACT] = x[:, :, ACT].clamp(-self.a_max, self.a_max)
        return out

    def violation(self, x):
        v = (np.abs(_np(x)[:, :, ACT]) - self.a_max).clip(min=0.0).max(-1)
        return _stats(v)

    def describe(self):
        return {"name": self.name, "a_max": self.a_max}


class StepNorm(ConstraintSet):
    """||a_t||_2 <= r: bounds the hand displacement per control step (not a box)."""

    name = "step_norm"

    def __init__(self, r):
        self.r = float(r)

    def project(self, x):
        out = x.clone()
        a = x[:, :, ACT]
        n = a.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        out[:, :, ACT] = a * torch.clamp(self.r / n, max=1.0)
        return out

    def violation(self, x):
        v = (np.linalg.norm(_np(x)[:, :, ACT], axis=-1) - self.r).clip(min=0.0)
        return _stats(v)

    def describe(self):
        return {"name": self.name, "r": self.r}


class _DesBox(ConstraintSet):
    kind = "s"

    def s_bounds(self, xn):
        """Absolute desired-hand bounds (B, H, 2), numpy."""
        raise NotImplementedError

    def bounds(self, x):
        lo, hi = self.s_bounds(_np(x))
        return torch.from_numpy(lo).to(x), torch.from_numpy(hi).to(x)

    def project(self, x):
        return BoxStack([self]).project(x)

    def violation(self, x):
        xn = _np(x)
        lo, hi = self.s_bounds(xn)
        d = xn[:, 0:1, DES] + np.cumsum(xn[:, :, ACT], axis=1)
        return _stats(np.maximum(lo - d, d - hi).clip(min=0.0).max(-1))


class WorkspaceBox(_DesBox):
    """Desired hand XY stays inside a fixed workspace box [lo, hi] (metres)."""

    name = "workspace"

    def __init__(self, lo, hi, **_):
        self.lo = np.asarray(lo, dtype=np.float64)
        self.hi = np.asarray(hi, dtype=np.float64)

    def s_bounds(self, xn):
        B, H = xn.shape[:2]
        return np.broadcast_to(self.lo, (B, H, 2)).copy(), np.broadcast_to(self.hi, (B, H, 2)).copy()

    def describe(self):
        return {"name": self.name, "lo": self.lo.tolist(), "hi": self.hi.tolist()}


class TrackingBox(_DesBox):
    """|des_t - hand_0|_inf <= d: within one plan the desired hand may not run
    away from the actual hand by more than d metres (targets the 'desired
    position runs off' failure mode of the baseline)."""

    name = "tracking"

    def __init__(self, d, **_):
        self.d = float(d)

    def s_bounds(self, xn):
        B, H = xn.shape[:2]
        c0 = np.broadcast_to(xn[:, 0:1, HAND], (B, H, 2))
        return c0 - self.d, c0 + self.d

    def describe(self):
        return {"name": self.name, "d": self.d}


class BoxStack(ConstraintSet):
    """Exact projection onto the intersection of all box-type sets (one QP)."""

    name = "box_stack"

    def __init__(self, sets, rho=0.3, max_iter=5000, tol=1e-11):
        self.a_sets = [s for s in sets if s.kind == "a"]
        self.s_sets = [s for s in sets if s.kind == "s"]
        assert len(self.a_sets) + len(self.s_sets) == len(sets) and sets
        self.rho, self.max_iter, self.tol = rho, max_iter, tol
        self._qp = {}
        self.last = {}

    def qp_bounds(self, xn):
        """(alo, ahi, slo, shi) as (B, H, 2) numpy; s-bounds are offsets from des_0."""
        alo = ahi = slo = shi = None
        for s in self.a_sets:
            lo, hi = s.a_bounds(xn)
            alo = lo if alo is None else np.maximum(alo, lo)
            ahi = hi if ahi is None else np.minimum(ahi, hi)
        if self.s_sets:
            des0 = xn[:, 0:1, DES]
            for s in self.s_sets:
                lo, hi = s.s_bounds(xn)
                slo = lo - des0 if slo is None else np.maximum(slo, lo - des0)
                shi = hi - des0 if shi is None else np.minimum(shi, hi - des0)
            empty = slo > shi                       # disjoint boxes: aim at the middle
            mid = 0.5 * (slo + shi)
            slo, shi = np.where(empty, mid, slo), np.where(empty, mid, shi)
            if alo is not None:                     # keep it reachable -> always feasible
                cl, ch = np.cumsum(alo, axis=1), np.cumsum(ahi, axis=1)
                slo, shi = np.clip(slo, cl, ch), np.clip(shi, cl, ch)
        return alo, ahi, slo, shi

    def _solver(self, H):
        key = H
        if key not in self._qp:
            self._qp[key] = BoxQP(H, bool(self.a_sets), bool(self.s_sets), self.rho, self.max_iter, self.tol)
        return self._qp[key]

    def project(self, x):
        xn = _np(x)
        B, H = xn.shape[:2]
        alo, ahi, slo, shi = self.qp_bounds(xn)
        cols = lambda v: None if v is None else v.transpose(1, 0, 2).reshape(H, B * 2)   # noqa: E731
        qp = self._solver(H)
        a = qp.solve(cols(xn[:, :, ACT]), cols(alo), cols(ahi), cols(slo), cols(shi))
        self.last = qp.last
        out = x.clone()
        out[:, :, ACT] = torch.from_numpy(a.reshape(H, B, 2).transpose(1, 0, 2).copy()).to(x)
        return out

    def violation(self, x):
        return {s.name: s.violation(x) for s in self.a_sets + self.s_sets}

    def qp_residual(self, x):
        """Max violation of the bounds the QP actually solves (after the
        reachability clip).  ~tol when the projection is exact; differs from
        violation() only for plans that start outside a box (runaway)."""
        xn = _np(x)
        alo, ahi, slo, shi = self.qp_bounds(xn)
        r = 0.0
        if alo is not None:
            a = xn[:, :, ACT]
            r = max(r, float(np.maximum(alo - a, a - ahi).max()))
        if slo is not None:
            s = np.cumsum(xn[:, :, ACT], axis=1)
            r = max(r, float(np.maximum(slo - s, s - shi).max()))
        return max(r, 0.0)

    def describe(self):
        return {"name": self.name, "rho": self.rho, "tol": self.tol,
                "sets": [s.describe() for s in self.a_sets + self.s_sets]}


class Intersection(ConstraintSet):
    """Π onto C_1 ∩ ... ∩ C_n.  Box-type sets are merged into one exact BoxStack;
    a remaining generic set (StepNorm) is combined with it by Dykstra."""

    name = "intersection"

    def __init__(self, sets, iters=200):
        self.sets = list(sets)
        boxes = [s for s in self.sets if s.kind in ("a", "s")]
        other = [s for s in self.sets if s.kind not in ("a", "s")]
        self.parts = ([BoxStack(boxes)] if boxes else []) + other
        self.iters = int(iters)

    def project(self, x):
        if len(self.parts) == 1:
            return self.parts[0].project(x)
        a = x[:, :, ACT].clone()
        p = [torch.zeros_like(a) for _ in self.parts]
        y = x.clone()
        for _ in range(self.iters):
            for i, s in enumerate(self.parts):
                y[:, :, ACT] = a + p[i]
                a_new = s.project(y)[:, :, ACT]
                p[i] = a + p[i] - a_new
                a = a_new
        out = x.clone()
        out[:, :, ACT] = a
        return out

    def violation(self, x):
        return {s.name: s.violation(x) for s in self.sets}

    def describe(self):
        return {"name": self.name, "parts": [s.describe() for s in self.parts],
                "dykstra_iters": self.iters if len(self.parts) > 1 else 0}


def total_violation(cset, x):
    """Flatten violation() to {'<set>/max', '<set>/frac_steps'} for logging."""
    v = cset.violation(x)
    if "max" in v:
        return {f"{cset.name}/{k}": val for k, val in v.items()}
    return {f"{n}/{k}": val for n, d in v.items() for k, val in d.items()}
