"""Constraint (2): Predict-then-Correct with a finite-time CBF (meeting_1008 p.12).

Follows SafeFlowMatcher (Yang, Jang, Han, ICLR 2026, Algorithm 1):

  Predict   tau^p_1 = Euler(v_theta, T^p steps from noise)            (T^p = 1: x0 + v(x0, 0))
  Correct   tau^c_0 = tau^p_1,  t in [0, 1] on T^c steps:
            v~_t  = alpha (1 - t) v_theta(tau^c_t, t)                  (vanishing time-scaled field)
            u*    = argmin_u ||u - v~_t||^2
                    s.t.  grad b_j(tau)^T u + eps sgn(b_j - delta) |b_j - delta|^rho >= 0   for every face j
            tau^c_{t+dt} = tau^c_t + dt u*

Intermediate latents of the prediction are never touched; only the predicted
path is corrected.

Adaptation to the D3IL plans
----------------------------
* Only the action channels are constrained (raw units, metres/step); the
  observation channels follow v~ unchanged and the t=0 observation stays fixed.
* Barriers come from the same box sets as constraint (1) (action_box,
  tracking, workspace).  Each box gives two linear faces, normalised by its
  half-width w so that b is O(1) and delta keeps the paper's meaning:
      b_up = (hi - q) / w,   b_lo = (q - lo) / w,   q = a_t   or   q = (L a)_t.
  The CBF condition on a linear face is a bound on u (or on L u):
      u <= w phi(b_up),  u >= -w phi(b_lo),   phi(b) = eps sgn(b - delta)|b - delta|^rho,
  so the per-step CBF-QP is again "box on u x box on L u" and is solved
  exactly by qp.BoxQP (same solver as constraint (1)).  Feasibility: the L u
  bounds are clipped into the range reachable under the u bounds, i.e. the
  action faces take priority (analogue of the paper's slack before t_w).
* No slack (t_w = 0, as the paper uses for its smallest T^c).
* Default eps: largest value for which a forward-Euler step can never carry a
  linear barrier from b > 0 to b < 0:  dt * eps <= min_{b>delta} b / (b - delta)^rho
  = (delta/(1-rho)) / (delta rho/(1-rho))^rho  (= 0.873 for delta=0.01, rho=0.9),
  times 0.9 for margin.  A violated face (b = -1) is restored in about 2 steps.
* T^p + T^c defaults to the baseline's ODE step count, so the number of
  network evaluations is identical to the baseline (10).
"""
from __future__ import annotations

import time

import numpy as np
import torch

from .qp import BoxQP
from .sets import ACT, DES, _np


def _phi(b, eps, rho, delta):
    z = b - delta
    return eps * np.sign(z) * np.abs(z) ** rho


def default_eps(dt, rho, delta, margin=0.9):
    b_star = delta / (1.0 - rho)
    return margin * b_star / (b_star - delta) ** rho / dt


class PCSampler:
    name = "pc"

    def __init__(self, sets, space, alpha=2.0, eps=None, rho=0.9, delta=0.01, tp=1, tc=None):
        self.a_sets = [s for s in sets if s.kind == "a"]
        self.s_sets = [s for s in sets if s.kind == "s"]
        if not (self.a_sets or self.s_sets):
            raise ValueError("Predict-then-Correct needs at least one box-type set (action_box / tracking / workspace)")
        self.space = space
        self.alpha, self.rho, self.delta = float(alpha), float(rho), float(delta)
        self.eps_arg = eps
        self.tp, self.tc = int(tp), (None if tc is None else int(tc))
        self._qp = {}
        self._t = 0.0
        self._n = 0
        self._active = []

    # ------------------------------------------------------------------ CBF
    def u_bounds(self, xn, eps):
        a = xn[:, :, ACT]
        s = np.cumsum(a, axis=1)
        des0 = xn[:, 0:1, DES]
        f = lambda b: _phi(b, eps, self.rho, self.delta)  # noqa: E731
        ulo = uhi = slo = shi = None
        for st in self.a_sets:
            lo, hi = st.a_bounds(xn)
            w = np.maximum((hi - lo) / 2, 1e-12)
            up, dn = w * f((hi - a) / w), -w * f((a - lo) / w)
            uhi = up if uhi is None else np.minimum(uhi, up)
            ulo = dn if ulo is None else np.maximum(ulo, dn)
        for st in self.s_sets:
            lo, hi = st.s_bounds(xn)
            lo, hi = lo - des0, hi - des0
            w = np.maximum((hi - lo) / 2, 1e-12)
            up, dn = w * f((hi - s) / w), -w * f((s - lo) / w)
            shi = up if shi is None else np.minimum(shi, up)
            slo = dn if slo is None else np.maximum(slo, dn)
        for lo_, hi_ in ((ulo, uhi), (slo, shi)):
            if lo_ is not None:
                bad = lo_ > hi_
                mid = 0.5 * (lo_ + hi_)
                lo_[bad], hi_[bad] = mid[bad], mid[bad]
        if ulo is not None and slo is not None:
            cl, ch = np.cumsum(ulo, axis=1), np.cumsum(uhi, axis=1)
            slo, shi = np.clip(slo, cl, ch), np.clip(shi, cl, ch)
        return ulo, uhi, slo, shi

    def _cbf(self, v, x, eps):
        t0 = time.perf_counter()
        xn = _np(self.space.to_raw(x))
        B, H = xn.shape[:2]
        s_act = self.space.scale[ACT].detach().to("cpu", torch.float64).numpy()
        v_raw = _np(v[:, :, ACT]) * s_act                         # d a_raw / dt
        ulo, uhi, slo, shi = self.u_bounds(xn, eps)
        cols = lambda q: None if q is None else q.transpose(1, 0, 2).reshape(H, B * 2)   # noqa: E731
        key = H
        if key not in self._qp:
            self._qp[key] = BoxQP(H, ulo is not None, slo is not None)
        u = self._qp[key].solve(cols(v_raw), cols(ulo), cols(uhi), cols(slo), cols(shi))
        u = u.reshape(H, B, 2).transpose(1, 0, 2) / s_act
        out = v.clone()
        out[:, :, ACT] = torch.from_numpy(u.copy()).to(v)
        self._active.append(float(np.mean(np.abs(u * s_act - v_raw) > 1e-9)))
        self._t += time.perf_counter() - t0
        self._n += 1
        return out

    # ------------------------------------------------------------ sampling
    @torch.no_grad()
    def solve(self, model, x, cond, act_dim, apply_conditioning, injector, n_steps):
        tp = self.tp
        tc = self.tc if self.tc is not None else max(n_steps - tp, 1)
        dt_c = 1.0 / tc
        eps = self.eps_arg if self.eps_arg is not None else default_eps(dt_c, self.rho, self.delta)
        self.eps_used, self.tc_used = eps, tc
        self._active = []
        injector.reset(tp + tc)
        k = 0

        def zero_cond(v):
            for tcnd in cond:
                v[:, tcnd, act_dim:] = 0.0
            return v

        # Predict: plain Euler, no safety term
        for t in torch.linspace(0.0, 1.0, tp + 1, device=x.device)[:-1]:
            x = injector.before_step(x, k, t)
            v = injector.velocity(zero_cond(model(x, t)), x, k, t)
            x = injector.after_step(x + v / tp, k, t)
            x = apply_conditioning(x, cond, act_dim).detach()
            k += 1
        # Correct: vanishing field + CBF-QP, t in [0, 1)
        for t in torch.linspace(0.0, 1.0, tc + 1, device=x.device)[:-1]:
            x = injector.before_step(x, k, t)
            v = zero_cond(model(x, t)) * (self.alpha * (1.0 - float(t)))
            v = injector.velocity(v, x, k, t)
            v = zero_cond(self._cbf(v, x, eps))
            x = injector.after_step(x + v * dt_c, k, t)
            x = apply_conditioning(x, cond, act_dim).detach()
            k += 1
        x = injector.finalize(x)
        return apply_conditioning(x, cond, act_dim)

    def stats(self):
        return {"pc_qp_calls": self._n, "pc_qp_ms_per_call": 1000.0 * self._t / max(self._n, 1),
                "pc_active_frac": float(np.mean(self._active)) if self._active else 0.0}

    def describe(self):
        return {"name": self.name, "alpha": self.alpha, "eps": self.eps_arg if self.eps_arg is not None else "auto(0.9*dt-safe)",
                "rho": self.rho, "delta": self.delta, "tp": self.tp, "tc": self.tc if self.tc is not None else "ode_steps-tp",
                "sets": [s.describe() for s in self.a_sets + self.s_sets]}
