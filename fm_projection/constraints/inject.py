"""Constraint-injection wrappers around the baseline flow-matching sampler.

The baseline integrates the learned velocity field v_θ with Euler steps
(HardFlow ``ConditionedODESolver``, ``value=None``):

    x_{k+1} = x_k + Δt · v_θ(x_k, t_k),      obs row t=0 re-conditioned each step.

``euler_solve`` below reproduces that loop *bit-for-bit* when no injector is
attached (checked by ``check_constraints.py``), and exposes hooks that the
injection methods plug into:

    before_step(x, k, t)      -> x        (e.g. guidance)
    velocity(v, x, k, t)      -> v        (e.g. a corrected field)
    after_step(x, k, t)       -> x        (e.g. method ①: projection every step)
    finalize(x)               -> x        (e.g. single projection at the end)

Method ① (slide 10): ``ProjectionInjector(cset, when="every")``

    x_{k+1} = Π_C( x_k + Δt · v_θ(x_k, t_k) ),   Π_C(y) = argmin_{z∈C} ||z − y||²

``when="final"`` applies Π_C once after the last step instead (the 'Final P'
ablation from the preliminary study).  Π_C is evaluated in raw units (metres);
the wrapper handles the normalised <-> raw conversion and keeps the t=0
observation row fixed.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from .sets import ConstraintSet, total_violation


class Injector:
    """No-op base: every hook is identity.  Subclass and override what you need."""

    name = "none"

    def reset(self, n_steps):
        pass

    def before_step(self, x, k, t):
        return x

    def velocity(self, v, x, k, t):
        return v

    def after_step(self, x, k, t):
        return x

    def finalize(self, x):
        return x

    def stats(self):
        return {}

    def describe(self):
        return {"name": self.name}


class RawSpace:
    """Converts between the model's normalised plan and raw units."""

    def __init__(self, center, scale, act_dim):
        self.center = center
        self.scale = scale
        self.act_dim = act_dim

    def to_raw(self, x):
        return x * self.scale + self.center

    def to_norm(self, x_raw):
        return (x_raw - self.center) / self.scale


class ProjectionInjector(Injector):
    """Method ①: snap the sample back onto C after every Euler step (or once at the end)."""

    def __init__(self, cset: ConstraintSet, space: RawSpace, when="every", skip_first=0):
        assert when in ("every", "final")
        self.cset = cset
        self.space = space
        self.when = when
        self.skip_first = int(skip_first)   # optionally leave the first k noisy steps unprojected
        self.name = f"proj_{when}"
        self._t = 0.0
        self._n = 0
        self._pre = []

    def reset(self, n_steps):
        self.n_steps = n_steps
        self._pre = []

    def _project(self, x):
        t0 = time.perf_counter()
        x_raw = self.space.to_raw(x)
        self._pre.append(total_violation(self.cset, x_raw))
        x = self.space.to_norm(self.cset.project(x_raw))
        self._t += time.perf_counter() - t0
        self._n += 1
        return x

    def after_step(self, x, k, t):
        if self.when == "every" and k >= self.skip_first:
            return self._project(x)
        return x

    def finalize(self, x):
        if self.when == "final":
            return self._project(x)
        return x

    def stats(self):
        out = {"proj_calls": self._n, "proj_ms_per_call": 1000.0 * self._t / max(self._n, 1)}
        if self._pre:   # violation seen *before* the last projection of this plan
            out.update({f"pre_{k}": v for k, v in self._pre[-1].items()})
        return out

    def describe(self):
        return {"name": self.name, "when": self.when, "skip_first": self.skip_first, "set": self.cset.describe()}


@torch.no_grad()
def euler_solve(model, x, cond, act_dim, t_span, apply_conditioning, injector: Injector | None = None):
    """Same loop as HardFlow ``ConditionedODESolver.forward`` (euler, value=None) + hooks."""
    inj = injector or Injector()
    assert len(t_span) > 1
    dt = t_span[1] - t_span[0]
    inj.reset(len(t_span))
    for k, t in enumerate(t_span):
        x = inj.before_step(x, k, t)
        v = model(x, t)
        for tc in cond:                               # == apply_conditioning_from_conditioned_x(v, zeros, cond, act_dim)
            v[:, tc, act_dim:] = 0.0
        v = inj.velocity(v, x, k, t)
        x = x + v * dt
        x = inj.after_step(x, k, t)
        x = apply_conditioning(x, cond, act_dim)      # keep the t=0 observation fixed after any edit
        x = x.detach()
    x = inj.finalize(x)
    x = apply_conditioning(x, cond, act_dim)
    return x


class ConstrainedPolicy:
    """Drop-in replacement for ``eval_pushing.Policy`` with constraint injection.

    ``injector``  hook-based method: (1) ProjectionInjector
    ``sampler``   optional replacement of the Euler loop (unused here)

    ``plan(obs)`` has the same signature and returns the same (H, 2 + obs_dim)
    raw-unit array.  With neither attached it reproduces the baseline exactly.
    """

    def __init__(self, policy, injector: Injector | None = None, sampler=None):
        self.base = policy
        self.injector = injector or Injector()
        self.sampler = sampler
        self.last_stats = {}
        # mirror the attributes eval scripts might touch
        self.model, self.norm, self.device = policy.model, policy.norm, policy.device
        self.obs_dim, self.act_dim, self.horizon = policy.obs_dim, policy.act_dim, policy.horizon

    @torch.no_grad()
    def plan(self, obs):
        p = self.base
        o = torch.from_numpy(np.asarray(obs, dtype=np.float32)).to(p.device)[None]
        o_n = (o - p.center[p.act_dim:]) / p.scale[p.act_dim:]
        cond = {0: o_n}
        x = torch.randn(1, p.horizon, p.obs_dim + p.act_dim, device=p.device)
        x = p.apply_conditioning(x, cond, p.act_dim)
        if self.sampler is None:
            x = euler_solve(p.model, x, cond, p.act_dim, p.t_span, p.apply_conditioning, self.injector)
        else:
            x = self.sampler.solve(p.model, x, cond, p.act_dim, p.apply_conditioning, self.injector, len(p.t_span))
        x = x * p.scale + p.center
        self.last_stats = {**self.injector.stats(), **(self.sampler.stats() if self.sampler else {})}
        return x[0].cpu().numpy()

    def describe(self):
        return {"sampler": self.sampler.describe() if self.sampler else "euler", "inject": self.injector.describe()}
