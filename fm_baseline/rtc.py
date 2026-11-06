"""RTC Eq. (2)-(5) and deterministic, simulated-delay chunk execution.

The simulator is stepped serially: inference completes on the host immediately,
but its result becomes available after d control steps. This is NOT a wall-clock
asynchronous controller. Only action channels are inpainted; current observations
remain conditioned. The full action/observation Jacobian is retained.
"""
import math
from time import perf_counter
import numpy as np


def soft_mask(horizon, execute_steps, delay):
    if not all(isinstance(v, int) and not isinstance(v, bool)
               for v in (horizon, execute_steps, delay)):
        raise ValueError('H, s and d must be integers')
    if not (0 <= delay <= execute_steps <= horizon - delay) or execute_steps < 1:
        raise ValueError('RTC requires 0 <= d <= s <= H-d and s >= 1')
    i = np.arange(horizon)
    c = (horizon - execute_steps - i) / (horizon - execute_steps - delay + 1)
    w = np.zeros(horizon, dtype=np.float64)
    middle = (i >= delay) & (i < horizon - execute_steps)
    w[middle] = c[middle] * np.expm1(c[middle]) / np.expm1(1.)
    w[i < delay] = 1.
    return w


def guided_plan(policy, obs, previous_actions, execute_steps, delay, beta=5., generator=None):
    """Generate a raw-unit plan using action-only RTC guidance (one VJP/step)."""
    import torch
    if not math.isfinite(beta) or beta < 0:
        raise ValueError('RTC beta must be finite and nonnegative')
    weights = soft_mask(policy.horizon, execute_steps, delay)
    previous_actions = np.asarray(previous_actions)
    if previous_actions.shape != (policy.horizon - execute_steps, policy.act_dim):
        raise ValueError('Previous actions must be the aligned unexecuted H-s suffix')
    if not np.isfinite(previous_actions).all():
        raise ValueError('Non-finite RTC target')
    if beta == 0 or not np.any(weights):
        return policy.plan(obs, generator=generator)
    p = policy
    # Freeze parameter gradients, not gradients with respect to the noisy input.
    params = list(p.model.parameters())
    requires_grad = [v.requires_grad for v in params]
    try:
        for v in params:
            v.requires_grad_(False)
        with torch.enable_grad():
            dtype = p.center.dtype
            observation = torch.as_tensor(obs, dtype=dtype, device=p.device)[None]
            cond = {0: (observation - p.center[p.act_dim:]) / p.scale[p.act_dim:]}
            target = torch.zeros(1, p.horizon, p.act_dim, dtype=dtype, device=p.device)
            target[:, :len(previous_actions)] = (
                torch.as_tensor(previous_actions, dtype=dtype, device=p.device)
                - p.center[:p.act_dim]) / p.scale[:p.act_dim]
            w = torch.as_tensor(weights, dtype=dtype, device=p.device)[None, :, None]
            x = torch.randn(1, p.horizon, p.obs_dim + p.act_dim,
                            device=p.device, dtype=dtype, generator=generator)
            x = p.apply_conditioning(x, cond, p.act_dim)
            n = len(p.t_span)
            if n < 1:
                raise ValueError('At least one Euler step required')
            for k in range(n):
                tau = k / n
                x = x.detach().requires_grad_(True)
                t = torch.as_tensor(tau, dtype=dtype, device=p.device)
                v = p.model(x, t)
                free = torch.ones_like(v)
                free[:, 0, p.act_dim:] = 0
                v = v * free
                clean_actions = (x + (1 - tau) * v)[..., :p.act_dim]
                error = ((target - clean_actions) * w).detach()
                g = torch.autograd.grad(clean_actions, x, grad_outputs=error)[0]
                # Handle tau=0 explicitly instead of evaluating infinity/NaN.
                strength = beta if k == 0 else min(beta, (tau*tau + (1-tau)**2) / (tau*(1-tau)))
                x = (x + (v + strength * g * free) / n).detach()
                x = p.apply_conditioning(x, cond, p.act_dim)
            result = (x * p.scale + p.center)[0].detach().cpu().numpy()
            if not np.isfinite(result).all():
                raise RuntimeError('RTC generated non-finite values')
            return result
    finally:
        for v, flag in zip(params, requires_grad):
            v.requires_grad_(flag)


class ChunkController:
    """Per-episode scheduler; inference launch every s steps, activation d later.

    Bootstrap inference occurs before the first control tick. At subsequent
    launches the old plan's first d remaining actions are committed. QP sees
    these exact actions as a fixed prefix and never rewrites them.
    """
    def __init__(self, policy, processor, execute_steps, delay=0, rtc=False, beta=5., fixed_rng=None):
        soft_mask(policy.horizon, execute_steps, delay)
        if not math.isfinite(beta) or beta < 0:
            raise ValueError('RTC beta must be finite and nonnegative')
        if delay and processor.mode in ('blend', 'lipo'):
            raise ValueError('Delayed evaluation supports none/qp, not legacy blending')
        if rtc and processor.mode in ('blend', 'lipo'):
            raise ValueError('RTC uses none/qp; legacy blending would duplicate stitching')
        self.policy, self.processor = policy, processor
        self.s, self.d, self.rtc, self.beta = execute_steps, delay, rtc, beta
        self.fixed_rng = fixed_rng
        self.tick = 0
        self.active = self.pending = None
        self.plan_time = 0.
        self.n_plans = 0

    def next_action(self, obs, history, generator=None):
        seam = False
        if self.pending is not None and self.tick == self.pending[0]:
            _, self.active, self.origin = self.pending
            self.pending = None
            seam = True
        if self.active is None or self.tick % self.s == 0:
            bootstrap = self.active is None
            old = None if bootstrap else self.active[self.tick - self.origin:].copy()
            if self.fixed_rng is not None:
                generator.set_state(self.fixed_rng)
            started = perf_counter()
            if self.rtc and not bootstrap:
                plan = guided_plan(self.policy, obs, old, self.s, self.d, self.beta, generator)
            else:
                plan = self.policy.plan(obs, generator=generator)
            self.plan_time += perf_counter() - started
            self.n_plans += 1
            actions = np.asarray(plan[:, :2], dtype=np.float64).copy()
            frozen = 0 if bootstrap else self.d
            if frozen:
                actions[:frozen] = old[:frozen]
            output = self.processor.process(actions, history, frozen_prefix=frozen)
            self.processor.diagnostics[-1].update(
                launch_step=self.tick, activation_step=self.tick + frozen,
                rtc_guided=bool(self.rtc and not bootstrap), frozen_prefix=frozen)
            if bootstrap or self.d == 0:
                self.active, self.origin = output, self.tick
                seam = True
            else:
                self.pending = (self.tick + self.d, output, self.tick)
        index = self.tick - self.origin
        if index >= len(self.active):
            raise RuntimeError('Action buffer underrun')
        action = self.active[index].copy()
        self.tick += 1
        return action, seam
