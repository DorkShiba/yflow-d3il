"""RTC equations (2)--(5), arXiv:2506.07339v2.

The baseline jointly generates actions and observations. Match only normalized
action channels to the previous chunk; allow the VJP to change free latent
coordinates, but always keep the current observation fixed. No weight updates.
"""
import math
import numpy as np
import torch


def soft_mask(horizon, executed, delay, *, device=None, dtype=torch.float32):
    """Paper Eq. (5); executed counts steps since the previous inference START."""
    if not 0 <= delay <= executed <= horizon - delay or executed < 1:
        raise ValueError("Require 0 <= delay <= executed <= horizon-delay and executed >= 1")
    overlap = horizon - executed
    i = torch.arange(horizon, device=device, dtype=dtype)
    c = (overlap - i) / (overlap - delay + 1)
    w = torch.zeros_like(i)
    mid = (i >= delay) & (i < overlap)
    w[mid] = c[mid] * torch.expm1(c[mid]) / math.expm1(1.0)
    w[i < delay] = 1
    return w


def guidance_weight(tau, beta):
    """Clipped coefficient; take the finite clipped limit at tau=0."""
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and nonnegative")
    if not 0 <= tau < 1:
        raise ValueError("tau must be in [0, 1)")
    if tau == 0:
        return beta
    return min(beta, (tau * tau + (1 - tau) ** 2) / (tau * (1 - tau)))


class RTCPolicy:
    def __init__(self, model, norm, cfg, ode_steps=10, device="cpu", beta=5.0):
        if ode_steps < 2:
            raise ValueError("Use at least two Euler steps, as in the baseline solver")
        guidance_weight(0, beta)
        self.model = model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.horizon, self.act_dim = cfg["horizon"], cfg["act_dim"]
        self.obs_dim, self.ode_steps = cfg["obs_dim"], ode_steps
        self.device, self.beta = device, beta
        self.center = torch.as_tensor(norm.center, dtype=torch.float32, device=device)
        self.scale = torch.as_tensor(norm.scale, dtype=torch.float32, device=device)
        if self.center.shape != (self.act_dim + self.obs_dim,) or self.scale.shape != self.center.shape:
            raise ValueError("Normalizer dimensions do not match the checkpoint")
        if not torch.isfinite(self.scale).all() or not (self.scale > 0).all():
            raise ValueError("Normalizer scale must be positive and finite")
        self.times = torch.linspace(0, 1, ode_steps + 1, device=device)[:-1]
        self.previous = None

    def reset(self):
        self.previous = None

    def plan(self, obs, *, executed=8, delay=0, rtc=True, generator=None, noise=None):
        w = soft_mask(self.horizon, executed, delay, device=self.device)
        obs = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device)
        if obs.shape != (self.obs_dim,) or not torch.isfinite(obs).all():
            raise ValueError("Invalid observation")
        obs_n = (obs - self.center[self.act_dim:]) / self.scale[self.act_dim:]
        shape = (1, self.horizon, self.act_dim + self.obs_dim)
        if noise is None:
            x = torch.randn(shape, device=self.device, generator=generator)
        else:
            x = noise.to(device=self.device, dtype=torch.float32).clone()
            if tuple(x.shape) != shape:
                raise ValueError("Invalid noise shape")
        x[:, 0, self.act_dim:] = obs_n
        # Multiplication, rather than an in-place edit, preserves the VJP graph.
        free = torch.ones_like(x)
        free[:, 0, self.act_dim:] = 0
        guided = rtc and self.previous is not None and self.beta > 0 and bool(w.any())
        if guided:
            target = torch.zeros_like(x)
            target[:, :self.horizon-executed, :self.act_dim] = self.previous[:, executed:, :self.act_dim]
            mask = torch.zeros_like(x)
            mask[:, :, :self.act_dim] = w[None, :, None]
        dt = self.times[1] - self.times[0]
        for t in self.times:
            if guided:
                # Re-enable gradients even when called inside a no_grad evaluation loop.
                with torch.enable_grad():
                    x = x.detach().requires_grad_(True)
                    v = self.model(x, t) * free
                    final_estimate = x + (1 - t) * v
                    residual = ((target - final_estimate) * mask).detach()
                    g, = torch.autograd.grad(final_estimate, x, grad_outputs=residual)
                    velocity = v.detach() + guidance_weight(float(t), self.beta) * g.detach() * free
            else:
                with torch.no_grad():
                    velocity = self.model(x, t) * free
            x = (x.detach() + dt * velocity).detach()
            x[:, 0, self.act_dim:] = obs_n
        if not torch.isfinite(x).all():
            raise FloatingPointError("RTC generated non-finite values")
        self.previous = x.clone()
        return (x[0] * self.scale + self.center).cpu().numpy()
