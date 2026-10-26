"""L2C2-inspired FM adaptation, not the paper's Hellinger-distance RL loss.

Kobayashi (2022), arXiv:2202.07152v1, equations 12, 14, 15.
The policy metric is replaced with a shared-noise action-chunk squared distance.
All inputs/outputs here are in baseline-normalized coordinates.
"""
from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class L2C2Config:
    weight: float = 0.01
    sigma: float = 1.0
    epsilon: float = 0.1
    batch_size: int = 4
    ode_steps: int = 10
    action_steps: int = 8

    def validate(self, horizon):
        for name in ("weight", "sigma", "epsilon"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.weight < 0 or self.sigma <= 0 or self.epsilon <= 0:
            raise ValueError("weight >= 0, sigma > 0 and epsilon > 0 required")
        if self.sigma + (self.sigma - 1) * self.epsilon <= 0:
            raise ValueError("Equation 15 requires a positive sampling half-width")
        if self.batch_size < 1 or self.ode_steps < 2:
            raise ValueError("batch_size >= 1 and ode_steps >= 2 required")
        if horizon < 2 or not 1 <= self.action_steps <= horizon:
            raise ValueError("Need a transition and 1 <= action_steps <= horizon")


def sample_neighbor(state, next_state, sigma, epsilon, generator=None):
    """Eq. 15 box and Eq. 14 relative L-inf distance.

    Constant coordinates contribute zero, avoiding 0/0. Use the sampled u
    directly on moving coordinates to avoid division by tiny state changes.
    No clipping/renormalization: doing so would change the paper's box.
    """
    half_width = sigma + (sigma - 1.0) * epsilon
    u = (2 * torch.rand(state.shape, dtype=state.dtype, device=state.device,
                        generator=generator) - 1) * half_width
    delta = next_state - state
    offset = delta * u
    relative = torch.where(delta != 0, u.abs(), torch.zeros_like(u))
    distance = relative.amax(dim=-1) + epsilon
    return state + offset, distance


def conditioned_plan(model, obs, noise, act_dim, ode_steps):
    """Differentiable counterpart of baseline ConditionedODESolver Euler.

    Replaces only t=0 observations; neither the state nor either policy branch
    is detached. The caller supplies identical latent noise to both branches.
    """
    mask = torch.zeros_like(noise)
    mask[:, 0, act_dim:] = 1
    condition = torch.zeros_like(noise)
    condition[:, 0, act_dim:] = obs
    x = noise * (1 - mask) + condition
    times = torch.linspace(0, 1, ode_steps + 1, device=noise.device)[:-1]
    dt = times[1] - times[0]
    for t in times:
        velocity = model(x, t) * (1 - mask)
        x = (x + dt * velocity) * (1 - mask) + condition
    return x


def policy_regularization(model, windows, act_dim, config, seed):
    """Return unweighted loss and diagnostics; preserve the FM RNG stream.

    Each window starts within one demonstration, so rows 0/1 are a valid
    transition without wrapping into another episode. Sample a small random
    subset of the current training batch, with deterministic step-local RNG.
    """
    config.validate(windows.shape[1])
    if config.weight == 0:
        return windows.new_zeros(()), {"action_distance": 0., "relative_distance": 0.}
    if any(isinstance(m, torch.nn.modules.dropout._DropoutNd) and m.p > 0
           for m in model.modules()):
        raise ValueError("Stochastic dropout would confound shared-noise policy comparison")
    devices = [windows.device.index] if windows.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        # Seed only the active device, leaving all unrelated CUDA generators alone.
        torch.random.default_generator.manual_seed(seed)
        if windows.is_cuda:
            with torch.cuda.device(windows.device):
                torch.cuda.manual_seed(seed)
        n = min(config.batch_size, len(windows))
        indices = torch.randperm(len(windows), device=windows.device)[:n]
        samples = windows[indices].detach()
        state, next_state = samples[:, 0, act_dim:], samples[:, 1, act_dim:]
        neighbor, distance = sample_neighbor(state, next_state, config.sigma, config.epsilon)
        noise = torch.randn_like(samples)
        # One batched call couples the two branches through the same latent z.
        plans = conditioned_plan(model, torch.cat([state, neighbor]),
                                 torch.cat([noise, noise]), act_dim, config.ode_steps)
        actions = plans[:, :config.action_steps, :act_dim]
        left, right = actions.chunk(2)
        # Half squared Euclidean action distance, averaged over executed steps.
        action_distance = 0.5 * (left - right).square().sum(-1).mean(-1)
        loss = (action_distance / distance).mean()
    return loss, {"action_distance": float(action_distance.detach().mean()),
                  "relative_distance": float(distance.detach().mean())}
