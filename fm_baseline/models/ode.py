"""현재 실험의 조건부 Euler 적분. 출처는 THIRD_PARTY_NOTICES.md 참조."""

import torch
from torch import nn

from .flow_matching import apply_conditioning_from_conditioned_x


class ConditionedODESolver(nn.Module):
    """첫 관측의 벡터장을 0으로 유지하며 기존 순서대로 Euler 적분한다."""

    def __init__(self, model, conditions, action_dim, value=None, ode_method="euler"):
        super().__init__()
        if ode_method != "euler" or value is not None:
            raise ValueError("Only unconditional-value Euler sampling is supported")
        self.model = model
        self.conditions = conditions
        self.action_dim = action_dim

    def forward(self, x, t_span, return_chain=False):
        """기존 다중 스텝 샘플러의 보폭·detach·chain 순서를 유지한다."""
        if len(t_span) < 2:
            raise ValueError("t_span needs at least two evenly spaced integration times")
        x0 = x.clone()
        dt = t_span[1] - t_span[0]
        x_chain = [x0]
        for t in t_span:
            dx_dt = self.model(x, t)
            dx_dt = apply_conditioning_from_conditioned_x(
                dx_dt, torch.zeros_like(x), self.conditions, self.action_dim
            )
            x = x + dx_dt * dt
            x = x.detach()
            x_chain.append(x)
        if return_chain:
            return x, torch.stack(x_chain, dim=1).detach()
        return x
