"""④ RTC (Real-Time Chunking, Eq. 2–5)의 soft mask와 행동 채널 guidance.

codex/rtc-qp-method123의 fm_baseline/rtc.py에서 가져왔다. 시뮬레이터는 직렬로
진행하며 추론 지연은 OOD 조건 E의 d step으로만 모사한다(벽시계 비동기 아님).
행동 채널만 이전 계획의 미실행 suffix로 inpainting하고, 현재 관측 조건은 유지한다.
"""
import numpy as np
import torch


def soft_mask(horizon, execute_steps, delay):
    """W_i: 앞 d개는 1(이미 실행이 확정), 그 뒤 H−s까지 지수 감소, 이후 0."""
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in (horizon, execute_steps, delay)):
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


def rtc_target(p, previous_actions, weights):
    """이전 계획의 미실행 행동 suffix (raw, (H−s, act_dim))를 정규화 공간 target으로."""
    dtype = p.center.dtype
    target = torch.zeros(1, p.horizon, p.act_dim, dtype=dtype, device=p.device)
    target[:, :len(previous_actions)] = (
        torch.as_tensor(previous_actions, dtype=dtype, device=p.device)
        - p.center[:p.act_dim]) / p.scale[:p.act_dim]
    w = torch.as_tensor(weights, dtype=dtype, device=p.device)[None, :, None]
    return target, w


def guided_velocity(p, x, t, target, w, tau, beta):
    """RTC로 보정한 FM 벡터장. 스텝마다 입력에 대한 VJP 한 번."""
    with torch.enable_grad():
        x = x.detach().requires_grad_(True)
        v = p.model(x, t)
        free = torch.ones_like(v)
        free[:, 0, p.act_dim:] = 0
        v = v * free
        clean_actions = (x + (1 - tau) * v)[..., :p.act_dim]
        error = ((target - clean_actions) * w).detach()
        g = torch.autograd.grad(clean_actions, x, grad_outputs=error)[0]
        # tau=0은 무한대/NaN 대신 beta로 처리한다.
        strength = beta if tau == 0 else min(beta, (tau * tau + (1 - tau) ** 2) / (tau * (1 - tau)))
        return (v + strength * g * free).detach()
