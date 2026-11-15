"""Stacking 관절 공간 ①②③④⑤ 조합 정책 (32조합 본실험용).

팀원의 ``joint.JointPolicy``(①②③⑤를 한꺼번에 켜고 끔)를 방법별로 켤 수 있게 일반화하고 ④ RTC를 더했다.
제약·PF·LiPo·QP 구현은 ``fm_constraint.joint``의 것을 그대로 쓰고 수정하지 않는다.

    ① projection  매 Euler 뒤 7관절 목표 증분 16개를 관절 위치 범위·축별 가속도 상자에 투영 (실패 시 계획 거부)
    ② pf2mp       7D 관절 KDE potential guidance (생성된 미래 관절 관측 채널에만)
    ③ filter      실행 직전 H=1 같은 제약 투영. 수정량 > intervention_tolerance_rad면 남은 큐 폐기 (eval 루프에서)
    ④ rtc         직전 계획의 미실행 행동 suffix로 행동 채널 inpainting guidance (main ``rtc.guided_velocity``).
                  **그리퍼 채널은 target 가중치 0** (2026-11-15 결정). 지연 0이라 soft mask는 앞 H−s 구간만
    ⑤ lipo        누적 관절 위치 3차 차분 최소화, 보정 ≤ lipo_bound_rad. ①이 켜져 있을 때만 ① 제약을 같은 QP에 넣는다

규칙(main ``policy.py``와 같다):
- 계획 단계 방법(①②④⑤)이 하나라도 켜지면 latent·정규화·QP 반환을 float64로 다룬다(신경망 입력만 float32).
  모두 꺼지면 팀원 baseline arm(``JointPolicy(enabled=False)``)과 같은 float32 경로를 그대로 탄다.
- 최종 계획의 제약 위반으로 계획을 거부하는 것은 ①이 켜져 있을 때뿐이다. 꺼져 있으면 측정만 한다.
- 그리퍼(행동 index 7)는 어떤 QP·guidance도 직접 수정하지 않는다.
"""

import time

import numpy as np
import torch

from .joint import JointConstraints, joint_lipo, solve_qp
from .rtc import guided_velocity, soft_mask

METHODS = ('projection', 'pf2mp', 'filter', 'rtc', 'lipo')
BITS = dict(zip(METHODS, (16, 8, 4, 2, 1)))
PLAN_METHODS = ('projection', 'pf2mp', 'rtc', 'lipo')
GRIPPER = 7
H, ACT = 16, 8


def combo_flags(number):
    return {name: bool(number & bit) for name, bit in BITS.items()}


def rtc_target(base, suffix):
    """직전 계획의 미실행 행동 (H−s, 8) raw → 정규화 target (1,H,8)과 가중치 (1,H,8). 그리퍼 열 가중치는 0."""
    dtype = base.center.dtype
    execute = H - len(suffix)
    target = torch.zeros(1, H, ACT, dtype=dtype, device=base.device)
    target[:, :len(suffix)] = (torch.as_tensor(suffix, dtype=dtype, device=base.device)
                               - base.center[:ACT]) / base.scale[:ACT]
    w = torch.as_tensor(soft_mask(H, execute, 0), dtype=dtype, device=base.device)[None, :, None].repeat(1, 1, ACT)
    w[..., GRIPPER] = 0
    return target, w


class JointComboPolicy:

    def __init__(self, base, settings, potential, methods, rtc_beta=5.0, ode_steps=10):
        self.base = base
        self.settings = settings
        self.potential = potential
        self.methods = dict(methods)
        if set(self.methods) != set(METHODS):
            raise ValueError(f'methods must be exactly {METHODS}')
        if self.methods['pf2mp'] and potential is None:
            raise ValueError('pf2mp needs the joint PF cache')
        self.rtc_beta = rtc_beta
        self.ode_steps = ode_steps
        self.planning = any(self.methods[m] for m in PLAN_METHODS)
        self.last_report = {}

    @torch.no_grad()
    def plan(self, obs, previous, history, noise, rtc_suffix=None):
        """obs (23,) raw, previous (7,) 직전 실행 관절 증분, history 실행된 목표 관절 최근 ≤3개(마지막 = 현재 목표),
        noise (1,16,31) 정규화 공간, rtc_suffix (8,8) raw 또는 None. 성공하면 raw (16,31), 실패하면 None."""
        p, s, m = self.base, self.settings, self.methods
        started = time.perf_counter()
        use_rtc = m['rtc'] and rtc_suffix is not None
        r = dict(status='ok', failure_stage=None, qp=[], lipo=None, pf_calls=0, model_calls=0,
                 projection_calls=0, rtc_guided=use_rtc, latency_ms=0.0, stages={})
        self.last_report = r
        observation = torch.as_tensor(obs, dtype=p.center.dtype, device=p.device)[None]
        cond = {0: (observation - p.center[ACT:]) / p.scale[ACT:]}
        x = p.apply_conditioning(noise.clone().to(p.device), cond, ACT)
        center = p.center.double() if self.planning else p.center
        scale = p.scale.double() if self.planning else p.scale
        if self.planning:
            x = x.double()
            cond = {0: cond[0].double()}
        anchor = np.asarray(history, dtype=np.float64)[-1]
        con = JointConstraints(anchor, previous, H, s)
        r['constraint_anchor_rad'] = anchor.tolist()
        if use_rtc:
            suffix = np.asarray(rtc_suffix, dtype=np.float64)
            if suffix.shape != (H - 8, ACT) or not np.isfinite(suffix).all():
                r.update(status='invalid_input', failure_stage='rtc')
                r['latency_ms'] = 1000 * (time.perf_counter() - started)
                return None
            target, weight = rtc_target(p, suffix)
            params = [q for q in getattr(p.model, 'parameters', lambda: [])() if q.requires_grad]
            for q in params:
                q.requires_grad_(False)
        times = torch.linspace(0, 1, self.ode_steps + 1, device=p.device)[:-1]
        dt = times[1] - times[0]
        try:
            for k, t in enumerate(times):
                if use_rtc:
                    velocity = guided_velocity(p, x.to(p.center.dtype), t, target, weight,
                                               k / self.ode_steps, self.rtc_beta).to(x.dtype)
                else:
                    velocity = p.model(x.to(p.center.dtype), t).to(x.dtype)
                velocity[:, 0, ACT:] = 0
                r['model_calls'] += 1
                if m['pf2mp']:
                    guidance, _ = self.potential.guidance(x, center, scale)
                    r['pf_calls'] += 1
                    velocity = velocity + s['pf_weight'] * guidance
                x = x + dt * velocity
                if not torch.isfinite(x).all():
                    r.update(status='invalid_policy_output', failure_stage='predict')
                    x = None
                    break
                if m['projection']:
                    raw = (x * scale + center)[0]
                    gripper_before = raw[:, GRIPPER].clone()
                    solved = solve_qp(raw[:, :7].double().cpu().numpy().reshape(-1), con.G, con.d, s['solver'])
                    value = solved.pop('value')
                    r['qp'].append(dict(step=k, **solved))
                    r['projection_calls'] += 1
                    if value is None:
                        r.update(status=solved['status'], failure_stage='projection')
                        x = None
                        break
                    x = x.clone()
                    x[0, :, :7] = (torch.as_tensor(value.reshape(H, 7), dtype=x.dtype, device=p.device)
                                   - center[:7]) / scale[:7]
                    assert torch.equal((x * scale + center)[0, :, GRIPPER], gripper_before)
                    if con.residual((x * scale + center)[0, :, :7].double().cpu().numpy()) \
                            > s['solver']['feasibility_tolerance_m']:
                        r.update(status='constraint_violation', failure_stage='projection_residual')
                        x = None
                        break
                x = p.apply_conditioning(x, cond, ACT).detach()
        finally:
            if use_rtc:
                for q in params:
                    q.requires_grad_(True)
        if x is not None and m['lipo']:
            raw = (x * scale + center)[0].cpu().numpy()
            G, d = (con.G, con.d) if m['projection'] else (np.zeros((0, H * 7)), np.zeros(0))
            solved = joint_lipo(raw[:, :7], history, s['lipo_bound_rad'], s['solver'], G, d)
            value = solved.pop('value')
            r['lipo'] = solved
            if value is None:
                r.update(status=solved['status'], failure_stage='lipo')
                x = None
            else:
                grip = x[0, :, GRIPPER].clone()
                x = x.clone()
                x[0, :, :7] = (torch.as_tensor(value, dtype=x.dtype, device=p.device) - center[:7]) / scale[:7]
                assert torch.equal(grip, x[0, :, GRIPPER])
        result = None
        if x is not None:
            result = (x * scale + center)[0].cpu().numpy()
            residual = con.residual(result[:, :7])
            r['final_residual_rad'] = residual
            if not np.isfinite(result).all():
                r.update(status='invalid_policy_output', failure_stage='final')
                result = None
            elif m['projection'] and residual > s['solver']['feasibility_tolerance_m']:
                r.update(status='constraint_violation', failure_stage='final')
                result = None
        r['latency_ms'] = 1000 * (time.perf_counter() - started)
        return result
