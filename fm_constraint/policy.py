"""baseline 모델을 재사용하는 Predict → Projection → VLSA → PC 정책."""

from time import perf_counter

import numpy as np
import torch

from .constraints import PlanConstraints
from .qp import solve_qp


class Policy:
    """baseline Policy를 감싸고, 활성 보정만 명시적으로 순서대로 적용한다.

    plan(obs, previous_action)는 성공 시 raw (H,D) 배열, 실패 시 None을
    반환한다. 마지막 실행의 상태·기하·중간 행동·비용은 last_report에 남긴다.
    이전 행동은 호출자가 실제 실행 직후 갱신하며 정책 내부에 숨겨두지 않는다.
    """

    def __init__(self, baseline_policy, config):
        self.base = baseline_policy
        self.config = config.values if hasattr(config, 'values') and not isinstance(config, dict) else config
        self.methods = self.config['methods']
        self.enabled = any(value['enabled'] for value in self.methods.values())
        self.last_report = {}

    def _project(self, x, constraints, label):
        """행동만 raw 단위에서 공동 투영하고, 모델 dtype 복귀 후 잔차를 확인한다."""
        p = self.base
        raw = (x * p.scale + p.center)[0].detach().cpu().numpy()
        solved = solve_qp(raw[:, :2].ravel(), constraints.G, constraints.d, self.config['solver'])
        self.last_report['qp_calls'] += 1
        self.last_report['qp_ms'] += solved['latency_ms']
        value = solved.pop('value')
        self.last_report['qp'].append({'stage': label, **solved})
        if solved['status'] != 'ok':
            self.last_report.update(status=solved['status'], failure_stage=label)
            return None
        # 위치에는 center가 필요하다. 수정하지 않는 관측은 재정규화하지 않아 반올림 오차를 피한다.
        actions = torch.as_tensor(value.reshape(1, -1, 2), device=x.device, dtype=x.dtype)
        x = x.clone()
        x[:, :, :2] = (actions - p.center[:2]) / p.scale[:2]
        restored = (x * p.scale + p.center)[0, :, :2].cpu().numpy()
        residual = constraints.violation(restored)['qp_max_m']
        if residual > self.config['solver']['feasibility_tolerance_m']:
            self.last_report.update(status='constraint_violation', failure_stage=label,
                                    dtype_residual_m=residual)
            return None
        return x

    def _record(self, label, x, constraints):
        """단계별 실행 대상 행동과 위반량을 저장한다. 모델을 호출하지 않는다."""
        raw = (x * self.base.scale + self.base.center)[0].detach().cpu().numpy()
        self.last_report['stages'][label] = {'actions_m': raw[:, :2].tolist(),
                                           **constraints.violation(raw[:, :2])}

    @torch.no_grad()
    def plan(self, obs, previous_action, noise=None):
        """현재 관측과 직전 실제 행동으로 전체 계획을 생성한다. noise는 비교용이다."""
        started = perf_counter()
        p, v = self.base, self.config
        self.last_report = {'status': 'ok', 'failure_stage': None, 'model_calls': 0,
                            'qp_calls': 0, 'qp_ms': 0., 'qp': [], 'stages': {},
                            'latency_ms': 0., 'stage_ms': {}, 'methods': {k: s['enabled'] for k, s in self.methods.items()}}
        try:
            constraints = PlanConstraints(obs, previous_action, p.horizon, v['constraints'], v['control']['dt_s'])
        except (ValueError, TypeError) as error:
            self.last_report.update(status='invalid_input', error=str(error), failure_stage='input')
            self.last_report['latency_ms'] = 1000 * (perf_counter() - started)
            return None
        self.last_report['geometry'] = constraints.geometry()
        if self.enabled and constraints.low is not None and (
                np.any(constraints.q0 < constraints.low - v['solver']['feasibility_tolerance_m'])
                or np.any(constraints.q0 > constraints.high + v['solver']['feasibility_tolerance_m'])):
            self.last_report.update(status='initial_outside_workspace', failure_stage='input')
            self.last_report['latency_ms'] = 1000 * (perf_counter() - started)
            return None
        # 모든 방법 off이면 baseline에 위임한다. 추가 noise를 뽑거나 제약으로 행동을 거부하지 않는다.
        if not self.enabled and noise is None and v['sampling']['predict_steps'] > 1:
            result = p.plan(obs)
            if not np.isfinite(result).all():
                self.last_report.update(status='invalid_policy_output', failure_stage='predict')
                self.last_report['latency_ms'] = 1000 * (perf_counter() - started)
                return None
            self.last_report['model_calls'] = v['sampling']['predict_steps']
            self.last_report['stages']['predict'] = {'actions_m': result[:, :2].tolist(),
                                                    **constraints.violation(result[:, :2])}
            self.last_report['latency_ms'] = 1000 * (perf_counter() - started)
            self.last_report['stage_ms']['predict'] = self.last_report['latency_ms']
            return result

        observation = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=p.device)[None]
        cond = {0: (observation - p.center[2:]) / p.scale[2:]}
        shape = (1, p.horizon, p.obs_dim + p.act_dim)
        if noise is None:
            x = torch.randn(*shape, device=p.device)
        else:
            x = torch.as_tensor(noise, device=p.device, dtype=p.center.dtype).clone()
            if x.shape != shape or not torch.isfinite(x).all():
                self.last_report.update(status='invalid_input', failure_stage='noise')
                self.last_report['latency_ms'] = 1000 * (perf_counter() - started)
                return None
        x = p.apply_conditioning(x, cond, p.act_dim)
        proj = self.methods['projection']
        steps = v['sampling']['predict_steps']
        times = torch.linspace(0., 1., steps + 1, device=p.device)[:-1]
        dt = times[1] - times[0] if steps > 1 else 1.0

        # 1. Predict: Euler 직후에만 선택적으로 Projection을 적용한다.
        # 첫 관측 벡터장을 0으로 하고 매 갱신 후 실제 관측 조건을 복원한다.
        stage_started = perf_counter()
        for k, t in enumerate(times):
            velocity = p.model(x, t)
            self.last_report['model_calls'] += 1
            velocity[:, 0, p.act_dim:] = 0.
            x = x + velocity * dt
            if proj['enabled'] and proj['mode'] == 'every' and k >= proj['skip_first']:
                x = self._project(x, constraints, 'projection')
                if x is None:
                    break
            x = p.apply_conditioning(x, cond, p.act_dim).detach()
        if x is not None and proj['enabled'] and proj['mode'] == 'final':
            x = self._project(x, constraints, 'projection_final')
        self.last_report['stage_ms']['predict'] = 1000 * (perf_counter() - stage_started)
        if x is not None:
            self._record('predict', x, constraints)

        # 2. VLSA: terminal 계획 전체를 모델 호출 없이 한 번 공동 보정한다.
        if x is not None and self.methods['vlsa']['enabled']:
            stage_started = perf_counter()
            x = self._project(x, constraints, 'vlsa')
            self.last_report['stage_ms']['vlsa'] = 1000 * (perf_counter() - stage_started)
            if x is not None:
                self._record('vlsa', x, constraints)

        # 3. PC Correct: 앞 단계 계획에서 시간만 다시 0으로 시작한다.
        # 새 noise나 두 번째 Predict를 만들지 않는다. 기하와 이전 행동도 고정이다.
        pc = self.methods['pc']
        if x is not None and pc['enabled']:
            stage_started = perf_counter()
            count = pc['correct_steps']
            dt_c = 1.0 / count
            b_star = pc['delta'] / (1 - pc['rho'])
            eps = (0.9 / dt_c * b_star / (b_star - pc['delta'])**pc['rho']
                   if pc['eps'] == 'auto' else pc['eps'])
            self.last_report['pc_eps'] = eps
            for t in torch.linspace(0., 1., count + 1, device=p.device)[:-1]:
                velocity = p.model(x, t) * (pc['alpha'] * (1 - float(t)))
                self.last_report['model_calls'] += 1
                velocity[:, 0, p.act_dim:] = 0.
                raw_actions = (x * p.scale + p.center)[0, :, :2].cpu().numpy().astype(np.float64)
                barrier = (constraints.d - constraints.G @ raw_actions.ravel()) / constraints.scales
                z = barrier - pc['delta']
                bound = constraints.scales * eps * np.sign(z) * np.abs(z)**pc['rho']
                # 벡터장은 dA/dτ이므로 scale만 곱한다. center를 더하면 미분이 아니다.
                raw_velocity = (velocity[0, :, :2].double() * p.scale[:2].double()).cpu().numpy()
                solved = solve_qp(raw_velocity.ravel(), constraints.G, bound, v['solver'])
                self.last_report['qp_calls'] += 1
                self.last_report['qp_ms'] += solved['latency_ms']
                value = solved.pop('value')
                self.last_report['qp'].append({'stage': 'pc', **solved})
                if solved['status'] != 'ok':
                    self.last_report.update(status=solved['status'], failure_stage='pc')
                    x = None
                    break
                corrected = torch.as_tensor(value.reshape(1, -1, 2), device=x.device, dtype=x.dtype)
                velocity[:, :, :2] = corrected / p.scale[:2]
                x = p.apply_conditioning(x + velocity * dt_c, cond, p.act_dim).detach()
            self.last_report['stage_ms']['pc'] = 1000 * (perf_counter() - stage_started)
            if x is not None:
                self._record('pc', x, constraints)

        result = None
        if x is not None:
            x = p.apply_conditioning(x, cond, p.act_dim)
            result = (x * p.scale + p.center)[0].cpu().numpy()
            if not np.isfinite(result).all():
                self.last_report.update(status='invalid_policy_output', failure_stage='final')
                result = None
            else:
                final = constraints.violation(result[:, :2])
                self.last_report['final'] = final
                # PC 뒤에 투영을 숨겨 넣지 않는다. 활성 방법의 실패를 그대로 보고한다.
                if self.enabled and final['qp_max_m'] > v['solver']['feasibility_tolerance_m']:
                    self.last_report.update(status='constraint_violation', failure_stage='final')
                    result = None
        self.last_report['latency_ms'] = 1000 * (perf_counter() - started)
        return result
