"""baseline 모델을 재사용하는 Predict[Projection] → PC → VLSA 정책."""

from time import perf_counter

import numpy as np
import torch

from .constraints import PlanConstraints
from .qp import solve_qp


# 설정 키의 나열 순서와 무관한 고정 실행 순서다. 과거 결과와의 비교에도 저장한다.
PIPELINE_ORDER = ('projection', 'pc', 'vlsa')


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
        if not np.isfinite(restored).all():
            # dtype 변환 중 overflow가 나면 잔차도 NaN/Inf가 될 수 있으므로
            # 위반 계산·JSON 기록 전에 유효하지 않은 출력으로 종료한다.
            self.last_report.update(status='invalid_policy_output', failure_stage=label)
            return None
        residual = constraints.violation(restored)['qp_max_m']
        if residual > self.config['solver']['feasibility_tolerance_m']:
            self.last_report.update(status='constraint_violation', failure_stage=label,
                                    dtype_residual_m=residual)
            return None
        return x

    def _record(self, label, x, constraints):
        """단계별 실행 대상 행동과 위반량을 저장한다. 모델을 호출하지 않는다."""
        raw = (x * self.base.scale + self.base.center)[0].detach().cpu().numpy()
        stages = self.last_report['stages']
        report = {'actions_m': raw[:, :2].tolist(), **constraints.violation(raw[:, :2])}
        if stages:
            # PC와 VLSA의 수정량은 바로 앞의 활성 단계 종료 행동에 대해 측정한다.
            # VLSA on/off와 관계없이 PC 입력은 항상 Predict 종료 계획이다.
            previous = next(reversed(stages.values()))['actions_m']
            report['intervention_norm_m'] = float(np.linalg.norm(raw[:, :2] - np.asarray(previous)))
        stages[label] = report

    @torch.no_grad()
    def plan(self, obs, previous_action, noise=None):
        """현재 관측과 직전 실행 행동으로 전체 계획을 생성한다.

        obs는 raw (obs_dim,), previous_action은 실제 실행한 XY 증분 (2,), 단위 m다.
        비교용 noise는 정규화 공간의 (1,H,act_dim+obs_dim)이며 내부에서 복사한다.
        성공하면 raw (H,act_dim+obs_dim)을 반환한다. 실패하면 None을 반환하고
        status·failure_stage·이미 끝난 단계의 행동과 비용을 last_report에 남긴다.
        """
        started = perf_counter()
        p, v = self.base, self.config
        self.last_report = {'status': 'ok', 'failure_stage': None, 'model_calls': 0,
                            'pipeline_order': list(PIPELINE_ORDER),
                            'qp_calls': 0, 'qp_ms': 0., 'qp': [], 'stages': {},
                            'latency_ms': 0., 'stage_ms': {}, 'methods': {k: s['enabled'] for k, s in self.methods.items()}}
        try:
            # 재계획 시점에 한 번만 구성한다. PC 반복이나 terminal VLSA에서
            # 물체 기하·이전 행동·제어 간격을 바꾸면 서로 다른 제약 문제가 된다.
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
            # 비유한 계획을 뒤 QP에 보내면 관측 채널의 오류가 행동 투영에 가려질 수 있다.
            if not torch.isfinite(x).all():
                self.last_report.update(status='invalid_policy_output', failure_stage='predict')
                x = None
                break
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

        # 2. PC Correct: Predict 종료 계획에서 시간만 다시 0으로 시작한다.
        # VLSA로 미리 feasible하게 만들지 않는다. PC 단독과 같은 입력을 받는다.
        # 새 noise나 두 번째 Predict를 만들지 않는다. 기하와 이전 행동도 고정이다.
        pc = self.methods['pc']
        if x is not None and pc['enabled']:
            stage_started = perf_counter()
            count = pc['correct_steps']
            # 보정 시간 τ의 간격이다. 물리 가속도 계산의 control.dt_s와 구별한다.
            dt_c = 1.0 / count
            b_star = pc['delta'] / (1 - pc['rho'])
            eps = (0.9 / dt_c * b_star / (b_star - pc['delta'])**pc['rho']
                   if pc['eps'] == 'auto' else pc['eps'])
            self.last_report['pc_eps'] = eps
            for t in torch.linspace(0., 1., count + 1, device=p.device)[:-1]:
                velocity = p.model(x, t) * (pc['alpha'] * (1 - float(t)))
                self.last_report['model_calls'] += 1
                velocity[:, 0, p.act_dim:] = 0.
                if not torch.isfinite(velocity).all():
                    self.last_report.update(status='invalid_policy_output', failure_stage='pc')
                    x = None
                    break
                raw_actions = (x * p.scale + p.center)[0, :, :2].cpu().numpy().astype(np.float64)
                # h는 공통 면의 raw 여유 거리를 scale로 나눈 무차원 장벽이다.
                # CBF의 G U <= s φ(h)는 행동 위치가 아니라 보정 벡터장의 제약이다.
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
                # 미래 관측은 명목 벡터장으로 갱신하되 첫 관측 조건은 복원한다.
                # 행동 보정값에 맞추려고 모델의 미래 관측 채널을 덮어쓰지 않는다.
                x = p.apply_conditioning(x + velocity * dt_c, cond, p.act_dim).detach()
                # solver 성공과 모델 dtype에서의 유한성은 별개다. 실패 iterate는
                # terminal VLSA로 복구하지 않고 여기서 계획 자체를 폐기한다.
                if not torch.isfinite(x).all():
                    self.last_report.update(status='invalid_policy_output', failure_stage='pc')
                    x = None
                    break
            self.last_report['stage_ms']['pc'] = 1000 * (perf_counter() - stage_started)
            if x is not None:
                self._record('pc', x, constraints)

        # 3. VLSA: PC on이면 PC 종료값, off이면 Predict 종료값의 전체 H개 행동을
        # 한 번에 공동 투영한다. PC의 유한 반복 뒤 남은 행동 잔차는 이 명시적인
        # 단계가 보정할 수 있지만, 앞 단계의 solver 실패(None)를 되살리지는 않는다.
        if x is not None and self.methods['vlsa']['enabled']:
            stage_started = perf_counter()
            x = self._project(x, constraints, 'vlsa')
            self.last_report['stage_ms']['vlsa'] = 1000 * (perf_counter() - stage_started)
            if x is not None:
                self._record('vlsa', x, constraints)

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
                # 판정 대상은 마지막 활성 단계의 반환 dtype 결과다. VLSA off에서
                # PC 잔차를 줄이려고 숨은 terminal 투영을 추가하지 않는다.
                if self.enabled and final['qp_max_m'] > v['solver']['feasibility_tolerance_m']:
                    self.last_report.update(status='constraint_violation', failure_stage='final')
                    result = None
        self.last_report['latency_ms'] = 1000 * (perf_counter() - started)
        return result
