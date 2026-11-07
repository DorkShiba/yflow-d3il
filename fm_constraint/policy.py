"""baseline 모델을 재사용하는 Predict[FM+PF → Euler → Projection] → VLSA 정책."""

from time import perf_counter

import numpy as np
import torch

from fm_baseline.rtc import guided_velocity, rtc_target, soft_mask

from .constraints import PlanConstraints
from .qp import solve_qp


# 설정 키의 나열 순서와 무관한 고정 실행 순서다. 과거 결과와의 비교에도 저장한다.
PIPELINE_ORDER = ('pf2mp', 'projection', 'vlsa')
PIPELINE_METADATA = {
    'pipeline_version': 'pf2mp_state_xy_v1',
    'predict_update_order': ['fm', 'pf2mp', 'euler', 'projection'],
    'post_predict_order': ['vlsa'],
}


class Policy:
    """baseline Policy를 감싸고, 활성 보정만 명시적으로 순서대로 적용한다.

    plan(obs, previous_action)는 성공 시 raw (H,D) 배열, 실패 시 None을
    반환한다. 마지막 실행의 상태·기하·중간 행동·비용은 last_report에 남긴다.
    이전 행동은 호출자가 실제 실행 직후 갱신하며 정책 내부에 숨겨두지 않는다.
    """

    def __init__(self, baseline_policy, config, potential=None):
        self.base = baseline_policy
        self.config = config.values if hasattr(config, 'values') and not isinstance(config, dict) else config
        self.methods = self.config['methods']
        self.enabled = any(value['enabled'] for value in self.methods.values())
        self.hard_constraints_enforced = any(self.methods[k]['enabled'] for k in ('projection', 'vlsa'))
        self.potential = potential
        if self.methods['pf2mp']['enabled'] and potential is None:
            raise ValueError('PF2MP enabled without prepared potential; call prepare_potential once')
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
        self.last_report['qp'][-1]['intervention_norm_m'] = float(np.linalg.norm(value.reshape(-1, 2) - raw[:, :2]))
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
            # VLSA 수정량은 Predict 종료 행동을 기준으로 측정한다.
            previous = next(reversed(stages.values()))['actions_m']
            report['intervention_norm_m'] = float(np.linalg.norm(raw[:, :2] - np.asarray(previous)))
        reconstructed = constraints.q0 + np.cumsum(raw[:, :2], axis=0)
        report['state_command_gap_max_m'] = float(np.linalg.norm(raw[:-1, 2:4] + raw[:-1, :2] - raw[1:, 2:4], axis=-1).max()) if len(raw) > 1 else 0.
        report['generated_vs_cumulative_command_max_m'] = float(np.linalg.norm(raw[1:, 2:4] - reconstructed[:-1], axis=-1).max()) if len(raw) > 1 else 0.
        if self.potential:
            phi, _ = self.potential.evaluate((x * self.base.scale + self.base.center)[:, 1:, 2:4])
            report['energy'] = float(-phi.sum())
        stages[label] = report

    @torch.no_grad()
    def plan(self, obs, previous_action, noise=None, rtc_suffix=None):
        """현재 관측과 직전 실행 행동으로 전체 계획을 생성한다.

        obs는 raw (obs_dim,), previous_action은 실제 실행한 XY 증분 (2,), 단위 m다.
        비교용 noise는 정규화 공간의 (1,H,act_dim+obs_dim)이며 내부에서 복사한다.
        성공하면 raw (H,act_dim+obs_dim)을 반환한다. 실패하면 None을 반환하고
        status·failure_stage·이미 끝난 단계의 행동과 비용을 last_report에 남긴다.
        """
        started = perf_counter()
        p, v = self.base, self.config
        self.last_report = {'status': 'ok', 'failure_stage': None, 'model_calls': 0,
                            'pipeline_order': list(PIPELINE_ORDER), **PIPELINE_METADATA,
                            'pf_calls': 0, 'pf_ms': 0., 'model_ms': 0., 'pf_steps': [],
                            'potential_cache': self.potential.cache_report if self.potential else None,
                            'hard_constraints_enforced': self.hard_constraints_enforced,
                            'qp_calls': 0, 'qp_ms': 0., 'qp': [], 'stages': {},
                            'latency_ms': 0., 'stage_ms': {}, 'methods': {k: s['enabled'] for k, s in self.methods.items()}}
        self.last_report['rtc_guided'] = rtc_suffix is not None
        if rtc_suffix is not None:
            execute = v['eval']['execute_steps']
            if rtc_suffix.shape != (p.horizon - execute, p.act_dim):
                raise ValueError('RTC requires the post-QP H-s action suffix')
            weights = soft_mask(p.horizon, execute, 0)
            target, mask = rtc_target(p, rtc_suffix, weights)
        try:
            # 재계획 시점에 한 번만 구성한다. Predict나 terminal VLSA에서
            # 물체 기하·이전 행동·제어 간격을 바꾸면 서로 다른 제약 문제가 된다.
            constraints = PlanConstraints(obs, previous_action, p.horizon, v['constraints'], v['control']['dt_s'])
        except (ValueError, TypeError) as error:
            self.last_report.update(status='invalid_input', error=str(error), failure_stage='input')
            self.last_report['latency_ms'] = 1000 * (perf_counter() - started)
            return None
        self.last_report['geometry'] = constraints.geometry()
        if self.hard_constraints_enforced and constraints.low is not None and (
                np.any(constraints.q0 < constraints.low - v['solver']['feasibility_tolerance_m'])
                or np.any(constraints.q0 > constraints.high + v['solver']['feasibility_tolerance_m'])):
            self.last_report.update(status='initial_outside_workspace', failure_stage='input')
            self.last_report['latency_ms'] = 1000 * (perf_counter() - started)
            return None
        # 모든 방법 off이면 baseline에 위임한다. 추가 noise를 뽑거나 제약으로 행동을 거부하지 않는다.
        if rtc_suffix is None and not self.enabled and noise is None and v['sampling']['predict_steps'] > 1:
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
            model_started = perf_counter()
            if rtc_suffix is None:
                velocity = p.model(x, t)
            else:
                velocity = guided_velocity(p, x, t, target, mask, k / steps, beta=5.)
            self.last_report['model_calls'] += 1
            self.last_report['model_ms'] += 1000 * (perf_counter() - model_started)
            velocity[:, 0, p.act_dim:] = 0.
            # PF는 갱신 전의 같은 x에서 계산한다. action 가산항은 항상0이다.
            pf_report = None
            if self.methods['pf2mp']['enabled']:
                pf_started = perf_counter()
                self.last_report['pf_calls'] += 1
                try:
                    guidance, pf_report = self.potential.guidance(x, p.center, p.scale, p.act_dim)
                    if (not torch.isfinite(guidance).all() or guidance[:, :, :p.act_dim].count_nonzero()
                            or guidance[:, 0, p.act_dim:].count_nonzero()):
                        raise ValueError('PF2MP violated finite/zero-action/conditioning invariant')
                    weight = self.methods['pf2mp']['weight']
                    pf_report.update(step=k, increment_norm=float((guidance.double() * weight * dt).norm()))
                    # weight=0에서 별도 덧셈에 의한 반올림도 피한다.
                    if weight:
                        velocity = velocity + weight * guidance
                except (ValueError, RuntimeError) as error:
                    self.last_report.update(status='invalid_potential', failure_stage='pf2mp', error=str(error))
                    x = None
                    break
                finally:
                    self.last_report['pf_ms'] += 1000 * (perf_counter() - pf_started)
            x = x + velocity * dt
            if pf_report is not None and torch.isfinite(x).all():
                q = (x * p.scale + p.center)[:, 1:, p.act_dim:p.act_dim + 2]
                phi, _ = self.potential.evaluate(q)
                pf_report['energy_after_euler'] = float(-phi.sum())
                self.last_report['pf_steps'].append(pf_report)
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

        # 이전 PC Correct 보존: schema v3에서는 실행하지 않는다.
        # 이전 구현은 변경 전 커밋 및 pf2mp_validation/before.json 기준으로 재현한다.
        # LEGACY_PC # 2. PC Correct: Predict 종료 계획에서 시간만 다시 0으로 시작한다.
        # LEGACY_PC # VLSA로 미리 feasible하게 만들지 않는다. PC 단독과 같은 입력을 받는다.
        # LEGACY_PC # 새 noise나 두 번째 Predict를 만들지 않는다. 기하와 이전 행동도 고정이다.
        # LEGACY_PC pc = self.methods['pc']
        # LEGACY_PC if x is not None and pc['enabled']:
        # LEGACY_PC     stage_started = perf_counter()
        # LEGACY_PC     count = pc['correct_steps']
        # LEGACY_PC     # 보정 시간 τ의 간격이다. 물리 가속도 계산의 control.dt_s와 구별한다.
        # LEGACY_PC     dt_c = 1.0 / count
        # LEGACY_PC     b_star = pc['delta'] / (1 - pc['rho'])
        # LEGACY_PC     eps = (0.9 / dt_c * b_star / (b_star - pc['delta'])**pc['rho']
        # LEGACY_PC            if pc['eps'] == 'auto' else pc['eps'])
        # LEGACY_PC     self.last_report['pc_eps'] = eps
        # LEGACY_PC     for t in torch.linspace(0., 1., count + 1, device=p.device)[:-1]:
        # LEGACY_PC         velocity = p.model(x, t) * (pc['alpha'] * (1 - float(t)))
        # LEGACY_PC         self.last_report['model_calls'] += 1
        # LEGACY_PC         velocity[:, 0, p.act_dim:] = 0.
        # LEGACY_PC         if not torch.isfinite(velocity).all():
        # LEGACY_PC             self.last_report.update(status='invalid_policy_output', failure_stage='pc')
        # LEGACY_PC             x = None
        # LEGACY_PC             break
        # LEGACY_PC         raw_actions = (x * p.scale + p.center)[0, :, :2].cpu().numpy().astype(np.float64)
        # LEGACY_PC         # h는 공통 면의 raw 여유 거리를 scale로 나눈 무차원 장벽이다.
        # LEGACY_PC         # CBF의 G U <= s φ(h)는 행동 위치가 아니라 보정 벡터장의 제약이다.
        # LEGACY_PC         barrier = (constraints.d - constraints.G @ raw_actions.ravel()) / constraints.scales
        # LEGACY_PC         z = barrier - pc['delta']
        # LEGACY_PC         bound = constraints.scales * eps * np.sign(z) * np.abs(z)**pc['rho']
        # LEGACY_PC         # 벡터장은 dA/dτ이므로 scale만 곱한다. center를 더하면 미분이 아니다.
        # LEGACY_PC         raw_velocity = (velocity[0, :, :2].double() * p.scale[:2].double()).cpu().numpy()
        # LEGACY_PC         solved = solve_qp(raw_velocity.ravel(), constraints.G, bound, v['solver'])
        # LEGACY_PC         self.last_report['qp_calls'] += 1
        # LEGACY_PC         self.last_report['qp_ms'] += solved['latency_ms']
        # LEGACY_PC         value = solved.pop('value')
        # LEGACY_PC         self.last_report['qp'].append({'stage': 'pc', **solved})
        # LEGACY_PC         if solved['status'] != 'ok':
        # LEGACY_PC             self.last_report.update(status=solved['status'], failure_stage='pc')
        # LEGACY_PC             x = None
        # LEGACY_PC             break
        # LEGACY_PC         corrected = torch.as_tensor(value.reshape(1, -1, 2), device=x.device, dtype=x.dtype)
        # LEGACY_PC         velocity[:, :, :2] = corrected / p.scale[:2]
        # LEGACY_PC         # 미래 관측은 명목 벡터장으로 갱신하되 첫 관측 조건은 복원한다.
        # LEGACY_PC         # 행동 보정값에 맞추려고 모델의 미래 관측 채널을 덮어쓰지 않는다.
        # LEGACY_PC         x = p.apply_conditioning(x + velocity * dt_c, cond, p.act_dim).detach()
        # LEGACY_PC         # solver 성공과 모델 dtype에서의 유한성은 별개다. 실패 iterate는
        # LEGACY_PC         # terminal VLSA로 복구하지 않고 여기서 계획 자체를 폐기한다.
        # LEGACY_PC         if not torch.isfinite(x).all():
        # LEGACY_PC             self.last_report.update(status='invalid_policy_output', failure_stage='pc')
        # LEGACY_PC             x = None
        # LEGACY_PC             break
        # LEGACY_PC     self.last_report['stage_ms']['pc'] = 1000 * (perf_counter() - stage_started)
        # LEGACY_PC     if x is not None:
        # LEGACY_PC         self._record('pc', x, constraints)
        # LEGACY_PC
        # LEGACY_PC # 3. VLSA: PC on이면 PC 종료값, off이면 Predict 종료값의 전체 H개 행동을
        # LEGACY_PC # 한 번에 공동 투영한다. PC의 유한 반복 뒤 남은 행동 잔차는 이 명시적인
        # LEGACY_PC # 단계가 보정할 수 있지만, 앞 단계의 solver 실패(None)를 되살리지는 않는다.

        # VLSA는 유효한 Predict 종료 계획에만 적용한다.
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
                # 잔차를 줄이려고 숨은 terminal 투영을 추가하지 않는다.
                if self.hard_constraints_enforced and final['qp_max_m'] > v['solver']['feasibility_tolerance_m']:
                    self.last_report.update(status='constraint_violation', failure_stage='final')
                    result = None
        self.last_report['latency_ms'] = 1000 * (perf_counter() - started)
        return result
