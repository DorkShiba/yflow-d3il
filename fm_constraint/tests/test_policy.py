"""통합 경로의 모델 호출 예산·조건 유지·실패 차단을 검증한다."""

from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from configs.config import ExperimentConfig, REPO
from fm_constraint.constraints import PlanConstraints
from fm_constraint.policy import Policy
from fm_constraint.qp import solve_qp


OBS = np.array([.5, 0., .5, 0., .4, 0., 0., 1., .6, 0., 0., 1.], dtype=np.float32)


def baseline(steps=10):
    def conditioning(x, cond, act_dim):
        x[:, 0, act_dim:] = cond[0]
        return x
    p = SimpleNamespace(horizon=4, obs_dim=12, act_dim=2, device='cpu',
                        center=torch.tensor([0., 0., .5, 0., .5, 0., .4, 0., 0., 0., .6, 0., 0., 0.]),
                        scale=torch.tensor([.01, .01] + [.1] * 12), apply_conditioning=conditioning)
    p.model = lambda x, t: .2 - .5 * x
    def plan(obs):
        cond = {0: (torch.from_numpy(obs)[None] - p.center[2:]) / p.scale[2:]}
        x = conditioning(torch.randn(1, 4, 14), cond, 2)
        ts = torch.linspace(0., 1., steps + 1)[:-1]
        for t in ts:
            vel = p.model(x, t)
            vel[:, 0, 2:] = 0
            x = x + vel * (ts[1] - ts[0])
        return (x * p.scale + p.center)[0].numpy()
    p.plan = plan
    return p


@pytest.mark.parametrize('flags', list(product((False, True), repeat=3)))
def test_all_combinations_obey_conditions_and_call_budget(flags):
    cfg = ExperimentConfig.from_yaml(REPO / 'configs/pushing.yaml')
    projection, pc, vlsa = flags
    for name, enabled in zip(('projection', 'pc', 'vlsa'), flags):
        cfg.values['methods'][name]['enabled'] = enabled
    torch.manual_seed(19)
    p = Policy(baseline(), cfg)
    plan = p.plan(OBS, [.001, -.001])
    assert plan is not None, p.last_report
    np.testing.assert_allclose(plan[0, 2:], OBS, atol=1e-7)
    assert p.last_report['model_calls'] == 10 + 9 * pc
    assert p.last_report['qp_calls'] == 10 * projection + 9 * pc + vlsa
    assert p.last_report['pipeline_order'] == ['projection', 'pc', 'vlsa']
    assert list(p.last_report['stages']) == ['predict'] + (['pc'] if pc else []) + (['vlsa'] if vlsa else [])
    if any(flags):
        assert p.last_report['final']['qp_max_m'] <= 1e-7


def test_all_off_exactly_delegates_without_extra_random_draw():
    cfg = ExperimentConfig.from_yaml(REPO / 'configs/pushing.yaml')
    torch.manual_seed(7)
    expected = baseline().plan(OBS)
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(7)
    actual = Policy(baseline(), cfg).plan(OBS, [0., 0.])
    np.testing.assert_array_equal(actual, expected)
    assert torch.equal(torch.get_rng_state(), expected_rng)


def test_without_pc_final_projection_matches_vlsa_and_every_then_vlsa_is_idempotent():
    cfg = ExperimentConfig.from_yaml(REPO / 'configs/pushing.yaml')
    noise = torch.randn(1, 4, 14, generator=torch.Generator().manual_seed(42))
    cfg.values['methods']['projection'].update(enabled=True, mode='final')
    final = Policy(baseline(), cfg).plan(OBS, [0., 0.], noise)
    cfg.values['methods']['projection']['enabled'] = False
    cfg.values['methods']['vlsa']['enabled'] = True
    terminal = Policy(baseline(), cfg).plan(OBS, [0., 0.], noise)
    np.testing.assert_array_equal(final, terminal)
    cfg.values['methods']['projection'].update(enabled=True, mode='every')
    both = Policy(baseline(), cfg).plan(OBS, [0., 0.], noise)
    cfg.values['methods']['vlsa']['enabled'] = False
    every = Policy(baseline(), cfg).plan(OBS, [0., 0.], noise)
    np.testing.assert_allclose(both, every, atol=1e-8)


def test_single_step_predict_and_invalid_inputs():
    cfg = ExperimentConfig.from_yaml(REPO / 'configs/pushing.yaml', {'sampling.predict_steps': 1})
    p = Policy(baseline(), cfg)
    plan = p.plan(OBS, [0., 0.], torch.zeros(1, 4, 14))
    np.testing.assert_allclose(plan[:, :2], .002, atol=1e-9)
    assert p.last_report['model_calls'] == 1
    assert p.plan(np.full(12, np.nan), [0., 0.]) is None
    assert p.last_report['status'] == 'invalid_input'


def test_infeasible_joint_constraints_return_no_plan():
    cfg = ExperimentConfig.from_yaml(REPO / 'configs/pushing.yaml', {'methods.vlsa.enabled': True})
    cfg.values['constraints']['object_envelope']['margin_m'] = .001
    obs = OBS.copy()
    obs[4:6] = obs[8:10] = [.8, .3]
    p = Policy(baseline(), cfg)
    assert p.plan(obs, [0., 0.]) is None
    assert p.last_report['status'] == 'infeasible'


def test_skipped_projection_does_not_add_hidden_final_projection():
    cfg = ExperimentConfig.from_yaml(REPO / 'configs/pushing.yaml', {
        'methods.projection.enabled': True, 'methods.projection.skip_first': 10})
    p = Policy(baseline(), cfg)
    noise = torch.zeros(1, 4, 14)
    noise[:, :, :2] = 100.
    assert p.plan(OBS, [0., 0.], noise) is None
    assert p.last_report['qp_calls'] == 0
    assert p.last_report['status'] == 'constraint_violation'


def order_case(vlsa=True):
    """PC 한 번으로 잔차가 남아 보정 순서를 수치로 구별할 수 있는 문제를 만든다."""
    cfg = ExperimentConfig.from_yaml(REPO / 'configs/pushing.yaml', {
        'sampling.predict_steps': 1, 'methods.pc.enabled': True,
        'methods.pc.correct_steps': 1, 'methods.pc.eps': .1,
        'methods.vlsa.enabled': vlsa, 'constraints.acceleration.enabled': False,
        'constraints.object_envelope.enabled': False})
    base = baseline(steps=1)
    base.model = lambda x, t: -.1 * x
    noise = torch.zeros(1, 4, 14)
    noise[:, :, :2] = 20.
    return cfg, base, noise


def test_pc_then_vlsa_matches_numeric_reference_and_differs_from_reversed_order():
    # 행동에 의존하는 모델을 써서 terminal 투영 전후의 PC 입력을 구별한다.
    cfg, base, noise = order_case()
    c = PlanConstraints(OBS, [0., 0.], 4, cfg.values['constraints'], .035)
    p = Policy(base, cfg)
    actual = p.plan(OBS, [0., 0.], noise)
    assert actual is not None, p.last_report
    pred = np.full(8, .18, dtype=np.float32).astype(np.float64)

    def one_pc(actions):
        # N_c=1, alpha=2이므로 명목 raw 벡터장은 -0.2 A다.
        # 본체의 stage 기록을 기대값으로 쓰지 않고 CBF 식과 QP로 직접 계산한다.
        h = (c.d - c.G @ actions) / c.scales
        z = h - .01
        bound = c.scales * .1 * np.sign(z) * np.abs(z)**.9
        solved = solve_qp(-.2 * actions, c.G, bound, cfg.values['solver'])
        assert solved['status'] == 'ok'
        return actions + solved['value']

    pc_actions = one_pc(pred)
    expected = solve_qp(pc_actions, c.G, c.d, cfg.values['solver'])['value']
    projected_first = solve_qp(pred, c.G, c.d, cfg.values['solver'])['value']
    reversed_order = one_pc(projected_first)
    np.testing.assert_allclose(actual[:, :2].ravel(), expected, atol=1e-7)
    assert np.linalg.norm(actual[:, :2].ravel() - reversed_order) > 1e-4
    np.testing.assert_allclose(p.last_report['stages']['predict']['actions_m'], pred.reshape(4, 2), atol=1e-7)
    np.testing.assert_allclose(p.last_report['stages']['pc']['actions_m'], pc_actions.reshape(4, 2), atol=1e-7)
    assert p.last_report['stages']['pc']['qp_max_m'] > 1e-7
    assert p.last_report['stages']['vlsa']['qp_max_m'] <= 1e-7
    assert p.last_report['stages']['vlsa']['intervention_norm_m'] > 1e-4
    np.testing.assert_array_equal(actual[0, 2:], OBS)


def test_pc_residual_is_rejected_without_explicit_terminal_vlsa():
    cfg, base, noise = order_case(vlsa=False)
    p = Policy(base, cfg)
    assert p.plan(OBS, [0., 0.], noise) is None
    assert p.last_report['status'] == 'constraint_violation'
    assert p.last_report['failure_stage'] == 'final'
    assert list(p.last_report['stages']) == ['predict', 'pc']
    assert p.last_report['qp_calls'] == 1


@pytest.mark.parametrize('stage', ['predict', 'pc', 'vlsa'])
@pytest.mark.parametrize('failure', ['solver_failure', 'infeasible'])
def test_solver_failure_stops_later_stages(monkeypatch, stage, failure):
    import fm_constraint.policy as module
    cfg, base, noise = order_case()
    if stage == 'predict':
        cfg.values['methods']['projection']['enabled'] = True
    fail_at = 2 if stage == 'vlsa' else 1
    calls = 0

    def failing_qp(*args):
        nonlocal calls
        calls += 1
        result = solve_qp(*args)
        if calls == fail_at:
            result.update(value=None, status=failure)
        return result

    monkeypatch.setattr(module, 'solve_qp', failing_qp)
    p = Policy(base, cfg)
    assert p.plan(OBS, [0., 0.], noise) is None
    assert p.last_report['status'] == failure
    assert p.last_report['failure_stage'] == ('projection' if stage == 'predict' else stage)
    assert calls == fail_at
    assert 'vlsa' not in p.last_report['stages']


@pytest.mark.parametrize('failure_call', [1, 2])
@pytest.mark.parametrize('channel', [0, 3])
def test_nonfinite_model_output_never_reaches_terminal_vlsa(failure_call, channel):
    cfg, base, noise = order_case()
    calls = 0

    def model(x, t):
        nonlocal calls
        calls += 1
        velocity = -.1 * x
        if calls == failure_call:
            velocity[:, 1, channel] = float('nan')
        return velocity

    base.model = model
    p = Policy(base, cfg)
    assert p.plan(OBS, [0., 0.], noise) is None
    assert p.last_report['status'] == 'invalid_policy_output'
    assert p.last_report['failure_stage'] == ('predict' if failure_call == 1 else 'pc')
    assert p.last_report['qp_calls'] == 0


@pytest.mark.parametrize('value,status', [(.2, 'constraint_violation'), (1e100, 'invalid_policy_output')])
def test_terminal_vlsa_checks_result_after_dtype_conversion(monkeypatch, value, status):
    import fm_constraint.policy as module
    cfg, base, noise = order_case()
    calls = 0

    def inaccurate_qp(*args):
        nonlocal calls
        calls += 1
        result = solve_qp(*args)
        if calls == 2:
            # solver 성공 상태만 믿지 않고 반환값의 잔차로 실패를 검출해야 한다.
            result['value'] = np.full(8, value)
        return result

    monkeypatch.setattr(module, 'solve_qp', inaccurate_qp)
    p = Policy(base, cfg)
    assert p.plan(OBS, [0., 0.], noise) is None
    assert p.last_report['status'] == status
    assert p.last_report['failure_stage'] == 'vlsa'
    assert calls == 2
    # 비유한 출력도 실패 보고 자체는 유효한 JSON으로 저장할 수 있어야 한다.
    import json
    json.dumps(p.last_report, allow_nan=False)


@pytest.mark.parametrize('predict_steps,correct_steps', [(1, 9), (5, 5), (10, 9)])
@pytest.mark.parametrize('mode,skip_first', [('every', 0), ('every', 2), ('final', 0)])
def test_explicit_call_budgets_with_terminal_vlsa(predict_steps, correct_steps, mode, skip_first):
    cfg = ExperimentConfig.from_yaml(REPO / 'configs/pushing.yaml', {
        'sampling.predict_steps': predict_steps, 'methods.pc.correct_steps': correct_steps,
        'methods.pc.enabled': True, 'methods.vlsa.enabled': True,
        'methods.projection.enabled': True, 'methods.projection.mode': mode,
        'methods.projection.skip_first': skip_first})
    p = Policy(baseline(predict_steps), cfg)
    assert p.plan(OBS, [0., 0.], torch.zeros(1, 4, 14)) is not None, p.last_report
    projection_calls = 1 if mode == 'final' else max(predict_steps - skip_first, 0)
    assert p.last_report['model_calls'] == predict_steps + correct_steps
    assert p.last_report['qp_calls'] == projection_calls + correct_steps + 1
