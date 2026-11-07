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
from fm_constraint.pf2mp import PotentialField


def potential(cfg):
    settings = cfg.values["methods"]["pf2mp"]
    return PotentialField(np.array([[.45, -.02], [.55, .02]]), np.array([[.45, -.02], [.55, .02]]), settings)


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


SUFFIX = np.full((2, 2), .001)          # horizon 4, execute 2 → H−s = 2
HISTORY = np.array([[.498, 0.], [.499, 0.], [.5, 0.]])


def config(**enabled):
    cfg = ExperimentConfig.from_yaml(REPO / 'configs/pushing.yaml', {'eval.execute_steps': 2})
    for name, value in enabled.items():
        cfg.values['methods'][name]['enabled'] = value
    return cfg


@pytest.mark.parametrize('flags', list(product((False, True), repeat=4)))
def test_all_plan_combinations_obey_conditions_and_call_budget(flags):
    projection, pf, rtc, lipo = flags
    cfg = config(projection=projection, pf2mp=pf, rtc=rtc, lipo=lipo)
    torch.manual_seed(19)
    p = Policy(baseline(), cfg, potential(cfg) if pf else None)
    plan = p.plan(OBS, [.001, -.001], torch.randn(1, 4, 14), rtc_suffix=SUFFIX, history=HISTORY)
    assert plan is not None, p.last_report
    np.testing.assert_allclose(plan[0, 2:], OBS, atol=1e-6)
    assert p.last_report['model_calls'] == 10
    assert p.last_report['qp_calls'] == 10 * projection + lipo
    assert p.last_report['pipeline_order'] == ['rtc', 'pf2mp', 'projection', 'lipo', 'filter']
    assert p.last_report['rtc_guided'] == rtc
    assert list(p.last_report['stages']) == ['predict'] + (['lipo'] if lipo else [])
    assert p.last_report['pf_calls'] == 10 * pf
    if projection:
        assert p.last_report['final']['qp_max_m'] <= 1e-7
    if lipo:
        assert p.last_report['lipo']['correction_max_m'] <= cfg.values['methods']['lipo']['bound_m'] + 1e-9
        assert p.last_report['lipo']['jerk_cost_after_m2'] <= p.last_report['lipo']['jerk_cost_before_m2'] + 1e-15


def test_rtc_pulls_actions_toward_suffix_and_is_inert_without_reference():
    noise = torch.randn(1, 4, 14, generator=torch.Generator().manual_seed(3))
    plain = Policy(baseline(), config()).plan(OBS, [0, 0], noise)
    no_ref = Policy(baseline(), config(rtc=True)).plan(OBS, [0, 0], noise)
    np.testing.assert_array_equal(plain, no_ref)
    target = np.full((2, 2), .005)
    guided = Policy(baseline(), config(rtc=True)).plan(OBS, [0, 0], noise, rtc_suffix=target)
    assert np.abs(guided[:2, :2] - target).max() < np.abs(plain[:2, :2] - target).max()
    cfg = config(rtc=True)
    cfg.values['methods']['rtc']['beta'] = 0.
    np.testing.assert_allclose(Policy(baseline(), cfg).plan(OBS, [0, 0], noise, rtc_suffix=target), plain, atol=1e-6)
    p = Policy(baseline(), config(rtc=True))
    assert p.plan(OBS, [0, 0], noise, rtc_suffix=np.zeros((3, 2))) is None
    assert p.last_report['failure_stage'] == 'rtc'


def test_lipo_keeps_frozen_prefix_and_projection_constraints():
    noise = torch.randn(1, 4, 14, generator=torch.Generator().manual_seed(5))
    frozen = np.array([[.002, -.001]])
    p = Policy(baseline(), config(projection=True, lipo=True))
    plan = p.plan(OBS, [0, 0], noise, frozen=frozen, history=HISTORY)
    assert plan is not None, p.last_report
    np.testing.assert_array_equal(plan[0, :2].astype(np.float32), frozen[0].astype(np.float32))
    assert p.last_report['lipo']['constrained'] and p.last_report['lipo']['frozen_prefix'] == 1
    constraints = PlanConstraints(OBS, [0, 0], 4, p.config['constraints'])
    keep = np.abs(constraints.G[:, 2:]).sum(1) > 0
    assert np.maximum(constraints.G[keep] @ plan[:, :2].ravel() - constraints.d[keep], 0).max() <= 1e-6


def test_lipo_failure_stops_the_plan(monkeypatch):
    import fm_constraint.policy as module
    monkeypatch.setattr(module, 'jerk_qp', lambda *a, **k: {'value': None, 'status': 'lipo_infeasible', 'latency_ms': 0.})
    p = Policy(baseline(), config(lipo=True))
    assert p.plan(OBS, [0, 0], torch.zeros(1, 4, 14)) is None
    assert p.last_report['failure_stage'] == 'lipo'


def test_all_off_exactly_delegates_without_extra_random_draw():
    cfg = ExperimentConfig.from_yaml(REPO / 'configs/pushing.yaml')
    torch.manual_seed(7)
    expected = baseline().plan(OBS)
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(7)
    actual = Policy(baseline(), cfg).plan(OBS, [0., 0.])
    np.testing.assert_array_equal(actual, expected)
    assert torch.equal(torch.get_rng_state(), expected_rng)


def test_single_step_predict_and_invalid_inputs():
    cfg = ExperimentConfig.from_yaml(REPO / 'configs/pushing.yaml', {'sampling.predict_steps': 1})
    p = Policy(baseline(), cfg)
    plan = p.plan(OBS, [0., 0.], torch.zeros(1, 4, 14))
    np.testing.assert_allclose(plan[:, :2], .002, atol=1e-9)
    assert p.last_report['model_calls'] == 1
    assert p.plan(np.full(12, np.nan), [0., 0.]) is None
    assert p.last_report['status'] == 'invalid_input'


def test_infeasible_joint_constraints_return_no_plan():
    cfg = ExperimentConfig.from_yaml(REPO / 'configs/pushing.yaml', {
        'methods.projection.enabled': True, 'methods.projection.mode': 'final'})
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




@pytest.mark.parametrize('steps', [1, 3])
def test_pf_action_is_indirect_and_one_step_does_not_change_actions(steps):
    cfg = ExperimentConfig.from_yaml(REPO / 'configs/pushing.yaml', {'sampling.predict_steps': steps})
    base = baseline(steps)
    def coupled_model(x, t):
        v = torch.zeros_like(x)
        v[:, :, :2] = .1 * x[:, :, 2:4]
        return v
    base.model = coupled_model
    noise = torch.ones(1, 4, 14) * .2
    plain = Policy(base, cfg).plan(OBS, [0, 0], noise)
    cfg.values['methods']['pf2mp']['enabled'] = True
    guided = Policy(base, cfg, potential(cfg)).plan(OBS, [0, 0], noise)
    np.testing.assert_array_equal(guided[0, 2:], OBS)
    if steps == 1:
        np.testing.assert_array_equal(guided[:, :2], plain[:, :2])
    else:
        assert not np.array_equal(guided[:, :2], plain[:, :2])
    assert not np.array_equal(guided[1:, 2:4], plain[1:, 2:4])


def test_zero_weight_is_identical_and_pf_only_measures_hard_violations():
    cfg = ExperimentConfig.from_yaml(REPO / 'configs/pushing.yaml')
    noise = torch.zeros(1, 4, 14)
    noise[:, :, :2] = 100
    plain = Policy(baseline(), cfg).plan(OBS, [0, 0], noise)
    cfg.values['methods']['pf2mp'].update(enabled=True, weight=0.)
    policy = Policy(baseline(), cfg, potential(cfg))
    actual = policy.plan(OBS, [0, 0], noise)
    np.testing.assert_array_equal(actual, plain)
    assert policy.last_report['final']['qp_max_m'] > 1e-7
    assert not policy.last_report['hard_constraints_enforced']


@pytest.mark.parametrize('stage', ['projection', 'projection_final'])
@pytest.mark.parametrize('failure', ['solver_failure', 'infeasible'])
def test_qp_failure_stops_pipeline(monkeypatch, stage, failure):
    import fm_constraint.policy as module
    cfg = ExperimentConfig.from_yaml(REPO / 'configs/pushing.yaml', {
        'methods.projection.enabled': True, 'methods.lipo.enabled': True, 'sampling.predict_steps': 1,
        'methods.projection.mode': 'every' if stage == 'projection' else 'final'})
    def failed(*args):
        result = solve_qp(*args)
        result.update(value=None, status=failure)
        return result
    monkeypatch.setattr(module, 'solve_qp', failed)
    p = Policy(baseline(1), cfg)
    assert p.plan(OBS, [0, 0], torch.zeros(1, 4, 14)) is None
    assert p.last_report['failure_stage'] == stage
    assert p.last_report['qp_calls'] == 1


@pytest.mark.parametrize('value,status', [(.2, 'constraint_violation'), (1e100, 'invalid_policy_output')])
def test_terminal_dtype_failure(monkeypatch, value, status):
    import fm_constraint.policy as module
    cfg = ExperimentConfig.from_yaml(REPO / 'configs/pushing.yaml', {
        'methods.projection.enabled': True, 'methods.projection.mode': 'final'})
    def inaccurate(*args):
        result = solve_qp(*args)
        result['value'] = np.full(8, value)
        return result
    monkeypatch.setattr(module, 'solve_qp', inaccurate)
    p = Policy(baseline(), cfg)
    assert p.plan(OBS, [0, 0], torch.zeros(1, 4, 14)) is None
    assert p.last_report['status'] == status
    import json
    json.dumps(p.last_report, allow_nan=False)


def test_nonfinite_pf_stops_before_qp():
    cfg = ExperimentConfig.from_yaml(REPO / 'configs/pushing.yaml', {
        'methods.pf2mp.enabled': True, 'methods.projection.enabled': True, 'methods.projection.mode': 'final'})
    field = potential(cfg)
    field.support[:] = float('nan')
    p = Policy(baseline(), cfg, field)
    assert p.plan(OBS, [0, 0], torch.zeros(1, 4, 14)) is None
    assert p.last_report['failure_stage'] == 'pf2mp'
    assert p.last_report['qp_calls'] == 0


@pytest.mark.parametrize('failure_call', [1, 2])
def test_nonfinite_model_output_stops_before_lipo(failure_call):
    cfg = ExperimentConfig.from_yaml(REPO / 'configs/pushing.yaml', {'methods.lipo.enabled': True})
    base = baseline()
    count = 0
    def model(x, t):
        nonlocal count
        count += 1
        value = torch.zeros_like(x)
        if count == failure_call:
            value[:, 1, 3] = float('nan')
        return value
    base.model = model
    p = Policy(base, cfg)
    assert p.plan(OBS, [0, 0], torch.zeros(1, 4, 14)) is None
    assert p.last_report['failure_stage'] == 'predict'
    assert p.last_report['qp_calls'] == 0


def test_combined_step_matches_independent_numeric_reference():
    cfg = ExperimentConfig.from_yaml(REPO / 'configs/pushing.yaml', {
        'methods.projection.enabled': True, 'methods.pf2mp.enabled': True,
        'sampling.predict_steps': 2})
    base, field = baseline(2), potential(cfg)
    noise = torch.ones(1, 4, 14) * .3
    x = noise.clone()
    cond = {0: (torch.from_numpy(OBS)[None] - base.center[2:]) / base.scale[2:]}
    x = base.apply_conditioning(x, cond, 2)
    constraints = PlanConstraints(OBS, [0, 0], 4, cfg.values['constraints'])
    for t in [0., .5]:
        velocity = base.model(x, t)
        velocity[:, 0, 2:] = 0
        q = x[:, 1:, 2:4].double() * base.scale[2:4] + base.center[2:4]
        # 단일 query별 Gaussian score와 최근접 거리 gradient를 본체와 독립 계산한다.
        array = q.numpy()
        support, safe = field.support.numpy(), field.safe.numpy()
        diff = array[..., None, :] - support
        logits = -(diff**2).sum(-1) / (2 * field.bandwidth**2)
        weight = np.exp(logits - logits.max(-1, keepdims=True))
        weight /= weight.sum(-1, keepdims=True)
        score = -(weight[..., None] * diff).sum(-2) / field.bandwidth**2
        distances = ((array[..., None, :] - safe)**2).sum(-1)
        nearest = safe[distances.argmin(-1)]
        delta = array - nearest
        score += field.alpha * delta / np.maximum(np.linalg.norm(delta, axis=-1, keepdims=True), 1e-15)
        guidance = torch.zeros_like(x)
        guidance[:, 1:, 2:4] = torch.as_tensor(score * base.scale[2:4].numpy(), dtype=x.dtype)
        y = x + .5 * (velocity + cfg.values['methods']['pf2mp']['weight'] * guidance)
        raw = (y * base.scale + base.center)[0].numpy()
        solved = solve_qp(raw[:, :2].ravel(), constraints.G, constraints.d, cfg.values['solver'])
        y[:, :, :2] = (torch.from_numpy(solved['value'].reshape(1, 4, 2)).float() - base.center[:2]) / base.scale[:2]
        x = base.apply_conditioning(y, cond, 2)
    actual = Policy(base, cfg, field).plan(OBS, [0, 0], noise)
    np.testing.assert_allclose(actual, (x * base.scale + base.center)[0].numpy(), atol=1e-7, rtol=0)
