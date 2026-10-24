"""통합 경로의 모델 호출 예산·조건 유지·실패 차단을 검증한다."""

from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from configs.config import ExperimentConfig, REPO
from fm_constraint.policy import Policy


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
    for name, enabled in zip(('projection', 'vlsa', 'pc'), flags):
        cfg.values['methods'][name]['enabled'] = enabled
    torch.manual_seed(19)
    p = Policy(baseline(), cfg)
    plan = p.plan(OBS, [.001, -.001])
    assert plan is not None, p.last_report
    np.testing.assert_allclose(plan[0, 2:], OBS, atol=1e-7)
    assert p.last_report['model_calls'] == 10 + 9 * flags[2]
    assert p.last_report['qp_calls'] == 10 * flags[0] + flags[1] + 9 * flags[2]
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


def test_final_projection_matches_vlsa_and_every_then_vlsa_is_idempotent():
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
