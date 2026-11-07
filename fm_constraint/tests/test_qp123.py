from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

from configs.config import ExperimentConfig, REPO
from fm_constraint.evaluation import run_evaluation
from fm_constraint.policy import Policy
from fm_constraint.tests.test_evaluation import FakeEnv, read_state, build_obs
from fm_constraint.tests.test_policy import baseline, potential
from fm_constraint.run_qp123 import ARMS


@pytest.mark.parametrize('flags', list(product((False, True), repeat=3)))
def test_eight_arms_execute_qp_and_log_actual_actions(flags):
    cfg = ExperimentConfig.from_yaml(REPO / 'configs/pushing.yaml')
    for name, flag in zip(('projection', 'pf2mp', 'vlsa'), flags):
        cfg.values['methods'][name]['enabled'] = flag
    policy = Policy(baseline(), cfg, potential(cfg) if flags[1] else None)
    env = FakeEnv()
    result, logs = run_evaluation(env, policy, [{'id': '0', 'context': {}}],
        read_state, build_obs, None, 'pushing',
        {'n_traj': 1, 'max_steps': 5, 'execute_steps': 2, 'seed': 0}, cfg.values,
        qp5_bound=.0003)
    assert result['method4'] is False and result['method5'] is True
    assert result['qp5_solved_count'] == 3
    assert result['smoothness']['actual_jerk_rms_m_s3']['n'] == 1
    assert not result['hard_constraints_enforced']
    rows = logs['c000_t0']
    np.testing.assert_allclose(np.array([r['next_command_state'] for r in rows]),
                               np.array(env.commands)[:, :2])
    for i, row in enumerate(rows):
        if i:
            np.testing.assert_array_equal(row['previous_action_m'], rows[i-1]['safe_action'])
        if row['plan_report']:
            diag = row['plan_report']['qp5']
            assert diag['solver_status'] == 'Solved'
            assert diag['raw_deviation_max_m'] <= .0003 + 1e-8
            assert diag['jerk_cost_after_m2'] <= diag['jerk_cost_before_m2'] + 1e-12
            assert 'after_violation' in diag


def test_noise_is_paired_and_failed_plan_never_reaches_qp():
    cfg = ExperimentConfig.from_yaml(REPO / 'configs/pushing.yaml')
    captured = []
    class Recorder:
        base = SimpleNamespace(device='cpu', horizon=4, act_dim=2, obs_dim=12)
        def plan(self, obs, previous, noise):
            captured.append(noise.clone())
            self.last_report = {'status': 'infeasible'}
            return None
    for _ in range(2):
        env = FakeEnv()
        result, _ = run_evaluation(env, Recorder(), [{'id': '0', 'context': {}}],
            read_state, build_obs, None, 'pushing',
            {'n_traj': 1, 'max_steps': 5, 'execute_steps': 2, 'seed': 9}, cfg.values,
            qp5_bound=.0003)
        assert not env.commands and result['qp5_solved_count'] == 0
        assert result['episodes'][0]['failure'] == 'infeasible'
    np.testing.assert_array_equal(captured[0], captured[1])
    assert len(set(ARMS)) == 8


def test_qp_failure_aborts_episode_without_executing_unfiltered_plan(monkeypatch):
    from fm_baseline.lipo import ChunkPostprocessor
    def fail(*args, **kwargs):
        raise RuntimeError('solver failed')
    monkeypatch.setattr(ChunkPostprocessor, 'process', fail)
    cfg = ExperimentConfig.from_yaml(REPO / 'configs/pushing.yaml')
    env = FakeEnv()
    result, _ = run_evaluation(env, Policy(baseline(), cfg), [{'id': '0', 'context': {}}],
        read_state, build_obs, None, 'pushing',
        {'n_traj': 1, 'max_steps': 5, 'execute_steps': 2, 'seed': 0}, cfg.values,
        qp5_bound=.0003)
    assert not env.commands
    assert result['episodes'][0]['failure'] == 'qp5_solver_failed'
    assert result['success_rate'] == 0 and result['filter_abort_rate'] == 1


@pytest.mark.parametrize('flags', list(product((False, True), repeat=3)))
def test_rtc_uses_post_qp_suffix_on_every_replan(flags, monkeypatch):
    from fm_baseline.lipo import ChunkPostprocessor

    cfg = ExperimentConfig.from_yaml(REPO / 'configs/pushing.yaml')
    cfg.values['eval']['execute_steps'] = 2
    for name, flag in zip(('projection', 'pf2mp', 'vlsa'), flags):
        cfg.values['methods'][name]['enabled'] = flag
    policy = Policy(baseline(), cfg, potential(cfg) if flags[1] else None)
    outputs, references = [], []
    process = ChunkPostprocessor.process
    plan = policy.plan

    def capture_qp(self, *args, **kwargs):
        output = process(self, *args, **kwargs)
        outputs.append(output.copy())
        return output

    def capture_plan(*args, rtc_suffix, **kwargs):
        references.append(rtc_suffix)
        return plan(*args, rtc_suffix=rtc_suffix, **kwargs)

    monkeypatch.setattr(ChunkPostprocessor, 'process', capture_qp)
    monkeypatch.setattr(policy, 'plan', capture_plan)
    result, logs = run_evaluation(FakeEnv(), policy, [{'id': '0', 'context': {}}],
        read_state, build_obs, None, 'pushing',
        {'n_traj': 1, 'max_steps': 5, 'execute_steps': 2, 'seed': 0}, cfg.values,
        qp5_bound=.0003, rtc=True)
    assert result['method4'] and result['method5']
    assert result['qp5_solved_count'] == 3
    assert references[0] is None
    for reference, previous in zip(references[1:], outputs):
        np.testing.assert_array_equal(reference, previous[2:])
    reports = [r['plan_report'] for r in logs['c000_t0'] if r['plan_report']]
    assert [r['rtc_guided'] for r in reports] == [False, True, True]
    assert all(r['model_calls'] == 10 and r['pf_calls'] == 10 * flags[1] for r in reports)


@pytest.mark.parametrize('task,count', [('pushing', 60), ('inserting', 82)])
def test_fixed_conditions_have_matched_context_ids_and_noise(task, count):
    from fm_constraint.fixed_conditions import load_condition, observe_boxes
    import json
    directory = REPO / 'fm_baseline/ood_benchmark' / task
    manifest = json.loads((directory / 'manifest.json').read_text())
    _, reference = load_condition(task, 'id')
    assert len(manifest['conditions']) == 14
    for name in manifest['conditions']:
        _, condition = load_condition(task, name)
        assert len(condition['episodes']) == count
        assert [(e['name'], e['noise_seed']) for e in condition['episodes']] == [
            (e['name'], e['noise_seed']) for e in reference['episodes']]
    boxes = [(np.zeros(3), None, 0.)]
    _, condition = load_condition(task, 'C_mid')
    noise = condition['factors']['C']
    seed = [condition['episodes'][0]['noise_seed'], 0]
    noisy = observe_boxes(boxes, noise, np.random.default_rng(seed))
    assert np.any(noisy[0][0] != boxes[0][0])
    assert np.all(boxes[0][0] == 0)
    np.testing.assert_array_equal(noisy[0][0], observe_boxes(boxes, noise, np.random.default_rng(seed))[0][0])


def test_reference_arm_matches_existing_rtc_guidance_for_identical_noise():
    import torch
    from fm_baseline.rtc import guided_plan
    from fm_constraint.tests.test_policy import OBS

    class Field(torch.nn.Module):
        def forward(self, x, t):
            return .2 - .5 * x

    base = baseline()
    base.model = Field()
    base.t_span = torch.linspace(0., 1., 11)[:-1]
    cfg = ExperimentConfig.from_yaml(REPO / 'configs/pushing.yaml')
    cfg.values['eval']['execute_steps'] = 2
    suffix = np.array([[.002, -.005], [.004, .001]])
    reference = guided_plan(base, OBS, suffix, 2, 0, generator=torch.Generator().manual_seed(21))
    noise = torch.randn((1, 4, 14), generator=torch.Generator().manual_seed(21))
    actual = Policy(base, cfg).plan(OBS, np.zeros(2), noise, rtc_suffix=suffix)
    np.testing.assert_allclose(actual, reference, atol=1e-7)
