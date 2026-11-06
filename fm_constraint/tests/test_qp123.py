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
