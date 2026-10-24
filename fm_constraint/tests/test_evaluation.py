"""모의 환경에서 실행 경계·이전 행동·실패·재계획 규칙을 검증한다."""

import numpy as np
import pytest

from configs.config import ExperimentConfig, REPO
from fm_constraint.evaluation import run_evaluation


class FakeEnv:
    def reset(self, random=False, context=None):
        self.p = np.array([.5, 0., .12])
        self.steps, self.commands = 0, []

    def step(self, command):
        self.commands.append(command.copy())
        self.p = command[:3].copy()
        self.steps += 1
        return None, 0., self.steps == 5, {'success': self.steps == 5, 'mode': 1, 'mean_distance': .01}


class FakePolicy:
    def __init__(self, actions=(.001, .002), fail=False):
        self.calls, self.actions, self.fail = [], actions, fail
        self.last_report = {}

    def plan(self, obs, previous):
        self.calls.append((obs.copy(), previous.copy()))
        self.last_report = {'status': 'infeasible' if self.fail else 'ok', 'model_calls': 10, 'qp_calls': 0}
        if self.fail:
            return None
        result = np.zeros((16, 14))
        result[:2, :2] = np.array(self.actions)[:, None]
        return result


def read_state(env, _):
    return env.p.copy(), [(np.array([.4, 0., 0.]), None, 0.), (np.array([.6, 0., 0.]), None, 0.)]


def build_obs(des, c, boxes):
    return np.r_[des, c[:2], np.array([[*b[0][:2], 0., 1.] for b in boxes]).ravel()]


@pytest.fixture
def config():
    return ExperimentConfig.from_yaml(REPO / 'configs/pushing.yaml').values


SETTINGS = {'n_traj': 1, 'max_steps': 5, 'execute_steps': 2}
CONTEXTS = [{'id': 'test', 'context': {}}]


def test_all_off_keeps_original_execution_order_and_previous_action(config):
    env, policy = FakeEnv(), FakePolicy()
    results, logs = run_evaluation(env, policy, CONTEXTS, read_state, build_obs, None,
                                   'pushing', SETTINGS, config)
    np.testing.assert_allclose(np.array(env.commands)[:, :2], .5 * np.array([[1, 0]]) + np.cumsum(np.array([.001, .002, .001, .002, .001])[:, None] * np.ones((1, 2)), axis=0))
    np.testing.assert_allclose([call[1] for call in policy.calls], [[0, 0], [.002, .002], [.002, .002]])
    assert results['success_rate'] == 1 and results['safe_success_rate'] is None
    assert results['action_attempts'] == results['executed_actions'] == 5
    assert logs['c000_t0'][0]['previous_action_m'] == [0., 0.]


def test_failed_plan_is_not_executed_and_counts_as_failure(config):
    env = FakeEnv()
    result, logs = run_evaluation(env, FakePolicy(fail=True), CONTEXTS, read_state, build_obs,
                                  None, 'pushing', SETTINGS, config)
    assert not env.commands
    assert result['success_rate'] == 0 and result['filter_abort_rate'] == 1
    assert result['episodes'][0]['failure'] == 'infeasible'
    assert result['action_attempts'] == result['executed_actions'] == 0
    assert logs['c000_t0'][0]['safe_action'] is None


def test_execution_recheck_invalidates_queue_and_uses_executed_previous(config):
    config['methods']['vlsa'].update(enabled=True, execution_recheck=True)
    config['constraints']['acceleration']['max_m_s2'] = .8
    env, policy = FakeEnv(), FakePolicy(actions=(.04, .04))
    result, logs = run_evaluation(env, policy, CONTEXTS, read_state, build_obs, None,
                                  'pushing', SETTINGS, config)
    executed = np.array([row['safe_action'] for row in logs['c000_t0']])
    assert len(policy.calls) == 5
    np.testing.assert_allclose([c[1] for c in policy.calls][1:], executed[:-1])
    lengths = np.linalg.norm(np.diff(executed, axis=0, prepend=np.zeros((1, 2))), axis=1) / .035**2
    assert lengths.max() <= .8 + 1e-5
    assert all(row['queue_invalidated'] for row in logs['c000_t0'])
    assert result['execution_intervention_rate'] == 1


def test_initial_previous_action_resets_between_episodes(config):
    policy = FakePolicy()
    run_evaluation(FakeEnv(), policy, CONTEXTS, read_state, build_obs, None,
                   'pushing', {**SETTINGS, 'n_traj': 2}, config)
    np.testing.assert_array_equal(policy.calls[3][1], [0., 0.])


def test_measured_and_command_workspace_are_distinct(config):
    class Overshoot(FakeEnv):
        def step(self, command):
            result = super().step(command)
            self.p[:2] = [1.5, 1.5]
            return result
    result, _ = run_evaluation(Overshoot(), FakePolicy(), CONTEXTS, read_state, build_obs,
                               None, 'pushing', SETTINGS, config)
    assert result['command_workspace_violation_rate'] == 0
    assert result['measured_workspace_violation_rate'] == 1


def test_contact_observer_accumulates_without_inventing_safety(config):
    def observe(env):
        return {'forbidden_contact': env.steps >= 2, 'min_distance': -.001 if env.steps >= 2 else .1}
    result, _ = run_evaluation(FakeEnv(), FakePolicy(), CONTEXTS, read_state, build_obs, None,
                               'pushing', SETTINGS, config, observe_safety=observe)
    assert result['success_rate'] == 1 and result['safe_success_rate'] == 0


def test_reporting_keeps_context_pairing_and_renders_saved_geometry(config, tmp_path, monkeypatch):
    import json
    import sys
    from fm_constraint.compute_metrics import main as compute
    from fm_constraint.render_rollouts import main as render
    env = FakeEnv()
    result, logs = run_evaluation(env, FakePolicy(), CONTEXTS, read_state, build_obs, None,
                                  'pushing', SETTINGS, config)
    result['checkpoint_sha256'] = 'same-model'
    directory = tmp_path / 'eval'
    directory.mkdir()
    (directory / 'results.json').write_text(json.dumps(result))
    (directory / 'rollouts.json').write_text(json.dumps(logs))
    monkeypatch.setattr(sys, 'argv', ['compute', '--eval-dir', str(directory),
                                     '--n-boot', '30', '--out-dir', str(tmp_path / 'summary')])
    compute()
    summary = json.loads((tmp_path / 'summary/summary.json').read_text())[0]
    assert summary['paired_success_difference'] == 0
    assert summary['pairing_status'] == 'matched_contexts_and_checkpoint'
    monkeypatch.setattr(sys, 'argv', ['render', '--eval-dir', str(directory), '--gif'])
    render()
    assert (directory / 'maps/c000_t0.png').stat().st_size > 0
    assert (directory / 'maps/c000_t0.gif').stat().st_size > 0
