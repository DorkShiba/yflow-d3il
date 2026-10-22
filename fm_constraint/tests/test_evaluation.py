import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fm_constraint.evaluation import evaluate, run_evaluation
from fm_constraint.config import ExperimentConfig, REPO


class FakeEnv:
    def reset(self, random=False, context=None):
        self.p = np.array([0., 0., .12])
        self.steps = 0
        self.commands = []

    def step(self, command):
        self.steps += 1
        self.commands.append(command.copy())
        self.p = command[:3].copy()
        return None, 0, self.steps == 5, {'success': self.steps == 5,
                                        'mode': 1, 'mean_distance': .01}


class FakePolicy:
    def __init__(self):
        self.conditions = []

    def plan(self, obs):
        self.conditions.append(obs.copy())
        plan = np.zeros((16, 14))
        plan[:, :2] = np.arange(1, 17)[:, None] * .001
        return plan


def read_state(env, _):
    return env.p.copy(), [(np.array([.4, .2, 0.]), None, 0.)] * 2


def build_obs(des, c, boxes):
    return np.r_[des, c[:2]]


SETTINGS = {'n_traj': 1, 'max_steps': 5, 'execute_steps': 2, 'comparison': 'none'}
CONTEXTS = [{'id': 'initial', 'context': {}}]


def test_disabled_vlsa_matches_baseline_execution_order(monkeypatch):
    def unexpected_call(*args, **kwargs):
        raise AssertionError('Disabled VLSA must not be called')
    monkeypatch.setattr('fm_constraint.evaluation.vlsa_constraint', unexpected_call)
    env, policy = FakeEnv(), FakePolicy()
    results, logs = run_evaluation(env, policy, CONTEXTS, read_state, build_obs,
                                   None, 'pushing', SETTINGS, vlsa={'enabled': False})
    # Independent copy of the original policy execution, with no extra RNG calls.
    reference, reference_policy = FakeEnv(), FakePolicy()
    reference.reset(random=False, context=copy.deepcopy(CONTEXTS[0]['context']))
    c, boxes = read_state(reference, None)
    des, queue, done = c[:2].copy(), [], False
    while not done:
        obs = build_obs(des, c, boxes)
        if not queue:
            plan = reference_policy.plan(obs)
            queue = [plan[k, :2].astype(np.float64) for k in range(2)]
        des += queue.pop(0)
        _, _, done, _ = reference.step(np.r_[des, .12, 0., 1., 0., 0.])
        c, boxes = read_state(reference, None)
    np.testing.assert_array_equal(env.commands, reference.commands)
    np.testing.assert_array_equal(policy.conditions, reference_policy.conditions)
    assert results['success_rate'] == 1
    assert results['safe_success_rate'] is None
    assert results['vlsa_enabled'] is False
    assert len(logs['c000_t0']) == 5


def test_vlsa_runs_for_every_action_and_updates_next_policy_condition(monkeypatch):
    calls = []
    def checked_vlsa(nominal, measured, command, config):
        assert config['enabled'] is True
        calls.append((measured.copy(), command.copy()))
        return {'action': nominal / 2, 'status': 'solved', 'h': [1.],
                'residual': 0., 'intervention_norm': float(np.linalg.norm(nominal / 2)),
                'latency_ms': .01}
    monkeypatch.setattr('fm_constraint.evaluation.vlsa_constraint', checked_vlsa)
    env, policy = FakeEnv(), FakePolicy()
    results, _ = run_evaluation(env, policy, CONTEXTS, read_state, build_obs, None,
                                'pushing', SETTINGS, vlsa={'enabled': True})
    assert len(calls) == 5
    np.testing.assert_allclose(policy.conditions[1][:2], [.0015, .0015])
    np.testing.assert_allclose(calls[2][1], [.0015, .0015])
    assert results['safe_success_rate'] is None
    assert results['vlsa_enabled'] is True


def test_vlsa_failure_counts_as_failed_episode_and_executes_no_action(monkeypatch):
    def failed_vlsa(*args):
        return {'action': None, 'status': 'infeasible', 'h': [0.],
                'residual': None, 'intervention_norm': None, 'latency_ms': .01}
    monkeypatch.setattr('fm_constraint.evaluation.vlsa_constraint', failed_vlsa)
    env = FakeEnv()
    results, logs = run_evaluation(env, FakePolicy(), CONTEXTS, read_state, build_obs,
                                   None, 'pushing', SETTINGS, vlsa={'enabled': True})
    assert results['success_rate'] == 0
    assert results['episodes'][0]['failure'] == 'infeasible'
    assert env.commands == []
    assert logs['c000_t0'][0]['safe_action'] is None


def test_real_vlsa_function_modifies_commands_in_rollout():
    config = {'enabled': True, 'gamma': 10, 'margin_m': 0.01, 'pusher_radius_m': 0.02,
              'command_delta_limit_m': .0005,
              'obstacles': [{'center_xy_m': [1., 0.], 'radius_m': .1}]}
    env, policy = FakeEnv(), FakePolicy()
    results, logs = run_evaluation(env, policy, CONTEXTS, read_state, build_obs, None,
                                   'pushing', SETTINGS, vlsa=config)
    np.testing.assert_allclose(np.asarray(env.commands)[:, :2], np.arange(1, 6)[:, None] * np.full((1, 2), .0005))
    np.testing.assert_allclose(policy.conditions[1][:2], [.001, .001])
    assert all(row['qp_status'] == 'solved' for row in logs['c000_t0'])
    assert results['vlsa_enabled'] is True


def test_stop_comparison_uses_vlsa_prediction_without_enabling_vlsa():
    config = {'enabled': False, 'gamma': 10, 'margin_m': 0.01, 'pusher_radius_m': 0.02,
              'command_delta_limit_m': .0005,
              'obstacles': [{'center_xy_m': [1., 0.], 'radius_m': .1}]}
    env = FakeEnv()
    results, logs = run_evaluation(env, FakePolicy(), CONTEXTS, read_state, build_obs, None,
                                   'pushing', {**SETTINGS, 'comparison': 'stop'}, vlsa=config)
    np.testing.assert_array_equal(np.asarray(env.commands)[:, :2], np.zeros((5, 2)))
    assert results['vlsa_enabled'] is False
    assert all(row['qp_status'] == 'stop_comparison' for row in logs['c000_t0'])


def test_contacts_accumulate_and_safe_success_is_distinct():
    def observer(env):
        return {'forbidden_contact': env.steps >= 2, 'min_distance': -.001 if env.steps >= 2 else .1}
    results, _ = run_evaluation(FakeEnv(), FakePolicy(), CONTEXTS, read_state, build_obs,
                                None, 'pushing', SETTINGS, observe_safety=observer)
    assert results['success_rate'] == 1
    assert results['safe_success_rate'] == 0
    assert results['forbidden_contact_free_rate'] == 0


def test_real_obstacles_are_never_silently_ignored():
    cfg = ExperimentConfig.from_yaml(REPO / 'configs/pushing.yaml', {
        'vlsa.geometry': 'circle', 'vlsa.workspace': None,
        'vlsa.obstacles': [{'id': 'a', 'center_xy_m': [.5, .2], 'radius_m': .03, 'height_m': .2}]})
    with pytest.raises(NotImplementedError, match='audit'):
        evaluate(cfg)


def test_rectangle_distinguishes_command_and_measured_hand():
    class OvershootEnv(FakeEnv):
        def step(self, command):
            result = super().step(command)
            self.p[:2] = [.05, .05]  # Tracking is intentionally outside the command fence.
            return result
    config = {'enabled': True, 'geometry': 'rectangle', 'margin_m': .001,
              'command_delta_limit_m': .001, 'obstacles': [],
              'workspace': {'x_min_m': -.01, 'x_max_m': .004,
                            'y_min_m': -.01, 'y_max_m': .004}}
    results, logs = run_evaluation(OvershootEnv(), FakePolicy(), CONTEXTS, read_state, build_obs,
                                   None, 'pushing', SETTINGS, vlsa=config)
    assert results['command_workspace_violation_rate'] == 0
    assert results['measured_workspace_violation_rate'] == 1
    assert results['command_workspace_success_rate'] == 1
    assert results['safe_success_rate'] is None
    assert min(logs['c000_t0'][-1]['command_workspace_h']) >= -1e-9
    assert min(logs['c000_t0'][-1]['measured_workspace_h']) < 0


def test_disabled_rectangle_still_measures_unmodified_commands():
    config = {'enabled': False, 'geometry': 'rectangle', 'margin_m': .001,
              'command_delta_limit_m': .001, 'obstacles': [],
              'workspace': {'x_min_m': -.01, 'x_max_m': .004,
                            'y_min_m': -.01, 'y_max_m': .004}}
    env = FakeEnv()
    results, _ = run_evaluation(env, FakePolicy(), CONTEXTS, read_state, build_obs,
                                None, 'pushing', SETTINGS, vlsa=config)
    assert results['command_workspace_violation_rate'] == 1
    assert results['command_workspace_success_rate'] == 0
    assert results['success_rate'] == 1
    np.testing.assert_allclose(env.commands[-1][:2], [.007, .007])


@pytest.mark.parametrize('task', ['pushing', 'inserting'])
def test_rectangle_evaluation_entry_point_runs_with_vlsa_enabled(task, tmp_path, monkeypatch):
    import fm_constraint.config as config_module
    cfg = ExperimentConfig.from_yaml(REPO / f'configs/{task}.yaml', {'vlsa.enabled': True})
    saved_model = cfg.check_checkpoint()
    cfg.values['eval'].update(n_contexts=1, n_traj=1, execute_steps=2, max_steps=5)
    cfg.values['paths']['output_root'] = str(tmp_path / 'fm_constraint/runs')
    d3il = tmp_path / 'd3il'
    d3il.mkdir()
    cfg.values['paths']['d3il_root'] = str(d3il)
    monkeypatch.setattr(config_module, 'REPO', tmp_path)
    context_file = d3il / 'environments/dataset/data/pushing/test_contexts.pkl'
    context_file.parent.mkdir(parents=True)
    with context_file.open('wb') as handle:
        np.save(handle, np.array([{}], dtype=object))

    class InBoundsEnv(FakeEnv):
        def __init__(self, **kwargs):
            pass

        def start(self):
            pass

        def reset(self, **kwargs):
            super().reset(**kwargs)
            self.p[:2] = [.525, -.28]

    baseline = SimpleNamespace(
        load_model=lambda *args: (saved_model, None, None, None, None, 1000000),
        Policy=lambda *args: FakePolicy(), setup_d3il=lambda root: (InBoundsEnv, None),
        read_state=read_state, build_obs=build_obs,
        load_contexts=lambda *args: [('demo.npz', {'robot': {'des_c_pos': np.zeros((6, 3))}}, {}, 'unused.npz')],
    )
    monkeypatch.setattr('fm_constraint.evaluation.import_baseline', lambda name: baseline)
    cwd = Path.cwd()
    results = evaluate(cfg)
    assert Path.cwd() == cwd
    assert results['vlsa_enabled'] is True
    assert results['success_rate'] == 1
    assert results['command_workspace_violation_rate'] == 0
    assert results['constraint_scope'] == 'command_workspace'
    outputs = list((tmp_path / 'fm_constraint/runs').glob('eval_*/results.json'))
    assert len(outputs) == 1
    assert json.loads(outputs[0].read_text())['safe_success_rate'] is None
