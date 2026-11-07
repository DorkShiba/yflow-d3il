"""순차 실험의 조합·분할·대응 난수·실행 필터와 보고 집계를 검증한다."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from configs.config import ExperimentConfig, REPO
from fm_constraint.evaluation import run_evaluation
from fm_constraint.method123 import COMBINATIONS, prepare, verify_inputs
from fm_constraint.method123_metrics import _aggregate, _frame
from fm_constraint.tests.test_evaluation import FakeEnv, FakePolicy, read_state, build_obs, CONTEXTS, SETTINGS
from fm_constraint.tests.test_policy import baseline


def test_execution_filter_independent_of_terminal_vlsa():
    config = ExperimentConfig.from_yaml(REPO / 'configs/pushing.yaml').values
    config['constraints']['acceleration']['max_m_s2'] = .8
    env, policy = FakeEnv(), FakePolicy(actions=(.04, .04))
    result, logs = run_evaluation(env, policy, CONTEXTS, read_state, build_obs, None,
                                   'pushing', SETTINGS, config, execution_filter=True)
    assert not config['methods']['vlsa']['enabled']
    assert result['executed_action_filter'] and result['execution_recheck']
    assert len(policy.calls) == 5
    actions = np.array([row['safe_action'] for row in logs['c000_t0']])
    np.testing.assert_allclose([call[1] for call in policy.calls][1:], actions[:-1])
    assert (np.linalg.norm(np.diff(actions, axis=0, prepend=np.zeros((1, 2))), axis=1) / .035**2).max() <= .8 + 1e-5
    assert all(row['queue_invalidated'] for row in logs['c000_t0'])


def test_noise_per_context_seed_ignores_prior_abort_and_global_rng():
    config = ExperimentConfig.from_yaml(REPO / 'configs/pushing.yaml').values
    contexts = [{'id': 'first', 'context': {}}, {'id': 'second', 'context': {}}]
    class Recorder(FakePolicy):
        def __init__(self, abort_first):
            super().__init__()
            self.base = baseline()
            self.abort_first = abort_first
            self.noises = []
        def plan(self, obs, previous, noise=None):
            self.noises.append(noise.clone())
            output = super().plan(obs, previous)
            if self.abort_first and len(self.noises) == 1:
                self.last_report['status'] = 'infeasible'
                return None
            return output
    settings = {**SETTINGS, 'n_traj': 2, 'sampling_seeds': [0, 1]}
    one, two = Recorder(False), Recorder(True)
    torch.manual_seed(12)
    result, logs = run_evaluation(FakeEnv(), one, contexts, read_state, build_obs, None,
                                  'pushing', settings, config)
    torch.manual_seed(999)
    second, _ = run_evaluation(FakeEnv(), two, contexts, read_state, build_obs, None,
                              'pushing', settings, config)
    # 첫 rollout은 정상 실행 시 3계획, 실패 시 1계획. 다음 seed의 첫 noise는 같다.
    torch.testing.assert_close(one.noises[3], two.noises[1], rtol=0, atol=0)
    torch.testing.assert_close(one.noises[6], two.noises[4], rtol=0, atol=0)
    assert result['episodes'][2]['noise_seed'] == second['episodes'][2]['noise_seed']
    assert len({row['noise_seed'] for row in result['episodes']}) == 4
    assert logs['c000_t0'][0]['plan_report']['noise_index'] == 0
    assert not second['episodes'][0]['success']
    assert second['episodes'][0]['constraint_evaluated_steps'] == 0


def test_group_holdout_and_manifest_are_frozen():
    import tempfile
    with tempfile.TemporaryDirectory(prefix='method123_test_', dir=REPO / 'fm_constraint/runs') as name:
        root = __import__('pathlib').Path(name)
        manifest = prepare(root, 'main', 'holdout', 'cpu', 2026)
        assert manifest['total_rollouts'] == 4800
        assert manifest['tasks']['pushing']['rollouts_per_job'] == 300
        assert manifest['tasks']['inserting']['rollouts_per_job'] == 300
        assert [row['number'] for row in manifest['tasks']['pushing']['jobs']] == list(COMBINATIONS)
        assert all(not row['rtc'] and not row['lipo'] for spec in manifest['tasks'].values() for row in spec['jobs'])
        tuning = json.loads((root / 'inserting_tuning_contexts.json').read_text())
        final = json.loads((root / 'inserting_final_contexts.json').read_text())
        assert len(tuning) == 20 and len(final) == 60
        assert not {row['context_group'] for row in tuning} & {row['context_group'] for row in final}
        assert prepare(root, 'main', 'holdout', 'cpu', 2026)['manifest_sha256'] == manifest['manifest_sha256']
        frozen = root / 'configs/pushing/resolved_config.yaml'
        frozen.write_text(frozen.read_text() + '\n# changed\n')
        with pytest.raises(ValueError, match='input changed'):
            verify_inputs(root, manifest)


def test_metrics_use_executed_steps_and_conditional_entropy():
    episodes = []
    for context, success, mode, bad, count in [('a', True, 0, 1, 2), ('a', True, 1, 0, 2),
                                              ('b', False, -1, 0, 0)]:
        episodes.append({'context_id': context, 'success': success, 'mode': mode,
                         'constraint_violation_steps': bad, 'constraint_evaluated_steps': count,
                         'action_change_m': .1 if count else None, 'command_jerk_m_s3': .2 if count else None,
                         'failure': None if success else 'infeasible'})
    frame = _frame(episodes, ['a', 'b'], 'pushing')
    result = _aggregate(frame)
    np.testing.assert_allclose(result, [50., 25., .1, .2, .5, 50.])
    assert not np.isfinite(frame[1, 5])
