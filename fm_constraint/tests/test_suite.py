"""32조합 순차 실험의 조합·분할·대응 난수·실행 필터와 보고 집계를 검증한다."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from configs.config import ExperimentConfig, REPO
from fm_constraint.evaluation import run_evaluation
from fm_constraint.suite import BITS, COMBINATIONS, COMPLEMENTARITY, prepare, verify_inputs
from fm_constraint.suite_metrics import _aggregate, _frame
from fm_constraint.tests.test_evaluation import FakeEnv, FakePolicy, read_state, build_obs, CONTEXTS, SETTINGS
from fm_constraint.tests.test_policy import baseline


def test_combination_bits_and_preregistered_complementarity():
    assert COMBINATIONS == tuple(range(32))
    assert BITS == {'projection': 16, 'pf2mp': 8, 'filter': 4, 'rtc': 2, 'lipo': 1}
    # 기존 method123 비교 쌍과 일치하고, 단독·all-off에는 비교가 없다.
    assert COMPLEMENTARITY[12] == (8, 4) and COMPLEMENTARITY[20] == (16, 4)
    assert COMPLEMENTARITY[24] == (16, 8) and COMPLEMENTARITY[28] == (24, 4)
    assert COMPLEMENTARITY[31] == (30, 1) and COMPLEMENTARITY[3] == (2, 1)
    assert not ({0, 1, 2, 4, 8, 16} & set(COMPLEMENTARITY))


def test_noise_per_context_seed_ignores_prior_abort_and_global_rng():
    config = ExperimentConfig.from_yaml(REPO / 'configs/pushing.yaml').values
    contexts = [{'id': 'first', 'context': {}}, {'id': 'second', 'context': {}}]
    class Recorder(FakePolicy):
        def __init__(self, abort_first):
            super().__init__()
            self.base = baseline()
            self.abort_first = abort_first
            self.noises = []
        def plan(self, obs, previous, noise=None, **kwargs):
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
    with tempfile.TemporaryDirectory(prefix='suite_test_', dir=REPO / 'fm_constraint/runs') as name:
        root = __import__('pathlib').Path(name)
        manifest = prepare(root, 'main', 'holdout', 'cpu', 2026)
        assert manifest['total_rollouts'] == 300 * 32 * 2
        assert manifest['tasks']['pushing']['rollouts_per_job'] == 300
        assert manifest['tasks']['inserting']['rollouts_per_job'] == 300
        assert [row['number'] for row in manifest['tasks']['pushing']['jobs']] == list(COMBINATIONS)
        jobs = manifest['tasks']['inserting']['jobs']
        assert [row['id'] for row in jobs][:3] == ['I00', 'I01', 'I02']
        assert jobs[19] == {'id': 'I19', 'number': 19, 'projection': True, 'pf2mp': False,
                            'filter': False, 'rtc': True, 'lipo': True}
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
