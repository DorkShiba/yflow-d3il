"""본실험(ID·OOD) runner: 평가 목록 고정, 인자 해석, noise 장비 독립성, 팀원 보고·통합 집계."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import fm_constraint.experiment as experiment
from configs.config import ExperimentConfig, REPO
from fm_constraint.evaluation import run_evaluation
from fm_constraint.tests.test_evaluation import FakeEnv, FakePolicy, read_state, build_obs, SETTINGS
from fm_constraint.tests.test_policy import baseline

D3IL = (REPO / '../d3il').resolve()


def test_argument_parsing():
    assert experiment.parse_indices('0-6', 30) == list(range(7))
    assert experiment.parse_indices('3,5,9', 30) == [3, 5, 9]
    assert experiment.parse_indices('all', 30) == list(range(30))
    assert experiment.parse_combos('P05,P17') == [5, 17] == experiment.parse_combos('5,17')
    assert experiment.parse_combos('16-31')[0] == 16 and experiment.combo_id('inserting', 19) == 'I19'
    assert experiment.combo_flags(19) == {'projection': True, 'pf2mp': False, 'filter': False, 'rtc': True, 'lipo': True}
    for bad in ('30', '-1', '5-40'):
        with pytest.raises(ValueError):
            experiment.parse_indices(bad, 30)


def test_frozen_list_has_30_contexts_with_disjoint_groups():
    selection = json.loads(experiment.CONTEXT_FILE.read_text())
    assert selection['noise_seeds'] == [0, 1, 2]
    for task in ('pushing', 'inserting'):
        rows = selection['tasks'][task]['contexts']
        assert [r['index'] for r in rows] == list(range(30))
        assert len({r['group'] for r in rows}) == 30 and len({r['context_sha256'] for r in rows}) == 30


@pytest.mark.skipif(not (D3IL / 'environments/dataset/data/pushing/test_contexts.pkl').exists(), reason='D3IL not found')
def test_selection_is_reproducible_and_matches_sources(tmp_path, monkeypatch):
    frozen = json.loads(experiment.CONTEXT_FILE.read_text())
    monkeypatch.setattr(experiment, 'CONTEXT_FILE', tmp_path / 'contexts.json')
    assert experiment.select(D3IL) == frozen
    for task in ('pushing', 'inserting'):
        assert len(experiment.load_selection(task, D3IL)) == 30


def test_noise_is_generated_on_cpu_and_shared_across_combinations():
    config = ExperimentConfig.from_yaml(REPO / 'configs/pushing.yaml').values

    class Recorder(FakePolicy):
        def __init__(self):
            super().__init__()
            self.base, self.noises = baseline(), []

        def plan(self, obs, previous, noise=None, **kwargs):
            self.noises.append(noise.clone())
            return super().plan(obs, previous)

    settings = {**SETTINGS, 'n_traj': 2, 'sampling_seeds': [0, 1]}
    contexts = [{'id': '3', 'context': {}}]
    first, second = Recorder(), Recorder()
    torch.manual_seed(1)
    run_evaluation(FakeEnv(), first, contexts, read_state, build_obs, None, 'pushing', settings, config)
    config['methods']['projection']['enabled'] = True     # 다른 조합, 같은 (context, seed)
    torch.manual_seed(999)
    run_evaluation(FakeEnv(), second, contexts, read_state, build_obs, None, 'pushing', settings, config)
    torch.testing.assert_close(first.noises[0], second.noises[0], rtol=0, atol=0)
    generator = torch.Generator(device='cpu').manual_seed(0)
    assert not torch.equal(first.noises[0], first.noises[3])   # seed 0과 1은 다르다
    if torch.cuda.is_available():
        gpu = Recorder()
        gpu.base.device = 'cuda'
        run_evaluation(FakeEnv(), gpu, contexts, read_state, build_obs, None, 'pushing', settings, config)
        torch.testing.assert_close(gpu.noises[0].cpu(), first.noises[0], rtol=0, atol=0)


def fake_unit(task, number, index, success, member='a'):
    episodes = [{'context_id': str(index), 'sampling_seed': s, 'success': success, 'mode': 1,
                 'constraint_violation_steps': 1, 'constraint_evaluated_steps': 10,
                 'action_change_m': .001, 'command_jerk_m_s3': 50., 'failure': None if success else 'max_steps'}
                for s in (0, 1, 2)]
    return {'protocol': 'id_v1', 'task': task, 'combo': experiment.combo_id(task, number), 'number': number,
            'context_index': index, 'member': member, 'code_fingerprint': 'same', 'wall_s': 1.,
            'device': {'gpu': 'X'}, 'episodes': episodes}


def write_units(root, units):
    for u in units:
        path = experiment.unit_dir(root, u['task'], u['number'], u['context_index']) / 'result.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(u))


def run_reports(tmp_path, monkeypatch, units, ood=None):
    """단위 결과를 로컬 결과 폴더에 쓰고 팀원별 보고서를 만든다. 결과 폴더는 지워 merge가 보고서만 쓰게 한다."""
    monkeypatch.setattr(experiment, 'RESULT_BASE', tmp_path / 'results')
    monkeypatch.setattr(experiment, 'REPORT_DIR', tmp_path / 'docs')
    experiment.REPORT_DIR.mkdir(exist_ok=True)
    write_units(experiment.result_root(ood), units)
    for member in sorted({u['member'] for u in units}):
        experiment.report(member, ood=ood)
    import shutil
    shutil.rmtree(tmp_path / 'results')


def test_member_report_and_merge_use_reports_only(tmp_path, monkeypatch):
    units = [fake_unit('pushing', n, i, success=(n in (1, 3) or i % 2 == 0), member='a' if i < 7 else 'b')
             for n in (0, 1, 2, 3) for i in range(30)]
    run_reports(tmp_path, monkeypatch, units)
    text = (tmp_path / 'docs/report_a.md').read_text()
    assert '| P00 | 0,1,2,3,4,5,6 | 21 |' in text and 'P03' in text and experiment.REPORT_MARKER in text
    assert len(experiment.parse_report(tmp_path / 'docs/report_b.md')['units']) == 4 * 23
    experiment.merge(n_boot=200)
    merged = (tmp_path / 'docs/report_results.md').read_text()
    assert '미완료' in merged and 'report_a.md' in merged         # P04..P31 없음, 입력 보고서 기록
    assert experiment.parse_report(tmp_path / 'docs/report_results.md') is None   # 통합본은 다시 읽히지 않는다
    data = json.loads((tmp_path / 'docs/report_results.json').read_text())
    assert data['meta']['reports'] == ['report_a.md', 'report_b.md']
    summary = data['pushing']
    assert summary['P01']['success_rate_pct'][0] == 100 and summary['P00']['success_rate_pct'][0] == 50
    # P03 = P02 + P01: Δ = 100 − max(50, 100) = 0
    assert summary['P03']['complementarity_success_pp'][0] == 0
    assert summary['P01']['constraint_violation_rate_pct'][0] == 10


def test_ood_reports_are_separate_from_id(tmp_path, monkeypatch):
    units = [{**fake_unit('pushing', 0, i, success=True), 'protocol': 'ood_v1_A_mid',
              'ood': {'sha256': 'f' * 64, 'condition': 'A_mid'}} for i in range(30)]
    run_reports(tmp_path, monkeypatch, units, ood='A_mid')
    assert (tmp_path / 'docs/report_ood_A_mid_a.md').exists()
    assert experiment.result_root('A_mid') == tmp_path / 'results/ood_v1/A_mid'
    with pytest.raises(SystemExit, match='No reports'):
        experiment.merge(n_boot=10)                              # ID 보고서는 없다
    experiment.merge(n_boot=10, ood='A_mid')
    assert json.loads((tmp_path / 'docs/report_ood_A_mid_results.json').read_text())['meta']['ood'] == 'A_mid'
    with pytest.raises(ValueError):
        experiment.protocol('Z_mid')
    for bad in ('results', 'ood_x'):
        with pytest.raises(ValueError):
            experiment.check_name(bad)


def test_merge_rejects_duplicate_units(tmp_path, monkeypatch):
    run_reports(tmp_path, monkeypatch, [fake_unit('pushing', 0, 0, True, member='a')])
    run_reports(tmp_path, monkeypatch, [fake_unit('pushing', 0, 0, True, member='b')])
    with pytest.raises(SystemExit, match='appears in both'):
        experiment.merge(n_boot=10)


def test_merge_rejects_mixed_code(tmp_path, monkeypatch):
    units = [fake_unit('pushing', 0, 0, True), {**fake_unit('pushing', 0, 1, True), 'code_fingerprint': 'other'}]
    run_reports(tmp_path, monkeypatch, units)
    with pytest.raises(SystemExit, match='code versions'):
        experiment.merge(n_boot=10)


def test_ood_context_ids_map_to_id_noise():
    from fm_constraint.evaluation import id_context_key, ood_context_id
    assert ood_context_id('pushing', '7') == 'test_007' and id_context_key('pushing', 'test_007') == '7'
    assert ood_context_id('inserting', 'env_005_06.npz') == id_context_key('inserting', 'env_005_06.npz') == 'env_005_06.npz'


def test_ood_case_with_sampling_seeds_uses_id_noise():
    """seed 0·1·2를 준 OOD 평가는 같은 context의 ID 평가와 같은 FM noise를 쓴다."""
    config = ExperimentConfig.from_yaml(REPO / 'configs/pushing.yaml').values

    class Recorder(FakePolicy):
        def __init__(self):
            super().__init__()
            self.base, self.noises = baseline(), []

        def plan(self, obs, previous, noise=None, **kwargs):
            self.noises.append(noise.clone())
            return super().plan(obs, previous)

    settings = {**SETTINGS, 'n_traj': 2, 'sampling_seeds': [0, 1]}
    case = {'id': 'pushing/A_mid/test_003', 'policy_seeds': [11, 12, 13, 14], 'noise': None, 'push': None,
            'delay_steps': 0, 'robot_start': None, 'factors': {}}
    id_run, ood_run = Recorder(), Recorder()
    run_evaluation(FakeEnv(), id_run, [{'id': '3', 'context': {}}], read_state, build_obs, None, 'pushing',
                   settings, config)
    results, _ = run_evaluation(FakeEnv(), ood_run, [{'id': 'test_003', 'noise_key': '3', 'context': {}, 'ood': case}],
                                read_state, build_obs, None, 'pushing', settings, config)
    assert [e['sampling_seed'] for e in results['episodes']] == [0, 1]
    torch.testing.assert_close(id_run.noises[0], ood_run.noises[0], rtol=0, atol=0)


def test_unit_lock_blocks_a_second_holder(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, 'LOCAL_BASE', tmp_path)
    with experiment.unit_lock('pushing', 5, 3, smoke=True) as first:
        assert first
        with experiment.unit_lock('pushing', 5, 3, smoke=True) as second:
            assert not second                         # 같은 단위는 동시에 잡을 수 없다
        with experiment.unit_lock('pushing', 5, 4, smoke=True) as other:
            assert other                              # 다른 단위는 독립적이다
    with experiment.unit_lock('pushing', 5, 3, smoke=True) as again:
        assert again                                  # 끝나면 풀린다
