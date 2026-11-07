"""중첩 설정·checkpoint 호환·출력 경로 보호를 검증한다."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from configs.config import ExperimentConfig, REPO, parse_config


def test_nested_overrides_and_cli_priority():
    cfg = parse_config('sample', argv=['--config', str(REPO / 'configs/pushing.yaml'),
                                      '--set', 'methods.pf2mp.enabled=true', '--no-pf2mp',
                                      '--set', 'constraints.object_envelope.n_sides=7', '--filter', '--rtc', '--lipo'])
    assert all(cfg.values['methods'][k]['enabled'] for k in ('filter', 'rtc', 'lipo'))
    assert not cfg.values['methods']['pf2mp']['enabled']
    assert cfg.values['constraints']['object_envelope']['n_sides'] == 7
    assert cfg.check_checkpoint()['obs_dim'] == 12
    assert '--no-save' in cfg.baseline_argv('sample')
    ExperimentConfig.from_yaml(REPO / 'configs/pushing.yaml', {'methods.pf2mp.weight': .04}).validate('sample')


@pytest.mark.parametrize('overrides', [
    {'vlsa.enabled': True}, {'methods.vlsa.enabled': True}, {'methods.pc.enabled': True},
    {'constraints.action_box.enabled': True}, {'schema_version': 3},
    {'methods.rtc.beta': -1.}, {'methods.lipo.bound_m': 0.}, {'methods.rtc.enabled': True, 'eval.execute_steps': 9},
    {'methods.pf2mp.enabled': 'false'}, {'train.batch_size': True}, {'train.lr': float('nan')},
    {'model.horizon': 15}, {'eval.execute_steps': 17}, {'sampling.predict_steps': 0},
    {'constraints.object_envelope.n_sides': 2}, {'constraints.acceleration.polygon_sides': 7},
    {'constraints.acceleration.max_m_s2': None}, {'constraints.acceleration.max_m_s2': True},
    {'paths.output_root': 'fm_baseline/runs'}, {'sample.no_save': False},
    {'constraints.workspace.low_m': [1, 0]}, {'control.dt_s': 0.}, {'methods.pf2mp.bandwidth_m': 0.},
    {'methods.pf2mp.weight': -1.}, {'methods.pf2mp.distance_alpha': 1.},
    {'methods.pf2mp.max_support': 0}, {'methods.pf2mp.density_quantile': 1.},
])
def test_invalid_configuration_is_rejected(overrides):
    with pytest.raises(ValueError):
        ExperimentConfig.from_yaml(REPO / 'configs/pushing.yaml', overrides).validate('sample')


def test_retired_cli_switches_are_rejected():
    for flag in ('--pc', '--vlsa'):
        with pytest.raises(SystemExit):
            parse_config('sample', argv=['--config', str(REPO / 'configs/pushing.yaml'), flag])


def test_model_mismatch_and_training_free_flags():
    with pytest.raises(ValueError, match='Checkpoint model'):
        ExperimentConfig.from_yaml(REPO / 'configs/pushing.yaml', {'model.dim': 64}).validate('sample')
    with pytest.raises(ValueError, match='Training-free'):
        ExperimentConfig.from_yaml(REPO / 'configs/pushing.yaml', {'methods.pf2mp.enabled': True}).validate('train')
    with pytest.raises(ValueError, match='dt_s'):
        ExperimentConfig.from_yaml(REPO / 'configs/pushing.yaml', {'control.dt_s': .04}).validate('eval')


def test_paths_and_symlinks_are_resolved(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = ExperimentConfig.from_yaml(REPO / 'configs/inserting.yaml')
    assert Path(cfg.values['paths']['data_root']) == REPO / 'data'
    import configs.config as module
    monkeypatch.setattr(module, 'REPO', tmp_path)
    (tmp_path / 'fm_constraint/runs').mkdir(parents=True)
    (tmp_path / 'fm_baseline').mkdir()
    (tmp_path / 'fm_constraint/runs/escape').symlink_to(tmp_path / 'fm_baseline')
    path = tmp_path / 'config.yaml'
    path.write_text('task: pushing\npaths:\n  output_root: fm_constraint/runs/escape\nconstraints:\n  acceleration:\n    enabled: false\n')
    with pytest.raises(ValueError, match='Outputs/resume'):
        ExperimentConfig.from_yaml(path)


def test_resume_conflict(tmp_path, monkeypatch):
    import configs.config as module
    monkeypatch.setattr(module, 'REPO', tmp_path)
    run = tmp_path / 'fm_constraint/runs/resume'
    run.mkdir(parents=True)
    (run / 'config.json').write_text(json.dumps({'args': {'task': 'pushing', 'lr': .1}}))
    path = tmp_path / 'config.yaml'
    path.write_text('task: pushing\ntrain:\n  resume: fm_constraint/runs/resume\nconstraints:\n  acceleration:\n    enabled: false\n')
    with pytest.raises(ValueError, match='Resume setting conflicts'):
        ExperimentConfig.from_yaml(path).validate('train')


def test_baseline_import_orders_and_collision():
    for order in [('train', 'sample', 'eval_pushing', 'eval_inserting'),
                  ('eval_inserting', 'eval_pushing', 'sample', 'train')]:
        code = ('from fm_constraint import import_baseline\n'
                f'for name in {order!r}:\n'
                ' m=import_baseline(name)\n'
                ' assert "/fm_baseline/" in m.__file__\n')
        subprocess.run([sys.executable, '-c', code], cwd=REPO, check=True, capture_output=True)
    code = ('import sys,types\nfrom fm_constraint import import_baseline\n'
            'sys.modules["train"]=types.ModuleType("train")\n'
            'try: import_baseline("sample")\n'
            'except ImportError: pass\nelse: raise AssertionError("collision accepted")\n')
    subprocess.run([sys.executable, '-c', code], cwd=REPO, check=True, capture_output=True)
