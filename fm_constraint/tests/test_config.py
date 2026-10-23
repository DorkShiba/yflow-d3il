import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from fm_constraint.config import ExperimentConfig, REPO, parse_config


def test_cli_yaml_and_checkpoint_compatibility():
    cfg = parse_config('sample', argv=['--config', str(REPO / 'configs/pushing.yaml'),
                                      '--set', 'sample.n_windows=2', '--device', 'cpu'])
    assert cfg.values['sample']['n_windows'] == 2
    assert '--no-save' in cfg.baseline_argv('sample')
    assert cfg.check_checkpoint()['obs_dim'] == 12
    with pytest.raises(ValueError, match='Checkpoint model'):
        ExperimentConfig.from_yaml(REPO / 'configs/pushing.yaml', {'model.dim': 64}).validate('sample')


@pytest.mark.parametrize('overrides', [
    {'model.padding': True}, {'train.batch_size': True}, {'train.lr': float('nan')},
    {'model.horizon': 15}, {'eval.execute_steps': 17}, {'eval.ode_steps': 1},
    {'paths.output_root': 'fm_baseline/runs'}, {'eval.comparison': 'typo'},
    {'vlsa.enabled': True}, {'vlsa.enabled': 'false'}, {'sample.no_save': False},
])
def test_invalid_settings_are_rejected(overrides):
    with pytest.raises(ValueError):
        ExperimentConfig.from_yaml(REPO / 'configs/pushing.yaml', overrides).validate('sample')


def test_unknown_yaml_key_and_cli_false(tmp_path):
    path = tmp_path / 'config.yaml'
    path.write_text('task: pushing\nmodel:\n  paddding: true\n')
    with pytest.raises(ValueError, match='Unknown setting'):
        ExperimentConfig.from_yaml(path)
    path.write_text('task: pushing\nvlsa:\n  enabled: true\n')
    cfg = parse_config('sample', argv=['--config', str(path), '--no-vlsa'])
    assert cfg.vlsa_enabled is False


def test_paths_do_not_depend_on_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = ExperimentConfig.from_yaml(REPO / 'configs/inserting.yaml')
    assert Path(cfg.values['paths']['data_root']) == REPO / 'data'


def test_resume_conflicts_are_not_silently_ignored(tmp_path, monkeypatch):
    import fm_constraint.config as module
    monkeypatch.setattr(module, 'REPO', tmp_path)
    run = tmp_path / 'fm_constraint/runs/resume'
    run.mkdir(parents=True)
    (run / 'config.json').write_text(json.dumps({'args': {'task': 'pushing', 'lr': 0.1}}))
    path = tmp_path / 'config.yaml'
    path.write_text('task: pushing\ntrain:\n  resume: fm_constraint/runs/resume\n')
    with pytest.raises(ValueError, match='Resume setting conflicts'):
        ExperimentConfig.from_yaml(path).validate('train')


def test_legacy_import_collision_and_independent_import_orders():
    for order in [('train', 'sample', 'eval_pushing', 'eval_inserting'),
                  ('eval_inserting', 'eval_pushing', 'sample', 'train')]:
        code = ('from fm_constraint import import_baseline\n'
                f'for name in {order!r}:\n'
                ' m = import_baseline(name)\n'
                ' assert "/fm_baseline/" in m.__file__\n'
                'import sys\n'
                'assert "/fm_baseline/" in sys.modules["train"].__file__\n')
        subprocess.run([sys.executable, '-c', code], cwd=REPO, check=True, capture_output=True)
    code = ('import sys,types\n'
            'from fm_constraint import import_baseline\n'
            'sys.modules["train"]=types.ModuleType("train")\n'
            'try: import_baseline("sample")\n'
            'except ImportError: pass\n'
            'else: raise AssertionError("collision was not rejected")\n')
    subprocess.run([sys.executable, '-c', code], cwd=REPO, check=True, capture_output=True)


def test_symlink_output_cannot_escape_owned_tree(tmp_path, monkeypatch):
    import fm_constraint.config as module
    monkeypatch.setattr(module, 'REPO', tmp_path)
    (tmp_path / 'fm_constraint/runs').mkdir(parents=True)
    baseline = tmp_path / 'fm_baseline'
    baseline.mkdir()
    (tmp_path / 'fm_constraint/runs/escape').symlink_to(baseline, target_is_directory=True)
    path = tmp_path / 'config.yaml'
    path.write_text('task: pushing\npaths:\n  output_root: fm_constraint/runs/escape\n')
    with pytest.raises(ValueError, match='Outputs/resume'):
        ExperimentConfig.from_yaml(path)


def test_vlsa_requires_explicit_units_and_is_evaluation_only(tmp_path):
    values = yaml.safe_load((REPO / 'configs/pushing.yaml').read_text())
    values['vlsa'].update(enabled=True, gamma=10., margin_m=0.01, pusher_radius_m=0.02,
                          geometry='circle', workspace=None, action_norm_enabled=False,
                          command_delta_limit_m=0.01,
                          obstacles=[{'id': 'a', 'center_xy_m': [0.5, 0.2],
                                      'radius_m': 0.03, 'height_m': 0.2}])
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(values))
    cfg = ExperimentConfig.from_yaml(path)
    cfg.validate('eval')
    with pytest.raises(ValueError, match='execution filter'):
        cfg.validate('train')


@pytest.mark.parametrize('task', ['pushing', 'inserting'])
def test_rectangle_enabled_without_external_obstacles(task):
    cfg = ExperimentConfig.from_yaml(REPO / f'configs/{task}.yaml', {'vlsa.enabled': True})
    cfg.validate('eval')
    assert cfg.values['vlsa']['obstacles'] == []
    assert cfg.values['vlsa']['gamma'] is None
    assert cfg.values['vlsa']['workspace_enabled'] is True
    assert cfg.values['vlsa']['action_norm_enabled'] is True
    assert cfg.values['vlsa']['max_step_displacement_m'] == .012


@pytest.mark.parametrize('radius', [None, 0, -.01, float('nan'), float('inf'), True, '0.012'])
def test_invalid_action_norm_limits_are_rejected(radius):
    with pytest.raises(ValueError, match='max_step_displacement_m'):
        ExperimentConfig.from_yaml(REPO / 'configs/pushing.yaml', {'vlsa.max_step_displacement_m': radius})


def test_action_norm_off_allows_null_and_workspace_off_allows_no_rectangle():
    cfg = ExperimentConfig.from_yaml(REPO / 'configs/pushing.yaml', {
        'vlsa.enabled': True, 'vlsa.workspace_enabled': False, 'vlsa.workspace': None,
        'vlsa.action_norm_enabled': False, 'vlsa.max_step_displacement_m': None})
    cfg.validate('eval')


@pytest.mark.parametrize('workspace', [
    {'xmin': 0, 'x_max_m': 1, 'y_min_m': 0, 'y_max_m': 1},
    {'x_min_m': 1, 'x_max_m': 0, 'y_min_m': 0, 'y_max_m': 1},
    {'x_min_m': 0, 'x_max_m': .01, 'y_min_m': 0, 'y_max_m': 1},
    {'x_min_m': float('nan'), 'x_max_m': 1, 'y_min_m': 0, 'y_max_m': 1},
])
def test_invalid_rectangle_configuration(workspace):
    with pytest.raises(ValueError, match='[Ww]orkspace'):
        ExperimentConfig.from_yaml(REPO / 'configs/pushing.yaml', {'vlsa.workspace': workspace})
