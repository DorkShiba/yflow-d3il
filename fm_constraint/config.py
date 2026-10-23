"""YAML configuration; all relative paths are relative to the repository root."""

import argparse
import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path

import yaml

from .vlsa import workspace_limits

REPO = Path(__file__).resolve().parents[1]


@dataclass
class ExperimentConfig:
    values: dict

    @classmethod
    def from_yaml(cls, path, overrides=None):
        with Path(path).open() as handle:
            supplied = yaml.safe_load(handle)
        if not isinstance(supplied, dict) or supplied.get('task') not in ('pushing', 'inserting'):
            raise ValueError('YAML must specify task: pushing or inserting')
        task = supplied['task']
        inserting = task == 'inserting'
        defaults = {
            'task': task, 'device': 'cpu',
            'paths': {
                'run_dir': f'fm_baseline/results/{task}_baseline',
                'output_root': 'fm_constraint/runs', 'data_root': 'data',
                'hardflow_root': '../HardFlow', 'd3il_root': '../d3il',
                'raw_root': f'/root/d3il-work/raw/{task}' + ('/all_data' if not inserting else ''),
                'contexts_file': 'fm_baseline/contexts/inserting_val.pkl',
            },
            'model': {'horizon': 16, 'dim': 64 if inserting else 32,
                      'dim_mults': [1, 2, 4, 8] if inserting else [1, 4, 8],
                      'attention': False, 'sn': False, 'sn_skip_output': False},
            'train': {'steps': 1000000, 'batch_size': 32, 'lr': 0.0002,
                      'scheduler': 'cosine', 'ema_decay': 0.995, 'seed': 42,
                      'log_every': 5000, 'save_every': 25000, 'val_batches': 8,
                      'val_seed': 1234, 'time_limit_hours': 0.0, 'tag': 'constraint',
                      'resume': None},
            'sample': {'n_windows': 256, 'batch_size': 64, 'split': 'val',
                       'seed': 0, 'no_save': True},
            'eval': {'checkpoint': 'baseline_ema.pt', 'weights': 'ema',
                     'ode_steps': 10, 'execute_steps': 8,
                     'max_steps': 2000 if inserting else 400,
                     'n_contexts': 0 if inserting else 30, 'n_traj': 4,
                     'seed': 0, 'comparison': 'none', 'check_obs': 0,
                     'export_contexts': None,
                     **({'split': 'val'} if inserting else {'contexts': 'test'})},
            'vlsa': {'enabled': False, 'geometry': 'rectangle', 'gamma': None,
                     'margin_m': None, 'pusher_radius_m': None,
                     'command_delta_limit_m': None, 'obstacles': [], 'workspace': None,
                     'workspace_enabled': True, 'action_norm_enabled': True,
                     'max_step_displacement_m': 0.012},
        }
        values = copy.deepcopy(defaults)
        for key, value in supplied.items():
            if key not in values:
                raise ValueError(f'Unknown setting: {key}')
            if isinstance(values[key], dict):
                if not isinstance(value, dict):
                    raise ValueError(f'{key} must be a mapping')
                unknown = value.keys() - values[key].keys()
                if unknown:
                    raise ValueError(f'Unknown setting: {key}.{sorted(unknown)[0]}')
                values[key].update(value)
            else:
                values[key] = value
        for dotted, value in (overrides or {}).items():
            parts = dotted.split('.')
            if len(parts) == 1 and dotted in ('device',):
                values[dotted] = value
            elif len(parts) == 2 and parts[0] in defaults and isinstance(defaults[parts[0]], dict) and parts[1] in defaults[parts[0]]:
                values[parts[0]][parts[1]] = value
            else:
                raise ValueError(f'Unknown or non-overridable setting: {dotted}')
        # Check scalar types before resolving paths; bool is not a numeric value.
        for group, fields in defaults.items():
            if not isinstance(fields, dict):
                if not isinstance(values[group], str):
                    raise ValueError(f'{group} must be a string')
                continue
            for key, default in fields.items():
                value = values[group][key]
                if default is None or key == 'max_step_displacement_m':
                    continue
                if isinstance(default, float):
                    valid = type(value) in (int, float) and math.isfinite(value)
                else:
                    valid = type(value) is type(default)
                if not valid:
                    raise ValueError(f'Invalid type for {group}.{key}')
        for key, value in values['paths'].items():
            values['paths'][key] = str((REPO / value).resolve())
        for group, key in [('train', 'resume'), ('eval', 'export_contexts')]:
            value = values[group][key]
            if value is not None:
                if not isinstance(value, str) or not value:
                    raise ValueError(f'{group}.{key} must be a nonempty path or null')
                values[group][key] = str((REPO / value).resolve())
        cfg = cls(values)
        cfg.validate()
        return cfg

    @property
    def task(self):
        return self.values['task']

    @property
    def vlsa_enabled(self):
        return self.values['vlsa']['enabled']

    def validate(self, command=None):
        v = self.values
        model, train, ev, safety = (v[k] for k in ('model', 'train', 'eval', 'vlsa'))
        for group, keys in [('model', ['horizon', 'dim']),
                            ('train', ['steps', 'batch_size', 'log_every', 'val_batches']),
                            ('sample', ['n_windows', 'batch_size']),
                            ('eval', ['ode_steps', 'execute_steps', 'max_steps', 'n_traj'])]:
            for key in keys:
                if v[group][key] <= 0:
                    raise ValueError(f'{group}.{key} must be positive')
        if ev['ode_steps'] < 2 or ev['execute_steps'] > model['horizon']:
            raise ValueError('Require ode_steps >= 2 and execute_steps <= horizon')
        for group, keys in [('train', ['seed', 'val_seed', 'save_every', 'time_limit_hours']),
                            ('sample', ['seed']), ('eval', ['seed', 'n_contexts', 'check_obs'])]:
            if any(v[group][key] < 0 for key in keys):
                raise ValueError(f'{group}: counts, seeds and time limits must be nonnegative')
        mults = model['dim_mults']
        if (not mults or any(type(x) is not int or x <= 0 for x in mults)
                or model['dim'] % 8 or model['horizon'] % (2 ** (len(mults) - 1))):
            raise ValueError('Invalid U-Net dim/dim_mults/horizon (GroupNorm and downsampling)')
        if train['lr'] <= 0 or not 0 <= train['ema_decay'] < 1 or train['scheduler'] not in ('none', 'cosine'):
            raise ValueError('Invalid lr, EMA decay or scheduler')
        if ev['weights'] not in ('ema', 'model') or ev['comparison'] not in ('none', 'stop'):
            raise ValueError('Invalid weights or comparison')
        if v['sample']['split'] not in ('train', 'val'):
            raise ValueError('sample.split must be train or val')
        if self.task == 'pushing' and ev['contexts'] not in ('train', 'test'):
            raise ValueError('eval.contexts must be train or test')
        if self.task == 'inserting' and ev['split'] not in ('train', 'val'):
            raise ValueError('eval.split must be train or val')
        if not isinstance(ev['checkpoint'], str) or not ev['checkpoint']:
            raise ValueError('eval.checkpoint must be a nonempty path')
        if safety['geometry'] not in ('circle', 'rectangle'):
            raise ValueError('Only circle and rectangle geometry are supported')
        radius = safety['max_step_displacement_m']
        if radius is not None and (type(radius) not in (int, float) or not math.isfinite(radius) or radius <= 0):
            raise ValueError('max_step_displacement_m must be a finite positive length in meters')
        if safety['action_norm_enabled'] and radius is None:
            raise ValueError('action_norm_enabled requires max_step_displacement_m')
        if safety['action_norm_enabled'] and safety['geometry'] != 'rectangle':
            raise ValueError('Action norm constraint currently requires rectangle geometry')
        if safety['workspace'] is not None:
            workspace_limits(safety['workspace'], safety['margin_m'])
        if safety['geometry'] == 'rectangle' and safety['obstacles']:
            raise ValueError('Rectangle command fence does not include circular obstacles')
        ids = set()
        for obstacle in safety['obstacles']:
            if not isinstance(obstacle, dict) or set(obstacle) != {'id', 'center_xy_m', 'radius_m', 'height_m'}:
                raise ValueError('Obstacle requires id, center_xy_m, radius_m, height_m')
            if not isinstance(obstacle['id'], str) or not obstacle['id'] or obstacle['id'] in ids:
                raise ValueError('Obstacle IDs must be nonempty and unique')
            ids.add(obstacle['id'])
            xy = obstacle['center_xy_m']
            if not isinstance(xy, list) or len(xy) != 2:
                raise ValueError('center_xy_m must have two coordinates')
            for number in [*xy, obstacle['radius_m'], obstacle['height_m']]:
                if type(number) not in (int, float) or not math.isfinite(number):
                    raise ValueError('Obstacle geometry must be finite')
            if obstacle['radius_m'] <= 0 or obstacle['height_m'] <= 0:
                raise ValueError('Obstacle radius and height must be positive')
        if self.vlsa_enabled or ev['comparison'] == 'stop':
            if self.vlsa_enabled and ev['comparison'] != 'none':
                raise ValueError('vlsa and stop comparison are mutually exclusive')
            required = ['margin_m', 'command_delta_limit_m']
            if safety['geometry'] == 'circle':
                required += ['gamma', 'pusher_radius_m']
            for key in required:
                number = safety[key]
                if type(number) not in (int, float) or not math.isfinite(number) or number < 0 or (key != 'margin_m' and number == 0):
                    raise ValueError(f'vlsa.{key} must be explicitly set to a valid physical value')
            if safety['geometry'] == 'circle' and not safety['obstacles']:
                raise ValueError('An enabled safety experiment requires obstacles')
            if safety['geometry'] == 'rectangle' and safety['workspace_enabled']:
                workspace_limits(safety['workspace'], safety['margin_m'])
        self.require_output(v['paths']['output_root'])
        if command in ('train', 'sample') and self.vlsa_enabled:
            raise ValueError('VLSA is an execution filter; use an evaluation entry point')
        if command in ('train', 'sample') and (ev['comparison'] != 'none' or ev['check_obs'] or ev['export_contexts']):
            raise ValueError('Comparison, observation checks and context export require evaluation')
        if command == 'train':
            if '/' in train['tag'] or '\\' in train['tag'] or train['tag'] in ('', '.', '..'):
                raise ValueError('train.tag must be a plain run label')
            if train['resume']:
                self.require_output(train['resume'])
                saved = json.loads((Path(train['resume']) / 'config.json').read_text())
                requested = {**train, **model, 'task': self.task}
                for key, value in saved['args'].items():
                    if key in requested and key not in ('resume', 'time_limit_hours'):
                        expected = ','.join(map(str, mults)) if key == 'dim_mults' else requested[key]
                        if value != expected:
                            raise ValueError(f'Resume setting conflicts with saved args: {key}')
        if command in ('sample', 'eval'):
            self.check_checkpoint()
            if command == 'sample' and not v['sample']['no_save']:
                self.require_output(v['paths']['run_dir'])
        if ev['export_contexts']:
            if self.task != 'inserting':
                raise ValueError('Context export is only available for inserting')
            self.require_output(ev['export_contexts'])

    def require_output(self, path):
        path = Path(path).resolve()
        owned = (REPO / 'fm_constraint/runs').resolve()
        if not path.is_relative_to(owned):
            raise ValueError(f'Outputs/resume must be inside {owned}: {path}')
        # Resolve symlinks and also protect custom input directories.
        for source in [REPO / 'fm_baseline', REPO / 'data', REPO / 'preprocess',
                       *(Path(self.values['paths'][key]) for key in ('data_root', 'hardflow_root', 'd3il_root'))]:
            if path.is_relative_to(source.resolve()):
                raise ValueError(f'Output overlaps a protected input: {source}')
        return path

    def check_checkpoint(self):
        root = Path(self.values['paths']['run_dir'])
        saved = json.loads((root / 'config.json').read_text())
        expected_dims = (12 if self.task == 'pushing' else 16, 2)
        if (saved['task'] != self.task or (saved['obs_dim'], saved['act_dim']) != expected_dims
                or saved.get('transition_order') != 'actions_then_obs'):
            raise ValueError('Checkpoint task, dimensions or transition order conflict')
        expected = {**saved['model'], 'horizon': saved['horizon'],
                    'sn': saved['spectral_norm'], 'sn_skip_output': saved['spectral_norm_skip_output']}
        for key, value in self.values['model'].items():
            if value != expected[key]:
                raise ValueError(f'Checkpoint model setting conflicts: {key}')
        checkpoint = (root / self.values['eval']['checkpoint']).resolve()
        if not checkpoint.is_file() or not (root / 'normalizer.npz').is_file():
            raise FileNotFoundError(f'Missing checkpoint or normalizer under {root}')
        return saved

    def baseline_argv(self, command):
        v, paths = self.values, self.values['paths']
        if command == 'train':
            options = {**v['train'], **v['model'], 'task': self.task, 'device': v['device'],
                       'data_root': paths['data_root'], 'hardflow_root': paths['hardflow_root'],
                       'out_root': paths['output_root']}
            options['dim_mults'] = ','.join(map(str, options['dim_mults']))
        elif command == 'sample':
            options = {**v['sample'], 'run_dir': paths['run_dir'],
                       'hardflow_root': paths['hardflow_root'],
                       'data_dir': str(Path(paths['data_root']) / f'{self.task}_sincos_v1'),
                       'device': v['device'], 'ode_steps': v['eval']['ode_steps'],
                       'ckpt': v['eval']['checkpoint'], 'weights': v['eval']['weights']}
        else:
            raise ValueError(f'Unknown baseline command: {command}')
        argv = []
        for key, value in options.items():
            if value is None or value is False:
                continue
            argv.append('--' + key.replace('_', '-'))
            if value is not True:
                argv.append(str(value))
        return argv

    def save(self, directory):
        directory = self.require_output(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'resolved_config.yaml').write_text(yaml.safe_dump(self.values, sort_keys=False))


def parse_config(command, task=None, argv=None):
    parser = argparse.ArgumentParser(description=f'FM constraint {command}')
    parser.add_argument('--config', required=True)
    parser.add_argument('--set', action='append', default=[], metavar='SECTION.KEY=VALUE')
    parser.add_argument('--device')
    parser.add_argument('--vlsa', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--check-obs', type=int)
    parser.add_argument('--export-contexts')
    args = parser.parse_args(argv)
    overrides = {}
    for item in args.set:
        if '=' not in item:
            parser.error('--set requires SECTION.KEY=VALUE')
        key, value = item.split('=', 1)
        overrides[key] = yaml.safe_load(value)
    for key, value in [('device', args.device), ('vlsa.enabled', args.vlsa),
                       ('eval.check_obs', args.check_obs), ('eval.export_contexts', args.export_contexts)]:
        if value is not None:
            overrides[key] = value
    cfg = ExperimentConfig.from_yaml(args.config, overrides)
    if task and cfg.task != task:
        raise ValueError(f'This entry point requires task: {task}')
    cfg.validate(command)
    return cfg
