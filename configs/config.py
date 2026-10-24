"""통합 FM 설정: 중첩 YAML과 CLI를 검증하고 경로를 저장소 기준으로 해석한다."""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


def _merge_settings(target, supplied, prefix=''):
    """알려진 키만 재귀 병합한다. 문자열 false를 bool로 묵시 변환하지 않는다."""
    if not isinstance(supplied, dict):
        raise ValueError(f'{prefix or "config"} must be a mapping')
    for key, value in supplied.items():
        name = f'{prefix}.{key}' if prefix else key
        if key not in target:
            raise ValueError(f'Unknown setting: {name}')
        default = target[key]
        if isinstance(default, dict):
            _merge_settings(default, value, name)
        else:
            if default is not None and name != 'methods.pc.eps':
                valid = (type(value) in (int, float) and math.isfinite(value)
                         if isinstance(default, float) else type(value) is type(default))
                if not valid:
                    raise ValueError(f'Invalid type for {name}')
            target[key] = value


@dataclass
class ExperimentConfig:
    values: dict

    @classmethod
    def from_yaml(cls, path, overrides=None):
        """CLI > YAML > 기본값 순서로 병합한다. 상대 경로는 REPO 기준이다."""
        with Path(path).open() as handle:
            supplied = yaml.safe_load(handle)
        if not isinstance(supplied, dict) or supplied.get('task') not in ('pushing', 'inserting'):
            raise ValueError('YAML must specify task: pushing or inserting')
        task = supplied['task']
        inserting = task == 'inserting'
        values = {
            'schema_version': 2, 'task': task, 'device': 'cpu',
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
                     'execute_steps': 8, 'max_steps': 2000 if inserting else 400,
                     'n_contexts': 0 if inserting else 30, 'n_traj': 4, 'seed': 0,
                     'check_obs': 0, 'export_contexts': None,
                     **({'split': 'val'} if inserting else {'contexts': 'test'})},
            'control': {'dt_s': 0.035},
            'sampling': {'predict_steps': 10},
            'methods': {
                'projection': {'enabled': False, 'mode': 'every', 'scope': 'predict', 'skip_first': 0},
                'vlsa': {'enabled': False, 'scope': 'terminal_plan', 'execution_recheck': False,
                         'invalidate_queue_on_change': True},
                'pc': {'enabled': False, 'correct_steps': 9, 'alpha': 2.0,
                       'rho': 0.9, 'delta': 0.01, 'eps': 'auto'},
            },
            'constraints': {
                'workspace': {'enabled': True, 'geometry': 'rectangle', 'source': 'research_bounds',
                              'low_m': [0.00 if inserting else 0.20, -0.40],
                              'high_m': [0.85, 0.75 if inserting else 0.50], 'margin_m': 0.01},
                'acceleration': {'enabled': True, 'max_m_s2': None,
                                 'approximation': 'inscribed_polygon', 'polygon_sides': 32},
                'object_envelope': {'enabled': True, 'geometry': 'regular_polygon', 'n_sides': 8,
                                    'normal_offset_rad': 0.0, 'center': 'mean_blocks',
                                    'include': 'all_blocks', 'footprint': 'centers',
                                    'margin_m': 0.10, 'update': 'per_plan', 'auto_expand': False},
            },
            'solver': {'formulation': 'linear_inequality_qp', 'backend': 'osqp', 'dtype': 'float64',
                       'feasibility_tolerance_m': 1e-7, 'intervention_tolerance_m': 1e-7,
                       'eps_abs': 1e-9, 'eps_rel': 1e-9, 'max_iter': 20000,
                       'on_failure': 'abort_episode'},
        }
        _merge_settings(values, supplied)
        for dotted, value in (overrides or {}).items():
            parts = dotted.split('.')
            node = values
            for part in parts[:-1]:
                if part not in node or not isinstance(node[part], dict):
                    raise ValueError(f'Unknown setting: {dotted}')
                node = node[part]
            _merge_settings(node, {parts[-1]: value}, '.'.join(parts[:-1]))
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
    def methods_enabled(self):
        return any(method['enabled'] for method in self.values['methods'].values())

    def validate(self, command=None):
        """형상·단위·지원 모드를 확인한다. 미지원 설정을 자동 대체하지 않는다."""
        v = self.values
        model, train, ev = (v[k] for k in ('model', 'train', 'eval'))
        if v['schema_version'] != 2:
            raise ValueError('Require schema_version: 2; migrate legacy settings explicitly')
        for group, keys in [('model', ['horizon', 'dim']),
                            ('train', ['steps', 'batch_size', 'log_every', 'val_batches']),
                            ('sample', ['n_windows', 'batch_size']),
                            ('eval', ['execute_steps', 'max_steps', 'n_traj']),
                            ('sampling', ['predict_steps']), ('solver', ['max_iter'])]:
            if any(v[group][key] <= 0 for key in keys):
                raise ValueError(f'{group}: counts must be positive')
        if ev['execute_steps'] > model['horizon']:
            raise ValueError('Require execute_steps <= horizon')
        for group, keys in [('train', ['seed', 'val_seed', 'save_every', 'time_limit_hours']),
                            ('sample', ['seed']), ('eval', ['seed', 'n_contexts', 'check_obs'])]:
            if any(v[group][key] < 0 for key in keys):
                raise ValueError(f'{group}: counts and seeds must be nonnegative')
        mults = model['dim_mults']
        if (not mults or any(type(x) is not int or x <= 0 for x in mults)
                or model['dim'] % 8 or model['horizon'] % (2 ** (len(mults) - 1))):
            raise ValueError('Invalid U-Net dim/dim_mults/horizon')
        if train['lr'] <= 0 or not 0 <= train['ema_decay'] < 1 or train['scheduler'] not in ('none', 'cosine'):
            raise ValueError('Invalid lr, EMA decay or scheduler')
        if ev['weights'] not in ('ema', 'model') or v['sample']['split'] not in ('train', 'val'):
            raise ValueError('Invalid weights or split')
        if self.task == 'pushing' and ev['contexts'] not in ('train', 'test'):
            raise ValueError('eval.contexts must be train or test')
        if self.task == 'inserting' and ev['split'] not in ('train', 'val'):
            raise ValueError('eval.split must be train or val')
        if not isinstance(ev['checkpoint'], str) or not ev['checkpoint']:
            raise ValueError('eval.checkpoint must be a nonempty path')
        if v['control']['dt_s'] <= 0:
            raise ValueError('control.dt_s must be positive')
        # 현재 D3IL은 1ms 내부 스텝 35회로 명령 하나를 실행한다. 다른 dt는 지원하지 않는다.
        if command == 'eval' and not math.isclose(v['control']['dt_s'], 0.035, abs_tol=1e-12):
            raise ValueError('D3IL evaluation requires control.dt_s=0.035')
        proj, vlsa, pc = (v['methods'][k] for k in ('projection', 'vlsa', 'pc'))
        if proj['mode'] not in ('every', 'final') or proj['scope'] != 'predict' or proj['skip_first'] < 0:
            raise ValueError('Invalid projection mode/scope/skip_first')
        if vlsa['scope'] != 'terminal_plan' or not vlsa['invalidate_queue_on_change']:
            raise ValueError('VLSA requires terminal_plan and queue invalidation on change')
        if proj['enabled'] and proj['mode'] == 'final' and vlsa['enabled']:
            raise ValueError('Projection final and terminal VLSA are identical; select one')
        if pc['correct_steps'] < 1 or pc['alpha'] <= 0 or not 0 < pc['rho'] < 1 or pc['delta'] <= 0:
            raise ValueError('Invalid PC steps/alpha/rho/delta')
        if pc['eps'] != 'auto' and (type(pc['eps']) not in (int, float)
                                   or not math.isfinite(pc['eps']) or pc['eps'] <= 0):
            raise ValueError('PC eps must be auto or a positive number')
        ws, acc, obj = (v['constraints'][k] for k in ('workspace', 'acceleration', 'object_envelope'))
        if ws['geometry'] != 'rectangle' or ws['margin_m'] < 0:
            raise ValueError('Workspace requires rectangle and nonnegative margin')
        if any(not isinstance(ws[k], list) or len(ws[k]) != 2 or
               any(type(x) not in (int, float) or not math.isfinite(x) for x in ws[k])
               for k in ('low_m', 'high_m')):
            raise ValueError('Workspace low/high must be finite XY lists')
        if any(lo + ws['margin_m'] >= hi - ws['margin_m'] for lo, hi in zip(ws['low_m'], ws['high_m'])):
            raise ValueError('Workspace must have positive width after margin')
        if acc['approximation'] != 'inscribed_polygon' or acc['polygon_sides'] < 4 or acc['polygon_sides'] % 2:
            raise ValueError('Acceleration polygon_sides must be even and >=4')
        limit = acc['max_m_s2']
        if limit is not None and (type(limit) not in (int, float) or not math.isfinite(limit) or limit <= 0):
            raise ValueError('Acceleration max_m_s2 must be finite and positive')
        if acc['enabled'] and limit is None:
            raise ValueError('Acceleration max_m_s2 must be set explicitly')
        if (obj['geometry'] != 'regular_polygon' or obj['n_sides'] < 3 or obj['margin_m'] <= 0
                or obj['center'] != 'mean_blocks' or obj['include'] != 'all_blocks'
                or obj['footprint'] != 'centers' or obj['update'] != 'per_plan' or obj['auto_expand']):
            raise ValueError('Object envelope requires a fixed-per-plan regular polygon around all block centers')
        solver = v['solver']
        if (solver['backend'] != 'osqp' or solver['dtype'] != 'float64'
                or solver['formulation'] != 'linear_inequality_qp' or solver['on_failure'] != 'abort_episode'):
            raise ValueError('Only float64 OSQP linear inequalities with abort_episode are supported')
        if any(solver[k] <= 0 for k in ('eps_abs', 'eps_rel', 'feasibility_tolerance_m', 'intervention_tolerance_m')):
            raise ValueError('Solver tolerances must be positive')
        self.require_output(v['paths']['output_root'])
        if command == 'train' and self.methods_enabled:
            raise ValueError('Training-free methods must be disabled during train')
        if command in ('train', 'sample') and (ev['check_obs'] or ev['export_contexts']):
            raise ValueError('Observation checks/context export require evaluation')
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
                raise ValueError('sample.no_save must be true; input runs are read-only')
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
                       'device': v['device'], 'ode_steps': v['sampling']['predict_steps'],
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
    """세 방법의 CLI 스위치와 임의 깊이의 --set 키를 명시적으로 병합한다."""
    parser = argparse.ArgumentParser(description=f'FM constraint {command}')
    parser.add_argument('--config', required=True)
    parser.add_argument('--set', action='append', default=[], metavar='SECTION.KEY=VALUE')
    parser.add_argument('--device')
    for method in ('projection', 'vlsa', 'pc'):
        parser.add_argument('--' + method, action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--check-obs', type=int)
    parser.add_argument('--export-contexts')
    args = parser.parse_args(argv)
    overrides = {}
    for item in args.set:
        if '=' not in item:
            parser.error('--set requires SECTION.KEY=VALUE')
        key, value = item.split('=', 1)
        overrides[key] = yaml.safe_load(value)
    for key, value in [('device', args.device), ('eval.check_obs', args.check_obs),
                       ('eval.export_contexts', args.export_contexts),
                       *[(f'methods.{name}.enabled', getattr(args, name))
                         for name in ('projection', 'vlsa', 'pc')]]:
        if value is not None:
            overrides[key] = value
    cfg = ExperimentConfig.from_yaml(args.config, overrides)
    if task and cfg.task != task:
        raise ValueError(f'This entry point requires task: {task}')
    cfg.validate(command)
    return cfg
