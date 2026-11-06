"""시뮬레이터 없이 제약 QP를 검사하거나 학습 시연의 가속도 한계를 선정한다."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from configs.config import ExperimentConfig
from .constraints import PlanConstraints
from .qp import solve_qp


def calibrate_acceleration(data_dir, dt, sides):
    """학습 split만 읽어 정확한 L2·내접 다각형 기준의 필요한 한계를 구한다.

    시연 첫 행동 앞에는 0을 붙인다. 파일 경계를 이어서 가속도를 만들지 않는다.
    추천 한계는 모든 학습 차분을 허용하는 값의 1.05배를 0.1m/s²로 올림한다.
    """
    files = sorted((Path(data_dir) / 'train').glob('*.npz'))
    if not files:
        raise FileNotFoundError(f'No training files under {data_dir}')
    angles = 2 * np.pi * np.arange(sides) / sides
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    norms, required, starts = [], [], []
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
        with np.load(path) as data:
            actions = np.asarray(data['actions'], dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != 2 or not len(actions) or not np.isfinite(actions).all():
            raise ValueError(f'Invalid training actions: {path}')
        acceleration = np.diff(actions, axis=0, prepend=np.zeros((1, 2))) / dt**2
        lengths = np.linalg.norm(acceleration, axis=1)
        norms.append(lengths)
        starts.append(float(lengths[0]))
        required.append((acceleration @ normals.T).max(axis=1) / np.cos(np.pi / sides))
    norms, required = np.concatenate(norms), np.concatenate(required)
    limit = float(np.ceil(float(required.max()) * 1.05 * 10) / 10)
    return {'training_files': len(files), 'samples': int(norms.size), 'dt_s': dt,
            'polygon_sides': sides, 'training_sha256': digest.hexdigest(),
            'l2_quantiles_m_s2': dict(zip(('p50', 'p95', 'p99', 'p999', 'max'),
                                        map(float, np.quantile(norms, [.5, .95, .99, .999, 1])))),
            'episode_start_max_m_s2': max(starts),
            'polygon_required_max_m_s2': float(required.max()), 'recommended_max_m_s2': limit,
            'training_polygon_retained_fraction': float(np.mean(required <= limit)),
            'selection': 'ceil(max_training_polygon_requirement * 1.05 * 10) / 10'}


def calibrate_envelope(data_dir, sides, offset=0.):
    """학습 episode 시작 목표를 포함하는 고정 margin을 1cm 여유와 함께 선정한다."""
    files = sorted((Path(data_dir) / 'train').glob('*.npz'))
    if not files:
        raise FileNotFoundError(f'No training files under {data_dir}')
    angles = offset + 2 * np.pi * np.arange(sides) / sides
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    starts, outside, total = [], 0, 0
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
        with np.load(path) as data:
            obs = np.asarray(data['obs'], dtype=np.float64)
        blocks = obs[:, 4:].reshape(len(obs), -1, 4)[:, :, :2]
        center = blocks.mean(axis=1)
        radius = np.einsum('tij,kj->tik', blocks - center[:, None], normals).max(axis=(1, 2))
        needed = ((obs[:, :2] - center) @ normals.T).max(axis=1) - radius
        starts.append(float(needed[0]))
        outside += int(np.sum(needed > .10))
        total += len(obs)
    margin = float(np.ceil((max(0., max(starts)) + .01) * 100) / 100)
    return {'training_files': len(files), 'training_sha256': digest.hexdigest(),
            'n_sides': sides, 'normal_offset_rad': offset,
            'maximum_training_start_required_margin_m': max(starts),
            'selected_margin_m': margin,
            'selection': 'ceil((max(0,maximum_training_start_required_margin_m)+0.01)*100)/100',
            'training_initial_positions_retained_fraction': float(np.mean(np.array(starts) <= margin)),
            'original_0_10_margin_outside_fraction_all_training_observations': outside / total,
            'scope': 'Fixed per-task configuration calibrated from training starts only; no runtime expansion'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    calibration = parser.add_mutually_exclusive_group()
    calibration.add_argument('--calibrate-acceleration', action='store_true')
    calibration.add_argument('--calibrate-envelope', action='store_true')
    parser.add_argument('--n-obs', type=int, default=16)
    args = parser.parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    v = cfg.values
    data_dir = Path(v['paths']['data_root']) / f'{cfg.task}_sincos_v1'
    if args.calibrate_acceleration:
        report = calibrate_acceleration(data_dir, v['control']['dt_s'],
                                        v['constraints']['acceleration']['polygon_sides'])
    elif args.calibrate_envelope:
        obj = v['constraints']['object_envelope']
        report = calibrate_envelope(data_dir, obj['n_sides'], obj['normal_offset_rad'])
    else:
        if args.n_obs < 1:
            parser.error('--n-obs must be positive')
        reports = []
        for path in sorted((data_dir / 'val').glob('*.npz'))[:args.n_obs]:
            with np.load(path) as data:
                obs, actions = data['obs'][0], data['actions'][:v['model']['horizon']]
            constraints = PlanConstraints(obs, np.zeros(2), len(actions), v['constraints'], v['control']['dt_s'])
            solved = solve_qp(actions.ravel(), constraints.G, constraints.d, v['solver'])
            solved.pop('value')
            reports.append({'file': path.name, 'nominal': constraints.violation(actions), 'qp': solved})
        report = {'task': cfg.task, 'observations': reports}
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = Path(v['paths']['output_root']) / f'check_{cfg.task}_{stamp}'
    cfg.save(output)
    (output / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2, allow_nan=False))
    print(f'Saved: {output}')


if __name__ == '__main__':
    main()
