"""태스크별 train KDE 캐시와 상태 위치 전용 potential guidance."""

from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from time import monotonic, perf_counter, sleep
import zipfile

import numpy as np
import torch

FORMAT_VERSION = 1
DENSITY_KEYS = ('bandwidth_m', 'max_support', 'seed', 'grid_step_m',
                'grid_padding_m', 'density_quantile')


def file_hash(path):
    """파일을 메모리에 통째로 올리지 않고 내용 hash를 계산한다."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _array_hash(*arrays):
    digest = hashlib.sha256()
    for value in arrays:
        digest.update(str((value.shape, str(value.dtype))).encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _log_density(points, support, bandwidth, chunk=256):
    """CPU float64에서 안정적인 Gaussian log-density를 계산한다."""
    out = []
    constant = math.log(len(support)) + math.log(2 * math.pi * bandwidth**2)
    for start in range(0, len(points), chunk):
        difference = points[start:start + chunk, None, :] - support[None, :, :]
        logits = -np.square(difference).sum(-1) / (2 * bandwidth**2)
        maximum = logits.max(-1)
        out.append(maximum + np.log(np.exp(logits - maximum[:, None]).sum(-1)) - constant)
    return np.concatenate(out)


def _audit_train(root, files, manifest):
    """모든 train 전이의 규격·단위·시연 내 일관성을 검사하고 위치를 모은다."""
    obs_dim = manifest['obs_dim']
    expected_fields = ['robot.des_c_pos.x', 'robot.des_c_pos.y', 'robot.c_pos.x', 'robot.c_pos.y']
    blocks = ['red-box', 'green-box'] if obs_dim == 12 else ['box-1', 'box-2', 'box-3']
    expected_fields += [f'{name}.{field}' for name in blocks for field in ('x', 'y', 'sin_yaw', 'cos_yaw')]
    if (obs_dim not in (12, 16) or manifest['action_dim'] != 2
            or manifest['obs_fields'] != expected_fields
            or manifest['action_fields'] != ['delta_des_c_pos.x', 'delta_des_c_pos.y'] or manifest['normalized'] is not False
            or not math.isclose(manifest['recording_interval_seconds'], .035, abs_tol=1e-9)
            or manifest['action_definition'] != 'des_c_pos[t+1, :2] - des_c_pos[t, :2]'):
        raise ValueError('PF2MP requires raw D3IL sincos XY-delta data')
    listed = {row['file'] for row in manifest['splits']['train']}
    train_sources = [row['source'] for row in manifest['splits']['train']]
    if len(listed) != len(train_sources) or len(set(train_sources)) != len(train_sources):
        raise ValueError('Duplicate train manifest entry/source')
    transitions = {row['file']: row['transitions'] for row in manifest['splits']['train']}
    if listed != {str(f.relative_to(root)) for f in files}:
        raise ValueError('Train files do not match manifest')
    val_sources = {row['source'] for row in manifest['splits']['val']}
    if val_sources.intersection(row['source'] for row in manifest['splits']['train']):
        raise ValueError('Train/val source overlap')
    train_groups = {row['context_group'] for row in manifest['splits']['train'] if 'context_group' in row}
    if train_groups.intersection(row.get('context_group') for row in manifest['splits']['val']):
        raise ValueError('Train/val context group overlap')
    points, max_delta, max_tracking, max_angle = [], 0., 0., 0.
    for path in files:
        with np.load(path, allow_pickle=False) as record:
            obs, action, nxt = (record[key] for key in ('obs', 'actions', 'next_obs'))
            time, next_time = (record[key] for key in ('timestamps', 'next_timestamps'))
            n = len(obs)
            if n != transitions[str(path.relative_to(root))]:
                raise ValueError(f'Transition count differs from manifest: {path}')
            if n == 0 or obs.shape != (n, obs_dim) or action.shape != (n, 2) or nxt.shape != obs.shape:
                raise ValueError(f'Invalid data shape: {path}')
            if time.shape != (n,) or next_time.shape != (n,):
                raise ValueError(f'Invalid timestamps: {path}')
            if any(x.dtype.kind != 'f' or not np.isfinite(x).all() for x in (obs, action, nxt, time, next_time)):
                raise ValueError(f'Invalid dtype/nonfinite train data: {path}')
            if not np.allclose(next_time - time, .035, atol=1e-7, rtol=0):
                raise ValueError(f'Invalid recording interval: {path}')
            if n > 1 and (not np.allclose(np.diff(time), .035, atol=1e-7, rtol=0)
                          or not np.allclose(nxt[:-1], obs[1:], atol=1e-6, rtol=0)):
                raise ValueError(f'Broken episode continuity: {path}')
            residual = float(np.abs(obs[:, :2] + action - nxt[:, :2]).max())
            angle_error = max(float(np.abs(arr[:, 6::4]**2 + arr[:, 7::4]**2 - 1).max()) for arr in (obs, nxt))
            if residual > 1e-6 or angle_error > 1e-5:
                raise ValueError(f'Invalid action/sincos: {path}')
            max_delta, max_angle = max(max_delta, residual), max(max_angle, angle_error)
            max_tracking = max(max_tracking, float(np.linalg.norm(obs[:, :2] - obs[:, 2:4], axis=-1).max()))
            points.append(obs[:, :2].astype(np.float64))
    points = np.concatenate(points)
    return points, {'train_files': len(files), 'train_rows': len(points), 'obs_dim': obs_dim,
                    'max_delta_error_m': max_delta, 'max_sincos_error': max_angle,
                    'max_tracking_distance_m': max_tracking,
                    'xy_min_m': points.min(0).tolist(), 'xy_max_m': points.max(0).tolist()}


@contextmanager
def _cache_lock(path, timeout):
    """프로세스 종료 시 OS가 해제하는 파일 lock으로 중복 생성을 막는다."""
    with path.open('a') as handle:
        started = monotonic()
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if monotonic() - started >= timeout:
                    raise TimeoutError(f'PF2MP cache lock timeout: {path}')
                sleep(.05)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _load_cache(path, identity, fingerprint):
    with np.load(path, allow_pickle=False) as record:
        support, safe = record['support'], record['safe_points']
        metadata = json.loads(str(record['metadata'].item()))
    if metadata['identity'] != identity or metadata['fingerprint'] != fingerprint:
        raise ValueError('PF2MP cache identity mismatch')
    if any(x.ndim != 2 or x.shape[1] != 2 or not len(x) or x.dtype != np.float64
           or not np.isfinite(x).all() for x in (support, safe)):
        raise ValueError('PF2MP cache contains invalid points')
    if metadata['arrays_sha256'] != _array_hash(support, safe):
        raise ValueError('PF2MP cache checksum mismatch')
    if not math.isfinite(metadata['log_threshold']):
        raise ValueError('PF2MP cache threshold is invalid')
    return support, safe, metadata


def ensure_potential_cache(dataset_dir, settings):
    """PF on에서만 호출한다. train/설정 hash가 같으면 저장된 KDE를 재사용한다."""
    started = perf_counter()
    root = Path(dataset_dir).resolve()
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    files = sorted((root / 'train').rglob('*.npz'))
    if not files:
        raise ValueError(f'No train NPZ for PF2MP: {root}')
    sources = [{'file': str(f.relative_to(root)), 'sha256': file_hash(f)} for f in files]
    identity = {'format_version': FORMAT_VERSION, 'feature': 'obs.command_xy.raw_m',
                'manifest_sha256': file_hash(manifest_path), 'train_sources': sources,
                'density': {key: settings[key] for key in DENSITY_KEYS}}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    directory = root / 'pf2mp_cache'
    directory.mkdir(exist_ok=True)
    path = directory / f'{fingerprint}.npz'
    event, corrupt = 'hit', None
    build_ms = load_ms = 0.
    with _cache_lock(directory / f'{fingerprint}.lock', settings['lock_timeout_s']):
        try:
            load_started = perf_counter()
            support, safe, metadata = _load_cache(path, identity, fingerprint)
        except FileNotFoundError:
            event = 'miss'
        except (ValueError, TypeError, KeyError, EOFError, UnicodeDecodeError, zipfile.BadZipFile) as error:
            corrupt = str(error)
            path.rename(path.with_suffix(f'.corrupt.{os.getpid()}.{monotonic():.6f}.npz'))
            event = 'rebuilt_corrupt'
        load_ms = 1000 * (perf_counter() - load_started)
        if event != 'hit':
            build_start = perf_counter()
            points, audit = _audit_train(root, files, manifest)
            # 지역 RNG를 사용하여 추론 torch/numpy RNG와 완전히 분리한다.
            rows = np.random.default_rng(settings['seed']).choice(
                len(points), size=min(settings['max_support'], len(points)), replace=False)
            support = points[rows].copy()
            bandwidth = settings['bandwidth_m']
            threshold = float(np.quantile(_log_density(points, support, bandwidth), settings['density_quantile']))
            step = settings['grid_step_m']
            low = points.min(0) - settings['grid_padding_m']
            high = points.max(0) + settings['grid_padding_m']
            axes = [np.arange(low[i], high[i] + step, step) for i in range(2)]
            if len(axes[0]) * len(axes[1]) > 1_000_000:
                raise ValueError('PF2MP grid exceeds one million cells; increase grid_step_m')
            grid = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 2)
            mask = _log_density(grid, support, bandwidth) > threshold
            safe = grid[mask].copy()
            if not len(safe):
                raise ValueError('PF2MP threshold leaves no safe grid points')
            on_boundary = ((grid[:, 0] == axes[0][0]) | (grid[:, 0] == axes[0][-1])
                           | (grid[:, 1] == axes[1][0]) | (grid[:, 1] == axes[1][-1]))
            if np.any(mask & on_boundary):
                raise ValueError('PF2MP high-density set touches grid boundary; increase grid_padding_m')
            # 생성 도중 source가 바뀌었으면 이전 fingerprint로 게시하지 않는다.
            if any(file_hash(root / row['file']) != row['sha256'] for row in sources) or file_hash(manifest_path) != identity['manifest_sha256']:
                raise ValueError('PF2MP training data changed during cache construction')
            metadata = {'identity': identity, 'fingerprint': fingerprint, 'audit': audit,
                        'log_threshold': threshold, 'grid_low_m': low.tolist(),
                        'grid_shape': [len(a) for a in axes],
                        'arrays_sha256': _array_hash(support, safe)}
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=directory, suffix='.npz', delete=False) as handle:
                    temporary = Path(handle.name)
                    np.savez_compressed(handle, support=support, safe_points=safe,
                                        metadata=json.dumps(metadata, sort_keys=True))
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            build_ms = 1000 * (perf_counter() - build_start)
    report = {'path': str(path), 'fingerprint': fingerprint, 'artifact_sha256': file_hash(path),
              'event': event, 'corruption': corrupt, 'build_ms': build_ms, 'load_ms': load_ms,
              'prepare_ms': 1000 * (perf_counter() - started), 'audit': metadata['audit'],
              'support_points': len(support), 'safe_grid_points': len(safe),
              'log_threshold': metadata['log_threshold']}
    return support, safe, report


class PotentialField:
    """에너지와 해석 gradient를 raw XY에서 평가하고 상태 위치에만 삽입한다."""

    def __init__(self, support, safe_points, settings, device='cpu', cache_report=None):
        self.support = torch.as_tensor(support, dtype=torch.float64, device=device)
        self.safe = torch.as_tensor(safe_points, dtype=torch.float64, device=device)
        self.bandwidth = settings['bandwidth_m']
        self.alpha = settings['distance_alpha']
        self.chunk_size = settings['query_chunk_size']
        self.cache_report = cache_report

    @torch.no_grad()
    def evaluate(self, q):
        """Potential 합이 아닌 점별 값·gradient를 반환한다. 거리0의 미분은0이다."""
        shape = q.shape
        values, gradients = [], []
        flat = q.reshape(-1, 2).double()
        if not torch.isfinite(flat).all():
            raise ValueError('Nonfinite PF2MP query')
        for points in flat.split(self.chunk_size):
            diff = points[:, None, :] - self.support[None, :, :]
            logits = -diff.square().sum(-1) / (2 * self.bandwidth**2)
            logp = torch.logsumexp(logits, -1) - math.log(len(self.support)) - math.log(2 * math.pi * self.bandwidth**2)
            score = -(logits.softmax(-1)[..., None] * diff).sum(1) / self.bandwidth**2
            distance2 = (points[:, None, :] - self.safe[None, :, :]).square().sum(-1)
            nearest = self.safe[distance2.argmin(-1)]
            delta = points - nearest
            distance = torch.linalg.vector_norm(delta, dim=-1)
            grad_distance = delta / distance.clamp_min(1e-15)[:, None]
            values.append(logp + self.alpha * distance)
            gradients.append(score + self.alpha * grad_distance)
        if not len(flat):
            return q.new_empty(shape[:-1]), q.new_empty(shape)
        return torch.cat(values).reshape(shape[:-1]), torch.cat(gradients).reshape(shape)

    @torch.no_grad()
    def guidance(self, x, center, scale, action_dim=2):
        """미래 목표 XY만 보정한다. action·첫 관측·나머지 상태의 가산항은0이다."""
        selected = slice(action_dim, action_dim + 2)
        q = x[:, 1:, selected].double() * scale[selected].double() + center[selected].double()
        phi, gradient = self.evaluate(q)
        guidance = torch.zeros_like(x)
        guidance[:, 1:, selected] = (gradient * scale[selected].double()).to(x.dtype)
        if not torch.isfinite(guidance).all() or not torch.isfinite(phi).all():
            raise ValueError('Nonfinite PF2MP potential/gradient')
        return guidance, {'energy': float(-phi.sum()), 'gradient_norm': float(guidance.double().norm())}


def prepare_potential(config):
    """공유 추론 진입점에서 한 번 준비한다. off에서는 파일을 전혀 읽지 않는다."""
    v = config.values if hasattr(config, 'values') and not isinstance(config, dict) else config
    if not v['methods']['pf2mp']['enabled']:
        return None
    settings = v['methods']['pf2mp']
    root = Path(v['paths']['data_root']) / f'{v["task"]}_sincos_v1'
    support, safe, report = ensure_potential_cache(root, settings)
    return PotentialField(support, safe, settings, v['device'], report)
