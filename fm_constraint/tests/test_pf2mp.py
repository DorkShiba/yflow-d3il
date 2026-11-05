"""밀도 score·위치 채널 마스크·캐시 재사용과 실패 처리를 검증한다."""

from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
import multiprocessing
from pathlib import Path

import numpy as np
import pytest
import torch

from configs.config import ExperimentConfig, REPO
from fm_constraint.pf2mp import (PotentialField, _cache_lock, ensure_potential_cache,
                                 prepare_potential)


@pytest.fixture
def settings():
    value = ExperimentConfig.from_yaml(REPO / 'configs/pushing.yaml').values['methods']['pf2mp']
    return {**value, 'max_support': 8, 'grid_step_m': .02, 'grid_padding_m': .2,
            'bandwidth_m': .04, 'query_chunk_size': 2, 'lock_timeout_s': .2}


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / 'pushing_sincos_v1'
    (root / 'train').mkdir(parents=True)
    manifest = json.loads((REPO / 'data/pushing_sincos_v1/manifest.json').read_text())
    manifest['splits'] = {'train': [{'file': 'train/a.npz', 'source': 'a.pkl', 'transitions': 8}],
                          'val': [{'file': 'val/b.npz', 'source': 'b.pkl', 'transitions': 8}]}
    (root / 'manifest.json').write_text(json.dumps(manifest))
    obs = np.zeros((9, 12), dtype=np.float32)
    obs[:, 0] = .4 + np.arange(9) * .002
    obs[:, 1] = np.arange(9) * .001
    obs[:, 2:4] = obs[:, :2]
    obs[:, 7] = obs[:, 11] = 1
    np.savez(root / 'train/a.npz', obs=obs[:-1], next_obs=obs[1:],
             actions=np.diff(obs[:, :2], axis=0),
             timestamps=np.arange(8) * .035, next_timestamps=np.arange(1, 9) * .035)
    return root


def test_potential_score_and_scaled_state_gradient(settings):
    field = PotentialField(np.array([[.4, -.01], [.55, .04]]),
                           np.array([[.4, 0.], [.55, .05]]), settings)
    q = torch.tensor([[.47, .021], [.6, -.1]], dtype=torch.float64)
    phi, grad = field.evaluate(q)
    for i in range(2):
        for j in range(2):
            plus, minus = q.clone(), q.clone()
            plus[i, j] += 1e-6
            minus[i, j] -= 1e-6
            numerical = (field.evaluate(plus)[0][i] - field.evaluate(minus)[0][i]) / 2e-6
            torch.testing.assert_close(grad[i, j], numerical, rtol=1e-7, atol=1e-7)
    center = torch.tensor([0, 0, .5, 0] + [0] * 10, dtype=torch.float64)
    scale = torch.tensor([.01, .02, .2, .3] + [.1] * 10, dtype=torch.float64)
    x = torch.zeros(1, 4, 14, dtype=torch.float64)
    x[:, 1:, 2:4] = torch.tensor([.1, .05])
    guidance, info = field.guidance(x, center, scale)
    assert guidance[:, :, :2].count_nonzero() == 0
    assert guidance[:, 0].count_nonzero() == 0
    assert guidance[:, :, 4:].count_nonzero() == 0
    altered = x.clone()
    altered[:, :, :2] = 1000
    assert field.guidance(altered, center, scale)[1]['energy'] == info['energy']
    plus, minus = x.clone(), x.clone()
    plus[0, 1, 2] += 1e-6
    minus[0, 1, 2] -= 1e-6
    # E=-Psi이므로 guidance는 -dE/dx다.
    numerical = -(field.guidance(plus, center, scale)[1]['energy'] - field.guidance(minus, center, scale)[1]['energy']) / 2e-6
    assert guidance[0, 1, 2] == pytest.approx(numerical, rel=1e-6)
    assert torch.isfinite(field.evaluate(torch.tensor([[1e3, -1e3]]))[1]).all()


def test_distance_zero_and_empty_horizon_are_finite(settings):
    field = PotentialField(np.array([[.4, 0.]]), np.array([[.4, 0.]]), settings)
    phi, gradient = field.evaluate(torch.tensor([[.4, 0.]], dtype=torch.float64))
    assert torch.isfinite(phi).all()
    torch.testing.assert_close(gradient, torch.zeros_like(gradient))
    guidance, info = field.guidance(torch.zeros(1, 1, 14), torch.zeros(14), torch.ones(14))
    assert guidance.count_nonzero() == 0 and info['energy'] == 0


def test_cache_roundtrip_invalidation_and_rng(dataset, settings):
    torch.manual_seed(19)
    np.random.seed(19)
    rng, nrng = torch.get_rng_state().clone(), deepcopy(np.random.get_state())
    a, safe_a, first = ensure_potential_cache(dataset, settings)
    b, safe_b, hit = ensure_potential_cache(dataset, settings)
    assert first['event'] == 'miss' and hit['event'] == 'hit'
    assert first['artifact_sha256'] == hit['artifact_sha256']
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(safe_a, safe_b)
    assert torch.equal(rng, torch.get_rng_state())
    assert nrng[0] == np.random.get_state()[0]
    np.testing.assert_array_equal(nrng[1], np.random.get_state()[1])
    assert hit['audit']['train_rows'] == 8
    _, _, weight = ensure_potential_cache(dataset, {**settings, 'weight': .7, 'distance_alpha': -3.})
    assert weight['event'] == 'hit' and weight['fingerprint'] == hit['fingerprint']
    _, _, bandwidth = ensure_potential_cache(dataset, {**settings, 'bandwidth_m': .03})
    assert bandwidth['fingerprint'] != hit['fingerprint']
    # Val 파일은 읽지도 않고 support에 포함하지 않는다.
    (dataset / 'val').mkdir()
    (dataset / 'val/b.npz').write_bytes(b'not read')
    assert ensure_potential_cache(dataset, settings)[2]['event'] == 'hit'
    path = dataset / 'train/a.npz'
    with np.load(path) as record:
        arrays = {key: record[key] for key in record.files}
    arrays['obs'][:, 4] = .7  # 유효하지만 원본 hash가 바뀌는 변경.
    arrays['next_obs'][:, 4] = .7
    np.savez(path, **arrays)
    changed = ensure_potential_cache(dataset, settings)[2]
    assert changed['event'] == 'miss' and changed['fingerprint'] != hit['fingerprint']


def test_corrupt_cache_is_quarantined_and_rebuilt(dataset, settings):
    first = ensure_potential_cache(dataset, settings)[2]
    Path(first['path']).write_bytes(b'broken cache')
    rebuilt = ensure_potential_cache(dataset, settings)[2]
    assert rebuilt['event'] == 'rebuilt_corrupt'
    assert rebuilt['corruption']
    assert list((dataset / 'pf2mp_cache').glob('*.corrupt.*.npz'))
    assert ensure_potential_cache(dataset, settings)[2]['event'] == 'hit'


def test_bad_train_is_not_published(dataset, settings):
    path = dataset / 'train/a.npz'
    with np.load(path) as record:
        arrays = {key: record[key] for key in record.files}
    arrays['actions'][0, 0] = np.nan
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match='nonfinite'):
        ensure_potential_cache(dataset, settings)
    assert not list((dataset / 'pf2mp_cache').glob('*.npz'))


def _build(args):
    return ensure_potential_cache(*args)[2]


def test_parallel_build_publishes_once(dataset, settings):
    with ProcessPoolExecutor(2, mp_context=multiprocessing.get_context('fork')) as pool:
        results = list(pool.map(_build, [(dataset, settings)] * 2))
    assert sorted(r['event'] for r in results) == ['hit', 'miss']
    assert results[0]['artifact_sha256'] == results[1]['artifact_sha256']


def test_lock_timeout_and_write_failure(dataset, settings, monkeypatch):
    import fm_constraint.pf2mp as module
    with _cache_lock(dataset / 'held.lock', .1):
        with pytest.raises(TimeoutError):
            with _cache_lock(dataset / 'held.lock', .01):
                pass
    def fail(*args, **kwargs):
        raise OSError('disk full')
    monkeypatch.setattr(module.os, 'replace', fail)
    with pytest.raises(OSError, match='disk full'):
        ensure_potential_cache(dataset, settings)
    assert not list((dataset / 'pf2mp_cache').glob('*.npz'))


def test_off_does_not_access_dataset(tmp_path):
    cfg = ExperimentConfig.from_yaml(REPO / 'configs/pushing.yaml', {'paths.data_root': str(tmp_path / 'missing')})
    assert prepare_potential(cfg) is None
    assert not (tmp_path / 'missing').exists()
