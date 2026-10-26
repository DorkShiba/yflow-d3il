"""Integration tests require the team's pinned HardFlow via HARDFLOW_ROOT."""
import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from fm_baseline import train as baseline
from fm_l2c2.train import configure, TrainingExtension
from fm_l2c2.regularizer import conditioned_plan


@pytest.fixture
def args(tmp_path):
    hf = os.environ.get('HARDFLOW_ROOT')
    if not hf:
        pytest.skip('HARDFLOW_ROOT needed for real-model integration')
    torch.set_num_threads(1)
    rng = np.random.default_rng(9)
    for split in ['train', 'val']:
        dest = tmp_path / 'data' / 'pushing_sincos_v1' / split
        dest.mkdir(parents=True)
        for i in range(2):
            np.savez(dest / f'{i}.npz', obs=rng.normal(size=(20, 12)).astype('float32'),
                     actions=rng.normal(0, .003, size=(20, 2)).astype('float32'))
    return baseline.parse_args(['--device', 'cpu', '--task', 'pushing', '--steps', '2',
        '--batch-size', '2', '--dim', '8', '--dim-mults', '1,2', '--val-batches', '1',
        '--log-every', '1', '--save-every', '1', '--hardflow-root', hf,
        '--data-root', str(tmp_path / 'data'), '--out-root', str(tmp_path / 'runs'),
        '--l2c2-batch-size', '1'], configure)


def run(args, extension=True):
    baseline.main(args, TrainingExtension if extension else None)
    folder = next(Path(args.out_root).iterdir()) if not args.resume else Path(args.resume)
    state = torch.load(folder / 'last.pt', map_location='cpu', weights_only=False)
    return folder, state


def equal_weights(a, b):
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)


def test_disabled_matches_unregularized_training(args, tmp_path):
    args.l2c2_weight = 0.
    _, plain = run(copy.deepcopy(args), extension=False)
    args.out_root = str(tmp_path / 'off')
    _, off = run(args)
    equal_weights(plain['model'], off['model'])
    equal_weights(plain['ema'], off['ema'])
    assert torch.equal(plain['torch_rng'], off['torch_rng'])


def test_real_training_resume_and_baseline_checkpoint_compatibility(args, tmp_path):
    continuous_args = copy.deepcopy(args)
    _, continuous = run(continuous_args)
    interrupted_args = copy.deepcopy(args)
    interrupted_args.out_root = str(tmp_path / 'interrupted')
    interrupted_args.time_limit_hours = 1e-12
    folder, first = run(interrupted_args)
    assert first['step'] == 1
    resume_args = copy.deepcopy(args)
    resume_args.resume = str(folder)
    resume_args.time_limit_hours = 0
    _, resumed = run(resume_args)
    equal_weights(continuous['model'], resumed['model'])
    equal_weights(continuous['ema'], resumed['ema'])
    assert torch.equal(continuous['torch_rng'], resumed['torch_rng'])
    cfg = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
    assert cfg['regularization']['name'] == 'l2c2_fm_shared_noise'
    # Existing evaluation loader, no L2C2-specific inference code.
    from fm_baseline.eval_pushing import load_model
    _, model, _, apply_conditioning, Solver, _ = load_model(
        str(folder), 'last.pt', 'ema', args.hardflow_root, 'cpu')
    obs, z = torch.randn(2, 12), torch.randn(2, 16, 14)
    original = Solver(model, {0: obs}, action_dim=2, value=None, ode_method='euler')
    with torch.no_grad():
        expected = original(apply_conditioning(z.clone(), {0: obs}, 2),
                            t_span=torch.linspace(0, 1, 11)[:-1])
        actual = conditioned_plan(model, obs, z, 2, 10)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_wrong_entrypoint_resume_is_rejected(args):
    folder, _ = run(args)
    args.resume = str(folder)
    with pytest.raises(ValueError, match='entry point'):
        baseline.main(args)


def test_enabled_changes_learned_weights(args, tmp_path):
    _, enabled = run(copy.deepcopy(args))
    args.out_root = str(tmp_path / 'off_compare')
    args.l2c2_weight = 0
    _, disabled = run(args)
    assert any(not torch.equal(enabled['model'][k], disabled['model'][k])
               for k in enabled['model'])


@pytest.mark.parametrize('task,dim,mults', [('pushing', 32, '1,4,8'),
                                         ('inserting', 64, '1,2,4,8')])
def test_default_architecture_matches_baseline(task, dim, mults):
    args = baseline.parse_args(['--task', task], configure)
    TrainingExtension(args)
    assert (args.dim, args.dim_mults) == (dim, mults)
