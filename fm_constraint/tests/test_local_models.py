"""내부 모델의 checkpoint 호환·CFM 수식·외부 import 차단을 검증한다."""

import json
import subprocess
import sys

import pytest
import torch

from configs.config import ExperimentConfig, REPO
from fm_baseline.models import (TemporalUnet, FlowMatcher, ConditionedODESolver,
                                apply_conditioning)


def test_cfm_loss_gradient_and_rng_match_explicit_formula():
    # CFM의 sigma=0에서도 원래 구현이 소비하던 epsilon 난수까지 보존한다.
    model = torch.nn.Linear(4, 4, bias=False)
    class Model(torch.nn.Module):
        def forward(self, x, t):
            return model(x)
    target = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 10
    cond = {0: target[:, 0, 2:].clone()}
    torch.manual_seed(19)
    noise = torch.randn_like(target)
    t = torch.rand(2).type_as(target)
    torch.randn_like(target)
    inputs = t[:, None, None] * target + (1 - t[:, None, None]) * noise
    apply_conditioning(inputs, cond, 2)
    expected = (model(inputs) - (target - noise)).square().mean()
    expected_gradient, = torch.autograd.grad(expected, model.weight)
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(19)
    loss, _ = FlowMatcher(Model(), 2).loss(target, cond)
    loss.backward()
    torch.testing.assert_close(loss, expected, rtol=0, atol=0)
    torch.testing.assert_close(model.weight.grad, expected_gradient, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), expected_rng)


def test_conditioned_euler_matches_explicit_steps_and_chain():
    class Model(torch.nn.Module):
        def forward(self, x, t):
            return .3 - .5 * x + t
    model = Model()
    x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 20
    cond = {0: torch.tensor([[.1, .2], [.3, .4]])}
    initial = apply_conditioning(x, cond, 2).clone()
    times = torch.linspace(0., 1., 11)[:-1]
    expected, chain = initial.clone(), [initial.clone()]
    for t in times:
        velocity = model(expected, t)
        velocity[:, 0, 2:] = 0
        expected = (expected + velocity * (times[1] - times[0])).detach()
        chain.append(expected)
    actual, actual_chain = ConditionedODESolver(model, cond, 2)(initial, times, return_chain=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_chain, torch.stack(chain, 1), rtol=0, atol=0)
    assert torch.equal(actual[:, 0, 2:], cond[0])
    with pytest.raises(ValueError):
        ConditionedODESolver(model, cond, 2, value=model)


@pytest.mark.parametrize('task', ['pushing', 'inserting'])
def test_real_checkpoint_loads_with_external_import_blocked(task):
    code = '''
import importlib.abc, sys, torch
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'hardflow' or fullname.startswith('hardflow.'):
            raise AssertionError('external package imported')
sys.meta_path.insert(0, Block())
from configs.config import ExperimentConfig, REPO
from fm_constraint import import_baseline, checkpoint_reads
cfg = ExperimentConfig.from_yaml(REPO / 'configs/TASK.yaml')
paths = cfg.values['paths']
assert 'hardflow_root' not in paths
with checkpoint_reads():
    mc, model, norm, conditioning, solver, step = import_baseline('eval_pushing').load_model(
        paths['run_dir'], 'baseline_ema.pt', 'ema', 'cpu')
x = torch.zeros(1, mc['horizon'], mc['act_dim'] + mc['obs_dim'])
assert model(x, torch.tensor(0.)).shape == x.shape
assert model.__class__.__module__ == 'fm_baseline.models.temporal_unet'
assert not any(n == 'hardflow' or n.startswith('hardflow.') for n in sys.modules)
'''.replace('TASK', task)
    subprocess.run([sys.executable, '-c', code], cwd=REPO, check=True, capture_output=True)

