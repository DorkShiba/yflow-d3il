import torch
import pytest

from fm_l2c2.regularizer import (L2C2Config, conditioned_plan,
                                policy_regularization, sample_neighbor)


class LinearPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(2.0))

    def forward(self, x, t):
        # Constant-in-flow-time derivative depending on the fixed observation.
        obs = x[:, :1, 2:4].expand(-1, x.shape[1], -1)
        return torch.cat([self.gain * obs, torch.zeros_like(x[:, :, 2:])], -1)


def test_neighbor_geometry_and_stationary_coordinates():
    state = torch.zeros(100, 4)
    next_state = state + torch.tensor([2., -3., 0., 1e-10])
    near, distance = sample_neighbor(state, next_state, 1., .1)
    relative = (near[:, [0, 1, 3]] / next_state[:, [0, 1, 3]]).abs()
    assert relative.max() <= 1
    assert torch.equal(near[:, 2], state[:, 2])
    torch.testing.assert_close(distance, relative.amax(-1) + .1)


def test_differentiable_policy_matches_analytic_solution():
    model = LinearPolicy()
    obs = torch.tensor([[.1, -.4], [.3, .6]])
    z = torch.randn(2, 16, 4)
    plan = conditioned_plan(model, obs, z, 2, 10)
    torch.testing.assert_close(plan[:, :, :2], z[:, :, :2] + 2 * obs[:, None])
    torch.testing.assert_close(plan[:, 0, 2:], obs)
    plan[:, :, :2].square().mean().backward()
    assert torch.isfinite(model.gain.grad) and model.gain.grad.abs() > 0


def test_regularizer_backprop_and_rng_isolation():
    model = LinearPolicy()
    windows = torch.randn(8, 16, 4)
    before = torch.get_rng_state().clone()
    config = L2C2Config()
    first, metrics = policy_regularization(model, windows, 2, config, 123)
    assert torch.equal(before, torch.get_rng_state())
    second, _ = policy_regularization(model, windows, 2, config, 123)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert first > 0 and metrics['relative_distance'] >= config.epsilon
    first.backward()
    assert model.gain.grad > 0  # Gradient descent must reduce local sensitivity.


def test_stationary_transition_has_zero_penalty_with_shared_noise():
    windows = torch.randn(4, 16, 4)
    windows[:, 1, 2:] = windows[:, 0, 2:]
    loss, _ = policy_regularization(LinearPolicy(), windows, 2, L2C2Config(), 8)
    assert loss.item() == 0 and torch.isfinite(loss)


def test_zero_weight_does_not_call_model_or_consume_rng():
    class NeverCall(torch.nn.Module):
        def forward(self, *args):
            raise AssertionError('disabled regularizer called policy')
    windows = torch.randn(4, 16, 4)
    before = torch.get_rng_state().clone()
    loss, _ = policy_regularization(NeverCall(), windows, 2, L2C2Config(weight=0), 5)
    assert loss.item() == 0
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize('kwargs', [{'weight': -1}, {'weight': float('nan')},
    {'epsilon': 0}, {'sigma': .01, 'epsilon': 1}, {'ode_steps': 1},
    {'action_steps': 17}, {'batch_size': 0}])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        L2C2Config(**kwargs).validate(16)
