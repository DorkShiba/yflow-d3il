"""RTC numerics, fixed-prefix QP and delayed execution (no MuJoCo required)."""
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'fm_baseline'))
from rtc import soft_mask, guided_plan, ChunkController
from lipo import ChunkPostprocessor, LipoConfig


def condition(x, cond, act_dim):
    out = x.clone()
    out[:, 0, act_dim:] = cond[0]
    return out


class LinearField(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[.1, .2, .3, .4],
            [.2, -.1, .4, .3], [.5, .4, .2, .1], [.3, .5, .1, .2]]))
    def forward(self, x, t):
        return x @ self.weight


def toy_policy(n=10):
    p = SimpleNamespace(horizon=16, act_dim=2, obs_dim=2, device='cpu',
        center=torch.tensor([.2, -.3, .4, -.5]), scale=torch.tensor([.5, .7, 2., 3.]),
        model=LinearField(), t_span=torch.arange(n) / n, apply_conditioning=condition)
    def plan(obs, generator=None):
        with torch.no_grad():
            cond = {0: (torch.tensor(obs, dtype=torch.float32)[None] - p.center[2:]) / p.scale[2:]}
            x = condition(torch.randn(1, 16, 4, generator=generator), cond, 2)
            for t in p.t_span:
                v = p.model(x, t); v[:, 0, 2:] = 0
                x = condition(x + v / n, cond, 2)
            return (x * p.scale + p.center)[0].numpy()
    p.plan = plan
    return p


class RTCNumerics(unittest.TestCase):
    def test_mask_and_invalid_schedules(self):
        w = soft_mask(16, 8, 2)
        np.testing.assert_array_equal(w[:2], 1)
        np.testing.assert_array_equal(w[8:], 0)
        self.assertTrue(np.all(np.diff(w[1:9]) < 0))
        self.assertAlmostEqual(w[2], (6/7)*np.expm1(6/7)/np.expm1(1))
        for args in [(16, 8, 9), (16, 15, 2), (16, 0, 0), (16, 8, -1)]:
            with self.assertRaises(ValueError): soft_mask(*args)

    def test_one_step_vjp_matches_closed_form(self):
        p = toy_policy(n=1)
        obs = np.array([.7, -.8], dtype=np.float32)
        old = np.tile([.1, -.2], (8, 1))
        g = torch.Generator().manual_seed(6)
        x = torch.randn(1, 16, 4, generator=g)
        cond = {0: (torch.tensor(obs)[None] - p.center[2:]) / p.scale[2:]}
        x = condition(x, cond, 2)
        v = p.model(x, torch.tensor(0.)).detach(); v[:, 0, 2:] = 0
        target = torch.zeros(1, 16, 2)
        target[:, :8] = (torch.tensor(old) - p.center[:2]) / p.scale[:2]
        residual = (target - (x+v)[..., :2]) * torch.tensor(soft_mask(16, 8, 2), dtype=torch.float32)[None, :, None]
        # d(clean_action)/d(input) = first two columns of I+W.
        jac = (torch.eye(4) + p.model.weight.detach())[:, :2]
        correction = residual @ jac.T
        correction[:, 0, 2:] = 0
        expected = condition(x + v + .2 * correction, cond, 2)
        expected = (expected * p.scale + p.center)[0].numpy()
        with torch.no_grad():
            out = guided_plan(p, obs, old, 8, 2, beta=.2,
                              generator=torch.Generator().manual_seed(6))
        np.testing.assert_allclose(out, expected, atol=3e-7)
        np.testing.assert_allclose(out[0, 2:], obs, atol=1e-7)
        self.assertTrue(p.model.weight.requires_grad)
        self.assertIsNone(p.model.weight.grad)

    def test_zero_guidance_preserves_baseline_and_bad_beta_rejected(self):
        p = toy_policy(); obs = [0., 0.]; old = np.zeros((8, 2))
        a = p.plan(obs, torch.Generator().manual_seed(9))
        b = guided_plan(p, obs, old, 8, 0, 0., torch.Generator().manual_seed(9))
        np.testing.assert_array_equal(a, b)
        for beta in [-1., float('nan'), float('inf')]:
            with self.assertRaises(ValueError): guided_plan(p, obs, old, 8, 0, beta)

    def test_qp_only_has_no_blending_and_freezes_committed_prefix(self):
        proc = ChunkPostprocessor('qp', 8, LipoConfig(path_bound_m=.0003))
        proc.previous_tail = np.full((8, 2), 100.)
        actions = np.random.default_rng(7).normal(0, .003, (16, 2))
        hist = np.array([[.398, .2], [.399, .2], [.4, .2]])
        out = proc.process(actions, hist, frozen_prefix=3)
        np.testing.assert_array_equal(out[:3], actions[:3])
        shift = np.cumsum(out-actions, axis=0)
        self.assertLessEqual(np.abs(shift).max(), .00030001)
        self.assertEqual(proc.diagnostics[-1]['blend_steps'], 0)
        self.assertEqual(proc.diagnostics[-1]['blend_deviation_max_m'], 0)
        self.assertLessEqual(proc.diagnostics[-1]['jerk_cost_after_m2'], proc.diagnostics[-1]['jerk_cost_before_m2'])


class ScheduleTests(unittest.TestCase):
    def test_delay_alignment_and_actual_qp_feedback(self):
        for d in (0, 1, 4, 8):
            p = SimpleNamespace(horizon=16)
            raw = np.tile([.001, -.0004], (16, 1))
            p.plan = lambda obs, generator=None: raw.copy()
            proc = ChunkPostprocessor('qp', 8)
            ctrl = ChunkController(p, proc, 8, d, rtc=True)
            history = [np.array([.4, .2])]
            targets = []
            def guided(policy, obs, previous, s, delay, beta, generator):
                targets.append((ctrl.tick, previous.copy(), ctrl.active.copy()))
                return -raw.copy()
            seams = []
            with patch('rtc.guided_plan', side_effect=guided):
                for tick in range(33):
                    # Future committed actions are the old active plan at launch.
                    expected = None
                    if ctrl.pending is not None and tick < ctrl.pending[0]:
                        expected = ctrl.active[tick - ctrl.origin].copy()
                    a, seam = ctrl.next_action(None, history[-3:])
                    if expected is not None: np.testing.assert_array_equal(a, expected)
                    history.append(history[-1] + a)
                    if seam: seams.append(tick)
            self.assertEqual(seams, [0] + [t+d for t in (8,16,24,32) if t+d < 33])
            for tick, target, active in targets:
                np.testing.assert_array_equal(target, active[8:])
            self.assertEqual(ctrl.n_plans, 5)
            self.assertTrue(all(row['solver_status'] == 'Solved' for row in proc.diagnostics))

    def test_naive_delay_runs_without_guidance(self):
        p = SimpleNamespace(horizon=16)
        count = [0]
        def plan(obs, generator=None):
            count[0] += 1
            return np.full((16, 2), count[0] * .001)
        p.plan = plan
        ctrl = ChunkController(p, ChunkPostprocessor('none'), 8, 2)
        out = [ctrl.next_action(None, [[0., 0.]])[0][0] for _ in range(20)]
        np.testing.assert_allclose(out, [.001]*10 + [.002]*8 + [.003]*2)


if __name__ == '__main__':
    unittest.main()
