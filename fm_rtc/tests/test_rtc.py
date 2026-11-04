import copy
import math
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from fm_rtc.policy import RTCPolicy, soft_mask, guidance_weight
from fm_rtc.execution import ChunkController
from fm_rtc.metrics import trajectory_metrics
from fm_rtc.compare import compare


class LinearField(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.2))

    def forward(self, x, t):
        return self.weight * x + .1


def policy(model=None, beta=5, steps=10):
    return RTCPolicy(model or LinearField(), SimpleNamespace(center=np.zeros(4), scale=np.ones(4)),
                     dict(horizon=16, act_dim=2, obs_dim=2), steps, "cpu", beta)


class PolicyTests(unittest.TestCase):
    def test_equation5_mask(self):
        w = soft_mask(16, 5, 4).numpy()
        np.testing.assert_array_equal(w[:4], 1)
        np.testing.assert_array_equal(w[11:], 0)
        c = np.arange(7, 0, -1) / 8
        np.testing.assert_allclose(w[4:11], c * np.expm1(c) / math.expm1(1), rtol=2e-6)
        self.assertTrue(np.all(np.diff(w) <= 0))

    def test_zero_delay_still_has_soft_guidance(self):
        w = soft_mask(16, 8, 0)
        self.assertTrue(bool((w[:8] > 0).all()))
        self.assertTrue(bool((w[8:] == 0).all()))
        self.assertTrue(bool((soft_mask(16, 8, 8)[:8] == 1).all()))
        with self.assertRaises(ValueError): soft_mask(16, 9, 8)

    def test_clipped_coefficient(self):
        self.assertEqual(guidance_weight(0, 5), 5)
        self.assertEqual(guidance_weight(.5, 5), 2)
        self.assertEqual(guidance_weight(.01, 5), 5)
        self.assertEqual(guidance_weight(.5, 0), 0)
        with self.assertRaises(ValueError): guidance_weight(0, float("nan"))

    def test_no_history_and_off_preserve_euler(self):
        noise = torch.randn(1, 16, 4, generator=torch.Generator().manual_seed(2))
        p = policy()
        actual = p.plan([.3, -.2], noise=noise)
        x = noise.clone(); x[:, 0, 2:] = torch.tensor([.3, -.2])
        with torch.no_grad():
            for t in p.times:
                v = p.model(x, t); v[:, 0, 2:] = 0
                x = (x + (p.times[1]-p.times[0]) * v).detach()
        np.testing.assert_array_equal(actual, x[0].numpy())
        np.testing.assert_array_equal(actual, p.plan([.3, -.2], noise=noise, rtc=False))

    def test_full_vjp_against_analytic_linear_field(self):
        p = policy(steps=2)
        previous = torch.arange(64, dtype=torch.float32).reshape(1, 16, 4) / 100
        p.previous = previous.clone()
        noise = torch.ones(1, 16, 4) * .3
        actual = p.plan([.7, -.4], noise=noise)
        x = noise.clone(); x[:, 0, 2:] = torch.tensor([.7, -.4])
        w = soft_mask(16, 8, 0)
        target = torch.zeros_like(x); target[:, :8, :2] = previous[:, 8:, :2]
        for tau in (0., .5):
            v = .2*x + .1; v[:, 0, 2:] = 0
            estimate = x + (1-tau)*v
            # Analytic Jacobian is 1 + (1-tau)*0.2 on the action channels.
            g = torch.zeros_like(x)
            g[:, :, :2] = (target-estimate)[:, :, :2] * w[None, :, None] * (1+(1-tau)*.2)
            x = x + .5*(v+guidance_weight(tau, 5)*g)
        np.testing.assert_allclose(actual, x[0].numpy(), rtol=2e-6, atol=2e-7)

    def test_reset_condition_and_frozen_weights(self):
        p = policy()
        before = copy.deepcopy(p.model.state_dict())
        noise = torch.zeros(1, 16, 4)
        first = p.plan([1, 2], noise=noise)
        with torch.no_grad():
            result = p.plan([3, 4], noise=noise)
        np.testing.assert_array_equal(result[0, 2:], [3, 4])
        for name, value in p.model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in p.model.parameters()))
        p.reset()
        np.testing.assert_array_equal(first, p.plan([1, 2], noise=noise))

    def test_beta_zero_exactly_disables_guidance(self):
        p = policy(beta=0)
        noise = torch.randn(1, 16, 4)
        first = p.plan([1, 2], noise=noise)
        np.testing.assert_array_equal(first, p.plan([1, 2], noise=noise))

    def test_episode_generators_are_independent(self):
        p = policy()
        a = p.plan([0, 0], generator=torch.Generator().manual_seed(4))
        torch.randn(100)
        p.reset()
        b = p.plan([0, 0], generator=torch.Generator().manual_seed(4))
        np.testing.assert_array_equal(a, b)


class TimestampPolicy:
    horizon = 16
    act_dim = 2

    def reset(self): self.calls = []

    def plan(self, obs, **kwargs):
        self.calls.append(obs)
        # Encode absolute planned time and inference-start time separately.
        return np.column_stack((obs + np.arange(16), np.full(16, obs)))


class ExecutionTests(unittest.TestCase):
    def test_delayed_switch_skips_stale_prefix(self):
        p = TimestampPolicy(); c = ChunkController(p, 8, 3)
        outputs, seams = zip(*(c.action(t) for t in range(35)))
        np.testing.assert_array_equal(np.asarray(outputs)[:, 0], np.arange(35))
        self.assertEqual(p.calls, [0, 8, 16, 24, 32])
        self.assertEqual(np.flatnonzero(seams).tolist(), [11, 19, 27])
        self.assertEqual(outputs[10][1], 0)
        self.assertEqual(outputs[11][1], 8)

    def test_zero_delay_baseline_schedule(self):
        c = ChunkController(TimestampPolicy(), 8, 0)
        pairs = [c.action(t) for t in range(25)]
        self.assertEqual([i for i, (_, s) in enumerate(pairs) if s], [8, 16, 24])

    def test_maximum_delay_no_underrun(self):
        c = ChunkController(TimestampPolicy(), 8, 8)
        values = [c.action(t)[0][0] for t in range(80)]
        np.testing.assert_array_equal(values, np.arange(80))


class MetricTests(unittest.TestCase):
    def test_cubic_has_exact_jerk(self):
        dt = .035
        time = np.arange(21)*dt
        p = np.column_stack((time**3, time*0))
        seams = np.zeros(20, dtype=bool); seams[8] = True
        m = trajectory_metrics(p[1:], p[1:], p[0], p[0], seams, dt)
        self.assertAlmostEqual(m["actual_jerk_rms_m_s3"], 6, places=9)
        self.assertAlmostEqual(m["actual_seam_jerk_rms_m_s3"], 6, places=9)
        self.assertEqual(m["tracking_gap_mean_m"], 0)

    def test_short_episode_reports_missing(self):
        p = np.zeros((2, 2))
        m = trajectory_metrics(p, p, [0,0], [0,0], [False,False], .035)
        self.assertIsNone(m["actual_jerk_rms_m_s3"])

    def test_mismatched_protocol_refused(self):
        with self.assertRaises(ValueError):
            compare(dict(method="baseline", protocol={"seed":0}), dict(method="rtc", protocol={"seed":1}))


if __name__ == "__main__":
    unittest.main()
