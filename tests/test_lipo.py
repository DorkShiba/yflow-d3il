import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'fm_baseline'))
from lipo import ChunkPostprocessor, LipoConfig, minimum_jerk
from smoothness import rollout_metrics
from compare_smoothness import compare, MATCH_FIELDS, METRICS


class LipoTests(unittest.TestCase):
    def test_baseline_preserves_actions_exactly(self):
        actions = np.random.default_rng(1).normal(size=(16, 2)).astype(np.float32)
        proc = ChunkPostprocessor('none')
        np.testing.assert_array_equal(proc.process(actions, [[0.4, -0.1]]), actions)

    def test_constant_velocity_has_zero_jerk(self):
        history = np.array([[-0.002, 0], [-0.001, 0], [0, 0]])
        ref = np.column_stack([np.arange(1, 17) * 0.001, np.zeros(16)])
        out, diag = minimum_jerk(ref, history, np.full(16, .0003))
        np.testing.assert_allclose(out, ref, atol=2e-7)
        self.assertLess(diag['jerk_cost_after_m2'], 1e-16)

    def test_qp_reduces_jerk_and_respects_per_sample_bounds(self):
        rng = np.random.default_rng(3)
        history = np.array([[-.002, 0], [-.001, 0], [0, 0]])
        ref = np.cumsum(rng.normal(0, .003, (16, 2)), axis=0)
        bounds = np.r_[np.full(4, .002), np.full(12, .0003)]
        out, diag = minimum_jerk(ref, history, bounds)
        self.assertTrue(np.all(np.abs(out - ref) <= bounds[:, None] + 1e-12))
        self.assertLess(diag['jerk_cost_after_m2'], diag['jerk_cost_before_m2'])

    def test_zero_bounds_leave_reference_unchanged(self):
        ref = np.random.default_rng(4).normal(0, .01, (16, 2))
        out, _ = minimum_jerk(ref, [[0, 0]], np.zeros(16))
        np.testing.assert_array_equal(out, ref)

    def test_blend_uses_absolute_unexecuted_previous_positions(self):
        proc = ChunkPostprocessor('blend', 8, LipoConfig(blend_steps=4))
        old = np.tile([.001, 0], (16, 1))
        proc.process(old, [[.4, .2]])
        anchor = np.array([.408, .2])
        new = np.tile([-.001, 0], (16, 1))
        out = proc.process(new, [anchor])
        positions = anchor + np.cumsum(out, axis=0)
        raw = anchor + np.cumsum(new, axis=0)
        prior = np.array([.4, .2]) + np.cumsum(old, axis=0)[8:]
        alpha = np.arange(1, 5)[:, None] / 4
        np.testing.assert_allclose(positions[:4], (1-alpha)*prior[:4]+alpha*raw[:4])
        np.testing.assert_allclose(positions[4:], raw[4:])

    def test_history_influences_boundary_optimization(self):
        ref = np.zeros((16, 2))
        flat, _ = minimum_jerk(ref, np.zeros((3, 2)), np.full(16, .002))
        moving, _ = minimum_jerk(ref, [[-.002, 0], [-.001, 0], [0, 0]], np.full(16, .002))
        self.assertGreater(moving[0, 0], flat[0, 0] + .0001)

    def test_new_episode_has_no_old_tail(self):
        actions = np.tile([.001, 0], (16, 1))
        a = ChunkPostprocessor('lipo').process(actions, [[0, 0]])
        b = ChunkPostprocessor('lipo').process(actions, [[0, 0]])
        np.testing.assert_array_equal(a, b)

    def test_reject_bad_inputs_and_overlap(self):
        with self.assertRaises(ValueError): LipoConfig(path_bound_m=-1)
        with self.assertRaises(ValueError): LipoConfig(blend_steps=1.5)
        with self.assertRaises(ValueError): ChunkPostprocessor('lipo', 16).process(np.zeros((16, 2)), [[0, 0]])
        with self.assertRaises(ValueError): ChunkPostprocessor().process(np.full((16, 2), np.nan), [[0, 0]])

    def test_fixed_multichunk_fixture_reduces_executed_jerk(self):
        chunks = np.random.default_rng(14).normal([.001, 0], .001, (40, 16, 2))
        results = {}
        for mode in ('none', 'blend', 'lipo'):
            processor = ChunkPostprocessor(mode)
            history = [np.array([.4, 0.])]
            actions, starts = [], []
            for chunk in chunks:
                output = processor.process(chunk, history[-3:])
                for k, action in enumerate(output[:8]):
                    history.append(history[-1] + action)
                    actions.append(action)
                    starts.append(k == 0)
            results[mode] = rollout_metrics(actions, history[1:], history[1:], starts,
                                             .035, history[0], history[0])
        for metric in ('desired_jerk_rms_m_s3', 'seam_velocity_jump_rms_m_s'):
            self.assertLess(results['lipo'][metric], results['none'][metric] * .5)

    def test_solver_failure_is_not_silent_fallback(self):
        with patch('clarabel.DefaultSolver') as cls:
            cls.return_value.solve.return_value.status = 'MaxIterations'
            with self.assertRaisesRegex(RuntimeError, 'MaxIterations'):
                minimum_jerk(np.zeros((16, 2)), [[0, 0]], np.ones(16))


class MetricsTests(unittest.TestCase):
    def test_cubic_trajectory_has_known_si_jerk(self):
        dt = .035
        t = np.arange(12) * dt
        q = np.column_stack([t ** 3, 2 * t ** 3])
        actions = np.diff(q, axis=0)
        starts = np.zeros(11, dtype=bool); starts[[0, 8]] = True
        m = rollout_metrics(actions, q[1:], q[1:], starts, dt, q[0], q[0])
        self.assertAlmostEqual(m['desired_jerk_rms_m_s3'], np.sqrt(180), places=8)
        self.assertAlmostEqual(m['actual_seam_jerk_rms_m_s3'], np.sqrt(180), places=8)
        self.assertEqual(m['seam_count'], 1)
        self.assertAlmostEqual(m['seam_velocity_jump_rms_m_s'], np.linalg.norm(actions[8]-actions[7])/dt)

    def test_short_episode_is_unavailable_not_zero(self):
        m = rollout_metrics([[.01, 0]], [[.01, 0]], [[.009, 0]], [True], .035, [0, 0], [0, 0])
        self.assertIsNone(m['actual_jerk_rms_m_s3'])
        self.assertIsNone(m['seam_velocity_jump_rms_m_s'])

    def test_seam_jerk_mask_covers_both_crossing_stencils(self):
        actions = np.zeros((12, 2)); actions[8:, 0] = .001
        q = np.cumsum(actions, axis=0)
        starts = np.zeros(12, dtype=bool); starts[[0, 8]] = True
        m = rollout_metrics(actions, q, q, starts, 1, [0, 0], [0, 0])
        self.assertAlmostEqual(m['desired_seam_jerk_rms_m_s3'], .001)

    def test_compare_rejects_mismatched_seed_and_handles_null(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp)/name for name in ('a','b')]
            metadata = {k: 'same' for k in MATCH_FIELDS}
            metadata.update(n_contexts=2, n_traj=1, paired_noise=True, postprocess='none', seed=0)
            rows = {f'c{ci:03d}_t0': {k: (None if 'seam' in k else 2.) for k in METRICS} for ci in range(2)}
            for path in paths:
                path.mkdir()
                (path/'results.json').write_text(json.dumps(metadata))
                (path/'smoothness.json').write_text(json.dumps({'episodes': rows}))
                np.savez(path/'rollouts.npz', successes=np.ones((2, 1)))
            result = compare(*paths, n_boot=20)
            self.assertEqual(result['metrics']['actual_jerk_rms_m_s3']['delta_ci95'], [0., 0.])
            self.assertIsNone(result['metrics']['seam_velocity_jump_rms_m_s']['delta'])
            metadata['seed'] = 1
            (paths[1]/'results.json').write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, 'seed'): compare(*paths)


if __name__ == '__main__':
    unittest.main()
