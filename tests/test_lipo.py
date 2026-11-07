import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'fm_baseline'))
from lipo import ChunkPostprocessor, LipoConfig
from smoothness import rollout_metrics
from compare_smoothness import compare, MATCH_FIELDS, METRICS


class LipoTests(unittest.TestCase):
    """⑤ LiPo (fixed-bound jerk QP, fm_constraint.lipo) through the baseline postprocessor."""

    def test_baseline_preserves_actions_exactly(self):
        actions = np.random.default_rng(1).normal(size=(16, 2)).astype(np.float32)
        proc = ChunkPostprocessor('none')
        np.testing.assert_array_equal(proc.process(actions, [[0.4, -0.1]]), actions)

    def test_constant_velocity_has_zero_jerk(self):
        history = np.array([[-0.002, 0], [-0.001, 0], [0, 0]])
        actions = np.tile([0.001, 0.], (16, 1))
        proc = ChunkPostprocessor('lipo')
        out = proc.process(actions, history)
        np.testing.assert_allclose(out, actions, atol=1e-9)
        self.assertLess(proc.diagnostics[-1]['jerk_cost_after_m2'], 1e-16)

    def test_qp_reduces_jerk_and_respects_position_bound(self):
        rng = np.random.default_rng(3)
        history = np.array([[-.002, 0], [-.001, 0], [0, 0]])
        actions = rng.normal(0, .003, (16, 2))
        proc = ChunkPostprocessor('lipo', config=LipoConfig(.0003))
        out = proc.process(actions, history)
        shift = np.cumsum(out, axis=0) - np.cumsum(actions, axis=0)
        self.assertTrue(np.all(np.abs(shift) <= .0003 + 1e-12))
        diag = proc.diagnostics[-1]
        self.assertLess(diag['jerk_cost_after_m2'], diag['jerk_cost_before_m2'])

    def test_history_influences_boundary_optimization(self):
        actions = np.tile([0.001, 0.], (16, 1))
        flat = ChunkPostprocessor('lipo', config=LipoConfig(.002)).process(actions, np.zeros((3, 2)))
        moving = ChunkPostprocessor('lipo', config=LipoConfig(.002)).process(actions, [[-.002, 0], [-.001, 0], [0, 0]])
        self.assertFalse(np.allclose(flat, moving))

    def test_reject_bad_inputs(self):
        with self.assertRaises(ValueError): LipoConfig(path_bound_m=0)
        with self.assertRaises(ValueError): ChunkPostprocessor('blend')
        with self.assertRaises(ValueError): ChunkPostprocessor('lipo', 17).process(np.zeros((16, 2)), [[0, 0]])
        with self.assertRaises(ValueError): ChunkPostprocessor().process(np.full((16, 2), np.nan), [[0, 0]])

    def test_fixed_multichunk_fixture_reduces_executed_jerk(self):
        chunks = np.random.default_rng(14).normal([.001, 0], .001, (40, 16, 2))
        results = {}
        for mode in ('none', 'lipo'):
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
        # 고정 jerk QP(blend 없음)는 이 fixture에서 78.7 → 41.4 m/s³ (약 47% 감소).
        # 이전 blend+QP 정의는 21.7이었다(fm_baseline/LIPO.md 이력).
        self.assertLess(results['lipo']['desired_jerk_rms_m_s3'], results['none']['desired_jerk_rms_m_s3'] * .6)

    def test_solver_failure_is_not_silent_fallback(self):
        with patch('lipo.jerk_qp', return_value={'status': 'solver_failure', 'solver_status': 'MaxIterations'}):
            with self.assertRaisesRegex(RuntimeError, 'MaxIterations'):
                ChunkPostprocessor('lipo').process(np.zeros((16, 2)), [[0, 0]])


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
