"""Exercise both real evaluator loops with a deterministic fake policy/environment.

This checks wiring and files, not D3IL physics or trained-policy performance.
"""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import pickle
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

BASE = Path(__file__).resolve().parents[1] / 'fm_baseline'
sys.path.insert(0, str(BASE))
from compare_smoothness import compare


class Generator:
    def __init__(self, device=None): pass
    def manual_seed(self, seed):
        self.rng = np.random.default_rng(seed)
        return self


class Env:
    def __init__(self, **kwargs):
        self.scene = SimpleNamespace(dt=.001, get_obj_pos=lambda obj: np.zeros(3))
        self.robot = SimpleNamespace(current_j_pos=np.arange(7, dtype=float))
        self.target_box_1 = self.target_box_2 = None
        self.target_box1 = self.target_box2 = self.target_box3 = None
        self.target_min_dist = .01
        self.n_substeps = 35
    def start(self): pass
    def seed(self, seed): pass
    def reset(self, **kwargs):
        self.c = np.array([.4, .0, .1]); self.steps = 0
    def step(self, command):
        self.c[:2] += .5 * (command[:2] - self.c[:2])
        self.steps += 1
        return None, 0., self.steps >= 20, {'success': True, 'mode': 1, 'mean_distance': .01}


class Policy:
    def __init__(self, *args): pass
    def plan(self, obs, generator=None):
        return np.column_stack([generator.rng.normal(.001, .001, (16, 2)), np.zeros((16, 16))])


def module(name):
    spec = importlib.util.spec_from_file_location(name, BASE / (name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class EvaluationIntegrationTests(unittest.TestCase):
    def test_sharded_rng_uses_global_context_index(self):
        from eval_smoothness import EvaluationDiagnostics
        from lipo import LipoConfig
        samples = []
        with patch.dict(sys.modules, {'torch': SimpleNamespace(Generator=Generator)}):
            for offset, local_index in [(0, 5), (3, 2)]:
                obj = EvaluationDiagnostics.__new__(EvaluationDiagnostics)
                obj.args = SimpleNamespace(seed=0, context_start=offset, paired_noise=True,
                                           device='cpu', postprocess='none', execute_steps=8)
                obj.config = LipoConfig()
                obj.metadata = {'context_hashes': {}}
                _, generator = obj.start(local_index, 1, Env(), [5])
                samples.append(generator.rng.normal(size=100))
        np.testing.assert_array_equal(*samples)

    def test_both_evaluators_write_paired_comparable_runs(self):
        fake_torch = SimpleNamespace(no_grad=lambda: lambda f: f, cuda=SimpleNamespace(is_available=lambda: False),
                                     manual_seed=lambda seed: None, Generator=Generator)
        train = SimpleNamespace(Normalizer=object, add_spectral_norm=None,
                                resolve_d3il=lambda x: x, resolve_data_dir=lambda x, cfg: x,
                                resolve_hardflow=lambda x, cfg: x)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {'torch': fake_torch, 'train': train}):
            root = Path(tmp)
            context_file = root / 'environments/dataset/data/pushing/test_contexts.pkl'
            context_file.parent.mkdir(parents=True)
            context_file.write_bytes(pickle.dumps([[0], [1]]))
            for task in ('pushing', 'inserting'):
                pushing = module('eval_pushing')
                mod = pushing if task == 'pushing' else module('eval_inserting')
                cfg = {'task': task, 'horizon': 16}
                run = root / task; run.mkdir()
                (run/'config.json').write_text(json.dumps(cfg))
                (run/'baseline_ema.pt').write_bytes(b'fake checkpoint')
                np.savez(run/'normalizer.npz', center=[0], scale=[1])
                paths = {}
                for mode in ('none', 'blend', 'lipo'):
                    output = root / f'{task}_{mode}'
                    args = SimpleNamespace(run_dir=str(run), ckpt='baseline_ema.pt', weights='ema',
                        d3il_root=str(root), hardflow_root=str(root), data_dir=str(root),
                        contexts='test', contexts_file=str(context_file), export_contexts=None, split='val', raw_root='',
                        context_start=0, n_contexts=2, n_traj=2, ode_steps=10, execute_steps=8, max_steps=20, seed=7,
                        check_obs=0, record_joints=True, tag=mode, device='cpu', postprocess=mode, lipo_blend_steps=4,
                        lipo_blend_bound_m=.002, lipo_path_bound_m=.0003, paired_noise=True, output_root=str(output))
                    boxes = [(np.zeros(3), np.array([1, 0, 0, 0]), 0.)] * (2 if task == 'pushing' else 3)
                    patches = dict(parse_args=lambda: args, load_model=lambda *a: (cfg.copy(), None, None, None, None, 0),
                                   setup_d3il=lambda *a: (Env, None), Policy=Policy,
                                   read_state=lambda env, _: (env.c.copy(), boxes))
                    if task == 'inserting':
                        patches['load_contexts'] = lambda *a, **kw: [(f'{ci}.npz', {'robot': {'des_c_pos': np.zeros((20, 3))}}, [ci], '') for ci in range(2)]
                    with patch.multiple(mod, **patches), contextlib.redirect_stdout(io.StringIO()):
                        mod.main()
                    paths[mode] = next(output.iterdir())
                    with np.load(paths[mode]/'rollouts.npz') as saved:
                        self.assertEqual(saved['c000_t0_joints'].shape, (20, 7))
                        np.testing.assert_array_equal(saved['c000_t0_joints'][0], np.arange(7))
                        self.assertEqual(saved['c000_t0_goals'].shape, (len(boxes), 3))
                    result = json.loads((paths[mode]/'results.json').read_text())
                    self.assertEqual(result['qp_solved_count'], 12 if mode == 'lipo' else 0)
                    self.assertEqual(result['smoothness']['seam_count']['mean'], 2)
                    self.assertEqual(result['smoothness']['actual_jerk_rms_m_s3']['n'], 4)
                report = compare(paths['none'], paths['lipo'], n_boot=50)
                self.assertEqual(report['metrics']['success_rate']['delta'], 0)
                self.assertEqual(report['metrics']['actual_jerk_rms_m_s3']['paired_episodes'], 4)
                if task in ('pushing', 'inserting'):
                    from run_lipo_sharded import merge_shards
                    base = paths['none']
                    result = json.loads((base/'results.json').read_text())
                    detail = json.loads((base/'smoothness.json').read_text())
                    shard_paths = []
                    for ci in range(2):
                        shard = root / f'{task}_merge_fixture_{ci}'; shard.mkdir()
                        meta = dict(result, context_start=ci, n_contexts=1)
                        if task == 'inserting': meta['context_files'] = result['context_files'][ci:ci+1]
                        meta['context_hashes'] = {k.replace(f'c{ci:03d}', 'c000', 1): v for k,v in result['context_hashes'].items() if k.startswith(f'c{ci:03d}')}
                        (shard/'results.json').write_text(json.dumps(meta))
                        rows = {section:{k.replace(f'c{ci:03d}', 'c000', 1):v for k,v in detail[section].items() if k.startswith(f'c{ci:03d}')} for section in ('episodes','plans')}
                        (shard/'smoothness.json').write_text(json.dumps(rows))
                        with np.load(base/'rollouts.npz') as z:
                            arrays = {k:z[k][ci:ci+1] for k in ('successes','modes','mean_distance','lengths','n_boxes','demo_len') if k in z.files}
                            arrays.update({k.replace(f'c{ci:03d}', 'c000', 1):z[k] for k in z.files if k.startswith(f'c{ci:03d}')})
                        np.savez(shard/'rollouts.npz', **arrays)
                        shard_paths.append(shard)
                    merged = merge_shards(shard_paths, root/f'{task}_merged_fixture')
                    with np.load(base/'rollouts.npz') as a, np.load(merged/'rollouts.npz') as b:
                        self.assertEqual(set(a.files), set(b.files))
                        for key in a.files: np.testing.assert_array_equal(a[key], b[key])
                    check = compare(base, merged, n_boot=20)
                    self.assertEqual(check['metrics']['actual_jerk_rms_m_s3']['delta'], 0.)
                    with self.assertRaisesRegex(ValueError, 'contiguous'):
                        merge_shards([shard_paths[0], shard_paths[0]], root/f'{task}_invalid_merge')



if __name__ == '__main__':
    unittest.main()
