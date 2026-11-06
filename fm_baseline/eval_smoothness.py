"""Shared LiPo evaluation options, provenance, and episode logging."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import random
import subprocess
import numpy as np
from lipo import ChunkPostprocessor, LipoConfig
from smoothness import rollout_metrics, summarize_metrics


def add_arguments(parser):
    parser.add_argument('--postprocess', choices=['none', 'blend', 'lipo', 'qp'], default='none')
    parser.add_argument('--rtc', action='store_true', help='Action-channel RTC soft-mask guidance')
    parser.add_argument('--rtc-beta', type=float, default=5.)
    parser.add_argument('--inference-delay-steps', type=int, default=0,
                        help='Simulated control-step delay, not wall-clock asynchronous inference')
    parser.add_argument('--lipo-blend-steps', type=int, default=4)
    parser.add_argument('--lipo-blend-bound-m', type=float, default=0.002)
    parser.add_argument('--lipo-path-bound-m', type=float, default=0.0003)
    parser.add_argument('--paired-noise', action='store_true', help='Independent per-episode RNG, shared across ablation arms')
    parser.add_argument('--output-root', default=None)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def git_head(path):
    try:
        return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


class EvaluationDiagnostics:
    def __init__(self, args, cfg, env, run_dir, context_source, hardflow_root, d3il_root):
        self.args = args
        self.config = LipoConfig(args.lipo_blend_steps, args.lipo_blend_bound_m, args.lipo_path_bound_m)
        if not 1 <= args.execute_steps <= cfg['horizon']:
            raise ValueError('execute_steps must be in [1, horizon]')
        if args.postprocess in ('blend', 'lipo') and self.config.blend_steps > cfg['horizon'] - args.execute_steps:
            raise ValueError('LiPo blend window must fit the unexecuted overlap')
        from rtc import soft_mask
        delay = getattr(args, 'inference_delay_steps', 0)
        rtc = getattr(args, 'rtc', False)
        soft_mask(cfg['horizon'], args.execute_steps, delay)
        beta = getattr(args, 'rtc_beta', 5.)
        if not np.isfinite(beta) or beta < 0:
            raise ValueError('RTC beta must be finite and nonnegative')
        if (rtc or delay) and args.postprocess in ('blend', 'lipo'):
            raise ValueError('RTC/delayed evaluation requires --postprocess none or qp')
        self.dt = float(env.scene.dt) * int(env.n_substeps)
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError('Invalid simulator control period')
        self.rows, self.plans = {}, {}
        self.metadata = {
            'metrics_version': 1, 'context_start': getattr(args, 'context_start', 0), 'task': cfg['task'], 'control_dt_s': self.dt,
            'postprocess': args.postprocess, 'lipo_config': asdict(self.config),
            'inference_delay_steps': delay, 'paired_noise': args.paired_noise,
            'rtc': rtc, 'rtc_beta': beta, 'rtc_target': 'normalized_action_channels',
            'execution_protocol': 'simulated_delay_v1',
            'bootstrap_protocol': 'initial_chunk_ready_before_first_step',
            'checkpoint_sha256': sha256(Path(run_dir) / args.ckpt),
            'normalizer_sha256': sha256(Path(run_dir) / 'normalizer.npz'),
            'config_sha256': sha256(Path(run_dir) / 'config.json'),
            'context_source_sha256': sha256(context_source) if context_source and Path(context_source).is_file() else None,
            'hardflow_commit': git_head(hardflow_root), 'd3il_commit': git_head(d3il_root),
            'source_sha256': {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
            'context_hashes': {},
            'rng_scheme': 'seedsequence(seed, context_index, trajectory_index)' if args.paired_noise else 'legacy_global',
        }

    def start(self, ci, ti, env, context):
        import torch
        encoded = json.dumps(context, default=lambda x: np.asarray(x).tolist(), sort_keys=True).encode()
        self.metadata['context_hashes'][f'c{ci:03d}_t{ti}'] = hashlib.sha256(encoded).hexdigest()
        generator = None
        if self.args.paired_noise:
            seed = int(np.random.SeedSequence([self.args.seed, ci + getattr(self.args, 'context_start', 0), ti]).generate_state(1)[0])
            random.seed(seed)
            np.random.seed(seed)
            env.seed(seed)
            generator = torch.Generator(device=self.args.device).manual_seed(seed)
        return ChunkPostprocessor(self.args.postprocess, self.args.execute_steps, self.config), generator

    def finish(self, key, processor, actions, desired, actual, starts, initial_des, initial_actual, rollouts):
        self.rows[key] = rollout_metrics(actions, desired, np.asarray(actual)[:, :2], starts,
                                         self.dt, initial_des, initial_actual)
        self.plans[key] = processor.diagnostics
        rollouts[key + '_plan_starts'] = np.asarray(starts, dtype=bool)
        rollouts[key + '_initial_des'] = np.asarray(initial_des, dtype=np.float64)
        rollouts[key + '_initial_c'] = np.asarray(initial_actual, dtype=np.float64)

    def save(self, results, out_dir):
        plans = [p for row in self.plans.values() for p in row]
        results.update(self.metadata)
        results['smoothness'] = summarize_metrics(self.rows)
        results['postprocess_ms'] = float(np.mean([p['postprocess_ms'] for p in plans])) if plans else None
        results['postprocess_max_ms'] = max((p['postprocess_ms'] for p in plans), default=None)
        qp_times = [p['qp_ms'] for p in plans if 'qp_ms' in p]
        results['qp_ms'] = float(np.mean(qp_times)) if qp_times else None
        results['qp_solved_count'] = sum(p['solver_status'] == 'Solved' for p in plans)
        with open(Path(out_dir) / 'smoothness.json', 'w') as f:
            json.dump({'episodes': self.rows, 'plans': self.plans}, f, indent=2, allow_nan=False)
