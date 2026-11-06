"""Eight combinations of methods 1/2/3, with RTC OFF and jerk QP ON."""
import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import subprocess
import sys

from configs.config import ExperimentConfig, REPO

METHODS = ('projection', 'pf2mp', 'vlsa')
ARMS = [''.join(map(str, bits)) for bits in itertools.product((0, 1), repeat=3)]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=['pushing', 'inserting'], default='pushing')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--n-contexts', type=int, default=0, help='0 = all available contexts')
    p.add_argument('--n-traj', type=int, default=4)
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--hardflow-root')
    p.add_argument('--d3il-root')
    p.add_argument('--data-root')
    p.add_argument('--output-root')
    p.add_argument('--qp-bound-m', type=float, default=0.0003)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--arm', choices=ARMS, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.n_contexts < 0 or args.n_traj < 1 or any(s < 0 for s in args.seeds):
        p.error('contexts >= 0, trajectories >= 1, seeds >= 0 required')
    if len(set(args.seeds)) != len(args.seeds):
        p.error('Duplicate seeds')
    if not 0 <= args.qp_bound_m < float('inf'):
        p.error('QP bound must be finite and nonnegative')
    if args.arm and (len(args.seeds) != 1 or not args.output_root):
        p.error('Single arm requires one seed and explicit output root')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = Path(args.output_root or REPO / 'codex' / 'qp123_runs' / stamp).resolve()
    overrides = {'device': args.device, 'paths.output_root': str(output),
                 'eval.n_contexts': args.n_contexts, 'eval.n_traj': args.n_traj,
                 'methods.vlsa.execution_recheck': False}
    for field in ('hardflow_root', 'd3il_root', 'data_root'):
        if getattr(args, field):
            overrides['paths.' + field] = str(Path(getattr(args, field)).resolve())
    cfg = ExperimentConfig.from_yaml(REPO / 'configs' / f'{args.task}.yaml', overrides)
    cfg.require_output(output)
    if args.arm:
        cfg.values['eval']['seed'] = args.seeds[0]
        for name, bit in zip(METHODS, args.arm):
            cfg.values['methods'][name]['enabled'] = bit == '1'
        from .evaluation import evaluate
        evaluate(cfg, qp5_bound=args.qp_bound_m)
        return
    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=False)
        (output / 'experiment.json').write_text(json.dumps({
            'method123_source': '5acb1034d0819453fcf6364477cb25f1c30e6d8b',
            'bit_order': METHODS, 'arms': ARMS, 'method4': False, 'method5': True,
            'args': vars(args), 'reference_arm': '000',
            'order': 'PF2MP / Euler / projection, then VLSA, then QP5',
        }, indent=2), encoding='utf-8')
    summary = []
    for seed in args.seeds:
        for arm in ARMS:
            dest = output / f'seed_{seed}' / f'm123_{arm}_m4off_m5on'
            cmd = [sys.executable, '-m', 'fm_constraint.run_qp123', '--task', args.task,
                   '--device', args.device, '--n-contexts', str(args.n_contexts),
                   '--n-traj', str(args.n_traj), '--seeds', str(seed), '--arm', arm,
                   '--qp-bound-m', str(args.qp_bound_m), '--output-root', str(dest)]
            for field in ('hardflow_root', 'd3il_root', 'data_root'):
                if getattr(args, field):
                    cmd += ['--' + field.replace('_', '-'), overrides['paths.' + field]]
            print(json.dumps(cmd), flush=True)
            if args.dry_run:
                continue
            if dest.exists():
                raise FileExistsError(dest)
            subprocess.run(cmd, cwd=REPO, check=True)
            files = list(dest.glob('*/results.json'))
            if len(files) != 1:
                raise RuntimeError(f'Expected one result under {dest}')
            result = json.loads(files[0].read_text())
            summary.append({'seed': seed, 'arm': arm, 'result': str(files[0]),
                            **{key: result[key] for key in ('success_rate', 'n_contexts',
                               'n_traj', 'plan_ms', 'postprocess_ms', 'smoothness',
                               'filter_abort_rate', 'executed_acceleration_qp_exceed_rate')}})
            (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'Output: {output}', flush=True)


if __name__ == '__main__':
    main()
