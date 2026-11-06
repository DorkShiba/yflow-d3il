"""Run baseline / QP / RTC / RTC+QP with matched simulated delay and RNG."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=['pushing', 'inserting'], default='pushing')
    p.add_argument('--run-dir')
    p.add_argument('--output-root')
    p.add_argument('--device', default='cpu')
    p.add_argument('--hardflow-root')
    p.add_argument('--d3il-root')
    p.add_argument('--n-contexts', type=int, default=3)
    p.add_argument('--n-traj', type=int, default=2)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--execute-steps', type=int, default=8)
    p.add_argument('--ode-steps', type=int, default=10)
    p.add_argument('--inference-delay-steps', type=int, default=0)
    p.add_argument('--rtc-beta', type=float, default=5.)
    p.add_argument('--qp-bound-m', type=float, default=.0003)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    here = Path(__file__).resolve().parent
    run = Path(args.run_dir or here / 'results' / f'{args.task}_baseline').resolve()
    output = Path(args.output_root or here.parent / 'codex' / 'rtc_runs' /
                  datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')).resolve()
    paths = {}
    for name, rtc, post in [('baseline', False, 'none'), ('qp', False, 'qp'),
                            ('rtc', True, 'none'), ('rtc_qp', True, 'qp')]:
        arm = output / name
        cmd = [sys.executable, str(here / f'eval_{args.task}.py'), '--run-dir', str(run),
               '--ckpt', 'baseline_ema.pt', '--paired-noise', '--postprocess', post,
               '--output-root', str(arm), '--tag', name, '--lipo-path-bound-m', str(args.qp_bound_m)]
        for field in ('device', 'n_contexts', 'n_traj', 'seed', 'execute_steps',
                      'ode_steps', 'inference_delay_steps', 'rtc_beta', 'hardflow_root', 'd3il_root'):
            value = getattr(args, field)
            if value is not None:
                cmd += ['--' + field.replace('_', '-'), str(value)]
        if rtc:
            cmd.append('--rtc')
        print(json.dumps(cmd, ensure_ascii=False), flush=True)
        if args.dry_run:
            continue
        if arm.exists():
            raise FileExistsError(f'Refusing to reuse output directory: {arm}')
        subprocess.run(cmd, check=True)
        candidates = list(arm.glob('*/results.json'))
        if len(candidates) != 1:
            raise RuntimeError(f'Expected one evaluation in {arm}')
        paths[name] = candidates[0].parent
    if not args.dry_run:
        from compare_smoothness import compare
        for name in ('qp', 'rtc', 'rtc_qp'):
            report = compare(paths['baseline'], paths[name])
            (output / f'comparison_{name}.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
