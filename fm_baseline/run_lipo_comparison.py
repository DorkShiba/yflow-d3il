"""Run frozen-policy baseline, blend-only and LiPo, then paired comparisons."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from compare_smoothness import compare


def main():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=['pushing', 'inserting'], default='pushing')
    p.add_argument('--run-dir', default=None)
    p.add_argument('--output-root', default=None)
    p.add_argument('--n-contexts', type=int, default=3)
    p.add_argument('--n-traj', type=int, default=2)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default=None)
    p.add_argument('--d3il-root', default=None)
    p.add_argument('--lipo-blend-steps', type=int, default=4)
    p.add_argument('--lipo-blend-bound-m', type=float, default=.002)
    p.add_argument('--lipo-path-bound-m', type=float, default=.0003)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = Path(args.output_root or here / 'runs' / f'lipo_comparison_{stamp}').resolve()
    run = Path(args.run_dir or here / 'results' / f'{args.task}_baseline').resolve()
    commands = {}
    for mode in ('none', 'blend', 'lipo'):
        cmd = [sys.executable, str(here / f'eval_{args.task}.py'), '--run-dir', str(run),
               '--ckpt', 'baseline_ema.pt', '--weights', 'ema', '--execute-steps', '8',
               '--ode-steps', '10', '--paired-noise', '--postprocess', mode,
               '--output-root', str(output / mode), '--tag', mode]
        for name in ('n_contexts', 'n_traj', 'seed', 'device', 'd3il_root',
                     'lipo_blend_steps', 'lipo_blend_bound_m', 'lipo_path_bound_m'):
            value = getattr(args, name)
            if value is not None:
                cmd.extend(['--' + name.replace('_', '-'), str(value)])
        commands[mode] = cmd
    if args.dry_run:
        print(json.dumps(commands, indent=2))
        return
    output.mkdir(parents=True, exist_ok=False)
    (output / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
    folders = {}
    for mode, command in commands.items():
        subprocess.run(command, check=True)
        runs = list((output / mode).glob('eval_*'))
        if len(runs) != 1:
            raise RuntimeError(f'Expected exactly one {mode} run, got {runs}')
        folders[mode] = runs[0]
    for mode in ('blend', 'lipo'):
        report = compare(folders['none'], folders[mode])
        (output / f'comparison_{mode}.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f'Paired comparisons saved: {output}')


if __name__ == '__main__':
    main()
