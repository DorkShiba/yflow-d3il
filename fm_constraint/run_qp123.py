"""Eight combinations of methods 1/2/3, with optional synchronous RTC and fixed jerk QP."""
import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import subprocess
import sys
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from configs.config import ExperimentConfig, REPO

METHODS = ('projection', 'pf2mp', 'vlsa')
ARMS = [''.join(map(str, bits)) for bits in itertools.product((0, 1), repeat=3)]


def add_protocol_options(parser):
    parser.add_argument('--rtc', action='store_true', help='Use synchronous RTC (delay=0, beta=5)')
    parser.add_argument('--condition', help='Frozen ID/OOD condition, e.g. id or ABC_mid')


def protocol_options(args):
    flags = ['--rtc'] if args.rtc else []
    if args.condition:
        flags += ['--condition', args.condition]
    return flags


def arm_name(arm, rtc):
    state = 'on' if rtc else 'off'
    return f'm123_{arm}_m4{state}_m5on'


def run_parallel(args, output, overrides):
    """One sequential worker per GPU; only the parent writes the summary."""
    devices = args.devices
    assignments = {device: [] for device in devices}
    for i, arm in enumerate(ARMS):
        slot = i % len(devices)
        if (i // len(devices)) % 2:
            slot = len(devices) - 1 - slot
        assignments[devices[slot]].append(arm)
    jobs = {}
    for device, arms in assignments.items():
        jobs[device] = []
        for seed in args.seeds:
            for arm in arms:
                dest = output / f'seed_{seed}' / arm_name(arm, args.rtc)
                cmd = [sys.executable, '-u', '-m', 'fm_constraint.run_qp123',
                       '--task', args.task, '--device', device, '--n-contexts', str(args.n_contexts),
                       '--n-traj', str(args.n_traj), '--seeds', str(seed), '--arm', arm,
                       '--qp-bound-m', str(args.qp_bound_m), '--output-root', str(dest), *protocol_options(args)]
                for field in ('hardflow_root', 'd3il_root', 'data_root'):
                    if getattr(args, field):
                        cmd += ['--' + field.replace('_', '-'), overrides['paths.' + field]]
                jobs[device].append((seed, arm, dest, cmd))
                if args.dry_run:
                    print(json.dumps(cmd), flush=True)
    if args.dry_run:
        return
    (output / 'logs').mkdir()
    (output / 'gpu_assignments.json').write_text(json.dumps(assignments, indent=2))
    active, lock, stop = {}, threading.Lock(), threading.Event()
    env = os.environ.copy()
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        env[key] = str(args.cpu_threads)

    def worker(device):
        rows, errors = [], []
        for seed, arm, dest, cmd in jobs[device]:
            if stop.is_set():
                break
            log_path = output / 'logs' / f'seed_{seed}_m123_{arm}.log'
            try:
                if dest.exists():
                    raise FileExistsError(dest)
                print(f'START {device} seed={seed} arm={arm} log={log_path}', flush=True)
                with log_path.open('w', encoding='utf-8') as log:
                    with lock:
                        if stop.is_set():
                            break
                        process = subprocess.Popen(cmd, cwd=REPO, env=env,
                                                   stdout=log, stderr=subprocess.STDOUT)
                        active[device] = process
                    code = process.wait()
                    with lock:
                        active.pop(device, None)
                if code:
                    raise RuntimeError(f'Child exit status {code}; see {log_path}')
                files = list(dest.glob('*/results.json'))
                if len(files) != 1:
                    raise RuntimeError(f'Expected one result under {dest}')
                result = json.loads(files[0].read_text())
                rows.append({'seed': seed, 'arm': arm, 'device': device, 'result': str(files[0]),
                             **{key: result[key] for key in ('success_rate', 'n_contexts', 'n_traj',
                                'plan_ms', 'postprocess_ms', 'smoothness', 'filter_abort_rate',
                                'executed_acceleration_qp_exceed_rate')}})
                print(f'DONE {device} seed={seed} arm={arm}', flush=True)
            except Exception as error:
                errors.append({'seed': seed, 'arm': arm, 'device': device, 'error': str(error)})
                print(f'FAILED {device} seed={seed} arm={arm}: {error}', flush=True)
                break  # Stop this GPU's remaining jobs; let other GPUs finish.
        return rows, errors

    pool = ThreadPoolExecutor(max_workers=len(devices))
    summary, errors = [], []
    try:
        futures = [pool.submit(worker, device) for device in devices]
        for future in as_completed(futures):
            rows, failures = future.result()
            summary.extend(rows); errors.extend(failures)
            summary.sort(key=lambda row: (row['seed'], row['arm']))
            (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
            (output / 'failures.json').write_text(json.dumps(errors, indent=2), encoding='utf-8')
    except BaseException:
        stop.set()
        with lock:
            processes = list(active.values())
        for process in processes:
            process.terminate()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    if errors:
        raise RuntimeError(f'{len(errors)} worker(s) failed; inspect {output / "failures.json"}')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=['pushing', 'inserting'], default='pushing')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--devices', nargs='+', help='Parallel GPUs, e.g. cuda:0 cuda:1 ... cuda:7')
    p.add_argument('--cpu-threads', type=int, default=2, help='Numerical CPU threads per parallel worker')
    p.add_argument('--n-contexts', type=int, default=0, help='0 = all available contexts')
    p.add_argument('--context-start', type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument('--n-traj', type=int, default=4)
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--hardflow-root')
    p.add_argument('--d3il-root')
    p.add_argument('--data-root')
    p.add_argument('--output-root')
    p.add_argument('--qp-bound-m', type=float, default=0.0003)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--arm', choices=ARMS, help=argparse.SUPPRESS)
    add_protocol_options(p)
    args = p.parse_args(argv)
    if args.condition:
        from .fixed_conditions import load_condition
        load_condition(args.task, args.condition)
    if args.devices and (len(args.devices) > 8 or len(set(args.devices)) != len(args.devices)
                         or any(not d.startswith('cuda:') or not d[5:].isdigit() for d in args.devices)):
        p.error('--devices requires 1 to 8 unique cuda:N devices')
    if args.cpu_threads < 1:
        p.error('--cpu-threads must be positive')
    if args.arm and (args.devices or args.dry_run):
        p.error('--arm cannot be combined with --devices or --dry-run')
    if args.context_start < 0 or (args.context_start and not args.arm):
        p.error('--context-start is nonnegative and only supported for single-arm jobs')
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
                 'eval.context_start': args.context_start,
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
        evaluate(cfg, qp5_bound=args.qp_bound_m, rtc=args.rtc, condition=args.condition)
        return
    if args.devices and not args.dry_run:
        if output.exists():
            raise FileExistsError(output)
        cfg.validate('eval')
        for name in ('hardflow_root', 'd3il_root'):
            if not Path(cfg.values['paths'][name]).is_dir():
                raise FileNotFoundError(f'{name}: {cfg.values["paths"][name]}')
        import torch
        if not torch.cuda.is_available() or any(int(d[5:]) >= torch.cuda.device_count() for d in args.devices):
            p.error('Requested GPUs are not visible to PyTorch; check nvidia-smi and CUDA_VISIBLE_DEVICES')
        # Build the shared training-only cache once before launching eight workers.
        # Workers still verify/reuse it under its existing file lock.
        from .pf2mp import prepare_potential
        cfg.values['device'] = 'cpu'
        cfg.values['methods']['pf2mp']['enabled'] = True
        print('Preparing shared PF2MP cache on CPU...', flush=True)
        prepare_potential(cfg, REPO / 'codex/pf2mp_cache')
    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=False)
        (output / 'experiment.json').write_text(json.dumps({
            'method123_source': '5acb1034d0819453fcf6364477cb25f1c30e6d8b',
            'bit_order': METHODS, 'arms': ARMS, 'method4': args.rtc, 'method5': True,
            'args': vars(args), 'reference_arm': '000',
            'order': ('RTC + ' if args.rtc else '') + 'FM + PF2MP / Euler / projection, then VLSA, then QP5',
        }, indent=2), encoding='utf-8')
    if args.devices:
        run_parallel(args, output, overrides)
        print(f'Output: {output}', flush=True)
        return
    summary = []
    for seed in args.seeds:
        for arm in ARMS:
            dest = output / f'seed_{seed}' / arm_name(arm, args.rtc)
            cmd = [sys.executable, '-m', 'fm_constraint.run_qp123', '--task', args.task,
                   '--device', args.device, '--n-contexts', str(args.n_contexts),
                   '--n-traj', str(args.n_traj), '--seeds', str(seed), '--arm', arm,
                   '--qp-bound-m', str(args.qp_bound_m), '--output-root', str(dest), *protocol_options(args)]
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
