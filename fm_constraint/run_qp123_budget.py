"""Time-limited multi-GPU evaluation, resumable at context boundaries."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

from configs.config import ExperimentConfig, REPO
from .run_qp123 import ARMS, add_protocol_options, protocol_options


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dataset_digest(root):
    h = hashlib.sha256()
    for path in sorted(Path(root).glob('*/*.npz')):
        h.update(path.relative_to(root).as_posix().encode('utf-8'))
        h.update(digest(path).encode('ascii'))
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def terminate(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def completed(job_dir):
    marker = job_dir / 'complete.json'
    if not marker.exists():
        return None
    record = json.loads(marker.read_text(encoding='utf-8'))
    for name, sha in record['files'].items():
        path = (job_dir / name).resolve()
        if not path.is_relative_to(job_dir.resolve()) or not path.is_file() or digest(path) != sha:
            raise ValueError(f'Completed result missing/changed: {path}')
    return record


def schedule(jobs, output, deadline, env):
    """A job is one seed/arm/context; complete markers are committed last."""
    stop, lock, active = threading.Event(), threading.Lock(), {}
    errors = []
    all_dirs = [directory for group in jobs.values() for directory, _ in group]
    rows = {str(d.relative_to(output)): completed(d) for d in all_dirs}
    rows = {k: v for k, v in rows.items() if v is not None}

    def progress(status):
        save_json(output / 'progress.json', {'status': status, 'total_jobs': len(all_dirs),
            'completed_jobs': len(rows), 'remaining_jobs': len(all_dirs) - len(rows), 'errors': errors})
        save_json(output / 'summary.json', rows)

    def worker(device):
        for directory, make_command in jobs[device]:
            key = str(directory.relative_to(output))
            if key in rows:
                continue
            if stop.is_set() or time.monotonic() >= deadline:
                stop.set()
                break
            directory.mkdir(parents=True, exist_ok=True)
            attempt = directory / ('attempt_' + uuid.uuid4().hex)
            log_path = directory / (attempt.name + '.log')
            print(f'START {device} {key}', flush=True)
            process = None
            try:
                with log_path.open('w', encoding='utf-8') as log:
                    with lock:
                        if stop.is_set() or time.monotonic() >= deadline:
                            stop.set()
                            break
                        process = subprocess.Popen(make_command(attempt), cwd=REPO, env=env,
                                                   stdout=log, stderr=subprocess.STDOUT)
                        active[device] = process
                    try:
                        code = process.wait(timeout=max(.001, deadline - time.monotonic()))
                    except subprocess.TimeoutExpired:
                        stop.set()
                        terminate(process)
                        break
                if code:
                    raise RuntimeError(f'Exit status {code}; see {log_path}')
                results = list(attempt.glob('*/results.json'))
                if len(results) != 1 or not (results[0].parent / 'rollouts.json').is_file():
                    raise ValueError(f'Incomplete evaluation output: {attempt}')
                result = json.loads(results[0].read_text(encoding='utf-8'))
                if result.get('n_contexts') != 1 or len(result.get('episodes', [])) != result.get('n_traj'):
                    raise ValueError('Unexpected episode count in completed job')
                artifacts = [results[0], results[0].parent / 'rollouts.json']
                record = {'result': results[0].relative_to(directory).as_posix(),
                          'files': {p.relative_to(directory).as_posix(): digest(p) for p in artifacts},
                          'success_rate': result['success_rate'], 'n_traj': result['n_traj'],
                          'smoothness': result['smoothness'], 'plan_ms': result['plan_ms'],
                          'postprocess_ms': result['postprocess_ms']}
                save_json(directory / 'complete.json', record)
                with lock:
                    rows[key] = record
                    progress('running')
                print(f'DONE {device} {key}', flush=True)
            except Exception as error:
                with lock:
                    errors.append({'job': key, 'error': str(error)})
                    progress('running')
                print(f'FAILED {key}: {error}', flush=True)
                break
            finally:
                if process is not None:
                    terminate(process)
                with lock:
                    active.pop(device, None)

    progress('running')
    pool = ThreadPoolExecutor(max_workers=len(jobs))
    try:
        futures = [pool.submit(worker, device) for device in jobs]
        for future in as_completed(futures):
            future.result()
    except BaseException:
        stop.set()
        with lock:
            processes = list(active.values())
        for process in processes:
            terminate(process)
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        status = ('complete' if len(rows) == len(all_dirs) else
                  'failed' if errors else 'stopped')
        progress(status)
    print(f'{status.upper()}: {len(rows)}/{len(all_dirs)} jobs; {output}', flush=True)
    if errors:
        raise RuntimeError('Some jobs failed; inspect progress.json and attempt logs before resuming')
    return status


def main(argv=None):
    started = time.monotonic()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=['pushing', 'inserting'], default='pushing')
    p.add_argument('--devices', nargs='+', default=['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'])
    p.add_argument('--max-hours', type=float, default=20.)
    p.add_argument('--cpu-threads', type=int, default=2)
    p.add_argument('--n-contexts', type=int, default=0)
    p.add_argument('--n-traj', type=int, default=4)
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--output-root', required=True)
    p.add_argument('--hardflow-root', required=True)
    p.add_argument('--d3il-root', required=True)
    p.add_argument('--data-root')
    p.add_argument('--qp-bound-m', type=float, default=.0003)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--prepare-only', action='store_true', help=argparse.SUPPRESS)
    add_protocol_options(p)
    args = p.parse_args(argv)
    if not 0 < args.max_hours <= 20 or args.cpu_threads < 1:
        p.error('Require 0 < max-hours <= 20 and positive cpu-threads')
    if args.n_contexts < 0 or args.n_traj < 1 or min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        p.error('Invalid contexts/trajectories/seeds')
    if not 0 <= args.qp_bound_m < float('inf'):
        p.error('QP bound must be finite and nonnegative')
    if (not 1 <= len(args.devices) <= 8 or
        any(not d.startswith('cuda:') or not d[5:].isdigit() for d in args.devices) or
        len({int(d[5:]) for d in args.devices}) != len(args.devices)):
        p.error('Require unique visible cuda:N devices')
    deadline = started + args.max_hours * 3600
    output = Path(args.output_root).resolve()
    overrides = {'paths.output_root': str(output), 'device': 'cpu',
                 'eval.n_traj': args.n_traj, 'methods.vlsa.execution_recheck': False}
    for key in ('hardflow_root', 'd3il_root', 'data_root'):
        if getattr(args, key):
            overrides['paths.' + key] = str(Path(getattr(args, key)).resolve())
    cfg = ExperimentConfig.from_yaml(REPO / 'configs' / f'{args.task}.yaml', overrides)
    cfg.require_output(output)
    if args.prepare_only:
        from .pf2mp import prepare_potential
        cfg.values['methods']['pf2mp']['enabled'] = True
        prepare_potential(cfg, REPO / 'codex/pf2mp_cache')
        return
    cfg.validate('eval')
    paths = cfg.values['paths']
    if args.condition:
        from .fixed_conditions import load_condition
        context_source, dataset = load_condition(args.task, args.condition)
        count = len(dataset['episodes'])
    elif args.task == 'pushing':
        import numpy as np
        context_source = Path(paths['d3il_root']) / 'environments/dataset/data/pushing/test_contexts.pkl'
        count = len(np.load(context_source, allow_pickle=True))
    else:
        context_source = Path(paths['contexts_file'])
        if not context_source.is_file():
            raise FileNotFoundError('Budgeted inserting runs require the saved inserting_val.pkl contexts')
        count = len(list((Path(paths['data_root']) / 'inserting_sincos_v1/val').glob('*.npz')))
    count = min(count, args.n_contexts) if args.n_contexts else count
    if not count:
        p.error('No evaluation contexts found')
    sources = [*Path(__file__).parent.glob('*.py'), * (REPO / 'fm_baseline').glob('*.py'),
               REPO / 'configs/config.py', REPO / 'configs' / f'{args.task}.yaml']
    identity = {'schema': 'qp123_context_jobs_v1', 'task': args.task, 'contexts': count,
                'method4': args.rtc, 'method5': True, 'condition': args.condition,
                'n_traj': args.n_traj, 'seeds': args.seeds, 'qp_bound_m': args.qp_bound_m,
                'source_sha256': {x.relative_to(REPO).as_posix(): digest(x) for x in sources},
                'checkpoint': digest(Path(paths['run_dir']) / 'baseline_ema.pt'),
                'normalizer': digest(Path(paths['run_dir']) / 'normalizer.npz'),
                'model_config': digest(Path(paths['run_dir']) / 'config.json'),
                'data_manifest': digest(Path(paths['data_root']) / f'{args.task}_sincos_v1/manifest.json'),
                'data_npz': dataset_digest(Path(paths['data_root']) / f'{args.task}_sincos_v1'),
                'contexts_sha256': digest(context_source) if context_source.is_file() else None}
    if args.dry_run:
        print(json.dumps({'contexts': count, 'total_jobs': count * 8 * len(args.seeds),
                          'max_hours': args.max_hours, 'devices': args.devices, 'identity': identity}, indent=2))
        return
    manifest = output / 'experiment.json'
    if args.resume:
        if not manifest.is_file() or json.loads(manifest.read_text()) != identity:
            raise ValueError('Resume identity mismatch (settings/code/model/data). Use the original experiment configuration.')
    else:
        output.mkdir(parents=True, exist_ok=False)
        save_json(manifest, identity)
    import torch
    if not torch.cuda.is_available() or any(int(d[5:]) >= torch.cuda.device_count() for d in args.devices):
        p.error('Requested GPUs are not visible to PyTorch')
    env = os.environ.copy()
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        env[key] = str(args.cpu_threads)
    prep = [sys.executable, '-u', '-m', 'fm_constraint.run_qp123_budget', '--prepare-only',
            '--task', args.task, '--output-root', str(output), '--hardflow-root', paths['hardflow_root'],
            '--d3il-root', paths['d3il_root'], '--data-root', paths['data_root']]
    print('Preparing shared PF2MP cache (included in time budget)...', flush=True)
    with (output / 'prepare.log').open('a', encoding='utf-8') as log:
        process = subprocess.Popen(prep, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            code = process.wait(timeout=max(.001, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            save_json(output / 'progress.json', {'status': 'stopped_during_cache_preparation'})
            print('Time budget reached during cache preparation; resume with the same output root.')
            return
        finally:
            terminate(process)
        if code:
            raise RuntimeError(f'Cache preparation failed; see {output / "prepare.log"}')
    jobs = {device: [] for device in args.devices}
    for i, arm in enumerate(ARMS):
        slot = i % len(args.devices)
        if (i // len(args.devices)) % 2:
            slot = len(args.devices) - 1 - slot
        device = args.devices[slot]
        for seed in args.seeds:
            for context in range(count):
                directory = output / f'seed_{seed}' / f'm123_{arm}' / f'context_{context:05d}'
                def command(attempt, device=device, arm=arm, seed=seed, context=context):
                    return [sys.executable, '-u', '-m', 'fm_constraint.run_qp123', '--arm', arm,
                        '--task', args.task, '--device', device, '--seeds', str(seed),
                        '--context-start', str(context), '--n-contexts', '1', '--n-traj', str(args.n_traj),
                        '--qp-bound-m', str(args.qp_bound_m), '--output-root', str(attempt),
                        '--hardflow-root', paths['hardflow_root'], '--d3il-root', paths['d3il_root'],
                        '--data-root', paths['data_root'], *protocol_options(args)]
                jobs[device].append((directory, command))
    save_json(output / 'gpu_assignments.json', {d: [str(x.relative_to(output)) for x, _ in group]
                                               for d, group in jobs.items()})
    schedule(jobs, output, deadline, env)


if __name__ == '__main__':
    main()
