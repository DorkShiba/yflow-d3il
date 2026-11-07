"""Parallel Pushing/Inserting comparison, preserving global context-index RNG seeds."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import numpy as np
from compare_smoothness import MATCH_FIELDS, compare
from smoothness import summarize_metrics


def merge_shards(folders, destination):
    records = []
    for folder in folders:
        folder = Path(folder)
        records.append((json.loads((folder/'results.json').read_text()), folder))
    records.sort(key=lambda pair: pair[0]['context_start'])
    merged = dict(records[0][0])
    offset = 0
    arrays, episodes, plans, context_hashes = {}, {}, {}, {}
    inserting = merged['task'] == 'inserting'
    mode_ids = range(1, 7) if inserting else range(4)
    top_keys = ['successes', 'modes', 'mean_distance', 'lengths']
    if inserting: top_keys += ['n_boxes', 'demo_len']
    top = {k: [] for k in top_keys}
    context_files = []
    weighted_plan_time, total_plans = 0., 0
    for result, folder in records:
        if result['context_start'] != offset:
            raise ValueError('Shard context ranges must be contiguous from zero')
        for key in (*MATCH_FIELDS, 'postprocess', 'lipo_config'):
            if key not in ('context_start', 'n_contexts', 'context_hashes') and result[key] != merged[key]:
                raise ValueError(f'Shard mismatch: {key}')
        if inserting: context_files.extend(result['context_files'])
        detail = json.loads((folder/'smoothness.json').read_text())
        def rekey(key):
            match = re.fullmatch(r'c(\d+)(_t\d+.*)', key)
            if not match:
                raise ValueError(f'Unknown rollout key: {key}')
            index = int(match[1])
            if not 0 <= index < result['n_contexts']:
                raise ValueError('Invalid local context index')
            return f'c{index + offset:03d}{match[2]}'
        for key, row in detail['episodes'].items(): episodes[rekey(key)] = row
        for key, row in detail['plans'].items(): plans[rekey(key)] = row
        for key, value in result['context_hashes'].items(): context_hashes[rekey(key)] = value
        count = sum(len(p) for p in detail['plans'].values())
        weighted_plan_time += result['plan_ms'] * count
        total_plans += count
        with np.load(folder/'rollouts.npz', allow_pickle=False) as z:
            for key in z.files:
                if key in top: top[key].append(z[key])
                else: arrays[rekey(key)] = z[key]
        offset += result['n_contexts']
    arrays.update({k: np.concatenate(v, axis=0) for k, v in top.items()})
    successes, modes = arrays['successes'], arrays['modes']
    if successes.shape != (offset, merged['n_traj']) or len(episodes) != successes.size:
        raise ValueError('Incomplete shards')
    flat_plans = [p for rows in plans.values() for p in rows]
    probs = np.stack([np.sum((modes == m) & (successes == 1), axis=1) for m in mode_ids], axis=1)
    probs = probs / (probs.sum(axis=1, keepdims=True) + 1e-12)
    entropy = float(-(probs * np.log(probs + 1e-12) / np.log(len(mode_ids))).sum(axis=1).mean())
    qp_times = [p['qp_ms'] for p in flat_plans if 'qp_ms' in p]
    merged.update(n_contexts=offset, context_start=0, context_hashes=context_hashes,
                  success_rate=float(successes.mean()), entropy=entropy,
                  score=.5*(float(successes.mean())+entropy),
                  mean_distance=float(np.mean(arrays['mean_distance'])),
                  mean_steps=float(np.mean(arrays['lengths'])),
                  mode_counts={str(m): int(np.sum((modes == m) & (successes == 1))) for m in mode_ids},
                  plan_ms=weighted_plan_time/total_plans, qp_ms=float(np.mean(qp_times)) if qp_times else None,
                  postprocess_ms=float(np.mean([p['postprocess_ms'] for p in flat_plans])),
                  postprocess_max_ms=max(p['postprocess_ms'] for p in flat_plans),
                  qp_solved_count=sum(p['solver_status']=='Solved' for p in flat_plans),
                  smoothness=summarize_metrics(episodes), shard_folders=[str(f) for _, f in records],
                  worker_wall_sec_sum=sum(r['wall_sec'] for r, _ in records))
    if inserting:
        merged.update(context_files=context_files, mean_demo_len=float(arrays['demo_len'].mean()),
                      mean_steps_success=float(arrays['lengths'][successes == 1].mean()) if successes.any() else None,
                      one_box_rate=float((arrays['n_boxes'] >= 1).mean()),
                      two_box_rate=float((arrays['n_boxes'] >= 2).mean()),
                      three_box_rate=float((arrays['n_boxes'] >= 3).mean()))
    merged.pop('wall_sec', None)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    (destination/'results.json').write_text(json.dumps(merged, indent=2, allow_nan=False)+'\n')
    (destination/'smoothness.json').write_text(json.dumps({'episodes':episodes,'plans':plans}, indent=2, allow_nan=False)+'\n')
    np.savez_compressed(destination/'rollouts.npz', **arrays)
    return destination


def main():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=['pushing', 'inserting'], default='pushing')
    p.add_argument('--output-root', required=True)
    p.add_argument('--d3il-root', required=True)
    p.add_argument('--n-contexts', type=int, default=30)
    p.add_argument('--n-traj', type=int, default=4)
    p.add_argument('--shard-contexts', type=int, default=3)
    p.add_argument('--workers', type=int, default=9)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    if min(args.n_contexts, args.n_traj, args.shard_contexts, args.workers) < 1:
        p.error('Counts must be positive')
    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    jobs = []
    for start in range(0, args.n_contexts, args.shard_contexts):
        for mode in ('none','blend','lipo'):
            name = f'{mode}_{start:03d}'
            cmd = [sys.executable, str(here/f'eval_{args.task}.py'), '--run-dir', str(here/'results'/f'{args.task}_baseline'),
                   '--ckpt', 'baseline_ema.pt', '--execute-steps','8','--ode-steps','10','--paired-noise',
                   '--postprocess',mode,'--seed',str(args.seed),'--context-start',str(start),
                   '--n-contexts',str(min(args.shard_contexts,args.n_contexts-start)),
                   '--n-traj',str(args.n_traj),'--device','cuda:0','--output-root',str(root/'shards'/name),
                   '--d3il-root',str(Path(args.d3il_root).resolve())]
            jobs.append({'name':name,'mode':mode,'command':cmd})
    (root/'commands.json').write_text(json.dumps(jobs,indent=2)+'\n')
    began = time.time()
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', MUJOCO_GL='egl')
    def execute(job):
        with open(root/(job['name']+'.log'),'w') as log:
            subprocess.run(job['command'], stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
        folders = list((root/'shards'/job['name']).glob('eval_*'))
        if len(folders)!=1: raise RuntimeError(f'Unexpected output count: {job}')
        return job['mode'], folders[0]
    folders = {m:[] for m in ('none','blend','lipo')}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(execute, job):job for job in jobs}
        for future in as_completed(futures):
            mode, folder = future.result()
            folders[mode].append(folder)
            print(f'completed {futures[future]["name"]} ({sum(map(len,folders.values()))}/{len(jobs)}) elapsed {time.time()-began:.0f}s',flush=True)
    combined = {mode:merge_shards(paths,root/'combined'/mode) for mode,paths in folders.items()}
    for mode in ('blend','lipo'):
        (root/f'comparison_{mode}.json').write_text(json.dumps(compare(combined['none'],combined[mode]),indent=2,allow_nan=False)+'\n')
    (root/'COMPLETE.json').write_text(json.dumps({'wall_sec':time.time()-began,'completed_utc':datetime.now(timezone.utc).isoformat(),'episodes_per_mode':args.n_contexts*args.n_traj,'workers':args.workers},indent=2)+'\n')
    print('COMPLETE',flush=True)


if __name__ == '__main__':
    main()
