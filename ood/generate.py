"""Pushing·Inserting OOD 평가 사례를 생성한다 (D3IL 시뮬레이터 필요).

    python -m ood.generate --d3il-root ../d3il            # data/ood/{pushing,inserting}.jsonl
    python -m ood.verify                                  # 해시·개수·seed 검사

행 하나 = (조건, 평가 context). context는 Pushing 공식 test 60개, Inserting
fm_baseline/contexts/inserting_val.pkl 82개다. B(로봇 시작)는 실제로 손을 옮겨
블록이 움직이지 않고(1 mm 미만) 손이 목표 5 mm 안에 도착한 경우만 채택한다.
기존 출력 파일은 덮어쓰지 않는다.
"""
import argparse
import json
import math
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'fm_baseline'))
from fm_constraint.tasks import get_task  # noqa: E402
from train import resolve_d3il  # noqa: E402

from .apply import CLEAR, context_to_json, file_sha256, shift_blocks  # noqa: E402
from .conditions import (LEVELS, POLICY_REPEATS, PUSH_PHASE, ROOT_SEED, conditions,  # noqa: E402
                         context_seeds, factor_values)

VERSION = 'v2'
HAND_TOL = 0.005        # m: 도착한 손과 목표의 최대 거리
BLOCK_TOL = 0.001       # m: 손을 옮기는 동안 블록 최대 이동
B_ANGLES = 128          # 시도할 방향 후보 수
B_SIM_TRIES = 8         # 시뮬레이터로 확인할 최대 후보 수


def path_clear(spec, a, b, blocks, radii, walls):
    """손의 직선 경로 a→b가 블록·벽과 떨어져 있는가. 이미 시작점 가까이 있는 블록은
    더 가까워지지만 않으면 된다(이후 시뮬레이터 검사가 접촉 여부를 확인한다)."""
    (x0, x1), (y0, y1) = spec.workspace
    if not (x0 + spec.hand_r <= b[0] <= x1 - spec.hand_r and y0 + spec.hand_r <= b[1] <= y1 - spec.hand_r):
        return False
    need = [min(r + spec.hand_r + CLEAR, np.linalg.norm(a - q)) for q, r in zip(blocks, radii)]
    for u in np.linspace(0.0, 1.0, 21):
        p = a + u * (b - a)
        if any(np.linalg.norm(p - q) < n - 1e-9 for q, n in zip(blocks, need)):
            return False
        if any(np.linalg.norm(np.maximum(np.abs(p - c) - h, 0.0)) < spec.hand_r + CLEAR for c, h in walls):
            return False
    return True


def robot_start(spec, env, ctx, hand0, d, seed, walls):
    """B: seed로 정한 방향 순서대로 후보를 보고, 시뮬레이터에서 통과한 첫 위치를 쓴다."""
    rng = random.Random(seed)
    angles = [rng.uniform(0.0, 2 * math.pi) for _ in range(B_ANGLES)]
    blocks = [np.asarray(spec.ctx_box_xy(ctx, i), dtype=float) for i in range(spec.n_boxes)]
    record = {'from': hand0.tolist(), 'to': None, 'angle_deg': None, 'tries': 0}
    for theta in angles:
        q = hand0 + d * np.array([math.cos(theta), math.sin(theta)])
        if not path_clear(spec, hand0, q, blocks, spec.block_radii(), walls):
            continue
        record['tries'] += 1
        spec.reset(env, ctx)
        before = [np.asarray(b[0][:2]) for b in spec.read(env).boxes]
        spec.move_hand(env, q)
        state = spec.read(env)
        moved = max([np.linalg.norm(np.asarray(b[0][:2]) - b0) for b, b0 in zip(state.boxes, before)] or [0.0])
        error = float(np.linalg.norm(state.c[:2] - q))
        if moved < BLOCK_TOL and error < HAND_TOL:
            record.update(to=q.tolist(), angle_deg=math.degrees(theta), hand_error_m=error,
                          block_moved_m=float(moved), hand_actual=state.c[:2].tolist())
            return record
        if record['tries'] >= B_SIM_TRIES:
            break
    return record


def generate_task(task, d3il, out_root):
    spec = get_task(task)
    spec.setup(d3il)
    env = spec.make_env(spec.max_steps)
    contexts = spec.contexts(argparse.Namespace(contexts='test', n_contexts=0, contexts_file=None, d3il_root=d3il), {})
    walls = spec.walls(env)
    spec.reset(env, contexts[0][1])
    hand0 = spec.read(env).c[:2].copy()                    # 모든 context에서 같은 시연 시작점
    rows, stats = [], {}
    shift_cache, start_cache = {}, {}
    for cond, selected in conditions():
        values = factor_values(task, selected)
        failures = {'A_blocks_not_shifted': 0, 'B_without_robot_shift': 0}
        for context_id, base in contexts:
            seeds = context_seeds(task, context_id)
            ctx, shift = base, None
            if values['A']:
                key = (context_id, values['A']['shift_m'])
                if key not in shift_cache:
                    shift_cache[key] = shift_blocks(spec, walls, base, values['A']['shift_m'], hand0)
                ctx, shift = shift_cache[key]
                failures['A_blocks_not_shifted'] += sum(e['face'] is None for e in shift)
            robot = None
            if values['B']:
                key = (context_id, values['B']['dist_m'], bool(values['A']), values['A'] and values['A']['shift_m'])
                if key not in start_cache:
                    start_cache[key] = robot_start(spec, env, ctx, hand0, values['B']['dist_m'], seeds['B'], walls)
                robot = start_cache[key]
                failures['B_without_robot_shift'] += robot['to'] is None
            rows.append({
                'id': f'{task}/{cond}/{context_id}', 'task': task, 'cond': cond, 'factors': values,
                'context_id': context_id, 'context': context_to_json(ctx),
                'shift': shift, 'robot': robot, 'robot_start': robot and robot['to'],
                'noise': values['C'] and {'pos_m': values['C']['pos_m'], 'yaw_deg': values['C']['yaw_deg'],
                                          'seed': seeds['C']},
                'push': values['D'] and {'dist_m': values['D']['dist_m'], 'phase': values['D']['phase'],
                                         'step': spec.push_step, 'seed': seeds['D']},
                'delay_steps': values['E']['delay_steps'] if values['E'] else 0,
                'policy_seeds': seeds['policy']})
        stats[cond] = {'factors': values, **{k: v for k, v in failures.items() if v}}
        print(f'  {task} {cond:9s} ' + ' '.join(f'{k}={v}' for k, v in failures.items() if v), flush=True)
    path = out_root / f'{task}.jsonl'
    with path.open('x') as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + '\n')
    return {'file': path.name, 'sha256': file_sha256(path), 'n_rows': len(rows), 'n_contexts': len(contexts),
            'contexts': 'd3il pushing test_contexts.pkl' if task == 'pushing' else 'fm_baseline/contexts/inserting_val.pkl',
            'demo_hand_start': hand0.tolist(), 'push_step': spec.push_step, 'levels': LEVELS[task],
            'conditions': stats}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--task', nargs='+', default=list(LEVELS), choices=sorted(LEVELS))
    parser.add_argument('--d3il-root', default=None)
    parser.add_argument('--out', default=str(REPO / 'data/ood'))
    args = parser.parse_args()
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    if (out_root / 'manifest.json').exists():
        raise SystemExit(f'{out_root} already holds a dataset; use a new --out')
    d3il = resolve_d3il(args.d3il_root)
    try:
        commit = subprocess.check_output(['git', '-C', d3il, 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    manifest = {'version': VERSION, 'created_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                'root_seed': ROOT_SEED, 'policy_repeats': POLICY_REPEATS, 'push_phase': PUSH_PHASE,
                'd3il_commit': commit, 'conditions': [name for name, _ in conditions()],
                'seed_rule': 'seed_for(root_seed, task, context_id, factor); noise value = '
                             'gauss(seed_for(noise_seed, rollout, step, block, component)); '
                             'policy_seeds = seed_for(root_seed, task, context_id, "policy", j)',
                'tasks': {}}
    cwd = os.getcwd()
    for task in args.task:
        print(f'== {task}', flush=True)
        manifest['tasks'][task] = generate_task(task, d3il, out_root)
        os.chdir(cwd)
    (out_root / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print('saved', out_root)


if __name__ == '__main__':
    main()
