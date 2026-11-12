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

from .apply import CLEAR, context_to_json, file_sha256, free, shift_blocks  # noqa: E402
from .conditions import (LEVELS, POLICY_REPEATS, PUSH_PHASE, ROOT_SEED, conditions,  # noqa: E402
                         context_seeds, factor_values)

VERSION = 'v3'
HAND_TOL = 0.005        # m: 도착한 손과 목표의 최대 거리
BLOCK_TOL = 0.001       # m: 손을 옮기는 동안 블록 최대 이동
B_ANGLES = 128          # 시도할 방향 후보 수
B_SIM_TRIES = 8         # 시뮬레이터로 확인할 최대 후보 수
A_FAR_FROM = 0.05       # m: 이보다 큰 A 이동(A_strong)은 shift_blocks_far 규칙을 쓴다
A_NEAR_MAX = 0.05       # m: A_strong에서 다른 블록 시작 범위와 둘 최소 거리
REACH_IN, REACH_OUT = 0.30, 0.81            # m: 손 도달 고리(로봇 베이스 기준), 2026-11-12 측정
TABLE_XY = ((-0.09, 0.89), (-0.98, 0.98))   # m: 테이블 윗면(lab_surrounding.xml)
A_LOWER_STEP = 0.01      # m: A_strong 배치가 안 되면 그 context만 이만큼씩 낮춘다
SETTLE_S = 2.0           # s: A_strong 배치 후 블록 안착 확인 시간
BLOCK_SETTLE_TOL = 0.002 # m


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


def shift_blocks_far(spec, walls, ctx, d, hand_xy):
    """A_strong(d > A_FAR_FROM): ``shift_blocks``와 같은 규칙(모든 블록, 가장 가까운 면부터)에 두 가지를 더한다.

    1. 다른 블록 시작 범위와의 최소 거리를 d 대신 min(d, A_NEAR_MAX)로 둔다. d로 두면 시작 범위가
       5~10 cm 간격인 Inserting에서 거의 모든 후보가 막힌다(18 cm에서 배치 가능 context 1%).
    2. 블록 중심이 팔 도달 고리 안(REACH_IN + m ≤ |p| ≤ REACH_OUT − m, 베이스 = 원점)에 있어야 한다.
       m = 블록 외접 반지름 + 손 반지름 + CLEAR, 손이 블록 뒤까지 들어가 밀 수 있어야 하기 때문이다.
       고리는 평가와 같은 제어(고정 z·자세, step당 8 mm)로 28방향을 재서 얻었다(오차 1 cm 기준).
    TaskSpec.workspace 대신 테이블 범위를 쓴다(고리가 더 좁다). weak·mid는 ``shift_blocks`` 그대로라 기존 배치가 같다.
    ``ood/apply.py``는 결과 code fingerprint에 들어가 있어 이미 끝난 ID·A_mid 결과와 맞추려고 여기 둔다.
    """
    radii = spec.block_radii()
    pos = [np.asarray(spec.ctx_box_xy(ctx, i), dtype=float).copy() for i in range(spec.n_boxes)]
    near_d = min(d, A_NEAR_MAX)
    workspace, spec.workspace = spec.workspace, TABLE_XY
    applied = []
    try:
        for i in range(spec.n_boxes):
            (x0, x1), (y0, y1) = spec.INIT_RANGE[i]
            p = pos[i]
            candidates = sorted([(p[0] - x0, np.array([x0 - d, p[1]]), 'x-'), (x1 - p[0], np.array([x1 + d, p[1]]), 'x+'),
                                 (p[1] - y0, np.array([p[0], y0 - d]), 'y-'), (y1 - p[1], np.array([p[0], y1 + d]), 'y+')],
                                key=lambda c: c[0])
            others = [(pos[j], radii[j]) for j in range(spec.n_boxes) if j != i]
            other_ranges = [spec.INIT_RANGE[j] for j in range(spec.n_boxes)
                            if j != i and spec.INIT_RANGE[j] != spec.INIT_RANGE[i]]
            margin = radii[i] + spec.hand_r + CLEAR
            for _, q, face in candidates:
                near_other = any(math.hypot(max(rx[0] - q[0], 0.0, q[0] - rx[1]), max(ry[0] - q[1], 0.0, q[1] - ry[1]))
                                 < near_d for rx, ry in other_ranges)
                reach = REACH_IN + margin <= float(np.linalg.norm(q)) <= REACH_OUT - margin
                if not near_other and reach and free(spec, walls, q, others, hand_xy, r=radii[i]):
                    pos[i] = q
                    applied.append({'box': i, 'face': face, 'from': p.tolist(), 'to': q.tolist()})
                    break
            else:
                applied.append({'box': i, 'face': None, 'from': p.tolist(), 'to': p.tolist()})
    finally:
        spec.workspace = workspace
    for i in range(spec.n_boxes):
        ctx = spec.ctx_set_box_xy(ctx, i, pos[i])
    return ctx, applied


def blocks_settle(spec, env, ctx):
    """배치를 D3IL에 넣고 손을 시작점에 둔 채 SETTLE_S초 기다려 모든 블록이 BLOCK_SETTLE_TOL 안에 있는가.
    Pushing의 일부 두 블록 배치는 약 1 s 뒤 블록 하나가 저절로 5 cm 밀려난다(원인 미상, 배치 조합에 의존)."""
    spec.reset(env, ctx)
    hand = spec.read(env).c[:2].copy()
    for _ in range(round(SETTLE_S / 0.2)):
        spec.move_hand(env, hand, duration=0.2)
    state = spec.read(env)
    return all(np.linalg.norm(np.asarray(b[0][:2]) - spec.ctx_box_xy(ctx, i)) < BLOCK_SETTLE_TOL
               for i, b in enumerate(state.boxes))


def shift_far_checked(spec, env, walls, ctx, d, hand_xy):
    """A_strong: shift_blocks_far 배치가 모든 블록을 옮기고 시뮬레이터에서 제자리에 있을 때까지
    그 context만 d를 A_LOWER_STEP씩 낮춘다(PLAN: 가능한 최대값으로 낮춘다). 각 기록에 실제 d(shift_m)를 남긴다."""
    k = 0
    while True:
        dk = round(d - k * A_LOWER_STEP, 6)
        new, applied = shift_blocks_far(spec, walls, ctx, dk, hand_xy)
        last = dk - A_LOWER_STEP <= A_FAR_FROM
        if (all(e['face'] is not None for e in applied) and blocks_settle(spec, env, new)) or last:
            return new, [{**e, 'shift_m': dk} for e in applied]
        k += 1


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
        failures = {'A_blocks_not_shifted': 0, 'A_lowered': 0, 'B_without_robot_shift': 0}
        for context_id, base in contexts:
            seeds = context_seeds(task, context_id)
            ctx, shift = base, None
            if values['A']:
                key = (context_id, values['A']['shift_m'])
                if key not in shift_cache:
                    d = values['A']['shift_m']
                    shift_cache[key] = (shift_far_checked(spec, env, walls, base, d, hand0) if d > A_FAR_FROM
                                        else shift_blocks(spec, walls, base, d, hand0))
                ctx, shift = shift_cache[key]
                failures['A_blocks_not_shifted'] += sum(e['face'] is None for e in shift)
                failures['A_lowered'] += any(e.get('shift_m', values['A']['shift_m']) < values['A']['shift_m'] for e in shift)
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
