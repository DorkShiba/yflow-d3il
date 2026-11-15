"""Stacking OOD A(블록 초기 위치) 사례를 생성한다 (D3IL 시뮬레이터 필요).

    MUJOCO_GL=disable python -m ood.generate_stacking --d3il-root ../d3il
    # data/ood_stacking/{stacking.jsonl, stacking_<cond>_contexts.pkl, max_shift.json, manifest.json}

Pushing·Inserting용 ``ood.generate``와 분리했다. ``ood/apply.py``는 본실험 code fingerprint에 들어가
수정하면 진행 중인 결과와 어긋나고, ``data/ood``는 이미 고정된 데이터이기 때문이다.

- 기준 배치: ``fm_baseline/results/stacking_baseline/id_contexts.pkl``의 test [0, 100).
  context ID는 ``fm_constraint/eval_stacking``과 같은 ``test_NNN``이다. dev/smoke는 쓰지 않는다.
- A_weak 1 cm, A_mid 5 cm: ``ood.apply.shift_blocks`` 그대로(가장 가까운 가능한 면, 다른 블록 시작 범위 쪽 금지).
- A_strong 18 cm: ``ood.generate.shift_blocks_far`` 규칙(팔 도달 고리, 다른 시작 범위와 5 cm). Pushing·Inserting과
  달리 d를 낮추지 않고 **옮길 수 있는 블록만** 18 cm 옮기며 나머지는 제자리(face=None)로 둔다(2026-11-15 결정).
- 모든 조건에서 배치 후 손·관절을 고정한 채 2 s 동안 블록이 2 mm 안에 머무는지 시뮬레이터로 확인한다.
  벗어난 블록은 원래 위치로 되돌리고 다시 확인하며, 기록에 ``reverted_unstable``로 남긴다.
- ``max_shift.json``: A_strong 규칙으로 세 블록을 같은 d로 배치했을 때 각 블록이 옮겨지는 최대 d
  (1 cm 간격, 최대 MAX_SCAN_M, 기하만 검사)와 세 블록이 모두 옮겨지는 최대 d를 기록한다.
기존 출력 폴더는 덮어쓰지 않는다.
"""
import argparse
import collections
import copy
import json
import pickle
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'fm_baseline'))
from fm_constraint.tasks import get_task  # noqa: E402
from train import resolve_d3il  # noqa: E402

from .apply import file_sha256, shift_blocks  # noqa: E402
from .conditions import NAMES, ROOT_SEED, context_seeds, seed_for  # noqa: E402
from .generate import BLOCK_SETTLE_TOL, SETTLE_S, shift_blocks_far  # noqa: E402

VERSION = 'stacking-v1'
TASK = 'stacking'
LEVELS_M = (0.01, 0.05, 0.18)              # weak, mid, strong: Pushing·Inserting v3와 같은 수치
FAR_FROM = 0.05                            # 이보다 크면 shift_blocks_far 규칙
CONTEXTS = REPO / 'fm_baseline/results/stacking_baseline/id_contexts.pkl'
TEST = range(0, 100)                       # protocol.json evaluation_contexts.test
MAX_SCAN_M = 0.18                          # A_strong 수치까지만 본다(더 멀면 다른 시작 범위를 넘어가 비단조)
DT = 0.03                                  # env n_substeps 30 x 1 ms
OPEN_WIDTH = 0.08                          # > 0.075: env.step이 손가락을 연다
COLORS = ('red', 'green', 'blue')


def to_pkl_context(ctx, like):
    """ctx_set_box_xy 결과를 원본 pkl과 같은 (tuple of [float32 pos, quat]) 형식으로."""
    return tuple([np.asarray(pos, dtype=like[i][0].dtype), np.asarray(quat, dtype=like[i][1].dtype)]
                 for i, (pos, quat) in enumerate(ctx))


def context_json(ctx):
    return [[np.asarray(pos, dtype=float).tolist(), np.asarray(quat, dtype=float).tolist()] for pos, quat in ctx]


def shift(spec, walls, ctx, d, hand_xy):
    if d > FAR_FROM:
        return shift_blocks_far(spec, walls, ctx, d, hand_xy)
    return shift_blocks(spec, walls, ctx, d, hand_xy)


def drift(spec, env, ctx):
    """배치를 넣고 관절 목표를 시작 자세로 고정한 채 SETTLE_S초 진행. 블록별 XY 이동량(m)."""
    spec.reset(env, ctx)
    env.robot.receiveState()
    q = np.array(env.robot.current_j_pos, dtype=np.float64)
    env.controller.setSetPoint(q.copy())
    for _ in range(round(SETTLE_S / DT)):
        env.step(np.r_[q, OPEN_WIDTH])
    state = spec.read(env)
    return [float(np.linalg.norm(np.asarray(b[0][:2]) - spec.ctx_box_xy(ctx, i))) for i, b in enumerate(state.boxes)]


def settled(spec, env, base, ctx, applied):
    """안 머무는 옮긴 블록을 원위치로 되돌리며 확인. 반환 (ctx, applied, 마지막 drift)."""
    applied = copy.deepcopy(applied)
    while True:
        moved = drift(spec, env, ctx)
        bad = [e['box'] for e in applied if e['face'] is not None and moved[e['box']] >= BLOCK_SETTLE_TOL]
        if not bad:
            return ctx, applied, moved
        worst = max(bad, key=lambda i: moved[i])
        e = applied[worst]
        e.update(face=None, to=e['from'], reverted_unstable=True, drift_m=moved[worst])
        ctx = to_pkl_context(spec.ctx_set_box_xy(ctx, worst, spec.ctx_box_xy(base, worst)), base)
        if all(x['face'] is None for x in applied):
            return ctx, applied, drift(spec, env, ctx)


def max_shift(spec, walls, ctx, hand_xy):
    """같은 d로 A_strong 배치를 했을 때 블록별로 옮겨지는 최대 d와, 세 블록이 모두 옮겨지는 최대 d. 1 cm 간격."""
    best = [0.0] * spec.n_boxes
    faces = [None] * spec.n_boxes
    feasible = [[] for _ in range(spec.n_boxes)]
    all_three = 0.0
    for k in range(1, round(MAX_SCAN_M * 100) + 1):
        d = k / 100
        _, applied = shift_blocks_far(spec, walls, ctx, d, hand_xy)
        for e in applied:
            if e['face'] is not None:
                best[e['box']], faces[e['box']] = d, e['face']
                feasible[e['box']].append(k)
        if all(e['face'] is not None for e in applied):
            all_three = d
    return best, faces, feasible, all_three


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--d3il-root', default=None)
    parser.add_argument('--out', default=str(REPO / 'data/ood_stacking'))
    parser.add_argument('--limit', type=int, default=0, help='앞쪽 context 몇 개만(동작 확인용)')
    args = parser.parse_args()
    out = Path(args.out).resolve()
    if (out / 'manifest.json').exists():
        raise SystemExit(f'{out} already holds a dataset; use a new --out')
    out.mkdir(parents=True, exist_ok=True)
    d3il = resolve_d3il(args.d3il_root)
    try:
        commit = subprocess.check_output(['git', '-C', d3il, 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None

    all_ctx = pickle.load(CONTEXTS.open('rb'))
    ids = list(TEST)[:args.limit] if args.limit else list(TEST)
    spec = get_task(TASK)
    spec.setup(d3il)
    env = spec.make_env(spec.max_steps)
    walls = spec.walls(env)
    spec.reset(env, all_ctx[0])
    start = spec.read(env)                      # read()가 손 높이를 기록: 30 cm > HAND_HIGH라 손 간섭 검사는 통과
    hand0 = start.c[:2].copy()

    rows, stats, pkls = [], {}, {}
    for k, name in enumerate(NAMES):
        cond, d = f'A_{name}', LEVELS_M[k]
        pkls[cond] = []
        st = collections.Counter()
        faces = collections.Counter()
        for ci in ids:
            base, cid = all_ctx[ci], f'test_{ci:03d}'
            ctx, applied = shift(spec, walls, base, d, hand0)
            # 평가에 들어가는 float32 배치 그대로 검사한다. 일부 배치는 1e-8 m 차이로 안착 여부가 바뀐다
            # (test_086 빨강: 블록을 z=0에 놓는 D3IL set_context의 침투 해소로 추정).
            ctx = to_pkl_context(ctx, base)
            ctx, applied, moved = settled(spec, env, base, ctx, applied)
            for e in applied:
                e.update(color=COLORS[e['box']], shift_m=d if e['face'] else 0.0, settle_drift_m=moved[e['box']])
                faces[(COLORS[e['box']], e['face'])] += 1
            n_moved = sum(e['face'] is not None for e in applied)
            st['contexts'] += 1
            st[f'blocks_moved_{n_moved}'] += 1
            st['reverted_unstable'] += sum(bool(e.get('reverted_unstable')) for e in applied)
            st['unsettled_after'] += any(m >= BLOCK_SETTLE_TOL for m in moved)
            pkls[cond].append(ctx)
            seeds = context_seeds(TASK, cid)
            rows.append({'id': f'{TASK}/{cond}/{cid}', 'task': TASK, 'cond': cond,
                         'factors': {'A': {'shift_m': d, 'level': name}, 'B': None, 'C': None, 'D': None, 'E': None},
                         'context_id': cid, 'context_index': ci, 'context': context_json(ctx),
                         'shift': applied, 'robot': None, 'robot_start': None, 'noise': None, 'push': None,
                         'delay_steps': 0, 'policy_seeds': seeds['policy']})
        stats[cond] = {'shift_m': d, 'rule': 'shift_blocks_far, movable blocks only' if d > FAR_FROM else 'shift_blocks',
                       **st, 'faces': {f'{c}:{f}': n for (c, f), n in sorted(faces.items(), key=str)}}
        print(f'{cond}: {dict(st)}', flush=True)

    # 최대 이동거리 (A_strong 규칙)
    per_context = []
    for ci in ids:
        best, faces, feasible, all_three = max_shift(spec, walls, all_ctx[ci], hand0)
        per_context.append({'context_id': f'test_{ci:03d}', 'max_shift_m': dict(zip(COLORS, best)),
                            'feasible_cm': dict(zip(COLORS, feasible)),
                            'face_at_max': dict(zip(COLORS, faces)), 'all_three_m': all_three})
    summary = {}
    for c in COLORS + ('all_three',):
        v = np.array([r['all_three_m'] if c == 'all_three' else r['max_shift_m'][c] for r in per_context])
        summary[c] = {'min': float(v.min()), 'median': float(np.median(v)), 'max': float(v.max()),
                      'n_ge_18cm': int((v >= 0.18 - 1e-9).sum()), 'n_zero': int((v == 0).sum()),
                      'hist_cm': {str(int(round(x * 100))): int(n) for x, n in sorted(collections.Counter(v.round(2)).items())}}
    (out / 'max_shift.json').write_text(json.dumps(
        {'rule': 'shift_blocks_far (reach ring, 5 cm from other start ranges, table, target wall), all blocks shifted '
                 'with the same d; 1 cm grid up to %.2f m; geometry only (no settle check)' % MAX_SCAN_M,
         'summary': summary, 'contexts': per_context}, indent=1) + '\n')
    print('max shift:', json.dumps({c: {k: s[k] for k in ('min', 'median', 'max', 'n_ge_18cm', 'n_zero')}
                                    for c, s in summary.items()}), flush=True)

    with (out / f'{TASK}.jsonl').open('x') as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + '\n')
    files = {f'{TASK}.jsonl': file_sha256(out / f'{TASK}.jsonl'), 'max_shift.json': file_sha256(out / 'max_shift.json')}
    for cond, ctxs in pkls.items():
        p = out / f'{TASK}_{cond}_contexts.pkl'
        with p.open('xb') as handle:
            pickle.dump(ctxs, handle)
        files[p.name] = file_sha256(p)
    manifest = {'version': VERSION, 'created_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                'root_seed': ROOT_SEED, 'd3il_commit': commit, 'task': TASK,
                'base_contexts': {'file': str(CONTEXTS.relative_to(REPO)), 'sha256': file_sha256(CONTEXTS),
                                  'indices': [ids[0], ids[-1] + 1], 'ids': 'test_NNN = index in base file'},
                'conditions': list(pkls), 'levels_m': dict(zip(pkls, LEVELS_M)),
                'demo_hand_start': start.c.tolist(), 'settle': {'seconds': SETTLE_S, 'tol_m': BLOCK_SETTLE_TOL},
                'stats': stats, 'files': files,
                'pkl_order': 'same order as base_contexts.indices; element i = test_{start+i:03d}',
                'policy_seed_rule': 'seed_for(root_seed, task, context_id, "policy", j) (runner may override)'}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print('saved', out)


if __name__ == '__main__':
    main()
