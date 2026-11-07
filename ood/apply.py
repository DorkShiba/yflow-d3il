"""OOD 사례(JSONL 한 행)를 생성·평가 시점에 적용한다.

배치 가능 여부는 기하로 판정한다. 블록(외접 반지름, 임의 yaw)은 다른 블록·손·
Inserting 미로 벽과 ``CLEAR`` 이상 떨어지고 TaskSpec.workspace 안에 있어야 한다.
구현 1(fm_pc/ood.py)의 shift·push 규칙을 그대로 옮겼고, 난수만 seed 규칙
(conditions.seed_for)으로 바꿨다.
"""
import copy
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np

from .conditions import gauss, seed_for

CLEAR = 0.01            # m: 물체 사이 여유 간격


# ---- context 직렬화 ---------------------------------------------------------------
def context_to_json(ctx):
    """D3IL 블록 태스크 context [pos, quat, ...] → JSON 리스트."""
    return [np.asarray(part, dtype=float).tolist() for part in ctx]


def context_from_json(value):
    return [np.asarray(part, dtype=float) for part in value]


# ---- 기하 -----------------------------------------------------------------------
def free(spec, walls, xy, others, hand_xy, old_xy=None, yaw=0.0, r=None):
    """xy에 반지름 r 블록을 둘 수 있는가. others: [(xy, radius)]."""
    (x0, x1), (y0, y1) = spec.workspace
    r = spec.block_half * math.sqrt(2.0) if r is None else r
    if not (x0 + r <= xy[0] <= x1 - r and y0 + r <= xy[1] <= y1 - r):
        return False
    if any(np.linalg.norm(xy - o) < r + ro + CLEAR for o, ro in others):
        return False
    if hand_xy is not None and not spec.hand_clear(xy, old_xy, yaw, np.asarray(hand_xy, dtype=float), r, CLEAR):
        return False
    return all(np.linalg.norm(np.maximum(np.abs(xy - c) - h, 0.0)) >= r + CLEAR for c, h in walls)


def shift_blocks(spec, walls, ctx, d, hand_xy):
    """A: 모든 블록을 시연 초기 범위 밖 d로, 가장 가까운 가능한 면부터 옮긴다(결정적).

    다른 블록의 시작 범위 쪽으로는 옮기지 않는다. 그건 '시연 밖'이 아니라 다른 ID 상황이다.
    반환: (새 context, [{box, face, from, to}]). 가능한 면이 없으면 face=None, 제자리.
    """
    radii = spec.block_radii()
    pos = [np.asarray(spec.ctx_box_xy(ctx, i), dtype=float).copy() for i in range(spec.n_boxes)]
    applied = []
    for i in range(spec.n_boxes):
        (x0, x1), (y0, y1) = spec.INIT_RANGE[i]
        p = pos[i]
        candidates = sorted([(p[0] - x0, np.array([x0 - d, p[1]]), 'x-'), (x1 - p[0], np.array([x1 + d, p[1]]), 'x+'),
                             (p[1] - y0, np.array([p[0], y0 - d]), 'y-'), (y1 - p[1], np.array([p[0], y1 + d]), 'y+')],
                            key=lambda c: c[0])
        others = [(pos[j], radii[j]) for j in range(spec.n_boxes) if j != i]
        other_ranges = [spec.INIT_RANGE[j] for j in range(spec.n_boxes)
                        if j != i and spec.INIT_RANGE[j] != spec.INIT_RANGE[i]]
        for _, q, face in candidates:
            near_other = any(math.hypot(max(rx[0] - q[0], 0.0, q[0] - rx[1]), max(ry[0] - q[1], 0.0, q[1] - ry[1])) < d
                             for rx, ry in other_ranges)
            if not near_other and free(spec, walls, q, others, hand_xy, r=radii[i]):
                pos[i] = q
                applied.append({'box': i, 'face': face, 'from': p.tolist(), 'to': q.tolist()})
                break
        else:
            applied.append({'box': i, 'face': None, 'from': p.tolist(), 'to': p.tolist()})
    for i in range(spec.n_boxes):
        ctx = spec.ctx_set_box_xy(ctx, i, pos[i])
    return ctx, applied


# ---- 데이터 파일 ------------------------------------------------------------------
def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_cases(root, task, condition):
    """data/ood/<task>.jsonl에서 한 조건의 사례를 context 순서대로 읽는다."""
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    entry = manifest['tasks'].get(task)
    if entry is None:
        raise ValueError(f'No OOD data for task {task!r} in {root}')
    if condition not in entry['conditions']:
        raise ValueError(f'Unknown OOD condition {condition!r}; available: {list(entry["conditions"])}')
    path = root / entry['file']
    digest = file_sha256(path)
    if digest != entry['sha256']:
        raise ValueError(f'OOD file checksum mismatch: {path}')
    with path.open() as handle:
        cases = [row for row in map(json.loads, handle) if row['cond'] == condition]
    return cases, {'file': str(path), 'sha256': digest, 'version': manifest['version'],
                   'condition': condition, 'factors': cases[0]['factors'] if cases else None}


# ---- 평가 중 적용 ------------------------------------------------------------------
class Episode:
    """rollout 하나의 OOD 상태. case=None이면 모든 훅이 no-op이다."""

    def __init__(self, case, rollout, spec=None, env=None, execute_steps=8):
        self.case, self.rollout, self.spec, self.env = case, rollout, spec, env
        self.execute_steps = execute_steps
        factors = (case or {}).get('factors') or {}
        self.noise = case.get('noise') if case else None
        self.push_spec = case.get('push') if case else None
        self.delay = int(case.get('delay_steps') or 0) if case else 0
        self.pushed = False
        self.events = {'push': None}
        self.factors = factors
        if self.delay and not 0 < self.delay < execute_steps:
            raise ValueError(f'delay_steps must be in 1..{execute_steps - 1}')

    @property
    def robot_start(self):
        return None if not self.case or self.case.get('robot_start') is None else np.asarray(self.case['robot_start'])

    def observe_boxes(self, boxes, step):
        """C: 정책·필터가 보는 블록 자세의 노이즈 사본. 실제 상태는 바꾸지 않는다."""
        if not self.noise:
            return boxes
        seed, sp, sy = self.noise['seed'], self.noise['pos_m'], math.radians(self.noise['yaw_deg'])
        out = []
        for i, (pos, quat, yaw) in enumerate(boxes):
            p = np.asarray(pos, dtype=float).copy()
            p[0] += gauss(seed, self.rollout, step, i, 'x', sigma=sp)
            p[1] += gauss(seed, self.rollout, step, i, 'y', sigma=sp)
            out.append((p, quat, float(yaw) + gauss(seed, self.rollout, step, i, 'yaw', sigma=sy)))
        return out

    def push_due(self, step, n_queued):
        if not self.push_spec or self.pushed or step < self.push_spec['step']:
            return False
        want = 0 if self.push_spec['phase'] == 'seam' else self.execute_steps // 2
        return n_queued == want

    def push(self, step, boxes, hand_xy, walls):
        """D: 손에 가장 가까운 블록부터, 가능한 방향을 찾을 때까지 d m 순간이동한다."""
        self.pushed = True
        spec, env = self.spec, self.env
        radii = spec.block_radii()
        xy = [np.asarray(b[0][:2], dtype=float) for b in boxes]
        rng = random.Random(seed_for(self.push_spec['seed'], self.rollout))
        angles = [rng.uniform(0.0, 2 * math.pi) for _ in range(64)]
        dist = self.push_spec['dist_m']
        record = {'step': step, 'box': None, 'from': None, 'to': None}
        for i in sorted(spec.pushable(boxes, hand_xy), key=lambda j: np.linalg.norm(xy[j] - hand_xy)):
            others = [(xy[j], radii[j]) for j in range(len(xy)) if j != i]
            record = {'step': step, 'box': i, 'from': xy[i].tolist(), 'to': None}
            for theta in angles:
                q = xy[i] + dist * np.array([math.cos(theta), math.sin(theta)])
                if free(spec, walls, q, others, hand_xy, xy[i], float(boxes[i][2]), r=radii[i]):
                    z = float(boxes[i][0][2])
                    env.scene.set_obj_pos_and_quat([q[0], q[1], z], np.asarray(boxes[i][1], dtype=float),
                                                   sim_obj=spec.box_objs(env)[i])
                    record['to'] = q.tolist()
                    break
            if record['to'] is not None:
                break
        self.events['push'] = record
        return record['to'] is not None

    def describe(self):
        return copy.deepcopy(self.events)
