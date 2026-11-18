"""벽 OOD(2026-11-18): 학습 때 없던 정적 벽과 ①·③의 벽 회피 부등식.

모델 관측에는 벽이 없다. 벽은 장면에만 추가하고, ① Projection과 ③ Filter의 제약 집합 C에
손(목표 TCP) 기준 반평면을 더한다. ②는 QP가 없어 해당 없음, ⑤는 ①이 켜졌을 때 ①의 C를 그대로 쓴다.

제약 (손 기준, 블록 크기는 넣지 않는다):
    금지 영역 = 벽 AABB를 rod 반지름 + margin만큼 키운 직사각형 B.
    시점 h의 목표 위치 q_h = q₀ + Σ_{i≤h} a_i 마다 B의 면 f_h 하나를 골라 반평면 n_fᵀ(q_h − c) ≥ e_f 를 건다.
    면 선택(시점별):
        q_h가 B 밖(부호 있는 거리 max ≥ −tol)이면 f_h = 부호 있는 거리가 가장 큰 면,
        q_h가 B 안이면 f_h = f_{h−1} (h = 0의 이전은 q₀의 면). 관통(tunneling)을 막는다.
    ①: 매 Euler 직후 그 시점의 nominal 계획으로 면을 다시 고른 뒤 투영한다.
       ⑤와 최종 판정은 마지막 Euler에서 고른 면을 쓴다.
    ③: 실행 직전 nominal 행동 하나(q₀ + a)로 같은 규칙을 적용한다.
    위반 측정: 같은 규칙으로 고른 면에 대한 침투 깊이(m). B 밖의 점은 어떤 규칙이든 0이다.

기존 실험 코드(CODE_FILES)는 바꾸지 않는다. wall_experiment.install()이 이 모듈의 클래스로
evaluation·policy 모듈의 이름을 바꿔 끼운다.
"""

import weakref

import numpy as np

from .constraints import PlanConstraints
from .policy import Policy
from .qp import solve_qp

# 벽 기하 (중심 xy, 반크기 xy, m). 2026-11-18 결정: Pushing 세로 2×10 cm, Inserting gate 안 2×2 cm 기둥.
WALLS = {
    'pushing': {'center_m': (0.525, 0.15), 'half_m': (0.01, 0.05)},
    'inserting': {'center_m': (0.525, 0.17), 'half_m': (0.01, 0.01)},
}
WALL_Z_M = 0.0            # 중심 높이. Inserting maze 벽과 같은 기준
WALL_HALF_Z_M = 0.04      # 반높이(바닥 −0.04, 위 0.04). 테이블에 묻혀도 정적 물체끼리는 접촉 계산이 없다
ROD_RADIUS_M = 0.01       # panda_rod rod:geom 반지름
MARGIN_M = 0.012          # 한 step 최대 이동(≈1.18 cm) 이상. smoke에서 실측 손–벽 접촉을 보고 조정한다
INSIDE_TOL_M = 1e-7       # solver feasibility tolerance와 같다

NORMALS = np.array([[-1., 0.], [1., 0.], [0., -1.], [0., 1.]])   # 왼쪽, 오른쪽, 아래, 위 (바깥 방향)
FACE_NAMES = ('x-', 'x+', 'y-', 'y+')

_STATE = {'task': None, 'margin': MARGIN_M}


def configure(task, margin=None):
    if task not in WALLS:
        raise ValueError(f'No wall defined for {task!r}')
    _STATE.update(task=task, margin=MARGIN_M if margin is None else float(margin))


def spec(task=None):
    """현재 설정의 벽과 키운 금지 영역(JSON 가능)."""
    task = task or _STATE['task']
    wall = WALLS[task]
    inflate = ROD_RADIUS_M + _STATE['margin']
    return {'task': task, 'center_m': list(wall['center_m']), 'half_m': list(wall['half_m']),
            'z_m': WALL_Z_M, 'half_z_m': WALL_HALF_Z_M, 'rod_radius_m': ROD_RADIUS_M,
            'margin_m': _STATE['margin'], 'inflated_half_m': [h + inflate for h in wall['half_m']],
            'constraint': 'hand_only', 'face_rule': 'per_step_max_signed_distance_inside_keeps_previous',
            'applies_to': ['projection', 'filter', 'lipo_via_projection']}


def _geometry():
    s = spec()
    center = np.asarray(s['center_m'], dtype=np.float64)
    half = np.asarray(s['inflated_half_m'], dtype=np.float64)
    return center, np.array([half[0], half[0], half[1], half[1]])


def signed_distances(points, center, extent):
    """(N,2) → (N,4) 각 면의 부호 있는 거리. 양수면 그 면의 바깥쪽이다."""
    return (np.asarray(points, dtype=np.float64) - center) @ NORMALS.T - extent


def select_faces(q0, points, center, extent, tol=INSIDE_TOL_M):
    """시점별 면 index (N,). 안쪽 점은 직전 시점의 면을 유지한다."""
    prev = int(np.argmax(signed_distances(np.asarray(q0)[None], center, extent)[0]))
    faces = np.empty(len(points), dtype=int)
    for h, s in enumerate(signed_distances(points, center, extent)):
        if s.max() >= -tol:
            prev = int(np.argmax(s))
        faces[h] = prev
    return faces


def penetration(q0, actions, center, extent):
    """규칙으로 고른 면에 대한 최대 침투 깊이(m). 모든 점이 금지 영역 밖이면 0."""
    points = np.asarray(q0) + np.cumsum(np.asarray(actions, dtype=np.float64).reshape(-1, 2), axis=0)
    faces = select_faces(q0, points, center, extent)
    s = signed_distances(points, center, extent)[np.arange(len(points)), faces]
    return float(np.maximum(-s, 0.).max(initial=0.))


class WallPlanConstraints(PlanConstraints):
    """PlanConstraints + 벽 반평면 H개. select(actions)로 면을 다시 고르면 G, d가 바뀐다."""

    _by_matrix = weakref.WeakValueDictionary()   # id(G) → 인스턴스 (③ 필터가 G로 찾는다)

    def __init__(self, obs, previous_action, horizon, settings, dt=0.035):
        super().__init__(obs, previous_action, horizon, settings, dt)
        self.base_G, self.base_d = self.G, self.d
        self.base_scales, self.base_names, self.base_steps = self.scales, self.names, self.steps
        self.wall_center, self.wall_extent = _geometry()
        cumulative = np.kron(np.tril(np.ones((horizon, horizon))), np.eye(2)).reshape(horizon, 2, 2 * horizon)
        self._cumulative = cumulative
        # 계획 전 기본값: 모든 시점이 q₀에 머무는 계획 기준. ①·③은 사용 전에 다시 고른다.
        self.select(np.zeros((horizon, 2)))

    def select(self, actions):
        actions = np.asarray(actions, dtype=np.float64).reshape(self.horizon, 2)
        points = self.q0 + np.cumsum(actions, axis=0)
        faces = select_faces(self.q0, points, self.wall_center, self.wall_extent)
        normals = NORMALS[faces]
        rows = -np.einsum('hi,hij->hj', normals, self._cumulative)
        bounds = signed_distances(self.q0[None], self.wall_center, self.wall_extent)[0][faces]
        self.faces = faces
        self.G = np.vstack([self.base_G, rows])
        self.d = np.concatenate([self.base_d, bounds])
        self.scales = np.concatenate([self.base_scales, self.wall_extent[faces]])
        self.names = np.concatenate([self.base_names, np.full(self.horizon, 'obstacle')])
        self.steps = np.concatenate([self.base_steps, np.arange(self.horizon)])
        WallPlanConstraints._by_matrix[id(self.G)] = self
        return self

    def violation(self, actions):
        G, d, names = self.G, self.d, self.names
        self.G, self.d, self.names = self.base_G, self.base_d, self.base_names
        try:
            report = super().violation(actions)
        finally:
            self.G, self.d, self.names = G, d, names
        depth = penetration(self.q0, actions, self.wall_center, self.wall_extent)
        report['obstacle_qp_max_m'] = depth
        report['qp_max_m'] = max(report['qp_max_m'], depth)
        return report

    def geometry(self):
        out = super().geometry()
        out.update(wall_center_m=self.wall_center.tolist(),
                   wall_inflated_half_m=[float(self.wall_extent[0]), float(self.wall_extent[2])],
                   wall_faces=[FACE_NAMES[f] for f in self.faces])
        return out


class WallPolicy(Policy):
    """① 투영 직전에 현재 nominal 계획으로 벽 면을 다시 고른다(매 Euler)."""

    def _project(self, x, constraints, label):
        if isinstance(constraints, WallPlanConstraints):
            raw = (x * self.base.scale + self.base.center)[0, :, :2].detach().cpu().numpy()
            constraints.select(raw)
            faces = self.last_report.setdefault('wall_faces', [])
            faces.append(''.join('LRBT'[f] for f in constraints.faces))
        return super()._project(x, constraints, label)


def filter_solve_qp(nominal, G, d, settings):
    """③: evaluation이 만든 H=1 제약이면 nominal 행동으로 면을 다시 고른 뒤 푼다."""
    owner = WallPlanConstraints._by_matrix.get(id(G))
    if owner is not None and owner.G is G and owner.horizon == 1:
        owner.select(np.asarray(nominal, dtype=np.float64).reshape(1, 2))
        G, d = owner.G, owner.d
    return solve_qp(nominal, G, d, settings)


# ---- 장면 --------------------------------------------------------------------------
LAST_ENV = {'env': None}


def wall_env_class(Env):
    """env.start() 전에 정적 벽을 장면에 넣고, 시뮬레이션 매 substep의 벽 접촉을 기록하는 Env."""

    class WallEnv(Env):
        def start(self, *args, **kwargs):
            s = spec()
            box = type(self.push_box1)   # D3IL 물체와 같은 Box 클래스(로더 등록이 클래스 단위)
            self.ood_wall = box(name='ood_wall', init_pos=[*s['center_m'], s['z_m']], init_quat=[0, 1, 0, 0],
                                size=[*s['half_m'], s['half_z_m']], rgba=[0.2, 0.2, 0.2, 1.0], static=True)
            self.scene.add_object(self.ood_wall)
            out = super().start(*args, **kwargs)
            self._wall_setup_contacts()
            self.wall_log = []
            LAST_ENV['env'] = self
            return out

        def _wall_setup_contacts(self):
            import mujoco
            model = self.scene.model
            names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or '' for i in range(model.ngeom)]
            self._wall_geoms = {i for i, n in enumerate(names) if n.startswith('ood_wall')}
            if not self._wall_geoms:
                raise RuntimeError('ood_wall geom not found after scene start')
            boxes = [getattr(self, a).name for a in ('push_box1', 'push_box2', 'push_box3') if hasattr(self, a)]
            self._box_geoms = {i for i, n in enumerate(names) if any(n.startswith(b + ':') or n == b for b in boxes)}
            body_of = model.geom_bodyid
            robot = {i for i, n in enumerate(names) if n.startswith('rod') or n.startswith('panda')}
            # 이름이 없는 robot geom도 rod/panda body 아래면 robot으로 본다.
            body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(body_of[i])) or '' for i in range(model.ngeom)]
            robot |= {i for i, n in enumerate(body_names) if n.startswith('rod') or n.startswith('panda')}
            self._robot_geoms = robot - self._wall_geoms
            self._wall_acc = {'hand': False, 'block': False}
            scene_step = self.scene.next_step

            def next_step(*args, **kwargs):
                out = scene_step(*args, **kwargs)
                data = self.scene.data
                for k in range(data.ncon):
                    g1, g2 = data.contact[k].geom1, data.contact[k].geom2
                    if g1 in self._wall_geoms or g2 in self._wall_geoms:
                        other = g2 if g1 in self._wall_geoms else g1
                        if other in self._robot_geoms:
                            self._wall_acc['hand'] = True
                        elif other in self._box_geoms:
                            self._wall_acc['block'] = True
                return out

            self.scene.next_step = next_step
            self.wall_geom_names = {'wall': sorted(names[i] for i in self._wall_geoms),
                                    'boxes': sorted(names[i] for i in self._box_geoms),
                                    'robot_count': len(self._robot_geoms)}

        def reset(self, *args, **kwargs):
            self.wall_log.append([])
            out = super().reset(*args, **kwargs)
            self.wall_log[-1] = []     # reset 안에서 step을 부르더라도 에피소드 기록에서 뺀다
            self._wall_acc = {'hand': False, 'block': False}
            return out

        def step(self, *args, **kwargs):
            self._wall_acc = {'hand': False, 'block': False}
            out = super().step(*args, **kwargs)
            if self.wall_log:
                self.wall_log[-1].append((self._wall_acc['hand'], self._wall_acc['block']))
            return out

    WallEnv.__name__ = f'Wall{Env.__name__}'
    return WallEnv


# ---- 에피소드 지표 -------------------------------------------------------------------
STALL_WINDOW = 40          # step (1.4 s)
STALL_EXTENT_M = 0.01      # 창 안에서 손 실측 위치 범위가 이보다 작으면 정지
STALL_NEAR_M = 0.03        # 그 동안 손이 (키우지 않은) 벽에서 이 거리 안


def episode_metrics(rows, contacts, tolerance):
    """rollout 기록 + 접촉 기록 → 벽 지표."""
    s = spec()
    center, half = np.asarray(s['center_m']), np.asarray(s['half_m'])
    done = [r for r in rows if r.get('safe_action') is not None]
    hand = np.asarray([r['next_measured_state'][:2] for r in done], dtype=float).reshape(-1, 2)
    gap = np.maximum(np.abs(hand - center) - half, 0.)
    distance = np.linalg.norm(gap, axis=1) - ROD_RADIUS_M     # 손(rod 표면)–벽 거리, 음수면 겹침
    stalled = False
    for k in range(0, max(len(hand) - STALL_WINDOW + 1, 0)):
        window = hand[k:k + STALL_WINDOW]
        if np.ptp(window, axis=0).max() < STALL_EXTENT_M and distance[k:k + STALL_WINDOW].min() < STALL_NEAR_M:
            stalled = True
            break
    command_in = [r['latest_executed_violation'].get('obstacle_qp_max_m', 0.) > tolerance for r in done]
    nominal_in = [r['latest_nominal_violation'].get('obstacle_qp_max_m', 0.) > tolerance for r in done]
    hand_contact = [c[0] for c in contacts]
    block_contact = [c[1] for c in contacts]
    return {'wall_hand_contact': bool(any(hand_contact)), 'wall_hand_contact_steps': int(sum(hand_contact)),
            'wall_block_contact': bool(any(block_contact)), 'wall_block_contact_steps': int(sum(block_contact)),
            'wall_command_violation': bool(any(command_in)), 'wall_command_violation_steps': int(sum(command_in)),
            'wall_nominal_violation_steps': int(sum(nominal_in)),
            'wall_min_hand_distance_m': float(distance.min()) if len(distance) else None,
            'wall_stalled': stalled, 'wall_margin_m': s['margin_m'],
            'wall_contact_steps_logged': len(contacts)}
