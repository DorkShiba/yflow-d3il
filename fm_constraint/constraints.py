"""목표 TCP 계획의 사각형·가속도·물체 포괄 정n각형 제약을 구성한다."""

import numpy as np


class PlanConstraints:
    """한 계획 동안 고정되는 G A <= d와 물리 단위 측정값을 보관한다.

    obs: (12,) 또는 (16,), [목표XY, 실제XY, 블록별 x/y/sin/cos].
    previous_action: 직전 실제 실행한 목표 증분 (2,), 단위 m.
    A는 (H,2)를 시간 우선 순서로 평탄화한다. 모든 QP 면 잔차는 m다.
    """

    def __init__(self, obs, previous_action, horizon, settings, dt=0.035):
        obs = np.asarray(obs, dtype=np.float64)
        previous = np.asarray(previous_action, dtype=np.float64)
        if (obs.shape not in ((12,), (16,)) or previous.shape != (2,)
                or not np.isfinite(obs).all() or not np.isfinite(previous).all()
                or type(horizon) is not int or horizon < 1 or not np.isfinite(dt) or dt <= 0):
            raise ValueError('Invalid observation, previous action, horizon or control dt')
        self.q0, self.p0 = obs[:2].copy(), obs[2:4].copy()
        self.blocks = obs[4:].reshape(-1, 4)[:, :2].copy()
        self.previous = previous.copy()
        self.horizon, self.settings, self.dt = horizon, settings, float(dt)
        self.low = self.high = self.normals = self.center = self.radius = None
        self.acc_normals = None
        self.acc_limit = settings['acceleration']['max_m_s2']
        eye = np.eye(2 * horizon).reshape(horizon, 2, 2 * horizon)
        cumulative = np.kron(np.tril(np.ones((horizon, horizon))), np.eye(2)).reshape(eye.shape)
        difference = eye.copy()
        difference[1:] -= eye[:-1]
        rows, bounds, scales, names, steps = [], [], [], [], []

        # 각 시점의 목표는 현재 목표 + 행동 누적이다. 미래 관측 채널의
        # 예측 위치를 사용하면 실행할 명령과 제약 대상이 달라지므로 쓰지 않는다.
        ws = settings['workspace']
        if ws['enabled']:
            self.low = np.asarray(ws['low_m'], dtype=float) + ws['margin_m']
            self.high = np.asarray(ws['high_m'], dtype=float) - ws['margin_m']
            normals = np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
            bound = np.array([self.high[0], -self.low[0], self.high[1], -self.low[1]])
            rows.append(np.einsum('ki,hij->hkj', normals, cumulative).reshape(-1, 2 * horizon))
            bounds.append(np.tile(bound - normals @ self.q0, horizon))
            scales.append(np.tile(np.repeat((self.high - self.low) / 2, 2), horizon))
            names.extend(['workspace'] * (4 * horizon))
            steps.extend(np.repeat(np.arange(horizon), 4))

        # 가속도 = 행동 차분 / 실제 제어 간격². 내부 원판의 정m각형을
        # 사용하므로 cos(pi/m)이 필요하다. 첫 차분의 이전 행동은 관측 상수다.
        acc = settings['acceleration']
        if acc['enabled']:
            m = acc['polygon_sides']
            angles = 2 * np.pi * np.arange(m) / m
            self.acc_normals = np.column_stack((np.cos(angles), np.sin(angles)))
            beta = acc['max_m_s2'] * self.dt**2 * np.cos(np.pi / m)
            bound = np.full((horizon, m), beta)
            bound[0] += self.acc_normals @ self.previous
            rows.append(np.einsum('ki,hij->hkj', self.acc_normals, difference).reshape(-1, 2 * horizon))
            bounds.append(bound.ravel())
            scales.append(np.full(horizon * m, beta))
            names.extend(['acceleration'] * (horizon * m))
            steps.extend(np.repeat(np.arange(horizon), m))

        # 성공한 블록도 포함한 모든 중심을 평균한다. 각 면을 공통 거리까지
        # 확장하여 정n각형을 만들고, margin은 바깥쪽 수직 거리로 더한다.
        # 이 기하는 PC 반복 중 갱신하지 않고 다음 재계획 때 다시 구성한다.
        obj = settings['object_envelope']
        if obj['enabled']:
            n = obj['n_sides']
            angles = obj['normal_offset_rad'] + 2 * np.pi * np.arange(n) / n
            self.normals = np.column_stack((np.cos(angles), np.sin(angles)))
            self.center = self.blocks.mean(axis=0)
            self.radius = float(np.max((self.blocks - self.center) @ self.normals.T) + obj['margin_m'])
            rows.append(np.einsum('ki,hij->hkj', self.normals, cumulative).reshape(-1, 2 * horizon))
            bounds.append(np.tile(self.radius + self.normals @ (self.center - self.q0), horizon))
            scales.append(np.full(horizon * n, self.radius))
            names.extend(['object_envelope'] * (horizon * n))
            steps.extend(np.repeat(np.arange(horizon), n))

        self.G = np.vstack(rows) if rows else np.empty((0, 2 * horizon))
        self.d = np.concatenate(bounds) if bounds else np.empty(0)
        self.scales = np.concatenate(scales) if scales else np.empty(0)
        self.names, self.steps = np.asarray(names), np.asarray(steps, dtype=int)
        if not np.isfinite(self.G).all() or not np.isfinite(self.d).all() or np.any(self.scales <= 0):
            raise ValueError('Constraint matrices must be finite with positive barrier scales')

    def violation(self, actions):
        """QP 잔차(m)와 정확한 L2 가속도(m/s²)를 측정하며 제약을 바꾸지 않는다."""
        actions = np.asarray(actions, dtype=np.float64)
        if actions.shape != (self.horizon, 2) or not np.isfinite(actions).all():
            raise ValueError('Expected finite actions of shape (H,2)')
        residual = np.maximum(self.G @ actions.ravel() - self.d, 0.)
        report = {'qp_max_m': float(residual.max(initial=0.))}
        for name in ('workspace', 'acceleration', 'object_envelope'):
            report[f'{name}_qp_max_m'] = float(residual[self.names == name].max(initial=0.))
        change = np.diff(actions, axis=0, prepend=self.previous[None])
        norm = np.linalg.norm(change, axis=1) / self.dt**2
        report['acceleration_max_m_s2'] = float(norm.max(initial=0.))
        report['acceleration_l2_excess_m_s2'] = (
            float(np.maximum(norm - self.acc_limit, 0).max(initial=0.))
            if self.settings['acceleration']['enabled'] else None)
        return report

    def geometry(self):
        """재생과 감사용 기하를 JSON으로 저장 가능한 형태로 반환한다."""
        return {'q0_m': self.q0.tolist(), 'p0_m': self.p0.tolist(),
                'previous_action_m': self.previous.tolist(), 'blocks_xy_m': self.blocks.tolist(),
                'dt_s': self.dt,
                'workspace_low_m': self.low.tolist() if self.low is not None else None,
                'workspace_high_m': self.high.tolist() if self.high is not None else None,
                'object_center_m': self.center.tolist() if self.center is not None else None,
                'object_normals': self.normals.tolist() if self.normals is not None else None,
                'object_inradius_m': self.radius}
