"""⑤ LiPo: 계획 끝에서 고정 보정 한도의 minimum-jerk QP (codex 브랜치의 mode="qp").

    min_z  || D³ [history ; q0 + L z] ||²
    s.t.   z_i = a_i                       (지연으로 실행이 확정된 앞 d개)
           | L (z − a) | ≤ bound          (나머지 시점·축의 위치 보정 ≤ bound_m)
           G z ≤ d                         (① Projection이 켜졌을 때만, 고정 행동만 걸린 면은 제외)

z, a는 (H,2) 행동을 시간 우선으로 펼친 벡터, L은 행동 누적(위치) 연산이다.
history는 실행된 목표 위치 최대 3개이며 마지막은 현재 목표 q0다.
①이 켜져 있으면 ① 투영 결과 a가 G a ≤ d를 만족하므로 z=a가 항상 해다
(지연으로 앞 d개를 바꾼 경우는 예외일 수 있어 그 면만 뺀다).

이 QP는 조건수가 나빠 OSQP(ADMM)가 반복 한도에 걸리거나 한도를 1e-9 m 수준으로
넘긴다. 그래서 master·codex의 LiPo와 같은 interior-point solver(Clarabel)로 mm 단위에서 푼다.
"""
from time import perf_counter

import numpy as np
from scipy import sparse

UNIT = 1e-3     # m → mm


def jerk_qp(actions, history, bound_m, settings, frozen=0, G=None, d=None):
    """반환: {'value': (H,2) 또는 None, 'status', ...}. 실패를 다른 값으로 대체하지 않는다."""
    import clarabel
    started = perf_counter()
    a = np.asarray(actions, dtype=np.float64)
    hist = np.asarray(history, dtype=np.float64).reshape(-1, 2)[-3:]
    H = len(a)
    report = {'value': None, 'status': 'invalid_input', 'solver_status': None, 'frozen_prefix': int(frozen),
              'constrained': G is not None, 'latency_ms': 0.}
    if a.shape != (H, 2) or not len(hist) or not np.isfinite(a).all() or not np.isfinite(hist).all() \
            or not 0 <= frozen <= H or bound_m <= 0:
        report['latency_ms'] = 1000 * (perf_counter() - started)
        return report
    n, h, f = 2 * H, len(hist), 2 * frozen
    L = np.kron(np.tril(np.ones((H, H))), np.eye(2))                     # 위치 = q0 + L z
    D3 = np.diff(np.eye(h + H), n=3, axis=0)
    Dh, Dp = np.kron(D3[:, :h], np.eye(2)), np.kron(D3[:, h:], np.eye(2))
    q0 = hist[-1]
    # 잔차 r = M z + c (mm). 3차 차분은 상수 이동에 불변이므로 q0 기준으로 둔다.
    M = Dp @ L / UNIT
    c = (Dh @ (hist - q0).ravel()) / UNIT
    P = sparse.triu(sparse.csc_matrix(2 * M.T @ M), format='csc')
    q = 2 * M.T @ c
    av = a.ravel()
    box = np.full(n - f, bound_m)
    # 등식(ZeroCone): 고정 행동. 부등식(Nonnegative): ±L(z−a) ≤ box, G z ≤ d.
    eq = np.eye(n)[:f]
    Lf = L[f:]
    ineq = [Lf, -Lf]
    rhs = [box + Lf @ av, box - Lf @ av]
    keep = None
    if G is not None:
        keep = np.ones(len(d), dtype=bool) if not frozen else _rows_after(G, frozen)
        # 위치 보정 ≤ bound이면 행동 변화 |z_i − a_i| ≤ 2·bound다. 이 변화로 닿을 수 없는 면
        # (여유 > ||G_row||₁·2·bound)은 해를 바꾸지 않으므로 뺀다. 모든 면을 넣으면 Clarabel이
        # InsufficientProgress로 멈추는 경우가 있었다(Inserting, 704면, 여유 1.7 cm).
        reach = np.abs(G).sum(axis=1) * 2 * bound_m
        keep &= d - G @ av <= reach + 1e-12
        ineq.append(G[keep])
        rhs.append(d[keep])
        report['constraint_rows'] = int(keep.sum())
    A = sparse.csc_matrix(np.vstack([eq, *ineq]) / UNIT)
    b = np.concatenate([av[:f], *rhs]) / UNIT
    cones = ([clarabel.ZeroConeT(f)] if f else []) + [clarabel.NonnegativeConeT(len(b) - f)]
    options = clarabel.DefaultSettings()
    options.verbose = False
    options.tol_gap_abs = options.tol_gap_rel = options.tol_feas = 1e-9
    try:
        solution = clarabel.DefaultSolver(P, q, A, b, cones, options).solve()
    except Exception as error:      # noqa: BLE001 - solver 내부 예외를 상태로 기록한다
        report.update(status='solver_failure', solver_status=str(error))
        report['latency_ms'] = 1000 * (perf_counter() - started)
        return report
    status = str(solution.status)
    report['solver_status'] = status
    x = np.asarray(solution.x, dtype=np.float64)
    if 'Infeasible' in status:
        report['status'] = 'lipo_infeasible'
    # AlmostSolved도 아래에서 보정 한도·① 잔차를 직접 검사한 뒤에만 받아들인다.
    elif status not in ('Solved', 'AlmostSolved') or not np.isfinite(x).all():
        report['status'] = 'solver_failure'
    else:
        # 허용 오차 이하의 위치 보정 초과를 한도로 자르고, 고정 구간은 정확히 원래 행동으로 둔다.
        correction = (L @ (x - av)).reshape(H, 2)
        limit = np.r_[np.zeros(frozen), np.full(H - frozen, bound_m)][:, None]
        if np.any(np.abs(correction) > limit + 1e-8):
            report['status'] = 'lipo_bound_violation'
        else:
            correction = np.clip(correction, -limit, limit)
            z = av + np.diff(np.vstack([np.zeros(2), correction]), axis=0).ravel()
            # 판정은 뺀 면까지 포함한 전체(고정 행동만 걸린 면 제외)로 한다.
            rows = (np.ones(len(d), dtype=bool) if not frozen else _rows_after(G, frozen)) if G is not None else None
            residual = float(np.maximum(G[rows] @ z - d[rows], 0).max(initial=0.)) if G is not None else 0.
            report['max_residual'] = residual
            if residual > settings['feasibility_tolerance_m']:
                report['status'] = 'constraint_violation'
            else:
                before = float(np.sum((M @ av + c) ** 2)) * UNIT ** 2
                after = float(np.sum((M @ z + c) ** 2)) * UNIT ** 2
                report.update(value=z.reshape(H, 2), status='ok', jerk_cost_before_m2=before,
                              jerk_cost_after_m2=after, correction_max_m=float(np.abs(correction).max(initial=0.)))
    report['latency_ms'] = 1000 * (perf_counter() - started)
    return report


def _rows_after(G, frozen):
    """고정 행동만 포함하는 면(자유 변수 계수가 모두 0)은 제외한다."""
    return np.abs(G[:, 2 * frozen:]).sum(axis=1) > 0
