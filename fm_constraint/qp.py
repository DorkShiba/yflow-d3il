"""OSQP 하나로 계획 투영과 CBF 벡터장 보정의 최소 수정 QP를 푼다."""

from time import perf_counter

import numpy as np
import osqp
from scipy import sparse


def solve_qp(nominal, G, d, settings):
    """min 0.5||z-nominal||², Gz<=d. 실패에서는 value=None을 반환한다.

    nominal은 (N,), G는 (M,N), d는 (M,). 단위는 호출 측에서 정한다.
    status와 물리 잔차를 모두 검사하며 미수렴 해를 실행에 사용하지 않는다.
    """
    started = perf_counter()
    nominal, G, d = (np.asarray(x, dtype=np.float64) for x in (nominal, G, d))
    report = {'value': None, 'status': 'invalid_input', 'solver_status': None,
              'iterations': 0, 'max_residual': None, 'intervention_norm': None, 'latency_ms': 0.}
    if (nominal.ndim != 1 or G.ndim != 2 or G.shape != (d.size, nominal.size)
            or d.ndim != 1 or not all(np.isfinite(x).all() for x in (nominal, G, d))):
        report['latency_ms'] = 1000 * (perf_counter() - started)
        return report
    # 이미 feasible한 입력은 정확한 최소 수정 해다. 불필요한 수치 변화와 반복을 피한다.
    if np.max(G @ nominal - d, initial=-np.inf) <= 0:
        report.update(value=nominal.copy(), status='ok', solver_status='already_feasible',
                      max_residual=0., intervention_norm=0.)
    else:
        solver = osqp.OSQP()
        try:
            solver.setup(P=sparse.eye(nominal.size, format='csc'), q=-nominal,
                         A=sparse.csc_matrix(G), l=np.full(d.size, -np.inf), u=d,
                         verbose=False, eps_abs=settings['eps_abs'], eps_rel=settings['eps_rel'],
                         eps_prim_inf=1e-10, eps_dual_inf=1e-10,
                         max_iter=settings['max_iter'], polishing=True)
            result = solver.solve(raise_error=False)
            report.update(solver_status=result.info.status, iterations=int(result.info.iter))
            if result.info.status_val in (3, 4):
                report['status'] = 'infeasible'
            elif result.info.status_val != 1 or result.x is None or not np.isfinite(result.x).all():
                report['status'] = 'solver_failure'
            else:
                residual = float(np.maximum(G @ result.x - d, 0).max(initial=0.))
                report['max_residual'] = residual
                if residual > settings['feasibility_tolerance_m']:
                    report['status'] = 'constraint_violation'
                else:
                    report.update(value=result.x.copy(), status='ok',
                                  intervention_norm=float(np.linalg.norm(result.x - nominal)))
        except (osqp.OSQPException, ValueError) as error:
            report.update(status='solver_failure', solver_status=str(error))
    report['latency_ms'] = 1000 * (perf_counter() - started)
    return report
