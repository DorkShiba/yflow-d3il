"""Execution-time VLSA command fence and numerical circle CBF-QP."""

from itertools import combinations
from time import perf_counter

import numpy as np


def workspace_limits(workspace, margin_m):
    """Validate and shrink an explicitly configured TCP-center rectangle."""
    keys = {'x_min_m', 'x_max_m', 'y_min_m', 'y_max_m'}
    if not isinstance(workspace, dict) or set(workspace) != keys:
        raise ValueError('workspace requires x_min_m, x_max_m, y_min_m, y_max_m')
    values = [workspace[k] for k in sorted(keys)] + [margin_m]
    if any(type(x) not in (int, float) or not np.isfinite(x) for x in values) or margin_m < 0:
        raise ValueError('Workspace coordinates and margin must be finite; margin >= 0')
    low = np.array([workspace['x_min_m'], workspace['y_min_m']], dtype=float) + margin_m
    high = np.array([workspace['x_max_m'], workspace['y_max_m']], dtype=float) - margin_m
    if np.any(low >= high):
        raise ValueError('Workspace must have positive width after applying margin')
    return low, high


def vlsa_constraint(nominal_delta, measured_xy, command_xy, config, *,
                    dt=0.035, tracking_gain=None, tolerance=1e-9):
    """Minimally modify one physical XY command increment.

    config is the YAML vlsa section. The caller controls its enabled flag.
    rectangle constrains command+delta exactly, without a tracking model.
    circle uses ideal p_dot=delta/dt or p_dot=k*(command+delta-measured).
    Actual hand/link safety is not established by the command fence.
    """
    started = perf_counter()
    margin = float(config['margin_m'])
    limit = float(config['command_delta_limit_m'])
    dt, tol = float(dt), float(tolerance)
    if not np.isfinite([margin, limit, dt, tol]).all() or margin < 0 or min(limit, dt, tol) <= 0:
        raise ValueError('Invalid physical VLSA parameters')
    nominal, p, q = (np.asarray(x, dtype=float) for x in (nominal_delta, measured_xy, command_xy))
    result = {'action': None, 'status': 'invalid_input', 'h': [],
              'residual': None, 'intervention_norm': None}
    if any(x.shape != (2,) or not np.isfinite(x).all() for x in (nominal, p, q)):
        result['latency_ms'] = 1000 * (perf_counter() - started)
        return result
    geometry = config.get('geometry', 'circle')
    if geometry == 'rectangle':
        if config.get('obstacles'):
            raise ValueError('Rectangle command fence does not include circular obstacles')
        low, high = workspace_limits(config['workspace'], margin)
        # h and constraint rows follow x_min, x_max, y_min, y_max.
        A = np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
        h = np.array([q[0] - low[0], high[0] - q[0], q[1] - low[1], high[1] - q[1]])
        b = -h
        result['constraint_names'] = ['workspace.x_min', 'workspace.x_max', 'workspace.y_min', 'workspace.y_max']
        result['measured_workspace_h'] = [p[0] - low[0], high[0] - p[0], p[1] - low[1], high[1] - p[1]]
    elif geometry == 'circle':
        centers = np.asarray([o['center_xy_m'] for o in config['obstacles']], dtype=float).reshape(-1, 2)
        radii = np.asarray([o['radius_m'] for o in config['obstacles']], dtype=float)
        gamma, pusher_radius = float(config['gamma']), float(config['pusher_radius_m'])
        gain = None if tracking_gain is None else float(tracking_gain)
        numbers = [gamma, pusher_radius] + ([] if gain is None else [gain])
        if (not np.isfinite(numbers).all() or min(numbers) <= 0
                or not np.isfinite(centers).all() or not np.isfinite(radii).all() or np.any(radii <= 0)):
            raise ValueError('Invalid physical circle CBF parameters')
        with np.errstate(over='ignore', invalid='ignore'):
            offset = p - centers
            h = np.sum(offset ** 2, axis=1) - (radii + pusher_radius + margin) ** 2
            factor = 1 / dt if gain is None else gain
            A = 2 * offset * factor
            b = -gamma * h
            if gain is not None:
                b = b - A @ (q - p)
    else:
        raise ValueError(f'Unsupported VLSA geometry: {geometry}')
    if not all(np.isfinite(x).all() for x in (h, A, b)):
        result['status'] = 'numerical_failure'
        result['latency_ms'] = 1000 * (perf_counter() - started)
        return result
    result['h'] = h.tolist()
    if np.any(h < (-tol if geometry == 'rectangle' else 0)):
        result['status'] = 'initial_unsafe'
    else:
        A = np.vstack([A, np.eye(2), -np.eye(2)])
        b = np.r_[b, [-limit] * 4]
        # The 2-D Euclidean projection has 0, 1, or 2 independent active constraints.
        candidates = [nominal]
        for a, rhs in zip(A, b):
            length2 = a @ a
            if length2 > 0:
                candidates.append(nominal + (rhs - a @ nominal) / length2 * a)
        for i, j in combinations(range(len(b)), 2):
            active = A[[i, j]]
            norms = np.linalg.norm(active, axis=1)
            if np.all(norms > 0):
                scaled = active / norms[:, None]
                if abs(np.linalg.det(scaled)) > 1e-12:
                    candidates.append(np.linalg.solve(scaled, b[[i, j]] / norms))
        feasible = [x for x in candidates if np.isfinite(x).all() and np.all(A @ x >= b - tol)]
        if feasible:
            action = min(feasible, key=lambda x: np.sum((x - nominal) ** 2))
            result.update(action=action, status='solved',
                          residual=float(max(0, np.max(b - A @ action))),
                          intervention_norm=float(np.linalg.norm(action - nominal)))
        else:
            result['status'] = 'infeasible'
    result['latency_ms'] = 1000 * (perf_counter() - started)
    return result
