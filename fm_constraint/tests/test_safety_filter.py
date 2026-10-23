import numpy as np
import pytest

from fm_constraint.vlsa import vlsa_constraint


def vlsa_config(**overrides):
    return {'enabled': True, 'obstacles': [{'center_xy_m': [0., 0.], 'radius_m': 0.1}],
            'gamma': 10, 'margin_m': 0.01, 'pusher_radius_m': 0.02,
            'command_delta_limit_m': 10, **overrides}


def test_random_halfspace_projection_matches_analytic_solution():
    rng = np.random.default_rng(42)
    config = vlsa_config()
    for _ in range(150):
        p = rng.normal(size=2)
        p = p / np.linalg.norm(p) * rng.uniform(.14, .4)
        nominal = rng.normal(size=2) * .04
        a = 2 * p / .035
        b = -10 * (p @ p - .13 ** 2)
        expected = nominal + max(0., b - a @ nominal) / (a @ a) * a
        result = vlsa_constraint(nominal, p, p, config)
        assert result['status'] == 'solved'
        np.testing.assert_allclose(result['action'], expected, atol=1e-10)
        assert result['residual'] <= 1e-9


def test_multiple_constraints_and_bounds_match_independent_optimizer():
    scipy = pytest.importorskip('scipy.optimize')
    obstacles = [{'center_xy_m': [0., 0.], 'radius_m': .1},
                 {'center_xy_m': [.3, 0.], 'radius_m': .1}]
    config = vlsa_config(obstacles=obstacles, command_delta_limit_m=.02)
    p, nominal = np.array([.15, .1]), np.array([.08, -.04])
    result = vlsa_constraint(nominal, p, p, config)
    centers = np.array([obstacle['center_xy_m'] for obstacle in obstacles])
    a = 2 * (p - centers) / .035
    b = -10 * (np.sum((p - centers) ** 2, axis=1) - .13 ** 2)
    reference = scipy.minimize(lambda x: .5 * np.sum((x - nominal) ** 2), [0., 0.],
                               jac=lambda x: x - nominal, method='SLSQP',
                               bounds=[(-.02, .02)] * 2,
                               constraints=[scipy.LinearConstraint(a, b, np.inf)],
                               options={'ftol': 1e-12})
    assert reference.success
    np.testing.assert_allclose(result['action'], reference.x, atol=1e-8)


def test_safe_action_is_unchanged_and_limits_are_enforced():
    config = vlsa_config()
    nominal = np.array([.001, 0.])
    np.testing.assert_array_equal(vlsa_constraint(nominal, [.2, 0], [.2, 0], config)['action'], nominal)
    bounded = vlsa_config(obstacles=[], margin_m=0, command_delta_limit_m=.01)
    np.testing.assert_allclose(vlsa_constraint([1., -1.], [0., 0.], [0., 0.], bounded)['action'], [.01, -.01])


def test_initial_unsafe_invalid_input_and_tracking_infeasibility():
    config = vlsa_config()
    assert vlsa_constraint([0., 0.], [0., 0.], [0., 0.], config)['status'] == 'initial_unsafe'
    assert vlsa_constraint([np.nan, 0.], [.2, 0.], [.2, 0.], config)['status'] == 'invalid_input'
    tracking = vlsa_config(obstacles=[{'center_xy_m': [1., 0.], 'radius_m': .9}],
                           margin_m=0, pusher_radius_m=.1, command_delta_limit_m=.01)
    result = vlsa_constraint([0., 0.], [0., 0.], [1., 0.], tracking, tracking_gain=5)
    assert result['status'] == 'infeasible'
    assert result['action'] is None


def test_rectangle_qp_matches_independent_box_projection():
    config = {'geometry': 'rectangle', 'workspace': {
        'x_min_m': .2, 'x_max_m': .85, 'y_min_m': -.4, 'y_max_m': .5},
        'margin_m': .01, 'command_delta_limit_m': .01, 'obstacles': []}
    rng = np.random.default_rng(42)
    low, high = np.array([.21, -.39]), np.array([.84, .49])
    points = [low, high, [low[0], high[1]], [high[0], low[1]],
              *rng.uniform(low, high, size=(150, 2))]
    for command in points:
        command = np.asarray(command)
        nominal = rng.normal(size=2) * .1
        expected = np.clip(nominal, np.maximum(-.01, low - command), np.minimum(.01, high - command))
        result = vlsa_constraint(nominal, command, command, config)
        assert result['status'] == 'solved'
        np.testing.assert_allclose(result['action'], expected, atol=1e-10)
        assert np.all(command + result['action'] >= low - 1e-9)
        assert np.all(command + result['action'] <= high + 1e-9)
        assert result['residual'] <= 1e-9


def test_rectangle_does_not_silently_recover_an_outside_command():
    config = {'geometry': 'rectangle', 'workspace': {
        'x_min_m': 0, 'x_max_m': 1, 'y_min_m': 0, 'y_max_m': 1},
        'margin_m': .01, 'command_delta_limit_m': .01, 'obstacles': []}
    result = vlsa_constraint([.01, 0], [.5, .5], [-.1, .5], config)
    assert result['status'] == 'initial_unsafe'
    assert result['action'] is None


def test_rectangle_boundary_allows_tangential_motion_and_blocks_outward_motion():
    config = {'geometry': 'rectangle', 'workspace': {
        'x_min_m': 0, 'x_max_m': 1, 'y_min_m': 0, 'y_max_m': 1},
        'margin_m': .01, 'command_delta_limit_m': .01, 'obstacles': []}
    result = vlsa_constraint([-.005, .004], [.01, .5], [.01, .5], config)
    np.testing.assert_allclose(result['action'], [0, .004], atol=1e-12)


@pytest.mark.parametrize('nominal', [[0, 0], [.005, 0], [.012, 0], [.02, .02], [1000, -2000]])
def test_action_norm_only_matches_radial_projection(nominal):
    config = {'geometry': 'rectangle', 'workspace_enabled': False, 'action_norm_enabled': True,
              'max_step_displacement_m': .012, 'margin_m': 0.,
              'command_delta_limit_m': 10000., 'obstacles': []}
    a = np.array(nominal, dtype=float)
    norm = np.linalg.norm(a)
    expected = a * min(1, .012 / norm) if norm else a
    result = vlsa_constraint(a, [0, 0], [0, 0], config)
    assert result['status'] == 'solved'
    np.testing.assert_allclose(result['action'], expected, atol=1e-12)


def test_joint_box_disk_projection_matches_independent_optimizer():
    scipy = pytest.importorskip('scipy.optimize')
    config = {'geometry': 'rectangle', 'workspace_enabled': True, 'action_norm_enabled': True,
              'max_step_displacement_m': .012, 'margin_m': .01, 'command_delta_limit_m': .01,
              'obstacles': [], 'workspace': {'x_min_m': .2, 'x_max_m': .85, 'y_min_m': -.4, 'y_max_m': .5}}
    low, high, radius = np.array([.21, -.39]), np.array([.84, .49]), .012
    rng = np.random.default_rng(12)
    points = [low, high, [low[0], high[1]], [high[0], low[1]], *rng.uniform(low, high, size=(40, 2))]
    for q in points:
        q, nominal = np.asarray(q), rng.normal(size=2) * .04
        lower, upper = np.maximum(-.01, low - q), np.minimum(.01, high - q)
        # Scale variables so the independent optimizer works in dimensionless units.
        target = nominal / radius
        ref = scipy.minimize(lambda y: .5 * np.sum((y - target) ** 2),
                             .1 * (lower + upper) / radius,
                             jac=lambda y: y - target, hess=lambda y: np.eye(2),
                             method='trust-constr', bounds=scipy.Bounds(lower / radius, upper / radius),
                             constraints=[scipy.NonlinearConstraint(
                                 lambda y: y @ y, -np.inf, 1., jac=lambda y: 2 * y,
                                 hess=lambda y, v: 2 * v[0] * np.eye(2))],
                             options={'gtol': 1e-13, 'xtol': 1e-13, 'barrier_tol': 1e-13,
                                      'initial_barrier_parameter': 1e-10, 'maxiter': 1000})
        assert ref.success, ref.message
        result = vlsa_constraint(nominal, q, q, config)
        assert result['status'] == 'solved'
        np.testing.assert_allclose(result['action'], ref.x * radius, atol=1e-7)
        assert np.linalg.norm(result['action']) <= radius + 1e-12
        assert result['residual'] <= 1e-9


def test_joint_projection_is_better_than_sequential_clipping():
    config = {'geometry': 'rectangle', 'workspace_enabled': True, 'action_norm_enabled': True,
              'max_step_displacement_m': .008, 'margin_m': 0., 'command_delta_limit_m': .01,
              'obstacles': [], 'workspace': {'x_min_m': 0., 'x_max_m': 1., 'y_min_m': 0., 'y_max_m': 1.}}
    nominal = np.array([.02, .02])
    result = vlsa_constraint(nominal, [.997, .5], [.997, .5], config)
    sequential = np.array([.003, .01])
    sequential *= .008 / np.linalg.norm(sequential)
    assert np.linalg.norm(result['action'] - nominal) < np.linalg.norm(sequential - nominal)
    np.testing.assert_allclose(result['action'], [.003, np.sqrt(.008 ** 2 - .003 ** 2)], atol=1e-12)
