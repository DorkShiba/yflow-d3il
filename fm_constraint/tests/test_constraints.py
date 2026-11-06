"""행렬식과 독립 물리 계산, 해석해, 별도 수치 solver를 비교한다."""

import numpy as np
import pytest
from scipy.optimize import minimize, LinearConstraint

from configs.config import ExperimentConfig, REPO
from fm_constraint.constraints import PlanConstraints
from fm_constraint.qp import solve_qp


@pytest.fixture
def config():
    return ExperimentConfig.from_yaml(REPO / 'configs/pushing.yaml').values


def observation(blocks=((.4, 0.), (.6, 0.)), q=(.5, 0.)):
    return np.r_[q, q, np.c_[blocks, np.zeros((len(blocks), 2))].ravel()]


@pytest.mark.parametrize('n', [3, 4, 7, 8, 32])
def test_envelope_contains_all_blocks_including_degenerate_positions(config, n):
    config['constraints']['object_envelope']['n_sides'] = n
    for blocks in (((.4, 0.), (.6, 0.)), ((.5, 0.), (.5, 0.)),
                   ((.3, 0.), (.5, 0.), (.7, 0.))):
        c = PlanConstraints(observation(blocks), [.001, -.002], 3, config['constraints'])
        projections = (c.blocks - c.center) @ c.normals.T
        assert projections.max() <= c.radius - .1 + 1e-14
        angles = np.arange(n) * 2 * np.pi / n + np.pi / n
        vertices = c.center + c.radius / np.cos(np.pi / n) * np.c_[np.cos(angles), np.sin(angles)]
        np.testing.assert_allclose(((vertices - c.center) @ c.normals.T).max(axis=1), c.radius)


def test_compiled_rows_match_independent_constraints_and_units(config):
    rng = np.random.default_rng(42)
    previous = np.array([.004, -.002])
    c = PlanConstraints(observation(), previous, 5, config['constraints'])
    a = rng.normal(0., .03, (5, 2))
    q = c.q0 + np.cumsum(a, axis=0)
    direct_ws = np.c_[q[:, 0]-c.high[0], c.low[0]-q[:, 0], q[:, 1]-c.high[1], c.low[1]-q[:, 1]].ravel()
    np.testing.assert_allclose((c.G @ a.ravel() - c.d)[c.names == 'workspace'], direct_ws)
    changes = np.diff(a, axis=0, prepend=previous[None])
    expected = changes @ c.acc_normals.T - c.acc_limit * c.dt**2 * np.cos(np.pi / len(c.acc_normals))
    np.testing.assert_allclose((c.G @ a.ravel() - c.d)[c.names == 'acceleration'], expected.ravel())
    v = c.violation(a)
    assert v['acceleration_max_m_s2'] == pytest.approx(np.linalg.norm(changes, axis=1).max() / .035**2)


@pytest.mark.parametrize('m', [4, 8, 32])
def test_acceleration_inner_vertices_obey_exact_l2(config, m):
    config['constraints']['acceleration']['polygon_sides'] = m
    c = PlanConstraints(observation(), [0, 0], 1, config['constraints'])
    radius = c.acc_limit * c.dt**2
    angles = np.arange(m) * 2 * np.pi / m + np.pi / m
    for a in radius * np.c_[np.cos(angles), np.sin(angles)]:
        assert c.violation(a[None])['acceleration_l2_excess_m_s2'] < 1e-10
        assert c.violation(a[None])['acceleration_qp_max_m'] < 1e-12


def test_qp_halfspace_and_infeasible_are_not_clipped(config):
    solved = solve_qp([2., 3.], np.array([[1., 1.]]), [1.], config['solver'])
    assert solved['status'] == 'ok'
    np.testing.assert_allclose(solved['value'], [0., 1.], atol=1e-7)
    failed = solve_qp([1.], np.array([[1.], [-1.]]), [0., -1.], config['solver'])
    assert failed['status'] == 'infeasible'
    assert failed['value'] is None


def test_joint_projection_matches_independent_slsqp(config):
    c = PlanConstraints(observation(), [.003, -.001], 4, config['constraints'])
    nominal = np.random.default_rng(5).normal(0., .15, 8)
    actual = solve_qp(nominal, c.G, c.d, config['solver'])
    reference = minimize(lambda a: .5 * np.sum((a - nominal)**2), np.zeros(8),
                         jac=lambda a: a - nominal, method='SLSQP',
                         constraints=[LinearConstraint(c.G, -np.inf, c.d)],
                         options={'ftol': 1e-12, 'maxiter': 1000})
    assert actual['status'] == 'ok' and reference.success
    np.testing.assert_allclose(actual['value'], reference.x, atol=2e-7)
    again = solve_qp(actual['value'], c.G, c.d, config['solver'])
    np.testing.assert_allclose(again['value'], actual['value'], atol=1e-8)


def test_training_calibration_never_differences_across_episodes(tmp_path):
    from fm_constraint.check_constraints import calibrate_acceleration, calibrate_envelope
    train = tmp_path / 'train'
    train.mkdir()
    for index, actions in enumerate(([[.001, 0.], [.002, 0.]], [[-.001, 0.], [-.001, 0.]])):
        np.savez(train / f'{index}.npz', actions=np.array(actions),
                 obs=np.stack([observation(), observation()]))
    r = calibrate_acceleration(tmp_path, .1, 4)
    assert r['samples'] == 4
    assert r['l2_quantiles_m_s2']['max'] == pytest.approx(.1)
    assert r['training_polygon_retained_fraction'] == 1.
    envelope = calibrate_envelope(tmp_path, 8)
    assert envelope['selected_margin_m'] > 0
    assert envelope['training_initial_positions_retained_fraction'] == 1.


def test_invalid_and_unconverged_qp_never_returns_executable_value(config):
    invalid = solve_qp([np.nan], [[1.]], [1.], config['solver'])
    assert invalid['status'] == 'invalid_input' and invalid['value'] is None
    c = PlanConstraints(observation(), [0., 0.], 4, config['constraints'])
    settings = {**config['solver'], 'max_iter': 1}
    unfinished = solve_qp(np.full(8, 10.), c.G, c.d, settings)
    assert unfinished['status'] == 'solver_failure' and unfinished['value'] is None
