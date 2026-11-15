"""Numerical and data-boundary regressions for the joint comparison publication."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from configs.config import ExperimentConfig, REPO
from fm_constraint.joint import (
    JointConstraints,
    JointPolicy,
    JointPotential,
    joint_lipo,
    solve_qp,
)
from fm_constraint.comparison_io import check_output_signature
from fm_constraint.calibrate_sorting import calibrate
from fm_constraint.recompute_comparison import paired_stats


@pytest.fixture
def settings():
    return json.loads((REPO / "configs/stacking_combo29.json").read_text())


def test_projection_fallback_same_problem_and_strict_residual(settings):
    q = (np.array(settings["joint_low_rad"]) + settings["joint_high_rad"]) / 2
    con = JointConstraints(q, np.zeros(7), 16, settings)
    nominal = np.random.default_rng(42).normal(size=112) * 0.02
    regular = solve_qp(nominal, con.G, con.d, settings["solver"])
    forced = solve_qp(nominal, con.G, con.d, dict(settings["solver"], max_iter=1))
    assert regular["status"] == forced["status"] == "ok"
    assert forced["primary_solver"]["status"] == "solver_failure"
    assert forced["max_residual"] <= 1e-8
    np.testing.assert_allclose(forced["value"], regular["value"], atol=2e-7, rtol=0)


def test_lipo_cumulative_bound_and_jerk(settings):
    q = (np.array(settings["joint_low_rad"]) + settings["joint_high_rad"]) / 2
    con = JointConstraints(q, np.zeros(7), 16, settings)
    nominal = np.random.default_rng(42).normal(size=112) * 0.02
    projected = solve_qp(nominal, con.G, con.d, settings["solver"])["value"].reshape(
        16, 7
    )
    result = joint_lipo(projected, [q], 0.0003, settings["solver"], con.G, con.d)
    assert result["status"] == "ok"
    assert con.residual(result["value"]) <= 1e-8
    assert (
        np.abs(np.cumsum(result["value"] - projected, axis=0)).max() <= 0.0003 + 1e-12
    )
    assert result["jerk_cost_after_rad2"] <= result["jerk_cost_before_rad2"] + 1e-12


def fake_base():
    def condition(x, cond, action_dim):
        x = x.clone()
        x[:, 0, action_dim:] = cond[0]
        return x

    return SimpleNamespace(
        center=torch.linspace(-0.01, 0.01, 31),
        scale=torch.linspace(0.07, 0.13, 31),
        device="cpu",
        apply_conditioning=condition,
        model=lambda x, t: -0.2 * x,
    )


def test_policy_keeps_double_solutions_exact_anchor_and_gripper(settings):
    q = (np.array(settings["joint_low_rad"]) + settings["joint_high_rad"]) / 2
    q += np.arange(7) * 1e-9
    obs = np.r_[q, 0.08, np.zeros(15)].astype(np.float32)
    base = fake_base()
    noise = torch.randn(1, 16, 31, generator=torch.Generator().manual_seed(1729))
    potential = SimpleNamespace(guidance=lambda x, c, s: (torch.zeros_like(x), {}))
    policy = JointPolicy(base, settings, potential, True)
    actions = policy.plan(obs, np.zeros(7), [q], noise)
    assert actions is not None and actions.dtype == np.float64
    assert policy.last_report["latent_dtype"] == "torch.float64"
    np.testing.assert_array_equal(policy.last_report["constraint_anchor_rad"], q)
    assert policy.last_report["final_residual_rad"] <= 1e-8
    np.testing.assert_array_equal(
        actions[:, 7], policy.last_report["stages"]["predict_gripper"]
    )
    off = JointPolicy(base, settings, None, False).plan(obs, np.zeros(7), [q], noise)
    assert off.dtype == np.float32
    # Independently reconstruct original float32 all-off Euler output.
    cond = {0: (torch.from_numpy(obs)[None] - base.center[8:]) / base.scale[8:]}
    x = base.apply_conditioning(noise.clone(), cond, 8)
    times = torch.linspace(0, 1, 11)[:-1]
    for t in times:
        v = base.model(x, t)
        v[:, 0, 8:] = 0
        x = base.apply_conditioning(x + (times[1] - times[0]) * v, cond, 8)
    np.testing.assert_array_equal(off, (x * base.scale + base.center)[0].numpy())


def test_potential_gradient_and_unmodified_channels(tmp_path):
    rng = np.random.default_rng(42)
    support = rng.normal(size=(12, 7))
    path = tmp_path / "cache.npz"
    np.savez(
        path,
        support=support,
        safe_anchors=support[:4],
        joint_center=np.zeros(7),
        joint_scale=np.ones(7),
        bandwidth=np.array(0.15),
    )
    potential = JointPotential(path, "cpu")
    point = torch.tensor(support[:1] + 0.013, dtype=torch.float64)
    _, grad = potential.evaluate(point)
    finite = []
    for j in range(7):
        plus, minus = point.clone(), point.clone()
        plus[0, j] += 1e-6
        minus[0, j] -= 1e-6
        finite.append(
            float((potential.evaluate(plus)[0] - potential.evaluate(minus)[0]) / 2e-6)
        )
    np.testing.assert_allclose(grad.numpy()[0], finite, atol=1e-4, rtol=1e-4)
    g, _ = potential.guidance(torch.randn(1, 16, 31), torch.zeros(31), torch.ones(31))
    assert not g[:, :, :8].count_nonzero() and not g[:, 0].count_nonzero()
    assert not g[:, :, 15:].count_nonzero()


def test_signature_rejects_changed_checkpoint(tmp_path):
    model = tmp_path / "model"
    model.write_bytes(b"first")
    output = tmp_path / "out"
    check_output_signature(output, {"count": 100}, [model])
    check_output_signature(output, {"count": 100}, [model])
    model.write_bytes(b"different")
    with pytest.raises(ValueError, match="different comparison"):
        check_output_signature(output, {"count": 100}, [model])


def test_sorting_config_and_train_only_calibration(tmp_path):
    cfg = ExperimentConfig.from_yaml(REPO / "configs/sorting2_combo29.yaml")
    cfg.validate("eval")
    assert cfg.check_checkpoint()["task"] == "sorting2"
    assert all(
        cfg.values["methods"][m]["enabled"]
        for m in ["projection", "pf2mp", "filter", "lipo"]
    )
    assert not cfg.values["methods"]["rtc"]["enabled"]
    from fm_constraint.comparison_io import file_sha256

    protocol = json.loads(
        (
            REPO
            / "docs/results/sorting2_stacking_combo29_id_20261115/sorting2/protocol.json"
        ).read_text()
    )
    assert (
        file_sha256(cfg.values["paths"]["contexts_file"])
        == protocol["cases"]["context_sha256"]
    )
    (tmp_path / "train").mkdir()
    (tmp_path / "val").mkdir()
    obs = np.zeros((4, 12))
    obs[:, :4] = [0.525, -0.3, 0.525, -0.3]
    obs[:, 4:6] = [0.5, 0]
    obs[:, 8:10] = [0.6, 0.1]
    np.savez(tmp_path / "train/a.npz", obs=obs, actions=np.zeros((4, 2)))
    # Deliberately invalid validation file must never influence calibration.
    (tmp_path / "val/b.npz").write_bytes(b"invalid")
    _, result = calibrate(
        tmp_path, [0.525, -0.2992917961963433], cfg.values["constraints"]
    )
    assert result["selected_fixed_margin_m"] == 0.41
    assert result["acceleration"]["training_files"] == 1


def test_sorting_view_preserves_arrays_and_timestamps(tmp_path):
    from preprocess.sorting_constraint_view import prepare

    source = tmp_path / "source"
    rows = {}
    for split in ["train", "val"]:
        (source / split).mkdir(parents=True)
        arrays = dict(
            obs=np.zeros((2, 12), dtype=np.float32),
            actions=np.zeros((2, 2), dtype=np.float32),
            next_obs=np.zeros((2, 12), dtype=np.float32),
            timestamps=np.array([0.0, 0.035]),
            next_timestamps=np.array([0.035, 0.07]),
        )
        np.savez(source / split / "a.npz", **arrays)
        rows[split] = [dict(file=f"{split}/a.npz", source="a.pkl", transitions=2)]
    (source / "manifest.json").write_text(
        json.dumps(dict(obs_dim=12, action_dim=2, splits=rows))
    )
    dest = tmp_path / "sorting_2_sincos_v1"
    result = prepare(source, dest)
    assert result["task"] == "sorting_2"
    for split in rows:
        with np.load(source / split / "a.npz") as a, np.load(
            dest / split / "a.npz"
        ) as b:
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key])


def test_published_paired_metrics_match_success_and_motion():
    root = REPO / "docs/results/sorting2_stacking_combo29_id_20261115"
    for task, arm in [("sorting2", "combo29"), ("stacking", "joint_combo29")]:
        records = json.loads((root / task / "rollout_metrics.json").read_text())
        summary = json.loads((root / task / "summary.json").read_text())
        for metric in ["success", "hand_xyz_jerk_rms"]:
            result = paired_stats(records, arm, metric)
            stored = summary["paired_differences"][metric]
            np.testing.assert_allclose(
                result["combo_minus_baseline"],
                stored["combo_minus_baseline"],
                atol=1e-10,
            )
            np.testing.assert_allclose(result["ci95"], stored["ci95"], atol=1e-10)
