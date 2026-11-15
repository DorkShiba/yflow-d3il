"""Verify the published inference inputs before running Stacking; no training data needed."""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from .comparison_io import file_sha256
from . import checkpoint_reads
from .joint import JointPolicy, JointPotential
from fm_baseline.eval_pushing import load_model, Policy

REPO = Path(__file__).resolve().parents[1]
BUNDLE = REPO / "fm_baseline/results/stacking_baseline"


def check_bundle():
    manifest = json.loads((BUNDLE / "sha256.json").read_text())
    for filename, expected in manifest.items():
        if file_sha256(BUNDLE / filename) != expected:
            raise ValueError(f"Stacking bundle hash mismatch: {filename}")
    contexts = pickle.loads((BUNDLE / "id_contexts.pkl").read_bytes())
    if len(contexts) != 142:
        raise ValueError("Expected the frozen 142 ID contexts")
    with np.load(BUNDLE / "normalizer.npz", allow_pickle=False) as norm, np.load(
        BUNDLE / "joint_pf_cache.npz", allow_pickle=False
    ) as cache:
        assert norm["center"].shape == norm["scale"].shape == (31,)
        assert (norm["scale"] > 0).all()
        np.testing.assert_array_equal(cache["joint_center"], norm["center"][8:15])
        np.testing.assert_array_equal(cache["joint_scale"], norm["scale"][8:15])
        assert cache["support"].shape == (3000, 7)
        assert cache["safe_anchors"].shape == (2951, 7)
        assert all(np.isfinite(cache[k]).all() for k in cache.files)
    return dict(passed=True, checked_files=len(manifest), contexts=len(contexts))


def check_plans(device):
    with checkpoint_reads():
        cfg, model, norm, condition, solver, step = load_model(
            str(BUNDLE), "baseline_ema.pt", "ema", device
        )
    assert (cfg["obs_dim"], cfg["act_dim"], cfg["horizon"]) == (23, 8, 16)
    base = Policy(model, norm, cfg, condition, solver, 10, device)
    settings = json.loads((REPO / "configs/stacking_combo31.json").read_text())
    q = (np.array(settings["joint_low_rad"]) + np.array(settings["joint_high_rad"])) / 2
    obs = norm.center[8:].copy()
    obs[:7] = q
    noise = torch.randn(1, 16, 31, generator=torch.Generator().manual_seed(42))
    potential = JointPotential(BUNDLE / "joint_pf_cache.npz", device)
    policy = JointPolicy(base, settings, potential, True)
    first = policy.plan(obs, np.zeros(7), [q], noise)
    if first is None:
        raise ValueError(f"First CPU plan failed: {policy.last_report['status']}")
    second = policy.plan(obs, np.zeros(7), [q], noise, rtc_suffix=first[8:, :7])
    if second is None or not np.isfinite(second).all():
        raise ValueError(f"RTC plan failed: {policy.last_report['status']}")
    assert policy.last_report["rtc_calls"] == 10
    return dict(checkpoint_step=step, rtc_calls=10, simulated_rollouts=0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--plan-check",
        action="store_true",
        help="Also run real model + QPs + RTC; no simulator",
    )
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    result = check_bundle()
    if args.plan_check:
        if args.device == "cpu":
            torch.set_num_threads(1)
        result.update(check_plans(args.device))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
