"""Prepare the same train-only joint limits and PF cache used by the Stacking ID comparison."""

import argparse, json, hashlib
from pathlib import Path
import numpy as np
import torch
from fm_constraint.tasks import setup_d3il
from fm_baseline.eval_stacking import setup


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--normalizer", required=True)
    p.add_argument("--d3il-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    data = Path(a.data_dir).resolve()
    normpath = Path(a.normalizer).resolve()
    D3 = Path(a.d3il_root).resolve()
    R = Path(a.output).resolve()
    R.mkdir(parents=True, exist_ok=False)
    h = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    manifest = json.loads((data / "manifest.json").read_text())
    assert manifest["split_counts"] == {"train": 1001, "val": 93}
    qparts = []
    accparts = []
    lo = np.full(31, np.inf, dtype=np.float32)
    hi = np.full(31, -np.inf, dtype=np.float32)
    for ep in manifest["episodes"]:
        if ep["split"] != "train":
            continue
        with np.load(data / "train" / (Path(ep["source"]).stem + ".npz")) as z:
            obs, act = z["obs"], z["actions"]
            qparts.append(obs[:, :7].astype(np.float64))
            accparts.append(
                np.abs(
                    np.diff(
                        act[:, :7].astype(np.float64), axis=0, prepend=np.zeros((1, 7))
                    )
                )
                / 0.03**2
            )
            x = np.c_[act, obs]
            lo = np.minimum(lo, x.min(0))
            hi = np.maximum(hi, x.max(0))
    with np.load(normpath) as z:
        np.testing.assert_array_equal(z["min"], lo)
        np.testing.assert_array_equal(z["max"], hi)
        center = z["center"][8:15].astype(np.float64)
        scale = z["scale"][8:15].astype(np.float64)
    accel = np.maximum(np.quantile(np.concatenate(accparts), 0.995, axis=0) * 1.1, 1e-6)
    setup_d3il(str(D3))
    Env, _, quat = setup(D3)
    env = Env(render=False, n_substeps=30, max_steps_per_episode=1000)
    env.start()
    low = env.robot.joint_pos_min.tolist()
    high = env.robot.joint_pos_max.tolist()
    settings = dict(
        dt_s=0.03,
        joint_low_rad=low,
        joint_high_rad=high,
        joint_acceleration_limit_rad_s2=accel.tolist(),
        pf_weight=0.01,
        pf_bandwidth_normalized=0.15,
        lipo_bound_rad=0.0003,
        intervention_tolerance_rad=1e-7,
        solver=dict(
            eps_abs=1e-9, eps_rel=1e-9, max_iter=20000, feasibility_tolerance_m=1e-8
        ),
    )
    (R / "joint_settings.json").write_text(json.dumps(settings, indent=2))
    points = (np.concatenate(qparts) - center) / scale
    support = points[
        np.random.default_rng(42).choice(
            len(points), min(3000, len(points)), replace=False
        )
    ]
    sp = torch.tensor(support, dtype=torch.float64, device=a.device)
    dens = []
    bandwidth = 0.15
    with torch.no_grad():
        for start in range(0, len(points), 256):
            query = torch.tensor(
                points[start : start + 256], dtype=torch.float64, device=a.device
            )
            logits = -(query[:, None, :] - sp[None]).square().sum(-1) / (
                2 * bandwidth**2
            )
            dens.extend(
                (
                    torch.logsumexp(logits, -1)
                    - np.log(len(support))
                    - 3.5 * np.log(2 * np.pi * bandwidth**2)
                )
                .cpu()
                .tolist()
            )
        threshold = float(np.quantile(dens, 0.05))
        safelog = []
        for start in range(0, len(support), 256):
            query = sp[start : start + 256]
            logits = -(query[:, None, :] - sp[None]).square().sum(-1) / (
                2 * bandwidth**2
            )
            safelog.extend(
                (
                    torch.logsumexp(logits, -1)
                    - np.log(len(support))
                    - 3.5 * np.log(2 * np.pi * bandwidth**2)
                )
                .cpu()
                .tolist()
            )
    safe = support[np.asarray(safelog) > threshold]
    assert len(safe) > 0
    np.savez_compressed(
        R / "joint_pf_cache.npz",
        support=support,
        safe_anchors=safe,
        joint_center=center,
        joint_scale=scale,
        bandwidth=np.array(bandwidth),
        log_density_threshold=np.array(threshold),
    )
    (R / "CALIBRATION.json").write_text(
        json.dumps(
            dict(
                train_episodes=1001,
                val_episodes=93,
                train_rows=len(points),
                pf_support=len(support),
                safe_train_anchors=len(safe),
                density_threshold=threshold,
                acceleration_quantile=0.995,
                acceleration_multiplier=1.1,
                all_train_only=True,
                normalizer_pass=True,
                source_manifest_sha256=h(data / "manifest.json"),
                raw_hashes_rechecked=False,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
