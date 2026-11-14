"""State-based stacking demos: desired joint delta plus next observed gripper width."""

import argparse, json, pickle, hashlib, importlib.util
from pathlib import Path
import numpy as np


def prepare(raw, metadata, d3il, output):
    raw, metadata, d3il, output = map(Path, [raw, metadata, d3il, output])
    spec = importlib.util.spec_from_file_location(
        "geo", d3il / "environments/dataset/geo_transform.py"
    )
    geo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geo)
    rows = list(map(json.loads, metadata.read_text().splitlines()))
    assert len(rows) == 1094
    output.mkdir(parents=True, exist_ok=False)
    [(output / s).mkdir() for s in ["train", "val"]]
    records = []
    contexts = {"train": set(), "val": set()}
    seen = set()
    for row in rows:
        split = {"train": "train", "eval": "val"}[row["split"]]
        p = raw / (row["source_episode"] + ".pkl")
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        assert digest == row["source_sha256"]
        assert p.name not in seen
        seen.add(p.name)
        d = pickle.load(p.open("rb"))
        q = np.asarray(d["robot"]["des_j_pos"])
        g = np.asarray(d["robot"]["gripper_width"]).reshape(-1, 1)
        ts = np.asarray(d["robot"]["time_stamp"]).reshape(-1)
        np.testing.assert_allclose(np.diff(ts), 0.03, atol=1e-6, rtol=0)
        parts = [q, g]
        for b in ["red-box", "green-box", "blue-box"]:
            yaw = geo.quat2euler(np.asarray(d[b]["quat"]))[:, -1:]
            parts.extend([np.asarray(d[b]["pos"]), np.sin(yaw), np.cos(yaw)])
        obs = np.concatenate(parts, axis=1).astype(np.float32)
        act = np.concatenate([np.diff(q, axis=0), g[1:]], axis=1).astype(np.float32)
        assert obs.shape == (len(act) + 1, 23) and act.shape[1] == 8
        assert np.isfinite(obs).all() and np.isfinite(act).all()
        np.testing.assert_allclose(obs[:-1, :7] + act[:, :7], obs[1:, :7], atol=1e-6)
        np.testing.assert_array_equal(act[:, 7], obs[1:, 7])
        np.savez_compressed(
            output / split / (p.stem + ".npz"),
            obs=obs[:-1],
            actions=act,
            next_obs=obs[1:],
            timestamps=ts[:-1],
        )
        ctx = tuple(
            np.round(
                np.concatenate([np.r_[v[0][:2], v[1]] for v in d["context"][:3]]), 6
            )
        )
        contexts[split].add(ctx)
        records.append(
            {
                "source": p.name,
                "sha256": digest,
                "split": split,
                "transitions": len(act),
            }
        )
    assert not contexts["train"] & contexts["val"], "train/val initial context overlap"
    manifest = {
        "task": "stacking",
        "obs_dim": 23,
        "act_dim": 8,
        "control_dt_s": 0.03,
        "obs_fields": "desired joint angles7, actual gripper width1, red/green/blue xyz+sin(yaw)+cos(yaw) each5",
        "actions": "delta desired joint angles7 + next recorded actual gripper width1 (not original gripper command)",
        "source": "Zeus-Omni/D3IL-Vision original stacking vision state mirror; release-provided train/eval split; not independently byte-checked against official Drive archive",
        "split_counts": {k: sum(r["split"] == k for r in records) for k in contexts},
        "exact_initial_context_overlap": 0,
        "episodes": records,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k != "episodes"}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--d3il", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    prepare(a.raw, a.metadata, a.d3il, a.output)
