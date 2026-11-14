"""Prepare official Sorting-2 state demonstrations for existing FM trainer."""

import argparse, hashlib, importlib.util, json, pickle
from pathlib import Path
import numpy as np


def prepare(raw, metadata, d3il, output):
    raw, metadata, d3il, output = map(Path, [raw, metadata, d3il, output])
    spec = importlib.util.spec_from_file_location(
        "d3il_geo", d3il / "environments/dataset/geo_transform.py"
    )
    geo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geo)
    split_files = {}
    for split, suffix in [("train", "train"), ("val", "eval")]:
        path = metadata / f"2_boxes_{suffix}_files.pkl"
        with path.open("rb") as f:
            names = pickle.load(f)
        names = [x.decode() if isinstance(x, bytes) else str(x) for x in names]
        assert len(names) == len(set(names))
        split_files[split] = names
    assert not set(split_files["train"]) & set(split_files["val"])
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "task": "sorting2",
        "obs_dim": 12,
        "action_dim": 2,
        "split_method": "official episode lists",
        "obs_fields": [
            "desired_x",
            "desired_y",
            "actual_x",
            "actual_y",
            "red_x",
            "red_y",
            "red_sin_yaw",
            "red_cos_yaw",
            "blue_x",
            "blue_y",
            "blue_sin_yaw",
            "blue_cos_yaw",
        ],
        "action_definition": "desired_xy[t+1]-desired_xy[t]",
        "recording_interval_seconds": 0.035,
        "splits": {},
    }
    contexts = {}
    stats = {}
    for split, names in split_files.items():
        (output / split).mkdir()
        records = []
        ctx = []
        first = []
        for name in names:
            path = raw / Path(name).name
            with path.open("rb") as f:
                d = pickle.load(f)
            des = np.asarray(d["robot"]["des_c_pos"])
            actual = np.asarray(d["robot"]["c_pos"])
            ts = np.asarray(d["robot"]["time_stamp"]).reshape(-1)
            np.testing.assert_allclose(np.diff(ts), 0.035, atol=1e-6, rtol=0)
            parts = [des[:, :2], actual[:, :2]]
            for box in ["red-box1", "blue-box1"]:
                quat = np.asarray(d[box]["quat"])
                assert np.all(np.linalg.norm(quat, axis=1) > 1e-12)
                yaw = geo.quat2euler(quat)[:, -1:]
                parts += [np.asarray(d[box]["pos"])[:, :2], np.sin(yaw), np.cos(yaw)]
            obs = np.concatenate(parts, axis=1).astype(np.float32)
            act = np.diff(des[:, :2], axis=0).astype(np.float32)
            assert obs.shape == (len(act) + 1, 12)
            assert np.isfinite(obs).all() and np.isfinite(act).all()
            np.testing.assert_allclose(obs[:-1, :2] + act, obs[1:, :2], atol=1e-6)
            np.savez_compressed(
                output / split / (path.stem + ".npz"),
                obs=obs[:-1],
                actions=act,
                next_obs=obs[1:],
                timestamps=ts[:-1],
            )
            key = hashlib.sha256(np.round(obs[0, 4:], 6).tobytes()).hexdigest()
            ctx.append(key)
            first.append(obs[0].tolist())
            records.append(
                {
                    "file": f"{split}/{path.stem}.npz",
                    "source": path.name,
                    "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "transitions": len(act),
                    "initial_objects_hash": key,
                }
            )
        report["splits"][split] = records
        contexts[split] = set(ctx)
        stats[split] = {
            "episodes": len(records),
            "transitions": sum(r["transitions"] for r in records),
            "initial_observations": first,
        }
    report["exact_initial_object_overlap"] = len(contexts["train"] & contexts["val"])
    (output / "manifest.json").write_text(json.dumps(report, indent=2))
    (output / "audit.json").write_text(json.dumps(stats, indent=2))
    print(
        json.dumps(
            {
                "counts": {
                    k: {x: v[x] for x in ["episodes", "transitions"]}
                    for k, v in stats.items()
                },
                "exact_initial_object_overlap": report["exact_initial_object_overlap"],
            },
            indent=2,
        )
    )
    if report["exact_initial_object_overlap"]:
        raise RuntimeError("Initial context overlap: review split before training")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--d3il", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    prepare(a.raw, a.metadata, a.d3il, a.output)
