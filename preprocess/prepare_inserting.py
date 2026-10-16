from pathlib import Path
from collections import defaultdict
import hashlib
import importlib.util
import json
import pickle
import numpy as np

base = Path("/root/d3il-work")
raw = base / "raw/inserting/data_3"
out = base / "codex/processed/inserting_sincos_v1"

spec = importlib.util.spec_from_file_location(
    "d3il_geo",
    base / "d3il/environments/dataset/geo_transform.py",
)
geo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geo)

files = sorted(raw.glob("*.pkl"))
assert len(files) == 800, len(files)

groups = defaultdict(list)
for path in files:
    with path.open("rb") as f:
        data = pickle.load(f)
    context = np.concatenate([
        np.asarray(x, dtype=np.float64).ravel()
        for x in data["context"]
    ])
    key = hashlib.sha256(
        np.round(context, 6).tobytes()
    ).hexdigest()
    groups[key].append(path)

keys = sorted(groups)
rng = np.random.default_rng(42)
order = rng.permutation(len(keys))
n_val = max(1, int(np.ceil(len(keys) * 0.1)))
val_groups = {keys[i] for i in order[:n_val]}

# 기존 결과 덮어쓰기 방지
out.mkdir(parents=True, exist_ok=False)
for split in ("train", "val"):
    (out / split).mkdir()

obs_names = [
    "robot.des_c_pos.x", "robot.des_c_pos.y",
    "robot.c_pos.x", "robot.c_pos.y",
]
for i in range(1, 4):
    obs_names += [
        f"box-{i}.x", f"box-{i}.y",
        f"box-{i}.sin_yaw", f"box-{i}.cos_yaw",
    ]

height = 0.12235930599088184
fixed_quat = np.array([0., 1., 0., 0.])
targets = np.array([
    [0.3575, 0.276, 0.0],
    [0.525, 0.4535, 0.0],
    [0.6925, 0.276, 0.0],
])

manifest = {
    "obs_dim": 16,
    "action_dim": 2,
    "obs_fields": obs_names,
    "action_fields": ["delta_des_c_pos.x", "delta_des_c_pos.y"],
    "normalized": False,
    "recording_interval_seconds": 0.035,
    "fixed_command_z": height,
    "fixed_command_quaternion_wxyz": fixed_quat.tolist(),
    "fixed_target_positions": targets.tolist(),
    "action_definition": "des_c_pos[t+1, :2] - des_c_pos[t, :2]",
    "split_method": "custom context-group split, rounded to 6 decimals",
    "split_seed": 42,
    "validation_group_fraction": 0.1,
    "source_root": str(raw),
    "splits": {"train": [], "val": []},
}

for key in keys:
    split = "val" if key in val_groups else "train"
    for path in groups[key]:
        with path.open("rb") as f:
            data = pickle.load(f)

        desired = np.asarray(data["robot"]["des_c_pos"])
        actual = np.asarray(data["robot"]["c_pos"])
        timestamps = np.asarray(data["robot"]["time_stamp"])
        quat = np.asarray(data["robot"]["des_c_quat"])
        quat = quat / np.linalg.norm(quat, axis=1, keepdims=True)

        np.testing.assert_allclose(desired[:, 2], height, atol=1e-8, rtol=0)
        np.testing.assert_allclose(
            np.abs(quat @ fixed_quat), 1.0, atol=1e-8, rtol=0
        )
        np.testing.assert_allclose(
            np.diff(timestamps), 0.035, atol=1e-6, rtol=0
        )

        parts = [desired[:, :2], actual[:, :2]]
        for i in range(1, 4):
            block = data[f"box-{i}"]
            block_quat = np.asarray(block["quat"])
            assert np.all(np.linalg.norm(block_quat, axis=1) > 1e-12)
            yaw = geo.quat2euler(block_quat)[:, -1:]
            parts.extend([
                block["pos"][:, :2], np.sin(yaw), np.cos(yaw),
            ])
            target = np.asarray(data[f"box-target-{i}"]["pos"])
            assert np.allclose(target, targets[i-1], atol=1e-8, rtol=0)

        states = np.concatenate(parts, axis=-1).astype(np.float32)
        actions = np.diff(desired[:, :2], axis=0).astype(np.float32)
        n = len(desired) - 1

        assert states.shape == (n + 1, 16), path.name
        assert actions.shape == (n, 2), path.name
        assert np.isfinite(states).all(), path.name
        assert np.isfinite(actions).all(), path.name

        relative = f"{split}/{path.stem}.npz"
        np.savez_compressed(
            out / relative,
            obs=states[:-1],
            actions=actions,
            next_obs=states[1:],
            timestamps=timestamps[:-1],
            next_timestamps=timestamps[1:],
        )
        manifest["splits"][split].append({
            "file": relative,
            "source": path.name,
            "context_group": key,
            "transitions": n,
        })

train_groups = {r["context_group"] for r in manifest["splits"]["train"]}
valid_groups = {r["context_group"] for r in manifest["splits"]["val"]}
assert not (train_groups & valid_groups)

total = 0
for split, records in manifest["splits"].items():
    count = sum(r["transitions"] for r in records)
    total += count
    print(f"{split}: {len(records)}개 시연, {count:,}개 상태·행동 쌍")

assert total == 792387, total
(out / "manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2)
)
print("학습·검증 context 그룹 중복: 0")
print("저장 완료:", out)
