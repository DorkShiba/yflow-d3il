from pathlib import Path
import importlib.util
import json
import pickle
import numpy as np

base = Path("/root/d3il-work")
raw = base / "raw/pushing/all_data"
meta = base / "d3il/environments/dataset/data/pushing"
out = base / "codex/processed/pushing_sincos_v1"

spec = importlib.util.spec_from_file_location(
    "d3il_geo",
    base / "d3il/environments/dataset/geo_transform.py",
)
geo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geo)

splits = {}
for split, filename in (
    ("train", "train_files.pkl"),
    ("val", "eval_files.pkl"),
):
    values = np.load(meta / filename, allow_pickle=True)
    names = [x.decode() if isinstance(x, bytes) else str(x) for x in values]
    assert len(names) == len(set(names))
    assert all((raw / name).is_file() for name in names)
    splits[split] = names

assert len(splits["train"]) == 1800
assert len(splits["val"]) == 200
assert not (set(splits["train"]) & set(splits["val"]))

# 기존 결과를 덮어쓰지 않음
out.mkdir(parents=True, exist_ok=False)

fields = [
    "robot.des_c_pos.x", "robot.des_c_pos.y",
    "robot.c_pos.x", "robot.c_pos.y",
]
for box in ("red-box", "green-box"):
    fields += [
        f"{box}.x", f"{box}.y",
        f"{box}.sin_yaw", f"{box}.cos_yaw",
    ]

manifest = {
    "obs_dim": 12,
    "action_dim": 2,
    "obs_fields": fields,
    "action_fields": ["delta_des_c_pos.x", "delta_des_c_pos.y"],
    "normalized": False,
    "recording_interval_seconds": 0.035,
    "action_definition": "des_c_pos[t+1, :2] - des_c_pos[t, :2]",
    "split_method": "official train_files.pkl / eval_files.pkl",
    "source_root": str(raw),
    "splits": {},
}

total = 0
for split, names in splits.items():
    (out / split).mkdir()
    records = []

    for name in names:
        with (raw / name).open("rb") as f:
            data = pickle.load(f)

        desired = np.asarray(data["robot"]["des_c_pos"])
        actual = np.asarray(data["robot"]["c_pos"])
        timestamps = np.asarray(data["robot"]["time_stamp"])
        parts = [desired[:, :2], actual[:, :2]]

        for box in ("red-box", "green-box"):
            quat = np.asarray(data[box]["quat"])
            assert np.isfinite(quat).all(), (name, box)
            assert np.all(np.linalg.norm(quat, axis=1) > 1e-12), (name, box)
            yaw = geo.quat2euler(quat)[:, -1:]
            parts.extend([
                data[box]["pos"][:, :2],
                np.sin(yaw),
                np.cos(yaw),
            ])

        states = np.concatenate(parts, axis=-1).astype(np.float32)
        actions = np.diff(desired[:, :2], axis=0).astype(np.float32)
        n = len(desired) - 1

        assert states.shape == (n + 1, 12), name
        assert actions.shape == (n, 2), name
        assert np.isfinite(states).all(), name
        assert np.isfinite(actions).all(), name
        np.testing.assert_allclose(
            np.diff(timestamps), 0.035, atol=1e-6, rtol=0
        )

        relative = f"{split}/{Path(name).stem}.npz"
        np.savez_compressed(
            out / relative,
            obs=states[:-1],
            actions=actions,
            next_obs=states[1:],
            timestamps=timestamps[:-1],
            next_timestamps=timestamps[1:],
        )

        # 저장 파일 재검사
        with np.load(out / relative, allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["obs"], states[:-1])
            np.testing.assert_array_equal(saved["next_obs"], states[1:])
            np.testing.assert_array_equal(saved["actions"], actions)
            np.testing.assert_array_equal(saved["timestamps"], timestamps[:-1])
            np.testing.assert_array_equal(saved["next_timestamps"], timestamps[1:])

            np.testing.assert_allclose(
                saved["obs"][:, :2] + saved["actions"],
                saved["next_obs"][:, :2],
                atol=1e-6, rtol=1e-5,
            )
            for key in ("obs", "next_obs"):
                for index in (6, 10):
                    pair = saved[key][:, index:index+2]
                    np.testing.assert_allclose(
                        np.sum(pair**2, axis=1), 1.0, atol=1e-6, rtol=0
                    )

        records.append({
            "file": relative,
            "source": name,
            "transitions": n,
        })
        total += n

    manifest["splits"][split] = records
    count = sum(r["transitions"] for r in records)
    print(f"{split}: {len(records)}개 시연, {count:,}쌍 저장·검사 통과")

assert total == 463386, total
(out / "manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2)
)
print(f"전체 {total:,}쌍 검사 통과")
print("저장 완료:", out)
