from pathlib import Path
import json
import numpy as np

base = Path("/root/d3il-work/codex/processed")

settings = [
    ("stacking_sincos_v1", 23, 8, 0.03, (11, 16, 21), 1095, 687552),
    ("inserting_sincos_v1", 16, 2, 0.035, (6, 10, 14), 800, 792387),
]

for name, obs_dim, act_dim, dt, angle_indices, expected_files, expected_pairs in settings:
    root = base / name
    manifest = json.loads((root / "manifest.json").read_text())
    sources = {}
    groups = {}
    total_files = total_pairs = 0

    for split in ("train", "val"):
        records = manifest["splits"][split]
        sources[split] = {r["source"] for r in records}
        groups[split] = {
            r["context_group"] for r in records if "context_group" in r
        }

        assert len(sources[split]) == len(records), "분할 내 파일 중복"
        listed = {r["file"] for r in records}
        actual = {
            str(p.relative_to(root))
            for p in (root / split).glob("*.npz")
        }
        assert listed == actual, (name, split, "파일 목록 불일치")

        for record in records:
            with np.load(root / record["file"], allow_pickle=False) as d:
                n = record["transitions"]
                obs, nxt, act = d["obs"], d["next_obs"], d["actions"]

                assert obs.shape == nxt.shape == (n, obs_dim)
                assert act.shape == (n, act_dim)
                assert obs.dtype == nxt.dtype == act.dtype == np.float32

                for key in d.files:
                    assert np.isfinite(d[key]).all(), (record["file"], key)

                for key in ("timestamps", "next_timestamps"):
                    assert d[key].shape == (n,)

                np.testing.assert_allclose(
                    d["next_timestamps"] - d["timestamps"],
                    dt, atol=1e-6, rtol=0,
                )
                np.testing.assert_allclose(
                    d["timestamps"][1:], d["next_timestamps"][:-1],
                    atol=1e-8, rtol=0,
                )
                np.testing.assert_array_equal(obs[1:], nxt[:-1])

                for state in (obs, nxt):
                    for i in angle_indices:
                        np.testing.assert_allclose(
                            np.sum(state[:, i:i+2] ** 2, axis=1),
                            1.0, atol=1e-6, rtol=0,
                        )

                motion_dim = 7 if name.startswith("stacking") else 2
                np.testing.assert_allclose(
                    obs[:, :motion_dim] + act[:, :motion_dim],
                    nxt[:, :motion_dim],
                    atol=1e-6, rtol=1e-5,
                )
                if name.startswith("stacking"):
                    np.testing.assert_allclose(
                        act[:, 7], nxt[:, 7], atol=1e-6, rtol=0,
                    )

                total_pairs += n
                total_files += 1

        print(f"{name} / {split}: {len(records)}개 검사 통과")

    assert not (sources["train"] & sources["val"]), "학습·검증 파일 중복"
    if name.startswith("inserting"):
        assert not (groups["train"] & groups["val"]), "초기 조건 그룹 중복"

    assert total_files == expected_files
    assert total_pairs == expected_pairs
    print(f"총 {total_files}개 시연, {total_pairs:,}쌍 정상\n")

print("두 태스크 전처리본 검사 완료")
