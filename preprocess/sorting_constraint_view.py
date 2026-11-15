"""Build the PF manifest view without changing Sorting2 observations/actions."""

import argparse
import json
from pathlib import Path

import numpy as np


def prepare(source, output):
    source, output = Path(source), Path(output)
    if output.name != "sorting_2_sincos_v1":
        raise ValueError("Constraint data directory must be named sorting_2_sincos_v1")
    manifest = json.loads((source / "manifest.json").read_text())
    if (manifest["obs_dim"], manifest["action_dim"]) != (12, 2):
        raise ValueError("Expected Sorting2 state/action dimensions 12/2")
    fields = [
        "robot.des_c_pos.x",
        "robot.des_c_pos.y",
        "robot.c_pos.x",
        "robot.c_pos.y",
    ]
    fields += [
        f"{box}.{field}"
        for box in ["red-box1", "blue-box1"]
        for field in ["x", "y", "sin_yaw", "cos_yaw"]
    ]
    result = dict(
        task="sorting_2",
        obs_dim=12,
        action_dim=2,
        obs_fields=fields,
        action_fields=["delta_des_c_pos.x", "delta_des_c_pos.y"],
        normalized=False,
        recording_interval_seconds=0.035,
        action_definition="des_c_pos[t+1, :2] - des_c_pos[t, :2]",
        splits={},
    )
    output.mkdir(parents=True, exist_ok=False)
    for split in ["train", "val"]:
        (output / split).mkdir()
        result["splits"][split] = []
        for row in manifest["splits"][split]:
            rel = Path(row["file"])
            if rel.is_absolute() or ".." in rel.parts or rel.parts[0] != split:
                raise ValueError("Invalid episode path in manifest")
            with np.load(source / rel) as data:
                arrays = {key: data[key] for key in data.files}
            if "next_timestamps" not in arrays:
                raise ValueError(
                    "Regenerate with preprocess/prepare_sorting.py for raw next_timestamps"
                )
            np.testing.assert_allclose(
                arrays["next_timestamps"] - arrays["timestamps"],
                0.035,
                atol=1e-6,
                rtol=0,
            )
            np.savez_compressed(output / rel, **arrays)
            result["splits"][split].append(
                dict(
                    file=str(rel), source=row["source"], transitions=row["transitions"]
                )
            )
    (output / "manifest.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prepare(args.source, args.output)


if __name__ == "__main__":
    main()
