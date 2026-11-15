"""Calibrate fixed Sorting2 constraints from train demonstrations and task support."""

import argparse
import copy
import itertools
import json
from pathlib import Path

import numpy as np
import yaml

from configs.config import ExperimentConfig
from .check_constraints import calibrate_acceleration, calibrate_envelope
from .tasks import Sorting


def calibrate(data_dir, reset_xy, constraints):
    """Use train only plus an externally measured canonical reset, never success rates."""
    data_dir = Path(data_dir)
    q0 = np.asarray(reset_xy, dtype=np.float64)
    if q0.shape != (2,) or not np.isfinite(q0).all():
        raise ValueError("Expected finite reset XY")
    result = copy.deepcopy(constraints)
    accel = calibrate_acceleration(data_dir, 0.035, 32)
    envelope = calibrate_envelope(data_dir, 8, 0.0)
    low, high = np.array([0.25, -0.35]), np.array([0.80, 0.45])
    for file in sorted((data_dir / "train").glob("*.npz")):
        with np.load(file) as data:
            xy = data["obs"][:, :4].reshape(-1, 2).astype(np.float64)
        low = np.minimum(low, xy.min(0))
        high = np.maximum(high, xy.max(0))
    low = np.floor(np.minimum(low, q0) * 100) / 100
    high = np.ceil(np.maximum(high, q0) * 100) / 100
    angles = np.arange(8) * 2 * np.pi / 8
    normals = np.c_[np.cos(angles), np.sin(angles)]
    corners = np.array(
        list(itertools.product(Sorting.INIT_BOX[0], Sorting.INIT_BOX[1]))
    )
    gap = float((q0 @ normals.T - (corners @ normals.T).min(0)).max())
    margin = float(
        np.ceil(
            (max(0.0, gap, envelope["maximum_training_start_required_margin_m"]) + 0.01)
            * 100
        )
        / 100
    )
    result["workspace"].update(
        low_m=(low - 0.01).tolist(), high_m=(high + 0.01).tolist(), margin_m=0.01
    )
    result["acceleration"]["max_m_s2"] = accel["recommended_max_m_s2"]
    result["object_envelope"]["margin_m"] = margin
    return result, dict(
        acceleration=accel,
        envelope_train=envelope,
        canonical_reset_xy_m=q0.tolist(),
        official_initial_support=Sorting.INIT_BOX,
        analytical_worst_initial_gap_m=gap,
        selected_fixed_margin_m=margin,
        effective_workspace_low_m=low.tolist(),
        effective_workspace_high_m=high.tolist(),
        scope="train/task-only selection; no runtime expansion or success-based filtering",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/sorting2_combo29.yaml")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--reset-xy",
        type=float,
        nargs=2,
        required=True,
        help="Canonical reset measured on development/smoke, not selected for test success",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    if cfg.task != "sorting_2":
        parser.error("Requires sorting_2 configuration")
    cfg.values["constraints"], report = calibrate(
        args.data_dir, args.reset_xy, cfg.values["constraints"]
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.yaml").write_text(yaml.safe_dump(cfg.values, sort_keys=False))
    (output / "calibration.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
