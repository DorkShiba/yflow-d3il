"""Render one verified D3IL replay into an intermediate camera clip and counters."""

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import glfw
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy.interpolate import RegularGridInterpolator

from density import CONTEXT, CUTOFFS, GOALS, PALETTE, load_support, log_density


def render_run(source, d3il, run_dir, output, method, seed, evaluate=False):
    sys.path.insert(0, str(source))
    from fm_constraint import import_baseline
    from ood.apply import context_from_json, load_cases

    if evaluate:
        import torch
        from configs.config import ExperimentConfig
        from fm_constraint.evaluation import evaluate as evaluate_policy
        from fm_constraint.experiment import combo_flags

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        settings = {
            f"methods.{k}.enabled": v for k, v in combo_flags(int(method[1:])).items()
        }
        settings.update(
            {
                "device": "cpu",
                "paths.d3il_root": str(d3il),
                "eval.max_steps": 2000,
                "eval.n_contexts": 0,
                "eval.execute_steps": 8,
                "sampling.predict_steps": 10,
                "eval.ood": "A_10",
            }
        )
        cfg = ExperimentConfig.from_yaml(source / "configs/inserting.yaml", settings)
        evaluate_policy(
            cfg, sampling_seeds=[seed], context_ids=[CONTEXT], output_directory=run_dir
        )

    result = json.loads((run_dir / "results.json").read_text())
    (episode,) = result["episodes"]
    expected_methods = {"pf2mp"} if method == "I08" else set()
    if (
        episode["context_id"] != CONTEXT
        or episode["sampling_seed"] != seed
        or result["ood"]["condition"] != "A_10"
        or result["task"] != "inserting"
        or {k for k, v in result["methods"].items() if v} != expected_methods
        or result["control_dt_s"] != 0.035
    ):
        raise ValueError(f"Unexpected rollout metadata at {run_dir}")
    (rows,) = json.loads((run_dir / "rollouts.json").read_text()).values()
    if not rows or len(rows) != episode["steps"]:
        raise ValueError(f"Incomplete rollout at {run_dir}")
    commands = np.array(
        [rows[0]["command_state"]] + [r["next_command_state"] for r in rows]
    )
    blocks = np.array(
        [rows[0]["execution_geometry"]["blocks_xy_m"]] + [r["boxes_xy_m"] for r in rows]
    )
    support = load_support(source)
    case = next(
        c
        for c in load_cases(source / "data/ood", "inserting", "A_10")[0]
        if c["context_id"] == CONTEXT
    )
    module = import_baseline("eval_inserting")
    InsertingEnv, angles = module.setup_d3il(str(d3il))
    env = InsertingEnv(render=False, n_substeps=35, max_steps_per_episode=2000)
    env.start()
    env.reset(random=False, context=copy.deepcopy(context_from_json(case["context"])))
    initial_hand, _ = module.read_state(env, angles)
    if not glfw.init():
        raise RuntimeError("GLFW requires a graphical session or Xvfb")
    glfw.window_hint(glfw.VISIBLE, 0)
    window = glfw.create_window(720, 504, "render", None, None)
    if not window:
        raise RuntimeError("Could not create GLFW window")
    glfw.make_context_current(window)
    model = env.scene.model
    data = env.scene.data
    model.geom_rgba[0, :3] = [0.16, 0.23, 0.25]
    for j in range(model.ngeom):
        if j == 24:
            model.geom_rgba[j, :3] = [0.64, 0.67, 0.68]
        if j >= 25 and 0.01 < model.geom_rgba[j, 3] < 0.99:
            model.geom_rgba[j, 3] = 1
    model.vis.headlight.ambient[:] = 0.22
    model.vis.global_.offwidth = 720
    model.vis.global_.offheight = 504
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [0.50, 0.02, 0.020]
    camera.azimuth = 90
    camera.elevation = -73
    camera.distance = 1.28
    options = mujoco.MjvOption()
    scene = mujoco.MjvScene(model, maxgeom=2000)
    render_context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)
    mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, render_context)
    viewport = mujoco.MjrRect(0, 0, 720, 504)
    rgb = np.empty((504, 720, 3), dtype=np.uint8)

    def render():
        mujoco.mjv_updateScene(
            model, data, options, None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene
        )
        mujoco.mjr_render(viewport, scene, render_context)
        mujoco.mjr_readPixels(rgb, None, viewport, render_context)
        return np.ascontiguousarray(rgb[::-1])

    # Table density interpolation in world XY; no display texture enters the physics model.
    table_id = next(
        j
        for j in range(model.ngeom)
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[j])
        == "table_plane"
    )
    height, width = 504, 720
    z = data.geom_xpos[table_id, 2] + model.geom_size[table_id, 2]
    render()
    eyes = scene.camera
    camera_position = np.mean([c.pos.astype(float) for c in eyes], axis=0)
    forward = eyes[0].forward.astype(float)
    up = eyes[0].up.astype(float)
    right = np.cross(forward, up)
    scale = (eyes[0].frustum_top - eyes[0].frustum_bottom) / (2 * eyes[0].frustum_near)

    def floor_pixels(points):
        points = np.atleast_2d(points)
        rel = np.c_[points, np.full(len(points), z)] - camera_position
        depth = rel @ forward
        return np.column_stack(
            (
                width * 0.5 + (rel @ right) / depth * height / (2 * scale) - 0.5,
                height * 0.5 - (rel @ up) / depth * height / (2 * scale) - 0.5,
            )
        )

    Y, X = np.mgrid[0:height, 0:width]
    ray = (
        forward
        + (2 * (X + 0.5) - width)[:, :, None] / height * scale * right
        + (height - 2 * (Y + 0.5))[:, :, None] / height * scale * up
    )
    world = camera_position + (z - camera_position[2]) / ray[:, :, 2:3] * ray
    bounds = (np.arange(-0.10, 0.91, 0.005), np.arange(-1.0, 1.005, 0.005))
    xy = np.stack(np.meshgrid(*bounds, indexing="ij"), axis=-1).reshape(-1, 2)
    interpolator = RegularGridInterpolator(
        bounds,
        np.reshape(log_density(xy, support), tuple(map(len, bounds))),
        bounds_error=False,
        fill_value=-1000,
    )
    heat = np.asarray(PALETTE, dtype=np.uint8)[
        np.searchsorted(CUTOFFS, interpolator(world[:, :, :2]))
    ]
    flags = (mujoco.mjtRndFlag.mjRND_SEGMENT, mujoco.mjtRndFlag.mjRND_IDCOLOR)
    track = floor_pixels(commands)
    geom_type = int(mujoco.mjtObj.mjOBJ_GEOM)

    def draw_frame(t):
        color = render().copy()
        for flag in flags:
            scene.flags[flag] = 1
        mujoco.mjr_render(viewport, scene, render_context)
        mujoco.mjr_readPixels(rgb, None, viewport, render_context)
        for flag in flags:
            scene.flags[flag] = 0
        segmentation_rgb = rgb[::-1].astype(np.int32)
        segmentation_ids = (
            segmentation_rgb[:, :, 0]
            + 256 * segmentation_rgb[:, :, 1]
            + 65536 * segmentation_rgb[:, :, 2]
        )
        table_tag = next(
            g.segid + 1
            for g in scene.geoms[: scene.ngeom]
            if g.objtype == geom_type and g.objid == table_id
        )
        mask = segmentation_ids == table_tag
        if t == 0 and mask.sum() <= 0.15 * height * width:
            raise ValueError("Table is outside the calibrated camera view")
        overlay = Image.fromarray(heat.copy())
        draw = ImageDraw.Draw(overlay)
        trail = [tuple(p) for p in track[max(0, t - 64) : t + 1 : 2]]
        if len(trail) > 1:
            draw.line(trail, fill=(255, 255, 255), width=5)
            draw.line(trail, fill=(38, 53, 63), width=2)
        pixel_x, pixel_y = track[t]
        draw.ellipse(
            (pixel_x - 6, pixel_y - 6, pixel_x + 6, pixel_y + 6),
            outline=(255, 255, 255),
            width=3,
        )
        draw.ellipse(
            (pixel_x - 4, pixel_y - 4, pixel_x + 4, pixel_y + 4),
            outline=(23, 51, 58),
            width=2,
        )
        # Preserve object occlusion; the palette applies to the visible table alone.
        overlay = np.asarray(overlay)
        color[mask] = overlay[mask]
        return color

    ffmpeg = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        "720x504",
        "-r",
        "25",
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "16",
        "-pix_fmt",
        "yuv420p",
        str(output.with_suffix(".mp4")),
    ]
    with subprocess.Popen(ffmpeg, stdin=subprocess.PIPE) as encoder:
        try:
            encoder.stdin.write(draw_frame(0).tobytes())
            max_error = 0.0
            for step, row in enumerate(rows, 1):
                xy = row["next_command_state"]
                _, _, _done, info = env.step(
                    np.r_[xy, initial_hand[2], 0.0, 1.0, 0.0, 0.0]
                )
                hand, boxes = module.read_state(env, angles)
                max_error = max(
                    max_error,
                    np.max(np.abs(hand - row["next_measured_state"])),
                    np.max(np.abs(np.array([v[0][:2] for v in boxes]) - blocks[step])),
                )
                if step % 4 == 0 or step == len(rows):
                    encoder.stdin.write(draw_frame(step).tobytes())
        finally:
            encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError("Camera clip encoding failed")
    if not max_error < 1e-7 or bool(info["success"]) != episode["success"]:
        raise ValueError(f"Replay diverged from recorded trajectory: {max_error=}")
    glfw.destroy_window(window)
    glfw.terminate()
    high = np.r_[0, np.cumsum(log_density(commands[1:], support) > CUTOFFS[-1])]
    reached = np.maximum.accumulate(
        np.linalg.norm(blocks - GOALS, axis=2) <= 0.01, axis=0
    ).sum(axis=1)
    np.savez(
        output.with_suffix(".npz"),
        high=high,
        reached=reached,
        success=episode["success"],
    )
    print(f"{method} seed {seed}: {len(rows)} steps, {max_error=}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "d3il", "run_dir", "output"):
        parser.add_argument(key, type=Path)
    parser.add_argument("method", choices=["I00", "I08"])
    parser.add_argument("seed", type=int)
    parser.add_argument("--evaluate", action="store_true")
    render_run(**vars(parser.parse_args()))
