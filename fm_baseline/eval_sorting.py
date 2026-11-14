"""Sorting-2 ID evaluation using the existing FM and LiPo implementations.
Reads current physical state after each step (not the stale pre-step obs).
"""

import argparse, copy, json, os, pickle, sys, time
from pathlib import Path
import numpy as np
import torch
from eval_pushing import load_model, Policy, build_obs
from eval_smoothness import add_arguments, EvaluationDiagnostics


def sorting_success(red, blue):
    return bool(
        0.3 < red[0] < 0.5
        and 0.22 < red[1] < 0.41
        and 0.525 < blue[0] < 0.725
        and 0.22 < blue[1] < 0.41
    )


def read_state(env, quat2euler):
    env.robot.receiveState()
    c = np.array(env.robot.current_c_pos, copy=True)
    boxes = []
    for obj in [env.red_box_1, env.blue_box_1]:
        p = np.asarray(env.scene.get_obj_pos(obj)).copy()
        q = np.asarray(env.scene.get_obj_quat(obj)).copy()
        yaw = float(quat2euler(q)[-1])
        boxes.append((p, q, yaw))
    return c, boxes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--ckpt", default="last.pt")
    p.add_argument("--weights", default="ema", choices=["ema", "model"])
    p.add_argument("--d3il-root", required=True)
    p.add_argument("--contexts-file", required=True)
    p.add_argument("--context-start", type=int, default=0)
    p.add_argument("--n-contexts", type=int, default=30)
    p.add_argument("--n-traj", type=int, default=4)
    p.add_argument("--ode-steps", type=int, default=10)
    p.add_argument("--execute-steps", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--tag", default="id")
    add_arguments(p)
    a = p.parse_args()
    for k in ["run_dir", "d3il_root", "contexts_file", "output_root"]:
        if getattr(a, k):
            setattr(a, k, os.path.abspath(getattr(a, k)))
    if a.n_contexts <= 0 or a.n_traj <= 0:
        raise ValueError("positive counts required")
    cfg, model, norm, condition, Solver, step = load_model(
        a.run_dir, a.ckpt, a.weights, a.device
    )
    assert cfg["task"] == "sorting2" and cfg["obs_dim"] == 12 and cfg["act_dim"] == 2
    sys.path.insert(0, a.d3il_root)
    os.chdir(a.d3il_root)
    from environments.d3il.envs.gym_sorting_env.gym_sorting.envs.sorting import (
        Sorting_Env,
    )
    from environments.d3il.d3il_sim.utils.geometric_transformation import quat2euler

    with open(a.contexts_file, "rb") as f:
        allcontexts = pickle.load(f)
    contexts = allcontexts[a.context_start : a.context_start + a.n_contexts]
    if len(contexts) != a.n_contexts:
        raise ValueError("Insufficient contexts; no silent shrinking")
    env = Sorting_Env(
        render=False, num_boxes=2, n_substeps=35, max_steps_per_episode=a.max_steps
    )
    env.start()
    policy = Policy(model, norm, cfg, condition, Solver, a.ode_steps, a.device)
    diag = EvaluationDiagnostics(a, cfg, env, a.run_dir, a.contexts_file, a.d3il_root)
    out = (
        Path(a.output_root or a.run_dir)
        / f"eval_sorting2_{a.tag}_{a.postprocess}_{time.time_ns()}"
    )
    out.mkdir(parents=True)
    successes = np.zeros((len(contexts), a.n_traj))
    lengths = np.zeros_like(successes, dtype=int)
    rollouts = {}
    plan_times = []
    t0 = time.time()
    for ci, ctx in enumerate(contexts):
        for ti in range(a.n_traj):
            processor, generator = diag.start(ci, ti, env, ctx)
            env.reset(random=False, context=copy.deepcopy(ctx))
            c, boxes = read_state(env, quat2euler)
            z = c[2:3].copy()
            des = c[:2].copy()
            initial_des = des.copy()
            initial_c = c[:2].copy()
            queue = []
            acts = []
            desired = []
            actual = []
            bxs = []
            starts = []
            done = False
            for k in range(a.max_steps):
                starts.append(not queue)
                if not queue:
                    obs = build_obs(des, c, boxes)
                    if a.device.startswith("cuda"):
                        torch.cuda.synchronize()
                    tp = time.perf_counter()
                    plan = policy.plan(obs, generator)
                    if a.device.startswith("cuda"):
                        torch.cuda.synchronize()
                    plan_times.append(1000 * (time.perf_counter() - tp))
                    history = np.vstack([initial_des, *desired[-3:]])[-3:]
                    processed = processor.process(plan[:, :2], history)
                    queue = list(processed[: a.execute_steps].copy())
                action = queue.pop(0)
                des = des + action
                if not np.isfinite(des).all():
                    raise RuntimeError("Nonfinite policy output")
                _, _, done, info = env.step(np.r_[des, z, [0.0, 1.0, 0.0, 0.0]])
                c, boxes = read_state(env, quat2euler)
                success = sorting_success(boxes[0][0], boxes[1][0])
                assert success == bool(info["success"])
                desired.append(des.copy())
                actual.append(c.copy())
                acts.append(action.copy())
                bxs.append(np.concatenate([np.r_[b[0], b[2]] for b in boxes]))
                if success or done:
                    break
            successes[ci, ti] = success
            lengths[ci, ti] = len(acts)
            key = f"c{ci:03d}_t{ti}"
            diag.finish(
                key,
                processor,
                acts,
                desired,
                actual,
                starts,
                initial_des,
                initial_c,
                rollouts,
            )
            for suffix, value in [
                ("act", acts),
                ("des", desired),
                ("c", actual),
                ("boxes", bxs),
            ]:
                rollouts[key + "_" + suffix] = np.asarray(value)
            print(
                f"context {ci} traj {ti} success {int(success)} steps {len(acts)} elapsed {time.time() - t0:.1f}",
                flush=True,
            )
    results = {
        "task": "sorting2",
        "checkpoint_step": step,
        "checkpoint": a.ckpt,
        "weights": a.weights,
        "n_contexts": len(contexts),
        "n_traj": a.n_traj,
        "seed": a.seed,
        "ode_steps": a.ode_steps,
        "execute_steps": a.execute_steps,
        "max_steps": a.max_steps,
        "success_rate": float(successes.mean()),
        "mean_steps": float(lengths.mean()),
        "plan_ms": float(np.mean(plan_times)),
        "wall_sec": time.time() - t0,
        "success_rule": "both color-specific XY rectangular goals; original D3IL rule",
    }
    diag.save(results, out)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    np.savez_compressed(
        out / "rollouts.npz", successes=successes, lengths=lengths, **rollouts
    )
    print("SAVED", out, flush=True)


if __name__ == "__main__":
    main()
