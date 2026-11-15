"""State FM stacking evaluation; desired joint deltas plus absolute gripper width.
No XY LiPo. Fresh state observations, original D3IL geometric success rule.
"""

import argparse, copy, hashlib, json, os, pickle, sys, time
from pathlib import Path
import numpy as np
import torch
if __package__:
    from .eval_pushing import load_model, Policy
else:
    from eval_pushing import load_model, Policy


def setup(root):
    sys.path.insert(0, str(root))
    os.chdir(root)
    from environments.d3il.envs.gym_stacking_env.gym_stacking.envs.stacking import (
        CubeStacking_Env,
        BlockContextManager,
    )
    from environments.d3il.d3il_sim.utils.geometric_transformation import quat2euler

    return CubeStacking_Env, BlockContextManager, quat2euler


def state(env, quat2euler):
    env.robot.receiveState()
    q = np.array(env.robot.current_j_pos, copy=True)
    c = np.array(env.robot.current_c_pos, copy=True)
    g = float(env.robot.gripper_width)
    boxes = np.array(
        [
            np.r_[
                env.scene.get_obj_pos(b),
                float(quat2euler(env.scene.get_obj_quat(b))[-1]),
            ]
            for b in [env.red_box, env.green_box, env.blue_box]
        ]
    )
    return q, c, g, boxes


def observation(des, g, boxes):
    return np.concatenate(
        [des, [g], *[np.r_[b[:3], np.sin(b[3]), np.cos(b[3])] for b in boxes]]
    ).astype(np.float32)


def full_success(boxes, target, radius=0.06):
    return bool(
        np.all(np.linalg.norm(boxes[:, :2] - target[:2], axis=1) <= radius)
        and min(abs(boxes[i, 2] - boxes[j, 2]) for i, j in [(0, 1), (0, 2), (1, 2)])
        > 0.03
    )


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    for n in ["run-dir", "d3il-root", "contexts-file", "output"]:
        p.add_argument("--" + n, required=True)
    p.add_argument("--ckpt", default="selected.pt")
    p.add_argument("--context-start", type=int, default=0)
    p.add_argument("--n-contexts", type=int, default=100)
    p.add_argument("--n-traj", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--ode-steps", type=int, default=10)
    p.add_argument("--execute-steps", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    for n in ["run_dir", "d3il_root", "contexts_file", "output"]:
        setattr(a, n, str(Path(getattr(a, n)).resolve()))
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    cfg, model, norm, condition, Solver, step = load_model(
        a.run_dir, a.ckpt, "ema", a.device
    )
    assert cfg["task"] == "stacking" and (cfg["obs_dim"], cfg["act_dim"]) == (23, 8)
    assert 0 < a.execute_steps <= cfg["horizon"]
    Env, _, quat2euler = setup(a.d3il_root)
    contexts = pickle.load(open(a.contexts_file, "rb"))[
        a.context_start : a.context_start + a.n_contexts
    ]
    assert len(contexts) == a.n_contexts
    env = Env(render=False, n_substeps=30, max_steps_per_episode=a.max_steps)
    env.start()
    policy = Policy(model, norm, cfg, condition, Solver, a.ode_steps, a.device)
    records = []
    arrays = {}
    start = time.time()
    successes = np.zeros((a.n_contexts, a.n_traj), bool)
    for ci, ctx in enumerate(contexts):
        for ti in range(a.n_traj):
            seed = int(
                np.random.SeedSequence(
                    [a.seed, a.context_start + ci, ti]
                ).generate_state(1)[0]
            )
            np.random.seed(seed)
            gen = torch.Generator(device=a.device).manual_seed(seed)
            env.reset(random=False, context=copy.deepcopy(ctx))
            q, c, g, boxes = state(env, quat2euler)
            des = q.copy()
            env.controller.setSetPoint(des.copy())
            initial_c = c.copy()
            initial_q = q.copy()
            target = np.array(env.scene.get_obj_pos(env.target_box))
            queue = []
            acts = []
            commands = []
            actual = []
            joints = []
            bxs = []
            grippers = []
            starts = []
            times = []
            full = []
            partial = []
            for k in range(a.max_steps):
                starts.append(not queue)
                if not queue:
                    if a.device.startswith("cuda"):
                        torch.cuda.synchronize()
                    t = time.perf_counter()
                    plan = policy.plan(observation(des, g, boxes), gen)
                    if a.device.startswith("cuda"):
                        torch.cuda.synchronize()
                    times.append((time.perf_counter() - t) * 1000)
                    queue = list(plan[: a.execute_steps, :8].copy())
                action = queue.pop(0)
                des = des + action[:7]
                if not np.isfinite(des).all() or not np.isfinite(action).all():
                    raise RuntimeError("Nonfinite policy output")
                _, _, done, info = env.step(np.r_[des, action[7]])
                q, c, g, boxes = state(env, quat2euler)
                s = full_success(boxes, target, env.pos_min_dist)
                assert s == bool(info["success"])
                acts.append(action)
                commands.append(des.copy())
                actual.append(c)
                joints.append(q)
                grippers.append(g)
                bxs.append(boxes)
                full.append(s)
                partial.append([bool(info["success_1"]), bool(info["success_2"])])
                if s or done:
                    break
            key = f"c{a.context_start + ci:03d}_t{ti}"
            successes[ci, ti] = s
            cc = np.vstack([initial_c, actual])
            jerk = np.diff(cc, n=3, axis=0) / 0.03**3
            velocities = np.asarray(acts)[:, :7] / 0.03
            boundaries = np.flatnonzero(starts)[1:]
            seams = np.linalg.norm(
                velocities[boundaries] - velocities[boundaries - 1], axis=1
            )
            rec = dict(
                key=key,
                context=a.context_start + ci,
                repeat=ti,
                seed=seed,
                success=s,
                success_1=bool(partial[-1][0]),
                success_2=bool(partial[-1][1]),
                steps=len(acts),
                hand_jerk_rms=float(np.sqrt(np.mean(np.sum(jerk**2, axis=1))))
                if len(jerk)
                else None,
                joint_boundary_velocity_change_mean=float(seams.mean())
                if len(seams)
                else None,
                plan_ms_mean=float(np.mean(times)),
            )
            records.append(rec)
            for suffix, value in [
                ("act", acts),
                ("des_j", commands),
                ("j", joints),
                ("c", actual),
                ("g", grippers),
                ("boxes", bxs),
                ("plan_starts", starts),
                ("plan_ms", times),
                ("full_success", full),
                ("partial", partial),
                ("initial_c", initial_c),
                ("initial_j", initial_q),
                ("target", target),
            ]:
                arrays[key + "_" + suffix] = np.asarray(value)
            print(json.dumps(rec), flush=True)
    result = dict(
        task="stacking",
        checkpoint_step=step,
        weights="ema",
        n_contexts=a.n_contexts,
        n_traj=a.n_traj,
        context_start=a.context_start,
        success_rate=float(successes.mean()),
        success_1_rate=float(np.mean([r["success_1"] for r in records])),
        success_2_rate=float(np.mean([r["success_2"] for r in records])),
        wall_sec=time.time() - start,
        control_dt_s=0.03,
        args=vars(a),
        checkpoint_sha256=sha(Path(a.run_dir) / a.ckpt),
        normalizer_sha256=sha(Path(a.run_dir) / "normalizer.npz"),
        contexts_sha256=sha(a.contexts_file),
        success_rule="all 3 boxes within .06 m XY target radius and all pairwise Z separation > .03 m; no stability dwell",
        partial_rule="historical entry of 1/2 distinct boxes into target XY radius, not stable stack levels",
        metrics_units={
            "hand_jerk_rms": "m/s^3",
            "joint_boundary_velocity_change_mean": "rad/s",
            "plan_ms_mean": "ms",
        },
        episodes=records,
    )
    np.savez_compressed(out / "rollouts.npz", successes=successes, **arrays)
    (out / "results.json").write_text(json.dumps(result, indent=2))
    print("SAVED", out, flush=True)


if __name__ == "__main__":
    main()
