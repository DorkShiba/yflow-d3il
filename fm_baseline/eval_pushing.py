import argparse
import copy
import json
import os
import pickle
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch

from smoothness import rollout_metrics, summarize_metrics

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from train import Normalizer, add_spectral_norm, resolve_d3il, resolve_data_dir, resolve_hardflow


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--ckpt", default="last.pt")
    p.add_argument("--weights", choices=["ema", "model"], default="ema")
    p.add_argument("--d3il-root", default=None)
    p.add_argument("--hardflow-root", default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--raw-dir", default="/root/d3il-work/raw/pushing/all_data")
    p.add_argument("--contexts", choices=["test", "train"], default="test")
    p.add_argument("--n-contexts", type=int, default=30)
    p.add_argument("--n-traj", type=int, default=1)
    p.add_argument("--ode-steps", type=int, default=10)
    p.add_argument("--execute-steps", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--paired-noise", action="store_true", help="Independent per-rollout policy RNG for paired ablations")
    p.add_argument("--replan-noise", choices=["fresh", "fixed"], default="fresh",
                   help="fixed reuses the initial latent tensor at every replan within a rollout")
    p.add_argument("--output-root", default=None, help="Store evaluation separately from checkpoint directory")
    p.add_argument("--check-obs", type=int, default=0)
    p.add_argument("--tag", default="")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--record-joints", action="store_true", help="Store post-step Panda arm joint positions for browser replay")
    p.add_argument("--timing-file", default=None)
    p.add_argument("--capture-dir", default=None)
    p.add_argument("--capture-camera", choices=["top", "oblique"], default="oblique")
    return p.parse_args()


def setup_d3il(root):
    os.chdir(root)
    for p in [root, os.path.join(root, "environments", "d3il")]:
        if p not in sys.path:
            sys.path.insert(0, p)
    from envs.gym_pushing_env.gym_pushing.envs.pushing import Block_Push_Env
    from environments.d3il.d3il_sim.utils.geometric_transformation import quat2euler
    return Block_Push_Env, quat2euler


def load_model(run_dir, ckpt_name, weights, hardflow_root, device):
    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    sys.path.insert(0, hardflow_root)
    from hardflow.models_flow.flow_matcher import apply_conditioning
    from hardflow.models_flow.flow_policy import ConditionedODESolver
    from hardflow.models_flow.unet import TemporalUnet

    obs_dim, act_dim, horizon = cfg["obs_dim"], cfg["act_dim"], cfg["horizon"]
    model = TemporalUnet(
        horizon=horizon,
        transition_dim=obs_dim + act_dim,
        cond_dim=obs_dim,
        dim=cfg["model"]["dim"],
        dim_mults=tuple(cfg["model"]["dim_mults"]),
        attention=cfg["model"]["attention"],
    )
    if cfg.get("spectral_norm"):
        skip = [model.final_conv[1]] if cfg.get("spectral_norm_skip_output") else []
        add_spectral_norm(model, skip)
    ckpt = torch.load(os.path.join(run_dir, ckpt_name), map_location="cpu")
    if ckpt.get(weights) is None:
        raise SystemExit(f"No {weights} weights in {ckpt_name}")
    model.load_state_dict(ckpt[weights])
    model = model.to(device).eval()
    nz = np.load(os.path.join(run_dir, "normalizer.npz"))
    norm = Normalizer(nz["center"], nz["scale"])
    return cfg, model, norm, apply_conditioning, ConditionedODESolver, ckpt.get("step")


class Policy:
    def __init__(self, model, norm, cfg, apply_conditioning, Solver, ode_steps, device):
        self.model = model
        self.norm = norm
        self.obs_dim = cfg["obs_dim"]
        self.act_dim = cfg["act_dim"]
        self.horizon = cfg["horizon"]
        self.apply_conditioning = apply_conditioning
        self.Solver = Solver
        self.device = device
        self.t_span = torch.linspace(0.0, 1.0, ode_steps + 1, device=device)[:-1]
        self.center = torch.from_numpy(norm.center).to(device)
        self.scale = torch.from_numpy(norm.scale).to(device)

    @torch.no_grad()
    def plan(self, obs, generator=None):
        o = torch.from_numpy(np.asarray(obs, dtype=np.float32)).to(self.device)[None]
        o_n = (o - self.center[self.act_dim:]) / self.scale[self.act_dim:]
        cond = {0: o_n}
        solver = self.Solver(self.model, cond, action_dim=self.act_dim, value=None, ode_method="euler")
        x = torch.randn(1, self.horizon, self.obs_dim + self.act_dim, device=self.device, generator=generator)
        x = self.apply_conditioning(x, cond, self.act_dim)
        x = solver(x, t_span=self.t_span)
        x = x * self.scale + self.center
        return x[0].cpu().numpy()


def read_state(env, quat2euler):
    env.robot.receiveState()
    c = np.asarray(env.robot.current_c_pos, dtype=np.float64).copy()
    boxes = []
    for b in [env.push_box1, env.push_box2]:
        pos = np.asarray(env.scene.get_obj_pos(b), dtype=np.float64).copy()
        quat = np.asarray(env.scene.get_obj_quat(b), dtype=np.float64).copy()
        yaw = float(np.asarray(quat2euler(quat)).reshape(-1)[-1])
        boxes.append((pos, quat, yaw))
    return c, boxes


def build_obs(des_xy, c, boxes):
    parts = [np.asarray(des_xy[:2]), c[:2]]
    for pos, _, yaw in boxes:
        parts.append([pos[0], pos[1], np.sin(yaw), np.cos(yaw)])
    return np.concatenate([np.asarray(p, dtype=np.float64).reshape(-1) for p in parts]).astype(np.float32)


def check_obs(env, quat2euler, cfg, raw_dir, n):
    val_dir = os.path.join(cfg["data_dir"], "val")
    files = sorted(f for f in os.listdir(val_dir) if f.endswith(".npz"))[:n]
    errs = []
    for f in files:
        name = f[:-4] + ".pkl"
        with open(os.path.join(raw_dir, name), "rb") as fh:
            data = pickle.load(fh)
        env.reset(random=False, context=copy.deepcopy(data["context"]))
        env.robot.beam_to_joint_pos(np.asarray(data["robot"]["j_pos"][0]).copy())
        env.scene.next_step(log=False)
        c, boxes = read_state(env, quat2euler)
        des0 = np.asarray(data["robot"]["des_c_pos"][0])
        ours = build_obs(des0, c, boxes)
        ref = np.load(os.path.join(val_dir, f))["obs"][0]
        e = np.abs(ours - ref)
        errs.append(e)
        print(f"check {f}  max abs err {e.max():.2e}  per-dim {np.round(e, 4).tolist()}", flush=True)
    errs = np.stack(errs)
    print(f"check summary  max {errs.max():.2e}  mean per-dim {np.round(errs.mean(0), 5).tolist()}", flush=True)


def main():
    args = parse_args()
    if args.timing_file:
        args.timing_file = os.path.abspath(args.timing_file)
    if args.replan_noise == "fixed" and not args.paired_noise:
        raise SystemExit("Fixed replan noise requires --paired-noise")
    if args.output_root:
        args.output_root = os.path.abspath(args.output_root)
    if args.capture_dir:
        args.capture_dir = os.path.abspath(args.capture_dir)
    cfg_path = os.path.join(args.run_dir, "config.json")
    run_dir = os.path.abspath(args.run_dir)
    with open(cfg_path, encoding="utf-8") as f:
        cfg0 = json.load(f)
    if cfg0["task"] != "pushing":
        raise SystemExit("This script evaluates pushing runs only")
    hardflow_root = resolve_hardflow(args.hardflow_root, cfg0)
    args.d3il_root = resolve_d3il(args.d3il_root)
    cfg, model, norm, apply_conditioning, Solver, ckpt_step = load_model(run_dir, args.ckpt, args.weights, hardflow_root, args.device)
    cfg["data_dir"] = resolve_data_dir(args.data_dir, cfg)

    Block_Push_Env, quat2euler = setup_d3il(args.d3il_root)
    env = Block_Push_Env(render=False, n_substeps=35, max_steps_per_episode=args.max_steps)
    env.start()
    if args.timing_file:
        with open(args.timing_file, "w") as timing:
            json.dump({"dt": float(env.scene.dt) * 35, "substeps": 35}, timing)
    capture = None
    if args.capture_dir:
        sys.path.insert(0, os.path.dirname(HERE))
        from visualizer.capture import Capture
        capture = Capture(env, args.capture_dir, args.capture_camera)

    if args.check_obs > 0:
        check_obs(env, quat2euler, cfg, args.raw_dir, args.check_obs)
        return

    ctx_file = os.path.join(args.d3il_root, "environments/dataset/data/pushing", f"{args.contexts}_contexts.pkl")
    contexts = np.load(ctx_file, allow_pickle=True)
    n_ctx = min(args.n_contexts, len(contexts))
    policy = Policy(model, norm, cfg, apply_conditioning, Solver, args.ode_steps, args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = os.path.join(args.output_root or run_dir, f"eval_pushing_{args.contexts}_{args.tag + '_' if args.tag else ''}{stamp}")
    os.makedirs(out_dir)

    successes = np.zeros((n_ctx, args.n_traj))
    modes = np.full((n_ctx, args.n_traj), -1)
    mean_dist = np.zeros((n_ctx, args.n_traj))
    lengths = np.zeros((n_ctx, args.n_traj), dtype=int)
    rollouts = {}
    diagnostics = {}
    t0 = time.time()
    plan_time = 0.0
    n_plans = 0

    for ci in range(n_ctx):
        for ti in range(args.n_traj):
            noise_generator = None
            if args.paired_noise:
                rollout_seed = int(np.random.SeedSequence([args.seed, ci, ti]).generate_state(1)[0])
                noise_generator = torch.Generator(device=args.device).manual_seed(rollout_seed)
            env.reset(random=False, context=copy.deepcopy(contexts[ci]))
            c, boxes = read_state(env, quat2euler)
            fixed_z = c[2:3].copy()
            des = c[:2].copy()
            done = False
            info = {}
            step = 0
            log_des, log_c, log_b, log_act = [], [], [], []
            log_joints = []
            log_boxes_xyz = []
            queue = []
            plan_starts = []
            while not done:
                obs = build_obs(des, c, boxes)
                plan_starts.append(not queue)
                if not queue:
                    if args.replan_noise == "fixed":
                        noise_generator.manual_seed(rollout_seed)
                    tp = time.time()
                    plan = policy.plan(obs, generator=noise_generator)
                    plan_time += time.time() - tp
                    n_plans += 1
                    queue = [plan[k, :2].astype(np.float64) for k in range(min(args.execute_steps, plan.shape[0]))]
                a = queue.pop(0)
                des = des + a
                command = np.concatenate([des, fixed_z, [0.0, 1.0, 0.0, 0.0]])
                _, _, done, info = env.step(command)
                c, boxes = read_state(env, quat2euler)
                log_des.append(des.copy()); log_c.append(c.copy()); log_act.append(a.copy())
                log_b.append(np.concatenate([boxes[0][0][:2], [boxes[0][2]], boxes[1][0][:2], [boxes[1][2]]]))
                if args.record_joints:
                    log_boxes_xyz.append(np.concatenate([b[0] for b in boxes]).copy())
                    log_joints.append(np.asarray(env.robot.current_j_pos, dtype=np.float64)[:7].copy())
                if capture:
                    capture.save(ci, ti, step)
                step += 1
            successes[ci, ti] = float(info.get("success", False))
            modes[ci, ti] = int(info.get("mode", -1))
            mean_dist[ci, ti] = float(info.get("mean_distance", np.nan))
            lengths[ci, ti] = step
            key = f"c{ci:03d}_t{ti}"
            diagnostics[key] = rollout_metrics(log_act, log_des, np.asarray(log_c)[:, :2], plan_starts)
            rollouts[key + "_plan_starts"] = np.asarray(plan_starts, dtype=bool)
            if args.record_joints:
                rollouts[key + "_goals"] = np.asarray([env.scene.get_obj_pos(g) for g in [env.target_box_1, env.target_box_2]], dtype=np.float64)
                rollouts[key + "_goal_threshold"] = np.asarray(env.target_min_dist)
                rollouts[key + "_goal_rule"] = np.asarray("any_permutation")
                rollouts[key + "_boxes_xyz"] = np.asarray(log_boxes_xyz, dtype=np.float64)
                rollouts[key + "_joints"] = np.asarray(log_joints, dtype=np.float64)
            rollouts[key + "_des"] = np.asarray(log_des, dtype=np.float32)
            rollouts[key + "_c"] = np.asarray(log_c, dtype=np.float32)
            rollouts[key + "_boxes"] = np.asarray(log_b, dtype=np.float32)
            rollouts[key + "_act"] = np.asarray(log_act, dtype=np.float32)
            print(f"context {ci:3d} traj {ti}  success {int(successes[ci, ti])}  mode {modes[ci, ti]:2d}  mean_dist {mean_dist[ci, ti]:.4f}  steps {step}  elapsed {time.time() - t0:.0f}s", flush=True)

    n_modes = 4
    mode_probs = np.zeros((n_ctx, n_modes))
    for ci in range(n_ctx):
        for m in range(n_modes):
            mode_probs[ci, m] = np.sum((modes[ci] == m) & (successes[ci] == 1)) / args.n_traj
    mode_probs = mode_probs / (mode_probs.sum(1, keepdims=True) + 1e-12)
    entropy = float(-(mode_probs * np.log(mode_probs + 1e-12) / np.log(n_modes)).sum(1).mean())

    results = {
        "run_dir": run_dir,
        "checkpoint": args.ckpt,
        "checkpoint_step": ckpt_step,
        "weights": args.weights,
        "caps": cfg.get("caps", {"enabled": False}),
        "smoothness": summarize_metrics(diagnostics),
        "contexts": args.contexts,
        "n_contexts": n_ctx,
        "n_traj": args.n_traj,
        "ode_steps": args.ode_steps,
        "execute_steps": args.execute_steps,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "noise_protocol": "per_rollout_v1" if args.paired_noise else "legacy_global",
        "replan_noise": args.replan_noise,
        "success_rate": float(successes.mean()),
        "entropy": entropy,
        "score": 0.5 * (float(successes.mean()) + entropy),
        "mean_distance": float(np.nanmean(mean_dist)),
        "mean_steps": float(lengths.mean()),
        "mode_counts": {str(m): int(np.sum((modes == m) & (successes == 1))) for m in range(n_modes)},
        "plan_ms": 1000.0 * plan_time / max(n_plans, 1),
        "wall_sec": time.time() - t0,
    }
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(out_dir, "smoothness_per_rollout.json"), "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2, allow_nan=False)
    if capture:
        capture.close()
    np.savez_compressed(os.path.join(out_dir, "rollouts.npz"), successes=successes, modes=modes, mean_distance=mean_dist, lengths=lengths, **rollouts)
    print(json.dumps(results, indent=2))
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
