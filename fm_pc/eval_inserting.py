import argparse
import copy
import glob
import importlib
import importlib.abc
import importlib.util
import json
import os
import pickle
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(os.path.dirname(HERE), "fm_baseline")   # unchanged baseline: train.py, contexts/
sys.path.insert(0, BASE)
sys.path.insert(0, HERE)
from eval_pushing import Policy, load_model
from constraints import add_constraint_args, build_policy_from_args, tag_suffix
from train import resolve_d3il, resolve_data_dir, resolve_hardflow

CONTEXTS_FILE = os.path.join(BASE, "contexts", "inserting_val.pkl")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--ckpt", default="last.pt")
    p.add_argument("--weights", choices=["ema", "model"], default="ema")
    p.add_argument("--d3il-root", default=None)
    p.add_argument("--hardflow-root", default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--raw-root", default="/root/d3il-work/raw/inserting")
    p.add_argument("--contexts-file", default=CONTEXTS_FILE)
    p.add_argument("--export-contexts", default=None)
    p.add_argument("--split", choices=["val", "train"], default="val")
    p.add_argument("--n-contexts", type=int, default=0)
    p.add_argument("--n-traj", type=int, default=1)
    p.add_argument("--ode-steps", type=int, default=10)
    p.add_argument("--execute-steps", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--check-obs", type=int, default=0)
    p.add_argument("--tag", default="")
    add_constraint_args(p)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


class D3ilSimAlias(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path, target=None):
        if name == "d3il_sim" or name.startswith("d3il_sim."):
            return importlib.util.spec_from_loader(name, self)
        return None

    def create_module(self, spec):
        return importlib.import_module("environments.d3il." + spec.name)

    def exec_module(self, module):
        pass


def setup_d3il(root):
    os.chdir(root)
    for p in [root, os.path.join(root, "environments", "d3il")]:
        if p not in sys.path:
            sys.path.insert(0, p)
    importlib.import_module("environments.d3il.d3il_sim")
    sys.meta_path.insert(0, D3ilSimAlias())
    from envs.gym_inserting_env.gym_inserting.envs.gate_insertion import Gate_Insertion_Env
    from d3il_sim.utils.geometric_transformation import quat2euler
    return Gate_Insertion_Env, quat2euler


def read_state(env, quat2euler):
    env.robot.receiveState()
    c = np.asarray(env.robot.current_c_pos, dtype=np.float64).copy()
    boxes = []
    for b in [env.push_box1, env.push_box2, env.push_box3]:
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


def find_raw(raw_root, name):
    hits = glob.glob(os.path.join(raw_root, "**", name), recursive=True)
    if not hits:
        raise FileNotFoundError(f"{name} not found under {raw_root}")
    return sorted(hits)[0]


def context_from_raw(data, quat2euler):
    if "context" in data:
        return copy.deepcopy(data["context"])
    ctx = []
    for key in ["box-1", "box-2", "box-3"]:
        pos = np.asarray(data[key]["pos"][0], dtype=np.float64)
        quat = np.asarray(data[key]["quat"][0], dtype=np.float64)
        deg = float(np.degrees(np.asarray(quat2euler(quat)).reshape(-1)[-1]))
        ctx += [np.array([pos[0], pos[1], deg]), quat]
    return ctx


def slim(data):
    return {"robot": {"des_c_pos": np.asarray(data["robot"]["des_c_pos"]), "j_pos": np.asarray(data["robot"]["j_pos"][:1])}}


def load_contexts(cfg, split, raw_root, quat2euler, n, contexts_file=None):
    split_dir = os.path.join(cfg["data_dir"], split)
    files = sorted(f for f in os.listdir(split_dir) if f.endswith(".npz"))
    if n > 0:
        files = files[:n]
    stored = {}
    if split == "val" and contexts_file and os.path.exists(contexts_file):
        with open(contexts_file, "rb") as fh:
            stored = pickle.load(fh)
    out = []
    for f in files:
        if f in stored:
            data, ctx = stored[f]["data"], stored[f]["context"]
        else:
            with open(find_raw(raw_root, f[:-4] + ".pkl"), "rb") as fh:
                data = pickle.load(fh)
            ctx = context_from_raw(data, quat2euler)
        out.append((f, data, ctx, os.path.join(split_dir, f)))
    return out


def main():
    args = parse_args()
    run_dir = os.path.abspath(args.run_dir)
    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as f:
        cfg0 = json.load(f)
    if cfg0["task"] != "inserting":
        raise SystemExit("This script evaluates inserting runs only")
    run_cwd = os.getcwd()
    cfg, model, norm, apply_conditioning, Solver, ckpt_step = load_model(run_dir, args.ckpt, args.weights, resolve_hardflow(args.hardflow_root, cfg0), args.device)
    cfg["data_dir"] = resolve_data_dir(args.data_dir, cfg)

    Env, quat2euler = setup_d3il(resolve_d3il(args.d3il_root))

    if args.export_contexts:
        items = load_contexts(cfg, "val", args.raw_root, quat2euler, 0, None)
        out = {f: {"context": ctx, "data": slim(data)} for f, data, ctx, _ in items}
        path = os.path.join(run_cwd, args.export_contexts) if not os.path.isabs(args.export_contexts) else args.export_contexts
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(out, fh)
        print(f"exported {len(out)} contexts to {path}")
        return

    env = Env(render=False, n_substeps=35, max_steps_per_episode=args.max_steps)
    env.start()

    if args.check_obs > 0:
        items = load_contexts(cfg, "val", args.raw_root, quat2euler, args.check_obs, args.contexts_file)
        errs = []
        for f, data, ctx, npz in items:
            env.reset(random=False, context=ctx)
            env.robot.beam_to_joint_pos(np.asarray(data["robot"]["j_pos"][0]).copy())
            env.scene.next_step(log=False)
            c, boxes = read_state(env, quat2euler)
            ours = build_obs(np.asarray(data["robot"]["des_c_pos"][0]), c, boxes)
            ref = np.load(npz)["obs"][0]
            e = np.abs(ours - ref)
            errs.append(e)
            print(f"check {f}  context_key {'context' in data}  max abs err {e.max():.2e}  per-dim {np.round(e, 4).tolist()}", flush=True)
        errs = np.stack(errs)
        print(f"check summary  max {errs.max():.2e}  mean per-dim {np.round(errs.mean(0), 5).tolist()}", flush=True)
        return

    items = load_contexts(cfg, args.split, args.raw_root, quat2euler, args.n_contexts, args.contexts_file)
    n_ctx = len(items)
    policy = Policy(model, norm, cfg, apply_conditioning, Solver, args.ode_steps, args.device)
    policy, constraint_info = build_policy_from_args(args, policy)
    args.tag = (args.tag or '') + tag_suffix(args)
    print('constraint injection:', json.dumps(constraint_info), flush=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = os.path.join(run_dir, f"eval_inserting_{args.split}_{args.tag + '_' if args.tag else ''}{stamp}")
    os.makedirs(out_dir)

    successes = np.zeros((n_ctx, args.n_traj))
    modes = np.zeros((n_ctx, args.n_traj), dtype=int)
    n_boxes = np.zeros((n_ctx, args.n_traj), dtype=int)
    mean_dist = np.zeros((n_ctx, args.n_traj))
    lengths = np.zeros((n_ctx, args.n_traj), dtype=int)
    demo_len = np.zeros(n_ctx, dtype=int)
    rollouts = {}
    t0 = time.time()
    plan_time = 0.0
    n_plans = 0

    for ci, (fname, data, ctx, _) in enumerate(items):
        demo_len[ci] = len(data["robot"]["des_c_pos"])
        for ti in range(args.n_traj):
            env.reset(random=False, context=copy.deepcopy(ctx))
            c, boxes = read_state(env, quat2euler)
            fixed_z = c[2:3].copy()
            des = c[:2].copy()
            done = False
            info = {}
            step = 0
            log_des, log_c, log_b, log_act = [], [], [], []
            queue = []
            while not done:
                obs = build_obs(des, c, boxes)
                if not queue:
                    tp = time.time()
                    plan = policy.plan(obs)
                    plan_time += time.time() - tp
                    n_plans += 1
                    queue = [plan[k, :2].astype(np.float64) for k in range(min(args.execute_steps, plan.shape[0]))]
                a = queue.pop(0)
                des = des + a
                command = np.concatenate([des, fixed_z, [0.0, 1.0, 0.0, 0.0]])
                _, _, done, info = env.step(command)
                c, boxes = read_state(env, quat2euler)
                log_des.append(des.copy()); log_c.append(c.copy()); log_act.append(a.copy())
                log_b.append(np.concatenate([np.r_[bx[0][:2], bx[2]] for bx in boxes]))
                step += 1
            successes[ci, ti] = float(info.get("success", False))
            modes[ci, ti] = int(info.get("mode", 0))
            n_boxes[ci, ti] = int(info.get("one_box_success", 0)) + int(info.get("two_box_success", 0)) + int(info.get("three_box_success", 0))
            mean_dist[ci, ti] = float(info.get("mean_distance", np.nan))
            lengths[ci, ti] = step
            key = f"c{ci:03d}_t{ti}"
            rollouts[key + "_des"] = np.asarray(log_des, dtype=np.float32)
            rollouts[key + "_c"] = np.asarray(log_c, dtype=np.float32)
            rollouts[key + "_boxes"] = np.asarray(log_b, dtype=np.float32)
            rollouts[key + "_act"] = np.asarray(log_act, dtype=np.float32)
            print(f"context {ci:3d} {fname} traj {ti}  success {int(successes[ci, ti])}  boxes {n_boxes[ci, ti]}  mode {modes[ci, ti]}  mean_dist {mean_dist[ci, ti]:.4f}  steps {step} (demo {demo_len[ci]})  elapsed {time.time() - t0:.0f}s", flush=True)

    n_modes = 6
    entropy = None
    if args.n_traj > 1:
        probs = np.zeros((n_ctx, n_modes))
        for ci in range(n_ctx):
            for m in range(n_modes):
                probs[ci, m] = np.sum((modes[ci] == m + 1) & (successes[ci] == 1)) / args.n_traj
        probs = probs / (probs.sum(1, keepdims=True) + 1e-12)
        entropy = float(-(probs * np.log(probs + 1e-12) / np.log(n_modes)).sum(1).mean())

    results = {
        "run_dir": run_dir,
        "checkpoint": args.ckpt,
        "checkpoint_step": ckpt_step,
        "weights": args.weights,
        "contexts": f"{args.split}_split_initial_states",
        "n_contexts": n_ctx,
        "n_traj": args.n_traj,
        "ode_steps": args.ode_steps,
        "execute_steps": args.execute_steps,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "success_rate": float(successes.mean()),
        "one_box_rate": float((n_boxes >= 1).mean()),
        "two_box_rate": float((n_boxes >= 2).mean()),
        "three_box_rate": float((n_boxes >= 3).mean()),
        "entropy": entropy,
        "mean_distance": float(np.nanmean(mean_dist)),
        "mean_steps": float(lengths.mean()),
        "mean_steps_success": float(lengths[successes == 1].mean()) if successes.any() else None,
        "mean_demo_len": float(demo_len.mean()),
        "mode_counts": {str(m): int(np.sum((modes == m) & (successes == 1))) for m in range(1, n_modes + 1)},
        "plan_ms": 1000.0 * plan_time / max(n_plans, 1),
        "sampler": args.sampler,
        "pc_params": args.pc_params if args.sampler == "pc" else None,
        "constraints": args.constraints if args.sampler != "euler" else None,
        "constraint_info": constraint_info,
        "constraint_params": args.constraint_params,
        "wall_sec": time.time() - t0,
        "context_files": [it[0] for it in items],
    }
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    np.savez_compressed(os.path.join(out_dir, "rollouts.npz"), successes=successes, modes=modes, n_boxes=n_boxes, mean_distance=mean_dist, lengths=lengths, demo_len=demo_len, **rollouts)
    print(json.dumps({k: v for k, v in results.items() if k != "context_files"}, indent=2))
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
