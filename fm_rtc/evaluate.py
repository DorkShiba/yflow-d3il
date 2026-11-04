"""Run baseline or baseline+RTC against the same D3IL contexts and EMA weights."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch

from .execution import ChunkController
from .metrics import trajectory_metrics
from .policy import RTCPolicy

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "fm_baseline"))
from eval_pushing import load_model
from train import resolve_data_dir, resolve_d3il, resolve_hardflow


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def revision(path):
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=["pushing", "inserting"], required=True)
    p.add_argument("--method", choices=["baseline", "rtc"], required=True)
    p.add_argument("--output", required=True, help="New output directory (never overwritten)")
    p.add_argument("--run-dir")
    p.add_argument("--hardflow-root")
    p.add_argument("--d3il-root")
    p.add_argument("--data-dir")
    p.add_argument("--contexts-file", default=str(REPO / "fm_baseline/contexts/inserting_val.pkl"))
    p.add_argument("--raw-root", default="/root/d3il-work/raw/inserting")
    p.add_argument("--ckpt", default="baseline_ema.pt")
    p.add_argument("--n-contexts", type=int, default=None)
    p.add_argument("--n-traj", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ode-steps", type=int, default=10)
    p.add_argument("--execute-steps", type=int, default=8)
    p.add_argument("--delay-steps", type=int, default=0,
                   help="Fixed simulated inference delay; 0 reproduces the baseline schedule")
    p.add_argument("--beta", type=float, default=5.0)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return p.parse_args(argv)


def episode(env, api, policy, context, args, generator, dt):
    env.reset(random=False, context=copy.deepcopy(context))
    c, boxes = api.read_state(env, api.quat2euler)
    des, z = c[:2].copy(), c[2:3].copy()
    initial_des, initial_c = des.copy(), c[:2].copy()
    control = ChunkController(policy, args.execute_steps, args.delay_steps,
                              args.method == "rtc", generator)
    desired, actual, actions, seams, box_log = [], [], [], [], []
    info = {}
    for _ in range(args.max_steps):
        obs = api.build_obs(des, c, boxes)
        action, seam = control.action(obs)
        action = action.astype(np.float64)
        des = des + action
        _, _, done, info = env.step(np.concatenate((des, z, [0., 1., 0., 0.])))
        c, boxes = api.read_state(env, api.quat2euler)
        desired.append(des.copy()); actual.append(c.copy())
        actions.append(action); seams.append(seam)
        box_log.append(np.concatenate([np.r_[b[0][:2], b[2]] for b in boxes]))
        if done:
            break
    desired, actual = np.asarray(desired), np.asarray(actual)
    metrics = trajectory_metrics(desired, actual, initial_des, initial_c, seams, dt)
    metrics.update(success=float(info.get("success", False)), mode=int(info.get("mode", -1)),
                   mean_distance=float(info.get("mean_distance", float("nan"))), steps=len(actions),
                   plan_ms=float(np.mean(control.plan_ms)), n_plans=len(control.plan_ms))
    arrays = dict(des=desired, c=actual, act=np.asarray(actions), boxes=np.asarray(box_log),
                  seams=np.asarray(seams), initial_des=initial_des, initial_c=initial_c,
                  plan_ms=np.asarray(control.plan_ms))
    return metrics, arrays


def main(argv=None):
    args = parse_args(argv)
    args.n_contexts = args.n_contexts if args.n_contexts is not None else (30 if args.task == "pushing" else 82)
    args.max_steps = args.max_steps if args.max_steps is not None else (400 if args.task == "pushing" else 2000)
    if min(args.n_contexts, args.n_traj, args.max_steps) < 1:
        raise ValueError("Context, trajectory and step counts must be positive")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    run = Path(args.run_dir or REPO / "fm_baseline/results" / (args.task + "_baseline")).resolve()
    cfg0 = json.loads((run / "config.json").read_text(encoding="utf-8"))
    if cfg0["task"] != args.task:
        raise ValueError("Checkpoint task mismatch")
    hf = str(Path(resolve_hardflow(args.hardflow_root, cfg0)).resolve())
    d3il = str(Path(resolve_d3il(args.d3il_root)).resolve())
    data = str(Path(resolve_data_dir(args.data_dir, cfg0)).resolve())
    contexts_file = str(Path(args.contexts_file).resolve())
    raw_root = str(Path(args.raw_root).resolve())
    cfg, model, norm, _, _, step = load_model(str(run), args.ckpt, "ema", hf, args.device)
    cfg["data_dir"] = data
    policy = RTCPolicy(model, norm, cfg, args.ode_steps, args.device, args.beta)
    # Validate delay/schedule before constructing the simulator.
    ChunkController(policy, args.execute_steps, args.delay_steps)
    if args.task == "pushing":
        import eval_pushing as api
        Env, api.quat2euler = api.setup_d3il(d3il)
        source = Path(d3il) / "environments/dataset/data/pushing/test_contexts.pkl"
        contexts = list(np.load(source, allow_pickle=True))
        names = [str(i) for i in range(len(contexts))]
    else:
        import eval_inserting as api
        Env, api.quat2euler = api.setup_d3il(d3il)
        source = Path(contexts_file)
        items = api.load_contexts(cfg, "val", raw_root, api.quat2euler, 0, contexts_file)
        contexts, names = [it[2] for it in items], [it[0] for it in items]
    if args.n_contexts > len(contexts):
        raise ValueError("Requested more contexts than available")
    contexts, names = contexts[:args.n_contexts], names[:args.n_contexts]
    env = Env(render=False, n_substeps=35, max_steps_per_episode=args.max_steps)
    env.start()
    dt = float(env.scene.dt) * int(env.n_substeps)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Invalid control dt")
    encoded = json.dumps(contexts, default=lambda x: np.asarray(x).tolist(), sort_keys=True).encode()
    protocol = dict(task=args.task, checkpoint_sha256=sha(run / args.ckpt),
                    normalizer_sha256=sha(run / "normalizer.npz"), config_sha256=sha(run / "config.json"),
                    checkpoint_step=step, context_sha256=hashlib.sha256(encoded).hexdigest(),
                    n_contexts=len(contexts), n_traj=args.n_traj, seed=args.seed,
                    ode_steps=args.ode_steps, execute_steps=args.execute_steps, delay_steps=args.delay_steps,
                    max_steps=args.max_steps, beta=args.beta, control_dt_s=dt,
                    rng="SeedSequence(seed,context,trajectory); per-plan torch noise",
                    guidance_channels="normalized action channels only",
                    hardflow_commit=revision(hf), d3il_commit=revision(d3il), repo_commit=revision(REPO),
                    source_sha256={p.name: sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
                    baseline_source_sha256={p.name: sha(p) for p in sorted((REPO / "fm_baseline").glob("*.py"))})
    output.mkdir(parents=True)
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    rows, rollouts = [], {}
    started = time.perf_counter()
    for ci, context in enumerate(contexts):
        for ti in range(args.n_traj):
            seed = int(np.random.SeedSequence([args.seed, ci, ti]).generate_state(1)[0])
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); env.seed(seed)
            generator = torch.Generator(device=args.device).manual_seed(seed)
            row, arrays = episode(env, api, policy, context, args, generator, dt)
            row.update(context=ci, trajectory=ti, context_name=names[ci])
            rows.append(row)
            key = f"c{ci:03d}_t{ti}"
            rollouts.update({key + "_" + k: v for k, v in arrays.items()})
            with (output / "episodes.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(f"{args.method} {key}: success={int(row['success'])} steps={row['steps']} "
                  f"actual_jerk={row['actual_jerk_rms_m_s3']} elapsed={time.perf_counter()-started:.0f}s", flush=True)
    n_modes = 4 if args.task == "pushing" else 6
    first_mode = 0 if args.task == "pushing" else 1
    entropies = []
    for ci in range(len(contexts)):
        counts = np.asarray([sum(r["success"] == 1 and r["mode"] == m and r["context"] == ci for r in rows)
                             for m in range(first_mode, first_mode+n_modes)], dtype=float)
        probs = counts / max(counts.sum(), 1)
        entropies.append(float(-(probs * np.log(probs + 1e-12)).sum() / np.log(n_modes)))
    success = float(np.mean([r["success"] for r in rows]))
    summary = {k: float(np.mean([r[k] for r in rows if r[k] is not None and np.isfinite(r[k])]))
               for k in rows[0] if k not in ("context", "trajectory", "context_name", "mode")
               and any(r[k] is not None and np.isfinite(r[k]) for r in rows)}
    result = dict(method=args.method, protocol=protocol, success_rate=success, entropy=float(np.mean(entropies)),
                  score=(success + float(np.mean(entropies)))/2, metrics=summary,
                  episodes=rows, wall_sec=time.perf_counter()-started,
                  torch_version=torch.__version__, device=args.device)
    np.savez_compressed(output / "rollouts.npz", **rollouts)
    (output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {output}; success_rate={success:.4f}", flush=True)


if __name__ == "__main__":
    main()
