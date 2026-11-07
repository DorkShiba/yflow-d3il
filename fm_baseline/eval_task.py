"""Baseline evaluation for any D3IL task in ``fm_constraint/tasks.py`` (in-distribution only).

    python fm_baseline/eval_task.py --task aligning --run-dir <run> --ckpt baseline_ema.pt --n-traj 4

Pushing and Inserting have their own evaluators (``eval_pushing.py``, ``eval_inserting.py``,
``python -m fm_constraint.eval_*`` with ``--ood``). This script keeps the other tasks
(avoiding, aligning, sorting_2/4/6, stacking) runnable: plan an H-step chunk from the latest
state, execute ``--execute-steps`` actions, re-plan. Outputs results.json and rollouts.npz.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from eval_pushing import Policy, load_model  # noqa: E402
from fm_constraint.tasks import TASKS, get_task  # noqa: E402
from train import resolve_d3il, resolve_data_dir  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=sorted(TASKS))
    p.add_argument("--run-dir", required=True)
    p.add_argument("--ckpt", default="baseline_ema.pt")
    p.add_argument("--weights", choices=["ema", "model"], default="ema")
    p.add_argument("--d3il-root", default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--contexts", choices=["test", "train"], default="test", help="official context file (tasks that have one)")
    p.add_argument("--contexts-file", default=None, help="override the task's context file")
    p.add_argument("--n-contexts", type=int, default=0, help="0 = all")
    p.add_argument("--n-traj", type=int, default=1)
    p.add_argument("--ode-steps", type=int, default=10)
    p.add_argument("--execute-steps", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=None, help="default: the task's episode limit")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def run_episode(env, policy, spec, ctx, execute_steps):
    spec.reset(env, ctx)
    s = spec.read(env)
    des, fixed = spec.init_des(env, s)
    done, info, step, queue = False, {}, 0, []
    log = {"des": [], "c": [], "b": [], "act": [], "seams": []}
    plan_s, n_plans = 0.0, 0
    while not done:
        if not queue:
            started = time.time()
            plan = policy.plan(spec.obs(des, s))
            plan_s += time.time() - started
            n_plans += 1
            queue = [plan[k, :spec.act_dim].astype(np.float64) for k in range(min(execute_steps, plan.shape[0]))]
            log["seams"].append(step)
        a = queue.pop(0)
        des = spec.apply(des, a)
        _, _, done, info = env.step(spec.command(des, fixed))
        s = spec.read(env)
        log["des"].append(np.asarray(des, dtype=np.float64).copy())
        log["c"].append(s.c.copy())
        log["act"].append(a.copy())
        log["b"].append(spec.log_state(s))
        step += 1
    return info, step, log, plan_s, n_plans


def main():
    args = parse_args()
    run_dir = os.path.abspath(args.run_dir)
    if args.contexts_file:
        args.contexts_file = os.path.abspath(args.contexts_file)
    spec = get_task(args.task)
    cfg, model, norm, apply_conditioning, Solver, ckpt_step = load_model(run_dir, args.ckpt, args.weights, args.device)
    if cfg["task"] != args.task:
        raise SystemExit(f"run dir is a '{cfg['task']}' model, not '{args.task}'")
    if (cfg["obs_dim"], cfg["act_dim"]) != (spec.obs_dim, spec.act_dim):
        raise SystemExit(f"checkpoint dims {(cfg['obs_dim'], cfg['act_dim'])} != task {(spec.obs_dim, spec.act_dim)}")
    cfg["data_dir"] = resolve_data_dir(args.data_dir, cfg)
    args.d3il_root = resolve_d3il(args.d3il_root)
    max_steps = args.max_steps or spec.max_steps
    spec.setup(args.d3il_root)
    env = spec.make_env(max_steps)
    contexts = spec.contexts(args, cfg)
    policy = Policy(model, norm, cfg, apply_conditioning, Solver, args.ode_steps, args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = os.path.join(run_dir, f"eval_{args.task}_{args.tag + '_' if args.tag else ''}{stamp}")
    os.makedirs(out_dir)

    n_ctx = len(contexts)
    outs, lengths, rollouts = {}, np.zeros((n_ctx, args.n_traj), dtype=int), {}
    t0, plan_time, n_plans = time.time(), 0.0, 0
    for ci, (name, ctx) in enumerate(contexts):
        for ti in range(args.n_traj):
            info, step, log, ps, npl = run_episode(env, policy, spec, ctx, args.execute_steps)
            plan_time += ps
            n_plans += npl
            outcome = spec.outcome(info)
            for k, v in outcome.items():
                outs.setdefault(k, np.full((n_ctx, args.n_traj), np.nan))[ci, ti] = v
            lengths[ci, ti] = step
            key = f"c{ci:03d}_t{ti}"
            rollouts.update({f"{key}_{k}": np.asarray(v, dtype=np.int32 if k == "seams" else np.float32)
                             for k, v in log.items()})
            print(f"context {ci:3d} {name} traj {ti}  " + "  ".join(f"{k} {v:g}" for k, v in outcome.items())
                  + f"  steps {step}  elapsed {time.time() - t0:.0f}s", flush=True)

    results = {"task": args.task, "run_dir": run_dir, "checkpoint": args.ckpt, "checkpoint_step": ckpt_step,
               "weights": args.weights, "contexts": [c for c, _ in contexts], "n_contexts": n_ctx, "n_traj": args.n_traj,
               "ode_steps": args.ode_steps, "execute_steps": args.execute_steps, "max_steps": max_steps,
               "seed": args.seed, **spec.summarize(outs), "mean_steps": float(lengths.mean()),
               "plan_ms": 1000.0 * plan_time / max(n_plans, 1), "wall_sec": time.time() - t0}
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=float)
    np.savez_compressed(os.path.join(out_dir, "rollouts.npz"), successes=outs["success"],
                        **{k + "s" if k == "mode" else k: v for k, v in outs.items() if k != "success"},
                        lengths=lengths, **rollouts)
    print(json.dumps({k: v for k, v in results.items() if k != "contexts"}, indent=2, default=float))
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
