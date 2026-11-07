"""Evaluate a state-based flow-matching policy on any D3IL task, with the OOD benchmark.

    python fm_pc/eval_task.py --task pushing --run-dir fm_baseline/results/pushing_baseline \\
        --ckpt baseline_ema.pt --n-contexts 30 --n-traj 4 --ood push --ood-push-dist 0.05

Tasks: see ``tasks.py``.  Protocol and outputs follow ``eval_pushing.py``
(results.json, rollouts.npz with ``_seams`` / ``_pobs`` / ``_event``), so
``ood_metrics.py`` scores every task the same way.  Method flags from
``constraints`` (``--sampler pc`` ..., ``--acg-scale`` ...) work for the tasks whose
action is the desired-hand XY increment (pushing, inserting, sorting_*, avoiding).
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
BASE = os.path.join(os.path.dirname(HERE), "fm_baseline")
sys.path.insert(0, BASE)
sys.path.insert(0, HERE)
from benchmark import run_task_episode
from constraints import add_constraint_args, build_policy_from_args, tag_suffix
from eval_pushing import Policy, load_model
from ood import OOD, add_ood_args, ood_tag, pack_rollout
from tasks import TASKS, get_task
from train import resolve_d3il, resolve_data_dir


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
    add_constraint_args(p)
    add_ood_args(p)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    if args.ood_file:                            # the D3IL setup changes the working directory
        args.ood_file = os.path.abspath(args.ood_file)
    run_dir = os.path.abspath(args.run_dir)
    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as f:
        cfg0 = json.load(f)
    if cfg0["task"] != args.task:
        raise SystemExit(f"run dir is a '{cfg0['task']}' model, not '{args.task}'")
    spec = get_task(args.task)
    args.d3il_root = resolve_d3il(args.d3il_root)
    cfg, model, norm, apply_conditioning, Solver, ckpt_step = load_model(run_dir, args.ckpt, args.weights, args.device)
    cfg["data_dir"] = resolve_data_dir(args.data_dir, cfg)
    if (cfg["obs_dim"], cfg["act_dim"]) != (spec.obs_dim, spec.act_dim):
        raise SystemExit(f"checkpoint dims {(cfg['obs_dim'], cfg['act_dim'])} != task {(spec.obs_dim, spec.act_dim)}")
    max_steps = args.max_steps or spec.max_steps

    spec.setup(args.d3il_root)
    env = spec.make_env(max_steps)
    contexts = spec.contexts(args, cfg)
    if args.ood_file:                            # evaluate exactly the episodes of the OOD test set
        from ood import load_ood_file
        eps = load_ood_file(args.ood_file)["episodes"]
        if args.n_contexts:
            eps = eps[:args.n_contexts]
        by_name = dict(spec.contexts(argparse.Namespace(**{**vars(args), "n_contexts": 0}), cfg))
        missing = [e["name"] for e in eps if e["name"] not in by_name]
        if missing:
            raise SystemExit(f"OOD file episodes not in the task's contexts: {missing[:3]} ...")
        contexts = [(e["name"], by_name[e["name"]]) for e in eps]
    n_ctx = len(contexts)

    policy = Policy(model, norm, cfg, apply_conditioning, Solver, args.ode_steps, args.device)
    policy, constraint_info = build_policy_from_args(args, policy)
    args.tag = (args.tag or "") + tag_suffix(args) + ood_tag(args)
    ood = OOD(args, spec, env, args.execute_steps, cfg["horizon"])
    print("constraint injection:", json.dumps(constraint_info), "| ood:", json.dumps(ood.describe()), flush=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = os.path.join(run_dir, f"eval_{args.task}_{args.tag + '_' if args.tag else ''}{stamp}")
    os.makedirs(out_dir)

    outs, lengths, rollouts = {}, np.zeros((n_ctx, args.n_traj), dtype=int), {}
    t0, plan_time, n_plans = time.time(), 0.0, 0
    for ci, (cname, ctx) in enumerate(contexts):
        for ti in range(args.n_traj):
            info, step, log, ps, npl, ev = run_task_episode(env, policy, spec, ood, ctx, ci, ti, args.execute_steps, cname)
            plan_time += ps
            n_plans += npl
            o = spec.outcome(info)
            for k, v in o.items():
                outs.setdefault(k, np.full((n_ctx, args.n_traj), np.nan))[ci, ti] = v
            lengths[ci, ti] = step
            rollouts.update(pack_rollout(f"c{ci:03d}_t{ti}", log, ev))
            print(f"context {ci:3d} {cname} traj {ti}  " + "  ".join(f"{k} {v:g}" for k, v in o.items())
                  + f"  steps {step}  elapsed {time.time() - t0:.0f}s", flush=True)

    summary = spec.summarize(outs)
    results = {
        "task": args.task, "run_dir": run_dir, "checkpoint": args.ckpt, "checkpoint_step": ckpt_step, "weights": args.weights,
        "contexts": [c for c, _ in contexts], "n_contexts": n_ctx, "n_traj": args.n_traj,
        "ode_steps": args.ode_steps, "execute_steps": args.execute_steps, "max_steps": max_steps, "seed": args.seed,
        **summary,
        "mean_steps": float(lengths.mean()),
        "plan_ms": 1000.0 * plan_time / max(n_plans, 1),
        "sampler": args.sampler, "constraint_info": constraint_info,
        "wall_sec": time.time() - t0,
        "ood": ood.describe(),
    }
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=float)
    np.savez_compressed(os.path.join(out_dir, "rollouts.npz"), successes=outs["success"],
                        **{k + "s" if k == "mode" else k: v for k, v in outs.items() if k != "success"},
                        lengths=lengths, **rollouts)
    print(json.dumps({k: v for k, v in results.items() if k not in ("contexts", "ood")}, indent=2, default=float))
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
