"""Offline check of the constraint-injection wrapper (no simulator needed).

1. inject=none through the wrapper == baseline ConditionedODESolver, bit-for-bit.
2. Plans from validation observations: violation, change of the executed
   actions and time, for none, pc and pc-noCBF (same sampler, boxes so wide
   that the CBF never acts: separates the constraint from the sampler).

    python fm_pc/check_constraints.py --run-dir fm_baseline/results/pushing_baseline --constraints action_box,tracking
"""
import argparse
import json
import os
import sys
import time  # noqa: F401

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(os.path.dirname(HERE), "fm_baseline")   # unchanged baseline: train.py, contexts/
sys.path.insert(0, BASE)
sys.path.insert(0, HERE)
from eval_pushing import Policy, load_model
from train import find_split_files, resolve_data_dir, resolve_hardflow
from constraints import (ConstrainedPolicy, build_pc_sampler, build_set, cumulative_des,
                         default_params, euler_solve, total_violation)
from constraints.sets import ACT


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--ckpt", default="baseline_ema.pt")
    p.add_argument("--hardflow-root", default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--n-obs", type=int, default=64)
    p.add_argument("--ode-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--constraints", default="action_box,workspace,tracking")
    p.add_argument("--constraint-params", default="{}")
    p.add_argument("--pc-params", default="{}", help="JSON overrides for method ② (alpha, eps, rho, delta, tp, tc)")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    with open(os.path.join(args.run_dir, "config.json"), encoding="utf-8") as f:
        cfg0 = json.load(f)
    hardflow_root = resolve_hardflow(args.hardflow_root, cfg0)
    cfg, model, norm, apply_conditioning, Solver, step = load_model(args.run_dir, args.ckpt, "ema", hardflow_root, args.device)
    cfg["data_dir"] = resolve_data_dir(args.data_dir, cfg)
    base = Policy(model, norm, cfg, apply_conditioning, Solver, args.ode_steps, args.device)
    params = default_params(norm, cfg["act_dim"])
    params.update(json.loads(args.constraint_params))
    names = [s for s in args.constraints.split(",") if s]
    print("params:", json.dumps(params))

    # observations from validation windows
    files = find_split_files(cfg["data_dir"], "val")
    rng = np.random.RandomState(args.seed)
    obs_list, gt_list = [], []
    for f in rng.choice(files, size=min(args.n_obs, len(files)), replace=False):
        z = np.load(f)
        i = rng.randint(len(z["obs"]) - 8)
        obs_list.append(z["obs"][i])
        gt_list.append(z["actions"][i:i + 8])          # demo actions for the 8 executed steps
    obs = np.stack(obs_list).astype(np.float32)
    gt = np.stack(gt_list).astype(np.float64)

    # ---- 1. bit-for-bit equivalence with the baseline solver ----------------
    o = torch.from_numpy(obs).to(args.device)
    o_n = (o - base.center[base.act_dim:]) / base.scale[base.act_dim:]
    cond = {0: o_n}
    torch.manual_seed(args.seed)
    x0 = torch.randn(len(obs), base.horizon, base.obs_dim + base.act_dim, device=args.device)
    x0 = apply_conditioning(x0, cond, base.act_dim)
    solver = Solver(model, cond, action_dim=base.act_dim, value=None, ode_method="euler")
    ref = solver(x0.clone(), t_span=base.t_span)
    ours = euler_solve(model, x0.clone(), cond, base.act_dim, base.t_span, apply_conditioning, None)
    diff = float((ref - ours).abs().max())
    print(f"[1] wrapper(none) vs baseline solver  max|diff| = {diff:.3e}  ->", "PASS" if diff == 0.0 else "FAIL")

    cset = build_set(names, params)

    # ---- 2. end-to-end plans ------------------------------------------------
    print("[2] plans from %d validation observations   (RMSE: executed 8 actions vs the demo, in units of a_max)" % len(obs))
    pcp = json.loads(args.pc_params)
    wide = {**params, "a_max": 1e3, "track_d": 1e3, "r": 1e3, "ws_lo": [-1e3, -1e3], "ws_hi": [1e3, 1e3]}
    configs = [("none", None), ("pc", params), ("pc-noCBF", wide)]
    results = {}
    for label, p_set in configs:
        smp = build_pc_sampler(names, p_set, pcp, norm, base.act_dim, args.device) if p_set else None
        use_pc = smp is not None
        pol = ConstrainedPolicy(base, sampler=smp)
        torch.manual_seed(args.seed)
        plans, stats, t0 = [], [], time.perf_counter()
        for ob in obs:
            plans.append(pol.plan(ob))
            stats.append(pol.last_stats)
        wall = (time.perf_counter() - t0) / len(obs)
        P = torch.from_numpy(np.stack(plans)).to(args.device)
        viol = total_violation(cset, P)
        results[label] = P
        Pn = P[:, :8, ACT].cpu().double().numpy()
        d_act = float(np.abs(Pn - results["none"][:, :8, ACT].cpu().double().numpy()).mean()) if label != "none" else 0.0
        rmse = float(np.sqrt(((Pn - gt) ** 2).mean()) / params["a_max"])
        qm = np.mean([s.get("pc_qp_ms_per_call", 0) for s in stats])
        act = np.mean([s.get("pc_active_frac", 0) for s in stats])
        extra = f"  pc-QP {qm:5.2f} ms/call, active {100 * act:4.1f}%" if use_pc else ""
        print(f"    {label:9s} plan {1000 * wall:6.1f} ms{extra}  RMSE {rmse:.3f}  "
              f"mean|Δa| {d_act:.2e}  viol: " + ", ".join(f"{k}={v:.4f}" for k, v in viol.items() if k.endswith("/max"))
              + "  frac: " + ", ".join(f"{k.split('/')[0]}={v:.3f}" for k, v in viol.items() if k.endswith("/frac_steps")))
    if smp is not None:
        print(f"    pc settings: {json.dumps(smp.describe())}  eps used {smp.eps_used:.3f}, T^c {smp.tc_used}")
    des = cumulative_des(results["pc"])
    print(f"    pc: desired-hand span of plans  x[{float(des[..., 0].min()):.3f},{float(des[..., 0].max()):.3f}]"
          f"  y[{float(des[..., 1].min()):.3f},{float(des[..., 1].max()):.3f}]")

if __name__ == "__main__":
    main()
