"""Offline check of the constraint-injection wrapper (no simulator needed).

1. inject=none through the wrapper == baseline ConditionedODESolver, bit-for-bit.
2. Each projection is exact: output feasible, idempotent, and (for the ADMM
   sets) matches a brute-force reference QP on a few samples.
3. Plans from validation observations: violation rate before/after, change of
   the executed actions, and projection time, for inject in {none, final, every}.

    python fm_projection/check_constraints.py --run-dir fm_baseline/results/pushing_baseline --constraints action_box,tracking
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(os.path.dirname(HERE), "fm_baseline")   # unchanged baseline: train.py, contexts/
sys.path.insert(0, BASE)
sys.path.insert(0, HERE)
from eval_pushing import Policy, load_model
from train import find_split_files, resolve_data_dir, resolve_hardflow
from constraints import (ConstrainedPolicy, build_injector, build_set, cumulative_des,
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
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def reference_qp(stack, x, n=4):
    """Brute-force SLSQP projection with exactly the bounds BoxStack uses."""
    from scipy.optimize import minimize
    xn = x[:n].detach().cpu().double().numpy()
    alo, ahi, slo, shi = stack.qp_bounds(xn)
    H = xn.shape[1]
    L = np.tril(np.ones((H, H)))
    out = np.zeros((xn.shape[0], H, 2))
    for b in range(xn.shape[0]):
        for d in range(2):
            a0 = xn[b, :, d]
            bnds = None if alo is None else list(zip(alo[b, :, d], ahi[b, :, d]))
            cons = [] if slo is None else [
                {"type": "ineq", "fun": lambda a, l=slo[b, :, d]: L @ a - l},
                {"type": "ineq", "fun": lambda a, h=shi[b, :, d]: h - L @ a}]
            res = minimize(lambda a: 0.5 * np.sum((a - a0) ** 2), np.zeros(H), jac=lambda a: a - a0,
                           bounds=bnds, constraints=cons, method="SLSQP", options={"ftol": 1e-16, "maxiter": 2000})
            out[b, :, d] = res.x
    return out


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

    # ---- 2. projection exactness -------------------------------------------
    cset = build_set(names, params)
    x_raw = ref * base.scale + base.center
    # make a deliberately bad sample: scale actions up 3x so everything violates
    bad = x_raw.clone()
    bad[:, :, ACT] = bad[:, :, ACT] * 3.0 + 0.004
    pr = cset.project(bad)
    pr2 = cset.project(pr)
    v_before = total_violation(cset, bad)
    v_after = total_violation(cset, pr)
    print("[2] violation before:", {k: round(v, 5) for k, v in v_before.items()})
    print("    violation after :", {k: round(v, 7) for k, v in v_after.items()},
          "(vs the original sets; > 0 only for plans that already start outside a box)")
    print(f"    idempotent max|P(P(x)) - P(x)| = {float((pr2 - pr).abs().max()):.2e}")
    print(f"    obs channels untouched: {bool(torch.equal(pr[:, :, 2:], bad[:, :, 2:]))}")
    for part in cset.parts:
        if hasattr(part, "qp_bounds"):
            try:
                ref_a = reference_qp(part, bad)
                e = float(np.abs(ref_a - part.project(bad[:4])[:, :, ACT].cpu().double().numpy()).max())
                print(f"    {part.name} ({'+'.join(x.name for x in part.a_sets + part.s_sets)}): QP vs SLSQP reference  max|diff| = {e:.2e}"
                      f"   QP bound residual after projection = {part.qp_residual(pr):.1e}")
            except ImportError:
                print("    scipy missing, reference QP skipped")
    one = bad[:1]
    cset.project(one)
    t0 = time.perf_counter()
    for _ in range(20):
        cset.project(one)
    print(f"    projection time (B=1): {1000 * (time.perf_counter() - t0) / 20:.2f} ms/call")

    # ---- 3. end-to-end plans ------------------------------------------------
    print("[3] plans from %d validation observations   (RMSE: executed 8 actions vs the demo, in units of a_max)" % len(obs))
    configs = [("none", "none"), ("final", "final"), ("every", "every")]
    results = {}
    for label, inject in configs:
        pol = ConstrainedPolicy(base, build_injector(inject, names, params, norm, base.act_dim, args.device))
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
        pm = np.mean([s.get("proj_ms_per_call", 0) for s in stats])
        print(f"    {label:9s} plan {1000 * wall:6.1f} ms  proj {pm:5.2f} ms/call  RMSE {rmse:.3f}  "
              f"mean|Δa| {d_act:.2e}  viol: " + ", ".join(f"{k}={v:.4f}" for k, v in viol.items() if k.endswith("/max"))
              + "  frac: " + ", ".join(f"{k.split('/')[0]}={v:.3f}" for k, v in viol.items() if k.endswith("/frac_steps")))
    des = cumulative_des(results["every"])
    print(f"    every: desired-hand span of plans  x[{float(des[..., 0].min()):.3f},{float(des[..., 0].max()):.3f}]"
          f"  y[{float(des[..., 1].min()):.3f},{float(des[..., 1].max()):.3f}]")

if __name__ == "__main__":
    main()
