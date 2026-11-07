"""Offline check of temporal-mixing perturbation guidance (constraints/acg.py).

No simulator.  Plans are generated from validation observations with the SAME
initial noise for every condition (paired), and the executed part of each plan
(first --exec-steps actions) is scored.

[1] equivalence  scale 0 == v_θ, inert hooks == v_θ, the clean half of the
                 batched [x, x] call == v_θ
[2] premise      the incoherent model alone (v_θ^IC) should be MORE jittery than
                 v_θ while keeping a similar path.  If it is not, guiding away
                 from it cannot target intra-chunk jitter at that site.
[3] guidance     v = (1+λ)v_θ − λ v_θ^IC per site and λ

Metrics (units of a_max = per-axis training action range; lower is smoother)
    dA      mean ||a_t − a_{t−1}||              executed steps (2nd diff of position)
    jerk    mean ||a_t − 2a_{t−1} + a_{t−2}||    executed steps (3rd diff of position)
    bJerk   ||a_0 − 2h_{−1} + h_{−2}||           across the chunk start, h = demo history
    flips   axis sign reversals per executed chunk (|a| > a_max/4 on both sides)
    offgrid mean distance of a/a_max to the data grid {−1, 0, 1}   (action-space OOD)
    >amax   fraction of executed steps with an axis beyond a_max   (action-space OOD)
    dev     mean |cumulative path − baseline path| with the same noise, metres
    div     mean pairwise distance between the K samples of one observation, metres
    rmse    executed actions vs the demo
    gap     ||v − v_IC|| / ||v|| averaged over guided calls

    python fm_pc/check_acg.py --run-dir fm_baseline/results/pushing_baseline --ckpt baseline_ema.pt
"""
import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(os.path.dirname(HERE), "fm_baseline")
sys.path.insert(0, BASE)
sys.path.insert(0, HERE)
from eval_pushing import Policy, load_model
from train import find_split_files, resolve_data_dir, resolve_hardflow
from constraints import (ACGModel, PerturbedModel, TemporalPerturbation, default_params, euler_solve,
                         site_modules, site_resolutions)
from constraints.sets import ACT


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--ckpt", default="baseline_ema.pt")
    p.add_argument("--hardflow-root", default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--n-obs", type=int, default=64)
    p.add_argument("--n-samples", type=int, default=8, help="noise samples per observation (K)")
    p.add_argument("--ode-steps", type=int, default=10)
    p.add_argument("--exec-steps", type=int, default=8)
    p.add_argument("--sites", default="each", help="'each' = every single site, or ';'-separated groups like 'up1,final;mid'")
    p.add_argument("--modes", default="center", help="comma-separated: center,skip")
    p.add_argument("--scales", default="0.1,0.3,1")
    p.add_argument("--t-range", default="0,1", help="t0,t1: flow times where guidance is applied")
    p.add_argument("--no-ic", action="store_true", help="skip the incoherent-model-alone rows")
    p.add_argument("--obs-noise", type=float, default=0.0,
                   help="Gaussian std added to the conditioned observation in normalised units ([-1, 1] range): "
                        "pushes the condition off the data to probe OOD behaviour")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="write all numbers to this JSON file")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_windows(cfg, n_obs, exec_steps, seed):
    files = find_split_files(cfg["data_dir"], "val")
    rng = np.random.RandomState(seed)
    obs, gt, hist = [], [], []
    for f in rng.choice(files, size=min(n_obs, len(files)), replace=False):
        z = np.load(f)
        i = rng.randint(2, len(z["obs"]) - exec_steps)
        obs.append(z["obs"][i])
        gt.append(z["actions"][i:i + exec_steps])
        hist.append(z["actions"][i - 2:i])          # executed history before the chunk (demo)
    return (np.stack(obs).astype(np.float32), np.stack(gt).astype(np.float64), np.stack(hist).astype(np.float64))


@torch.no_grad()
def generate(field, base, x0, cond, apply_conditioning):
    x = euler_solve(field, x0.clone(), cond, base.act_dim, base.t_span, apply_conditioning, None)
    return (x * base.scale + base.center)[:, :, ACT].cpu().double().numpy()


def metrics(a, a_ref, gt, hist, a_max, n_obs, k, e):
    """a: (N*K, H, 2) raw actions; executed part = first e steps."""
    a = a.reshape(n_obs, k, -1, 2)
    ref = a_ref.reshape(n_obs, k, -1, 2)
    ex = a[:, :, :e]
    d1 = np.linalg.norm(np.diff(ex, axis=2), axis=-1)
    d2 = np.linalg.norm(np.diff(ex, n=2, axis=2), axis=-1)
    h = hist[:, None]                                                    # (N, 1, 2, 2)
    bj = np.linalg.norm(ex[:, :, 0] - 2 * h[:, :, 1] + h[:, :, 0], axis=-1)
    big = np.abs(ex) > a_max / 4
    flips = ((ex[:, :, 1:] * ex[:, :, :-1] < 0) & big[:, :, 1:] & big[:, :, :-1]).sum(axis=(2, 3))
    u = ex / a_max
    offgrid = np.min(np.stack([np.abs(u + 1), np.abs(u), np.abs(u - 1)]), axis=0).mean()
    amax = (np.abs(ex) > a_max * (1 + 1e-4)).any(axis=-1).mean()
    path, path_ref = np.cumsum(ex, axis=2), np.cumsum(ref[:, :, :e], axis=2)
    dev = np.linalg.norm(path - path_ref, axis=-1).mean()
    flat = path.reshape(n_obs, k, -1)
    div = np.mean([np.linalg.norm(flat[:, i] - flat[:, j], axis=-1).mean() / np.sqrt(e)
                   for i, j in itertools.combinations(range(k), 2)]) if k > 1 else 0.0
    rmse = np.sqrt(((ex - gt[:, None]) ** 2).mean())
    return {"dA": d1.mean() / a_max, "jerk": d2.mean() / a_max, "bJerk": bj.mean() / a_max,
            "flips": flips.mean(), "offgrid": offgrid, ">amax": amax, "dev": dev, "div": div, "rmse": rmse / a_max}


def fmt_row(label, m, base=None, extra=""):
    cols = ["dA", "jerk", "bJerk", "flips", "offgrid", ">amax", "dev", "div", "rmse"]
    out = f"  {label:28s}"
    for c in cols:
        v = m[c]
        if base is not None and c in ("dA", "jerk", "bJerk", "flips", "div") and base[c] > 0:
            out += f" {v:7.3f}({100 * (v / base[c] - 1):+4.0f}%)"
        else:
            out += f" {v:7.3f}       " if c in ("dA", "jerk", "bJerk", "flips", "div") else f" {v:7.3f}"
    return out + extra


def header():
    return ("  " + "condition".ljust(28) + "".join(f" {c:>15s}" for c in ["dA", "jerk", "bJerk", "flips"])
            + "".join(f" {c:>7s}" for c in ["offgrid", ">amax", "dev"]) + f" {'div':>15s} {'rmse':>7s}")


def main():
    args = parse_args()
    with open(os.path.join(args.run_dir, "config.json"), encoding="utf-8") as f:
        cfg0 = json.load(f)
    hardflow_root = resolve_hardflow(args.hardflow_root, cfg0)
    cfg, model, norm, apply_conditioning, Solver, _ = load_model(args.run_dir, args.ckpt, "ema", hardflow_root, args.device)
    cfg["data_dir"] = resolve_data_dir(args.data_dir, cfg)
    base = Policy(model, norm, cfg, apply_conditioning, Solver, args.ode_steps, args.device)
    a_max = default_params(norm, cfg["act_dim"])["a_max"]
    tdim = cfg["obs_dim"] + cfg["act_dim"]
    res = site_resolutions(model, cfg["horizon"], tdim, args.device)
    print("sites (temporal length):", ", ".join(f"{s}={r}" for s, r in res.items()))

    obs, gt, hist = load_windows(cfg, args.n_obs, args.exec_steps, args.seed)
    n, k, e = len(obs), args.n_samples, args.exec_steps
    o = torch.from_numpy(np.repeat(obs, k, axis=0)).to(args.device)
    o_n = (o - base.center[base.act_dim:]) / base.scale[base.act_dim:]
    if args.obs_noise > 0:
        g = torch.Generator(device="cpu").manual_seed(args.seed + 1)
        o_n = o_n + args.obs_noise * torch.randn(o_n.shape, generator=g).to(o_n.device)
    cond = {0: o_n}
    torch.manual_seed(args.seed)
    x0 = apply_conditioning(torch.randn(n * k, base.horizon, tdim, device=args.device), cond, base.act_dim)

    # ---- [1] equivalence ----------------------------------------------------
    t = base.t_span[3]
    with torch.no_grad():
        v = model(x0, t)
        pert = TemporalPerturbation(model, list(site_modules(model)), "center")
        v_inert = model(x0, t)
        acg0 = ACGModel(model, pert, 0.0)(x0, t)
        pert.split = n * k
        v_half = model(torch.cat([x0, x0]), t)[: n * k]
        pert.split = None
        pert.remove()
    d_inert, d_zero, d_half = (float((v - u).abs().max()) for u in (v_inert, acg0, v_half))
    print(f"[1] inert hooks |Δ|={d_inert:.1e}  scale 0 |Δ|={d_zero:.1e}  clean half of [x,x] |Δ|={d_half:.1e}  ->",
          "PASS" if d_inert == 0 and d_zero == 0 and d_half < 1e-5 else "FAIL")

    # ---- conditions -----------------------------------------------------------
    groups = [[s] for s in res] if args.sites == "each" else [g.split(",") for g in args.sites.split(";")]
    modes = [m for m in args.modes.split(",") if m]
    scales = [float(s) for s in args.scales.split(",") if s]
    t_lo, t_hi = (float(s) for s in args.t_range.split(","))
    out_t = {"t_range": [t_lo, t_hi], "obs_noise": args.obs_noise}
    t0 = time.perf_counter()
    a_base = generate(model, base, x0, cond, apply_conditioning)
    m_base = metrics(a_base, a_base, gt, hist, a_max, n, k, e)
    out = {"n_obs": n, "n_samples": k, "exec_steps": e, "a_max": a_max, "sites": res, **out_t,
           "baseline": m_base, "runs": []}

    print(f"[2]/[3] {n} observations x {k} noise samples, executed {e} steps; (±%) = change vs baseline")
    print(header())
    print(fmt_row("baseline v_θ", m_base))
    for mode in modes:
        for g in groups:
            pert = TemporalPerturbation(model, g, mode)
            label = f"{mode}:{'+'.join(g)}"
            if not args.no_ic:
                a_ic = generate(PerturbedModel(model, pert), base, x0, cond, apply_conditioning)
                m_ic = metrics(a_ic, a_base, gt, hist, a_max, n, k, e)
                print(fmt_row(f"IC   {label}", m_ic, m_base))
                out["runs"].append({"kind": "ic", "mode": mode, "sites": g, **m_ic})
            for lam in scales:
                f = ACGModel(model, pert, lam, t_lo, t_hi)
                a_g = generate(f, base, x0, cond, apply_conditioning)
                m_g = metrics(a_g, a_base, gt, hist, a_max, n, k, e)
                st = f.stats()
                print(fmt_row(f"ACG  {label} λ={lam:g}", m_g, m_base, f"  gap {st['acg_rel_gap']:.3f}"))
                out["runs"].append({"kind": "acg", "mode": mode, "sites": g, "scale": lam, **m_g, **st})
            pert.remove()
    print(f"done in {time.perf_counter() - t0:.1f} s")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1, default=float)
        print("saved", args.out)


if __name__ == "__main__":
    main()
