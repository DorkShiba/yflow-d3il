import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import Normalizer, add_spectral_norm, find_split_files, load_split, resolve_data_dir, resolve_hardflow, set_seed, window_index


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--split", choices=["train", "val"], default="val")
    p.add_argument("--n-windows", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--ode-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hardflow-root", default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--weights", choices=["auto", "ema", "model"], default="auto")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    with open(os.path.join(args.run_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    hardflow_root = resolve_hardflow(args.hardflow_root, cfg)
    cfg["data_dir"] = resolve_data_dir(args.data_dir, cfg)
    sys.path.insert(0, hardflow_root)
    from hardflow.models_flow.flow_matcher import apply_conditioning
    from hardflow.models_flow.flow_policy import ConditionedODESolver
    from hardflow.models_flow.unet import TemporalUnet

    set_seed(args.seed)
    obs_dim, act_dim, horizon = cfg["obs_dim"], cfg["act_dim"], cfg["horizon"]
    transition_dim = obs_dim + act_dim

    nz = np.load(os.path.join(args.run_dir, "normalizer.npz"))
    norm = Normalizer(nz["center"], nz["scale"])

    model = TemporalUnet(
        horizon=horizon,
        transition_dim=transition_dim,
        cond_dim=obs_dim,
        dim=cfg["model"]["dim"],
        dim_mults=tuple(cfg["model"]["dim_mults"]),
        attention=cfg["model"]["attention"],
    )
    if cfg.get("spectral_norm"):
        skip = [model.final_conv[1]] if cfg.get("spectral_norm_skip_output") else []
        add_spectral_norm(model, skip)
    ckpt_path = args.ckpt
    if ckpt_path is None:
        for name in ["last.pt", "checkpoint.pt", "baseline_ema.pt"]:
            if os.path.exists(os.path.join(args.run_dir, name)):
                ckpt_path = name
                break
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(args.run_dir, ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    has_ema = ckpt.get("ema") is not None
    use = "ema" if (args.weights == "ema" or (args.weights == "auto" and has_ema)) else "model"
    if use == "ema" and not has_ema:
        raise SystemExit("No EMA weights in checkpoint")
    model.load_state_dict(ckpt[use])
    model = model.to(args.device).eval()

    files = find_split_files(cfg["data_dir"], args.split)
    eps, _ = load_split(files, obs_dim, act_dim, horizon)
    eps_n = [norm.normalize(e) for e in eps]
    index = window_index(eps_n, horizon)
    rng = np.random.default_rng(args.seed)
    n = min(args.n_windows, len(index))
    rows = rng.choice(len(index), size=n, replace=False)
    gt_n = np.stack([eps_n[e][s:s + horizon] for e, s in index[rows]], axis=0)
    gt = np.stack([eps[e][s:s + horizon] for e, s in index[rows]], axis=0)

    t_span = torch.linspace(0.0, 1.0, args.ode_steps + 1, device=args.device)[:-1]
    outs = []
    t0 = time.time()
    with torch.no_grad():
        for b in range(0, n, args.batch_size):
            g = torch.from_numpy(gt_n[b:b + args.batch_size]).to(args.device)
            cond = {0: g[:, 0, act_dim:]}
            solver = ConditionedODESolver(model, cond, action_dim=act_dim, value=None, ode_method="euler")
            x = torch.randn(g.shape[0], horizon, transition_dim, device=args.device)
            x = apply_conditioning(x, cond, act_dim)
            x = solver(x, t_span=t_span)
            outs.append(x.cpu().numpy())
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    gen_n = np.concatenate(outs, axis=0)
    gen = gen_n * norm.scale + norm.center

    shape_ok = gen_n.shape == (n, horizon, transition_dim)
    finite_ok = bool(np.isfinite(gen_n).all())
    cond_err_n = float(np.abs(gen_n[:, 0, act_dim:] - gt_n[:, 0, act_dim:]).max())
    cond_err = float(np.abs(gen[:, 0, act_dim:] - gt[:, 0, act_dim:]).max())
    out_of_range = float((np.abs(gen_n) > 1.0).mean())
    out_of_range_far = float((np.abs(gen_n) > 1.5).mean())

    a_gen, a_gt = gen[:, :, :act_dim], gt[:, :, :act_dim]
    o_gen, o_gt = gen[:, 1:, act_dim:], gt[:, 1:, act_dim:]
    mean_act = np.concatenate([e[:, :act_dim] for e in eps], axis=0).mean(axis=0)

    def rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))

    print(f"run: {args.run_dir}")
    print(f"checkpoint: {os.path.basename(ckpt_path)}  step: {ckpt.get('step')}  weights: {use}")
    print(f"split: {args.split}  windows: {n}  ode_steps: {args.ode_steps}  spectral_norm: {cfg.get('spectral_norm')}")
    print(f"time: {elapsed:.2f}s total  {1000 * elapsed / n:.2f} ms/window")
    print(f"shape: {gen_n.shape}  ok: {shape_ok}")
    print(f"finite: {finite_ok}")
    print(f"cond t0 max abs err  normalized: {cond_err_n:.3e}  raw: {cond_err:.3e}")
    print(f"normalized |x|>1: {out_of_range:.4f}  |x|>1.5: {out_of_range_far:.4f}")
    print(f"action rmse first step  model: {rmse(a_gen[:, 0], a_gt[:, 0]):.6f}  zero: {rmse(0 * a_gt[:, 0], a_gt[:, 0]):.6f}  mean: {rmse(mean_act + 0 * a_gt[:, 0], a_gt[:, 0]):.6f}")
    print(f"action rmse horizon     model: {rmse(a_gen, a_gt):.6f}  zero: {rmse(0 * a_gt, a_gt):.6f}  mean: {rmse(mean_act + 0 * a_gt, a_gt):.6f}")
    print(f"obs rmse t>=1           model: {rmse(o_gen, o_gt):.6f}  hold t0: {rmse(np.repeat(gt[:, :1, act_dim:], horizon - 1, axis=1), o_gt):.6f}")
    print(f"gen action abs mean: {np.abs(a_gen).mean():.6f}  gt action abs mean: {np.abs(a_gt).mean():.6f}")

    if not args.no_save:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = os.path.join(args.run_dir, f"samples_{args.split}_{stamp}.npz")
        np.savez(
            path,
            generated=gen.astype(np.float32),
            ground_truth=gt.astype(np.float32),
            window_index=index[rows],
            ode_steps=args.ode_steps,
            seed=args.seed,
        )
        print(f"saved: {path}")
    if not (shape_ok and finite_ok and cond_err_n < 1e-5):
        raise SystemExit("CHECK FAILED")
    print("CHECK PASSED")


if __name__ == "__main__":
    main()
