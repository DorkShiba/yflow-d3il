import argparse
import copy
import csv
import json
import math
import os
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from glob import glob

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm

TASK_DIMS = {"pushing": (12, 2), "inserting": (16, 2)}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=sorted(TASK_DIMS), default="pushing")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--dim-mults", default="1,4,8")
    p.add_argument("--attention", action="store_true")
    p.add_argument("--scheduler", choices=["none", "cosine"], default="none")
    p.add_argument("--ema-decay", type=float, default=0.0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--val-batches", type=int, default=8)
    p.add_argument("--val-seed", type=int, default=1234)
    p.add_argument("--time-limit-hours", type=float, default=0.0)
    p.add_argument("--sn", action="store_true")
    p.add_argument("--sn-skip-output", action="store_true")
    p.add_argument("--tag", default="smoke")
    p.add_argument("--resume", default=None)
    p.add_argument("--work-root", default="/root/d3il-work/codex")
    p.add_argument("--data-root", default=None)
    p.add_argument("--hardflow-root", default=None)
    p.add_argument("--out-root", default=None)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_split_files(task_dir, split):
    files = sorted(glob(os.path.join(task_dir, split, "**", "*.npz"), recursive=True))
    if not files:
        listing = sorted(os.listdir(task_dir)) if os.path.isdir(task_dir) else "missing"
        raise FileNotFoundError(f"No npz files for split '{split}' under {task_dir}. Contents: {listing}")
    return files


def load_split(files, obs_dim, act_dim, horizon):
    episodes = []
    skipped = 0
    for f in files:
        with np.load(f) as d:
            obs = np.asarray(d["obs"], dtype=np.float32)
            act = np.asarray(d["actions"], dtype=np.float32)
        if obs.ndim != 2 or obs.shape[1] != obs_dim:
            raise ValueError(f"{f}: obs shape {obs.shape}, expected (*, {obs_dim})")
        if act.ndim != 2 or act.shape[1] != act_dim:
            raise ValueError(f"{f}: actions shape {act.shape}, expected (*, {act_dim})")
        if obs.shape[0] != act.shape[0]:
            raise ValueError(f"{f}: obs length {obs.shape[0]} != actions length {act.shape[0]}")
        if not (np.isfinite(obs).all() and np.isfinite(act).all()):
            raise ValueError(f"{f}: non-finite values")
        if obs.shape[0] < horizon:
            skipped += 1
            continue
        episodes.append(np.concatenate([act, obs], axis=1))
    if not episodes:
        raise ValueError("No episode is long enough for the horizon")
    return episodes, skipped


def window_index(episodes, horizon):
    idx = []
    for e, ep in enumerate(episodes):
        for s in range(ep.shape[0] - horizon + 1):
            idx.append((e, s))
    return np.asarray(idx, dtype=np.int64)


class Normalizer:
    def __init__(self, center, scale):
        self.center = center.astype(np.float32)
        self.scale = scale.astype(np.float32)

    @classmethod
    def fit(cls, episodes):
        allx = np.concatenate(episodes, axis=0)
        lo = allx.min(axis=0)
        hi = allx.max(axis=0)
        center = (hi + lo) / 2.0
        scale = (hi - lo) / 2.0
        small = scale < 1e-8
        scale[small] = 1.0
        return cls(center, scale), lo, hi, small

    def normalize(self, x):
        return (x - self.center) / self.scale


class WindowSampler:
    def __init__(self, episodes_n, horizon, device):
        lengths = [e.shape[0] for e in episodes_n]
        offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]])
        starts = [o + np.arange(n - horizon + 1) for o, n in zip(offsets, lengths)]
        self.data = torch.from_numpy(np.concatenate(episodes_n, axis=0)).to(device)
        self.starts = torch.from_numpy(np.concatenate(starts)).to(device)
        self.ar = torch.arange(horizon, device=device)
        self.n = int(self.starts.shape[0])

    def batch(self, rows):
        pos = self.starts[rows][:, None] + self.ar[None, :]
        return self.data[pos]


def add_spectral_norm(model, skip_modules):
    wrapped = []
    for name, m in list(model.named_modules()):
        if any(m is s for s in skip_modules):
            continue
        if isinstance(m, nn.ConvTranspose1d):
            spectral_norm(m, name="weight", dim=1)
            wrapped.append(name)
        elif isinstance(m, (nn.Conv1d, nn.Linear)):
            spectral_norm(m, name="weight", dim=0)
            wrapped.append(name)
    return wrapped


def build_model(TemporalUnet, horizon, obs_dim, act_dim, sn, sn_skip_output, dim=32, dim_mults=(1, 4, 8), attention=False):
    model = TemporalUnet(
        horizon=horizon,
        transition_dim=obs_dim + act_dim,
        cond_dim=obs_dim,
        dim=dim,
        dim_mults=dim_mults,
        attention=attention,
    )
    layers = []
    if sn:
        skip = [model.final_conv[1]] if sn_skip_output else []
        layers = add_spectral_norm(model, skip)
    return model, layers


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.model = copy.deepcopy(model)
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for e, m in zip(self.model.parameters(), model.parameters()):
            e.mul_(self.decay).add_(m.detach(), alpha=1.0 - self.decay)
        for e, m in zip(self.model.buffers(), model.buffers()):
            e.copy_(m)

    @torch.no_grad()
    def refresh_sn(self, horizon, transition_dim, device, iters=20):
        self.model.train()
        x = torch.zeros(1, horizon, transition_dim, device=device)
        t = torch.zeros(1, device=device)
        for _ in range(iters):
            self.model(x, t)
        self.model.eval()


def validate(FlowMatcher, model, sampler, args, act_dim):
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(args.val_seed)
    torch.cuda.manual_seed_all(args.val_seed)
    g = np.random.default_rng(args.val_seed)
    was_training = model.training
    model.eval()
    flow = FlowMatcher(model=model, action_dim=act_dim, flow_matching_type="cfm")
    losses = []
    with torch.no_grad():
        for _ in range(args.val_batches):
            rows = torch.from_numpy(g.integers(0, sampler.n, size=args.batch_size)).to(sampler.data.device)
            x = sampler.batch(rows)
            loss, _ = flow.loss(x, {0: x[:, 0, act_dim:]})
            losses.append(loss.item())
    model.train(was_training)
    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    return float(np.mean(losses))


def lr_at(step, args):
    if args.scheduler == "cosine":
        return 0.5 * args.lr * (1.0 + math.cos(math.pi * min(step, args.steps) / args.steps))
    return args.lr


def save_state(path, model, ema, optimizer, step, config, data_rng, extra=None):
    state = {
        "model": model.state_dict(),
        "ema": ema.model.state_dict() if ema is not None else None,
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": config,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "data_rng": data_rng.bit_generator.state,
    }
    if extra:
        state.update(extra)
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def main():
    args = parse_args()
    if args.resume:
        with open(os.path.join(args.resume, "config.json"), encoding="utf-8") as f:
            saved = json.load(f)
        keep = {"device", "resume", "time_limit_hours", "work_root", "data_root", "hardflow_root", "out_root"}
        for k, v in saved["args"].items():
            if k not in keep:
                setattr(args, k, v)

    data_root = args.data_root or os.path.join(args.work_root, "processed")
    hardflow_root = args.hardflow_root or os.path.join(args.work_root, "HardFlow")
    out_root = args.out_root or os.path.join(args.work_root, "fm_baseline", "runs")
    sys.path.insert(0, hardflow_root)
    from hardflow.models_flow.flow_matcher import FlowMatcher
    from hardflow.models_flow.unet import TemporalUnet

    set_seed(args.seed)
    obs_dim, act_dim = TASK_DIMS[args.task]
    transition_dim = obs_dim + act_dim
    task_dir = os.path.join(data_root, f"{args.task}_sincos_v1")

    train_files = find_split_files(task_dir, "train")
    val_files = find_split_files(task_dir, "val")
    train_eps, train_skip = load_split(train_files, obs_dim, act_dim, args.horizon)
    val_eps, val_skip = load_split(val_files, obs_dim, act_dim, args.horizon)

    norm, lo, hi, small = Normalizer.fit(train_eps)
    train_s = WindowSampler([norm.normalize(e) for e in train_eps], args.horizon, args.device)
    val_s = WindowSampler([norm.normalize(e) for e in val_eps], args.horizon, args.device)

    print(f"task: {args.task}")
    print(f"train files: {len(train_files)}  used: {len(train_eps)}  skipped short: {train_skip}  windows: {train_s.n}")
    print(f"val files: {len(val_files)}  used: {len(val_eps)}  skipped short: {val_skip}  windows: {val_s.n}")
    print(f"constant channels: {np.flatnonzero(small).tolist()}")

    model, sn_layers = build_model(TemporalUnet, args.horizon, obs_dim, act_dim, args.sn, args.sn_skip_output, args.dim, tuple(int(v) for v in str(args.dim_mults).split(",")), args.attention)
    model = model.to(args.device)
    flow = FlowMatcher(model=model, action_dim=act_dim, flow_matching_type="cfm")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    ema = EMA(model, args.ema_decay) if args.ema_decay > 0 else None
    data_rng = np.random.default_rng(args.seed)

    print(f"device: {args.device}")
    print(f"parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"spectral_norm: {args.sn}  skip_output: {args.sn_skip_output}  wrapped layers: {len(sn_layers)}")
    print(f"steps: {args.steps}  scheduler: {args.scheduler}  ema: {args.ema_decay}  save_every: {args.save_every}")

    start_step = 0
    if args.resume:
        run_dir = args.resume
        state = torch.load(os.path.join(run_dir, "last.pt"), map_location=args.device)
        model.load_state_dict(state["model"])
        if ema is not None:
            ema.model.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"].cpu())
        if state["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
        data_rng.bit_generator.state = state["data_rng"]
        start_step = int(state["step"])
        with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as f:
            config = json.load(f)
        print(f"resumed: {run_dir}  from step {start_step}")
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        sn_tag = ("sn_skipout" if args.sn_skip_output else "sn") if args.sn else "nosn"
        run_dir = os.path.join(out_root, f"{args.task}_{args.tag}_{sn_tag}_{stamp}")
        os.makedirs(run_dir, exist_ok=False)
        config = {
            "task": args.task,
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "transition_order": "actions_then_obs",
            "horizon": args.horizon,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "lr": args.lr,
            "optimizer": "Adam",
            "weight_decay": 0.0,
            "scheduler": args.scheduler,
            "ema_decay": args.ema_decay,
            "model": {"name": "TemporalUnet", "dim": args.dim, "dim_mults": [int(v) for v in str(args.dim_mults).split(",")], "attention": args.attention},
            "flow_matching_type": "cfm",
            "conditioning": "obs at t=0 via apply_conditioning",
            "spectral_norm": args.sn,
            "spectral_norm_skip_output": args.sn_skip_output,
            "spectral_norm_layers": sn_layers,
            "val_batches": args.val_batches,
            "val_seed": args.val_seed,
            "data_dir": task_dir,
            "hardflow_root": hardflow_root,
            "n_train_files": len(train_files),
            "n_val_files": len(val_files),
            "n_train_windows": train_s.n,
            "n_val_windows": val_s.n,
            "torch": torch.__version__,
            "created_utc": stamp,
            "args": vars(args),
        }
        with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        for mf in glob(os.path.join(task_dir, "*.json")):
            shutil.copy2(mf, os.path.join(run_dir, os.path.basename(mf)))
        np.savez(
            os.path.join(run_dir, "normalizer.npz"),
            center=norm.center,
            scale=norm.scale,
            min=lo.astype(np.float32),
            max=hi.astype(np.float32),
            constant=small,
        )
        with open(os.path.join(run_dir, "metrics.csv"), "w", newline="") as f:
            csv.writer(f).writerow(["step", "lr", "train_loss_mean", "val_loss", "val_loss_ema", "steps_per_sec", "elapsed_sec"])
        v0 = validate(FlowMatcher, model, val_s, args, act_dim)
        print(f"step 0  val {v0:.6f}", flush=True)
        with open(os.path.join(run_dir, "metrics.csv"), "a", newline="") as f:
            csv.writer(f).writerow([0, f"{args.lr:.3e}", "", f"{v0:.6f}", "", "", "0.0"])

    print(f"run_dir: {run_dir}", flush=True)
    metrics_path = os.path.join(run_dir, "metrics.csv")
    limit_sec = args.time_limit_hours * 3600.0 if args.time_limit_hours > 0 else float("inf")
    t0 = time.time()
    t_log = t0
    s_log = start_step
    loss_sum = torch.zeros((), device=args.device)
    n_sum = 0
    bad = torch.zeros((), dtype=torch.bool, device=args.device)
    step = start_step
    model.train()

    def checkpoint(step, final):
        if ema is not None:
            ema.refresh_sn(args.horizon, transition_dim, args.device)
        save_state(os.path.join(run_dir, "last.pt"), model, ema, optimizer, step, config, data_rng)
        if final or (args.save_every and step % args.save_every == 0):
            torch.save(
                {"model": model.state_dict(), "ema": ema.model.state_dict() if ema is not None else None, "step": step, "config": config},
                os.path.join(run_dir, f"ckpt_{step:07d}.pt"),
            )

    stopped_by_time = False
    while step < args.steps:
        step += 1
        for g in optimizer.param_groups:
            g["lr"] = lr_at(step - 1, args)
        rows = torch.from_numpy(data_rng.integers(0, train_s.n, size=args.batch_size)).to(args.device)
        x = train_s.batch(rows)
        loss, _ = flow.loss(x, {0: x[:, 0, act_dim:]})
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
        bad = bad | ~torch.isfinite(loss.detach()) | ~torch.isfinite(gnorm)
        optimizer.step()
        if ema is not None:
            ema.update(model)
        loss_sum += loss.detach()
        n_sum += 1

        is_save = args.save_every and step % args.save_every == 0
        if step % args.log_every == 0 or step == args.steps or is_save:
            if bool(bad):
                raise FloatingPointError(f"Non-finite loss or gradient before step {step}")
            now = time.time()
            sps = (step - s_log) / max(now - t_log, 1e-9)
            t_log, s_log = now, step
            tr = float(loss_sum.item() / max(n_sum, 1))
            loss_sum = torch.zeros((), device=args.device)
            n_sum = 0
            va = validate(FlowMatcher, model, val_s, args, act_dim)
            va_e = ""
            if ema is not None:
                ema.refresh_sn(args.horizon, transition_dim, args.device)
                va_e = f"{validate(FlowMatcher, ema.model, val_s, args, act_dim):.6f}"
            el = now - t0
            eta_h = (args.steps - step) / max(sps, 1e-9) / 3600.0
            print(f"step {step}  lr {optimizer.param_groups[0]['lr']:.2e}  train {tr:.6f}  val {va:.6f}  val_ema {va_e or '-'}  {sps:.1f} it/s  elapsed {el / 3600:.2f}h  eta {eta_h:.2f}h", flush=True)
            with open(metrics_path, "a", newline="") as f:
                csv.writer(f).writerow([step, f"{optimizer.param_groups[0]['lr']:.3e}", f"{tr:.6f}", f"{va:.6f}", va_e, f"{sps:.2f}", f"{el:.1f}"])
        if is_save:
            checkpoint(step, False)
            print(f"saved checkpoint at step {step}", flush=True)
        if time.time() - t0 > limit_sec:
            stopped_by_time = True
            break

    checkpoint(step, True)
    if stopped_by_time:
        print(f"TIME LIMIT reached at step {step}. Resume with: --resume {run_dir}", flush=True)
    else:
        print(f"DONE step {step}", flush=True)
    print(f"saved: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
