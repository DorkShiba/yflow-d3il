"""Summarise constraint experiments: one table + one figure.

Reads eval_* folders (results.json + rollouts.npz) of the SAME task and
checkpoint and reports, per folder:

  D3IL metrics      success rate, mode entropy, score = (success + entropy) / 2,
                    block-target mean distance, episode length, plan time
  constraint        computed from the executed trajectories (rollouts.npz):
                      runaway episodes  max_t |des_t - hand_t|_inf > --track-d
                      gap steps         fraction of steps with that gap > --track-d
                      max gap           per-episode max of |des_t - hand_t|_inf (m)
                      a_max steps       fraction of steps with |a_t,i| > a_max
                      action jerk       mean ||a_t - a_{t-1}||_2 / a_max
  uncertainty       95% CI by bootstrap over CONTEXTS (the n_traj rollouts of
                    one context are not independent), and the paired
                    difference to the baseline (--baseline-index, default 0)
                    on the same resampled contexts.

    python compute_metrics.py \\
        --eval-dir ../fm_baseline/results/pushing_baseline/eval_pushing_test_v100_2026... --label baseline \\
        --eval-dir ../fm_baseline/results/pushing_baseline/eval_pushing_test_v100_pc_pc_... --label pc \\
        --eval-dir ../fm_baseline/results/pushing_baseline/eval_pushing_test_v100_pc_nocbf_... --label pc-noCBF

Outputs (default <run dir>/summary_<stamp>/): summary.md, summary.csv,
summary.json, summary.png
"""
import argparse
import csv
import json
import os
from datetime import datetime, timezone

import numpy as np

SURFACE, INK, INK2, GRID, MARK, REF = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0", "#2a78d6", "#8a8984"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-dir", action="append", required=True)
    p.add_argument("--label", action="append", default=None)
    p.add_argument("--baseline-index", type=int, default=0)
    p.add_argument("--track-d", type=float, default=0.10)
    p.add_argument("--a-max", type=float, default=None, help="default: action range in the run's normalizer.npz")
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def load(d):
    d = os.path.abspath(d)
    with open(os.path.join(d, "results.json"), encoding="utf-8") as f:
        res = json.load(f)
    z = np.load(os.path.join(d, "rollouts.npz"))
    return d, res, {k: z[k] for k in z.files}


def episode_arrays(roll, n_ctx, n_traj, track_d, a_max):
    keys = ["runaway", "gap_frac", "max_gap", "amax_frac", "jerk"]
    out = {k: np.full((n_ctx, n_traj), np.nan) for k in keys}
    for ci in range(n_ctx):
        for ti in range(n_traj):
            ep = f"c{ci:03d}_t{ti}"
            if ep + "_des" not in roll:
                continue
            des, hand, act = roll[ep + "_des"][:, :2], roll[ep + "_c"][:, :2], roll[ep + "_act"]
            gap = np.abs(des.astype(np.float64) - hand).max(axis=1)
            out["max_gap"][ci, ti] = gap.max()
            out["runaway"][ci, ti] = float(gap.max() > track_d)
            out["gap_frac"][ci, ti] = float((gap > track_d).mean())
            out["amax_frac"][ci, ti] = float((np.abs(act) > a_max * (1 + 1e-4)).any(axis=1).mean())
            out["jerk"][ci, ti] = float(np.linalg.norm(np.diff(act, axis=0), axis=1).mean() / a_max) if len(act) > 1 else 0.0
    return out


def context_entropy(modes, succ, task):
    """Per-context normalised mode entropy over successful rollouts (D3IL definition)."""
    mode_ids = range(4) if task == "pushing" else range(1, 7)
    n_modes = len(mode_ids)
    n_traj = succ.shape[1]
    ent = np.zeros(succ.shape[0])
    for ci in range(succ.shape[0]):
        p = np.array([np.sum((modes[ci] == m) & (succ[ci] == 1)) / n_traj for m in mode_ids])
        p = p / (p.sum() + 1e-12)
        ent[ci] = float(-(p * np.log(p + 1e-12) / np.log(n_modes)).sum())
    return ent


def boot_ci(per_ctx, idx):
    """per_ctx: (n_ctx,) context means; idx: (B, n_ctx) resampled context indices."""
    stats = np.nanmean(per_ctx[idx], axis=1)
    return float(np.nanpercentile(stats, 2.5)), float(np.nanpercentile(stats, 97.5))


def main():
    args = parse_args()
    runs = [load(d) for d in args.eval_dir]
    labels = args.label or [os.path.basename(d) for d, _, _ in runs]
    if len(labels) != len(runs):
        raise SystemExit("give one --label per --eval-dir")
    task = "pushing" if "context_files" not in runs[0][1] else "inserting"
    shapes = {r["successes"].shape for _, _, r in runs}
    if len(shapes) != 1:
        raise SystemExit(f"eval folders differ in (contexts, rollouts): {shapes}")
    n_ctx, n_traj = shapes.pop()
    run_dir = os.path.dirname(runs[0][0])
    a_max = args.a_max
    if a_max is None:
        a_max = float(np.load(os.path.join(run_dir, "normalizer.npz"))["scale"][:2].max())

    rng = np.random.RandomState(args.seed)
    idx = rng.randint(0, n_ctx, size=(args.n_boot, n_ctx))      # shared by all runs -> paired
    rows, per_ctx = [], {}
    for (d, res, roll), lab in zip(runs, labels):
        succ = roll["successes"].astype(float)
        ep = episode_arrays(roll, n_ctx, n_traj, args.track_d, a_max)
        ent = context_entropy(roll["modes"], succ, task)
        pc = {"success": succ.mean(1), "entropy": ent, "mean_distance": roll["mean_distance"].mean(1),
              "steps": roll["lengths"].mean(1), **{k: np.nanmean(v, 1) for k, v in ep.items()}}
        pc["score"] = 0.5 * (pc["success"] + pc["entropy"])
        per_ctx[lab] = pc
        row = {"label": lab, "eval_dir": os.path.basename(d), "n": int(succ.size),
               "inject": res.get("inject", "none"), "sampler": res.get("sampler", "euler"),
               "constraints": res.get("constraints"), "plan_ms": res.get("plan_ms")}
        for k, v in pc.items():
            row[k] = float(np.nanmean(v))
            row[k + "_lo"], row[k + "_hi"] = boot_ci(v, idx)
        rows.append(row)
    base = labels[args.baseline_index]
    for row in rows:
        for k in ["success", "score", "runaway", "max_gap"]:
            diff = per_ctx[row["label"]][k] - per_ctx[base][k]
            row[f"d_{k}"] = float(np.nanmean(diff))
            row[f"d_{k}_lo"], row[f"d_{k}_hi"] = boot_ci(diff, idx)

    out = args.out_dir or os.path.join(run_dir, "summary_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"task": task, "n_ctx": n_ctx, "n_traj": n_traj, "track_d": args.track_d, "a_max": a_max,
                   "n_boot": args.n_boot, "baseline": base, "rows": rows}, f, indent=2)
    with open(os.path.join(out, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    pct = lambda r, k: f"{100 * r[k]:.1f} [{100 * r[k + '_lo']:.1f}, {100 * r[k + '_hi']:.1f}]"   # noqa: E731
    lines = [f"# {task}: {n_ctx} contexts x {n_traj} rollouts, 95% CI = bootstrap over contexts ({args.n_boot}x)",
             f"track_d = {args.track_d} m, a_max = {a_max:.5f} m, baseline = {base}", "",
             "| run | success % | Δ success %p vs base | entropy | score | mean dist (m) | steps | runaway eps % | Δ runaway %p | gap>d steps % | max gap (m) | a>a_max steps % | jerk (/a_max) | plan ms |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        dsu = f"{100 * r['d_success']:+.1f} [{100 * r['d_success_lo']:+.1f}, {100 * r['d_success_hi']:+.1f}]" if r["label"] != base else "–"
        dru = f"{100 * r['d_runaway']:+.1f} [{100 * r['d_runaway_lo']:+.1f}, {100 * r['d_runaway_hi']:+.1f}]" if r["label"] != base else "–"
        lines.append(f"| {r['label']} | {pct(r, 'success')} | {dsu} | {r['entropy']:.3f} | {r['score']:.3f} | {r['mean_distance']:.4f} | "
                     f"{r['steps']:.1f} | {pct(r, 'runaway')} | {dru} | {100 * r['gap_frac']:.2f} | {r['max_gap']:.4f} | "
                     f"{100 * r['amax_frac']:.2f} | {r['jerk']:.3f} | {r['plan_ms'] or float('nan'):.1f} |")
    lines += ["", "Δ: paired difference on the same resampled contexts; an interval that contains 0 means no detectable difference."]
    md = "\n".join(lines)
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(md + "\n")
    print(md)

    if not args.no_plot:
        plot(rows, base, task, n_ctx, n_traj, args.track_d, os.path.join(out, "summary.png"))
    print(f"\nsaved: {out}")


def plot(rows, base, task, n_ctx, n_traj, track_d, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [("success", "Success rate (%)", 100, "higher is better"),
              ("runaway", f"Runaway episodes (%)\nmax |des - hand| > {track_d:g} m", 100, "lower is better"),
              ("max_gap", "Max desired-hand gap (m)\nper episode, mean", 1, "lower is better"),
              ("entropy", "Mode entropy", 1, "behaviour diversity"),
              ("mean_distance", "Block-target distance (m)", 1, "lower is better"),
              ("jerk", "Action change\n|a_t - a_(t-1)| / a_max", 1, "lower = smoother")]
    n = len(rows)
    fig, axes = plt.subplots(2, 3, figsize=(12, 1.3 + 0.55 * n * 2 + 1.2), facecolor=SURFACE)
    ys = np.arange(n)[::-1]
    for ax, (k, title, s, hint) in zip(axes.ravel(), panels):
        ax.set_facecolor(SURFACE)
        v = np.array([r[k] for r in rows]) * s
        lo = np.array([r[k + "_lo"] for r in rows]) * s
        hi = np.array([r[k + "_hi"] for r in rows]) * s
        bi = [r["label"] for r in rows].index(base)
        ax.axvline(v[bi], color=REF, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.hlines(ys, lo, hi, color=MARK, lw=2, zorder=2)
        ax.plot(v, ys, "o", ms=7, color=MARK, mec=SURFACE, mew=1.5, zorder=3)
        span = max(hi.max() - lo.min(), 1e-9)
        dec = int(max(0, 2 - np.floor(np.log10(span))))           # ~3 significant digits of the span
        for y, x, h in zip(ys, v, hi):
            ax.text(h + 0.03 * span, y, f"{x:.{dec}f}", va="center", ha="left", fontsize=8.5, color=INK2)
        ax.set_xlim(lo.min() - 0.08 * span, hi.max() + 0.28 * span)
        ax.set_yticks(ys)
        ax.set_yticklabels([r["label"] for r in rows], fontsize=9.5, color=INK)
        ax.set_ylim(-0.6, n - 0.4)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.text(1.0, 1.02, hint, transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=INK2)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.tick_params(axis="x", labelsize=8.5, colors=INK2, length=0)
        ax.tick_params(axis="y", length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.suptitle(f"{task.capitalize()}: {n_ctx} contexts x {n_traj} rollouts   "
                 f"dot = mean, line = 95% CI (bootstrap over contexts), dashed = {base}",
                 x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
