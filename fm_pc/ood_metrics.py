"""Score OOD-benchmark evaluations: seam vs in-chunk jerk, OOD distance, task metrics.

    python fm_pc/ood_metrics.py \\
        --eval-dir <run>/eval_pushing_test_id_...      --label ID \\
        --eval-dir <run>/eval_pushing_test_ood-push4cm_...  --label push4cm

All runs must use the same task and checkpoint; the first one is the reference
for paired differences (same contexts, same trajectory index).

Per rollout (a_t executed action = desired-hand increment, S = steps where a new
chunk starts, a_max = training action range; position jerk = Δ³ des = Δ² a):

  J_seam    mean_{s in S, s>=2} ||a_s − 2a_{s−1} + a_{s−2}|| / a_max     jerk across a chunk seam
  jump      mean_{s in S, s>=1} ||a_s − a_{s−1}|| / a_max                 action jump at a seam
  J_in      mean_{t: t, t−1 not in S} ||a_t − 2a_{t−1} + a_{t−2}|| / a_max   jerk inside chunks
  seam/in   J_seam / J_in
  JerkRMS   sqrt(mean ||Δ³ des / dt³||²)  (m/s³, dt = 0.035 s)              whole episode
  ATV       mean ||a_t − a_{t−1}|| / a_max                                 whole episode
  J_event   jerk at the first seam at or after the push (OOD-3 only)
  OOD       distance (normalised obs units) from each planning observation to
            its nearest training observation (k=1, random subset of the train split);
            mean over plans, and at the first plan after the push

Uncertainty: 95% bootstrap over contexts (rollouts of one context are not
independent); Δ = paired difference to the reference on the same contexts.
"""
import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
from scipy.spatial import cKDTree

DT = 0.035
# tasks whose action is not a pure position increment: (number of motion dims, control period)
MOTION = {"stacking": (7, 0.030)}        # 7 joint increments (+ absolute gripper width, excluded)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", action="append", required=True)
    p.add_argument("--label", action="append", default=None)
    p.add_argument("--data-dir", default=None, help="processed data dir (default: the run's config data_dir, else <repo>/data/<task>_sincos_v1)")
    p.add_argument("--n-train", type=int, default=200000, help="train observations used for the OOD distance")
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--out", default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_eval(d):
    with open(os.path.join(d, "results.json"), encoding="utf-8") as f:
        res = json.load(f)
    roll = np.load(os.path.join(d, "rollouts.npz"))
    return res, roll


def train_tree(task, data_dir, center, scale, n, rng, saved=None):
    if not data_dir:
        cands = [saved, os.path.join(REPO, "data", f"{task}_sincos_v1")]
        data_dir = next((c for c in cands if c and os.path.isdir(os.path.join(c, "train"))), cands[-1])
    files = sorted(glob.glob(os.path.join(data_dir, "train", "*.npz")))
    if not files:
        raise SystemExit(f"no training data under {data_dir}/train")
    obs = np.concatenate([np.load(f)["obs"] for f in files]).astype(np.float64)
    if len(obs) > n:
        obs = obs[rng.choice(len(obs), n, replace=False)]
    return cKDTree((obs - center) / scale)


def rollout_metrics(roll, key, a_max, exe, tree, center, scale, md=None, dt=DT):
    a = roll[key + "_act"].astype(np.float64)[:, :md]
    des = roll[key + "_des"].astype(np.float64)[:, :md]
    T = len(a)
    seams = roll[key + "_seams"].astype(int) if key + "_seams" in roll else np.arange(0, T, exe)
    S = set(int(s) for s in seams)
    j3 = np.full(T, np.nan)
    if T > 2:
        j3[2:] = np.linalg.norm(a[2:] - 2 * a[1:-1] + a[:-2], axis=1) / a_max
    j_seam = [j3[s] for s in seams if 2 <= s < T]
    jump = [np.linalg.norm(a[s] - a[s - 1]) / a_max for s in seams if 1 <= s < T]
    j_in = [j3[t] for t in range(2, T) if t not in S and (t - 1) not in S]
    jr = np.diff(des, n=3, axis=0) / dt ** 3 if T > 3 else np.zeros((1, a.shape[1]))
    out = {
        "J_seam": np.mean(j_seam) if j_seam else np.nan,
        "jump": np.mean(jump) if jump else np.nan,
        "J_in": np.mean(j_in) if j_in else np.nan,
        "JerkRMS": float(np.sqrt((np.linalg.norm(jr, axis=1) ** 2).mean())),
        "ATV": float(np.linalg.norm(np.diff(a, axis=0), axis=1).mean() / a_max) if T > 1 else 0.0,
        "J_event": np.nan, "OOD": np.nan, "OOD_event": np.nan,
    }
    ev = int(roll[key + "_event"][0]) if key + "_event" in roll else -1
    if ev >= 0:
        after = [s for s in seams if s >= ev and 2 <= s < T]
        if after:
            out["J_event"] = j3[after[0]]
    if tree is not None and key + "_pobs" in roll and len(roll[key + "_pobs"]):
        po = (roll[key + "_pobs"].astype(np.float64) - center) / scale
        dist, _ = tree.query(po, k=1)
        out["OOD"] = float(dist.mean())
        if ev >= 0:
            k = int(np.searchsorted(seams, ev))        # plan k starts executing at seams[k] >= ev
            if k < len(dist):
                out["OOD_event"] = float(dist[k])
    return out


def per_context(res, roll, a_max, exe, tree, center, scale, md=None, dt=DT):
    n_ctx, n_traj = int(res["n_contexts"]), int(res["n_traj"])
    keys = ["J_seam", "jump", "J_in", "JerkRMS", "ATV", "J_event", "OOD", "OOD_event"]
    m = {k: np.full((n_ctx, n_traj), np.nan) for k in keys}
    for ci in range(n_ctx):
        for ti in range(n_traj):
            key = f"c{ci:03d}_t{ti}"
            if key + "_act" not in roll:
                continue
            r = rollout_metrics(roll, key, a_max, exe, tree, center, scale, md, dt)
            for k in keys:
                m[k][ci, ti] = r[k]
    m["success"] = roll["successes"].astype(float)
    return m


def ctx_mean(x):
    with np.errstate(all="ignore"):
        return np.nanmean(x, axis=1)


def boot(vals, idx):
    with np.errstate(all="ignore"):
        bs = np.nanmean(vals[idx], axis=1)
    return np.nanpercentile(bs, [2.5, 97.5])


def main():
    args = parse_args()
    labels = args.label or [os.path.basename(d.rstrip("/")) for d in args.eval_dir]
    if len(labels) != len(args.eval_dir):
        raise SystemExit("give one --label per --eval-dir")
    rng = np.random.default_rng(args.seed)
    runs = [load_eval(d) for d in args.eval_dir]
    res0 = runs[0][0]
    run_dir = os.path.dirname(os.path.abspath(args.eval_dir[0].rstrip("/")))
    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    task, act_dim = cfg["task"], cfg["act_dim"]
    nz = np.load(os.path.join(run_dir, "normalizer.npz"))
    md, dt = MOTION.get(task, (act_dim, DT))
    a_max = float(np.asarray(nz["scale"])[:md].max())
    center, scale = np.asarray(nz["center"])[act_dim:], np.asarray(nz["scale"])[act_dim:]
    tree = train_tree(task, args.data_dir, center, scale, args.n_train, rng, cfg.get("data_dir"))

    metrics = []
    for res, roll in runs:
        exe = int(res.get("execute_steps", 8))
        metrics.append(per_context(res, roll, a_max, exe, tree, center, scale, md, dt))
    n_ctx = min(m["success"].shape[0] for m in metrics)
    idx = rng.integers(0, n_ctx, size=(args.n_boot, n_ctx))

    cols = ["success", "J_seam", "J_in", "jump", "JerkRMS", "ATV", "J_event", "OOD", "OOD_event"]
    rows = []
    base_ctx = {k: ctx_mean(metrics[0][k][:n_ctx]) for k in cols}
    for lab, (res, _), m in zip(labels, runs, metrics):
        row = {"label": lab, "ood": (res.get("ood") or {}).get("kind", "none"), "n": int(np.isfinite(m["J_in"][:n_ctx]).sum()),
               "entropy": res.get("entropy"), "score": res.get("score")}
        for k in cols:
            c = ctx_mean(m[k][:n_ctx])
            with np.errstate(all="ignore"):
                row[k] = float(np.nanmean(c))
            row[k + "_ci"] = boot(c[:, None], idx).tolist() if np.isfinite(c).any() else [np.nan, np.nan]
            if lab != labels[0] and np.isfinite(c).any() and np.isfinite(base_ctx[k]).any():
                dlt = c - base_ctx[k]
                row["d_" + k] = float(np.nanmean(dlt))
                row["d_" + k + "_ci"] = boot(dlt[:, None], idx).tolist()
        row["seam_over_in"] = row["J_seam"] / row["J_in"] if row["J_in"] else np.nan
        pushes = [e.get("push", {}) for e in (res.get("ood") or {}).get("events", {}).values()]
        if pushes:
            row["push_applied"] = float(np.mean([p.get("to") is not None for p in pushes]))
        rows.append(row)

    def f(r, k, pct=False, d=3):
        v = r.get(k)
        if v is None or not np.isfinite(v):
            return "–"
        s = f"{100 * v:.1f}" if pct else f"{v:.{d}f}"
        dk = "d_" + k
        if dk in r and np.isfinite(r[dk]):
            lo, hi = r[dk + "_ci"]
            s += f" ({100 * r[dk]:+.1f} [{100 * lo:+.1f},{100 * hi:+.1f}])" if pct else f" ({r[dk]:+.{d}f} [{lo:+.{d}f},{hi:+.{d}f}])"
        return s

    unit = "rad (joints)" if task in MOTION else "m"
    head = f"# {task}: {n_ctx} contexts x {res0['n_traj']} rollouts, a_max {a_max:.5f} {unit}; (Δ [95% CI]) = paired vs '{labels[0]}', bootstrap over contexts"
    lines = [head, "",
             "| run | OOD kind | success % | entropy | J_seam | J_in | seam/in | jump | JerkRMS (unit/s³) | ATV | J_event | OOD dist | OOD at event |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        ent = "–" if r["entropy"] is None else f"{r['entropy']:.3f}"
        lines.append(f"| {r['label']} | {r['ood']} | {f(r, 'success', True)} | {ent} | {f(r, 'J_seam')} | {f(r, 'J_in')} | "
                     f"{r['seam_over_in']:.2f} | {f(r, 'jump')} | {f(r, 'JerkRMS', d=1)} | {f(r, 'ATV')} | {f(r, 'J_event')} | "
                     f"{f(r, 'OOD')} | {f(r, 'OOD_event')} |")
    lines += ["", "J_* and jump in units of a_max (Δ² of the executed action = Δ³ of the desired hand position).",
              "OOD dist: nearest-neighbour distance of the planning observation to the training observations (normalised units)."]
    md = "\n".join(lines)
    out = args.out or os.path.join(run_dir, "ood_summary_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write(md + "\n")
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"task": task, "a_max": a_max, "eval_dirs": args.eval_dir, "rows": rows}, fh, indent=1, default=float)
    print(md)
    print("\nsaved", out)


if __name__ == "__main__":
    main()
