"""Build the OOD test-condition datasets (no training needed).

For each task and each condition one file ``fm_baseline/ood_benchmark/<task>/<cond>.pkl``
fixes, per evaluation context, everything the OOD changes:

  A  initial block position  every block starts d outside its demo init range
                             (nearest feasible face; ood.OOD.initial_context)
  B  robot start             the hand starts d away from the fixed demo start (random
                             direction; straight path and end point clear of blocks and
                             walls; executed in the simulator and checked: blocks unmoved,
                             hand within 5 mm of the target)
  C  observation noise       Gaussian noise on block x, y (sigma_pos) and yaw (sigma_yaw) at
                             every re-plan; the file fixes sigma and a per-episode seed

Conditions: id; A, B, C alone at weak / mid / strong; AB, AC, BC, ABC at mid (2^3 design).
Every factor uses its own random stream per context (common random numbers): a combined
cell reuses the single-factor placements / directions / noise draws where they stay feasible.

    python fm_pc/make_ood_dataset.py --task pushing inserting aligning
    python fm_pc/eval_task.py --task pushing --run-dir <run> --n-traj 4 \\
        --ood-file fm_baseline/ood_benchmark/pushing/A_mid.pkl
"""
import argparse
import copy
import json
import os
import pickle
import sys
from datetime import datetime, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(REPO, "fm_baseline")]
from ood import CLEAR, OOD
from tasks import get_task
from train import resolve_d3il

VERSION = "v1"
# (weak, mid, strong).  A, B in metres; C = (sigma_pos m, sigma_yaw deg).
# A = 0.5b / 1b / 2b (b = block edge; capped at the largest shift feasible for every context:
# Aligning 12 cm -- from 13 cm the target cannot leave its range in 59/60 contexts), C = 5 / 10 / 20 % of b.
LEVELS = {
    "pushing":   {"A": (0.025, 0.05, 0.10), "B": (0.02, 0.05, 0.10), "C": ((0.0025, 2.5), (0.005, 5.0), (0.010, 10.0))},
    "inserting": {"A": (0.025, 0.05, 0.10), "B": (0.02, 0.05, 0.10), "C": ((0.0025, 2.5), (0.005, 5.0), (0.010, 10.0))},
    "aligning":  {"A": (0.05, 0.10, 0.12),  "B": (0.02, 0.05, 0.10), "C": ((0.005, 2.5), (0.010, 5.0), (0.020, 10.0))},
}
NAMES = ("weak", "mid", "strong")
HAND_TOL = 0.005          # m: reached start must be within this of the target
BLOCK_TOL = 0.001         # m: blocks must not move while the hand goes to its start


def conditions():
    out = [("id", {})]
    for f in "ABC":
        out += [(f"{f}_{n}", {f: k}) for k, n in enumerate(NAMES)]
    for combo in ("AB", "AC", "BC", "ABC"):
        out.append((f"{combo}_mid", {f: 1 for f in combo}))
    return out


def factor_values(task, sel):
    lv = LEVELS[task]
    v = {"A": None, "B": None, "C": None}
    if "A" in sel:
        v["A"] = {"shift_m": lv["A"][sel["A"]], "level": NAMES[sel["A"]]}
    if "B" in sel:
        v["B"] = {"dist_m": lv["B"][sel["B"]], "level": NAMES[sel["B"]]}
    if "C" in sel:
        p, y = lv["C"][sel["C"]]
        v["C"] = {"pos_m": p, "yaw_deg": y, "level": NAMES[sel["C"]]}
    return v


def path_clear(spec, a, b, blocks, radii, walls):
    """Straight hand path a -> b keeps clear of blocks and walls.  A block that is already
    closer to the demo start than the (conservative, circumscribed) clearance only must not
    get any closer; the simulator check afterwards confirms that nothing was touched."""
    (x0, x1), (y0, y1) = spec.workspace
    if not (x0 + spec.hand_r <= b[0] <= x1 - spec.hand_r and y0 + spec.hand_r <= b[1] <= y1 - spec.hand_r):
        return False
    need = [min(r + spec.hand_r + CLEAR, np.linalg.norm(a - q)) for q, r in zip(blocks, radii)]
    for u in np.linspace(0.0, 1.0, 21):
        p = a + u * (b - a)
        if any(np.linalg.norm(p - q) < n - 1e-9 for q, n in zip(blocks, need)):
            return False
        if any(np.linalg.norm(np.maximum(np.abs(p - c) - h, 0.0)) < spec.hand_r + CLEAR for c, h in walls):
            return False
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", nargs="+", default=list(LEVELS), choices=sorted(LEVELS))
    p.add_argument("--contexts", default="test", help="context split (Inserting always uses its validation file)")
    p.add_argument("--n-contexts", type=int, default=0, help="0 = all")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--d3il-root", default=None)
    p.add_argument("--out-dir", default=os.path.join(REPO, "fm_baseline", "ood_benchmark"))
    args = p.parse_args()
    d3il = resolve_d3il(args.d3il_root)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for task in args.task:
        spec = get_task(task)
        spec.setup(d3il)
        env = spec.make_env(spec.max_steps)
        cargs = argparse.Namespace(contexts_file=None, contexts=args.contexts, n_contexts=args.n_contexts, d3il_root=d3il)
        contexts = spec.contexts(cargs, {})
        walls = spec.walls(env)
        radii = spec.block_radii()
        spec.reset(env, contexts[0][1])
        hand0 = spec.read(env).c[:2].copy()               # fixed demo start (same for every context)
        out_dir = os.path.join(args.out_dir, task)
        os.makedirs(out_dir, exist_ok=True)
        manifest = {"task": task, "version": VERSION, "created_utc": stamp, "seed": args.seed, "n_episodes": len(contexts),
                    "contexts": args.contexts if task != "inserting" else "fm_baseline/contexts/inserting_val.pkl",
                    "demo_hand_start": hand0.tolist(), "levels": LEVELS[task], "conditions": {}}
        print(f"== {task}: {len(contexts)} contexts, hand start {np.round(hand0, 3).tolist()}", flush=True)

        for cond, sel in conditions():
            fv = factor_values(task, sel)
            episodes, n_a_fail, n_b_fail, a_moved = [], 0, 0, []
            for ci, (name, ctx) in enumerate(contexts):
                ep = {"name": name, "context": copy.deepcopy(ctx), "robot_start": None, "hand_actual": None,
                      "noise_seed": int(np.random.default_rng([args.seed, ci, 3]).integers(2 ** 31)), "shift": None, "robot": None}
                if fv["A"]:
                    o = OOD(argparse.Namespace(ood="shift", ood_shift=fv["A"]["shift_m"], ood_noise_pos=0, ood_noise_yaw=0,
                                               ood_push_dist=0, ood_push_step=None, ood_push_phase="seam", ood_delay=0,
                                               ood_seed=args.seed, ood_file=None), spec, env, 8, 16)
                    o.start_episode(ci, 0)
                    ep["context"] = o.initial_context(ctx, hand0)
                    ep["shift"] = o.events[o.key]["shift"]
                    n_a_fail += sum(e["face"] is None for e in ep["shift"])
                    a_moved += [float(np.linalg.norm(np.subtract(e["to"], e["from"]))) for e in ep["shift"] if e["face"]]
                if fv["B"]:
                    d = fv["B"]["dist_m"]
                    blocks = [np.asarray(spec.ctx_box_xy(ep["context"], i), dtype=float) for i in range(spec.n_boxes)]
                    rng = np.random.default_rng([args.seed, ci, 2])
                    rec = {"from": hand0.tolist(), "to": None, "angle_deg": None, "tries": 0}
                    for th in rng.uniform(0, 2 * np.pi, 128):
                        q = hand0 + d * np.array([np.cos(th), np.sin(th)])
                        if not path_clear(spec, hand0, q, blocks, radii, walls):
                            continue
                        rec["tries"] += 1
                        spec.reset(env, ep["context"])           # execute and check in the simulator
                        before = [np.asarray(b[0][:2]) for b in spec.read(env).boxes]
                        spec.move_hand(env, q)
                        st = spec.read(env)
                        moved = max([np.linalg.norm(np.asarray(b[0][:2]) - b0) for b, b0 in zip(st.boxes, before)] or [0.0])
                        err = float(np.linalg.norm(st.c[:2] - q))
                        if moved < BLOCK_TOL and err < HAND_TOL:
                            rec.update(to=q.tolist(), angle_deg=float(np.rad2deg(th)), hand_error_m=err, block_moved_m=float(moved))
                            ep["robot_start"], ep["hand_actual"] = q.tolist(), st.c[:2].tolist()
                            break
                        if rec["tries"] >= 8:
                            break
                    if ep["robot_start"] is None:
                        n_b_fail += 1
                    ep["robot"] = rec
                episodes.append(ep)
            data = {"task": task, "cond": cond, "version": VERSION, "seed": args.seed, "factors": fv, "created_utc": stamp,
                    "demo_hand_start": hand0.tolist(), "episodes": episodes}
            with open(os.path.join(out_dir, f"{cond}.pkl"), "wb") as fh:
                pickle.dump(data, fh)
            info = {"factors": fv, "file": f"{cond}.pkl"}
            if fv["A"]:
                info["A_blocks_not_shifted"] = n_a_fail
                info["A_mean_move_m"] = float(np.mean(a_moved)) if a_moved else None
            if fv["B"]:
                info["B_episodes_without_robot_shift"] = n_b_fail
                errs = [e["robot"].get("hand_error_m") for e in episodes if e["robot_start"] is not None]
                info["B_max_hand_error_m"] = float(max(errs)) if errs else None
            manifest["conditions"][cond] = info
            print(f"  {cond:10s} " + "  ".join(f"{k}={v}" for k, v in info.items() if k not in ("factors", "file")), flush=True)
        with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=1)
        print("saved", out_dir, flush=True)


if __name__ == "__main__":
    main()
