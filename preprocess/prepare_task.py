"""Preprocess D3IL Avoiding / Aligning / Sorting (2, 4, 6) / Stacking for the flow-matching baseline.

Same output format as prepare_pushing.py (one npz per demo with obs, actions,
next_obs; manifest.json) and the same conventions: block yaw as (sin, cos),
action = increment of the desired command.  The raw pickle keys are the ones the
official D3IL loaders read (environments/dataset/*_dataset.py).  Layouts must
match fm_pc/tasks.py (the evaluator builds the same observation online).

  task       obs                                                         action
  avoiding   des xy, hand xy                                        (4)  Δ des xy (2)
  aligning   des xyz, hand xyz, box (x y sin cos), target (x y sin cos) (14)  Δ des xyz (3)
  sorting_n  des xy, hand xy, red_1..k, blue_1..k (x y sin cos)    (4+4n)  Δ des xy (2)
  stacking   des joints 7, gripper width, red/green/blue (x y z sin cos) (23)  Δ des joints 7, gripper width[t+1] (8)

Splits: the official train/eval file lists where D3IL ships them (aligning
900/100, stacking 1000/95: environments/dataset/data/<task>/{train,eval}_files.pkl;
sorting: {n}_boxes_{train,eval}_files.pkl if present -- not shipped).  Otherwise
(avoiding has no split; sorting lists come from D3IL's split_data.py, whose order
depends on os.listdir) the file names are SORTED and then split like split_data.py:
random.seed(42), shuffle, first 10 % -> eval.  The list used is written to the manifest.

    python preprocess/prepare_task.py --task stacking --raw /root/d3il-work/raw/stacking/all_data \\
        --d3il /root/d3il-work/d3il --out data/stacking_sincos_v1
"""
import argparse
import json
import os
import pickle
import random
from pathlib import Path

import numpy as np

TASKS = ("avoiding", "aligning", "sorting_2", "sorting_4", "sorting_6", "stacking")


def load_geo(d3il):
    import importlib.util
    spec = importlib.util.spec_from_file_location("d3il_geo", Path(d3il) / "environments/dataset/geo_transform.py")
    geo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geo)
    return geo


def yaw(geo, quat):
    q = np.asarray(quat, dtype=np.float64)
    return np.array([float(np.asarray(geo.quat2euler(r)).reshape(-1)[-1]) for r in q])


def xy_sc(geo, d, key, z=False):
    pos = np.asarray(d[key]["pos"], dtype=np.float64)
    y = yaw(geo, d[key]["quat"])
    cols = [pos[:, 0], pos[:, 1]] + ([pos[:, 2]] if z else []) + [np.sin(y), np.cos(y)]
    return np.stack(cols, axis=1)


def build(task, d, geo):
    """(input_state (T, obs_dim), command (T, cmd_dim)) -> obs = state[:-1], action from command."""
    r = d["robot"]
    if task == "avoiding":
        des = np.asarray(r["des_c_pos"], dtype=np.float64)[:, :2]
        st = np.concatenate([des, np.asarray(r["c_pos"], dtype=np.float64)[:, :2]], 1)
        return st, des[1:] - des[:-1]
    if task == "aligning":
        des = np.asarray(r["des_c_pos"], dtype=np.float64)[:, :3]
        st = np.concatenate([des, np.asarray(r["c_pos"], dtype=np.float64)[:, :3],
                             xy_sc(geo, d, "push-box"), xy_sc(geo, d, "target-box")], 1)
        return st, des[1:] - des[:-1]
    if task.startswith("sorting_"):
        k = int(task.split("_")[1]) // 2
        des = np.asarray(r["des_c_pos"], dtype=np.float64)[:, :2]
        parts = [des, np.asarray(r["c_pos"], dtype=np.float64)[:, :2]]
        parts += [xy_sc(geo, d, f"red-box{i}") for i in range(1, k + 1)]
        parts += [xy_sc(geo, d, f"blue-box{i}") for i in range(1, k + 1)]
        return np.concatenate(parts, 1), des[1:] - des[:-1]
    if task == "stacking":
        dj = np.asarray(r["des_j_pos"], dtype=np.float64)
        g = np.asarray(r["gripper_width"], dtype=np.float64).reshape(-1, 1)
        st = np.concatenate([dj, g] + [xy_sc(geo, d, f"{c}-box", z=True) for c in ("red", "green", "blue")], 1)
        return st, np.concatenate([dj[1:] - dj[:-1], g[1:]], 1)
    raise ValueError(task)


FIELDS = {
    "avoiding": ["des.x", "des.y", "hand.x", "hand.y"],
    "aligning": ["des.x", "des.y", "des.z", "hand.x", "hand.y", "hand.z",
                 "box.x", "box.y", "box.sin_yaw", "box.cos_yaw", "target.x", "target.y", "target.sin_yaw", "target.cos_yaw"],
    "stacking": [f"des_j{i}" for i in range(7)] + ["gripper_width"]
                + [f"{c}.{f}" for c in ("red", "green", "blue") for f in ("x", "y", "z", "sin_yaw", "cos_yaw")],
}


def fields(task):
    if task.startswith("sorting_"):
        k = int(task.split("_")[1]) // 2
        boxes = [f"red_{i}" for i in range(1, k + 1)] + [f"blue_{i}" for i in range(1, k + 1)]
        return ["des.x", "des.y", "hand.x", "hand.y"] + [f"{b}.{f}" for b in boxes for f in ("x", "y", "sin_yaw", "cos_yaw")]
    return FIELDS[task]


def splits(task, raw, d3il):
    meta = Path(d3il) / "environments/dataset/data" / task.split("_")[0]
    if task.startswith("sorting_"):
        n = task.split("_")[1]
        cands = (meta / f"{n}_boxes_train_files.pkl", meta / f"{n}_boxes_eval_files.pkl")
    else:
        cands = (meta / "train_files.pkl", meta / "eval_files.pkl")
    if all(c.is_file() for c in cands):
        lists = []
        for c in cands:
            with open(c, "rb") as fh:
                lists.append([x.decode() if isinstance(x, bytes) else str(x) for x in pickle.load(fh)])
        return {"train": lists[0], "val": lists[1]}, f"official {cands[0].name} / {cands[1].name}"
    names = sorted(f for f in os.listdir(raw) if f.endswith(".pkl"))
    rng = random.Random(42)
    rng.shuffle(names)
    k = int(len(names) * 0.1)
    return {"train": names[k:], "val": names[:k]}, "sorted file names, random.seed(42) shuffle, first 10% -> val (split_data.py style)"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=TASKS)
    p.add_argument("--raw", required=True, help="directory with the raw demo pickles")
    p.add_argument("--d3il", required=True, help="official D3IL repository (geo_transform.py, split lists)")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0, help="first N files per split (testing)")
    args = p.parse_args()
    raw, out = Path(args.raw), Path(args.out)
    geo = load_geo(args.d3il)
    sp, how = splits(args.task, raw, args.d3il)
    out.mkdir(parents=True, exist_ok=False)            # never overwrite an existing dataset
    names_f = fields(args.task)
    manifest = {"task": args.task, "obs_dim": len(names_f), "action_dim": None, "obs_fields": names_f,
                "normalized": False, "split_method": how, "source_root": str(raw), "splits": {}}
    for split, files in sp.items():
        if args.limit:
            files = files[:args.limit]
        (out / split).mkdir()
        recs = []
        for name in files:
            with open(raw / name, "rb") as fh:
                d = pickle.load(fh)
            st, act = build(args.task, d, geo)
            if not (np.isfinite(st).all() and np.isfinite(act).all()):
                raise ValueError(f"{name}: non-finite values")
            obs, nxt = st[:-1], st[1:]
            assert len(obs) == len(act) and obs.shape[1] == len(names_f)
            f = f"{split}/{Path(name).stem}.npz"
            np.savez(out / f, obs=obs.astype(np.float32), actions=act.astype(np.float32), next_obs=nxt.astype(np.float32))
            recs.append({"file": f, "source": name, "n": int(len(act))})
            manifest["action_dim"] = int(act.shape[1])
        manifest["splits"][split] = recs
        print(f"{split}: {len(recs)} demos, {sum(r['n'] for r in recs)} pairs")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("saved", out)


if __name__ == "__main__":
    main()
