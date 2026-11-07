"""Check that prepare_task.py builds the same observation layout the evaluator builds online.

Runs a short episode per task in the simulator (small random actions), records the
state in the raw D3IL pickle format (robot des_c_pos / c_pos / des_j_pos /
gripper_width, <box>.pos / .quat), converts it with prepare_task.build and compares
with tasks.TaskSpec.obs on the same states.  Needs the simulator, not the dataset.

    python preprocess/check_prepare_task.py [--tasks avoiding aligning ...]
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [HERE, REPO, os.path.join(REPO, "fm_baseline")]
import prepare_task
from fm_constraint.tasks import get_task
from train import resolve_d3il

RAW_KEYS = {
    "avoiding": [], "aligning": ["push-box", "target-box"], "stacking": ["red-box", "green-box", "blue-box"],
}


def raw_keys(task):
    if task.startswith("sorting_"):
        k = int(task.split("_")[1]) // 2
        return [f"red-box{i}" for i in range(1, k + 1)] + [f"blue-box{i}" for i in range(1, k + 1)]
    return RAW_KEYS[task]


def run(task, d3il, steps, seed):
    spec = get_task(task)
    spec.setup(d3il)
    env = spec.make_env(steps + 5)
    if task == "avoiding":
        ctx = None
    else:
        ctx = spec.contexts(argparse.Namespace(contexts_file=None, contexts="test", n_contexts=1, d3il_root=d3il), {})[0][1]
    spec.reset(env, ctx)
    s = spec.read(env)
    des, fixed = spec.init_des(env, s)
    rng = np.random.default_rng(seed)
    rec = {"robot": {"des_c_pos": [], "c_pos": [], "des_j_pos": [], "gripper_width": []}}
    for k in raw_keys(task):
        rec[k] = {"pos": [], "quat": []}
    online = []
    for t in range(steps):
        online.append(spec.obs(des, s))
        r = rec["robot"]
        if task == "stacking":
            r["des_j_pos"].append(des[:7].copy()); r["gripper_width"].append(des[7])
        else:
            r["des_c_pos"].append(np.r_[des, fixed][:3].copy()); r["c_pos"].append(s.c.copy())
        for k, b in zip(raw_keys(task), spec.box_objs(env)):
            rec[k]["pos"].append(np.asarray(env.scene.get_obj_pos(b), dtype=np.float64).copy())
            rec[k]["quat"].append(np.asarray(env.scene.get_obj_quat(b), dtype=np.float64).copy())
        a = rng.uniform(-1, 1, spec.act_dim) * (0.01 if task == "stacking" else 0.005)
        if task == "stacking":
            a[7] = des[7]                                   # gripper width is absolute; keep it
        des = spec.apply(des, a)
        env.step(spec.command(des, fixed))
        s = spec.read(env)
    rec = {k: {f: np.asarray(v) for f, v in d.items() if len(v)} for k, d in rec.items()}
    st, act = prepare_task.build(task, rec, prepare_task.load_geo(d3il))
    online = np.asarray(online, dtype=np.float64)
    err = np.abs(st - online).max()
    names = prepare_task.fields(task)
    ok = st.shape == online.shape == (steps, spec.obs_dim) and len(names) == spec.obs_dim and act.shape[1] == spec.act_dim and err < 1e-5
    print(f"{task:10s} obs {st.shape} act {act.shape} fields {len(names)}  max |offline - online| {err:.2e}  {'OK' if ok else 'MISMATCH'}")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=list(prepare_task.TASKS))
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--d3il-root", default=None)
    args = p.parse_args()
    d3il = resolve_d3il(args.d3il_root)
    res = [run(t, d3il, args.steps, i) for i, t in enumerate(args.tasks)]
    sys.exit(0 if all(res) else 1)


if __name__ == "__main__":
    main()
