"""Reproducible test contexts for D3IL Sorting (2 / 4 / 6 boxes).

D3IL's ``environments/dataset/data/sorting/get_contexts.py`` seeds ``random`` and
``numpy`` with 0, but the slot samplers are gym ``Box`` spaces with their own,
unseeded generators, so its ``{n}_test_contexts.pkl`` cannot be regenerated and
is not shipped.  This script calls the same sampler (``env.manager.sample()``:
six start slots, uniform position and yaw, then a Python ``random.shuffle``) after
seeding every slot space, and writes ``fm_baseline/contexts/sorting_{n}_test_contexts.pkl``.

    python fm_pc/make_sorting_contexts.py --num-boxes 2 4 6 [--d3il-root ...]
"""
import argparse
import os
import pickle
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "fm_baseline"))
sys.path.insert(0, HERE)
from tasks import get_task
from train import resolve_d3il


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-boxes", type=int, nargs="+", default=[2, 4, 6])
    p.add_argument("--n", type=int, default=1200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--d3il-root", default=None)
    p.add_argument("--out-dir", default=os.path.join(REPO, "fm_baseline", "contexts"))
    args = p.parse_args()
    out_dir = os.path.abspath(args.out_dir)
    d3il = resolve_d3il(args.d3il_root)
    for n in args.num_boxes:
        spec = get_task(f"sorting_{n}")
        spec.setup(d3il)
        env = spec.make_env(spec.max_steps)
        m = env.manager
        for i in range(1, 7):
            getattr(m, f"box{i}_space").seed(1000 * args.seed + i)
        random.seed(args.seed)
        np.random.seed(args.seed)
        ctx = [m.sample() for _ in range(args.n)]
        os.makedirs(out_dir, exist_ok=True)
        f = os.path.join(out_dir, f"sorting_{n}_test_contexts.pkl")
        with open(f, "wb") as fh:
            pickle.dump(ctx, fh)
        print(f"sorting_{n}: {len(ctx)} contexts -> {f}; first: {[np.round(e[0], 3).tolist() for e in ctx[0][:n]]}")


if __name__ == "__main__":
    main()
