"""CLI glue: build a feasible set + injector from argparse options.

    --constraints a,b,...            which sets form C (intersection):
                                       action_box   |a_i| <= a_max
                                       step_norm    ||a_t||_2 <= r
                                       workspace    desired hand inside [ws_lo, ws_hi]
                                       tracking     |des_t - hand_0|_inf <= track_d
    --constraint-params JSON         overrides for the parameters below
    --sampler euler|pc               pc = method ② Predict-then-Correct (CBF)
    --pc-constraints a,b             barrier sets for ② (default: same as --constraints)
    --pc-params JSON                 alpha, eps, rho, delta, tp, tc (see constraints/pc.py)

Default parameters are derived from the run's ``normalizer.npz`` so they are
never 'magic numbers': ``a_max`` is the per-axis action range seen in training
(0.00833 m), and the workspace box is the training range of the desired hand
position.  ``track_d`` defaults to 0.10 m (the baseline README counts a
desired–actual hand gap above 10 cm as a failure pattern).
"""
from __future__ import annotations

import json

import numpy as np
import torch

from .inject import ConstrainedPolicy, Injector, RawSpace
from .pc import PCSampler
from .sets import ActionBox, Intersection, StepNorm, TrackingBox, WorkspaceBox

SET_NAMES = ("action_box", "step_norm", "workspace", "tracking")


def add_constraint_args(p):
    p.add_argument("--constraints", default="action_box",
                   help="comma-separated subset of %s" % ",".join(SET_NAMES))
    p.add_argument("--constraint-params", default="{}", help="JSON overrides, e.g. '{\"track_d\": 0.08}'")
    p.add_argument("--sampler", choices=["euler", "pc"], default="euler",
                   help="pc: method ② Predict-then-Correct with a finite-time CBF")
    p.add_argument("--pc-constraints", default=None, help="barrier sets for ② (default: --constraints)")
    p.add_argument("--pc-params", default="{}", help='JSON, e.g. \'{"alpha": 2.0, "rho": 0.9, "delta": 0.01, "tp": 1}\'')
    return p


def default_params(norm, act_dim=2):
    """Parameters implied by the training data (normalizer center/scale)."""
    center, scale = np.asarray(norm.center), np.asarray(norm.scale)
    a_max = float(scale[:act_dim].max())
    des_lo = (center - scale)[act_dim:act_dim + 2]
    des_hi = (center + scale)[act_dim:act_dim + 2]
    return {
        "a_max": a_max,
        "r": a_max * np.sqrt(2.0),          # diagonal step of the data = a_max on both axes
        "ws_lo": des_lo.tolist(),
        "ws_hi": des_hi.tolist(),
        "track_d": 0.10,
        "dykstra_iters": 200,   # only used when step_norm is combined with box sets
    }


def build_set(names, params):
    sets = []
    for n in names:
        if n == "action_box":
            sets.append(ActionBox(params["a_max"]))
        elif n == "step_norm":
            sets.append(StepNorm(params["r"]))
        elif n == "workspace":
            sets.append(WorkspaceBox(params["ws_lo"], params["ws_hi"]))
        elif n == "tracking":
            sets.append(TrackingBox(params["track_d"]))
        else:
            raise SystemExit(f"unknown constraint set '{n}', choose from {SET_NAMES}")
    if not sets:
        raise SystemExit("--constraints is empty")
    # always an Intersection: box-type sets become one exact cached QP
    return Intersection(sets, iters=params["dykstra_iters"])


def make_space(norm, act_dim, device):
    center = torch.from_numpy(np.asarray(norm.center, dtype=np.float32)).to(device)
    scale = torch.from_numpy(np.asarray(norm.scale, dtype=np.float32)).to(device)
    return RawSpace(center, scale, act_dim)


def build_pc_sampler(names, params, pc_params, norm, act_dim, device):
    return PCSampler(build_set(names, params).sets, make_space(norm, act_dim, device), **pc_params)


def _names(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def build_policy_from_args(args, policy):
    """Wrap a baseline ``Policy``.  Returns (policy_or_wrapper, description_dict)."""
    params = default_params(policy.norm, policy.act_dim)
    params.update(json.loads(args.constraint_params))
    names = _names(args.constraints)
    if args.sampler != "pc":
        return policy, {"name": "none"}
    pc_names = _names(args.pc_constraints) if args.pc_constraints else names
    sampler = build_pc_sampler(pc_names, params, json.loads(args.pc_params), policy.norm, policy.act_dim, policy.device)
    wrapped = ConstrainedPolicy(policy, Injector(), sampler)
    return wrapped, wrapped.describe()


def tag_suffix(args):
    if args.sampler != "pc":
        return ""
    tag = "_pc_" + args.constraints.replace(",", "+")
    if args.pc_constraints:
        tag += "_pcC-" + args.pc_constraints.replace(",", "+")
    return tag
