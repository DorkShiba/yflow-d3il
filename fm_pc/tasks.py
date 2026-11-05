"""Task definitions for the D3IL OOD benchmark (state-based flow-matching policies).

One ``TaskSpec`` per D3IL task gathers everything that differs between tasks,
so the episode loop (``benchmark.run_task_episode``), the OOD conditions
(``ood.OOD``) and the metrics are shared:

    environment class and constructor      make_env
    test contexts                          contexts
    reading the scene                      read      -> State
    observation used for training          obs       (same layout as the preprocessing)
    desired command and action             init_des, apply, command
    objects the OOD conditions move        n_boxes, box_objs, INIT_RANGE, workspace, walls
    success / mode / entropy               outcome, summarize

Conventions shared with the processed data (``preprocess/``): block orientation
enters as (sin yaw, cos yaw); the action is the increment of the desired command
(desired hand XY, or desired joints for Stacking); observations start with the
desired command followed by the measured robot state.
"""
from __future__ import annotations

import copy
import importlib
import importlib.abc
import importlib.util
import os
import sys
from dataclasses import dataclass, field

import numpy as np


@dataclass
class State:
    c: np.ndarray                                   # measured end-effector position (xyz)
    boxes: list = field(default_factory=list)       # [(pos xyz, quat wxyz, yaw rad)] in obs order
    extra: dict = field(default_factory=dict)       # task specific (joints, gripper, target, ...)


class _D3ilSimAlias(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Some D3IL env files import ``d3il_sim.*`` while the simulator uses
    ``environments.d3il.d3il_sim.*``; map both to one module (no D3IL edits)."""

    def find_spec(self, name, path, target=None):
        if name == "d3il_sim" or name.startswith("d3il_sim."):
            return importlib.util.spec_from_loader(name, self)
        return None

    def create_module(self, spec):
        return importlib.import_module("environments.d3il." + spec.name)

    def exec_module(self, module):
        pass


def setup_d3il(root):
    """Import paths for the official D3IL repository (unmodified)."""
    os.chdir(root)
    for p in [root, os.path.join(root, "environments", "d3il")]:
        if p not in sys.path:
            sys.path.insert(0, p)
    importlib.import_module("environments.d3il.d3il_sim")
    if not any(isinstance(f, _D3ilSimAlias) for f in sys.meta_path):
        sys.meta_path.insert(0, _D3ilSimAlias())
    from environments.d3il.d3il_sim.utils.geometric_transformation import quat2euler
    return quat2euler


def yaw_of(quat2euler, quat):
    return float(np.asarray(quat2euler(np.asarray(quat, dtype=np.float64))).reshape(-1)[-1])


def box_feats(boxes):
    return [v for pos, _q, yaw in boxes for v in (pos[0], pos[1], np.sin(yaw), np.cos(yaw))]


def mode_entropy(modes, successes, mode_ids):
    """D3IL per-context entropy of the mode distribution among successful rollouts."""
    n_ctx, n_traj = successes.shape
    k = len(mode_ids)
    if n_traj < 2 or k < 2:
        return None
    ent = []
    for ci in range(n_ctx):
        p = np.array([np.sum((modes[ci] == m) & (successes[ci] == 1)) for m in mode_ids], dtype=float)
        if p.sum() == 0:
            ent.append(0.0)
            continue
        p /= p.sum()
        ent.append(float(-(p * np.log(p + 1e-12)).sum() / np.log(k)))
    return float(np.mean(ent))


class TaskSpec:
    name = "base"
    obs_dim = act_dim = 0
    n_boxes = 0
    box_attrs = ()                    # env attributes holding the movable blocks, in obs order
    INIT_RANGE = []                   # per block ((x0, x1), (y0, y1)) of the demo initial positions
    workspace = ((0.0, 1.0), (-0.5, 0.5))
    max_steps = 400
    push_step = 80
    hand_r = 0.02
    block_half = 0.025

    def __init__(self):
        self.quat2euler = None

    # ---- environment --------------------------------------------------------
    def setup(self, d3il_root):
        self.quat2euler = setup_d3il(d3il_root)

    def make_env(self, max_steps):
        raise NotImplementedError

    def contexts(self, args, cfg):
        """[(name, context)] in evaluation order."""
        raise NotImplementedError

    def reset(self, env, ctx):
        env.reset(random=False, context=copy.deepcopy(ctx))

    def box_objs(self, env):
        return [getattr(env, a) for a in self.box_attrs]

    # context = [pos_1 (x, y, deg), quat_1, pos_2, quat_2, ...] for the D3IL block tasks
    def ctx_box_xy(self, ctx, i):
        return np.asarray(ctx[2 * i][:2], dtype=float)

    def ctx_set_box_xy(self, ctx, i, xy):
        ctx = list(ctx)
        ctx[2 * i] = np.array([xy[0], xy[1], ctx[2 * i][2]])
        return ctx

    def walls(self, env):
        return []

    # ---- state / observation / command ---------------------------------------
    def read(self, env):
        env.robot.receiveState()
        c = np.asarray(env.robot.current_c_pos, dtype=np.float64).copy()
        boxes = []
        for b in self.box_objs(env):
            pos = np.asarray(env.scene.get_obj_pos(b), dtype=np.float64).copy()
            quat = np.asarray(env.scene.get_obj_quat(b), dtype=np.float64).copy()
            boxes.append((pos, quat, yaw_of(self.quat2euler, quat)))
        return State(c=c, boxes=boxes)

    def init_des(self, env, s: State):
        """(desired command, fixed part of the robot command)"""
        return s.c[:2].copy(), s.c[2:3].copy()

    def obs(self, des, s: State):
        return np.concatenate([np.asarray(des[:2]), s.c[:2], box_feats(s.boxes)]).astype(np.float32)

    def apply(self, des, a):
        return des + np.asarray(a, dtype=np.float64)

    def command(self, des, fixed):
        return np.concatenate([des, fixed, [0.0, 1.0, 0.0, 0.0]])

    def hand_xy(self, s: State):
        return s.c[:2]

    def log_state(self, s: State):
        return np.concatenate([np.r_[p[:2], y] for p, _q, y in s.boxes]) if s.boxes else np.zeros(0)

    # ---- outcome -------------------------------------------------------------
    def outcome(self, info):
        return {"success": float(bool(info.get("success", False))), "mode": int(info.get("mode", -1)),
                "mean_distance": float(info.get("mean_distance", np.nan))}

    def summarize(self, out):
        """out: dict of (n_ctx, n_traj) arrays from ``outcome``."""
        s = out["success"]
        return {"success_rate": float(s.mean())}


# ======================================================================================
class Pushing(TaskSpec):
    name, obs_dim, act_dim, n_boxes = "pushing", 12, 2, 2
    box_attrs = ("push_box1", "push_box2")
    INIT_RANGE = [((0.40, 0.50), (-0.15, 0.00)), ((0.55, 0.65), (-0.15, 0.00))]
    workspace = ((0.21, 0.84), (-0.39, 0.49))
    max_steps, push_step = 400, 80

    def make_env(self, max_steps):
        from envs.gym_pushing_env.gym_pushing.envs.pushing import Block_Push_Env
        env = Block_Push_Env(render=False, n_substeps=35, max_steps_per_episode=max_steps)
        env.start()
        return env

    def contexts(self, args, cfg):
        f = os.path.join(args.d3il_root, "environments/dataset/data/pushing", f"{args.contexts}_contexts.pkl")
        ctx = np.load(f, allow_pickle=True)
        n = len(ctx) if not args.n_contexts else min(args.n_contexts, len(ctx))
        return [(f"{args.contexts}_{i:03d}", ctx[i]) for i in range(n)]

    def summarize(self, out):
        s, m = out["success"], out["mode"]
        ent = mode_entropy(m, s, range(4))
        r = {"success_rate": float(s.mean()), "entropy": ent}
        if ent is not None:
            r["score"] = 0.5 * (r["success_rate"] + ent)
        return r


class Inserting(TaskSpec):
    name, obs_dim, act_dim, n_boxes = "inserting", 16, 2, 3
    box_attrs = ("push_box1", "push_box2", "push_box3")
    INIT_RANGE = [((0.35, 0.50), (-0.20, -0.15)), ((0.55, 0.70), (-0.10, -0.05)), ((0.35, 0.50), (0.00, 0.05))]
    workspace = ((0.01, 0.84), (-0.39, 0.74))
    max_steps, push_step = 2000, 300

    def make_env(self, max_steps):
        from envs.gym_inserting_env.gym_inserting.envs.gate_insertion import Gate_Insertion_Env
        env = Gate_Insertion_Env(render=False, n_substeps=35, max_steps_per_episode=max_steps)
        env.start()
        return env

    def walls(self, env):
        out = []
        for i in range(1, 40):
            obj = getattr(env, f"maze_{i}", None)
            if obj is None:
                continue
            try:
                pos = np.asarray(env.scene.get_obj_pos(obj), dtype=float)
            except Exception:                     # defined but not added to the scene (maze_1, maze_2)
                continue
            out.append((pos[:2], np.asarray(obj.size, dtype=float)[:2]))
        return out

    def contexts(self, args, cfg):
        import pickle
        base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fm_baseline")
        f = getattr(args, "contexts_file", None) or os.path.join(base, "contexts", "inserting_val.pkl")
        with open(f, "rb") as fh:
            stored = pickle.load(fh)
        names = sorted(stored)
        if args.n_contexts:
            names = names[:args.n_contexts]
        return [(n, stored[n]["context"]) for n in names]

    def outcome(self, info):
        o = super().outcome(info)
        o["mode"] = int(info.get("mode", 0))
        o["n_boxes"] = int(info.get("one_box_success", 0)) + int(info.get("two_box_success", 0)) + int(info.get("three_box_success", 0))
        return o

    def summarize(self, out):
        s, m = out["success"], out["mode"]
        ent = mode_entropy(m, s, range(1, 7))
        nb = out["n_boxes"]
        return {"success_rate": float(s.mean()), "entropy": ent,
                "one_box_rate": float((nb >= 1).mean()), "two_box_rate": float((nb >= 2).mean()),
                "three_box_rate": float((nb >= 3).mean())}


TASKS = {"pushing": Pushing, "inserting": Inserting}


def get_task(name):
    if name not in TASKS:
        raise SystemExit(f"unknown task {name}; available: {sorted(TASKS)}")
    return TASKS[name]()
