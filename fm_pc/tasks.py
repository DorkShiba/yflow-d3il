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

    HAND_QUAT = [0.0, 1.0, 0.0, 0.0]               # rod pointing down (all rod tasks)

    def move_hand(self, env, xy, duration=2.0):
        """OOD-B: drive the hand (same height and orientation) to xy before the episode starts
        (2 s: the impedance controller settles within ~3 mm for a 10 cm move; 1 s leaves up to 2 cm)."""
        z = float(np.asarray(env.robot.current_c_pos)[2])
        env.robot.gotoCartPositionAndQuat(desiredPos=[float(xy[0]), float(xy[1]), z], desiredQuat=self.HAND_QUAT, duration=duration)

    def pushable(self, boxes, hand_xy):
        """Indices of the blocks OOD-3 may teleport (default: all)."""
        return list(range(len(boxes)))

    def block_radii(self):
        """Circumscribed radius of each block (any yaw), in obs order."""
        return [self.block_half * np.sqrt(2.0)] * self.n_boxes

    def hand_clear(self, xy, old_xy, yaw, hand_xy, r, clear):
        """May a block (bounding radius r) be placed at xy, given the hand?  Default: no overlap."""
        return np.linalg.norm(xy - hand_xy) >= r + self.hand_r + clear

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


# ======================================================================================
class Avoiding(TaskSpec):
    """No movable objects; the start pose is fixed and nothing is randomised at reset.
    A 'context' is one rollout index (None), or ``{"start": [x, y]}`` for OOD-1.
    OOD on the hand: shift = start pose moved d away from the demo start,
    obsnoise = noise on the measured hand xy, push = desired hand displaced by d."""
    name, obs_dim, act_dim, n_boxes = "avoiding", 4, 2, 0
    START = (0.525, -0.28)
    # six static cylinders (centre, AABB half-extent = radius)
    OBSTACLES = [((0.50, -0.10), 0.03), ((0.425, 0.08), 0.025), ((0.575, 0.08), 0.025),
                 ((0.35, 0.26), 0.025), ((0.50, 0.26), 0.025), ((0.65, 0.26), 0.025)]
    workspace = ((0.25, 0.80), (-0.35, 0.33))          # below the finish line y = 0.35
    max_steps, push_step = 250, 24

    def make_env(self, max_steps):
        from envs.gym_avoiding_env.gym_avoiding.envs.avoiding import ObstacleAvoidanceEnv
        env = ObstacleAvoidanceEnv(render=False, n_substeps=35, max_steps_per_episode=max_steps)
        env.start()
        return env

    def contexts(self, args, cfg):
        n = args.n_contexts or 100
        return [(f"rollout_{i:03d}", None) for i in range(n)]

    def reset(self, env, ctx):
        env.reset()
        if isinstance(ctx, dict) and "start" in ctx:
            self.move_hand(env, ctx["start"], duration=0.5)

    def walls(self, env):
        return [(np.asarray(c, dtype=float), np.array([r, r])) for c, r in self.OBSTACLES]

    def shift_start(self, env, ctx, ood):
        """OOD-1: start d away from the (fixed) demo start, sideways or backwards (seeded)."""
        d = ood.args.ood_shift
        dirs = [("x-", (-1.0, 0.0)), ("x+", (1.0, 0.0)), ("y-", (0.0, -1.0))]
        name, v = dirs[int(ood.rng.integers(len(dirs)))]
        start = [self.START[0] + d * v[0], self.START[1] + d * v[1]]
        ood.events[ood.key] = {"shift": [{"face": name, "from": list(self.START), "to": start}]}
        return {"start": start}

    def outcome(self, info):
        enc, succ = info if isinstance(info, tuple) else (np.zeros(9), False)
        enc = np.asarray(enc).astype(int)
        return {"success": float(bool(succ)), "mode": int(enc.dot(1 << np.arange(len(enc))))}

    def summarize(self, out):
        s, m = out["success"].ravel(), out["mode"].ravel()
        r = {"success_rate": float(s.mean()), "entropy": None}
        ok = m[s == 1]
        if len(ok):
            _, cnt = np.unique(ok, return_counts=True)
            p = cnt / cnt.sum()
            r["entropy"] = float(-(p * np.log(p)).sum() / np.log(24))
            r["score"] = 0.8 * r["success_rate"] + 0.2 * r["entropy"]
        return r


class Aligning(TaskSpec):
    """Push a 10 cm open box onto a target pose.  Block 0 = push box, block 1 = target
    (visual only: part of the context and the observation, never teleported)."""
    name, obs_dim, act_dim, n_boxes = "aligning", 14, 3, 2
    box_attrs = ("push_box", "target_box")
    INIT_RANGE = [((0.40, 0.60), (-0.25, -0.10)), ((0.40, 0.60), (0.20, 0.35))]
    workspace = ((0.25, 0.80), (-0.40, 0.50))
    block_half = 0.055
    max_steps, push_step = 400, 80

    def make_env(self, max_steps):
        from envs.gym_aligning_env.gym_aligning.envs.aligning import Robot_Push_Env
        env = Robot_Push_Env(render=False, n_substeps=35, max_steps_per_episode=max_steps, if_vision=False)
        env.start()
        return env

    def contexts(self, args, cfg):
        f = args.contexts_file or os.path.join(args.d3il_root, "environments/dataset/data/aligning", f"{args.contexts}_contexts.pkl")
        ctx = np.load(f, allow_pickle=True)
        n = len(ctx) if not args.n_contexts else min(args.n_contexts, len(ctx))
        return [(f"{args.contexts}_{i:03d}", ctx[i]) for i in range(n)]

    def pushable(self, boxes, hand_xy):
        return [0]

    # the push box is open (walls 5 mm thick, inner half 4.5 cm); the rod (r 1 cm) works inside it
    OUTER, INNER, ROD_R = 0.05, 0.045, 0.01

    def hand_clear(self, xy, old_xy, yaw, hand_xy, r, clear):
        """Rod inside the box: the box may move only so that the rod stays inside (5 mm from the walls)."""
        def rel(c):
            c_, s_ = np.cos(yaw), np.sin(yaw)
            d = hand_xy - np.asarray(c, dtype=float)
            return np.abs(np.array([c_ * d[0] + s_ * d[1], -s_ * d[0] + c_ * d[1]]))
        if old_xy is not None and rel(old_xy).max() < self.OUTER:
            return rel(xy).max() <= self.INNER - self.ROD_R - 0.005
        return super().hand_clear(xy, old_xy, yaw, hand_xy, r, clear)

    def init_des(self, env, s):
        return s.c[:3].copy(), np.zeros(0)

    def obs(self, des, s):
        return np.concatenate([np.asarray(des[:3]), s.c[:3], box_feats(s.boxes)]).astype(np.float32)

    def command(self, des, fixed):
        return np.concatenate([des, [0.0, 1.0, 0.0, 0.0]])

    def summarize(self, out):
        s, m = out["success"], out["mode"]
        ent = mode_entropy(m, s, range(2))
        r = {"success_rate": float(s.mean()), "entropy": ent, "mean_distance": float(np.nanmean(out["mean_distance"]))}
        if ent is not None:
            r["score"] = 0.5 * (r["success_rate"] + ent)
        return r


class Sorting(TaskSpec):
    """Sort k red and k blue blocks into their regions.  The six start slots are shuffled
    per context, so every block shares one demo start region (bounding box of the slots).
    Context = list of 6 [pos (x, y, deg), quat]; red_1..red_k take ctx[0:k], blue ctx[k:2k]."""
    act_dim = 2
    INIT_BOX = ((0.40, 0.65), (-0.15, 0.10))
    # static walls of the two target regions (centre, half-extent), sorting_objects.py
    TARGET_WALLS = [((0.4, 0.41), (0.1, 0.01)), ((0.3, 0.32), (0.005, 0.1)), ((0.5, 0.32), (0.005, 0.1)),
                    ((0.4, 0.22), (0.1, 0.005)), ((0.625, 0.41), (0.1, 0.01)), ((0.525, 0.32), (0.005, 0.1)),
                    ((0.725, 0.32), (0.005, 0.1)), ((0.625, 0.22), (0.1, 0.005))]
    workspace = ((0.25, 0.80), (-0.35, 0.45))
    block_half = 0.03
    STEPS = {2: (500, 100), 4: (700, 150), 6: (1200, 250)}

    def __init__(self, n):
        super().__init__()
        self.n = n
        self.k = n // 2
        self.name = f"sorting_{n}"
        self.n_boxes = n
        self.obs_dim = 4 + 4 * n
        self.box_attrs = tuple(f"red_box_{i}" for i in range(1, self.k + 1)) + tuple(f"blue_box_{i}" for i in range(1, self.k + 1))
        self.INIT_RANGE = [self.INIT_BOX] * n
        self.max_steps, self.push_step = self.STEPS[n]

    def make_env(self, max_steps):
        from envs.gym_sorting_env.gym_sorting.envs.sorting import Sorting_Env
        env = Sorting_Env(render=False, n_substeps=35, max_steps_per_episode=max_steps, num_boxes=self.n, if_vision=False)
        env.start()
        return env

    def contexts(self, args, cfg):
        """Official files are generated by D3IL's get_contexts.py (not shipped); see make_sorting_contexts.py."""
        import pickle
        base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fm_baseline", "contexts")
        f = args.contexts_file or os.path.join(base, f"sorting_{self.n}_{args.contexts}_contexts.pkl")
        with open(f, "rb") as fh:
            ctx = pickle.load(fh)
        n = min(args.n_contexts or 30, len(ctx))
        return [(f"{args.contexts}_{i:03d}", ctx[i]) for i in range(n)]

    def reset(self, env, ctx):
        env.reset(random=False, context=copy.deepcopy(ctx), if_vision=False)

    def ctx_box_xy(self, ctx, i):
        return np.asarray(ctx[i][0][:2], dtype=float)

    def ctx_set_box_xy(self, ctx, i, xy):
        ctx = [list(e) for e in ctx]
        ctx[i][0] = np.array([xy[0], xy[1], ctx[i][0][2]])
        return ctx

    def walls(self, env):
        return [(np.asarray(c, dtype=float), np.asarray(h, dtype=float)) for c, h in self.TARGET_WALLS]

    def outcome(self, info):
        return {"success": float(bool(info.get("success", False))), "mode": int(info.get("mode", -1))}

    def summarize(self, out):
        s, m = out["success"], out["mode"]
        from math import comb
        ids = sorted(set(m[s == 1].astype(int).tolist())) or [0]
        n_mode = comb(self.n, self.k)            # orders of k red and k blue placements
        r = {"success_rate": float(s.mean()), "n_modes_seen": len(ids), "n_modes_possible": n_mode}
        if s.shape[1] > 1:
            ent = []
            for ci in range(s.shape[0]):
                p = np.array([np.sum((m[ci] == k) & (s[ci] == 1)) for k in ids], dtype=float)
                if p.sum() > 0:
                    p /= p.sum()
                    ent.append(float(-(p * np.log(p + 1e-12)).sum() / np.log(n_mode)))
            r["entropy"] = float(np.mean(ent)) if ent else 0.0
        return r


class Stacking(TaskSpec):
    """Stack red, green, blue on the target (joint-position control + binary gripper).
    obs = [desired joints 7, gripper 1, per block (x, y, z, sin yaw, cos yaw)];
    action = [desired-joint increment 7, gripper width (absolute)].
    Context = 4 [pos (x, y, deg), quat] pairs: red, green, blue, target (target is ignored by D3IL)."""
    name, obs_dim, act_dim, n_boxes = "stacking", 23, 8, 3
    box_attrs = ("red_box", "green_box", "blue_box")
    INIT_RANGE = [((0.35, 0.45), (-0.25, -0.15)), ((0.35, 0.45), (-0.10, 0.00)), ((0.55, 0.60), (-0.20, 0.00))]
    TARGET = ((0.5, 0.2), (0.05, 0.05))
    workspace = ((0.25, 0.75), (-0.35, 0.35))
    block_half = 0.05                     # blue is 6 x 10 cm (radii per block below)
    HALF = [(0.03, 0.03), (0.03, 0.03), (0.03, 0.05)]   # stacking_objects.py: red, green 6 cm cubes, blue 6 x 10 cm
    hand_r = 0.06                         # gripper fingers
    HAND_HIGH = 0.12                      # tcp above this clears any block resting on the table (6 cm tall)
    ON_TABLE_Z = 0.045                    # resting centre z ~0.03; above this a block is held or stacked
    max_steps, push_step = 1000, 150
    motion_dims, dt = 7, 0.030            # jerk metrics on the 7 joints; n_substeps 30 x 1 ms

    def make_env(self, max_steps):
        from envs.gym_stacking_env.gym_stacking.envs.stacking import CubeStacking_Env
        env = CubeStacking_Env(render=False, max_steps_per_episode=max_steps)
        env.start()
        return env

    def contexts(self, args, cfg):
        f = args.contexts_file or os.path.join(args.d3il_root, "environments/dataset/data/stacking", f"{args.contexts}_contexts.pkl")
        ctx = np.load(f, allow_pickle=True)
        n = len(ctx) if not args.n_contexts else min(args.n_contexts, len(ctx))
        return [(f"{args.contexts}_{i:03d}", ctx[i]) for i in range(n)]

    def ctx_box_xy(self, ctx, i):
        return np.asarray(ctx[i][0][:2], dtype=float)

    def ctx_set_box_xy(self, ctx, i, xy):
        ctx = [list(e) for e in ctx]
        ctx[i][0] = np.array([xy[0], xy[1], ctx[i][0][2]])
        return ctx

    def walls(self, env):
        return [(np.asarray(self.TARGET[0], dtype=float), np.asarray(self.TARGET[1], dtype=float))]

    def pushable(self, boxes, hand_xy):
        """Only blocks resting on the table and not between the fingers."""
        return [i for i, (p, _q, _y) in enumerate(boxes)
                if p[2] < self.ON_TABLE_Z and np.linalg.norm(np.asarray(p[:2]) - hand_xy) > self.hand_r + 0.02]

    def block_radii(self):
        return [float(np.hypot(*h)) for h in self.HALF]

    def move_hand(self, env, xy, duration=1.0):
        raise NotImplementedError("Stacking OOD-B needs the gripper orientation / IK; not implemented yet")

    def hand_clear(self, xy, old_xy, yaw, hand_xy, r, clear):
        if self._hand_z is not None and self._hand_z > self.HAND_HIGH:
            return True
        return super().hand_clear(xy, old_xy, yaw, hand_xy, r, clear)

    _hand_z = None

    def read(self, env):
        s = super().read(env)
        self._hand_z = float(s.c[2])
        env.robot.receiveState()
        s.extra["j"] = np.asarray(env.robot.current_j_pos, dtype=np.float64).copy()
        s.extra["g"] = float(env.robot.gripper_width)
        return s

    def init_des(self, env, s):
        return np.r_[s.extra["j"], s.extra["g"]], np.zeros(0)

    def obs(self, des, s):
        feats = [v for pos, _q, yaw in s.boxes for v in (pos[0], pos[1], pos[2], np.sin(yaw), np.cos(yaw))]
        return np.concatenate([np.asarray(des[:8]), feats]).astype(np.float32)

    def apply(self, des, a):
        a = np.asarray(a, dtype=np.float64)
        return np.r_[des[:7] + a[:7], a[7]]

    def command(self, des, fixed):
        return np.asarray(des, dtype=np.float64)

    def log_state(self, s):
        return np.concatenate([np.r_[p[:3], y] for p, _q, y in s.boxes] + [[s.extra["g"]]])

    MODE3 = {"rgb": 0, "rbg": 1, "grb": 2, "gbr": 3, "brg": 4, "bgr": 5}

    def outcome(self, info):
        mode = str(info.get("mode", ""))
        return {"success": float(bool(info.get("success", False))), "success_1": float(bool(info.get("success_1", False))),
                "success_2": float(bool(info.get("success_2", False))), "mode": self.MODE3.get(mode[:3], -1)}

    def summarize(self, out):
        s, s1, s2 = out["success"], out["success_1"], out["success_2"]
        r = {"success_rate": float(s.mean()), "success_1_rate": float(s1.mean()), "success_2_rate": float(s2.mean()),
             "entropy": mode_entropy(out["mode"], s, range(6))}
        r["score"] = r["success_1_rate"] + r["success_2_rate"] + r["success_rate"]
        return r


TASKS = {"pushing": Pushing, "inserting": Inserting, "avoiding": Avoiding, "aligning": Aligning,
         "sorting_2": lambda: Sorting(2), "sorting_4": lambda: Sorting(4), "sorting_6": lambda: Sorting(6),
         "stacking": Stacking}


def get_task(name):
    if name not in TASKS:
        raise SystemExit(f"unknown task {name}; available: {sorted(TASKS)}")
    return TASKS[name]()
