"""OOD benchmark for the D3IL Pushing / Inserting evaluations.

The policy, checkpoint and execution protocol (plan 16, execute 8, Euler 10)
are unchanged; only the test situation moves outside the demonstrations.
Exactly one condition is active per run (``--ood``):

  none      in-distribution reference (bit-identical to the old loop)
  shift     OOD-1  every block starts ``--ood-shift`` metres OUTSIDE the demo
                   range of initial positions, through its nearest feasible face
  obsnoise  OOD-2  Gaussian noise on the block pose the policy sees at every
                   re-plan (x, y: ``--ood-noise-pos`` m, yaw: ``--ood-noise-yaw``
                   deg).  The true state is untouched.
  push      OOD-3  at the first re-plan after step ``--ood-push-step`` the block
                   closest to the hand is teleported ``--ood-push-dist`` metres in
                   a random feasible direction.  Phase ``seam`` (default): right
                   before the re-plan, so the next chunk is planned from the
                   pushed state.  Phase ``mid``: half a chunk earlier, so the old
                   chunk keeps running blind for execute/2 steps.
  delay     OOD-4  inference delay of ``--ood-delay`` control steps: the next
                   chunk is planned from the observation d steps before it starts,
                   and execution resumes at index d of the new chunk (naive async,
                   as the RTC paper's baseline).

Every random draw uses a dedicated ``numpy.random.Generator`` seeded by
(--ood-seed, context, trajectory), so the policy noise streams (torch / numpy
globals) are exactly the ones of the in-distribution run.

Logged per rollout (rollouts.npz):  ``<ep>_seams`` steps where a new chunk starts
executing, ``<ep>_pobs`` the observation each chunk was planned from, ``<ep>_event``
step of the push (−1 if none).  ``results.json`` gets an ``ood`` block with the
settings and every applied shift / push.  ``ood_metrics.py`` scores them.

Placement feasibility (shift, push) is geometric: the block (half-size 2.5 cm,
any yaw -> circumscribed radius) must keep ``CLEAR`` from the other blocks, the
hand, the Inserting maze walls that are in the scene, and stay inside the
research workspace of the constraint study.
"""
from __future__ import annotations

import copy
import json
import time

import numpy as np

KINDS = ("none", "shift", "obsnoise", "push", "delay")

BLOCK_HALF = 0.025
BLOCK_R = BLOCK_HALF * np.sqrt(2.0)        # any yaw
CLEAR = 0.01                               # free gap between bodies
HAND_R = 0.02                              # pusher rod radius (conservative)


def add_ood_args(p):
    g = p.add_argument_group("OOD benchmark (fm_pc/ood.py)")
    g.add_argument("--ood", choices=KINDS, default="none")
    g.add_argument("--ood-shift", type=float, default=0.02, help="OOD-1: metres outside the demo init range")
    g.add_argument("--ood-noise-pos", type=float, default=0.005, help="OOD-2: std of block x, y noise (m)")
    g.add_argument("--ood-noise-yaw", type=float, default=5.0, help="OOD-2: std of block yaw noise (deg)")
    g.add_argument("--ood-push-dist", type=float, default=0.04, help="OOD-3: teleport distance (m)")
    g.add_argument("--ood-push-step", type=int, default=None, help="OOD-3: earliest step (default: the task's push_step)")
    g.add_argument("--ood-push-phase", choices=["seam", "mid"], default="seam")
    g.add_argument("--ood-delay", type=int, default=2, help="OOD-4: inference delay in control steps")
    g.add_argument("--ood-seed", type=int, default=0)
    return p


def ood_tag(args):
    k = args.ood
    if k == "none":
        return ""
    if k == "shift":
        return f"_ood-shift{100 * args.ood_shift:g}cm"
    if k == "obsnoise":
        return f"_ood-noise{1000 * args.ood_noise_pos:g}mm{args.ood_noise_yaw:g}deg"
    if k == "push":
        return f"_ood-push{100 * args.ood_push_dist:g}cm{'' if args.ood_push_phase == 'seam' else '-mid'}"
    return f"_ood-delay{args.ood_delay}"


class OOD:
    def __init__(self, args, spec, env, execute_steps, horizon):
        """``spec``: tasks.TaskSpec (blocks, demo init ranges, workspace, walls)."""
        self.kind, self.spec, self.task, self.env = args.ood, spec, spec.name, env
        self.args, self.n_boxes, self.exe, self.horizon = args, spec.n_boxes, execute_steps, horizon
        self.push_step = args.ood_push_step if args.ood_push_step is not None else spec.push_step
        self.delay = args.ood_delay if self.kind == "delay" else 0
        d_max = min(horizon - execute_steps, execute_steps - 1)
        if self.kind == "delay" and not 0 < self.delay <= d_max:
            raise SystemExit(f"--ood-delay must be in 1..{d_max} (horizon {horizon}, execute {execute_steps})")
        self.walls = self._walls()
        self.block_r = spec.block_half * np.sqrt(2.0)        # circumscribed radius, any yaw
        self.events = {}
        self.rng = None

    # ---- geometry -------------------------------------------------------------
    def _walls(self):
        return self.spec.walls(self.env)

    def free(self, xy, others, hand_xy, old_xy=None, yaw=0.0):
        (x0, x1), (y0, y1) = self.spec.workspace
        r = self.block_r
        if not (x0 + r <= xy[0] <= x1 - r and y0 + r <= xy[1] <= y1 - r):
            return False
        for o in others:
            if np.linalg.norm(xy - o) < 2 * r + CLEAR:
                return False
        if hand_xy is not None and not self.spec.hand_clear(xy, old_xy, yaw, np.asarray(hand_xy, dtype=float), r, CLEAR):
            return False
        for c, h in self.walls:          # circle vs axis-aligned box
            d = np.maximum(np.abs(xy - c) - h, 0.0)
            if np.linalg.norm(d) < r + CLEAR:
                return False
        return True

    # ---- episode hooks --------------------------------------------------------
    def start_episode(self, ci, ti):
        self.rng = np.random.default_rng([self.args.ood_seed, ci, ti, 7])
        self.key = f"c{ci:03d}_t{ti}"
        self.pushed = False
        self.event_step = -1

    def initial_context(self, ctx, hand_xy):
        """OOD-1: move every block d outside its demo init range (nearest feasible face first)."""
        if self.kind != "shift":
            return ctx
        d = self.args.ood_shift
        ctx = copy.deepcopy(ctx)
        pos = [np.asarray(self.spec.ctx_box_xy(ctx, i), dtype=float).copy() for i in range(self.n_boxes)]
        applied = []
        for i in range(self.n_boxes):
            (x0, x1), (y0, y1) = self.spec.INIT_RANGE[i]
            p = pos[i]
            cands = [(p[0] - x0, np.array([x0 - d, p[1]]), "x-"), (x1 - p[0], np.array([x1 + d, p[1]]), "x+"),
                     (p[1] - y0, np.array([p[0], y0 - d]), "y-"), (y1 - p[1], np.array([p[0], y1 + d]), "y+")]
            cands.sort(key=lambda c: c[0])
            others = [pos[j] for j in range(self.n_boxes) if j != i]
            own = self.spec.INIT_RANGE[i]
            other_ranges = [self.spec.INIT_RANGE[j] for j in range(self.n_boxes) if j != i and self.spec.INIT_RANGE[j] != own]
            for _, q, face in cands:
                # do not move a block towards another block's start region: that is a different
                # in-distribution situation (e.g. green where red usually starts), not "outside the demos"
                near_other = any(np.hypot(max(rx[0] - q[0], 0.0, q[0] - rx[1]), max(ry[0] - q[1], 0.0, q[1] - ry[1])) < d
                                 for rx, ry in other_ranges)
                if not near_other and self.free(q, others, hand_xy):
                    pos[i] = q
                    applied.append({"box": i, "face": face, "from": p.tolist(), "to": q.tolist()})
                    break
            else:
                applied.append({"box": i, "face": None, "from": p.tolist(), "to": p.tolist()})
        for i in range(self.n_boxes):
            ctx = self.spec.ctx_set_box_xy(ctx, i, pos[i])
        self.events[self.key] = {"shift": applied}
        return ctx

    def observe_boxes(self, boxes):
        """OOD-2: noisy copy of the block poses for the policy (true state untouched)."""
        if self.kind != "obsnoise":
            return boxes
        a, out = self.args, []
        for pos, quat, yaw in boxes:
            p = np.asarray(pos, dtype=float).copy()
            p[:2] += self.rng.normal(0.0, a.ood_noise_pos, 2)
            out.append((p, quat, float(yaw + np.deg2rad(self.rng.normal(0.0, a.ood_noise_yaw)))))
        return out

    def observe_state(self, st):
        """OOD-2 on a tasks.State: noisy blocks, or the measured hand when the task has no blocks."""
        if self.kind != "obsnoise":
            return st
        st = copy.copy(st)
        if st.boxes:
            st.boxes = self.observe_boxes(st.boxes)
        else:
            st.c = np.asarray(st.c, dtype=float).copy()
            st.c[:2] += self.rng.normal(0.0, self.args.ood_noise_pos, 2)
        return st

    def push_hand(self, step, des, hand_xy):
        """OOD-3 for tasks without blocks: displace the desired hand by ``--ood-push-dist``
        (random direction that stays in the workspace and clear of the obstacles)."""
        self.pushed = True
        dist = self.args.ood_push_dist
        rec = {"step": step, "hand_from": np.asarray(des[:2]).tolist(), "hand_to": None}
        (x0, x1), (y0, y1) = self.spec.workspace
        for th in self.rng.uniform(0, 2 * np.pi, 64):
            q = np.asarray(des[:2], dtype=float) + dist * np.array([np.cos(th), np.sin(th)])
            ok = x0 <= q[0] <= x1 and y0 <= q[1] <= y1
            for c, h in self.walls:
                if np.linalg.norm(np.maximum(np.abs(q - c) - h, 0.0)) < self.spec.hand_r + CLEAR:
                    ok = False
            if ok:
                des = np.asarray(des, dtype=float).copy()
                des[:2] = q
                rec["hand_to"] = q.tolist()
                self.event_step = step
                break
        self.events.setdefault(self.key, {})["push"] = rec
        return des

    def push_due(self, step, n_queued):
        if self.kind != "push" or self.pushed or step < self.push_step:
            return False
        want = 0 if self.args.ood_push_phase == "seam" else self.exe // 2
        return n_queued == want

    def push(self, step, box_objs, boxes, hand_xy):
        """OOD-3: teleport the block nearest the hand by ``--ood-push-dist`` (random feasible direction;
        falls back to the next-nearest block when the nearest one has none)."""
        self.pushed = True
        xy = [np.asarray(b[0][:2], dtype=float) for b in boxes]
        cand = self.spec.pushable(boxes, hand_xy)
        if not cand:
            self.events.setdefault(self.key, {})["push"] = {"step": step, "box": None, "from": None, "to": None}
            return False
        dist = self.args.ood_push_dist
        rec = None
        # nearest block first; if it has no free direction (walls, hand, other blocks), try the next one
        for i in sorted(cand, key=lambda j: np.linalg.norm(xy[j] - hand_xy)):
            others = [xy[j] for j in range(len(xy)) if j != i]
            r = {"step": step, "box": i, "from": xy[i].tolist(), "to": None}
            rec = rec or r
            for th in self.rng.uniform(0, 2 * np.pi, 64):
                q = xy[i] + dist * np.array([np.cos(th), np.sin(th)])
                if self.free(q, others, hand_xy, xy[i], float(boxes[i][2])):
                    z = float(boxes[i][0][2])
                    self.env.scene.set_obj_pos_and_quat([q[0], q[1], z], np.asarray(boxes[i][1], dtype=float), sim_obj=box_objs[i])
                    r["to"] = q.tolist()
                    self.event_step = step
                    rec = r
                    break
            if rec["to"] is not None:
                break
        self.events.setdefault(self.key, {})["push"] = rec
        return rec["to"] is not None

    def describe(self):
        a = self.args
        d = {"kind": self.kind, "seed": a.ood_seed}
        if self.kind == "shift":
            d.update(shift_m=a.ood_shift, init_range=self.spec.INIT_RANGE)
        elif self.kind == "obsnoise":
            d.update(noise_pos_m=a.ood_noise_pos, noise_yaw_deg=a.ood_noise_yaw)
        elif self.kind == "push":
            d.update(push_dist_m=a.ood_push_dist, push_step=self.push_step, phase=a.ood_push_phase)
        elif self.kind == "delay":
            d.update(delay_steps=self.delay)
        d["events"] = self.events
        return json.loads(json.dumps(d, default=float))


def run_episode(env, policy, ood, ctx, ci, ti, read_state, build_obs, box_objs, execute_steps, log_boxes):
    """One rollout.  With ``ood.kind == 'none'`` this is the original evaluation loop, call for call."""
    ood.start_episode(ci, ti)
    if ood.kind == "shift":
        env.reset(random=False, context=copy.deepcopy(ctx))
        c0, _ = read_state(env)
        ctx = ood.initial_context(ctx, c0[:2])
    env.reset(random=False, context=copy.deepcopy(ctx))
    c, boxes = read_state(env)
    fixed_z = c[2:3].copy()
    des = c[:2].copy()
    done, info, step = False, {}, 0
    log = {"des": [], "c": [], "b": [], "act": [], "seams": [], "pobs": []}
    queue, pending, plan_s, n_plans = [], None, 0.0, 0
    d = ood.delay
    while not done:
        if ood.push_due(step, len(queue)):
            ood.push(step, box_objs, boxes, c[:2])
            c, boxes = read_state(env)
        need_plan = (not queue and pending is None) if d == 0 else ((not queue and pending is None) or (len(queue) == d and pending is None and step > 0))
        if need_plan:
            obs = build_obs(des, c, ood.observe_boxes(boxes))
            tp = time.time()
            plan = policy.plan(obs)
            plan_s += time.time() - tp
            n_plans += 1
            log["pobs"].append(obs)
            if not queue:                       # synchronous start (and every chunk when d == 0)
                queue = [plan[k, :2].astype(np.float64) for k in range(min(execute_steps, plan.shape[0]))]
                log["seams"].append(step)
            else:                               # async: becomes active when the queue runs out
                pending = plan
        if not queue and pending is not None:
            queue = [pending[k, :2].astype(np.float64) for k in range(d, min(d + execute_steps, pending.shape[0]))]
            pending = None
            log["seams"].append(step)
        a = queue.pop(0)
        des = des + a
        command = np.concatenate([des, fixed_z, [0.0, 1.0, 0.0, 0.0]])
        _, _, done, info = env.step(command)
        c, boxes = read_state(env)
        log["des"].append(des.copy()); log["c"].append(c.copy()); log["act"].append(a.copy())
        log["b"].append(log_boxes(boxes))
        step += 1
    return info, step, log, plan_s, n_plans, ood.event_step


def pack_rollout(key, log, event_step):
    return {key + "_des": np.asarray(log["des"], dtype=np.float32),
            key + "_c": np.asarray(log["c"], dtype=np.float32),
            key + "_boxes": np.asarray(log["b"], dtype=np.float32),
            key + "_act": np.asarray(log["act"], dtype=np.float32),
            key + "_seams": np.asarray(log["seams"], dtype=np.int32),
            key + "_pobs": np.asarray(log["pobs"], dtype=np.float32),
            key + "_event": np.asarray([event_step], dtype=np.int32)}
