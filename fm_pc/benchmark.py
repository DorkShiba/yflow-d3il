"""Task-generic rollout loop for the OOD benchmark (any ``tasks.TaskSpec``).

Same protocol as ``fm_baseline/eval_*.py``: plan an H-step chunk from the latest
state, execute its first ``execute_steps`` actions, re-plan.  OOD hooks follow
``ood.OOD``; with ``--ood none`` the loop for Pushing reproduces
``fm_pc/eval_pushing.py`` bit for bit (checked by ``check_benchmark.py``).
"""
from __future__ import annotations

import copy
import time

import numpy as np


def run_task_episode(env, policy, spec, ood, ctx, ci, ti, execute_steps):
    ood.start_episode(ci, ti)
    if ood.kind == "shift":
        spec.reset(env, ctx)
        s0 = spec.read(env)
        ctx = ood.initial_context(ctx, spec.hand_xy(s0)) if spec.n_boxes else spec.shift_start(env, ctx, ood)
    spec.reset(env, ctx)
    s = spec.read(env)
    des, fixed = spec.init_des(env, s)
    done, info, step = False, {}, 0
    log = {"des": [], "c": [], "b": [], "act": [], "seams": [], "pobs": []}
    queue, pending, plan_s, n_plans = [], None, 0.0, 0
    d, A = ood.delay, spec.act_dim
    while not done:
        if ood.push_due(step, len(queue)):
            if spec.n_boxes:
                ood.push(step, spec.box_objs(env), s.boxes, spec.hand_xy(s))
                s = spec.read(env)
            else:
                des = ood.push_hand(step, des, spec.hand_xy(s))
        need_plan = (not queue and pending is None) or (d > 0 and len(queue) == d and pending is None and step > 0)
        if need_plan:
            obs = spec.obs(des, ood.observe_state(s))
            tp = time.time()
            plan = policy.plan(obs)
            plan_s += time.time() - tp
            n_plans += 1
            log["pobs"].append(obs)
            if not queue:
                queue = [plan[k, :A].astype(np.float64) for k in range(min(execute_steps, plan.shape[0]))]
                log["seams"].append(step)
            else:
                pending = plan
        if not queue and pending is not None:
            queue = [pending[k, :A].astype(np.float64) for k in range(d, min(d + execute_steps, pending.shape[0]))]
            pending = None
            log["seams"].append(step)
        a = queue.pop(0)
        des = spec.apply(des, a)
        _, _, done, info = env.step(spec.command(des, fixed))
        s = spec.read(env)
        log["des"].append(np.asarray(des, dtype=np.float64).copy()); log["c"].append(s.c.copy()); log["act"].append(a.copy())
        log["b"].append(spec.log_state(s))
        step += 1
    return info, step, log, plan_s, n_plans, ood.event_step
