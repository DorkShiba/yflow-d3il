"""Paired 23-observation / 8-action Stacking ID comparison with joint-space constraints."""

import argparse, copy, gzip, hashlib, json, pickle, time
from pathlib import Path
import numpy as np
import torch
from fm_constraint.tasks import setup_d3il
from fm_constraint.joint import solve_qp
from fm_baseline.eval_pushing import load_model, Policy as BasePolicy
from fm_baseline.eval_stacking import setup, state, observation, full_success
from fm_constraint.joint import (
    JointPolicy,
    JointPotential,
    JointConstraints,
    JointRTCReference,
)
from fm_constraint import checkpoint_reads
from fm_constraint.comparison_io import check_output_signature


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", default="fm_baseline/results/stacking_baseline")
    p.add_argument("--checkpoint", default="baseline_ema.pt")
    p.add_argument("--settings", default="configs/stacking_combo29.json")
    p.add_argument(
        "--cache", default="fm_baseline/results/stacking_baseline/joint_pf_cache.npz"
    )
    p.add_argument("--d3il-root", required=True)
    p.add_argument(
        "--contexts-file",
        default="fm_baseline/results/stacking_baseline/id_contexts.pkl",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--count", type=int, default=100)
    p.add_argument("--repeats", type=int, default=4)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    assert a.start >= 0 and a.count > 0 and a.repeats > 0 and a.steps > 0
    out = Path(a.output).resolve()
    D3 = Path(a.d3il_root).resolve()
    RUN = Path(a.run_dir).resolve()
    contexts_file = Path(a.contexts_file).resolve()
    cache = Path(a.cache).resolve()
    settings_path = Path(a.settings).resolve()
    settings = json.loads(settings_path.read_text())
    rtc_enabled = settings.get("rtc", {}).get("enabled", False)
    if rtc_enabled and (
        not np.isfinite(settings["rtc"]["beta"]) or settings["rtc"]["beta"] < 0
    ):
        raise ValueError("RTC beta must be finite and nonnegative")
    combo_arm = "joint_combo31" if rtc_enabled else "joint_combo29"
    with checkpoint_reads():
        c, model, norm, cond, solver, step = load_model(
            str(RUN), a.checkpoint, "ema", a.device
        )
    assert (c["obs_dim"], c["act_dim"], c["horizon"]) == (23, 8, 16)
    check_output_signature(
        out,
        dict(
            task="stacking",
            settings=settings,
            start=a.start,
            count=a.count,
            repeats=a.repeats,
            steps=a.steps,
            device=a.device,
            rtc_reference_rule="unexecuted_joint_suffix_reset_on_filter_v1",
        ),
        [
            RUN / a.checkpoint,
            RUN / "config.json",
            RUN / "normalizer.npz",
            contexts_file,
            cache,
            Path(__file__),
            Path(__file__).with_name("joint.py"),
            Path(__file__).with_name("rtc.py"),
        ],
    )
    setup_d3il(str(D3))
    Env, _, quat = setup(D3)
    env = Env(render=False, n_substeps=30, max_steps_per_episode=a.steps)
    env.start()
    contexts = pickle.load(contexts_file.open("rb"))
    potential = JointPotential(cache, a.device)
    base = BasePolicy(model, norm, c, cond, solver, 10, a.device)
    assert a.start + a.count <= len(contexts)
    order = (
        ["baseline", combo_arm] if (a.start // 5) % 2 == 0 else [combo_arm, "baseline"]
    )
    for arm in order:
        d = out / arm
        if (d / "COMPLETE.json").exists():
            saved = json.loads((d / "results.json").read_text())
            assert (
                saved["context_start"] == a.start
                and saved["n_contexts"] == a.count
                and saved["n_traj"] == a.repeats
                and saved["max_steps"] == a.steps
            )
            continue
        if d.exists():
            d.rename(out / (arm + ".partial." + str(time.time_ns())))
        d.mkdir()
        enabled = arm == combo_arm
        policy = JointPolicy(base, settings, potential if enabled else None, enabled)
        records = []
        arrays = {}
        reports = {}
        wall = time.time()
        for ci in range(a.start, a.start + a.count):
            for ti in range(a.repeats):
                payload = json.dumps(
                    ["stacking", f"test_{ci:03d}", ti], separators=(",", ":")
                )
                seed = int.from_bytes(
                    hashlib.sha256(payload.encode()).digest()[:8], "big"
                ) % (2**63 - 1)
                gen = torch.Generator(device="cpu").manual_seed(seed)
                env.reset(random=False, context=copy.deepcopy(contexts[ci]))
                q, tcp, g, boxes = state(env, quat)
                des = q.copy()
                env.controller.setSetPoint(des.copy())
                initial_q = q.copy()
                initial_c = tcp.copy()
                initial_boxes = boxes.copy()
                initial_gripper = g
                target = np.array(env.scene.get_obj_pos(env.target_box))
                previous = np.zeros(7)
                history = [des.copy()]
                queue = []
                reference = JointRTCReference()
                acts = []
                commands = []
                actual = []
                joints = []
                bxs = []
                grips = []
                seams = []
                full = []
                partials = []
                plan_times = []
                plan_logs = []
                filter_logs = []
                failure = None
                success = False
                done = False
                plan_id = -1
                for k in range(a.steps):
                    seam = False
                    if not queue:
                        obs = observation(des, g, boxes)
                        noise = torch.randn(
                            1, 16, 31, generator=gen, dtype=base.center.dtype
                        ).to(base.device)
                        synchronize(a.device)
                        started = time.perf_counter()
                        suffix = reference.suffix() if enabled and rtc_enabled else None
                        plan = policy.plan(
                            obs,
                            previous.copy(),
                            np.asarray(history[-3:]),
                            noise,
                            rtc_suffix=suffix,
                        )
                        synchronize(a.device)
                        ms = 1000 * (time.perf_counter() - started)
                        plan_times.append(ms)
                        plan_id += 1
                        plan_logs.append(
                            dict(
                                step=k,
                                plan_id=plan_id,
                                noise_seed=seed,
                                noise_index=plan_id,
                                obs=obs.tolist(),
                                previous_joint_action_rad=previous.tolist(),
                                history_joint_commands_rad=np.asarray(
                                    history[-3:]
                                ).tolist(),
                                report=copy.deepcopy(policy.last_report),
                                rtc_suffix_rad=(
                                    None if suffix is None else suffix.tolist()
                                ),
                            )
                        )
                        if plan is None:
                            failure = policy.last_report["status"]
                            break
                        reference.install(plan)
                        queue = [
                            (plan_id, index, row[:8].astype(np.float64))
                            for index, row in enumerate(plan[:8])
                        ]
                        seam = True
                    active, index, action = queue.pop(0)
                    reference.executed()
                    nominal = action.copy()
                    filtered = None
                    con = JointConstraints(des, previous, 1, settings)
                    if enabled:
                        filtered = solve_qp(
                            action[:7], con.G, con.d, settings["solver"]
                        )
                        value = filtered.pop("value")
                        entry = dict(
                            step=k,
                            plan_id=active,
                            plan_index=index,
                            nominal_action=nominal.tolist(),
                            solver=filtered,
                            safe_action=None,
                            queue_invalidated=False,
                        )
                        if value is None:
                            filter_logs.append(entry)
                            failure = filtered["status"]
                            break
                        action = action.copy()
                        action[:7] = value
                        assert action[7] == nominal[7]
                        entry["safe_action"] = action.tolist()
                        if (
                            np.linalg.norm(action[:7] - nominal[:7])
                            > settings["intervention_tolerance_rad"]
                        ):
                            queue = []
                            reference.clear()
                            entry["queue_invalidated"] = True
                        entry["executed_residual_rad"] = con.residual(action[:7])
                        assert (
                            entry["executed_residual_rad"]
                            <= settings["solver"]["feasibility_tolerance_m"]
                        )
                        filter_logs.append(entry)
                    des = des + action[:7]
                    history.append(des.copy())
                    previous = action[:7].copy()
                    _, _, done, info = env.step(np.r_[des, action[7]])
                    q, tcp, g, boxes = state(env, quat)
                    success = full_success(boxes, target, env.pos_min_dist)
                    assert success == bool(info["success"])
                    acts.append(action)
                    commands.append(des.copy())
                    actual.append(tcp.copy())
                    joints.append(q.copy())
                    bxs.append(boxes.copy())
                    grips.append(g)
                    seams.append(seam)
                    full.append(success)
                    partials.append([bool(info["success_1"]), bool(info["success_2"])])
                    if success or done:
                        break
                success = bool(success and failure is None)
                key = f"c{ci:03d}_t{ti}"
                n = len(acts)
                arract = np.asarray(acts).reshape(-1, 8)
                arrc = np.asarray(actual).reshape(-1, 3)
                arrj = np.asarray(joints).reshape(-1, 7)
                starts = np.flatnonzero(seams)
                boundary = (
                    np.linalg.norm(
                        (arract[starts[1:], :7] - arract[starts[1:] - 1, :7]) / 0.03,
                        axis=1,
                    )
                    if len(starts) > 1
                    else np.empty(0)
                )

                def rms_diff(x, degree, dt):
                    value = np.diff(x, n=degree, axis=0) / dt**degree
                    return (
                        float(np.sqrt(np.mean(np.sum(value**2, axis=1))))
                        if len(value)
                        else None
                    )

                record = dict(
                    key=key,
                    context=ci,
                    repeat=ti,
                    noise_seed=seed,
                    success=success,
                    success_1=partials[-1][0] if n else False,
                    success_2=partials[-1][1] if n else False,
                    steps=n,
                    failure=failure,
                    termination_reason=failure
                    or (
                        "success"
                        if success
                        else "step_budget" if n == a.steps else "environment_done"
                    ),
                    hand_xyz_jerk_rms=rms_diff(np.vstack([initial_c, arrc]), 3, 0.03),
                    actual_joint_jerk_rms=rms_diff(
                        np.vstack([initial_q, arrj]), 3, 0.03
                    ),
                    command_joint_jerk_rms=(
                        float(
                            np.sqrt(
                                np.mean(
                                    np.sum(
                                        (np.diff(arract[:, :7], n=2, axis=0) / 0.03**3)
                                        ** 2,
                                        axis=1,
                                    )
                                )
                            )
                        )
                        if n >= 3
                        else None
                    ),
                    boundary_joint_velocity_jump_rms=(
                        float(np.sqrt(np.mean(boundary**2))) if len(boundary) else None
                    ),
                    boundary_joint_velocity_jump_mean=(
                        float(boundary.mean()) if len(boundary) else None
                    ),
                    plan_ms_mean=float(np.mean(plan_times)),
                    plan_calls=len(plan_times),
                    plan_ms_sum=float(sum(plan_times)),
                    filter_ms_sum=float(
                        sum(row["solver"]["latency_ms"] for row in filter_logs)
                    ),
                    qp_calls=sum(
                        len(row["report"]["qp"])
                        + int(row["report"]["lipo"] is not None)
                        for row in plan_logs
                    )
                    + len(filter_logs),
                    pf_calls=sum(row["report"]["pf_calls"] for row in plan_logs),
                    rtc_guided_plans=sum(
                        row["report"]["rtc_guided"] for row in plan_logs
                    ),
                    filter_interventions=sum(
                        row["queue_invalidated"] for row in filter_logs
                    ),
                )
                for suffix, value in [
                    ("act", arract),
                    ("des_j", np.asarray(commands).reshape(-1, 7)),
                    ("j", arrj),
                    ("c", arrc),
                    ("g", grips),
                    ("boxes", np.asarray(bxs).reshape(-1, 3, 4)),
                    ("plan_starts", seams),
                    ("plan_ms", plan_times),
                    ("full_success", full),
                    ("partial", np.asarray(partials).reshape(-1, 2)),
                    ("initial_c", initial_c),
                    ("initial_j", initial_q),
                    ("initial_boxes", initial_boxes),
                    ("initial_gripper", initial_gripper),
                    ("target", target),
                ]:
                    arrays[key + "_" + suffix] = np.asarray(value)
                records.append(record)
                reports[key] = dict(plans=plan_logs, filters=filter_logs)
                print(arm, json.dumps(record), flush=True)
        result = dict(
            task="stacking",
            profile="joint_space_analogue_v2_float64",
            pipeline_order=["rtc", "pf2mp", "projection", "lipo", "filter"],
            inference_delay_steps=0,
            arm=arm,
            context_start=a.start,
            n_contexts=a.count,
            n_traj=a.repeats,
            max_steps=a.steps,
            checkpoint_step=step,
            checkpoint_sha256=hashlib.sha256(
                (RUN / a.checkpoint).read_bytes()
            ).hexdigest(),
            normalizer_sha256=hashlib.sha256(
                (RUN / "normalizer.npz").read_bytes()
            ).hexdigest(),
            contexts_sha256=hashlib.sha256(contexts_file.read_bytes()).hexdigest(),
            success_rate=float(np.mean([r["success"] for r in records])),
            wall_seconds=time.time() - wall,
            gpu_name=(
                torch.cuda.get_device_name(torch.device(a.device))
                if a.device.startswith("cuda")
                else None
            ),
            paired_same_process_gpu=a.device.startswith("cuda"),
            methods=dict(
                projection=enabled,
                pf2mp=enabled,
                filter=enabled,
                rtc=enabled and rtc_enabled,
                lipo=enabled,
            ),
            episodes=records,
        )
        np.savez_compressed(d / "rollouts.npz", **arrays)
        with gzip.open(d / "raw_reports.json.gz", "wt") as f:
            json.dump(reports, f, allow_nan=False)
        (d / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False))
        (d / "COMPLETE.json").write_text(
            json.dumps(
                dict(
                    episodes=len(records),
                    context_start=a.start,
                    repeats=a.repeats,
                    max_steps=a.steps,
                )
            )
        )
        (out / "PROGRESS.json").write_text(
            json.dumps(
                dict(
                    arm=arm,
                    wall_seconds=result["wall_seconds"],
                    success_rate=result["success_rate"],
                )
            )
        )
        print("SAVED", d, flush=True)
    (out / "COMPLETE.json").write_text(
        json.dumps(dict(arms=order, contexts=a.count, repeats=a.repeats, steps=a.steps))
    )


if __name__ == "__main__":
    main()
