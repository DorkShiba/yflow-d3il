"""Paired Sorting2 baseline / 1+2+3+5 ID evaluation using the main XY policy."""

import argparse, copy, json, pickle, time, hashlib
from pathlib import Path
import numpy as np
import torch
from configs.config import ExperimentConfig
from fm_baseline.eval_pushing import load_model, Policy as BasePolicy
from fm_constraint.policy import Policy
from fm_constraint.pf2mp import prepare_potential
from fm_constraint.tasks import get_task
from fm_constraint.evaluation import run_evaluation
from fm_constraint import checkpoint_reads
from fm_constraint.comparison_io import check_output_signature


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/sorting2_combo29.yaml")
    p.add_argument("--run-dir")
    p.add_argument("--data-root")
    p.add_argument("--checkpoint")
    p.add_argument("--d3il-root", required=True)
    p.add_argument("--contexts-file", required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--count", type=int, default=100)
    p.add_argument("--repeats", type=int, default=4)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    assert a.start >= 0 and a.count > 0 and a.repeats > 0 and a.steps > 0
    out = Path(a.output).resolve()
    D3 = Path(a.d3il_root).resolve()
    contexts_file = Path(a.contexts_file).resolve()
    overrides = {"device": a.device, "paths.d3il_root": str(D3)}
    if a.run_dir:
        overrides["paths.run_dir"] = str(Path(a.run_dir).resolve())
    if a.data_root:
        overrides["paths.data_root"] = str(Path(a.data_root).resolve())
    if a.checkpoint:
        overrides["eval.checkpoint"] = a.checkpoint
    cfg = ExperimentConfig.from_yaml(a.config, overrides)
    v = cfg.values
    v["eval"].update(n_contexts=a.count, n_traj=a.repeats, max_steps=a.steps)
    cfg.validate("eval")
    assert cfg.task == "sorting_2"
    with checkpoint_reads():
        m, model, norm, condition, solver, step = load_model(
            v["paths"]["run_dir"], v["eval"]["checkpoint"], "ema", a.device
        )
    check_output_signature(
        out,
        dict(
            task="sorting_2",
            config=v,
            start=a.start,
            count=a.count,
            repeats=a.repeats,
            steps=a.steps,
        ),
        [
            Path(v["paths"]["run_dir"]) / v["eval"]["checkpoint"],
            Path(v["paths"]["run_dir"]) / "normalizer.npz",
            contexts_file,
        ],
    )
    spec = get_task("sorting_2")
    spec.setup(str(D3))
    env = spec.make_env(a.steps)
    contexts = pickle.load(contexts_file.open("rb"))
    assert a.start + a.count <= len(contexts)
    items = [
        dict(id=f"test_{i:03d}", context=contexts[i])
        for i in range(a.start, a.start + a.count)
    ]

    def read_state(env, quat2euler):
        state = spec.read(env)
        return state.c, state.boxes

    def build_obs(des, c, boxes):
        return np.concatenate(
            [
                des,
                c[:2],
                *[np.r_[p[:2], np.sin(yaw), np.cos(yaw)] for p, q, yaw in boxes],
            ]
        ).astype(np.float32)

    # Load one frozen train-only cache; use the main policy and main closed-loop implementation.
    potential = prepare_potential(cfg)
    order = ["baseline", "combo29"] if a.start % 2 == 0 else ["combo29", "baseline"]
    for arm in order:
        if (out / arm / "COMPLETE.json").exists():
            saved = json.loads((out / arm / "results.json").read_text())
            assert (
                saved["context_ids"] == [i["id"] for i in items]
                and saved["max_steps"] == a.steps
                and saved["n_traj"] == a.repeats
            )
            continue
        if (out / arm).exists():
            (out / arm).rename(out / (arm + ".partial." + str(time.time_ns())))
        config = copy.deepcopy(v)
        for name in config["methods"]:
            config["methods"][name]["enabled"] = arm == "combo29" and name != "rtc"
        base = BasePolicy(model, norm, m, condition, solver, 10, a.device)
        policy = Policy(base, config, potential if arm == "combo29" else None)
        settings = dict(config["eval"], sampling_seeds=list(range(a.repeats)))
        t = time.time()
        results, logs = run_evaluation(
            env,
            policy,
            items,
            read_state,
            build_obs,
            spec.quat2euler,
            "sorting_2",
            settings,
            config,
            spec=spec,
        )
        d = out / arm
        d.mkdir()
        results.update(
            gpu_name=(
                torch.cuda.get_device_name(torch.device(a.device))
                if a.device.startswith("cuda")
                else None
            ),
            wall_seconds=time.time() - t,
            checkpoint_step=step,
            checkpoint_sha256=hashlib.sha256(
                (Path(v["paths"]["run_dir"]) / v["eval"]["checkpoint"]).read_bytes()
            ).hexdigest(),
            normalizer_sha256=hashlib.sha256(
                (Path(v["paths"]["run_dir"]) / "normalizer.npz").read_bytes()
            ).hexdigest(),
            context_ids=[i["id"] for i in items],
            max_steps=a.steps,
            seed_scheme="main run_evaluation CPU noise hash(task, context_id, repeat_seed)",
            combo=0 if arm == "baseline" else 29,
        )
        (d / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False))
        (d / "rollouts.json").write_text(json.dumps(logs, allow_nan=False))
        (d / "resolved_config.json").write_text(json.dumps(config, indent=2))
        (d / "COMPLETE.json").write_text(
            json.dumps(
                dict(
                    episodes=len(results["episodes"]),
                    context_ids=results["context_ids"],
                    max_steps=a.steps,
                )
            )
        )
        (out / "PROGRESS.json").write_text(
            json.dumps(
                dict(
                    arm=arm,
                    complete=True,
                    wall_seconds=results["wall_seconds"],
                    success_rate=results["success_rate"],
                    episodes=len(results["episodes"]),
                )
            )
        )
        print(arm, "SAVED", results["wall_seconds"], flush=True)
    (out / "COMPLETE.json").write_text(
        json.dumps(dict(arms=order, contexts=a.count, repeats=a.repeats, steps=a.steps))
    )


if __name__ == "__main__":
    main()
