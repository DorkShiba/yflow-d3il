"""Exercise the actual evaluation entrypoint with a lightweight fake simulator."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from fm_rtc import evaluate
from fm_rtc.compare import compare


class Field(torch.nn.Module):
    def forward(self, x, t): return .01 * x


class FakeEnv:
    def __init__(self, render, n_substeps, max_steps_per_episode):
        self.n_substeps = n_substeps
        self.scene = SimpleNamespace(dt=.001)
        self.limit = max_steps_per_episode
    def start(self): pass
    def seed(self, value): pass
    def reset(self, random, context):
        self.c = np.r_[context, .1].astype(float)
        self.steps = 0
    def step(self, command):
        self.c += .3*(command[:3]-self.c)
        self.steps += 1
        return None, 0, self.steps >= self.limit, dict(success=True, mode=0, mean_distance=.01)


class EvaluationTests(unittest.TestCase):
    def test_saved_paired_results_and_rollouts(self):
        import eval_pushing as api
        root = evaluate.REPO / "codex/rtc/test_tmp"
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as folder:
            folder = Path(folder)
            run, hf, d3il = folder/"run", folder/"hf", folder/"d3il"
            for d in (run, hf, d3il): d.mkdir()
            cfg = dict(task="pushing", horizon=16, act_dim=2, obs_dim=2)
            (run/"config.json").write_text(json.dumps(cfg))
            (run/"baseline_ema.pt").write_bytes(b"test-only checkpoint placeholder")
            np.savez(run/"normalizer.npz", center=np.zeros(4), scale=np.ones(4))
            ctx = d3il/"environments/dataset/data/pushing/test_contexts.pkl"
            ctx.parent.mkdir(parents=True)
            with ctx.open("wb") as f: np.save(f, np.array([[.1,.2],[.2,.3]]))
            norm = SimpleNamespace(center=np.zeros(4), scale=np.ones(4))
            common = ["--task", "pushing", "--run-dir", str(run), "--hardflow-root", str(hf),
                      "--d3il-root", str(d3il), "--n-contexts", "2", "--n-traj", "2",
                      "--max-steps", "24", "--device", "cpu", "--delay-steps", "3"]
            boxes = [(np.zeros(3), np.array([1.,0,0,0]), 0.)]*2
            with patch.object(evaluate, "load_model", return_value=(cfg,Field(),norm,None,None,1000000)), \
                 patch.object(api, "setup_d3il", return_value=(FakeEnv,None)), \
                 patch.object(api, "read_state", side_effect=lambda env, quat: (env.c.copy(),boxes)), \
                 patch.object(api, "build_obs", side_effect=lambda des,c,boxes: c[:2].astype("float32")), \
                 redirect_stdout(io.StringIO()):
                for method in ("baseline", "rtc"):
                    evaluate.main(common+["--method",method,"--output",str(folder/method)])
            results = [json.loads((folder/m/"results.json").read_text()) for m in ("baseline","rtc")]
            report = compare(*results, n_boot=100)
            self.assertEqual(report["metrics"]["success"]["mean_context_difference"], 0)
            self.assertEqual(len(results[0]["episodes"]), 4)
            for method in ("baseline","rtc"):
                with np.load(folder/method/"rollouts.npz") as saved:
                    self.assertEqual(saved["c000_t0_des"].shape, (24,2))
                    self.assertEqual(np.flatnonzero(saved["c000_t0_seams"]).tolist(), [11,19])
            with self.assertRaises(FileExistsError):
                evaluate.main(common+["--method","rtc","--output",str(folder/"rtc")])


if __name__ == "__main__":
    unittest.main()
