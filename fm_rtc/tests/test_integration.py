"""Optional real-checkpoint checks; set HARDFLOW_ROOT to the pinned checkout."""
import os
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

from fm_rtc.policy import RTCPolicy


@unittest.skipUnless(os.environ.get("HARDFLOW_ROOT"), "Set HARDFLOW_ROOT for actual checkpoint tests")
class CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.repo = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(cls.repo / "fm_baseline"))

    def check_task(self, task):
        from eval_pushing import Policy, load_model
        run = self.repo / "fm_baseline/results" / (task + "_baseline")
        cfg, model, norm, cond, solver, step = load_model(
            str(run), "baseline_ema.pt", "ema", os.environ["HARDFLOW_ROOT"], "cpu")
        self.assertEqual(step, 1000000)
        path = next((self.repo / "data" / (task + "_sincos_v1") / "val").glob("*.npz"))
        with np.load(path) as data:
            obs = data["obs"][0].copy()
        legacy = Policy(model, norm, cfg, cond, solver, 10, "cpu")
        rtc = RTCPolicy(model, norm, cfg, 10, "cpu")
        torch.manual_seed(519)
        reference = legacy.plan(obs)
        torch.manual_seed(519)
        off = rtc.plan(obs, rtc=False)
        np.testing.assert_array_equal(off, reference)
        rtc.reset()
        torch.manual_seed(519)
        np.testing.assert_array_equal(rtc.plan(obs), reference)
        state = {k: v.clone() for k, v in model.state_dict().items()}
        for delay in (0, 3):
            guided = rtc.plan(obs, delay=delay, generator=torch.Generator().manual_seed(10))
            self.assertTrue(np.isfinite(guided).all())
            np.testing.assert_allclose(guided[0, 2:], obs, atol=1e-7)
        for k, v in model.state_dict().items():
            torch.testing.assert_close(v, state[k], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_pushing_checkpoint(self): self.check_task("pushing")

    def test_inserting_checkpoint(self): self.check_task("inserting")


if __name__ == "__main__":
    unittest.main()
