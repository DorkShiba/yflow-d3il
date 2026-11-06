import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest

from fm_constraint.run_qp123_budget import schedule, completed


def test_deadline_terminates_children_and_resume_skips_completed(tmp_path, monkeypatch):
    processes = []
    original = subprocess.Popen
    def launch(*args, **kwargs):
        child = original(*args, **kwargs)
        processes.append(child)
        return child
    monkeypatch.setattr('fm_constraint.run_qp123_budget.subprocess.Popen', launch)
    directory = tmp_path / 'context_0'
    jobs = {'cuda:0': [(directory, lambda _: [sys.executable, '-c', 'import time; time.sleep(30)'])]}
    started = time.monotonic()
    assert schedule(jobs, tmp_path, started + .5, os.environ.copy()) == 'stopped'
    assert time.monotonic() - started < 10
    assert all(child.poll() is not None for child in processes)
    assert completed(directory) is None

    def successful(attempt):
        code = ("import json,sys; from pathlib import Path; p=Path(sys.argv[1])/'eval'; "
                "p.mkdir(parents=True); "
                "(p/'results.json').write_text(json.dumps(dict(n_contexts=1,n_traj=1,episodes=[{}],"
                "success_rate=1,smoothness={},plan_ms=1,postprocess_ms=1))); "
                "(p/'rollouts.json').write_text('{}')")
        return [sys.executable, '-c', code, str(attempt)]
    jobs['cuda:0'] = [(directory, successful)]
    assert schedule(jobs, tmp_path, time.monotonic() + 30, os.environ.copy()) == 'complete'
    launched = len(processes)
    assert completed(directory)['success_rate'] == 1
    assert schedule(jobs, tmp_path, time.monotonic() + 30, os.environ.copy()) == 'complete'
    assert len(processes) == launched
    record = completed(directory)
    (directory / record['result']).write_text('{}')
    with pytest.raises(ValueError, match='missing/changed'):
        completed(directory)


def test_failed_job_is_not_marked_complete(tmp_path):
    jobs = {'cuda:0': [(tmp_path / 'failed', lambda _: [sys.executable, '-c', 'raise SystemExit(2)'])]}
    with pytest.raises(RuntimeError, match='Some jobs failed'):
        schedule(jobs, tmp_path, time.monotonic() + 30, os.environ.copy())
    assert completed(tmp_path / 'failed') is None
    assert json.loads((tmp_path / 'progress.json').read_text())['status'] == 'failed'


def test_sharded_context_uses_identical_episode_noise():
    from types import SimpleNamespace
    from configs.config import ExperimentConfig, REPO
    from fm_constraint.evaluation import run_evaluation
    from fm_constraint.tests.test_evaluation import FakeEnv, read_state, build_obs
    cfg = ExperimentConfig.from_yaml(REPO / 'configs/pushing.yaml')
    class Recorder:
        base = SimpleNamespace(device='cpu', horizon=4, act_dim=2, obs_dim=12)
        def __init__(self):
            self.noises = []
        def plan(self, obs, previous, noise):
            self.noises.append(noise.clone())
            self.last_report = {'status': 'infeasible'}
            return None
    settings = {'n_traj': 2, 'max_steps': 5, 'execute_steps': 2, 'seed': 3}
    contexts = [{'id': str(i), 'context': {}} for i in range(3)]
    whole, shard = Recorder(), Recorder()
    run_evaluation(FakeEnv(), whole, contexts, read_state, build_obs, None,
                   'pushing', settings, cfg.values, qp5_bound=.0003)
    result, _ = run_evaluation(FakeEnv(), shard, contexts[1:2], read_state, build_obs, None,
                   'pushing', {**settings, 'context_start': 1}, cfg.values, qp5_bound=.0003)
    for expected, actual in zip(whole.noises[2:4], shard.noises):
        np.testing.assert_array_equal(expected, actual)
    assert result['episodes'][0]['global_context_index'] == 1
