import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from fm_constraint.run_qp123 import run_parallel


@pytest.mark.parametrize('fail', [False, True])
def test_gpu_workers_are_serial_and_failures_are_visible(tmp_path, monkeypatch, fail):
    active, peak, launched = {}, {}, []
    lock = threading.Lock()
    class Process:
        def __init__(self, cmd, **kwargs):
            self.device = cmd[cmd.index('--device') + 1]
            self.arm = cmd[cmd.index('--arm') + 1]
            self.seed = cmd[cmd.index('--seeds') + 1]
            assert kwargs['env']['OMP_NUM_THREADS'] == '2'
            with lock:
                launched.append((self.device, self.arm, self.seed))
                active[self.device] = active.get(self.device, 0) + 1
                peak[self.device] = max(peak.get(self.device, 0), active[self.device])
            dest = Path(cmd[cmd.index('--output-root') + 1]) / 'evaluation'
            dest.mkdir(parents=True)
            result = {k: 0 for k in ('success_rate', 'n_contexts', 'n_traj', 'plan_ms',
                      'postprocess_ms', 'smoothness', 'filter_abort_rate',
                      'executed_acceleration_qp_exceed_rate')}
            (dest / 'results.json').write_text(json.dumps(result))
        def wait(self):
            time.sleep(.005)
            with lock:
                active[self.device] -= 1
            return 2 if fail and self.device == 'cuda:0' else 0
    monkeypatch.setattr('fm_constraint.run_qp123.subprocess.Popen', Process)
    args = SimpleNamespace(rtc=False, condition=None, devices=['cuda:0', 'cuda:1'], seeds=[0, 1, 2],
        task='pushing', n_contexts=0, n_traj=4, qp_bound_m=.0003,
        hardflow_root=None, d3il_root=None, data_root=None, dry_run=False, cpu_threads=2)
    if fail:
        with pytest.raises(RuntimeError, match='worker'):
            run_parallel(args, tmp_path, {})
    else:
        run_parallel(args, tmp_path, {})
    assert peak == {'cuda:0': 1, 'cuda:1': 1}
    rows = json.loads((tmp_path / 'summary.json').read_text())
    errors = json.loads((tmp_path / 'failures.json').read_text())
    assert len(rows) == (12 if fail else 24)
    assert len(errors) == int(fail)
    assert len({(r['seed'], r['arm']) for r in rows}) == len(rows)
    assert all(r['arm'] in ('001', '010', '101', '110') for r in rows if r['device'] == 'cuda:1')


def test_eight_gpu_dry_run_has_24_unique_jobs_and_writes_nothing(tmp_path, capsys):
    args = SimpleNamespace(rtc=False, condition=None, devices=[f'cuda:{i}' for i in range(8)], seeds=[0, 1, 2],
        task='pushing', n_contexts=0, n_traj=4, qp_bound_m=.0003,
        hardflow_root=None, d3il_root=None, data_root=None, dry_run=True, cpu_threads=2)
    run_parallel(args, tmp_path, {})
    commands = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(commands) == 24
    pairs = [(c[c.index('--device') + 1], c[c.index('--arm') + 1]) for c in commands]
    assert len(set(pairs)) == 8
    assert all(pairs.count(pair) == 3 for pair in set(pairs))
    assert len({c[c.index('--output-root') + 1] for c in commands}) == 24
    assert not list(tmp_path.iterdir())


def test_four_gpus_each_get_one_pf_on_and_one_pf_off(tmp_path, capsys):
    args = SimpleNamespace(rtc=False, condition=None, devices=[f'cuda:{i}' for i in range(4)], seeds=[0, 1, 2],
        task='pushing', n_contexts=0, n_traj=4, qp_bound_m=.0003,
        hardflow_root=None, d3il_root=None, data_root=None, dry_run=True, cpu_threads=2)
    run_parallel(args, tmp_path, {})
    commands = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(commands) == 24
    expected = [('000', '111'), ('001', '110'), ('010', '101'), ('011', '100')]
    for i, arms in enumerate(expected):
        jobs = [c for c in commands if c[c.index('--device') + 1] == f'cuda:{i}']
        assert len(jobs) == 6
        assert {c[c.index('--arm') + 1] for c in jobs} == set(arms)
