"""분할 실행 스크립트의 그룹 분리·병렬 잠금·실패 전파를 GPU 없이 검증한다."""

import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from configs.config import REPO

SCRIPT = REPO / 'scripts/run_inserting_gpu_group.sh'


@pytest.fixture
def fake_python(tmp_path):
    path = tmp_path / 'fake-python'
    path.write_text('#!' + sys.executable + '\n' + '''
import json, os, sys, time
from pathlib import Path
args = sys.argv[1:]
if args[0] == '-c':
    print(Path(args[-1]).resolve())
    sys.exit(0)
if args[0] == '-':
    print('Fake GPU for shell control-flow tests')
    sys.exit(0)
command = args[args.index('fm_constraint.method123') + 1]
root = Path(args[args.index('--output') + 1])
events = root / 'events'
events.mkdir(exist_ok=True)
if command == 'prepare':
    manifest = root / 'experiment_manifest.json'
    kind = 'hit' if manifest.exists() else 'created'
    time.sleep(.1)
    if kind == 'created':
        manifest.write_text('{}')
    (events / ('prepare.' + str(os.getpid()) + '.json')).write_text(json.dumps({'kind':kind}))
elif command == 'run':
    job = args[args.index('--job') + 1]
    (events / (job + '.begin')).write_text(str(time.time()))
    time.sleep(float(os.environ.get('FAKE_JOB_PAUSE', '.02')))
    if os.environ.get('FAKE_RUN_FAIL') == job:
        sys.exit(7)
    (events / (job + '.end')).write_text(str(time.time()))
else:
    raise AssertionError('Unexpected command: ' + command)
''')
    path.chmod(0o755)
    return path


def _command(root, group):
    return ['bash', str(SCRIPT), '--group', str(group), '--output', str(root)]


def test_two_groups_share_manifest_but_run_disjoint_jobs(tmp_path, fake_python):
    root = tmp_path / 'shared'
    env = {**os.environ, 'PYTHON': str(fake_python)}
    processes = [subprocess.Popen(_command(root, group), cwd=REPO, env=env,
                                  text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE) for group in (1, 2)]
    for group, process in enumerate(processes, start=1):
        stdout, stderr = process.communicate(timeout=20)
        assert process.returncode == 0, stdout + stderr
        expected = ['I00', 'I04', 'I08', 'I12'] if group == 1 else ['I16', 'I20', 'I24', 'I28']
        assert [line.split()[3] for line in stdout.splitlines() if line.startswith('[group')] == expected
    assert sorted(path.stem for path in (root / 'events').glob('I*.end')) == ['I00', 'I04', 'I08', 'I12', 'I16', 'I20', 'I24', 'I28']
    preparation = [json.loads(path.read_text())['kind'] for path in (root / 'events').glob('prepare.*.json')]
    assert sorted(preparation) == ['created', 'hit']


def test_same_group_is_rejected_while_first_is_running(tmp_path, fake_python):
    root = tmp_path / 'shared'
    env = {**os.environ, 'PYTHON': str(fake_python), 'FAKE_JOB_PAUSE': '.3'}
    first = subprocess.Popen(_command(root, 1), cwd=REPO, env=env,
                             text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 10
        while not (root / 'events/I00.begin').exists():
            assert first.poll() is None
            assert time.monotonic() < deadline
            time.sleep(.01)
        second = subprocess.run(_command(root, 1), cwd=REPO, env=env, text=True, capture_output=True, timeout=10)
        assert second.returncode != 0
        assert 'already running' in second.stderr
    finally:
        stdout, stderr = first.communicate(timeout=20)
        assert first.returncode == 0, stdout + stderr


def test_whole_suite_lock_blocks_group(tmp_path, fake_python):
    root = tmp_path / 'shared'
    root.mkdir()
    with (root / '.suite.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        result = subprocess.run(_command(root, 2), cwd=REPO,
                                env={**os.environ, 'PYTHON': str(fake_python)}, text=True, capture_output=True)
    assert result.returncode != 0
    assert 'Whole-suite runner' in result.stderr
    assert not (root / 'events').exists()


def test_failed_pipeline_stops_remaining_jobs(tmp_path, fake_python):
    root = tmp_path / 'shared'
    result = subprocess.run(_command(root, 1), cwd=REPO,
                            env={**os.environ, 'PYTHON': str(fake_python), 'FAKE_RUN_FAIL': 'I04'},
                            text=True, capture_output=True)
    assert result.returncode == 7
    assert (root / 'events/I00.end').exists()
    assert (root / 'events/I04.begin').exists()
    assert not (root / 'events/I08.begin').exists()


def test_invalid_group_does_not_prepare(tmp_path, fake_python):
    root = tmp_path / 'shared'
    result = subprocess.run(_command(root, 3), cwd=REPO,
                            env={**os.environ, 'PYTHON': str(fake_python)}, text=True, capture_output=True)
    assert result.returncode == 2
    assert not root.exists()
