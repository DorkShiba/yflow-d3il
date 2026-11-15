"""Stacking 본실험(ID·OOD A): 32조합 × (30 context × noise 3) × {id, A_weak, A_mid, A_strong}.

    python -m fm_constraint.stacking_experiment select            # 평가 context 30개 고정 (한 번, commit)
    python -m fm_constraint.stacking_experiment run --cond id,A_weak --combos all --contexts 0-6 \\
        --device cuda:0 --name hyeongbin                          # 맡은 context 실행
    python -m fm_constraint.stacking_experiment report --cond A_weak --name hyeongbin
        # docs/stacking/report_<cond>_<name>.md (git에 올리는 것은 이 보고서뿐)
    python -m fm_constraint.stacking_experiment merge --cond A_weak
        # docs/stacking/report_<cond>_results.{md,json}: 30 context bootstrap CI, Complementarity Δ, A 블록 수별 성공률

Pushing·Inserting 본실험(``fm_constraint.experiment``)과 코드 fingerprint·결과·보고서 위치를 분리했다
(그쪽 CODE_FILES를 건드리지 않기 위해). 평가 단위는 (조건, 조합, context 번호) 하나이며 noise seed 0·1·2의
rollout 3개를 담는다. 단위마다 즉시 저장하므로 중단 후 같은 명령을 다시 실행하면 완료된 단위를 건너뛴다.

- 모델: ``fm_baseline/results/stacking_baseline/baseline_ema.pt`` (35만 step EMA), train-only normalizer.
- 방법: ``fm_constraint.joint_combo`` (관절 공간 ①②③④⑤, 그리퍼 미수정. ④는 팀원 ``joint.joint_rtc_velocity``). 수치 설정은 ``configs/stacking_combo29.json``.
- FM noise: 팀원 비교 실험과 같은 CPU 해시 규칙 ``sha256(["stacking","test_NNN",seed])``. OOD도 같은 noise.
- context: ``id_contexts.pkl``의 test [0,100)에서 고정 seed로 30개. OOD A는 ``data/ood_stacking``의 같은 index 배치.
- 위반율: 실제 실행한 관절 증분이 실행 직전 목표 관절·직전 실행 증분 기준의 관절 위치 범위·축별 가속도
  상자(H=1)를 ``feasibility_tolerance`` 넘게 벗어난 step 비율. 분모는 실행 step 전체.
"""

import argparse
import copy
import fcntl
import hashlib
import json
import os
import pickle
import platform
import subprocess
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

from configs.config import REPO
from .experiment import REPORT_FORMAT, REPORT_MARKER, _mean, check_name, fmt, parse_indices
from .joint_combo import combo_flags

PROTOCOL = 'stacking_v1'
SELECTION_SEED = 20261108
SELECTION_STREAM = 2                      # pushing 0, inserting 1 (fm_constraint.experiment.select)
N_CONTEXTS = 30
SEEDS = (0, 1, 2)
CONDITIONS = ('id', 'A_weak', 'A_mid', 'A_strong')
CONFIG = REPO / 'configs/stacking_experiment.json'
CONTEXT_FILE = REPO / 'configs/stacking_experiment_contexts.json'
RESULT_BASE = REPO / 'experiments' / PROTOCOL           # 로컬 전용(.gitignore)
LOCAL_BASE = REPO / 'fm_constraint/runs' / f'experiment_{PROTOCOL}'
REPORT_DIR = REPO / 'docs/stacking'
CODE_FILES = ('configs/stacking_experiment.json', 'configs/stacking_combo29.json',
              'configs/stacking_experiment_contexts.json', 'requirements.txt',
              'fm_constraint/stacking_experiment.py', 'fm_constraint/joint_combo.py', 'fm_constraint/joint.py',
              'fm_constraint/qp.py', 'fm_constraint/rtc.py',
              'fm_baseline/eval_pushing.py', 'fm_baseline/eval_stacking.py',
              'fm_baseline/models/__init__.py', 'fm_baseline/models/temporal_unet.py',
              'fm_baseline/models/flow_matching.py', 'fm_baseline/models/ode.py')
MODE3 = {'rgb': 0, 'rbg': 1, 'grb': 2, 'gbr': 3, 'brg': 4, 'bgr': 5}
# context_row 열 / 보고 지표
ROW = ('success', 'goal1', 'goal2', 'violation_steps', 'evaluated_steps', 'action_change_rad', 'command_jerk_rad_s3',
       'hand_jerk_m_s3', 'boundary_jump_rad_s', 'entropy', 'abort', 'plan_ms')
METRICS = ('success_rate_pct', 'goal_entry_1_pct', 'goal_entry_2_pct', 'constraint_violation_rate_pct',
           'action_change_rad', 'command_jerk_rad_s3', 'hand_jerk_m_s3', 'boundary_jump_rad_s',
           'behavior_entropy', 'method_abort_rate_pct', 'plan_ms')


# ---- 공통 --------------------------------------------------------------------------
def load_config():
    cfg = json.loads(CONFIG.read_text())
    cfg['settings'] = json.loads((REPO / cfg['settings_file']).read_text())
    return cfg


def check_cond(cond):
    if cond not in CONDITIONS:
        raise ValueError(f'Unknown condition {cond!r}; available: {CONDITIONS}')
    return cond


def combo_id(number):
    return f'S{number:02d}'


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def code_fingerprint():
    digest = hashlib.sha256()
    for relative in CODE_FILES:
        digest.update(relative.encode() + b'\0' + (REPO / relative).read_bytes() + b'\0')
    return digest.hexdigest()


def dirty_code_files():
    out = subprocess.check_output(['git', 'status', '--porcelain', '--', *CODE_FILES], cwd=REPO, text=True)
    return [line[3:] for line in out.splitlines() if line.strip()]


def report_path(cond, name):
    return REPORT_DIR / f'report_{check_cond(cond)}_{name}.md'


def base_contexts(cfg):
    return pickle.load((REPO / cfg['contexts_file']).open('rb'))


def context_sha(context):
    """stacking context ([pos(x,y,deg), quat] × 4)를 6자리로 반올림해 hash한다."""
    values = [[np.round(np.asarray(v, dtype=float), 6).tolist() for v in part] for part in context]
    return hashlib.sha256(json.dumps(values).encode()).hexdigest()


def noise_seed(test_index, seed):
    """팀원 비교 실험(fm_constraint/eval_stacking.py)과 같은 규칙. repeat 자리에 noise seed 0·1·2."""
    payload = json.dumps(['stacking', f'test_{test_index:03d}', seed], separators=(',', ':'))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], 'big') % (2 ** 63 - 1)


# ---- select ------------------------------------------------------------------------
def select():
    if CONTEXT_FILE.exists():
        raise FileExistsError(f'{CONTEXT_FILE} exists; the evaluation list is frozen')
    cfg = load_config()
    contexts = base_contexts(cfg)
    low, high = cfg['test_range']
    chosen = sorted(int(i) for i in np.random.default_rng([SELECTION_SEED, SELECTION_STREAM])
                    .permutation(np.arange(low, high))[:N_CONTEXTS])
    out = {'protocol': PROTOCOL, 'selection_seed': SELECTION_SEED, 'selection_stream': SELECTION_STREAM,
           'n_contexts': N_CONTEXTS, 'noise_seeds': list(SEEDS),
           'source': f'{cfg["contexts_file"]} test [{low},{high}) (stacking baseline protocol.json)',
           'source_sha256': sha256_file(REPO / cfg['contexts_file']),
           'contexts': [{'index': k, 'test_index': i, 'context_id': f'test_{i:03d}',
                         'context_sha256': context_sha(contexts[i])} for k, i in enumerate(chosen)]}
    CONTEXT_FILE.write_text(json.dumps(out, indent=1) + '\n')
    print(f'Saved {CONTEXT_FILE}: {chosen}')


def load_selection(cfg):
    sel = json.loads(CONTEXT_FILE.read_text())
    if sel['source_sha256'] != sha256_file(REPO / cfg['contexts_file']):
        raise ValueError(f'{cfg["contexts_file"]} differs from {CONTEXT_FILE.name}')
    return sel['contexts']


def condition_contexts(cfg, cond):
    """조건의 context 배열(test index로 접근)과 OOD 메타데이터."""
    base = base_contexts(cfg)
    if cond == 'id':
        return {i: c for i, c in enumerate(base)}, None
    root = REPO / cfg['ood_root']
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['base_contexts']['sha256'] != sha256_file(REPO / cfg['contexts_file']):
        raise ValueError(f'{root} was generated from different base contexts')
    name = f'stacking_{cond}_contexts.pkl'
    path = root / name
    sha = sha256_file(path)
    if manifest['files'][name] != sha:
        raise ValueError(f'{path} does not match its manifest')
    start = manifest['base_contexts']['indices'][0]
    ctxs = pickle.load(path.open('rb'))
    rows = [json.loads(line) for line in (root / 'stacking.jsonl').open()]
    rows = {r['context_index']: r for r in rows if r['cond'] == cond}
    meta = {'cond': cond, 'file': str(path.relative_to(REPO)), 'sha256': sha, 'manifest_version': manifest['version'],
            'level_m': manifest['levels_m'][cond], 'shift': {i: rows[i]['shift'] for i in rows}}
    return {start + k: c for k, c in enumerate(ctxs)}, meta


# ---- 평가 --------------------------------------------------------------------------
class Runtime:
    """프로세스당 한 번: 모델·환경·PF cache를 읽는다."""

    def __init__(self, cfg, device, max_steps):
        import torch
        from fm_baseline.eval_pushing import Policy as BasePolicy, load_model
        from fm_baseline.eval_stacking import setup
        from . import checkpoint_reads
        from .joint import JointPotential
        from .tasks import setup_d3il
        self.torch = torch
        self.cfg, self.settings, self.device = cfg, cfg['settings'], device
        run_dir = REPO / cfg['run_dir']
        with checkpoint_reads():
            c, model, norm, cond, solver, step = load_model(str(run_dir), cfg['checkpoint'], 'ema', device)
        assert (c['obs_dim'], c['act_dim'], c['horizon']) == (23, 8, 16)
        self.checkpoint_step = step
        self.base = BasePolicy(model, norm, c, cond, solver, cfg['ode_steps'], device)
        cache = REPO / cfg['pf_cache']
        self.cache_sha256 = sha256_file(cache) if cache.exists() else None
        self.potential = JointPotential(cache, device) if cache.exists() else None
        d3il = Path(os.path.expanduser(cfg['d3il_root']))
        d3il = d3il if d3il.is_absolute() else (REPO / d3il).resolve()
        cwd = os.getcwd()
        setup_d3il(str(d3il))
        Env, _, self.quat = setup(d3il)          # setup은 cwd를 D3IL로 바꾼다
        self.env = Env(render=False, n_substeps=30, max_steps_per_episode=max_steps)
        self.env.start()
        os.chdir(cwd)
        self.max_steps = max_steps

    def policy(self, number):
        from .joint_combo import JointComboPolicy
        flags = combo_flags(number)
        if flags['pf2mp'] and self.potential is None:
            raise FileNotFoundError(f'PF cache {self.cfg["pf_cache"]} is missing (needed by ②)')
        return JointComboPolicy(self.base, self.settings, self.potential if flags['pf2mp'] else None, flags,
                                rtc_beta=self.cfg['rtc_beta'], ode_steps=self.cfg['ode_steps'])


def rms_diff(x, degree, dt):
    value = np.diff(np.asarray(x, dtype=float), n=degree, axis=0) / dt ** degree
    return float(np.sqrt(np.mean(np.sum(value ** 2, axis=1)))) if len(value) else None


def run_episode(rt, number, context, test_index, seed):
    """rollout 하나. 팀원 eval_stacking 루프 + 방법별 플래그·④ suffix·관절 위반 측정."""
    from fm_baseline.eval_stacking import full_success, observation, state
    from .joint import JointConstraints, solve_qp
    torch, env, s = rt.torch, rt.env, rt.settings
    flags = combo_flags(number)
    policy = rt.policy(number)
    dt, execute, tol = s['dt_s'], rt.cfg['execute_steps'], s['solver']['feasibility_tolerance_m']
    nseed = noise_seed(test_index, seed)
    gen = torch.Generator(device='cpu').manual_seed(nseed)
    env.reset(random=False, context=copy.deepcopy(context))
    q, tcp, g, boxes = state(env, rt.quat)
    des = q.copy()
    env.controller.setSetPoint(des.copy())
    initial_q, initial_c = q.copy(), tcp.copy()
    target = np.array(env.scene.get_obj_pos(env.target_box))
    previous = np.zeros(7)
    history = [des.copy()]
    queue, active = [], None                  # active: 실행 중인 계획의 행동 (16,8)
    acts, joints, tcps, bxs, seams, viol, viol_pos, viol_acc, filt = [], [], [], [], [], [], [], [], []
    plan_ms, rtc_guided, interventions = [], 0, 0
    failure, success, info, plan_id = None, False, {}, -1
    for k in range(rt.max_steps):
        seam = False
        if not queue:
            obs = observation(des, g, boxes)
            noise = torch.randn(1, 16, 31, generator=gen, dtype=rt.base.center.dtype).to(rt.base.device)
            suffix = active[execute:, :7].copy() if (flags['rtc'] and active is not None) else None
            if rt.device.startswith('cuda'):
                torch.cuda.synchronize(rt.device)
            started = perf_counter()
            plan = policy.plan(obs, previous.copy(), np.asarray(history[-3:]), noise, rtc_suffix=suffix)
            if rt.device.startswith('cuda'):
                torch.cuda.synchronize(rt.device)
            plan_ms.append(1000 * (perf_counter() - started))
            rtc_guided += int(policy.last_report.get('rtc_guided', False))
            plan_id += 1
            if plan is None:
                failure = policy.last_report['status']
                break
            active = np.asarray(plan[:, :8], dtype=np.float64)
            queue = [active[i].copy() for i in range(execute)]
            seam = True
        action = queue.pop(0)
        nominal = action.copy()
        con = JointConstraints(des, previous, 1, s)
        correction = 0.0
        if flags['filter']:
            filtered = solve_qp(action[:7], con.G, con.d, s['solver'])
            value = filtered.pop('value')
            if value is None:
                failure = filtered['status']
                break
            action = action.copy()
            action[:7] = value
            assert action[7] == nominal[7]
            correction = float(np.linalg.norm(action[:7] - nominal[:7]))
            if correction > s['intervention_tolerance_rad']:
                queue, active = [], None
                interventions += 1
        excess = np.maximum(con.G @ action[:7] - con.d, 0)
        viol_pos.append(bool(excess[:14].max() > tol))
        viol_acc.append(bool(excess[14:].max() > tol))
        viol.append(viol_pos[-1] or viol_acc[-1])
        filt.append(correction)
        des = des + action[:7]
        history.append(des.copy())
        previous = action[:7].copy()
        _, _, done, info = env.step(np.r_[des, action[7]])
        q, tcp, g, boxes = state(env, rt.quat)
        success = full_success(boxes, target, env.pos_min_dist)
        acts.append(action)
        joints.append(q.copy())
        tcps.append(tcp.copy())
        bxs.append(boxes.copy())
        seams.append(seam)
        if success or done:
            break
    success = bool(success and failure is None)
    n = len(acts)
    act = np.asarray(acts, dtype=float).reshape(-1, 8)
    starts = np.flatnonzero(seams)
    boundary = (np.linalg.norm((act[starts[1:], :7] - act[starts[1:] - 1, :7]) / dt, axis=1)
                if len(starts) > 1 else np.empty(0))
    mode = str(info.get('mode', ''))
    episode = {
        'sampling_seed': seed, 'noise_seed': nseed, 'success': success,
        'success_1': bool(info.get('success_1', False)), 'success_2': bool(info.get('success_2', False)),
        'mode': MODE3.get(mode[:3], -1), 'mode_string': mode, 'steps': n, 'failure': failure,
        'termination': failure or ('success' if success else 'step_budget' if n == rt.max_steps else 'environment_done'),
        'constraint_violation_steps': int(sum(viol)), 'constraint_evaluated_steps': n,
        'constraint_violation_steps_by_type': {'joint_position': int(sum(viol_pos)), 'acceleration': int(sum(viol_acc))},
        # main 정의(evaluation.py)와 같은 형태: 연속 행동 차이 norm 평균, 2차 차분 norm 평균 / dt³ (rad 단위)
        'action_change_rad': float(np.linalg.norm(np.diff(act[:, :7], axis=0), axis=-1).mean()) if n > 1 else None,
        'command_jerk_rad_s3': (float(np.linalg.norm(np.diff(act[:, :7], n=2, axis=0), axis=-1).mean() / dt ** 3)
                                if n > 2 else None),
        'hand_jerk_m_s3': rms_diff(np.vstack([initial_c, np.asarray(tcps).reshape(-1, 3)]), 3, dt) if n else None,
        'actual_joint_jerk_rad_s3': rms_diff(np.vstack([initial_q, np.asarray(joints).reshape(-1, 7)]), 3, dt) if n else None,
        'boundary_jump_rad_s': float(np.sqrt(np.mean(boundary ** 2))) if len(boundary) else None,
        'plan_calls': len(plan_ms), 'plan_ms_mean': float(np.mean(plan_ms)) if plan_ms else None,
        'rtc_guided_plans': rtc_guided, 'filter_interventions': interventions,
        'filter_correction_max_rad': float(max(filt)) if filt else 0.0,
    }
    arrays = {'action': act.astype(np.float32), 'joint': np.asarray(joints, np.float32).reshape(-1, 7),
              'tcp': np.asarray(tcps, np.float32).reshape(-1, 3), 'boxes': np.asarray(bxs, np.float32).reshape(-1, 3, 4),
              'seam': np.asarray(seams, bool), 'violation': np.asarray(viol, bool),
              'filter_rad': np.asarray(filt, np.float32), 'initial_joint': initial_q, 'initial_tcp': initial_c}
    return episode, arrays


# ---- run ---------------------------------------------------------------------------
def device_info(device):
    import torch
    info = {'requested': device, 'torch': torch.__version__, 'python': platform.python_version(),
            'cpu': platform.processor() or platform.machine()}
    if device.startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError(f'{device} requested but CUDA is not available')
        info.update(gpu=torch.cuda.get_device_name(torch.device(device).index or 0), cuda=torch.version.cuda)
    return info


def unit_dir(root, cond, number, index):
    return root / cond / combo_id(number) / f'ctx_{index:02d}'


@contextmanager
def unit_lock(cond, number, index, smoke):
    path = LOCAL_BASE / 'locks' / ('smoke' if smoke else 'full') / f'{cond}_{combo_id(number)}_ctx_{index:02d}.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            handle.write(f'{os.getpid()}\n')
            handle.flush()
            yield True
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def run(conds, combos, indices, device, name, max_steps=None, allow_dirty=False, pf_cache=None):
    check_name(name)
    for cond in conds:
        check_cond(cond)
    smoke = max_steps is not None
    dirty = dirty_code_files()
    if dirty and not (smoke or allow_dirty):
        raise SystemExit(f'Uncommitted changes in result-affecting files: {dirty}. Commit first (or --allow-dirty)')
    cfg = load_config()
    if pf_cache:
        cfg['pf_cache'] = str(Path(pf_cache).resolve())   # 본실험에서도 아래 sha256 고정 검사를 통과해야 한다
    if not smoke and cfg['pf_cache_sha256'] is None:
        raise SystemExit('configs/stacking_experiment.json pf_cache_sha256 is not pinned; '
                         'pin the shared PF cache before the main run')
    rt = Runtime(cfg, device, max_steps or cfg['max_steps'])
    if not smoke and rt.cache_sha256 != cfg['pf_cache_sha256']:
        raise SystemExit(f'PF cache sha256 {rt.cache_sha256} != pinned {cfg["pf_cache_sha256"]}')
    rows = load_selection(cfg)
    fingerprint = code_fingerprint()
    info = device_info(device)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    root = (LOCAL_BASE / 'smoke') if smoke else RESULT_BASE
    base = base_contexts(cfg)
    for row in rows:
        if context_sha(base[row['test_index']]) != row['context_sha256']:
            raise ValueError(f'context {row["context_id"]} differs from {CONTEXT_FILE.name}')
    for cond in conds:
        contexts, ood = condition_contexts(cfg, cond)
        for number in combos:
            for index in indices:
                with unit_lock(cond, number, index, smoke) as acquired:
                    if not acquired:
                        print(f'BUSY {cond} {combo_id(number)} ctx_{index:02d}', flush=True)
                        continue
                    run_unit(rt, cfg, cond, number, rows[index], contexts, ood, name, fingerprint, info, commit,
                             root, smoke)


def run_unit(rt, cfg, cond, number, row, contexts, ood, name, fingerprint, info, commit, root, smoke):
    index, ti = row['index'], row['test_index']
    out = unit_dir(root, cond, number, index)
    result_path = out / 'result.json'
    if result_path.exists():
        saved = json.loads(result_path.read_text())
        if saved['code_fingerprint'] != fingerprint:
            raise ValueError(f'{result_path} was produced by different code; do not mix results')
        print(f'SKIP {cond} {combo_id(number)} ctx_{index:02d} (done by {saved["member"]})', flush=True)
        return
    print(f'RUN  {cond} {combo_id(number)} ctx_{index:02d} ({row["context_id"]}) seeds {list(SEEDS)}', flush=True)
    started = perf_counter()
    episodes, arrays = [], {}
    for seed in SEEDS:
        episode, arr = run_episode(rt, number, contexts[ti], ti, seed)
        episodes.append(episode)
        arrays.update({f't{seed}_{k}': v for k, v in arr.items()})
        print(f'  seed {seed} success={int(episode["success"])} goal={int(episode["success_1"])}'
              f'{int(episode["success_2"])} steps={episode["steps"]} failure={episode["failure"]}', flush=True)
    shift = (ood or {}).get('shift', {}).get(ti)
    record = {'protocol': PROTOCOL, 'cond': cond, 'task': 'stacking', 'combo': combo_id(number), 'number': number,
              'methods': combo_flags(number), 'context_index': index, 'context_id': row['context_id'],
              'context_sha256': row['context_sha256'], 'noise_seeds': list(SEEDS), 'member': name,
              'ood': None if ood is None else {k: ood[k] for k in ('cond', 'file', 'sha256', 'manifest_version', 'level_m')},
              'shift': shift, 'blocks_moved': None if shift is None else sum(e['face'] is not None for e in shift),
              'device': info, 'code_fingerprint': fingerprint, 'git_commit': commit,
              'checkpoint_step': rt.checkpoint_step, 'pf_cache_sha256': rt.cache_sha256,
              'max_steps': rt.max_steps, 'smoke': smoke,
              'finished_utc': datetime.now(timezone.utc).isoformat(), 'wall_s': perf_counter() - started,
              'episodes': episodes}
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'trajectories.npz', **arrays)
    result_path.write_text(json.dumps(record, indent=1, allow_nan=False) + '\n')   # 마지막에 쓴다 = 완료 표시
    print(f'DONE {cond} {combo_id(number)} ctx_{index:02d} success '
          f'{np.mean([e["success"] for e in episodes]):.2f} {record["wall_s"]:.0f}s', flush=True)


# ---- 집계 ---------------------------------------------------------------------------
def context_row(episodes):
    successes = [e for e in episodes if e['success']]
    counts = np.array([sum(e['mode'] == m for e in successes) for m in range(6)], dtype=float)
    entropy = np.nan
    if counts.sum():
        p = counts[counts > 0] / counts.sum()
        entropy = float(-(p * np.log(p)).sum() / np.log(6))
    vals = lambda key: _mean(np.array([np.nan if e[key] is None else e[key] for e in episodes], float), 0)  # noqa: E731
    return [np.mean([e['success'] for e in episodes]), np.mean([e['success_1'] for e in episodes]),
            np.mean([e['success_2'] for e in episodes]),
            sum(e['constraint_violation_steps'] for e in episodes), sum(e['constraint_evaluated_steps'] for e in episodes),
            vals('action_change_rad'), vals('command_jerk_rad_s3'), vals('hand_jerk_m_s3'), vals('boundary_jump_rad_s'),
            entropy, np.mean([e['failure'] is not None for e in episodes]), vals('plan_ms_mean')]


def aggregate(frame):
    """(…, context, len(ROW)) → (…, len(METRICS)). 위반율은 step 분모를 합쳐서 계산한다."""
    num, den = frame[..., 3].sum(-1), frame[..., 4].sum(-1)
    violation = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
    return np.stack([frame[..., 0].mean(-1) * 100, frame[..., 1].mean(-1) * 100, frame[..., 2].mean(-1) * 100,
                     violation * 100, _mean(frame[..., 5], -1), _mean(frame[..., 6], -1), _mean(frame[..., 7], -1),
                     _mean(frame[..., 8], -1), _mean(frame[..., 9], -1), frame[..., 10].mean(-1) * 100,
                     _mean(frame[..., 11], -1)], axis=-1)


def _finite_or_none(values):
    return [float(v) if np.isfinite(v) else None for v in np.asarray(values, dtype=float)]


def flags_text(number):
    return ''.join('●' if f else '·' for f in combo_flags(number).values())


TABLE_HEAD = ('| 조합 | ①②③④⑤ | 성공 % | 목표 진입 1개 % | 목표 진입 2개 % | 위반 % | action change mrad | '
              'jerk rad/s³ | 손 jerk m/s³ | 경계 변화 rad/s | entropy | 방법 실패 % | 계획 ms |')
TABLE_SEP = '|---|---|' + '---:|' * 11


def report(cond, name, out=None):
    """한 팀원이 실행한 단위만 모은 보고서. 점추정 표 + merge용 context별 원시 데이터."""
    check_name(name)
    check_cond(cond)
    units = [json.loads(p.read_text()) for p in sorted((RESULT_BASE / cond).glob('S*/ctx_*/result.json'))]
    units = [u for u in units if u['member'] == name]
    if not units:
        raise SystemExit(f'No {cond} results by {name} under {RESULT_BASE / cond}')
    fingerprints = {u['code_fingerprint'] for u in units}
    lines = [f'# Stacking 실험 보고 ({cond}): {name}', '',
             f'- protocol `{PROTOCOL}`, 조건 `{cond}`, noise seed {list(SEEDS)}, 생성 {datetime.now(timezone.utc).isoformat()}',
             f'- code fingerprint: {", ".join(sorted(f[:12] for f in fingerprints))}'
             + (' ⚠ 여러 개: 코드가 다른 결과가 섞였다' if len(fingerprints) > 1 else ''),
             f'- git commit: {", ".join(sorted({u["git_commit"][:8] for u in units}))}',
             f'- PF cache sha256: {", ".join(sorted({str(u["pf_cache_sha256"])[:12] for u in units}))}',
             f'- 장비: {", ".join(sorted({u["device"].get("gpu", "cpu") for u in units}))}']
    if cond != 'id':
        lines.append(f'- OOD 파일 sha256: {", ".join(sorted({u["ood"]["sha256"][:12] for u in units}))}')
    lines += ['', '이 팀원이 실행한 context만의 점추정이다(신뢰구간 없음). 전체 CI는 merge 결과를 본다.', '',
              TABLE_HEAD.replace('| 계획 ms |', '| 계획 ms | context |'), TABLE_SEP + '---|']
    for number in sorted({u['number'] for u in units}):
        group = sorted((u for u in units if u['number'] == number), key=lambda u: u['context_index'])
        v = aggregate(np.array([context_row(u['episodes']) for u in group], dtype=float))
        lines.append(f'| {combo_id(number)} | {flags_text(number)} | ' + ' | '.join(
            [fmt(v[0]), fmt(v[1]), fmt(v[2]), fmt(v[3]), fmt(v[4], 1e3), fmt(v[5]), fmt(v[6]), fmt(v[7]), fmt(v[8]),
             fmt(v[9]), fmt(v[10])]) + f' | {",".join(str(u["context_index"]) for u in group)} |')
    data = {'format': REPORT_FORMAT, 'protocol': f'{PROTOCOL}_{cond}', 'cond': cond, 'member': name,
            'noise_seeds': list(SEEDS), 'row_columns': list(ROW), 'units': []}
    for u in sorted(units, key=lambda u: (u['number'], u['context_index'])):
        data['units'].append({
            'number': u['number'], 'context_index': u['context_index'], 'context_id': u['context_id'],
            'code_fingerprint': u['code_fingerprint'], 'git_commit': u['git_commit'],
            'pf_cache_sha256': u['pf_cache_sha256'], 'ood_sha256': (u['ood'] or {}).get('sha256'),
            'blocks_moved': u['blocks_moved'], 'faces': [e['face'] for e in u['shift']] if u['shift'] else None,
            'gpu': u['device'].get('gpu', 'cpu'), 'wall_s': u['wall_s'],
            'seeds': sorted(e['sampling_seed'] for e in u['episodes']),
            'row': _finite_or_none(context_row(u['episodes']))})
    text = json.dumps(data, separators=(',', ':'), allow_nan=False)
    lines += ['', '<details><summary>merge용 원시 데이터 (context별 지표, 자동 생성)</summary>', '', REPORT_MARKER, '',
              '```json', text.replace('},{', '},\n{'), '```', '', '</details>', '']
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = Path(out) if out else report_path(cond, name)
    path.write_text('\n'.join(lines))
    print(f'Saved {path}')


def parse_report(path):
    text = Path(path).read_text()
    if REPORT_MARKER not in text:
        return None
    block = text.split(REPORT_MARKER, 1)[1]
    start = block.index('```json\n') + len('```json\n')
    data = json.loads(block[start:block.index('\n```', start)])
    if data.get('format') != REPORT_FORMAT:
        raise SystemExit(f'{path}: unknown report format {data.get("format")!r}')
    return data


def merge(cond, out=None, n_boot=2000, seed=2026, report_dir=None):
    """docs/stacking/report_<cond>_*.md만 읽어 조합별 추정값 [95% CI], Complementarity Δ, A 블록 수별 성공률."""
    check_cond(cond)
    units, sources, owner = [], [], {}
    for path in sorted(Path(report_dir or REPORT_DIR).glob(f'report_{cond}_*.md')):
        data = parse_report(path)
        if data is None or data['protocol'] != f'{PROTOCOL}_{cond}':
            continue
        if data['noise_seeds'] != list(SEEDS) or data['row_columns'] != list(ROW):
            raise SystemExit(f'{path.name}: different seeds or row layout')
        sources.append((path.name, data['member'], len(data['units'])))
        for u in data['units']:
            key = (u['number'], u['context_index'])
            if key in owner:
                raise SystemExit(f'{combo_id(u["number"])} ctx_{u["context_index"]:02d} in both {owner[key]} and {path.name}')
            owner[key] = path.name
            units.append({**u, 'row': [np.nan if v is None else v for v in u['row']]})
    if not units:
        raise SystemExit(f'No {cond} reports in {report_dir or REPORT_DIR}')
    for key in ('code_fingerprint', 'pf_cache_sha256', 'ood_sha256'):
        if len({u[key] for u in units}) > 1:
            raise SystemExit(f'Reports mix several {key} values; rerun mismatched units first')
    lines = [f'# Stacking 통합 결과 ({cond}, merge)', '',
             f'protocol `{PROTOCOL}`, context {N_CONTEXTS}개 × noise seed {list(SEEDS)}, context bootstrap {n_boot}회'
             f'(seed {seed}), code fingerprint {units[0]["code_fingerprint"][:12]}, PF cache {str(units[0]["pf_cache_sha256"])[:12]}'
             + (f', OOD 파일 {units[0]["ood_sha256"][:12]}' if cond != 'id' else ''), '',
             '입력 보고서: ' + ', '.join(f'`{f}`({m}, 단위 {n}개)' for f, m, n in sources), '']
    expected = list(range(N_CONTEXTS))
    table, missing = {}, []
    for number in range(32):
        group = {u['context_index']: u for u in units if u['number'] == number}
        absent = [i for i in expected if i not in group]
        if absent:
            missing.append(f'{combo_id(number)}: {len(absent)}개')
            continue
        if any(u['seeds'] != list(SEEDS) for u in group.values()):
            raise SystemExit(f'{combo_id(number)}: seeds incomplete')
        table[number] = np.array([group[i]['row'] for i in expected], dtype=float)
    if missing:
        lines += ['미완료(누락 context 수): ' + ', '.join(missing), '']
    summary = {}
    if table:
        index = np.random.default_rng(seed).integers(N_CONTEXTS, size=(n_boot, N_CONTEXTS))
        values = {n: aggregate(f) for n, f in table.items()}
        draws = {n: aggregate(f[index]) for n, f in table.items()}

        def cell(v, s, scale=1.):
            s = s[np.isfinite(s)]
            if not np.isfinite(v) or not len(s):
                return '—', [None, None, None]
            lo, hi = np.quantile(s, [.025, .975])
            return f'{v * scale:.3g} [{lo * scale:.3g}, {hi * scale:.3g}]', [float(v), float(lo), float(hi)]

        lines += [TABLE_HEAD.replace('| 계획 ms |', '| 계획 ms | Δ 성공 %p (A, B) |'), TABLE_SEP + '---|']
        for number in sorted(table):
            cells, row = [], {}
            for i, metric in enumerate(METRICS):
                text, row[metric] = cell(values[number][i], draws[number][:, i], 1e3 if metric == 'action_change_rad' else 1.)
                cells.append(text)
            low = number & -number
            delta = 'N/A'
            if number != low and number - low in table and low in table:
                a, b = number - low, low
                v = values[number][0] - max(values[a][0], values[b][0])
                sd = draws[number][:, 0] - np.maximum(draws[a][:, 0], draws[b][:, 0])
                lo, hi = np.quantile(sd, [.025, .975])
                delta = f'{v:+.3g} [{lo:+.3g}, {hi:+.3g}] ({combo_id(a)}, {combo_id(b)})'
                row['complementarity_success_pp'] = [float(v), float(lo), float(hi)]
            lines.append(f'| {combo_id(number)} | {flags_text(number)} | ' + ' | '.join(cells) + f' | {delta} |')
            summary[combo_id(number)] = row
        lines.append('')
        if cond != 'id':
            moved = {u['context_index']: u['blocks_moved'] for u in units}
            groups = sorted({moved[i] for i in expected if i in moved})
            lines += ['## 옮겨진 블록 수별 성공률 (점추정, %)', '',
                      'A 배치에서 실제로 옮겨진 블록 수로 context를 나눴다. 방향(x±/y±)별 분석은 보고서 원시 데이터의 `faces`를 쓴다.', '',
                      '| 조합 | ' + ' | '.join(f'{g}개 (context {sum(moved[i] == g for i in expected)})' for g in groups) + ' |',
                      '|---|' + '---:|' * len(groups)]
            for number in sorted(table):
                f = table[number]
                lines.append(f'| {combo_id(number)} | ' + ' | '.join(
                    fmt(f[[moved[i] == g for i in expected], 0].mean() * 100) for g in groups) + ' |')
            lines.append('')
    path = Path(out) if out else REPORT_DIR / f'report_{cond}_results.md'
    path.write_text('\n'.join(lines))
    meta = {'protocol': PROTOCOL, 'cond': cond, 'n_boot': n_boot, 'bootstrap_seed': seed,
            'code_fingerprint': units[0]['code_fingerprint'], 'pf_cache_sha256': units[0]['pf_cache_sha256'],
            'ood_sha256': units[0]['ood_sha256'], 'reports': [f for f, _, _ in sources]}
    path.with_suffix('.json').write_text(json.dumps({'meta': meta, 'combos': summary}, indent=1))
    print(f'Saved {path} and {path.with_suffix(".json").name}')


def parse_list(text, allowed):
    items = [x.strip() for x in text.split(',') if x.strip()]
    for item in items:
        if item not in allowed:
            raise SystemExit(f'Unknown {item!r}; available {allowed}')
    return items


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('select')
    r = sub.add_parser('run')
    r.add_argument('--cond', required=True, help="id, A_weak, A_mid, A_strong (쉼표로 여러 개, 적은 순서대로 실행)")
    r.add_argument('--combos', default='all', help="'all', '0-31', '5,17' 또는 'S05,S17'")
    r.add_argument('--contexts', required=True, help="context 번호 0..29: '0-6', '7,9'")
    r.add_argument('--device', required=True)
    r.add_argument('--name', required=True)
    r.add_argument('--max-steps', type=int, default=None, help='동작 확인용. 결과는 로컬 smoke 폴더에만')
    r.add_argument('--allow-dirty', action='store_true')
    r.add_argument('--pf-cache', default=None, help='PF cache 경로(기본: config). 본실험은 고정 sha256과 같아야 한다')
    p = sub.add_parser('report')
    p.add_argument('--cond', required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--out')
    m = sub.add_parser('merge')
    m.add_argument('--cond', required=True)
    m.add_argument('--out')
    args = parser.parse_args()
    if args.command == 'select':
        select()
    elif args.command == 'run':
        combos = parse_indices(','.join(x.strip().lstrip('S') for x in args.combos.split(',')), 32)
        run(parse_list(args.cond, CONDITIONS), combos, parse_indices(args.contexts, N_CONTEXTS), args.device,
            args.name, args.max_steps, args.allow_dirty, args.pf_cache)
    elif args.command == 'report':
        report(args.cond, args.name, args.out)
    else:
        merge(args.cond, args.out)


if __name__ == '__main__':
    main()
