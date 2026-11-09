"""본실험(ID·OOD): 32조합 × (30 context × noise 3) × Pushing·Inserting.

    python -m fm_constraint.experiment select                 # 평가 목록 생성 (한 번, commit)
    python -m fm_constraint.experiment run --task pushing --combos 0-31 --contexts 0-6 \\
        --device cuda:0 --name hyeongbin [--ood A_mid]         # 맡은 context 실행
    python -m fm_constraint.experiment report --name hyeongbin [--ood A_mid]
        # docs/report_hyeongbin.md (OOD: docs/report_ood_A_mid_hyeongbin.md)
    python -m fm_constraint.experiment merge [--ood A_mid]
        # 누구나: docs/report_*.md만 읽어 docs/report_results.{md,json}
        #         (OOD: docs/report_ood_A_mid_results.{md,json})

평가 단위는 (task, 조합, context 번호) 하나이며 noise seed 0, 1, 2의 rollout 3개를 담는다.
OOD도 ID와 같은 30 context·같은 seed·같은 FM noise를 쓰고, 초기 배치 등 OOD 조건만 다르다.
단위마다 결과를 바로 저장하므로 중단 후 같은 명령을 다시 실행하면 완료된 단위를 건너뛴다.

결과는 모두 로컬에만 둔다(git 제외):
    experiments/id_v1/<task>/<조합>/ctx_NN/{result.json, trajectories.npz}
    experiments/ood_v1/<조건>/<task>/<조합>/ctx_NN/...
    fm_constraint/runs/experiment_<protocol>/...   전체 rollouts.json
git에는 보고서(docs/report_*.md)만 올린다. 보고서에 context별 원시 지표가 들어 있어
merge는 보고서만으로 30 context bootstrap CI를 계산한다.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import subprocess
from time import perf_counter

import numpy as np

from configs.config import METHODS, ExperimentConfig, REPO

PROTOCOL = 'id_v1'
OOD_PROTOCOL = 'ood_v1'
REPORT_FORMAT = 'yflow-report-v1'
REPORT_MARKER = '<!-- yflow-report-data: merge가 읽는다. 수정하지 않는다 -->'
SELECTION_SEED = 20261108
N_CONTEXTS = 30
SEEDS = (0, 1, 2)
TASKS = ('pushing', 'inserting')
BITS = dict(zip(METHODS, (16, 8, 4, 2, 1)))
CONTEXT_FILE = REPO / 'configs/experiment_contexts.json'
RESULT_BASE = REPO / 'experiments'            # 단위 결과(로컬 전용, .gitignore)
LOCAL_BASE = REPO / 'fm_constraint/runs'        # 전체 rollout 로그·잠금(로컬 전용)
REPORT_DIR = REPO / 'docs'                      # git에 올리는 보고서
# 결과에 영향을 주는 코드·설정. 팀원이 결과를 push해 HEAD가 바뀌어도 이 hash는 같아야 한다.
CODE_FILES = ('configs/config.py', 'configs/pushing.yaml', 'configs/inserting.yaml',
              'configs/experiment_contexts.json', 'requirements.txt',
              'fm_constraint/evaluation.py', 'fm_constraint/policy.py', 'fm_constraint/constraints.py',
              'fm_constraint/qp.py', 'fm_constraint/pf2mp.py', 'fm_constraint/rtc.py', 'fm_constraint/lipo.py',
              'fm_constraint/experiment.py', 'fm_constraint/__init__.py', 'ood/apply.py',
              'fm_baseline/eval_pushing.py', 'fm_baseline/eval_inserting.py', 'fm_baseline/train.py',
              'fm_baseline/models/__init__.py', 'fm_baseline/models/temporal_unet.py',
              'fm_baseline/models/flow_matching.py', 'fm_baseline/models/ode.py')
METRICS = ('success_rate_pct', 'constraint_violation_rate_pct', 'action_change_m',
           'command_jerk_m_s3', 'behavior_entropy', 'method_abort_rate_pct')
LOWER_BETTER = {'constraint_violation_rate_pct', 'action_change_m', 'command_jerk_m_s3', 'method_abort_rate_pct'}


# ---- 공통 --------------------------------------------------------------------------
def ood_conditions():
    """본실험에서 쓸 수 있는 OOD 조건 이름('id'는 ID 실험이므로 제외)."""
    from ood.conditions import conditions
    return [name for name, _ in conditions() if name != 'id']


def check_ood(ood):
    if ood is not None and ood not in ood_conditions():
        raise ValueError(f'Unknown OOD condition {ood!r}; available: {ood_conditions()} (ID는 --ood 없이)')
    return ood


def protocol(ood=None):
    """결과·보고서를 구분하는 이름. ID: id_v1, OOD: ood_v1_<조건>."""
    return PROTOCOL if ood is None else f'{OOD_PROTOCOL}_{check_ood(ood)}'


def result_root(ood=None):
    return RESULT_BASE / (PROTOCOL if ood is None else f'{OOD_PROTOCOL}/{check_ood(ood)}')


def local_root(ood=None):
    return LOCAL_BASE / f'experiment_{protocol(ood)}'


def report_path(name, ood=None):
    """팀원 보고서: report_<이름>.md, OOD는 report_ood_<조건>_<이름>.md. 통합본은 name='results'."""
    return REPORT_DIR / (f'report_{name}.md' if ood is None else f'report_ood_{check_ood(ood)}_{name}.md')


def check_name(name):
    if not name or not all(c.isascii() and (c.isalnum() or c in '_-') for c in name):
        raise ValueError('--name must be ASCII letters/digits/_/-')
    if name == 'results' or name.startswith('ood'):
        raise ValueError(f'--name {name!r} is reserved (통합 보고서·OOD 파일 이름과 겹친다)')
    return name


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def context_sha(context):
    """초기 상태 배열을 6자리로 반올림해 hash한다(같은 초기 상태 확인용)."""
    values = [np.round(np.asarray(part, dtype=float), 6).tolist() for part in context]
    return sha256_bytes(json.dumps(values).encode())


def code_fingerprint():
    digest = hashlib.sha256()
    for relative in CODE_FILES:
        digest.update(relative.encode() + b'\0' + (REPO / relative).read_bytes() + b'\0')
    return digest.hexdigest()


def parse_indices(text, upper):
    """'0-6', '3,5,9', 'all' → 정렬된 고유 정수 목록."""
    if text == 'all':
        return list(range(upper))
    values = set()
    for part in text.split(','):
        part = part.strip()
        if '-' in part:
            low, high = (int(x) for x in part.split('-'))
            values.update(range(low, high + 1))
        elif part:
            values.add(int(part))
    if not values or min(values) < 0 or max(values) >= upper:
        raise ValueError(f'Indices must be within 0..{upper - 1}: {text}')
    return sorted(values)


def parse_combos(text):
    """'all', '0-31', '5,17', 'P05,I17' → 조합 번호 목록. P/I 접두사는 무시한다(태스크는 --task)."""
    return parse_indices(','.join(part.strip().lstrip('PI') for part in text.split(',')), 32)


def combo_id(task, number):
    return f'{"P" if task == "pushing" else "I"}{number:02d}'


def combo_flags(number):
    return {name: bool(number & bit) for name, bit in BITS.items()}


def candidate_contexts(task, d3il_root):
    """선택 후보 [(context_id, context, group)]. 정렬 순서가 고정이어야 선택이 재현된다."""
    if task == 'pushing':
        path = Path(d3il_root) / 'environments/dataset/data/pushing/test_contexts.pkl'
        contexts = np.load(path, allow_pickle=True)
        # evaluation.py의 Pushing context ID는 test 배열의 index 문자열이다.
        return [(str(i), ctx, str(i)) for i, ctx in enumerate(contexts)]
    stored = pickle.loads((REPO / 'fm_baseline/contexts/inserting_val.pkl').read_bytes())
    manifest = json.loads((REPO / 'data/inserting_sincos_v1/manifest.json').read_text())
    groups = {}
    for row in manifest['splits']['val']:
        groups.setdefault(row['context_group'], []).append(Path(row['file']).name)
    # 그룹마다 대표 1개(이름순 첫 번째)만 후보로 둔다. 같은 그룹이 두 번 뽑히지 않는다.
    return [(min(files), stored[min(files)]['context'], group) for group, files in sorted(groups.items())]


# ---- select ------------------------------------------------------------------------
def select(d3il_root):
    """태스크마다 고정 seed로 30개를 뽑아 번호 0..29를 붙인다. 이미 있으면 덮어쓰지 않는다."""
    if CONTEXT_FILE.exists():
        raise FileExistsError(f'{CONTEXT_FILE} exists; the evaluation list is frozen')
    out = {'protocol': PROTOCOL, 'selection_seed': SELECTION_SEED, 'n_contexts': N_CONTEXTS,
           'noise_seeds': list(SEEDS), 'tasks': {}}
    for offset, task in enumerate(TASKS):
        candidates = candidate_contexts(task, d3il_root)
        unique, seen = [], set()
        for cid, ctx, group in candidates:
            digest = context_sha(ctx)
            if digest not in seen:
                seen.add(digest)
                unique.append((cid, digest, group))
        order = np.random.default_rng([SELECTION_SEED, offset]).permutation(len(unique))[:N_CONTEXTS]
        # 번호는 context ID 순서(Pushing은 숫자 순)로 붙인다.
        chosen = sorted((unique[i] for i in order), key=lambda row: (len(row[0]), row[0]) if task == 'pushing' else row[0])
        out['tasks'][task] = {
            'source': ('d3il environments/dataset/data/pushing/test_contexts.pkl (official test)' if task == 'pushing'
                       else 'fm_baseline/contexts/inserting_val.pkl, one per val context_group (validation)'),
            'n_candidates': len(unique),
            'contexts': [{'index': k, 'context_id': cid, 'context_sha256': digest, 'group': group}
                         for k, (cid, digest, group) in enumerate(chosen)]}
    CONTEXT_FILE.write_text(json.dumps(out, indent=1) + '\n')
    print(f'Saved {CONTEXT_FILE}')
    return out


def load_selection(task, d3il_root):
    """목록을 읽고 현재 D3IL·저장소의 초기 상태가 목록과 같은지 확인한다."""
    selection = json.loads(CONTEXT_FILE.read_text())
    rows = selection['tasks'][task]['contexts']
    current = {cid: ctx for cid, ctx, _ in candidate_contexts(task, d3il_root)}
    for row in rows:
        if row['context_id'] not in current or context_sha(current[row['context_id']]) != row['context_sha256']:
            raise ValueError(f'{task} context {row["index"]} ({row["context_id"]}) differs from '
                             f'{CONTEXT_FILE.name}; check the D3IL checkout (commit 1d9c718)')
    return rows


# ---- run ---------------------------------------------------------------------------
def device_info(device):
    import torch
    info = {'requested': device, 'torch': torch.__version__, 'python': platform.python_version(),
            'cpu': platform.processor() or platform.machine()}
    if device.startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError(f'{device} requested but torch.cuda.is_available() is False')
        index = torch.device(device).index or 0
        if index >= torch.cuda.device_count():
            raise RuntimeError(f'{device} requested but only {torch.cuda.device_count()} GPU(s) visible')
        # 실제로 계산이 GPU에서 되는지 확인한다.
        value = (torch.ones(8, device=device) * 2).sum().item()
        if value != 16:
            raise RuntimeError('GPU smoke computation failed')
        info.update(gpu=torch.cuda.get_device_name(index), cuda=torch.version.cuda)
    return info


def unit_dir(root, task, number, index):
    return root / task / combo_id(task, number) / f'ctx_{index:02d}'


def slim_trajectories(rollouts, tolerance):
    """git에 올릴 최소 궤적: 실행 행동, 시작 목표·손 위치, seam·위반·필터 수정량(seed마다).

    목표 위치 궤적은 시작 목표 + 행동 누적으로 복원한다. 실측 손·블록·계획 보고 전체는
    용량이 커서(Inserting rollout 1개에 최대 약 8 MB) 로컬 rollouts.json에만 남긴다.
    """
    arrays = {}
    for key, rows in rollouts.items():
        done = [row for row in rows if row.get('safe_action') is not None]
        t = 't' + key.split('_t')[-1]
        arrays[f'{t}_action'] = np.asarray([r['safe_action'] for r in done], np.float32).reshape(-1, 2)
        arrays[f'{t}_start_command'] = np.asarray(rows[0].get('command_state', [np.nan] * 2), np.float64)
        arrays[f'{t}_start_hand'] = np.asarray(rows[0].get('measured_state', [np.nan] * 3)[:2], np.float64)
        arrays[f'{t}_seam'] = np.asarray([bool(r.get('seam')) for r in done])
        arrays[f'{t}_violation'] = np.asarray([r['latest_executed_violation']['qp_max_m'] > tolerance for r in done])
        arrays[f'{t}_filter_m'] = np.asarray([r.get('intervention_norm_m', 0.) for r in done], np.float32)
    return arrays


def run(task, combos, indices, device, name, max_steps=None, ood=None):
    check_name(name)
    check_ood(ood)
    rows = None
    fingerprint = code_fingerprint()
    smoke = max_steps is not None
    tracked = (local_root(ood) / 'smoke') if smoke else result_root(ood)
    info = device_info(device)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    for number in combos:
        for index in indices:
            with unit_lock(task, number, index, smoke, ood) as acquired:
                if not acquired:
                    # 다른 프로세스(다른 GPU 등)가 지금 이 단위를 실행 중이다. 건너뛰고 다음 단위로 간다.
                    print(f'BUSY {combo_id(task, number)} ctx_{index:02d} (running in another process)', flush=True)
                    continue
                rows = run_unit(task, number, index, device, name, max_steps, rows, fingerprint, info, commit,
                                tracked, ood)


def check_ood_base(task, rows, ood_root):
    """OOD 파일의 원래 배치('id' 조건)가 평가 목록의 초기 상태와 같은지 확인한다."""
    from ood.apply import context_from_json, load_cases
    from .evaluation import ood_context_id
    cases, _ = load_cases(ood_root, task, 'id')
    by_id = {case['context_id']: case for case in cases}
    for row in rows:
        case = by_id.get(ood_context_id(task, row['context_id']))
        if case is None or context_sha(context_from_json(case['context'])) != row['context_sha256']:
            raise ValueError(f'{task} context {row["index"]} ({row["context_id"]}) is missing or differs in '
                             f'{ood_root}; the OOD file does not match {CONTEXT_FILE.name}')


def run_unit(task, number, index, device, name, max_steps, rows, fingerprint, info, commit, tracked, ood=None):
    """잠금을 잡은 상태에서 단위 하나를 실행한다. 이미 완료됐으면 건너뛴다. 평가 목록을 반환한다."""
    from .evaluation import evaluate, ood_context_id
    smoke = max_steps is not None
    out = unit_dir(tracked, task, number, index)
    result_path = out / 'result.json'
    if result_path.exists():
        saved = json.loads(result_path.read_text())
        if saved['code_fingerprint'] != fingerprint:
            raise ValueError(f'{result_path} was produced by different code; do not mix results')
        print(f'SKIP {combo_id(task, number)} ctx_{index:02d} (done by {saved["member"]})', flush=True)
        return rows
    overrides = {**{f'methods.{k}.enabled': v for k, v in combo_flags(number).items()},
                 'device': device, 'eval.n_contexts': 0, 'eval.execute_steps': 8,
                 'sampling.predict_steps': 10, 'eval.ood': ood,
                 'paths.output_root': str(local_root(ood))}
    if smoke:
        overrides['eval.max_steps'] = max_steps
    cfg = ExperimentConfig.from_yaml(REPO / f'configs/{task}.yaml', overrides)
    if rows is None:
        rows = load_selection(task, cfg.values['paths']['d3il_root'])
        if ood is not None:
            check_ood_base(task, rows, cfg.values['paths']['ood_root'])
    row = rows[index]
    # OOD 파일의 context ID(Pushing 'test_007')로 같은 초기 배치의 OOD 사례를 고른다.
    # FM noise는 evaluation이 ID context ID로 만들므로 ID 결과와 같은 noise다.
    eval_id = row['context_id'] if ood is None else ood_context_id(task, row['context_id'])
    local = unit_dir(local_root(ood) / ('smoke' if smoke else 'full'), task, number, index)
    if local.exists():
        # 중단된 이전 시도는 지우지 않고 이름을 바꿔 보존한다.
        local.rename(local.with_name(local.name + '.interrupted.' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')))
    started = perf_counter()
    print(f'RUN  {protocol(ood)} {combo_id(task, number)} ctx_{index:02d} ({row["context_id"]}) seeds {list(SEEDS)}',
          flush=True)
    result = evaluate(cfg, sampling_seeds=list(SEEDS), context_ids=[eval_id], output_directory=local)
    rollouts = json.loads((local / 'rollouts.json').read_text())
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'trajectories.npz',
                        **slim_trajectories(rollouts, cfg.values['solver']['feasibility_tolerance_m']))
    summary = {k: result[k] for k in ('success_rate', 'plan_ms', 'plan_attempts', 'model_calls', 'qp_calls',
                                      'pf_calls', 'executed_actions', 'execution_intervention_rate',
                                      'filter_abort_rate', 'mean_steps')}
    record = {'protocol': protocol(ood), 'ood': result.get('ood'), 'task': task, 'combo': combo_id(task, number), 'number': number,
              'methods': combo_flags(number), 'context_index': index, 'context_id': row['context_id'],
              'context_sha256': row['context_sha256'], 'noise_seeds': list(SEEDS), 'member': name,
              'device': info, 'code_fingerprint': fingerprint, 'git_commit': commit,
              'max_steps': cfg.values['eval']['max_steps'],
              'finished_utc': datetime.now(timezone.utc).isoformat(),
              'wall_s': perf_counter() - started, 'summary': summary, 'episodes': result['episodes']}
    # result.json은 마지막에 쓴다. 이 파일이 있으면 완료된 단위다.
    result_path.write_text(json.dumps(record, indent=1, allow_nan=False) + '\n')
    print(f'DONE {combo_id(task, number)} ctx_{index:02d} success {result["success_rate"]:.2f} '
          f'{record["wall_s"]:.0f}s', flush=True)
    return rows


@contextmanager
def unit_lock(task, number, index, smoke, ood=None):
    """같은 장비에서 동시에 실행한 프로세스끼리 같은 단위를 두 번 돌리지 않게 한다(비차단 flock).

    잠금 파일은 로컬(fm_constraint/runs)에 있으므로 서로 다른 장비 사이에는 적용되지 않는다.
    장비 사이의 중복은 README의 context 분담으로 막는다.
    """
    path = local_root(ood) / 'locks' / ('smoke' if smoke else 'full') / f'{combo_id(task, number)}_ctx_{index:02d}.lock'
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


# ---- 집계 ---------------------------------------------------------------------------
def _mean(values, axis):
    finite = np.isfinite(values)
    total = np.where(finite, values, 0).sum(axis=axis)
    count = finite.sum(axis=axis)
    return np.divide(total, count, out=np.full_like(total, np.nan, dtype=float), where=count > 0)


def context_row(episodes, task):
    """context 하나의 [성공, 위반 step, 평가 step, action change, jerk, entropy, 방법 실패]."""
    modes = range(4) if task == 'pushing' else range(1, 7)
    successes = [e for e in episodes if e['success']]
    counts = np.array([sum(e['mode'] == m for e in successes) for m in modes], dtype=float)
    entropy = np.nan
    if counts.sum():
        p = counts[counts > 0] / counts.sum()
        entropy = float(-(p * np.log(p)).sum() / np.log(len(modes)))
    nan = lambda v: np.nan if v is None else v  # noqa: E731
    return [np.mean([e['success'] for e in episodes]),
            sum(e['constraint_violation_steps'] for e in episodes),
            sum(e['constraint_evaluated_steps'] for e in episodes),
            _mean(np.array([nan(e['action_change_m']) for e in episodes], dtype=float), 0),
            _mean(np.array([nan(e['command_jerk_m_s3']) for e in episodes], dtype=float), 0),
            entropy, np.mean([e['failure'] not in (None, 'max_steps') for e in episodes])]


def aggregate(frame):
    """(…, context, 7) → (…, 6): 성공 %, 위반 %(step 분모 유지), action change, jerk, entropy, 실패 %."""
    numerator, denominator = frame[..., 1].sum(-1), frame[..., 2].sum(-1)
    violation = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
    return np.stack([frame[..., 0].mean(-1) * 100, violation * 100, _mean(frame[..., 3], -1),
                     _mean(frame[..., 4], -1), _mean(frame[..., 5], -1), frame[..., 6].mean(-1) * 100], axis=-1)


def load_units(root):
    units = []
    for path in sorted(Path(root).glob('*/*/ctx_*/result.json')):
        units.append(json.loads(path.read_text()))
    return units


def fmt(value, unit=1., digits=3):
    return '—' if value is None or not np.isfinite(value) else f'{value * unit:.{digits}g}'


def _finite_or_none(values):
    return [float(v) if np.isfinite(v) else None for v in np.asarray(values, dtype=float)]


def report_data(units, name, ood):
    """보고서에 넣는 merge용 원시 데이터: 단위마다 context_row 7개 값."""
    out = {'format': REPORT_FORMAT, 'protocol': protocol(ood), 'ood': ood, 'member': name,
           'noise_seeds': list(SEEDS), 'units': []}
    for u in sorted(units, key=lambda u: (u['task'], u['number'], u['context_index'])):
        out['units'].append({
            'task': u['task'], 'number': u['number'], 'context_index': u['context_index'],
            'context_id': u.get('context_id'), 'code_fingerprint': u['code_fingerprint'],
            'git_commit': u.get('git_commit'), 'gpu': u['device'].get('gpu', 'cpu'), 'wall_s': u['wall_s'],
            'ood_sha256': (u.get('ood') or {}).get('sha256'),
            'seeds': sorted(e['sampling_seed'] for e in u['episodes']),
            'row': _finite_or_none(context_row(u['episodes'], u['task']))})
    return out


def parse_report(path):
    """보고서의 원시 데이터 블록. 없으면(통합본 등) None."""
    text = Path(path).read_text()
    if REPORT_MARKER not in text:
        return None
    block = text.split(REPORT_MARKER, 1)[1]
    start = block.index('```json\n') + len('```json\n')
    data = json.loads(block[start:block.index('\n```', start)])
    if data.get('format') != REPORT_FORMAT:
        raise SystemExit(f'{path}: unknown report format {data.get("format")!r}')
    return data


def report(name, out=None, ood=None):
    """한 팀원이 실행한 단위만 모은 보고서. 점추정 표 + merge용 context별 원시 데이터.

    결과(result.json)는 로컬에만 있으므로, git에는 이 보고서만 올린다.
    """
    check_name(name)
    root = result_root(ood)
    units = [u for u in load_units(root) if u['member'] == name]
    if not units:
        raise SystemExit(f'No results by {name} under {root}')
    fingerprints = {u['code_fingerprint'] for u in units}
    ood_files = {(u.get('ood') or {}).get('sha256') for u in units}
    title = 'ID' if ood is None else f'OOD {ood}'
    lines = [f'# 실험 보고 ({title}): {name}', '',
             f'- protocol `{protocol(ood)}`, noise seed {list(SEEDS)}, 생성 {datetime.now(timezone.utc).isoformat()}',
             f'- code fingerprint: {", ".join(sorted(f[:12] for f in fingerprints))}'
             + (' ⚠ 여러 개: 코드가 다른 결과가 섞였다' if len(fingerprints) > 1 else ''),
             f'- git commit: {", ".join(sorted({(u.get("git_commit") or "?")[:8] for u in units}))}',
             f'- 장비: {", ".join(sorted({u["device"].get("gpu", "cpu") for u in units}))}']
    if ood is not None:
        factors = (units[0].get('ood') or {}).get('factors')
        lines += [f'- OOD 조건 `{ood}`: {json.dumps({k: v for k, v in (factors or {}).items() if v}, ensure_ascii=False)}',
                  f'- OOD 파일 sha256: {", ".join(sorted(str(f)[:12] for f in ood_files))}'
                  + (' ⚠ 여러 개' if len(ood_files) > 1 else '')]
    lines += ['', '값은 이 팀원이 실행한 context만의 점추정이다(신뢰구간 없음). 전체 CI는 merge 결과를 본다.', '']
    for task in TASKS:
        rows = [u for u in units if u['task'] == task]
        if not rows:
            continue
        lines += [f'## {task}', '', '| 조합 | context 번호 | rollout | 성공 % | 위반 % | action change mm | jerk m/s³ | entropy | 방법 실패 % | 평균 시간/context s |',
                  '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for number in sorted({u['number'] for u in rows}):
            group = sorted((u for u in rows if u['number'] == number), key=lambda u: u['context_index'])
            frame = np.array([context_row(u['episodes'], task) for u in group], dtype=float)
            values = aggregate(frame)
            indices = ','.join(str(u['context_index']) for u in group)
            lines.append(f'| {combo_id(task, number)} | {indices} | {sum(len(u["episodes"]) for u in group)} | '
                         f'{fmt(values[0])} | {fmt(values[1])} | {fmt(values[2], 1e3)} | {fmt(values[3])} | '
                         f'{fmt(values[4])} | {fmt(values[5])} | {np.mean([u["wall_s"] for u in group]):.0f} |')
        lines.append('')
    data = json.dumps(report_data(units, name, ood), indent=None, separators=(',', ':'), allow_nan=False)
    lines += ['<details><summary>merge용 원시 데이터 (context별 지표, 자동 생성)</summary>', '',
              REPORT_MARKER, '', '```json', data.replace('},{', '},\n{'), '```', '', '</details>', '']
    path = Path(out) if out else report_path(name, ood)
    path.write_text('\n'.join(lines))
    print(f'Saved {path}')


def collect_reports(ood=None, report_dir=None):
    """report_dir의 보고서 중 같은 protocol의 단위를 모은다. 같은 단위가 두 번 있으면 멈춘다."""
    units, sources, owner = [], [], {}
    for path in sorted(Path(report_dir or REPORT_DIR).glob('report_*.md')):
        data = parse_report(path)
        if data is None or data['protocol'] != protocol(ood):
            continue
        if data['noise_seeds'] != list(SEEDS):
            raise SystemExit(f'{path.name}: noise seeds {data["noise_seeds"]} != {list(SEEDS)}')
        sources.append((path.name, data['member'], len(data['units'])))
        for u in data['units']:
            key = (u['task'], u['number'], u['context_index'])
            if key in owner:
                raise SystemExit(f'{combo_id(u["task"], u["number"])} ctx_{u["context_index"]:02d} appears in both '
                                 f'{owner[key]} and {path.name}')
            owner[key] = path.name
            units.append({**u, 'row': [np.nan if v is None else v for v in u['row']]})
    return units, sources


def merge(out=None, n_boot=2000, seed=2026, ood=None, report_dir=None):
    """모든 팀원의 보고서(docs/report_*.md)를 합쳐 조합별 추정값 [95% CI]와 Complementarity Δ를 계산한다.

    결과 파일(result.json) 없이 보고서만 있으면 된다. 누구나 실행할 수 있다.
    """
    units, sources = collect_reports(ood, report_dir)
    if not units:
        raise SystemExit(f'No reports for {protocol(ood)} in {report_dir or REPORT_DIR}')
    fingerprints = {u['code_fingerprint'] for u in units}
    if len(fingerprints) > 1:
        raise SystemExit(f'Results from {len(fingerprints)} code versions; rerun mismatched units first')
    ood_files = {u.get('ood_sha256') for u in units}
    if len(ood_files) > 1:
        raise SystemExit(f'Results from {len(ood_files)} different OOD files')
    title = 'ID' if ood is None else f'OOD {ood}'
    lines = [f'# 통합 결과 ({title}, merge)', '', f'protocol `{protocol(ood)}`, context bootstrap {n_boot}회(seed {seed}), '
             f'code fingerprint {next(iter(fingerprints))[:12]}'
             + (f', OOD 파일 sha256 {str(next(iter(ood_files)))[:12]}' if ood is not None else ''), '',
             '입력 보고서: ' + ', '.join(f'`{f}`({m}, 단위 {n}개)' for f, m, n in sources), '']
    summary = {}
    for task in TASKS:
        expected = list(range(N_CONTEXTS))
        table, missing = {}, []
        for number in range(32):
            group = {u['context_index']: u for u in units if u['task'] == task and u['number'] == number}
            absent = [i for i in expected if i not in group]
            if absent:
                missing.append(f'{combo_id(task, number)}: {len(absent)}개')
                continue
            for u in group.values():
                if u['seeds'] != list(SEEDS):
                    raise SystemExit(f'{combo_id(task, number)} ctx_{u["context_index"]}: seeds incomplete')
            table[number] = np.array([group[i]['row'] for i in expected], dtype=float)
        lines += [f'## {task}', '']
        if missing:
            lines += ['미완료(누락 context 수): ' + ', '.join(missing), '']
        if not table:
            continue
        index = np.random.default_rng(seed).integers(N_CONTEXTS, size=(n_boot, N_CONTEXTS))
        values = {n: aggregate(f) for n, f in table.items()}
        draws = {n: aggregate(f[index]) for n, f in table.items()}

        def estimate(v, s):
            s = s[np.isfinite(s)]
            return (v, *np.quantile(s, [.025, .975])) if np.isfinite(v) and len(s) else (np.nan, np.nan, np.nan)

        lines += ['| 조합 | ①②③④⑤ | 성공 % | 위반 % | action change mm / jerk m/s³ | entropy | 방법 실패 % | Δ 성공 %p (A, B) |',
                  '|---|---|---:|---:|---:|---:|---:|---|']
        summary[task] = {}
        for number in sorted(table):
            cells = []
            row = {}
            for i, metric in enumerate(METRICS):
                v, lo, hi = estimate(values[number][i], draws[number][:, i])
                row[metric] = [v, lo, hi]
                scale = 1e3 if metric == 'action_change_m' else 1.
                cells.append('—' if not np.isfinite(v) else f'{v * scale:.3g} [{lo * scale:.3g}, {hi * scale:.3g}]')
            low = number & -number
            delta = 'N/A'
            if number != low and number - low in table and low in table:
                a, b = number - low, low
                v = values[number][0] - max(values[a][0], values[b][0])
                s = draws[number][:, 0] - np.maximum(draws[a][:, 0], draws[b][:, 0])
                lo, hi = np.quantile(s, [.025, .975])
                delta = f'{v:+.3g} [{lo:+.3g}, {hi:+.3g}] ({combo_id(task, a)}, {combo_id(task, b)})'
                row['complementarity_success_pp'] = [v, lo, hi]
            flags = ''.join('●' if f else '·' for f in combo_flags(number).values())
            lines.append(f'| {combo_id(task, number)} | {flags} | {cells[0]} | {cells[1]} | {cells[2]} / {cells[3]} | '
                         f'{cells[4]} | {cells[5]} | {delta} |')
            summary[task][combo_id(task, number)] = row
        lines.append('')
    path = Path(out) if out else report_path('results', ood)
    path.write_text('\n'.join(lines))
    meta = {'protocol': protocol(ood), 'ood': ood, 'n_boot': n_boot, 'bootstrap_seed': seed,
            'code_fingerprint': next(iter(fingerprints)), 'ood_sha256': next(iter(ood_files)),
            'reports': [f for f, _, _ in sources]}
    path.with_suffix('.json').write_text(json.dumps({'meta': meta, **summary}, indent=1, default=float))
    print(f'Saved {path} and {path.with_suffix(".json").name}')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    s = sub.add_parser('select')
    s.add_argument('--d3il-root', default=str((REPO / '../d3il').resolve()))
    r = sub.add_parser('run')
    r.add_argument('--task', choices=TASKS, required=True)
    r.add_argument('--combos', default='all', help="'all', '0-31', '5,17' 또는 'P05,P17'")
    r.add_argument('--contexts', required=True, help="context 번호 0..29: '0-6', '7,9'")
    r.add_argument('--device', required=True, help="cuda:0 등. cpu는 명시해야 한다")
    r.add_argument('--name', required=True, help='팀원 이름(영문). report_<name>.md에 쓰인다')
    r.add_argument('--max-steps', type=int, default=None, help='동작 확인용. 결과는 로컬 smoke 폴더에만 저장')
    p = sub.add_parser('report')
    p.add_argument('--name', required=True)
    p.add_argument('--out')
    m = sub.add_parser('merge')
    m.add_argument('--out')
    for command in (r, p, m):
        command.add_argument('--ood', default=None, help='OOD 조건(예: A_mid). 생략하면 ID 실험')
    args = parser.parse_args()
    if args.command == 'select':
        select(args.d3il_root)
    elif args.command == 'run':
        run(args.task, parse_combos(args.combos), parse_indices(args.contexts, N_CONTEXTS),
            args.device, args.name, args.max_steps, args.ood)
    elif args.command == 'report':
        report(args.name, args.out, args.ood)
    else:
        merge(args.out, ood=args.ood)


if __name__ == '__main__':
    main()
