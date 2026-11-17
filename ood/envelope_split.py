"""OOD A 결과를 '시작 손이 블록 중심 다각형 제약 안/밖'으로 나눠 집계한다 (보고서만 읽음, 시뮬레이터 불필요).

    python -m ood.envelope_split --cond A_strong          # docs/report_ood_A_strong_envelope_split.{md,json}
    python -m ood.envelope_split --cond A_10 --task pushing

평가 제약 object_envelope(configs/<task>.yaml)는 매 계획마다 **현재 블록 위치**의 평균을 중심으로, 블록 범위 +
margin(Pushing 0.25 m, Inserting 0.14 m)인 정다각형으로 다시 만든다. A로 블록을 멀리 옮기면 다각형도 따라가
고정 시작 손(manifest demo_hand_start)이 밖에 남을 수 있다. 이 제약은 계획의 첫 step에도 걸리고, 첫 행동은 가속도
제약 때문에 시작점에서 β = a_max·dt²·cos(π/m)(Pushing 1.67 cm, Inserting 1.73 cm)까지만 움직일 수 있다. 따라서
**시작 손이 다각형 밖으로 β보다 멀면** ①투영·③필터는 첫 계획부터 실행 불가능(infeasible)해 rollout이 방법 실패로
끝난다(A_strong Pushing 10 context 전부 확인. β 안쪽으로 1.6 cm 밖이던 A_mid context는 정상 실행). 이는 방법 성능이
아니라 제약 정의가 그 배치에 적용되지 않는 경우라, 주 비교는 '적용 가능' context로 하고 '적용 불가' context는
따로 보고한다(2026-11-17 결정).

판정은 fm_constraint.constraints와 같은 기하를 OOD 행의 블록 위치(shift[].to)로 계산한다. 결과·코드는 바꾸지 않는다.
경계(β와 2 mm 이내)인 context는 보고서에 표시한다. 실제 시작 손은 시뮬레이터 측정값이라 판정이 바뀔 수 있다.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from fm_constraint.evaluation import ood_context_id
from fm_constraint.experiment import (N_CONTEXTS, TASKS, aggregate, collect_reports, combo_flags, combo_id,
                                      fmt)

REPO = Path(__file__).resolve().parents[1]
OOD_ROOT = REPO / 'data/ood'
N_BOOT, SEED = 2000, 2026


def step0_reach(task):
    """첫 행동이 시작점에서 움직일 수 있는 최대 거리 β (m). constraints.PlanConstraints의 가속도 행과 같은 식."""
    cfg = yaml.safe_load((REPO / f'configs/{task}.yaml').read_text())
    acc = cfg['constraints']['acceleration']
    return float(acc['max_m_s2'] * cfg['control']['dt_s'] ** 2 * np.cos(np.pi / acc['polygon_sides']))


def outside_contexts(task, cond):
    """실험 context 번호(0..29) 중 시작 손이 object_envelope 밖으로 β보다 먼 것(제약 적용 불가)과 context별 거리."""
    obj = yaml.safe_load((REPO / f'configs/{task}.yaml').read_text())['constraints']['object_envelope']
    beta = step0_reach(task)
    manifest = json.loads((OOD_ROOT / 'manifest.json').read_text())
    hand = np.asarray(manifest['tasks'][task]['demo_hand_start'], dtype=float)
    angles = obj['normal_offset_rad'] + 2 * np.pi * np.arange(obj['n_sides']) / obj['n_sides']
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    rows = {r['context_id']: r for r in map(json.loads, (OOD_ROOT / manifest['tasks'][task]['file']).open())
            if r['cond'] == cond}
    selected = json.loads((REPO / 'configs/experiment_contexts.json').read_text())['tasks'][task]['contexts']
    out, faces, excess = [], {}, {}
    for c in selected:
        row = rows[ood_context_id(task, c['context_id'])]
        blocks = np.array([e['to'] for e in row['shift']], dtype=float)
        center = blocks.mean(axis=0)
        radius = float(np.max((blocks - center) @ normals.T) + obj['margin_m'])
        e = float(np.max((hand - center) @ normals.T) - radius)     # > 0: 다각형 밖으로 이만큼(m)
        excess[c['index']] = e
        if e > beta:
            out.append(c['index'])
        faces[c['index']] = [x['face'] for x in row['shift']]
    geom = {'margin_m': obj['margin_m'], 'n_sides': obj['n_sides'], 'hand_start': hand.tolist(), 'step0_reach_m': beta}
    return out, faces, excess, geom


def boot(frame, index):
    """context 묶음 bootstrap: (n_boot, 6)."""
    return aggregate(frame[index])


def ci(value, draws):
    draws = draws[np.isfinite(draws)]
    if not np.isfinite(value) or not len(draws):
        return np.nan, np.nan, np.nan
    lo, hi = np.quantile(draws, [.025, .975])
    return value, lo, hi


def cell(est, digits=3, sign=False):
    v, lo, hi = (0.0 if abs(x) < 1e-9 else x for x in est)   # 부동소수 잔차(-3.6e-15 등)를 0으로
    if not np.isfinite(v):
        return '—'
    f = (lambda x: f'{x:+.{digits}g}') if sign else (lambda x: f'{x:.{digits}g}')
    return f'{f(v)} [{f(lo)}, {f(hi)}]'


def split_task(task, cond, units):
    out, faces, excess, geom = outside_contexts(task, cond)
    beta = geom['step0_reach_m']
    near = {i: e for i, e in excess.items() if e > 0}
    border = sorted(i for i, e in excess.items() if abs(e - beta) < 0.002)
    inside = [i for i in range(N_CONTEXTS) if i not in out]
    table = {}
    for number in range(32):
        group = {u['context_index']: u for u in units if u['task'] == task and u['number'] == number}
        if len(group) == N_CONTEXTS:
            table[number] = np.array([group[i]['row'] for i in range(N_CONTEXTS)], dtype=float)
    missing = 32 - len(table)
    rng = np.random.default_rng(SEED)
    idx_in = np.asarray(inside)[rng.integers(len(inside), size=(N_BOOT, len(inside)))] if inside else None
    idx_out = np.asarray(out)[rng.integers(len(out), size=(N_BOOT, len(out)))] if out else None
    base = table.get(0)
    lines = [f'## {task}', '',
             f'- **제약 적용 불가**(시작 손이 다각형 밖으로 β = {beta * 100:.2f} cm보다 멂): {len(out)}개 context {out}',
             f'- **적용 가능**: {len(inside)}개 context',
             '- 다각형 밖까지의 거리(cm, 밖인 context만): '
             + (', '.join(f'{i}: {e * 100:.1f}' for i, e in sorted(near.items())) or '없음'),
             f'- 다각형: 정{geom["n_sides"]}각형, margin {geom["margin_m"]} m, 시작 손 {np.round(geom["hand_start"], 4).tolist()}']
    if border:
        lines.append(f'- **경계**(β와 2 mm 이내, 판정이 바뀔 수 있음): {border}')
    if out:
        pattern = {}
        for i in out:
            key = '+'.join(str(f) for f in faces[i])
            pattern[key] = pattern.get(key, 0) + 1
        lines.append(f'- 적용 불가 context의 블록 이동 방향: {pattern} (y+ = 목표 쪽)')
    if missing:
        lines.append(f'- 30 context가 모두 모이지 않은 조합 {missing}개는 표에서 뺐다.')
    lines += ['',
              '성공·위반·jerk·방법 실패는 각 그룹 안에서 집계했다(위반은 step 합 분모). [ ]는 그룹 안 context bootstrap '
              f'{N_BOOT}회 95% 구간, Δ는 같은 context에서 baseline({combo_id(task, 0)})과의 짝 차이다.', '',
              f'| 조합 | ①②③④⑤ | 성공 % 전체 | 성공 % 적용 가능 | Δ 성공 %p 적용 가능 (vs {combo_id(task, 0)}) | '
              '성공 % 적용 불가 | 방법 실패 % 가능 / 불가 | 위반 % 가능 | jerk m/s³ 가능 |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    summary = {}
    for number, frame in sorted(table.items()):
        flags = ''.join('●' if f else '·' for f in combo_flags(number).values())
        allv = aggregate(frame)
        vin = aggregate(frame[inside]) if inside else np.full(6, np.nan)
        vout = aggregate(frame[out]) if out else np.full(6, np.nan)
        s_in = ci(vin[0], boot(frame, idx_in)[:, 0]) if inside else (np.nan,) * 3
        s_out = ci(vout[0], boot(frame, idx_out)[:, 0]) if out else (np.nan,) * 3
        if base is not None and inside:
            d = vin[0] - aggregate(base[inside])[0]
            dd = boot(frame, idx_in)[:, 0] - boot(base, idx_in)[:, 0]
            delta = ci(d, dd)
        else:
            delta = (np.nan,) * 3
        lines.append(f'| {combo_id(task, number)} | {flags} | {fmt(allv[0])} | {cell(s_in)} | '
                     f'{"—" if number == 0 else cell(delta, sign=True)} | {cell(s_out)} | '
                     f'{fmt(vin[5])} / {fmt(vout[5])} | {fmt(vin[1])} | {fmt(vin[3])} |')
        summary[combo_id(task, number)] = {
            'success_all_pct': float(allv[0]), 'success_inside_pct': list(map(float, s_in)),
            'delta_success_inside_vs_00_pp': list(map(float, delta)), 'success_outside_pct': list(map(float, s_out)),
            'abort_inside_pct': float(vin[5]), 'abort_outside_pct': float(vout[5]),
            'violation_inside_pct': float(vin[1]), 'jerk_inside_m_s3': float(vin[3])}
    lines.append('')
    return lines, {'outside': out, 'inside': inside, 'geometry': geom, 'borderline': border,
                   'excess_m': {str(i): e for i, e in excess.items()},
                   'outside_faces': {str(i): faces[i] for i in out}, 'combos': summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--cond', required=True, help='OOD A 조건, 예: A_strong, A_10, A_mid')
    parser.add_argument('--task', nargs='+', default=list(TASKS), choices=TASKS)
    parser.add_argument('--out', default=None, help='출력 경로(확장자 제외). 기본 docs/report_ood_<cond>_envelope_split')
    args = parser.parse_args()
    units, sources = collect_reports(args.cond)
    if not units:
        raise SystemExit(f'No reports for {args.cond}')
    units = [{**u, 'row': [np.nan if v is None else v for v in u['row']]} for u in units]
    fingerprints = sorted({u['code_fingerprint'][:12] for u in units})
    ood_files = sorted({str(u.get('ood_sha256'))[:12] for u in units})
    lines = [f'# OOD {args.cond}: 블록 중심 다각형 제약 적용 가능/불가 context로 나눈 결과', '',
             '①투영·③필터가 쓰는 `object_envelope`는 매 계획마다 현재 블록 위치를 중심으로 다시 만든다. 블록을 멀리 옮기면 '
             '고정 시작 손이 다각형 밖에 남는다. 첫 행동은 가속도 제약 때문에 β(Pushing 1.67 cm, Inserting 1.73 cm)까지만 '
             '움직일 수 있어, 손이 다각형 밖으로 β보다 멀면 ①·③을 켠 조합은 첫 계획부터 실행 불가능해 방법 실패로 끝난다. '
             '이는 방법 성능이 아니라 제약 정의가 그 배치에 적용되지 않는 경우라, **주 비교는 적용 가능 context로 하고 '
             '적용 불가 context는 따로 보고한다**(2026-11-17 결정). 코드·결과는 바꾸지 않고 팀원 보고서만 다시 집계했다.', '',
             f'입력 보고서: ' + ', '.join(f'`{f}`({m}, 단위 {n}개)' for f, m, n in sources),
             f'code fingerprint {", ".join(fingerprints)}, OOD 파일 sha256 {", ".join(ood_files)}. '
             f'`python -m ood.envelope_split --cond {args.cond}`로 다시 만든다.', '']
    data = {'cond': args.cond, 'reports': [f for f, _, _ in sources], 'code_fingerprints': fingerprints,
            'ood_sha256': ood_files, 'bootstrap': {'n': N_BOOT, 'seed': SEED, 'unit': 'context within group'},
            'tasks': {}}
    for task in args.task:
        task_lines, task_data = split_task(task, args.cond, units)
        lines += task_lines
        data['tasks'][task] = task_data
    out = Path(args.out) if args.out else REPO / f'docs/report_ood_{args.cond}_envelope_split'
    out.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    out.with_suffix('.json').write_text(json.dumps(data, indent=1, ensure_ascii=False) + '\n')
    print('saved', out.with_suffix('.md'), out.with_suffix('.json'))


if __name__ == '__main__':
    main()
