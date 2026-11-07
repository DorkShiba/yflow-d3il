"""fm_constraint 평가 결과의 OOD 지표: seam·chunk 내부 jerk, push 직후 jerk, 성공률.

    python -m ood.metrics --eval-dir fm_constraint/runs/<id> --label ID \\
                          --eval-dir fm_constraint/runs/<D_mid> --label D_mid

첫 번째 실행이 기준이다. 모든 실행은 같은 태스크·context 목록이어야 하며
(같은 OOD 파일의 서로 다른 조건, 또는 같은 조건의 서로 다른 방법) Δ는 같은
context끼리의 대응 차이다. 불확실성은 context 단위 95% bootstrap이다.

rollout마다 (a_t: 실행 행동 = 목표 손 XY 증분, S: 새 chunk가 시작한 step, a_max: 학습 행동 scale 최대):

  J_seam   mean_{s∈S, s≥2} ||a_s − 2a_{s−1} + a_{s−2}|| / a_max      chunk 경계의 jerk
  jump     mean_{s∈S, s≥1} ||a_s − a_{s−1}|| / a_max                  chunk 경계의 행동 점프
  J_in     mean_{t: t, t−1 ∉ S} ||a_t − 2a_{t−1} + a_{t−2}|| / a_max  chunk 내부 jerk
  JerkRMS  sqrt(mean ||Δ³ des / dt³||²) (m/s³)                         전체 episode
  J_event  push(D) 이후 첫 seam의 jerk
"""
import argparse
import json
from pathlib import Path

import numpy as np

DT = 0.035
COLUMNS = ('success', 'J_seam', 'J_in', 'jump', 'JerkRMS', 'J_event')


def rollout_metrics(records, a_max, push_step=None, dt=DT):
    rows = [r for r in records if r.get('safe_action') is not None]
    a = np.asarray([r['safe_action'] for r in rows], dtype=float).reshape(-1, 2)
    steps = len(a)
    seams = [k for k, r in enumerate(rows) if r.get('seam')]
    seam_set = set(seams)
    j3 = np.full(steps, np.nan)
    if steps > 2:
        j3[2:] = np.linalg.norm(a[2:] - 2 * a[1:-1] + a[:-2], axis=1) / a_max
    out = {'J_seam': np.nan, 'jump': np.nan, 'J_in': np.nan, 'JerkRMS': np.nan, 'J_event': np.nan}
    values = [j3[s] for s in seams if s >= 2]
    out['J_seam'] = float(np.mean(values)) if values else np.nan
    values = [np.linalg.norm(a[s] - a[s - 1]) / a_max for s in seams if s >= 1]
    out['jump'] = float(np.mean(values)) if values else np.nan
    values = [j3[t] for t in range(2, steps) if t not in seam_set and t - 1 not in seam_set]
    out['J_in'] = float(np.mean(values)) if values else np.nan
    if steps > 3:
        jerk = np.diff(np.cumsum(a, axis=0), n=3, axis=0) / dt ** 3
        out['JerkRMS'] = float(np.sqrt((np.linalg.norm(jerk, axis=1) ** 2).mean()))
    if push_step is not None:
        after = [s for s in seams if s >= push_step and s >= 2]
        if after:
            out['J_event'] = float(j3[after[0]])
    return out


def load_run(directory):
    directory = Path(directory)
    results = json.loads((directory / 'results.json').read_text())
    rollouts = json.loads((directory / 'rollouts.json').read_text())
    scale = np.load(Path(results['checkpoint']).parent / 'normalizer.npz')['scale']
    a_max = float(np.asarray(scale)[:2].max())
    contexts = sorted({e['context_index'] for e in results['episodes']})
    n_traj = max(e['trajectory'] for e in results['episodes']) + 1
    table = {k: np.full((len(contexts), n_traj), np.nan) for k in COLUMNS}
    for e in results['episodes']:
        ci, ti = e['context_index'], e['trajectory']
        push = (e.get('ood_events') or {}).get('push') or {}
        metrics = rollout_metrics(rollouts[f'c{ci:03d}_t{ti}'], a_max,
                                  push['step'] if push.get('to') is not None else None)
        metrics['success'] = float(e['success'])
        for k in COLUMNS:
            table[k][ci, ti] = metrics[k]
    return results, table


def context_mean(x):
    with np.errstate(all='ignore'):
        return np.nanmean(x, axis=1)


def bootstrap(values, index):
    with np.errstate(all='ignore'):
        return np.nanpercentile(np.nanmean(values[index], axis=1), [2.5, 97.5]).tolist()


def summarize(eval_dirs, labels, n_boot=5000, seed=0):
    runs = [load_run(d) for d in eval_dirs]
    if len({r['task'] for r, _ in runs}) != 1:
        raise ValueError('All runs must use the same task')
    n_ctx = min(t['success'].shape[0] for _, t in runs)
    index = np.random.default_rng(seed).integers(0, n_ctx, size=(n_boot, n_ctx))
    base = {k: context_mean(runs[0][1][k][:n_ctx]) for k in COLUMNS}
    rows = []
    for label, (results, table) in zip(labels, runs):
        ood = results.get('ood') or {}
        row = {'label': label, 'condition': ood.get('condition', 'none'), 'n_contexts': n_ctx}
        for k in COLUMNS:
            c = context_mean(table[k][:n_ctx])
            finite = np.isfinite(c).any()
            row[k] = float(np.nanmean(c)) if finite else None
            row[k + '_ci'] = bootstrap(c[:, None], index) if finite else None
            if label != labels[0] and finite and np.isfinite(base[k]).any():
                delta = c - base[k]
                row['d_' + k] = float(np.nanmean(delta))
                row['d_' + k + '_ci'] = bootstrap(delta[:, None], index)
        row['seam_over_in'] = row['J_seam'] / row['J_in'] if row['J_seam'] and row['J_in'] else None
        pushes = [(e.get('ood_events') or {}).get('push') for e in results['episodes']]
        if any(pushes):
            row['push_applied'] = float(np.mean([bool(p and p.get('to') is not None) for p in pushes]))
        rows.append(row)
    return runs[0][0]['task'], rows


def to_markdown(task, rows, reference):
    def cell(row, key, pct=False, digits=3):
        value = row.get(key)
        if value is None:
            return '–'
        text = f'{100 * value:.1f}' if pct else f'{value:.{digits}f}'
        if row.get('d_' + key) is not None:
            d, (lo, hi) = row['d_' + key], row['d_' + key + '_ci']
            text += (f' ({100 * d:+.1f} [{100 * lo:+.1f},{100 * hi:+.1f}])' if pct
                     else f' ({d:+.{digits}f} [{lo:+.{digits}f},{hi:+.{digits}f}])')
        return text
    lines = [f"# {task}: (Δ [95% CI]) = paired vs '{reference}', bootstrap over contexts", '',
             '| run | condition | success % | J_seam | J_in | seam/in | jump | JerkRMS (m/s³) | J_event |',
             '|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        ratio = '–' if r['seam_over_in'] is None else f"{r['seam_over_in']:.2f}"
        lines.append(f"| {r['label']} | {r['condition']} | {cell(r, 'success', True)} | {cell(r, 'J_seam')} | "
                     f"{cell(r, 'J_in')} | {ratio} | {cell(r, 'jump')} | {cell(r, 'JerkRMS', digits=1)} | "
                     f"{cell(r, 'J_event')} |")
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-dir', action='append', required=True)
    parser.add_argument('--label', action='append', default=None)
    parser.add_argument('--n-boot', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', default=None, help='directory for summary.md / summary.json')
    args = parser.parse_args()
    labels = args.label or [Path(d).name for d in args.eval_dir]
    if len(labels) != len(args.eval_dir):
        raise SystemExit('give one --label per --eval-dir')
    task, rows = summarize(args.eval_dir, labels, args.n_boot, args.seed)
    markdown = to_markdown(task, rows, labels[0])
    print(markdown)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / 'summary.md').write_text(markdown + '\n')
        (out / 'summary.json').write_text(json.dumps({'task': task, 'eval_dirs': args.eval_dir, 'rows': rows}, indent=1))


if __name__ == '__main__':
    main()
