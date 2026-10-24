"""통합/이전 평가 결과를 읽고 context 단위 신뢰구간과 비교 표를 저장한다."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np


def context_metrics(episodes, task):
    """같은 context 반복을 묶어 성공률·정규화 모드 엔트로피·score를 계산한다."""
    groups = {}
    for episode in episodes:
        groups.setdefault(str(episode['context_id']), []).append(episode)
    mode_ids = range(4) if task == 'pushing' else range(1, 7)
    result = {}
    for key, rows in groups.items():
        success = float(np.mean([r['success'] for r in rows]))
        counts = np.array([sum(r['success'] and r['mode'] == mode for r in rows) for mode in mode_ids], dtype=float)
        probability = counts / (counts.sum() + 1e-12)
        entropy = float(-np.sum(probability * np.log(probability + 1e-12)) / np.log(len(mode_ids)))
        result[key] = np.array([success, entropy, .5 * (success + entropy)])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval-dir', action='append', required=True)
    parser.add_argument('--label', action='append')
    parser.add_argument('--n-boot', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out-dir')
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()
    if args.n_boot < 2 or args.label and len(args.label) != len(args.eval_dir):
        parser.error('Require n-boot>=2 and one label per eval-dir')
    records = []
    for i, directory in enumerate(args.eval_dir):
        path = Path(directory).resolve()
        result = json.loads((path / 'results.json').read_text())
        context = context_metrics(result.get('episodes', []), result.get('task'))
        records.append((path, result, context))
    baseline = records[0]
    rng = np.random.default_rng(args.seed)
    rows = []
    for i, (path, result, contexts) in enumerate(records):
        row = {'label': args.label[i] if args.label else path.name, 'directory': str(path),
               **{key: result.get(key) for key in ('task', 'success_rate', 'entropy', 'score',
                                                  'plan_ms', 'model_calls', 'qp_calls', 'filter_abort_rate',
                                                  'executed_acceleration_qp_exceed_rate')}}
        metric_names = ('success', 'entropy', 'score')
        row['pairing_status'] = 'unavailable_without_episode_contexts'
        for metric in metric_names:
            row[f'{metric}_ci95'] = None
            row[f'paired_{metric}_difference'] = None
            row[f'paired_{metric}_difference_ci95'] = None
        if contexts:
            values = np.stack(list(contexts.values()))
            indices = rng.integers(len(values), size=(args.n_boot, len(values)))
            intervals = np.quantile(values[indices].mean(axis=1), [.025, .975], axis=0)
            for j, metric in enumerate(metric_names):
                row[f'{metric}_ci95'] = intervals[:, j].tolist()
            same_checkpoint = (result.get('checkpoint_sha256') is not None and
                               result.get('checkpoint_sha256') == baseline[1].get('checkpoint_sha256'))
            if set(contexts) == set(baseline[2]) and same_checkpoint and result.get('task') == baseline[1].get('task'):
                keys = sorted(contexts)
                differences = np.stack([contexts[key] - baseline[2][key] for key in keys])
                # 동일 context의 차이 벡터를 재표집하며 세 지표의 대응을 유지한다.
                paired = rng.integers(len(keys), size=(args.n_boot, len(keys)))
                intervals = np.quantile(differences[paired].mean(axis=1), [.025, .975], axis=0)
                for j, metric in enumerate(metric_names):
                    row[f'paired_{metric}_difference'] = float(differences[:, j].mean())
                    row[f'paired_{metric}_difference_ci95'] = intervals[:, j].tolist()
                row['pairing_status'] = 'matched_contexts_and_checkpoint'
            else:
                row['pairing_status'] = 'context_task_or_checkpoint_mismatch'
        rows.append(row)
    output = Path(args.out_dir) if args.out_dir else Path('fm_constraint/runs') / ('summary_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir(parents=True, exist_ok=True)
    (output / 'summary.json').write_text(json.dumps(rows, indent=2, allow_nan=False))
    with (output / 'summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['| label | success | entropy | score | plan ms | filter abort |', '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        values = [row[key] for key in ('label', 'success_rate', 'entropy', 'score', 'plan_ms', 'filter_abort_rate')]
        lines.append('| ' + ' | '.join('—' if x is None else str(x) if isinstance(x, str) else f'{x:.5f}' for x in values) + ' |')
    (output / 'summary.md').write_text('\n'.join(lines) + '\n')
    if not args.no_plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(max(6, len(rows) * 1.8), 4))
        positions = np.arange(len(rows))
        for i, metric in enumerate(('success_rate', 'entropy', 'score')):
            ax.bar(positions + (i - 1) * .25, [r[metric] if r[metric] is not None else np.nan for r in rows],
                   width=.25, label=metric)
        ax.set_xticks(positions, [r['label'] for r in rows], rotation=20)
        ax.set_ylim(0, 1.05)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / 'summary.png', dpi=140)
        plt.close(fig)
    print('\n'.join(lines))
    print(f'Saved: {output}')


if __name__ == '__main__':
    main()
