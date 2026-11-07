"""실제 실행 지표와 context 묶음 bootstrap으로 16개 실험 결과를 집계한다."""

import json
from pathlib import Path

import numpy as np

NAMES = ('success_rate_pct', 'constraint_violation_rate_pct', 'action_change_m',
         'command_jerk_m_s3', 'behavior_entropy', 'method_abort_rate_pct')
LOWER_BETTER = {'constraint_violation_rate_pct', 'action_change_m', 'command_jerk_m_s3', 'method_abort_rate_pct'}


def _mean(values, axis):
    finite = np.isfinite(values)
    total = np.where(finite, values, 0).sum(axis=axis)
    count = finite.sum(axis=axis)
    return np.divide(total, count, out=np.full_like(total, np.nan, dtype=float), where=count > 0)


def _frame(episodes, contexts, task):
    rows = []
    modes = range(4) if task == 'pushing' else range(1, 7)
    for context in contexts:
        group = [row for row in episodes if row['context_id'] == context]
        if not group:
            raise ValueError(f'Missing context: {context}')
        successes = [row for row in group if row['success']]
        counts = np.array([sum(row['mode'] == mode for row in successes) for mode in modes], dtype=float)
        entropy = np.nan
        if counts.sum():
            probabilities = counts[counts > 0] / counts.sum()
            entropy = float(-(probabilities * np.log(probabilities)).sum() / np.log(len(modes)))
        rows.append([np.mean([row['success'] for row in group]),
                     sum(row['constraint_violation_steps'] for row in group),
                     sum(row['constraint_evaluated_steps'] for row in group),
                     _mean(np.array([row['action_change_m'] if row['action_change_m'] is not None else np.nan for row in group]), 0),
                     _mean(np.array([row['command_jerk_m_s3'] if row['command_jerk_m_s3'] is not None else np.nan for row in group]), 0),
                     entropy, np.mean([row['failure'] not in (None, 'max_steps') for row in group])])
    return np.asarray(rows, dtype=float)


def _aggregate(frame):
    """마지막 두 축을 context×성분으로 집계하며 step 비율의 분모를 유지한다."""
    numerator, denominator = frame[..., 1].sum(-1), frame[..., 2].sum(-1)
    violation = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
    return np.stack([frame[..., 0].mean(-1) * 100, violation * 100,
                     _mean(frame[..., 3], -1), _mean(frame[..., 4], -1),
                     _mean(frame[..., 5], -1), frame[..., 6].mean(-1) * 100], axis=-1)


def _estimate(value, samples):
    valid = samples[np.isfinite(samples)]
    return {'value': float(value) if np.isfinite(value) else None,
            'ci95': np.quantile(valid, [.025, .975]).tolist() if len(valid) else None,
            'valid_bootstrap_samples': len(valid)}


def summarize(root):
    """16개 완료 결과를 대조한 뒤 태스크별 추정값과 대응 차이를 저장한다."""
    from .method123 import verify_inputs, COMPLEMENTARITY
    from .pf2mp import file_hash
    manifest = json.loads((root / 'experiment_manifest.json').read_text())
    verify_inputs(root, manifest)
    summary = {'protocol_version': 'method123_v1', 'manifest_sha256': manifest['manifest_sha256'],
               'bootstrap_unit': 'context_cluster_all_seeds', 'n_bootstrap': 2000,
               'bootstrap_seed': 2026, 'tasks': {}}
    for task, spec in manifest['tasks'].items():
        indices = np.random.default_rng(2026).integers(len(spec['context_ids']), size=(2000, len(spec['context_ids'])))
        values, draws, results = {}, {}, {}
        expected_keys = {(context, seed) for context in spec['context_ids'] for seed in manifest['sampling_seeds']}
        for job in spec['jobs']:
            folder = root / job['id']
            complete = json.loads((folder / 'COMPLETE.json').read_text())
            if (complete['manifest_sha256'] != manifest['manifest_sha256']
                    or complete['results_sha256'] != file_hash(folder / 'results.json')
                    or complete['rollouts_sha256'] != file_hash(folder / 'rollouts.json')):
                raise ValueError(f'Incomplete or changed results: {job["id"]}')
            result = json.loads((folder / 'results.json').read_text())
            keys = [(row['context_id'], row['sampling_seed']) for row in result['episodes']]
            if len(keys) != len(expected_keys) or set(keys) != expected_keys:
                raise ValueError(f'Incomplete paired keys: {job["id"]}')
            frame = _frame(result['episodes'], spec['context_ids'], task)
            number = job['number']
            values[number], draws[number] = _aggregate(frame), _aggregate(frame[indices])
            results[number] = {'job': job, 'metrics': {name: _estimate(values[number][i], draws[number][:, i]) for i, name in enumerate(NAMES)},
                               'entropy_contexts_with_success': int(np.isfinite(frame[:, 5]).sum()),
                               'contexts': len(frame), 'rollout_attempts': len(result['episodes']),
                               'plan_ms': result['plan_ms'], 'model_calls': result['model_calls'],
                               'pf_calls': result['pf_calls'], 'qp_calls': result['qp_calls'],
                               'safe_success_rate': result['safe_success_rate']}
        for number, row in results.items():
            row['difference_from_baseline'] = {
                name: _estimate(values[number][i] - values[0][i], draws[number][:, i] - draws[0][:, i])
                for i, name in enumerate(NAMES)}
            row['complementarity'] = None
            if number in COMPLEMENTARITY:
                a, b = COMPLEMENTARITY[number]
                row['complementarity'] = {'parts': [a, b], 'metrics': {}}
                for i, name in enumerate(NAMES):
                    direction = -1 if name in LOWER_BETTER else 1
                    value = direction * values[number][i] - max(direction * values[a][i], direction * values[b][i])
                    samples = direction * draws[number][:, i] - np.maximum(direction * draws[a][:, i], direction * draws[b][:, i])
                    row['complementarity']['metrics'][name] = _estimate(value, samples)
        summary['tasks'][task] = list(results.values())
    (root / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    lines = ['# Method 1/2/3 results', '', 'RTC=OFF, LiPo=OFF, terminal VLSA=OFF. Values are estimates [95% context bootstrap CI].', '']
    for task, rows in summary['tasks'].items():
        lines += [f'## {task}', '', '| ID | Projection | PF2MP | Executed filter | Success % | Violation % | Action change m | Jerk m/s³ | Entropy | Method abort % |',
                  '|---|---|---|---|---:|---:|---:|---:|---:|---:|']
        for row in rows:
            cells = [row['job']['id'], *['ON' if row['job'][key] else 'OFF' for key in ('projection', 'pf2mp', 'executed_action_filter')]]
            for name in NAMES:
                estimate = row['metrics'][name]
                if estimate['value'] is None:
                    cells.append('N/A')
                else:
                    low, high = estimate['ci95']
                    cells.append(f'{estimate["value"]:.5g} [{low:.5g}, {high:.5g}]')
            lines.append('| ' + ' | '.join(cells) + ' |')
        lines.append('')
    lines += ['Complementarity Δ, paired baseline differences, bootstrap validity counts and context coverage are saved in summary.json.', '']
    (root / 'summary.md').write_text('\n'.join(lines))
    print(f'Saved: {root / "summary.md"}')
