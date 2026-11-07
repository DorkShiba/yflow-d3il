"""data/ood의 해시·개수·조건·seed 규칙을 검사한다. 시뮬레이터는 쓰지 않는다.

    python -m ood.verify [--root data/ood]
"""
import argparse
import json
import sys
from pathlib import Path

from .apply import file_sha256
from .conditions import conditions, context_seeds, factor_values

REPO = Path(__file__).resolve().parents[1]


def verify(root):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    names = [name for name, _ in conditions()]
    errors = []
    if manifest['conditions'] != names:
        errors.append('condition list differs from ood.conditions')
    for task, entry in manifest['tasks'].items():
        path = root / entry['file']
        if file_sha256(path) != entry['sha256']:
            errors.append(f'{task}: sha256 mismatch')
        rows = [json.loads(line) for line in path.open()]
        if len(rows) != entry['n_rows'] or len(rows) != len(names) * entry['n_contexts']:
            errors.append(f'{task}: row count {len(rows)}')
        if len({row['id'] for row in rows}) != len(rows):
            errors.append(f'{task}: duplicate ids')
        selected = dict(conditions())
        for row in rows:
            seeds = context_seeds(task, row['context_id'])
            if row['factors'] != factor_values(task, selected[row['cond']]):
                errors.append(f'{row["id"]}: factors')
            if row['policy_seeds'] != seeds['policy']:
                errors.append(f'{row["id"]}: policy seeds')
            if row['noise'] and row['noise']['seed'] != seeds['C']:
                errors.append(f'{row["id"]}: noise seed')
            if row['push'] and row['push']['seed'] != seeds['D']:
                errors.append(f'{row["id"]}: push seed')
            if bool(row['factors']['B']) != (row['robot'] is not None):
                errors.append(f'{row["id"]}: robot start record')
        by_context = {}
        for row in rows:
            by_context.setdefault(row['context_id'], {})[row['cond']] = row
        for context_id, group in by_context.items():
            # 조합 조건은 단일 요인과 같은 배치·방향·노이즈를 재사용해야 한다(common random numbers).
            if group['AB_mid']['context'] != group['A_mid']['context']:
                errors.append(f'{context_id}: AB_mid blocks differ from A_mid')
            if group['BC_mid']['robot_start'] != group['B_mid']['robot_start']:
                errors.append(f'{context_id}: BC_mid start differs from B_mid')
            if group['C_mid']['context'] != group['id']['context']:
                errors.append(f'{context_id}: C_mid changed the context')
    return manifest, errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=str(REPO / 'data/ood'))
    manifest, errors = verify(parser.parse_args().root)
    for error in errors[:50]:
        print('FAIL', error)
    counts = {task: entry['n_rows'] for task, entry in manifest['tasks'].items()}
    print('OK' if not errors else f'{len(errors)} errors', counts)
    sys.exit(1 if errors else 0)


if __name__ == '__main__':
    main()
