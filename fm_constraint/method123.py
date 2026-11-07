"""16개 순차 실험의 고정 context·seed 목록, 실행, 보고 지표 집계를 관리한다."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
import subprocess

import numpy as np

from configs.config import ExperimentConfig, REPO
from .evaluation import evaluate
from .pf2mp import file_hash

COMBINATIONS = (0, 4, 8, 12, 16, 20, 24, 28)
# 결과를 보기 전에 고정하는 상보성 비교: triple은 Projection+PF와 Filter를 비교한다.
COMPLEMENTARITY = {12: (8, 4), 20: (16, 4), 24: (16, 8), 28: (24, 4)}


def _context_digest(value):
    """형태가 다른 context의 수치 배열을 정규화하여 같은 초기 조건을 찾는다."""
    if isinstance(value, np.ndarray) and value.dtype != object:
        return np.asarray(value, dtype=float).round(6).tolist()
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_context_digest(item) for item in value]
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _train_digest(root):
    """모든 train 파일 이름·내용을 묶어 조합 사이 데이터 변경을 감지한다."""
    return _digest([(str(path.relative_to(root)), file_hash(path))
                    for path in sorted((root / 'train').rglob('*.npz'))])


def prepare(root, stage, inserting_set, device, split_seed, max_steps=None, context_limit=None):
    """시뮬레이터 없이 공통 목록과 설정을 동결한다. 기존 계획은 덮어쓰지 않는다."""
    root = ExperimentConfig.from_yaml(REPO / 'configs/pushing.yaml').require_output(root)
    request = dict(stage=stage, inserting_set=inserting_set, device=device, split_seed=split_seed,
                   max_steps=max_steps, context_limit=context_limit)
    manifest_path = root / 'experiment_manifest.json'
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if saved['request'] != request:
            raise ValueError('Existing experiment has different settings; use a new output directory')
        verify_inputs(root, saved)
        print(f'Reusing frozen plan: {manifest_path}')
        return saved
    if stage != 'smoke' and (max_steps is not None or context_limit is not None):
        raise ValueError('max_steps/context_limit overrides are only allowed for smoke checks')
    root.mkdir(parents=True, exist_ok=True)
    seeds = list(range(5)) if stage == 'main' else [100, 101] if stage == 'smoke' else [100, 101, 102]
    manifest = {'protocol_version': 'method123_v1', 'request': request,
                'created_at': datetime.now(timezone.utc).isoformat(), 'sampling_seeds': seeds,
                'seeding': 'SHA256 JSON [task,context_id,sampling_seed], independent per rollout FM generator',
                'rtc': False, 'lipo': False, 'terminal_vlsa': False, 'tasks': {}, 'source_hashes': {},
                'runtime_versions': {name: version(name) for name in ['torch', 'numpy', 'osqp', 'mujoco', 'PyYAML']},
                'complementarity': {str(k): list(v) for k, v in COMPLEMENTARITY.items()}}
    for relative in ['fm_constraint/method123.py', 'fm_constraint/evaluation.py', 'fm_constraint/policy.py',
                     'fm_constraint/pf2mp.py', 'fm_constraint/constraints.py', 'fm_constraint/qp.py',
                     'configs/config.py', 'fm_constraint/method123_metrics.py', 'scripts/run_method123.sh']:
        manifest['source_hashes'][relative] = file_hash(REPO / relative)
    manifest['git_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    for task in ['pushing', 'inserting']:
        cfg = ExperimentConfig.from_yaml(REPO / f'configs/{task}.yaml', {'device': device})
        v, settings, paths = cfg.values, cfg.values['eval'], cfg.values['paths']
        settings.update(n_contexts=0, n_traj=len(seeds), execute_steps=8, seed=0,
                        check_obs=0, export_contexts=None)
        settings['max_steps'] = (max_steps or 10) if stage == 'smoke' else (400 if task == 'pushing' else 2000)
        v['sampling']['predict_steps'] = 10
        v['methods']['projection'].update(enabled=False, mode='every', skip_first=0)
        v['methods']['pf2mp']['enabled'] = False
        v['methods']['vlsa'].update(enabled=False, execution_recheck=False)
        cfg.validate('eval')
        dataset_root = Path(paths['data_root']) / f'{task}_sincos_v1'
        inputs = {str(Path(paths['run_dir']) / name): file_hash(Path(paths['run_dir']) / name)
                  for name in [settings['checkpoint'], 'normalizer.npz', 'config.json']}
        inputs[str(dataset_root / 'manifest.json')] = file_hash(dataset_root / 'manifest.json')
        if task == 'pushing':
            source_dir = Path(paths['d3il_root']) / 'environments/dataset/data/pushing'
            train_path, test_path = source_dir / 'train_contexts.pkl', source_dir / 'test_contexts.pkl'
            train, test = np.load(train_path, allow_pickle=True), np.load(test_path, allow_pickle=True)
            train_hash = {_digest(_context_digest(c)) for c in train}
            test_hash = {_digest(_context_digest(c)) for c in test}
            if train_hash & test_hash:
                raise ValueError('Pushing tuning/test contexts overlap')
            settings['contexts'] = 'test' if stage == 'main' else 'train'
            source = test if stage == 'main' else train
            limit = len(source) if stage == 'main' else (5 if stage == 'smoke' else 10)
            limit = context_limit or limit
            ids = [str(i) for i in range(min(limit, len(source)))]
            tuning = [{'id': str(i), 'initial_condition': _context_digest(c)} for i, c in enumerate(train)]
            final = [{'id': str(i), 'initial_condition': _context_digest(c)} for i, c in enumerate(test)]
            selected = [{'id': key, 'initial_condition': _context_digest(source[int(key)])} for key in ids]
            for path in [train_path, test_path]:
                inputs[str(path)] = file_hash(path)
        else:
            settings['split'] = 'val'
            data_manifest = Path(paths['data_root']) / 'inserting_sincos_v1/manifest.json'
            data = json.loads(data_manifest.read_text())
            groups = {}
            for row in data['splits']['val']:
                groups.setdefault(row['context_group'], []).append(Path(row['file']).name)
            ordered = sorted(groups)
            order = np.random.default_rng(split_seed).permutation(len(ordered))
            if len(ordered) <= 20:
                raise ValueError('Not enough Inserting groups for 20 tuning groups and a holdout')
            tuning_groups = {ordered[i] for i in order[:20]}
            representatives = [{'id': min(groups[group]), 'context_group': group} for group in ordered]
            tuning = [row for row in representatives if row['context_group'] in tuning_groups]
            final = [row for row in representatives if row['context_group'] not in tuning_groups]
            if stage == 'main' and inserting_set == 'all-val':
                selected = [{'id': Path(row['file']).name, 'context_group': row['context_group']}
                            for row in sorted(data['splits']['val'], key=lambda row: row['file'])]
            elif stage == 'main':
                selected = final
            else:
                selected = tuning[:context_limit or (5 if stage == 'smoke' else 10)]
            ids = [row['id'] for row in selected]
            inputs[str(data_manifest)] = file_hash(data_manifest)
            inputs[paths['contexts_file']] = file_hash(paths['contexts_file'])
        if not ids:
            raise ValueError(f'No contexts for {task}')
        for label, content in [('tuning', tuning), ('final', final), ('selected', selected)]:
            (root / f'{task}_{label}_contexts.json').write_text(json.dumps(content, indent=2))
        cfg.save(root / 'configs' / task)
        frozen = root / 'configs' / task / 'resolved_config.yaml'
        inputs[str(frozen)] = file_hash(frozen)
        for label in ['tuning', 'final', 'selected']:
            path = root / f'{task}_{label}_contexts.json'
            inputs[str(path)] = file_hash(path)
        jobs = [{'id': f'{"P" if task == "pushing" else "I"}{number:02d}', 'number': number,
                 'projection': bool(number & 16), 'pf2mp': bool(number & 8),
                 'executed_action_filter': bool(number & 4), 'rtc': False, 'lipo': False}
                for number in COMBINATIONS]
        manifest['tasks'][task] = {'dataset_root': str(dataset_root), 'train_sha256': _train_digest(dataset_root),
                                   'context_ids': ids, 'context_list_sha256': _digest(selected),
                                   'config': str(frozen), 'inputs': inputs, 'jobs': jobs,
                                   'rollouts_per_job': len(ids) * len(seeds),
                                   'evaluation_label': ('validation_all_not_independent_test' if task == 'inserting' and inserting_set == 'all-val'
                                                        else 'heldout_contexts' if stage == 'main' else 'tuning_function_check' if stage == 'smoke' else 'tuning_pilot')}
    manifest['total_rollouts'] = sum(row['rollouts_per_job'] * 8 for row in manifest['tasks'].values())
    manifest['manifest_sha256'] = _digest(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f'Frozen plan: {manifest_path}')
    for task, row in manifest['tasks'].items():
        print(f'{task}: {len(row["context_ids"])} contexts × {len(seeds)} seeds × 8 combinations')
    print(f'Total: 16 sequential jobs, {manifest["total_rollouts"]} rollouts; RTC=OFF LiPo=OFF terminal VLSA=OFF')
    return manifest


def verify_inputs(root, manifest):
    """중단 후 재개할 때 코드·모델·목록을 바꿔 결과가 섞이는 것을 막는다."""
    check = {k: v for k, v in manifest.items() if k != 'manifest_sha256'}
    if _digest(check) != manifest['manifest_sha256']:
        raise ValueError('Experiment manifest was modified')
    for relative, expected in manifest['source_hashes'].items():
        if file_hash(REPO / relative) != expected:
            raise ValueError(f'Code changed since plan was frozen: {relative}')
    if any(version(name) != expected for name, expected in manifest['runtime_versions'].items()):
        raise ValueError('Runtime dependency versions changed since plan was frozen')
    for row in manifest['tasks'].values():
        if _train_digest(Path(row['dataset_root'])) != row['train_sha256']:
            raise ValueError('Train dataset changed since plan was frozen')
        for path, expected in row['inputs'].items():
            if file_hash(path) != expected:
                raise ValueError(f'Experiment input changed: {path}')


def run(root, job_id):
    """한 조합의 모든 context×seed를 실행하며 알고리즘 실패도 결과로 보존한다."""
    manifest = json.loads((root / 'experiment_manifest.json').read_text())
    verify_inputs(root, manifest)
    task = 'pushing' if job_id.startswith('P') else 'inserting'
    spec = manifest['tasks'][task]
    job = next((row for row in spec['jobs'] if row['id'] == job_id), None)
    if job is None:
        raise ValueError(f'Unsupported combination: {job_id}')
    output = root / job_id
    completed = output / 'COMPLETE.json'
    if completed.exists():
        saved = json.loads(completed.read_text())
        if (saved['manifest_sha256'] != manifest['manifest_sha256']
                or saved['results_sha256'] != file_hash(output / 'results.json')
                or saved['rollouts_sha256'] != file_hash(output / 'rollouts.json')):
            raise ValueError(f'Completed result differs from frozen plan: {job_id}')
        print(f'SKIP completed {job_id}')
        return
    if output.exists():
        # 중단된 출력은 지우지 않고 이력으로 보존한 뒤 같은 평가 키를 재실행한다.
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        output.rename(root / f'{job_id}.interrupted.{stamp}')
    cfg = ExperimentConfig.from_yaml(spec['config'], {
        'methods.projection.enabled': job['projection'], 'methods.pf2mp.enabled': job['pf2mp'],
        'paths.output_root': str(root)})
    metadata = {'experiment_protocol': 'method123_v1', 'combination_id': job_id,
                'experiment_manifest_sha256': manifest['manifest_sha256'],
                'context_list_sha256': spec['context_list_sha256'],
                'experiment_methods': {k: job[k] for k in ('projection', 'pf2mp', 'executed_action_filter', 'rtc', 'lipo')},
                'evaluation_label': spec['evaluation_label'], 'terminal_vlsa': False,
                'execution_queue_rule': 'discard_remaining_queue_after_modified_action',
                'git_commit': manifest['git_commit'], 'source_hashes': manifest['source_hashes'],
                'runtime_versions': manifest['runtime_versions']}
    result = evaluate(cfg, execution_filter=job['executed_action_filter'], sampling_seeds=manifest['sampling_seeds'],
                      context_ids=spec['context_ids'], output_directory=output, experiment_metadata=metadata)
    expected = {(context, seed) for context in spec['context_ids'] for seed in manifest['sampling_seeds']}
    actual = [(row['context_id'], row['sampling_seed']) for row in result['episodes']]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError(f'Missing/duplicate evaluation keys: {job_id}')
    completed.write_text(json.dumps({'manifest_sha256': manifest['manifest_sha256'],
                                     'results_sha256': file_hash(output / 'results.json'),
                                     'rollouts_sha256': file_hash(output / 'rollouts.json')}, indent=2))
    print(f'COMPLETE {job_id}: {len(actual)} rollout attempts (including method failures)')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'summarize'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=['smoke', 'pilot', 'main'], default='main')
    parser.add_argument('--inserting-set', choices=['holdout', 'all-val'], default='holdout')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--split-seed', type=int, default=2026)
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--context-limit', type=int)
    parser.add_argument('--job')
    args = parser.parse_args()
    root = ExperimentConfig.from_yaml(REPO / 'configs/pushing.yaml').require_output(args.output)
    if args.split_seed < 0 or any(value is not None and value < 1 for value in [args.max_steps, args.context_limit]):
        parser.error('split-seed must be nonnegative; limits must be positive')
    if args.command == 'prepare':
        prepare(root, args.stage, args.inserting_set, args.device, args.split_seed, args.max_steps, args.context_limit)
    elif args.command == 'run':
        if not args.job:
            parser.error('run requires --job')
        run(root, args.job)
    else:
        from .method123_metrics import summarize
        summarize(root)


if __name__ == '__main__':
    main()
