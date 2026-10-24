"""baseline 데이터 도구와 통합 정책으로 시뮬레이터 없는 생성을 검사한다."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from configs.config import parse_config
from . import checkpoint_reads, import_baseline
from .policy import Policy


def main():
    cfg = parse_config('sample')
    v, paths = cfg.values, cfg.values['paths']
    trainer = import_baseline('train')
    baseline = import_baseline('eval_pushing')
    with checkpoint_reads():
        mc, model, norm, conditioning, solver, checkpoint_step = baseline.load_model(
            paths['run_dir'], v['eval']['checkpoint'], v['eval']['weights'], paths['hardflow_root'], v['device'])
    base = baseline.Policy(model, norm, mc, conditioning, solver, v['sampling']['predict_steps'], v['device'])
    policy = Policy(base, cfg)
    files = trainer.find_split_files(str(Path(paths['data_root']) / f'{cfg.task}_sincos_v1'), v['sample']['split'])
    episodes, _ = trainer.load_split(files, mc['obs_dim'], mc['act_dim'], mc['horizon'])
    index = trainer.window_index(episodes, mc['horizon'])
    rng = np.random.default_rng(v['sample']['seed'])
    rows = rng.choice(len(index), size=min(v['sample']['n_windows'], len(index)), replace=False)
    torch.manual_seed(v['sample']['seed'])
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = Path(paths['output_root']) / f'sample_{cfg.task}_{stamp}'
    cfg.save(output)
    reports, generated, truth, first_previous = [], [], [], []
    started = perf_counter()
    # 우선 window 하나씩 읽기 쉬운 정책 경로로 검사한다. batch_size 설정은
    # baseline 학습/기존 sample 설정 호환용이며 통합 경로는 serial 실행이다.
    for episode_index, start in index[rows]:
        ep = episodes[episode_index]
        target = ep[start:start + mc['horizon']]
        previous = ep[start - 1, :2].copy() if start else np.zeros(2, dtype=np.float32)
        plan = policy.plan(target[0, 2:], previous)
        reports.append({'episode_index': int(episode_index), 'start': int(start),
                        'previous_action_m': previous.tolist(), **policy.last_report})
        if plan is not None:
            generated.append(plan)
            truth.append(target)
            first_previous.append(previous)
    elapsed = perf_counter() - started
    checkpoint = Path(paths['run_dir']) / v['eval']['checkpoint']
    result = {'schema_version': 2, 'task': cfg.task, 'attempts': len(rows),
              'generated': len(generated), 'failures': len(rows) - len(generated),
              'failure_rate': (len(rows) - len(generated)) / len(rows),
              'checkpoint_step': checkpoint_step, 'weights': v['eval']['weights'],
              'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              'normalizer_sha256': hashlib.sha256((Path(paths['run_dir']) / 'normalizer.npz').read_bytes()).hexdigest(),
              'seed': v['sample']['seed'], 'execution': 'serial_windows', 'elapsed_s': elapsed,
              'model_calls': sum(r['model_calls'] for r in reports),
              'qp_calls': sum(r['qp_calls'] for r in reports), 'windows': reports}
    if generated:
        gen, gt = np.stack(generated), np.stack(truth)
        # 실패 window를 RMSE 분모에 몰래 포함하거나 제외한 사실을 숨기지 않는다.
        # RMSE는 생성 성공 window만 대상으로 하며 실패율은 위에서 별도 기록한다.
        result.update(shape=list(gen.shape), finite=bool(np.isfinite(gen).all()),
                      condition_max_error=float(np.max(np.abs(gen[:, 0, 2:] - gt[:, 0, 2:]))),
                      action_rmse=float(np.sqrt(np.mean((gen[:, :, :2] - gt[:, :, :2])**2))),
                      observation_rmse=float(np.sqrt(np.mean((gen[:, 1:, 2:] - gt[:, 1:, 2:])**2))),
                      zero_action_rmse=float(np.sqrt(np.mean(gt[:, :, :2]**2))))
        np.savez_compressed(output / 'samples.npz', generated=gen, truth=gt,
                            previous_actions=np.stack(first_previous))
    (output / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({k: value for k, value in result.items() if k != 'windows'}, indent=2))
    print(f'Saved: {output}')


if __name__ == '__main__':
    main()
