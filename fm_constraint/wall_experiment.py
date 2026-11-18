"""벽 OOD 본실험: 32조합 × (ID 30 context × noise 3) × Pushing·Inserting, 장면에 학습 때 없던 벽.

    python -m fm_constraint.wall_experiment run --task pushing --combos all --contexts 0-29 \\
        --device cuda:0 --name hyeongbin [--smoke 120 --margin 0.012]
    python -m fm_constraint.wall_experiment report --name hyeongbin   # docs/report_wall_hyeongbin.md
    python -m fm_constraint.wall_experiment merge                     # docs/report_wall_results.{md,json}
    python -m fm_constraint.wall_experiment summary --task pushing [--smoke]   # 벽 지표 표

ID 실험(fm_constraint.experiment)과 같은 context·seed·FM noise를 쓰며 벽만 다르다. 실행 루프·조합·보고서는
experiment 모듈을 그대로 쓰고, 이 모듈이 다음을 바꿔 끼운다(기존 CODE_FILES는 수정하지 않는다).
    장면      fm_baseline eval 모듈의 setup_d3il → 벽을 넣는 Env (fm_constraint.wall.wall_env_class)
    ①·⑤      policy/evaluation의 PlanConstraints → WallPlanConstraints, Policy → WallPolicy(매 Euler 면 재선택)
    ③        evaluation.solve_qp → filter_solve_qp(실행 직전 nominal로 면 재선택)
    결과      experiments/wall_v1/, fm_constraint/runs/experiment_wall_v1/, docs/report_wall_*.md
fingerprint = 기존 CODE_FILES + wall.py + wall_experiment.py.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from . import experiment as ex
from . import wall

PROTOCOL = 'wall_v1'
WALL_FILES = ('fm_constraint/wall.py', 'fm_constraint/wall_experiment.py')


def install(task, margin=None):
    """experiment·evaluation·policy 모듈의 이름을 벽 버전으로 바꾼다. 프로세스당 한 번."""
    from . import evaluation, import_baseline, policy
    wall.configure(task, margin)

    ex.CODE_FILES = tuple(f for f in ex.CODE_FILES if f not in WALL_FILES) + WALL_FILES
    ex.protocol = lambda ood=None: PROTOCOL
    ex.result_root = lambda ood=None: ex.RESULT_BASE / PROTOCOL
    # smoke에서 margin을 바꾸면 결과 폴더를 따로 둔다(같은 fingerprint라 SKIP되지 않게).
    local = f'experiment_{PROTOCOL}' + ('' if margin is None else f'_margin{1e3 * float(margin):g}mm')
    ex.local_root = lambda ood=None: ex.LOCAL_BASE / local
    ex.report_path = lambda name, ood=None: ex.REPORT_DIR / f'report_wall_{name}.md'

    policy.PlanConstraints = wall.WallPlanConstraints
    evaluation.PlanConstraints = wall.WallPlanConstraints
    evaluation.Policy = wall.WallPolicy
    evaluation.solve_qp = wall.filter_solve_qp

    for name in ('eval_pushing', 'eval_inserting'):
        module = import_baseline(name)
        if getattr(module.setup_d3il, '_wall', False):
            continue
        original = module.setup_d3il

        def setup(root, _original=original):
            Env, quat2euler = _original(root)
            return wall.wall_env_class(Env), quat2euler

        setup._wall = True
        module.setup_d3il = setup

    if not getattr(evaluation.evaluate, '_wall', False):
        original_evaluate = evaluation.evaluate

        def evaluate(cfg, **kwargs):
            kwargs['experiment_metadata'] = {**(kwargs.get('experiment_metadata') or {}), 'wall': wall.spec()}
            result = original_evaluate(cfg, **kwargs)
            env = wall.LAST_ENV['env']
            output = Path(kwargs['output_directory'])
            rollouts = json.loads((output / 'rollouts.json').read_text())
            episodes = result['episodes']
            logs = env.wall_log[-len(episodes):]
            if len(rollouts) != len(episodes) or len(logs) != len(episodes):
                raise RuntimeError('wall contact log does not match the episodes')
            tolerance = cfg.values['solver']['feasibility_tolerance_m']
            for episode, rows, log in zip(episodes, rollouts.values(), logs):
                episode.update(wall.episode_metrics(rows, log, tolerance))
            result['wall_geoms'] = env.wall_geom_names
            (output / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False))
            return result

        evaluate._wall = True
        evaluation.evaluate = evaluate


# ---- 요약 ---------------------------------------------------------------------------
COLUMNS = (('success', '성공 %'), ('wall_hand_contact', '손–벽 접촉 %'), ('wall_block_contact', '블록–벽 접촉 %'),
           ('wall_command_violation', '명령 벽 침범 %'), ('wall_stalled', '벽 앞 정지 %'), ('abort', '방법 실패 %'))


def _flags(episode):
    out = {k: bool(episode.get(k)) for k, _ in COLUMNS if k != 'abort'}
    out['abort'] = episode['failure'] is not None and episode['failure'] != 'max_steps'
    return out


def summary(task, smoke=False, out=None):
    root = ex.local_root() / 'smoke' if smoke else ex.result_root()
    units = [json.loads(p.read_text()) for p in sorted((root / task).glob('*/ctx_*/result.json'))]
    if not units:
        raise SystemExit(f'No results under {root / task}')
    lines = [f'# 벽 OOD 요약: {task}' + (' (smoke)' if smoke else ''), '',
             f'- 벽: {json.dumps(wall.spec(task) if wall._STATE["task"] else wall.WALLS[task])}',
             f'- 단위 {len(units)}개, fingerprint {", ".join(sorted({u["code_fingerprint"][:12] for u in units}))}', '',
             '| 조합 | rollout | ' + ' | '.join(label for _, label in COLUMNS) + ' | 최소 손–벽 거리 mm (중앙값) |',
             '|---|---:|' + '---:|' * (len(COLUMNS) + 1)]
    groups = {'① off': [], '① on': []}
    for number in sorted({u['number'] for u in units}):
        episodes = [e for u in units if u['number'] == number for e in u['episodes']]
        groups['① on' if number & 16 else '① off'] += episodes
        flags = [_flags(e) for e in episodes]
        distances = [e['wall_min_hand_distance_m'] for e in episodes if e.get('wall_min_hand_distance_m') is not None]
        lines.append(f'| {ex.combo_id(task, number)} | {len(episodes)} | '
                     + ' | '.join(f'{100 * np.mean([f[k] for f in flags]):.1f}' for k, _ in COLUMNS)
                     + (f' | {1e3 * np.median(distances):.1f} |' if distances else ' | – |'))
    lines += ['', '| 묶음 | rollout | ' + ' | '.join(label for _, label in COLUMNS) + ' |',
              '|---|---:|' + '---:|' * len(COLUMNS)]
    for label, episodes in groups.items():
        if episodes:
            flags = [_flags(e) for e in episodes]
            lines.append(f'| {label} | {len(episodes)} | '
                         + ' | '.join(f'{100 * np.mean([f[k] for f in flags]):.1f}' for k, _ in COLUMNS) + ' |')
    text = '\n'.join(lines) + '\n'
    if out:
        Path(out).write_text(text)
        print(f'Saved {out}')
    print(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run')
    r.add_argument('--task', choices=ex.TASKS, required=True)
    r.add_argument('--combos', default='all')
    r.add_argument('--contexts', required=True)
    r.add_argument('--device', required=True)
    r.add_argument('--name', required=True)
    r.add_argument('--smoke', type=int, default=None, metavar='MAX_STEPS',
                   help='동작 확인: 최대 step, 결과는 로컬 smoke 폴더에만')
    r.add_argument('--margin', type=float, default=None,
                   help=f'smoke 전용: 벽 키움 여유(m, 기본 {wall.MARGIN_M}). 본실험은 wall.MARGIN_M만 쓴다')
    p = sub.add_parser('report')
    p.add_argument('--name', required=True)
    p.add_argument('--out')
    m = sub.add_parser('merge')
    m.add_argument('--out')
    s = sub.add_parser('summary')
    s.add_argument('--task', choices=ex.TASKS, required=True)
    s.add_argument('--smoke', action='store_true')
    s.add_argument('--out')
    args = parser.parse_args()
    if args.command == 'run':
        if args.margin is not None and args.smoke is None:
            parser.error('--margin is only for --smoke runs (the main run uses wall.MARGIN_M)')
        install(args.task, args.margin)
        ex.run(args.task, ex.parse_combos(args.combos), ex.parse_indices(args.contexts, ex.N_CONTEXTS),
               args.device, args.name, args.smoke)
    elif args.command == 'report':
        install('pushing')
        ex.report(args.name, args.out)
        path = Path(args.out) if args.out else ex.report_path(args.name)
        path.write_text(path.read_text().replace('# 실험 보고 (ID)', '# 실험 보고 (벽 OOD)', 1))
    elif args.command == 'merge':
        install('pushing')
        ex.merge(args.out)
    else:
        install(args.task)
        summary(args.task, args.smoke, args.out)


if __name__ == '__main__':
    main()
