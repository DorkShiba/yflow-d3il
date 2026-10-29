"""저장된 궤적과 당시 제약 기하를 지도 PNG/GIF로 재생한다. 정책을 다시 실행하지 않는다."""

import argparse
import copy
import os
import json
from pathlib import Path

import numpy as np


def replay_camera(directory, episode, rows, indices, size, config_path=None):
    """동일 context에서 실제 실행 행동만 재생하고 카메라와 좌표 오차를 반환한다.

    모델과 제약을 다시 실행하지 않는다. MuJoCo GL backend는 프로세스 시작
    전에 설정해야 한다. 영상 생성 실패를 지도-only 성공으로 숨기지 않는다.
    """
    from configs.config import ExperimentConfig
    from . import import_baseline
    cfg = ExperimentConfig.from_yaml(config_path or directory / 'resolved_config.yaml')
    v, settings, paths = cfg.values, cfg.values['eval'], cfg.values['paths']
    result = json.loads((directory / 'results.json').read_text())
    entry = next((e for e in result.get('episodes', [])
                  if f"c{e['context_index']:03d}_t{e['trajectory']}" == episode), None)
    if entry is None:
        raise ValueError('Camera replay requires an episode/context identity in results.json')
    module = import_baseline('eval_pushing' if cfg.task == 'pushing' else 'eval_inserting')
    old_cwd = Path.cwd()
    images, q_error, p_error = {}, 0., 0.
    try:
        Env, quat = module.setup_d3il(paths['d3il_root'])
        if cfg.task == 'pushing':
            source = Path(paths['d3il_root']) / 'environments/dataset/data/pushing' / f"{settings['contexts']}_contexts.pkl"
            contexts = np.load(source, allow_pickle=True)
            context = contexts[entry['context_index']]
        else:
            mc = {'data_dir': str(Path(paths['data_root']) / 'inserting_sincos_v1')}
            contexts = module.load_contexts(mc, settings['split'], paths['raw_root'], quat,
                                           settings['n_contexts'], paths['contexts_file'])
            context = next(c for name, _, c, _ in contexts if name == entry['context_id'])
        env = Env(render=False, n_substeps=35, max_steps_per_episode=10**7)
        env.start()
        env.reset(random=False, context=copy.deepcopy(context))
        measured, _ = module.read_state(env, quat)
        q, z = measured[:2].copy(), measured[2:3].copy()
        for i, row in enumerate(rows):
            q += np.asarray(row['safe_action'], dtype=np.float64)
            env.step(np.r_[q, z, 0., 1., 0., 0.])
            measured, _ = module.read_state(env, quat)
            q_error = max(q_error, float(np.max(np.abs(q - row['next_command_state'][:2]))))
            p_error = max(p_error, float(np.max(np.abs(measured[:2] - row['next_measured_state'][:2]))))
            if i in indices:
                images[i] = np.asarray(env.bp_cam.get_image(width=size, height=size, depth=False))[..., :3].astype(np.uint8)
    finally:
        os.chdir(old_cwd)
    return images, {'command_max_error_m': q_error, 'measured_xy_max_error_m': p_error,
                    'within_0_1mm': p_error <= 1e-4}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval-dir', required=True)
    parser.add_argument('--episode', default='c000_t0')
    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--gif', action='store_true')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--map-only', action='store_true', help='저장된 좌표만 재생; 기본 모드')
    mode.add_argument('--camera', action='store_true', help='동일 context에서 MuJoCo 카메라 영상을 재생')
    parser.add_argument('--size', type=int, default=480)
    parser.add_argument('--config', help='카메라 재생용 설정; 기본은 평가 폴더의 resolved_config.yaml')
    parser.add_argument('--out-dir')
    args = parser.parse_args()
    if args.stride < 1 or args.fps < 1:
        parser.error('stride and fps must be positive')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle
    from PIL import Image

    directory = Path(args.eval_dir).resolve()
    # 영상은 실제 실행 행동만 재생한다. 제목의 순서는 결과 메타데이터에서 읽고,
    # 필드가 없는 과거 파일에 현재 정책의 순서를 임의로 붙이지 않는다.
    result_path = directory / 'results.json'
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    order = result.get('pipeline_order')
    order_label = ' → '.join(order) if order else 'unknown order'
    if (directory / 'rollouts.json').exists():
        rows = json.loads((directory / 'rollouts.json').read_text())[args.episode]
    elif (directory / 'rollouts.npz').exists():
        # 이전 로그에는 새 제약 기하가 없다. 좌표만 재생하고 제약은 그리지 않는다.
        with np.load(directory / 'rollouts.npz') as z:
            q, p = z[args.episode + '_des'], z[args.episode + '_c']
            rows = [{'next_command_state': x[:2].tolist(), 'next_measured_state': y[:3].tolist(),
                     'safe_action': True} for x, y in zip(q, p)]
    else:
        raise FileNotFoundError('No saved rollouts; summary-only legacy results cannot be replayed')
    rows = [r for r in rows if r.get('safe_action') is not None]
    if not rows:
        raise ValueError('Episode has no executed actions to render')
    output = Path(args.out_dir) if args.out_dir else directory / 'maps'
    output.mkdir(parents=True, exist_ok=True)
    q = np.array([r['next_command_state'][:2] for r in rows])
    p = np.array([r['next_measured_state'][:2] for r in rows])
    frames = []
    indices = list(range(0, len(rows), args.stride)) if args.gif else []
    indices = sorted(set(indices + [len(rows) - 1]))
    camera, replay_check = ({}, None)
    if args.camera:
        camera, replay_check = replay_camera(directory, args.episode, rows, set(indices), args.size, args.config)
        (output / f'{args.episode}_replay_check.json').write_text(json.dumps(replay_check, indent=2))
    for end in indices:
        if args.camera:
            fig, (camera_ax, ax) = plt.subplots(1, 2, figsize=(12, 6))
            camera_ax.imshow(camera[end])
            camera_ax.set_axis_off()
        else:
            fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(q[:end + 1, 0], q[:end + 1, 1], color='darkorange', label='command')
        ax.plot(p[:end + 1, 0], p[:end + 1, 1], color='royalblue', label='measured')
        # 실행 전 최신 관측으로 측정한 기하다. 계획 고정 기하와 혼동하지 않게 표시한다.
        geo = rows[end].get('execution_geometry', {})
        if geo.get('workspace_low_m') is not None:
            lo, hi = np.array(geo['workspace_low_m']), np.array(geo['workspace_high_m'])
            ax.add_patch(Rectangle(lo, *(hi - lo), fill=False, color='black', label='workspace'))
        if geo.get('object_normals') is not None:
            normals, center, radius = np.array(geo['object_normals']), np.array(geo['object_center_m']), geo['object_inradius_m']
            vertices = np.array([np.linalg.solve(np.stack((normals[k], normals[(k + 1) % len(normals)])), [radius, radius]) + center for k in range(len(normals))])
            ax.add_patch(Polygon(vertices, fill=False, color='seagreen', label='latest object envelope'))
        if geo.get('blocks_xy_m'):
            blocks = np.asarray(geo['blocks_xy_m'])
            ax.scatter(blocks[:, 0], blocks[:, 1], c='seagreen', marker='s', label='blocks')
        ax.set(xlabel='x (m)', ylabel='y (m)', title=f'{args.episode}: step {end + 1}\n{order_label}')
        ax.set_aspect('equal', adjustable='datalim')
        ax.legend(loc='best')
        fig.tight_layout()
        if end == len(rows) - 1:
            fig.savefig(output / f'{args.episode}.png', dpi=140)
        if args.gif:
            fig.canvas.draw()
            frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()))
        plt.close(fig)
    if frames:
        frames[0].save(output / f'{args.episode}.gif', save_all=True, append_images=frames[1:], duration=1000 / args.fps, loop=0)
    if replay_check is not None:
        print(json.dumps(replay_check))
    print(f'Saved: {output}')


if __name__ == '__main__':
    main()
