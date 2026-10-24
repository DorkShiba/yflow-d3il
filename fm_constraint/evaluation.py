"""통합 정책의 공통 closed-loop 평가. 태스크별 환경·관측은 baseline에서 가져온다."""

import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
from time import perf_counter

import numpy as np

from . import checkpoint_reads, import_baseline
from .constraints import PlanConstraints
from .policy import Policy
from .qp import solve_qp


def run_evaluation(env, policy, contexts, read_state, build_obs, quat2euler,
                   task, settings, config, observe_safety=None):
    """계획 실패를 실행하지 않고, 실제 실행 행동으로만 이전 행동을 갱신한다.

    config는 schema v2의 resolved 값이다. 접촉 관측이 없으면 물리적
    safe success는 null로 둔다. all-off에서는 제약 위반을 측정만 한다.
    """
    if not contexts:
        raise ValueError('No evaluation contexts')
    methods, definitions = config['methods'], config['constraints']
    enabled = any(m['enabled'] for m in methods.values())
    recheck = methods['vlsa']['enabled'] and methods['vlsa']['execution_recheck']
    dt, tolerance = config['control']['dt_s'], config['solver']['feasibility_tolerance_m']
    ws = definitions['workspace']
    low = np.asarray(ws['low_m']) + ws['margin_m'] if ws['enabled'] else None
    high = np.asarray(ws['high_m']) - ws['margin_m'] if ws['enabled'] else None
    episodes, rollouts = [], {}
    plan_seconds = 0.
    plan_count = model_calls = qp_calls = attempts = executed = interventions = 0
    acceleration_violations = 0

    for ci, item in enumerate(contexts):
        for ti in range(settings['n_traj']):
            env.reset(random=False, context=copy.deepcopy(item['context']))
            c, boxes = read_state(env, quat2euler)
            fixed_z, des = c[2:3].copy(), c[:2].copy()
            # episode 시작만 0이다. 재계획에서는 직전 실제 행동을 계속 사용한다.
            previous = np.zeros(2)
            queue, logs = [], []
            plan_id, steps, failure = -1, 0, None
            info, done = {}, False
            command_min_h = float(np.min(np.r_[des - low, high - des])) if low is not None else None
            measured_min_h = float(np.min(np.r_[c[:2] - low, high - c[:2]])) if low is not None else None
            forbidden, min_distance = False, None
            if observe_safety:
                measured = observe_safety(env)
                forbidden = bool(measured['forbidden_contact'])
                min_distance = float(measured['min_distance'])
            while not done and steps < settings['max_steps']:
                obs = build_obs(des, c, boxes)
                new_plan_report = None
                if not queue:
                    started = perf_counter()
                    plan = policy.plan(obs, previous.copy())
                    plan_seconds += perf_counter() - started
                    plan_count += 1
                    plan_id += 1
                    new_plan_report = copy.deepcopy(policy.last_report)
                    model_calls += new_plan_report.get('model_calls', 0)
                    qp_calls += new_plan_report.get('qp_calls', 0)
                    if plan is None:
                        failure = new_plan_report.get('status', 'invalid_policy_output')
                        logs.append({'step': steps, 'plan_id': plan_id, 'plan_report': new_plan_report,
                                     'safe_action': None, 'failure': failure})
                        break
                    if (plan.ndim != 2 or plan.shape[0] < settings['execute_steps']
                            or plan.shape[1] < 2 or not np.isfinite(plan).all()):
                        failure = 'invalid_policy_output'
                        logs.append({'step': steps, 'plan_id': plan_id, 'plan_report': new_plan_report,
                                     'safe_action': None, 'failure': failure})
                        break
                    queue = [plan[k, :2].astype(np.float64) for k in range(settings['execute_steps'])]
                nominal = queue.pop(0)
                attempts += 1
                record = {'step': steps, 'plan_id': plan_id, 'measured_state': c.tolist(),
                          'command_state': des.tolist(), 'previous_action_m': previous.tolist(),
                          'nominal_action': nominal.tolist(), 'safe_action': None,
                          'plan_report': new_plan_report, 'queue_invalidated': False}
                # 실행 시점 측정은 최신 블록을 기준으로 한다. 계획 중 고정한
                # 기하의 보장은 plan_report에 남으며 이 값과 구분해 해석한다.
                current = PlanConstraints(obs, previous, 1, definitions, dt)
                record['latest_nominal_violation'] = current.violation(nominal[None])
                action = nominal
                if recheck:
                    filtered = solve_qp(nominal, current.G, current.d, config['solver'])
                    qp_calls += 1
                    action = filtered.pop('value')
                    record['execution_filter'] = {**filtered, 'geometry': current.geometry()}
                    if action is None:
                        failure = filtered['status']
                        record['failure'] = failure
                        logs.append(record)
                        break
                    # 실제 실행이 수정되면 남은 누적 목표는 원래 계획과 다르다.
                    # 다음 행동부터 최신 관측으로 재계획하도록 큐를 폐기한다.
                    if np.linalg.norm(action - nominal) > config['solver']['intervention_tolerance_m']:
                        queue.clear()
                        record['queue_invalidated'] = True
                modification = float(np.linalg.norm(action - nominal))
                interventions += int(modification > config['solver']['intervention_tolerance_m'])
                measurement = current.violation(action[None])
                acceleration_violations += int(measurement['acceleration_qp_max_m'] > tolerance)
                record.update(safe_action=action.tolist(), intervention_norm_m=modification,
                              latest_executed_violation=measurement,
                              execution_geometry=current.geometry())
                des = des + action
                command = np.concatenate([des, fixed_z, [0., 1., 0., 0.]])
                _, _, done, info = env.step(command)
                previous = action.copy()
                c, boxes = read_state(env, quat2euler)
                steps += 1
                executed += 1
                record.update(next_command_state=des.tolist(), next_measured_state=c.tolist(),
                              boxes_xy_m=[np.asarray(b[0][:2]).tolist() for b in boxes])
                if low is not None:
                    command_h = np.r_[des - low, high - des]
                    measured_h = np.r_[c[:2] - low, high - c[:2]]
                    command_min_h = min(command_min_h, float(command_h.min()))
                    measured_min_h = min(measured_min_h, float(measured_h.min()))
                    record.update(command_workspace_h=command_h.tolist(), measured_workspace_h=measured_h.tolist())
                if observe_safety:
                    measured = observe_safety(env)
                    forbidden |= bool(measured['forbidden_contact'])
                    distance = float(measured['min_distance'])
                    min_distance = min(min_distance, distance)
                logs.append(record)
            if not done and failure is None:
                failure = 'max_steps'
            success = bool(info.get('success', False)) and failure is None
            distance = float(info.get('mean_distance', float('nan')))
            episodes.append({'context_id': item['id'], 'context_index': ci, 'trajectory': ti,
                             'success': success, 'mode': int(info.get('mode', -1 if task == 'pushing' else 0)),
                             'mean_distance': distance if np.isfinite(distance) else None,
                             'steps': steps, 'failure': failure,
                             'forbidden_contact': forbidden if observe_safety else None,
                             'safe_success': (success and not forbidden) if observe_safety else None,
                             'min_distance': min_distance,
                             'command_workspace_min_h': command_min_h, 'measured_workspace_min_h': measured_min_h,
                             'command_workspace_violation': command_min_h < -tolerance if low is not None else None,
                             'measured_workspace_violation': measured_min_h < -tolerance if low is not None else None,
                             'n_boxes': sum(int(info.get(k, 0)) for k in ('one_box_success', 'two_box_success', 'three_box_success'))})
            key = f'c{ci:03d}_t{ti}'
            rollouts[key] = logs
            print(f'{key} success={int(success)} steps={steps} failure={failure}', flush=True)

    mode_values = list(range(4)) if task == 'pushing' else list(range(1, 7))
    entropies = []
    for ci in range(len(contexts)):
        counts = np.array([sum(e['context_index'] == ci and e['success'] and e['mode'] == mode
                               for e in episodes) for mode in mode_values], dtype=float)
        probs = counts / (counts.sum() + 1e-12)
        entropies.append(float(-(probs * np.log(probs + 1e-12) / np.log(len(mode_values))).sum()))
    entropy = float(np.mean(entropies)) if task == 'pushing' or settings['n_traj'] > 1 else None
    success_rate = float(np.mean([e['success'] for e in episodes]))
    distances = [e['mean_distance'] for e in episodes if e['mean_distance'] is not None]
    results = {'schema_version': 2, 'task': task, 'n_contexts': len(contexts), 'n_traj': settings['n_traj'],
               'methods': {name: m['enabled'] for name, m in methods.items()},
               'execution_recheck': bool(recheck), 'success_rate': success_rate, 'entropy': entropy,
               'score': 0.5 * (success_rate + entropy) if entropy is not None else None,
               'mean_distance': float(np.mean(distances)) if distances else None,
               'mean_steps': float(np.mean([e['steps'] for e in episodes])),
               'safe_success_rate': float(np.mean([e['safe_success'] for e in episodes])) if observe_safety else None,
               'forbidden_contact_free_rate': float(np.mean([not e['forbidden_contact'] for e in episodes])) if observe_safety else None,
               'safety_measurement': 'adapter_cumulative_internal_steps' if observe_safety else 'unavailable',
               'plan_ms': 1000 * plan_seconds / max(plan_count, 1), 'plan_attempts': plan_count,
               'model_calls': model_calls, 'qp_calls': qp_calls,
               'action_attempts': attempts, 'executed_actions': executed,
               'execution_intervention_rate': interventions / executed if executed else None,
               'executed_acceleration_qp_exceed_rate': acceleration_violations / executed if executed else None,
               'filter_abort_rate': float(np.mean([e['failure'] is not None and e['failure'] != 'max_steps' for e in episodes])),
               'episodes': episodes}
    if low is not None:
        for label in ('command', 'measured'):
            results[f'{label}_workspace_violation_rate'] = float(np.mean([e[f'{label}_workspace_violation'] for e in episodes]))
        results['command_workspace_success_rate'] = float(np.mean([e['success'] and not e['command_workspace_violation'] for e in episodes]))
    if task == 'inserting':
        for n, word in enumerate(('one', 'two', 'three'), 1):
            results[f'{word}_box_rate'] = float(np.mean([e['n_boxes'] >= n for e in episodes]))
    return results, rollouts


def evaluate(cfg):
    """baseline 환경·모델 로더를 재사용해 두 태스크를 동일한 평가 루프로 실행한다."""
    cfg.validate('eval')
    v, paths, settings = cfg.values, cfg.values['paths'], cfg.values['eval']
    if not Path(paths['d3il_root']).is_dir():
        raise FileNotFoundError(f'D3IL checkout not found: {paths["d3il_root"]}')
    pushing = import_baseline('eval_pushing')
    module = pushing if cfg.task == 'pushing' else import_baseline('eval_inserting')
    old_cwd = Path.cwd()
    try:
        with checkpoint_reads():
            model_cfg, model, norm, conditioning, solver, checkpoint_step = pushing.load_model(
                paths['run_dir'], settings['checkpoint'], settings['weights'], paths['hardflow_root'], v['device'])
        model_cfg['data_dir'] = str(Path(paths['data_root']) / f'{cfg.task}_sincos_v1')
        Env, quat2euler = module.setup_d3il(paths['d3il_root'])
        if settings['export_contexts']:
            items = module.load_contexts(model_cfg, 'val', paths['raw_root'], quat2euler, 0, None)
            output = cfg.require_output(settings['export_contexts'])
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open('xb') as handle:
                pickle.dump({f: {'context': ctx, 'data': module.slim(data)}
                             for f, data, ctx, _ in items}, handle)
            return
        env = Env(render=False, n_substeps=35, max_steps_per_episode=settings['max_steps'])
        env.start()
        if settings['check_obs'] and cfg.task == 'pushing':
            module.check_obs(env, quat2euler, model_cfg, paths['raw_root'], settings['check_obs'])
            return
        if cfg.task == 'pushing':
            context_path = Path(paths['d3il_root']) / 'environments/dataset/data/pushing' / f'{settings["contexts"]}_contexts.pkl'
            contexts = np.load(context_path, allow_pickle=True)
            if settings['n_contexts']:
                contexts = contexts[:settings['n_contexts']]
            items = [{'id': str(i), 'context': ctx} for i, ctx in enumerate(contexts)]
        else:
            raw_items = module.load_contexts(model_cfg, 'val' if settings['check_obs'] else settings['split'],
                                            paths['raw_root'], quat2euler,
                                            settings['check_obs'] or settings['n_contexts'], paths['contexts_file'])
            if settings['check_obs']:
                for name, data, ctx, npz in raw_items[:settings['check_obs']]:
                    env.reset(random=False, context=ctx)
                    env.robot.beam_to_joint_pos(np.asarray(data['robot']['j_pos'][0]).copy())
                    env.scene.next_step(log=False)
                    c, boxes = module.read_state(env, quat2euler)
                    obs = module.build_obs(np.asarray(data['robot']['des_c_pos'][0]), c, boxes)
                    with np.load(npz) as record:
                        error = float(np.abs(obs - record['obs'][0]).max())
                    print(f'{name}: max observation error {error:.6g}')
                return
            items = [{'id': name, 'context': ctx, 'demo_length': len(data['robot']['des_c_pos'])}
                     for name, data, ctx, _ in raw_items]
        base = pushing.Policy(model, norm, model_cfg, conditioning, solver, v['sampling']['predict_steps'], v['device'])
        policy = Policy(base, cfg)
        import torch
        torch.manual_seed(settings['seed'])
        np.random.seed(settings['seed'])
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        output = Path(paths['output_root']) / f'eval_{cfg.task}_{stamp}'
        cfg.save(output)
        checkpoint = Path(paths['run_dir']) / settings['checkpoint']
        metadata = {'task': cfg.task, 'checkpoint': str(checkpoint.resolve()),
                    'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    'checkpoint_step': checkpoint_step, 'weights': settings['weights'],
                    'seed': settings['seed'], 'predict_steps': v['sampling']['predict_steps'],
                    'constraint_scope': 'command_plan', 'schema_version': 2,
                    'normalizer_sha256': hashlib.sha256((Path(paths['run_dir']) / 'normalizer.npz').read_bytes()).hexdigest(),
                    'control_dt_s': v['control']['dt_s'],
                    'execute_steps': settings['execute_steps'], 'max_steps': settings['max_steps']}
        (output / 'model_config.json').write_text((Path(paths['run_dir']) / 'config.json').read_text())
        results, rollouts = run_evaluation(env, policy, items, module.read_state, module.build_obs,
                                          quat2euler, cfg.task, settings, config=v)
        results.update(metadata)
        (output / 'results.json').write_text(json.dumps(results, indent=2, allow_nan=False))
        (output / 'rollouts.json').write_text(json.dumps(rollouts, allow_nan=False))
        print(f'Saved: {output}')
        return results
    finally:
        os.chdir(old_cwd)
