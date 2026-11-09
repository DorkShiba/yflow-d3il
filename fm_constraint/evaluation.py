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
from .policy import PIPELINE_ORDER, PIPELINE_METADATA, Policy
from .pf2mp import prepare_potential
from .qp import solve_qp
from .tasks import get_task
from ood.apply import Episode, context_from_json, load_cases


def id_context_key(task, ood_context_id):
    """OOD 파일의 context ID → 같은 초기 배치의 ID 평가 context ID.

    Pushing ID 평가는 test 배열 index('7'), OOD 파일은 'test_007'이다. Inserting은 같다.
    """
    if task == 'pushing':
        prefix, _, number = ood_context_id.rpartition('_')
        if prefix != 'test' or not number.isdigit():
            raise ValueError(f'Unexpected Pushing OOD context ID {ood_context_id!r}')
        return str(int(number))
    return ood_context_id


def ood_context_id(task, id_context_id):
    """id_context_key의 역. ID 평가 목록의 context ID → OOD 파일의 context ID."""
    return f'test_{int(id_context_id):03d}' if task == 'pushing' else id_context_id


def run_evaluation(env, policy, contexts, read_state, build_obs, quat2euler,
                   task, settings, config, observe_safety=None, spec=None):
    """계획 실패를 실행하지 않고, 실제 실행 행동으로만 이전 행동을 갱신한다.

    config는 schema v4의 resolved 값이다. 접촉 관측이 없으면 물리적
    safe success는 null로 둔다. all-off에서는 제약 위반을 측정만 한다.

    context에 'ood' 사례(data/ood JSONL 한 행)가 있으면 OOD 조건을 적용한다.
    정책 seed는 sampling_seeds가 주어지면 그것을(본실험: ID와 같은 seed 0·1·2, FM noise는
    item['noise_key']로 ID와 같은 값), 없으면 사례의 policy_seeds를 쓴다. 관측 노이즈(C)는 정책과 실행 직전
    필터가 보는 관측에만 넣는다. 제약 위반 측정은 항상 실제 상태로 한다.
    지연(E)이 있으면 남은 행동이 d개일 때 다음 계획을 세우고 새 chunk는 index d부터 실행한다.

    ④ RTC: 활성 계획에서 아직 실행하지 않은 H−s개 행동을 다음 계획의 target으로 넘긴다.
    ③이 행동을 수정해 큐를 폐기하면 참조가 끊기므로 다음 계획은 RTC 없이 만든다.
    ⑤ LiPo: 실행된 목표 위치 최근 3개와, 지연 중 실행이 확정된 d개 행동(고정 구간)을 넘긴다.
    """
    if not contexts:
        raise ValueError('No evaluation contexts')
    methods, definitions = config['methods'], config['constraints']
    recheck = methods['filter']['enabled']
    use_rtc = methods['rtc']['enabled']
    seeds = settings.get('sampling_seeds')
    if seeds is not None and (not seeds or len(set(seeds)) != len(seeds)
                              or any(type(seed) is not int or seed < 0 for seed in seeds)):
        raise ValueError('sampling_seeds must contain distinct nonnegative integers')
    dt, tolerance = config['control']['dt_s'], config['solver']['feasibility_tolerance_m']
    ws = definitions['workspace']
    low = np.asarray(ws['low_m']) + ws['margin_m'] if ws['enabled'] else None
    high = np.asarray(ws['high_m']) - ws['margin_m'] if ws['enabled'] else None
    episodes, rollouts = [], {}
    plan_seconds = 0.
    plan_count = model_calls = qp_calls = attempts = executed = interventions = pf_calls = 0
    pf_ms = 0.
    acceleration_violations = 0

    walls = spec.walls(env) if spec is not None and any('ood' in item for item in contexts) else []
    for ci, item in enumerate(contexts):
        case = item.get('ood')
        if seeds is not None:
            item_seeds = seeds
        else:
            item_seeds = case['policy_seeds'][:settings['n_traj']] if case else None
        for ti in range(len(item_seeds) if item_seeds is not None else settings['n_traj']):
            # 조합·이전 rollout의 길이와 독립적인 FM 전용 RNG를 사용한다.
            generator = None
            sampling_seed = item_seeds[ti] if item_seeds is not None else None
            noise_seed = None
            if item_seeds is not None:
                import torch
                if case and seeds is None:
                    noise_seed = int(sampling_seed) % (2**63 - 1)
                else:
                    # noise_key: OOD 사례가 ID와 같은 FM noise를 쓰도록 ID 평가의 context ID를 넣는다.
                    key = item.get('noise_key', item['id'])
                    payload = json.dumps([task, key, sampling_seed], separators=(',', ':'))
                    noise_seed = int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], 'big') % (2**63 - 1)
                # noise는 장비와 무관하게 CPU 난수로 만든 뒤 옮긴다. CUDA 난수열은 CPU와 달라서
                # 장비가 다른 팀원 사이에 같은 seed의 noise가 달라지는 것을 막는다.
                generator = torch.Generator(device='cpu').manual_seed(noise_seed)
            episode = Episode(case, ti, spec, env, settings['execute_steps'])
            env.reset(random=False, context=copy.deepcopy(item['context']))
            if episode.robot_start is not None:
                spec.move_hand(env, episode.robot_start)
            c, boxes = read_state(env, quat2euler)
            fixed_z, des = c[2:3].copy(), c[:2].copy()
            # episode 시작만 0이다. 재계획에서는 직전 실제 행동을 계속 사용한다.
            previous = np.zeros(2)
            queue, logs = [], []
            pending, delay = None, episode.delay
            # 활성 계획의 행동 (H,2)과 다음에 실행할 index. ④ RTC의 suffix에 쓴다.
            active_actions, active_index = None, 0
            des_history = [des.copy()]
            plan_id, active_plan, steps, failure = -1, -1, 0, None
            info, done = {}, False
            command_min_h = float(np.min(np.r_[des - low, high - des])) if low is not None else None
            measured_min_h = float(np.min(np.r_[c[:2] - low, high - c[:2]])) if low is not None else None
            forbidden, min_distance = False, None
            if observe_safety:
                measured = observe_safety(env)
                forbidden = bool(measured['forbidden_contact'])
                min_distance = float(measured['min_distance'])
            while not done and steps < settings['max_steps']:
                if episode.push_due(steps, len(queue)):
                    episode.push(steps, boxes, c[:2], walls)
                    c, boxes = read_state(env, quat2euler)
                # obs: 정책·필터가 보는 관측(C 노이즈 포함), true_obs: 위반 측정용 실제 상태.
                true_obs = build_obs(des, c, boxes)
                obs = build_obs(des, c, episode.observe_boxes(boxes, steps)) if episode.noise else true_obs
                new_plan_report = None
                seam = False
                if (not queue and pending is None) or (
                        delay and len(queue) == delay and pending is None and steps > 0):
                    extra = {'history': np.asarray(des_history[-3:]),
                             'frozen': np.asarray(queue) if queue else None, 'rtc_suffix': None}
                    if use_rtc and active_actions is not None:
                        suffix = active_actions[active_index:active_index + len(active_actions) - settings['execute_steps']]
                        if len(suffix) == len(active_actions) - settings['execute_steps']:
                            extra['rtc_suffix'] = suffix.copy()
                    started = perf_counter()
                    if generator is None:
                        plan = policy.plan(obs, previous.copy(), **extra)
                    else:
                        base = policy.base
                        noise = torch.randn(1, base.horizon, base.act_dim + base.obs_dim,
                                            generator=generator, dtype=base.center.dtype).to(base.device)
                        plan = policy.plan(obs, previous.copy(), noise=noise, **extra)
                    plan_seconds += perf_counter() - started
                    plan_count += 1
                    plan_id += 1
                    new_plan_report = copy.deepcopy(policy.last_report)
                    if generator is not None:
                        new_plan_report.update(sampling_seed=sampling_seed, noise_seed=noise_seed,
                                               noise_index=plan_id)
                    model_calls += new_plan_report.get('model_calls', 0)
                    qp_calls += new_plan_report.get('qp_calls', 0)
                    pf_calls += new_plan_report.get('pf_calls', 0)
                    pf_ms += new_plan_report.get('pf_ms', 0.)
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
                    if not queue:
                        queue = [plan[k, :2].astype(np.float64) for k in range(settings['execute_steps'])]
                        active_plan, seam = plan_id, True
                        active_actions, active_index = plan[:, :2].astype(np.float64).copy(), 0
                    else:
                        # 지연: 남은 d개를 실행하는 동안 계산했다고 보고, 새 계획은 index d부터 쓴다.
                        pending = (plan_id, plan)
                if not queue and pending is not None:
                    active_plan, plan = pending
                    queue = [plan[k, :2].astype(np.float64) for k in range(delay, delay + settings['execute_steps'])]
                    pending, seam = None, True
                    active_actions, active_index = plan[:, :2].astype(np.float64).copy(), delay
                nominal = queue.pop(0)
                active_index += 1
                attempts += 1
                record = {'step': steps, 'plan_id': active_plan, 'seam': seam, 'measured_state': c.tolist(),
                          'command_state': des.tolist(), 'previous_action_m': previous.tolist(),
                          'nominal_action': nominal.tolist(), 'safe_action': None,
                          'plan_report': new_plan_report, 'queue_invalidated': False}
                # 실행 시점 측정은 최신 블록을 기준으로 한다. 계획 중 고정한
                # 기하의 보장은 plan_report에 남으며 이 값과 구분해 해석한다.
                current = PlanConstraints(obs, previous, 1, definitions, dt)
                measured_at = current if obs is true_obs else PlanConstraints(true_obs, previous, 1, definitions, dt)
                record['latest_nominal_violation'] = measured_at.violation(nominal[None])
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
                        pending, active_actions = None, None
                        record['queue_invalidated'] = True
                modification = float(np.linalg.norm(action - nominal))
                interventions += int(modification > config['solver']['intervention_tolerance_m'])
                measurement = measured_at.violation(action[None])
                acceleration_violations += int(measurement['acceleration_qp_max_m'] > tolerance)
                record.update(safe_action=action.tolist(), intervention_norm_m=modification,
                              latest_executed_violation=measurement,
                              execution_geometry=measured_at.geometry())
                des = des + action
                des_history.append(des.copy())
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
                             'sampling_seed': sampling_seed, 'noise_seed': noise_seed,
                             'success': success, 'mode': int(info.get('mode', -1 if task == 'pushing' else 0)),
                             'mean_distance': distance if np.isfinite(distance) else None,
                             'steps': steps, 'failure': failure,
                             'forbidden_contact': forbidden if observe_safety else None,
                             'safe_success': (success and not forbidden) if observe_safety else None,
                             'min_distance': min_distance,
                             'command_workspace_min_h': command_min_h, 'measured_workspace_min_h': measured_min_h,
                             'command_workspace_violation': command_min_h < -tolerance if low is not None else None,
                             'measured_workspace_violation': measured_min_h < -tolerance if low is not None else None,
                             'n_boxes': sum(int(info.get(k, 0)) for k in ('one_box_success', 'two_box_success', 'three_box_success')),
                             'ood_case': case['id'] if case else None,
                             'ood_events': episode.describe() if case else None})
            key = f'c{ci:03d}_t{ti}'
            # 문서 지표는 실제 실행 step과 rollout 내부의 유효 차분으로 계산한다.
            performed = [row for row in logs if row.get('safe_action') is not None]
            violations = [row['latest_executed_violation'] for row in performed]
            action_array = np.asarray([row['safe_action'] for row in performed], dtype=float).reshape(-1, 2)
            episodes[-1].update(
                constraint_violation_steps=sum(row['qp_max_m'] > tolerance for row in violations),
                constraint_evaluated_steps=len(performed),
                constraint_violation_steps_by_type={name: sum(row[f'{name}_qp_max_m'] > tolerance for row in violations)
                                                    for name in ('workspace', 'acceleration', 'object_envelope')},
                action_change_m=float(np.linalg.norm(np.diff(action_array, axis=0), axis=-1).mean()) if len(action_array) > 1 else None,
                command_jerk_m_s3=float(np.linalg.norm(np.diff(action_array, n=2, axis=0), axis=-1).mean() / dt**3) if len(action_array) > 2 else None)
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
    if seeds is not None:
        # 프로토콜 실행에서는 무성공 context를 entropy 0으로 채우지 않는다.
        valid = [value for ci, value in enumerate(entropies)
                 if any(e['success'] and e['context_index'] == ci for e in episodes)]
        entropy = float(np.mean(valid)) if valid else None
    success_rate = float(np.mean([e['success'] for e in episodes]))
    distances = [e['mean_distance'] for e in episodes if e['mean_distance'] is not None]
    results = {'schema_version': 4, 'pipeline_order': list(PIPELINE_ORDER), **PIPELINE_METADATA,
               'task': task, 'n_contexts': len(contexts), 'n_traj': len(seeds) if seeds is not None else settings['n_traj'],
               'sampling_seeds': seeds,
               'methods': {name: m['enabled'] for name, m in methods.items()},
               'execution_recheck': bool(recheck), 'success_rate': success_rate, 'entropy': entropy,
               'score': 0.5 * (success_rate + entropy) if entropy is not None else None,
               'mean_distance': float(np.mean(distances)) if distances else None,
               'mean_steps': float(np.mean([e['steps'] for e in episodes])),
               'safe_success_rate': float(np.mean([e['safe_success'] for e in episodes])) if observe_safety else None,
               'forbidden_contact_free_rate': float(np.mean([not e['forbidden_contact'] for e in episodes])) if observe_safety else None,
               'safety_measurement': 'adapter_cumulative_internal_steps' if observe_safety else 'unavailable',
               'plan_ms': 1000 * plan_seconds / max(plan_count, 1), 'plan_attempts': plan_count,
               'model_calls': model_calls, 'qp_calls': qp_calls, 'pf_calls': pf_calls, 'pf_ms': pf_ms,
               'hard_constraints_enforced': methods['projection']['enabled'],
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


def evaluate(cfg, *, sampling_seeds=None, context_ids=None,
             output_directory=None, experiment_metadata=None):
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
                paths['run_dir'], settings['checkpoint'], settings['weights'], v['device'])
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
        ood_info, spec = None, None
        if settings['ood'] is not None:
            # OOD 사례는 context·로봇 시작·노이즈·push·지연·정책 seed를 모두 파일에서 가져온다.
            cases, ood_info = load_cases(paths['ood_root'], cfg.task, settings['ood'])
            if settings['n_contexts']:
                cases = cases[:settings['n_contexts']]
            delays = {case['delay_steps'] for case in cases}
            if max(delays) + settings['execute_steps'] > v['model']['horizon']:
                raise ValueError('OOD delay_steps + execute_steps must not exceed the horizon')
            items = [{'id': case['context_id'], 'noise_key': id_context_key(cfg.task, case['context_id']),
                      'context': context_from_json(case['context']), 'ood': case} for case in cases]
            spec = get_task(cfg.task)
            spec.quat2euler = quat2euler
        if context_ids is not None:
            available = {item['id']: item for item in items}
            if len(set(context_ids)) != len(context_ids) or any(key not in available for key in context_ids):
                raise ValueError('Invalid or unavailable evaluation context IDs')
            items = [available[key] for key in context_ids]
        settings = {**settings}
        if sampling_seeds is not None:
            settings.update(sampling_seeds=list(sampling_seeds), n_traj=len(sampling_seeds))
        base = pushing.Policy(model, norm, model_cfg, conditioning, solver, v['sampling']['predict_steps'], v['device'])
        policy = Policy(base, cfg, prepare_potential(cfg))
        import torch
        torch.manual_seed(settings['seed'])
        np.random.seed(settings['seed'])
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        output = (cfg.require_output(output_directory) if output_directory is not None else
                  Path(paths['output_root']) / f'eval_{cfg.task}_{stamp}')
        if (output / 'results.json').exists():
            raise FileExistsError(f'Results already exist: {output}')
        cfg.save(output)
        checkpoint = Path(paths['run_dir']) / settings['checkpoint']
        metadata = {'task': cfg.task, 'checkpoint': str(checkpoint.resolve()),
                    'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    'checkpoint_step': checkpoint_step, 'weights': settings['weights'],
                    'seed': settings['seed'], 'predict_steps': v['sampling']['predict_steps'],
                    'potential_cache': policy.potential.cache_report if policy.potential else None,
                    'constraint_scope': 'command_plan', 'schema_version': 4,
                    'normalizer_sha256': hashlib.sha256((Path(paths['run_dir']) / 'normalizer.npz').read_bytes()).hexdigest(),
                    'control_dt_s': v['control']['dt_s'],
                    'execute_steps': settings['execute_steps'], 'max_steps': settings['max_steps']}
        (output / 'model_config.json').write_text((Path(paths['run_dir']) / 'config.json').read_text())
        results, rollouts = run_evaluation(env, policy, items, module.read_state, module.build_obs,
                                          quat2euler, cfg.task, settings, config=v,
                                          spec=spec)
        results.update(metadata)
        results['ood'] = ood_info
        if experiment_metadata:
            results.update(experiment_metadata)
        if context_ids is not None or sampling_seeds is not None:
            replay = {'context_ids': [item['id'] for item in items],
                      'sampling_seeds': sampling_seeds, 'methods': {k: m['enabled'] for k, m in v['methods'].items()}}
            (output / 'evaluation_protocol.json').write_text(json.dumps(replay, indent=2))
            (output / 'evaluation_contexts.pkl').write_bytes(pickle.dumps(items))
        (output / 'results.json').write_text(json.dumps(results, indent=2, allow_nan=False))
        (output / 'rollouts.json').write_text(json.dumps(rollouts, allow_nan=False))
        print(f'Saved: {output}')
        return results
    finally:
        os.chdir(old_cwd)
