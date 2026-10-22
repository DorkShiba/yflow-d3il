"""Shared rollout, with baseline policy/observation code imported unchanged."""

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
from .vlsa import vlsa_constraint, workspace_limits


def run_evaluation(env, policy, contexts, read_state, build_obs, quat2euler,
                   task, settings, vlsa=None, observe_safety=None):
    """Run prepared contexts. Safety observations must be supplied, never invented.

    observe_safety(env) must return cumulative episode forbidden_contact and
    min_distance, including internal physics steps and reset state. A simulator
    adapter is responsible for that instrumentation. Without it safety rates
    are null, even when there are no explicitly configured obstacles.
    """
    if not contexts:
        raise ValueError('No evaluation contexts')
    vlsa_enabled = vlsa is not None and vlsa['enabled']
    stop_comparison = settings.get('comparison') == 'stop'
    if vlsa_enabled and stop_comparison:
        raise ValueError('VLSA and its stop comparison are mutually exclusive')
    if stop_comparison and vlsa is None:
        raise ValueError('The stop comparison requires VLSA geometry and parameters')
    workspace = None
    if vlsa is not None and vlsa.get('geometry') == 'rectangle' and vlsa.get('workspace') is not None:
        workspace = workspace_limits(vlsa['workspace'], vlsa['margin_m'])
    episodes, rollouts = [], {}
    plan_seconds, n_plans = 0.0, 0
    for ci, item in enumerate(contexts):
        for ti in range(settings['n_traj']):
            env.reset(random=False, context=copy.deepcopy(item['context']))
            c, boxes = read_state(env, quat2euler)
            fixed_z, des = c[2:3].copy(), c[:2].copy()
            queue, logs = [], []
            info, done, failure, steps = {}, False, None, 0
            command_min_h, measured_min_h = None, None
            if workspace is not None:
                low, high = workspace
                command_min_h = float(np.min(np.r_[des - low, high - des]))
                measured_min_h = float(np.min(np.r_[c[:2] - low, high - c[:2]]))
            forbidden, min_distance = False, None
            if observe_safety:
                measured = observe_safety(env)
                forbidden = bool(measured['forbidden_contact'])
                min_distance = float(measured['min_distance'])
            while not done and steps < settings['max_steps']:
                obs = build_obs(des, c, boxes)
                if not queue:
                    started = perf_counter()
                    plan = policy.plan(obs)
                    plan_seconds += perf_counter() - started
                    n_plans += 1
                    if (plan.ndim != 2 or plan.shape[0] < settings['execute_steps']
                            or plan.shape[1] < 2 or not np.isfinite(plan).all()):
                        failure = 'invalid_policy_output'
                        break
                    queue = [plan[k, :2].astype(np.float64) for k in range(settings['execute_steps'])]
                nominal = queue.pop(0)
                record = {'step': steps, 'measured_state': c.tolist(), 'command_state': des.tolist(),
                          'nominal_action': nominal.tolist(), 'h': [], 'qp_status': 'disabled',
                          'qp_residual': None, 'filter_latency_ms': 0.0, 'intervention_norm': 0.0}
                if workspace is not None:
                    record['current_command_workspace_h'] = np.column_stack((des - low, high - des)).ravel().tolist()
                    record['current_measured_workspace_h'] = np.column_stack((c[:2] - low, high - c[:2])).ravel().tolist()
                action = nominal
                # VLSA acts on each physical command, after sampling and before execution.
                # Its own flag does not select or replace any other constraint method.
                if vlsa_enabled or stop_comparison:
                    filtered = vlsa_constraint(nominal, c[:2], des, vlsa)
                    record.update(h=filtered['h'], qp_status=filtered['status'],
                                  qp_residual=filtered['residual'],
                                  filter_latency_ms=filtered['latency_ms'],
                                  intervention_norm=filtered['intervention_norm'])
                    if 'constraint_names' in filtered:
                        record['constraint_names'] = filtered['constraint_names']
                    if filtered['action'] is None:
                        failure = filtered['status']
                        record['safe_action'] = None
                        logs.append(record)
                        break
                    action = filtered['action']
                    if stop_comparison and filtered['intervention_norm'] > 1e-8:
                        action = np.zeros(2)
                        record['qp_status'] = 'stop_comparison'
                        record['intervention_norm'] = float(np.linalg.norm(nominal))
                des = des + action
                command = np.concatenate([des, fixed_z, [0.0, 1.0, 0.0, 0.0]])
                _, _, done, info = env.step(command)
                c, boxes = read_state(env, quat2euler)
                steps += 1
                record.update(safe_action=action.tolist(), next_command_state=des.tolist(),
                              next_measured_state=c.tolist(),
                              boxes=np.concatenate([np.r_[bx[0][:2], bx[2]] for bx in boxes]).tolist())
                if workspace is not None:
                    command_h = np.column_stack((des - low, high - des)).ravel()
                    measured_h = np.column_stack((c[:2] - low, high - c[:2])).ravel()
                    command_min_h = min(command_min_h, float(command_h.min()))
                    measured_min_h = min(measured_min_h, float(measured_h.min()))
                    record.update(command_workspace_h=command_h.tolist(), measured_workspace_h=measured_h.tolist())
                if observe_safety:
                    measured = observe_safety(env)
                    forbidden |= bool(measured['forbidden_contact'])
                    distance = float(measured['min_distance'])
                    min_distance = min(min_distance, distance)
                    record.update(forbidden_contact=bool(measured['forbidden_contact']), min_distance=distance)
                logs.append(record)
            if not done and failure is None:
                failure = 'max_steps'
            success = bool(info.get('success', False)) and failure is None
            distance = float(info.get('mean_distance', float('nan')))
            episode = {'context_id': item['id'], 'context_index': ci, 'trajectory': ti,
                       'success': success, 'mode': int(info.get('mode', -1 if task == 'pushing' else 0)),
                       'mean_distance': distance if np.isfinite(distance) else None,
                       'steps': steps, 'failure': failure,
                       'forbidden_contact': forbidden if observe_safety else None,
                       'safe_success': (success and not forbidden) if observe_safety else None,
                       'min_distance': min_distance,
                       'command_workspace_min_h': command_min_h,
                       'measured_workspace_min_h': measured_min_h,
                       'command_workspace_violation': command_min_h < -1e-9 if workspace is not None else None,
                       'measured_workspace_violation': measured_min_h < -1e-9 if workspace is not None else None,
                       'n_boxes': sum(int(info.get(k, 0)) for k in
                                      ('one_box_success', 'two_box_success', 'three_box_success')),
                       'demo_length': item.get('demo_length')}
            episodes.append(episode)
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
    results = {
        'n_contexts': len(contexts), 'n_traj': settings['n_traj'],
        'vlsa_enabled': bool(vlsa_enabled),
        'success_rate': success_rate, 'entropy': entropy,
        'score': None if entropy is None else 0.5 * (success_rate + entropy),
        'mean_distance': float(np.mean(distances)) if distances else None,
        'mean_steps': float(np.mean([e['steps'] for e in episodes])),
        'mode_counts': {str(m): sum(e['success'] and e['mode'] == m for e in episodes) for m in mode_values},
        'safe_success_rate': float(np.mean([e['safe_success'] for e in episodes])) if observe_safety else None,
        'forbidden_contact_free_rate': float(np.mean([not e['forbidden_contact'] for e in episodes])) if observe_safety else None,
        'safety_measurement': 'adapter_cumulative_internal_steps' if observe_safety else 'unavailable',
        'plan_ms': 1000 * plan_seconds / max(n_plans, 1),
        'episodes': episodes,
    }
    results.update(
        command_workspace_violation_rate=float(np.mean([e['command_workspace_violation'] for e in episodes])) if workspace is not None else None,
        measured_workspace_violation_rate=float(np.mean([e['measured_workspace_violation'] for e in episodes])) if workspace is not None else None,
        command_workspace_success_rate=float(np.mean([e['success'] and not e['command_workspace_violation'] for e in episodes])) if workspace is not None else None,
        workspace_measurement='reset_and_control_steps' if workspace is not None else 'unavailable',
    )
    if task == 'inserting':
        for n, word in [(1, 'one'), (2, 'two'), (3, 'three')]:
            results[f'{word}_box_rate'] = float(np.mean([e['n_boxes'] >= n for e in episodes]))
        successful = [e['steps'] for e in episodes if e['success']]
        results['mean_steps_success'] = float(np.mean(successful)) if successful else None
        lengths = [i['demo_length'] for i in contexts if i.get('demo_length') is not None]
        results['mean_demo_len'] = float(np.mean(lengths)) if lengths else None
    return results, rollouts


def evaluate(cfg):
    """Run the command fence; physical obstacle CBF evaluation still needs an audit."""
    cfg.validate('eval')
    v, paths, settings = cfg.values, cfg.values['paths'], cfg.values['eval']
    # Do not silently run imagined obstacles or unvalidated velocity constraints.
    if v['vlsa']['obstacles'] or ((cfg.vlsa_enabled or settings['comparison'] != 'none') and v['vlsa']['geometry'] != 'rectangle'):
        raise NotImplementedError(
            'Physical obstacle/CBF evaluation requires the D3IL geometry, tracking-model '
            'and internal-step audit (PLAN stages 4-6). Numerical vlsa_constraint is available; '
            'no unvalidated safety experiment is executed.')
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
        policy = pushing.Policy(model, norm, model_cfg, conditioning, solver, settings['ode_steps'], v['device'])
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
                    'seed': settings['seed'], 'ode_steps': settings['ode_steps'],
                    'constraint_scope': 'command_workspace' if v['vlsa']['geometry'] == 'rectangle' else 'none',
                    'execute_steps': settings['execute_steps'], 'max_steps': settings['max_steps']}
        (output / 'model_config.json').write_text((Path(paths['run_dir']) / 'config.json').read_text())
        results, rollouts = run_evaluation(env, policy, items, module.read_state, module.build_obs,
                                          quat2euler, cfg.task, settings, vlsa=v['vlsa'])
        results.update(metadata)
        (output / 'results.json').write_text(json.dumps(results, indent=2, allow_nan=False))
        (output / 'rollouts.json').write_text(json.dumps(rollouts, allow_nan=False))
        print(f'Saved: {output}')
        return results
    finally:
        os.chdir(old_cwd)
