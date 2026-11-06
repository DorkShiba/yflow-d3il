"""Paired context-bootstrap comparison of baseline and LiPo evaluations."""
import argparse
import json
from pathlib import Path
import numpy as np

MATCH_FIELDS = ('metrics_version', 'context_start', 'task', 'checkpoint_sha256', 'normalizer_sha256',
                'config_sha256', 'context_source_sha256', 'context_hashes', 'hardflow_commit',
                'd3il_commit', 'weights', 'n_contexts', 'n_traj', 'ode_steps',
                'execute_steps', 'max_steps', 'seed', 'control_dt_s', 'rng_scheme',
                'source_sha256')
METRICS = ('desired_jerk_rms_m_s3', 'actual_jerk_rms_m_s3',
           'seam_velocity_jump_rms_m_s', 'desired_seam_jerk_rms_m_s3',
           'actual_seam_jerk_rms_m_s3', 'tracking_gap_mean_m',
           'desired_path_length_m', 'actual_path_length_m', 'duration_s')


def load_run(path):
    path = Path(path)
    result = json.loads((path / 'results.json').read_text())
    detail = json.loads((path / 'smoothness.json').read_text())
    with np.load(path / 'rollouts.npz', allow_pickle=False) as z:
        success = z['successes'].copy()
    return result, detail['episodes'], success


def compare(baseline, candidate, n_boot=5000, seed=0):
    a, ar, ass = load_run(baseline)
    b, br, bss = load_run(candidate)
    if n_boot < 1:
        raise ValueError('n_boot must be positive')
    if a.get('postprocess') != 'none':
        raise ValueError('The baseline must use --postprocess none')
    if not a.get('paired_noise') or not b.get('paired_noise'):
        raise ValueError('Both runs must use --paired-noise')
    for key in MATCH_FIELDS:
        if key not in a or key not in b or a[key] != b[key]:
            raise ValueError(f'Incompatible paired evaluations: {key}')
    if a.get('replan_noise', 'fresh') != b.get('replan_noise', 'fresh'):
        raise ValueError('Incompatible paired evaluations: replan_noise')
    for key in ('inference_delay_steps', 'execution_protocol', 'bootstrap_protocol'):
        if a.get(key) != b.get(key):
            raise ValueError(f'Incompatible paired evaluations: {key}')
    nctx, ntraj = a['n_contexts'], a['n_traj']
    if ass.shape != (nctx, ntraj) or bss.shape != ass.shape:
        raise ValueError('Success array shape mismatch')
    keys = [f'c{ci:03d}_t{ti}' for ci in range(nctx) for ti in range(ntraj)]
    if set(keys) != set(ar) or set(keys) != set(br):
        raise ValueError('Missing/unpaired episodes')
    rng = np.random.default_rng(seed)
    result = {'baseline': str(baseline), 'candidate': str(candidate),
              'candidate_mode': b['postprocess'], 'candidate_rtc': b.get('rtc', False), 'n_boot': n_boot,
              'aggregation': 'equal episodes within context; paired bootstrap over contexts',
              'metrics': {}, 'timing': {label: {k: r.get(k) for k in ('plan_ms', 'qp_ms', 'postprocess_ms', 'postprocess_max_ms')}
                                       for label, r in [('baseline', a), ('candidate', b)]}}
    for metric in ('success_rate', *METRICS):
        if metric == 'success_rate':
            av, bv = ass, bss
        else:
            av, bv = [np.array([rows[k][metric] if rows[k][metric] is not None else np.nan for k in keys]).reshape(nctx, ntraj)
                      for rows in (ar, br)]
        # Only paired available episodes contribute, never turn short-run nulls into zeros.
        valid = np.isfinite(av) & np.isfinite(bv)
        counts = valid.sum(axis=1)
        present = counts > 0
        ac = np.where(valid, av, 0).sum(axis=1)[present] / counts[present]
        bc = np.where(valid, bv, 0).sum(axis=1)[present] / counts[present]
        if not len(ac):
            result['metrics'][metric] = {'paired_episodes': 0, 'delta': None}
            continue
        diff = bc - ac
        boot = diff[rng.integers(0, len(diff), size=(n_boot, len(diff)))].mean(axis=1)
        am, bm = float(ac.mean()), float(bc.mean())
        result['metrics'][metric] = {'baseline': am, 'candidate': bm, 'delta': bm - am,
                                     'relative_change_pct': 100 * (bm / am - 1) if am else None,
                                     'delta_ci95': np.quantile(boot, [0.025, 0.975]).tolist() if len(diff) > 1 else None,
                                     'paired_contexts': len(diff), 'paired_episodes': int(valid.sum())}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', required=True)
    p.add_argument('--candidate', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--n-boot', type=int, default=5000)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    result = compare(args.baseline, args.candidate, args.n_boot, args.seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    for name, row in result['metrics'].items():
        print(name, json.dumps(row))


if __name__ == '__main__':
    main()
