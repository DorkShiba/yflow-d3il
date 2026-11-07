"""Render the frozen three-arm evaluation; never selects parameters."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    modes = ['none', 'blend', 'lipo']
    labels = ['Baseline', 'Blend only', 'LiPo']
    results, episodes, success, plan_diagnostics = {}, {}, {}, {}
    for mode in modes:
        folder = args.input/'combined'/mode
        results[mode] = json.loads((folder/'results.json').read_text())
        detail = json.loads((folder/'smoothness.json').read_text())
        episodes[mode] = detail['episodes']
        flat = [p for rows in detail['plans'].values() for p in rows]
        plan_diagnostics[mode] = {'plans':len(flat), 'solved':sum(p['solver_status']=='Solved' for p in flat),
                                  'max_qp_correction_m':max(p['qp_correction_max_m'] for p in flat),
                                  'max_raw_deviation_m':max(p['raw_deviation_max_m'] for p in flat),
                                  'postprocess_median_ms':float(np.median([p['postprocess_ms'] for p in flat])),
                                  'postprocess_p95_ms':float(np.quantile([p['postprocess_ms'] for p in flat], .95))}
        with np.load(folder/'rollouts.npz', allow_pickle=False) as z:
            success[mode] = z['successes'].copy()
    comparisons = {mode:json.loads((args.input/f'comparison_{mode}.json').read_text()) for mode in modes[1:]}
    baseline = results['none']
    nctx, ntraj = baseline['n_contexts'], baseline['n_traj']
    metrics = [('success_rate', 'Success rate', '%', 100),
               ('desired_jerk_rms_m_s3', 'Desired-position jerk RMS', 'm/s³', 1),
               ('actual_jerk_rms_m_s3', 'Actual-hand jerk RMS', 'm/s³', 1),
               ('seam_velocity_jump_rms_m_s', 'Chunk-seam velocity change RMS', 'm/s', 1)]
    rng = np.random.default_rng(0)
    bootstrap = rng.integers(0,nctx,size=(5000,nctx))
    summary = {}
    for key, title, unit, scale in metrics:
        summary[key] = {}
        for mode in modes:
            if key == 'success_rate': arr = success[mode]
            else: arr = np.array([[episodes[mode][f'c{ci:03d}_t{ti}'][key] for ti in range(ntraj)] for ci in range(nctx)])
            per_context = arr.mean(axis=1)
            lo, hi = np.quantile(per_context[bootstrap].mean(axis=1), [.025,.975])
            summary[key][mode] = {'mean':float(per_context.mean()), 'ci95':[float(lo),float(hi)]}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(2,2,figsize=(10,7))
    colors = ['#697386','#38938a','#3366cc']
    for ax,(key,title,unit,scale) in zip(axes.flat,metrics):
        for i,mode in enumerate(modes):
            row = summary[key][mode]; val = row['mean']*scale
            lo,hi = np.array(row['ci95'])*scale
            ax.errorbar(i,val,yerr=[[val-lo],[hi-val]],fmt='o',color=colors[i],capsize=5,markersize=8)
            ax.annotate(f'{val:.2f}' if key!='seam_velocity_jump_rms_m_s' else f'{val:.3f}', (i,val), xytext=(12,0),textcoords='offset points',va='center',fontsize=10)
        ax.set_xticks(range(3),labels); ax.set_xlim(-.4,2.7)
        ax.set_title(title,loc='left',fontweight='bold'); ax.set_ylabel(unit)
        ax.grid(axis='y',alpha=.2); ax.set_ylim(bottom=0)
        if key=='success_rate': ax.set_ylim(0,105)
    fig.suptitle(f'Inserting: {nctx} contexts × {ntraj} trajectories per method',fontsize=16,fontweight='bold',x=.07,ha='left')
    fig.text(.07,.025,'Fixed EMA policy · 95% context-bootstrap CI · zero-delay discrete XY LiPo',fontsize=10,color='#555555')
    fig.tight_layout(rect=[0,.045,1,.95]);fig.savefig(args.output/'comparison.png',dpi=180);plt.close(fig)
    selected = ['desired_jerk_rms_m_s3','actual_jerk_rms_m_s3','seam_velocity_jump_rms_m_s','actual_seam_jerk_rms_m_s3','tracking_gap_mean_m','duration_s']
    lines=['# LiPo Inserting 비교 결과','', '2026-11-04 · RTX 3090 · 동일 EMA 체크포인트 · 검증 분할 초기 배치 평가','',
           f'고정 설정: context {nctx}개 × {ntraj}회, seed 0, 16스텝 생성·8스텝 실행, Euler 10스텝, paired noise.',
           'LiPo: blend 4스텝, blend bound 2 mm, path bound 0.3 mm. 평가 결과에 따른 파라미터 조정 없음.','',
           '| 지표 | Baseline | Blend only | LiPo |','|---|---:|---:|---:|']
    counts=[f'{int(success[m].sum())}/{success[m].size} ({100*success[m].mean():.2f}%)' for m in modes]
    lines.append('| 성공률 | '+' | '.join(counts)+' |')
    for key in selected:
        values=[results[m]['smoothness'][key]['mean'] for m in modes]
        lines.append(f'| {key} | '+' | '.join(f'{v:.6g}' for v in values)+' |')
    for key in ('plan_ms','postprocess_ms','qp_ms'):
        values=[results[m].get(key) for m in modes]
        lines.append(f'| {key} | '+' | '.join('—' if v is None else f'{v:.3f}' for v in values)+' |')
    lines += ['', '## LiPo − Baseline: paired context-bootstrap 95% CI','',
              '| 지표 | 평균 차이 | 95% CI | 상대 변화 |','|---|---:|---:|---:|']
    for key in ['success_rate',*selected]:
        row=comparisons['lipo']['metrics'][key]
        scale=100 if key=='success_rate' else 1
        unit=' %p' if key=='success_rate' else ''
        lo,hi=row['delta_ci95']
        lines.append(f'| {key} | {row["delta"]*scale:.5g}{unit} | [{lo*scale:.5g}, {hi*scale:.5g}]{unit} | {row["relative_change_pct"]:.2f}% |')
    lines += ['', '## 해석 범위','',
              '- 실제 손 지표는 D3IL 시뮬레이터의 XY 위치를 35 ms마다 샘플링한 유한차분이다. 물리 로봇이나 substep 고주파 진동의 개선을 입증하지 않는다.',
              '- 이 결과는 검증 시연 82개 초기 배치의 비교다. OOD·지연·다른 태스크는 평가하지 않았다.',
              '- 성공률 CI가 0을 포함하면 동일 성능 또는 비열등성을 입증한 것으로 해석하지 않는다.',
              '- 전체 episode smoothness에는 성공 여부·길이·경로 차이가 영향을 줄 수 있다. 경계 지표, 이동량, 성공률을 함께 본다.',
              '- 시간은 14개 평가 프로세스가 같은 GPU를 공유한 조건이다. 단독 배포 지연 벤치마크가 아니다. QP 시간은 후처리 시간에 포함된다.',
              '- 분할 실행은 global context index로 난수를 고정하고, context별 raw rollout과 QP 진단을 합쳐 분석했다.',
              '', '## 근거 파일','',
              '- `protocol.json`: 사전 고정 설정과 소스·체크포인트 해시',
              '- `comparison.png`: context-bootstrap CI를 포함한 비교 그림',
              '- `summary.json`: 핵심 수치와 paired 비교 결과',
              '- `artifacts/main/combined/{none,blend,lipo}/`: 원본 궤적·episode/plan별 지표',
              '- `artifacts/main/commands.json`: 모든 shard 실행 명령',
              '- `artifacts/main/COMPLETE.json`: 완료 시각·실행 시간',
              '- `artifacts/environment.txt`, `artifacts/source.tar.gz`: 환경과 실행 소스 스냅샷','']
    (args.output/'REPORT_KO.md').write_text('\n'.join(lines))
    payload={'summary':summary,'comparisons':comparisons,'success_counts':{m:int(success[m].sum()) for m in modes},
             'complete':json.loads((args.input/'COMPLETE.json').read_text()),
             'checkpoint_sha256':baseline['checkpoint_sha256'],'source_sha256':baseline['source_sha256'],
             'plan_diagnostics':plan_diagnostics}
    (args.output/'summary.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    hashes={str(p.relative_to(args.input)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((args.input/'combined').rglob('*')) if p.is_file()}
    (args.output/'artifact_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(json.dumps({'success_counts':payload['success_counts'],'lipo':comparisons['lipo']['metrics']},indent=2))


if __name__=='__main__': main()
