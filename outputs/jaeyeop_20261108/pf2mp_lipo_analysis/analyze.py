import json,pathlib,hashlib,collections,csv
import numpy as np
BASE=pathlib.Path('/root/d3il-work/yflow-d3il')
OUT=pathlib.Path('/root/d3il-work/pf2mp_lipo_analysis'); OUT.mkdir(exist_ok=True)
FULL=BASE/'fm_constraint/runs/experiment_id_v1/full'
allrows=[]; series={}; manifests=[]; comparisons=[]
def stats(a):
 return {'mean':float(np.mean(a)), 'p95':float(np.quantile(a,.95)), 'max':float(np.max(a))} if len(a) else None
for task,prefix in [('inserting','I'),('pushing','P')]:
 for ctx in (17,18,19):
  logs={}
  for num in (0,1,8,9):
   combo=f'{prefix}{num:02d}'; path=FULL/task/combo/f'ctx_{ctx:02d}'
   raw=(path/'rollouts.json').read_bytes(); logs[num]=json.loads(raw)
   manifests.append({'path':str(path/'rollouts.json'),'sha256':hashlib.sha256(raw).hexdigest()})
   result=json.loads((path/'results.json').read_text())
   for ep in result['episodes']:
    seed=ep['sampling_seed']; rows=logs[num][f'c000_t{seed}']; done=[r for r in rows if r.get('safe_action') is not None]
    arr={k:np.asarray([r[k] for r in done]) for k in ('command_state','measured_state','safe_action','next_command_state','next_measured_state','boxes_xy_m')}
    reports=[r['plan_report'] for r in rows if r.get('plan_report')]
    lp=[r for r in reports if 'lipo' in r]
    before=[]; after=[]; delta8=[]; corr=[]; drift=[]
    for r in lp:
     l=r['lipo']
     if l['status']!='ok': continue
     a=np.array(r['stages']['predict']['actions_m']); b=np.array(r['stages']['lipo']['actions_m']); d=b-a
     before.append(l['jerk_cost_before_m2']);after.append(l['jerk_cost_after_m2']);corr.append(l['correction_max_m'])
     delta8.append(d[:8].sum(0));drift.append(np.cumsum(d,axis=0).max(0))
    gap=np.linalg.norm(arr['next_command_state']-arr['next_measured_state'][:,:2],axis=1)
    row={'task':task,'ctx':ctx,'seed':seed,'combo':combo,**{k:ep[k] for k in ('success','steps','failure','mean_distance','n_boxes','action_change_m','command_jerk_m_s3','constraint_violation_steps_by_type')},'plans':len(reports),'plan_status':dict(collections.Counter(r['status'] for r in reports)),'gap_mm':stats(gap*1000),'final_gap_mm':float(gap[-1]*1000),'hand_last200_path_mm':float(np.linalg.norm(np.diff(arr['next_measured_state'][-200:,:2],axis=0),axis=1).sum()*1000),'command_last200_path_mm':float(np.linalg.norm(np.diff(arr['next_command_state'][-200:],axis=0),axis=1).sum()*1000),'box_last200_max_displacement_mm':float(np.linalg.norm(arr['boxes_xy_m'][-1]-arr['boxes_xy_m'][max(0,len(done)-200)],axis=1).max()*1000),'lipo_status':dict(collections.Counter(r['lipo']['status'] for r in lp)),'lipo_bound_mm':stats(np.array(corr)*1000),'plan_jerk_reduction_pct':float((1-sum(after)/sum(before))*100) if before else None,'executed_chunk_correction_sum_mm':(np.array(delta8).sum(0)*1000).tolist() if delta8 else None}
    allrows.append(row)
    series[(task,ctx,seed,num)]=arr
  for seed in (0,1,2):
   aa=logs[8][f'c000_t{seed}'];bb=logs[9][f'c000_t{seed}']; a=series[(task,ctx,seed,8)];b=series[(task,ctx,seed,9)];n=min(len(a['safe_action']),len(b['safe_action']))
   c={'task':task,'ctx':ctx,'seed':seed,'first_predict_max_abs_diff':float(np.abs(np.array(aa[0]['plan_report']['stages']['predict']['actions_m'])-np.array(bb[0]['plan_report']['stages']['predict']['actions_m'])).max()),'noise_seed_equal':aa[0]['plan_report']['noise_seed']==bb[0]['plan_report']['noise_seed']}
   for key in ('next_command_state','next_measured_state','boxes_xy_m'):
    d=np.linalg.norm(a[key][:n]-b[key][:n],axis=-1)
    if d.ndim>1:d=d.max(axis=1)
    c[key+'_first_diverge_step']={str(th):int(np.where(d>th)[0][0]) if np.any(d>th) else None for th in (.001,.01,.05,.1)}
    c[key+'_diff_mm_at_steps']={str(s):float(d[s]*1000) for s in (0,7,8,50,100,200,400,800) if s<n}
   comparisons.append(c)
(OUT/'metrics.json').write_text(json.dumps(allrows,indent=2));(OUT/'paired_comparisons.json').write_text(json.dumps(comparisons,indent=2));(OUT/'source_manifest.json').write_text(json.dumps(manifests,indent=2))
np.savez_compressed(OUT/'timeseries.npz',**{f'{t}_ctx{c}_s{s}_{n:02d}_{k}':v for (t,c,s,n),arr in series.items() for k,v in arr.items()})
for task in ('inserting','pushing'):
 print('\nTASK',task)
 for ctx in (17,18,19):
  for seed in (0,1,2):
   rr=[r for r in allrows if (r['task'],r['ctx'],r['seed'])==(task,ctx,seed)]
   print(ctx,seed,' | '.join(f"{r['combo']} succ={int(r['success'])} nbox={r['n_boxes']} steps={r['steps']} gap_max={r['gap_mm']['max']:.1f}mm gap_final={r['final_gap_mm']:.1f}mm jerk={r['command_jerk_m_s3']:.1f}" for r in rr))
print('output',OUT)
