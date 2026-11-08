import sys,pathlib,json,numpy as np,torch
BASE=pathlib.Path('/root/d3il-work/yflow-d3il');sys.path.insert(0,str(BASE))
from configs.config import ExperimentConfig
from fm_constraint.pf2mp import prepare_potential
cfg=ExperimentConfig.from_yaml(BASE/'configs/inserting.yaml',{'methods.pf2mp.enabled':True,'device':'cuda:0'})
potential=prepare_potential(cfg)
allrows=[]
for ctx in (17,18,19):
 d=json.loads((BASE/f'fm_constraint/runs/experiment_id_v1/full/inserting/I09/ctx_{ctx}/rollouts.json').read_text())
 for seed in (0,1,2):
  rows=d[f'c000_t{seed}']; reps=[r['plan_report'] for r in rows if r.get('plan_report')];pre=[];post=[];costpre=[];costpost=[];sameenergy=[]
  for r in reps:
   a=np.array(r['stages']['predict']['actions_m']);b=np.array(r['stages']['lipo']['actions_m']);q=np.array(r['geometry']['q0_m'])
   pre.append(q+np.cumsum(a,axis=0));post.append(q+np.cumsum(b,axis=0));costpre.append(r['lipo']['jerk_cost_before_m2']);costpost.append(r['lipo']['jerk_cost_after_m2']);sameenergy.append(r['stages']['predict']['energy']==r['stages']['lipo']['energy'])
  pa=torch.tensor(np.array(pre),device='cuda:0');pb=torch.tensor(np.array(post),device='cuda:0')
  e0=-potential.evaluate(pa)[0].sum(-1).cpu().numpy();e1=-potential.evaluate(pb)[0].sum(-1).cpu().numpy();de=e1-e0
  a=np.array([r['safe_action'] for r in rows]);j=np.linalg.norm(np.diff(a,n=2,axis=0),axis=1)/.035**3;seam=np.array([r['seam'] for r in rows]);s=seam[2:]|seam[1:-1]
  allrows.append({'ctx':ctx,'seed':seed,'plans':len(reps),'higher_potential_fraction':float(np.mean(de>1e-8)),'potential_change_mean':float(de.mean()),'potential_change_max_abs':float(abs(de).max()),'potential_before_mean':float(e0.mean()),'all_logged_state_energy_unchanged':all(sameenergy),'jerk_cost_improved_fraction':float(np.mean(np.array(costpost)<=np.array(costpre)+1e-12)),'executed_jerk_seam':float(j[s].mean()),'executed_jerk_interior':float(j[~s].mean())})
print(json.dumps(allrows,indent=2));pathlib.Path('/root/d3il-work/pf2mp_lipo_analysis/potential_diagnostics.json').write_text(json.dumps(allrows,indent=2))
