import ast,json,sys,time,hashlib,shutil
import numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parent
VIEWER=ROOT.parents[1]
sys.path.insert(0,str(VIEWER))
from visualizer.media import read_rollouts
store=ROOT/'viewer_store';store.mkdir(exist_ok=True)
walls=[]
for sample in (VIEWER/'visualizer/samples').glob('*/rollouts.json'):
 records=json.loads(sample.read_text())
 if records and records[0].get('walls'):
  walls=records[0]['walls'];break
assert len(walls)==17, 'Bundled verified Inserting wall metadata missing'
dashboard=dict(default_task='pushing',tasks={})
for task in ['pushing','inserting']:
 if not (ROOT/task/'lipo/rollouts.npz').exists():continue
 panels=[];res=[]
 for mode,label in [('none','Baseline'),('blend','Blending'),('lipo','LiPo')]:
  folder=ROOT/task/mode
  verify=json.loads((folder/'verification.json').read_text())
  id=hashlib.md5(('lipo20261104'+task+mode).encode()).hexdigest();dest=store/id;dest.mkdir(exist_ok=True)
  rs=read_rollouts(folder/'rollouts.npz',.035)
  for r in rs:r.update(joint_source=verify['joint_source'],walls=walls if task=='inserting' else [])
  with np.load(folder/'rollouts.npz') as z:
   for r in rs:
    pos=np.vstack([z[r['key']+'_initial_c'],np.asarray(r['c'])[:,:2]])
    jerk=np.linalg.norm(np.diff(pos,n=3,axis=0)/r['dt']**3,axis=1)
    r['jerk_xy']=[None,None]+jerk.tolist()
    a=np.asarray(r['act']);dv=np.linalg.norm(np.diff(a,axis=0),axis=1)/r['dt']
    r['seam_dv']=[None]+[float(v) if r['plan_starts'][i+1] else None for i,v in enumerate(dv)]
    r['jerk_rms']=float(np.sqrt(np.mean(jerk**2)))
  (dest/'rollouts.json').write_text(json.dumps(rs));shutil.copy2(folder/'rollouts.npz',dest/'rollouts.npz')
  job=dict(id=id,kind='import',status='completed',created=time.time(),label=label,spec=dict(task=task,dt=.035,label=label,verification=verify),robot=False,rollouts=['c000_t0'])
  (dest/'job.json').write_text(json.dumps(job));panels.append(dict(job=id,key='c000_t0',label=label,view='scene'))
  res.append(json.loads((ROOT.parent/f'lipo_{task}_20261104/artifacts/main/combined'/mode/'results.json').read_text()))
 rows=[['성공률',*[f"{r['success_rate']*100:.2f}%" for r in res]],['손 jerk RMS (m/s³)',*[f"{r['smoothness']['actual_jerk_rms_m_s3']['mean']:.2f}" for r in res]],['계산 시간 (ms/chunk)',*[f"{r['plan_ms']+r['postprocess_ms']:.2f}" for r in res]]]
 rows.insert(2,['청크 경계 명령 속도 변화 RMS (m/s)',*[f"{r['smoothness']['seam_velocity_jump_rms_m_s']['mean']:.4f}" for r in res]])
 dashboard['tasks'][task]=dict(title=f'{task.title()} · Baseline / Blending / LiPo',panels=panels,note='동일 context 0 / trajectory 0 · 사전 고정한 첫 사례 · 원본 손·블록 좌표와 정확히 일치한 명령 replay에서 관절 기록 · 한 장면은 전체 성능을 대표하지 않습니다. 벽은 D3IL 원본 배치, 로봇·작업대는 웹 표시용 모델입니다.',metric_rows=rows,metric_note=f"전체 {res[0]['n_contexts']} contexts × 4회 / 방법 · 시뮬레이터 XY · 지연 0 · 계산 시간은 병렬 평가 환경")
(store/'dashboard.json').write_text(json.dumps(dashboard,ensure_ascii=False,indent=2))
