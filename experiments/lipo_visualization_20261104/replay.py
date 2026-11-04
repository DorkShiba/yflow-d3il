"""Replay fixed c000_t0 commands; admit joint logs only if original states match."""
import os, sys, json, random, copy, hashlib
from pathlib import Path
import numpy as np
ROOT=Path('/opt/yflow-lipo-20261104')
sys.path.insert(0,str(ROOT/'repo/fm_baseline'))
task,mode=sys.argv[1:3]
module=__import__('eval_'+task)
source=(ROOT if task=='pushing' else Path('/opt/yflow-inserting-20261104'))/'main/combined'/mode
result=json.loads((source/'results.json').read_text())
z=np.load(source/'rollouts.npz'); key='c000_t0'
Env,quat=module.setup_d3il(str(ROOT/'d3il'))
env=Env(render=False,n_substeps=35,max_steps_per_episode=result['max_steps']);env.start()
if task=='pushing':
    context=np.load(ROOT/'d3il/environments/dataset/data/pushing/test_contexts.pkl',allow_pickle=True)[0]
else:
    cfg={'data_dir':str(ROOT/'repo/data/inserting_sincos_v1')}
    context=module.load_contexts(cfg,'val','',quat,1,str(ROOT/'repo/fm_baseline/contexts/inserting_val.pkl'))[0][2]
seed=int(np.random.SeedSequence([0,0,0]).generate_state(1)[0]);random.seed(seed);np.random.seed(seed);env.seed(seed)
env.reset(random=False,context=copy.deepcopy(context))
c,_=module.read_state(env,quat);fixed_z=c[2:3].copy()
joints=[];actual=[];blocks=[];xyz=[]
for des in z[key+'_des']:
    _,_,done,info=env.step(np.r_[des,fixed_z,[0.,1.,0.,0.]])
    c,boxes=module.read_state(env,quat)
    actual.append(c);blocks.append(np.concatenate([np.r_[b[0][:2],b[2]] for b in boxes]));xyz.append(np.concatenate([b[0] for b in boxes]));joints.append(np.asarray(env.robot.current_j_pos).copy())
err_c=float(np.max(np.abs(np.asarray(actual)-z[key+'_c'])))
err_b=float(np.max(np.abs(np.asarray(blocks,dtype=np.float32)-z[key+'_boxes'])))
assert err_c<1e-9 and err_b<1e-7,(err_c,err_b)
assert bool(info.get('success'))==bool(z['successes'][0,0])
out=Path('/opt/yflow-lipo-videos-20261104')/task/mode;out.mkdir(parents=True,exist_ok=True)
data={k:z[k] for k in z.files if k.startswith(key+'_')}
data.update({k:z[k][:1,:1] for k in ['successes','modes','lengths','mean_distance']})
data[key+'_joints']=np.asarray(joints);data[key+'_boxes_xyz']=np.asarray(xyz)
np.savez_compressed(out/'rollouts.npz',**data)
manifest=dict(task=task,mode=mode,key=key,selection='first context, first trajectory; fixed before viewing',source_sha256=hashlib.sha256((source/'rollouts.npz').read_bytes()).hexdigest(),max_hand_error_m=err_c,max_boxes_error=err_b,success=bool(info.get('success')),steps=len(joints),joint_source='deterministic command replay verified against original hand and block logs; no policy rerun',dt=.035)
(out/'verification.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest),flush=True)
