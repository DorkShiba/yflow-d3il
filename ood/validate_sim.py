"""MuJoCo initialization screening. Does not prove task solvability/controller readiness.
Runs real D3IL scenes. Robot qpos is held kinematically during object settling.
Results never set rollout_ready true. Candidates are immutable.
"""
import argparse, collections, hashlib, importlib, json, math, os, sys, time
from pathlib import Path
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def make_env(task):
    name='sorting' if task.startswith('sorting') else task
    cls={'sorting':'Sorting_Env','stacking':'CubeStacking_Env','avoiding':'ObstacleAvoidanceEnv'}[name]
    mod=importlib.import_module(f'environments.d3il.envs.gym_{name}_env.gym_{name}.envs.{name}')
    kw={'render':False}
    if name=='sorting':kw['num_boxes']=int(task[-1])
    e=getattr(mod,cls)(**kw);e.start();e.reset();return e


def screen(task,rows,output):
    e=make_env(task);m,d=e.scene.model,e.scene.data
    q0=d.qpos.copy();arm=np.array(e.robot.joint_indices);jid=np.array([m.joint(n).id for n in e.robot.joint_names]);dofs=m.jnt_dofadr[jid]
    tcp=m.body(e.robot.add_id2model_key('tcp')).id
    mujoco.mj_forward(m,d);p0=d.xpos[tcp].copy();R0=Rotation.from_matrix(d.xmat[tcp].reshape(3,3))
    lower,upper=m.jnt_range[jid].T
    names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(b)) or '' for b in m.geom_bodyid]
    robot_geoms={i for i,n in enumerate(names) if '_rb' in n}
    blocks={b['name'] for r in rows for b in r['blocks']}
    block_geoms={i for i,n in enumerate(names) if n in blocks}
    support_geoms={i for i,n in enumerate(names) if n in ['ground','table_plane','platform']}
    # Existing self contact of a closed gripper is measured, not silently generalized.
    baseline_self={tuple(sorted((int(c.geom1),int(c.geom2)))) for c in d.contact if int(c.geom1) in robot_geoms and int(c.geom2) in robot_geoms}
    metadata={'task':task,'timestep':m.opt.timestep,'canonical_tcp':p0.tolist(),'canonical_arm_qpos':q0[arm].tolist(),'baseline_self_contact_pairs':[list(p) for p in sorted(baseline_self)],'geom_sizes':{names[i]:m.geom_size[i].tolist() for i in block_geoms},'settle_seconds':.6,'settling_robot':'kinematically fixed; controller integration NOT tested','thresholds':{'xy_drift_m':.002,'linear_speed_m_s':.01,'angular_speed_rad_s':.1,'penetration_m':.001,'ik_position_m':.001,'ik_angle_rad':.01},'source_reset_z':'preserved before dynamic settling; transient block-support_body penetration allowed only before settling'}
    (output/(task+'_scene.json')).write_text(json.dumps(metadata,indent=2))
    cache={};ikcache={};counts=collections.Counter();started=time.time()
    f=(output/(task+'_results.jsonl')).open('w')
    def ik(start):
        key=json.dumps(start,sort_keys=True)
        if key in ikcache:return ikcache[key]
        target=p0+np.array(start['xy_offset_m']+[0.]);rot=Rotation.from_rotvec([0,0,start['yaw_offset_rad']])*R0
        d.qpos[:]=q0
        def residual(q):
            d.qpos[arm]=q;mujoco.mj_forward(m,d)
            return np.r_[d.xpos[tcp]-target,.2*(rot.inv()*Rotation.from_matrix(d.xmat[tcp].reshape(3,3))).as_rotvec(),1e-5*(q-q0[arm])]
        if np.linalg.norm(target-p0)+abs(start['yaw_offset_rad'])<1e-12:q=q0[arm].copy()
        else:q=least_squares(residual,np.clip(q0[arm],lower+1e-6,upper-1e-6),bounds=(lower+1e-6,upper-1e-6),max_nfev=150,gtol=1e-9,ftol=1e-9,xtol=1e-9).x
        err=residual(q);ans={'q':q.tolist(),'position_error_m':float(np.linalg.norm(err[:3])),'angle_error_rad':float(np.linalg.norm(err[3:6])/.2)};ikcache[key]=ans;return ans
    for row in rows:
        key=json.dumps([row['blocks'],row['robot_start']],sort_keys=True)
        if key not in cache:
            sol=ik(row['robot_start']);qarm=np.array(sol['q']);fail=[]
            if sol['position_error_m']>.001 or sol['angle_error_rad']>.01:fail.append('ik_residual')
            mujoco.mj_resetData(m,d);d.qpos[:]=q0;d.qpos[arm]=qarm;d.qvel[:]=0
            for b in row['blocks']:
                body=m.body(b['name']);j=int(m.body_jntadr[body.id]);adr=int(m.jnt_qposadr[j]);d.qpos[adr:adr+2]=b['xy']
                a=b['yaw_rad'];d.qpos[adr+3:adr+7]=[math.cos(a/2),0,0,math.sin(a/2)]
            mujoco.mj_forward(m,d)
            # Preserve reset z; this also exposes source reset penetration/settling drift.
            bad_initial=[]
            def bad_contacts(initial=False):
                bad=[]
                for c in d.contact:
                    a,b=int(c.geom1),int(c.geom2);pair=tuple(sorted((a,b)))
                    if c.dist>=-.001:continue
                    if initial and ((a in block_geoms and names[b]=='support_body') or (b in block_geoms and names[a]=='support_body')):continue
                    if (a in block_geoms and b in support_geoms) or (b in block_geoms and a in support_geoms):continue
                    if a in robot_geoms and b in robot_geoms and pair in baseline_self:continue
                    if a in robot_geoms or b in robot_geoms or a in block_geoms or b in block_geoms:bad.append([a,b,float(c.dist),names[a],names[b]])
                return bad
            bad_initial=bad_contacts(initial=True)
            if bad_initial:fail.append('initial_collision')
            # Fixed-arm settling, not policy execution or controller tracking.
            fixed=np.array([i for i in range(m.nq) if not any(int(m.jnt_qposadr[int(m.body_jntadr[m.body(b['name']).id])])<=i<int(m.jnt_qposadr[int(m.body_jntadr[m.body(b['name']).id])])+7 for b in row['blocks'])])
            fixedq=d.qpos[fixed].copy()
            for _ in range(round(.6/m.opt.timestep)):
                d.qpos[fixed]=fixedq
                # Freeze all non-object dofs; free objects are the first six-dof joints.
                for j in range(m.njnt):
                    if m.jnt_type[j]!=mujoco.mjtJoint.mjJNT_FREE:d.qvel[m.jnt_dofadr[j]]=0
                d.ctrl[:]=0;mujoco.mj_step(m,d)
            d.qpos[fixed]=fixedq;mujoco.mj_forward(m,d)
            bad_final=bad_contacts()
            if bad_final:fail.append('settled_collision')
            states=[]
            for b in row['blocks']:
                body=m.body(b['name']);j=int(m.body_jntadr[body.id]);v=int(m.jnt_dofadr[j]);p=d.xpos[body.id].copy();q=d.xquat[body.id].copy();drift=float(np.linalg.norm(p[:2]-b['xy']));speed=float(np.linalg.norm(d.qvel[v:v+3]));omega=float(np.linalg.norm(d.qvel[v+3:v+6]));yaw=Rotation.from_quat(q[[1,2,3,0]]).as_euler('xyz')[2];angle=abs(math.atan2(math.sin(yaw-b['yaw_rad']),math.cos(yaw-b['yaw_rad'])))
                if drift>.002:fail.append('xy_drift')
                if speed>.01 or omega>.1:fail.append('unstable')
                if angle>.02:fail.append('yaw_drift')
                # Require actual supporting contacts; no reliance on an infinite-plane XY box.
                geoms={i for i in block_geoms if names[i]==b['name']}
                supported=any((int(c.geom1) in geoms and int(c.geom2) in support_geoms) or (int(c.geom2) in geoms and int(c.geom1) in support_geoms) for c in d.contact)
                if not supported:fail.append('unsupported')
                states.append({'name':b['name'],'xyz':p.tolist(),'quat_wxyz':q.tolist(),'xy_drift_m':drift,'linear_speed':speed,'angular_speed':omega,'yaw_drift_rad':angle,'supported':supported})
            if not np.isfinite(d.qpos).all():fail.append('nonfinite')
            cache[key]={'initialization_screen':'pass' if not fail else 'fail','reasons':sorted(set(fail)),'ik':sol,'blocks_after_settle':states,'initial_bad_contacts':bad_initial,'settled_bad_contacts':bad_final,'rollout_ready':False,'not_checked':['controller reset integration','workspace and object reachability and goal path feasibility','empirical training distribution']}
        result={'id':row['id'],'family':row['perturbation']['family'],'level':row['perturbation']['level'],**cache[key]}
        counts[result['initialization_screen']]+=1;f.write(json.dumps(result,allow_nan=False)+'\n')
        if sum(counts.values())%100==0:f.flush();print(task,dict(counts),'seconds',round(time.time()-started,1),flush=True)
    f.close();print(task,'DONE',dict(counts),flush=True)
    (output/(task+'_complete.json')).write_text(json.dumps({'counts':dict(counts),'cases':len(rows),'unique_physical_resets':len(cache),'elapsed_s':time.time()-started},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--d3il',required=True);p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--task',required=True);p.add_argument('--limit',type=int,default=0);a=p.parse_args()
    root=Path(a.d3il).resolve();inp=Path(a.input).resolve();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(root));os.chdir(root)
    rows=[json.loads(x) for x in (inp/(a.task+'.jsonl')).read_text().splitlines()]
    if a.limit:rows=rows[:a.limit]
    screen(a.task,rows,out)
