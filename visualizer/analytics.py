"""Rollout diagnostics; unavailable signals stay null, not zero."""
import numpy as np

TASKS = {
    'pushing': {'goals': [[.42,.3,0],[.63,.3,0]], 'threshold': .05, 'rule': 'any_permutation',
                'description': '두 블록이 서로 다른 목표에 도착 · 색상 교차 허용', 'revision': '1d9c71850d4fc1477cee17b557c8b97d70b13071'},
    'inserting': {'goals': [[.3575,.276,0],[.525,.4535,0],[.6925,.276,0]], 'threshold': .01, 'rule': 'matched',
                  'description': '각 블록이 같은 색 지정 목표에 도착', 'revision': '1d9c71850d4fc1477cee17b557c8b97d70b13071'}
}

def enrich(r, task=None):
    n=r['steps'];dt=r['dt'];actual=np.asarray(r['c'])[:,:2];desired=np.asarray(r['des'])
    task_spec=TASKS.get(task)
    # Known evaluator jobs may use checked task geometry. Unknown imports get no invented goals.
    spec=r.get('goal_spec') or (dict(task_spec,source='D3IL task definition at recorded revision') if task_spec else None)
    r['goal_spec']=spec
    actions=np.asarray(r['act']) if r.get('act') is not None else None
    series=dict(tracking_mm=(np.linalg.norm(desired-actual,axis=1)*1000).tolist(),
                speed_m_s=[None]+(np.linalg.norm(np.diff(actual,axis=0),axis=1)/dt).tolist(),
                action_mm=(np.linalg.norm(actions,axis=1)*1000).tolist() if actions is not None else [None]*n,
                action_delta_mm=([None]+(np.linalg.norm(np.diff(actions,axis=0),axis=1)*1000).tolist()) if actions is not None else [None]*n)
    if r.get('joints') is not None:
        series['joint_speed_rad_s']=[None]+(np.linalg.norm(np.diff(np.asarray(r['joints']),axis=0),axis=1)/dt).tolist()
    else:series['joint_speed_rad_s']=[None]*n
    if spec:
        goals=np.asarray(spec['goals']);count=len(goals)
        xyz=r.get('boxes_xyz')
        boxes=np.asarray(xyz if xyz is not None else r['boxes']).reshape(n,count,3)
        dims=3 if xyz is not None else 2
        d=np.linalg.norm(boxes[:,:,None,:dims]-goals[None,None,:,:dims],axis=-1)
        if spec['rule']=='any_permutation':
            # Match unique goals by minimum worst distance, the same feasibility test as env success.
            direct=np.stack([d[:,0,0],d[:,1,1]],axis=1)
            cross=np.stack([d[:,0,1],d[:,1,0]],axis=1)
            choose=cross.max(1)<direct.max(1)
            selected=np.where(choose[:,None],cross,direct)
            assigned=np.where(choose[:,None],np.array([[1,0]]),np.array([[0,1]])).tolist()
            series['mean_distance_mm']=d.min(2).mean(1).tolist() # official pushing mean-distance definition
            series['mean_distance_mm']=(np.asarray(series['mean_distance_mm'])*1000).tolist()
        else:
            selected=np.stack([d[:,i,i] for i in range(count)],axis=1)
            assigned=[list(range(count)) for _ in range(n)]
            series['mean_distance_mm']=(selected.mean(1)*1000).tolist()
        series['goal_distances_mm']=(selected*1000).tolist()
        series['goal_assignment']=assigned
        series['reached_count']=(selected<=spec['threshold']).sum(1).tolist()
        r['distance_kind']='3D' if dims==3 else 'XY projection'
    else:
        series.update(mean_distance_mm=[None]*n,goal_distances_mm=None,goal_assignment=None,reached_count=None)
        r['distance_kind']=None
    r['series']=series
    def mean(values):
        values=[v for v in values if v is not None]
        return float(np.mean(values)) if values else None
    r['summary']=dict(duration_s=n*dt,success=r['success'],tracking_mean_mm=mean(series['tracking_mm']),
                      action_delta_mean_mm=mean(series['action_delta_mm']),
                      final_distance_mm=series['mean_distance_mm'][-1],path_length_m=float(np.linalg.norm(np.diff(actual,axis=0),axis=1).sum()))
    return r
