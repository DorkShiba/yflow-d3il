"""Generate reproducible D3IL evaluation candidates, not expert demonstrations.
Only analytical support and conservative block separation are checked here.
Every case remains blocked from rollout until simulator/IK validation.
"""
import argparse
import hashlib
import json
import math
import random
from pathlib import Path

COMMIT = '1d9c71850d4fc1477cee17b557c8b97d70b13071'
SORT_REGIONS = [(x, x+.1, y, y+.05) for x in (.4, .55) for y in (-.15, -.05, .05)]
STACK_REGIONS = [(.35,.45,-.25,-.15),(.35,.45,-.1,0),(.55,.6,-.2,0)]
TASKS = ['sorting2','sorting4','sorting6','stacking','avoiding']

def seed_for(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str,parts)).encode()).digest()[:8], 'big')

def distance_to_region(x,y,r):
    return math.hypot(max(r[0]-x,0,x-r[1]),max(r[2]-y,0,y-r[3]))

def distance_to_support(x,y,regions):
    return min(distance_to_region(x,y,r) for r in regions)

def valid_separation(blocks):
    return all(math.hypot(a['xy'][0]-b['xy'][0],a['xy'][1]-b['xy'][1]) > a['bounding_radius_m']+b['bounding_radius_m']+.002
               for i,a in enumerate(blocks) for b in blocks[i+1:])

def initial(task,rng):
    if task=='avoiding': return []
    if task.startswith('sorting'):
        n=int(task[-1]); regions=rng.sample(SORT_REGIONS,n)
        names=[f'red_{i+1}' for i in range(n//2)]+[f'blue_{i+1}' for i in range(n//2)]
        radii=[math.hypot(.03,.03)]*n
    else:
        regions=STACK_REGIONS; names=['red_box','green_box','blue_box'];radii=[math.hypot(.03,.03)]*2+[math.hypot(.03,.05)]
    for _ in range(10000):
        blocks=[dict(name=name,xy=[rng.uniform(r[0],r[1]),rng.uniform(r[2],r[3])],yaw_rad=rng.uniform(-math.pi/2,math.pi/2),bounding_radius_m=rad)
                for name,r,rad in zip(names,regions,radii)]
        if valid_separation(blocks):return blocks
    raise RuntimeError('Could not sample separated initial blocks')

def conditions(task):
    result=[('id',0,0.)]
    specs=[]
    if task!='avoiding':specs.append(('object_position', [.03,.06,.10] if task.startswith('sorting') else [.015,.03,.06]))
    specs.append(('robot_xy',[.02,.05,.10]))
    if task=='stacking':specs.append(('robot_yaw',[math.radians(x) for x in [5,10,20]]))
    specs.append(('observation_xy',[.0025,.005,.01] if task=='avoiding' else [.003,.006,.012]))
    if task!='avoiding':specs.append(('observation_yaw',[math.radians(x) for x in [2.5,5,10]]))
    for family,values in specs:
        result += [(family,i+1,v) for i,v in enumerate(values)]
    return result

def perturb(case,family,level,value,root_seed,index):
    task=case['task'];rng=random.Random(seed_for(root_seed,task,case['split'],index,family))
    if family=='object_position':
        bidx=index%len(case['blocks']); b=case['blocks'][bidx]
        regions=SORT_REGIONS if task.startswith('sorting') else [STACK_REGIONS[bidx]]
        for _ in range(10000):
            r=rng.choice(regions);edge=rng.randrange(4)
            if edge<2: xy=[r[edge]+(-value if edge==0 else value),rng.uniform(r[2],r[3])]
            else: xy=[rng.uniform(r[0],r[1]),r[edge]+(-value if edge==2 else value)]
            b['xy']=xy
            if abs(distance_to_support(*xy,regions)-value)<1e-10 and valid_separation(case['blocks']):break
        else:raise RuntimeError('Could not sample support exterior')
        case['perturbation'].update(block_index=bidx,distance_outside_support_m=value)
    elif family=='robot_xy':
        # Balanced directions; all levels share direction for each base case.
        angle=(0 if index%2==0 else math.pi) if task=='avoiding' else (index%8)*math.pi/4
        case['robot_start']['xy_offset_m']=[value*math.cos(angle),value*math.sin(angle)]
    elif family=='robot_yaw':case['robot_start']['yaw_offset_rad']=value*(1 if index%2==0 else -1)
    elif family.startswith('observation_'):
        case['noise'].update(kind=family,sigma=value,target='actual_hand_xy' if task=='avoiding' else 'block_xy' if family.endswith('xy') else 'block_yaw')

def noise_sample(case,rollout,replan,entity,component):
    """Method-independent paired noise. Add before representation/normalization.
    Never inject into command anchors, simulator truth, rewards or metric streams.
    Noise is independently indexed, so methods terminating early do not shift RNG.
    """
    if not 0 <= rollout < case['repeats']:raise ValueError('rollout out of range')
    if replan<0:raise ValueError('negative replan')
    rng=random.Random(seed_for(case['noise']['seed'],rollout,replan,entity,component))
    return rng.gauss(0,case['noise']['sigma'])

def generate(out,dev=20,test=100,repeats=4,seed=20261106):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    manifest={'schema_version':1,'status':'CANDIDATES_ONLY_SIMULATOR_VALIDATION_REQUIRED','source_commit':COMMIT,'seed':seed,'dev_cases_per_condition':dev,'test_cases_per_condition':test,'repeats':repeats,'files':{},'task_counts':{},'checks':['finite values','source reset support membership or exact exterior distance','conservative XY block separation','unique IDs and disjoint split seeds'],'not_checked':['empirical training split support','workspace and wall geometry','robot IK and joint limits','robot-object collision','settling stability','path feasibility','policy and controller reset synchronization'],'sampling_note':'ID means source reset-distribution candidates conditioned on conservative separation; not official test contexts or a reproduced empirical distribution.'}
    for task in TASKS:
        rows=[]
        for split,n in [('dev',dev),('test',test)]:
            for i in range(n):
                base=initial(task,random.Random(seed_for(seed,task,split,i,'base')))
                for family,level,value in conditions(task):
                    case={'id':f'{task}/{split}/{family}/L{level}/{i:04d}','task':task,'split':split,'base_id':f'{task}/{split}/{i:04d}','blocks':json.loads(json.dumps(base)),'robot_start':{'reference':'canonical task reset','xy_offset_m':[0.,0.],'yaw_offset_rad':0.,'preserve_height':True,'preserve_gripper':True},'perturbation':{'family':family,'level':level},'noise':{'kind':'none','sigma':0.,'seed':seed_for(seed,task,split,i,'noise'),'sampling':'independent per coordinate per replan','target':None},'repeats':repeats,'policy_seeds':[seed_for(seed,task,split,i,j,'policy') for j in range(repeats)],'validation':{'analytical':'passed','simulator':'not_run','rollout_ready':False}}
                    perturb(case,family,level,value,seed,i);rows.append(case)
        path=out/(task+'.jsonl');path.write_text(''.join(json.dumps(r,sort_keys=True,allow_nan=False)+'\n' for r in rows))
        manifest['files'][path.name]=hashlib.sha256(path.read_bytes()).hexdigest();manifest['task_counts'][task]={'conditions':len(conditions(task)),'cases':len(rows),'planned_rollouts_per_method':len(rows)*repeats}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--dev',type=int,default=20);p.add_argument('--test',type=int,default=100);p.add_argument('--repeats',type=int,default=4);a=p.parse_args()
    if min(a.dev,a.test,a.repeats)<=0:p.error('counts must be positive')
    print(json.dumps(generate(a.output,a.dev,a.test,a.repeats),indent=2))
