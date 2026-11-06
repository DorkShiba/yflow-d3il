"""Verify the complete candidate package; this is NOT simulator validation."""
import argparse, hashlib, json, math
from pathlib import Path
try:
    from .generate import SORT_REGIONS, STACK_REGIONS, distance_to_support, valid_separation
except ImportError:
    from generate import SORT_REGIONS, STACK_REGIONS, distance_to_support, valid_separation

def verify(root):
    root=Path(root);m=json.loads((root/'manifest.json').read_text());seen=set();count=0
    for name,digest in m['files'].items():
        path=root/name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,name
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows)==m['task_counts'][path.stem]['cases']
        for c in rows:
            assert c['id'] not in seen;seen.add(c['id']);count+=1
            assert c['validation']=={'analytical':'passed','simulator':'not_run','rollout_ready':False}
            assert valid_separation(c['blocks'])
            changed=c['perturbation'].get('block_index')
            for i,b in enumerate(c['blocks']):
                assert all(math.isfinite(v) for v in b['xy']+[b['yaw_rad']])
                assert -math.pi/2 <= b['yaw_rad'] <= math.pi/2
                regions=SORT_REGIONS if c['task'].startswith('sorting') else [STACK_REGIONS[i]]
                d=distance_to_support(*b['xy'],regions)
                expected=c['perturbation']['distance_outside_support_m'] if changed==i else 0.
                assert abs(d-expected)<1e-9,(c['id'],i,d)
            assert len(c['policy_seeds'])==c['repeats']
        # Every family uses identical random policy seeds and canonical base ID.
        groups={}
        for c in rows:groups.setdefault(c['base_id'],[]).append(c)
        for group in groups.values():
            assert len({tuple(c['policy_seeds']) for c in group})==1
            assert len({c['noise']['seed'] for c in group})==1
    result={'verified_cases':count,'sha256':'passed','support_and_separation':'passed','paired_seeds':'passed','simulator_validation':'NOT RUN','rollout_ready_cases':0}
    (root/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();print(json.dumps(verify(a.root),indent=2))
