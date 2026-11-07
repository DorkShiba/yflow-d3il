"""Independent checks on downloaded raw rollouts and frozen experiment inputs."""
from pathlib import Path
import hashlib
import json
import numpy as np

root = Path(__file__).resolve().parent
main = root / 'artifacts/main'
manifest = json.loads((root/'artifact_sha256.json').read_text())
assert all(hashlib.sha256((main/p).read_bytes()).hexdigest() == sha for p,sha in manifest.items())
protocol = json.loads((root/'protocol.json').read_text())
maxerr = 0.; total = 0; qps = 0
serial_differences = {}
for mode in ('none','blend','lipo'):
    folder = main/'combined'/mode
    result = json.loads((folder/'results.json').read_text())
    detail = json.loads((folder/'smoothness.json').read_text())
    dt = result['control_dt_s']
    assert result['source_sha256'] == protocol['source_sha256']
    assert result['checkpoint_sha256'] == protocol['checkpoint_sha256']
    with np.load(folder/'rollouts.npz') as z:
        assert z['successes'].shape == (82,4)
        assert len(result['context_files']) == len(set(result['context_files'])) == 82
        for episode,metric in detail['episodes'].items():
            for label,suffix,init in [('desired','_des','_initial_des'),('actual','_c','_initial_c')]:
                p = np.vstack([z[episode+init][:2],z[episode+suffix][:,:2]])
                j = (p[3:]-3*p[2:-1]+3*p[1:-2]-p[:-3])/dt**3
                rms = float(np.sqrt(np.mean((j*j).sum(axis=1))))
                maxerr = max(maxerr,abs(rms-metric[label+'_jerk_rms_m_s3']))
                assert np.isclose(rms,metric[label+'_jerk_rms_m_s3'],rtol=1e-9,atol=1e-9)
            np.testing.assert_allclose(np.diff(np.vstack([z[episode+'_initial_des'],z[episode+'_des']]),axis=0),z[episode+'_act'],atol=1e-12)
            total += 1
        smoke = root/'artifacts/smoke/combined'/mode/'rollouts.npz'
        with np.load(smoke) as s:
            diffs = [float(np.max(np.abs(s[k]-z[k]))) for k in s.files if k.endswith('_des') or k.endswith('_c')]
            serial_differences[mode] = max(diffs)
        if mode == 'lipo':
            for rows in detail['plans'].values():
                for row in rows:
                    assert row['solver_status'] == 'Solved'
                    limit = .002 if row['blend_steps'] else .0003
                    assert row['qp_correction_max_m'] <= limit+1e-8
                    assert row['jerk_cost_after_m2'] <= row['jerk_cost_before_m2']+max(1e-16,row['jerk_cost_before_m2']*1e-6)
                    qps += 1
verification = {'raw_rollouts_checked':total,'lipo_qps_checked':qps,'max_independent_jerk_rms_error':maxerr,
                'artifact_hashes_verified':len(manifest),'protocol_source_and_checkpoint_match':True,
                'smoke_vs_main_max_position_difference_m':serial_differences}
(root/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
print(json.dumps(verification,indent=2))
