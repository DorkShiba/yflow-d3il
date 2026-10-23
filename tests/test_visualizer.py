import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
import numpy as np

TMP=tempfile.TemporaryDirectory()
os.environ['VISUALIZER_STORE']=TMP.name
from fastapi.testclient import TestClient
from visualizer.app import app
from visualizer.media import read_rollouts, frame, bounds

class VisualizerTest(unittest.TestCase):
    def setUp(self): self.client=TestClient(app)
    def sample(self, n=20, invalid=False):
        b=io.BytesIO(); t=np.arange(n)*.001
        c=np.column_stack([t,t,np.ones(n)*.1]); c[0,0]=np.nan if invalid else 0
        np.savez_compressed(b,c000_t0_c=c,c000_t0_des=c[:,:2],c000_t0_boxes=np.zeros((n,6)),successes=np.ones((1,1)))
        return b.getvalue()
    def upload(self):
        r=self.client.post('/api/import',files={'file':('rollouts.npz',self.sample())},data={'dt':'.035','label':'OFF'})
        self.assertEqual(r.status_code,200,r.text);return r.json()
    def test_import_and_reject_bad_data(self):
        j=self.upload(); r=self.client.get(f'/api/jobs/{j["id"]}/rollouts').json()[0]
        self.assertEqual(r['steps'],20)
        self.assertEqual(r['dt'],.035)
        im=frame(r,100,'OFF',bounds([r]));self.assertEqual(im.size,(640,480))
        bad=self.client.post('/api/import',files={'file':('bad.npz',self.sample(invalid=True))})
        self.assertEqual(bad.status_code,400)
    def test_local_guard(self):
        self.assertEqual(self.client.post('/api/evaluate',headers={'Origin':'https://evil.example'},json={}).status_code,403)
        self.assertEqual(self.client.get('/api/health',headers={'Host':'evil.example'}).status_code,403)
    def test_missing_environment(self):
        r=self.client.post('/api/evaluate',json={'run_dir':'/missing','d3il_root':'/missing','hardflow_root':'/missing'})
        self.assertEqual(r.status_code,400)
    def test_fresh_store_has_matched_default_pair(self):
        dashboard=self.client.get('/api/dashboard').json()
        panels=dashboard['tasks']['pushing']['panels']
        self.assertEqual(len(panels),2)
        jobs={j['id']:j for j in self.client.get('/api/jobs').json()}
        for p in panels:
            spec=jobs[p['job']]['spec']
            self.assertEqual(spec['checkpoint_step'],100000)
            self.assertEqual(spec['training_seed'],42)
            r=self.client.get(f'/api/jobs/{p["job"]}/rollouts').json()[0]
            self.assertEqual(r['key'],p['key'])
            self.assertEqual(len(r['act']),r['steps'])
            self.assertNotIn('joints',r)
            self.assertEqual(r['distance_kind'],'XY projection')
    def test_captured_goals_xyz_roundtrip(self):
        data=io.BytesIO()
        np.savez_compressed(data,c000_t0_c=np.zeros((2,3)),c000_t0_des=np.zeros((2,2)),
            c000_t0_boxes=np.zeros((2,6)),c000_t0_boxes_xyz=np.tile([0,0,.1,1,0,.1],(2,1)),
            c000_t0_goals=[[0,0,0],[1,0,0]],c000_t0_goal_threshold=.05,c000_t0_goal_rule='matched')
        result=self.client.post('/api/import',files={'file':('record.npz',data.getvalue())})
        self.assertEqual(result.status_code,200,result.text)
        r=self.client.get('/api/jobs/'+result.json()['id']+'/rollouts').json()[0]
        self.assertEqual(r['distance_kind'],'3D')
        self.assertEqual(r['series']['mean_distance_mm'],[100,100])
        self.assertEqual(r['goal_spec']['source'],'Recorded simulator goal positions')
    def test_composite_export(self):
        j=self.upload()
        for fmt,layout in [('mp4','row'),('gif','column')]:
            spec={'panels':[{'job':j['id'],'key':'c000_t0','label':'OFF'},{'job':j['id'],'key':'c000_t0','label':'T+S'}], 'layout':layout,'format':fmt,'start':0,'end':.7}
            response=self.client.post('/api/export',json=spec);self.assertEqual(response.status_code,200,response.text)
            id=response.json()['id']
            for _ in range(200):
                job=next(x for x in self.client.get('/api/jobs').json() if x['id']==id)
                if job['status'] in ('completed','failed'):break
                time.sleep(.05)
            self.assertEqual(job['status'],'completed',job)
            media=Path(TMP.name)/id/f'comparison.{fmt}'
            probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(media)]))['streams'][0]
            self.assertEqual((probe['width'],probe['height']),(1280,480) if fmt=='mp4' else (480,720))
            subprocess.run(['ffmpeg','-v','error','-i',str(media),'-f','null','-'],check=True)
            self.assertEqual(self.client.get(f'/api/jobs/{id}/download').status_code,200)
    def test_finished_rollout_holds_last_state(self):
        r=read_rollouts(io.BytesIO(self.sample()),.035)[0]
        a=frame(r,.7,'OFF',bounds([r])).crop((0,50,640,480))
        b=frame(r,5,'OFF',bounds([r])).crop((0,50,640,480))
        self.assertEqual(a.tobytes(),b.tobytes())

    def test_scene_capture_validation_and_encoding(self):
        from PIL import Image
        j=self.upload()
        for fmt in ('mp4','gif'):
            response=self.client.post('/api/export',json={'panels':[{'job':j['id'],'key':'c000_t0','label':'OFF','view':'scene'}], 'format':fmt,'end':.1})
            self.assertEqual(response.status_code,200,response.text)
            job=response.json();id=job['id'];capture=job['capture']
            self.assertEqual(job['status'],'capturing')
            self.assertEqual(self.client.post(f'/api/jobs/{id}/encode').status_code,409)
            image=Image.new('RGB',(capture['width'],capture['height']),'white');buffer=io.BytesIO();image.save(buffer,format='JPEG')
            self.assertEqual(self.client.post(f'/api/jobs/{id}/export-frames/0',content=b'not a JPEG').status_code,400)
            self.assertEqual(self.client.post(f'/api/jobs/{id}/export-frames/1',content=buffer.getvalue()).status_code,409)
            for i in range(capture['count']):
                self.assertEqual(self.client.post(f'/api/jobs/{id}/export-frames/{i}',content=buffer.getvalue()).status_code,200)
            self.assertEqual(self.client.post(f'/api/jobs/{id}/export-frames/0',content=buffer.getvalue()).status_code,409)
            self.assertEqual(self.client.post(f'/api/jobs/{id}/encode').status_code,200)
            for _ in range(100):
                job=next(x for x in self.client.get('/api/jobs').json() if x['id']==id)
                if job['status'] in ('completed','failed'):break
                time.sleep(.05)
            self.assertEqual(job['status'],'completed',job)
            media=Path(TMP.name)/id/f'comparison.{fmt}'
            streams=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(media)]))['streams']
            self.assertEqual(streams[0]['codec_name'],'h264' if fmt=='mp4' else 'gif')
            subprocess.run(['ffmpeg','-v','error','-i',str(media),'-f','null','-'],check=True)
            self.assertFalse((Path(TMP.name)/id/'export-frames').exists())

    def test_running_job_cancellation(self):
        import sys
        from visualizer.app import new_job, run_job, command, POOL
        j=new_job('test',{})
        future=POOL.submit(run_job,j,lambda:command(j,[sys.executable,'-c','import time; time.sleep(30)']))
        for _ in range(100):
            if j['status']=='running':break
            time.sleep(.01)
        response=self.client.post(f'/api/jobs/{j["id"]}/cancel')
        self.assertEqual(response.status_code,200)
        future.result(timeout=5)
        self.assertEqual(j['status'],'cancelled')

    def test_optional_joint_record(self):
        raw=self.sample()
        with np.load(io.BytesIO(raw)) as source:
            arrays={key:source[key] for key in source.files}
        arrays['c000_t0_joints']=np.zeros((20,7))
        b=io.BytesIO();np.savez_compressed(b,**arrays)
        r=read_rollouts(io.BytesIO(b.getvalue()),.035)[0]
        self.assertEqual(np.asarray(r['joints']).shape,(20,7))
        arrays['c000_t0_joints']=np.zeros((20,6))
        b=io.BytesIO();np.savez_compressed(b,**arrays)
        with self.assertRaisesRegex(ValueError,'joint record'):
            read_rollouts(io.BytesIO(b.getvalue()),.035)

    def test_vendored_model(self):
        from visualizer.app import ROOT
        model_path=ROOT/'visualizer/static/panda'
        model=json.loads((model_path/'model.json').read_text())
        joints=[]
        def visit(body):
            if body['joint']: joints.append(body['joint'])
            for visual in body['visuals']:
                self.assertTrue((model_path/visual['file']).is_file())
                self.assertIn(visual['material'],model['materials'])
            for child in body['children']:visit(child)
        visit(model['body'])
        self.assertEqual(len([j for j in joints if j.startswith('joint')]),7)
        self.assertEqual(len(model['home']),9)

    def test_bad_export(self):
        self.assertEqual(self.client.post('/api/export',json={'panels':[],'end':1}).status_code,422)

if __name__=='__main__': unittest.main()
