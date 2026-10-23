import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import FileResponse, Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from .media import read_rollouts, bounds, frame
from .analytics import enrich, TASKS

ROOT = Path(__file__).resolve().parents[1]
STORE = Path(os.environ.get('VISUALIZER_STORE', ROOT / '.visualizer')).resolve()
STORE.mkdir(parents=True, exist_ok=True)
# Ship recorded examples so a fresh teammate checkout opens a real comparison.
# Existing local jobs and dashboard choices always win.
SAMPLES = Path(__file__).parent / 'samples'
for sample in SAMPLES.glob('*/job.json'):
    destination = STORE / sample.parent.name
    if not destination.exists():
        shutil.copytree(sample.parent, destination)
if not (STORE/'dashboard.json').exists() and (SAMPLES/'dashboard.json').exists():
    shutil.copy2(SAMPLES/'dashboard.json', STORE/'dashboard.json')
POOL = ThreadPoolExecutor(max_workers=1)
JOBS = {}
PROCS = {}
LOCK = threading.RLock()
app = FastAPI(title='D3IL Visualizer')

@app.middleware('http')
async def local_only(request: Request, call_next):
    host = request.url.hostname
    origin = request.headers.get('origin')
    if host not in ('localhost', '127.0.0.1', 'testserver') or (origin and urlparse(origin).netloc != request.headers.get('host')):
        return JSONResponse({'detail':'Local same-origin access only'},status_code=403)
    response = await call_next(request)
    if request.url.path in ('/', '/index.html', '/app.js', '/charts.js', '/scene3d.js', '/style.css', '/scene-export.js'):
        response.headers['Cache-Control'] = 'no-cache'
    return response

def job_dir(id):
    if not id.isalnum() or len(id)!=32:
        raise HTTPException(404,'Unknown job')
    path = STORE/id
    if not path.is_dir(): raise HTTPException(404,'Unknown job')
    return path

def persist(j):
    tmp=STORE/j['id']/'job.tmp'; tmp.write_text(json.dumps(j)); tmp.replace(STORE/j['id']/'job.json')

def new_job(kind, spec):
    j=dict(id=uuid.uuid4().hex, kind=kind, status='queued', created=time.time(), spec=spec)
    (STORE/j['id']).mkdir(); JOBS[j['id']]=j; persist(j)
    return j

for saved in STORE.glob('*/job.json'):
    j=json.loads(saved.read_text())
    if j['status'] in ('queued','running','capturing'):
        j.update(status='interrupted', error='Server stopped; submit a new job to retry')
        persist(j)
    JOBS[j['id']]=j

def run_job(j, fn):
    with LOCK:
        if j['status']=='cancelled': return
        j['status']='running'; persist(j)
    try:
        fn()
        with LOCK:
            if j['status']!='cancelled': j['status']='completed'
    except Exception as e:
        with LOCK:
            if j['status']!='cancelled': j.update(status='failed',error=str(e))
    finally:
        with LOCK: persist(j)

def command(j, args, env=None, stdin=None):
    with open(STORE/j['id']/'log.txt','ab') as log:
        with LOCK:
            if j['status']=='cancelled': raise RuntimeError('Cancelled')
            p=subprocess.Popen(args, cwd=ROOT, env=env, stdout=log, stderr=log,
                               stdin=subprocess.PIPE if stdin else None, start_new_session=True)
            PROCS[j['id']]=p
        try:
            if stdin:
                try: stdin(p.stdin)
                finally: p.stdin.close()
            if p.wait()!=0: raise RuntimeError('Process failed; see job log')
        finally:
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM); p.wait()
            with LOCK: PROCS.pop(j['id'],None)

class EvalSpec(BaseModel):
    run_dir: str
    checkpoint: str='baseline_ema.pt'
    python: str=sys.executable
    d3il_root: str
    hardflow_root: str
    data_dir: str=str(ROOT/'data')
    task: str='pushing'
    contexts: str='test'
    seed: int=Field(default=0,ge=0,le=2**31-1)
    n_contexts: int=Field(default=1,ge=1,le=30)
    execute_steps: int=Field(default=8,ge=1,le=16)
    max_steps: int=Field(default=400,ge=1,le=2000)
    noise: str='fresh'
    robot: bool=False
    camera: str='oblique'
    label: str=''
    gl: str='osmesa' if sys.platform=='linux' else 'glfw'

@app.get('/api/health')
def health():
    return dict(ffmpeg=bool(shutil.which('ffmpeg')), python=sys.executable, root=str(ROOT), device='cpu')

@app.get('/api/checkpoints')
def checkpoints():
    found=[]
    for base in [ROOT/'fm_baseline/results',ROOT/'fm_baseline/runs']:
        for cfg in base.glob('*/config.json'):
            if not (cfg.parent/'normalizer.npz').exists(): continue
            for ck in cfg.parent.glob('*.pt'):
                found.append(dict(run_dir=str(cfg.parent),checkpoint=ck.name,config=json.loads(cfg.read_text())))
    return found

@app.get('/api/jobs')
def jobs():
    with LOCK: return sorted(JOBS.values(),key=lambda j:j['created'],reverse=True)

@app.get('/api/jobs/{id}/log')
def log(id:str):
    p=job_dir(id)/'log.txt'
    return {'text':p.read_text(errors='replace')[-12000:] if p.exists() else ''}

@app.post('/api/jobs/{id}/cancel')
def cancel(id:str):
    job_dir(id)
    with LOCK:
        j=JOBS[id]
        if j['status'] not in ('queued','running','capturing'): raise HTTPException(409,'Job already finished')
        j['status']='cancelled'; persist(j)
        p=PROCS.get(id)
        if p and p.poll() is None: os.killpg(p.pid,signal.SIGTERM)
    return j

@app.post('/api/evaluate')
def evaluate(s:EvalSpec):
    if s.task not in ('pushing','inserting') or s.noise not in ('fresh','fixed') or s.camera not in ('top','oblique') or s.gl not in ('osmesa','glfw'):
        raise HTTPException(400,'Unsupported option')
    if s.contexts not in (('test','train') if s.task=='pushing' else ('val',)):
        raise HTTPException(400,'Invalid context split')
    if s.task=='inserting' and s.noise=='fixed': raise HTTPException(400,'Inserting fixed noise is not implemented')
    run=Path(s.run_dir).expanduser().resolve()
    if Path(s.checkpoint).name!=s.checkpoint: raise HTTPException(400,'Checkpoint must be a filename')
    required=[run/s.checkpoint,run/'config.json',run/'normalizer.npz']
    if not all(p.is_file() for p in required): raise HTTPException(400,'Checkpoint, config.json and normalizer.npz are required')
    cfg=json.loads(required[1].read_text())
    if cfg.get('task')!=s.task: raise HTTPException(400,'Checkpoint task mismatch')
    for directory in [s.d3il_root,s.hardflow_root,s.data_dir]:
        if not Path(directory).expanduser().is_dir(): raise HTTPException(400,'Dependency directory missing: '+directory)
    if not shutil.which(s.python): raise HTTPException(400,'Python executable not found')
    s.run_dir=str(run)
    j=new_job('evaluation',s.model_dump())
    def work():
        out=STORE/j['id']
        manifest=dict(spec=s.model_dump(),config=cfg,sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in required},
                      source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'fm_baseline').glob('*.py')})
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
        args=[s.python,str(ROOT/f'fm_baseline/eval_{s.task}.py'),'--run-dir',str(run),'--ckpt',s.checkpoint,'--device','cpu',
              '--d3il-root',str(Path(s.d3il_root).expanduser().resolve()),'--hardflow-root',str(Path(s.hardflow_root).expanduser().resolve()),
              '--data-dir',str(Path(s.data_dir).expanduser().resolve()),'--n-contexts',str(s.n_contexts),'--n-traj','1',
              '--execute-steps',str(s.execute_steps),'--max-steps',str(s.max_steps),'--seed',str(s.seed),'--paired-noise','--record-joints','--output-root',str(out),'--timing-file',str(out/'timing.json')]
        args += ['--contexts',s.contexts,'--replan-noise',s.noise] if s.task=='pushing' else ['--split',s.contexts]
        if s.robot: args+=['--capture-dir',str(out/'frames'),'--capture-camera',s.camera]
        env=os.environ.copy(); env.update(MUJOCO_GL=s.gl,OMP_NUM_THREADS='2',PYTHONUNBUFFERED='1')
        command(j,args,env)
        source=next(out.glob('eval_*/rollouts.npz'))
        for p in required:
            if hashlib.sha256(p.read_bytes()).hexdigest()!=manifest['sha256'][p.name]:
                raise RuntimeError('Checkpoint inputs changed during evaluation')
        rolls=read_rollouts(source,json.loads((out/'timing.json').read_text())['dt'])
        (out/'rollouts.json').write_text(json.dumps(rolls))
        j.update(rollouts=[r['key'] for r in rolls],label=s.label or f'{s.task} / {s.checkpoint}',robot=s.robot)
    POOL.submit(run_job,j,work)
    return j

@app.post('/api/import')
async def import_file(file:UploadFile=File(...),dt:float=Form(.035),label:str=Form('Imported rollout')):
    raw=await file.read(64*1024*1024+1)
    if len(raw)>64*1024*1024: raise HTTPException(413,'Maximum upload size: 64 MiB')
    import zipfile
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            if sum(i.file_size for i in z.infolist())>256*1024*1024: raise ValueError('Expanded archive too large')
        rolls=read_rollouts(io.BytesIO(raw),dt)
    except Exception as e: raise HTTPException(400,str(e))
    j=new_job('import',dict(dt=dt,label=label))
    out=STORE/j['id']; (out/'rollouts.npz').write_bytes(raw); (out/'rollouts.json').write_text(json.dumps(rolls))
    j.update(status='completed',label=label,robot=False,rollouts=[r['key'] for r in rolls]); persist(j)
    return j

@app.get('/api/jobs/{id}/rollouts')
def rollouts(id:str):
    p=job_dir(id)/'rollouts.json'
    if not p.exists(): raise HTTPException(404,'Rollouts unavailable')
    records=json.loads(p.read_text())
    if any('act' not in r for r in records):
        sources=list(job_dir(id).glob('eval_*/rollouts.npz')) or list(job_dir(id).glob('rollouts.npz'))
        if sources:
            restored={r['key']:r for r in read_rollouts(sources[0], records[0]['dt'])}
            records=[dict(restored.get(r['key'],{}), **r) for r in records]
    return [enrich(r, JOBS[id].get('spec', {}).get('task')) for r in records]

@app.get('/api/jobs/{id}/frames/{key}/{index}')
def robot_frame(id:str,key:str,index:int):
    import re
    if not re.fullmatch(r'c\d+_t\d+',key) or index<0: raise HTTPException(400,'Invalid frame')
    p=job_dir(id)/'frames'/key/f'{index:06d}.jpg'
    if not p.exists(): raise HTTPException(404,'Frame unavailable')
    return FileResponse(p)


@app.get('/api/dashboard')
def dashboard():
    config=STORE/'dashboard.json'
    if not config.exists(): return {'tasks':{},'default_task':'pushing'}
    data=json.loads(config.read_text())
    for group in data.get('tasks',{}).values():
        group['panels']=[p for p in group.get('panels',[]) if p.get('job') in JOBS and JOBS[p['job']]['status']=='completed']
    return data

class Panel(BaseModel):
    job: str
    key: str
    label: str=Field(max_length=80)
    view: str='trajectory'
class Export(BaseModel):
    panels:list[Panel]=Field(min_length=1,max_length=4)
    layout:str='grid'
    format:str='mp4'
    start:float=Field(default=0,ge=0)
    end:float=Field(gt=0,le=120)
    speed:float=Field(default=1,ge=.25,le=4)

@app.post('/api/export')
def export(s:Export):
    if s.layout not in ('row','column','grid') or s.format not in ('mp4','gif') or s.end<=s.start:
        raise HTTPException(400,'Invalid export settings')
    if s.format=='gif' and (s.end-s.start)/s.speed>15: raise HTTPException(400,'GIF maximum output duration is 15 seconds')
    if not shutil.which('ffmpeg'): raise HTTPException(400,'Install FFmpeg first')
    selected=[]
    for p in s.panels:
        rs=rollouts(p.job); r=next((r for r in rs if r['key']==p.key),None)
        if not r: raise HTTPException(400,'Unknown rollout')
        if p.view not in ('trajectory','robot','scene'): raise HTTPException(400,'Unknown view')
        if p.view=='robot' and not JOBS[p.job].get('robot'): raise HTTPException(400,'Robot frames unavailable')
        selected.append(r)
    if s.end>max(r['steps']*r['dt'] for r in selected)+.001: raise HTTPException(400,'End exceeds rollout duration')
    if any(p.view=='scene' for p in s.panels):
        import math
        fps=15 if s.format=='gif' else 30
        count=math.ceil((s.end-s.start)/s.speed*fps)
        if count>1800: raise HTTPException(400,'3D MP4 output is limited to 60 seconds; shorten the interval')
        n=len(selected);cols=n if s.layout=='row' else 1 if s.layout=='column' else min(2,n)
        cell=(480,360) if s.format=='gif' else (640,480)
        j=new_job('export',s.model_dump())
        j.update(status='capturing',frames_received=0,frame_bytes=0,
                 capture=dict(width=cols*cell[0],height=math.ceil(n/cols)*cell[1],cell=list(cell),cols=cols,count=count,fps=fps),
                 label='3D '+s.format.upper())
        (STORE/j['id']/'export-frames').mkdir();persist(j)
        return j
    j=new_job('export',s.model_dump())
    def work():
        from PIL import Image
        import math
        n=len(selected); cols=n if s.layout=='row' else 1 if s.layout=='column' else min(2,n)
        rows=math.ceil(n/cols); cell=(480,360) if s.format=='gif' else (640,480)
        w,h=cols*cell[0],rows*cell[1]; fps=15 if s.format=='gif' else 30
        out=STORE/j['id']/('comparison.'+s.format)
        args=['ffmpeg','-y','-v','error','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{w}x{h}','-framerate',str(fps),'-i','pipe:0']
        args+=['-vf','split[a][b];[a]palettegen[p];[b][p]paletteuse'] if s.format=='gif' else ['-vf','scale=out_range=tv:out_color_matrix=bt709,format=yuv420p','-c:v','libx264','-pix_fmt','yuv420p','-color_range','tv','-colorspace','bt709','-color_primaries','bt709','-color_trc','bt709','-movflags','+faststart']
        box=bounds(selected)
        def feed(pipe):
            count=math.ceil((s.end-s.start)/s.speed*fps)
            for i in range(count):
                if j['status']=='cancelled': raise RuntimeError('Cancelled')
                im=Image.new('RGB',(w,h),'#ffffff'); t=s.start+i/fps*s.speed
                for k,(r,p) in enumerate(zip(selected,s.panels)):
                    folder=STORE/p.job/'frames'/p.key if p.view=='robot' else None
                    im.paste(frame(r,t,p.label,box,cell,folder),((k%cols)*cell[0],(k//cols)*cell[1]))
                pipe.write(im.tobytes())
        command(j,args+[str(out)],stdin=feed)
        (STORE/j['id']/'manifest.json').write_text(json.dumps(s.model_dump(),indent=2))
        j['download']=f'/api/jobs/{j["id"]}/download'
    POOL.submit(run_job,j,work)
    return j

@app.post('/api/jobs/{id}/export-frames/{index}')
async def export_frame(id:str,index:int,request:Request):
    from PIL import Image
    job_dir(id);j=JOBS[id]
    if j['status']!='capturing' or not j.get('capture'): raise HTTPException(409,'Export is not capturing')
    raw=bytearray()
    async for chunk in request.stream():
        raw.extend(chunk)
        if len(raw)>2*1024*1024: raise HTTPException(413,'Frame exceeds 2 MiB')
    try:
        with Image.open(io.BytesIO(raw)) as im:
            if im.format!='JPEG' or im.size!=(j['capture']['width'],j['capture']['height']):
                raise ValueError('Invalid frame dimensions or format')
            im.verify()
    except Exception as e: raise HTTPException(400,str(e))
    with LOCK:
        if j['status']!='capturing' or index!=j['frames_received'] or index>=j['capture']['count']:
            raise HTTPException(409,'Frames must arrive once, in order')
        if j['frame_bytes']+len(raw)>256*1024*1024: raise HTTPException(413,'Capture exceeds 256 MiB; shorten the interval')
        (STORE/id/'export-frames'/f'{index:06d}.jpg').write_bytes(raw)
        j['frames_received']+=1;j['frame_bytes']+=len(raw)
        if j['frames_received']%15==0:persist(j)
    return {'received':j['frames_received']}

@app.post('/api/jobs/{id}/encode')
def encode_frames(id:str):
    job_dir(id);j=JOBS[id]
    with LOCK:
        if j['status']!='capturing' or j.get('frames_received')!=j.get('capture',{}).get('count'):
            raise HTTPException(409,'Capture is incomplete or already submitted')
        j['status']='queued';persist(j)
    def work():
        folder=STORE/id;spec=j['spec'];out=folder/('comparison.'+spec['format'])
        args=['ffmpeg','-y','-v','error','-framerate',str(j['capture']['fps']),'-i',str(folder/'export-frames/%06d.jpg')]
        args+=['-vf','split[a][b];[a]palettegen[p];[b][p]paletteuse'] if spec['format']=='gif' else ['-vf','scale=out_range=tv:out_color_matrix=bt709,format=yuv420p','-c:v','libx264','-pix_fmt','yuv420p','-color_range','tv','-colorspace','bt709','-color_primaries','bt709','-color_trc','bt709','-movflags','+faststart']
        command(j,args+[str(out)])
        (folder/'manifest.json').write_text(json.dumps(dict(**spec,renderer='browser Three.js',background='white',capture=j['capture']),indent=2))
        j['download']=f'/api/jobs/{id}/download'
        shutil.rmtree(folder/'export-frames')
    POOL.submit(run_job,j,work)
    return j

@app.get('/api/jobs/{id}/download')
def download(id:str):
    folder=job_dir(id); j=JOBS[id]
    if j['status']!='completed' or j['kind']!='export': raise HTTPException(404,'Export not ready')
    p=folder/('comparison.'+j['spec']['format']); return FileResponse(p,filename=p.name)

@app.get('/api/jobs/{id}/manifest')
def manifest(id:str):
    folder=job_dir(id); p=folder/'manifest.json'
    return FileResponse(p if p.exists() else folder/'job.json',filename='manifest.json')

app.mount('/',StaticFiles(directory=Path(__file__).parent/'static',html=True),name='ui')
