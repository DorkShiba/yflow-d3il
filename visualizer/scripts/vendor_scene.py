"""Fetch pinned upstream visual-only Panda assets and Three.js for offline use."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request
import xml.etree.ElementTree as ET

DEST=Path(__file__).resolve().parents[1]/'static'
PIN=''
def get(url):
    with urllib.request.urlopen(url,timeout=60) as r:return r.read()
def main():
    version='0.170.0'
    vendor=DEST/'vendor';vendor.mkdir(exist_ok=True)
    for name,remote in [('three.module.js','build/three.module.js'),('OrbitControls.js','examples/jsm/controls/OrbitControls.js'),('OBJLoader.js','examples/jsm/loaders/OBJLoader.js'),('THREE-LICENSE.txt','LICENSE')]:
        (vendor/name).write_bytes(get(f'https://cdn.jsdelivr.net/npm/three@{version}/{remote}'))
    asset=DEST/'panda';asset.mkdir(exist_ok=True)
    lock=asset/'provenance.json'
    revision=json.loads(lock.read_text())['revision'] if lock.exists() else json.loads(get('https://api.github.com/repos/google-deepmind/mujoco_menagerie/commits/main'))['sha']
    base=f'https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{revision}/franka_emika_panda/'
    xml=get(base+'panda.xml');(asset/'panda.xml').write_bytes(xml);root=ET.fromstring(xml)
    meshes={m.get('name',Path(m.get('file')).stem):m.get('file') for m in root.findall('asset/mesh')}
    used=sorted({meshes[g.get('mesh')] for g in root.findall('.//geom') if g.get('class')=='visual' and g.get('mesh')})
    def fetch(name):
        raw=get(base+'assets/'+name);(asset/name).write_bytes(raw);return name,hashlib.sha256(raw).hexdigest()
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex: checksums=dict(ex.map(fetch,used))
    for name in ['LICENSE','README.md']:(asset/name).write_bytes(get(base+name))
    def numbers(node,name,default):return [float(x) for x in node.get(name,default).split()]
    def body(node):
        j=node.find('joint')
        return dict(name=node.get('name'),pos=numbers(node,'pos','0 0 0'),quat=numbers(node,'quat','1 0 0 0'),
                    joint=j.get('name') if j is not None else None,
                    visuals=[dict(file=meshes[g.get('mesh')],material=g.get('material')) for g in node.findall('geom') if g.get('class')=='visual'],
                    children=[body(n) for n in node.findall('body')])
    model=dict(body=body(root.find('worldbody/body')),materials={m.get('name'):numbers(m,'rgba','1 1 1 1') for m in root.findall('asset/material')},home=numbers(root.find('keyframe/key'),'qpos',''))
    (asset/'model.json').write_text(json.dumps(model))
    lock.write_text(json.dumps(dict(source='google-deepmind/mujoco_menagerie/franka_emika_panda',revision=revision,three=version,sha256=checksums),indent=2))
    print(f'Vendored {len(used)} visual meshes, revision {revision}')
if __name__=='__main__':main()
