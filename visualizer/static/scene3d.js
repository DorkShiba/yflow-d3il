import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';
import { OBJLoader } from './vendor/OBJLoader.js';

const loader = new OBJLoader();
let assets;
function loadAssets() {
  if (!assets) assets = (async () => {
    const response = await fetch('/panda/model.json');
    if (!response.ok) throw Error('Panda 모델을 불러오지 못했습니다');
    const model = await response.json();
    const files = new Set();
    function visit(b) { b.visuals.forEach(v => files.add(v.file)); b.children.forEach(visit); }
    visit(model.body);
    const meshes = new Map(await Promise.all([...files].map(async file => [file, await loader.loadAsync('/panda/' + file)])));
    return { model, meshes };
  })();
  return assets;
}

export class Scene3D {
  constructor(host, { rollout = null, task = 'pushing', onError = () => {} } = {}) {
    this.host = host; this.rollout = rollout; this.task = task; this.disposed = false;
    this.armJoints = []; this.fingers = []; this.paths = []; this.goalLines = []; this.step = 0;
    this.scene = new THREE.Scene(); this.scene.background = new THREE.Color('#ffffff');
    this.camera = new THREE.PerspectiveCamera(38, 1, .01, 30);
    this.camera.up.set(0, 0, 1);
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.25;
    this.renderer.domElement.setAttribute('aria-label', 'Panda 로봇팔과 블록의 3D 장면');
    host.append(this.renderer.domElement);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(.35, 0, .25); this.controls.minDistance = .5; this.controls.maxDistance = 4;
    this.controls.maxPolarAngle = Math.PI / 2 - .02;
    this.controls.addEventListener('change', () => this.render());
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0xc5c7ca, 2.0));
    const key = new THREE.DirectionalLight(0xffffff, 2.8);
    key.position.set(1, -.7, 2); key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    Object.assign(key.shadow.camera, { left: -1.2, right: 1.2, top: 1.2, bottom: -1.2, near: .1, far: 6 });
    key.shadow.normalBias = .015; this.scene.add(key);
    const fill = new THREE.DirectionalLight(0xffffff, 1.0); fill.position.set(-1, .5, 1); this.scene.add(fill);
    this.addBox([1.3, 1.05, .055], [.4, 0, -.034], '#e8ebee');
    const grid = new THREE.GridHelper(1.1, 22, 0xc4cbd3, 0xd4d9df);
    grid.rotation.x = Math.PI / 2; grid.position.set(.4, 0, -.004); this.scene.add(grid);
    this.blocks = [];
    const count = rollout ? rollout.boxes[0].length / 3 : task === 'inserting' ? 3 : 2;
    const colors = ['#b64b3b', '#36836b', '#8062a4'];
    for (let i = 0; i < count; i++) {
      this.blocks.push(this.addBox([.055, .055, .055], [.43 + i * .12, -.12, .0275], colors[i]));
      if (!rollout) {
        const goal = this.addBox([.075, .075, .004], [.43 + i * .12, .2, .001], colors[i]);
        goal.material.transparent = true; goal.material.opacity = .25;
      }
    }
    if (rollout?.goal_spec) {
      rollout.goal_spec.goals.forEach((g, i) => {
        const ring = new THREE.Mesh(new THREE.RingGeometry(rollout.goal_spec.threshold * .92, rollout.goal_spec.threshold, 48), new THREE.MeshBasicMaterial({ color: colors[i], side: THREE.DoubleSide, transparent: true, opacity: .85 }));
        ring.position.set(g[0], g[1], .004); this.scene.add(ring);
        const tile = this.addBox([.06,.06,.004], [g[0],g[1],.001], colors[i]); tile.material.transparent=true; tile.material.opacity=.25;
        this.addLabel('G'+(i+1), [g[0],g[1],.09], colors[i]);
        const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(),new THREE.Vector3()]), new THREE.LineBasicMaterial({color: colors[i],transparent:true,opacity:.65}));
        this.goalLines.push(line);this.scene.add(line);
      });
    }
    if (!rollout && task === 'inserting') {
      for (let i = 0; i < 4; i++) this.addBox([.022, .18, .05], [.37 + i * .12, .20, .025], '#718197');
    }
    this.hand = new THREE.Mesh(new THREE.SphereGeometry(.012, 20, 12), new THREE.MeshStandardMaterial({ color: '#25669b' }));
    this.target = new THREE.Mesh(new THREE.SphereGeometry(.008, 16, 10), new THREE.MeshStandardMaterial({ color: '#697680' }));
    this.hand.visible = this.target.visible = !!rollout;
    this.scene.add(this.hand, this.target);
    if (rollout) {
      this.addPath(rollout.c.map(p => [p[0], p[1], p[2] ?? .07]), '#25669b');
      this.addPath(rollout.des.map((p, i) => [p[0], p[1], rollout.c[i][2] ?? .07]), '#697680');
      for (let j = 0; j < count; j++) this.addPath(rollout.boxes.map(p => [p[j * 3], p[j * 3 + 1], .03]), colors[j]);
    }
    this.resizeObserver = new ResizeObserver(() => this.resize()); this.resizeObserver.observe(host);
    this.setCamera('oblique'); this.resize(); this.setStep(0);
    this.ready = loadAssets().then(({ model, meshes }) => {
      if (this.disposed) return;
      this.home = model.home;
      const materials = {};
      for (const [name, rgba] of Object.entries(model.materials)) materials[name] = new THREE.MeshStandardMaterial({ color: new THREE.Color(...rgba.slice(0, 3)), roughness: .45, metalness: .12 });
      if (rollout && !rollout.joints) for (const m of Object.values(materials)) {m.transparent=true;m.opacity=.4;m.depthWrite=false;}
      this.robotMaterials = Object.values(materials);
      const build = b => {
        const origin = new THREE.Group(); origin.position.fromArray(b.pos);
        origin.quaternion.set(b.quat[1], b.quat[2], b.quat[3], b.quat[0]).normalize();
        const motion = new THREE.Group(); origin.add(motion);
        if (b.joint?.startsWith('joint')) this.armJoints[Number(b.joint.slice(5)) - 1] = motion;
        if (b.joint?.startsWith('finger')) { this.fingers.push(motion); motion.position.y = .025; }
        for (const visual of b.visuals) {
          const mesh = meshes.get(visual.file).clone(true);
          mesh.traverse(obj => { if (obj.isMesh) { obj.material = materials[visual.material]; obj.castShadow = obj.receiveShadow = true; obj.userData.sharedGeometry = true; } });
          motion.add(mesh);
        }
        b.children.forEach(child => motion.add(build(child))); return origin;
      };
      this.robot = build(model.body); this.scene.add(this.robot);
      this.setStep(this.step); this.render(); host.dataset.ready = 'true';
    }).catch(e => { if (!this.disposed) onError(e); });
  }
  addLabel(text, position, color) {
    const canvas=document.createElement('canvas');canvas.width=128;canvas.height=64;
    const ctx=canvas.getContext('2d');ctx.fillStyle='#ffffffee';ctx.fillRect(0,0,128,64);ctx.font='bold 34px system-ui';ctx.fillStyle=color;ctx.textAlign='center';ctx.fillText(text,64,44);
    const sprite=new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(canvas),depthTest:false}));sprite.position.fromArray(position);sprite.scale.set(.075,.0375,1);this.scene.add(sprite);
  }
  addBox(size, pos, color) {
    const box = new THREE.Mesh(new THREE.BoxGeometry(...size), new THREE.MeshStandardMaterial({ color, roughness: .65, metalness: .08 }));
    box.position.fromArray(pos); box.castShadow = box.receiveShadow = true; this.scene.add(box); return box;
  }
  addPath(points, color) {
    const geometry = new THREE.BufferGeometry().setFromPoints(points.map(p => new THREE.Vector3(...p)));
    const line = new THREE.Line(geometry, new THREE.LineBasicMaterial({ color, transparent: true, opacity: .8 }));
    this.paths.push(line); this.scene.add(line);
  }
  pose(angles) { this.armJoints.forEach((joint, i) => joint.rotation.z = angles[i]); }
  setStep(index) {
    this.step = index;
    const r = this.rollout;
    if (r) {
      const k = Math.min(Math.max(index, 0), r.steps - 1), b = r.boxes[k];
      this.blocks.forEach((block, i) => { block.position.set(b[i * 3], b[i * 3 + 1], .0275); block.rotation.z = b[i * 3 + 2]; });
      this.hand.position.set(r.c[k][0], r.c[k][1], r.c[k][2] ?? .07);
      this.target.position.set(r.des[k][0], r.des[k][1], r.c[k][2] ?? .07);
      this.paths.forEach(path => path.geometry.setDrawRange(0, k + 1));
      this.goalLines.forEach((line,i)=>{
        const goal=this.rollout.goal_spec.goals[r.series?.goal_assignment?.[k]?.[i] ?? i];
        const a=line.geometry.attributes.position;a.setXYZ(0,b[i*3],b[i*3+1],.035);a.setXYZ(1,goal[0],goal[1],.006);a.needsUpdate=true;line.geometry.computeBoundingSphere();
      });
      if (this.home) this.pose(r.joints?.[k] || this.home);
    } else if (this.home) this.pose(this.home);
    this.render();
  }
  animatePreview(seconds) {
    if (this.rollout || !this.home) return;
    const q = [...this.home]; q[0] += .3 * Math.sin(seconds * .7); q[1] -= .15 * Math.sin(seconds * .9); q[3] -= .12 * Math.sin(seconds * .9);
    this.pose(q); this.render();
  }
  setCamera(view) {
    const positions = { oblique: [1.6, -1.65, 1.25], top: [.4, 0, 2.4], front: [1.9, 0, .8] };
    this.camera.position.fromArray(positions[view]); this.controls.target.set(.35, 0, .25); this.controls.update(); this.render();
  }
  resize() {
    if (this.disposed) return;
    const w = this.host.clientWidth, h = this.host.clientHeight;
    if (!w || !h) return;
    this.camera.aspect = w / h; this.camera.updateProjectionMatrix(); this.renderer.setSize(w, h, false); this.render();
  }
  render() { if (!this.disposed) this.renderer.render(this.scene, this.camera); }
  dispose() {
    this.disposed = true; this.resizeObserver.disconnect(); this.controls.dispose();
    this.scene.traverse(obj => {
      if (!obj.userData.sharedGeometry) { obj.geometry?.dispose(); if (obj.material) (Array.isArray(obj.material) ? obj.material : [obj.material]).forEach(m => {m.map?.dispose();m.dispose();}); }
    });
    this.robotMaterials?.forEach(m => {m.map?.dispose();m.dispose();}); this.renderer.dispose(); this.renderer.forceContextLoss(); this.host.replaceChildren();
  }
}
