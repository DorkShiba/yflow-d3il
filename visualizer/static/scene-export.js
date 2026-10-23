import {Scene3D} from './scene3d.js?v=light-1';
const BLOCKS=['#b64b3b','#36836b','#8062a4'];
const sample=(r,t)=>Math.min(Math.max(Math.floor(t/r.dt+1e-9)-1,0),r.steps-1);
function trajectory(ctx,r,k,rect,box){
 const [x,y,w,h]=rect,[x0,y0,x1,y1]=box,scale=Math.min((w-50)/(x1-x0),(h-40)/(y1-y0));
 const xy=p=>[x+w/2+(p[0]-(x0+x1)/2)*scale,y+h/2-(p[1]-(y0+y1)/2)*scale];
 r.goal_spec?.goals.forEach((g,j)=>{const [gx,gy]=xy(g);ctx.strokeStyle=BLOCKS[j];ctx.lineWidth=2;ctx.beginPath();ctx.arc(gx,gy,r.goal_spec.threshold*scale,0,Math.PI*2);ctx.stroke();ctx.fillStyle=BLOCKS[j];ctx.font='13px Arial';ctx.fillText('G'+(j+1),gx+8,gy-8)});
 const paths=[[r.c,'#25669b'],[r.des,'#697680']];for(let j=0;j<r.boxes[0].length;j+=3)paths.push([r.boxes.map(b=>b.slice(j,j+2)),BLOCKS[j/3]]);
 for(const [points,color]of paths){ctx.strokeStyle=color;ctx.lineWidth=2;ctx.beginPath();points.slice(0,k+1).forEach((p,i)=>{const[a,b]=xy(p);i?ctx.lineTo(a,b):ctx.moveTo(a,b)});ctx.stroke();const[a,b]=xy(points[k]);ctx.fillStyle=color;ctx.beginPath();ctx.arc(a,b,4,0,Math.PI*2);ctx.fill()}
}
function loadImage(url){return new Promise((resolve,reject)=>{const image=new Image();image.onload=()=>resolve(image);image.onerror=()=>reject(Error('녹화 프레임을 불러오지 못했습니다.'));image.src=url})}
function jpeg(canvas){return new Promise((resolve,reject)=>canvas.toBlob(blob=>blob?resolve(blob):reject(Error('영상 프레임 생성 실패')),'image/jpeg',.93))}
export async function captureScenes(job,panels,box,api,progress){
 const {width,height,cell,cols,count,fps}=job.capture,[cw,ch]=cell,spec=job.spec;
 const canvas=document.createElement('canvas');canvas.width=width;canvas.height=height;const ctx=canvas.getContext('2d');
 const views=[];
 try{
  for(const p of panels){
   if(p.view!=='scene'){views.push(null);continue}
   const host=document.createElement('div');host.className='export-host';host.style.width=cw+'px';host.style.height=(ch-90)+'px';document.body.append(host);
   const scene=new Scene3D(host,{rollout:p.r,task:p.job.spec.task});views.push(scene);await scene.ready;
   if(!scene.robot)throw Error('3D 모델이 준비되지 않았습니다.');
   scene.renderer.setPixelRatio(1);scene.resize();
   if(p.scene){scene.camera.position.copy(p.scene.camera.position);scene.camera.quaternion.copy(p.scene.camera.quaternion);scene.controls.target.copy(p.scene.controls.target);scene.controls.update()}
  }
  for(let i=0;i<count;i++){
   const t=spec.start+i/fps*spec.speed;ctx.fillStyle='#fff';ctx.fillRect(0,0,width,height);
   for(let j=0;j<panels.length;j++){
    const p=panels[j],r=p.r,k=sample(r,t),x=j%cols*cw,y=Math.floor(j/cols)*ch;
    ctx.save();ctx.beginPath();ctx.rect(x,y,cw,ch);ctx.clip();
    if(views[j]){views[j].setStep(k);ctx.drawImage(views[j].renderer.domElement,x,y+48,cw,ch-90)}
    else if(p.view==='trajectory')trajectory(ctx,r,k,[x,y+48,cw,ch-90],box);
    else{const image=await loadImage(`/api/jobs/${p.job.id}/frames/${r.key}/${k}`);ctx.drawImage(image,x,y+48,cw,ch-90)}
    ctx.fillStyle='#20252b';ctx.font='600 18px Arial, sans-serif';ctx.fillText(p.label,x+20,y+26,cw-40);
    ctx.font='12px Arial, sans-serif';ctx.fillStyle='#59636e';ctx.fillText(`${t.toFixed(2)} s  |  ${k+1} / ${r.steps}`,x+20,y+44);
    const distance=r.series?.mean_distance_mm?.[k];const suffix=distance==null?'':`  |  Goal ${r.distance_kind==='3D'?'3D':'XY'} ${distance.toFixed(1)} mm`;
    ctx.fillText((r.joints?'Recorded joints':'No joint record - reference pose')+suffix,x+20,y+ch-16,cw-40);
    ctx.strokeStyle='#d7dce1';ctx.lineWidth=1;ctx.strokeRect(x+.5,y+.5,cw-1,ch-1);ctx.restore();
   }
   const blob=await jpeg(canvas);await api(`/api/jobs/${job.id}/export-frames/${i}`,{method:'POST',headers:{'Content-Type':'image/jpeg'},body:blob});
   progress(`프레임 저장 ${i+1} / ${count} (${Math.round((i+1)/count*100)}%)`);
  }
  await api(`/api/jobs/${job.id}/encode`,{method:'POST'});progress('영상 인코딩 중입니다. 완료 후 작업과 결과에서 다운로드하세요.');
 }catch(e){await api(`/api/jobs/${job.id}/cancel`,{method:'POST'}).catch(()=>{});throw e}
 finally{for(const view of views)if(view){const host=view.host;view.dispose();host.remove()}}
}
