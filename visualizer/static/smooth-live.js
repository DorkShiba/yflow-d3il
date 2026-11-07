// Unfiltered backward finite differences at the original 35 ms control period.
export function smoothLimits(rollouts){
 const max=(key,min)=>Math.max(min,...rollouts.flatMap(r=>(r[key]||[]).filter(v=>v!==null)))*1.05;
 return {jerk:max('jerk_xy',1),seam:max('seam_dv',.01)};
}
export function drawSmoothness(ctx,r,time,rect,limits,color='#28679c'){
 const [ox,oy,w,h]=rect;ctx.save();ctx.translate(ox,oy);ctx.scale(w/640,h/260);
 ctx.fillStyle='#fff';ctx.fillRect(0,0,640,260);
 const k=Math.min(Math.max(Math.floor(time/r.dt+1e-9)-1,0),r.steps-1),finished=time>=r.steps*r.dt;
 const value=r.jerk_xy?.[k];const window=(r.jerk_xy||[]).slice(Math.max(0,k-Math.round(1/r.dt)+1),k+1).filter(v=>v!==null);
 const rms=window.length?Math.sqrt(window.reduce((s,v)=>s+v*v,0)/window.length):null;
 ctx.fillStyle='#202b36';ctx.font='600 17px Arial';
 ctx.fillText(`Hand XY jerk  ${!finished&&value!=null?value.toFixed(1):'—'} m/s³`,20,23);
 ctx.font='13px Arial';ctx.fillStyle=color;
 ctx.fillText(`Last 1 s RMS: ${!finished&&rms!=null?rms.toFixed(1):'—'}   |   Episode RMS: ${r.jerk_rms?.toFixed(1)??'—'}`,20,44);
 const end=Math.max(2,time),start=end-2,left=58,width=560;
 const tx=t=>left+(t-start)/2*width;
 for(const [key,top,height,max,label]of [['jerk_xy',65,86,limits.jerk,'m/s³'],['seam_dv',184,43,limits.seam,'m/s']]){
  const py=v=>top+height-v/max*height;
  ctx.font='11px Arial';ctx.textAlign='right';ctx.fillStyle='#596979';
  for(const f of [0,.5,1]){const y=py(f*max);ctx.fillText((f*max).toFixed(key==='seam_dv'?2:0),left-7,y+4);ctx.strokeStyle='#e2e8ee';ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(left+width,y);ctx.stroke()}
  ctx.textAlign='left';ctx.fillText(label,5,top-5);
  ctx.save();ctx.beginPath();ctx.rect(left,top,width,height);ctx.clip();
  for(let i=1;i<r.steps;i++){const t=(i+1)*r.dt;if(t<start||t>end||t>time)continue;if(r.plan_starts?.[i]){ctx.strokeStyle='#acb7c3';ctx.setLineDash([3,3]);ctx.beginPath();ctx.moveTo(tx(t),top);ctx.lineTo(tx(t),top+height);ctx.stroke();ctx.setLineDash([])}}
  ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=1.8;ctx.beginPath();let active=false;
  (r[key]||[]).forEach((v,i)=>{const t=(i+1)*r.dt;if(v==null||t<start||t>end||t>time){active=false;return}const x=tx(t),y=py(v);if(key==='seam_dv'){ctx.moveTo(x,py(0));ctx.lineTo(x,y)}else{active?ctx.lineTo(x,y):ctx.moveTo(x,y);active=true}});ctx.stroke();
  const current=r[key]?.[k];if(!finished&&current!=null){ctx.beginPath();ctx.arc(tx((k+1)*r.dt),py(current),3.5,0,Math.PI*2);ctx.fill()}
  ctx.strokeStyle='#28313b';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(tx(time),top);ctx.lineTo(tx(time),top+height);ctx.stroke();ctx.restore();
 }
 ctx.fillStyle='#334557';ctx.font='13px Arial';ctx.fillText('Chunk boundary command Δv   ·   dashed lines = new chunk',20,173);
 ctx.font='11px Arial';ctx.fillStyle='#667483';ctx.fillText(`${start.toFixed(1)} s`,left,244);ctx.textAlign='right';ctx.fillText(`${end.toFixed(1)} s`,left+width,244);ctx.textAlign='left';
 ctx.fillText('Same axes across methods · 2 s window · unfiltered · lower = smoother',20,258);
 ctx.restore();
}
