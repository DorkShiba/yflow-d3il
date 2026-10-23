export const COLORS = ['#28679c','#c16b25','#377d65','#866393'];
export const METRICS = [
 ['mean_distance_mm','목표까지 평균 거리','mm','Pushing: 각 블록의 가까운 목표까지 평균. Inserting: 지정 목표까지 평균.'],
 ['tracking_mm','목표 손 ↔ 실제 손','mm','명령된 손 위치와 실제 손 위치의 XY 차이'],
 ['action_delta_mm','행동 변화량','mm','연속 XY 명령의 차이. 물리적 jerk가 아닙니다. 첫 전이는 없음.'],
 ['action_mm','행동 크기','mm','한 제어 스텝의 XY 위치 명령 크기'],
 ['speed_m_s','손 이동 속도','m/s','인접한 실제 손 XY 위치의 차이 / 제어 간격'],
 ['joint_speed_rad_s','관절 속도 노름','rad/s','관절 기록이 있는 경우에만 표시. 기록 없음은 0이 아닙니다.']
];
export function drawChart(canvas, panels, key, unit, currentTime, seek) {
 const w=Math.max(canvas.clientWidth,240),h=180,dpr=Math.min(devicePixelRatio,2);
 if(canvas.width!==Math.round(w*dpr)||canvas.height!==h*dpr){canvas.width=Math.round(w*dpr);canvas.height=h*dpr}
 const c=canvas.getContext('2d');c.setTransform(dpr,0,0,dpr,0,0);c.fillStyle='#fff';c.fillRect(0,0,w,h);
 const x0=52,y0=18,pw=w-68,ph=130;
 const end=Math.max(.001,...panels.map(p=>p.r.steps*p.r.dt));
 let max=0,available=false;
 for(const p of panels)for(const v of p.r.series?.[key]||[])if(v!==null&&Number.isFinite(v)){available=true;max=Math.max(max,v)}
 max=Math.max(max*1.12,.001);
 c.font='11px system-ui';c.lineWidth=1;
 for(let i=0;i<=4;i++){let y=y0+i*ph/4;c.strokeStyle='#e1e5e9';c.beginPath();c.moveTo(x0,y);c.lineTo(x0+pw,y);c.stroke();c.fillStyle='#58636e';c.textAlign='right';c.fillText((max*(1-i/4)).toFixed(max<1?3:1),x0-7,y+4)}
 c.textAlign='left';c.fillText(unit,4,12);c.fillText('0 s',x0,y0+ph+22);c.textAlign='right';c.fillText(end.toFixed(2)+' s',x0+pw,y0+ph+22);
 panels.forEach((p,index)=>{
  c.strokeStyle=COLORS[index];c.lineWidth=2;c.beginPath();let active=false;
  (p.r.series?.[key]||[]).forEach((v,i)=>{if(v===null||!Number.isFinite(v)){active=false;return}const x=x0+(i+1)*p.r.dt/end*pw,y=y0+ph-v/max*ph;if(active)c.lineTo(x,y);else c.moveTo(x,y);active=true});c.stroke();
 });
 const x=x0+Math.min(currentTime,end)/end*pw;c.strokeStyle='#626c77';c.setLineDash([4,4]);c.beginPath();c.moveTo(x,y0);c.lineTo(x,y0+ph);c.stroke();c.setLineDash([]);
 if(!available){c.fillStyle='#66717d';c.textAlign='center';c.fillText('기록 없음',x0+pw/2,y0+ph/2)}
 canvas.onclick=e=>{const rect=canvas.getBoundingClientRect();seek(Math.max(0,Math.min(end,(e.clientX-rect.left-x0)/pw*end)))};
}
