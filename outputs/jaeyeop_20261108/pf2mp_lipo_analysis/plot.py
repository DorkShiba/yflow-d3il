import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np,json,pathlib
from matplotlib.patches import Circle
p=pathlib.Path('/root/d3il-work/pf2mp_lipo_analysis'); a=np.load(p/'timeseries.npz');g=np.array([[.3575,.276],[.525,.4535],[.6925,.276]])
colors=['#c74242','#26804b','#386fc7']
def get(c,s,n,k):return a[f'inserting_ctx{c}_s{s}_{n:02d}_{k}']
fig,axs=plt.subplots(2,3,figsize=(15,9),layout='constrained')
for row,(c,s) in enumerate([(17,0),(18,2)]):
 for col,n in enumerate((8,9)):
  ax=axs[row,col];h=get(c,s,n,'next_measured_state');b=get(c,s,n,'boxes_xy_m');ax.plot(h[:,0]*100,h[:,1]*100,color='#888',lw=.7,alpha=.7,label='Measured hand')
  for j in range(3):
   ax.plot(b[:,j,0]*100,b[:,j,1]*100,color=colors[j],lw=1.3,label=f'Box {j+1}');ax.scatter(* (b[-1,j]*100),color=colors[j],s=25);ax.scatter(*(b[0,j]*100),color=colors[j],s=20,marker='x');ax.add_patch(Circle(g[j]*100,1,fill=False,color=colors[j],ls=':'))
  ax.set(title=f'Context {c}, seed {s}: '+('PF2MP' if n==8 else 'PF2MP + LiPo'),xlabel='X (cm)',ylabel='Y (cm)');ax.set_aspect('equal');ax.grid(alpha=.15);ax.legend(fontsize=7,loc='upper right')
 ax=axs[row,2]
 for n,ls in ((8,'--'),(9,'-')):
  dist=np.linalg.norm(get(c,s,n,'boxes_xy_m')-g,axis=2)*1000
  for j in range(3):ax.plot(dist[:,j],color=colors[j],ls=ls,lw=1,label=f'Box {j+1}: '+('PF' if n==8 else 'PF+LiPo'))
 ax.axhline(10,color='k',lw=.6,ls=':');ax.set(title='XY goal distance (3D success rule)',xlabel='Control step',ylabel='Distance (mm)',yscale='log');ax.legend(fontsize=7,ncol=2);ax.grid(alpha=.15)
fig.savefig(p/'trajectories.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained');c,s=17,0
for k,label in [('next_command_state','Command'),('next_measured_state','Measured hand'),('boxes_xy_m','Maximum box')]:
 x=get(c,s,8,k);y=get(c,s,9,k);n=min(len(x),len(y));d=np.linalg.norm(x[:n]-y[:n],axis=-1)
 if d.ndim==2:d=d.max(1)
 ax[0,0].plot(d[:120]*1000,label=label)
ax[0,0].set(title='Small changes amplify early (ctx17, seed0)',xlabel='Control step',ylabel='PF vs PF+LiPo difference (mm)');ax[0,0].legend()
for n,color,label in ((8,'#377eb8','PF2MP'),(9,'#e76f51','PF2MP + LiPo')):
 h=get(c,s,n,'next_measured_state');q=get(c,s,n,'next_command_state');act=get(c,s,n,'safe_action')
 ax[0,1].plot(np.linalg.norm(q-h[:,:2],axis=1)*1000,color=color,label=label,lw=.8)
 jerk=np.linalg.norm(np.diff(act,n=2,axis=0),axis=1)/.035**3
 ax[1,0].plot(np.arange(len(jerk))+2,np.convolve(jerk,np.ones(40)/40,mode='same'),color=color,label=label,lw=.9)
 b=get(c,s,n,'boxes_xy_m');travel=np.linalg.norm(np.diff(b,axis=0),axis=2).sum(1)
 ax[1,1].plot(np.arange(len(travel))+1,np.cumsum(travel)*1000,color=color,label=label,lw=1)
ax[0,1].set(title='Command-to-hand tracking gap',xlabel='Control step',ylabel='Gap (mm)')
ax[1,0].set(title='Executed command jerk (40-step moving average)',xlabel='Control step',ylabel='Jerk (m/s³)')
ax[1,1].set(title='Cumulative box movement (sum of three boxes)',xlabel='Control step',ylabel='Path length (mm)')
for axes in ax.flat:axes.grid(alpha=.2);axes.legend(fontsize=8)
fig.savefig(p/'divergence.png',dpi=160)
# Additional numeric diagnostics; XY is not the 3D success distance.
metrics=json.loads((p/'metrics.json').read_text());extra=[]
for c in (17,18,19):
 for s in (0,1,2):
  r=next(r for r in metrics if (r['task'],r['ctx'],r['seed'],r['combo'])==('inserting',c,s,'I09'))
  b=get(c,s,9,'boxes_xy_m');d=np.linalg.norm(b-g,axis=2)*1000
  extra.append({'ctx':c,'seed':s,'final_xy_goal_distance_mm':d[-1].tolist(),'first_step_xy_under_10mm':[int(np.where(d[:,j]<=10)[0][0]) if np.any(d[:,j]<=10) else None for j in range(3)]})
(p/'goal_diagnostics.json').write_text(json.dumps(extra,indent=2))
print('plots saved')
