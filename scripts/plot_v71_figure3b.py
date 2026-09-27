from pathlib import Path
import sys
import json
import hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tmp/plot_v51_packages'))
sys.path.insert(0,str(ROOT/'output/rewrite_v66/python_packages'))
import pymupdf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from scipy.interpolate import PchipInterpolator

DATA=ROOT/'output/greenfield_cfe_20260915'
OUT=ROOT/'output/figures/paper_v71'
QA=ROOT/'output/rewrite_v71'
OUT.mkdir(parents=True,exist_ok=True)
QA.mkdir(parents=True,exist_ok=True)
frontier=pd.read_csv(DATA/'figure3_frontier.csv')
summary=pd.read_csv(DATA/'full_matching_cost_summary.csv')
BLUE,ORANGE,TEAL='#244B68','#C45D32','#6F9C91'
INK,GRAY,GRID='#29363D','#89959E','#E5EAED'
SSPS=['ssp126','ssp245','ssp585']
SHORT=['1-2.6','2-4.5','5-8.5']
COLORS=[TEAL,BLUE,ORANGE]
plt.rcParams.update({'font.family':'Arial','font.size':7,'axes.labelsize':7,
    'xtick.labelsize':6.5,'ytick.labelsize':6.5,'legend.fontsize':6.3,
    'legend.frameon':False,'axes.linewidth':.55,'axes.unicode_minus':False,
    'text.parse_math':False,'text.color':INK,'axes.labelcolor':INK,
    'xtick.color':INK,'ytick.color':INK,'pdf.fonttype':42,'savefig.facecolor':'white'})

def frame(ax,grid='y'):
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(length=2,width=.5,pad=3)
    ax.set_axisbelow(True)
    if grid:ax.grid(axis=grid,color=GRID,lw=.45)

fig=plt.figure(figsize=(396/72,300/72))
fig.text(.045,.96,'b',fontsize=12,weight='bold')
ax=fig.add_axes([.20,.44,.75,.45])
inset=ax.inset_axes([.12,.56,.35,.37])
audit=[]
for j,(ssp,col) in enumerate(zip(SSPS,COLORS)):
    q=frontier[frontier.state.eq('ACCESS-CM2_'+ssp)&frontier.feasible].sort_values('wind_pv_scale')
    x=q.wind_pv_scale.to_numpy()*100
    y=q.annual_cost_usd.to_numpy()/1e9
    k=np.argmin(y)
    r=summary[summary['mode'].eq('S3')&summary.state.eq('ACCESS-CM2_'+ssp)].iloc[0]
    assert np.isclose(x[k],100)
    assert np.isclose(y[k],r.total_annual_cost_usd/1e9,rtol=2e-6)
    # 最低费用点左右分别保形插值，保留真实最优点及两侧变化方向。
    for selection in [slice(0,k+1),slice(k,None)]:
        xx,yy=x[selection],y[selection]
        interp=PchipInterpolator(xx,yy)
        assert np.allclose(interp(xx),yy)
        dense=np.linspace(xx.min(),xx.max(),240)
        smooth=interp(dense)
        assert smooth.min()>=y[k]-1e-9
        ax.plot(dense,smooth,color=col,lw=1.45,solid_capstyle='round',zorder=3)
        inset.plot(dense,smooth,color=col,lw=1.15,solid_capstyle='round',zorder=3)
    ax.scatter(x[k],y[k],marker='D',s=24,color=col,edgecolor='white',lw=.65,zorder=6)
    inset.scatter(x,y,color=col,s=7,edgecolor='white',lw=.25,zorder=4)
    inset.scatter(x[k],y[k],marker='D',s=22,color=col,edgecolor='white',lw=.5,zorder=5)
    audit.append(dict(state='ACCESS-CM2_'+ssp,points=len(q),minimum_capacity_pct=float(x[k]),minimum_cost_bn=float(y[k])))
ax.set(xlim=(75,162),ylim=(180,248),xticks=[80,100,120,140,160],yticks=[180,200,220,240],
    xlabel='Wind–PV capacity / optimum (%)',ylabel='Annual cost (US$ bn/yr)')
frame(ax)
ax.axvline(100,ymin=0,ymax=.39,color=GRAY,ls=(0,(2,3)),lw=.55,alpha=.65,zorder=1)
ax.add_patch(Rectangle((94,184),12,13,fill=False,edgecolor=GRAY,lw=.55,linestyle=(0,(3,2)),zorder=2))
ax.text(.975,.095,'100% hourly matching',transform=ax.transAxes,ha='right',va='bottom',fontsize=7)
inset.set(xlim=(94,106),ylim=(184,197),xticks=[95,100,105],yticks=[185,190,195])
inset.set_title('Near the minimum',fontsize=6.1,pad=4)
inset.tick_params(labelsize=5.8,length=1.5,pad=2)
inset.set_facecolor('white')
frame(inset)
for spine in inset.spines.values():spine.set_edgecolor(GRAY)
ax.legend(handles=[Line2D([],[],color=c,lw=1.6,label='SSP'+s) for c,s in zip(COLORS,SHORT)],
    loc='upper center',bbox_to_anchor=(.5,1.20),ncol=3,fontsize=7,columnspacing=1.4,handlelength=2.2)

# 下方费用分解保留原版坐标、颜色与数值。
bar=fig.add_axes([.20,.12,.75,.17])
for j,ssp in enumerate(SSPS):
    r=summary[summary['mode'].eq('S3')&summary.state.eq('ACCESS-CM2_'+ssp)].iloc[0]
    values=[(r.pv_annual_cost_usd+r.wind_annual_cost_usd)/1e9,r.delivery_annual_cost_usd/1e9,r.battery_annual_cost_usd/1e9]
    left=0
    for val,col in zip(values,[BLUE,ORANGE,TEAL]):
        bar.barh(j,val,left=left,height=.55,color=col,edgecolor='white',lw=.3)
        left+=val
    bar.text(left+2,j,f'{left:.2f}',va='center',fontsize=6.7)
bar.set(yticks=range(3),yticklabels=SHORT,ylim=(2.5,-.5),xlim=(0,225),xticks=[0,100,200],xlabel='Cost at complete-matching optimum (US$ bn/yr)')
bar.set_xlabel(bar.get_xlabel(),fontsize=6.8,labelpad=3)
frame(bar,'x')
fig.legend(handles=[Patch(fc=c,label=l) for c,l in zip([BLUE,ORANGE,TEAL],['Wind–PV','Transmission delivery','Storage'])],
    loc='lower center',bbox_to_anchor=(.56,-.005),ncol=3,fontsize=6.5,columnspacing=.8,handlelength=1)
fig.savefig(OUT/'figure3b_panel.pdf')
fig.savefig(QA/'figure3b_preview.png',dpi=240)
plt.close(fig)

original=pymupdf.open(ROOT/'output/figures/paper_v66/figure3_complete_hourly_matching.pdf')
doc=pymupdf.open()
bounds=original[0].rect
page=doc.new_page(width=bounds.width,height=bounds.height)
target=pymupdf.Rect(0,377,393,677)
for clip in [pymupdf.Rect(0,0,bounds.width,377),pymupdf.Rect(393,377,bounds.width,677),pymupdf.Rect(0,677,bounds.width,bounds.height)]:
    page.show_pdf_page(clip,original,0,clip=clip)
new=pymupdf.open(OUT/'figure3b_panel.pdf')
page.show_pdf_page(target,new,0)
doc.save(OUT/'figure3_complete_hourly_matching.pdf')
page.get_pixmap(matrix=pymupdf.Matrix(4000/bounds.width,4000/bounds.width),alpha=False).save(OUT/'figure3_complete_hourly_matching.png')
(QA/'figure_audit.json').write_text(json.dumps({'input_sha256':hashlib.sha256((DATA/'figure3_frontier.csv').read_bytes()).hexdigest(),
    'panel':'Figure 3b','interpolation':'Separate shape-preserving curves on both sides of sampled minimum',
    'results':audit,'data_changed':False},indent=2),encoding='utf8')
print(json.dumps(audit))
