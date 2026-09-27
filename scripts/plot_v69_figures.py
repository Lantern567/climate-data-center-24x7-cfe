from pathlib import Path
import sys
import json
import hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tmp/plot_v51_packages'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap

DATA=ROOT/'output/commissioning_learning_20260919'
OUT=ROOT/'output/figures/paper_v69'
QA=ROOT/'output/rewrite_v69'
OUT.mkdir(parents=True,exist_ok=True)
QA.mkdir(parents=True,exist_ok=True)
s=pd.read_csv(DATA/'commissioning_global_results.csv')
contrib=pd.read_csv(DATA/'technology_contributions.csv').set_index('state')
assert json.loads((DATA/'independent_validation.json').read_text())['passed']
BLUE,ORANGE,TEAL,PURPLE='#244B68','#C45D32','#6F9C91','#7C658C'
INK,GRAY,GRID='#24282D','#89959E','#E6E8EB'
LC=[GRAY,BLUE,ORANGE]
MC=[BLUE,ORANGE,TEAL]
LEARNS=['Low','Medium','High']
MARKERS=['s','o','^']
MODELS=['ACCESS-CM2','GFDL-ESM4','MIROC6']
SSPS=['ssp126','ssp245','ssp585']
SSL=['1-2.6','2-4.5','5-8.5']
STATES=[m+'_'+p for m in MODELS for p in SSPS]
REP='ACCESS-CM2_ssp245'
main=s[s.learning.isin(LEARNS)&s.doublings.eq(2)]
end=main[main.year.eq(2050)].set_index(['state','learning'])
base=main[main.year.eq(2030)&main.learning.eq('Medium')].set_index('state')
plt.rcParams.update({'font.family':'Arial','font.size':7.1,'axes.labelsize':7,
    'xtick.labelsize':6.4,'ytick.labelsize':6.4,'legend.fontsize':6.3,
    'axes.linewidth':.55,'xtick.major.width':.5,'ytick.major.width':.5,
    'xtick.major.size':2.2,'ytick.major.size':2.2,'legend.frameon':False,
    'axes.unicode_minus':False,'text.parse_math':False,'text.color':INK,
    'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,'pdf.fonttype':42})
cmap=LinearSegmentedColormap.from_list('retained_teal',['#F7F5EF','#B4D1C9','#386F72'])

def series(state=REP,learning='High'):
    return main[main.state.eq(state)&main.learning.eq(learning)].sort_values('year')

def frame(ax,axis='y'):
    ax.spines[['top','right']].set_visible(False)
    if axis:
        ax.set_axisbelow(True)
        ax.grid(axis=axis,color=GRID,lw=.45)

def letter(ax,l):
    ax.text(-.16,1.045,'('+l+')',transform=ax.transAxes,fontsize=9,weight='bold')

def combos(ax,xpos=-.24,fontsize=6):
    ax.set_yticks(range(9),SSL*3)
    ax.set_ylim(8.6,-.6)
    ax.tick_params(axis='y',length=0)
    for i,col in enumerate(MC):
        ax.axhspan(i*3-.45,i*3+2.45,color=col,alpha=.045,lw=0,zorder=0)
        ax.text(xpos,i*3+1,['ACCESS','GFDL','MIROC'][i],
            transform=ax.get_yaxis_transform(),rotation=90,ha='right',va='center',fontsize=fontsize,color=col)

def save(fig,name):
    fig.savefig(OUT/(name+'.pdf'),facecolor='white')
    fig.savefig(OUT/(name+'.png'),dpi=400,facecolor='white')
    fig.savefig(QA/(name+'_preview.png'),dpi=175,facecolor='white')
    plt.close(fig)

# 图4复用原版的不等宽六面板，左上价格响应面与右上三条学习曲线组成面板a。
fig=plt.figure(figsize=(190/25.4,245/25.4))
for l,x,y in [('a',.043,.973),('b',.043,.601),('c',.595,.601),
              ('d',.043,.345),('e',.043,.169),('f',.595,.345)]:
    fig.text(x,y,'('+l+')',fontsize=9,weight='bold',ha='left',va='bottom')
ax=fig.add_axes([.12,.715,.495,.242])
r=base.loc[REP]
xx,yy=np.meshgrid(np.linspace(0,60,151),np.linspace(0,65,151))
wp=(r.pv_capital+r.wind_capital)/r.cost
wb=r.battery_capital/r.cost
zz=wp*xx+wb*yy
mesh=ax.pcolormesh(xx,yy,zz,cmap=cmap,vmin=0,vmax=50,shading='auto',rasterized=True)
cs=ax.contour(xx,yy,zz,levels=[10,20,30,40],colors='white',linewidths=.8,linestyles='--')
ax.clabel(cs,fmt=lambda v:f'{v:.0f}%',fontsize=6,inline=True)
for learn,col,m,pr,wr,br in zip(LEARNS,LC,MARKERS,[.1,.2,.25],[.05,.1,.15],[.1,.2,.3]):
    t=np.arange(21)/10
    px=100*(1-((1-pr)**t*r.pv_capital+(1-wr)**t*r.wind_capital)/(r.pv_capital+r.wind_capital))
    py=100*(1-(1-br)**t)
    ax.plot(px,py,color=col,lw=1.3,marker='o',ms=2.2,mec='white',mew=.25)
    ax.plot(px[-1],py[-1],marker=m,color=col,ms=5,mec='white',mew=.65)
    ax.annotate('2050 '+learn[0],(px[-1],py[-1]),xytext=(6,0),textcoords='offset points',color=col,fontsize=6,va='center')
ax.scatter(0,0,marker='D',s=22,facecolor='white',edgecolor=INK,zorder=5,clip_on=False)
ax.text(.025,.965,'2030 capacity held fixed',transform=ax.transAxes,va='top',fontsize=6.5)
ax.set(xlim=(0,60),ylim=(0,65),xticks=[0,20,40,60],yticks=[0,20,40,60],
    xlabel='Wind–PV CAPEX reduction (%)',ylabel='Battery CAPEX reduction (%)')
frame(ax,None)
ca=fig.add_axes([.12,.674,.495,.008])
cb=fig.colorbar(mesh,cax=ca,orientation='horizontal',ticks=[0,10,20,30,40,50])
cb.set_label('Annualized cost reduction at fixed capacity (%)',fontsize=6.6,labelpad=3)
cb.outline.set_linewidth(.45)
for j,(learn,col) in enumerate(zip(LEARNS,LC)):
    ax=fig.add_axes([.735,.891-j*.099,.222,.066])
    q=series(learning=learn)
    band=main[main.learning.eq(learn)].groupby('year').saving_pct.agg(['min','max'])
    ax.fill_between(band.index,100-band['max'],100-band['min'],color=col,alpha=.2,lw=0)
    ax.plot(q.year,100-q.saving_pct,color=col,lw=1.25)
    ax.scatter(2050,100-q.saving_pct.iloc[-1],color=col,s=12,zorder=4)
    ax.text(.03,.17,learn+' learning',color=col,transform=ax.transAxes,fontsize=6.1)
    ax.annotate(f'{100-q.saving_pct.iloc[-1]:.1f}',(2050,100-q.saving_pct.iloc[-1]),xytext=(4,0),textcoords='offset points',fontsize=6,color=col,va='center')
    ax.set(xlim=(2030,2054),ylim=(60,105),yticks=[60,80,100],xticks=[2030,2040,2050])
    if j<2:ax.set_xticklabels([])
    else:ax.set_xlabel('Commissioning year',fontsize=6.3,labelpad=3)
    if j==1:ax.set_ylabel('Annualized cost (2030 = 100)',fontsize=6.4,labelpad=7)
    frame(ax)
fig.text(.846,.975,'Optimized capacity',ha='center',fontsize=6.5)
fig.legend(handles=[Line2D([],[],color=c,marker=m,lw=1,ms=3,label=l+' learning ('+l[0]+')')
    for l,c,m in zip(LEARNS,LC,MARKERS)],loc='center',bbox_to_anchor=(.53,.633),ncol=3,fontsize=6.4,columnspacing=1.2)

ax=fig.add_axes([.12,.444,.395,.143])
years=np.arange(2030,2051)
rates=[[.05,.1,.15],[.1,.2,.25],[.1,.2,.3]]
mat=np.array([100*(1-(1-rate)**((years-2030)/10)) for group in rates for rate in group])
mesh=ax.pcolormesh(np.arange(22)+2029.5,np.arange(10),mat,cmap=cmap,vmin=0,vmax=60,edgecolor='white',lw=.35)
ax.invert_yaxis()
ax.set_yticks(np.arange(9)+.5,[tech+' '+l[0] for tech in ['Wind','PV','Battery'] for l in LEARNS])
ax.set_xticks([2030,2035,2040,2045,2050])
ax.set_xlabel('Commissioning year',fontsize=6.5)
ax.tick_params(length=0,pad=3)
for sp in ax.spines.values():sp.set_visible(False)
for i,row in enumerate(mat):ax.text(2051.2,i+.5,f'{row[-1]:.1f}',va='center',fontsize=5.9)
ax.text(2051.2,-.3,'2050',fontsize=5.9)
ca=fig.add_axes([.12,.397,.395,.008])
cb=fig.colorbar(mesh,cax=ca,orientation='horizontal',ticks=[0,20,40,60])
cb.set_label('Equipment CAPEX reduction vs 2030 (%)',fontsize=6.5,labelpad=3)
cb.outline.set_linewidth(.45)

ax=fig.add_axes([.697,.436,.26,.145])
for i,st in enumerate(STATES):
    vals=[end.loc[(st,l),'equivalent_annual_growth_pct'] for l in ['Medium','High']]
    ax.plot(vals,[i,i],color=MC[i//3],alpha=.45,lw=.9)
    for v,m in zip(vals,['o','D']):ax.scatter(v,i,color=MC[i//3],marker=m,s=15,zorder=3)
combos(ax,xpos=-.24)
ax.set(xlim=(1.25,2.01),xticks=[1.3,1.5,1.7,1.9],xlabel='Budget-supported annual growth (%)')
ax.xaxis.label.set_fontsize(6.4)
frame(ax,'x')
ax.text(.5,1.06,'2050 · fixed annualized budget',transform=ax.transAxes,ha='center',fontsize=6.1,color=GRAY)
ax.legend(handles=[Line2D([],[],marker=m,color=INK,ls='',ms=3,label=l) for m,l in [('o','Medium'),('D','High')]],
    loc='upper center',bbox_to_anchor=(.5,-.24),ncol=2,fontsize=6,handletextpad=.35,columnspacing=1.1)

ax=fig.add_axes([.14,.236,.375,.096])
heat=np.array([series(st).saving_pct.to_numpy() for st in STATES])
mesh=ax.pcolormesh(np.arange(2029.5,2051.5),np.arange(-.5,9.5),heat,cmap=cmap,vmin=0,vmax=35,edgecolor='white',lw=.25)
combos(ax,xpos=-.22,fontsize=5.1)
ax.set_xticks([2030,2040,2050])
ax.set_xlabel('Commissioning year · high learning',fontsize=6.2,labelpad=3)
for sp in ax.spines.values():sp.set_visible(False)
ca=fig.add_axes([.14,.195,.375,.007])
cb=fig.colorbar(mesh,cax=ca,orientation='horizontal',ticks=[0,10,20,30])
cb.set_label('Optimized annualized cost reduction (%)',fontsize=6.1,labelpad=2)
cb.outline.set_linewidth(.45)

ax=fig.add_axes([.12,.055,.395,.100])
for i,year in enumerate([2035,2040,2045,2050]):
    for j,(learn,alpha) in enumerate([('Low',.38),('High',1)]):
        row=series(learning=learn).set_index('year').loc[year]
        cap=(row.pv_capital+row.wind_capital+row.battery_capital)/row.demand_mwh
        op=(row.cost-row.pv_capital-row.wind_capital-row.battery_capital)/row.demand_mwh
        y=i+(j-.5)*.34
        ax.barh(y,cap,height=.29,color=BLUE,alpha=alpha)
        ax.barh(y,op,left=cap,height=.29,color=ORANGE,alpha=alpha)
        ax.text(cap+op+3,y,f'{cap+op:.0f}',va='center',fontsize=5.8)
ax.set(yticks=range(4),yticklabels=[2035,2040,2045,2050],ylim=(3.6,-.6),xlim=(0,305),xticks=[0,100,200,300],xlabel='Annualized supply cost (US$/MWh)')
ax.xaxis.label.set_fontsize(6.3)
frame(ax,None)
ax.legend(handles=[Patch(fc=BLUE,label='Capital'),Patch(fc=ORANGE,label='O&M + delivery')],loc='upper left',bbox_to_anchor=(0,-.27),ncol=2,fontsize=6,handlelength=1,columnspacing=.9)
fig.text(.515,.007,'Light: low learning · Dark: high learning',ha='right',fontsize=5.9,color=GRAY)

ax=fig.add_axes([.655,.153,.302,.163])
for learn,col,m in zip(LEARNS,LC,MARKERS):
    row=end.loc[(REP,learn)]
    vals=[row[f'growth{g}_extra_budget_pct'] for g in [2,5,8]]
    ax.plot([2,5,8],vals,color=col,marker=m,lw=1.3,ms=3.5,label=learn)
ax.set(xticks=[2,5,8],xlim=(1.5,8.5),ylim=(0,340),yticks=[0,100,200,300],xlabel='Equivalent annual business growth (%)',ylabel='Additional annualized budget (%)')
ax.xaxis.label.set_fontsize(6.3)
ax.yaxis.label.set_fontsize(6.5)
ax.legend(loc='upper left',fontsize=6)
frame(ax)
fig.text(.657,.104,'High learning · nine-climate range',fontsize=6.3,weight='bold')
for i,g in enumerate([2,5,8]):
    vals=end.xs('High',level='learning')[f'growth{g}_extra_budget_pct']
    fig.text(.657,.084-i*.019,f'{g}% / year',fontsize=6.2)
    fig.text(.957,.084-i*.019,f'+{vals.min():.2f}–{vals.max():.2f}%',ha='right',fontsize=6.2,color=ORANGE)
fig.text(.657,.016,'Budget: 2030 baseline business scale\nSame-climate annualized cost',fontsize=5.8,color=GRAY,linespacing=1.4)
save(fig,'figure4_future_technology')

# 图7保留原版四个散点面板、模型颜色和情景排列。
fig,axs=plt.subplots(2,2,figsize=(180/25.4,160/25.4))
fig.subplots_adjust(left=.13,right=.965,bottom=.13,top=.89,hspace=.48,wspace=.43)
fig.legend(handles=[Line2D([],[],marker='o',color=c,ls='',ms=4,label=m) for m,c in zip(MODELS,MC)],loc='upper center',bbox_to_anchor=(.54,.99),ncol=3)
for ax,key,label,l in zip(axs.ravel(),['pv_mw','wind_mw','battery_mwh','configuration_saving_pp'],
    ['PV capacity change (%)','Wind capacity change (%)','Battery capacity change (%)','Additional saving from reoptimization (pp)'],'abcd'):
    for mi,(model,col) in enumerate(zip(MODELS,MC)):
        for si,ssp in enumerate(SSPS):
            for li,(learn,mark) in enumerate(zip(LEARNS,MARKERS)):
                st=model+'_'+ssp
                row=end.loc[(st,learn)]
                val=row[key] if key=='configuration_saving_pp' else 100*(row[key]/base.loc[st,key]-1)
                ax.scatter(si+(mi-1)*.23+(li-1)*.06,val,color=col,marker=mark,s=19,edgecolors='white',linewidths=.3,zorder=4)
    ax.set(xticks=range(3),xticklabels=['SSP'+x for x in SSL],xlim=(-.48,2.48),ylabel=label)
    if key!='configuration_saving_pp':
        ax.axhline(0,color=GRAY,lw=.7,ls='--')
        ax.set_ylim(-12,12)
    else:ax.set_ylim(0,.25)
    frame(ax)
    letter(ax,l)
fig.legend(handles=[Line2D([],[],marker=m,color=INK,ls='',ms=4,label=l+' learning') for m,l in zip(MARKERS,LEARNS)],loc='lower center',bbox_to_anchor=(.54,.025),ncol=3)
save(fig,'figure7_cost_capacity')

# 图8复用三乘三气候分面，曲线仍按业务增长目标区分。
fig,axs=plt.subplots(3,3,figsize=(180/25.4,185/25.4),sharex=True,sharey=True)
fig.subplots_adjust(left=.11,right=.975,bottom=.075,top=.91,hspace=.40,wspace=.26)
for i,ssp in enumerate(SSPS):
    for j,model in enumerate(MODELS):
        ax=axs[i,j]
        q=series(model+'_'+ssp,'High')
        for g,col in zip([2,5,8],LC):ax.plot(q.year,100+q[f'growth{g}_extra_budget_pct'],color=col,lw=1.5)
        ax.axhline(100,color=GRAY,ls='--',lw=.6)
        ax.set(xlim=(2030,2050),xticks=[2030,2040,2050],ylim=(0,350),yticks=[0,100,200,300])
        ax.set_title(model+'\nSSP'+SSL[i],fontsize=7,pad=6)
        if i==2:ax.set_xlabel('Commissioning year')
        if j==0:ax.set_ylabel('Annualized cost (2030 = 100)')
        frame(ax)
        letter(ax,chr(97+i*3+j))
fig.legend(handles=[Line2D([],[],color=c,lw=1.5,label=f'{g}% annual business growth') for c,g in zip(LC,[2,5,8])],loc='upper center',bbox_to_anchor=(.54,.99),ncol=3,fontsize=6.5)
save(fig,'figure8_annual_spending')

# 图9保留原版折线、分组点图、正负贡献和时间曲线的四面板结构。
fig,axs=plt.subplots(2,2,figsize=(190/25.4,175/25.4))
fig.subplots_adjust(left=.12,right=.96,bottom=.105,top=.90,hspace=.38,wspace=.44)
fig.legend(handles=[Line2D([],[],color=c,lw=1.5,label=l+' learning') for c,l in zip(LC,LEARNS)],loc='upper center',bbox_to_anchor=(.53,.99),ncol=3)
ax=axs[0,0]
for learn,col,m in zip(LEARNS,LC,MARKERS):
    q=s[s.state.eq(REP)&s.year.eq(2050)&s.learning.eq(learn)].sort_values('doublings')
    ax.plot(q.doublings,q.supported_business_increase_pct,color=col,marker=m,ms=4,lw=1.5)
ax.set(xticks=[1,2,3],xticklabels=['2×','4×','8×'],ylim=(0,80),xlabel='Industry deployment in 2050 / 2030',ylabel='Budget-supported business increase (%)')
ax.text(.02,.96,'ACCESS-CM2 · SSP2-4.5',transform=ax.transAxes,va='top',fontsize=6.5)
ax=axs[0,1]
for i,st in enumerate(STATES):
    vals=[end.loc[(st,l),'growth5_extra_budget_pct'] for l in LEARNS]
    ax.plot([min(vals),max(vals)],[i,i],color=GRID,lw=1.5)
    for val,col,m in zip(vals,LC,MARKERS):ax.scatter(val,i,color=col,marker=m,s=20,zorder=4)
combos(ax)
ax.set(xlim=(70,145),xticks=[80,100,120,140],xlabel='Additional annualized budget (%)')
ax.text(.98,1.045,'2050 · 5% annual business growth',ha='right',transform=ax.transAxes,fontsize=6.5)
ax=axs[1,0]
for i,g in enumerate([2,5,8]):
    row=end.loc[(REP,'High')]
    factor=(1+g/100)**20
    gross=100*(factor-1)
    wind=-factor*contrib.loc[REP,'renewables_saving_pp']
    batt=-factor*contrib.loc[REP,'battery_saving_pp']
    net=gross+wind+batt
    assert np.isclose(net,row[f'growth{g}_extra_budget_pct'])
    ax.axvspan(i-.36,i+.36,color=GRID,alpha=.4,lw=0)
    ax.bar(i,gross,width=.42,color='#A4B1BB')
    ax.bar(i,wind,width=.42,color=TEAL)
    ax.bar(i,batt,bottom=wind,width=.42,color=PURPLE)
    ax.scatter(i,net,marker='D',color=INK,s=24,zorder=4)
    ax.annotate(f'+{net:.1f}%',(i,net),xytext=(0,7),textcoords='offset points',ha='center',fontsize=6.4)
ax.axhline(0,color=GRAY,lw=.8)
ax.set(xticks=range(3),xticklabels=['2%','5%','8%'],ylim=(-180,430),xlabel='Equivalent annual business growth',ylabel='Change relative to 2030 budget (%)')
ax.legend(handles=[Patch(fc='#A4B1BB',label='Business growth'),Patch(fc=TEAL,label='Wind–PV learning'),Patch(fc=PURPLE,label='Battery learning'),Line2D([],[],marker='D',color=INK,ls='',ms=4,label='Net budget increase')],loc='lower left',bbox_to_anchor=(-.025,1.01),ncol=2,fontsize=5.8,columnspacing=.8,handlelength=1)
ax=axs[1,1]
for learn,col in zip(LEARNS,LC):
    for st in STATES:
        q=series(st,learn)
        ax.plot(q.year,q.supported_business_increase_pct,color=col,lw=.65,alpha=.22)
    q=series(REP,learn)
    ax.plot(q.year,q.supported_business_increase_pct,color=col,lw=1.7)
    ax.text(2050.3,q.supported_business_increase_pct.iloc[-1],f'{q.supported_business_increase_pct.iloc[-1]:.1f}%',color=col,fontsize=6.3,va='center')
ax.set(xlim=(2030,2054),ylim=(0,55),xticks=[2030,2035,2040,2045,2050],xlabel='Commissioning year',ylabel='Budget-supported business increase (%)')
ax.text(.03,.96,'Thin: nine climates\nThick: ACCESS-CM2 · SSP2-4.5',transform=ax.transAxes,va='top',fontsize=6,color=GRAY,linespacing=1.5)
for ax,l in zip(axs.ravel(),'abcd'):
    frame(ax,'x' if l=='b' else 'y')
    letter(ax,l)
save(fig,'figure9_business_growth')
audit={'source_sha256':hashlib.sha256((DATA/'commissioning_global_results.csv').read_bytes()).hexdigest(),
       'figures':[4,7,8,9],'layout_reference':'v67 / v66 retained panels',
       'surface':'Exact repricing of fixed 2030 capacities; optimized paths shown separately',
       'decomposition_identity':'passed','scientific_data':'unchanged from v68'}
(QA/'figure_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
print(json.dumps(audit))
