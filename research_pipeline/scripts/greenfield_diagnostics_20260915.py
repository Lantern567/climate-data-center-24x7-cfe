"""Hourly and cost evidence for the retained six-panel Figure 3 layout."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,json,os,time
import numpy as np
import pandas as pd
from greenfield_optimise_20260915 import ROOT,OUT,ENDS,T,Kernel,load_pool

CACHE={}

def data(state):
    if state in CACHE:return CACHE[state]
    metadata=json.loads((OUT/'pool_metadata.json').read_text())['S3'];rows=[]
    for code,meta in metadata.items():
        r=json.loads((OUT/'solutions/S3'/state/(code+'.json')).read_text())
        d,pv,w=load_pool(meta,state);mean=meta['reference_mean_load_mw']
        x=np.array([r['pv_mw'],r['wind_mw'],r['battery_mwh']])/mean
        rows.append((code,meta,r,d,pv,w,x))
    CACHE[state]=rows;return rows

def scores(load,gap):
    records=[]
    for y,a,z in zip(range(1995,2015),ENDS[:-1],ENDS[1:]):
        d=load[a:z];g=gap[a:z];cd=np.r_[0,np.cumsum(d,dtype=float)];cg=np.r_[0,np.cumsum(g,dtype=float)]
        r=dict(weather_year=y,demand_mwh=float(d.sum()),gap_mwh=float(g.sum()),annual_coverage=1-float(g.sum()/d.sum()))
        for h in [24,168]:
            energy=cg[h:]-cg[:-h];den=cd[h:]-cd[:-h]
            r[f'min_{h}h_coverage']=1-float((energy/den).max());r[f'worst_{h}h_gap_mwh']=float(energy.max())
        records.append(r)
    return records

def surface(task):
    rw,re=task;target=OUT/'diagnostics/surface'/f'{rw:.5f}_{re:.5f}.json';target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():return json.loads(target.read_text())
    kernel=Kernel();load=np.zeros(T);gap=np.zeros(T);totalcost=0.;full=True
    for code,meta,r,d,pv,w,x in data('ACCESS-CM2_ssp245'):
        scale=meta['reference_mean_load_mw'];s=x*np.array([rw,rw,re])
        met,g,_,_=kernel.dispatch(d,pv,w,ENDS,s);gap+=g*scale;load+=d*scale
        full=full and met[:,1].max()<2e-7 and met[:,4].max()<2e-6
        totalcost+=rw*(r['pv_annual_cost_usd']+r['wind_annual_cost_usd']+r['delivery_annual_cost_usd'])+re*r['battery_annual_cost_usd']
    byyear=scores(load,gap)
    result=dict(wind_pv_scale=rw,storage_scale=re,annual_cost_usd=totalcost,full_matching=bool(full),
                **{k:float(np.mean([r[k] for r in byyear])) for k in byyear[0] if k!='weather_year'})
    target.write_text(json.dumps(result));return result

def frontier(task):
    state,rw=task;folder=OUT/'diagnostics/frontier'/state;folder.mkdir(parents=True,exist_ok=True);target=folder/f'{rw:.5f}.json'
    if target.exists():return json.loads(target.read_text())
    kernel=Kernel();costs=dict(wind_pv=0.,battery=0.,delivery=0.);energy=0.;feasible=True;uniform_scale=0.
    for code,meta,r,d,pv,w,x in data(state):
        gen=(pv.astype(float)*x[0]+w.astype(float)*x[1]).astype('float32')
        need=kernel.requirement(d,gen,ENDS,rw)[0]
        if not np.isfinite(need):feasible=False;break
        capacity=need*meta['reference_mean_load_mw'];energy+=capacity
        ratio=need/x[2] if x[2]>1e-9 else (0. if need<1e-7 else np.inf)
        uniform_scale=max(uniform_scale,ratio)
        costs['wind_pv']+=rw*(r['pv_annual_cost_usd']+r['wind_annual_cost_usd'])
        costs['delivery']+=rw*r['delivery_annual_cost_usd'];costs['battery']+=capacity*44565.883753861046
    result=dict(state=state,wind_pv_scale=rw,feasible=feasible,battery_mwh=energy if feasible else None,
                uniform_storage_scale=uniform_scale if feasible and np.isfinite(uniform_scale) else None,
                **{k+'_annual_cost_usd':v if feasible else None for k,v in costs.items()},annual_cost_usd=sum(costs.values()) if feasible else None)
    target.write_text(json.dumps(result));return result

def counterfactual(state):
    folder=OUT/'diagnostics/counterfactual';folder.mkdir(parents=True,exist_ok=True);target=folder/(state+'.json')
    if target.exists():return json.loads(target.read_text())
    kernel=Kernel();load=np.zeros(T);gaps={'annual_reference':np.zeros(T),'no_battery':np.zeros(T)}
    zero_floor=np.zeros(T);zero_pools=[];discharge=0.;cost=0.;battery_cost=0.
    for code,meta,r,d,pv,w,x in data(state):
        scale=meta['reference_mean_load_mw'];load+=d*scale
        plans={'annual_reference':np.array([meta['reference_pv_mw']/scale,meta['reference_wind_mw']/scale,0.]),
               'no_battery':np.array([x[0],x[1],0.])}
        for name,plan in plans.items():
            _,g,_,_=kernel.dispatch(d,pv,w,ENDS,plan);gaps[name]+=g*scale
        zero=(pv==0)&(w==0)&(d>0);zero_floor+=np.where(zero,d*scale,0)
        if zero.any():zero_pools.append(code)
        discharge+=float(np.asarray(r['yearly_metrics'])[:,8].mean());cost+=r['total_annual_cost_usd'];battery_cost+=r['battery_annual_cost_usd']
    result=dict(state=state,annual_battery_discharge_mwh=discharge,battery_share_of_load_pct=100*discharge/(load.sum()/20),
                battery_cost_share_pct=100*battery_cost/cost,zero_resource_pools=zero_pools,
                capacity_only_unavoidable_annual_gap_mwh=float(zero_floor.sum()/20),
                cases={name:scores(load,g) for name,g in gaps.items()})
    target.write_text(json.dumps(result));return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--part',choices=['surface','frontier','counterfactual','all'],default='all');args=p.parse_args()
    assert json.loads((OUT/'summary_validation.json').read_text())['passed']
    tasks=[]
    if args.part in ['surface','all']:
        tasks += [('surface',(float(a),float(b))) for a in np.round(np.linspace(.4,1.4,21),5) for b in np.unique(np.r_[np.round(np.linspace(0,1.4,21),5),1.])]
    if args.part in ['frontier','all']:
        tasks += [('frontier',('ACCESS-CM2_'+s,float(r))) for s in ['ssp126','ssp245','ssp585'] for r in np.round(np.linspace(.6,1.6,26),5)]
    if args.part in ['counterfactual','all']:
        tasks += [('counterfactual',state) for state in ['historical']+[m+'_'+s for m in ['ACCESS-CM2','GFDL-ESM4','MIROC6'] for s in ['ssp126','ssp245','ssp585']]]
    outputs={};count=0
    with ProcessPoolExecutor(max_workers=32) as pool:
        fs={pool.submit(globals()[kind],task):(kind,task) for kind,task in tasks}
        for f in as_completed(fs):
            kind,task=fs[f];outputs.setdefault(kind,[]).append(f.result());count+=1
            if count%20==0:print(json.dumps({'completed':count,'total':len(tasks)}),flush=True)
    for kind,rows in outputs.items():
        if kind!='counterfactual':pd.DataFrame(rows).to_csv(OUT/('figure3_'+kind+'.csv'),index=False)
        else:(OUT/'figure3_counterfactual.json').write_text(json.dumps(rows,indent=2))
    if 'surface' in outputs:
        ref=next(r for r in outputs['surface'] if r['wind_pv_scale']==1 and r['storage_scale']==1)
        assert ref['full_matching'] and ref['annual_coverage']>1-1e-9 and ref['min_168h_coverage']>1-1e-9
    (OUT/('diagnostics_'+args.part+'_complete.json')).write_text(json.dumps(dict(passed=True,tasks=count)))

if __name__=='__main__':main()
