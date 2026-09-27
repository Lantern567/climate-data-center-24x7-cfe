"""Globally optimal fixed-capacity renewable/battery plans via exact LP cuts."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,ctypes,json,os,time,warnings
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import lil_matrix
warnings.filterwarnings('ignore',message='Unrecognized options detected.*threads.*')

ROOT=Path('/data2/ljt/datacenter_climate_result1_20260912')
OUT=ROOT/'greenfield_cfe_20260915'
ENDS=np.r_[0,np.cumsum([8784 if y%4==0 else 8760 for y in range(1995,2015)])].astype('int64')
T=int(ENDS[-1]);ETA=.85;DURATION=4.
MAPS={}

class Kernel:
    def __init__(self):
        self.lib=ctypes.CDLL(str(OUT/'greenfield_kernel.so'))
        f32=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS')
        f64=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        i64=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
        self.lib.gf_cuts.argtypes=[f32,f32,f32,i64,ctypes.c_int]+[ctypes.c_double]*5+[f64]
        self.lib.gf_dispatch.argtypes=[f32,f32,i64,ctypes.c_int]+[ctypes.c_double]*4+[f64]*3
        self.lib.gf_requirement.argtypes=[f32,f32,i64,ctypes.c_int]+[ctypes.c_double]*3+[f64]
    def cuts(self,d,pv,w,ends,x):
        out=np.empty((len(ends)-1,3,4))
        self.lib.gf_cuts(d,pv,w,ends,len(ends)-1,*map(float,x),DURATION,ETA,out)
        return out
    def dispatch(self,d,pv,w,ends,x):
        gen=(x[0]*pv.astype(float)+x[1]*w.astype(float)).astype('float32')
        out=np.empty((len(ends)-1,10));gap=np.empty(len(d));soc=np.empty(len(d))
        self.lib.gf_dispatch(d,gen,ends,len(ends)-1,1.,float(x[2]),DURATION,ETA,out,gap,soc)
        return out,gap,soc,gen
    def requirement(self,d,g,ends,multiplier):
        out=np.empty(2)
        self.lib.gf_requirement(d,g,ends,len(ends)-1,float(multiplier),DURATION,ETA,out)
        return out

def optimise(d,pv,w,cost,ends=ENDS,route='joint'):
    k=Kernel();cost=np.asarray(cost,float);cost_scale=float(cost.max());c=cost/cost_scale
    A=[];b=[]
    for a,z in zip(ends[:-1],ends[1:]):
        row=np.array([-float(pv[a:z].mean()),-float(w[a:z].mean()),0.]);rhs=-float(d[a:z].mean())
        if row.min()>=0 and rhs<0:return dict(feasible=False,reason='no generation resource in a weather year')
        scale=max(1.,float(abs(row).max()),abs(rhs));A.append(row/scale);b.append(rhs/scale)
    bounds=[(0,None),(0,None),(0,0) if route=='capacity_only' else (0,None)]
    previous=-np.inf
    for step in range(1200):
        lp=linprog(c,A_ub=np.array(A),b_ub=np.array(b),bounds=bounds,method='highs',
                   options={'dual_feasibility_tolerance':1e-9,'primal_feasibility_tolerance':1e-9,'threads':1})
        if lp.status==2:return dict(feasible=False,reason='infeasible resource and storage route',iterations=step+1)
        if not lp.success:raise RuntimeError(lp.message)
        assert lp.fun>=previous-1e-7;previous=lp.fun
        x=lp.x;cuts=k.cuts(d,pv,w,ends,x).reshape(-1,4)
        bad=cuts[:,0]<-2e-7
        close_enough=(cuts[:,0]/np.maximum(1.,np.abs(cuts[:,1:])@np.maximum(x,0))).min()>-1e-8
        if not bad.any() or close_enough:
            delivered=x*(1+2e-6)
            metrics,gap,soc,gen=k.dispatch(d,pv,w,ends,delivered)
            if metrics[:,1].max()<2e-7 and metrics[:,4].max()<2e-6:
                return dict(feasible=True,x=delivered.tolist(),lower_bound_cost=float(lp.fun*cost_scale),
                            annual_cost=float(delivered@cost),relative_optimality_gap=float((delivered@cost)/(lp.fun*cost_scale)-1),
                            iterations=step+1,constraints=len(A),max_normalised_hourly_gap=float(metrics[:,1].max()),
                            max_normalised_cyclic_error=float(metrics[:,4].max()),
                            metrics=metrics.tolist())
            if not bad.any():raise RuntimeError('Separating constraints passed but independent dispatch failed')
        for row in cuts[bad]:
            gradient=row[1:];rhs=float(row[0]-gradient@x)
            scale=max(1.,float(abs(gradient).max()),abs(rhs))
            A.append(-gradient/scale);b.append(rhs/scale)
    raise RuntimeError('Cut generation did not converge')

def direct_lp(d,pv,w,cost,ends):
    n=len(d);nv=3+4*n
    # P, W, E, charge, discharge, state of charge, curtailment.
    objective=np.zeros(nv);objective[:3]=cost
    eq=lil_matrix((2*n,nv));rhs=np.zeros(2*n)
    ub=lil_matrix((3*n,nv));bound=np.zeros(3*n);leg=ETA**.5
    for a,z in zip(ends[:-1],ends[1:]):
        for t in range(a,z):
            prev=z-1 if t==a else t-1
            eq[t,0]=pv[t];eq[t,1]=w[t];eq[t,3+t]=-1;eq[t,3+n+t]=1;eq[t,3+3*n+t]=-1;rhs[t]=d[t]
            eq[n+t,3+2*n+t]=1;eq[n+t,3+2*n+prev]-=1;eq[n+t,3+t]=-leg;eq[n+t,3+n+t]=1/leg
            ub[t,3+t]=1;ub[t,2]=-1/DURATION
            ub[n+t,3+n+t]=1;ub[n+t,2]=-1/DURATION
            ub[2*n+t,3+2*n+t]=1;ub[2*n+t,2]=-1
    lp=linprog(objective,A_ub=ub.tocsr(),b_ub=bound,A_eq=eq.tocsr(),b_eq=rhs,bounds=(0,None),method='highs',options={'threads':1})
    assert lp.success,lp.message
    return lp

def validate():
    rng=np.random.default_rng(81547);checks=[]
    for n in [12,48,240]:
        for j in range(8):
            ends=np.array([0,n//2,n],dtype='int64')
            d=rng.uniform(.4,1.6,n).astype('float32');pv=np.maximum(0,np.sin(np.arange(n)*np.pi/12))*rng.uniform(.3,1,n)
            w=rng.uniform(0,.8,n);w[rng.uniform(0,1,n)<.3]=0
            pv=pv.astype('float32');w=w.astype('float32');cost=np.array([1.14,1.67,.446])
            got=optimise(d,pv,w,cost,ends);ref=direct_lp(d,pv,w,cost,ends)
            assert got['feasible'] and np.isclose(got['lower_bound_cost'],ref.fun,rtol=2e-6,atol=2e-6)
            x=np.array(got['x']);gen=(pv.astype(float)*x[0]+w.astype(float)*x[1]).astype('float32')
            required=Kernel().requirement(d,gen,ends,1.)[0]
            assert np.isclose(required,x[2],rtol=4e-5,atol=3e-5),(required,x[2])
            checks.append(dict(hours=n,case=j,lp_cost=float(ref.fun),cut_cost=got['annual_cost'],iterations=got['iterations']))
    report=dict(passed=True,independent_full_dispatch_lp_cases=len(checks),checks=checks,
                objective='all new PV, wind and 4-hour battery capacity costed; all hourly gaps zero with cyclic SOC')
    (OUT/'optimiser_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

def array(path):
    path=str(path)
    if path not in MAPS:MAPS[path]=np.load(path,mmap_mode='r')
    return MAPS[path]

def load_pool(meta,state):
    d=array(OUT/meta['demand'][state]['file'])[meta['demand'][state]['row']]
    p=array(OUT/meta['pv'][state]['file'])[meta['pv'][state]['row']]
    w=array(OUT/meta['wind'][state]['file'])[meta['wind'][state]['row']]
    scale=meta['reference_mean_load_mw']
    return np.ascontiguousarray(d/scale,dtype='float32'),np.ascontiguousarray(p*meta.get('pv_factor',1.),dtype='float32'),np.ascontiguousarray(w,dtype='float32')

def job(task):
    mode,state,pool_id=task;start=time.time()
    folder=OUT/'solutions'/mode/state;folder.mkdir(parents=True,exist_ok=True);target=folder/(pool_id+'.json')
    if target.exists():return json.loads(target.read_text())
    if 'metadata' not in MAPS:MAPS['metadata']=json.loads((OUT/'pool_metadata.json').read_text())
    meta=MAPS['metadata'][mode][pool_id];d,pv,w=load_pool(meta,state);scale=meta['reference_mean_load_mw']
    if 'prices' not in MAPS:MAPS['prices']=json.loads((ROOT/'recovery_investment_20260914/inputs/price_interface_2030.json').read_text())['technologies']
    prices=MAPS['prices'];fee=meta['delivery_usd_mwh'];hours_year=T/20
    unit=np.array([prices['utility_pv']['annual_usd_mw_year']+fee*float(pv.mean(dtype=float))*hours_year,
                   prices['landbased_wind']['annual_usd_mw_year']+fee*float(w.mean(dtype=float))*hours_year,
                   prices['battery_4h']['annual_usd_mwh_year']])
    got=optimise(d,pv,w,unit)
    result=dict(mode=mode,state=state,pool_id=pool_id,sites=meta['sites'],reference_mean_load_mw=scale,
                future_or_historical_demand_mwh_year=float(d.sum(dtype=float))*scale/20,**got)
    if got['feasible']:
        x=np.array(got['x'])*scale;metrics=np.array(got.pop('metrics'))*scale;result.pop('metrics',None)
        pv_cost=x[0]*prices['utility_pv']['annual_usd_mw_year'];wind_cost=x[1]*prices['landbased_wind']['annual_usd_mw_year'];storage_cost=x[2]*prices['battery_4h']['annual_usd_mwh_year']
        produced=(x[0]*float(pv.mean(dtype=float))+x[1]*float(w.mean(dtype=float)))*hours_year;delivery_cost=fee*produced
        result.update(pv_mw=x[0],wind_mw=x[1],battery_mwh=x[2],battery_mw=x[2]/4,
                      pv_annual_cost_usd=pv_cost,wind_annual_cost_usd=wind_cost,battery_annual_cost_usd=storage_cost,
                      delivery_annual_cost_usd=delivery_cost,total_annual_cost_usd=pv_cost+wind_cost+storage_cost+delivery_cost,
                      lower_bound_total_annual_cost_usd=got['lower_bound_cost']*scale,
                      total_capex_usd=1000*(x[0]*prices['utility_pv']['capex_usd_kw']+x[1]*prices['landbased_wind']['capex_usd_kw']+x[2]/4*prices['battery_4h']['capex_usd_kw']),
                      renewable_generation_mwh_year=produced,curtailed_mwh_year=float(metrics[:,9].mean()),
                      validated_gap_mwh=float(metrics[:,0].sum()),validated_max_hourly_gap_mw=float(metrics[:,1].max()),
                      validated_max_soc_closure_mwh=float(metrics[:,4].max()),
                      yearly_demand_mwh=[float(d[a:z].sum(dtype=float))*scale for a,z in zip(ENDS[:-1],ENDS[1:])],
                      yearly_generation_mwh=[float(pv[a:z].sum(dtype=float))*x[0]+float(w[a:z].sum(dtype=float))*x[1] for a,z in zip(ENDS[:-1],ENDS[1:])],
                      yearly_metrics=metrics.tolist())
    result['seconds']=time.time()-start
    tmp=target.with_suffix('.partial');tmp.write_text(json.dumps(result));tmp.replace(target)
    return {k:v for k,v in result.items() if k!='yearly_metrics'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--validate',action='store_true');p.add_argument('--mode',action='append');p.add_argument('--workers',type=int,default=128);args=p.parse_args()
    if args.validate:validate();return
    assert json.loads((OUT/'optimiser_validation.json').read_text())['passed']
    metadata=json.loads((OUT/'pool_metadata.json').read_text())
    tasks=[(mode,state,pool_id) for mode,pools in metadata.items() if not args.mode or mode in args.mode for state in next(iter(pools.values()))['demand'] for pool_id in pools]
    done=[];failed=[];started=time.time();workers=min(args.workers,len(os.sched_getaffinity(0)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        fs={pool.submit(job,t):t for t in tasks}
        for f in as_completed(fs):
            task=fs[f]
            try:done.append(f.result())
            except Exception as exc:failed.append(dict(task=task,error=repr(exc)));print(json.dumps(failed[-1]),flush=True)
            if (len(done)+len(failed))%25==0:
                report=dict(pid=os.getpid(),workers=workers,complete=len(done),failed=failed,total=len(tasks),elapsed_seconds=time.time()-started,last=task)
                (OUT/'optimisation_status.json').write_text(json.dumps(report));print(json.dumps({k:v for k,v in report.items() if k!='failed'}),flush=True)
    pd.DataFrame(done).to_csv(OUT/'pool_cost_results.csv',index=False)
    report=dict(passed=not failed,complete=len(done),failed=failed,total=len(tasks),seconds=time.time()-started)
    (OUT/'optimisation_complete.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))

if __name__=='__main__':main()
