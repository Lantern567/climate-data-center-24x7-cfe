"""Reoptimize full-hourly new-build plans under future technology prices.

Shares valid cutting planes across objectives, never across physical pools.
Every returned plan is independently replayed by the original dispatch kernel.
"""
from pathlib import Path
import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[name]='1'
import argparse,json,time,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
from scipy.optimize import linprog
import greenfield_optimise_20260915 as G

OUT=G.ROOT/'complete_matching_technology_20260917'
META=None
PRICES=None

class WarmSolver:
    def __init__(self,d,pv,w):
        self.d,self.pv,self.w=d,pv,w
        self.k=G.Kernel();self.A=[];self.b=[]
        for lo,hi in zip(G.ENDS[:-1],G.ENDS[1:]):
            row=np.array([-float(pv[lo:hi].mean()),-float(w[lo:hi].mean()),0.])
            rhs=-float(d[lo:hi].mean());scale=max(1.,float(abs(row).max()),abs(rhs))
            self.A.append(row/scale);self.b.append(rhs/scale)
    def solve(self,cost):
        scale=float(max(cost));c=cost/scale
        for step in range(1200):
            lp=linprog(c,A_ub=np.asarray(self.A),b_ub=np.asarray(self.b),bounds=(0,None),method='highs',
                       options={'dual_feasibility_tolerance':1e-9,'primal_feasibility_tolerance':1e-9,'threads':1})
            if not lp.success:raise RuntimeError(lp.message)
            x=lp.x;cuts=self.k.cuts(self.d,self.pv,self.w,G.ENDS,x).reshape(-1,4)
            bad=cuts[:,0]<-2e-7
            close=(cuts[:,0]/np.maximum(1.,np.abs(cuts[:,1:])@np.maximum(x,0))).min()>-1e-8
            if not bad.any() or close:
                delivered=x*(1+2e-6)
                metrics,gap,soc,gen=self.k.dispatch(self.d,self.pv,self.w,G.ENDS,delivered)
                if metrics[:,1].max()<2e-7 and metrics[:,4].max()<2e-6:
                    upper=float(delivered@cost);lower=float(lp.fun*scale)
                    assert upper/lower-1<3e-6
                    return delivered,metrics,dict(iterations=step+1,lower_bound=lower,
                        relative_optimality_gap=upper/lower-1,max_gap=float(metrics[:,1].max()),
                        max_cyclic_error=float(metrics[:,4].max()))
                if not bad.any():raise RuntimeError('No violating cuts but dispatch failed')
            for row in cuts[bad]:
                grad=row[1:];rhs=float(row[0]-grad@x)
                norm=max(1.,float(abs(grad).max()),abs(rhs))
                self.A.append(-grad/norm);self.b.append(rhs/norm)
        raise RuntimeError('No convergence')

def prices(full=True):
    global PRICES
    if PRICES is None:PRICES=pd.read_csv(OUT/'annual_capacity_price_paths.csv')
    p=PRICES.copy()
    if not full:p=p[((p.path=='Moderate')&(p.year==2030))|(p.year==2050)]
    rows=p.to_dict('records')
    base=next(r for r in rows if r['path']=='Moderate' and r['year']==2030)
    rows=[base]+[r for r in rows if r is not base]
    if full:
        advanced=next(r for r in rows if r['path']=='Advanced' and r['year']==2050)
        rows += [dict(base,path='RenewablesOnly',year=2050,pv=advanced['pv'],wind=advanced['wind']),
                 dict(base,path='StorageOnly',year=2050,battery=advanced['battery'])]
    return rows

def job(task):
    global META
    mode,state,code=task;started=time.time()
    dest=OUT/'solutions'/mode/state/(code+'.json');dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():return {'task':task,'cached':True}
    if META is None:META=json.loads((G.OUT/'pool_metadata.json').read_text())
    meta=META[mode][code];d,pv,w=G.load_pool(meta,state);scale=meta['reference_mean_load_mw']
    base=json.loads((G.OUT/'solutions'/mode/state/(code+'.json')).read_text())
    x0=np.array([base['pv_mw'],base['wind_mw'],base['battery_mwh']])/scale
    solver=WarmSolver(d,pv,w)
    fee=meta['delivery_usd_mwh'];hours=G.T/20
    annual_gen=np.array([float(pv.mean(dtype=float))*hours,float(w.mean(dtype=float))*hours,0.])
    demand=float(d.sum(dtype=float))*scale/20
    rows=[]
    for price in prices(mode=='S3'):
        cap=np.array([price['pv'],price['wind'],price['battery']])
        cost=cap+fee*annual_gen
        x,metrics,diag=solver.solve(cost)
        total=float(x@cost)*scale;frozen=float(x0@cost)*scale
        assert total<=frozen*(1+4e-6),(task,price,total,frozen)
        components=x*scale*cap;delivery=float(x@annual_gen)*fee*scale
        assert np.isclose(components.sum()+delivery,total,rtol=1e-12)
        if price['path']=='Moderate' and price['year']==2030:
            assert np.isclose(total,base['total_annual_cost_usd'],rtol=5e-6),(task,total,base['total_annual_cost_usd'])
        rows.append(dict(mode=mode,state=state,pool=code,path=price['path'],year=int(price['year']),
            demand_mwh=demand,cost=total,baseline_cost=base['total_annual_cost_usd'],frozen_cost=frozen,
            pv_cost=components[0],wind_cost=components[1],battery_cost=components[2],delivery_cost=delivery,
            pv_mw=x[0]*scale,wind_mw=x[1]*scale,battery_mwh=x[2]*scale,battery_mw=x[2]*scale/4,
            hourly_gap_mw=diag['max_gap']*scale,cyclic_error_mwh=diag['max_cyclic_error']*scale,
            annual_gap_mwh=float(metrics[:,0].mean())*scale,
            relative_optimality_gap=diag['relative_optimality_gap'],iterations=diag['iterations']))
    result=dict(task=task,seconds=time.time()-started,rows=rows,passed=True)
    tmp=dest.with_suffix('.partial');tmp.write_text(json.dumps(result));tmp.replace(dest)
    return dict(task=task,seconds=result['seconds'],price_cases=len(rows))

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=64);p.add_argument('--pilot',action='store_true');args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    metadata=json.loads((G.OUT/'pool_metadata.json').read_text())
    states=[m+'_'+s for m in ['ACCESS-CM2','GFDL-ESM4','MIROC6'] for s in ['ssp126','ssp245','ssp585']]
    tasks=[('S3',state,code) for state in states for code in metadata['S3']]
    if args.pilot:tasks=[('S3','ACCESS-CM2_ssp245',code) for code in ['USA','DEU','CHN']]
    else:tasks += [(mode,'ACCESS-CM2_ssp245',code) for mode in ['S4','S5','S1','S2'] for code in metadata[mode]]
    print(json.dumps(dict(stage='start',tasks=len(tasks),workers=args.workers)),flush=True)
    errors=[];done=0;start=time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures={ex.submit(job,t):t for t in tasks}
        for f in as_completed(futures):
            try:result=f.result()
            except Exception as e:
                result=dict(task=futures[f],error=str(e),traceback=traceback.format_exc());errors.append(result)
            done+=1
            if args.pilot or 'error' in result:print(json.dumps(result),flush=True)
            if done%25==0 or done==len(tasks):
                status=dict(done=done,total=len(tasks),errors=len(errors),seconds=time.time()-start)
                (OUT/'status.json').write_text(json.dumps(status));print(json.dumps(status),flush=True)
    (OUT/('pilot_complete.json' if args.pilot else 'complete.json')).write_text(json.dumps(dict(passed=not errors,errors=errors,tasks=len(tasks),seconds=time.time()-start),indent=2))
    if errors:raise RuntimeError(f'{len(errors)} failed tasks')

if __name__=='__main__':main()
