"""Quantify persistent low-CFE matching from paired site-hourly climate replays.

Primary daily shortfall threshold: 50% of daily facility demand. Sensitivities:
25% and 75%. Events are runs of >=3 or >=7 complete UTC calendar days. These are
wind/PV matching shortfalls, not realised grid outages. No adaptation is applied.
"""
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
import gc
import hashlib
import importlib.util
import json
import multiprocessing
import os
import resource
import time
import numpy as np
import pandas as pd

ROOT = Path('/data2/ljt/datacenter_climate_result1_20260912')
OUT = ROOT/'persistence_20260914'
MODELS = ['ACCESS-CM2','GFDL-ESM4','MIROC6']
SSPS = ['ssp126','ssp245','ssp585']
YEARS = range(1995,2015)
VERSION = 'daily_low_matching_site_paired_v1'
IO_GATE = None

def runs(mask, energy):
    n, days = mask.shape
    run = np.zeros(n,dtype=np.int32)
    total = np.zeros(n,dtype=float)
    longest = np.zeros(n,dtype=np.int32)
    results = {f'{m}_{w}':np.zeros(n,dtype=float) for w in [3,7] for m in ['days','events','energy']}
    for t in range(days+1):
        active = mask[:,t] if t<days else np.zeros(n,dtype=bool)
        ending = (~active)&(run>0)
        for w in [3,7]:
            selected = ending&(run>=w)
            results[f'days_{w}'][selected] += run[selected]
            results[f'events_{w}'][selected] += 1
            results[f'energy_{w}'][selected] += total[selected]
        run[ending]=0;total[ending]=0
        if t<days:
            run[active]+=1
            total[active]+=energy[active,t]
            np.maximum(longest,run,out=longest)
    results['longest_days']=longest
    results['low_matching_days']=mask.sum(axis=1)
    return results

def load_inputs(model,ssp,year):
    spec=importlib.util.spec_from_file_location('geo',Path(__file__).with_name('66_compute_geographic_climate_conditioning.py'))
    g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
    inp=ROOT/'inputs'
    with np.load(inp/f'paired_{year}.npz',allow_pickle=False) as z:
        ids=pd.Index(z['ids'].astype(str));hist=z['historical']
        if model in z and ssp=='ssp245':future=z[model]
        else:
            compact=inp/'compact_demand'/(model if ssp=='ssp245' else f'{model}_{ssp}')
            receipt=json.loads((compact/'complete.json').read_text())
            if ssp!='ssp245':assert receipt['validation_passed']
            assert np.array_equal(np.load(compact/'ids.npy').astype(str),ids.to_numpy())
            future=np.load(compact/f'{year}.npy')
    g.SUPPLY_H=inp/'supply/historical'
    g.SUPPLY_F=inp/'supply'/model if ssp=='ssp245' else ROOT/'new_ssp_supply_replay_pv_corrected'/model/ssp
    if ssp!='ssp245':assert json.loads((g.SUPPLY_F/'pv_correction_complete.json').read_text())['passed']
    ladder=pd.read_parquet(inp/'mixes.parquet')
    mix=ladder[ladder.scenario.eq('S3')].drop_duplicates('logical_site_id').set_index('logical_site_id').wind_energy_share
    sites=pd.read_csv(inp/'sites.csv',low_memory=False)
    sites=sites[sites.logical_site_id.isin(mix.index)&sites.supply_country_area_code.notna()].sort_values('logical_site_id').reset_index(drop=True)
    sites['region']=sites.continent.where(~sites.continent.isin(['Unknown','Antarctica']))
    regions=sites.dropna(subset=['region']).groupby('supply_country_area_code').region.agg(lambda v:v.mode().iloc[0])
    take=ids.get_indexer(sites.logical_site_id.astype(str));assert (take>=0).all()
    mix_values=mix.reindex(sites.logical_site_id).to_numpy(float)
    groups,pools=g.pool_spec('S3',year,sites,regions,None)
    return g,hist,future,sites,take,mix_values,groups,pools

def job(model,ssp,year):
    target=OUT/model/ssp/str(year);target.mkdir(parents=True,exist_ok=True)
    if (target/'validation.json').exists():
        report=json.loads((target/'validation.json').read_text())
        if report.get('passed') and report.get('version')==VERSION:return report
    started=time.perf_counter()
    with IO_GATE if IO_GATE is not None else nullcontext():
        g,hist,future,sites,take,mix_values,groups,pools=load_inputs(model,ssp,year)
    n=len(sites);hours=hist.shape[1];days=hours//24;assert hours%24==0
    daily_load={k:np.zeros((n,days),dtype=float) for k in ['historical','future']}
    daily_gap={k:np.zeros((n,days),dtype=float) for k in daily_load}
    tails={k:{w:np.zeros(n) for w in g.WINDOWS} for k in daily_load}
    deficit_hours={k:np.zeros(n) for k in daily_load}
    covered=np.zeros(n,dtype=bool)
    for key,positions_all in groups.items():
        pv_h,wind_h,pv_f,wind_f=pools[key]
        for share in np.unique(mix_values[positions_all]):
            subset=positions_all[mix_values[positions_all]==share]
            shapes={'historical':g.portfolio_shape(pv_h,wind_h,float(share)),
                    'future':g.portfolio_shape(pv_f,wind_f,float(share))}
            for offset in range(0,len(subset),128):
                ix=subset[offset:offset+128]
                loads={'historical':np.asarray(hist[take[ix]],dtype=float),
                       'future':np.asarray(future[take[ix]],dtype=float)}
                reference=loads['historical'].sum(axis=1)
                for case in loads:
                    load=loads[case]
                    gap=np.maximum(load-reference[:,None]*shapes[case][None,:],0.)
                    daily_load[case][ix]=load.reshape(len(ix),days,24).sum(axis=2)
                    daily_gap[case][ix]=gap.reshape(len(ix),days,24).sum(axis=2)
                    deficit_hours[case][ix]=(gap>1e-9).sum(axis=1)
                    for w,value in g.rolling_maxima(gap).items():tails[case][w][ix]=value
                covered[ix]=True
    assert covered.all() and n==8317
    del hist,future,loads,load,gap,groups,pools;gc.collect()
    reference_energy=daily_load['historical'].sum(axis=1)
    weights=reference_energy/reference_energy.sum()
    physical=[];events=[];country=[];site_output=[]
    for case in ['historical','future']:
        d=daily_load[case];gap=daily_gap[case]
        annual_load=d.sum();annual_gap=gap.sum()
        row={'model':model,'ssp':ssp,'weather_year':year,'case':case,'sites':n,'hours':hours,
             'demand_mwh':annual_load,'uncovered_mwh':annual_gap,'coverage':1-annual_gap/annual_load,
             'uncovered_equivalent_h':annual_gap/(reference_energy.sum()/hours),
             'shortfall_hours_weighted':np.dot(weights,deficit_hours[case]),
             'historical_mean_load_mw':reference_energy.sum()/hours}
        row.update({f'worst_{w}h_mwh':tails[case][w].sum() for w in g.WINDOWS})
        row['worst_168h_average_gap_mw']=row['worst_168h_mwh']/168
        row['worst_168h_equivalent_h']=row['worst_168h_mwh']/(reference_energy.sum()/hours)
        physical.append(row)
        for threshold in [.25,.5,.75]:
            mask=np.divide(gap,d,out=np.zeros_like(gap),where=d>0)>=threshold
            result=runs(mask,gap)
            result['exposed_3']=(result['events_3']>0).astype(float)
            result['exposed_7']=(result['events_7']>0).astype(float)
            out={'model':model,'ssp':ssp,'weather_year':year,'case':case,'shortfall_threshold':threshold,
                 'annual_uncovered_mwh':annual_gap,'sites':n}
            for metric,values in result.items():
                out[metric+'_weighted']=float(np.dot(weights,values))
                if metric.startswith('energy_'):out[metric+'_mwh']=values.sum()
                if metric.startswith('exposed_'):out[metric+'_sites']=int(values.sum())
            for w in [3,7]:out[f'energy_share_{w}_pct']=100*result[f'energy_{w}'].sum()/annual_gap
            events.append(out)
            if threshold==.5:
                sd=pd.DataFrame({'model':model,'ssp':ssp,'weather_year':year,'case':case,
                    'logical_site_id':sites.logical_site_id,'country':sites.supply_country_area_code,
                    'historical_demand_mwh':reference_energy,'annual_uncovered_mwh':gap.sum(axis=1),**result})
                site_output.append(sd)
                for code,ix in sites.groupby('supply_country_area_code').indices.items():
                    w=weights[ix]/weights[ix].sum()
                    crow={'model':model,'ssp':ssp,'weather_year':year,'case':case,'country':code,
                          'historical_demand_mwh':reference_energy[ix].sum(),'sites':len(ix)}
                    for metric,values in result.items():
                        crow[metric+'_weighted']=float(np.dot(w,values[ix]))
                        if metric.startswith('energy_'):crow[metric+'_mwh']=values[ix].sum()
                    country.append(crow)
            assert np.all(result['days_7']<=result['days_3']) and np.all(result['days_3']<=result['low_matching_days'])
            assert np.all(result['energy_3']<=gap.sum(axis=1)+1e-6)
    reference_dir=ROOT/'results'/model/str(year) if ssp=='ssp245' else ROOT/'new_ssp_results'/model/ssp/str(year)
    old=pd.read_csv(reference_dir/'summary.csv')
    a=pd.DataFrame(physical).set_index('case')
    b=old[old.scenario.eq('S3')].set_index('case')
    errors={}
    for case,old_case in [('historical','historical'),('future','joint_frozen_volume')]:
        for col in ['demand_mwh','uncovered_mwh','coverage',*[f'worst_{w}h_mwh' for w in g.WINDOWS]]:
            actual=float(a.loc[case,col]);expected=float(b.loc[old_case,col])
            assert np.isclose(actual,expected,rtol=2e-9,atol=1e-5),(model,ssp,year,case,col,actual,expected)
            errors[case+'_'+col]=abs(actual-expected)
    for name,rows in [('physical',physical),('events',events),('countries',country)]:
        frame=pd.DataFrame(rows);assert np.isfinite(frame.select_dtypes('number')).all().all()
        frame.to_csv(target/(name+'.csv'),index=False)
    pd.concat(site_output,ignore_index=True).to_csv(target/'site_primary_metrics.csv.gz',index=False,compression='gzip')
    report={'passed':True,'version':VERSION,'model':model,'ssp':ssp,'weather_year':year,
            'sites':n,'days':days,'seconds':time.perf_counter()-started,
            'max_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
            'baseline_errors':errors,'original_summary_sha256':hashlib.sha256((reference_dir/'summary.csv').read_bytes()).hexdigest(),
            'primary_shortfall_threshold':.5,'sensitivity_thresholds':[.25,.75],
            'events_use_complete_UTC_calendar_days':True,'events_do_not_cross_weather_year_boundaries':True,
            'weights':'within-weather-year historical facility energy; shared between paired climates',
            'event_energy_aggregation':'sum site-local low-matching events; no contemporaneous global event implied'}
    (target/'validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ['model','ssp','weather_year','seconds','max_rss_gib']}),flush=True)
    return report

def main():
    global IO_GATE
    parser=argparse.ArgumentParser();parser.add_argument('--pilot',action='store_true');args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    # The run kernel must correctly count separate events and a year-end event.
    test=runs(np.array([[1,1,1,0,1,1,1,1,1,1,1]],dtype=bool),np.ones((1,11)))
    assert test['events_3'][0]==2 and test['days_3'][0]==10 and test['events_7'][0]==1 and test['energy_7'][0]==7
    if args.pilot:
        print(json.dumps(job('ACCESS-CM2','ssp245',2000),indent=2));return
    started=time.perf_counter()
    pilot=json.loads((OUT/'ACCESS-CM2/ssp245/2000/validation.json').read_text())
    available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))/1024**2
    per_worker=max(2.5,pilot['max_rss_gib']*1.5)
    workers=max(1,min(180,len(os.sched_getaffinity(0)),int(max(1,available-32)/per_worker)))
    IO_GATE=multiprocessing.BoundedSemaphore(2)
    print(json.dumps({'stage':'parallel_start','workers':workers,'parallel_input_readers':2,'available_memory_gib':available,'reserved_gib':32,'budget_gib_per_worker':per_worker}),flush=True)
    reports=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(job,m,s,y) for y in YEARS for m in MODELS for s in SSPS]
        for f in as_completed(futures):reports.append(f.result())
    for name in ['physical','events','countries']:
        frame=pd.concat([pd.read_csv(OUT/m/s/str(y)/(name+'.csv')) for m in MODELS for s in SSPS for y in YEARS],ignore_index=True)
        frame.to_csv(OUT/(name+'_by_year.csv'),index=False)
        group=['model','ssp','case']+(['shortfall_threshold'] if name=='events' else ['country'] if name=='countries' else [])
        means=frame.drop(columns=['weather_year']).groupby(group).mean(numeric_only=True).reset_index()
        means.to_csv(OUT/(name+'_mean.csv'),index=False)
        index=['model','ssp']+(['shortfall_threshold'] if name=='events' else ['country'] if name=='countries' else [])
        h=means[means.case.eq('historical')].drop(columns='case').set_index(index)
        f=means[means.case.eq('future')].drop(columns='case').set_index(index)
        (f-h).reset_index().to_csv(OUT/(name+'_change.csv'),index=False)
    report={'passed':all(q['passed'] for q in reports),'version':VERSION,'jobs':len(reports),'workers':workers,
            'parallel_input_readers':2,'input_order':'weather year first to reuse historical arrays and supply cache',
            'primary_shortfall_threshold':.5,'sensitivity_thresholds':[.25,.75],'elapsed_seconds':time.perf_counter()-started,
            'maximum_job_rss_gib':max(q['max_rss_gib'] for q in reports),'climate_period':'2031–2050',
            'original_matching_results_reproduced':True,'results':reports}
    (OUT/'calculation_validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2),flush=True)

if __name__=='__main__':main()
