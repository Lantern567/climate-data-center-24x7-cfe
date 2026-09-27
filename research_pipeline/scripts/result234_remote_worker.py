"""Extend v31 mechanisms and engineering surfaces with the unchanged original kernels."""
from pathlib import Path
from math import factorial
import argparse, ast, hashlib, importlib.util, json, resource, sys, time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'supplemental_20260913'
INPUT=BASE/'inputs'
EXPERIMENTS=('main_robust','S3','S4','S5')


def module(name, filename):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(filename))
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    return m


def compare(actual, reference, keys, columns, label, cost_tolerance=None):
    actual=actual.copy();reference=reference.copy()
    # CSV readers round floating grid labels differently from in-memory float32 promotion.
    # Canonicalise labels only; metric values and physical calculations are unchanged.
    for key in keys:
        if key in ['additional_procurement_pct','storage_hours_mean_load']:
            actual[key]=actual[key].round(10);reference[key]=reference[key].round(10)
    assert not actual.duplicated(keys).any() and not reference.duplicated(keys).any()
    a=actual.set_index(keys).sort_index();b=reference.set_index(keys).sort_index()
    assert a.index.equals(b.index),label+' index mismatch'
    errors={}
    for col in columns:
        errors[col]=float(np.max(np.abs(a[col].to_numpy()-b[col].to_numpy())))
        atol=np.full(len(a),1e-4)
        if cost_tolerance is not None:
            if 'metric' in a.index.names:
                atol[a.index.get_level_values('metric')=='annual_procurement_restoration_usd']=cost_tolerance
            elif col=='annual_procurement_restoration_usd':atol[:]=cost_tolerance
        assert np.allclose(a[col],b[col],rtol=2e-9,atol=atol),f'{label}/{col}: {errors[col]}'
    return errors


def run_mechanism(year, ids, historical, future, h_supply, f_supply, out):
    """Execute original script69 function bodies, adapting only input/output access."""
    original=Path(__file__).with_name('69_fixed_volume_mechanisms.py').read_text(encoding='utf-8-sig')
    changes={
        "pd.read_csv(P/'weather_robust_design/site_portfolio_designs.csv')":"pd.read_csv(INPUT/'design.csv')",
        "pd.read_csv(P/'climate_coupling/demand_supply_country_crosswalk/facility_to_era5_grid_with_country.csv',low_memory=False)":"pd.read_csv(ROOT/'inputs/sites.csv',low_memory=False)",
        "json.loads((ROOT/'output/24x7_cfe_facility/adaptation_economics_geo/S3/adaptation_economics.json').read_text())":"json.loads((INPUT/'old_geoS3_economics.json').read_text())",
        'for year in range(1995,2015):':'for year in [YEAR]:',
        "pd.Index(np.load(CACHE/f'historical_{year}_ids.npy',allow_pickle=True))":'IDS',
        "pd.Index(np.load(CACHE/f'future_ACCESS-CM2_{year}_ids.npy',allow_pickle=True))":'IDS',
        "np.load(CACHE/f'historical_{year}.npy',mmap_mode='r')":'HISTORICAL',
        "np.load(CACHE/f'future_ACCESS-CM2_{year}.npy',mmap_mode='r')":'FUTURE',
        "['replay_results_pv_corrected','replay_results_future_ssp245']":'[str(H_SUPPLY),str(F_SUPPLY)]',
    }
    for old,new in changes.items():
        assert original.count(old)==1,old
        original=original.replace(old,new)
    tree=ast.parse(original)
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
    namespace=dict(Path=Path,np=np,pd=pd,json=json,time=time,factorial=factorial,
        ROOT=ROOT,INPUT=INPUT,P=INPUT,SUP=INPUT,OUT=out,YEAR=year,IDS=ids,
        HISTORICAL=historical.astype(np.float64),FUTURE=future.astype(np.float64),
        H_SUPPLY=h_supply,F_SUPPLY=f_supply)
    exec(compile(ast.Module(body=functions,type_ignores=[]),'original_script69_adapted_io','exec'),namespace)
    namespace['main']()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('model');parser.add_argument('ssp');parser.add_argument('year',type=int)
    args=parser.parse_args();model,ssp,year=args.model,args.ssp,args.year
    assert model in ['ACCESS-CM2','GFDL-ESM4','MIROC6'] and ssp in ['ssp126','ssp245','ssp585'] and 1995<=year<=2014
    started=time.perf_counter();out=BASE/'results'/model/ssp/str(year);out.mkdir(parents=True,exist_ok=True)
    marker=out/'complete.json'
    if marker.exists():print(marker.read_text());return
    with np.load(ROOT/'inputs'/f'paired_{year}.npz',allow_pickle=False) as data:
        ids=pd.Index(data['ids'].astype(str));historical=data['historical']
        if ssp=='ssp245' and model in data:future=data[model]
        else:
            compact=ROOT/'inputs/compact_demand'/(model if ssp=='ssp245' else f'{model}_{ssp}')
            record=json.loads((compact/'complete.json').read_text())
            if ssp!='ssp245':assert record['validation_passed'] and record['model']==model and record['ssp']==ssp
            assert np.array_equal(np.load(compact/'ids.npy').astype(str),ids.to_numpy())
            future=np.load(compact/f'{year}.npy',mmap_mode='r')
    assert historical.shape==future.shape and np.isfinite(historical).all() and np.isfinite(future).all()
    h_supply=ROOT/'inputs/supply/historical'
    f_supply=ROOT/'inputs/supply'/model if ssp=='ssp245' else ROOT/'new_ssp_supply_replay_pv_corrected'/model/ssp
    if ssp != 'ssp245':
        correction=json.loads((f_supply/'pv_correction_complete.json').read_text())
        assert correction['passed'] and correction['weather_years']==20
    mechanism=out/'mechanism';mechanism.mkdir(exist_ok=True)
    run_mechanism(year,ids,historical,future,h_supply,f_supply,mechanism)
    checks={};baseline=model=='ACCESS-CM2' and ssp=='ssp245'
    counter=pd.read_csv(mechanism/'procurement_only_counterfactual_costs.csv')
    positive=counter[counter.extra_procurement_pct>0]
    # Script69 bisects an initial [0,0.25] expansion interval 32 times.
    # One discrete solver step is about US$1.8, even with identical physical
    # outputs. Bound cost-only comparisons by two steps; physical checks stay strict.
    assert counter.extra_procurement_pct.max()<=25
    cost_quantum=float(.25/2**32*(positive.annual_procurement_restoration_usd/positive.extra_procurement_pct*100).max()) if len(positive) else 0.
    if baseline:
        for name,keys,cols in [
            ('four_component_shapley_by_year.csv',['year','metric','channel'],['contribution','total_delta']),
            ('counterfactual_country_pool_cells.csv',['year','mask','metric'],['value']),
            ('procurement_only_counterfactual_costs.csv',['year','mask'],['extra_procurement_pct','annual_procurement_restoration_usd']),
            ('window_128_sensitivity_by_year.csv',['year','accounting','metric'],['historical','future','change_pct'])]:
            ref=pd.read_csv(INPUT/'baseline_mechanism'/name);ref=ref[ref.year.eq(year)]
            checks[name]=compare(pd.read_csv(mechanism/name),ref,keys,cols,'mechanism',cost_tolerance=2*cost_quantum+1e-4)
    print(json.dumps({'stage':'mechanism_verified','model':model,'ssp':ssp,'year':year}),flush=True)

    m=module('adapt64','64_compute_adaptation_frontier_and_compound_events.py')
    m.GEO.SUPPLY_H=h_supply;m.GEO.SUPPLY_F=f_supply
    geographic_country_cf=m.GEO.country_cf
    design=pd.read_csv(INPUT/'design.csv');ladder=pd.read_parquet(ROOT/'inputs/mixes.parquet')
    master=pd.read_csv(ROOT/'inputs/sites.csv',low_memory=False)
    summary=[]
    for experiment in EXPERIMENTS:
        # The original main surface predates the geographic extension: script64
        # stored country capacity factors as float32; script66's network path
        # retains float64. Reuse both original paths instead of relaxing checks.
        m.GEO.country_cf=m.country_cf if experiment=='main_robust' else geographic_country_cf
        scenario='S3' if experiment=='main_robust' else experiment
        mixes=(design.set_index('logical_site_id').wind_share_robust_mean_20y if experiment=='main_robust'
               else ladder[ladder.scenario.eq(scenario)].drop_duplicates('logical_site_id').set_index('logical_site_id').wind_energy_share)
        sites=master[master.logical_site_id.isin(mixes.index)].sort_values('logical_site_id').reset_index(drop=True)
        sites['region']=sites.continent.where(~sites.continent.isin(['Unknown','Antarctica']))
        regions=sites.dropna(subset=['region']).groupby('supply_country_area_code').region.agg(lambda v:v.mode().iloc[0])
        take=ids.get_indexer(sites.logical_site_id.astype(str));assert (take>=0).all()
        load_h=np.ascontiguousarray(historical[take],dtype=np.float32)
        load_f=np.ascontiguousarray(future[take],dtype=np.float32)
        groups,pools=m.GEO.pool_spec(scenario,year,sites,regions,None)
        gen_h,gen_f=m.build_generation(mixes.reindex(sites.logical_site_id).to_numpy(float),load_h,groups,pools)
        country_groups=sites.groupby('supply_country_area_code',observed=True).indices
        positions=[country_groups[c] for c in sorted(country_groups)]
        ch=m.aggregate_country(load_h,positions);cf=m.aggregate_country(load_f,positions)
        gh=m.aggregate_country(gen_h,positions);gf=m.aggregate_country(gen_f,positions)
        hist_metrics=m.no_storage_metrics(ch,gh)
        surface=m.storage_surface(cf,gf,m.VOLUME_GRID_PCT,m.STORAGE_GRID_H,.85)
        for key,value in hist_metrics.items():surface['historical_'+key]=value
        surface.insert(0,'weather_year',year);surface.insert(0,'model',model)
        assert len(surface)==len(m.VOLUME_GRID_PCT)*len(m.STORAGE_GRID_H)
        assert np.isfinite(surface.select_dtypes('number')).all().all()
        assert surface.coverage.between(0,1).all()
        ref=pd.read_csv(INPUT/'baseline_surfaces'/f'{experiment}.csv');ref=ref[ref.weather_year.eq(year)]
        cols=[c for c in surface if c.startswith('historical_')]
        if baseline:cols += ['coverage','uncovered_mwh','worst_24h_mwh','worst_168h_mwh','chronological_worst_24h_mwh','chronological_worst_168h_mwh']
        candidate=out/f'surface_{experiment}.candidate.csv';surface.to_csv(candidate,index=False)
        checks['surface_'+experiment]=compare(surface,ref,['weather_year','additional_procurement_pct','storage_hours_mean_load'],cols,experiment)
        candidate.replace(out/f'surface_{experiment}.csv')
        summary.append({'experiment':experiment,'sites':len(sites),'countries':len(positions),'historical_mwh':float(ch.sum(dtype=np.float64)),'future_mwh':float(cf.sum(dtype=np.float64))})
        print(json.dumps({'stage':'surface_verified','experiment':experiment,'year':year,'elapsed_seconds':time.perf_counter()-started}),flush=True)
        del load_h,load_f,gen_h,gen_f,ch,cf,gh,gf

    # Physical totals retain annual generation; demand totals retain all source sites.
    annuals=[]
    for state,source in [('historical',h_supply),('future',f_supply)]:
        frame=pd.read_parquet(source/'annual_country_supply_1995_2014.parquet')
        frame=frame[frame.weather_year.eq(year)].copy();frame.insert(0,'climate_state',state);annuals.append(frame)
    pd.concat(annuals).to_csv(out/'physical_supply_by_country.csv',index=False)
    d=master.set_index('logical_site_id').reindex(ids).rename_axis('logical_site_id').reset_index()
    d['historical_mwh']=historical.sum(axis=1,dtype=np.float64);d['future_mwh']=future.sum(axis=1,dtype=np.float64)
    d[['logical_site_id','supply_country_area_code','historical_mwh','future_mwh']].to_csv(out/'demand_by_site.csv',index=False)
    report={'model':model,'ssp':ssp,'year':year,'validation_passed':True,'experiments':summary,
        'seconds':time.perf_counter()-started,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
        'baseline_checks':checks,'methods':'Original script69 exact Shapley and script64 full country-pooled surfaces; unchanged float64/float32 conventions.',
        'storage_efficiency':.85,'scope':'S3 robust design plus S3/S4/S5 year2000 designs; no new S1/S2 supply or compound-threshold evidence.'}
    report['procurement_cost_solver_step_usd']=cost_quantum
    tmp=marker.with_suffix('.partial');tmp.write_text(json.dumps(report,indent=2));tmp.replace(marker)
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    import fcntl
    model,ssp,year=sys.argv[1],sys.argv[2],int(sys.argv[3])
    locks=ROOT/'pv_correction_audit_20260913/locks';locks.mkdir(parents=True,exist_ok=True)
    lock=(locks/f'result234_{model}_{ssp}_{year}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    main()
