"""Prepare operating boundaries, physical resource rows and complete cost bases."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import json,time
import numpy as np
import pandas as pd
from greenfield_prepare_20260915 import ROOT,OUT,T,COMBOS,state_name

STATES=['historical']+[state_name(*x) for x in COMBOS]

def reference(file,row):return dict(file=str(Path(file).relative_to(OUT)),row=int(row))

def aggregate_load(task):
    mode,state,groups=task
    dest=OUT/'pooled_demand'/mode/(state+'.npy');dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():return mode,state
    a=np.empty((len(groups),T),dtype='float32')
    # Demand is unchanged by the physical-capacity supply correction. Reuse the
    # already verified national aggregates, avoiding three reads of 60 GB of
    # facility loads. The old normalised generation arrays are never reused.
    experiment='main_robust' if mode=='S3' else mode
    m,s=('ACCESS-CM2','ssp245') if state=='historical' else state.rsplit('_',1)
    cursor=0
    for year in range(1995,2015):
        source=ROOT/'full_matching_20260915/hourly'/experiment/m/s/str(year)
        assert json.loads((source/'prepared.json').read_text())['passed']
        with np.load(source/'country_hourly.npz') as z:
            assert z['country'].astype(str).tolist()==[k for k,indices in groups]
            values=z['historical_load' if state=='historical' else 'future_load']
            a[:,cursor:cursor+values.shape[1]]=values;cursor+=values.shape[1]
    assert cursor==T
    assert np.isfinite(a).all() and (a>=0).all();np.save(dest,a)
    return mode,state

def main():
    assert json.loads((OUT/'physical_inputs_complete.json').read_text())['passed']
    assert json.loads((OUT/'fallback_complete.json').read_text())['passed']
    master=pd.read_csv(OUT/'inputs/sites.csv',low_memory=False)
    master['position']=np.arange(len(master))
    historical=np.load(OUT/'loads/historical/demand.npy',mmap_mode='r')
    means=historical.mean(axis=1,dtype='float64');master['reference_mean']=means
    np.save(OUT/'inputs/reference_mean_load_mw.npy',means)
    defaults=json.loads((ROOT/'inputs/pv_bias_correction_validation.json').read_text())['factor_construction']['default_factor']
    factors=pd.read_csv(OUT/'inputs/pv_country_correction_factors.csv').set_index('country_area_code').pv_correction_factor
    # Supplement the zero-fleet countries using fixed demand-weighted local
    # candidate sites, with the same allocation in historical and future states.
    fallback_audit=[]
    for state in STATES:
        file=OUT/'fallback'/((state if state!='historical' else 'ACCESS-CM2_ssp245')+'.npz')
        with np.load(file) as f:
            gids=f['grid'].astype(str);pv=f['pv' if state!='historical' else 'historical_pv'];wind=f['wind' if state!='historical' else 'historical_wind']
        folder=OUT/'resources/S3'/state;keys=json.loads((folder/'keys.json').read_text())
        p=np.load(folder/'pv.npy',mmap_mode='r+');w=np.load(folder/'wind.npy',mmap_mode='r+')
        for country in ['GNQ','MDV','SOM']:
            block=master[master.supply_country_area_code.eq(country)];idx=keys.index(country)
            weights=block.groupby('era5_grid_id').reference_mean.sum();weights/=weights.sum()
            pp=np.zeros(T);ww=np.zeros(T)
            for grid,weight in weights.items():
                r=int(np.flatnonzero(gids==grid)[0]);pp+=weight*pv[r];ww+=weight*wind[r]
            factor=float(factors.get(country,defaults));p[idx]=pp*factor;w[idx]=ww
            assert p[idx].mean()>0 and np.isfinite(p[idx]).all()
            fallback_audit.append(dict(state=state,country=country,sites=len(block),pv_factor=factor,weights=weights.to_dict()))
        p.flush();w.flush();del p,w
    ladder=pd.read_parquet(ROOT/'inputs/mixes.parquet')
    design=pd.read_csv(ROOT/'supplemental_20260913/inputs/design.csv')
    assumptions=pd.read_csv(ROOT/'recovery_investment_20260914/inputs/network_assumptions.csv').set_index('experiment')
    selected={};groups={}
    for mode in ['S1','S2','S3','S4','S5']:
        mix=(design.set_index('logical_site_id').wind_share_robust_mean_20y if mode=='S3' else
             ladder[ladder.scenario.eq(mode)].drop_duplicates('logical_site_id').set_index('logical_site_id').wind_energy_share)
        sites=master[master.logical_site_id.isin(mix.index)].copy();sites['original_wind_energy_share']=sites.logical_site_id.map(mix)
        assert sites.original_wind_energy_share.notna().all();selected[mode]=sites
        if mode in ['S3','S4','S5']:
            groups[mode]=[(str(k),v.position.tolist()) for k,v in sites.groupby('supply_country_area_code',sort=True)]
    jobs=[(mode,state,g) for mode,g in groups.items() for state in STATES]
    with ProcessPoolExecutor(max_workers=4) as pool:
        fs=[pool.submit(aggregate_load,j) for j in jobs]
        for f in as_completed(fs):print(json.dumps({'aggregated':f.result()}),flush=True)
    prices=json.loads((ROOT/'recovery_investment_20260914/inputs/price_interface_2030.json').read_text())['technologies']
    metadata={};arrays={}
    def arr(path):
        path=str(path)
        if path not in arrays:arrays[path]=np.load(path,mmap_mode='r')
        return arrays[path]
    for mode,sites in selected.items():
        metadata[mode]={};states=STATES if mode in groups else ['historical','ACCESS-CM2_ssp245']
        delivery=float(assumptions.loc['main_robust' if mode=='S3' else mode,'delivery_usd_mwh'])
        blocks=list(sites.groupby('supply_country_area_code',sort=True)) if mode in groups else [(r.logical_site_id,sites.loc[[i]]) for i,r in sites.iterrows()]
        for pool_row,(code,block) in enumerate(blocks):
            code=str(code);positions=block.position.to_numpy();mean=float(means[positions].sum())
            assert mean>0
            meta=dict(sites=len(block),reference_mean_load_mw=mean,delivery_usd_mwh=delivery,demand={},pv={},wind={})
            if mode=='S1':meta['pv_factor']=float(factors.get(block.supply_country_area_code.iloc[0],defaults))
            for state in states:
                if mode in groups:
                    meta['demand'][state]=reference(OUT/'pooled_demand'/mode/(state+'.npy'),pool_row)
                else:meta['demand'][state]=reference(OUT/'loads'/state/'demand.npy',positions[0])
                if mode in ['S1','S2']:
                    folder=OUT/'inputs'/((('onsite' if mode=='S1' else 'local')+'_h') if state=='historical' else (('onsite' if mode=='S1' else 'local')+'_f'))
                    ids=arr(folder/'era5_grid_id.npy').astype(str);ri=int(np.flatnonzero(ids==block.era5_grid_id.iloc[0])[0])
                    meta['pv'][state]=reference(folder/'pv_cf.npy',ri);meta['wind'][state]=reference(folder/'wind_cf.npy',ri)
                else:
                    folder=OUT/'resources'/mode/state;keys=json.loads((folder/'keys.json').read_text())
                    if mode=='S3':ri=keys.index(code)
                    elif mode=='S5':ri=0
                    else:
                        weights=block.groupby('continent').reference_mean.sum();weights/=weights.sum()
                        if len(weights)==1:ri=keys.index(weights.index[0])
                        else:
                            mixed=OUT/'mixed_resources'/state;mixed.mkdir(parents=True,exist_ok=True)
                            for tech in ['pv','wind']:
                                value=sum(float(weight)*arr(folder/(tech+'.npy'))[keys.index(region)].astype(float) for region,weight in weights.items())
                                np.save(mixed/(code+'_'+tech+'.npy'),value.astype('float32')[None,:])
                                meta[tech][state]=reference(mixed/(code+'_'+tech+'.npy'),0)
                            continue
                    for tech in ['pv','wind']:meta[tech][state]=reference(folder/(tech+'.npy'),ri)
            mh={tech:float(arr(OUT/meta[tech]['historical']['file'])[meta[tech]['historical']['row']].mean(dtype=float)) for tech in ['pv','wind']}
            mh['pv']*=meta.get('pv_factor',1.)
            share=float(np.average(block.original_wind_energy_share,weights=block.reference_mean))
            if mh['wind']<=0:share=0.
            elif mh['pv']<=0:share=1.
            assert mh['pv']>0 or mh['wind']>0
            p=mean*(1-share)/mh['pv'] if mh['pv']>0 else 0.
            w=mean*share/mh['wind'] if mh['wind']>0 else 0.
            meta.update(reference_pv_mw=p,reference_wind_mw=w,original_wind_energy_share=share,
                        reference_annual_cost_usd=p*prices['utility_pv']['annual_usd_mw_year']+w*prices['landbased_wind']['annual_usd_mw_year']+delivery*mean*T/20,
                        reference_capex_usd=1000*(p*prices['utility_pv']['capex_usd_kw']+w*prices['landbased_wind']['capex_usd_kw']))
            metadata[mode][code]=meta
        print(json.dumps({'mode_metadata':mode,'pools':len(metadata[mode]),'sites':len(sites)}),flush=True)
    (OUT/'pool_metadata.json').write_text(json.dumps(metadata))
    (OUT/'pool_metadata_validation.json').write_text(json.dumps(dict(passed=True,fallback=fallback_audit,
        modes={m:dict(pools=len(p),sites=sum(r['sites'] for r in p.values()),states=len(next(iter(p.values()))['demand'])) for m,p in metadata.items()},
        cost_basis='all new assets at ATB Moderate2030, annualised capital + FOM + gross-output delivery service',
        power='4h battery modules, equal AC charging/discharging rating E/4',
        resource_constraints='existing spatial resource weights; zero-fleet countries supplemented using local build potential',
        operating_boundaries='individual sites for S1/S2; national demand pools for S3/S4/S5; no network congestion or land limit',
        capacity_fixed_across_twenty_weather_samples=True,wind_pv_mix_optimised=True),indent=2))

if __name__=='__main__':main()
