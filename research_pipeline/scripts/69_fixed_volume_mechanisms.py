"""Paired country-pool diagnostic with actual renewable-volume changes.

Three changing factors are cooling-related annual demand (A), wind/PV annual
generation (V), and joint normalised load/generation profiles (T). Other demand
is fixed. Exact Shapley allocation includes every interaction once. Capacities
are held fixed within each historical/future weather-year pair; this is NOT a
common installed portfolio across all weather years. Annual-normalised results
are computed alongside the fixed-pair-capacity sensitivity.
"""
from pathlib import Path
from itertools import combinations
from math import factorial
import json, time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]; P=ROOT/'data/24x7_cfe/processed'
OUT=ROOT/'output/rewrite_20260908'; OUT.mkdir(exist_ok=True,parents=True)
CACHE=Path(r'C:\Users\Lantern567\.codex\visualizations\2026\08\24\01a03476-4217-7c61-a36b-347bab3591bf\geo_climate_cache\demand')
SUP=P/'wind_solar_supply_20y_fixed_2025'

def windows(gap):
    cs=np.pad(np.cumsum(gap,axis=1,dtype=np.float64),((0,0),(1,0)))
    return {str(w)+'h':(cs[:,w:]-cs[:,:-w]).max(axis=1).sum() for w in (24,128,168)}

def main():
    design=pd.read_csv(P/'weather_robust_design/site_portfolio_designs.csv').set_index('logical_site_id')
    sites=pd.read_csv(P/'climate_coupling/demand_supply_country_crosswalk/facility_to_era5_grid_with_country.csv',low_memory=False)
    sites=sites[sites.logical_site_id.isin(design.index)].copy()
    groups=list(sites.groupby('supply_country_area_code'))
    rows=[]; attrs=[]; cost_rows=[]; snapshots=[]
    assumptions=json.loads((ROOT/'output/24x7_cfe_facility/adaptation_economics_geo/S3/adaptation_economics.json').read_text())['global_cost_assumptions']
    for year in range(1995,2015):
        start=time.time()
        ids_h=pd.Index(np.load(CACHE/f'historical_{year}_ids.npy',allow_pickle=True))
        ids_f=pd.Index(np.load(CACHE/f'future_ACCESS-CM2_{year}_ids.npy',allow_pickle=True))
        hist=np.load(CACHE/f'historical_{year}.npy',mmap_mode='r')
        future=np.load(CACHE/f'future_ACCESS-CM2_{year}.npy',mmap_mode='r')
        supply=[]
        for directory in ['replay_results_pv_corrected','replay_results_future_ssp245']:
            fn=next((SUP/directory/'hourly_country_supply'/f'weather_year={year}').glob('*.parquet'))
            df=pd.read_parquet(fn,columns=['country_area_code','time_utc','solar_pv_generation_power_mw','wind_generation_power_mw'])
            supply.append({c:b.sort_values('time_utc')[['solar_pv_generation_power_mw','wind_generation_power_mw']].to_numpy().T for c,b in df.groupby('country_area_code')})
        dh=[]; df=[]; sh=[]; sf_norm=[]; sf_volume=[]; sf_raw=[]
        for country,b in groups:
            ih=ids_h.get_indexer(b.logical_site_id); iff=ids_f.get_indexer(b.logical_site_id)
            assert (ih>=0).all() and (iff>=0).all()
            lh=np.asarray(hist[ih]); lf=np.asarray(future[iff])
            energy=lh.sum(axis=1); wind=design.loc[b.logical_site_id,'wind_share_robust_mean_20y'].to_numpy()
            ph=supply[0][country]; pf=supply[1][country]
            eh=ph.sum(axis=1); ef=pf.sum(axis=1)
            # Reproduce technology availability convention in the prior kernel.
            if eh[0]<=0: wind=np.ones_like(wind)
            if eh[1]<=0: wind=np.zeros_like(wind)
            bought=np.array([np.dot(energy,1-wind),np.dot(energy,wind)])
            shape_h=np.divide(ph,eh[:,None],out=np.zeros_like(ph),where=eh[:,None]>0)
            shape_f=np.divide(pf,ef[:,None],out=np.zeros_like(pf),where=ef[:,None]>0)
            ratio=np.divide(ef,eh,out=np.ones_like(ef),where=eh>0)
            dh.append(lh.sum(axis=0)); df.append(lf.sum(axis=0))
            sh.append((bought[:,None]*shape_h).sum(axis=0))
            sf_norm.append((bought[:,None]*shape_f).sum(axis=0))
            sf_volume.append((bought[:,None]*ratio[:,None]*shape_h).sum(axis=0))
            sf_raw.append((bought[:,None]*ratio[:,None]*shape_f).sum(axis=0))
        dh,df,sh,sf_norm,sf_volume,sf_raw=map(np.asarray,[dh,df,sh,sf_norm,sf_volume,sf_raw])
        scale=df.sum(axis=1)/dh.sum(axis=1)
        baseline_gap=np.maximum(dh-sh,0)
        baseline={'uncovered_mwh':baseline_gap.sum(),'coverage_loss_pp':0.,**windows(baseline_gap)}
        base_coverage=1-baseline_gap.sum()/dh.sum()
        values={}; cost_values={}
        for mask in range(8):
            a,v,t=bool(mask&1),bool(mask&2),bool(mask&4)
            d=(df/scale[:,None] if t else dh)*(scale[:,None] if a else 1)
            s=sf_raw if v and t else sf_volume if v else sf_norm if t else sh
            gap=np.maximum(d-s,0)
            vals={'uncovered_mwh':gap.sum(),'coverage_loss_pp':(base_coverage-(1-gap.sum()/d.sum()))*100,**windows(gap)}
            values[mask]=vals
            for k,x in vals.items(): rows.append(dict(year=year,mask=mask,metric=k,value=x))
            # Cost attribution is restricted to procurement-only annual restoration.
            target_uncovered=(1-base_coverage)*d.sum()
            lo=0.; hi=.25
            while np.maximum(d-s*(1+hi),0).sum()>target_uncovered and hi<64: hi*=2
            if gap.sum()<=target_uncovered: hi=0.
            else:
                for _ in range(32):
                    mid=(lo+hi)/2
                    if np.maximum(d-s*(1+mid),0).sum()>target_uncovered: lo=mid
                    else: hi=mid
            cost=hi*dh.sum()*assumptions['renewable_procurement_lcoe_usd_per_mwh']
            cost_values[mask]=cost
            cost_rows.append(dict(year=year,mask=mask,extra_procurement_pct=hi*100,annual_procurement_restoration_usd=cost))
        for metric in values[0]:
            contributions=[]
            for factor,label in enumerate(['cooling_annual_volume','renewable_annual_volume','joint_timing']):
                value=0.
                for mask in range(8):
                    if mask&(1<<factor): continue
                    n=mask.bit_count(); weight=factorial(n)*factorial(2-n)/factorial(3)
                    value+=weight*(values[mask|(1<<factor)][metric]-values[mask][metric])
                contributions.append(value)
                attrs.append(dict(year=year,metric=metric,channel=label,contribution=value,total_delta=values[7][metric]-values[0][metric]))
            assert np.isclose(sum(contributions),values[7][metric]-values[0][metric],atol=1e-5)
            attrs.append(dict(year=year,metric=metric,channel='other_demand',contribution=0.,total_delta=values[7][metric]-values[0][metric]))
        ctotal=cost_values[7]-cost_values[0]
        csum=0.
        for factor,label in enumerate(['cooling_annual_volume','renewable_annual_volume','joint_timing']):
            value=sum(factorial(mask.bit_count())*factorial(2-mask.bit_count())/6*(cost_values[mask|(1<<factor)]-cost_values[mask]) for mask in range(8) if not mask&(1<<factor))
            csum+=value
            attrs.append(dict(year=year,metric='annual_procurement_restoration_usd',channel=label,contribution=value,total_delta=ctotal))
        assert np.isclose(csum,ctotal,atol=1e-4)
        attrs.append(dict(year=year,metric='annual_procurement_restoration_usd',channel='other_demand',contribution=0.,total_delta=ctotal))
        for tag,mask in [('normalised_annual_supply',5),('fixed_pair_capacity',7)]:
            for metric in ['uncovered_mwh','24h','128h','168h']:
                snapshots.append(dict(year=year,accounting=tag,metric=metric,historical=values[0][metric],future=values[mask][metric],change_pct=(values[mask][metric]/values[0][metric]-1)*100))
        pd.DataFrame(rows).to_csv(OUT/'counterfactual_country_pool_cells.csv',index=False)
        pd.DataFrame(attrs).to_csv(OUT/'four_component_shapley_by_year.csv',index=False)
        pd.DataFrame(cost_rows).to_csv(OUT/'procurement_only_counterfactual_costs.csv',index=False)
        pd.DataFrame(snapshots).to_csv(OUT/'window_128_sensitivity_by_year.csv',index=False)
        print(f'{year}: four components, 128-hour window and procurement-cost attribution complete ({time.time()-start:.1f}s)',flush=True)
    avg=pd.DataFrame(attrs).groupby(['metric','channel'])[['contribution','total_delta']].mean().reset_index()
    avg['contribution_share_pct']=avg.contribution/avg.total_delta*100
    avg.to_csv(OUT/'four_component_shapley_mean.csv',index=False)
    print(avg.to_string(index=False),flush=True)

if __name__=='__main__': main()
