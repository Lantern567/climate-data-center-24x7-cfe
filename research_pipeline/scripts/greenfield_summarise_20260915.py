"""Summarise verified complete hourly matching costs without mixing baselines."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from greenfield_optimise_20260915 import ROOT,OUT

def main():
    completed=json.loads((OUT/'optimisation_complete.json').read_text());assert completed['passed']
    metadata=json.loads((OUT/'pool_metadata.json').read_text())
    rows=[];yearly=[]
    for path in sorted((OUT/'solutions').glob('*/*/*.json')):
        r=json.loads(path.read_text());assert r['feasible'],(r['mode'],r['state'],r['pool_id'])
        assert r['relative_optimality_gap']<3e-6
        assert r['validated_max_hourly_gap_mw']<max(1e-5,r['reference_mean_load_mw']*2e-7)
        assert r['validated_max_soc_closure_mwh']<max(1e-4,r['reference_mean_load_mw']*2e-6)
        m=metadata[r['mode']][r['pool_id']]
        for year,metrics,demand,generation in zip(range(1995,2015),r.pop('yearly_metrics'),r['yearly_demand_mwh'],r['yearly_generation_mwh']):
            yearly.append(dict(mode=r['mode'],state=r['state'],pool_id=r['pool_id'],weather_year=year,
                               demand_mwh=demand,generation_mwh=generation,
                               annual_cost_usd=r['pv_annual_cost_usd']+r['wind_annual_cost_usd']+r['battery_annual_cost_usd']+m['delivery_usd_mwh']*generation,
                               gap_mwh=metrics[0],peak_gap_mw=metrics[1],worst24_mwh=metrics[2],worst168_mwh=metrics[3],
                               closure_mwh=metrics[4],charge_mwh=metrics[7],discharge_mwh=metrics[8],curtailment_mwh=metrics[9]))
        r.update(reference_annual_cost_usd=m['reference_annual_cost_usd'],reference_capex_usd=m['reference_capex_usd'],
                 reference_pv_mw=m['reference_pv_mw'],reference_wind_mw=m['reference_wind_mw'])
        rows.append(r)
    frame=pd.DataFrame(rows);assert len(frame)==completed['complete']
    fields=['sites','reference_mean_load_mw','future_or_historical_demand_mwh_year','pv_mw','wind_mw','battery_mwh','battery_mw',
            'pv_annual_cost_usd','wind_annual_cost_usd','battery_annual_cost_usd','delivery_annual_cost_usd','total_annual_cost_usd',
            'total_capex_usd','renewable_generation_mwh_year','curtailed_mwh_year','reference_annual_cost_usd','reference_capex_usd',
            'reference_pv_mw','reference_wind_mw','validated_gap_mwh']
    summary=frame.groupby(['mode','state'],as_index=False)[fields].sum()
    for i,r in summary.iterrows():
        h=summary[summary['mode'].eq(r['mode'])&summary.state.eq('historical')].iloc[0]
        summary.loc[i,'total_usd_per_mwh']=r.total_annual_cost_usd/r.future_or_historical_demand_mwh_year
        summary.loc[i,'full_matching_premium_over_reference_pct']=100*(r.total_annual_cost_usd/r.reference_annual_cost_usd-1)
        summary.loc[i,'climate_cost_increment_usd_year']=r.total_annual_cost_usd-h.total_annual_cost_usd
        summary.loc[i,'climate_cost_increment_pct']=100*(r.total_annual_cost_usd/h.total_annual_cost_usd-1)
        summary.loc[i,'battery_cost_share_pct']=100*r.battery_annual_cost_usd/r.total_annual_cost_usd
        summary.loc[i,'delivery_cost_share_pct']=100*r.delivery_annual_cost_usd/r.total_annual_cost_usd
        summary.loc[i,'storage_hours_reference_mean_load']=r.battery_mwh/r.reference_mean_load_mw
        summary.loc[i,'renewable_generation_to_load_ratio']=r.renewable_generation_mwh_year/r.future_or_historical_demand_mwh_year
        summary.loc[i,'curtailment_pct_generation']=100*r.curtailed_mwh_year/r.renewable_generation_mwh_year
    summary.to_csv(OUT/'full_matching_cost_summary.csv',index=False)
    frame.to_csv(OUT/'verified_pool_results.csv',index=False)
    pd.DataFrame(yearly).to_csv(OUT/'verified_yearly_dispatch.csv',index=False)
    assert np.allclose(summary.total_annual_cost_usd,summary[['pv_annual_cost_usd','wind_annual_cost_usd','battery_annual_cost_usd','delivery_annual_cost_usd']].sum(axis=1))
    report=dict(passed=True,pool_plans=len(frame),mode_climate_cases=len(summary),
                maximum_hourly_gap_mw=float(frame.validated_max_hourly_gap_mw.max()),total_gap_mwh=float(frame.validated_gap_mwh.sum()),
                maximum_relative_optimality_gap=float(frame.relative_optimality_gap.max()),
                full_matching_premium_denominator='complete annual cost of historical mean-annual-energy reference wind/PV assets, no storage',
                climate_cost_increment_denominator='cost of a separate historical-climate plan satisfying the same 100% hourly target',
                no_free_existing_assets=True,all_twenty_weather_years_satisfied=True)
    (OUT/'summary_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    print(summary[['mode','state','total_annual_cost_usd','total_usd_per_mwh','full_matching_premium_over_reference_pct','climate_cost_increment_pct','battery_cost_share_pct']].to_string(index=False),flush=True)

if __name__=='__main__':main()
