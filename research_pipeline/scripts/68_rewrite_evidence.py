"""Recompute evidence for the September 2026 Chinese manuscript revision.

Read-only with respect to prior results. Cost learning changes CAPEX and FOM
only: resource capacity factors, dispatch, efficiency and delivery are fixed.
Every cost scenario is minimised again over the existing engineering frontier.
These are historical-performance restoration costs, NOT 100% CFE system costs.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / 'data/24x7_cfe/processed'
OUT = ROOT / 'output/rewrite_20260908'
OUT.mkdir(parents=True, exist_ok=True)

def crf(n, r=.07):
    return r * (1+r)**n / ((1+r)**n-1)

def main():
    costs = pd.read_csv(P/'costs/atb_2024_v3_costs_compact.csv')
    def pick(tech, scenario, year):
        a = costs[(costs.technology_key==tech)&(costs.scenario==scenario)&(costs.year==year)]
        assert len(a)==1
        return a.iloc[0]
    def ann(tech, scenario, year):
        a=pick(tech,scenario,year)
        return (a.capex_2022usd_per_kw*crf(a.technical_life_years)+a.fixed_om_2022usd_per_kw_year)*1000
    scenarios=[('fixed_2030','Moderate',2030),('moderate_2050','Moderate',2050),('advanced_2050','Advanced',2050)]
    rows=[]; country_rows=[]; checks=[]
    for geo in ['S1','S2','S3','S4','S5']:
        saved=json.loads((ROOT/f'output/24x7_cfe_facility/adaptation_economics_geo/{geo}/adaptation_economics.json').read_text())
        a=saved['global_cost_assumptions']
        frontier=pd.read_csv(P/f'adaptation_economics_geo/{geo}/adaptation_cost_frontier.csv')
        for label, scenario, year in scenarios:
            wind=ann('landbased_wind',scenario,year)/(8760*pick('landbased_wind','Moderate',2030).capacity_factor_fraction)
            pv=ann('utility_pv',scenario,year)/(8760*pick('utility_pv','Moderate',2030).capacity_factor_fraction)
            price=a['global_wind_energy_share']*wind+(1-a['global_wind_energy_share'])*pv+a['delivery_cost_basis']['usd_per_mwh']
            battery=ann('battery_4h',scenario,year)/4
            base_bill=a['historical_procurement_mwh']*price
            fixed_base=a['historical_procurement_mwh']*a['renewable_procurement_lcoe_usd_per_mwh']
            for target, f in frontier.groupby('restoration_target'):
                f=f.copy()
                f['new_cost']=f.additional_procurement_pct/100*a['historical_procurement_mwh']*price+f.storage_hours_mean_load*a['future_mean_load_mw']*battery
                best=f.loc[f.new_cost.idxmin()]
                old=next(x for x in saved['global_restoration_summary'] if x['restoration_target']==target)
                old_cost=old['least_cost_billion_usd_per_year']*1e9
                if label=='fixed_2030':
                    assert np.isclose(best.new_cost,old_cost,rtol=1e-8)
                    checks.append(f'{geo}/{target}: published baseline reproduced')
                rows.append(dict(geography=geo,technology_scenario=label,target=target,
                    wind_price=wind,pv_price=pv,delivered_price_usd_mwh=price,battery_usd_mwh_year=battery,
                    extra_procurement_pct=best.additional_procurement_pct,storage_hours=best.storage_hours_mean_load,
                    restoration_cost_usd_year=best.new_cost,reference_procurement_bill_usd_year=fixed_base,
                    restoration_pct_reference_bill=best.new_cost/fixed_base*100,
                    restoration_cost_reduction_pct=(1-best.new_cost/old_cost)*100,
                    repriced_procurement_plus_restoration_usd_year=base_bill+best.new_cost,
                    repriced_total_change_pct=((base_bill+best.new_cost)/fixed_base-1)*100))
            countries=pd.read_csv(P/f'adaptation_economics_geo/{geo}/country_resilience_costs.csv')
            scale=ann('geothermal_hydroflash',scenario,year)/ann('geothermal_hydroflash','Moderate',2030)
            major=countries[countries.major_market].copy()
            unit=major.resilience_premium_usd_per_mean_mw_year.to_numpy()*scale
            country_rows.append(dict(geography=geo,technology_scenario=label,clean_firm_price_ratio=scale,
                countries=len(major),total_premium_usd=major.resilience_premium_billion_usd_per_year.sum()*1e9*scale,
                unit_premium_min=unit.min(),unit_premium_max=unit.max(),unit_premium_p90_p10_ratio=np.quantile(unit,.9)/np.quantile(unit,.1),
                unit_premium_p90_p10_gap=np.quantile(unit,.9)-np.quantile(unit,.1)))
    econ=pd.DataFrame(rows); econ.to_csv(OUT/'technology_cost_recalculation.csv',index=False)
    pd.DataFrame(country_rows).to_csv(OUT/'country_cost_scaling.csv',index=False)
    # Annual physical supply is not normalised: retain the generation-volume effect.
    supply_root=P/'wind_solar_supply_20y_fixed_2025'
    hist=pd.read_parquet(supply_root/'replay_results_pv_corrected/annual_country_supply_1995_2014.parquet')
    supply=[]
    for model,folder in [('ACCESS-CM2','replay_results_future_ssp245'),('GFDL-ESM4','replay_results_future_GFDL-ESM4'),('MIROC6','replay_results_future_MIROC6')]:
        future=pd.read_parquet(supply_root/folder/'annual_country_supply_1995_2014.parquet')
        pair=hist.merge(future,on=['weather_year','country_area_code'],suffixes=('_h','_f'),validate='one_to_one')
        for tech in ['solar_pv','wind','wind_solar']:
            col=tech+'_generation_energy_mwh'
            annual=pair.groupby('weather_year')[[col+'_h',col+'_f']].sum()
            country=pair.groupby('country_area_code')[[col+'_h',col+'_f']].mean()
            supply.append(dict(model=model,technology=tech,countries=len(country),years=len(annual),
                historical_twh=annual[col+'_h'].mean()/1e6,future_twh=annual[col+'_f'].mean()/1e6,
                change_pct=(annual[col+'_f'].mean()/annual[col+'_h'].mean()-1)*100,
                countries_increase=int((country[col+'_f']>country[col+'_h']).sum()),
                countries_decrease=int((country[col+'_f']<country[col+'_h']).sum())))
    pd.DataFrame(supply).to_csv(OUT/'physical_supply_changes.csv',index=False)
    # Demand components on the same 8,360-site scope, paired by weather year.
    droot=P/'climate_coupling'
    h=pd.read_csv(droot/'demand_replay_1995_2014/global_annual_demand_replay_1995_2014.csv')
    demand=[]
    for model,folder in [('ACCESS-CM2','demand_replay_future_ssp245'),('GFDL-ESM4','demand_future_GFDL-ESM4'),('MIROC6','demand_future_MIROC6'),('MRI-ESM2-0','demand_future_MRI-ESM2-0'),('MPI-ESM1-2-HR','demand_future_MPI-ESM1-2-HR')]:
        f=pd.read_csv(droot/folder/'global_annual_demand_replay_1995_2014.csv')
        pair=h.merge(f,on=['weather_year','demand_scope','pue_scenario'],suffixes=('_h','_f'),validate='one_to_one')
        for (scope,pue), b in pair.groupby(['demand_scope','pue_scenario']):
            vals={}
            for c in ['facility_energy_twh','it_energy_twh','climate_variable_overhead_twh']:
                vals[c+'_h']=b[c+'_h'].mean(); vals[c+'_f']=b[c+'_f'].mean(); vals[c+'_delta']=vals[c+'_f']-vals[c+'_h']
            total=vals['facility_energy_twh_delta']; cooling=vals['climate_variable_overhead_twh_delta']; it=vals['it_energy_twh_delta']
            demand.append(dict(model=model,scope=scope,pue_scenario=pue,years=len(b),**vals,
                facility_change_pct=100*total/vals['facility_energy_twh_h'],cooling_share_new_demand_pct=100*cooling/total,
                other_delta_twh=total-cooling-it))
    pd.DataFrame(demand).to_csv(OUT/'demand_component_changes.csv',index=False)
    # Existing decomposition: contribution ratios are computed AFTER weather averaging.
    decomp=[]
    for model in ['ACCESS-CM2','GFDL-ESM4','MIROC6']:
        df=pd.read_csv(P/f'level_shape_{model}/level_shape_channels.csv')
        for metric,b in df.groupby('metric'):
            m=b.mean(numeric_only=True)
            for channel in ['demand_level','demand_shape','supply_level','supply_shape','interaction_frozen_volume']:
                value=m.get(channel,np.nan); joint=m.get('joint_frozen_volume',np.nan)
                decomp.append(dict(model=model,metric=metric,channel=channel,mean_contribution=value,mean_joint=joint,
                    share_joint_pct=100*value/joint if joint else np.nan))
    pd.DataFrame(decomp).to_csv(OUT/'existing_decomposition_audit.csv',index=False)
    (OUT/'checks.json').write_text(json.dumps({'checks':checks,'scope':'Restoration of historical performance; learning changes prices only; national clean-firm scaling is a common-price sensitivity, not country-specific technology diffusion.'},indent=2),encoding='utf-8')
    print(econ[econ.geography.eq('S3')][['technology_scenario','target','restoration_cost_usd_year','restoration_cost_reduction_pct','repriced_total_change_pct']].to_string(index=False))
    print(pd.DataFrame(supply).to_string(index=False))
    print(pd.DataFrame(demand).query("scope=='global_reconciled_allocation'")[['model','pue_scenario','facility_change_pct','cooling_share_new_demand_pct','other_delta_twh']].to_string(index=False))

if __name__=='__main__': main()
