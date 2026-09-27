"""Reprice verified engineering surfaces for the v11 Results 3.

Does not overwrite prior analyses. Main S3 uses the 20-weather-year robust mix;
the five-network comparison uses its separate year-2000 fixed-mix experiment.
All surfaces have national aggregation and power-unconstrained batteries.
Consequently they are conditional portfolio planning estimates, not feasible
independent-site or actual intercontinental dispatch costs.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / 'data/24x7_cfe/processed'
OUT = ROOT / 'output/rewrite_v11/adaptation'
OUT.mkdir(parents=True, exist_ok=True)
NAMES = {'S1': 'Independent site', 'S2': 'Metro cluster', 'S3': 'National network',
         'S4': 'Regional platform', 'S5': 'Global platform'}
METRICS = {'annual': ('coverage', 'historical_coverage', True),
           '24h': ('worst_24h_mwh', 'historical_worst_24h_mwh', False),
           '168h': ('worst_168h_mwh', 'historical_worst_168h_mwh', False),
           '24h_chronological': ('chronological_worst_24h_mwh', 'historical_worst_24h_mwh', False),
           '168h_chronological': ('chronological_worst_168h_mwh', 'historical_worst_168h_mwh', False)}


def crossing(x, y, level, higher):
    x, y = np.asarray(x, float), np.asarray(y, float)
    reached = y >= level if higher else y <= level
    if reached[0]:
        return float(x[0])
    if not reached.any():
        return float('nan')
    j = np.flatnonzero(reached)[0]
    return float(x[j-1] + (level-y[j-1])*(x[j]-x[j-1])/(y[j]-y[j-1]))


def frontier(surface, target):
    metric, hist, higher = METRICS[target]
    base = surface[(surface.additional_procurement_pct == 0) & (surface.storage_hours_mean_load == 0)].iloc[0]
    level = float(base[hist])
    zero = surface[surface.storage_hours_mean_load == 0].sort_values('additional_procurement_pct')
    intercept = crossing(zero.additional_procurement_pct, zero[metric], level, higher)
    if intercept == 0:
        return pd.DataFrame([{'expansion_pct': 0., 'storage_hours': 0.}])
    rows = []
    for x, block in surface.groupby('additional_procurement_pct', observed=True):
        if np.isfinite(intercept) and x >= intercept:
            continue
        block = block.sort_values('storage_hours_mean_load')
        h = crossing(block.storage_hours_mean_load, block[metric], level, higher)
        if np.isfinite(h):
            rows.append({'expansion_pct': float(x), 'storage_hours': h})
    if np.isfinite(intercept):
        rows.append({'expansion_pct': intercept, 'storage_hours': 0.})
    return pd.DataFrame(rows).sort_values('expansion_pct').reset_index(drop=True)


def price(f, a):
    f = f.copy()
    f['renewable_cost_usd_year'] = f.expansion_pct/100 * a['historical_demand_mwh'] * a['busbar_usd_mwh']
    f['delivery_cost_usd_year'] = f.expansion_pct/100 * a['historical_demand_mwh'] * a['delivery_usd_mwh']
    f['storage_cost_usd_year'] = f.storage_hours * a['future_mean_load_mw'] * a['battery_usd_mwh_year']
    f['total_cost_usd_year'] = f[['renewable_cost_usd_year', 'delivery_cost_usd_year', 'storage_cost_usd_year']].sum(axis=1)
    f['cost_usd_per_reference_mwh'] = f.total_cost_usd_year/a['historical_demand_mwh']
    f['least_cost_candidate'] = False
    f.loc[f.total_cost_usd_year.idxmin(), 'least_cost_candidate'] = True
    return f


def main():
    interface_path = ROOT/'output/rewrite_v11/future/price_interface_2030.json'
    interface = json.loads(interface_path.read_text(encoding='utf-8'))
    tech = interface['technologies']
    design = pd.read_csv(P/'weather_robust_design/site_portfolio_designs.csv')
    ladder = pd.read_parquet(P/'cfe_geographic_scenarios/site_scenario_year_cfe.parquet',
        columns=['weather_year', 'scenario', 'logical_site_id', 'wind_energy_share'])
    ladder = ladder[ladder.weather_year.eq(2000)].merge(design[['logical_site_id', 'mean_annual_demand_mwh']], on='logical_site_id', validate='many_to_one')
    assumptions, fronts, summary, yearly, audits = [], [], [], [], []
    for key in ['main_robust', 'S1', 'S2', 'S3', 'S4', 'S5']:
        main_case = key == 'main_robust'
        scenario = 'S3' if main_case else key
        source = P/'climate_adaptation_ACCESS-CM2' if main_case else P/f'climate_adaptation_ACCESS-CM2_geo/{scenario}'
        old = json.loads((ROOT/f'output/24x7_cfe_facility/adaptation_economics_geo/{scenario}/adaptation_economics.json').read_text())['global_cost_assumptions']
        d = design if main_case else ladder[ladder.scenario.eq(scenario)]
        mix = np.average(d.wind_share_robust_mean_20y if main_case else d.wind_energy_share, weights=d.mean_annual_demand_mwh)
        surface = pd.read_csv(source/'adaptation_surface_mean.csv')
        base = surface[(surface.additional_procurement_pct == 0) & (surface.storage_hours_mean_load == 0)].iloc[0]
        demand_f = base.uncovered_mwh/(1-base.coverage)
        a = dict(experiment=key, scenario=scenario, network=NAMES[scenario], sites=len(d),
            mix_design='robust_mean_20y' if main_case else 'fixed_reference_2000', wind_energy_share=float(mix),
            historical_demand_mwh=float(d.mean_annual_demand_mwh.sum()),
            future_mean_load_mw=float(demand_f/(365.25*24)),
            busbar_usd_mwh=float(mix*tech['landbased_wind']['energy_usd_mwh']+(1-mix)*tech['utility_pv']['energy_usd_mwh']),
            delivery_usd_mwh=old['delivery_cost_basis']['usd_per_mwh']*interface['usd2022_to_2023'],
            resource_distance_km=old['delivery_cost_basis']['distance_km'],
            battery_usd_mwh_year=tech['battery_4h']['annual_usd_mwh_year'],
            dollar_year=interface['dollar_year'], engineering_source=str(source.relative_to(ROOT)).replace('\\','/'),
            aggregation='national demand/supply pools for every network; conditional planning estimates')
        a['delivered_usd_mwh'] = a['busbar_usd_mwh']+a['delivery_usd_mwh']
        assumptions.append(a)
        for target in METRICS:
            f = price(frontier(surface, target), a)
            f.insert(0,'target',target); f.insert(0,'experiment',key)
            fronts.append(f)
            best = f.loc[f.least_cost_candidate].iloc[0].to_dict()
            best.update(scenario=scenario, network=NAMES[scenario], sites=len(d),
                distance_km=a['resource_distance_km'], historical_demand_mwh=a['historical_demand_mwh'],
                expansion_only_pct=float(f.iloc[-1].expansion_pct) if f.iloc[-1].storage_hours == 0 else None,
                storage_only_hours=float(f.iloc[0].storage_hours) if f.iloc[0].expansion_pct == 0 else None,
                expansion_only_cost_usd_year=float(f.iloc[-1].total_cost_usd_year) if f.iloc[-1].storage_hours == 0 else None,
                storage_only_cost_usd_year=float(f.iloc[0].total_cost_usd_year) if f.iloc[0].expansion_pct == 0 else None)
            summary.append(best)
            audits.append(dict(experiment=key, target=target, frontier_nonempty=len(f)>0,
                cost_nonnegative=bool((f.total_cost_usd_year>=0).all()), unique_minimum_count=int(f.least_cost_candidate.sum())))
        years = pd.read_csv(source/'adaptation_surface_by_year.csv')
        for yr, yf in years.groupby('weather_year', observed=True):
            b = yf[(yf.additional_procurement_pct == 0)&(yf.storage_hours_mean_load == 0)].iloc[0]
            ay = dict(a)
            ay['historical_demand_mwh'] = float(b.historical_uncovered_mwh/(1-b.historical_coverage))
            ay['future_mean_load_mw'] = float(b.uncovered_mwh/(1-b.coverage)/(8784 if yr%4==0 else 8760))
            for target in ['annual','24h','168h']:
                f = price(frontier(yf,target),ay)
                best = f.loc[f.least_cost_candidate].iloc[0].to_dict()
                best.update(experiment=key, target=target, weather_year=int(yr))
                yearly.append(best)
    af=pd.DataFrame(assumptions); ff=pd.concat(fronts, ignore_index=True); sf=pd.DataFrame(summary); yf=pd.DataFrame(yearly)
    af.to_csv(OUT/'assumptions.csv',index=False); ff.to_csv(OUT/'engineering_frontiers.csv',index=False)
    sf.to_csv(OUT/'cost_summary.csv',index=False); yf.to_csv(OUT/'weather_year_costs.csv',index=False)
    ranges=yf.groupby(['experiment','target']).agg(cost_mean_usd_year=('total_cost_usd_year','mean'),
        cost_min_usd_year=('total_cost_usd_year','min'), cost_max_usd_year=('total_cost_usd_year','max'),
        cost_p10_usd_year=('total_cost_usd_year',lambda x:x.quantile(.1)), cost_p90_usd_year=('total_cost_usd_year',lambda x:x.quantile(.9)),
        unit_cost_mean=('cost_usd_per_reference_mwh','mean'), unit_cost_min=('cost_usd_per_reference_mwh','min'),
        unit_cost_max=('cost_usd_per_reference_mwh','max'), n=('weather_year','size')).reset_index()
    ranges.to_csv(OUT/'weather_year_ranges.csv',index=False)
    verification=dict(price_interface_sha256=hashlib.sha256(interface_path.read_bytes()).hexdigest(),
        checks=audits, weather_results=len(yf), expected_weather_results=6*3*20,
        interpolation='linear crossings at all sampled expansion values plus exact interpolated zero-storage intercept; minimum over piecewise-linear frontier vertices',
        corrected=['scenario-specific fixed wind mix','scenario-specific historical demand sample','zero-storage crossing within each weather-year grid','ATB2025 Moderate2030 prices and dollar basis'],
        limitations=['national aggregation also applied to S1/S2; no implementable independent-site cost inference',
            'power unconstrained; full state of charge at event onset for tail lower envelope',
            'greedy chronological tail sensitivity is a feasible policy only under unconstrained power, not a proven upper bound for a real 4h battery',
            'five distances are discrete scenario attributes; sample and mix also vary',
            'mean-surface optimum and average weather-year optimum are different estimands'])
    assert len(yf)==360
    assert sf.total_cost_usd_year.ge(0).all()
    (OUT/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf-8')
    print(sf[sf.experiment.eq('main_robust')][['target','expansion_pct','storage_hours','total_cost_usd_year','expansion_only_pct','storage_only_hours']].to_string(index=False))
    print('Created',OUT)


if __name__=='__main__':
    main()
