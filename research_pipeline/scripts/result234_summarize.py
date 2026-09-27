"""Reprice every new physical frontier; never extrapolate old costs by climate ratios."""
from pathlib import Path
import argparse,calendar,importlib.util,json,sys,time
import numpy as np
import pandas as pd
from result234_remote_worker import BASE,INPUT,EXPERIMENTS,compare


def pricing_module():
    spec=importlib.util.spec_from_file_location('price72',Path(__file__).with_name('72_capacity_storage_distance.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def economics(surfaces, out, baseline_check=False, assumptions_override=None, name_prefix=''):
    m=pricing_module();assumptions=pd.read_csv(INPUT/'baseline_economics/assumptions.csv').set_index('experiment')
    if assumptions_override is not None:assumptions=assumptions_override
    paths=pd.read_csv(INPUT/'prices/technology_price_paths.csv')
    p0=paths[paths.scenario.eq('Moderate')&paths.year.eq(2030)].iloc[0]
    summaries=[];frontiers=[];weather=[];price_rows=[];shapley=[];frequencies=[]
    for experiment, years in surfaces.items():
        cols=['coverage','uncovered_mwh','worst_24h_mwh','worst_168h_mwh','chronological_worst_24h_mwh','chronological_worst_168h_mwh','historical_coverage','historical_uncovered_mwh','historical_worst_24h_mwh','historical_worst_168h_mwh']
        mean=years.groupby(['additional_procurement_pct','storage_hours_mean_load'],as_index=False)[cols].mean()
        base=mean[(mean.additional_procurement_pct==0)&(mean.storage_hours_mean_load==0)].iloc[0]
        a=assumptions.loc[experiment].to_dict()
        a['future_mean_load_mw']=float(base.uncovered_mwh/(1-base.coverage)/(365.25*24))
        for target in m.METRICS:
            frontier=m.frontier(mean,target)
            if frontier.empty:raise RuntimeError(f'{experiment}/{target}: target outside original engineering grid; no extrapolation')
            priced=m.price(frontier,a);best=priced.loc[priced.least_cost_candidate].iloc[0].to_dict()
            summaries.append(dict(experiment=experiment,target=target,**best))
            frontiers.append(priced.assign(experiment=experiment,target=target))
            def solution(p):
                b=dict(a);b['busbar_usd_mwh']=a['wind_energy_share']*p['wind']+(1-a['wind_energy_share'])*p['pv'];b['battery_usd_mwh_year']=p['battery']
                f=m.price(frontier,b);best=f.loc[f.least_cost_candidate].iloc[0]
                return dict(cost_usd_year=float(best.total_cost_usd_year),extra_capacity_pct=float(best.expansion_pct),storage_hours=float(best.storage_hours),
                    renewable_cost_usd=float(best.renewable_cost_usd_year+best.delivery_cost_usd_year),storage_cost_usd=float(best.storage_cost_usd_year),
                    delivered_usd_mwh=b['busbar_usd_mwh']+b['delivery_usd_mwh'])
            base_sol=solution(p0);reference_bill=a['historical_demand_mwh']*base_sol['delivered_usd_mwh']
            for p in paths.to_dict('records'):
                sol=solution(p);bill=a['historical_demand_mwh']*sol['delivered_usd_mwh']
                price_rows.append(dict(case=experiment,scenario=p['scenario'],year=p['year'],target=target,**sol,
                    reference_cost_usd_year=base_sol['cost_usd_year'],cost_reduction_pct=100*(1-sol['cost_usd_year']/base_sol['cost_usd_year']) if base_sol['cost_usd_year']>0 else np.nan,
                    reference_electricity_bill_usd=reference_bill,repriced_base_bill_usd=bill,
                    base_plus_restoration_change_pct=100*((bill+sol['cost_usd_year'])/reference_bill-1),
                    climate_premium_pct_same_year_bill=100*sol['cost_usd_year']/bill))
            for scenario in ['Conservative','Moderate','Advanced']:
                pf=paths[paths.scenario.eq(scenario)&paths.year.eq(2050)].iloc[0]
                c0=base_sol['cost_usd_year'];cr=solution(dict(wind=pf.wind,pv=pf.pv,battery=p0.battery))['cost_usd_year']
                cb=solution(dict(wind=p0.wind,pv=p0.pv,battery=pf.battery))['cost_usd_year'];both=solution(pf)['cost_usd_year']
                renew=.5*((c0-cr)+(cb-both));battery=.5*((c0-cb)+(cr-both))
                assert np.isclose(renew+battery,c0-both,atol=1e-5)
                shapley.append(dict(case=experiment,target=target,scenario=scenario,baseline_cost_usd=c0,renewables_savings_usd=renew,storage_savings_usd=battery,cost_2050_usd=both))
        for year, surface in years.groupby('weather_year'):
            b=surface[(surface.additional_procurement_pct==0)&(surface.storage_hours_mean_load==0)].iloc[0]
            ay=dict(a);ay['historical_demand_mwh']=float(b.historical_uncovered_mwh/(1-b.historical_coverage))
            ay['future_mean_load_mw']=float(b.uncovered_mwh/(1-b.coverage)/(8784 if calendar.isleap(int(year)) else 8760))
            for target in ['annual','24h','168h']:
                f=m.price(m.frontier(surface,target),ay);best=f.loc[f.least_cost_candidate].iloc[0].to_dict()
                weather.append(dict(experiment=experiment,target=target,weather_year=int(year),**best))
                metric,hist,higher=m.METRICS[target];tol=1e-10 if higher else .001
                reached=surface[metric]>=surface[hist]-tol if higher else surface[metric]<=surface[hist]+tol
                frequencies.append(surface[['additional_procurement_pct','storage_hours_mean_load']].assign(experiment=experiment,target=target,weather_year=int(year),restored=reached.to_numpy()))
    out.mkdir(parents=True,exist_ok=True)
    frames={'cost_summary.csv':pd.DataFrame(summaries),'engineering_frontiers.csv':pd.concat(frontiers),
        'weather_year_costs.csv':pd.DataFrame(weather),'future_restoration_paths.csv':pd.DataFrame(price_rows),'technology_savings_shapley.csv':pd.DataFrame(shapley)}
    frequency=pd.concat(frequencies).groupby(['experiment','target','additional_procurement_pct','storage_hours_mean_load'],as_index=False).restored.agg(['sum','count','mean']).reset_index()
    frames['restoration_sample_frequencies.csv']=frequency
    checks={}
    if baseline_check:
        ref=pd.read_csv(INPUT/'baseline_economics/cost_summary.csv');ref=ref[ref.experiment.isin(EXPERIMENTS)]
        checks['costs']=compare(frames['cost_summary.csv'],ref,['experiment','target'],['expansion_pct','storage_hours','total_cost_usd_year'],'repriced baseline')
        ref=pd.read_csv(INPUT/'baseline_economics/weather_year_costs.csv');ref=ref[ref.experiment.isin(EXPERIMENTS)]
        checks['weather_costs']=compare(frames['weather_year_costs.csv'],ref,['experiment','target','weather_year'],['total_cost_usd_year'],'weather baseline')
        ref=pd.read_csv(INPUT/'prices/future_restoration_paths.csv');ref=ref[ref['case'].isin(EXPERIMENTS)]
        actual=frames['future_restoration_paths.csv'];actual=actual.merge(ref[['case','scenario','year','target']],on=['case','scenario','year','target'],validate='one_to_one')
        checks['future_prices']=compare(actual,ref,['case','scenario','year','target'],['cost_usd_year','extra_capacity_pct','storage_hours'],'future price baseline')
    for name,frame in frames.items():frame.to_csv(out/(name_prefix+name),index=False)
    return checks


def summarize(model,ssp):
    roots=[BASE/'results'/model/ssp/str(y) for y in range(1995,2015)]
    for year,root in zip(range(1995,2015),roots):
        r=json.loads((root/'complete.json').read_text());assert r['model']==model and r['ssp']==ssp and r['year']==year and r['validation_passed']
    out=BASE/'summary'/model/ssp;out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():(out/'complete.json').replace(out/'complete_before_refresh.json')
    surfaces={e:pd.concat([pd.read_csv(p/f'surface_{e}.csv') for p in roots],ignore_index=True) for e in EXPERIMENTS}
    checks=economics(surfaces,out,baseline_check=model=='ACCESS-CM2' and ssp=='ssp245')
    attributes=pd.concat([pd.read_csv(p/'mechanism/four_component_shapley_by_year.csv') for p in roots],ignore_index=True)
    attributes.to_csv(out/'mechanisms_by_year.csv',index=False)
    mean=attributes.groupby(['metric','channel'],as_index=False)[['contribution','total_delta']].mean()
    mean['contribution_share_pct']=np.where(np.abs(mean.total_delta)>1e-9,mean.contribution/mean.total_delta*100,np.nan)
    mean.to_csv(out/'mechanisms_mean.csv',index=False)
    for name in ['procurement_only_counterfactual_costs.csv','window_128_sensitivity_by_year.csv']:
        pd.concat([pd.read_csv(p/'mechanism'/name) for p in roots]).to_csv(out/name,index=False)
    old_price=json.loads((INPUT/'old_geoS3_economics.json').read_text())['global_cost_assumptions']['renewable_procurement_lcoe_usd_per_mwh']
    a=pd.read_csv(INPUT/'baseline_economics/assumptions.csv').set_index('experiment').loc['main_robust']
    counter=pd.read_csv(out/'procurement_only_counterfactual_costs.csv')
    counter['annual_procurement_restoration_2023usd']=counter.annual_procurement_restoration_usd*a.delivered_usd_mwh/old_price
    counter.to_csv(out/'limited_wind_pv_counterfactual_costs_2023usd.csv',index=False)
    cost_attrs=attributes[attributes.metric.eq('annual_procurement_restoration_usd')].copy()
    cost_attrs[['contribution','total_delta']]*=a.delivered_usd_mwh/old_price
    cost_attrs.to_csv(out/'limited_wind_pv_cost_shapley_2023usd.csv',index=False)
    for name in ['physical_supply_by_country.csv','demand_by_site.csv']:
        pd.concat([pd.read_csv(p/name).assign(weather_year=y) for p,y in zip(roots,range(1995,2015))]).to_csv(out/name,index=False)
    # Audit legacy pricing samples against the actual engineering participants.
    # Retain the reproduced v31 series and publish this sensitivity separately.
    demand=pd.read_csv(out/'demand_by_site.csv')
    assumptions=pd.read_csv(INPUT/'baseline_economics/assumptions.csv').set_index('experiment')
    aligned=assumptions.copy();design=pd.read_csv(INPUT/'design.csv')
    ladder=pd.read_parquet(BASE.parent/'inputs/mixes.parquet')
    prices=pd.read_csv(INPUT/'prices/technology_price_paths.csv');p0=prices[prices.scenario.eq('Moderate')&prices.year.eq(2030)].iloc[0]
    sample_audit=[]
    for e in EXPERIMENTS:
        mix=design.set_index('logical_site_id').wind_share_robust_mean_20y if e=='main_robust' else ladder[ladder.scenario.eq(e)].drop_duplicates('logical_site_id').set_index('logical_site_id').wind_energy_share
        d=demand[demand.logical_site_id.isin(mix.index)].copy();d['wind']=d.logical_site_id.map(mix)
        assert d.wind.notna().all() and d.groupby('weather_year').logical_site_id.nunique().nunique()==1
        total=d.groupby('weather_year').historical_mwh.sum().mean();wind=np.average(d.wind,weights=d.historical_mwh)
        aligned.loc[e,'sites']=d.logical_site_id.nunique()
        aligned.loc[e,'historical_demand_mwh']=total;aligned.loc[e,'wind_energy_share']=wind
        aligned.loc[e,'busbar_usd_mwh']=wind*p0.wind+(1-wind)*p0.pv
        aligned.loc[e,'delivered_usd_mwh']=aligned.loc[e,'busbar_usd_mwh']+aligned.loc[e,'delivery_usd_mwh']
        sample_audit.append(dict(experiment=e,legacy_pricing_sites=int(assumptions.loc[e,'sites']),engineering_sites=d.logical_site_id.nunique(),
            legacy_reference_mwh=assumptions.loc[e,'historical_demand_mwh'],engineering_reference_mwh=total,
            legacy_wind_share=assumptions.loc[e,'wind_energy_share'],engineering_wind_share=wind))
    pd.DataFrame(sample_audit).to_csv(out/'economic_reference_sample_audit.csv',index=False)
    aligned.loc[list(EXPERIMENTS)].to_csv(out/'engineering_sample_assumptions.csv')
    economics(surfaces,out,assumptions_override=aligned,name_prefix='engineering_sample_')
    report={'model':model,'ssp':ssp,'weather_years':20,'validated_jobs':20,'experiments':list(EXPERIMENTS),'validation_passed':True,'checks':checks,'updated_at_unix':time.time(),'analysis_version':'result234_v3_pv_corrected_engineering_sample_pricing'}
    tmp=out/'complete.partial';tmp.write_text(json.dumps(report,indent=2));tmp.replace(out/'complete.json');return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--baseline',action='store_true');parser.add_argument('--model');parser.add_argument('--ssp');args=parser.parse_args()
    if args.baseline:
        data={e:pd.read_csv(INPUT/'baseline_surfaces'/f'{e}.csv') for e in EXPERIMENTS}
        checks=economics(data,BASE/'pricing_baseline',True)
        report={'passed':True,'checks':checks};(BASE/'pricing_baseline_gate.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
    else:print(json.dumps(summarize(args.model,args.ssp)))
