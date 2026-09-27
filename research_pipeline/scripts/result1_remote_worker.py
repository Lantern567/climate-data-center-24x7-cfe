"""Portable paired matching worker. Reuses script 66 without altering its equations."""
from pathlib import Path
import importlib.util
import json
import resource
import time
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

def main():
    model, year = sys.argv[1], int(sys.argv[2])
    ssp = sys.argv[3] if len(sys.argv)>3 else 'ssp245'
    started = time.perf_counter()
    spec = importlib.util.spec_from_file_location('geo66', Path(__file__).with_name('66_compute_geographic_climate_conditioning.py'))
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)
    inp = ROOT / 'inputs'
    with np.load(inp / f'paired_{year}.npz', allow_pickle=False) as z:
        ids = pd.Index(z['ids'].astype(str))
        hist = z['historical']
        if model in z and ssp == 'ssp245':
            fut = z[model]
        else:
            compact=inp/'compact_demand'/(model if ssp=='ssp245' else f'{model}_{ssp}')
            assert (compact/'complete.json').exists()
            assert np.array_equal(np.load(compact/'ids.npy').astype(str),ids.to_numpy())
            fut=np.load(compact/f'{year}.npy',mmap_mode='r')
            if ssp=='ssp245':
                archived=pd.read_csv(inp/f'archived_demand_{model}.csv')
                reference=archived[archived.weather_year.eq(year)&archived.demand_scope.eq('global_reconciled_allocation')&archived.pue_scenario.eq('type_linked_proxy')]
                assert len(reference)==1
                assert np.isclose(np.sum(fut,dtype=np.float64)/1e6,reference.facility_energy_twh.iloc[0],rtol=1e-7,atol=1e-6), 'Rebuilt demand disagrees with archived original annual result'
            else:
                receipt=json.loads((compact/'complete.json').read_text())
                assert receipt['model']==model and receipt['ssp']==ssp and receipt['validation_passed']
                assert np.isclose(np.sum(fut,dtype=np.float64),receipt['annual_total_mwh'][str(year)],rtol=1e-12,atol=1e-6)
    def demand(root, y, sites, cache, label):
        assert y == year
        rows = ids.get_indexer(sites.logical_site_id.astype(str))
        assert (rows >= 0).all()
        return np.asarray((hist if label == 'historical' else fut)[rows], dtype=np.float64)
    g.load_demand = demand
    # This label is internal to original script 66; actual selected model is explicit above.
    g.SUPPLY_H = inp / 'supply' / 'historical'
    g.SUPPLY_F = inp / 'supply' / model if ssp=='ssp245' else ROOT/'new_ssp_supply_replay_pv_corrected'/model/ssp
    if ssp != 'ssp245':
        correction = json.loads((g.SUPPLY_F/'pv_correction_complete.json').read_text())
        assert correction['passed'] and correction['weather_years'] == 20
    sites = pd.read_csv(inp / 'sites.csv', low_memory=False)
    sites = sites[sites.supply_country_area_code.notna()].sort_values('logical_site_id').reset_index(drop=True)
    sites['region'] = sites.continent.where(~sites.continent.isin(['Unknown', 'Antarctica']))
    regions = sites.dropna(subset=['region']).groupby('supply_country_area_code').region.agg(lambda v: v.mode().iloc[0])
    ladder = pd.read_parquet(inp / 'mixes.parquet')
    summaries, mechanisms, countries = [], [], []
    for scenario in ['S3', 'S4', 'S5']:
        mix = ladder[ladder.scenario.eq(scenario)].drop_duplicates('logical_site_id').set_index('logical_site_id').wind_energy_share
        g.run_scenario(scenario, (year,), sites, mix, regions, None, inp, summaries, mechanisms, countries)
    out = ROOT / 'results' / model / str(year) if ssp=='ssp245' else ROOT/'new_ssp_results'/model/ssp/str(year)
    out.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summaries)
    mech = pd.DataFrame(mechanisms)
    assert np.isfinite(summary.select_dtypes('number')).all().all()
    assert summary.coverage.between(0, 1).all()
    assert np.allclose(mech.joint_pp, mech.demand_level_pp + mech.demand_shape_pp + mech.supply_timing_pp + mech.interaction_pp, atol=1e-8)
    baseline = pd.read_csv(inp / 'baseline_v31.csv')
    baseline = baseline[baseline.weather_year.eq(year) & baseline.scenario.isin(['S3','S4','S5'])]
    reference_cases = list(summary.case.unique()) if model == 'ACCESS-CM2' and ssp=='ssp245' else ['historical']
    a = summary[summary.case.isin(reference_cases)].set_index(['scenario','weather_year','case']).sort_index()
    b = baseline[baseline.case.isin(reference_cases)].set_index(['scenario','weather_year','case']).sort_index()
    assert a.index.equals(b.index)
    checks = {}
    for col in ['sites','demand_mwh','matched_mwh','coverage','uncovered_mwh'] + [f'worst_{w}h_mwh' for w in g.WINDOWS]:
        checks[col] = float(np.max(np.abs(a[col].to_numpy()-b[col].to_numpy())))
        if not np.allclose(a[col],b[col],rtol=2e-9,atol=1e-5):
            raise AssertionError(f'baseline mismatch {model} {year} {col}: {checks[col]}')
    for name, frame in [('summary',summary),('mechanisms',mech),('countries',pd.DataFrame(countries))]:
        frame.insert(0,'model',model)
        frame.insert(1,'ssp',ssp)
        frame.to_csv(out / f'{name}.csv',index=False)
    report = {'model':model,'ssp':ssp,'year':year,'scenarios':['S3','S4','S5'],
              'seconds':time.perf_counter()-started,'max_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
              'baseline_max_absolute_errors':checks,'validation_passed':True,
              'supply_level_correction':'pv_empirical_country_bias_correction_v1'}
    (out/'complete.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)

if __name__ == '__main__':
    import fcntl
    model,year=sys.argv[1],int(sys.argv[2]);ssp=sys.argv[3] if len(sys.argv)>3 else 'ssp245'
    locks=ROOT/'pv_correction_audit_20260913/locks';locks.mkdir(parents=True,exist_ok=True)
    lock=(locks/f'matching_{model}_{ssp}_{year}.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX)
    marker=(ROOT/'results'/model/str(year) if ssp=='ssp245' else ROOT/'new_ssp_results'/model/ssp/str(year))/'complete.json'
    if marker.exists():
        record=json.loads(marker.read_text());assert record['validation_passed']
        if ssp!='ssp245':assert record.get('supply_level_correction')=='pv_empirical_country_bias_correction_v1'
        print(json.dumps({'reused':True,'model':model,'ssp':ssp,'year':year}),flush=True)
    else:main()
