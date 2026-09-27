"""Potential onsite resources for countries with no fleet; original QDM rule."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import ast,importlib.util,json,sys
import numpy as np
import pandas as pd
from greenfield_prepare_20260915 import ROOT,OUT,COMBOS,state_name

GRIDS=['era5_g0384','era5_g0389','era5_g0408','era5_g0458']
Q=np.round(np.arange(.05,.96,.05),2)
QC=[f'q{int(round(v*100)):02d}' for v in Q]

def work(combo):
    m,s=combo;state=state_name(m,s)
    dest=OUT/'fallback'/f'{state}.npz';dest.parent.mkdir(exist_ok=True)
    if dest.exists():return state
    grids=pd.read_csv(OUT/'inputs/facility_grid_catalog.csv').set_index('era5_grid_id').loc[GRIDS]
    cells=pd.read_csv(ROOT/'nex_points/cells.csv')
    lookup={(round(r.era5_latitude,2),round(r.era5_longitude,2)):r.cell_id for r in cells.itertuples()}
    ids=[lookup[(round(r.era5_latitude,2),round(r.era5_longitude,2))] for r in grids.itertuples()]
    deltas=ROOT/'inputs/deltas' if m=='ACCESS-CM2' and s=='ssp245' else (OUT/'inputs/ssp245_deltas' if s=='ssp245' else ROOT/'new_ssp_supply_deltas'/m/s)
    tables={}
    for v in ['tas','sfcWind','rsds']:
        repaired=OUT/'inputs'/f'fallback_{m}_{s}_{v}.json'
        frame=pd.read_json(repaired) if repaired.exists() else pd.read_parquet(deltas/f'delta_{m}_{s}_{v}.parquet')
        frame=frame.loc[frame.cell_id.isin(ids)]
        assert len(frame)==12*len(set(ids)) and np.isfinite(frame[QC]).all().all()
        tables[v]={(str(r.cell_id),int(r.month)):np.array([getattr(r,c) for c in QC]) for r in frame.itertuples()}
    # Reuse the exact established monthly quantile-shift function.
    tree=ast.parse((ROOT/'research_pipeline/scripts/55_extract_supply_future_cf.py').read_text(encoding='utf-8-sig'))
    func=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_shift')
    ns=dict(np=np,DELTA_TABLES=tables,QUANTILES=Q)
    exec(compile(ast.Module(body=[func],type_ignores=[]),'original_qdm_shift','exec'),ns)
    shift=ns['_shift']
    spec=importlib.util.spec_from_file_location('gf_resource',ROOT/'research_pipeline/supply_resource_models.py')
    resource=importlib.util.module_from_spec(spec);sys.modules[spec.name]=resource;spec.loader.exec_module(resource)
    times=pd.date_range('1995-01-01','2015-01-01',freq='h',tz='UTC');month=times.month.to_numpy()
    pv=[];wind=[];hist_pv=[];hist_wind=[]
    for grid,cid,r in zip(GRIDS,ids,grids.itertuples()):
        with np.load(OUT/'inputs/fallback_raw/weather'/f'{grid}.npz') as w,np.load(OUT/'inputs/fallback_raw/resource'/f'{grid}.npz') as raw:
            kwargs=dict(surface_pressure_pa=w['sp_pa'],wind_u_100m_m_s=raw['u100_ms'],wind_v_100m_m_s=raw['v100_ms'],
                        surface_solar_radiation_downwards_j_m2=raw['ssrd_j_m2'],total_sky_direct_solar_radiation_at_surface_j_m2=raw['fdir_j_m2'])
            h,_=resource.interval_capacity_factors(times,r.era5_latitude,r.era5_longitude,w['t2m_k'],**kwargs)
            speed=np.hypot(raw['u100_ms'],raw['v100_ms']);f_speed=shift(speed,month,cid,'sfcWind',True)
            factor=np.divide(f_speed,speed,out=np.ones_like(f_speed),where=speed>1e-6)
            ssrd=raw['ssrd_j_m2'];sf=shift(ssrd,month,cid,'rsds',True)
            radiation_factor=np.divide(sf,ssrd,out=np.ones_like(sf),where=ssrd>1e-6)
            kwargs.update(wind_u_100m_m_s=raw['u100_ms']*factor,wind_v_100m_m_s=raw['v100_ms']*factor,
                          surface_solar_radiation_downwards_j_m2=np.maximum(sf,0),
                          total_sky_direct_solar_radiation_at_surface_j_m2=np.maximum(raw['fdir_j_m2']*radiation_factor,0))
            f,_=resource.interval_capacity_factors(times,r.era5_latitude,r.era5_longitude,shift(w['t2m_k'],month,cid,'tas',False),**kwargs)
            pv.append(f['pv_cf']);wind.append(f['wind_cf']);hist_pv.append(h['pv_cf']);hist_wind.append(h['wind_cf'])
    np.savez_compressed(dest,grid=GRIDS,cell_id=ids,pv=np.asarray(pv,dtype='float32'),wind=np.asarray(wind,dtype='float32'),
                        historical_pv=np.asarray(hist_pv,dtype='float32'),historical_wind=np.asarray(hist_wind,dtype='float32'))
    return state

if __name__=='__main__':
    with ProcessPoolExecutor(max_workers=9) as p:
        results=list(p.map(work,COMBOS))
    report=dict(passed=True,states=results,grids=GRIDS,method='original monthly QDM shift and physical resource conversion',
                purpose='new-build potential where original installed-capacity table has no fleet')
    (OUT/'fallback_complete.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
