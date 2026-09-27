"""Physical-capacity inputs for greenfield CFE planning; all output is on data2."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse, importlib.util, json, os, sys, time, zipfile
import numpy as np
import pandas as pd

ROOT=Path('/data2/ljt/datacenter_climate_result1_20260912')
OUT=ROOT/'greenfield_cfe_20260915'
YEARS=list(range(1995,2015))
HOURS=[8784 if y%4==0 else 8760 for y in YEARS]
ENDS=np.r_[0,np.cumsum(HOURS)]
T=int(ENDS[-1])
COMBOS=[(m,s) for m in ['ACCESS-CM2','GFDL-ESM4','MIROC6'] for s in ['ssp245','ssp126','ssp585']]

def state_name(m,s):return m+'_'+s
def geo_module():
    spec=importlib.util.spec_from_file_location('gf_geo',ROOT/'research_pipeline/scripts/66_compute_geographic_climate_conditioning.py')
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m

def source_supply(state):
    if state=='historical':return ROOT/'inputs/supply/historical'
    m,s=state.rsplit('_',1)
    return ROOT/'inputs/supply'/m if s=='ssp245' else ROOT/'new_ssp_supply_replay_pv_corrected'/m/s

def write_array(path,shape):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        assert np.load(path,mmap_mode='r').shape==shape;return
    a=np.lib.format.open_memmap(path,mode='w+',dtype='float32',shape=shape);a.flush();del a

def unpack(label):
    folder=OUT/'inputs'/label;folder.mkdir(exist_ok=True)
    with np.load(OUT/'inputs'/(label+'.npz')) as z:
        for name in z.files:
            dest=folder/(name+'.npy')
            if not dest.exists():np.save(dest,z[name])
    return label

def init():
    with zipfile.ZipFile(OUT/'inputs/fallback_raw.zip') as z:z.extractall(OUT/'inputs')
    with ProcessPoolExecutor(max_workers=4) as p:
        for r in p.map(unpack,['onsite_h','onsite_f','local_h','local_f']):print('unpacked '+r,flush=True)
    with np.load(ROOT/'inputs/paired_1995.npz') as z:ids=z['ids'].astype(str)
    master=pd.read_csv(ROOT/'inputs/sites.csv',low_memory=False).set_index('logical_site_id').reindex(ids).reset_index()
    assert master.era5_grid_id.notna().all()
    master.to_csv(OUT/'inputs/sites.csv',index=False)
    np.save(OUT/'inputs/site_ids.npy',ids)
    regions=master[~master.continent.isin(['Unknown','Antarctica'])].groupby('supply_country_area_code').continent.agg(lambda x:x.mode().iloc[0])
    states=['historical']+[state_name(*j) for j in COMBOS]
    for state in states:
        write_array(OUT/'loads'/state/'demand.npy',(len(ids),T))
        annual=pd.read_parquet(source_supply(state)/'annual_country_supply_1995_2014.parquet')
        countries=sorted(annual.country_area_code.unique())
        for mode,keys in [('S3',countries),('S4',sorted(regions.unique())),('S5',['GLOBAL'])]:
            folder=OUT/'resources'/mode/state;folder.mkdir(parents=True,exist_ok=True)
            (folder/'keys.json').write_text(json.dumps(keys))
            for tech in ['pv','wind']:write_array(folder/(tech+'.npy'),(len(keys),T))
    print(json.dumps({'initialised':True,'sites':len(ids),'hours':T,'states':len(states)}),flush=True)

def year_job(year):
    marker=OUT/'prepared_years_v2'/f'{year}.json';marker.parent.mkdir(exist_ok=True)
    if marker.exists():return json.loads(marker.read_text())
    start=time.time();pos=YEARS.index(year);a,b=map(int,ENDS[pos:pos+2])
    ids=np.load(OUT/'inputs/site_ids.npy');master=pd.read_csv(OUT/'inputs/sites.csv',low_memory=False)
    regions=master[~master.continent.isin(['Unknown','Antarctica'])].groupby('supply_country_area_code').continent.agg(lambda x:x.mode().iloc[0])
    geo=geo_module()
    with np.load(ROOT/'inputs'/f'paired_{year}.npz',allow_pickle=False) as z:
        assert np.array_equal(z['ids'].astype(str),ids)
        for state in ['historical']+[state_name(*j) for j in COMBOS]:
            if state=='historical':d=z['historical']
            else:
                m,s=state.rsplit('_',1)
                if s=='ssp245' and m in z:d=z[m]
                else:
                    folder=ROOT/'inputs/compact_demand'/(m if s=='ssp245' else state)
                    assert np.array_equal(np.load(folder/'ids.npy').astype(str),ids)
                    d=np.load(folder/f'{year}.npy',mmap_mode='r')
            assert d.shape==(len(ids),b-a) and np.isfinite(d).all() and (d>=0).all()
            part=OUT/'year_chunks'/state;part.mkdir(parents=True,exist_ok=True)
            np.save(part/f'demand_{year}.npy',d);del d
            src=source_supply(state)
            codes,pv,wind=geo.country_cf(src,year)
            regional=geo.regional_cf(src,year,regions)
            gp,gw=geo.global_cf(src,year)
            pools={'S3':{str(c):(pv[i],wind[i]) for i,c in enumerate(codes)},'S4':regional,'S5':{'GLOBAL':(gp,gw)}}
            for mode in pools:
                folder=OUT/'resources'/mode/state;keys=json.loads((folder/'keys.json').read_text())
                assert set(keys)==set(pools[mode])
                for t,tech in enumerate(['pv','wind']):
                    value=np.stack([pools[mode][k][t] for k in keys]).astype('float32')
                    assert np.isfinite(value).all() and value.min()>=0 and value.max()<1.00001
                    out=np.load(folder/(tech+'.npy'),mmap_mode='r+');out[:,a:b]=value;out.flush();del out
    record=dict(passed=True,year=year,hours=b-a,seconds=time.time()-start,states=10)
    marker.write_text(json.dumps(record,indent=2));return record

def combine(state):
    started=time.time();ids=np.load(OUT/'inputs/site_ids.npy')
    values=np.empty((len(ids),T),dtype='float32')
    for year,a,b in zip(YEARS,ENDS[:-1],ENDS[1:]):values[:,a:b]=np.load(OUT/'year_chunks'/state/f'demand_{year}.npy')
    assert np.isfinite(values).all() and (values>=0).all()
    path=OUT/'loads'/state/'demand.npy';tmp=path.with_name('demand.complete.npy')
    np.save(tmp,values);tmp.replace(path)
    return dict(state=state,seconds=time.time()-started)

def main():
    p=argparse.ArgumentParser();p.add_argument('--init-only',action='store_true');args=p.parse_args()
    init()
    if args.init_only:return
    rows=[]
    with ProcessPoolExecutor(max_workers=10) as pool:
        fs={pool.submit(year_job,y):y for y in YEARS}
        for f in as_completed(fs):
            rows.append(f.result());state=dict(completed=len(rows),total=20,last=fs[f],pid=os.getpid())
            (OUT/'preparation_status.json').write_text(json.dumps(state));print(json.dumps(state),flush=True)
    with ProcessPoolExecutor(max_workers=6) as pool:
        for r in pool.map(combine,['historical']+[state_name(*j) for j in COMBOS]):print(json.dumps({'assembled':r}),flush=True)
    (OUT/'physical_inputs_complete.json').write_text(json.dumps(dict(passed=True,years=20,states=10,hours=T,normalised_by_weather_year=False,rows=rows),indent=2))

if __name__=='__main__':main()
