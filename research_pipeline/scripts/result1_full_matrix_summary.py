"""Build the complete 3-model x 3-SSP Result-1 evidence table for figure revision."""
from pathlib import Path
import argparse
import json
import time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
MODELS=['ACCESS-CM2','GFDL-ESM4','MIROC6']
SSPS=['ssp126','ssp245','ssp585']
YEARS=list(range(1995,2015))

def collect():
    frames=[];jobs=[];missing=[]
    for model in MODELS:
        for ssp in SSPS:
            for year in YEARS:
                folder=ROOT/'results'/model/str(year) if ssp=='ssp245' else ROOT/'new_ssp_results'/model/ssp/str(year)
                marker=folder/'complete.json'
                if not marker.exists():missing.append([model,ssp,year]);continue
                record=json.loads(marker.read_text())
                assert record['validation_passed'] and record['model']==model and record['ssp']==ssp and record['year']==year
                if ssp!='ssp245':
                    assert record.get('supply_level_correction')=='pv_empirical_country_bias_correction_v1'
                frame=pd.read_csv(folder/'summary.csv')
                assert frame.model.eq(model).all() and frame.ssp.eq(ssp).all() and frame.weather_year.eq(year).all()
                assert set(frame.scenario)=={'S3','S4','S5'}
                frames.append(frame);jobs.append(record)
    status={'updated_at_unix':time.time(),'validated_jobs':len(jobs),'expected_jobs':180,'missing':missing,'complete':not missing}
    (ROOT/'full_matrix_status.json').write_text(json.dumps(status,indent=2))
    if missing:return False
    allrows=pd.concat(frames,ignore_index=True)
    assert not allrows.duplicated(['model','ssp','scenario','weather_year','case']).any()
    rows=[];paired=[]
    def ratio(a,b):return float(a/b) if b!=0 else None
    for (model,ssp,scenario),part in allrows.groupby(['model','ssp','scenario']):
        h=part[part.case.eq('historical')].set_index('weather_year').sort_index()
        f=part[part.case.eq('joint_frozen_volume')].set_index('weather_year').sort_index()
        r=part[part.case.eq('joint_resized')].set_index('weather_year').sort_index()
        assert h.index.tolist()==YEARS and h.index.equals(f.index) and h.index.equals(r.index)
        assert h.sites.nunique()==1 and (h.sites==f.sites).all()
        annual=(h.coverage-f.coverage)*100;week=f.worst_168h_mwh-h.worst_168h_mwh
        row={'model':model,'ssp':ssp,'scenario':scenario,'weather_years_completed':20,'complete_20_year_sample':True,'sites':int(h.sites.iloc[0]),'coverage_loss_pp':float(annual.mean()),'coverage_positive_years':int((annual>0).sum()),'worst_168h_delta_gwh':float(week.mean()/1000),'worst_168h_positive_years':int((week>0).sum()),'annual_loss_remaining_pct':ratio(100*(h.coverage-r.coverage).mean(),(h.coverage-f.coverage).mean()),'worst_168h_increment_remaining_pct':ratio(100*(r.worst_168h_mwh-h.worst_168h_mwh).mean(),week.mean())}
        for w in [1,6,24,72,168]:row[f'worst_{w}h_change_pct']=100*(f[f'worst_{w}h_mwh'].mean()/h[f'worst_{w}h_mwh'].mean()-1)
        rows.append(row)
        for year in YEARS:
            d={'model':model,'ssp':ssp,'scenario':scenario,'weather_year':year,'coverage_loss_pp':float(annual.loc[year]),'worst_168h_delta_gwh':float(week.loc[year]/1000)}
            for w in [1,6,24,72,168]:d[f'worst_{w}h_change_pct']=100*(f.loc[year,f'worst_{w}h_mwh']-h.loc[year,f'worst_{w}h_mwh'])/h[f'worst_{w}h_mwh'].mean()
            paired.append(d)
    assert len(rows)==27 and len(paired)==540
    out=ROOT/'full_matrix';out.mkdir(exist_ok=True)
    allrows.to_csv(out/'paired_summary_all_cases.csv',index=False)
    pd.DataFrame(rows).to_csv(out/'model_ssp_scenario_summary.csv',index=False)
    pd.DataFrame(paired).to_csv(out/'weather_year_paired_changes.csv',index=False)
    report={'validated_jobs':180,'summary_rows':27,'paired_year_rows':540,'models':MODELS,'ssps':SSPS,'scenarios':['S3','S4','S5'],'weather_years':YEARS,'complete':True,'scope':'Result1 S3/S4/S5 only; weather years and SSPs are not independent climate realizations'}
    report['supply_level_correction']='pv_empirical_country_bias_correction_v1'
    (out/'complete.json').write_text(json.dumps(report,indent=2))
    return True

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--once',action='store_true');args=p.parse_args()
    started=time.time()
    while time.time()-started<72*3600:
        if collect() or args.once:break
        time.sleep(60)
