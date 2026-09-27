"""Build reader-facing numbers from the completed persistence matrix."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path('/data2/ljt/datacenter_climate_result1_20260912/persistence_20260914')
validation=json.loads((ROOT/'calculation_validation.json').read_text())
assert validation['passed'] and validation['jobs']==180
ev=pd.read_csv(ROOT/'events_mean.csv')
ph=pd.read_csv(ROOT/'physical_mean.csv')
metrics={}
def summarize(frame,field,scale=1):
    h=frame[frame.case.eq('historical')].set_index(['model','ssp'])[field].sort_index()/scale
    f=frame[frame.case.eq('future')].set_index(['model','ssp'])[field].sort_index()/scale
    assert h.index.equals(f.index) and len(h)==9 and np.allclose(h,h.iloc[0],rtol=1e-10,atol=1e-8)
    d=f-h
    return {'historical':float(h.iloc[0]),'future_min':float(f.min()),'future_max':float(f.max()),
            'change_min':float(d.min()),'change_max':float(d.max()),'positive_combinations':int((d>1e-9).sum()),
            'negative_combinations':int((d<-1e-9).sum()),
            'focal_historical':float(h.loc[('ACCESS-CM2','ssp245')]),'focal_future':float(f.loc[('ACCESS-CM2','ssp245')]),
            'focal_change':float(d.loc[('ACCESS-CM2','ssp245')]),
            'by_combination':[{'model':m,'ssp':s,'historical':float(h.loc[(m,s)]),'future':float(f.loc[(m,s)]),'change':float(d.loc[(m,s)])} for m,s in h.index]}
primary=ev[ev.shortfall_threshold.eq(.5)]
for name,field,scale in [('persistent_days','days_3_weighted',1),('persistent_events','events_3_weighted',1),
    ('longest_spell_days','longest_days_weighted',1),('low_matching_days','low_matching_days_weighted',1),
    ('persistent_energy_twh','energy_3_mwh',1e6),('persistent_energy_share_pct','energy_share_3_pct',1),
    ('week_spells_days','days_7_weighted',1),('week_spells_energy_twh','energy_7_mwh',1e6),
    ('persistent_exposure_pct','exposed_3_weighted',.01)]:metrics[name]=summarize(primary,field,scale)
for name,field,scale in [('annual_uncovered_twh','uncovered_mwh',1e6),('annual_equivalent_h','uncovered_equivalent_h',1),
    ('weekly_deficit_gwh','worst_168h_mwh',1e3),('weekly_mean_gap_gw','worst_168h_average_gap_mw',1e3),
    ('weekly_equivalent_h','worst_168h_equivalent_h',1),('shortfall_hours','shortfall_hours_weighted',1)]:metrics[name]=summarize(ph,field,scale)
sensitivity={str(t):{name:summarize(ev[ev.shortfall_threshold.eq(t)],field,scale) for name,field,scale in
    [('persistent_days','days_3_weighted',1),('persistent_energy_twh','energy_3_mwh',1e6)]} for t in [.25,.5,.75]}
report={'passed':True,'primary_threshold':.5,'minimum_spell_days':3,'metrics':metrics,'sensitivity':sensitivity,
        'source_jobs':180,'interpretation':'uncovered energy within site-local low-matching spells; not realised outages; no coincident global event implied'}
(ROOT/'narrative_numbers.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:{x:y for x,y in v.items() if x!='by_combination'} for k,v in metrics.items()},indent=2))
