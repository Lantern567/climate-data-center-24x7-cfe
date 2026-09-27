from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / 'source_manifest.json').read_text())
for item in manifest:
    raw = (ROOT / item['path']).read_bytes()
    assert len(raw) == item['bytes'], item['path']
    assert hashlib.sha256(raw).hexdigest() == item['sha256'], item['path']

decomp = pd.read_csv(ROOT / 'output/multimodel_analysis_20260913/result234_mechanisms_mean.csv')
keys = ['climate_model', 'climate_ssp', 'metric']
groups = decomp.groupby(keys)
assert groups.total_delta.nunique().eq(1).all()
np.testing.assert_allclose(groups.contribution.sum(), groups.total_delta.first(), rtol=1e-9, atol=1e-5)

data = pd.read_csv(ROOT / 'output/commissioning_learning_20260919/commissioning_global_results.csv')
components = ['pv_capital', 'wind_capital', 'battery_capital', 'pv_fom', 'wind_fom', 'battery_fom', 'delivery_cost']
np.testing.assert_allclose(data[components].sum(axis=1), data.cost, rtol=1e-10)
np.testing.assert_allclose(100 * (1 - data.cost / data.baseline_cost), data.saving_pct, atol=1e-10)
np.testing.assert_allclose(data.baseline_cost / data.cost, data.supported_business_multiplier, atol=1e-10)
np.testing.assert_allclose(100 * (data.supported_business_multiplier - 1), data.supported_business_increase_pct, atol=1e-10)
for growth in [2, 5, 8]:
    factor = (1 + growth / 100) ** (data.year - 2030)
    np.testing.assert_allclose(100 * (factor * data.cost / data.baseline_cost - 1), data[f'growth{growth}_extra_budget_pct'], atol=1e-9)
contrib = pd.read_csv(ROOT / 'output/commissioning_learning_20260919/technology_contributions.csv')
np.testing.assert_allclose(contrib.renewables_saving_pp + contrib.battery_saving_pp, contrib.total_saving_pct, atol=1e-10)
plans = pd.read_csv(ROOT / 'output/commissioning_learning_20260919/pool_plans.csv')
assert len(plans) == 99615 and plans.pool.nunique() == 145
assert plans.hourly_gap_mw.max() == 0 and plans.cyclic_error_mwh.max() == 0
plan_keys = ['state', 'learning', 'doublings', 'year']
totals = plans.groupby(plan_keys).cost.sum()
reference = data.set_index(plan_keys).cost
assert totals.index.difference(reference.index).empty
np.testing.assert_allclose(totals, reference.reindex(totals.index), rtol=1e-10)
extra = reference.index.difference(totals.index)
assert (extra.get_level_values('year') == 2030).all()
for state, learning, doublings, year in extra:
    np.testing.assert_allclose(reference.loc[(state, learning, doublings, year)], totals.loc[(state, 'Base', 2, 2030)], rtol=1e-10)
report = {'passed': True, 'verified_files': len(manifest), 'decomposition_cases': len(groups), 'global_learning_rows': len(data), 'pool_plans': len(plans)}
out = ROOT / 'verification'
out.mkdir(exist_ok=True)
(out / 'release_checks.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report))
