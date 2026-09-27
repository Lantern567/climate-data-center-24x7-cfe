from pathlib import Path
import os

for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '1'

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import itertools
import json
import time
import numpy as np
import pandas as pd
import greenfield_optimise_20260915 as G
from complete_matching_technology_20260917 import WarmSolver

LEARNING = {'Low': np.array([.10, .05, .10]),
            'Medium': np.array([.20, .10, .20]),
            'High': np.array([.25, .15, .30])}
STATES = [m + '_' + s for m in ['ACCESS-CM2', 'GFDL-ESM4', 'MIROC6']
          for s in ['ssp126', 'ssp245', 'ssp585']]
REP = 'ACCESS-CM2_ssp245'
OUT = None
META = None
PARAMETERS = None


def configure(root):
    global OUT, META, PARAMETERS
    G.ROOT = Path(root)
    G.OUT = G.ROOT / 'greenfield_cfe_20260915'
    OUT = G.ROOT / 'commissioning_learning_20260919'
    OUT.mkdir(parents=True, exist_ok=True)
    META = json.loads((G.OUT / 'pool_metadata.json').read_text())['S3']
    PARAMETERS = json.loads((G.ROOT / 'recovery_investment_20260914/inputs/price_interface_2030.json').read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix('.partial')
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    partial.replace(path)


def price_basis():
    keys = ['utility_pv', 'landbased_wind', 'battery_4h']
    p = PARAMETERS['technologies']
    divisor = np.array([1., 1., 4.])
    capex = np.array([p[k]['capex_usd_kw'] for k in keys]) * 1000 / divisor
    fom = np.array([p[k]['fom_usd_kw_year'] for k in keys]) * 1000 / divisor
    lives = np.array([p[k]['technical_life_years'] for k in keys])
    rate = PARAMETERS['real_discount_rate']
    crf = rate / (1 - (1 + rate) ** (-lives.astype(float)))
    return capex, fom, crf


def cases(state):
    values = [dict(learning='Base', doublings=2, year=2030, factor=np.ones(3))]
    speeds = [2, 1, 3] if state == REP else [2]
    for speed, label, year in itertools.product(speeds, LEARNING, range(2031, 2051)):
        factor = (1 - LEARNING[label]) ** (speed * (year - 2030) / 20)
        values.append(dict(learning=label, doublings=speed, year=year, factor=factor))
    high = (1 - LEARNING['High']) ** 2
    values.extend([
        dict(learning='RenewablesOnly', doublings=2, year=2050, factor=np.r_[high[:2], 1.]),
        dict(learning='StorageOnly', doublings=2, year=2050, factor=np.r_[1., 1., high[2]]),
    ])
    return values


def solve_pool(task):
    state, code = task
    target = OUT / 'solutions' / state / (code + '.json')
    if target.exists():
        assert json.loads(target.read_text())['passed']
        return dict(task=task, cached=True)
    started = time.time()
    meta = META[code]
    d, pv, wind = G.load_pool(meta, state)
    scale = meta['reference_mean_load_mw']
    base = json.loads((G.OUT / 'solutions/S3' / state / (code + '.json')).read_text())
    x0 = np.array([base['pv_mw'], base['wind_mw'], base['battery_mwh']]) / scale
    capex0, fom, crf = price_basis()
    generation = np.array([pv.mean(dtype=float), wind.mean(dtype=float), 0.]) * G.T / 20
    delivery = generation * meta['delivery_usd_mwh']
    solver = WarmSolver(d, pv, wind)
    rows = []
    for case in cases(state):
        capex = capex0 * case['factor']
        capital = capex * crf
        objective = capital + fom + delivery
        x, metrics, diag = solver.solve(objective)
        capacity = x * scale
        cost = float(capacity @ objective)
        frozen = float(x0 @ objective * scale)
        assert cost <= frozen * (1 + 5e-6)
        if case['year'] == 2030:
            assert np.isclose(cost, base['total_annual_cost_usd'], rtol=5e-6)
        row = dict(state=state, pool=code, learning=case['learning'],
                   doublings=case['doublings'], year=case['year'],
                   cost=cost, lower_bound=diag['lower_bound'] * scale,
                   baseline_cost=base['total_annual_cost_usd'], frozen_cost=frozen,
                   demand_mwh=float(d.sum(dtype=float)) * scale / 20,
                   pv_mw=capacity[0], wind_mw=capacity[1], battery_mwh=capacity[2],
                   pv_capital=capacity[0] * capital[0], wind_capital=capacity[1] * capital[1],
                   battery_capital=capacity[2] * capital[2],
                   pv_fom=capacity[0] * fom[0], wind_fom=capacity[1] * fom[1],
                   battery_fom=capacity[2] * fom[2],
                   delivery_cost=float(capacity @ delivery),
                   initial_capex=float(capacity @ capex),
                   pv_capex_factor=case['factor'][0], wind_capex_factor=case['factor'][1],
                   battery_capex_factor=case['factor'][2],
                   relative_optimality_gap=diag['relative_optimality_gap'],
                   hourly_gap_mw=diag['max_gap'] * scale,
                   cyclic_error_mwh=diag['max_cyclic_error'] * scale,
                   annual_gap_mwh=float(metrics[:, 0].mean()) * scale)
        assert np.isclose(sum(row[k] for k in ['pv_capital', 'wind_capital', 'battery_capital',
            'pv_fom', 'wind_fom', 'battery_fom', 'delivery_cost']), cost, rtol=1e-12)
        rows.append(row)
    write_json(target, dict(passed=True, task=task, seconds=time.time()-started, rows=rows))
    return dict(task=task, seconds=time.time()-started, plans=len(rows))


def aggregate():
    records = []
    for state, code in itertools.product(STATES, META):
        result = json.loads((OUT / 'solutions' / state / (code + '.json')).read_text())
        assert result['passed']
        records.extend(result['rows'])
    f = pd.DataFrame(records)
    assert not f.duplicated(['state', 'pool', 'learning', 'doublings', 'year']).any()
    assert f.relative_optimality_gap.max() < 3e-6
    assert f.hourly_gap_mw.max() < .1
    f.to_csv(OUT / 'pool_plans.csv', index=False)
    additive = ['cost', 'lower_bound', 'baseline_cost', 'frozen_cost', 'demand_mwh',
                'pv_mw', 'wind_mw', 'battery_mwh', 'pv_capital', 'wind_capital', 'battery_capital',
                'pv_fom', 'wind_fom', 'battery_fom', 'delivery_cost', 'initial_capex', 'annual_gap_mwh']
    s = f.groupby(['state', 'learning', 'doublings', 'year'])[additive].sum().reset_index()
    base = s[s.learning.eq('Base')].copy()
    copies = []
    for label in LEARNING:
        for speed in [1, 2, 3]:
            q = base.copy() if speed == 2 else base[base.state.eq(REP)].copy()
            q['learning'], q['doublings'] = label, speed
            copies.append(q)
    s = pd.concat([s, *copies], ignore_index=True)
    s['saving_pct'] = 100 * (1 - s.cost / s.baseline_cost)
    s['supported_business_multiplier'] = s.baseline_cost / s.cost
    s.loc[s.year.eq(2030), 'supported_business_multiplier'] = 1.
    s['supported_business_increase_pct'] = 100 * (s.supported_business_multiplier - 1)
    elapsed = s.year - 2030
    s['equivalent_annual_growth_pct'] = np.nan
    idx = elapsed.gt(0)
    s.loc[idx, 'equivalent_annual_growth_pct'] = 100 * (s.loc[idx, 'supported_business_multiplier'] ** (1/elapsed[idx]) - 1)
    s['usd_mwh'] = s.cost / s.demand_mwh
    s['configuration_saving_pp'] = 100 * (s.frozen_cost - s.cost) / s.baseline_cost
    for g in [2, 5, 8]:
        multiplier = (1 + g/100) ** elapsed
        s[f'growth{g}_cost'] = s.cost * multiplier
        s[f'growth{g}_extra_budget_pct'] = 100 * (s[f'growth{g}_cost'] / s.baseline_cost - 1)
        s[f'growth{g}_offset_fraction_pct'] = np.where(elapsed.gt(0),
            100 * multiplier * (s.baseline_cost - s.cost) / (s.baseline_cost * (multiplier - 1)), np.nan)
    for _, block in s[s.learning.isin(LEARNING)].groupby(['state', 'learning', 'doublings']):
        assert np.max(np.diff(block.sort_values('year').cost)) < block.cost.max() * 5e-6
    assert np.allclose(s.loc[idx, 'supported_business_multiplier'] * s.loc[idx, 'cost'], s.loc[idx, 'baseline_cost'], rtol=1e-12)
    s.to_csv(OUT / 'commissioning_global_results.csv', index=False)
    shapley = []
    for state in STATES:
        q = s[s.state.eq(state) & s.year.eq(2050) & s.doublings.eq(2)].set_index('learning')
        baseline = float(q.loc['High', 'baseline_cost'])
        both, ren, bat = (float(q.loc[l, 'cost']) for l in ['High', 'RenewablesOnly', 'StorageOnly'])
        r = .5 * ((baseline-ren) + (bat-both))
        b = .5 * ((baseline-bat) + (ren-both))
        assert np.isclose(r+b, baseline-both, rtol=1e-12)
        shapley.append(dict(state=state, renewables_saving_pp=100*r/baseline,
            battery_saving_pp=100*b/baseline, total_saving_pct=100*(baseline-both)/baseline))
    pd.DataFrame(shapley).to_csv(OUT / 'technology_contributions.csv', index=False)
    report = dict(passed=True, pool_plans=len(f), pools=len(META), climate_states=len(STATES),
        max_relative_optimality_gap=float(f.relative_optimality_gap.max()),
        max_hourly_gap_mw=float(f.hourly_gap_mw.max()), max_cyclic_error_mwh=float(f.cyclic_error_mwh.max()),
        commissioning_years=[2030,2050], weather_samples=20, price_formula='CAPEX0*(1-learning_rate)^(doublings*(year-2030)/20)',
        objective='independent new-build annualized capital + fixed O&M + gross-generation delivery',
        budget='same-climate baseline-scale 2030 annualized complete-matching cost',
        learning_rates={k:v.tolist() for k,v in LEARNING.items()}, order=['PV','wind','battery'],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(OUT / 'aggregate_validation.json', report)
    print(json.dumps(report), flush=True)


def validate():
    data = pd.read_csv(OUT / 'pool_plans.csv')
    capex0, fom, crf = price_basis()
    checks, growth_checks, direct_checks = [], [], []
    kernel = G.Kernel()
    for state, code in itertools.product(STATES, ['USA', 'DEU', 'CHN']):
        meta = META[code]
        d, pv, wind = G.load_pool(meta, state)
        scale = meta['reference_mean_load_mw']
        row = data[data.state.eq(state) & data.pool.eq(code) & data.learning.eq('High') & data.year.eq(2050) & data.doublings.eq(2)].iloc[0]
        cost = capex0*(1-LEARNING['High'])**2*crf + fom
        cost += meta['delivery_usd_mwh']*np.array([pv.mean(dtype=float),wind.mean(dtype=float),0.])*G.T/20
        cold = G.optimise(d,pv,wind,cost)
        error = abs(cold['annual_cost']*scale/row.cost-1)
        assert error < 6e-6
        checks.append(dict(state=state,pool=code,relative_cost_error=error))
        x = row[['pv_mw','wind_mw','battery_mwh']].to_numpy(float)/scale
        for multiplier in [(1.02)**20,(1.05)**20,(1.08)**20,row.baseline_cost/row.cost]:
            metrics,_,_,_ = kernel.dispatch(np.ascontiguousarray(d*multiplier,dtype='float32'),pv,wind,G.ENDS,x*multiplier)
            assert metrics[:,1].max() < 2e-6 and metrics[:,4].max() < 2e-5
            growth_checks.append(dict(state=state,pool=code,multiplier=multiplier,max_normalized_hourly_gap=float(metrics[:,1].max())))
        if state == REP:
            # 使用真实逐时样本对照显式线性规划，核对容量求解器。
            ends=np.array([0,168],dtype='int64')
            small=G.optimise(d[:168],pv[:168],wind[:168],cost,ends)
            direct=G.direct_lp(d[:168],pv[:168],wind[:168],cost,ends)
            error=abs(small['lower_bound_cost']/direct.fun-1)
            assert error<5e-6
            direct_checks.append(dict(pool=code,hours=168,relative_objective_error=error))
    write_json(OUT/'independent_validation.json',dict(passed=True,cold_start_checks=checks,scaled_dispatch_checks=growth_checks,real_hourly_explicit_lp_checks=direct_checks))
    print(json.dumps(dict(passed=True,cold_start_checks=len(checks),scaled_dispatch_checks=len(growth_checks),explicit_lp_checks=len(direct_checks))),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',default=str(G.ROOT))
    p.add_argument('--workers',type=int,default=32)
    p.add_argument('--stage',choices=['solve','aggregate','validate'],default='solve')
    p.add_argument('--pilot',action='store_true')
    args=p.parse_args()
    configure(args.root)
    if args.stage=='aggregate':aggregate();return
    if args.stage=='validate':validate();return
    tasks=list(itertools.product(STATES,META))
    if args.pilot:tasks=[(REP,c) for c in ['USA','DEU','CHN']]
    started=time.time()
    print(json.dumps(dict(tasks=len(tasks),workers=args.workers)),flush=True)
    with ProcessPoolExecutor(max_workers=args.workers,initializer=configure,initargs=(args.root,)) as ex:
        futures={ex.submit(solve_pool,task):task for task in tasks}
        for n,future in enumerate(as_completed(futures),1):
            result=future.result()
            if n%10==0 or n==len(tasks) or args.pilot:
                report=dict(done=n,total=len(tasks),seconds=time.time()-started,last=result)
                write_json(OUT/'status.json',report)
                print(json.dumps(report),flush=True)
    write_json(OUT/('pilot_complete.json' if args.pilot else 'complete.json'),dict(passed=True,tasks=len(tasks),seconds=time.time()-started))


if __name__=='__main__':
    main()
