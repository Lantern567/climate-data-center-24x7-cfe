# -*- coding: utf-8 -*-
"""Read the S1--S5 panel and answer the questions the ladder was built for.

Four things the geographic scenarios are supposed to settle, per the proposal:

  * how much hourly coverage each widening of the procurement boundary buys;
  * whether that gain diminishes (H5) or stays roughly linear;
  * how much loose annual-certificate accounting overstates coverage (S5 - S3);
  * whether the tail moves more than the mean as the boundary widens (H1).

It also separates two things that look identical in a ranked list of weak
markets but are not: a country that scores low because its weather and load
shapes genuinely misalign, and one that scores low because the supply model
covers wind and solar only and that country's clean fleet is hydro, geothermal
or nuclear.  Reporting them together would dress a data-scope limit up as a
climate finding.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "24x7_cfe" / "processed"
PANEL = PROCESSED / "cfe_geographic_scenarios"
SUPPLY = PROCESSED / "wind_solar_supply_20y_fixed_2025" / "replay_results_pv_corrected"
DEFAULT_OUTPUT = ROOT / "output" / "24x7_cfe_facility" / "cfe_geographic_scenarios"

LADDER = ["S1", "S2", "S3", "S4", "S5"]
LABEL = {"S1": "场内", "S2": "本地直供", "S3": "同网区",
         "S4": "跨区（大洲上界）", "S5": "无地理约束"}
# A fleet this lopsided cannot express a wind-solar portfolio at all, so a low
# score says more about the modelled technology set than about the climate.
WIND_SHARE_FLOOR = 0.15
MIN_DEMAND_SHARE = 0.002


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel", type=Path, default=PANEL / "site_scenario_year_cfe.parquet")
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return p.parse_args()


def weighted(frame: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    out = (frame.assign(_w=frame.cfe_score * frame.annual_demand_mwh)
           .groupby(by, as_index=False)
           .agg(demand_mwh=("annual_demand_mwh", "sum"),
                numerator=("_w", "sum"),
                sites=("logical_site_id", "nunique"),
                wind_share=("wind_energy_share", "mean"),
                unmatched_mwh=("unmatched_energy_mwh", "sum"),
                worst_24h=("worst_24h_unmatched_mwh", "sum"),
                worst_168h=("worst_168h_unmatched_mwh", "sum"),
                longest_run_30pct=("longest_run_above_30pct", "mean")))
    out["cfe"] = out.numerator / out.demand_mwh
    return out.drop(columns="numerator")


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(args.panel)
    present = [s for s in LADDER if s in set(panel.scenario)]

    # ---- 1. the ladder -----------------------------------------------------
    by_year = weighted(panel, ["scenario", "weather_year"])
    ladder = by_year.groupby("scenario").agg(
        cfe_mean=("cfe", "mean"), cfe_min=("cfe", "min"), cfe_max=("cfe", "max"),
        cfe_cv=("cfe", lambda s: s.std(ddof=1) / s.mean()),
        wind_share=("wind_share", "mean"), sites=("sites", "first"),
        demand_mwh=("demand_mwh", "mean"),
    ).reindex(present)
    ladder["cfe_range_pp"] = (ladder.cfe_max - ladder.cfe_min) * 100
    print("=" * 68)
    print("地理阶梯：需求加权小时 CFE 分数（20 个天气年）\n")
    for scenario in present:
        row = ladder.loc[scenario]
        print(f"  {scenario} {LABEL[scenario]:<16s} CFE {row.cfe_mean:.4f}"
              f"  极差 {row.cfe_range_pp:4.2f} pp  CV {row.cfe_cv*100:5.3f}%"
              f"  风电份额 {row.wind_share:.2f}  站点 {int(row.sites):,}")

    # S1 and S2 are NOT nested, so their difference is not a boundary effect.
    # S1 lets a site build whatever it wants on its own land, so both
    # technologies are always available; S2 restricts it to the fleet that
    # already exists nearby, and 692 of 2,256 served cells have solar capacity
    # but no wind within the radius.  Those sites are forced into a pure-solar
    # portfolio, which matches hours badly.  Reading S1 -> S2 as "widening the
    # boundary costs 9 pp" would be wrong: the step also removes technology
    # freedom.  It is reported, but flagged.
    if {"S1", "S2"} <= set(present):
        ref = panel[panel.weather_year.eq(panel.weather_year.min())]
        s2 = ref[ref.scenario.eq("S2")][["logical_site_id", "wind_energy_share",
                                         "cfe_score", "annual_demand_mwh"]]
        s1 = ref[ref.scenario.eq("S1")][["logical_site_id", "cfe_score"]]
        joined = s2.merge(s1, on="logical_site_id", suffixes=("_s2", "_s1"))
        forced = joined.wind_energy_share == 0
        print("\nS1 与 S2 不是嵌套关系——差异同时包含地理与技术自由度两个维度")
        for label, mask in (("周边有风电项目", ~forced), ("周边无风电项目（被迫纯光伏）", forced)):
            block = joined[mask]
            share = block.annual_demand_mwh.sum() / joined.annual_demand_mwh.sum()
            delta = ((block.cfe_score_s2 - block.cfe_score_s1)
                     * block.annual_demand_mwh).sum() / block.annual_demand_mwh.sum()
            print(f"  {label:<26s} 需求占比 {share:5.1%}   S1→S2 {delta*100:+6.2f} pp")

    print("\n每扩大一级买到多少（百分点）")
    steps = []
    for a, b in zip(present, present[1:]):
        # Compare on the sites both scenarios actually serve, otherwise the step
        # mixes a coverage change into what should be a boundary effect.
        shared = (set(panel[panel.scenario.eq(a)].logical_site_id)
                  & set(panel[panel.scenario.eq(b)].logical_site_id))
        sub = panel[panel.logical_site_id.isin(shared)]
        pair = weighted(sub, ["scenario", "weather_year"]).pivot(
            index="weather_year", columns="scenario", values="cfe")
        gain = (pair[b] - pair[a]) * 100
        steps.append({"from": a, "to": b, "gain_pp_mean": float(gain.mean()),
                      "gain_pp_min": float(gain.min()), "gain_pp_max": float(gain.max()),
                      "shared_sites": len(shared)})
        print(f"  {a} → {b:<3s} {gain.mean():+6.2f} pp  "
              f"（{gain.min():+.2f} — {gain.max():+.2f}，共同站点 {len(shared):,}）")

    if len(steps) >= 2:
        print("\nH5 收益递减检验：")
        for earlier, later in zip(steps, steps[1:]):
            ratio = later["gain_pp_mean"] / earlier["gain_pp_mean"] \
                if earlier["gain_pp_mean"] else float("nan")
            verdict = "递减" if ratio < 0.8 else ("近似线性" if ratio < 1.2 else "递增")
            print(f"  {later['from']}→{later['to']} 相对 {earlier['from']}→{earlier['to']}"
                  f"：{ratio:.2f}× → {verdict}")

    # ---- 2. tail vs mean ---------------------------------------------------
    print("\n" + "=" * 68)
    print("尾部 vs 均值：各指标的二十年变异系数\n")
    tail_rows = []
    for scenario in present:
        block = by_year[by_year.scenario.eq(scenario)]
        cv = {name: block[name].std(ddof=1) / block[name].mean()
              for name in ("cfe", "unmatched_mwh", "worst_24h", "worst_168h",
                           "longest_run_30pct")}
        cv["scenario"] = scenario
        cv["tail_over_mean"] = cv["worst_168h"] / cv["cfe"]
        tail_rows.append(cv)
        print(f"  {scenario}: CFE {cv['cfe']*100:5.2f}%  未匹配 {cv['unmatched_mwh']*100:5.2f}%"
              f"  最差24h {cv['worst_24h']*100:5.2f}%  最差168h {cv['worst_168h']*100:5.2f}%"
              f"  最长连续 {cv['longest_run_30pct']*100:5.2f}%"
              f"   → 尾部/均值 {cv['tail_over_mean']:.1f}×")

    # ---- 3. climate-vulnerable vs scope-limited ----------------------------
    annual = pd.read_parquet(SUPPLY / "annual_country_supply_1995_2014.parquet")
    capacity = (annual[annual.weather_year.eq(2000)]
                .set_index("country_area_code")
                [["solar_pv_installed_capacity_mw", "wind_installed_capacity_mw"]])
    headline = "S3" if "S3" in present else present[-1]
    country = weighted(panel[panel.scenario.eq(headline)],
                       ["supply_country_area_code"]).set_index("supply_country_area_code")
    country = country.join(capacity)
    total_capacity = (country.solar_pv_installed_capacity_mw
                      + country.wind_installed_capacity_mw)
    country["wind_capacity_share"] = np.where(
        total_capacity > 0, country.wind_installed_capacity_mw / total_capacity, np.nan)
    big = country[country.demand_mwh >= country.demand_mwh.sum() * MIN_DEMAND_SHARE]
    scope_limited = big[big.wind_capacity_share < WIND_SHARE_FLOOR].sort_values("cfe")
    balanced = big[big.wind_capacity_share >= WIND_SHARE_FLOOR].sort_values("cfe")

    print("\n" + "=" * 68)
    print(f"国别格局（{headline}，占需求 ≥{MIN_DEMAND_SHARE*100:.1f}% 的 {len(big)} 个市场）\n")
    print(f"【A】机队结构受限（风电占装机 <{WIND_SHARE_FLOOR:.0%}）——低分主要是只建模风光的产物")
    for code, row in scope_limited.iterrows():
        print(f"     {code}  CFE {row.cfe:.3f}  风电占装机 {row.wind_capacity_share:.1%}"
              f"  需求份额 {row.demand_mwh/big.demand_mwh.sum()*100:.2f}%")
    print(f"\n【B】风光机队均衡，CFE 最低的 8 个——这才是形状/气候脆弱")
    for code, row in balanced.head(8).iterrows():
        print(f"     {code}  CFE {row.cfe:.3f}  风电占装机 {row.wind_capacity_share:.1%}"
              f"  需求份额 {row.demand_mwh/big.demand_mwh.sum()*100:.2f}%")
    print(f"\n  A 类占大市场需求 {scope_limited.demand_mwh.sum()/big.demand_mwh.sum()*100:.1f}%,"
          f" B 类 {balanced.demand_mwh.sum()/big.demand_mwh.sum()*100:.1f}%")
    print(f"  CFE 与风电装机占比相关 Pearson {big.cfe.corr(big.wind_capacity_share):+.3f}")

    payload = {
        "scenarios_present": present,
        "ladder": ladder.reset_index().to_dict("records"),
        "steps": steps,
        "tail_vs_mean": tail_rows,
        "scope_limited_markets": scope_limited.index.tolist(),
        "most_shape_vulnerable_markets": balanced.head(8).index.tolist(),
        "headline_scenario": headline,
    }
    (args.output_root / "geographic_ladder_analysis.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), "utf-8")
    by_year.to_csv(args.output_root / "scenario_year_summary.csv", index=False)
    country.to_csv(args.output_root / "country_scenario_summary.csv")


if __name__ == "__main__":
    main()
