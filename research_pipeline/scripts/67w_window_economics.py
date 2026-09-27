# -*- coding: utf-8 -*-
"""Price the dense-window restoration surface produced by 64w_window_sweep.py.

Follows the exact path that produced the published global_restoration_summary
in script 67: average the surface over weather years first, build the
restored-fraction field, take the storage=0 intercept as the procurement-only
endpoint, and interpolate the storage target at every volume level.  Only the
window set is extended.

The 24 h and 168 h rows must reproduce the published numbers; that check is
printed first and is the gate on trusting the intermediate windows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "24x7_cfe" / "processed"
DESIGN = PROCESSED / "weather_robust_design" / "site_portfolio_designs.csv"
PUBLISHED = ROOT / "output" / "24x7_cfe_facility" / "adaptation_economics_geo"
HOURS_PER_YEAR = 365.25 * 24.0


def interpolation_target(curve, metric, historical):
    curve = curve.sort_values("storage_hours_mean_load")
    x = curve.storage_hours_mean_load.to_numpy(dtype=float)
    y = curve[metric].to_numpy(dtype=float)
    reached = y <= historical
    if not reached.any():
        return np.nan, False
    index = int(np.flatnonzero(reached)[0])
    if index == 0:
        return float(x[0]), True
    x0, x1 = x[index - 1], x[index]
    y0, y1 = y[index - 1], y[index]
    if y1 == y0:
        return float(x1), True
    return float(x0 + (historical - y0) / (y1 - y0) * (x1 - x0)), True


def restoration_intercept(surface, field):
    zero = surface[np.isclose(surface.storage_hours_mean_load, 0.0)].sort_values(
        "additional_procurement_pct")
    x = zero.additional_procurement_pct.to_numpy(dtype=float)
    value = zero[field].to_numpy(dtype=float)
    for index in range(len(x) - 1):
        if (value[index] - 1.0) * (value[index + 1] - 1.0) <= 0:
            if np.isclose(value[index + 1], value[index]):
                return float(x[index])
            return float(x[index] + (1.0 - value[index])
                         * (x[index + 1] - x[index])
                         / (value[index + 1] - value[index]))
    return float("nan")


def solve_window(surface_mean, base, window, historical_procurement_mwh,
                 future_mean_load_mw, procurement_lcoe, battery_annuity):
    metric = "worst_" + str(window) + "h_mwh"
    hist_col = "historical_worst_" + str(window) + "h_mwh"
    penalty = float(base[metric]) - float(base[hist_col])
    if penalty <= 0:
        return None
    field = metric + "_restored_fraction"
    surface_mean = surface_mean.copy()
    surface_mean[field] = (float(base[metric]) - surface_mean[metric]) / penalty
    intercept = restoration_intercept(surface_mean, field)
    if not np.isfinite(intercept):
        return None

    rows = []
    for volume_value, curve in surface_mean.groupby("additional_procurement_pct",
                                                    sort=True):
        value, reachable = interpolation_target(curve, metric,
                                                float(base[hist_col]))
        rows.append({"additional_procurement_pct": float(volume_value),
                     "storage_hours_mean_load": value,
                     "reachable_within_grid": reachable})
    targets = pd.DataFrame(rows)
    block = targets[targets.additional_procurement_pct.lt(intercept)
                    & targets.storage_hours_mean_load.gt(1e-8)
                    & targets.reachable_within_grid]
    endpoint = pd.DataFrame([{"additional_procurement_pct": intercept,
                              "storage_hours_mean_load": 0.0,
                              "reachable_within_grid": True}])
    frontier = pd.concat([block, endpoint], ignore_index=True).sort_values(
        "additional_procurement_pct").reset_index(drop=True)

    frontier["procurement_cost_billion_usd_per_year"] = (
        frontier.additional_procurement_pct / 100.0
        * historical_procurement_mwh * procurement_lcoe / 1e9)
    frontier["storage_cost_billion_usd_per_year"] = (
        frontier.storage_hours_mean_load * future_mean_load_mw
        * battery_annuity / 1e9)
    frontier["total_cost_billion_usd_per_year"] = (
        frontier.procurement_cost_billion_usd_per_year
        + frontier.storage_cost_billion_usd_per_year)

    optimum = frontier.loc[frontier.total_cost_billion_usd_per_year.idxmin()]
    storage_only = frontier.iloc[0]
    procurement_only = frontier.iloc[-1]
    return {
        "window_hours": window,
        "procurement_only_pct": float(procurement_only.additional_procurement_pct),
        "procurement_only_cost_billion_usd_per_year": float(
            procurement_only.total_cost_billion_usd_per_year),
        "storage_only_hours_mean_load": float(storage_only.storage_hours_mean_load),
        "storage_only_cost_billion_usd_per_year": float(
            storage_only.total_cost_billion_usd_per_year),
        "least_cost_billion_usd_per_year": float(
            optimum.total_cost_billion_usd_per_year),
        "least_cost_procurement_pct": float(optimum.additional_procurement_pct),
        "least_cost_storage_hours_mean_load": float(
            optimum.storage_hours_mean_load),
        "historical_worst_mwh": float(base[hist_col]),
        "future_worst_mwh": float(base[metric]),
        "climate_increment_mwh": penalty,
        "climate_penalty_pct": 100.0 * penalty / float(base[hist_col]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="S3")
    parser.add_argument("--model", default="ACCESS-CM2")
    args = parser.parse_args()

    suffix = "" if args.scenario == "S3" else "_" + args.scenario
    surface_file = (PROCESSED / ("climate_adaptation_windows_" + args.model + suffix)
                    / "adaptation_surface_windows_by_year.csv")
    panel = pd.read_csv(surface_file)
    numeric = [c for c in panel.columns
               if c not in ("scenario", "model", "weather_year")]
    surface_mean = panel.groupby(
        ["additional_procurement_pct", "storage_hours_mean_load"],
        as_index=False)[numeric].mean()
    base = surface_mean[(surface_mean.additional_procurement_pct == 0)
                        & (surface_mean.storage_hours_mean_load == 0)].iloc[0]

    published = json.loads(
        (PUBLISHED / args.scenario / "adaptation_economics.json").read_text(
            encoding="utf-8"))
    assumptions = published["global_cost_assumptions"]
    procurement_lcoe = float(assumptions["renewable_procurement_lcoe_usd_per_mwh"])
    battery_annuity = float(assumptions["battery_4h_annuity_usd_per_mwh_year"])

    design = pd.read_csv(DESIGN)
    historical_procurement_mwh = float(design.mean_annual_demand_mwh.sum())
    future_demand_mwh = float(base.uncovered_mwh / (1.0 - base.coverage))
    future_mean_load_mw = future_demand_mwh / HOURS_PER_YEAR

    windows = sorted(int(c.split("_")[1][:-1]) for c in panel.columns
                     if c.startswith("worst_") and c.endswith("h_mwh"))
    results = [solve_window(surface_mean, base, w, historical_procurement_mwh,
                            future_mean_load_mw, procurement_lcoe,
                            battery_annuity) for w in windows]
    table = pd.DataFrame([r for r in results if r is not None])
    table["cheapest_tool"] = np.where(
        table.least_cost_storage_hours_mean_load > 1e-6,
        np.where(table.least_cost_procurement_pct > 1e-6, "mixed", "storage"),
        "procurement")

    out_dir = ROOT / "output" / "24x7_cfe_facility" / ("adaptation_windows" + suffix)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "window_economics_mean.csv", index=False)

    print("=== VALIDATION against published " + args.scenario + " ===")
    pub = {r["restoration_target"]: r
           for r in published["global_restoration_summary"]}
    gate = True
    for target, window in (("24h", 24), ("168h", 168)):
        row = table[table.window_hours.eq(window)]
        if target not in pub or row.empty:
            continue
        row = row.iloc[0]
        for key in ("procurement_only_pct", "storage_only_hours_mean_load",
                    "procurement_only_cost_billion_usd_per_year",
                    "storage_only_cost_billion_usd_per_year",
                    "least_cost_billion_usd_per_year"):
            new = float(row[key])
            old = float(pub[target][key])
            delta = 100.0 * (new - old) / old if old else float("nan")
            ok = abs(delta) < 1.0
            gate = gate and ok
            print("  " + ("OK  " if ok else "DIFF") + " " + target + " " + key
                  + ": new=" + format(new, ".4f") + " published=" + format(old, ".4f")
                  + " delta=" + format(delta, "+.2f") + "%")
    print("  GATE: " + ("PASS" if gate else "FAIL"))

    print()
    print("=== WINDOW SWEEP (" + args.scenario + ", 20-year mean surface) ===")
    print("  W(h)   加容量%   加容量(亿$)    储能(h)   储能(亿$)   最低成本(亿$)"
          "  最便宜工具    气候增量(GWh)  罚分%")
    for row in table.itertuples():
        print("  {:>4d}   {:>7.3f}   {:>10.2f}   {:>8.3f}   {:>9.2f}   {:>11.2f}"
              "   {:<11s}   {:>11.1f}   {:>5.2f}".format(
                  int(row.window_hours), row.procurement_only_pct,
                  row.procurement_only_cost_billion_usd_per_year * 10,
                  row.storage_only_hours_mean_load,
                  row.storage_only_cost_billion_usd_per_year * 10,
                  row.least_cost_billion_usd_per_year * 10,
                  row.cheapest_tool, row.climate_increment_mwh / 1000.0,
                  row.climate_penalty_pct))
    print()
    print("wrote " + str(out_dir / "window_economics_mean.csv"))


if __name__ == "__main__":
    main()
