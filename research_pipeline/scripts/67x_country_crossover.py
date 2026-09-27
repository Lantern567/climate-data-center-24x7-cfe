# -*- coding: utf-8 -*-
"""Locate each country's storage-to-capacity crossover window.

Uses the per-country tails from 64x_country_windows.py.  For every country and
window:

  procurement endpoint  smallest volume uplift whose worst-window residual falls
                        back to the historical value, interpolated on the volume
                        grid used by the published surface
  storage endpoint      solved in closed form, E = (tail - historical)
                        / (mean_load * sqrt(eta)), because a full store delivers
                        E * mean_load * sqrt(eta) into the event

Both endpoints are then priced with the published S3 cost assumptions, and the
crossover is the window where the storage cost stops being the cheaper of the
two.  Countries whose two cost curves never cross inside the swept window range
are reported as such rather than extrapolated.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "24x7_cfe" / "processed"
PUBLISHED = ROOT / "output" / "24x7_cfe_facility" / "adaptation_economics_geo"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Reuse script 64's reader so the efficiency here cannot drift from the surface.
S64 = _load("s64", Path(__file__).resolve().parent
            / "64_compute_adaptation_frontier_and_compound_events.py")
round_trip_efficiency = S64.round_trip_efficiency


def solve_country(block, procurement_lcoe, battery_annuity, leg):
    """Endpoints for one country and one window."""
    block = block.sort_values("additional_procurement_pct")
    historical = float(block.historical_tail_mwh.iloc[0])
    mean_load = float(block.mean_load_mw.iloc[0])
    demand = float(block.historical_demand_mwh.iloc[0])
    volume = block.additional_procurement_pct.to_numpy(dtype=float)
    tail = block.tail_mwh.to_numpy(dtype=float)
    if historical <= 0 or mean_load <= 0 or tail[0] <= historical:
        return None

    storage_hours = (tail[0] - historical) / (mean_load * leg)
    reached = tail <= historical
    if not reached.any():
        procurement_pct = float("nan")
    else:
        index = int(np.flatnonzero(reached)[0])
        if index == 0:
            procurement_pct = 0.0
        else:
            x0, x1 = volume[index - 1], volume[index]
            y0, y1 = tail[index - 1], tail[index]
            procurement_pct = (x1 if np.isclose(y1, y0)
                               else x0 + (historical - y0) * (x1 - x0) / (y1 - y0))
    return {
        "storage_hours": storage_hours,
        "storage_cost_usd_per_year": storage_hours * mean_load * battery_annuity,
        "procurement_pct": procurement_pct,
        "procurement_cost_usd_per_year": (procurement_pct / 100.0 * demand
                                          * procurement_lcoe),
        "mean_load_mw": mean_load,
        "historical_demand_mwh": demand,
        "climate_increment_mwh": tail[0] - historical,
        "historical_tail_mwh": historical,
    }


def crossover_window(frame):
    """First window where storage stops being cheaper, by linear interpolation."""
    frame = frame.sort_values("window_hours")
    w = frame.window_hours.to_numpy(dtype=float)
    diff = (frame.storage_cost_usd_per_year
            - frame.procurement_cost_usd_per_year).to_numpy()
    good = np.isfinite(diff)
    w, diff = w[good], diff[good]
    if len(w) < 2:
        return np.nan, "insufficient"
    if diff[0] >= 0:
        return np.nan, "capacity always cheaper"
    if diff[-1] < 0:
        return np.nan, "storage always cheaper"
    for i in range(len(w) - 1):
        if diff[i] < 0 <= diff[i + 1]:
            return float(w[i] - diff[i] * (w[i + 1] - w[i])
                         / (diff[i + 1] - diff[i])), "crosses"
    return np.nan, "no crossing"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="S3")
    parser.add_argument("--model", default="ACCESS-CM2")
    args = parser.parse_args()

    suffix = "" if args.scenario == "S3" else "_" + args.scenario
    source = (PROCESSED / ("climate_adaptation_windows_" + args.model + suffix)
              / "country_window_tails_mean.csv")
    tails = pd.read_csv(source)

    published = json.loads(
        (PUBLISHED / args.scenario / "adaptation_economics.json").read_text(
            encoding="utf-8"))
    assumptions = published["global_cost_assumptions"]
    procurement_lcoe = float(assumptions["renewable_procurement_lcoe_usd_per_mwh"])
    battery_annuity = float(assumptions["battery_4h_annuity_usd_per_mwh_year"])
    leg = float(np.sqrt(round_trip_efficiency()))

    rows = []
    for (country, window), block in tails.groupby(
            ["supply_country_area_code", "window_hours"], observed=True):
        result = solve_country(block, procurement_lcoe, battery_annuity, leg)
        if result is None:
            continue
        result["supply_country_area_code"] = country
        result["window_hours"] = int(window)
        rows.append(result)
    endpoints = pd.DataFrame(rows)

    summary = []
    for country, block in endpoints.groupby("supply_country_area_code",
                                            observed=True):
        window, status = crossover_window(block)
        summary.append({
            "supply_country_area_code": country,
            "crossover_hours": window,
            "status": status,
            "demand_mwh": float(block.historical_demand_mwh.iloc[0]),
            "windows_solved": int(len(block)),
        })
    table = pd.DataFrame(summary).sort_values("demand_mwh", ascending=False)
    table["demand_share"] = table.demand_mwh / table.demand_mwh.sum()

    out_dir = ROOT / "output" / "24x7_cfe_facility" / ("adaptation_windows" + suffix)
    out_dir.mkdir(parents=True, exist_ok=True)
    endpoints.to_csv(out_dir / "country_window_endpoints.csv", index=False)
    table.to_csv(out_dir / "country_crossover.csv", index=False)

    crossing = table[table.status.eq("crosses")]
    weight = crossing.demand_mwh
    print("countries solved: " + str(len(table)))
    print(table.status.value_counts().to_string())
    if len(crossing):
        print()
        print("crossover window (h) among countries that cross:")
        print("  demand-weighted mean {:.1f}".format(
            float((crossing.crossover_hours * weight).sum() / weight.sum())))
        for q in (5, 25, 50, 75, 95):
            print("  p{:<3d} {:.1f}".format(
                q, float(np.percentile(crossing.crossover_hours, q))))
        print("  min {:.1f}  max {:.1f}".format(
            crossing.crossover_hours.min(), crossing.crossover_hours.max()))
        print()
        print("ten largest markets:")
        for row in table.head(10).itertuples():
            value = ("{:.1f} h".format(row.crossover_hours)
                     if np.isfinite(row.crossover_hours) else row.status)
            print("  {:<4s} demand share {:5.2f}%   {}".format(
                row.supply_country_area_code, 100 * row.demand_share, value))
    print()
    print("wrote " + str(out_dir / "country_crossover.csv"))


if __name__ == "__main__":
    main()
