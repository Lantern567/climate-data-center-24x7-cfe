# -*- coding: utf-8 -*-
"""Per-country window sweep: where does the storage-to-capacity flip sit?

64w_window_sweep.py answers this for the demand-weighted global aggregate.
This script keeps the procurement countries separate, which is the pooling
boundary the published method already uses ("within procurement country only"),
so a country-level crossover is the natural granularity rather than a new
assumption.

For each country, window and procurement level it records the event-targeted
worst-window residual with no storage.  The storage endpoint is then solved in
closed form downstream: a full store delivers E * mean_load * sqrt(eta) into
the event, so restoring the historical worst window needs
E = (tail - historical) / (mean_load * sqrt(eta)).  No grid interpolation is
required on the storage axis, so the storage grid is not swept here.
"""
from __future__ import annotations

import argparse
import importlib.util
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HERE = Path(__file__).resolve().parent
S64 = _load("s64", HERE / "64_compute_adaptation_frontier_and_compound_events.py")
S64W = _load("s64w", HERE / "64w_window_sweep.py")
GEO = S64.GEO

WINDOWS = S64W.SURFACE_WINDOWS


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="ACCESS-CM2")
    parser.add_argument("--demand-future", type=Path,
                        default=Path(r"D:\facility_demand_future_1995_2014"))
    parser.add_argument("--design-file", type=Path, default=S64.DESIGN)
    parser.add_argument("--design-column", default="wind_share_robust_mean_20y")
    parser.add_argument("--year", type=int, action="append")
    parser.add_argument("--scenario", default="S3",
                        choices=["S1", "S2", "S3", "S4", "S5"])
    parser.add_argument("--mix-from-ladder", action="store_true")
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    suffix = "" if args.scenario == "S3" else "_" + args.scenario
    if args.output_root is None:
        args.output_root = (S64.PROCESSED
                            / ("climate_adaptation_windows_" + args.model + suffix))
    return args


def main():
    args = parse_args()
    years = tuple(sorted(set(args.year))) if args.year else S64.REPLAY_YEARS
    args.output_root.mkdir(parents=True, exist_ok=True)

    if args.scenario == "S3" and not args.mix_from_ladder:
        design = pd.read_csv(args.design_file)
        mixes = design.set_index("logical_site_id")[args.design_column]
    else:
        ladder = pd.read_parquet(
            GEO.LADDER,
            columns=["scenario", "weather_year", "logical_site_id",
                     "wind_energy_share"])
        ladder = ladder[ladder.scenario.eq(args.scenario)
                        & ladder.weather_year.eq(GEO.REFERENCE_YEAR)]
        mixes = ladder.drop_duplicates("logical_site_id").set_index(
            "logical_site_id")["wind_energy_share"]

    sites = pd.read_csv(S64.CROSSWALK / "facility_to_era5_grid_with_country.csv",
                        low_memory=False)
    sites = (sites[sites.logical_site_id.isin(mixes.index)]
             .sort_values("logical_site_id").reset_index(drop=True))
    sites["region"] = sites.continent.where(
        ~sites.continent.isin(["Unknown", "Antarctica"]))
    country_to_region = (sites.dropna(subset=["region"])
                         .groupby("supply_country_area_code").region
                         .agg(lambda values: values.mode().iloc[0]))
    large_pool = None
    if args.scenario == "S1":
        large_pool = GEO.load_npz_pair(GEO.ONSITE_H, GEO.ONSITE_F,
                                       "onsite_supply_cf.npz",
                                       GEO.DEFAULT_CACHE, "S1_onsite")
    elif args.scenario == "S2":
        large_pool = GEO.load_npz_pair(GEO.LOCAL_H, GEO.LOCAL_F,
                                       "local_supply_pool_cf.npz",
                                       GEO.DEFAULT_CACHE, "S2_local_100km")

    site_ids = pd.Index(sites.logical_site_id)
    site_mix = mixes.reindex(site_ids).to_numpy(dtype=float)
    country_groups = sites.groupby("supply_country_area_code",
                                   observed=True).indices
    country_codes = pd.Index(sorted(country_groups))
    country_positions = [country_groups[code] for code in country_codes]

    volume_values = np.asarray(S64.VOLUME_GRID_PCT, dtype=np.float32)
    frames = []
    for year in years:
        started = datetime.now(UTC)
        hist = S64.load_demand_columns(S64.DEMAND_HIST, year, site_ids,
                                       (S64.LOAD_COLUMN,))
        future = S64.load_demand_columns(args.demand_future, year, site_ids,
                                         (S64.LOAD_COLUMN,))
        load_h = hist.pop(S64.LOAD_COLUMN)
        load_f = future.pop(S64.LOAD_COLUMN)
        groups, pools = GEO.pool_spec(args.scenario, year, sites,
                                      country_to_region, large_pool)
        gen_h, gen_f = S64.build_generation(site_mix, load_h, groups, pools)

        cl_h = S64.aggregate_country(load_h, country_positions)
        cg_h = S64.aggregate_country(gen_h, country_positions)
        cl_f = S64.aggregate_country(load_f, country_positions)
        cg_f = S64.aggregate_country(gen_f, country_positions)
        del load_h, load_f, gen_h, gen_f

        countries, hours = cl_f.shape
        gap_h = np.maximum(cl_h - cg_h, 0.0).astype(np.float32)
        hist_worst = S64.rolling_worst_by_site(gap_h, WINDOWS, chunk=countries)

        records = []
        for value in volume_values:
            gap = np.maximum(cl_f - cg_f * (1.0 + value / 100.0), 0.0)
            worst = S64.rolling_worst_by_site(gap.astype(np.float32), WINDOWS,
                                              chunk=countries)
            for window in WINDOWS:
                records.append(pd.DataFrame({
                    "supply_country_area_code": country_codes,
                    "additional_procurement_pct": float(value),
                    "window_hours": window,
                    "tail_mwh": worst[window],
                    "historical_tail_mwh": hist_worst[window],
                }))
        frame = pd.concat(records, ignore_index=True)
        frame["weather_year"] = year
        frame["mean_load_mw"] = frame.supply_country_area_code.map(
            pd.Series(cl_f.sum(axis=1, dtype=np.float64) / hours,
                      index=country_codes))
        frame["historical_demand_mwh"] = frame.supply_country_area_code.map(
            pd.Series(cl_h.sum(axis=1, dtype=np.float64), index=country_codes))
        frames.append(frame)
        elapsed = (datetime.now(UTC) - started).total_seconds()
        print("  " + str(year) + " complete in " + str(round(elapsed)) + " s",
              flush=True)

    panel = pd.concat(frames, ignore_index=True)
    mean = panel.groupby(
        ["supply_country_area_code", "window_hours",
         "additional_procurement_pct"], as_index=False)[
        ["tail_mwh", "historical_tail_mwh", "mean_load_mw",
         "historical_demand_mwh"]].mean()
    out = args.output_root / "country_window_tails_mean.csv"
    mean.to_csv(out, index=False)
    print("wrote " + str(out))
    print("countries=" + str(mean.supply_country_area_code.nunique())
          + " windows=" + str(len(WINDOWS)) + " rows=" + str(len(mean)))


if __name__ == "__main__":
    main()
