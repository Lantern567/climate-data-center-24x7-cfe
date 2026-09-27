# -*- coding: utf-8 -*-
"""Dense-window sweep of the volume-storage restoration surface.

Reproduces the event-targeted tail bound of script 64 but for a dense set of
rolling windows instead of only 24 h and 168 h, so the crossover between the
procurement-only and storage-only endpoints can be located.

The chronological greedy annual dispatch is skipped: window targets do not use
it, so the 8760-step loop is unnecessary here.  Everything else -- the pooling
boundary, the volume/storage grids, the round-trip efficiency and the
event-targeted lower-bound convention -- is taken unchanged from script 64.
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
GEO = S64.GEO

SURFACE_WINDOWS = (3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 42, 48, 60,
                   72, 96, 120, 144, 168)


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


def surface_for_year(load_f, gen_f, load_h, gen_h, volume_grid, storage_grid,
                     efficiency):
    countries, hours = load_f.shape
    volume_values = np.asarray(volume_grid, dtype=np.float32)
    storage_values = np.asarray(storage_grid, dtype=np.float32)
    volume, storage = np.meshgrid(volume_values, storage_values, indexing="ij")
    volume = volume.ravel()
    storage = storage.ravel()
    leg = float(np.sqrt(efficiency))
    mean_load = load_f.sum(axis=1, dtype=np.float64) / hours

    tail_by_volume = {w: [] for w in SURFACE_WINDOWS}
    for value in volume_values:
        gap = np.maximum(load_f - gen_f * (1.0 + value / 100.0), 0.0)
        worst = S64.rolling_worst_by_site(gap.astype(np.float32),
                                          SURFACE_WINDOWS,
                                          chunk=max(1, countries))
        for w in SURFACE_WINDOWS:
            tail_by_volume[w].append(worst[w])
    volume_index = np.repeat(np.arange(len(volume_values)), len(storage_values))
    deliverable = (mean_load[:, None] * storage[None, :] * leg).astype(np.float64)

    gap_h = np.maximum(load_h - gen_h, 0.0).astype(np.float32)
    hist_worst = S64.rolling_worst_by_site(gap_h, SURFACE_WINDOWS,
                                           chunk=max(1, countries))

    future_uncovered = float(np.maximum(load_f - gen_f, 0.0).sum(dtype=np.float64))
    future_demand = float(load_f.sum(dtype=np.float64))
    hist_uncovered = float(gap_h.sum(dtype=np.float64))
    hist_demand = float(load_h.sum(dtype=np.float64))

    frame = {
        "additional_procurement_pct": volume.astype(float),
        "storage_hours_mean_load": storage.astype(float),
        "coverage": 1.0 - future_uncovered / future_demand,
        "uncovered_mwh": future_uncovered,
        "historical_coverage": 1.0 - hist_uncovered / hist_demand,
        "historical_uncovered_mwh": hist_uncovered,
    }
    for w in SURFACE_WINDOWS:
        stacked = np.stack(tail_by_volume[w])
        frame["worst_" + str(w) + "h_mwh"] = np.maximum(
            stacked[volume_index].T - deliverable, 0.0).sum(axis=0)
        frame["historical_worst_" + str(w) + "h_mwh"] = float(hist_worst[w].sum())
    return pd.DataFrame(frame)


def main():
    args = parse_args()
    years = tuple(sorted(set(args.year))) if args.year else S64.REPLAY_YEARS
    args.output_root.mkdir(parents=True, exist_ok=True)
    efficiency = S64.round_trip_efficiency()

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

    out_file = args.output_root / "adaptation_surface_windows_by_year.csv"
    rows = []
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

        country_load_h = S64.aggregate_country(load_h, country_positions)
        country_gen_h = S64.aggregate_country(gen_h, country_positions)
        country_load_f = S64.aggregate_country(load_f, country_positions)
        country_gen_f = S64.aggregate_country(gen_f, country_positions)
        del load_h, load_f, gen_h, gen_f

        surface = surface_for_year(country_load_f, country_gen_f,
                                   country_load_h, country_gen_h,
                                   S64.VOLUME_GRID_PCT, S64.STORAGE_GRID_H,
                                   efficiency)
        surface.insert(0, "weather_year", year)
        surface.insert(0, "model", args.model)
        surface.insert(0, "scenario", args.scenario)
        rows.append(surface)
        pd.concat(rows, ignore_index=True).to_csv(out_file, index=False)
        elapsed = (datetime.now(UTC) - started).total_seconds()
        print("  " + str(year) + " complete in " + str(round(elapsed)) + " s",
              flush=True)

    panel = pd.concat(rows, ignore_index=True)
    panel.to_csv(out_file, index=False)
    print("wrote " + str(out_file))
    print("rows=" + str(len(panel)) + " windows=" + str(SURFACE_WINDOWS))


if __name__ == "__main__":
    main()
