# -*- coding: utf-8 -*-
"""Complete the climate-resilience calculations required by Results 1, 3 and 4.

The script adds three pieces of evidence that are deliberately kept separate.

1. Facility-level deficit duration-severity curves retain the manuscript's
   buyer-level matching boundary and extend the existing 24 h and 168 h metrics
   to 1--720 h windows.
2. A volume-storage response surface pools portfolios and storage only within
   each procurement country.  Annual coverage uses chronological greedy
   dispatch.  Daily and weekly targets use an event-targeted lower bound in
   which the store is full at the start of the relevant worst window; the
   chronological tail outcome is retained as a sensitivity.  The surface
   therefore represents a regional planning portfolio, not unconstrained
   global balancing or a claim about optimal market dispatch.
3. Compound high-cooling/low-renewable exposure is measured with paired,
   site-specific historical percentile thresholds.  The future construction
   preserves the historical hour pairing, so the calculation tests the effect
   of changing marginal climate distributions and must not be interpreted as a
   change in the heat-renewable dependence structure.

All headline calculations use the robust twenty-weather-year wind/solar mix and
the ACCESS-CM2 SSP2-4.5 climate perturbation unless command-line arguments say
otherwise.
"""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import importlib.util

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def _load_geography_module():
    """Reuse script 66's procurement-pool builder (see script 63 for rationale)."""
    path = Path(__file__).with_name("66_compute_geographic_climate_conditioning.py")
    spec = importlib.util.spec_from_file_location("geo66", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GEO = _load_geography_module()


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "24x7_cfe" / "processed"
CROSSWALK = PROCESSED / "climate_coupling" / "demand_supply_country_crosswalk"
SUPPLY_ROOT = PROCESSED / "wind_solar_supply_20y_fixed_2025"
SUPPLY_HIST = SUPPLY_ROOT / "replay_results_pv_corrected"
DESIGN = PROCESSED / "weather_robust_design" / "site_portfolio_designs.csv"
COSTS = PROCESSED / "costs" / "atb_2024_v3_costs_compact.csv"
DEMAND_HIST = Path(r"D:\facility_demand_replay_1995_2014")

METHOD_VERSION = "adaptation_frontier_compound_v1"
LOAD_COLUMN = "global_reconciled_allocation_facility_load_mw"
COOLING_COLUMN = "global_reconciled_allocation_climate_variable_overhead_load_mw"
REPLAY_YEARS = tuple(range(1995, 2015))
DURATION_WINDOWS = (1, 6, 24, 72, 168, 336, 720)
VOLUME_GRID_PCT = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0)
STORAGE_GRID_H = (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5,
                  2.0, 3.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 400.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="ACCESS-CM2")
    parser.add_argument("--demand-future", type=Path,
                        default=Path(r"D:\facility_demand_future_1995_2014"))
    parser.add_argument("--supply-future", type=Path,
                        default=SUPPLY_ROOT / "replay_results_future_ssp245")
    parser.add_argument("--design-file", type=Path, default=DESIGN)
    parser.add_argument("--design-column", default="wind_share_robust_mean_20y")
    parser.add_argument("--year", type=int, action="append")
    parser.add_argument("--scenario", default="S3",
                        choices=["S1", "S2", "S3", "S4", "S5"],
                        help="Procurement geography. S3 reproduces the published run.")
    parser.add_argument("--mix-from-ladder", action="store_true",
                        help="Use the scenario ladder's frozen wind share even for "
                             "S3, so all five geographies share one mix convention.")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--validation-root", type=Path, default=None)
    args = parser.parse_args()
    suffix = "" if args.scenario == "S3" else f"_{args.scenario}"
    if args.output_root is None:
        args.output_root = PROCESSED / f"climate_adaptation_{args.model}{suffix}"
    if args.validation_root is None:
        args.validation_root = (ROOT / "output" / "24x7_cfe_facility"
                                / f"climate_adaptation_{args.model}{suffix}")
    return args


def calendar(year: int) -> pd.DatetimeIndex:
    return pd.date_range(f"{year}-01-01", f"{year + 1}-01-01", freq="h",
                         inclusive="left", tz="UTC")


def round_trip_efficiency() -> float:
    costs = pd.read_csv(COSTS)
    row = costs[(costs.technology_key == "battery_4h")
                & (costs.scenario == "Moderate")
                & (costs.year == 2030)].iloc[0]
    return float(row.round_trip_efficiency_fraction)


def country_cf(root: Path, year: int) -> tuple[pd.Index, np.ndarray, np.ndarray]:
    partition = root / "hourly_country_supply" / f"weather_year={year}"
    files = list(partition.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no country supply parquet under {partition}")
    frame = pd.read_parquet(files[0])
    annual = pd.read_parquet(root / "annual_country_supply_1995_2014.parquet")
    annual = annual[annual.weather_year.eq(year)].set_index("country_area_code")
    hours = len(calendar(year))
    frame = frame.sort_values(["country_area_code", "time_utc"])
    codes = pd.Index(frame.country_area_code.unique())
    pv = frame.solar_pv_generation_power_mw.to_numpy().reshape(len(codes), hours)
    wind = frame.wind_generation_power_mw.to_numpy().reshape(len(codes), hours)
    pv_cap = annual.loc[codes, "solar_pv_installed_capacity_mw"].to_numpy()
    wind_cap = annual.loc[codes, "wind_installed_capacity_mw"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        pv_cf = np.where(pv_cap[:, None] > 0,
                         pv / np.maximum(pv_cap[:, None], 1e-9), 0.0)
        wind_cf = np.where(wind_cap[:, None] > 0,
                           wind / np.maximum(wind_cap[:, None], 1e-9), 0.0)
    return codes, pv_cf.astype(np.float32), wind_cf.astype(np.float32)


def portfolio_shape(pv_cf: np.ndarray, wind_cf: np.ndarray, mix: float) -> np.ndarray:
    pv_energy = float(pv_cf.sum(dtype=np.float64))
    wind_energy = float(wind_cf.sum(dtype=np.float64))
    if pv_energy <= 0 and wind_energy <= 0:
        return np.zeros_like(pv_cf, dtype=np.float32)
    if wind_energy <= 0:
        mix = 0.0
    elif pv_energy <= 0:
        mix = 1.0
    shape = np.zeros_like(pv_cf, dtype=np.float64)
    if mix < 1.0:
        shape += (1.0 - mix) * pv_cf / pv_energy
    if mix > 0.0:
        shape += mix * wind_cf / wind_energy
    return shape.astype(np.float32)


def site_block_order(handle: pq.ParquetFile, hours: int) -> pd.Index:
    order: list[str] = []
    counts: list[int] = []
    for group in range(handle.metadata.num_row_groups):
        ids = (handle.read_row_group(group, columns=["logical_site_id"])
               .column("logical_site_id").to_numpy(zero_copy_only=False))
        edges = np.flatnonzero(ids[1:] != ids[:-1]) + 1
        starts = np.concatenate([[0], edges])
        lengths = np.diff(np.concatenate([starts, [len(ids)]]))
        for site, length in zip(ids[starts], lengths):
            if order and order[-1] == site:
                counts[-1] += int(length)
            else:
                order.append(str(site))
                counts.append(int(length))
    if set(counts) != {hours}:
        raise RuntimeError(f"demand file does not contain {hours} hours per site")
    return pd.Index(order)


def load_demand_columns(root: Path, year: int, site_ids: pd.Index,
                        columns: tuple[str, ...]) -> dict[str, np.ndarray]:
    hours = len(calendar(year))
    path = root / f"facility_hourly_demand_replay_{year}.parquet"
    handle = pq.ParquetFile(path)
    present = site_block_order(handle, hours)
    rows = pd.Series(range(len(present)), index=present).reindex(site_ids).to_numpy()
    if pd.isna(rows).any():
        raise RuntimeError(f"{int(pd.isna(rows).sum())} sites absent from {path}")
    take = rows.astype(int)
    table = handle.read(columns=list(columns))
    output: dict[str, np.ndarray] = {}
    for column in columns:
        values = (table.column(column).to_numpy(zero_copy_only=False)
                  .astype(np.float32, copy=False).reshape(-1, hours))
        output[column] = np.ascontiguousarray(values[take])
    return output


def build_generation(site_mix: np.ndarray, load_h: np.ndarray,
                     groups: dict, pools: dict) -> tuple[np.ndarray, np.ndarray]:
    """Build historical and frozen-volume future generation per facility.

    `groups`/`pools` come from script 66's pool_spec, so one routine serves all
    five procurement geographies rather than the host country alone.
    """
    n, hours = load_h.shape
    gen_h = np.zeros((n, hours), dtype=np.float32)
    gen_f = np.zeros((n, hours), dtype=np.float32)
    procured = load_h.sum(axis=1, dtype=np.float64)
    for code, positions in groups.items():
        positions = np.asarray(positions)
        pv_h, wind_h, pv_f, wind_f = pools[code]
        mixes = site_mix[positions]
        for mix in np.unique(mixes[~np.isnan(mixes)]):
            which = positions[mixes == mix]
            historical_shape = portfolio_shape(pv_h, wind_h, float(mix))
            future_shape = portfolio_shape(pv_f, wind_f, float(mix))
            gen_h[which] = procured[which, None] * historical_shape[None, :]
            gen_f[which] = procured[which, None] * future_shape[None, :]
    return gen_h, gen_f


def rolling_worst_by_site(values: np.ndarray, windows: tuple[int, ...],
                          chunk: int = 512) -> dict[int, np.ndarray]:
    """Return each site's largest rolling sum for every requested window."""
    n, hours = values.shape
    result = {window: np.empty(n, dtype=np.float64) for window in windows}
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        block = values[start:stop]
        cumulative = np.empty((stop - start, hours + 1), dtype=np.float64)
        cumulative[:, 0] = 0.0
        np.cumsum(block, axis=1, dtype=np.float64, out=cumulative[:, 1:])
        for window in windows:
            result[window][start:stop] = (
                cumulative[:, window:] - cumulative[:, :-window]).max(axis=1)
    return result


def aggregate_country(values: np.ndarray, country_positions: list[np.ndarray]) -> np.ndarray:
    return np.stack([values[position].sum(axis=0, dtype=np.float64)
                     for position in country_positions]).astype(np.float32)


def no_storage_metrics(load: np.ndarray, generation: np.ndarray) -> dict[str, float]:
    gap = np.maximum(load - generation, 0.0).astype(np.float32)
    worst = rolling_worst_by_site(gap, (24, 168), chunk=max(1, len(load)))
    demand = float(load.sum(dtype=np.float64))
    uncovered = float(gap.sum(dtype=np.float64))
    return {
        "coverage": 1.0 - uncovered / demand,
        "uncovered_mwh": uncovered,
        "worst_24h_mwh": float(worst[24].sum()),
        "worst_168h_mwh": float(worst[168].sum()),
    }


def storage_surface(load: np.ndarray, generation: np.ndarray,
                    volume_grid_pct: tuple[float, ...],
                    storage_grid_h: tuple[float, ...],
                    efficiency: float) -> pd.DataFrame:
    """Evaluate country-pooled storage for a volume-storage Cartesian grid."""
    volume_values = np.asarray(volume_grid_pct, dtype=np.float32)
    storage_values = np.asarray(storage_grid_h, dtype=np.float32)
    volume, storage = np.meshgrid(volume_values, storage_values,
                                  indexing="ij")
    volume = volume.ravel()
    storage = storage.ravel()
    scenarios = len(volume)
    countries, hours = load.shape
    mean_load = load.sum(axis=1, dtype=np.float64) / hours
    capacity = (mean_load[:, None] * storage[None, :]).astype(np.float32)
    scale = (1.0 + volume / 100.0)[None, :]
    leg = np.float32(np.sqrt(efficiency))
    soc = np.zeros((countries, scenarios), dtype=np.float32)
    total_gap = np.zeros(scenarios, dtype=np.float64)
    roll24 = np.zeros((countries, scenarios), dtype=np.float32)
    roll168 = np.zeros((countries, scenarios), dtype=np.float32)
    worst24 = np.zeros((countries, scenarios), dtype=np.float32)
    worst168 = np.zeros((countries, scenarios), dtype=np.float32)
    ring24 = np.zeros((24, countries, scenarios), dtype=np.float32)
    ring168 = np.zeros((168, countries, scenarios), dtype=np.float32)

    for hour in range(hours):
        demand = load[:, hour, None]
        supplied = generation[:, hour, None] * scale
        surplus = np.maximum(supplied - demand, 0.0)
        deficit = np.maximum(demand - supplied, 0.0)
        soc += np.minimum(surplus * leg, capacity - soc)
        released = np.minimum(soc * leg, deficit)
        soc -= released / leg
        residual = deficit - released
        total_gap += residual.sum(axis=0, dtype=np.float64)

        slot24 = hour % 24
        roll24 += residual - ring24[slot24]
        ring24[slot24] = residual
        if hour >= 23:
            np.maximum(worst24, roll24, out=worst24)

        slot168 = hour % 168
        roll168 += residual - ring168[slot168]
        ring168[slot168] = residual
        if hour >= 167:
            np.maximum(worst168, roll168, out=worst168)

    # Tail-targeted lower bound.  For each procurement country and volume level,
    # calculate the no-storage worst window, then allow a fully charged store to
    # deliver up to E * mean load * sqrt(eta) into that event.  This is the
    # minimum energy-capacity interpretation of restoring a historical tail; it
    # deliberately does not assume that greedy annual dispatch saves energy for
    # the event.  The chronological greedy tail above is retained separately.
    tail24_by_volume = []
    tail168_by_volume = []
    for value in volume_values:
        gap = np.maximum(load - generation * (1.0 + value / 100.0), 0.0)
        worst = rolling_worst_by_site(gap.astype(np.float32), (24, 168),
                                      chunk=max(1, countries))
        tail24_by_volume.append(worst[24])
        tail168_by_volume.append(worst[168])
    tail24_by_volume = np.stack(tail24_by_volume)
    tail168_by_volume = np.stack(tail168_by_volume)
    volume_index = np.repeat(np.arange(len(volume_values)), len(storage_values))
    deliverable = (mean_load[:, None] * storage[None, :] * leg).astype(np.float64)
    targeted24 = np.maximum(
        tail24_by_volume[volume_index].T - deliverable, 0.0).sum(axis=0)
    targeted168 = np.maximum(
        tail168_by_volume[volume_index].T - deliverable, 0.0).sum(axis=0)

    demand_total = float(load.sum(dtype=np.float64))
    return pd.DataFrame({
        "additional_procurement_pct": volume.astype(float),
        "storage_hours_mean_load": storage.astype(float),
        "coverage": 1.0 - total_gap / demand_total,
        "uncovered_mwh": total_gap,
        "worst_24h_mwh": targeted24,
        "worst_168h_mwh": targeted168,
        "chronological_worst_24h_mwh": worst24.sum(axis=0, dtype=np.float64),
        "chronological_worst_168h_mwh": worst168.sum(axis=0, dtype=np.float64),
    })


def compound_site_metrics(cooling_h: np.ndarray, cooling_f: np.ndarray,
                          gen_h: np.ndarray, gen_f: np.ndarray,
                          gap_f: np.ndarray) -> pd.DataFrame:
    """Measure paired high-cooling and low-renewable hours per site-year."""
    cooling_threshold = np.quantile(cooling_h, 0.9, axis=1).astype(np.float32)
    supply_threshold = np.quantile(gen_h, 0.1, axis=1).astype(np.float32)
    high_h = cooling_h >= cooling_threshold[:, None]
    high_f = cooling_f >= cooling_threshold[:, None]
    low_h = gen_h <= supply_threshold[:, None]
    low_f = gen_f <= supply_threshold[:, None]
    joint_h = high_h & low_h
    joint_f = high_f & low_f
    max_h = rolling_worst_by_site(joint_h, (168,), chunk=512)[168]
    max_f = rolling_worst_by_site(joint_f, (168,), chunk=512)[168]
    p_high_h = high_h.mean(axis=1)
    p_low_h = low_h.mean(axis=1)
    p_high_f = high_f.mean(axis=1)
    p_low_f = low_f.mean(axis=1)
    p_joint_h = joint_h.mean(axis=1)
    p_joint_f = joint_f.mean(axis=1)
    gap_total = gap_f.sum(axis=1, dtype=np.float64)
    gap_compound = (gap_f * joint_f).sum(axis=1, dtype=np.float64)
    gap_share = np.divide(gap_compound, gap_total,
                          out=np.zeros_like(gap_compound), where=gap_total > 0)
    concentration = np.divide(gap_share, p_joint_f,
                              out=np.full_like(gap_share, np.nan), where=p_joint_f > 0)
    dependence_h = np.divide(p_joint_h, p_high_h * p_low_h,
                             out=np.full_like(p_joint_h, np.nan),
                             where=(p_high_h * p_low_h) > 0)
    dependence_f = np.divide(p_joint_f, p_high_f * p_low_f,
                             out=np.full_like(p_joint_f, np.nan),
                             where=(p_high_f * p_low_f) > 0)
    return pd.DataFrame({
        "high_cooling_probability_historical": p_high_h,
        "high_cooling_probability_future": p_high_f,
        "low_renewable_probability_historical": p_low_h,
        "low_renewable_probability_future": p_low_f,
        "compound_probability_historical": p_joint_h,
        "compound_probability_future": p_joint_f,
        "compound_probability_change_pp": (p_joint_f - p_joint_h) * 100,
        "compound_dependence_ratio_historical": dependence_h,
        "compound_dependence_ratio_future": dependence_f,
        "max_compound_hours_168h_historical": max_h,
        "max_compound_hours_168h_future": max_f,
        "max_compound_hours_168h_change": max_f - max_h,
        "future_uncovered_energy_compound_share": gap_share,
        "future_compound_deficit_concentration": concentration,
    })


def weighted_country_summary(panel: pd.DataFrame) -> pd.DataFrame:
    value_columns = [column for column in panel.columns if column not in {
        "model", "weather_year", "logical_site_id", "supply_country_area_code",
        "continent", "demand_future_mwh"}]
    rows = []
    for code, block in panel.groupby("supply_country_area_code", observed=True):
        weights = block.demand_future_mwh.to_numpy(dtype=float)
        row = {
            "supply_country_area_code": code,
            "demand_future_mwh": float(weights.sum()),
        }
        for column in value_columns:
            values = block[column].to_numpy(dtype=float)
            ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
            row[column] = (float(np.average(values[ok], weights=weights[ok]))
                           if ok.any() else np.nan)
        rows.append(row)
    result = pd.DataFrame(rows)
    result["demand_share"] = (result.demand_future_mwh
                              / result.demand_future_mwh.sum())
    return result.sort_values("demand_share", ascending=False)


def interpolation_target(curve: pd.DataFrame, metric: str,
                         historical: float) -> tuple[float, bool]:
    curve = curve.sort_values("storage_hours_mean_load")
    x = curve.storage_hours_mean_load.to_numpy(dtype=float)
    y = curve[metric].to_numpy(dtype=float)
    if metric == "coverage":
        reached = y >= historical
    else:
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
    fraction = (historical - y0) / (y1 - y0)
    return float(x0 + fraction * (x1 - x0)), True


def pareto_points(surface: pd.DataFrame, restoration_column: str,
                  target: float) -> pd.DataFrame:
    feasible = surface[surface[restoration_column] >= target - 1e-9].copy()
    if feasible.empty:
        return feasible
    values = feasible[["additional_procurement_pct",
                       "storage_hours_mean_load"]].to_numpy()
    keep = np.ones(len(values), dtype=bool)
    for index, point in enumerate(values):
        dominates = ((values[:, 0] <= point[0]) & (values[:, 1] <= point[1])
                     & ((values[:, 0] < point[0]) | (values[:, 1] < point[1])))
        if dominates.any():
            keep[index] = False
    return feasible.loc[keep].sort_values(
        ["additional_procurement_pct", "storage_hours_mean_load"])


def main() -> None:
    args = parse_args()
    years = tuple(sorted(set(args.year))) if args.year else REPLAY_YEARS
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.validation_root.mkdir(parents=True, exist_ok=True)
    efficiency = round_trip_efficiency()

    if args.scenario == "S3" and not args.mix_from_ladder:
        design = pd.read_csv(args.design_file)
        mixes = design.set_index("logical_site_id")[args.design_column]
    else:
        ladder = pd.read_parquet(
            GEO.LADDER,
            columns=["scenario", "weather_year", "logical_site_id", "wind_energy_share"])
        ladder = ladder[ladder.scenario.eq(args.scenario)
                        & ladder.weather_year.eq(GEO.REFERENCE_YEAR)]
        mixes = ladder.drop_duplicates("logical_site_id").set_index(
            "logical_site_id")["wind_energy_share"]

    sites = pd.read_csv(CROSSWALK / "facility_to_era5_grid_with_country.csv",
                        low_memory=False)
    sites = (sites[sites.logical_site_id.isin(mixes.index)]
             .sort_values("logical_site_id").reset_index(drop=True))
    sites["region"] = sites.continent.where(~sites.continent.isin(["Unknown", "Antarctica"]))
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
    country_groups = sites.groupby("supply_country_area_code", observed=True).indices
    country_codes = pd.Index(sorted(country_groups))
    country_positions = [country_groups[code] for code in country_codes]

    duration_rows: list[dict[str, float | int | str]] = []
    surface_rows: list[pd.DataFrame] = []
    compound_rows: list[pd.DataFrame] = []
    baseline_checks: list[dict[str, float | int]] = []

    for year in years:
        started = datetime.now(UTC)
        hist = load_demand_columns(DEMAND_HIST, year, site_ids,
                                   (LOAD_COLUMN, COOLING_COLUMN))
        future = load_demand_columns(args.demand_future, year, site_ids,
                                     (LOAD_COLUMN, COOLING_COLUMN))
        load_h = hist.pop(LOAD_COLUMN)
        cooling_h = hist.pop(COOLING_COLUMN)
        load_f = future.pop(LOAD_COLUMN)
        cooling_f = future.pop(COOLING_COLUMN)
        groups, pools = GEO.pool_spec(args.scenario, year, sites,
                                      country_to_region, large_pool)
        gen_h, gen_f = build_generation(site_mix, load_h, groups, pools)
        demand_h = load_h.sum(axis=1, dtype=np.float64)
        demand_f = load_f.sum(axis=1, dtype=np.float64)
        resize_factor = np.divide(demand_f, demand_h,
                                  out=np.ones_like(demand_f), where=demand_h > 0)
        gen_resized = (gen_f * resize_factor[:, None]).astype(np.float32)

        scenario_arrays = {
            "historical": (load_h, gen_h),
            "future_frozen_volume": (load_f, gen_f),
            "future_resized_volume": (load_f, gen_resized),
        }
        for scenario, (load, generation) in scenario_arrays.items():
            gap = np.maximum(load - generation, 0.0).astype(np.float32)
            worst = rolling_worst_by_site(gap, DURATION_WINDOWS)
            for window in DURATION_WINDOWS:
                duration_rows.append({
                    "model": args.model,
                    "weather_year": year,
                    "scenario": scenario,
                    "duration_hours": window,
                    "worst_window_uncovered_mwh": float(worst[window].sum()),
                })
            if scenario == "future_frozen_volume":
                gap_f = gap
            else:
                del gap

        compound = compound_site_metrics(cooling_h, cooling_f,
                                         gen_h, gen_f, gap_f)
        compound.insert(0, "continent", sites.continent.to_numpy())
        compound.insert(0, "supply_country_area_code",
                        sites.supply_country_area_code.to_numpy())
        compound.insert(0, "logical_site_id", site_ids.to_numpy())
        compound.insert(0, "weather_year", year)
        compound.insert(0, "model", args.model)
        compound["demand_future_mwh"] = demand_f
        compound_rows.append(compound)

        country_load_h = aggregate_country(load_h, country_positions)
        country_gen_h = aggregate_country(gen_h, country_positions)
        country_load_f = aggregate_country(load_f, country_positions)
        country_gen_f = aggregate_country(gen_f, country_positions)
        historical_metrics = no_storage_metrics(country_load_h, country_gen_h)
        surface = storage_surface(country_load_f, country_gen_f,
                                  VOLUME_GRID_PCT, STORAGE_GRID_H, efficiency)
        surface.insert(0, "weather_year", year)
        surface.insert(0, "model", args.model)
        for key, value in historical_metrics.items():
            surface[f"historical_{key}"] = value
        surface_rows.append(surface)

        baseline_gap = np.maximum(load_h - gen_h, 0.0).astype(np.float32)
        future_gap = gap_f
        base_uncovered = float(baseline_gap.sum(dtype=np.float64))
        future_uncovered = float(future_gap.sum(dtype=np.float64))
        baseline_checks.append({
            "weather_year": year,
            "facility_coverage_historical": 1.0 - base_uncovered / float(demand_h.sum()),
            "facility_coverage_future": 1.0 - future_uncovered / float(demand_f.sum()),
        })
        del (load_h, load_f, cooling_h, cooling_f, gen_h, gen_f, gen_resized,
             gap_f, baseline_gap, future_gap, country_load_h, country_gen_h,
             country_load_f, country_gen_f)
        elapsed = (datetime.now(UTC) - started).total_seconds()
        print(f"  {year} complete in {elapsed:.0f} s", flush=True)

    duration = pd.DataFrame(duration_rows)
    duration.to_csv(args.output_root / "duration_severity_by_year.csv", index=False)
    duration_mean = (duration.groupby(["model", "scenario", "duration_hours"],
                                      as_index=False)
                     .worst_window_uncovered_mwh.mean())
    historical_duration = (duration_mean[duration_mean.scenario.eq("historical")]
                           .set_index("duration_hours")
                           .worst_window_uncovered_mwh)
    duration_mean["change_relative_to_historical_pct"] = [
        (row.worst_window_uncovered_mwh / historical_duration.loc[row.duration_hours] - 1)
        * 100 for row in duration_mean.itertuples()
    ]
    duration_mean.to_csv(args.output_root / "duration_severity_mean.csv", index=False)

    compound_panel = pd.concat(compound_rows, ignore_index=True)
    compound_panel.to_parquet(args.output_root / "compound_site_year.parquet", index=False)
    compound_country = weighted_country_summary(compound_panel)
    compound_country.to_csv(args.output_root / "compound_country_summary.csv", index=False)

    surface_panel = pd.concat(surface_rows, ignore_index=True)
    surface_panel.to_csv(args.output_root / "adaptation_surface_by_year.csv", index=False)
    numeric = ["coverage", "uncovered_mwh", "worst_24h_mwh", "worst_168h_mwh",
               "chronological_worst_24h_mwh", "chronological_worst_168h_mwh",
               "historical_coverage", "historical_uncovered_mwh",
               "historical_worst_24h_mwh", "historical_worst_168h_mwh"]
    surface_mean = (surface_panel.groupby(
        ["model", "additional_procurement_pct", "storage_hours_mean_load"],
        as_index=False)[numeric].mean())
    base = surface_mean[(surface_mean.additional_procurement_pct == 0)
                        & (surface_mean.storage_hours_mean_load == 0)].iloc[0]
    penalty_coverage = base.historical_coverage - base.coverage
    penalty_24 = base.worst_24h_mwh - base.historical_worst_24h_mwh
    penalty_168 = base.worst_168h_mwh - base.historical_worst_168h_mwh
    surface_mean["annual_coverage_restored_fraction"] = np.divide(
        surface_mean.coverage - base.coverage, penalty_coverage)
    surface_mean["worst_24h_restored_fraction"] = np.divide(
        base.worst_24h_mwh - surface_mean.worst_24h_mwh, penalty_24)
    surface_mean["worst_168h_restored_fraction"] = np.divide(
        base.worst_168h_mwh - surface_mean.worst_168h_mwh, penalty_168)
    surface_mean.to_csv(args.output_root / "adaptation_surface_mean.csv", index=False)

    target_rows = []
    for volume_value, volume_curve in surface_mean.groupby(
            "additional_procurement_pct", sort=True):
        for label, metric, historical_column in (
            ("annual", "coverage", "historical_coverage"),
            ("24h", "worst_24h_mwh", "historical_worst_24h_mwh"),
            ("168h", "worst_168h_mwh", "historical_worst_168h_mwh"),
        ):
            value, reachable = interpolation_target(
                volume_curve, metric, float(base[historical_column]))
            target_rows.append({
                "target": label,
                "additional_procurement_pct": float(volume_value),
                "storage_hours_mean_load": value,
                "reachable_within_grid": reachable,
            })
    targets = pd.DataFrame(target_rows)
    targets.to_csv(args.output_root / "storage_restoration_targets.csv", index=False)

    frontier_rows = []
    for metric_label, column in (
        ("annual", "annual_coverage_restored_fraction"),
        ("24h", "worst_24h_restored_fraction"),
        ("168h", "worst_168h_restored_fraction"),
    ):
        for target in (0.5, 0.8, 1.0):
            points = pareto_points(surface_mean, column, target)
            if points.empty:
                frontier_rows.append(pd.DataFrame([{
                    "metric": metric_label,
                    "restoration_target": target,
                    "reachable": False,
                    "additional_procurement_pct": np.nan,
                    "storage_hours_mean_load": np.nan,
                    "achieved_restoration_fraction": np.nan,
                }]))
            else:
                block = points[["additional_procurement_pct",
                                "storage_hours_mean_load", column]].copy()
                block = block.rename(columns={column: "achieved_restoration_fraction"})
                block.insert(0, "reachable", True)
                block.insert(0, "restoration_target", target)
                block.insert(0, "metric", metric_label)
                frontier_rows.append(block)
    frontiers = pd.concat(frontier_rows, ignore_index=True)
    frontiers.to_csv(args.output_root / "adaptation_pareto_frontiers.csv", index=False)

    checks = pd.DataFrame(baseline_checks)
    mean_hist = float(checks.facility_coverage_historical.mean())
    mean_future = float(checks.facility_coverage_future.mean())
    duration_lookup = duration_mean.set_index(["scenario", "duration_hours"])
    rel24 = float(duration_lookup.loc[("future_frozen_volume", 24),
                                     "change_relative_to_historical_pct"])
    rel168 = float(duration_lookup.loc[("future_frozen_volume", 168),
                                      "change_relative_to_historical_pct"])
    compound_weight = compound_panel.demand_future_mwh.to_numpy(dtype=float)
    compound_hist = float(np.average(
        compound_panel.compound_probability_historical, weights=compound_weight))
    compound_future = float(np.average(
        compound_panel.compound_probability_future, weights=compound_weight))
    target_map = targets[targets.additional_procurement_pct.eq(0)].set_index("target")
    reference_channels = pd.read_csv(
        PROCESSED / f"level_shape_{args.model}" / "level_shape_channels.csv")
    reference_channels = reference_channels[
        reference_channels.weather_year.isin(years)]
    ref_coverage = reference_channels[reference_channels.metric.eq("cfe")]
    ref_24 = reference_channels[reference_channels.metric.eq("worst_24h_mwh")]
    ref_168 = reference_channels[reference_channels.metric.eq("worst_168h_mwh")]
    expected_hist_coverage = float(ref_coverage.baseline.mean() * 100)
    expected_future_coverage = float(
        (ref_coverage.baseline + ref_coverage.joint_frozen_volume).mean() * 100)
    expected_24 = float(ref_24.joint_frozen_volume.mean()
                        / ref_24.baseline.mean() * 100)
    expected_168 = float(ref_168.joint_frozen_volume.mean()
                         / ref_168.baseline.mean() * 100)
    payload = {
        "schema_version": "1.0",
        "method_version": METHOD_VERSION,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "model": args.model,
        "weather_years": list(years),
        "sites": int(len(site_ids)),
        "procurement_regions": int(len(country_codes)),
        "round_trip_efficiency": efficiency,
        "storage_starts_empty": True,
        "storage_power_unconstrained": True,
        "storage_pooling_boundary": "within procurement country only",
        "tail_flexibility_interpretation": (
            "event-targeted minimum energy capacity with storage full at the "
            "start of the worst window; chronological greedy tail retained as sensitivity"),
        "compound_thresholds": {
            "high_cooling": "site-year historical 90th percentile",
            "low_renewable": "site-year historical 10th percentile",
            "future_pairing": "historical paired-hour rank structure retained",
        },
        "duration_severity": {
            "windows_hours": list(DURATION_WINDOWS),
            "future_frozen_penalty_24h_pct": rel24,
            "future_frozen_penalty_168h_pct": rel168,
        },
        "storage_restoration_targets_hours_mean_load": {
            key: (None if pd.isna(target_map.loc[key, "storage_hours_mean_load"])
                  else float(target_map.loc[key, "storage_hours_mean_load"]))
            for key in ("annual", "24h", "168h")
        },
        "compound_exposure": {
            "historical_probability_pct": compound_hist * 100,
            "future_probability_pct": compound_future * 100,
            "change_pp": (compound_future - compound_hist) * 100,
        },
        "facility_coverage_check": {
            "historical_mean_pct": mean_hist * 100,
            "future_mean_pct": mean_future * 100,
            "change_pp": (mean_future - mean_hist) * 100,
        },
        "hard_checks": {
            "historical_coverage_matches_existing": abs(
                mean_hist * 100 - expected_hist_coverage) < 0.01,
            "future_coverage_matches_existing": abs(
                mean_future * 100 - expected_future_coverage) < 0.01,
            "duration_24h_matches_existing": abs(rel24 - expected_24) < 0.05,
            "duration_168h_matches_existing": abs(rel168 - expected_168) < 0.05,
            "surface_values_finite": bool(np.isfinite(surface_mean[numeric]).all().all()),
            "compound_probability_positive": bool(compound_hist > 0),
            "annual_target_reachable": bool(
                target_map.loc["annual", "reachable_within_grid"]),
        },
    }
    payload["all_hard_checks_pass"] = all(payload["hard_checks"].values())
    (args.validation_root / "adaptation_frontier_compound.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), "utf-8")
    print(json.dumps({
        "duration_severity": payload["duration_severity"],
        "storage_targets": payload["storage_restoration_targets_hours_mean_load"],
        "compound_exposure": payload["compound_exposure"],
        "hard_checks": payload["hard_checks"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
