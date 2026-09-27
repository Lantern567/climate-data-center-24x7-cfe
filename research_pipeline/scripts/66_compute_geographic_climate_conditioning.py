# -*- coding: utf-8 -*-
"""Evaluate the ACCESS-CM2 climate perturbation under S1--S5 procurement geographies.

Geography is a conditioning variable, not the scientific treatment.  Historical
portfolio mixes are taken from the completed S1--S5 experiment, frozen at their
year-2000 optimum, and then replayed with paired historical and future demand and
supply.  The script reports annual matching, persistent deficit windows, a
level/shape/timing decomposition and resizing performance for every geography.

S4 remains a continental copper-plate upper bound because hourly transmission
headroom and congestion are unavailable.  Tail storage values are event-targeted
energy lower bounds: storage is assumed full at the start of each site's worst
window, unconstrained in power, and has 0.85 discharge efficiency.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "24x7_cfe" / "processed"
COUPLING = PROCESSED / "climate_coupling"
CROSSWALK = COUPLING / "demand_supply_country_crosswalk"
LADDER = PROCESSED / "cfe_geographic_scenarios" / "site_scenario_year_cfe.parquet"
SUPPLY_H = PROCESSED / "wind_solar_supply_20y_fixed_2025" / "replay_results_pv_corrected"
SUPPLY_F = PROCESSED / "wind_solar_supply_20y_fixed_2025" / "replay_results_future_ssp245"
DEMAND_H = Path(r"D:\facility_demand_replay_1995_2014")
DEMAND_F = Path(r"D:\facility_demand_future_1995_2014")
ONSITE_H = Path(r"D:\facility_onsite_supply_cf")
ONSITE_F = Path(r"D:\facility_onsite_supply_cf_future_ACCESS-CM2")
LOCAL_H = Path(r"D:\facility_local_supply_pools")
LOCAL_F = Path(r"D:\facility_local_supply_pools_future_ACCESS-CM2")
DEFAULT_OUTPUT = PROCESSED / "geographic_climate_ACCESS-CM2"
DEFAULT_CACHE = Path(os.environ.get(
    "GEO_CLIMATE_CACHE_ROOT",
    r"C:\Users\Lantern567\.codex\visualizations\2026\08\24\01a03476-4217-7c61-a36b-347bab3591bf\geo_climate_cache"))

YEARS = tuple(range(1995, 2015))
REFERENCE_YEAR = 2000
WINDOWS = (1, 6, 24, 72, 168)
# Counterfactual cell -> column stem used in the per-country tail output.  The
# stems match the annual `*_pp` columns so the tail decomposition reads the same
# way as the coverage one.
TAIL_LABELS = {
    "historical": "historical",
    "demand_level": "demand_level",
    "demand_full_hist_supply": "demand_total",
    "historical_demand_future_supply": "supply_timing",
    "joint_frozen_volume": "future",
    "joint_resized": "resized",
}
EFFICIENCY = 0.85
DEMAND_COLUMN = "global_reconciled_allocation_facility_load_mw"
METHOD_VERSION = "geographic_climate_conditioning_ACCESS-CM2_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", action="append", choices=["S1", "S2", "S3", "S4", "S5"])
    parser.add_argument("--year", type=int, action="append")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    return parser.parse_args()


def calendar(year: int) -> pd.DatetimeIndex:
    return pd.date_range(f"{year}-01-01", f"{year + 1}-01-01", freq="h",
                         inclusive="left", tz="UTC")


def year_slice(year: int) -> tuple[int, int]:
    start = sum(len(calendar(y)) for y in YEARS if y < year)
    return start, start + len(calendar(year))


def load_demand(root: Path, year: int, sites: pd.DataFrame,
                cache_root: Path, cache_label: str) -> np.ndarray:
    cache = cache_root / "demand"
    cache.mkdir(parents=True, exist_ok=True)
    values_path = cache / f"{cache_label}_{year}.npy"
    ids_path = cache / f"{cache_label}_{year}_ids.npy"
    if not values_path.exists() or not ids_path.exists():
        path = root / f"facility_hourly_demand_replay_{year}.parquet"
        frame = pq.read_table(path, columns=["logical_site_id", DEMAND_COLUMN]).to_pandas()
        frame = frame.sort_values("logical_site_id", kind="stable")
        hours = len(calendar(year))
        ids = pd.Index(frame.logical_site_id.unique())
        values = frame[DEMAND_COLUMN].to_numpy(dtype=np.float64).reshape(-1, hours)
        np.save(values_path, values)
        np.save(ids_path, ids.to_numpy())
        del frame, values
        gc.collect()
    ids = pd.Index(np.load(ids_path, allow_pickle=True))
    values = np.load(values_path, mmap_mode="r")
    order = pd.Series(np.arange(len(ids)), index=ids)
    rows = order.reindex(sites.logical_site_id).to_numpy()
    if pd.isna(rows).any():
        raise ValueError(f"Demand replay {year} does not cover every analysis site")
    return np.asarray(values[rows.astype(int)], dtype=np.float64)


def portfolio_shape(pv: np.ndarray, wind: np.ndarray, mix: float) -> np.ndarray:
    e_pv, e_wind = float(pv.sum()), float(wind.sum())
    if e_pv <= 0 and e_wind <= 0:
        return np.zeros_like(pv, dtype=np.float64)
    if e_wind <= 0:
        mix = 0.0
    elif e_pv <= 0:
        mix = 1.0
    out = np.zeros_like(pv, dtype=np.float64)
    if mix < 1.0:
        out += (1.0 - mix) * pv / e_pv
    if mix > 0.0:
        out += mix * wind / e_wind
    return out


def rolling_maxima(gap: np.ndarray) -> dict[int, np.ndarray]:
    cumulative = np.concatenate(
        [np.zeros((len(gap), 1), dtype=np.float64),
         np.cumsum(gap, axis=1, dtype=np.float64)], axis=1)
    return {
        window: np.max(cumulative[:, window:] - cumulative[:, :-window], axis=1)
        for window in WINDOWS
    }


def annual_case(load: np.ndarray, shape: np.ndarray,
                procurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    generation = procurement[:, None] * shape[None, :]
    matched = np.minimum(load, generation).sum(axis=1)
    gap = np.maximum(load - generation, 0.0)
    return matched, gap


def country_cf(root: Path, year: int) -> tuple[pd.Index, np.ndarray, np.ndarray]:
    partition = root / "hourly_country_supply" / f"weather_year={year}"
    frame = pd.read_parquet(next(partition.glob("*.parquet")))
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
        pv = np.where(pv_cap[:, None] > 0, pv / np.maximum(pv_cap[:, None], 1e-9), 0.0)
        wind = np.where(wind_cap[:, None] > 0, wind / np.maximum(wind_cap[:, None], 1e-9), 0.0)
    return codes, pv, wind


def regional_cf(root: Path, year: int, country_to_region: pd.Series
                ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    partition = root / "hourly_country_supply" / f"weather_year={year}"
    frame = pd.read_parquet(next(partition.glob("*.parquet")))
    annual = pd.read_parquet(root / "annual_country_supply_1995_2014.parquet")
    annual = annual[annual.weather_year.eq(year)].set_index("country_area_code")
    frame["region"] = frame.country_area_code.map(country_to_region)
    frame = frame[frame.region.notna()]
    pools: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for region, block in frame.groupby("region"):
        codes = block.country_area_code.unique()
        pv_cap = float(annual.loc[codes, "solar_pv_installed_capacity_mw"].sum())
        wind_cap = float(annual.loc[codes, "wind_installed_capacity_mw"].sum())
        summed = block.groupby("time_utc")[["solar_pv_generation_power_mw",
                                             "wind_generation_power_mw"]].sum().sort_index()
        pv = (summed.solar_pv_generation_power_mw.to_numpy() / pv_cap
              if pv_cap > 0 else np.zeros(len(summed)))
        wind = (summed.wind_generation_power_mw.to_numpy() / wind_cap
                if wind_cap > 0 else np.zeros(len(summed)))
        pools[str(region)] = (pv, wind)
    return pools


def global_cf(root: Path, year: int) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_parquet(root / "hourly_global_supply_1995_2014.parquet")
    frame = frame[frame.weather_year.eq(year)].sort_values("time_utc")
    annual = pd.read_csv(root / "annual_global_supply_1995_2014.csv")
    row = annual[annual.weather_year.eq(year)].iloc[0]
    return (frame.solar_pv_generation_power_mw.to_numpy()
            / row.solar_pv_installed_capacity_mw,
            frame.wind_generation_power_mw.to_numpy()
            / row.wind_installed_capacity_mw)


def load_npz_pair(hist_root: Path, future_root: Path, filename: str,
                  cache_root: Path, cache_name: str) -> dict[str, np.ndarray]:
    """Materialise compressed pools once, then reopen them as low-RAM memmaps."""
    target = cache_root / cache_name
    target.mkdir(parents=True, exist_ok=True)
    required = [target / f"{name}.npy" for name in
                ("ids", "pv_h", "wind_h", "pv_f", "wind_f", "capacity")]
    if not all(path.exists() for path in required):
        print(f"  extracting {cache_name} compressed pools to memmap cache", flush=True)
        with np.load(hist_root / filename) as archive:
            ids_h = archive["era5_grid_id"].astype(str)
            np.save(target / "ids.npy", ids_h)
            np.save(target / "pv_h.npy", archive["pv_cf"].astype(np.float32))
            np.save(target / "wind_h.npy", archive["wind_cf"].astype(np.float32))
            capacity = ((archive["pv_capacity_mw"] + archive["wind_capacity_mw"]).astype(float)
                        if "pv_capacity_mw" in archive else np.ones(len(ids_h)))
            np.save(target / "capacity.npy", capacity)
        gc.collect()
        with np.load(future_root / filename) as archive:
            ids_f = archive["era5_grid_id"].astype(str)
            if not np.array_equal(ids_h, ids_f):
                raise ValueError(f"Historical and future pool order differs for {filename}")
            np.save(target / "pv_f.npy", archive["pv_cf"].astype(np.float32))
            np.save(target / "wind_f.npy", archive["wind_cf"].astype(np.float32))
        gc.collect()
    return {
        "ids": np.load(target / "ids.npy", mmap_mode="r"),
        "pv_h": np.load(target / "pv_h.npy", mmap_mode="r"),
        "wind_h": np.load(target / "wind_h.npy", mmap_mode="r"),
        "pv_f": np.load(target / "pv_f.npy", mmap_mode="r"),
        "wind_f": np.load(target / "wind_f.npy", mmap_mode="r"),
        "capacity": np.load(target / "capacity.npy", mmap_mode="r"),
    }


def pool_spec(scenario: str, year: int, sites: pd.DataFrame,
              country_to_region: pd.Series, large_pool: dict[str, np.ndarray] | None
             ) -> tuple[dict[str, np.ndarray], dict[str, tuple[np.ndarray, np.ndarray,
                                                               np.ndarray, np.ndarray]]]:
    """Return site groups and paired historical/future PV-wind pools."""
    if scenario in ("S1", "S2"):
        assert large_pool is not None
        start, stop = year_slice(year)
        row_lookup = pd.Series(np.arange(len(large_pool["ids"])), index=large_pool["ids"])
        groups: dict[str, np.ndarray] = {}
        pools: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        for key, positions in sites.groupby("era5_grid_id").indices.items():
            if key not in row_lookup.index:
                continue
            row = int(row_lookup[key])
            if scenario == "S2" and large_pool["capacity"][row] <= 0:
                continue
            groups[str(key)] = np.asarray(positions)
            pools[str(key)] = (large_pool["pv_h"][row, start:stop],
                               large_pool["wind_h"][row, start:stop],
                               large_pool["pv_f"][row, start:stop],
                               large_pool["wind_f"][row, start:stop])
        return groups, pools

    if scenario == "S3":
        codes_h, pv_h, wind_h = country_cf(SUPPLY_H, year)
        codes_f, pv_f, wind_f = country_cf(SUPPLY_F, year)
        row_h = pd.Series(np.arange(len(codes_h)), index=codes_h)
        row_f = pd.Series(np.arange(len(codes_f)), index=codes_f)
        groups = {str(k): np.asarray(v) for k, v in
                  sites.groupby("supply_country_area_code").indices.items()
                  if k in row_h.index and k in row_f.index}
        pools = {key: (pv_h[int(row_h[key])], wind_h[int(row_h[key])],
                       pv_f[int(row_f[key])], wind_f[int(row_f[key])]) for key in groups}
        return groups, pools

    if scenario == "S4":
        hist = regional_cf(SUPPLY_H, year, country_to_region)
        future = regional_cf(SUPPLY_F, year, country_to_region)
        groups = {str(k): np.asarray(v) for k, v in sites.groupby("region").indices.items()
                  if str(k) in hist and str(k) in future}
        pools = {key: (*hist[key], *future[key]) for key in groups}
        return groups, pools

    if scenario == "S5":
        pv_h, wind_h = global_cf(SUPPLY_H, year)
        pv_f, wind_f = global_cf(SUPPLY_F, year)
        return {"GLOBAL": np.arange(len(sites))}, {
            "GLOBAL": (pv_h, wind_h, pv_f, wind_f)}
    raise ValueError(scenario)


def summarize_case(case: str, demand: np.ndarray, matched: np.ndarray,
                   tails: dict[int, np.ndarray], scenario: str, year: int) -> dict[str, float | str | int]:
    row: dict[str, float | str | int] = {
        "scenario": scenario, "weather_year": year, "case": case,
        "sites": int(len(demand)), "demand_mwh": float(demand.sum()),
        "matched_mwh": float(matched.sum()),
        "coverage": float(matched.sum() / demand.sum()),
        "uncovered_mwh": float(demand.sum() - matched.sum()),
    }
    for window, values in tails.items():
        row[f"worst_{window}h_mwh"] = float(values.sum())
    return row


def run_scenario(scenario: str, years: tuple[int, ...], sites_master: pd.DataFrame,
                 mixes: pd.Series, country_to_region: pd.Series,
                 large_pool: dict[str, np.ndarray] | None,
                 cache_root: Path,
                 summary_rows: list[dict], mechanism_rows: list[dict],
                 country_rows: list[dict]) -> None:
    eligible_ids = mixes.index
    sites = sites_master[sites_master.logical_site_id.isin(eligible_ids)].copy()
    sites = sites.sort_values("logical_site_id").reset_index(drop=True)
    mix_values = mixes.reindex(sites.logical_site_id).to_numpy(dtype=float)
    if np.isnan(mix_values).any():
        raise ValueError(f"Missing frozen mix within {scenario} analysis sites")

    for year in years:
        load_h = load_demand(DEMAND_H, year, sites, cache_root, "historical")
        load_f = load_demand(DEMAND_F, year, sites, cache_root, "future_ACCESS-CM2")
        demand_h = load_h.sum(axis=1)
        demand_f = load_f.sum(axis=1)
        load_level = load_h * (demand_f / np.maximum(demand_h, 1e-12))[:, None]
        groups, pools = pool_spec(scenario, year, sites, country_to_region, large_pool)

        n = len(sites)
        matched = {name: np.zeros(n) for name in
                   ("historical", "demand_level", "demand_full_hist_supply",
                    "historical_demand_future_supply", "joint_frozen_volume",
                    "joint_resized")}
        # Every counterfactual cell needs rolling maxima, not just the three
        # endpoints: Result 2 decomposes the *tail* into demand level, demand
        # shape and supply timing, and that decomposition is only available if
        # each intermediate cell carries its own window statistics.
        tails = {case: {window: np.zeros(n) for window in WINDOWS}
                 for case in matched}
        covered = np.zeros(n, dtype=bool)

        for key, positions_all in groups.items():
            if key not in pools:
                continue
            positions_all = np.asarray(positions_all)
            pv_h, wind_h, pv_f, wind_f = pools[key]
            for mix in np.unique(mix_values[positions_all]):
                positions = positions_all[mix_values[positions_all] == mix]
                if len(positions) == 0:
                    continue
                shape_h = portfolio_shape(pv_h, wind_h, float(mix))
                shape_f = portfolio_shape(pv_f, wind_f, float(mix))
                sub_h, sub_f = load_h[positions], load_f[positions]
                sub_level = load_level[positions]
                proc_h, proc_f = demand_h[positions], demand_f[positions]

                cases = (
                    ("historical", sub_h, shape_h, proc_h),
                    ("demand_level", sub_level, shape_h, proc_h),
                    ("demand_full_hist_supply", sub_f, shape_h, proc_h),
                    ("historical_demand_future_supply", sub_h, shape_f, proc_h),
                    ("joint_frozen_volume", sub_f, shape_f, proc_h),
                    ("joint_resized", sub_f, shape_f, proc_f),
                )
                for case, load, shape, procured in cases:
                    matched[case][positions], gap = annual_case(load, shape, procured)
                    values = rolling_maxima(gap)
                    for window in WINDOWS:
                        tails[case][window][positions] = values[window]
                    del gap, values
                covered[positions] = True

        if not covered.all():
            missing = int((~covered).sum())
            raise ValueError(f"{scenario} {year}: {missing} frozen-mix sites lack paired pools")

        demand_by_case = {
            "historical": demand_h,
            "demand_level": demand_f,
            "demand_full_hist_supply": demand_f,
            "historical_demand_future_supply": demand_h,
            "joint_frozen_volume": demand_f,
            "joint_resized": demand_f,
        }
        for case in matched:
            case_tails = tails.get(case, {})
            summary_rows.append(summarize_case(
                case, demand_by_case[case], matched[case], case_tails, scenario, year))

        coverage = {case: float(matched[case].sum() / demand_by_case[case].sum())
                    for case in matched}
        level = coverage["demand_level"] - coverage["historical"]
        shape = coverage["demand_full_hist_supply"] - coverage["demand_level"]
        supply = coverage["historical_demand_future_supply"] - coverage["historical"]
        joint = coverage["joint_frozen_volume"] - coverage["historical"]
        mechanism_rows.append({
            "scenario": scenario, "weather_year": year,
            "historical_coverage": coverage["historical"],
            "future_coverage": coverage["joint_frozen_volume"],
            "demand_level_pp": level * 100.0,
            "demand_shape_pp": shape * 100.0,
            "supply_timing_pp": supply * 100.0,
            "interaction_pp": (joint - level - shape - supply) * 100.0,
            "joint_pp": joint * 100.0,
        })

        site_frame = pd.DataFrame({
            "country": sites.supply_country_area_code.to_numpy(),
            "d_h": demand_h, "d_f": demand_f,
            **{f"m_{case}": values for case, values in matched.items()},
            # Result 4 prices the tail per procurement pool, so the rolling-window
            # maxima have to leave this loop grouped by country as well as globally.
            # Summing site maxima matches how script 63 aggregates site firming.
            **{f"t_{case}_{window}": values[window]
               for case, values in tails.items() for window in WINDOWS},
        })
        grouped = site_frame.groupby("country", as_index=False).sum(numeric_only=True)
        for row in grouped.itertuples(index=False):
            c_hist = row.m_historical / row.d_h
            c_level = row.m_demand_level / row.d_f
            c_shape = row.m_demand_full_hist_supply / row.d_f
            c_supply = row.m_historical_demand_future_supply / row.d_h
            c_joint = row.m_joint_frozen_volume / row.d_f
            d_level = c_level - c_hist
            d_shape = c_shape - c_level
            d_supply = c_supply - c_hist
            country_rows.append({
                "scenario": scenario, "weather_year": year, "country": row.country,
                "historical_demand_mwh": row.d_h, "future_demand_mwh": row.d_f,
                "historical_coverage": c_hist, "future_coverage": c_joint,
                "demand_level_pp": d_level * 100.0,
                "demand_shape_pp": d_shape * 100.0,
                "supply_timing_pp": d_supply * 100.0,
                "interaction_pp": (c_joint - c_hist - d_level - d_shape - d_supply) * 100.0,
                "joint_pp": (c_joint - c_hist) * 100.0,
                **{f"worst_{window}h_{label}_mwh": getattr(row, f"t_{case}_{window}")
                   for window in WINDOWS for case, label in TAIL_LABELS.items()},
            })

        print(f"  {scenario} {year} complete", flush=True)
        del load_h, load_f, load_level, matched, tails
        gc.collect()


def main() -> None:
    args = parse_args()
    scenarios = tuple(args.scenario) if args.scenario else ("S1", "S2", "S3", "S4", "S5")
    years = tuple(sorted(set(args.year))) if args.year else YEARS
    args.output_root.mkdir(parents=True, exist_ok=True)

    sites = pd.read_csv(CROSSWALK / "facility_to_era5_grid_with_country.csv", low_memory=False)
    sites = sites[sites.supply_country_area_code.notna()].copy()
    sites = sites.sort_values("logical_site_id").reset_index(drop=True)
    sites["region"] = sites.continent.where(~sites.continent.isin(["Unknown", "Antarctica"]))
    country_to_region = (sites.dropna(subset=["region"])
                         .groupby("supply_country_area_code").region
                         .agg(lambda values: values.mode().iloc[0]))

    ladder = pd.read_parquet(
        LADDER, columns=["scenario", "weather_year", "logical_site_id", "wind_energy_share"])
    ladder = ladder[ladder.weather_year.eq(REFERENCE_YEAR)]
    mixes = {scenario: block.drop_duplicates("logical_site_id").set_index("logical_site_id")
             .wind_energy_share for scenario, block in ladder.groupby("scenario")}

    summary_rows: list[dict] = []
    mechanism_rows: list[dict] = []
    country_rows: list[dict] = []
    for scenario in scenarios:
        large_pool = None
        if scenario == "S1":
            large_pool = load_npz_pair(
                ONSITE_H, ONSITE_F, "onsite_supply_cf.npz",
                args.cache_root, "S1_onsite")
        elif scenario == "S2":
            large_pool = load_npz_pair(
                LOCAL_H, LOCAL_F, "local_supply_pool_cf.npz",
                args.cache_root, "S2_local_100km")
        run_scenario(scenario, years, sites, mixes[scenario], country_to_region,
                     large_pool, args.cache_root, summary_rows, mechanism_rows, country_rows)
        del large_pool
        gc.collect()

    summary = pd.DataFrame(summary_rows)
    mechanisms = pd.DataFrame(mechanism_rows)
    countries = pd.DataFrame(country_rows)
    summary.to_csv(args.output_root / "geographic_climate_summary_by_year.csv", index=False)
    mechanisms.to_csv(args.output_root / "geographic_mechanism_by_year.csv", index=False)
    countries.to_csv(args.output_root / "geographic_country_mechanism_by_year.csv", index=False)

    mean_summary = summary.groupby(["scenario", "case"], as_index=False).mean(numeric_only=True)
    mean_summary.to_csv(args.output_root / "geographic_climate_summary_mean.csv", index=False)
    mean_mechanisms = mechanisms.groupby("scenario", as_index=False).mean(numeric_only=True)
    mean_mechanisms.to_csv(args.output_root / "geographic_mechanism_mean.csv", index=False)
    mean_countries = countries.groupby(["scenario", "country"], as_index=False).mean(numeric_only=True)
    mean_countries.to_csv(args.output_root / "geographic_country_mechanism_mean.csv", index=False)

    penalty_rows = []
    for scenario in scenarios:
        block = mean_summary[mean_summary.scenario.eq(scenario)].set_index("case")
        historical = block.loc["historical"]
        future = block.loc["joint_frozen_volume"]
        resized = block.loc["joint_resized"]
        row = {
            "scenario": scenario,
            "historical_coverage": historical.coverage,
            "future_coverage": future.coverage,
            "resized_coverage": resized.coverage,
            "coverage_penalty_pp": (future.coverage - historical.coverage) * 100.0,
            "coverage_penalty_resized_pp": (resized.coverage - historical.coverage) * 100.0,
            "uncovered_change_pct": (future.uncovered_mwh / historical.uncovered_mwh - 1.0) * 100.0,
            "uncovered_change_resized_pct": (resized.uncovered_mwh / historical.uncovered_mwh - 1.0) * 100.0,
        }
        for window in WINDOWS:
            h = historical[f"worst_{window}h_mwh"]
            f = future[f"worst_{window}h_mwh"]
            r = resized[f"worst_{window}h_mwh"]
            row[f"worst_{window}h_change_pct"] = (f / h - 1.0) * 100.0
            row[f"worst_{window}h_change_resized_pct"] = (r / h - 1.0) * 100.0
        penalty_rows.append(row)
    penalties = pd.DataFrame(penalty_rows)
    penalties.to_csv(args.output_root / "geographic_climate_penalties_mean.csv", index=False)

    validation = {
        "schema_version": "1.0", "method_version": METHOD_VERSION,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "model": "ACCESS-CM2", "scenario": "SSP2-4.5", "future_window": "2031-2050",
        "weather_years": list(years), "procurement_geographies": list(scenarios),
        "mix_reference_year": REFERENCE_YEAR, "mix_frozen": True,
        "s4_interpretation": "continental copper-plate upper bound",
        "tail_storage_interpretation": "event-targeted energy lower bound",
        "hard_checks": {
            "all_scenarios_present": set(penalties.scenario) == set(scenarios),
            "coverage_within_unit_interval": bool(summary.coverage.between(0, 1).all()),
            "uncovered_non_negative": bool((summary.uncovered_mwh >= -1e-6).all()),
            "mechanism_closure": bool(np.allclose(
                mechanisms.joint_pp,
                mechanisms.demand_level_pp + mechanisms.demand_shape_pp
                + mechanisms.supply_timing_pp + mechanisms.interaction_pp,
                atol=1e-8)),
        },
    }
    validation["all_hard_checks_pass"] = all(validation["hard_checks"].values())
    (args.output_root / "geographic_climate_validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(validation, indent=2, ensure_ascii=False), flush=True)
    print(penalties.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
