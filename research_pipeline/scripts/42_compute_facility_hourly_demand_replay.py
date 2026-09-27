# -*- coding: utf-8 -*-
"""Twenty-year facility hourly demand replay (1995--2014 weather, fixed demand).

Answers "how much electricity would the SAME data-centre fleet, running the SAME
computing workload, draw under twenty different historical weather years?".  It
does not reconstruct what data centres actually consumed in 1995--2014 -- the
fleet did not exist at today's scale.

Design constraints carried over from the three-year production run
(``35_compute_facility_hourly_climate_demand.py``):

  * weather acts only through PUE; the IT workload itself is weather-invariant;
  * the interval [t, t+1) takes the trapezoidal mean of the NON-LINEAR PUE
    evaluated at both endpoints -- never PUE of the averaged weather;
  * the two demand scopes (``site_marginal`` / ``global_reconciled_allocation``)
    are produced side by side and must never be summed together.

Two things are new here:

  * The IT template is re-expressed on tau = (month, weekday, hour-of-day) so it
    can be instantiated on any calendar.  The published 2023--2025 templates are
    indexed by hour-of-year and cannot address a 1995 calendar directly.
  * A SINGLE demand scale is shared by all twenty weather years (g = 1 in the
    proposal's notation).  Per-year recalibration would defeat the whole point:
    the experiment isolates weather, so demand scale must not move with it.

Storage: the fact table keeps only quantities that are not exact duplicates or
exact identities of one another.  ``*_energy_mwh`` equals ``*_load_mw`` because
the interval is one hour, facility overhead is ``facility - it`` and PUE is
``facility / it``; Parquet compresses each column independently and performs no
cross-column de-duplication, so storing them costs full price for zero
information.  Dropping the twelve redundant columns takes the twenty-year table
from ~134 GiB to ~47 GiB with nothing lost.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "research_pipeline" / "scripts"


def _load_by_path(name: str, filename: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# The three-year demand engine and the weather validator are validated scripts,
# not importable modules.  Load them by path so the type constants, the PUE
# archetype loader and the humidity formula are reused verbatim instead of being
# copied into a second definition that can silently drift.
BASE = _load_by_path("dcwatch_climate_demand", "26_compute_dcwatch_climate_demand_supply.py")
WEATHER = _load_by_path("facility_weather_validator", "41_validate_facility_weather_replay.py")

FACILITY_TYPES: tuple[str, ...] = BASE.FACILITY_TYPES
TYPE_LINKED_ARCHETYPE: dict[str, str] = BASE.TYPE_LINKED_ARCHETYPE
SCENARIO: str = BASE.SCENARIO
TEMPLATE_YEARS = (2023, 2024, 2025)
REPLAY_YEARS = tuple(range(1995, 2015))

METHOD_VERSION = "facility_hourly_demand_replay_1995_2014_v1"
DEMAND_SCOPES = ("site_marginal", "global_reconciled_allocation")
PUE_SCENARIOS = ("type_linked_proxy", "all_advanced", "all_liquid", "all_conventional")
SCENARIO_ARCHETYPE = {
    "all_advanced": "advanced_hyperscale",
    "all_liquid": "liquid",
    "all_conventional": "conventional",
}
PUE_VALIDATION_STATUS = "engineering_scenario_only_not_globally_validated"
SITE_STATUS = "modeled_site_marginal_not_metered"
GLOBAL_STATUS = "global_p50_conditionally_allocated_not_site_prediction"
RESULT_ROLE = "demand_weather_replay_fixed_workload"

DEFAULT_WEATHER_ROOT = Path(
    os.environ.get("FACILITY_WEATHER_ROOT") or r"D:\era5_facility_weather_1995_2014"
)
DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get("FACILITY_DEMAND_REPLAY_ROOT")
    or r"D:\facility_demand_replay_1995_2014"
)
DEFAULT_SUMMARY_ROOT = (
    ROOT / "data" / "24x7_cfe" / "processed" / "climate_coupling" / "demand_replay_1995_2014"
)

ARRAY_NAMES = {"2m_temperature": "t2m_k", "2m_dewpoint_temperature": "d2m_k",
               "surface_pressure": "sp_pa"}
EXPECTED_ENDPOINTS = 175_321
EXPECTED_INTERVALS = 175_320
WET_BULB_CLIP = (-20.0, 50.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weather-root", type=Path, default=DEFAULT_WEATHER_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--summary-root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    p.add_argument("--mapping", type=Path, default=BASE.DEFAULT_MAPPING)
    p.add_argument("--grid-catalog", type=Path, default=BASE.DEFAULT_CATALOG)
    p.add_argument("--type-annual", type=Path, default=BASE.DEFAULT_TYPE_ANNUAL)
    p.add_argument("--type-summary-root", type=Path, default=BASE.HIER)
    p.add_argument("--global-hourly", type=Path, default=BASE.DEFAULT_GLOBAL)
    p.add_argument("--scenario", default=SCENARIO)
    p.add_argument("--year", type=int, action="append",
                   help="Restrict to these weather years (repeatable).")
    p.add_argument("--limit-grids", type=int)
    p.add_argument("--compression-level", type=int, default=7)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


# --------------------------------------------------------------- tau template
def calendar(year: int) -> pd.DatetimeIndex:
    return pd.date_range(f"{year}-01-01", f"{year + 1}-01-01", freq="h",
                         inclusive="left", tz="UTC")


def build_tau_template(root: Path) -> pd.DataFrame:
    """Median normalized load on tau = (month, weekday, hour-of-day).

    The published templates are indexed by hour-of-year on the 2023--2025
    calendars, which cannot address a 1995 calendar.  The proposal defines the
    workload template on (season, day type, hour position) precisely so it can
    be carried to another calendar, so that is the index rebuilt here.
    """
    frames = []
    for year in TEMPLATE_YEARS:
        wide = BASE.load_type_profiles(root, year)
        index = calendar(year)
        frame = pd.DataFrame(wide.to_numpy(dtype=float), columns=list(FACILITY_TYPES))
        frame["month"] = index.month
        frame["weekday"] = index.dayofweek
        frame["hour"] = index.hour
        frames.append(frame)
    tau = (
        pd.concat(frames, ignore_index=True)
        .groupby(["month", "weekday", "hour"])[list(FACILITY_TYPES)]
        .median()
    )
    if len(tau) != 12 * 7 * 24:
        raise ValueError(f"tau template is incomplete: {len(tau)} cells")
    if not np.isfinite(tau.to_numpy(dtype=float)).all():
        raise ValueError("tau template contains non-finite values")
    return tau


def instantiate_template(tau: pd.DataFrame, year: int) -> np.ndarray:
    index = calendar(year)
    key = pd.MultiIndex.from_arrays([index.month, index.dayofweek, index.hour])
    values = tau.reindex(key).to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"tau template failed to cover calendar {year}")
    values = values / values.mean(axis=0)
    if not np.allclose(values.mean(axis=0), 1.0, atol=1e-9):
        raise ValueError(f"Instantiated template for {year} is not unit mean")
    return values


def localize(reference: np.ndarray, year: int, timezones: list[str]) -> dict[str, np.ndarray]:
    """Address the annual template in each site's civil time, preserving DST.

    Mirrors ``26_compute_dcwatch_climate_demand_supply.localize_profiles``: the
    template is treated cyclically at UTC-year boundaries and sub-hour zone
    offsets are handled by linear interpolation between adjacent template hours.
    """
    utc_index = calendar(year)
    # (month, day, hour) -> hour-of-year lookup for this template year.  The
    # original implementation rebuilt the timestamp with pd.to_datetime once per
    # time zone, which costs ~5 ms x 227 zones x 20 years and dominated the run.
    lut = np.full((13, 32, 24), -1, dtype=np.int32)
    lut[utc_index.month, utc_index.day, utc_index.hour] = np.arange(len(utc_index))
    out: dict[str, np.ndarray] = {}
    for timezone in timezones:
        local = utc_index.tz_convert(timezone)
        left = lut[local.month, local.day, local.hour]
        if (left < 0).any() or (left >= len(reference)).any():
            raise ValueError(f"Local template mapping failed for {timezone}/{year}")
        fraction = np.asarray(local.minute, dtype=float) / 60.0
        right = (left + 1) % len(reference)
        out[timezone] = (
            reference[left] * (1.0 - fraction)[:, None] + reference[right] * fraction[:, None]
        )
    return out


# ------------------------------------------------------------------- weather
def endpoint_pue(path: Path, pue_module: Any) -> dict[str, np.ndarray]:
    """Return each archetype's PUE evaluated at all 175,321 hourly endpoints."""
    with np.load(path) as payload:
        temperature = payload[ARRAY_NAMES["2m_temperature"]].astype(np.float64)
        dewpoint = payload[ARRAY_NAMES["2m_dewpoint_temperature"]].astype(np.float64)
    if temperature.shape != (EXPECTED_ENDPOINTS,) or dewpoint.shape != (EXPECTED_ENDPOINTS,):
        raise ValueError(f"{path.name}: expected {EXPECTED_ENDPOINTS} endpoints")
    humidity = WEATHER.relative_humidity_percent(temperature, dewpoint)
    wet_bulb = pue_module.stull_wet_bulb_c(temperature - 273.15, np.clip(humidity, 0.0, 100.0))
    wet_bulb = np.clip(wet_bulb, *WET_BULB_CLIP)
    return {name: pue_module.reduced_order_pue(wet_bulb, archetype)
            for name, archetype in pue_module.PUE_ARCHETYPES.items()}


def interval_pue(endpoints: dict[str, np.ndarray], start: int, stop: int) -> dict[str, np.ndarray]:
    """Trapezoidal mean of the non-linear PUE over each [t, t+1) interval."""
    return {name: 0.5 * (values[start:stop] + values[start + 1:stop + 1])
            for name, values in endpoints.items()}


def year_slices() -> dict[int, tuple[int, int]]:
    """Interval index range covered by each weather year."""
    endpoints = pd.date_range("1995-01-01", "2015-01-01", freq="h", tz="UTC")
    if len(endpoints) != EXPECTED_ENDPOINTS:
        raise ValueError("Replay window is not 175,321 endpoints")
    interval_start = endpoints[:-1]
    bounds: dict[int, tuple[int, int]] = {}
    cursor = 0
    for year in REPLAY_YEARS:
        hours = len(calendar(year))
        bounds[year] = (cursor, cursor + hours)
        if interval_start[cursor].year != year:
            raise ValueError(f"Interval bookkeeping drifted at {year}")
        cursor += hours
    if cursor != EXPECTED_INTERVALS:
        raise ValueError(f"Intervals sum to {cursor}, expected {EXPECTED_INTERVALS}")
    return bounds


# -------------------------------------------------------------------- schema
def fact_schema() -> pa.Schema:
    metadata = {
        b"method_version": METHOD_VERSION.encode(),
        b"result_role": RESULT_ROLE.encode(),
        b"interval_semantics": b"UTC [t,t+1); interval_hours=1",
        b"hourly_pue_scenario": b"type_linked_proxy",
        b"pue_validation_status": PUE_VALIDATION_STATUS.encode(),
        b"contains_metered_facility_load": b"false",
        b"is_historical_actual_demand": b"false",
        b"demand_scale_fixed_across_weather_years": b"true",
        b"site_marginal_interpretation": SITE_STATUS.encode(),
        b"global_reconciled_interpretation": GLOBAL_STATUS.encode(),
        # Every dropped column is an exact duplicate or exact identity of a
        # retained one.  Recovering them downstream is a one-liner.
        b"omitted_derivable_columns": (
            b"*_energy_mwh == *_load_mw (interval_hours=1); "
            b"*_facility_overhead_load_mw == facility - it; "
            b"*_pue_type_linked == facility / it"
        ),
    }
    fields = [
        pa.field("weather_year", pa.int16()),
        pa.field("logical_site_id", pa.string()),
        pa.field("era5_grid_id", pa.string()),
        pa.field("time_utc", pa.timestamp("ns", tz="UTC")),
    ]
    for scope in DEMAND_SCOPES:
        fields.extend([
            pa.field(f"{scope}_it_load_mw", pa.float32()),
            pa.field(f"{scope}_facility_load_mw", pa.float32()),
            pa.field(f"{scope}_climate_variable_overhead_load_mw", pa.float32()),
        ])
    return pa.schema(fields, metadata=metadata)


# ---------------------------------------------------------------------- main
def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    years = tuple(sorted(set(args.year))) if args.year else REPLAY_YEARS
    if not set(years).issubset(set(REPLAY_YEARS)):
        raise ValueError(f"Years must be drawn from {REPLAY_YEARS[0]}..{REPLAY_YEARS[-1]}")

    pue_module = BASE.load_pue_module()
    _, strata = BASE.load_spatial_strata(args.mapping)
    type_medians = BASE.load_type_medians(args.type_annual, args.scenario)
    tau = build_tau_template(args.type_summary_root)

    catalog = pd.read_csv(args.mapping, low_memory=False)
    if len(catalog) != 8_360 or catalog["logical_site_id"].nunique() != 8_360:
        raise ValueError("Expected 8,360 unique logical sites")
    weight_columns = [f"spatial_weight_mw_{name}" for name in FACILITY_TYPES]
    type_totals = catalog[weight_columns].sum()
    if (type_totals <= 0.0).any():
        raise ValueError("Every facility type must have positive total spatial weight")
    catalog = catalog.sort_values(["era5_grid_id", "logical_site_id"]).reset_index(drop=True)

    timezones = sorted(strata["timezone_name"].unique())
    type_index = {name: i for i, name in enumerate(FACILITY_TYPES)}

    # ---- pass 1: one demand scale shared by all twenty weather years --------
    # The published 2023--2025 targets pin "today's" demand level.  Anchoring the
    # twenty-year MEAN to that level keeps g = 1 while letting only the calendar
    # (leap days, weekday counts) move IT energy between years.
    target_mean_mw = float(np.mean([
        BASE.global_target_mean_mw(args.global_hourly, args.scenario, y)
        for y in TEMPLATE_YEARS
    ]))
    templates: dict[int, dict[str, np.ndarray]] = {}
    raw_year_mean: dict[int, float] = {}
    for year in REPLAY_YEARS:
        localized = localize(instantiate_template(tau, year), year, timezones)
        # float32 halves the 1.5 GiB resident template set; the values are unit
        # scale so seven significant digits are far beyond what the float32
        # output needs, and the products below are accumulated in float64.
        if year in years:
            templates[year] = {k: v.astype(np.float32) for k, v in localized.items()}
        raw = np.zeros(len(calendar(year)), dtype=float)
        for row in strata.itertuples(index=False):
            mass = float(type_medians[str(row.facility_type)]) * float(row.within_type_spatial_share)
            raw += localized[str(row.timezone_name)][:, type_index[str(row.facility_type)]] * mass
        raw_year_mean[year] = float(raw.mean())
    hours_per_year = {y: len(calendar(y)) for y in REPLAY_YEARS}
    total_hours = sum(hours_per_year.values())
    pooled_raw_mean = sum(raw_year_mean[y] * hours_per_year[y] for y in REPLAY_YEARS) / total_hours
    annual_scale_factor = target_mean_mw / pooled_raw_mean
    if not np.isfinite(annual_scale_factor) or annual_scale_factor <= 0.0:
        raise ValueError("Invalid fixed demand scale")
    print(f"target mean IT = {target_mean_mw / 1000:.4f} GW | "
          f"fixed scale = {annual_scale_factor:.6f}")

    # ---- per-site constants -------------------------------------------------
    site_weights = catalog[weight_columns].to_numpy(dtype=float)
    site_anchor = catalog["mean_it_mw_p50"].to_numpy(dtype=float)
    site_timezone = catalog["timezone_name"].astype(str).to_numpy()
    site_grid = catalog["era5_grid_id"].astype(str).to_numpy()
    site_id = catalog["logical_site_id"].astype(str).to_numpy()
    # global_reconciled_allocation mass: the type-level posterior median spread
    # over sites by within-type spatial share, then scaled once.
    global_mass = (
        site_weights
        / type_totals.to_numpy(dtype=float)[None, :]
        * type_medians.to_numpy(dtype=float)[None, :]
        * annual_scale_factor
    )
    type_baseline = np.asarray([
        1.0
        + pue_module.PUE_ARCHETYPES[TYPE_LINKED_ARCHETYPE[name]].fixed_overhead
        + pue_module.PUE_ARCHETYPES[TYPE_LINKED_ARCHETYPE[name]].flow_base
        for name in FACILITY_TYPES
    ], dtype=float)

    grid_ids = sorted(catalog["era5_grid_id"].unique())
    if args.limit_grids is not None:
        grid_ids = grid_ids[: args.limit_grids]
    selected = np.isin(site_grid, grid_ids)
    site_rows = {g: np.flatnonzero(site_grid == g) for g in grid_ids}
    # Sites in one grid usually share a time zone but not always (a 0.25 deg
    # cell can straddle a zone border), so group by zone inside each grid.
    grid_timezone_groups: dict[str, dict[str, np.ndarray]] = {}
    for grid, rows_for_grid in site_rows.items():
        zones = site_timezone[rows_for_grid]
        grid_timezone_groups[grid] = {
            str(zone): np.flatnonzero(zones == zone) for zone in np.unique(zones)
        }

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.summary_root.mkdir(parents=True, exist_ok=True)
    schema = fact_schema()
    bounds = year_slices()

    annual_records: list[dict[str, Any]] = []
    global_records: list[dict[str, Any]] = []
    row_counts: dict[int, int] = {}
    diagnostics = {"nonfinite": 0, "negative": 0,
                   "pue_min": float("inf"), "pue_max": float("-inf"),
                   "site_anchor_max_abs_mw": 0.0}

    # Endpoint PUE is weather-year independent, so it is computed once per grid
    # and sliced per year.  Holding one grid's four archetype arrays costs about
    # 5.6 MB, which is why grids are the outer loop and years the inner one.
    for grid_number, grid_id in enumerate(grid_ids, start=1):
        rows = site_rows[grid_id]
        endpoints = endpoint_pue(args.weather_root / f"{grid_id}.npz", pue_module)
        for year in years:
            start, stop = bounds[year]
            pue = interval_pue(endpoints, start, stop)
            hours = stop - start
            times = calendar(year)
            type_pue = np.column_stack([pue[TYPE_LINKED_ARCHETYPE[n]] for n in FACILITY_TYPES])

            # Vectorised over sites.  For one site the three quantities are
            #   it      = sum_k L[:,k] w[k]                  = L @ w
            #   fac     = sum_k L[:,k] P[:,k] w[k]           = (L*P) @ w
            #   climate = sum_k L[:,k](P[:,k]-b[k]) w[k]     = (L*(P-b)) @ w
            # so a whole timezone group collapses to three (hours,5)@(5,n)
            # matrix products instead of a Python loop over sites.
            climate_pue = type_pue - type_baseline[None, :]
            payload: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
            for scope in DEMAND_SCOPES:
                it = np.empty((len(rows), hours))
                fac = np.empty((len(rows), hours))
                climate = np.empty((len(rows), hours))
                masses = site_weights if scope == "site_marginal" else global_mass
                for timezone, positions in grid_timezone_groups[grid_id].items():
                    localized = templates[year][timezone].astype(np.float64)
                    block = masses[rows[positions]].T          # (5, n_sites)
                    it_block = localized @ block
                    fac_block = (localized * type_pue) @ block
                    climate_block = (localized * climate_pue) @ block
                    if scope == "site_marginal":
                        # Anchor each site's annual mean IT to its own P50.
                        scale = site_anchor[rows[positions]] / it_block.mean(axis=0)
                        it_block = it_block * scale
                        fac_block = fac_block * scale
                        climate_block = climate_block * scale
                    it[positions] = it_block.T
                    fac[positions] = fac_block.T
                    climate[positions] = climate_block.T
                payload[scope] = (it, fac, climate)

                ratio = fac / it
                diagnostics["pue_min"] = min(diagnostics["pue_min"], float(ratio.min()))
                diagnostics["pue_max"] = max(diagnostics["pue_max"], float(ratio.max()))
                for array in (it, fac, climate):
                    diagnostics["nonfinite"] += int((~np.isfinite(array)).sum())
                diagnostics["negative"] += int((it < 0).sum()) + int((fac < 0).sum())
                if scope == "site_marginal":
                    drift = np.abs(it.mean(axis=1) - site_anchor[rows]).max()
                    diagnostics["site_anchor_max_abs_mw"] = max(
                        diagnostics["site_anchor_max_abs_mw"], float(drift)
                    )

                # Annual roll-up carries all four PUE scenarios; only the central
                # one is written hourly, exactly as in the three-year run.
                for scenario in PUE_SCENARIOS:
                    if scenario == "type_linked_proxy":
                        facility_scenario = fac
                        climate_scenario = climate
                    else:
                        archetype = SCENARIO_ARCHETYPE[scenario]
                        arr = pue[archetype]
                        facility_scenario = it * arr[None, :]
                        base = pue_module.PUE_ARCHETYPES[archetype]
                        climate_scenario = it * (
                            arr[None, :] - (1.0 + base.fixed_overhead + base.flow_base)
                        )
                    annual_records.append({
                        "weather_year": np.full(len(rows), year, dtype=np.int16),
                        "logical_site_id": site_id[rows],
                        "era5_grid_id": np.full(len(rows), grid_id),
                        "demand_scope": np.full(len(rows), scope),
                        "pue_scenario": np.full(len(rows), scenario),
                        "hours": np.full(len(rows), hours, dtype=np.int32),
                        "mean_it_mw": it.mean(axis=1),
                        "it_energy_mwh": it.sum(axis=1),
                        "mean_facility_load_mw": facility_scenario.mean(axis=1),
                        "peak_facility_load_mw": facility_scenario.max(axis=1),
                        "facility_energy_mwh": facility_scenario.sum(axis=1),
                        "climate_variable_overhead_mwh": climate_scenario.sum(axis=1),
                    })

            table = pa.Table.from_pydict({
                "weather_year": np.full(len(rows) * hours, year, dtype=np.int16),
                "logical_site_id": np.repeat(site_id[rows], hours),
                "era5_grid_id": np.repeat(np.asarray([grid_id]), len(rows) * hours),
                "time_utc": np.tile(times.to_numpy(), len(rows)),
                **{
                    f"{scope}_{field}": payload[scope][position].reshape(-1).astype(np.float32)
                    for scope in DEMAND_SCOPES
                    for position, field in enumerate(
                        ("it_load_mw", "facility_load_mw", "climate_variable_overhead_load_mw")
                    )
                },
            }, schema=schema)
            writer = WRITERS.get(year)
            if writer is None:
                path = args.output_root / f"facility_hourly_demand_replay_{year}.parquet"
                if path.exists() and not args.force:
                    raise FileExistsError(f"Output exists; use --force: {path}")
                writer = pq.ParquetWriter(
                    path, schema, compression="zstd",
                    compression_level=args.compression_level,
                    use_dictionary=True, write_statistics=True, data_page_version="2.0",
                )
                WRITERS[year] = writer
            writer.write_table(table)
            row_counts[year] = row_counts.get(year, 0) + len(rows) * hours

        if grid_number % 100 == 0 or grid_number == len(grid_ids):
            elapsed = time.perf_counter() - started
            rate = grid_number / elapsed
            print(f"  {grid_number}/{len(grid_ids)} grids | {elapsed / 60:.1f} min | "
                  f"ETA {(len(grid_ids) - grid_number) / rate / 60:.1f} min", flush=True)

    for writer in WRITERS.values():
        writer.close()

    annual = pd.concat([pd.DataFrame(block) for block in annual_records], ignore_index=True)
    annual.to_parquet(args.summary_root / "facility_annual_demand_replay_1995_2014.parquet",
                      index=False)
    grouped = (
        annual.groupby(["weather_year", "demand_scope", "pue_scenario"], observed=True)
        .agg(hours=("hours", "first"),
             sites=("logical_site_id", "nunique"),
             it_energy_twh=("it_energy_mwh", lambda s: s.sum() / 1e6),
             facility_energy_twh=("facility_energy_mwh", lambda s: s.sum() / 1e6),
             climate_variable_overhead_twh=("climate_variable_overhead_mwh",
                                            lambda s: s.sum() / 1e6))
        .reset_index()
    )
    grouped["mean_it_gw"] = grouped["it_energy_twh"] * 1e3 / grouped["hours"]
    grouped["mean_facility_load_gw"] = grouped["facility_energy_twh"] * 1e3 / grouped["hours"]
    grouped["load_weighted_pue"] = grouped["facility_energy_twh"] / grouped["it_energy_twh"]
    grouped.to_csv(args.summary_root / "global_annual_demand_replay_1995_2014.csv", index=False)

    payload = {
        "method_version": METHOD_VERSION,
        "result_role": RESULT_ROLE,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "weather_root": str(args.weather_root),
        "output_root": str(args.output_root),
        "weather_years": list(years),
        "sites": int(selected.sum()),
        "grids": len(grid_ids),
        "demand_scenario": args.scenario,
        "target_mean_it_mw": target_mean_mw,
        "annual_scale_factor": annual_scale_factor,
        "demand_scale_fixed_across_weather_years": True,
        "template_construction": "tau=(month, weekday, hour) median of 2023-2025 published profiles",
        "row_counts": {str(k): int(v) for k, v in sorted(row_counts.items())},
        "total_rows": int(sum(row_counts.values())),
        "diagnostics": diagnostics,
        "pue_validation_status": PUE_VALIDATION_STATUS,
        "is_historical_actual_demand": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.summary_root / "demand_replay_run_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({k: payload[k] for k in
                      ("total_rows", "annual_scale_factor", "diagnostics", "elapsed_seconds")},
                     indent=2, ensure_ascii=False))


WRITERS: dict[int, pq.ParquetWriter] = {}


if __name__ == "__main__":
    main()
