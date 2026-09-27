#!/usr/bin/env python3
"""Aggregate fixed-2025 capacity factors into 1995--2014 hourly supply.

The formal output is modeled physical generation from a fixed 2025 on-grid
PV/wind fleet under twenty historical ERA5 weather years.  It is not observed
historical generation and it does not establish PPA ownership or electrical
deliverability to a data centre.

The script never materializes the roughly 3.77 billion grid-hour records.
Instead it reads one compact grid resource file at a time, accumulates country
and global hourly arrays, and writes year-partitioned results plus annual grid,
country and global metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
BASE_ROOT = (
    ROOT
    / "data"
    / "24x7_cfe"
    / "processed"
    / "wind_solar_supply_20y_fixed_2025"
)
DEFAULT_ALLOCATION = BASE_ROOT / "fixed_2025_supply_grid_allocation.parquet"
DEFAULT_CATALOG = BASE_ROOT / "supply_asset_grid_catalog_2025.parquet"
DEFAULT_UNALLOCATED = BASE_ROOT / "unallocated_2025_capacity.parquet"
DEFAULT_RESOURCE_ROOT = BASE_ROOT / "resource_cf_1995_2014"
DEFAULT_OUTPUT_ROOT = BASE_ROOT / "replay_results"
METHOD_VERSION = "fixed_2025_on_grid_supply_weather_replay_1995_2014_v1"
RESOURCE_METHOD_VERSION = "cds_era5_arco_geo_fixed_supply_resource_1995_2014_v1"
RESULT_ROLE = "supply_weather_replay_fixed_capacity"
CAPACITY_REFERENCE_YEAR = 2025
START = pd.Timestamp("1995-01-01 00:00:00", tz="UTC")
END = pd.Timestamp("2015-01-01 00:00:00", tz="UTC")
TIME_INDEX = pd.date_range(START, END, freq="h", inclusive="left")
EXPECTED_INTERVAL_COUNT = 175_320
YEARS = tuple(range(1995, 2015))
LOW_OUTPUT_THRESHOLD = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allocation", type=Path, default=DEFAULT_ALLOCATION)
    parser.add_argument("--grid-catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--unallocated", type=Path, default=DEFAULT_UNALLOCATED)
    parser.add_argument("--resource-root", type=Path, default=DEFAULT_RESOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_scalar(value: Any) -> Any:
    """Convert NumPy/pandas scalars losslessly; reject anything else.

    A stray ``np.bool_`` or ``np.float64`` must not sink a validation record
    written after hours of aggregation, but an unexpected object still has to
    fail loudly rather than be stringified into the evidence file.
    """

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Refusing to serialise {type(value).__name__} into JSON evidence")


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
            default=json_scalar,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def expected_resource_names(has_solar: bool, has_wind: bool) -> set[str]:
    names: set[str] = set()
    if has_solar:
        names.add("pv_cf")
    if has_wind:
        names.add("wind_cf")
    return names


def validate_inputs(
    allocation: pd.DataFrame,
    catalog: pd.DataFrame,
    resource_root: Path,
) -> dict[str, Any]:
    required_allocation = {
        "supply_era5_grid_id",
        "country_area_code",
        "country_area",
        "technology",
        "allocated_on_grid_capacity_mw",
        "allocated_all_grid_capacity_mw",
        "allocated_on_grid_capacity_known_start_sensitivity_mw",
        "capacity_reference_year",
        "capacity_fixed_across_weather_years",
        "result_role",
    }
    required_catalog = {
        "supply_era5_grid_id",
        "has_solar_pv",
        "has_wind",
        "era5_latitude",
        "era5_longitude",
    }
    missing_allocation = required_allocation - set(allocation)
    missing_catalog = required_catalog - set(catalog)
    if missing_allocation or missing_catalog:
        raise ValueError(
            f"Input columns missing: allocation={sorted(missing_allocation)}, "
            f"catalog={sorted(missing_catalog)}"
        )
    if allocation["capacity_reference_year"].ne(CAPACITY_REFERENCE_YEAR).any():
        raise ValueError("Allocation does not use fixed 2025 capacity")
    if not allocation["capacity_fixed_across_weather_years"].fillna(False).all():
        raise ValueError("Allocation is not marked fixed across weather years")
    if allocation["result_role"].ne(RESULT_ROLE).any():
        raise ValueError("Allocation result role mismatch")
    if catalog["supply_era5_grid_id"].duplicated().any():
        raise ValueError("Grid catalog contains duplicate identifiers")
    if set(allocation["supply_era5_grid_id"]) - set(catalog["supply_era5_grid_id"]):
        raise ValueError("Allocation contains grids absent from the catalog")

    failures: list[dict[str, str]] = []
    total_bytes = 0
    for row in catalog.itertuples(index=False):
        grid_id = str(row.supply_era5_grid_id)
        resource_path = resource_root / f"{grid_id}.npz"
        metadata_path = resource_root / f"{grid_id}.metadata.json"
        if not resource_path.exists() or not metadata_path.exists():
            failures.append({"grid_id": grid_id, "reason": "resource_or_metadata_missing"})
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (
                metadata.get("method_version") != RESOURCE_METHOD_VERSION
                or int(metadata.get("interval_count", -1)) != EXPECTED_INTERVAL_COUNT
                or metadata.get("output_sha256") != sha256_file(resource_path)
            ):
                raise ValueError("metadata provenance mismatch")
            with np.load(resource_path) as archive:
                expected = expected_resource_names(bool(row.has_solar_pv), bool(row.has_wind))
                if set(archive.files) != expected:
                    raise ValueError("resource arrays do not match catalog technology flags")
                for name in archive.files:
                    values = archive[name]
                    if values.shape != (EXPECTED_INTERVAL_COUNT,) or not np.isfinite(values).all():
                        raise ValueError(f"{name} has invalid length or non-finite values")
                    upper = 0.97 if name == "wind_cf" else 1.0
                    if (values < 0.0).any() or (values > upper + 1e-7).any():
                        raise ValueError(f"{name} is outside physical model bounds")
            total_bytes += resource_path.stat().st_size
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            failures.append({"grid_id": grid_id, "reason": f"{type(exc).__name__}: {exc}"})
    if failures:
        raise RuntimeError(
            f"Resource inventory is incomplete or invalid for {len(failures):,} grids. "
            f"Examples: {failures[:20]}"
        )
    return {
        "catalog_grid_count": int(len(catalog)),
        "validated_resource_grid_count": int(len(catalog)),
        "resource_bytes": total_bytes,
        "resource_coverage_pass": True,
    }


def year_slices() -> dict[int, slice]:
    result: dict[int, slice] = {}
    for year in YEARS:
        start = int((pd.Timestamp(f"{year}-01-01", tz="UTC") - START) / pd.Timedelta(hours=1))
        stop = int((pd.Timestamp(f"{year + 1}-01-01", tz="UTC") - START) / pd.Timedelta(hours=1))
        result[year] = slice(start, stop)
    return result


def load_resources(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {name: np.asarray(archive[name], dtype=np.float64) for name in archive.files}


def annual_cf_rows(
    grid_id: str,
    resources: dict[str, np.ndarray],
    slices: dict[int, slice],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for year, interval_slice in slices.items():
        row: dict[str, Any] = {
            "supply_era5_grid_id": grid_id,
            "weather_year": year,
            "hour_count": interval_slice.stop - interval_slice.start,
        }
        if "pv_cf" in resources:
            values = resources["pv_cf"][interval_slice]
            row.update(
                {
                    "pv_capacity_factor": float(values.mean()),
                    "pv_full_load_hours": float(values.sum()),
                    "pv_low_output_hours_cf_below_0p05": int(
                        (values < LOW_OUTPUT_THRESHOLD).sum()
                    ),
                }
            )
        else:
            row.update(
                {
                    "pv_capacity_factor": np.nan,
                    "pv_full_load_hours": np.nan,
                    "pv_low_output_hours_cf_below_0p05": pd.NA,
                }
            )
        if "wind_cf" in resources:
            values = resources["wind_cf"][interval_slice]
            row.update(
                {
                    "wind_capacity_factor": float(values.mean()),
                    "wind_full_load_hours": float(values.sum()),
                    "wind_low_output_hours_cf_below_0p05": int(
                        (values < LOW_OUTPUT_THRESHOLD).sum()
                    ),
                }
            )
        else:
            row.update(
                {
                    "wind_capacity_factor": np.nan,
                    "wind_full_load_hours": np.nan,
                    "wind_low_output_hours_cf_below_0p05": pd.NA,
                }
            )
        rows.append(row)
    return rows


def accumulate_supply(
    allocation: pd.DataFrame,
    catalog: pd.DataFrame,
    resource_root: Path,
) -> tuple[
    list[str],
    dict[str, str],
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    pd.DataFrame,
]:
    countries = sorted(allocation["country_area_code"].dropna().astype(str).unique())
    country_index = {code: index for index, code in enumerate(countries)}
    country_names = (
        allocation[["country_area_code", "country_area"]]
        .drop_duplicates("country_area_code")
        .set_index("country_area_code")["country_area"]
        .astype(str)
        .to_dict()
    )
    country_solar = np.zeros((len(countries), EXPECTED_INTERVAL_COUNT), dtype=np.float64)
    country_wind = np.zeros_like(country_solar)
    sensitivity = {
        "all_grid_solar": np.zeros(EXPECTED_INTERVAL_COUNT, dtype=np.float64),
        "all_grid_wind": np.zeros(EXPECTED_INTERVAL_COUNT, dtype=np.float64),
        "known_start_on_grid_solar": np.zeros(EXPECTED_INTERVAL_COUNT, dtype=np.float64),
        "known_start_on_grid_wind": np.zeros(EXPECTED_INTERVAL_COUNT, dtype=np.float64),
    }
    slices = year_slices()
    cf_rows: list[dict[str, Any]] = []
    total_grids = len(catalog)
    grouped_allocation = {
        grid_id: group
        for grid_id, group in allocation.groupby("supply_era5_grid_id", sort=False)
    }
    started = time.perf_counter()
    for position, catalog_row in enumerate(catalog.itertuples(index=False), start=1):
        grid_id = str(catalog_row.supply_era5_grid_id)
        resources = load_resources(resource_root / f"{grid_id}.npz")
        cf_rows.extend(annual_cf_rows(grid_id, resources, slices))
        weights = grouped_allocation.get(grid_id)
        if weights is None:
            raise ValueError(f"Catalog grid lacks allocation rows: {grid_id}")
        for weight in weights.itertuples(index=False):
            technology = str(weight.technology)
            resource_name = "pv_cf" if technology == "solar_pv" else "wind_cf"
            values = resources[resource_name]
            country_position = country_index[str(weight.country_area_code)]
            primary_capacity = float(weight.allocated_on_grid_capacity_mw)
            all_grid_capacity = float(weight.allocated_all_grid_capacity_mw)
            known_capacity = weight.allocated_on_grid_capacity_known_start_sensitivity_mw
            known_capacity = 0.0 if pd.isna(known_capacity) else float(known_capacity)
            if technology == "solar_pv":
                country_solar[country_position] += values * primary_capacity
                sensitivity["all_grid_solar"] += values * all_grid_capacity
                sensitivity["known_start_on_grid_solar"] += values * known_capacity
            elif technology == "wind":
                country_wind[country_position] += values * primary_capacity
                sensitivity["all_grid_wind"] += values * all_grid_capacity
                sensitivity["known_start_on_grid_wind"] += values * known_capacity
            else:
                raise ValueError(f"Unexpected modeled technology: {technology}")
        if position == 1 or position % 250 == 0 or position == total_grids:
            elapsed = time.perf_counter() - started
            print(
                f"Aggregation progress {position:,}/{total_grids:,} grids "
                f"({elapsed / 60.0:.1f} min)"
            )
    annual_grid = pd.DataFrame(cf_rows).merge(
        catalog[
            [
                "supply_era5_grid_id",
                "era5_latitude",
                "era5_longitude",
                "has_solar_pv",
                "has_wind",
            ]
        ],
        on="supply_era5_grid_id",
        how="left",
        validate="many_to_one",
    )
    return countries, country_names, country_solar, country_wind, sensitivity, annual_grid


def capacities_by_country(
    allocation: pd.DataFrame, countries: list[str]
) -> pd.DataFrame:
    grouped = (
        allocation.groupby(["country_area_code", "technology"], as_index=False)[
            "allocated_on_grid_capacity_mw"
        ]
        .sum()
        .pivot(
            index="country_area_code",
            columns="technology",
            values="allocated_on_grid_capacity_mw",
        )
        .reindex(countries)
        .fillna(0.0)
    )
    grouped.columns.name = None
    for technology in ["solar_pv", "wind"]:
        if technology not in grouped:
            grouped[technology] = 0.0
    return grouped[["solar_pv", "wind"]]


def write_country_hourly(
    output_root: Path,
    countries: list[str],
    country_names: dict[str, str],
    solar: np.ndarray,
    wind: np.ndarray,
) -> list[Path]:
    paths: list[Path] = []
    slices = year_slices()
    hourly_root = output_root / "hourly_country_supply"
    for year, interval_slice in slices.items():
        year_root = hourly_root / f"weather_year={year}"
        year_root.mkdir(parents=True, exist_ok=True)
        path = year_root / "country_hourly_supply.parquet"
        temporary = path.with_suffix(path.suffix + ".tmp")
        if temporary.exists():
            temporary.unlink()
        writer: pq.ParquetWriter | None = None
        times = TIME_INDEX[interval_slice]
        try:
            for country_position, country_code in enumerate(countries):
                solar_values = solar[country_position, interval_slice].astype(np.float32)
                wind_values = wind[country_position, interval_slice].astype(np.float32)
                total_values = solar_values + wind_values
                frame = pd.DataFrame(
                    {
                        "time_utc": times,
                        "weather_year": np.int16(year),
                        "country_area_code": country_code,
                        "country_area": country_names[country_code],
                        "solar_pv_generation_power_mw": solar_values,
                        "wind_generation_power_mw": wind_values,
                        "wind_solar_generation_power_mw": total_values,
                        "solar_pv_generation_energy_mwh": solar_values,
                        "wind_generation_energy_mwh": wind_values,
                        "wind_solar_generation_energy_mwh": total_values,
                        "interval_hours": np.float32(1.0),
                        "capacity_reference_year": np.int16(CAPACITY_REFERENCE_YEAR),
                        "capacity_fixed": True,
                        "result_role": RESULT_ROLE,
                        "is_historical_actual_generation": False,
                    }
                )
                table = pa.Table.from_pandas(frame, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(
                        temporary, table.schema, compression="zstd", use_dictionary=True
                    )
                writer.write_table(table)
        finally:
            if writer is not None:
                writer.close()
        temporary.replace(path)
        paths.append(path)
        print(f"Wrote country-hourly partition {year}: {path}")
    return paths


def global_hourly_frame(
    country_solar: np.ndarray,
    country_wind: np.ndarray,
    sensitivity: dict[str, np.ndarray],
) -> pd.DataFrame:
    primary_solar = country_solar.sum(axis=0)
    primary_wind = country_wind.sum(axis=0)
    frame = pd.DataFrame(
        {
            "time_utc": TIME_INDEX,
            "weather_year": TIME_INDEX.year.astype(np.int16),
            "solar_pv_generation_power_mw": primary_solar,
            "wind_generation_power_mw": primary_wind,
            "wind_solar_generation_power_mw": primary_solar + primary_wind,
            "solar_pv_generation_energy_mwh": primary_solar,
            "wind_generation_energy_mwh": primary_wind,
            "wind_solar_generation_energy_mwh": primary_solar + primary_wind,
            "all_grid_solar_pv_generation_power_mw": sensitivity["all_grid_solar"],
            "all_grid_wind_generation_power_mw": sensitivity["all_grid_wind"],
            "known_start_sensitivity_solar_pv_generation_power_mw": sensitivity[
                "known_start_on_grid_solar"
            ],
            "known_start_sensitivity_wind_generation_power_mw": sensitivity[
                "known_start_on_grid_wind"
            ],
            "interval_hours": 1.0,
            "capacity_reference_year": CAPACITY_REFERENCE_YEAR,
            "capacity_fixed": True,
            "result_role": RESULT_ROLE,
            "is_historical_actual_generation": False,
        }
    )
    return frame


def annual_country_frame(
    countries: list[str],
    country_names: dict[str, str],
    solar: np.ndarray,
    wind: np.ndarray,
    capacities: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for year, interval_slice in year_slices().items():
        hours = interval_slice.stop - interval_slice.start
        for position, code in enumerate(countries):
            solar_energy = float(solar[position, interval_slice].sum())
            wind_energy = float(wind[position, interval_slice].sum())
            solar_capacity = float(capacities.loc[code, "solar_pv"])
            wind_capacity = float(capacities.loc[code, "wind"])
            rows.append(
                {
                    "weather_year": year,
                    "country_area_code": code,
                    "country_area": country_names[code],
                    "hour_count": hours,
                    "solar_pv_installed_capacity_mw": solar_capacity,
                    "wind_installed_capacity_mw": wind_capacity,
                    "solar_pv_generation_energy_mwh": solar_energy,
                    "wind_generation_energy_mwh": wind_energy,
                    "wind_solar_generation_energy_mwh": solar_energy + wind_energy,
                    "solar_pv_mean_generation_power_mw": solar_energy / hours,
                    "wind_mean_generation_power_mw": wind_energy / hours,
                    "solar_pv_capacity_factor": solar_energy / (solar_capacity * hours)
                    if solar_capacity > 0
                    else np.nan,
                    "wind_capacity_factor": wind_energy / (wind_capacity * hours)
                    if wind_capacity > 0
                    else np.nan,
                    "capacity_reference_year": CAPACITY_REFERENCE_YEAR,
                    "capacity_fixed": True,
                    "result_role": RESULT_ROLE,
                    "is_historical_actual_generation": False,
                }
            )
    return pd.DataFrame(rows)


def annual_global_frame(
    global_hourly: pd.DataFrame,
    allocation: pd.DataFrame,
) -> pd.DataFrame:
    solar_capacity = float(
        allocation.loc[
            allocation["technology"].eq("solar_pv"), "allocated_on_grid_capacity_mw"
        ].sum()
    )
    wind_capacity = float(
        allocation.loc[
            allocation["technology"].eq("wind"), "allocated_on_grid_capacity_mw"
        ].sum()
    )
    rows: list[dict[str, Any]] = []
    for year, group in global_hourly.groupby("weather_year", sort=True):
        hours = len(group)
        solar_energy = float(group["solar_pv_generation_energy_mwh"].sum())
        wind_energy = float(group["wind_generation_energy_mwh"].sum())
        all_grid_solar_energy = float(group["all_grid_solar_pv_generation_power_mw"].sum())
        all_grid_wind_energy = float(group["all_grid_wind_generation_power_mw"].sum())
        known_solar_energy = float(
            group["known_start_sensitivity_solar_pv_generation_power_mw"].sum()
        )
        known_wind_energy = float(
            group["known_start_sensitivity_wind_generation_power_mw"].sum()
        )
        rows.append(
            {
                "weather_year": int(year),
                "hour_count": hours,
                "solar_pv_installed_capacity_mw": solar_capacity,
                "wind_installed_capacity_mw": wind_capacity,
                "solar_pv_generation_twh": solar_energy / 1e6,
                "wind_generation_twh": wind_energy / 1e6,
                "wind_solar_generation_twh": (solar_energy + wind_energy) / 1e6,
                "solar_pv_mean_generation_power_gw": solar_energy / hours / 1e3,
                "wind_mean_generation_power_gw": wind_energy / hours / 1e3,
                "solar_pv_capacity_factor": solar_energy / (solar_capacity * hours),
                "wind_capacity_factor": wind_energy / (wind_capacity * hours),
                "all_grid_solar_pv_generation_twh": all_grid_solar_energy / 1e6,
                "all_grid_wind_generation_twh": all_grid_wind_energy / 1e6,
                "known_start_sensitivity_solar_pv_generation_twh": known_solar_energy / 1e6,
                "known_start_sensitivity_wind_generation_twh": known_wind_energy / 1e6,
                "capacity_reference_year": CAPACITY_REFERENCE_YEAR,
                "capacity_fixed": True,
                "result_role": RESULT_ROLE,
                "is_historical_actual_generation": False,
            }
        )
    return pd.DataFrame(rows)


def low_output_events(
    global_hourly: pd.DataFrame,
    allocation: pd.DataFrame,
) -> pd.DataFrame:
    capacities = {
        "solar_pv": float(
            allocation.loc[
                allocation["technology"].eq("solar_pv"), "allocated_on_grid_capacity_mw"
            ].sum()
        ),
        "wind": float(
            allocation.loc[
                allocation["technology"].eq("wind"), "allocated_on_grid_capacity_mw"
            ].sum()
        ),
    }
    series = {
        "solar_pv": global_hourly["solar_pv_generation_power_mw"].to_numpy(dtype=float)
        / capacities["solar_pv"],
        "wind": global_hourly["wind_generation_power_mw"].to_numpy(dtype=float)
        / capacities["wind"],
        "wind_solar": global_hourly["wind_solar_generation_power_mw"].to_numpy(dtype=float)
        / (capacities["solar_pv"] + capacities["wind"]),
    }
    events: list[dict[str, Any]] = []
    times = pd.DatetimeIndex(global_hourly["time_utc"])
    for technology, cf in series.items():
        mask = cf < LOW_OUTPUT_THRESHOLD
        padded = np.concatenate(([False], mask, [False])).astype(np.int8)
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        stops = np.flatnonzero(changes == -1)
        for sequence, (start, stop) in enumerate(zip(starts, stops, strict=True), start=1):
            values = cf[start:stop]
            events.append(
                {
                    "technology": technology,
                    "event_sequence": sequence,
                    "start_time_utc": times[start],
                    "end_time_exclusive_utc": times[stop - 1] + pd.Timedelta(hours=1),
                    "duration_hours": int(stop - start),
                    "minimum_capacity_factor": float(values.min()),
                    "mean_capacity_factor": float(values.mean()),
                    "threshold_capacity_factor": LOW_OUTPUT_THRESHOLD,
                    "capacity_reference_year": CAPACITY_REFERENCE_YEAR,
                    "result_role": RESULT_ROLE,
                }
            )
    return pd.DataFrame(events)


def numeric_json(value: Any) -> int | float | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    rounded = round(number, 10)
    return int(rounded) if rounded.is_integer() else rounded


def build_validation(
    allocation: pd.DataFrame,
    catalog: pd.DataFrame,
    input_validation: dict[str, Any],
    global_hourly: pd.DataFrame,
    annual_country: pd.DataFrame,
    annual_global: pd.DataFrame,
    output_paths: dict[str, Path],
    country_hourly_paths: list[Path],
    input_paths: dict[str, Path],
) -> dict[str, Any]:
    country_annual_sum = annual_country.groupby("weather_year", as_index=False)[
        ["solar_pv_generation_energy_mwh", "wind_generation_energy_mwh"]
    ].sum()
    global_energy = annual_global.assign(
        solar_pv_generation_energy_mwh=lambda frame: frame["solar_pv_generation_twh"] * 1e6,
        wind_generation_energy_mwh=lambda frame: frame["wind_generation_twh"] * 1e6,
    )
    comparison = country_annual_sum.merge(
        global_energy[
            [
                "weather_year",
                "solar_pv_generation_energy_mwh",
                "wind_generation_energy_mwh",
            ]
        ],
        on="weather_year",
        suffixes=("_country", "_global"),
        validate="one_to_one",
    )
    differences = {}
    for technology in ["solar_pv", "wind"]:
        country_values = comparison[f"{technology}_generation_energy_mwh_country"]
        global_values = comparison[f"{technology}_generation_energy_mwh_global"]
        differences[technology] = float(np.max(np.abs(country_values - global_values)))
    solar_capacity = float(
        allocation.loc[
            allocation["technology"].eq("solar_pv"), "allocated_on_grid_capacity_mw"
        ].sum()
    )
    wind_capacity = float(
        allocation.loc[
            allocation["technology"].eq("wind"), "allocated_on_grid_capacity_mw"
        ].sum()
    )
    hard_checks = {
        "resource_coverage_complete": input_validation["resource_coverage_pass"],
        "global_hour_count_exact": len(global_hourly) == EXPECTED_INTERVAL_COUNT,
        "weather_years_exact": set(global_hourly["weather_year"]) == set(YEARS),
        "year_hour_counts_exact": all(
            len(group) == (8784 if int(year) in {1996, 2000, 2004, 2008, 2012} else 8760)
            for year, group in global_hourly.groupby("weather_year")
        ),
        "power_nonnegative": bool(
            global_hourly[
                [
                    "solar_pv_generation_power_mw",
                    "wind_generation_power_mw",
                    "wind_solar_generation_power_mw",
                ]
            ]
            .ge(-1e-7)
            .all()
            .all()
        ),
        "solar_power_within_capacity": bool(
            global_hourly["solar_pv_generation_power_mw"].max() <= solar_capacity + 1e-5
        ),
        "wind_power_within_capacity": bool(
            global_hourly["wind_generation_power_mw"].max() <= wind_capacity * 0.97 + 1e-5
        ),
        "hourly_power_energy_identity": bool(
            np.allclose(
                global_hourly["wind_solar_generation_power_mw"],
                global_hourly["wind_solar_generation_energy_mwh"],
                rtol=0.0,
                atol=1e-8,
            )
        ),
        # The tolerance is a NumPy scalar, so the comparison must be cast back
        # to a builtin bool for the validation record to serialise.
        "country_global_solar_energy_conserved": bool(
            differences["solar_pv"]
            <= max(1e-4, annual_global["solar_pv_generation_twh"].max() * 1e6 * 1e-10)
        ),
        "country_global_wind_energy_conserved": bool(
            differences["wind"]
            <= max(1e-4, annual_global["wind_generation_twh"].max() * 1e6 * 1e-10)
        ),
        "capacity_reference_fixed_2025": bool(
            annual_global["capacity_reference_year"].eq(CAPACITY_REFERENCE_YEAR).all()
            and annual_global["capacity_fixed"].all()
        ),
        "result_not_labeled_actual": bool(
            ~annual_global["is_historical_actual_generation"].any()
        ),
    }
    files = {**output_paths}
    for path in country_hourly_paths:
        files[f"country_hourly_{path.parent.name.split('=', 1)[1]}"] = path

    def evidence(path: Path) -> dict[str, Any]:
        # The multi-decade capacity-factor archive is routinely kept off the
        # repository tree, so evidence must record such paths absolutely
        # rather than fail.
        try:
            location = {"relative_path": str(path.relative_to(ROOT))}
        except ValueError:
            location = {"absolute_path": str(path), "outside_repository_root": True}
        return {
            **location,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    output_evidence = {name: evidence(path) for name, path in files.items()}
    input_evidence = {name: evidence(path) for name, path in input_paths.items()}
    return {
        "schema_version": "1.0",
        "method_version": METHOD_VERSION,
        "completed_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "result_role": RESULT_ROLE,
        "capacity_reference_year": CAPACITY_REFERENCE_YEAR,
        "weather_years": list(YEARS),
        "is_historical_actual_generation": False,
        "data_center_demand_used": False,
        "input_validation": input_validation,
        "input_files": input_evidence,
        "output_files": output_evidence,
        "energy_aggregation_max_absolute_difference_mwh": {
            key: numeric_json(value) for key, value in differences.items()
        },
        "hard_checks": hard_checks,
        "all_hard_checks_pass": all(hard_checks.values()),
        "validation_status": "PASS_WITH_HOURLY_EXTERNAL_VALIDATION_GAP"
        if all(hard_checks.values())
        else "FAIL",
        "limitations": [
            "The replay holds 2025 installed capacity constant in every weather year and is not historical actual generation.",
            "Country capacity is official IRENA capacity; within-country spatial allocation uses EIA/GEM asset weights.",
            "Capacity without a valid spatial anchor is excluded and reported separately.",
            "Hourly external validation remains incomplete until ISO/RTO or ENTSO-E hourly observed generation is added.",
            "Grid-connected physical supply is not evidence of a PPA, transmission deliverability or data-centre entitlement.",
        ],
    }


def main() -> None:
    args = parse_args()
    for name in [
        "allocation",
        "grid_catalog",
        "unallocated",
        "resource_root",
        "output_root",
    ]:
        setattr(args, name, getattr(args, name).resolve())
    if len(TIME_INDEX) != EXPECTED_INTERVAL_COUNT:
        raise AssertionError("Registered replay time index is inconsistent")
    args.output_root.mkdir(parents=True, exist_ok=True)
    final_validation_path = args.output_root / "fixed_2025_replay_validation.json"
    if final_validation_path.exists() and not args.force:
        existing = json.loads(final_validation_path.read_text(encoding="utf-8"))
        if existing.get("all_hard_checks_pass"):
            print(
                f"Validated replay already exists: {final_validation_path}. "
                "Use --force to rebuild."
            )
            return

    allocation = pd.read_parquet(args.allocation)
    catalog = pd.read_parquet(args.grid_catalog)
    input_validation = validate_inputs(allocation, catalog, args.resource_root)
    (
        countries,
        country_names,
        country_solar,
        country_wind,
        sensitivity,
        annual_grid,
    ) = accumulate_supply(allocation, catalog, args.resource_root)
    capacities = capacities_by_country(allocation, countries)
    country_hourly_paths = write_country_hourly(
        args.output_root,
        countries,
        country_names,
        country_solar,
        country_wind,
    )
    global_hourly = global_hourly_frame(country_solar, country_wind, sensitivity)
    annual_country = annual_country_frame(
        countries,
        country_names,
        country_solar,
        country_wind,
        capacities,
    )
    annual_global = annual_global_frame(global_hourly, allocation)
    events = low_output_events(global_hourly, allocation)

    output_paths = {
        "global_hourly": args.output_root / "hourly_global_supply_1995_2014.parquet",
        "annual_grid": args.output_root / "annual_grid_capacity_factor_1995_2014.parquet",
        "annual_country": args.output_root / "annual_country_supply_1995_2014.parquet",
        "annual_global_parquet": args.output_root / "annual_global_supply_1995_2014.parquet",
        "annual_global_csv": args.output_root / "annual_global_supply_1995_2014.csv",
        "low_output_events": args.output_root / "low_output_event_catalog.parquet",
    }
    atomic_parquet(global_hourly, output_paths["global_hourly"])
    atomic_parquet(annual_grid, output_paths["annual_grid"])
    atomic_parquet(annual_country, output_paths["annual_country"])
    atomic_parquet(annual_global, output_paths["annual_global_parquet"])
    atomic_csv(annual_global, output_paths["annual_global_csv"])
    atomic_parquet(events, output_paths["low_output_events"])

    input_paths = {
        "allocation": args.allocation,
        "grid_catalog": args.grid_catalog,
        "unallocated": args.unallocated,
        "resource_manifest": args.resource_root / "resource_manifest.csv",
        "resource_summary": args.resource_root / "resource_summary.json",
    }
    missing_evidence = [name for name, path in input_paths.items() if not path.exists()]
    if missing_evidence:
        raise FileNotFoundError(f"Required input evidence missing: {missing_evidence}")
    validation = build_validation(
        allocation,
        catalog,
        input_validation,
        global_hourly,
        annual_country,
        annual_global,
        output_paths,
        country_hourly_paths,
        input_paths,
    )
    atomic_json(validation, final_validation_path)
    if not validation["all_hard_checks_pass"]:
        raise RuntimeError(f"Fixed-capacity replay validation failed: {final_validation_path}")
    print(
        json.dumps(
            {
                "status": validation["validation_status"],
                "weather_years": list(YEARS),
                "country_count": len(countries),
                "grid_count": len(catalog),
                "annual_global": str(output_paths["annual_global_csv"]),
                "validation": str(final_validation_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
