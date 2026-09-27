#!/usr/bin/env python3
"""Compute the 2023--2025 climate demand and unit-resource baseline.

The calculation uses the complete DCWatch inventory rather than a market
sample.  A common UTC ERA5 realization drives cooling demand, fixed-tilt PV,
and wind output.  Generic workload templates are interpreted in each site's
IANA civil time before they are aggregated on the UTC axis.

Absolute D1 scale comes from the hierarchical joint posterior: type-level
annual medians determine the composition and the published global hourly P50
determines each year's mean IT load.  Only one constant annual reconciliation
factor is used.  The old timezone-free global reference profile is not imposed
hour by hour, so the new UTC profile is allowed to emerge from geography.

PUE and renewable conversion functions are engineering scenarios, not
facility-specific calibrated predictions.  Capacity-factor columns are valid
only as unit-nameplate weather-response coefficients.

DEPRECATED DIAGNOSTIC: the legacy local wind/PV portfolio blocks below solve
for nameplate capacity by forcing annual modeled generation to equal data-
center demand.  Their capacities, generation, deficits, and CFE scores are not
formal supply results and must never be cited as installed or eligible supply.
Formal supply requires an external plant-year inventory of actual capacity.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
HIER = ROOT / "data" / "24x7_cfe" / "processed" / "demand_hierarchical"
COUPLING = ROOT / "data" / "24x7_cfe" / "processed" / "climate_coupling"
DEFAULT_MAPPING = COUPLING / "facility_to_era5_grid.csv"
DEFAULT_CATALOG = COUPLING / "era5_grid_catalog.csv"
DEFAULT_WEATHER = (
    ROOT
    / "data"
    / "24x7_cfe"
    / "raw"
    / "openmeteo"
    / "era5_2023_2025_dcwatch_grid"
)
DEFAULT_INTERVAL_BOUNDARY = (
    DEFAULT_WEATHER
    / "boundaries"
    / "era5_interval_boundary_2026-01-01_00.parquet"
)
DEFAULT_TYPE_ANNUAL = HIER / "sampled_type_annual_draws_2023.parquet"
DEFAULT_GLOBAL = HIER / "global_hourly_demand_summary_2023_2025.csv"
DEFAULT_OUTPUT = COUPLING / "historical_2023_2025"
DEFAULT_VALIDATION = (
    ROOT / "output" / "24x7_cfe_facility" / "climate_coupled_validation"
)
PUE_MODULE_PATH = ROOT / "research_pipeline" / "scripts" / "05_build_hourly_inputs.py"
SCENARIO = "transport__capacity_utilization_rho_+0.0"
YEARS = (2023, 2024, 2025)
FACILITY_TYPES = (
    "colocation_interconnection",
    "hyperscale_cloud",
    "ai_hpc",
    "enterprise_internal",
    "telecom_edge",
)
TYPE_LINKED_ARCHETYPE = {
    "colocation_interconnection": "colocation",
    "hyperscale_cloud": "advanced_hyperscale",
    "ai_hpc": "advanced_hyperscale",
    "enterprise_internal": "conventional",
    "telecom_edge": "conventional",
}
PUE_POLICIES = (
    "type_linked_proxy",
    "all_advanced",
    "all_liquid",
    "all_conventional",
)
WEATHER_COLUMNS = (
    "era5_grid_id",
    "time",
    "requested_latitude",
    "requested_longitude",
    "response_latitude",
    "response_longitude",
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "surface_pressure",
    "wind_speed_100m",
    "shortwave_radiation",
    "direct_normal_irradiance",
    "diffuse_radiation",
)
METHOD_VERSION = "full_dcwatch_localtime_demand_unit_resource_v3"
PUE_VALIDATION_STATUS = "engineering_scenario_only_not_globally_validated"
RESOURCE_VALIDATION_STATUS = (
    "unit_capacity_resource_proxy_only__actual_capacity_not_loaded__"
    "demand_scaled_portfolios_deprecated"
)
WIND_AVAILABILITY = 0.97
WIND_OTHER_LOSS_FRACTION = 0.0
WIND_SCENARIOS = ("iec_i", "iec_ii", "iec_iii", "cubic_conservative")
CENTRAL_WIND_SCENARIO = "iec_ii"
# NREL WIND Toolkit 100 m, 2 MW composite power curves (IEC I/II/III):
# King, Clifton & Hodge (2014), NREL/TP-5D00-61714, doi:10.2172/1159354.
_WIND_SPEED_NODES = np.arange(0.0, 26.0, 1.0)
WIND_POWER_CURVES = {
    "iec_i": {
        "speed_nodes_m_s": _WIND_SPEED_NODES,
        "power_fraction": np.asarray(
            [
                0.0,
                0.0,
                0.0,
                0.0043,
                0.0323,
                0.0771,
                0.1426,
                0.2329,
                0.3528,
                0.5024,
                0.6732,
                0.8287,
                0.9264,
                0.9774,
                0.9946,
                0.9990,
                0.9999,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            dtype=float,
        ),
        "cut_out_m_s": 26.0,
    },
    "iec_ii": {
        "speed_nodes_m_s": _WIND_SPEED_NODES,
        "power_fraction": np.asarray(
            [
                0.0,
                0.0,
                0.0,
                0.0052,
                0.0423,
                0.1031,
                0.1909,
                0.3127,
                0.4731,
                0.6693,
                0.8554,
                0.9641,
                0.9942,
                0.9994,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            dtype=float,
        ),
        "cut_out_m_s": 26.0,
    },
    "iec_iii": {
        "speed_nodes_m_s": np.arange(0.0, 23.0, 1.0),
        "power_fraction": np.asarray(
            [
                0.0,
                0.0,
                0.0,
                0.0054,
                0.0530,
                0.1351,
                0.2508,
                0.4033,
                0.5952,
                0.7849,
                0.9178,
                0.9796,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            dtype=float,
        ),
        "cut_out_m_s": 23.0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--grid-catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--weather-root", type=Path, default=DEFAULT_WEATHER)
    parser.add_argument(
        "--interval-boundary",
        type=Path,
        default=DEFAULT_INTERVAL_BOUNDARY,
        help="ERA5 2026-01-01 00 UTC right endpoint for the final 2025 interval.",
    )
    parser.add_argument("--type-annual", type=Path, default=DEFAULT_TYPE_ANNUAL)
    parser.add_argument("--type-summary-root", type=Path, default=HIER)
    parser.add_argument("--global-hourly", type=Path, default=DEFAULT_GLOBAL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validation-root", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--scenario", default=SCENARIO)
    parser.add_argument("--year", type=int, action="append", choices=YEARS)
    parser.add_argument("--limit-grids", type=int)
    parser.add_argument("--no-grid-hourly", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_pue_module() -> Any:
    name = "dcwatch_reduced_order_pue"
    spec = importlib.util.spec_from_file_location(name, PUE_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load PUE module {PUE_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_spatial_strata(mapping_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    weight_columns = [f"spatial_weight_mw_{name}" for name in FACILITY_TYPES]
    columns = [
        "logical_site_id",
        "era5_grid_id",
        "era5_latitude",
        "era5_longitude",
        "timezone_name",
        "mean_it_mw_p50",
        *weight_columns,
    ]
    mapping = pd.read_csv(mapping_path, usecols=columns, low_memory=False)
    if len(mapping) != 8_360 or mapping["logical_site_id"].nunique() != 8_360:
        raise ValueError("Expected the complete 8,360-site logical inventory")
    if mapping["timezone_name"].isna().any():
        raise ValueError("Every logical site must have an IANA time zone")
    if not np.isfinite(mapping[weight_columns].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite spatial type weights")

    long_parts: list[pd.DataFrame] = []
    for facility_type in FACILITY_TYPES:
        part = mapping[
            [
                "era5_grid_id",
                "era5_latitude",
                "era5_longitude",
                "timezone_name",
                f"spatial_weight_mw_{facility_type}",
            ]
        ].copy()
        part = part.rename(
            columns={f"spatial_weight_mw_{facility_type}": "raw_spatial_weight_mw"}
        )
        part["facility_type"] = facility_type
        long_parts.append(part)
    long = pd.concat(long_parts, ignore_index=True)
    strata = (
        long.groupby(
            [
                "era5_grid_id",
                "era5_latitude",
                "era5_longitude",
                "timezone_name",
                "facility_type",
            ],
            as_index=False,
            sort=True,
            observed=True,
        )["raw_spatial_weight_mw"]
        .sum()
    )
    type_totals = strata.groupby("facility_type", observed=True)[
        "raw_spatial_weight_mw"
    ].transform("sum")
    if (type_totals <= 0.0).any():
        raise ValueError("A facility type has zero spatial weight")
    strata["within_type_spatial_share"] = strata["raw_spatial_weight_mw"] / type_totals
    checks = strata.groupby("facility_type", observed=True)["within_type_spatial_share"].sum()
    if not np.allclose(checks.to_numpy(), 1.0, atol=1e-10):
        raise ValueError(f"Within-type spatial shares do not sum to one: {checks.to_dict()}")
    return mapping, strata


def load_type_medians(path: Path, scenario: str) -> pd.Series:
    frame = pd.read_parquet(path, columns=["sampled_facility_type", "scenario", "draw", "mean_it_gw"])
    frame = frame[frame["scenario"].eq(scenario)].copy()
    medians = frame.groupby("sampled_facility_type", observed=True)["mean_it_gw"].median() * 1_000.0
    if set(medians.index) != set(FACILITY_TYPES):
        raise ValueError(f"Incomplete type-level annual posterior: {medians.to_dict()}")
    return medians.reindex(FACILITY_TYPES)


def load_type_profiles(root: Path, year: int) -> pd.DataFrame:
    path = root / f"type_hourly_path_summary_{year}.csv"
    frame = pd.read_csv(
        path,
        usecols=["reference_year", "facility_type", "hour_of_year", "normalized_load_mean"],
    )
    frame = frame[frame["reference_year"].eq(year)]
    wide = frame.pivot(index="hour_of_year", columns="facility_type", values="normalized_load_mean")
    wide = wide.reindex(columns=FACILITY_TYPES).sort_index()
    expected = 8_784 if pd.Timestamp(f"{year}-12-31").is_leap_year else 8_760
    if wide.shape != (expected, len(FACILITY_TYPES)) or wide.isna().any().any():
        raise ValueError(f"Incomplete type profile for {year}: {wide.shape}")
    if not np.allclose(wide.mean(axis=0).to_numpy(), 1.0, atol=2e-6):
        raise ValueError(f"Type profiles for {year} are not unit mean")
    return wide


def localize_profiles(
    reference: pd.DataFrame, year: int, timezones: list[str]
) -> dict[str, np.ndarray]:
    """Return [hour, type] templates addressed in each site's civil time."""

    utc_index = pd.date_range(
        f"{year}-01-01", f"{year + 1}-01-01", freq="h", inclusive="left", tz="UTC"
    )
    values = reference.to_numpy(dtype=float)
    localized: dict[str, np.ndarray] = {}
    for timezone in timezones:
        local = utc_index.tz_convert(timezone)
        # Force month/day/hour onto the current template year.  This treats the
        # annual template cyclically at UTC-year boundaries and preserves DST.
        local_reference = pd.to_datetime(
            {
                "year": np.full(len(local), year),
                "month": local.month,
                "day": local.day,
                "hour": local.hour,
            },
            errors="raise",
        )
        left_index = np.asarray(
            ((local_reference - pd.Timestamp(f"{year}-01-01")) / pd.Timedelta(hours=1)).astype(int),
            dtype=int,
        )
        if (left_index < 0).any() or (left_index >= len(values)).any():
            raise ValueError(f"Local template mapping failed for {timezone}/{year}")
        fraction = np.asarray(local.minute, dtype=float) / 60.0
        right_index = (left_index + 1) % len(values)
        localized[timezone] = (
            values[left_index] * (1.0 - fraction[:, None])
            + values[right_index] * fraction[:, None]
        )
    return localized


def global_target_mean_mw(path: Path, scenario: str, year: int) -> float:
    frame = pd.read_csv(
        path, usecols=["calendar_year", "scenario", "it_load_gw_p50"]
    )
    subset = frame[frame["calendar_year"].eq(year) & frame["scenario"].eq(scenario)]
    expected = 8_784 if pd.Timestamp(f"{year}-12-31").is_leap_year else 8_760
    if len(subset) != expected:
        raise ValueError(f"Incomplete published global demand target for {year}")
    return float(subset["it_load_gw_p50"].mean() * 1_000.0)


def prepare_year_context(
    year: int,
    strata: pd.DataFrame,
    type_medians_mw: pd.Series,
    type_summary_root: Path,
    global_hourly_path: Path,
    scenario: str,
) -> dict[str, Any]:
    reference = load_type_profiles(type_summary_root, year)
    timezones = sorted(strata["timezone_name"].unique())
    localized = localize_profiles(reference, year, timezones)
    type_index = {name: index for index, name in enumerate(FACILITY_TYPES)}
    hours = len(reference)
    raw_global = np.zeros(hours, dtype=float)
    for row in strata.itertuples(index=False):
        facility_type = str(row.facility_type)
        mass_mw = (
            float(type_medians_mw[facility_type])
            * float(row.within_type_spatial_share)
        )
        raw_global += localized[str(row.timezone_name)][:, type_index[facility_type]] * mass_mw
    target_mean = global_target_mean_mw(global_hourly_path, scenario, year)
    scale = target_mean / float(raw_global.mean())
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"Invalid annual D1 reconciliation factor for {year}")
    return {
        "year": year,
        "hours": hours,
        "utc_index": pd.date_range(
            f"{year}-01-01", f"{year + 1}-01-01", freq="h", inclusive="left", tz="UTC"
        ),
        "localized": localized,
        "type_index": type_index,
        "type_medians_mw": type_medians_mw,
        "raw_global_it_mw": raw_global,
        "target_mean_it_mw": target_mean,
        "annual_scale_factor": scale,
        "expected_global_it_mw": raw_global * scale,
    }


def circular_longitude_difference(left: float, right: float) -> float:
    return abs(((left - right + 180.0) % 360.0) - 180.0)


def load_weather(
    path: Path,
    expected_grid_id: str,
    expected_latitude: float,
    expected_longitude: float,
) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=list(WEATHER_COLUMNS))
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    frame = frame.sort_values("time", kind="stable").reset_index(drop=True)
    if frame["era5_grid_id"].nunique() != 1 or str(frame["era5_grid_id"].iloc[0]) != expected_grid_id:
        raise ValueError(f"Weather grid identifier mismatch in {path}")
    requested_latitude = float(frame["requested_latitude"].iloc[0])
    requested_longitude = float(frame["requested_longitude"].iloc[0])
    if (
        not np.allclose(frame["requested_latitude"], requested_latitude, atol=1e-9)
        or not np.allclose(frame["requested_longitude"], requested_longitude, atol=1e-9)
        or abs(requested_latitude - expected_latitude) > 1e-6
        or circular_longitude_difference(requested_longitude, expected_longitude) > 1e-6
    ):
        raise ValueError(f"Requested weather coordinates do not match the catalog in {path}")
    response_latitude = float(frame["response_latitude"].iloc[0])
    response_longitude = float(frame["response_longitude"].iloc[0])
    if (
        abs(response_latitude - expected_latitude) > 0.251
        or circular_longitude_difference(response_longitude, expected_longitude) > 0.251
    ):
        raise ValueError(f"ERA5 response coordinates are too far from catalog grid in {path}")
    expected = pd.date_range(
        "2023-01-01", "2026-01-01", freq="h", inclusive="left", tz="UTC"
    )
    if not pd.DatetimeIndex(frame["time"]).equals(expected):
        raise ValueError(f"Weather time coverage is not complete 2023--2025 in {path}")
    numeric = frame[list(WEATHER_COLUMNS[2:])].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"Non-finite weather values in {path}")
    metadata_path = path.with_suffix(".metadata.json")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Weather metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    parameters = metadata.get("parameters", {})
    if (
        metadata.get("method_version") != "openmeteo_era5_dcwatch_grid_2023_2025_v1"
        or parameters.get("models") != "era5"
        or parameters.get("timezone") != "GMT"
        or parameters.get("wind_speed_unit") != "ms"
        or int(metadata.get("normalized_rows", -1)) != len(expected)
    ):
        raise ValueError(f"Weather metadata provenance check failed: {metadata_path}")
    return frame


def load_interval_boundary(path: Path, catalog: pd.DataFrame) -> pd.DataFrame:
    boundary = pd.read_parquet(path)
    required = set(WEATHER_COLUMNS)
    missing = sorted(required - set(boundary.columns))
    if missing:
        raise ValueError(f"Interval boundary lacks columns: {missing}")
    boundary["time"] = pd.to_datetime(boundary["time"], utc=True)
    expected_time = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
    if len(boundary) != len(catalog) or boundary["era5_grid_id"].nunique() != len(catalog):
        raise ValueError("Interval boundary does not cover every ERA5 grid")
    if not boundary["time"].eq(expected_time).all():
        raise ValueError("Interval boundary timestamp must be 2026-01-01 00:00 UTC")
    lookup = catalog.set_index("era5_grid_id")[["era5_latitude", "era5_longitude"]]
    joined = boundary.join(lookup, on="era5_grid_id", how="left", validate="many_to_one")
    if joined[["era5_latitude", "era5_longitude"]].isna().any().any():
        raise ValueError("Interval boundary contains unknown grid identifiers")
    latitude_error = (
        joined["requested_latitude"].astype(float) - joined["era5_latitude"].astype(float)
    ).abs()
    longitude_error = np.abs(
        (
            joined["requested_longitude"].astype(float)
            - joined["era5_longitude"].astype(float)
            + 180.0
        )
        % 360.0
        - 180.0
    )
    if (latitude_error > 1e-6).any() or (longitude_error > 1e-6).any():
        raise ValueError("Interval boundary coordinates do not match the grid catalog")
    numeric = boundary[list(WEATHER_COLUMNS[2:])].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Interval boundary contains non-finite values")
    return boundary.set_index("era5_grid_id", verify_integrity=True)


def interval_weather_for_year(
    weather: pd.DataFrame,
    boundary_row: pd.Series,
    year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return left/right endpoints for hourly intervals labelled by the left time."""

    boundary_frame = pd.DataFrame([boundary_row.to_dict()])
    boundary_frame.insert(0, "era5_grid_id", str(boundary_row.name))
    extended = pd.concat([weather, boundary_frame], ignore_index=True)
    extended = extended.sort_values("time", kind="stable").reset_index(drop=True)
    start = pd.Timestamp(f"{year}-01-01", tz="UTC")
    end = pd.Timestamp(f"{year + 1}-01-01", tz="UTC")
    left = extended[(extended["time"] >= start) & (extended["time"] < end)].reset_index(drop=True)
    right = extended[(extended["time"] > start) & (extended["time"] <= end)].reset_index(drop=True)
    if len(left) != len(right) or not np.array_equal(
        right["time"].to_numpy(),
        (left["time"] + pd.Timedelta(hours=1)).to_numpy(),
    ):
        raise ValueError(f"Cannot construct continuous hourly intervals for {year}")
    return left, right


def fixed_tilt_pv_cf(
    times: pd.DatetimeIndex,
    latitude: float,
    longitude: float,
    ghi: np.ndarray,
    dni: np.ndarray,
    dhi: np.ndarray,
    air_temperature_c: np.ndarray,
) -> np.ndarray:
    """Isotropic transposition plus a transparent PVWatts-style AC proxy."""

    day = times.dayofyear.to_numpy(dtype=float)
    utc_hour = times.hour.to_numpy(dtype=float) + times.minute.to_numpy(dtype=float) / 60.0
    b_angle = 2.0 * np.pi * (day - 81.0) / 364.0
    equation_of_time_min = (
        9.87 * np.sin(2.0 * b_angle) - 7.53 * np.cos(b_angle) - 1.5 * np.sin(b_angle)
    )
    solar_time = utc_hour + longitude / 15.0 + equation_of_time_min / 60.0
    hour_angle = np.radians(15.0 * (solar_time - 12.0))
    declination = np.radians(23.45 * np.sin(np.radians(360.0 * (284.0 + day) / 365.0)))
    latitude_rad = np.radians(latitude)
    north = (
        np.cos(latitude_rad) * np.sin(declination)
        - np.sin(latitude_rad) * np.cos(declination) * np.cos(hour_angle)
    )
    upward = (
        np.sin(latitude_rad) * np.sin(declination)
        + np.cos(latitude_rad) * np.cos(declination) * np.cos(hour_angle)
    )
    tilt_deg = float(np.clip(abs(latitude), 10.0, 60.0))
    tilt = np.radians(tilt_deg)
    equator_facing_north_component = -1.0 if latitude >= 0.0 else 1.0
    cosine_incidence = (
        upward * np.cos(tilt)
        + north * equator_facing_north_component * np.sin(tilt)
    )
    beam = np.maximum(dni, 0.0) * np.maximum(cosine_incidence, 0.0)
    sky = np.maximum(dhi, 0.0) * (1.0 + np.cos(tilt)) / 2.0
    ground = np.maximum(ghi, 0.0) * 0.20 * (1.0 - np.cos(tilt)) / 2.0
    poa = np.maximum(beam + sky + ground, 0.0)
    cell_temperature = air_temperature_c + (45.0 - 20.0) / 800.0 * poa
    temperature_factor = np.clip(1.0 - 0.004 * (cell_temperature - 25.0), 0.0, 1.25)
    dc_per_ac = 1.34 * 0.86 * poa / 1_000.0 * temperature_factor
    return np.clip(dc_per_ac, 0.0, 1.0)


def density_corrected_wind_cf_cubic(
    wind_speed_100m: np.ndarray,
    air_temperature_c: np.ndarray,
    surface_pressure_hpa: np.ndarray,
) -> np.ndarray:
    density = surface_pressure_hpa * 100.0 / (287.05 * (air_temperature_c + 273.15))
    equivalent_speed = np.maximum(wind_speed_100m, 0.0) * np.cbrt(density / 1.225)
    factor = np.zeros_like(equivalent_speed, dtype=float)
    ramp = (equivalent_speed >= 3.0) & (equivalent_speed < 12.0)
    factor[ramp] = (equivalent_speed[ramp] ** 3 - 3.0**3) / (12.0**3 - 3.0**3)
    rated = (equivalent_speed >= 12.0) & (equivalent_speed < 25.0)
    factor[rated] = 1.0
    return np.clip(
        factor * WIND_AVAILABILITY * (1.0 - WIND_OTHER_LOSS_FRACTION),
        0.0,
        WIND_AVAILABILITY,
    )


def density_corrected_wind_cf(
    wind_speed_100m: np.ndarray,
    air_temperature_c: np.ndarray,
    surface_pressure_hpa: np.ndarray,
    scenario: str,
) -> np.ndarray:
    """Apply density correction and a named NREL composite power curve."""

    if scenario == "cubic_conservative":
        return density_corrected_wind_cf_cubic(
            wind_speed_100m, air_temperature_c, surface_pressure_hpa
        )
    if scenario not in WIND_POWER_CURVES:
        raise KeyError(f"Unknown wind scenario: {scenario}")
    density = surface_pressure_hpa * 100.0 / (287.05 * (air_temperature_c + 273.15))
    actual_speed = np.maximum(np.asarray(wind_speed_100m, dtype=float), 0.0)
    equivalent_speed = actual_speed * np.cbrt(density / 1.225)
    curve = WIND_POWER_CURVES[scenario]
    power = np.interp(
        equivalent_speed,
        curve["speed_nodes_m_s"],
        curve["power_fraction"],
        left=0.0,
        right=1.0,
    )
    # Turbine cut-out logic responds to actual hub-height wind.  Density only
    # shifts the aerodynamic power-curve coordinate below cut-out.
    power[actual_speed >= float(curve["cut_out_m_s"])] = 0.0
    return np.clip(
        power * WIND_AVAILABILITY * (1.0 - WIND_OTHER_LOSS_FRACTION),
        0.0,
        WIND_AVAILABILITY,
    )


def longest_true_run(mask: np.ndarray) -> int:
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False])).astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return int((ends - starts).max()) if len(starts) else 0


def rolling_max(values: np.ndarray, window: int) -> float:
    if len(values) < window:
        return float(values.sum())
    cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=float)))
    return float(np.max(cumulative[window:] - cumulative[:-window]))


def design_energy_matched_portfolio(
    load_mw: np.ndarray,
    pv_cf: np.ndarray,
    wind_cf: np.ndarray,
    pv_share: float,
) -> dict[str, float]:
    resource = pv_share * pv_cf + (1.0 - pv_share) * wind_cf
    if resource.sum() <= 0.0:
        raise ValueError("Renewable resource has zero baseline-period output")
    total_capacity_mw = float(load_mw.sum() / resource.sum())
    return {
        "pv_nameplate_share": float(pv_share),
        "pv_capacity_mw": float(total_capacity_mw * pv_share),
        "wind_capacity_mw": float(total_capacity_mw * (1.0 - pv_share)),
        "total_capacity_mw": total_capacity_mw,
    }


def evaluate_fixed_portfolio(
    load_mw: np.ndarray,
    pv_cf: np.ndarray,
    wind_cf: np.ndarray,
    design: dict[str, float],
) -> tuple[dict[str, float | int], np.ndarray, np.ndarray]:
    supply = design["pv_capacity_mw"] * pv_cf + design["wind_capacity_mw"] * wind_cf
    matched = np.minimum(load_mw, supply)
    deficit = np.maximum(load_mw - supply, 0.0)
    metrics: dict[str, float | int] = {
        **design,
        "annual_load_mwh": float(load_mw.sum()),
        "annual_supply_mwh": float(supply.sum()),
        "matched_energy_mwh": float(matched.sum()),
        "unmatched_energy_mwh": float(deficit.sum()),
        "curtailed_energy_mwh": float(np.maximum(supply - load_mw, 0.0).sum()),
        "cfe_score": float(matched.sum() / load_mw.sum()),
        "deficit_hours": int((deficit > 1e-9).sum()),
        "longest_deficit_streak_hours": longest_true_run(deficit > 1e-9),
        "worst_24h_deficit_mwh": rolling_max(deficit, 24),
        "worst_168h_deficit_mwh": rolling_max(deficit, 168),
    }
    return metrics, supply, matched


def choose_baseline_portfolios(
    load_mw: np.ndarray, pv_cf: np.ndarray, wind_cf: np.ndarray
) -> dict[str, dict[str, float]]:
    """Choose one fixed design on the designated historical design year."""

    shares = np.linspace(0.0, 1.0, 21)
    resources = shares[:, None] * pv_cf[None, :] + (1.0 - shares[:, None]) * wind_cf[None, :]
    resource_energy = resources.sum(axis=1)
    feasible = resource_energy > 1e-12
    if not feasible.any():
        raise ValueError("Both wind and PV have zero design-year output")
    capacities = np.full(len(shares), np.inf, dtype=float)
    scores = np.full(len(shares), -np.inf, dtype=float)
    capacities[feasible] = load_mw.sum() / resource_energy[feasible]
    feasible_supplies = resources[feasible] * capacities[feasible, None]
    scores[feasible] = (
        np.minimum(feasible_supplies, load_mw[None, :]).sum(axis=1) / load_mw.sum()
    )
    best_score = scores.max()
    tied = np.flatnonzero(np.isclose(scores, best_score, atol=1e-12, rtol=0.0))
    best_index = int(tied[np.argmin(capacities[tied])])
    designs: dict[str, dict[str, float]] = {}
    if wind_cf.sum() > 1e-12:
        designs["all_wind_fixed_2023_design"] = design_energy_matched_portfolio(
            load_mw, pv_cf, wind_cf, 0.0
        )
    if pv_cf.sum() > 1e-12:
        designs["all_pv_fixed_2023_design"] = design_energy_matched_portfolio(
            load_mw, pv_cf, wind_cf, 1.0
        )
    designs["optimal_mix_fixed_2023_design"] = design_energy_matched_portfolio(
            load_mw, pv_cf, wind_cf, float(shares[best_index])
        )
    return designs


def cell_it_components(
    cell_strata: pd.DataFrame,
    context: dict[str, Any],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    hours = int(context["hours"])
    total = np.zeros(hours, dtype=float)
    by_type = {name: np.zeros(hours, dtype=float) for name in FACILITY_TYPES}
    scale = float(context["annual_scale_factor"])
    for row in cell_strata.itertuples(index=False):
        facility_type = str(row.facility_type)
        type_index = int(context["type_index"][facility_type])
        mass = (
            float(context["type_medians_mw"][facility_type])
            * float(row.within_type_spatial_share)
            * scale
        )
        shape = context["localized"][str(row.timezone_name)][:, type_index]
        component = mass * shape
        by_type[facility_type] += component
        total += component
    return total, by_type


def main() -> None:
    args = parse_args()
    years = tuple(sorted(set(args.year or YEARS)))
    if args.limit_grids is not None and args.limit_grids <= 0:
        raise ValueError("--limit-grids must be positive")
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.validation_root.mkdir(parents=True, exist_ok=True)

    mapping, strata = load_spatial_strata(args.mapping)
    catalog = pd.read_csv(args.grid_catalog)
    if set(catalog["era5_grid_id"].astype(str)) != set(strata["era5_grid_id"].astype(str)):
        raise ValueError("Mapping and ERA5 catalog grid-ID sets differ")
    interval_boundary = load_interval_boundary(args.interval_boundary, catalog)
    type_medians = load_type_medians(args.type_annual, args.scenario)
    contexts = {
        year: prepare_year_context(
            year,
            strata,
            type_medians,
            args.type_summary_root,
            args.global_hourly,
            args.scenario,
        )
        for year in years
    }
    # Persist the climate-independent D1 UTC bridge before checking weather.
    # This makes the corrected demand axis independently auditable and keeps
    # a failed/incomplete weather download from hiding completed demand work.
    legacy_global = pd.read_csv(
        args.global_hourly,
        usecols=["calendar_year", "hour_of_year", "scenario", "it_load_gw_p50"],
    )
    demand_utc_parts: list[pd.DataFrame] = []
    for year, context in contexts.items():
        legacy = legacy_global[
            legacy_global["calendar_year"].eq(year)
            & legacy_global["scenario"].eq(args.scenario)
        ].sort_values("hour_of_year")
        legacy_mw = legacy["it_load_gw_p50"].to_numpy(dtype=float) * 1_000.0
        reconstructed = np.asarray(context["expected_global_it_mw"], dtype=float)
        demand_utc_parts.append(
            pd.DataFrame(
                {
                    "calendar_year": year,
                    "time_utc": context["utc_index"],
                    "d1_it_mw_localtime_reconstructed": reconstructed,
                    "legacy_timezone_free_reference_p50_mw": legacy_mw,
                    "localtime_minus_legacy_reference_mw": reconstructed - legacy_mw,
                    "annual_scale_factor": float(context["annual_scale_factor"]),
                }
            )
        )
    demand_utc = pd.concat(demand_utc_parts, ignore_index=True)
    demand_utc_path = args.output_root / "global_hourly_d1_it_demand_utc_2023_2025.parquet"
    demand_utc.to_parquet(demand_utc_path, index=False, compression="zstd")
    pue_module = load_pue_module()
    grid_ids = sorted(strata["era5_grid_id"].unique())
    complete_inventory = args.limit_grids is None
    if args.limit_grids is not None:
        grid_ids = grid_ids[: args.limit_grids]

    expected_weather = [args.weather_root / f"{grid_id}.parquet" for grid_id in grid_ids]
    missing_weather = [path for path in expected_weather if not path.exists()]
    if missing_weather:
        raise FileNotFoundError(
            f"{len(missing_weather):,} selected ERA5 files are missing; examples: {missing_weather[:10]}"
        )

    grid_hour_paths = {
        year: args.output_root / f"grid_hourly_climate_demand_unit_resource_{year}.parquet"
        for year in years
    }
    for path in grid_hour_paths.values():
        if path.exists() and not args.force and not args.no_grid_hourly:
            raise FileExistsError(f"Output exists; use --force: {path}")
    writers: dict[int, pq.ParquetWriter | None] = {year: None for year in years}
    buffers: dict[int, list[pd.DataFrame]] = {year: [] for year in years}
    annual_rows: list[dict[str, Any]] = []
    portfolio_rows: list[dict[str, Any]] = []
    continuous_portfolio_rows: list[dict[str, Any]] = []
    global_arrays: dict[int, dict[str, np.ndarray]] = {}
    for year, context in contexts.items():
        hours = int(context["hours"])
        global_arrays[year] = {
            "it": np.zeros(hours),
            "type_linked": np.zeros(hours),
            "advanced": np.zeros(hours),
            "liquid": np.zeros(hours),
            "conventional": np.zeros(hours),
            **{
                f"optimal_supply_{scenario}": np.zeros(hours)
                for scenario in WIND_SCENARIOS
            },
            **{
                f"local_matched_{scenario}": np.zeros(hours)
                for scenario in WIND_SCENARIOS
            },
        }

    def flush(year: int) -> None:
        if not buffers[year] or args.no_grid_hourly:
            buffers[year].clear()
            return
        frame = pd.concat(buffers[year], ignore_index=True)
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if writers[year] is None:
            writers[year] = pq.ParquetWriter(
                grid_hour_paths[year], table.schema, compression="zstd", use_dictionary=True
            )
        writers[year].write_table(table)
        buffers[year].clear()

    design_year = 2023 if 2023 in years else min(years)
    try:
        for grid_number, grid_id in enumerate(grid_ids, start=1):
            cell_strata = strata[strata["era5_grid_id"].eq(grid_id)]
            cell_catalog = catalog[catalog["era5_grid_id"].eq(grid_id)]
            if len(cell_catalog) != 1:
                raise ValueError(f"Catalog record missing or duplicated for {grid_id}")
            catalog_row = cell_catalog.iloc[0]
            latitude = float(catalog_row["era5_latitude"])
            longitude = float(catalog_row["era5_longitude"])
            weather = load_weather(
                args.weather_root / f"{grid_id}.parquet",
                grid_id,
                latitude,
                longitude,
            )
            boundary_row = interval_boundary.loc[grid_id]
            year_data: dict[int, dict[str, Any]] = {}
            for year in years:
                context = contexts[year]
                left, right = interval_weather_for_year(weather, boundary_row, year)
                if len(left) != int(context["hours"]):
                    raise ValueError(f"Incomplete {year} weather intervals for {grid_id}")
                times = pd.DatetimeIndex(left["time"])
                solar_times = times + pd.Timedelta(minutes=30)
                temperature_left = left["temperature_2m"].to_numpy(dtype=float)
                temperature_right = right["temperature_2m"].to_numpy(dtype=float)
                humidity_left = left["relative_humidity_2m"].to_numpy(dtype=float)
                humidity_right = right["relative_humidity_2m"].to_numpy(dtype=float)
                temperature = 0.5 * (temperature_left + temperature_right)
                wet_bulb_left = pue_module.stull_wet_bulb_c(
                    temperature_left, humidity_left
                )
                wet_bulb_right = pue_module.stull_wet_bulb_c(
                    temperature_right, humidity_right
                )
                wet_bulb = 0.5 * (wet_bulb_left + wet_bulb_right)
                archetype_pue = {
                    name: 0.5
                    * (
                        pue_module.reduced_order_pue(
                            np.clip(wet_bulb_left, -20.0, 50.0), archetype
                        )
                        + pue_module.reduced_order_pue(
                            np.clip(wet_bulb_right, -20.0, 50.0), archetype
                        )
                    )
                    for name, archetype in pue_module.PUE_ARCHETYPES.items()
                }
                it_load, it_by_type = cell_it_components(cell_strata, context)
                type_linked_load = np.zeros_like(it_load)
                type_linked_climate_variable_overhead = np.zeros_like(it_load)
                for facility_type, component in it_by_type.items():
                    archetype_name = TYPE_LINKED_ARCHETYPE[facility_type]
                    archetype = pue_module.PUE_ARCHETYPES[archetype_name]
                    baseline_pue = 1.0 + archetype.fixed_overhead + archetype.flow_base
                    type_linked_load += component * archetype_pue[archetype_name]
                    type_linked_climate_variable_overhead += component * (
                        archetype_pue[archetype_name] - baseline_pue
                    )
                advanced_load = it_load * archetype_pue["advanced_hyperscale"]
                liquid_load = it_load * archetype_pue["liquid"]
                conventional_load = it_load * archetype_pue["conventional"]
                pv_cf = fixed_tilt_pv_cf(
                    solar_times,
                    latitude,
                    longitude,
                    right["shortwave_radiation"].to_numpy(dtype=float),
                    right["direct_normal_irradiance"].to_numpy(dtype=float),
                    right["diffuse_radiation"].to_numpy(dtype=float),
                    temperature,
                )
                wind_cf: dict[str, np.ndarray] = {}
                for scenario in WIND_SCENARIOS:
                    wind_left = density_corrected_wind_cf(
                        left["wind_speed_100m"].to_numpy(dtype=float),
                        temperature_left,
                        left["surface_pressure"].to_numpy(dtype=float),
                        scenario,
                    )
                    wind_right = density_corrected_wind_cf(
                        right["wind_speed_100m"].to_numpy(dtype=float),
                        temperature_right,
                        right["surface_pressure"].to_numpy(dtype=float),
                        scenario,
                    )
                    wind_cf[scenario] = 0.5 * (wind_left + wind_right)
                all_resource_arrays = [pv_cf, *wind_cf.values()]
                if any((values < 0.0).any() or (values > 1.0).any() for values in all_resource_arrays):
                    raise ValueError(
                        f"Resource capacity factor outside [0,1] for {grid_id}/{year}"
                    )
                advanced_archetype = pue_module.PUE_ARCHETYPES["advanced_hyperscale"]
                advanced_nonclimate_pue = (
                    1.0
                    + advanced_archetype.fixed_overhead
                    + advanced_archetype.flow_base
                )
                d0_advanced_climate_variable_overhead = (
                    archetype_pue["advanced_hyperscale"] - advanced_nonclimate_pue
                )
                year_data[year] = {
                    "times": times,
                    "temperature": temperature,
                    "wet_bulb": wet_bulb,
                    "temperature_domain_exceedance_hours": int(
                        (
                            (temperature_left < -20.0)
                            | (temperature_left > 50.0)
                            | (temperature_right < -20.0)
                            | (temperature_right > 50.0)
                        ).sum()
                    ),
                    "rh_domain_exceedance_hours": int(
                        (
                            (humidity_left < 5.0)
                            | (humidity_left > 99.0)
                            | (humidity_right < 5.0)
                            | (humidity_right > 99.0)
                        ).sum()
                    ),
                    "dni_above_solar_constant_hours": int(
                        (right["direct_normal_irradiance"].to_numpy(dtype=float) > 1_361.0).sum()
                    ),
                    "archetype_pue": archetype_pue,
                    "it_load": it_load,
                    "type_linked_load": type_linked_load,
                    "type_linked_climate_variable_overhead": type_linked_climate_variable_overhead,
                    "advanced_load": advanced_load,
                    "liquid_load": liquid_load,
                    "conventional_load": conventional_load,
                    "pv_cf": pv_cf,
                    "wind_cf": wind_cf,
                    "d0_advanced_climate_variable_overhead": d0_advanced_climate_variable_overhead,
                }

            design_data = year_data[design_year]
            designs_by_wind = {
                scenario: choose_baseline_portfolios(
                    design_data["type_linked_load"],
                    design_data["pv_cf"],
                    design_data["wind_cf"][scenario],
                )
                for scenario in WIND_SCENARIOS
            }
            for scenario, designs in designs_by_wind.items():
                continuous_load = np.concatenate(
                    [year_data[year]["type_linked_load"] for year in years]
                )
                continuous_pv = np.concatenate([year_data[year]["pv_cf"] for year in years])
                continuous_wind = np.concatenate(
                    [year_data[year]["wind_cf"][scenario] for year in years]
                )
                for portfolio_label, design in designs.items():
                    metrics, _, _ = evaluate_fixed_portfolio(
                        continuous_load, continuous_pv, continuous_wind, design
                    )
                    continuous_portfolio_rows.append(
                        {
                            "period": f"{min(years)}_{max(years)}_continuous",
                            "design_year": design_year,
                            "era5_grid_id": grid_id,
                            "era5_latitude": latitude,
                            "era5_longitude": longitude,
                            "logical_site_count": int(catalog_row["logical_site_count"]),
                            "wind_scenario": scenario,
                            "portfolio": portfolio_label,
                            "portfolio_scope": "deprecated_demand_scaled_capacity_diagnostic_only",
                            "resource_validation_status": RESOURCE_VALIDATION_STATUS,
                            **metrics,
                        }
                    )

            for year in years:
                data = year_data[year]
                it_load = data["it_load"]
                type_linked_load = data["type_linked_load"]
                advanced_load = data["advanced_load"]
                liquid_load = data["liquid_load"]
                conventional_load = data["conventional_load"]
                pv_cf = data["pv_cf"]
                archetype_pue = data["archetype_pue"]
                optimal_supply_by_wind: dict[str, np.ndarray] = {}
                local_matched_by_wind: dict[str, np.ndarray] = {}
                for scenario, designs in designs_by_wind.items():
                    for portfolio_label, design in designs.items():
                        metrics, supply, matched = evaluate_fixed_portfolio(
                            type_linked_load,
                            pv_cf,
                            data["wind_cf"][scenario],
                            design,
                        )
                        portfolio_rows.append(
                            {
                                "calendar_year": year,
                                "evaluation_role": (
                                    "design_sample" if year == design_year else "out_of_sample_weather_validation"
                                ),
                                "design_year": design_year,
                                "era5_grid_id": grid_id,
                                "era5_latitude": latitude,
                                "era5_longitude": longitude,
                                "logical_site_count": int(catalog_row["logical_site_count"]),
                                "wind_scenario": scenario,
                                "portfolio": portfolio_label,
                                "portfolio_scope": "deprecated_demand_scaled_capacity_diagnostic_only",
                                "resource_validation_status": RESOURCE_VALIDATION_STATUS,
                                **metrics,
                            }
                        )
                        if portfolio_label == "optimal_mix_fixed_2023_design":
                            optimal_supply_by_wind[scenario] = supply
                            local_matched_by_wind[scenario] = matched

                central_wind = data["wind_cf"][CENTRAL_WIND_SCENARIO]
                d0_climate_overhead = data["d0_advanced_climate_variable_overhead"]
                equal_mix_resource = 0.5 * (pv_cf + central_wind)
                climate_pv_correlation = (
                    float(np.corrcoef(d0_climate_overhead, pv_cf)[0, 1])
                    if np.std(d0_climate_overhead) > 0.0 and np.std(pv_cf) > 0.0
                    else float("nan")
                )
                climate_wind_correlation = (
                    float(np.corrcoef(d0_climate_overhead, central_wind)[0, 1])
                    if np.std(d0_climate_overhead) > 0.0 and np.std(central_wind) > 0.0
                    else float("nan")
                )
                compound_threshold_mask = (
                    d0_climate_overhead >= np.quantile(d0_climate_overhead, 0.90)
                ) & (equal_mix_resource <= np.quantile(equal_mix_resource, 0.10))
                annual_row: dict[str, Any] = {
                    "calendar_year": year,
                    "era5_grid_id": grid_id,
                    "era5_latitude": latitude,
                    "era5_longitude": longitude,
                    "logical_site_count": int(catalog_row["logical_site_count"]),
                    "timezone_count": int(catalog_row["timezone_count"]),
                    "d1_it_energy_mwh": float(it_load.sum()),
                    "d1_mean_it_mw": float(it_load.mean()),
                    "d1_peak_it_mw": float(it_load.max()),
                    "type_linked_facility_energy_mwh": float(type_linked_load.sum()),
                    "type_linked_facility_overhead_energy_mwh": float(
                        (type_linked_load - it_load).sum()
                    ),
                    "type_linked_climate_variable_overhead_energy_mwh": float(
                        data["type_linked_climate_variable_overhead"].sum()
                    ),
                    "type_linked_load_weighted_pue": float(type_linked_load.sum() / it_load.sum()),
                    "type_linked_facility_peak_mw": float(type_linked_load.max()),
                    "all_advanced_load_weighted_pue": float(advanced_load.sum() / it_load.sum()),
                    "all_liquid_load_weighted_pue": float(liquid_load.sum() / it_load.sum()),
                    "all_conventional_load_weighted_pue": float(conventional_load.sum() / it_load.sum()),
                    "d0_advanced_mean_pue": float(archetype_pue["advanced_hyperscale"].mean()),
                    "d0_advanced_p99_pue": float(np.quantile(archetype_pue["advanced_hyperscale"], 0.99)),
                    "d0_advanced_facility_mwh_per_1mw_it": float(archetype_pue["advanced_hyperscale"].sum()),
                    "d0_colocation_mean_pue": float(archetype_pue["colocation"].mean()),
                    "d0_colocation_p99_pue": float(np.quantile(archetype_pue["colocation"], 0.99)),
                    "d0_colocation_facility_mwh_per_1mw_it": float(archetype_pue["colocation"].sum()),
                    "d0_liquid_mean_pue": float(archetype_pue["liquid"].mean()),
                    "d0_liquid_p99_pue": float(np.quantile(archetype_pue["liquid"], 0.99)),
                    "d0_liquid_facility_mwh_per_1mw_it": float(archetype_pue["liquid"].sum()),
                    "d0_conventional_mean_pue": float(archetype_pue["conventional"].mean()),
                    "d0_conventional_p99_pue": float(np.quantile(archetype_pue["conventional"], 0.99)),
                    "d0_conventional_facility_mwh_per_1mw_it": float(archetype_pue["conventional"].sum()),
                    "d0_advanced_climate_variable_overhead_mean_mw_per_1mw_it": float(
                        d0_climate_overhead.mean()
                    ),
                    "d0_advanced_climate_variable_overhead_p99_mw_per_1mw_it": float(
                        np.quantile(d0_climate_overhead, 0.99)
                    ),
                    "temperature_interval_mean_c": float(data["temperature"].mean()),
                    "wet_bulb_interval_mean_c": float(data["wet_bulb"].mean()),
                    "stull_temperature_domain_exceedance_hours": data[
                        "temperature_domain_exceedance_hours"
                    ],
                    "stull_rh_domain_exceedance_hours": data["rh_domain_exceedance_hours"],
                    "dni_above_solar_constant_hours": data["dni_above_solar_constant_hours"],
                    "pv_capacity_factor": float(pv_cf.mean()),
                    "wind_capacity_factor": float(central_wind.mean()),
                    **{
                        f"wind_capacity_factor_{scenario}": float(values.mean())
                        for scenario, values in data["wind_cf"].items()
                    },
                    **{
                        f"wind_resource_zero_in_design_year_{scenario}": bool(
                            year_data[design_year]["wind_cf"][scenario].sum() <= 1e-12
                        )
                        for scenario in WIND_SCENARIOS
                    },
                    "pv_resource_zero_in_design_year": bool(
                        year_data[design_year]["pv_cf"].sum() <= 1e-12
                    ),
                    "pv_low_output_hours_cf_below_0p05": int((pv_cf < 0.05).sum()),
                    "wind_low_output_hours_cf_below_0p05": int((central_wind < 0.05).sum()),
                    "compound_low_wind_pv_hours": int(
                        ((pv_cf < 0.05) & (central_wind < 0.05)).sum()
                    ),
                    "d0_climate_overhead_pv_cf_correlation": climate_pv_correlation,
                    "d0_climate_overhead_wind_cf_correlation": climate_wind_correlation,
                    "high_d0_climate_overhead_low_equal_mix_resource_hours_p90_p10": int(
                        compound_threshold_mask.sum()
                    ),
                    "pue_validation_status": PUE_VALIDATION_STATUS,
                    "resource_validation_status": RESOURCE_VALIDATION_STATUS,
                }
                for archetype_name, pue_values in archetype_pue.items():
                    archetype = pue_module.PUE_ARCHETYPES[archetype_name]
                    annual_row[f"d0_{archetype_name}_annual_pue_plausible"] = bool(
                        archetype.plausible_annual_min
                        <= float(pue_values.mean())
                        <= archetype.plausible_annual_max
                    )
                annual_rows.append(annual_row)

                global_arrays[year]["it"] += it_load
                global_arrays[year]["type_linked"] += type_linked_load
                global_arrays[year]["advanced"] += advanced_load
                global_arrays[year]["liquid"] += liquid_load
                global_arrays[year]["conventional"] += conventional_load
                for scenario in WIND_SCENARIOS:
                    global_arrays[year][f"optimal_supply_{scenario}"] += (
                        optimal_supply_by_wind[scenario]
                    )
                    global_arrays[year][f"local_matched_{scenario}"] += (
                        local_matched_by_wind[scenario]
                    )

                if not args.no_grid_hourly:
                    hourly_payload: dict[str, Any] = {
                        "calendar_year": year,
                        "era5_grid_id": grid_id,
                        "time_utc": data["times"],
                        "it_load_mw": it_load.astype(np.float32),
                        "facility_load_type_linked_mw": type_linked_load.astype(np.float32),
                        "facility_load_all_advanced_mw": advanced_load.astype(np.float32),
                        "facility_load_all_liquid_mw": liquid_load.astype(np.float32),
                        "facility_load_all_conventional_mw": conventional_load.astype(np.float32),
                        "pue_type_linked": (type_linked_load / it_load).astype(np.float32),
                        "pue_advanced": archetype_pue["advanced_hyperscale"].astype(np.float32),
                        "pue_liquid": archetype_pue["liquid"].astype(np.float32),
                        "pue_conventional": archetype_pue["conventional"].astype(np.float32),
                        "d0_advanced_climate_variable_overhead_mw_per_1mw_it": (
                            d0_climate_overhead.astype(np.float32)
                        ),
                        "pv_cf_fixed_tilt": pv_cf.astype(np.float32),
                    }
                    hourly_payload.update(
                        {
                            f"wind_cf_{scenario}": values.astype(np.float32)
                            for scenario, values in data["wind_cf"].items()
                        }
                    )
                    buffers[year].append(pd.DataFrame(hourly_payload))
                    if len(buffers[year]) >= 12:
                        flush(year)
            if grid_number == 1 or grid_number % 50 == 0 or grid_number == len(grid_ids):
                print(f"Processed {grid_number:,}/{len(grid_ids):,} ERA5 grid cells.")
        for year in years:
            flush(year)
    finally:
        for writer in writers.values():
            if writer is not None:
                writer.close()

    annual = pd.DataFrame(annual_rows).sort_values(["calendar_year", "era5_grid_id"])
    portfolios = pd.DataFrame(portfolio_rows).sort_values(
        ["calendar_year", "era5_grid_id", "wind_scenario", "portfolio"]
    )
    continuous_portfolios = pd.DataFrame(continuous_portfolio_rows).sort_values(
        ["era5_grid_id", "wind_scenario", "portfolio"]
    )
    annual_path = args.output_root / "grid_annual_climate_demand_unit_resource.csv"
    deprecated_root = args.output_root / "deprecated_hypothetical_capacity"
    portfolio_path = (
        deprecated_root / "DEPRECATED_grid_local_renewable_portfolios.csv"
    )
    continuous_portfolio_path = (
        deprecated_root
        / "DEPRECATED_grid_continuous_2023_2025_renewable_portfolios.csv"
    )
    atomic_csv(annual, annual_path)
    atomic_csv(portfolios, portfolio_path)
    atomic_csv(continuous_portfolios, continuous_portfolio_path)

    global_hourly_parts: list[pd.DataFrame] = []
    global_annual_rows: list[dict[str, Any]] = []
    max_global_reconciliation_error = 0.0
    for year, arrays in global_arrays.items():
        context = contexts[year]
        expected = np.asarray(context["expected_global_it_mw"], dtype=float)
        if complete_inventory:
            error = float(np.max(np.abs(arrays["it"] - expected)))
            max_global_reconciliation_error = max(max_global_reconciliation_error, error)
        else:
            error = float("nan")
        hourly_payload: dict[str, Any] = {
            "calendar_year": year,
            "time_utc": context["utc_index"],
            "d1_it_load_mw": arrays["it"],
            "facility_load_type_linked_mw": arrays["type_linked"],
            "facility_load_all_advanced_mw": arrays["advanced"],
            "facility_load_all_liquid_mw": arrays["liquid"],
            "facility_load_all_conventional_mw": arrays["conventional"],
        }
        for wind_scenario in WIND_SCENARIOS:
            supply = arrays[f"optimal_supply_{wind_scenario}"]
            local_match = arrays[f"local_matched_{wind_scenario}"]
            pooled_match = np.minimum(arrays["type_linked"], supply)
            hourly_payload[f"local_optimal_supply_{wind_scenario}_mw"] = supply
            hourly_payload[f"same_grid_matched_{wind_scenario}_mw"] = local_match
            hourly_payload[f"unconstrained_pool_matched_{wind_scenario}_mw"] = pooled_match
        global_hourly_parts.append(pd.DataFrame(hourly_payload))
        total_load = float(arrays["type_linked"].sum())
        for wind_scenario in WIND_SCENARIOS:
            supply = arrays[f"optimal_supply_{wind_scenario}"]
            local_match = arrays[f"local_matched_{wind_scenario}"]
            pooled_match = np.minimum(arrays["type_linked"], supply)
            global_annual_rows.append(
                {
                    "calendar_year": year,
                    "evaluation_role": (
                        "design_sample" if year == design_year else "out_of_sample_weather_validation"
                    ),
                    "design_year": design_year,
                    "wind_scenario": wind_scenario,
                    "inventory_complete": complete_inventory,
                    "grids_processed": len(grid_ids),
                    "d1_mean_it_gw": float(arrays["it"].mean() / 1_000.0),
                    "d1_it_energy_twh": float(arrays["it"].sum() / 1_000_000.0),
                    "type_linked_mean_facility_load_gw": float(arrays["type_linked"].mean() / 1_000.0),
                    "type_linked_facility_energy_twh": float(total_load / 1_000_000.0),
                    "type_linked_facility_overhead_energy_twh": float(
                        (total_load - arrays["it"].sum()) / 1_000_000.0
                    ),
                    "type_linked_load_weighted_pue": float(total_load / arrays["it"].sum()),
                    "type_linked_peak_facility_load_gw": float(arrays["type_linked"].max() / 1_000.0),
                    "type_linked_p99_facility_load_gw": float(
                        np.quantile(arrays["type_linked"], 0.99) / 1_000.0
                    ),
                    "all_advanced_load_weighted_pue": float(
                        arrays["advanced"].sum() / arrays["it"].sum()
                    ),
                    "all_liquid_load_weighted_pue": float(
                        arrays["liquid"].sum() / arrays["it"].sum()
                    ),
                    "all_conventional_load_weighted_pue": float(
                        arrays["conventional"].sum() / arrays["it"].sum()
                    ),
                    "fixed_2023_same_grid_cfe_score": float(local_match.sum() / total_load),
                    "fixed_2023_unconstrained_pool_cfe_score": float(
                        pooled_match.sum() / total_load
                    ),
                    "fixed_2023_same_grid_unmatched_energy_twh": float(
                        (total_load - local_match.sum()) / 1_000_000.0
                    ),
                    "fixed_2023_supply_to_load_ratio": float(supply.sum() / total_load),
                    "annual_scale_factor_from_type_medians_to_global_p50": float(
                        context["annual_scale_factor"]
                    ),
                    "published_global_p50_mean_it_gw": float(
                        context["target_mean_it_mw"] / 1_000.0
                    ),
                    "global_it_reconciliation_max_abs_mw": error,
                    "pue_validation_status": PUE_VALIDATION_STATUS,
                    "resource_validation_status": RESOURCE_VALIDATION_STATUS,
                }
            )
    global_hourly = pd.concat(global_hourly_parts, ignore_index=True)
    global_hourly_path = (
        deprecated_root
        / "DEPRECATED_global_hourly_climate_demand_supply_2023_2025.parquet"
    )
    global_hourly.to_parquet(global_hourly_path, index=False, compression="zstd")
    formal_global_hourly_path = (
        args.output_root / "global_hourly_climate_demand_2023_2025.parquet"
    )
    formal_global_hourly_columns = [
        "calendar_year",
        "time_utc",
        "d1_it_load_mw",
        "facility_load_type_linked_mw",
        "facility_load_all_advanced_mw",
        "facility_load_all_liquid_mw",
        "facility_load_all_conventional_mw",
    ]
    global_hourly[formal_global_hourly_columns].to_parquet(
        formal_global_hourly_path, index=False, compression="zstd"
    )
    global_annual = pd.DataFrame(global_annual_rows).sort_values(
        ["calendar_year", "wind_scenario"]
    )
    global_annual_path = (
        deprecated_root / "DEPRECATED_global_annual_climate_demand_supply.csv"
    )
    atomic_csv(global_annual, global_annual_path)
    formal_global_annual_path = args.output_root / "global_annual_climate_demand.csv"
    formal_global_annual_columns = [
        "calendar_year",
        "inventory_complete",
        "grids_processed",
        "d1_mean_it_gw",
        "d1_it_energy_twh",
        "type_linked_mean_facility_load_gw",
        "type_linked_facility_energy_twh",
        "type_linked_facility_overhead_energy_twh",
        "type_linked_load_weighted_pue",
        "type_linked_peak_facility_load_gw",
        "type_linked_p99_facility_load_gw",
        "all_advanced_load_weighted_pue",
        "all_liquid_load_weighted_pue",
        "all_conventional_load_weighted_pue",
        "annual_scale_factor_from_type_medians_to_global_p50",
        "published_global_p50_mean_it_gw",
        "global_it_reconciliation_max_abs_mw",
        "pue_validation_status",
    ]
    formal_global_annual = global_annual[
        global_annual["wind_scenario"].eq(CENTRAL_WIND_SCENARIO)
    ][formal_global_annual_columns].copy()
    atomic_csv(formal_global_annual, formal_global_annual_path)

    design_portfolios = portfolios[portfolios["evaluation_role"].eq("design_sample")]
    wind_cf_columns = [f"wind_capacity_factor_{scenario}" for scenario in WIND_SCENARIOS]
    pue_plausibility_columns = [
        column for column in annual.columns if column.endswith("_annual_pue_plausible")
    ]
    validation = {
        "method_version": METHOD_VERSION,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "inventory_complete": complete_inventory,
        "logical_sites": int(len(mapping)),
        "era5_grids_available_in_mapping": int(strata["era5_grid_id"].nunique()),
        "era5_grids_processed": int(len(grid_ids)),
        "iana_timezones": int(strata["timezone_name"].nunique()),
        "years": list(years),
        "design_year": design_year,
        "out_of_sample_weather_validation_years": [
            year for year in years if year != design_year
        ],
        "global_it_reconciliation_max_abs_mw": (
            max_global_reconciliation_error if complete_inventory else None
        ),
        "annual_load_weighted_pue_range": {
            "minimum": float(
                annual[
                    [
                        "type_linked_load_weighted_pue",
                        "all_advanced_load_weighted_pue",
                        "all_liquid_load_weighted_pue",
                        "all_conventional_load_weighted_pue",
                    ]
                ].min().min()
            ),
            "maximum": float(
                annual[
                    [
                        "type_linked_load_weighted_pue",
                        "all_advanced_load_weighted_pue",
                        "all_liquid_load_weighted_pue",
                        "all_conventional_load_weighted_pue",
                    ]
                ].max().max()
            ),
        },
        "capacity_factor_bounds_passed": bool(
            annual["pv_capacity_factor"].between(0.0, 1.0).all()
            and annual[wind_cf_columns].apply(lambda values: values.between(0.0, 1.0)).all().all()
        ),
        "design_year_portfolio_energy_balance_max_abs_mwh": float(
            (design_portfolios["annual_supply_mwh"] - design_portfolios["annual_load_mwh"])
            .abs()
            .max()
        ),
        "portfolio_load_accounting_max_abs_mwh": float(
            (
                portfolios["matched_energy_mwh"]
                + portfolios["unmatched_energy_mwh"]
                - portfolios["annual_load_mwh"]
            )
            .abs()
            .max()
        ),
        "portfolio_supply_accounting_max_abs_mwh": float(
            (
                portfolios["matched_energy_mwh"]
                + portfolios["curtailed_energy_mwh"]
                - portfolios["annual_supply_mwh"]
            )
            .abs()
            .max()
        ),
        "same_grid_not_above_unconstrained_pool_passed": bool(
            (
                global_annual["fixed_2023_same_grid_cfe_score"]
                <= global_annual["fixed_2023_unconstrained_pool_cfe_score"] + 1e-12
            ).all()
        ),
        "continuous_period_tail_metrics_rows": int(len(continuous_portfolios)),
        "design_year_zero_wind_resource_grid_scenarios": int(
            annual[annual["calendar_year"].eq(design_year)][
                [
                    f"wind_resource_zero_in_design_year_{scenario}"
                    for scenario in WIND_SCENARIOS
                ]
            ]
            .to_numpy(dtype=bool)
            .sum()
        ),
        "design_year_zero_pv_resource_grids": int(
            annual[annual["calendar_year"].eq(design_year)][
                "pv_resource_zero_in_design_year"
            ].sum()
        ),
        "pue_archetype_grid_year_plausibility_pass_rate": float(
            annual[pue_plausibility_columns].to_numpy(dtype=bool).mean()
        ),
        "stull_temperature_domain_exceedance_grid_hours": int(
            annual["stull_temperature_domain_exceedance_hours"].sum()
        ),
        "stull_rh_domain_exceedance_grid_hours": int(
            annual["stull_rh_domain_exceedance_hours"].sum()
        ),
        "dni_above_solar_constant_grid_hours": int(
            annual["dni_above_solar_constant_hours"].sum()
        ),
        "time_axis": (
            "Interval-start UTC labels [t,t+1); workload templates selected in site IANA "
            "civil time including DST and fractional-hour offsets. Instantaneous PUE/wind "
            "responses are trapezoidal means of endpoints t and t+1. Radiation for [t,t+1) "
            "uses Open-Meteo timestamp t+1 (preceding-hour mean), with solar geometry at t+30m."
        ),
        "absolute_scale": (
            "type-level joint-posterior medians with one annual factor to the "
            "published global P50 mean; no hourly forcing to the timezone-free reference curve"
        ),
        "pue_validation_status": PUE_VALIDATION_STATUS,
        "resource_validation_status": RESOURCE_VALIDATION_STATUS,
    }
    validation_path = (
        args.validation_root
        / "deprecated_hypothetical_capacity"
        / "DEPRECATED_internal_accounting_and_proxy_validation.json"
    )
    atomic_json(validation, validation_path)

    metadata = {
        "method_version": METHOD_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "scenario": args.scenario,
        "years": list(years),
        "design_year": design_year,
        "formal_supply_status": "actual_installed_capacity_not_loaded",
        "hypothetical_demand_scaled_portfolios_deprecated": True,
        "evaluation_design": (
            "Capacity factors are formal unit-nameplate weather coefficients. "
            "The legacy 2023 demand-scaled PV/wind portfolio is a deprecated "
            "diagnostic and is excluded from supply, generation, and CFE claims."
        ),
        "market_sample_used": False,
        "inputs": {
            "mapping": {"path": str(args.mapping), "sha256": sha256_file(args.mapping)},
            "grid_catalog": {"path": str(args.grid_catalog), "sha256": sha256_file(args.grid_catalog)},
            "type_annual": {"path": str(args.type_annual), "sha256": sha256_file(args.type_annual)},
            "global_hourly": {"path": str(args.global_hourly), "sha256": sha256_file(args.global_hourly)},
            "type_hourly_profiles": {
                str(year): {
                    "path": str(args.type_summary_root / f"type_hourly_path_summary_{year}.csv"),
                    "sha256": sha256_file(
                        args.type_summary_root / f"type_hourly_path_summary_{year}.csv"
                    ),
                }
                for year in years
            },
            "pue_module": {
                "path": str(PUE_MODULE_PATH),
                "sha256": sha256_file(PUE_MODULE_PATH),
            },
            "weather_root": str(args.weather_root),
            "interval_boundary": {
                "path": str(args.interval_boundary),
                "sha256": sha256_file(args.interval_boundary),
            },
        },
        "type_annual_medians_mw": {name: float(value) for name, value in type_medians.items()},
        "type_linked_pue_mapping": TYPE_LINKED_ARCHETYPE,
        "pue_parameters": {
            name: asdict(archetype) for name, archetype in pue_module.PUE_ARCHETYPES.items()
        },
        "pv_parameters": {
            "hourly_interval_alignment": (
                "API radiation timestamp t+1 represents mean over demand interval [t,t+1); "
                "solar position evaluated at interval midpoint t+30 minutes"
            ),
            "surface_tilt_deg": "clip(abs(latitude), 10, 60)",
            "surface_azimuth": "equator_facing",
            "transposition": "beam_plus_isotropic_sky_plus_albedo_0.20",
            "dc_ac_ratio": 1.34,
            "system_derate": 0.86,
            "NOCT_c": 45.0,
            "temperature_coefficient_per_c": -0.004,
        },
        "wind_parameters": {
            "hub_height_m": 100.0,
            "air_density_reference_kg_m3": 1.225,
            "instantaneous_to_interval": (
                "calculate nonlinear CF at t and t+1, then trapezoidal mean"
            ),
            "central_scenario": CENTRAL_WIND_SCENARIO,
            "scenarios": list(WIND_SCENARIOS),
            "availability": WIND_AVAILABILITY,
            "other_loss_fraction": WIND_OTHER_LOSS_FRACTION,
            "nrel_composite_curve_source": (
                "King, Clifton & Hodge 2014, NREL/TP-5D00-61714, doi:10.2172/1159354"
            ),
            "nrel_curves": {
                name: {
                    "speed_nodes_m_s": curve["speed_nodes_m_s"].tolist(),
                    "power_fraction": curve["power_fraction"].tolist(),
                    "cut_out_m_s": float(curve["cut_out_m_s"]),
                }
                for name, curve in WIND_POWER_CURVES.items()
            },
            "cubic_conservative": {
                "cut_in_m_s": 3.0,
                "rated_m_s": 12.0,
                "cut_out_m_s": 25.0,
            },
        },
        "outputs": {
            "global_hourly_d1_it_utc_bridge": str(demand_utc_path),
            "grid_annual": str(annual_path),
            "formal_global_hourly_demand": str(formal_global_hourly_path),
            "formal_global_annual_demand": str(formal_global_annual_path),
            "deprecated_grid_local_portfolios": str(portfolio_path),
            "deprecated_grid_continuous_portfolios": str(continuous_portfolio_path),
            "deprecated_global_hourly_with_hypothetical_supply": str(global_hourly_path),
            "deprecated_global_annual_with_hypothetical_supply": str(global_annual_path),
            "grid_hourly_by_year": (
                {str(year): str(path) for year, path in grid_hour_paths.items()}
                if not args.no_grid_hourly
                else {}
            ),
            "validation": str(validation_path),
        },
    }
    metadata_path = (
        deprecated_root / "DEPRECATED_coupled_historical_metadata.json"
    )
    atomic_json(metadata, metadata_path)
    print(f"Wrote coupled historical outputs under {args.output_root}")
    print(formal_global_annual.to_string(index=False))


if __name__ == "__main__":
    main()
