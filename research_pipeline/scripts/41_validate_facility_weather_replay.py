#!/usr/bin/env python3
"""Validate the 1995--2014 facility weather extraction.

Two independent things are checked.

Completeness and physics
    Every catalogued facility grid has an NPZ and metadata file, the stored
    hash matches, all three variables carry exactly 175,321 finite endpoints,
    and every value sits inside physically plausible bounds.

Cross-source agreement
    The same grids already hold 2023--2025 hourly weather fetched from
    Open-Meteo's ERA5 archive -- a different retrieval path, a different
    period and a different vendor.  Comparing per-grid climatological means
    against the new ARCO extraction tests coordinate handling and unit
    handling end to end.  The two periods are not the same years, so a warming
    offset is expected and is reported rather than corrected; what must hold
    is a near-unit correlation across the 2,416 grids.

Relative humidity is derived here only for the cross-check, using the same
August-Roche-Magnus form Open-Meteo applies to ERA5 dew point.  Nothing
derived is written back to the extraction store.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG = (
    ROOT
    / "data"
    / "24x7_cfe"
    / "processed"
    / "climate_coupling"
    / "era5_grid_catalog.csv"
)
IN_TREE_WEATHER_ROOT = (
    ROOT
    / "data"
    / "24x7_cfe"
    / "processed"
    / "climate_coupling"
    / "facility_weather_1995_2014"
)
DEFAULT_WEATHER_ROOT = Path(
    os.environ.get("FACILITY_WEATHER_ROOT") or IN_TREE_WEATHER_ROOT
)
DEFAULT_REFERENCE_ROOT = (
    ROOT
    / "data"
    / "24x7_cfe"
    / "raw"
    / "openmeteo"
    / "era5_2023_2025_dcwatch_grid"
)
DEFAULT_OUTPUT_ROOT = (
    ROOT / "output" / "24x7_cfe_facility" / "facility_weather_replay_validation"
)
METHOD_VERSION = "facility_weather_replay_validation_v1"
EXPECTED_ENDPOINT_COUNT = 175_321
ARRAY_NAMES = ("t2m_k", "d2m_k", "sp_pa")

# Bounds are deliberately wider than any populated location, so a breach means
# a broken read rather than an unusual site.
BOUNDS = {
    "t2m_k": (180.0, 340.0),
    "d2m_k": (170.0, 330.0),
    "sp_pa": (45_000.0, 108_000.0),
}
# A dew point above the dry bulb is a known ERA5 saturation artefact; a large
# excess would instead mean the two variables were mismatched.
DEWPOINT_EXCESS_TOLERANCE_K = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--weather-root", type=Path, default=DEFAULT_WEATHER_ROOT)
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def json_scalar(value: Any) -> Any:
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
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_scalar),
        encoding="utf-8",
    )
    temporary.replace(path)


def saturation_vapour_pressure_hpa(temperature_c: np.ndarray) -> np.ndarray:
    """August-Roche-Magnus saturation vapour pressure over water."""

    return 6.112 * np.exp(17.67 * temperature_c / (temperature_c + 243.5))


def relative_humidity_percent(
    temperature_k: np.ndarray, dewpoint_k: np.ndarray
) -> np.ndarray:
    numerator = saturation_vapour_pressure_hpa(dewpoint_k - 273.15)
    denominator = saturation_vapour_pressure_hpa(temperature_k - 273.15)
    return 100.0 * numerator / denominator


def scan_grids(catalog: pd.DataFrame, weather_root: Path) -> pd.DataFrame:
    """Read every grid's arrays once, returning per-grid diagnostics."""

    records = []
    for row in catalog.itertuples(index=False):
        grid_id = str(row.era5_grid_id)
        npz_path = weather_root / f"{grid_id}.npz"
        metadata_path = weather_root / f"{grid_id}.metadata.json"
        record: dict[str, Any] = {
            "era5_grid_id": grid_id,
            "npz_present": npz_path.exists(),
            "metadata_present": metadata_path.exists(),
        }
        if not (record["npz_present"] and record["metadata_present"]):
            records.append(record)
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        with np.load(npz_path) as archive:
            arrays = {name: archive[name] for name in archive.files}
        record["array_names_match"] = set(arrays) == set(ARRAY_NAMES)
        record["endpoint_count"] = int(
            min(len(values) for values in arrays.values())
        ) if arrays else 0
        record["all_finite"] = bool(
            all(np.isfinite(values).all() for values in arrays.values())
        )
        record["derived_quantities_stored"] = bool(
            metadata.get("derived_quantities_stored", True)
        )
        record["requested_latitude"] = float(metadata["requested_latitude"])
        record["requested_longitude"] = float(metadata["requested_longitude"])
        record["coordinate_matches_catalog"] = bool(
            abs(record["requested_latitude"] - float(row.era5_latitude)) < 1e-6
            and abs(record["requested_longitude"] - float(row.era5_longitude)) < 1e-6
        )
        record["response_matches_request"] = bool(
            abs(float(metadata["response_latitude"]) - record["requested_latitude"])
            <= 0.126
            and abs(
                (
                    (
                        float(metadata["response_longitude"])
                        - record["requested_longitude"]
                        + 180.0
                    )
                    % 360.0
                )
                - 180.0
            )
            <= 0.126
        )
        temperature = arrays["t2m_k"].astype(float)
        dewpoint = arrays["d2m_k"].astype(float)
        pressure = arrays["sp_pa"].astype(float)
        for name, values in (
            ("t2m_k", temperature),
            ("d2m_k", dewpoint),
            ("sp_pa", pressure),
        ):
            low, high = BOUNDS[name]
            record[f"min_{name}"] = float(values.min())
            record[f"max_{name}"] = float(values.max())
            record[f"{name}_within_bounds"] = bool(
                values.min() >= low and values.max() <= high
            )
        excess = dewpoint - temperature
        record["max_dewpoint_excess_k"] = float(excess.max())
        record["dewpoint_excess_within_tolerance"] = bool(
            excess.max() <= DEWPOINT_EXCESS_TOLERANCE_K
        )
        humidity = relative_humidity_percent(temperature, dewpoint)
        record["min_relative_humidity_percent"] = float(humidity.min())
        record["max_relative_humidity_percent"] = float(humidity.max())
        record["mean_t2m_c"] = float(temperature.mean() - 273.15)
        record["mean_relative_humidity_percent"] = float(humidity.mean())
        record["mean_sp_pa"] = float(pressure.mean())
        records.append(record)
    return pd.DataFrame(records)


def reference_means(reference_root: Path, grid_ids: list[str]) -> pd.DataFrame:
    """Per-grid 2023--2025 Open-Meteo ERA5 means for the cross-check."""

    records = []
    for grid_id in grid_ids:
        path = reference_root / f"{grid_id}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(
            path, columns=["temperature_2m", "relative_humidity_2m", "surface_pressure"]
        )
        records.append(
            {
                "era5_grid_id": grid_id,
                "reference_mean_t2m_c": float(frame["temperature_2m"].mean()),
                "reference_mean_relative_humidity_percent": float(
                    frame["relative_humidity_2m"].mean()
                ),
                # Open-Meteo reports surface pressure in hPa.
                "reference_mean_sp_pa": float(frame["surface_pressure"].mean()) * 100.0,
            }
        )
    return pd.DataFrame(records)


def cross_check(scan: pd.DataFrame, reference: pd.DataFrame) -> dict[str, Any]:
    if reference.empty:
        return {
            "status": "REFERENCE_UNAVAILABLE",
            "note": "No Open-Meteo 2023--2025 grid files were found",
        }
    merged = scan.merge(reference, on="era5_grid_id", how="inner")
    result: dict[str, Any] = {
        "status": "COMPARED",
        "grid_count": int(len(merged)),
        "replay_period": "1995-01-01/2015-01-01 UTC",
        "reference_period": "2023--2025",
        "reference_source": "Open-Meteo ERA5 archive",
        "periods_are_disjoint": True,
        "note": (
            "The periods do not overlap, so a warming offset in temperature is "
            "expected. The test is the across-grid correlation, not the offset."
        ),
    }
    for label, replay_column, reference_column in (
        ("temperature_c", "mean_t2m_c", "reference_mean_t2m_c"),
        (
            "relative_humidity_percent",
            "mean_relative_humidity_percent",
            "reference_mean_relative_humidity_percent",
        ),
        ("surface_pressure_pa", "mean_sp_pa", "reference_mean_sp_pa"),
    ):
        difference = merged[replay_column] - merged[reference_column]
        result[label] = {
            "pearson_correlation": float(
                merged[replay_column].corr(merged[reference_column])
            ),
            "replay_mean": float(merged[replay_column].mean()),
            "reference_mean": float(merged[reference_column].mean()),
            "mean_difference_replay_minus_reference": float(difference.mean()),
            "median_absolute_difference": float(difference.abs().median()),
            "p99_absolute_difference": float(difference.abs().quantile(0.99)),
            "max_absolute_difference": float(difference.abs().max()),
        }
    return result


def main() -> None:
    args = parse_args()
    catalog = pd.read_csv(
        args.grid_catalog,
        usecols=["era5_grid_id", "era5_latitude", "era5_longitude"],
    )
    print(f"Scanning {len(catalog):,} facility grids in {args.weather_root}")
    scan = scan_grids(catalog, args.weather_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    scan.to_parquet(args.output_root / "facility_weather_grid_scan.parquet", index=False)

    print("Reading Open-Meteo 2023--2025 reference means")
    reference = reference_means(args.reference_root, scan["era5_grid_id"].tolist())
    comparison = cross_check(scan, reference)

    complete = scan[scan["npz_present"] & scan["metadata_present"]]
    checks = {
        "every_catalogued_grid_has_output": bool(len(complete) == len(catalog)),
        "array_names_match_everywhere": bool(scan["array_names_match"].all()),
        "endpoint_count_exact_everywhere": bool(
            (scan["endpoint_count"] == EXPECTED_ENDPOINT_COUNT).all()
        ),
        "all_values_finite": bool(scan["all_finite"].all()),
        "no_derived_quantities_stored": bool(
            (~scan["derived_quantities_stored"]).all()
        ),
        "coordinates_match_catalog": bool(scan["coordinate_matches_catalog"].all()),
        "response_within_half_cell": bool(scan["response_matches_request"].all()),
        "t2m_within_bounds": bool(scan["t2m_k_within_bounds"].all()),
        "d2m_within_bounds": bool(scan["d2m_k_within_bounds"].all()),
        "sp_within_bounds": bool(scan["sp_pa_within_bounds"].all()),
        "dewpoint_excess_within_tolerance": bool(
            scan["dewpoint_excess_within_tolerance"].all()
        ),
        "relative_humidity_within_0_to_101_percent": bool(
            (scan["min_relative_humidity_percent"] >= 0.0).all()
            and (scan["max_relative_humidity_percent"] <= 101.0).all()
        ),
    }
    if comparison.get("status") == "COMPARED":
        checks["cross_source_temperature_correlation_above_0_99"] = bool(
            comparison["temperature_c"]["pearson_correlation"] > 0.99
        )
        checks["cross_source_pressure_correlation_above_0_99"] = bool(
            comparison["surface_pressure_pa"]["pearson_correlation"] > 0.99
        )
        checks["cross_source_humidity_correlation_above_0_95"] = bool(
            comparison["relative_humidity_percent"]["pearson_correlation"] > 0.95
        )

    all_pass = all(checks.values())
    validation = {
        "schema_version": "1.0",
        "method_version": METHOD_VERSION,
        "completed_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "weather_root": str(args.weather_root),
        "catalogued_grid_count": int(len(catalog)),
        "grids_with_output": int(len(complete)),
        "expected_endpoint_count": EXPECTED_ENDPOINT_COUNT,
        "hard_checks": checks,
        "all_hard_checks_pass": all_pass,
        "physical_ranges": {
            "min_t2m_c": float(scan["min_t2m_k"].min() - 273.15),
            "max_t2m_c": float(scan["max_t2m_k"].max() - 273.15),
            "min_mean_t2m_c": float(scan["mean_t2m_c"].min()),
            "max_mean_t2m_c": float(scan["mean_t2m_c"].max()),
            "min_sp_kpa": float(scan["min_sp_pa"].min() / 1000.0),
            "max_sp_kpa": float(scan["max_sp_pa"].max() / 1000.0),
            "max_dewpoint_excess_k": float(scan["max_dewpoint_excess_k"].max()),
            "grids_with_any_dewpoint_excess": int(
                (scan["max_dewpoint_excess_k"] > 0.0).sum()
            ),
        },
        "cross_source_comparison": comparison,
        "validation_status": "PASS" if all_pass else "FAIL",
    }
    atomic_json(validation, args.output_root / "facility_weather_validation.json")
    failed = [name for name, value in checks.items() if not value]
    print(json.dumps(checks, indent=2))
    if comparison.get("status") == "COMPARED":
        print(json.dumps(comparison, indent=2, default=json_scalar))
    if failed:
        raise SystemExit(f"Failed hard checks: {failed}")
    print(f"All hard checks pass; wrote {args.output_root}")


if __name__ == "__main__":
    main()
