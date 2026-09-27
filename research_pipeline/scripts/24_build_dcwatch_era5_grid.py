#!/usr/bin/env python3
"""Map the full DCWatch logical-site inventory to native ERA5 grid cells.

This script deliberately does not use ``market_sample.csv``.  Every logical
site in the audited 8,360-site inventory is retained and assigned to its
nearest 0.25-degree ERA5 grid centre.  Facility posterior medians are used
only to construct within-type spatial weights; they are not treated as the
global demand total because a sum of marginal medians is not the median of a
joint posterior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from timezonefinder import TimezoneFinder


ROOT = Path(__file__).resolve().parents[2]
HIER = ROOT / "data" / "24x7_cfe" / "processed" / "demand_hierarchical"
DEFAULT_SITES = HIER / "logical_sites_enriched_2023_standardized.csv"
DEFAULT_DEMAND = HIER / "facility_demand_hierarchical_2023.csv"
DEFAULT_OUTPUT = ROOT / "data" / "24x7_cfe" / "processed" / "climate_coupling"
DEFAULT_SCENARIO = "transport__capacity_utilization_rho_+0.0"
FACILITY_TYPES = (
    "colocation_interconnection",
    "hyperscale_cloud",
    "ai_hpc",
    "enterprise_internal",
    "telecom_edge",
)
METHOD_VERSION = "dcwatch_full_inventory_to_era5_native_grid_with_iana_time_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", type=Path, default=DEFAULT_SITES)
    parser.add_argument("--facility-demand", type=Path, default=DEFAULT_DEMAND)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--grid-resolution-deg", type=float, default=0.25)
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
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def nearest_grid(values: pd.Series, resolution: float) -> np.ndarray:
    rounded = np.round(values.to_numpy(dtype=float) / resolution) * resolution
    rounded[np.isclose(rounded, 0.0)] = 0.0
    return np.round(rounded, 6)


def assign_timezones(mapping: pd.DataFrame) -> pd.DataFrame:
    """Assign an IANA civil-time zone at each original site coordinate."""

    finder = TimezoneFinder(in_memory=True)
    names: list[str | None] = []
    for latitude, longitude in mapping[["lat", "lng"]].itertuples(index=False, name=None):
        names.append(finder.timezone_at(lat=float(latitude), lng=float(longitude)))
    result = mapping.copy()
    result["timezone_name"] = names
    if result["timezone_name"].isna().any():
        missing = int(result["timezone_name"].isna().sum())
        raise ValueError(f"TimezoneFinder could not assign {missing} site time zones")
    result["timezone_mapping_method"] = "timezonefinder_at_original_site_coordinate"
    return result


def haversine_km(
    latitude_a: np.ndarray,
    longitude_a: np.ndarray,
    latitude_b: np.ndarray,
    longitude_b: np.ndarray,
) -> np.ndarray:
    radius_km = 6371.0088
    lat_a = np.radians(latitude_a)
    lat_b = np.radians(latitude_b)
    delta_lat = lat_b - lat_a
    delta_lon = np.radians(longitude_b - longitude_a)
    term = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat_a) * np.cos(lat_b) * np.sin(delta_lon / 2.0) ** 2
    )
    return 2.0 * radius_km * np.arcsin(np.minimum(np.sqrt(term), 1.0))


def weighted_quantile(
    values: np.ndarray, weights: np.ndarray, probabilities: list[float]
) -> list[float]:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = np.maximum(weights[order], 0.0)
    if not np.isfinite(sorted_weights).all() or sorted_weights.sum() <= 0:
        return [float(np.quantile(sorted_values, probability)) for probability in probabilities]
    cumulative = np.cumsum(sorted_weights)
    cumulative = (cumulative - 0.5 * sorted_weights) / sorted_weights.sum()
    return [
        float(np.interp(probability, cumulative, sorted_values))
        for probability in probabilities
    ]


def load_inputs(
    sites_path: Path, demand_path: Path, scenario: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    site_columns = [
        "logical_site_id",
        "name",
        "country_for_aggregation",
        "continent",
        "lat",
        "lng",
        "demand_evidence_level",
        "capacity_completion_method",
    ]
    sites = pd.read_csv(sites_path, usecols=site_columns, low_memory=False)
    if sites.empty or sites["logical_site_id"].duplicated().any():
        raise ValueError("Logical-site inventory is empty or has duplicate logical_site_id values")
    if sites[["lat", "lng"]].isna().any().any():
        raise ValueError("Every logical site must have latitude and longitude")
    if not sites["lat"].between(-90.0, 90.0).all():
        raise ValueError("Latitude outside [-90, 90]")
    if not sites["lng"].between(-180.0, 180.0).all():
        raise ValueError("Longitude outside [-180, 180]")

    probability_columns = [f"posterior_type_share_{name}" for name in FACILITY_TYPES]
    demand_columns = [
        "logical_site_id",
        "scenario",
        "mean_it_mw_p10",
        "mean_it_mw_p50",
        "mean_it_mw_p90",
        *probability_columns,
    ]
    demand = pd.read_csv(demand_path, usecols=demand_columns, low_memory=False)
    demand = demand[demand["scenario"].eq(scenario)].copy()
    if len(demand) != len(sites) or demand["logical_site_id"].duplicated().any():
        raise ValueError(
            f"Scenario {scenario!r} must contain one demand record per logical site"
        )
    probabilities = demand[probability_columns].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or (probabilities < 0.0).any():
        raise ValueError("Invalid posterior facility-type probabilities")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=2e-3):
        raise ValueError("Posterior facility-type shares do not sum to one")
    return sites, demand


def build_mapping(
    sites: pd.DataFrame,
    demand: pd.DataFrame,
    resolution: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if resolution <= 0.0 or not np.isclose((1.0 / resolution) % 1.0, 0.0):
        raise ValueError("Grid resolution must be a positive divisor of one degree")
    mapping = sites.merge(
        demand.drop(columns="scenario"),
        on="logical_site_id",
        how="left",
        validate="one_to_one",
    )
    mapping["era5_latitude"] = nearest_grid(mapping["lat"], resolution)
    mapping["era5_longitude"] = nearest_grid(mapping["lng"], resolution)
    mapping["era5_latitude"] = mapping["era5_latitude"].clip(-90.0, 90.0)
    longitude = mapping["era5_longitude"].to_numpy(dtype=float)
    longitude = np.where(longitude > 180.0, longitude - 360.0, longitude)
    longitude = np.where(longitude < -180.0, longitude + 360.0, longitude)
    mapping["era5_longitude"] = longitude

    cells = (
        mapping[["era5_latitude", "era5_longitude"]]
        .drop_duplicates()
        .sort_values(["era5_latitude", "era5_longitude"], kind="stable")
        .reset_index(drop=True)
    )
    cells.insert(0, "era5_grid_id", [f"era5_g{index:04d}" for index in range(1, len(cells) + 1)])
    mapping = mapping.merge(
        cells,
        on=["era5_latitude", "era5_longitude"],
        how="left",
        validate="many_to_one",
    )
    mapping = assign_timezones(mapping)
    mapping["site_to_grid_distance_km"] = haversine_km(
        mapping["lat"].to_numpy(dtype=float),
        mapping["lng"].to_numpy(dtype=float),
        mapping["era5_latitude"].to_numpy(dtype=float),
        mapping["era5_longitude"].to_numpy(dtype=float),
    )

    for facility_type in FACILITY_TYPES:
        mapping[f"spatial_weight_mw_{facility_type}"] = (
            mapping["mean_it_mw_p50"]
            * mapping[f"posterior_type_share_{facility_type}"]
        )

    group = mapping.groupby("era5_grid_id", sort=False, observed=True)
    catalog = cells.merge(
        group.agg(
            logical_site_count=("logical_site_id", "size"),
            country_count=("country_for_aggregation", "nunique"),
            continent_count=("continent", "nunique"),
            timezone_count=("timezone_name", "nunique"),
            site_mean_it_mw_p10_sum=("mean_it_mw_p10", "sum"),
            site_mean_it_mw_p50_sum=("mean_it_mw_p50", "sum"),
            site_mean_it_mw_p90_sum=("mean_it_mw_p90", "sum"),
            max_site_to_grid_distance_km=("site_to_grid_distance_km", "max"),
        ).reset_index(),
        on="era5_grid_id",
        how="left",
        validate="one_to_one",
    )
    for facility_type in FACILITY_TYPES:
        weights = group[f"spatial_weight_mw_{facility_type}"].sum().rename(
            f"spatial_weight_mw_{facility_type}"
        )
        catalog = catalog.merge(
            weights.reset_index(), on="era5_grid_id", how="left", validate="one_to_one"
        )
    country_lists = group["country_for_aggregation"].agg(
        lambda values: ";".join(sorted({str(value) for value in values.dropna()}))
    )
    catalog = catalog.merge(
        country_lists.rename("countries").reset_index(),
        on="era5_grid_id",
        how="left",
        validate="one_to_one",
    )
    timezone_lists = group["timezone_name"].agg(
        lambda values: ";".join(sorted({str(value) for value in values.dropna()}))
    )
    dominant_timezone_rows: list[dict[str, str]] = []
    for grid_id, subset in mapping.groupby("era5_grid_id", sort=False, observed=True):
        weights = subset.groupby("timezone_name", observed=True)["mean_it_mw_p50"].sum()
        dominant_timezone_rows.append(
            {"era5_grid_id": str(grid_id), "dominant_timezone": str(weights.idxmax())}
        )
    catalog = catalog.merge(
        timezone_lists.rename("timezones").reset_index(),
        on="era5_grid_id",
        how="left",
        validate="one_to_one",
    ).merge(
        pd.DataFrame(dominant_timezone_rows),
        on="era5_grid_id",
        how="left",
        validate="one_to_one",
    )

    mapping.insert(0, "mapping_method", "nearest_native_era5_grid_centre")
    mapping.insert(1, "grid_resolution_deg", resolution)
    mapping = mapping.sort_values("logical_site_id", kind="stable").reset_index(drop=True)
    catalog.insert(1, "mapping_method", "nearest_native_era5_grid_centre")
    catalog.insert(2, "grid_resolution_deg", resolution)
    return mapping, catalog


def main() -> None:
    args = parse_args()
    sites, demand = load_inputs(args.sites, args.facility_demand, args.scenario)
    mapping, catalog = build_mapping(sites, demand, args.grid_resolution_deg)
    mapping_path = args.output_root / "facility_to_era5_grid.csv"
    catalog_path = args.output_root / "era5_grid_catalog.csv"
    metadata_path = args.output_root / "facility_to_era5_grid.metadata.json"
    atomic_csv(mapping, mapping_path)
    atomic_csv(catalog, catalog_path)

    distances = mapping["site_to_grid_distance_km"].to_numpy(dtype=float)
    site_weights = mapping["mean_it_mw_p50"].to_numpy(dtype=float)
    unweighted = np.quantile(distances, [0.5, 0.9, 0.99]).tolist()
    weighted = weighted_quantile(distances, site_weights, [0.5, 0.9, 0.99])
    metadata: dict[str, Any] = {
        "method_version": METHOD_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "scenario": args.scenario,
        "grid_resolution_deg": args.grid_resolution_deg,
        "mapping_method": "nearest_native_era5_grid_centre",
        "input_files": {
            "logical_sites": {"path": str(args.sites), "sha256": sha256_file(args.sites)},
            "facility_demand": {"path": str(args.facility_demand), "sha256": sha256_file(args.facility_demand)},
        },
        "counts": {
            "logical_sites": int(len(mapping)),
            "unique_era5_grid_cells": int(len(catalog)),
            "sites_with_coordinates": int(mapping[["lat", "lng"]].notna().all(axis=1).sum()),
            "antarctic_sites_latitude_below_minus_60": int(mapping["lat"].lt(-60.0).sum()),
            "iana_timezones": int(mapping["timezone_name"].nunique()),
            "sites_without_timezone": int(mapping["timezone_name"].isna().sum()),
            "grid_cells_crossing_timezone_boundaries": int(catalog["timezone_count"].gt(1).sum()),
        },
        "site_to_grid_distance_km": {
            "median": float(unweighted[0]),
            "p90": float(unweighted[1]),
            "p99": float(unweighted[2]),
            "maximum": float(distances.max()),
            "site_p50_demand_weighted_median": float(weighted[0]),
            "site_p50_demand_weighted_p90": float(weighted[1]),
            "site_p50_demand_weighted_p99": float(weighted[2]),
        },
        "spatial_weight_note": (
            "Site marginal P50 demand times posterior type share is used only "
            "for within-type spatial allocation. Absolute hourly global demand "
            "must be reconciled to the hierarchical joint-posterior global P50."
        ),
        "market_sample_used": False,
        "time_axis_note": (
            "Demand templates are interpreted in each site's IANA civil time. "
            "The coupled model converts the common UTC ERA5 axis to local time "
            "before selecting a workload-template hour."
        ),
        "outputs": {
            "facility_mapping": str(mapping_path),
            "grid_catalog": str(catalog_path),
        },
    }
    atomic_json(metadata, metadata_path)
    print(
        f"Mapped {len(mapping):,} logical sites to {len(catalog):,} native "
        f"{args.grid_resolution_deg:g}-degree ERA5 cells."
    )
    print(f"Facility mapping: {mapping_path}")
    print(f"Grid catalog: {catalog_path}")


if __name__ == "__main__":
    main()
