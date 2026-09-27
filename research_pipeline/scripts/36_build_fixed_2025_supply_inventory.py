#!/usr/bin/env python3
"""Build the fixed-2025 wind/PV inventory for the 1995--2014 replay.

IRENASTAT 2025 year-end capacity is the national capacity anchor.  EIA-860
provides the United States spatial weights and GEM provides spatial weights
elsewhere.  Asset capacities used as weights are never interpreted as the
published national capacity total.  Solar thermal capacity is retained in the
capacity evidence but is excluded from the PV generation model.

The main allocation contains on-grid capacity.  An all-grid allocation is
also retained as a physical-supply sensitivity.  Capacity without a valid
spatial anchor remains explicitly unallocated; it is never moved to a data
centre or a fabricated power-plant location.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
import time
import unicodedata
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SUPPLY_ROOT = (
    ROOT / "data" / "24x7_cfe" / "processed" / "wind_solar_supply_2023_2025"
)
DEFAULT_GEM = SUPPLY_ROOT / "gem_historical_vintage_asset_snapshot_2023_2025.parquet"
DEFAULT_EIA = SUPPLY_ROOT / "eia_us_generator_year_capacity_2023_2025.parquet"
DEFAULT_IRENA_METADATA = (
    ROOT
    / "data"
    / "24x7_cfe"
    / "raw"
    / "supply_assets"
    / "irena"
    / "2026-h1"
    / "country_capacity_metadata.json"
)
DEFAULT_RAW_ROOT = (
    ROOT
    / "data"
    / "24x7_cfe"
    / "raw"
    / "supply_assets"
    / "irena"
    / "2026-h1-fixed-replay-detail"
)
DEFAULT_OUTPUT_ROOT = (
    ROOT
    / "data"
    / "24x7_cfe"
    / "processed"
    / "wind_solar_supply_20y_fixed_2025"
)
IRENA_COUNTRY_URL = (
    "https://pxweb.irena.org/api/v1/en/IRENASTAT/"
    "Power%20Capacity%20and%20Generation/"
    "Country_ELECCAP_2026_H1_v-PX%201.px"
)
IRENA_TERMS_URL = "https://www.irena.org/terms-and-conditions"
IRENA_RELEASE = "IRENASTAT 2026 H1"
CAPACITY_REFERENCE_YEAR = 2025
WEATHER_START = "1995-01-01T00:00:00Z"
WEATHER_END_EXCLUSIVE = "2015-01-01T00:00:00Z"
EXPECTED_INTERVAL_COUNT = 175_320
EXPECTED_WEATHER_ENDPOINTS = EXPECTED_INTERVAL_COUNT + 1
GRID_RESOLUTION_DEG = 0.25
METHOD_VERSION = "fixed_2025_irena_capacity_eia_gem_spatial_allocation_v1"
TARGET_IRENA_TECHNOLOGIES = (
    "Solar energy",
    "Solar photovoltaic",
    "Solar thermal energy",
    "Wind energy",
    "Onshore wind energy",
    "Offshore wind energy",
)
PRIMARY_TECHNOLOGY_MAP = {
    "solar_pv": "Solar photovoltaic",
    "wind": "Wind energy",
}

# GEM names that do not normalize to the IRENA country/area labels.  Values
# are IRENA/ISO-like three-character codes.  Any code absent from IRENA is
# retained in the anchor audit but receives no allocated capacity.
GEM_COUNTRY_CODE_ALIASES = {
    "American Samoa": "ASM",
    "Antigua and Barbuda": "ATG",
    "Bermuda": "BMU",
    "Bonaire, Sint Eustatius, and Saba": "BES",
    "Bosnia and Herzegovina": "BIH",
    "British Indian Ocean Territory": "IOT",
    "Brunei": "BRN",
    "Cayman Islands": "CYM",
    "Central African Republic": "CAF",
    "Cook Islands": "COK",
    "Czech Republic": "CZE",
    "DR Congo": "COD",
    "Dominican Republic": "DOM",
    "Equatorial Guinea": "GNQ",
    "French Guiana": "GUF",
    "French Polynesia": "PYF",
    "Hong Kong": "HKG",
    "Iran": "IRN",
    "Jersey": "JEY",
    "Laos": "LAO",
    "Moldova": "MDA",
    "New Caledonia": "NCL",
    "Norfolk Island": "NFK",
    "Papua New Guinea": "PNG",
    "Russia": "RUS",
    "Saint Kitts and Nevis": "KNA",
    "Saint Vincent and the Grenadines": "VCT",
    "Sao Tome and Principe": "STP",
    "Solomon Islands": "SLB",
    "South Korea": "KOR",
    "Syria": "SYR",
    "Taiwan": "TWN",
    "The Gambia": "GMB",
    "United Arab Emirates": "ARE",
    "United Kingdom": "GBR",
    "United States": "USA",
    "Virgin Islands (U.S.)": "VIR",
    "Western Sahara": "ESH",
    "Åland Islands": "ALA",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gem", type=Path, default=DEFAULT_GEM)
    parser.add_argument("--eia", type=Path, default=DEFAULT_EIA)
    parser.add_argument("--irena-metadata", type=Path, default=DEFAULT_IRENA_METADATA)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--refresh-irena", action="store_true")
    parser.add_argument("--offline", action="store_true")
    return parser.parse_args()


def atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_text(value: Any) -> str:
    ascii_text = (
        unicodedata.normalize("NFKD", str(value))
        .encode("ascii", "ignore")
        .decode("ascii")
        .casefold()
    )
    return re.sub(r"[^a-z0-9]+", "", ascii_text)


def variable(metadata: dict[str, Any], label: str) -> dict[str, Any]:
    matches = [
        item
        for item in metadata["variables"]
        if normalized_text(item.get("text")) == normalized_text(label)
    ]
    if len(matches) != 1:
        raise ValueError(f"Cannot identify IRENA metadata variable {label!r}")
    return matches[0]


def category_code(item: dict[str, Any], label: str) -> str:
    lookup = {
        normalized_text(text): str(code)
        for code, text in zip(item["values"], item["valueTexts"], strict=True)
    }
    key = normalized_text(label)
    if key not in lookup:
        raise ValueError(f"IRENA variable {item['code']} lacks category {label!r}")
    return lookup[key]


def detailed_irena_query(metadata: dict[str, Any]) -> dict[str, Any]:
    area = variable(metadata, "Country/area")
    technology = variable(metadata, "Technology")
    grid = variable(metadata, "Grid connection")
    year = variable(metadata, "Year")
    return {
        "query": [
            {
                "code": area["code"],
                "selection": {"filter": "all", "values": ["*"]},
            },
            {
                "code": technology["code"],
                "selection": {
                    "filter": "item",
                    "values": [
                        category_code(technology, label)
                        for label in TARGET_IRENA_TECHNOLOGIES
                    ],
                },
            },
            {
                "code": grid["code"],
                "selection": {"filter": "all", "values": ["*"]},
            },
            {
                "code": year["code"],
                "selection": {
                    "filter": "item",
                    "values": [category_code(year, str(CAPACITY_REFERENCE_YEAR))],
                },
            },
        ],
        "response": {"format": "json-stat2"},
    }


def fetch_json(url: str, payload: dict[str, Any]) -> Any:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "academic-fixed-capacity-weather-replay/1.0",
        },
    )
    errors: list[str] = []
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return json.loads(response.read().decode("utf-8-sig"))
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            errors.append(f"attempt {attempt}: {exc}")
            if attempt < 3:
                time.sleep(2**attempt)
    raise RuntimeError(f"IRENA request failed: {'; '.join(errors)}")


def load_detailed_irena(
    metadata: dict[str, Any], raw_root: Path, refresh: bool, offline: bool
) -> tuple[dict[str, Any], dict[str, Any], str]:
    query = detailed_irena_query(metadata)
    request_path = raw_root / "country_capacity_2025_detailed_request.json"
    response_path = raw_root / "country_capacity_2025_detailed_response.json"
    atomic_json(query, request_path)
    if response_path.exists() and not refresh:
        response = json.loads(response_path.read_text(encoding="utf-8-sig"))
        retrieved_at = datetime.fromtimestamp(
            response_path.stat().st_mtime, tz=UTC
        ).replace(microsecond=0).isoformat()
    else:
        if offline:
            raise FileNotFoundError(f"Offline IRENA response missing: {response_path}")
        response = fetch_json(IRENA_COUNTRY_URL, query)
        retrieved_at = datetime.now(UTC).replace(microsecond=0).isoformat()
        atomic_json(response, response_path)
    return query, response, retrieved_at


def ordered_categories(dimension: dict[str, Any], expected_size: int) -> list[str]:
    index = dimension["category"].get("index")
    if isinstance(index, dict):
        values = [code for code, _ in sorted(index.items(), key=lambda pair: pair[1])]
    elif isinstance(index, list):
        values = [str(code) for code in index]
    else:
        values = list(dimension["category"].get("label", {}))
    if len(values) != expected_size:
        raise ValueError("IRENA JSON-stat category length mismatch")
    return values


def parse_irena_jsonstat(dataset: dict[str, Any], retrieved_at: str) -> pd.DataFrame:
    dimension_names = list(dataset["id"])
    sizes = [int(value) for value in dataset["size"]]
    codes: dict[str, list[str]] = {}
    labels: dict[str, dict[str, str]] = {}
    for name, size in zip(dimension_names, sizes, strict=True):
        dimension = dataset["dimension"][name]
        codes[name] = ordered_categories(dimension, size)
        labels[name] = {
            str(code): str(label)
            for code, label in dimension["category"].get("label", {}).items()
        }
    values = dataset.get("value", [])
    expected_cells = math.prod(sizes)
    if isinstance(values, list) and len(values) != expected_cells:
        raise ValueError("IRENA JSON-stat value length mismatch")
    records: list[dict[str, Any]] = []
    combinations = itertools.product(*(codes[name] for name in dimension_names))
    for flat_index, combination in enumerate(combinations):
        raw_value = (
            values[flat_index]
            if isinstance(values, list)
            else values.get(str(flat_index), values.get(flat_index))
        )
        record: dict[str, Any] = {"installed_capacity_mw": raw_value}
        for name, code in zip(dimension_names, combination, strict=True):
            record[f"{name}__code"] = code
            record[f"{name}__label"] = labels[name].get(code, code)
        records.append(record)
    raw = pd.DataFrame.from_records(records)
    dimensions = [column[: -len("__label")] for column in raw if column.endswith("__label")]
    technology_dim = next(name for name in dimensions if normalized_text(name) == "technology")
    grid_dim = next(name for name in dimensions if normalized_text(name) == "gridconnection")
    year_dim = next(name for name in dimensions if normalized_text(name) == "year")
    area_dim = next(
        name for name in dimensions if name not in {technology_dim, grid_dim, year_dim}
    )
    result = pd.DataFrame(
        {
            "country_area_code": raw[f"{area_dim}__code"].astype("string"),
            "country_area": raw[f"{area_dim}__label"].astype("string"),
            "technology_code": raw[f"{technology_dim}__code"].astype("string"),
            "technology": raw[f"{technology_dim}__label"].astype("string"),
            "grid_connection_code": raw[f"{grid_dim}__code"].astype("string"),
            "grid_connection": raw[f"{grid_dim}__label"].astype("string"),
            "year": pd.to_numeric(raw[f"{year_dim}__label"], errors="raise").astype("Int16"),
            "installed_capacity_mw": pd.to_numeric(
                raw["installed_capacity_mw"], errors="coerce"
            ).astype("Float64"),
        }
    )
    result["is_source_aggregate_area"] = result["country_area_code"].isin(
        ["REA", "OCA"]
    )
    result["source_dataset"] = "IRENASTAT Electricity Capacity by Country/area"
    result["source_release"] = IRENA_RELEASE
    result["source_url"] = IRENA_COUNTRY_URL
    result["source_terms_url"] = IRENA_TERMS_URL
    result["source_retrieved_at_utc"] = retrieved_at
    return result.sort_values(
        ["country_area_code", "technology", "grid_connection"], kind="stable"
    ).reset_index(drop=True)


def snap_grid(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return np.round(numeric / GRID_RESOLUTION_DEG) * GRID_RESOLUTION_DEG


def normalize_longitudes(values: np.ndarray) -> np.ndarray:
    normalized = ((np.asarray(values, dtype=float) + 180.0) % 360.0) - 180.0
    return np.where(np.isclose(normalized, -180.0), 180.0, normalized)


def country_lookup(irena: pd.DataFrame) -> tuple[dict[str, str], dict[str, str]]:
    country_rows = irena.loc[
        ~irena["is_source_aggregate_area"], ["country_area_code", "country_area"]
    ].drop_duplicates()
    by_name = {
        normalized_text(row.country_area): str(row.country_area_code)
        for row in country_rows.itertuples(index=False)
    }
    official_names = {
        str(row.country_area_code): str(row.country_area)
        for row in country_rows.itertuples(index=False)
    }
    return by_name, official_names


def map_gem_country_codes(
    names: pd.Series, by_name: dict[str, str]
) -> pd.Series:
    def one(value: Any) -> str | None:
        text = str(value)
        direct = by_name.get(normalized_text(text))
        return direct if direct is not None else GEM_COUNTRY_CODE_ALIASES.get(text)

    result = names.map(one).astype("string")
    if result.isna().any():
        missing = sorted(names[result.isna()].dropna().astype(str).unique().tolist())
        raise ValueError(f"Unmapped GEM country/area names: {missing}")
    return result


def build_spatial_anchors(
    gem_path: Path,
    eia_path: Path,
    by_name: dict[str, str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    gem = pd.read_parquet(gem_path)
    gem_2025 = gem[gem["proxy_target_year"].eq(CAPACITY_REFERENCE_YEAR)].copy()
    operating = gem_2025[gem_2025["status_base"].eq("operating")].copy()
    known_future = operating["start_year"].notna() & operating["start_year"].gt(
        CAPACITY_REFERENCE_YEAR
    )
    operating = operating.loc[~known_future].copy()
    operating = operating.loc[operating["country_area"].ne("United States")].copy()
    solar_pv = operating["tracker"].eq("solar") & operating["technology_detail"].isin(
        ["PV", "Assumed PV"]
    )
    wind = operating["tracker"].eq("wind")
    operating = operating.loc[solar_pv | wind].copy()
    operating["technology"] = np.where(solar_pv.loc[operating.index], "solar_pv", "wind")
    operating["capacity_weight_mw"] = pd.to_numeric(
        operating["capacity_reported_mw"], errors="coerce"
    )
    valid = (
        operating["capacity_weight_mw"].gt(0.0)
        & pd.to_numeric(operating["latitude"], errors="coerce").between(-90.0, 90.0)
        & pd.to_numeric(operating["longitude"], errors="coerce").between(-180.0, 180.0)
    )
    gem_invalid = int((~valid).sum())
    operating = operating.loc[valid].copy()
    operating["country_area_code"] = map_gem_country_codes(
        operating["country_area"], by_name
    )
    operating["anchor_source"] = "GEM 2026-02 next-release proxy for 2025"
    operating["anchor_release_status"] = "next_release_proxy_not_exact_year_end"
    operating["known_operating_by_2025"] = operating["start_year"].notna() & operating[
        "start_year"
    ].le(CAPACITY_REFERENCE_YEAR)
    gem_missing_start = int(operating["start_year"].isna().sum())
    gem_anchor = operating[
        [
            "country_area_code",
            "country_area",
            "technology",
            "latitude",
            "longitude",
            "capacity_weight_mw",
            "known_operating_by_2025",
            "anchor_source",
            "anchor_release_status",
        ]
    ].copy()

    eia = pd.read_parquet(eia_path)
    eia_2025 = eia[eia["year"].eq(CAPACITY_REFERENCE_YEAR)].copy()
    eia_2025 = eia_2025.loc[eia_2025["is_formal_operating"].fillna(False)].copy()
    known_eia_future = eia_2025["operating_year"].notna() & eia_2025[
        "operating_year"
    ].gt(CAPACITY_REFERENCE_YEAR)
    eia_2025 = eia_2025.loc[~known_eia_future].copy()
    eia_2025["capacity_weight_mw"] = pd.to_numeric(
        eia_2025["nameplate_capacity_mw"], errors="coerce"
    )
    valid_eia = (
        eia_2025["technology_family"].isin(["solar", "wind"])
        & eia_2025["capacity_weight_mw"].gt(0.0)
        & pd.to_numeric(eia_2025["latitude"], errors="coerce").between(-90.0, 90.0)
        & pd.to_numeric(eia_2025["longitude"], errors="coerce").between(-180.0, 180.0)
    )
    eia_invalid = int((~valid_eia).sum())
    eia_2025 = eia_2025.loc[valid_eia].copy()
    eia_anchor = pd.DataFrame(
        {
            "country_area_code": "USA",
            "country_area": "United States",
            "technology": eia_2025["technology_family"].replace({"solar": "solar_pv"}),
            "latitude": eia_2025["latitude"],
            "longitude": eia_2025["longitude"],
            "capacity_weight_mw": eia_2025["capacity_weight_mw"],
            "known_operating_by_2025": True,
            "anchor_source": "EIA-860 generator-year",
            "anchor_release_status": eia_2025["release_status"].astype("string"),
        }
    )
    anchors = pd.concat([gem_anchor, eia_anchor], ignore_index=True)
    anchors["era5_latitude"] = snap_grid(anchors["latitude"])
    anchors["era5_longitude"] = normalize_longitudes(snap_grid(anchors["longitude"]))
    anchors["known_capacity_weight_mw"] = np.where(
        anchors["known_operating_by_2025"], anchors["capacity_weight_mw"], 0.0
    )
    grouped = (
        anchors.groupby(
            [
                "country_area_code",
                "technology",
                "era5_latitude",
                "era5_longitude",
            ],
            as_index=False,
            observed=True,
        )
        .agg(
            capacity_weight_mw=("capacity_weight_mw", "sum"),
            known_capacity_weight_mw=("known_capacity_weight_mw", "sum"),
            anchor_asset_count=("capacity_weight_mw", "size"),
            known_start_asset_count=("known_operating_by_2025", "sum"),
            anchor_sources=("anchor_source", lambda values: " | ".join(sorted(set(values)))),
            anchor_release_statuses=(
                "anchor_release_status",
                lambda values: " | ".join(sorted(set(values.dropna().astype(str)))),
            ),
        )
        .sort_values(
            ["country_area_code", "technology", "era5_latitude", "era5_longitude"],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    audit = {
        "gem_2025_rows": int(len(gem_2025)),
        "gem_operating_rows_before_date_filter": int(
            gem_2025["status_base"].eq("operating").sum()
        ),
        "gem_known_post_2025_operating_rows_excluded": int(known_future.sum()),
        "gem_non_us_pv_wind_anchor_rows": int(len(gem_anchor)),
        "gem_anchor_rows_with_missing_start_year_retained": gem_missing_start,
        "gem_invalid_capacity_or_coordinate_rows_excluded": gem_invalid,
        "eia_2025_rows": int(len(eia[eia["year"].eq(CAPACITY_REFERENCE_YEAR)])),
        "eia_formal_operating_anchor_rows": int(len(eia_anchor)),
        "eia_invalid_capacity_or_coordinate_rows_excluded": eia_invalid,
        "combined_anchor_rows": int(len(anchors)),
        "combined_country_technology_grid_rows": int(len(grouped)),
        "usa_gem_replaced_by_eia_not_added": True,
    }
    return grouped, audit


def irena_country_capacity(irena: pd.DataFrame) -> pd.DataFrame:
    base = irena.loc[~irena["is_source_aggregate_area"]].copy()
    grouped = (
        base.groupby(
            ["country_area_code", "country_area", "technology", "grid_connection"],
            as_index=False,
            observed=True,
        )["installed_capacity_mw"]
        .sum(min_count=1)
    )
    rows: list[pd.DataFrame] = []
    for model_technology, irena_technology in PRIMARY_TECHNOLOGY_MAP.items():
        selected = grouped[grouped["technology"].eq(irena_technology)].copy()
        pivot = selected.pivot_table(
            index=["country_area_code", "country_area"],
            columns="grid_connection",
            values="installed_capacity_mw",
            aggfunc="sum",
        ).reset_index()
        pivot.columns.name = None
        for column in ["OnGrid", "OffGrid"]:
            if column not in pivot:
                pivot[column] = pd.NA
        pivot["technology"] = model_technology
        pivot["irena_on_grid_capacity_mw"] = pd.to_numeric(
            pivot["OnGrid"], errors="coerce"
        )
        pivot["irena_off_grid_capacity_mw"] = pd.to_numeric(
            pivot["OffGrid"], errors="coerce"
        )
        pivot["irena_all_grid_capacity_mw"] = pivot[
            ["irena_on_grid_capacity_mw", "irena_off_grid_capacity_mw"]
        ].sum(axis=1, min_count=1)
        rows.append(
            pivot[
                [
                    "country_area_code",
                    "country_area",
                    "technology",
                    "irena_on_grid_capacity_mw",
                    "irena_off_grid_capacity_mw",
                    "irena_all_grid_capacity_mw",
                ]
            ]
        )
    return pd.concat(rows, ignore_index=True)


def build_allocation(
    anchors: pd.DataFrame,
    capacities: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    totals = (
        anchors.groupby(["country_area_code", "technology"], as_index=False)
        .agg(
            country_capacity_weight_mw=("capacity_weight_mw", "sum"),
            country_known_capacity_weight_mw=("known_capacity_weight_mw", "sum"),
            country_anchor_asset_count=("anchor_asset_count", "sum"),
        )
    )
    allocation = anchors.merge(
        totals,
        on=["country_area_code", "technology"],
        how="left",
        validate="many_to_one",
    ).merge(
        capacities,
        on=["country_area_code", "technology"],
        how="left",
        validate="many_to_one",
    )
    allocation["central_spatial_weight"] = (
        allocation["capacity_weight_mw"] / allocation["country_capacity_weight_mw"]
    )
    allocation["known_start_spatial_weight"] = np.where(
        allocation["country_known_capacity_weight_mw"].gt(0.0),
        allocation["known_capacity_weight_mw"]
        / allocation["country_known_capacity_weight_mw"],
        np.nan,
    )
    for scope in ["on_grid", "all_grid"]:
        capacity_column = f"irena_{scope}_capacity_mw"
        allocation[f"allocated_{scope}_capacity_mw"] = (
            allocation[capacity_column] * allocation["central_spatial_weight"]
        )
        allocation[f"allocated_{scope}_capacity_known_start_sensitivity_mw"] = (
            allocation[capacity_column] * allocation["known_start_spatial_weight"]
        )

    # A location alone is not an installed-capacity observation.  GEM/EIA may
    # contain coordinates for a country/technology pair for which the detailed
    # IRENA 2025 table reports zero capacity or has no matching country record.
    # Such rows must not trigger an ERA5 extraction or enter the replay.  Where
    # IRENA reports positive all-grid capacity but no on-grid component, the
    # primary on-grid contribution is explicitly zero (rather than NaN).
    allocation = allocation.loc[
        allocation["allocated_all_grid_capacity_mw"].fillna(0.0).gt(0.0)
    ].copy()
    capacity_columns = [
        "allocated_on_grid_capacity_mw",
        "allocated_all_grid_capacity_mw",
        "allocated_on_grid_capacity_known_start_sensitivity_mw",
        "allocated_all_grid_capacity_known_start_sensitivity_mw",
    ]
    allocation[capacity_columns] = allocation[capacity_columns].fillna(0.0)
    allocation["capacity_reference_year"] = CAPACITY_REFERENCE_YEAR
    allocation["capacity_fixed_across_weather_years"] = True
    allocation["result_role"] = "supply_weather_replay_fixed_capacity"
    allocation["capacity_anchor_source"] = IRENA_RELEASE
    allocation["spatial_allocation_note"] = (
        "IRENA country capacity allocated across EIA-860 USA or GEM non-USA "
        "operating asset grids using reported capacity only as a spatial weight"
    )
    allocation = allocation.sort_values(
        ["era5_latitude", "era5_longitude", "country_area_code", "technology"],
        kind="stable",
    ).reset_index(drop=True)

    located_keys = totals[["country_area_code", "technology"]].drop_duplicates()
    unallocated = capacities.merge(
        located_keys.assign(has_spatial_anchor=True),
        on=["country_area_code", "technology"],
        how="left",
        validate="one_to_one",
    )
    unallocated["has_spatial_anchor"] = (
        unallocated["has_spatial_anchor"].eq(True).astype("boolean")
    )
    unallocated["unallocated_on_grid_capacity_mw"] = np.where(
        unallocated["has_spatial_anchor"], 0.0, unallocated["irena_on_grid_capacity_mw"]
    )
    unallocated["unallocated_all_grid_capacity_mw"] = np.where(
        unallocated["has_spatial_anchor"], 0.0, unallocated["irena_all_grid_capacity_mw"]
    )
    unallocated = unallocated.loc[
        unallocated[["unallocated_on_grid_capacity_mw", "unallocated_all_grid_capacity_mw"]]
        .fillna(0.0)
        .gt(0.0)
        .any(axis=1)
    ].reset_index(drop=True)

    coordinates = allocation[["era5_latitude", "era5_longitude"]].drop_duplicates()
    coordinates = coordinates.sort_values(
        ["era5_latitude", "era5_longitude"], kind="stable"
    ).reset_index(drop=True)
    coordinates.insert(
        0,
        "supply_era5_grid_id",
        [f"supply_g{index:05d}" for index in range(1, len(coordinates) + 1)],
    )
    allocation = allocation.merge(
        coordinates,
        on=["era5_latitude", "era5_longitude"],
        how="left",
        validate="many_to_one",
    )
    technology_flags = (
        allocation.groupby("supply_era5_grid_id", as_index=False)
        .agg(
            has_solar_pv=("technology", lambda values: bool((values == "solar_pv").any())),
            has_wind=("technology", lambda values: bool((values == "wind").any())),
            allocated_on_grid_capacity_mw=("allocated_on_grid_capacity_mw", "sum"),
            allocated_all_grid_capacity_mw=("allocated_all_grid_capacity_mw", "sum"),
            anchor_asset_count=("anchor_asset_count", "sum"),
        )
    )
    catalog = coordinates.merge(
        technology_flags,
        on="supply_era5_grid_id",
        how="left",
        validate="one_to_one",
    )
    catalog["capacity_reference_year"] = CAPACITY_REFERENCE_YEAR
    catalog["weather_start_utc"] = WEATHER_START
    catalog["weather_end_exclusive_utc"] = WEATHER_END_EXCLUSIVE
    catalog["expected_generation_intervals"] = EXPECTED_INTERVAL_COUNT
    catalog["expected_weather_endpoints"] = EXPECTED_WEATHER_ENDPOINTS
    catalog["grid_resolution_deg"] = GRID_RESOLUTION_DEG
    catalog["weather_source_planned"] = "CDS ERA5 ARCO geo-chunked"
    return allocation, catalog, unallocated


def hierarchy_checks(irena: pd.DataFrame) -> list[dict[str, Any]]:
    base = irena.loc[~irena["is_source_aggregate_area"]].copy()
    grouped = (
        base.groupby(["technology", "grid_connection"], as_index=False)[
            "installed_capacity_mw"
        ]
        .sum(min_count=1)
        .set_index(["technology", "grid_connection"])["installed_capacity_mw"]
    )
    checks = []
    for parent, children in [
        ("Solar energy", ["Solar photovoltaic", "Solar thermal energy"]),
        ("Wind energy", ["Onshore wind energy", "Offshore wind energy"]),
    ]:
        for grid_connection in ["OnGrid", "OffGrid"]:
            parent_value = grouped.get((parent, grid_connection), np.nan)
            child_values = [grouped.get((child, grid_connection), np.nan) for child in children]
            observed_children = [float(value) for value in child_values if pd.notna(value)]
            child_sum = sum(observed_children) if observed_children else np.nan
            difference = (
                float(child_sum - parent_value)
                if pd.notna(parent_value) and pd.notna(child_sum)
                else np.nan
            )
            checks.append(
                {
                    "parent": parent,
                    "children": children,
                    "grid_connection": grid_connection,
                    "parent_country_sum_mw": None
                    if pd.isna(parent_value)
                    else round(float(parent_value), 6),
                    "children_country_sum_mw": None
                    if pd.isna(child_sum)
                    else round(float(child_sum), 6),
                    "difference_mw": None if pd.isna(difference) else round(difference, 6),
                    "pass": bool(pd.isna(difference) or abs(difference) <= 0.05),
                }
            )
    return checks


def json_number(value: Any) -> int | float | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    rounded = round(number, 6)
    return int(rounded) if rounded.is_integer() else rounded


def build_validation(
    irena: pd.DataFrame,
    allocation: pd.DataFrame,
    catalog: pd.DataFrame,
    unallocated: pd.DataFrame,
    anchor_audit: dict[str, Any],
    paths: dict[str, Path],
    inputs: dict[str, Path],
) -> dict[str, Any]:
    hierarchy = hierarchy_checks(irena)
    allocation_checks = []
    for technology, scope in itertools.product(
        PRIMARY_TECHNOLOGY_MAP, ["on_grid", "all_grid"]
    ):
        subset = allocation[allocation["technology"].eq(technology)]
        allocated = float(subset[f"allocated_{scope}_capacity_mw"].sum())
        country_capacity = (
            subset[["country_area_code", f"irena_{scope}_capacity_mw"]]
            .drop_duplicates()[f"irena_{scope}_capacity_mw"]
            .sum()
        )
        difference = allocated - float(country_capacity)
        allocation_checks.append(
            {
                "technology": technology,
                "scope": scope,
                "allocated_capacity_mw": round(allocated, 6),
                "irena_capacity_in_countries_with_spatial_anchors_mw": round(
                    float(country_capacity), 6
                ),
                "difference_mw": round(difference, 6),
                "pass": abs(difference) <= max(0.01, abs(float(country_capacity)) * 1e-10),
            }
        )
    capacity_summary = []
    capacity_base = irena.loc[~irena["is_source_aggregate_area"]]
    for technology in TARGET_IRENA_TECHNOLOGIES:
        for grid_connection in ["OnGrid", "OffGrid"]:
            values = capacity_base.loc[
                capacity_base["technology"].eq(technology)
                & capacity_base["grid_connection"].eq(grid_connection),
                "installed_capacity_mw",
            ]
            capacity_summary.append(
                {
                    "technology": technology,
                    "grid_connection": grid_connection,
                    "nonaggregate_country_sum_mw": json_number(values.sum(min_count=1)),
                    "reported_country_cells": int(values.notna().sum()),
                }
            )
    unallocated_summary = []
    for technology in PRIMARY_TECHNOLOGY_MAP:
        row = unallocated[unallocated["technology"].eq(technology)]
        total_on_grid = float(
            capacity_base.loc[
                capacity_base["technology"].eq(PRIMARY_TECHNOLOGY_MAP[technology])
                & capacity_base["grid_connection"].eq("OnGrid"),
                "installed_capacity_mw",
            ].sum(min_count=1)
        )
        missing = float(row["unallocated_on_grid_capacity_mw"].sum())
        unallocated_summary.append(
            {
                "technology": technology,
                "unallocated_on_grid_capacity_mw": round(missing, 6),
                "total_on_grid_capacity_mw": round(total_on_grid, 6),
                "capacity_weighted_spatial_coverage_fraction": round(
                    1.0 - missing / total_on_grid, 10
                )
                if total_on_grid > 0
                else None,
            }
        )
    outputs = {
        name: {
            "relative_path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for name, path in paths.items()
        if path.exists()
    }
    input_files = {
        name: {
            "relative_path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for name, path in inputs.items()
    }
    hard_checks = {
        "irena_hierarchy_all_pass": all(item["pass"] for item in hierarchy),
        "allocation_all_pass": all(item["pass"] for item in allocation_checks),
        "catalog_unique_grid_ids": not catalog["supply_era5_grid_id"].duplicated().any(),
        "catalog_coordinates_valid": bool(
            catalog["era5_latitude"].between(-90.0, 90.0).all()
            and catalog["era5_longitude"].between(-180.0, 180.0).all()
        ),
        "allocated_capacity_nonnegative": bool(
            allocation[
                ["allocated_on_grid_capacity_mw", "allocated_all_grid_capacity_mw"]
            ]
            .fillna(0.0)
            .ge(0.0)
            .all()
            .all()
        ),
        "usa_anchor_is_eia_only": bool(
            allocation.loc[allocation["country_area_code"].eq("USA"), "anchor_sources"]
            .str.contains("EIA-860", regex=False)
            .all()
        ),
    }
    return {
        "schema_version": "1.0",
        "method_version": METHOD_VERSION,
        "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "result_role": "supply_weather_replay_fixed_capacity",
        "capacity_reference_year": CAPACITY_REFERENCE_YEAR,
        "capacity_fixed_across_weather_years": True,
        "weather_period": {
            "start_utc": WEATHER_START,
            "end_exclusive_utc": WEATHER_END_EXCLUSIVE,
            "generation_interval_count": EXPECTED_INTERVAL_COUNT,
            "interval_hours": 1,
            "weather_endpoint_count": EXPECTED_WEATHER_ENDPOINTS,
        },
        "main_capacity_scope": "IRENA on-grid solar photovoltaic plus wind",
        "is_historical_actual_generation": False,
        "data_center_demand_used": False,
        "input_files": input_files,
        "output_files": outputs,
        "anchor_audit": anchor_audit,
        "irena_capacity_summary": capacity_summary,
        "irena_parent_child_checks": hierarchy,
        "allocation_checks": allocation_checks,
        "unallocated_capacity": unallocated_summary,
        "grid_catalog": {
            "unique_grid_count": int(len(catalog)),
            "solar_grid_count": int(catalog["has_solar_pv"].sum()),
            "wind_grid_count": int(catalog["has_wind"].sum()),
            "co_located_solar_wind_grid_count": int(
                (catalog["has_solar_pv"] & catalog["has_wind"]).sum()
            ),
        },
        "hard_checks": hard_checks,
        "all_hard_checks_pass": all(hard_checks.values()),
        "limitations": [
            "IRENA country capacity is real reported year-end capacity; its within-country spatial allocation is a proxy based on EIA/GEM asset locations.",
            "GEM solar MW has mixed AC/DC rating and is used only as a relative spatial weight.",
            "GEM 2026-02 is a next-release proxy for 2025, not an exact 2025-12-31 asset snapshot.",
            "GEM operating rows with missing start year are retained in the central spatial allocation and separated in a known-start sensitivity.",
            "Solar thermal capacity is not assigned a PV capacity factor and is excluded from modeled PV generation.",
            "Capacity without an EIA/GEM spatial anchor remains unallocated rather than being placed at a fabricated location.",
            "On-grid capacity is physical grid-connected supply, not evidence of a data-centre PPA or deliverability right.",
        ],
    }


def main() -> None:
    args = parse_args()
    if args.offline and args.refresh_irena:
        raise ValueError("--offline and --refresh-irena cannot be combined")
    args.raw_root.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(args.irena_metadata.read_text(encoding="utf-8-sig"))
    query, response, retrieved_at = load_detailed_irena(
        metadata, args.raw_root, args.refresh_irena, args.offline
    )
    irena = parse_irena_jsonstat(response, retrieved_at)
    irena_path = args.output_root / "irena_country_2025_detailed_capacity.parquet"
    atomic_parquet(irena, irena_path)

    by_name, _ = country_lookup(irena)
    anchors, anchor_audit = build_spatial_anchors(args.gem, args.eia, by_name)
    capacities = irena_country_capacity(irena)
    allocation, catalog, unallocated = build_allocation(anchors, capacities)

    allocation_path = args.output_root / "fixed_2025_supply_grid_allocation.parquet"
    catalog_path = args.output_root / "supply_asset_grid_catalog_2025.csv"
    catalog_parquet_path = args.output_root / "supply_asset_grid_catalog_2025.parquet"
    unallocated_path = args.output_root / "unallocated_2025_capacity.parquet"
    atomic_parquet(allocation, allocation_path)
    atomic_csv(catalog, catalog_path)
    atomic_parquet(catalog, catalog_parquet_path)
    atomic_parquet(unallocated, unallocated_path)

    validation_path = args.output_root / "capacity_allocation_validation.json"
    request_path = args.raw_root / "country_capacity_2025_detailed_request.json"
    response_path = args.raw_root / "country_capacity_2025_detailed_response.json"
    paths = {
        "irena_detailed": irena_path,
        "allocation": allocation_path,
        "grid_catalog_csv": catalog_path,
        "grid_catalog_parquet": catalog_parquet_path,
        "unallocated": unallocated_path,
        "irena_request": request_path,
        "irena_response": response_path,
    }
    inputs = {
        "gem": args.gem,
        "eia": args.eia,
        "irena_metadata": args.irena_metadata,
    }
    validation = build_validation(
        irena,
        allocation,
        catalog,
        unallocated,
        anchor_audit,
        paths,
        inputs,
    )
    atomic_json(validation, validation_path)
    if not validation["all_hard_checks_pass"]:
        raise RuntimeError(f"Capacity allocation validation failed: {validation_path}")
    print(
        json.dumps(
            {
                "method_version": METHOD_VERSION,
                "grid_count": len(catalog),
                "allocation_rows": len(allocation),
                "unallocated_rows": len(unallocated),
                "validation": str(validation_path),
                "all_hard_checks_pass": validation["all_hard_checks_pass"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
