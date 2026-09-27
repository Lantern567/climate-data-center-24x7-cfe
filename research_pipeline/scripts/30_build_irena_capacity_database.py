#!/usr/bin/env python3
"""Build the official IRENASTAT 2023--2025 wind/solar capacity database.

The output is a capacity table, not electricity generation and not capacity
inferred from data-centre demand.  It keeps IRENA's On-grid and Off-grid
observations separate.  Official global totals are independently validated
against the IRENASTAT Region table (World, On-grid + Off-grid).
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_ROOT = (
    ROOT
    / "data"
    / "24x7_cfe"
    / "raw"
    / "supply_assets"
    / "irena"
    / "2026-h1"
)
DEFAULT_OUTPUT_ROOT = (
    ROOT
    / "data"
    / "24x7_cfe"
    / "processed"
    / "wind_solar_supply_2023_2025"
)

API_ROOT = (
    "https://pxweb.irena.org/api/v1/en/IRENASTAT/"
    "Power%20Capacity%20and%20Generation"
)
COUNTRY_URL = f"{API_ROOT}/Country_ELECCAP_2026_H1_v-PX%201.px"
REGION_URL = f"{API_ROOT}/Region_ELECCAP_2026_H1_v-PX%201.px"
TERMS_URL = "https://www.irena.org/terms-and-conditions"
YEARS = (2023, 2024, 2025)
TARGET_TECHNOLOGIES = ("Solar energy", "Wind energy")
SOURCE_RELEASE = "IRENASTAT 2026 H1"
USER_AGENT = "wind-solar-research-database/1.0"

# Published values returned by the official World-region API selection at the
# time this database was built.  They are an explicit regression test rather
# than a country-sum estimate.
EXPECTED_WORLD_TOTAL_MW = {
    (2023, "Solar energy"): 1_422_025.58,
    (2024, "Solar energy"): 1_880_396.38,
    (2025, "Solar energy"): 2_391_584.49,
    (2023, "Wind energy"): 1_018_224.49,
    (2024, "Wind energy"): 1_132_689.94,
    (2025, "Wind energy"): 1_291_367.86,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Fetch fresh API responses even when raw JSON already exists.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use saved raw JSON only; fail if any required file is absent.",
    )
    return parser.parse_args()


def atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_json(url: str, payload: dict[str, Any] | None = None) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    errors: list[str] = []
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return json.loads(response.read().decode("utf-8-sig"))
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            errors.append(f"attempt {attempt}: {exc}")
            if attempt < 3:
                time.sleep(attempt * 2)
    raise RuntimeError(f"Failed to retrieve {url}: {'; '.join(errors)}")


def load_or_fetch(
    path: Path,
    url: str,
    payload: dict[str, Any] | None,
    refresh: bool,
    offline: bool,
) -> Any:
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    if offline:
        raise FileNotFoundError(f"Offline raw input is missing: {path}")
    result = fetch_json(url, payload)
    atomic_json(result, path)
    return result


def variable(metadata: dict[str, Any], predicate: Any) -> dict[str, Any]:
    matches = [item for item in metadata["variables"] if predicate(item)]
    if len(matches) != 1:
        codes = [item.get("code") for item in metadata.get("variables", [])]
        raise ValueError(f"Expected one metadata variable, found {len(matches)}; {codes}")
    return matches[0]


def normalized(value: Any) -> str:
    return " ".join(str(value).strip().casefold().split())


def codes_for_texts(item: dict[str, Any], requested: tuple[str, ...]) -> list[str]:
    lookup = {
        normalized(label): str(code)
        for code, label in zip(item["values"], item["valueTexts"], strict=True)
    }
    missing = [label for label in requested if normalized(label) not in lookup]
    if missing:
        raise ValueError(f"{item['code']} lacks requested labels: {missing}")
    return [lookup[normalized(label)] for label in requested]


def selection_metadata(metadata: dict[str, Any], area_kind: str) -> dict[str, Any]:
    technology = variable(
        metadata, lambda item: normalized(item.get("text")) == "technology"
    )
    grid = variable(
        metadata, lambda item: normalized(item.get("text")) == "grid connection"
    )
    year = variable(metadata, lambda item: normalized(item.get("text")) == "year")
    if area_kind == "country":
        area = variable(
            metadata,
            lambda item: normalized(item.get("text")) in {"country/area", "country area"},
        )
        area_values = ["*"]
        area_filter = "all"
    else:
        area = variable(
            metadata,
            lambda item: "world" in {normalized(label) for label in item.get("valueTexts", [])},
        )
        area_values = codes_for_texts(area, ("World",))
        area_filter = "item"
    return {
        "area": area,
        "area_values": area_values,
        "area_filter": area_filter,
        "technology": technology,
        "technology_values": codes_for_texts(technology, TARGET_TECHNOLOGIES),
        "grid": grid,
        "year": year,
        "year_values": codes_for_texts(year, tuple(str(year) for year in YEARS)),
    }


def make_query(metadata: dict[str, Any], area_kind: str) -> dict[str, Any]:
    selected = selection_metadata(metadata, area_kind)
    return {
        "query": [
            {
                "code": selected["area"]["code"],
                "selection": {
                    "filter": selected["area_filter"],
                    "values": selected["area_values"],
                },
            },
            {
                "code": selected["technology"]["code"],
                "selection": {
                    "filter": "item",
                    "values": selected["technology_values"],
                },
            },
            {
                "code": selected["grid"]["code"],
                "selection": {"filter": "all", "values": ["*"]},
            },
            {
                "code": selected["year"]["code"],
                "selection": {
                    "filter": "item",
                    "values": selected["year_values"],
                },
            },
        ],
        "response": {"format": "json-stat2"},
    }


def ordered_categories(dimension: dict[str, Any], expected_size: int) -> list[str]:
    category = dimension["category"]
    index = category.get("index")
    if isinstance(index, dict):
        result = [code for code, _ in sorted(index.items(), key=lambda pair: pair[1])]
    elif isinstance(index, list):
        result = [str(code) for code in index]
    else:
        labels = category.get("label", {})
        result = list(labels)
    if len(result) != expected_size:
        raise ValueError(
            f"Dimension {dimension.get('label')} has {len(result)} categories, "
            f"expected {expected_size}"
        )
    return result


def parse_jsonstat2(dataset: dict[str, Any]) -> pd.DataFrame:
    dimensions = list(dataset["id"])
    sizes = [int(value) for value in dataset["size"]]
    if len(dimensions) != len(sizes):
        raise ValueError("JSON-stat2 id/size lengths differ")
    codes: dict[str, list[str]] = {}
    labels: dict[str, dict[str, str]] = {}
    for name, size in zip(dimensions, sizes, strict=True):
        dimension = dataset["dimension"][name]
        codes[name] = ordered_categories(dimension, size)
        labels[name] = {
            str(code): str(label)
            for code, label in dimension["category"].get("label", {}).items()
        }

    raw_values = dataset.get("value", [])
    cell_count = math.prod(sizes)
    if isinstance(raw_values, list) and len(raw_values) != cell_count:
        raise ValueError(f"JSON-stat2 has {len(raw_values)} values for {cell_count} cells")

    records: list[dict[str, Any]] = []
    products = itertools.product(*(codes[name] for name in dimensions))
    for flat_index, combination in enumerate(products):
        if isinstance(raw_values, list):
            value = raw_values[flat_index]
        else:
            value = raw_values.get(str(flat_index), raw_values.get(flat_index))
        row: dict[str, Any] = {"value": value}
        for name, code in zip(dimensions, combination, strict=True):
            row[f"{name}__code"] = code
            row[f"{name}__label"] = labels[name].get(code, code)
        records.append(row)
    return pd.DataFrame.from_records(records)


def find_dimension(frame: pd.DataFrame, wanted: str) -> str:
    suffix = "__label"
    candidates = [column[: -len(suffix)] for column in frame if column.endswith(suffix)]
    if wanted == "area":
        excluded = {"technology", "grid connection", "year"}
        matches = [name for name in candidates if normalized(name) not in excluded]
    else:
        matches = [name for name in candidates if normalized(name) == wanted]
    if len(matches) != 1:
        raise ValueError(f"Cannot identify {wanted} dimension from {candidates}")
    return matches[0]


def build_country_table(
    dataset: dict[str, Any], retrieved_at: str
) -> pd.DataFrame:
    raw = parse_jsonstat2(dataset)
    area = find_dimension(raw, "area")
    technology = find_dimension(raw, "technology")
    grid = find_dimension(raw, "grid connection")
    year = find_dimension(raw, "year")
    capacity = pd.DataFrame(
        {
            "country_area_code": raw[f"{area}__code"].astype("string"),
            "country_area": raw[f"{area}__label"].astype("string"),
            "technology_code": raw[f"{technology}__code"].astype("string"),
            "technology": raw[f"{technology}__label"].astype("string"),
            "grid_connection_code": raw[f"{grid}__code"].astype("string"),
            "grid_connection": raw[f"{grid}__label"].astype("string"),
            "year": pd.to_numeric(raw[f"{year}__label"], errors="raise").astype("Int16"),
            "installed_capacity_mw": pd.to_numeric(raw["value"], errors="coerce").astype("Float64"),
        }
    )
    # The source's Country/area dimension includes the explicit regional
    # aggregates Eurasia (REA) and Oceania (OCA).  Flag them so downstream
    # country aggregation cannot silently double count those rows.
    capacity["is_source_aggregate_area"] = capacity["country_area_code"].isin(
        {"REA", "OCA"}
    ).astype("boolean")
    capacity["unit"] = pd.Series("MW", index=capacity.index, dtype="string")
    capacity["capacity_measure"] = pd.Series(
        "maximum_net_generating_capacity_at_calendar_year_end",
        index=capacity.index,
        dtype="string",
    )
    capacity["observation_type"] = pd.Series(
        "official_reported_capacity", index=capacity.index, dtype="string"
    )
    capacity["is_reported_value"] = capacity["installed_capacity_mw"].notna().astype("boolean")
    capacity["source_dataset"] = pd.Series(
        "IRENASTAT Electricity Capacity by Country/area",
        index=capacity.index,
        dtype="string",
    )
    capacity["source_release"] = pd.Series(SOURCE_RELEASE, index=capacity.index, dtype="string")
    capacity["source_url"] = pd.Series(COUNTRY_URL, index=capacity.index, dtype="string")
    capacity["source_terms_url"] = pd.Series(TERMS_URL, index=capacity.index, dtype="string")
    capacity["source_retrieved_at_utc"] = pd.Series(
        retrieved_at, index=capacity.index, dtype="string"
    )
    return capacity.sort_values(
        ["year", "country_area_code", "technology", "grid_connection"], kind="stable"
    ).reset_index(drop=True)


def world_totals(dataset: dict[str, Any]) -> pd.DataFrame:
    raw = parse_jsonstat2(dataset)
    area = find_dimension(raw, "area")
    technology = find_dimension(raw, "technology")
    grid = find_dimension(raw, "grid connection")
    year = find_dimension(raw, "year")
    frame = pd.DataFrame(
        {
            "region": raw[f"{area}__label"],
            "technology": raw[f"{technology}__label"],
            "grid_connection": raw[f"{grid}__label"],
            "year": pd.to_numeric(raw[f"{year}__label"], errors="raise").astype(int),
            "capacity_mw": pd.to_numeric(raw["value"], errors="coerce"),
        }
    )
    if set(frame["region"]) != {"World"}:
        raise ValueError(f"World validation selected unexpected regions: {set(frame['region'])}")
    return (
        frame.groupby(["year", "technology"], as_index=False, dropna=False)["capacity_mw"]
        .sum(min_count=1)
        .sort_values(["year", "technology"], kind="stable")
        .reset_index(drop=True)
    )


def json_number(value: Any) -> int | float | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    number = round(number, 6)
    return int(number) if number.is_integer() else number


def build_validation(
    capacity: pd.DataFrame,
    world: pd.DataFrame,
    raw_root: Path,
    output_path: Path,
    country_metadata: dict[str, Any],
    region_metadata: dict[str, Any],
) -> dict[str, Any]:
    key = [
        "country_area_code",
        "technology_code",
        "grid_connection_code",
        "year",
    ]
    validations: list[dict[str, Any]] = []
    for row in world.itertuples(index=False):
        expected = EXPECTED_WORLD_TOTAL_MW[(int(row.year), str(row.technology))]
        observed = float(row.capacity_mw)
        difference = observed - expected
        validations.append(
            {
                "year": int(row.year),
                "technology": str(row.technology),
                "observed_world_on_plus_off_grid_mw": round(observed, 6),
                "expected_world_on_plus_off_grid_mw": expected,
                "difference_mw": round(difference, 6),
                "absolute_tolerance_mw": 0.01,
                "pass": abs(difference) <= 0.01,
            }
        )

    coverage: list[dict[str, Any]] = []
    for (year, technology, grid), group in capacity.groupby(
        ["year", "technology", "grid_connection"], dropna=False, sort=True
    ):
        coverage.append(
            {
                "year": int(year),
                "technology": str(technology),
                "grid_connection": str(grid),
                "selected_country_area_count": int(len(group)),
                "reported_value_count": int(group["installed_capacity_mw"].notna().sum()),
                "missing_value_count": int(group["installed_capacity_mw"].isna().sum()),
                "reported_capacity_mw_unvalidated_country_area_sum": json_number(
                    group["installed_capacity_mw"].sum(min_count=1)
                ),
            }
        )

    raw_files = {}
    for path in sorted(raw_root.glob("*.json")):
        if path.name == "irena_validation.json":
            continue
        raw_files[path.name] = {
            "relative_path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }

    return {
        "schema_version": "1.0",
        "dataset_scope": "IRENASTAT country/area wind and solar year-end capacity, 2023-2025",
        "source_release": SOURCE_RELEASE,
        "source_urls": {"country": COUNTRY_URL, "world_validation": REGION_URL},
        "source_metadata": {
            "country_title": country_metadata.get("title"),
            "region_title": region_metadata.get("title"),
            "country_updated": country_metadata.get("updated"),
            "region_updated": region_metadata.get("updated"),
            "unit": "MW",
        },
        "raw_files": raw_files,
        "output": {
            "relative_path": str(output_path.relative_to(ROOT)),
            "bytes": output_path.stat().st_size,
            "sha256": sha256(output_path),
            "row_count": int(len(capacity)),
            "reported_value_count": int(capacity["installed_capacity_mw"].notna().sum()),
            "missing_value_count": int(capacity["installed_capacity_mw"].isna().sum()),
            "columns": list(capacity.columns),
            "duplicate_primary_key_count": int(capacity.duplicated(key, keep=False).sum()),
            "negative_capacity_count": int((capacity["installed_capacity_mw"] < 0).fillna(False).sum()),
            "unexpected_year_count": int((~capacity["year"].isin(YEARS)).sum()),
            "unexpected_technology_count": int(
                (~capacity["technology"].isin(TARGET_TECHNOLOGIES)).sum()
            ),
            "source_aggregate_area_codes": sorted(
                capacity.loc[
                    capacity["is_source_aggregate_area"], "country_area_code"
                ].dropna().unique().tolist()
            ),
        },
        "coverage_by_year_technology_grid": coverage,
        "official_world_region_validation": {
            "method": "sum IRENASTAT Region=World On-grid and Off-grid cells; do not sum Country/area rows",
            "checks": validations,
            "all_pass": all(item["pass"] for item in validations),
        },
        "limitations": [
            "This table is installed capacity at calendar-year end, not electricity generation and not weather-derived output.",
            "Missing source cells remain null; no capacity is imputed.",
            "On-grid and Off-grid remain separate in the parquet; combine only when the analysis explicitly needs total capacity.",
            "Country/area includes the source aggregates Eurasia (REA) and Oceania (OCA); they are explicitly flagged and its row sum is not used as the official world total.",
            "IRENA terms and conditions require attribution and impose reuse conditions; this extract does not change those terms.",
        ],
    }


def main() -> None:
    args = parse_args()
    if args.refresh and args.offline:
        raise ValueError("--refresh and --offline cannot be combined")
    args.raw_root.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)

    retrieved_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    country_metadata_path = args.raw_root / "country_capacity_metadata.json"
    region_metadata_path = args.raw_root / "region_capacity_metadata.json"
    country_metadata = load_or_fetch(
        country_metadata_path, COUNTRY_URL, None, args.refresh, args.offline
    )
    region_metadata = load_or_fetch(
        region_metadata_path, REGION_URL, None, args.refresh, args.offline
    )

    country_query = make_query(country_metadata, "country")
    region_query = make_query(region_metadata, "region")
    country_request_path = args.raw_root / "country_capacity_request.json"
    region_request_path = args.raw_root / "world_capacity_request.json"
    atomic_json(country_query, country_request_path)
    atomic_json(region_query, region_request_path)

    country_response_path = args.raw_root / "country_capacity_response.json"
    region_response_path = args.raw_root / "world_capacity_response.json"
    country_response = load_or_fetch(
        country_response_path, COUNTRY_URL, country_query, args.refresh, args.offline
    )
    region_response = load_or_fetch(
        region_response_path, REGION_URL, region_query, args.refresh, args.offline
    )

    capacity = build_country_table(country_response, retrieved_at)
    world = world_totals(region_response)
    output_path = args.output_root / "irena_country_year_capacity_2023_2025.parquet"
    atomic_parquet(capacity, output_path)

    validation = build_validation(
        capacity,
        world,
        args.raw_root,
        output_path,
        country_metadata,
        region_metadata,
    )
    validation_path = args.output_root / "irena_validation.json"
    atomic_json(validation, validation_path)
    # Also keep the validation beside the immutable raw API evidence requested
    # for this source release.
    atomic_json(validation, args.raw_root / "irena_validation.json")

    print(
        json.dumps(
            {
                "output": str(output_path),
                "rows": len(capacity),
                "reported_values": int(capacity["installed_capacity_mw"].notna().sum()),
                "world_validation_pass": validation["official_world_region_validation"]["all_pass"],
                "validation": str(validation_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
