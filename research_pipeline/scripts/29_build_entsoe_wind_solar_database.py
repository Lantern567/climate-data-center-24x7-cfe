#!/usr/bin/env python3
"""Build the 2023--2025 ENTSO-E wind/solar actual-supply database.

Inputs are the official ENTSO-E Power Statistics CSV files for Net Generating
Capacity and Monthly Domestic Values.  The script retains only Solar, Wind
Onshore, and Wind Offshore.  Source gaps are described but never imputed.

The two output tables contain reported observations, not capacity inferred
from data-centre demand and not weather-modelled generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_ROOT = (
    ROOT
    / "data"
    / "24x7_cfe"
    / "raw"
    / "supply_assets"
    / "entsoe"
    / "power_statistics"
)
DEFAULT_OUTPUT_ROOT = (
    ROOT
    / "data"
    / "24x7_cfe"
    / "processed"
    / "wind_solar_supply_2023_2025"
)
YEARS = (2023, 2024, 2025)
TECHNOLOGIES = ("Solar", "Wind Onshore", "Wind Offshore")
POWER_STATISTICS_PAGE = "https://www.entsoe.eu/data/power-stats/"
MONTHLY_GUIDELINES_URL = (
    "https://eepublicdownloads.entsoe.eu/clean-documents/Publications/"
    "Statistics/MS_guidelines2016.pdf"
)
CAPACITY_DESCRIPTION_URL = (
    "https://transparencyplatform.zendesk.com/hc/en-us/articles/"
    "16648300912916-Installed-Generation-Capacity-Aggregated-14-1-A"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def source_url(kind: str, year: int) -> str:
    return (
        "https://www.entsoe.eu/publications/data/power-stats/"
        f"{year}/{kind}_{year}.csv"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_official_csv(path: Path) -> pd.DataFrame:
    """Read ENTSO-E CSV despite delimiter changes between releases."""
    first_line = path.open("r", encoding="utf-8-sig").readline()
    delimiter = "\t" if first_line.count("\t") > first_line.count(",") else ","
    return pd.read_csv(path, sep=delimiter, encoding="utf-8-sig")


def nullable_string(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().replace("", pd.NA)


def require_columns(frame: pd.DataFrame, expected: set[str], path: Path) -> None:
    missing = expected.difference(frame.columns)
    if missing:
        raise ValueError(f"{path.name} lacks required columns: {sorted(missing)}")


def read_capacity(raw_root: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    sources: list[dict[str, Any]] = []
    for year in YEARS:
        path = raw_root / f"net_generation_capacity_{year}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        raw = read_official_csv(path)
        require_columns(
            raw,
            {
                "MeasureItem",
                "MeasureItemCategoryID",
                "MeasureItemID",
                "Category",
                "Country",
                "Year",
                "Representativity",
                "ProvidedValue",
                "ProvidedValueCode",
            },
            path,
        )
        selected = raw.loc[raw["Category"].isin(TECHNOLOGIES)].copy()
        if selected.empty:
            raise ValueError(f"{path.name} contains no selected wind/solar rows")
        selected["source_file"] = path.name
        selected["source_url"] = source_url("net_generation_capacity", year)
        if "CreationDate" not in selected:
            selected["CreationDate"] = pd.NA
        frames.append(selected)
        sources.append(
            {
                "dataset": "Net Generating Capacity",
                "year": year,
                "file": str(path.relative_to(ROOT)),
                "url": source_url("net_generation_capacity", year),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    raw_all = pd.concat(frames, ignore_index=True)
    capacity = pd.DataFrame(
        {
            "country_code": nullable_string(raw_all["Country"]),
            "year": pd.to_numeric(raw_all["Year"], errors="coerce").astype("Int16"),
            "technology": nullable_string(raw_all["Category"]),
            "technology_code": nullable_string(raw_all["MeasureItemID"]),
            "capacity_mw": pd.to_numeric(raw_all["ProvidedValue"], errors="coerce"),
            "unit": pd.Series("MW", index=raw_all.index, dtype="string"),
            "representativity_pct": pd.to_numeric(
                raw_all["Representativity"], errors="coerce"
            ).astype("Float64"),
            "provided_value_code": nullable_string(raw_all["ProvidedValueCode"]),
            "creation_date_raw": nullable_string(raw_all["CreationDate"]),
            "measure_item": nullable_string(raw_all["MeasureItem"]),
            "measure_item_category_id": pd.to_numeric(
                raw_all["MeasureItemCategoryID"], errors="coerce"
            ).astype("Int16"),
            "observation_status": pd.Series(
                "reported_actual", index=raw_all.index, dtype="string"
            ),
            "contains_imputation": pd.Series(
                False, index=raw_all.index, dtype="boolean"
            ),
            "source_dataset": pd.Series(
                "ENTSO-E Power Statistics / Net Generating Capacity",
                index=raw_all.index,
                dtype="string",
            ),
            "source_file": nullable_string(raw_all["source_file"]),
            "source_url": nullable_string(raw_all["source_url"]),
        }
    )
    capacity = capacity.sort_values(
        ["year", "country_code", "technology"], kind="stable"
    ).reset_index(drop=True)
    return capacity, sources


def build_series_inventory(generation: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    keys = ["country_code", "year", "technology"]
    for key, group in generation.groupby(keys, dropna=False, sort=True):
        observed = sorted(
            {int(month) for month in group["month"].dropna().tolist()}
        )
        missing = [month for month in range(1, 13) if month not in observed]
        records.append(
            {
                "country_code": key[0],
                "year": int(key[1]),
                "technology": key[2],
                "series_row_count": int(len(group)),
                "observed_month_count": len(observed),
                "observed_months": observed,
                "missing_month_count": len(missing),
                "missing_months": missing,
                "series_complete_12_months": len(missing) == 0,
                "series_has_duplicate_months": len(group) != len(observed),
            }
        )
    return pd.DataFrame.from_records(records)


def read_generation(raw_root: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    sources: list[dict[str, Any]] = []
    for year in YEARS:
        path = raw_root / f"monthly_domestic_values_{year}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        raw = read_official_csv(path)
        area_column = "Country" if "Country" in raw.columns else "Area"
        require_columns(
            raw,
            {
                "MeasureItem",
                "Month",
                "Year",
                "Category",
                area_column,
                "MeasureItemID",
                "Representativity",
                "ProvidedValue",
                "ProvidedValueCode",
            },
            path,
        )
        selected = raw.loc[raw["Category"].isin(TECHNOLOGIES)].copy()
        if selected.empty:
            raise ValueError(f"{path.name} contains no selected wind/solar rows")
        selected = selected.rename(columns={area_column: "country_code_raw"})
        selected["source_file"] = path.name
        selected["source_url"] = source_url("monthly_domestic_values", year)
        if "CreationDate" not in selected:
            selected["CreationDate"] = pd.NA
        frames.append(selected)
        sources.append(
            {
                "dataset": "Monthly Domestic Values",
                "year": year,
                "file": str(path.relative_to(ROOT)),
                "url": source_url("monthly_domestic_values", year),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    raw_all = pd.concat(frames, ignore_index=True)
    generation = pd.DataFrame(
        {
            "country_code": nullable_string(raw_all["country_code_raw"]),
            "year": pd.to_numeric(raw_all["Year"], errors="coerce").astype("Int16"),
            "month": pd.to_numeric(raw_all["Month"], errors="coerce").astype("Int8"),
            "technology": nullable_string(raw_all["Category"]),
            "technology_code": nullable_string(raw_all["MeasureItemID"]),
            "generation_gwh": pd.to_numeric(raw_all["ProvidedValue"], errors="coerce"),
            "unit": pd.Series("GWh", index=raw_all.index, dtype="string"),
            "representativity_pct": pd.to_numeric(
                raw_all["Representativity"], errors="coerce"
            ).astype("Float64"),
            "provided_value_code": nullable_string(raw_all["ProvidedValueCode"]),
            "creation_date_raw": nullable_string(raw_all["CreationDate"]),
            "measure_item": nullable_string(raw_all["MeasureItem"]),
            "observation_status": pd.Series(
                "reported_actual", index=raw_all.index, dtype="string"
            ),
            "contains_imputation": pd.Series(
                False, index=raw_all.index, dtype="boolean"
            ),
            "source_dataset": pd.Series(
                "ENTSO-E Power Statistics / Monthly Domestic Values",
                index=raw_all.index,
                dtype="string",
            ),
            "source_file": nullable_string(raw_all["source_file"]),
            "source_url": nullable_string(raw_all["source_url"]),
        }
    )
    generation["period"] = (
        generation["year"].astype("string")
        + "-"
        + generation["month"].astype("string").str.zfill(2)
    ).astype("string")
    inventory = build_series_inventory(generation)
    generation = generation.merge(
        inventory,
        on=["country_code", "year", "technology"],
        how="left",
        validate="many_to_one",
    )
    generation = generation.sort_values(
        ["year", "country_code", "technology", "month"], kind="stable"
    ).reset_index(drop=True)
    return generation, sources


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is pd.NA or value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if math.isnan(float(value)) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json_ready(frame.to_dict(orient="records"))


def technology_counts(frame: pd.DataFrame) -> list[dict[str, Any]]:
    result = (
        frame.groupby(["year", "technology"], observed=True)
        .size()
        .rename("row_count")
        .reset_index()
    )
    return records(result)


def country_counts(frame: pd.DataFrame) -> list[dict[str, Any]]:
    result = (
        frame.groupby("year", observed=True)["country_code"]
        .nunique()
        .rename("country_count")
        .reset_index()
    )
    return records(result)


def key_set(frame: pd.DataFrame) -> set[tuple[str, int, str]]:
    return {
        (str(row.country_code), int(row.year), str(row.technology))
        for row in frame[["country_code", "year", "technology"]]
        .drop_duplicates()
        .itertuples(index=False)
    }


def build_validation(
    capacity: pd.DataFrame,
    generation: pd.DataFrame,
    source_files: list[dict[str, Any]],
    capacity_path: Path,
    generation_path: Path,
) -> dict[str, Any]:
    capacity_key = ["country_code", "year", "technology"]
    generation_key = ["country_code", "year", "month", "technology"]
    series_columns = [
        "country_code",
        "year",
        "technology",
        "series_row_count",
        "observed_month_count",
        "observed_months",
        "missing_month_count",
        "missing_months",
        "series_complete_12_months",
        "series_has_duplicate_months",
    ]
    series = generation[series_columns].drop_duplicates(
        ["country_code", "year", "technology"]
    )
    incomplete = series.loc[~series["series_complete_12_months"]].copy()

    positive_capacity = capacity.loc[capacity["capacity_mw"] > 0]
    capacity_pairs = key_set(positive_capacity)
    generation_pairs = key_set(generation)
    missing_generation_pairs = sorted(capacity_pairs.difference(generation_pairs))
    generation_without_capacity = sorted(generation_pairs.difference(key_set(capacity)))

    capacity_totals = (
        capacity.groupby(["year", "technology"], observed=True)["capacity_mw"]
        .sum(min_count=1)
        .rename("reported_capacity_sum_mw")
        .reset_index()
    )
    generation_totals = (
        generation.groupby(["year", "technology"], observed=True)["generation_gwh"]
        .sum(min_count=1)
        .rename("sum_of_reported_months_gwh")
        .reset_index()
    )
    month_count_distribution = (
        series.groupby(["year", "observed_month_count"], observed=True)
        .size()
        .rename("series_count")
        .reset_index()
    )

    capacity_duplicate_count = int(capacity.duplicated(capacity_key).sum())
    generation_duplicate_count = int(generation.duplicated(generation_key).sum())
    capacity_null_values = int(capacity["capacity_mw"].isna().sum())
    generation_null_values = int(generation["generation_gwh"].isna().sum())
    invalid_years = sorted(
        {
            int(value)
            for value in pd.concat([capacity["year"], generation["year"]])
            .dropna()
            .unique()
            if int(value) not in YEARS
        }
    )
    invalid_technologies = sorted(
        set(capacity["technology"].dropna().astype(str))
        .union(set(generation["technology"].dropna().astype(str)))
        .difference(TECHNOLOGIES)
    )
    invalid_month_count = int(
        (~generation["month"].between(1, 12, inclusive="both")).sum()
    )
    units = {
        "capacity": sorted(capacity["unit"].dropna().astype(str).unique()),
        "generation": sorted(generation["unit"].dropna().astype(str).unique()),
    }
    hard_checks_passed = not any(
        [
            capacity_duplicate_count,
            generation_duplicate_count,
            capacity_null_values,
            generation_null_values,
            invalid_years,
            invalid_technologies,
            invalid_month_count,
            units["capacity"] != ["MW"],
            units["generation"] != ["GWh"],
        ]
    )
    source_gaps_present = bool(len(incomplete) or missing_generation_pairs)

    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": (
            "PASS_WITH_SOURCE_GAPS"
            if hard_checks_passed and source_gaps_present
            else "PASS"
            if hard_checks_passed
            else "FAIL"
        ),
        "hard_checks_passed": hard_checks_passed,
        "actual_database": True,
        "contains_modeled_values": False,
        "contains_imputed_values": False,
        "scope": {
            "years": list(YEARS),
            "technologies": list(TECHNOLOGIES),
            "spatial_granularity": "country",
            "capacity_time_granularity": "annual",
            "generation_time_granularity": "monthly",
        },
        "interpretation": {
            "capacity": (
                "Reported country-level net generating capacity from ENTSO-E "
                "Power Statistics; values are MW."
            ),
            "generation": (
                "Reported country-level monthly actual net generation by "
                "production type from Monthly Domestic Values; values are GWh."
            ),
            "missing_data_policy": (
                "No filling or interpolation. Output rows are only source "
                "observations; observed and missing months are explicitly listed."
            ),
            "representativity": (
                "Representativity is retained exactly as supplied and expressed "
                "as a percentage field."
            ),
        },
        "official_references": {
            "power_statistics_page": POWER_STATISTICS_PAGE,
            "monthly_domestic_values_unit_and_definition": MONTHLY_GUIDELINES_URL,
            "installed_capacity_unit_and_definition": CAPACITY_DESCRIPTION_URL,
        },
        "source_files": source_files,
        "outputs": {
            "capacity": {
                "file": str(capacity_path.relative_to(ROOT)),
                "rows": int(len(capacity)),
                "columns": list(capacity.columns),
            },
            "generation": {
                "file": str(generation_path.relative_to(ROOT)),
                "rows": int(len(generation)),
                "columns": list(generation.columns),
            },
        },
        "capacity_validation": {
            "rows": int(len(capacity)),
            "countries_by_year": country_counts(capacity),
            "rows_by_year_and_technology": technology_counts(capacity),
            "reported_totals_by_year_and_technology": records(capacity_totals),
            "units": units["capacity"],
            "duplicate_country_year_technology_keys": capacity_duplicate_count,
            "null_capacity_values": capacity_null_values,
            "negative_capacity_values": int((capacity["capacity_mw"] < 0).sum()),
            "representativity_values_pct": sorted(
                float(value)
                for value in capacity["representativity_pct"].dropna().unique()
            ),
        },
        "generation_validation": {
            "rows": int(len(generation)),
            "countries_by_year": country_counts(generation),
            "rows_by_year_and_technology": technology_counts(generation),
            "sum_of_reported_months_by_year_and_technology": records(
                generation_totals
            ),
            "units": units["generation"],
            "duplicate_country_year_month_technology_keys": generation_duplicate_count,
            "null_generation_values": generation_null_values,
            "negative_generation_values": int((generation["generation_gwh"] < 0).sum()),
            "invalid_month_values": invalid_month_count,
            "series_count": int(len(series)),
            "complete_12_month_series": int(
                series["series_complete_12_months"].sum()
            ),
            "incomplete_series_count": int(len(incomplete)),
            "month_count_distribution": records(month_count_distribution),
            "incomplete_series": records(incomplete),
            "representativity_values_pct": sorted(
                float(value)
                for value in generation["representativity_pct"].dropna().unique()
            ),
        },
        "cross_dataset_coverage": {
            "positive_capacity_series_without_any_generation_rows_count": len(
                missing_generation_pairs
            ),
            "positive_capacity_series_without_any_generation_rows": [
                {"country_code": item[0], "year": item[1], "technology": item[2]}
                for item in missing_generation_pairs
            ],
            "generation_series_without_capacity_row_count": len(
                generation_without_capacity
            ),
            "generation_series_without_capacity_row": [
                {"country_code": item[0], "year": item[1], "technology": item[2]}
                for item in generation_without_capacity
            ],
        },
        "schema_checks": {
            "invalid_years": invalid_years,
            "invalid_technologies": invalid_technologies,
        },
        "source_gap_note": (
            "PASS_WITH_SOURCE_GAPS means the transformation and schema checks "
            "passed, while the official source has incomplete monthly series or "
            "cross-table coverage differences. These gaps remain unfilled."
        ),
    }


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root.resolve()
    output_root = args.output_root.resolve()
    capacity, capacity_sources = read_capacity(raw_root)
    generation, generation_sources = read_generation(raw_root)

    capacity_path = output_root / "entsoe_country_year_capacity_2023_2025.parquet"
    generation_path = (
        output_root / "entsoe_country_month_generation_2023_2025.parquet"
    )
    validation_path = output_root / "entsoe_validation.json"
    atomic_parquet(capacity, capacity_path)
    atomic_parquet(generation, generation_path)
    validation = build_validation(
        capacity,
        generation,
        capacity_sources + generation_sources,
        capacity_path,
        generation_path,
    )
    atomic_json(validation, validation_path)

    print(
        json.dumps(
            {
                "status": validation["status"],
                "capacity_rows": len(capacity),
                "generation_rows": len(generation),
                "incomplete_monthly_series": validation["generation_validation"][
                    "incomplete_series_count"
                ],
                "capacity_output": str(capacity_path),
                "generation_output": str(generation_path),
                "validation_output": str(validation_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
