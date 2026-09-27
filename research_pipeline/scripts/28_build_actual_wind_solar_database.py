#!/usr/bin/env python3
"""Build traceable 2023--2025 wind/solar capacity and generation databases.

The pipeline deliberately keeps four evidence layers separate:

1. GEM project/phase locations and the current-release operating cohort;
2. GEM country commissioning-year summaries;
3. Ember monthly installed capacity and annual observed generation;
4. EIA generator-level capacity and plant/technology/month observed generation.

No data-center demand is used to infer installed capacity.  The GEM 2026 map
cannot reproduce a complete historical stock because it omits assets retired
before that release and because some operating phases have no start year.  Its
2023--2025 reconstructions are therefore explicitly labelled survivor lower
bounds, not historical snapshots.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import shutil
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = ROOT / "data" / "24x7_cfe" / "raw" / "supply_assets"
DEFAULT_OUTPUT = (
    ROOT
    / "data"
    / "24x7_cfe"
    / "processed"
    / "wind_solar_supply_2023_2025"
)
YEARS = (2023, 2024, 2025)
RETRIEVED_DATE = "2026-08-06"

GEM_ROOT = RAW_ROOT / "gem" / "2026-02"
EMBER_ROOT = RAW_ROOT / "ember" / f"retrieved_{RETRIEVED_DATE}"
EIA_ROOT = RAW_ROOT / "eia"

RAW_URLS = {
    GEM_ROOT / "GlobalSolarPowerTrackerSummaryData-Feb2026.xlsx": (
        "https://docs.google.com/spreadsheets/d/"
        "1cxeu_WZ4GaKWET0RMvm6ERdd9RrJTEWt7IJzb0_66n8/export?format=xlsx"
    ),
    GEM_ROOT / "GlobalWindPowerTrackerSummaryData-Feb2026.xlsx": (
        "https://docs.google.com/spreadsheets/d/"
        "1q89UKfHE21hFEySRKS7PuO2JTN7rGu0zfXvDJU2smHg/export?format=xlsx"
    ),
    GEM_ROOT / "gspt_map_2026-02.geojson": (
        "https://publicgemdata.nyc3.cdn.digitaloceanspaces.com/"
        "interim_maps/gspt_map_2026-02.geojson"
    ),
    GEM_ROOT / "gwpt_map_2026-02.geojson": (
        "https://publicgemdata.nyc3.cdn.digitaloceanspaces.com/"
        "interim_maps/gwpt_map_2026-02.geojson"
    ),
    EMBER_ROOT / "monthly_capacity_wind_solar_public_release_file.csv": (
        "https://files.ember-energy.org/public-downloads/capacity/outputs/"
        "monthly_capacity_wind_solar_public_release_file.csv"
    ),
    EMBER_ROOT / "ember_monthly_capacity_methodology.pdf": (
        "https://files.ember-energy.org/public-downloads/capacity/outputs/"
        "ember_monthly_capacity_methodology.pdf"
    ),
    EMBER_ROOT / "release_generation_yearly_global.csv": (
        "https://files.ember-energy.org/public-downloads/generation/outputs/"
        "release_generation_yearly_global.csv"
    ),
    EMBER_ROOT / "ember_electricity_data_methodology.pdf": (
        "https://files.ember-energy.org/public-downloads/"
        "ember_electricity_data_methodology.pdf"
    ),
    EIA_ROOT / "eia8602023.zip": (
        "https://www.eia.gov/electricity/data/eia860/archive/xls/eia8602023.zip"
    ),
    EIA_ROOT / "eia8602024.zip": (
        "https://www.eia.gov/electricity/data/eia860/xls/eia8602024.zip"
    ),
    EIA_ROOT / "eia8602025ER.zip": (
        "https://www.eia.gov/electricity/data/eia860/xls/eia8602025ER.zip"
    ),
    EIA_ROOT / "f923_2023.zip": (
        "https://www.eia.gov/electricity/data/eia923/archive/xls/f923_2023.zip"
    ),
    EIA_ROOT / "f923_2024.zip": (
        "https://www.eia.gov/electricity/data/eia923/archive/xls/f923_2024.zip"
    ),
    EIA_ROOT / "f923_2025er.zip": (
        "https://www.eia.gov/electricity/data/eia923/xls/f923_2025er.zip"
    ),
}

GEM_HISTORICAL_URLS = {
    "gem_solar_release_2024-06.csv": (
        "https://raw.githubusercontent.com/GlobalEnergyMonitor/interim-maps/"
        "635131df55f9a240c927602391755a0afdd279d4/trackers/solar/data.csv"
    ),
    "gem_wind_release_2024-06.csv": (
        "https://raw.githubusercontent.com/GlobalEnergyMonitor/interim-maps/"
        "635131df55f9a240c927602391755a0afdd279d4/trackers/wind/data.csv"
    ),
    "gem_solar_release_2025-02.csv": (
        "https://raw.githubusercontent.com/GlobalEnergyMonitor/interim-maps/"
        "db543b1fd159fa172db6aa9ccdd16765858a881c/trackers/solar/"
        "compilation_output/Solar-map-file-2025-02-04.csv"
    ),
    "gem_wind_release_2025-02.csv": (
        "https://raw.githubusercontent.com/GlobalEnergyMonitor/interim-maps/"
        "db543b1fd159fa172db6aa9ccdd16765858a881c/trackers/wind/"
        "compilation_output/Wind-map-file-2025-02-04.csv"
    ),
}

IRENA_COUNTRY_URL = (
    "https://pxweb.irena.org/api/v1/en/IRENASTAT/"
    "Power%20Capacity%20and%20Generation/Country_ELECCAP_2026_H1_v-PX%201.px"
)
IRENA_REGION_URL = (
    "https://pxweb.irena.org/api/v1/en/IRENASTAT/"
    "Power%20Capacity%20and%20Generation/Region_ELECCAP_2026_H1_v-PX%201.px"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--download-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download an official raw file only when it is absent locally.",
    )
    parser.add_argument(
        "--skip-gem-assets",
        action="store_true",
        help="Skip the two large GEM GeoJSON files during a fast diagnostic run.",
    )
    return parser.parse_args()


def atomic_json(payload: dict[str, Any], path: Path) -> None:
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


def download_file(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".download")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "wind-solar-research-database/1.0"},
    )
    with urllib.request.urlopen(request, timeout=180) as response, temporary.open(
        "wb"
    ) as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    if temporary.stat().st_size == 0:
        raise RuntimeError(f"Downloaded an empty file: {url}")
    temporary.replace(path)


def ensure_raw_files(download_missing: bool) -> None:
    missing = [path for path in RAW_URLS if not path.exists()]
    if missing and not download_missing:
        joined = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Required raw files are missing:\n{joined}")
    for path in missing:
        print(f"Downloading {RAW_URLS[path]} -> {path}", flush=True)
        download_file(RAW_URLS[path], path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_column(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).replace("\xa0", " ")).strip()


def clean_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().replace({"": pd.NA, ".": pd.NA})


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def json_safe_number(value: Any) -> int | float | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    if number.is_integer():
        return int(number)
    return number


def iter_geojson_features(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[dict]:
    """Stream a FeatureCollection without loading a 150 MB file into memory."""

    decoder = json.JSONDecoder()
    marker = re.compile(r'"features"\s*:\s*\[')
    buffer = ""
    started = False
    finished = False
    with path.open("r", encoding="utf-8") as handle:
        while not finished:
            chunk = handle.read(chunk_size)
            if chunk:
                buffer += chunk
            elif not buffer:
                break

            if not started:
                match = marker.search(buffer)
                if match is None:
                    if not chunk:
                        raise ValueError(f"No features array in {path}")
                    buffer = buffer[-128:]
                    continue
                buffer = buffer[match.end() :]
                started = True

            while started:
                buffer = buffer.lstrip()
                if buffer.startswith(","):
                    buffer = buffer[1:]
                    continue
                if buffer.startswith("]"):
                    finished = True
                    break
                try:
                    feature, end = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    break
                if not isinstance(feature, dict):
                    raise ValueError(f"Invalid feature in {path}")
                yield feature
                buffer = buffer[end:]

            if not chunk and not finished:
                raise ValueError(f"Truncated GeoJSON: {path}")


def parse_start_year(value: Any) -> int | None:
    if value is None or value == "":
        return None
    match = re.search(r"(?:19|20)\d{2}", str(value))
    return int(match.group()) if match else None


def build_gem_assets(output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rows: list[dict[str, Any]] = []
    specifications = [
        (
            "solar",
            GEM_ROOT / "gspt_map_2026-02.geojson",
            "mixed_reported_mw_not_uniform_mwac",
            False,
        ),
        (
            "wind",
            GEM_ROOT / "gwpt_map_2026-02.geojson",
            "nameplate_mw",
            True,
        ),
    ]
    for family, path, capacity_basis, ac_comparable in specifications:
        print(f"Streaming {path.name}", flush=True)
        for feature in iter_geojson_features(path):
            properties = feature.get("properties") or {}
            geometry = feature.get("geometry") or {}
            coordinates = geometry.get("coordinates") or [None, None]
            longitude = properties.get("Longitude")
            latitude = properties.get("Latitude")
            if longitude is None and len(coordinates) >= 1:
                longitude = coordinates[0]
            if latitude is None and len(coordinates) >= 2:
                latitude = coordinates[1]
            rows.append(
                {
                    "technology_family": family,
                    "technology_detail": properties.get("tech-type"),
                    "country_area": properties.get("country-area1"),
                    "subnational": properties.get("subnational"),
                    "project_name": properties.get("name"),
                    "phase_name": properties.get("unit-name"),
                    "project_id": properties.get("project-id"),
                    "phase_id": properties.get("unit-id"),
                    "capacity_mw_reported": json_safe_number(
                        properties.get("capacity")
                    ),
                    "capacity_basis": capacity_basis,
                    "is_capacity_ac_comparable": ac_comparable,
                    "status": properties.get("status"),
                    "start_year_raw": (
                        None
                        if properties.get("start-year") in (None, "")
                        else str(properties.get("start-year"))
                    ),
                    "start_year": parse_start_year(properties.get("start-year")),
                    "latitude": json_safe_number(latitude),
                    "longitude": json_safe_number(longitude),
                    "location_accuracy": properties.get("location-accuracy"),
                    "owner": properties.get("owner"),
                    "operator": properties.get("operator"),
                    "source_record_url": properties.get("url"),
                    "source_dataset": (
                        "GEM Global Solar Power Tracker"
                        if family == "solar"
                        else "GEM Global Wind Power Tracker"
                    ),
                    "source_release": "2026-02",
                    "source_snapshot_interpretation": "current_release_asset_phase",
                    "license_note": (
                        "Mixed: GEM CC BY 4.0 default; some GSPT records supplied "
                        "by TransitionZero may be CC BY-NC 4.0, but the public map "
                        "does not expose the identifier needed to separate them."
                        if family == "solar"
                        else "CC BY 4.0 under GEM default licensing."
                    ),
                }
            )

    assets = pd.DataFrame(rows)
    assets["start_year"] = assets["start_year"].astype("Int64")
    assets["status"] = clean_text(assets["status"]).str.lower()
    atomic_parquet(assets, output_root / "gem_asset_phase_snapshot_2026-02.parquet")

    survivor_rows: list[pd.DataFrame] = []
    operating_known = assets.loc[
        assets["status"].eq("operating") & assets["start_year"].notna()
    ].copy()
    for year in YEARS:
        selected = operating_known.loc[operating_known["start_year"].le(year)].copy()
        selected.insert(0, "snapshot_year", year)
        selected.insert(
            1,
            "snapshot_interpretation",
            "currently_operating_in_2026_and_commissioned_by_year_survivor_lower_bound",
        )
        survivor_rows.append(selected)
    survivors = pd.concat(survivor_rows, ignore_index=True)
    atomic_parquet(
        survivors,
        output_root / "gem_operating_asset_year_survivor_lower_bound_2023_2025.parquet",
    )

    operating = assets.loc[assets["status"].eq("operating")]
    missing_start = operating["start_year"].isna()
    validation = {
        "asset_phase_rows": int(len(assets)),
        "phase_id_missing": int(assets["phase_id"].isna().sum()),
        "phase_id_duplicate_rows": int(assets["phase_id"].duplicated(keep=False).sum()),
        "coordinate_missing_rows": int(
            (assets["latitude"].isna() | assets["longitude"].isna()).sum()
        ),
        "by_technology": {},
    }
    for family in ("solar", "wind"):
        family_assets = assets.loc[assets["technology_family"].eq(family)]
        family_operating = family_assets.loc[family_assets["status"].eq("operating")]
        family_missing = family_operating["start_year"].isna()
        validation["by_technology"][family] = {
            "asset_phase_rows": int(len(family_assets)),
            "operating_phase_rows": int(len(family_operating)),
            "operating_missing_start_year_rows": int(family_missing.sum()),
            "operating_missing_start_year_share": float(family_missing.mean()),
            "operating_missing_start_year_capacity_mw_reported": float(
                family_operating.loc[family_missing, "capacity_mw_reported"].sum()
            ),
            "current_operating_capacity_mw_reported": float(
                family_operating["capacity_mw_reported"].sum()
            ),
            "survivor_lower_bound": {
                str(year): {
                    "phase_rows": int(
                        (
                            family_operating["start_year"].notna()
                            & family_operating["start_year"].le(year)
                        ).sum()
                    ),
                    "capacity_mw_reported": float(
                        family_operating.loc[
                            family_operating["start_year"].notna()
                            & family_operating["start_year"].le(year),
                            "capacity_mw_reported",
                        ].sum()
                    ),
                }
                for year in YEARS
            },
        }
    validation["operating_missing_start_year_rows"] = int(missing_start.sum())
    return assets, survivors, validation


def year_column(frame: pd.DataFrame, year: int) -> Any:
    for column in frame.columns:
        value = clean_column(column)
        if value in {str(year), f"{year}.0"}:
            return column
    raise KeyError(f"No {year} column found: {frame.columns.tolist()}")


def named_column(frame: pd.DataFrame, pattern: str) -> Any:
    expression = re.compile(pattern, flags=re.IGNORECASE)
    for column in frame.columns:
        if expression.search(clean_column(column)):
            return column
    raise KeyError(f"No column matching {pattern!r}")


def build_gem_country_year(output_root: Path) -> tuple[pd.DataFrame, dict]:
    specifications = [
        (
            "solar",
            GEM_ROOT / "GlobalSolarPowerTrackerSummaryData-Feb2026.xlsx",
            "Yearly added utility-scale capa",
            "MWac",
        ),
        (
            "wind",
            GEM_ROOT / "GlobalWindPowerTrackerSummaryData-Feb2026.xlsx",
            "New Wind Power Capacity Added b",
            "MW",
        ),
    ]
    records: list[dict[str, Any]] = []
    for family, path, sheet, unit in specifications:
        frame = pd.read_excel(path, sheet_name=sheet, header=4)
        country_column = frame.columns[0]
        pre_2007 = named_column(frame, r"^Pre-2007$")
        unknown = named_column(frame, r"Operational but Unknown Year")
        total = named_column(frame, r"^Total")
        frame = frame.loc[frame[country_column].notna()].copy()
        frame[country_column] = clean_text(frame[country_column])
        for _, row in frame.iterrows():
            country = row[country_column]
            if pd.isna(country):
                continue
            for year in YEARS:
                cumulative_columns = [pre_2007] + [
                    year_column(frame, candidate)
                    for candidate in range(2007, year + 1)
                ]
                records.append(
                    {
                        "technology_family": family,
                        "country_area": str(country),
                        "is_global_total": str(country).strip() == "Global Total",
                        "year": year,
                        "commissioned_in_year_mw": json_safe_number(
                            row[year_column(frame, year)]
                        ),
                        "cumulative_known_start_capacity_mw": json_safe_number(
                            pd.to_numeric(row[cumulative_columns], errors="coerce").sum()
                        ),
                        "operating_unknown_start_year_mw": json_safe_number(row[unknown]),
                        "published_total_capacity_mw": json_safe_number(row[total]),
                        "published_total_column": clean_column(total),
                        "capacity_unit": unit,
                        "scope": (
                            "utility_scale_solar_1MW_or_greater"
                            if family == "solar"
                            else "wind_farm_phase_10MW_or_greater"
                        ),
                        "historical_interpretation": (
                            "current_2026_release_operating_cohort_grouped_by_"
                            "commissioning_year; cumulative value is a survivor_"
                            "lower_bound, not a complete historical stock"
                        ),
                        "source_release": "2026-02",
                    }
                )
    result = pd.DataFrame(records)
    atomic_parquet(
        result,
        output_root / "gem_country_year_utility_scale_capacity_2023_2025.parquet",
    )
    global_rows = result.loc[result["is_global_total"]]
    validation = {
        "rows": int(len(result)),
        "countries_or_areas_including_global_total": {
            family: int(
                result.loc[result["technology_family"].eq(family), "country_area"].nunique()
            )
            for family in ("solar", "wind")
        },
        "global_totals": global_rows.to_dict(orient="records"),
        "duplicate_key_rows": int(
            result.duplicated(
                ["technology_family", "country_area", "year"], keep=False
            ).sum()
        ),
    }
    return result, validation


def build_ember(output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    capacity_path = EMBER_ROOT / "monthly_capacity_wind_solar_public_release_file.csv"
    capacity = pd.read_csv(capacity_path, low_memory=False)
    capacity = capacity.loc[capacity["Year"].isin(YEARS)].copy()
    capacity = capacity.rename(
        columns={
            "Month": "month",
            "Year": "year",
            "ISO 3 Code": "iso3",
            "Area": "area",
            "Is aggregate area": "is_aggregate_area",
            "Ember region": "ember_region",
            "Continent": "continent",
            "Source": "source",
            "Is aggregate source": "is_aggregate_source",
            "Unit": "capacity_unit",
            "Installed Capacity": "installed_capacity",
            "Capacity additions (month-on-month)": "capacity_additions_month_on_month",
            "Capacity additions (year-to-date)": "capacity_additions_year_to_date",
        }
    )
    capacity["source_dataset"] = "Ember Monthly Wind and Solar Capacity Data"
    capacity["retrieved_at"] = RETRIEVED_DATE
    atomic_parquet(capacity, output_root / "ember_monthly_capacity_2023_2025.parquet")

    december = capacity.loc[capacity["month"].eq(12)].copy()
    atomic_parquet(
        december, output_root / "ember_country_year_capacity_2023_2025.parquet"
    )

    generation_path = EMBER_ROOT / "release_generation_yearly_global.csv"
    generation = pd.read_csv(generation_path, low_memory=False)
    generation = generation.loc[
        generation["Year"].isin(YEARS)
        & generation["Electricity source"].isin(["Solar", "Wind"])
    ].copy()
    generation = generation.rename(
        columns={
            "Area": "area",
            "ISO 3 code": "iso3",
            "Year": "year",
            "Area type": "area_type",
            "Electricity source": "electricity_source",
            "Is aggregated source": "is_aggregated_source",
            "Generation (TWh)": "generation_twh",
            "Generation YoY change (TWh)": "generation_yoy_change_twh",
            "Generation YoY change (%)": "generation_yoy_change_percent",
            "Share of generation (%)": "share_of_generation_percent",
            "Share of generation YoY change (% points)": (
                "share_of_generation_yoy_change_percentage_points"
            ),
            "Emissions (MtCO2e)": "emissions_mtco2e",
            "Continent": "continent",
            "Ember region": "ember_region",
            "EU member": "eu_member",
            "OECD member": "oecd_member",
            "G20 member": "g20_member",
            "G7 member": "g7_member",
            "ASEAN member": "asean_member",
        }
    )
    generation["source_dataset"] = "Ember Yearly Electricity Data"
    generation["retrieved_at"] = RETRIEVED_DATE
    atomic_parquet(
        generation, output_root / "ember_country_year_generation_2023_2025.parquet"
    )

    world_capacity = december.loc[
        december["area"].eq("World")
        & (
            december["source"].eq("Solar")
            | (
                december["source"].eq("Wind")
                & december["is_aggregate_source"].eq(1)
            )
        ),
        ["year", "source", "capacity_unit", "installed_capacity"],
    ]
    world_generation = generation.loc[
        generation["area"].eq("World"),
        ["year", "electricity_source", "generation_twh"],
    ]
    validation = {
        "monthly_capacity_rows": int(len(capacity)),
        "december_capacity_rows": int(len(december)),
        "annual_generation_rows": int(len(generation)),
        "capacity_years": sorted(int(value) for value in capacity["year"].unique()),
        "generation_years": sorted(int(value) for value in generation["year"].unique()),
        "capacity_units": sorted(str(value) for value in capacity["capacity_unit"].dropna().unique()),
        "capacity_duplicate_key_rows": int(
            capacity.duplicated(
                ["year", "month", "area", "source", "capacity_unit"], keep=False
            ).sum()
        ),
        "generation_duplicate_key_rows": int(
            generation.duplicated(
                ["year", "area", "electricity_source"], keep=False
            ).sum()
        ),
        "world_december_capacity": world_capacity.to_dict(orient="records"),
        "world_annual_generation": world_generation.to_dict(orient="records"),
    }
    return capacity, december, generation, validation


def read_zip_excel(
    zip_path: Path,
    member_pattern: str,
    *,
    sheet_name: str | int,
    header: int,
) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as archive:
        matches = [
            name
            for name in archive.namelist()
            if member_pattern.lower() in name.lower() and name.lower().endswith(".xlsx")
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one {member_pattern!r} workbook in {zip_path}, got {matches}"
            )
        workbook = io.BytesIO(archive.read(matches[0]))
    frame = pd.read_excel(workbook, sheet_name=sheet_name, header=header)
    frame.columns = [clean_column(column) for column in frame.columns]
    return frame


def standardized_generator_id(series: pd.Series) -> pd.Series:
    return clean_text(series).str.replace(r"\.0$", "", regex=True)


def build_eia_capacity(output_root: Path) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    specifications = {
        2023: (EIA_ROOT / "eia8602023.zip", 1, "final"),
        2024: (EIA_ROOT / "eia8602024.zip", 1, "final"),
        2025: (EIA_ROOT / "eia8602025ER.zip", 2, "early_release_provisional"),
    }
    frames: list[pd.DataFrame] = []
    plants_by_year: dict[int, pd.DataFrame] = {}
    for year, (path, header, release_status) in specifications.items():
        generators = read_zip_excel(
            path, "3_1_Generator", sheet_name="Operable", header=header
        )
        plants = read_zip_excel(path, "2___Plant", sheet_name=0, header=header)
        plants = plants[
            [
                "Plant Code",
                "Latitude",
                "Longitude",
                "NERC Region",
                "Balancing Authority Code",
                "Balancing Authority Name",
                "Transmission or Distribution System Owner",
            ]
        ].copy()
        plants["Plant Code"] = pd.to_numeric(plants["Plant Code"], errors="coerce").astype(
            "Int64"
        )
        plants = plants.dropna(subset=["Plant Code"]).drop_duplicates("Plant Code")
        plants_by_year[year] = plants.copy()

        selected = generators.loc[
            generators["Energy Source 1"].astype("string").str.strip().isin(["SUN", "WND"])
        ].copy()
        selected["Plant Code"] = pd.to_numeric(
            selected["Plant Code"], errors="coerce"
        ).astype("Int64")
        selected["Generator ID"] = standardized_generator_id(selected["Generator ID"])
        selected = selected.merge(plants, on="Plant Code", how="left", validate="many_to_one")
        result = pd.DataFrame(
            {
                "year": year,
                "release_status": release_status,
                "plant_code": selected["Plant Code"],
                "generator_id": selected["Generator ID"],
                "plant_name": clean_text(selected["Plant Name"]),
                "state": clean_text(selected["State"]),
                "county": clean_text(selected["County"]),
                "technology_family": selected["Energy Source 1"].map(
                    {"SUN": "solar", "WND": "wind"}
                ),
                "technology_detail": clean_text(selected["Technology"]),
                "energy_source_code": clean_text(selected["Energy Source 1"]),
                "prime_mover": clean_text(selected["Prime Mover"]),
                "nameplate_capacity_mw": numeric(selected["Nameplate Capacity (MW)"]),
                "summer_capacity_mw": numeric(selected["Summer Capacity (MW)"]),
                "winter_capacity_mw": numeric(selected["Winter Capacity (MW)"]),
                "status": clean_text(selected["Status"]),
                "is_formal_operating": clean_text(selected["Status"]).eq("OP"),
                "operating_month": numeric(selected["Operating Month"]).astype("Int64"),
                "operating_year": numeric(selected["Operating Year"]).astype("Int64"),
                "planned_retirement_month": numeric(
                    selected["Planned Retirement Month"]
                ).astype("Int64"),
                "planned_retirement_year": numeric(
                    selected["Planned Retirement Year"]
                ).astype("Int64"),
                "sector_name": clean_text(selected["Sector Name"]),
                "ownership_code": clean_text(selected["Ownership"]),
                "latitude": numeric(selected["Latitude"]),
                "longitude": numeric(selected["Longitude"]),
                "nerc_region": clean_text(selected["NERC Region"]),
                "balancing_authority_code": clean_text(
                    selected["Balancing Authority Code"]
                ),
                "balancing_authority_name": clean_text(
                    selected["Balancing Authority Name"]
                ),
                "transmission_distribution_owner": clean_text(
                    selected["Transmission or Distribution System Owner"]
                ),
                "source_dataset": "EIA-860",
                "source_granularity": "generator_year",
            }
        )

        wind = read_zip_excel(path, "3_2_Wind", sheet_name=0, header=header)
        wind["Plant Code"] = pd.to_numeric(wind["Plant Code"], errors="coerce").astype(
            "Int64"
        )
        wind["Generator ID"] = standardized_generator_id(wind["Generator ID"])
        wind = wind[
            [
                "Plant Code",
                "Generator ID",
                "Number of Turbines",
                "Predominant Turbine Manufacturer",
                "Predominant Turbine Model Number",
                "Design Wind Speed (mph)",
                "Wind Quality Class",
                "Turbine Hub Height (Feet)",
            ]
        ].rename(
            columns={
                "Plant Code": "plant_code",
                "Generator ID": "generator_id",
                "Number of Turbines": "number_of_turbines",
                "Predominant Turbine Manufacturer": "turbine_manufacturer",
                "Predominant Turbine Model Number": "turbine_model",
                "Design Wind Speed (mph)": "design_wind_speed_mph",
                "Wind Quality Class": "wind_quality_class",
                "Turbine Hub Height (Feet)": "turbine_hub_height_ft",
            }
        )
        solar = read_zip_excel(path, "3_3_Solar", sheet_name=0, header=header)
        solar["Plant Code"] = pd.to_numeric(
            solar["Plant Code"], errors="coerce"
        ).astype("Int64")
        solar["Generator ID"] = standardized_generator_id(solar["Generator ID"])
        solar = solar[
            [
                "Plant Code",
                "Generator ID",
                "Single-Axis Tracking?",
                "Dual-Axis Tracking?",
                "Fixed Tilt?",
                "Bifacial?",
                "Azimuth Angle",
                "Tilt Angle",
                "DC Net Capacity (MW)",
            ]
        ].rename(
            columns={
                "Plant Code": "plant_code",
                "Generator ID": "generator_id",
                "Single-Axis Tracking?": "single_axis_tracking",
                "Dual-Axis Tracking?": "dual_axis_tracking",
                "Fixed Tilt?": "fixed_tilt",
                "Bifacial?": "bifacial",
                "Azimuth Angle": "azimuth_angle_degrees",
                "Tilt Angle": "tilt_angle_degrees",
                "DC Net Capacity (MW)": "dc_net_capacity_mw",
            }
        )
        result = result.merge(
            wind.drop_duplicates(["plant_code", "generator_id"]),
            on=["plant_code", "generator_id"],
            how="left",
            validate="one_to_one",
        ).merge(
            solar.drop_duplicates(["plant_code", "generator_id"]),
            on=["plant_code", "generator_id"],
            how="left",
            validate="one_to_one",
        )
        for column in [
            "number_of_turbines",
            "design_wind_speed_mph",
            "turbine_hub_height_ft",
            "azimuth_angle_degrees",
            "tilt_angle_degrees",
            "dc_net_capacity_mw",
        ]:
            result[column] = numeric(result[column])
        for column in [
            "turbine_manufacturer",
            "turbine_model",
            "wind_quality_class",
            "single_axis_tracking",
            "dual_axis_tracking",
            "fixed_tilt",
            "bifacial",
        ]:
            result[column] = clean_text(result[column])
        frames.append(result)

    capacity = pd.concat(frames, ignore_index=True)
    atomic_parquet(
        capacity, output_root / "eia_us_generator_year_capacity_2023_2025.parquet"
    )
    return capacity, plants_by_year


MONTHS = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}


def build_eia_generation(
    output_root: Path,
    capacity: pd.DataFrame,
    plants_by_year: dict[int, pd.DataFrame],
) -> tuple[pd.DataFrame, dict]:
    specifications = {
        2023: (EIA_ROOT / "f923_2023.zip", 5, "final"),
        2024: (EIA_ROOT / "f923_2024.zip", 5, "final"),
        2025: (EIA_ROOT / "f923_2025er.zip", 6, "early_release_provisional"),
    }
    frames: list[pd.DataFrame] = []
    annual_checks: dict[str, Any] = {}
    for year, (path, header, release_status) in specifications.items():
        frame = read_zip_excel(
            path,
            "Schedules_2_3_4_5",
            sheet_name="Page 1 Generation and Fuel Data",
            header=header,
        )
        if year == 2025:
            warning_columns = [
                column for column in frame if column.startswith("Early release data")
            ]
            frame = frame.drop(columns=warning_columns)
        selected = frame.loc[
            frame["Reported Fuel Type Code"]
            .astype("string")
            .str.strip()
            .isin(["SUN", "WND"])
        ].copy()
        selected["Plant Id"] = pd.to_numeric(
            selected["Plant Id"], errors="coerce"
        ).astype("Int64")
        selected["raw_row_id"] = range(len(selected))

        monthly_total = pd.Series(0.0, index=selected.index)
        any_month = pd.Series(False, index=selected.index)
        long_rows: list[pd.DataFrame] = []
        for month, month_name in MONTHS.items():
            values = numeric(selected[f"Netgen {month_name}"])
            monthly_total = monthly_total.add(values.fillna(0.0), fill_value=0.0)
            any_month |= values.notna()
            part = pd.DataFrame(
                {
                    "year": year,
                    "month": month,
                    "release_status": release_status,
                    "raw_row_id": selected["raw_row_id"],
                    "plant_code": selected["Plant Id"],
                    "plant_name": clean_text(selected["Plant Name"]),
                    "state": clean_text(selected["Plant State"]),
                    "operator_name": clean_text(selected["Operator Name"]),
                    "operator_id": numeric(selected["Operator Id"]).astype("Int64"),
                    "sector_name": clean_text(selected["Sector Name"]),
                    "technology_family": selected["Reported Fuel Type Code"].map(
                        {"SUN": "solar", "WND": "wind"}
                    ),
                    "fuel_type_code": clean_text(
                        selected["Reported Fuel Type Code"]
                    ),
                    "prime_mover": clean_text(selected["Reported Prime Mover"]),
                    "balancing_authority_code": clean_text(
                        selected["Balancing Authority Code"]
                    ),
                    "respondent_frequency": clean_text(
                        selected["Respondent Frequency"]
                    ),
                    "net_generation_mwh": values,
                    "source_dataset": "EIA-923",
                    "source_granularity": "plant_prime_mover_fuel_month",
                }
            )
            long_rows.append(part)
        long_frame = pd.concat(long_rows, ignore_index=True)
        plant_locations = plants_by_year[year][
            [
                "Plant Code",
                "Latitude",
                "Longitude",
                "NERC Region",
                "Balancing Authority Name",
            ]
        ].rename(
            columns={
                "Plant Code": "plant_code",
                "Latitude": "latitude",
                "Longitude": "longitude",
                "NERC Region": "nerc_region",
                "Balancing Authority Name": "balancing_authority_name",
            }
        )
        long_frame = long_frame.merge(
            plant_locations,
            on="plant_code",
            how="left",
            validate="many_to_one",
        )
        annual_reported = numeric(selected["Net Generation (Megawatthours)"])
        comparable = any_month & annual_reported.notna()
        differences = (monthly_total.loc[comparable] - annual_reported.loc[comparable]).abs()
        annual_checks[str(year)] = {
            "source_rows": int(len(selected)),
            "annual_comparable_rows": int(comparable.sum()),
            "annual_month_sum_max_abs_difference_mwh": (
                float(differences.max()) if len(differences) else None
            ),
            "annual_month_sum_rows_over_1mwh": int((differences > 1.0).sum()),
        }
        frames.append(long_frame)

    generation = pd.concat(frames, ignore_index=True)
    atomic_parquet(
        generation,
        output_root / "eia_us_plant_month_generation_2023_2025.parquet",
    )

    coverage: dict[str, Any] = {}
    for year in YEARS:
        year_generation = generation.loc[
            generation["year"].eq(year) & generation["net_generation_mwh"].notna()
        ]
        year_capacity = capacity.loc[
            capacity["year"].eq(year) & capacity["is_formal_operating"]
        ]
        coverage[str(year)] = {}
        for family in ("solar", "wind"):
            observed_plants = set(
                year_generation.loc[
                    year_generation["technology_family"].eq(family), "plant_code"
                ].dropna()
            )
            subset = year_capacity.loc[
                year_capacity["technology_family"].eq(family)
            ]
            denominator = float(subset["nameplate_capacity_mw"].sum())
            numerator = float(
                subset.loc[
                    subset["plant_code"].isin(observed_plants), "nameplate_capacity_mw"
                ].sum()
            )
            coverage[str(year)][family] = {
                "formal_operating_nameplate_capacity_mw": denominator,
                "capacity_at_plants_with_eia923_observation_mw": numerator,
                "coverage_share": numerator / denominator if denominator else None,
            }

    validation = {
        "generator_capacity_rows": int(len(capacity)),
        "capacity_duplicate_key_rows": int(
            capacity.duplicated(["year", "plant_code", "generator_id"], keep=False).sum()
        ),
        "capacity_missing_coordinate_rows": int(
            (capacity["latitude"].isna() | capacity["longitude"].isna()).sum()
        ),
        "capacity_status_counts": (
            capacity.groupby(["year", "technology_family", "status"], dropna=False)
            .size()
            .rename("rows")
            .reset_index()
            .to_dict(orient="records")
        ),
        "generation_rows": int(len(generation)),
        "generation_years": sorted(int(value) for value in generation["year"].unique()),
        "generation_months": sorted(int(value) for value in generation["month"].unique()),
        "generation_missing_value_rows": int(generation["net_generation_mwh"].isna().sum()),
        "generation_negative_value_rows": int(generation["net_generation_mwh"].lt(0).sum()),
        "annual_consistency": annual_checks,
        "eia923_observation_coverage_of_eia860_operating_capacity": coverage,
        "provisional_warning": (
            "EIA 2025 early-release files are not fully edited and explicitly "
            "should not be aggregated to state, regional, or national totals."
        ),
    }
    return generation, validation


def source_license(path: Path) -> str:
    relative = path.relative_to(RAW_ROOT).as_posix().lower()
    if relative.startswith("gem/"):
        if "gspt_map" in relative or "solar" in relative:
            return (
                "Mixed/conditional: GEM default CC BY 4.0; some TransitionZero "
                "solar records may be CC BY-NC 4.0 and are not identifiable in map export."
            )
        return "GEM default CC BY 4.0; retain source-specific exceptions if any."
    if relative.startswith("ember/"):
        return "CC BY 4.0"
    if relative.startswith("eia/"):
        return "United States Government public data; cite EIA."
    if relative.startswith("entsoe/"):
        return "See ENTSO-E Power Statistics/Transparency Platform terms and source metadata."
    if relative.startswith("irena/"):
        return "IRENA terms require attribution and impose reuse conditions; not treated as unrestricted CC BY."
    return "See source metadata."


def source_url(path: Path) -> str | None:
    if path in RAW_URLS:
        return RAW_URLS[path]
    relative = path.relative_to(RAW_ROOT).as_posix()
    if relative.startswith("entsoe/power_statistics/"):
        match = re.search(r"_(202[345])\.csv$", path.name)
        if match:
            return (
                "https://www.entsoe.eu/publications/data/power-stats/"
                f"{match.group(1)}/{path.name}"
            )
    if relative.startswith("gem/historical_vintages/"):
        return GEM_HISTORICAL_URLS.get(path.name) or (
            "https://github.com/GlobalEnergyMonitor/interim-maps"
        )
    if relative.startswith("irena/2026-h1/"):
        return IRENA_REGION_URL if path.name.startswith(("region_", "world_")) else IRENA_COUNTRY_URL
    return None


def build_manifest(output_root: Path) -> dict:
    files: list[dict[str, Any]] = []
    for path in sorted(item for item in RAW_ROOT.rglob("*") if item.is_file()):
        files.append(
            {
                "relative_path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "source_url": source_url(path),
                "retrieved_at": RETRIEVED_DATE,
                "license_note": source_license(path),
            }
        )
    manifest = {
        "schema_version": "1.0",
        "built_at_utc": datetime.now(UTC).isoformat(),
        "period": list(YEARS),
        "principle": (
            "Observed capacity/generation only; no installed capacity inferred from "
            "data-center electricity demand."
        ),
        "files": files,
    }
    atomic_json(manifest, output_root / "source_manifest.json")
    return manifest


def output_inventory(output_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(output_root.glob("*.parquet")):
        frame = pd.read_parquet(path)
        rows.append(
            {
                "file": path.name,
                "rows": int(len(frame)),
                "columns": int(len(frame.columns)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "year_min": (
                    int(frame["year"].min())
                    if "year" in frame and frame["year"].notna().any()
                    else None
                ),
                "year_max": (
                    int(frame["year"].max())
                    if "year" in frame and frame["year"].notna().any()
                    else None
                ),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    ensure_raw_files(args.download_missing)

    validation: dict[str, Any] = {
        "schema_version": "1.0",
        "built_at_utc": datetime.now(UTC).isoformat(),
        "period": list(YEARS),
        "boundary": {
            "demand_used_to_infer_capacity": False,
            "observed_generation_kept_separate_from_modeled_generation": True,
            "gem_historical_assets_are_complete_snapshots": False,
            "eia_2025_is_final": False,
        },
    }

    if not args.skip_gem_assets:
        _, _, validation["gem_assets"] = build_gem_assets(output_root)
    _, validation["gem_country_year"] = build_gem_country_year(output_root)
    _, _, _, validation["ember"] = build_ember(output_root)
    eia_capacity, plants_by_year = build_eia_capacity(output_root)
    _, validation["eia"] = build_eia_generation(
        output_root, eia_capacity, plants_by_year
    )
    manifest = build_manifest(output_root)
    validation["source_manifest_file_count"] = len(manifest["files"])
    validation["outputs"] = output_inventory(output_root)
    validation["all_target_years_present"] = all(
        item["year_min"] == 2023 and item["year_max"] == 2025
        for item in validation["outputs"]
        if item["file"]
        not in {
            "gem_asset_phase_snapshot_2026-02.parquet",
        }
        and item["year_min"] is not None
    )
    atomic_json(validation, output_root / "validation.json")
    print(json.dumps(validation, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
