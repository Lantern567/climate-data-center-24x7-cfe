#!/usr/bin/env python3
"""Build historical GEM wind/solar asset-vintage proxy snapshots.

This script downloads four immutable CSV files from commits in Global Energy
Monitor's official ``interim-maps`` repository and harmonises their asset-phase
fields.  The temporal mapping is deliberately conservative:

* target 2023 -> GEM 2024-06 release (next-release proxy, not exact year-end);
* target 2024 -> GEM 2025-02 release (next-release proxy, not exact year-end);
* target 2025 -> GEM 2026-02 release (next-release proxy, not exact year-end).

The source ``Capacity (MW)`` / ``capacity-(mw)`` value is retained as reported.
In particular, the solar map vintages do not provide a uniform MWac rating, so
the column must not be treated as directly aggregable AC supply capacity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_ROOT = (
    ROOT
    / "data"
    / "24x7_cfe"
    / "raw"
    / "supply_assets"
    / "gem"
    / "historical_vintages"
)
DEFAULT_OUTPUT_ROOT = (
    ROOT
    / "data"
    / "24x7_cfe"
    / "processed"
    / "wind_solar_supply_2023_2025"
)
OUTPUT_NAME = "gem_historical_vintage_asset_snapshot_2023_2025.parquet"
VALIDATION_NAME = "gem_historical_vintages_validation.json"
MANIFEST_NAME = "manifest.json"
CURRENT_SNAPSHOT_PATH = (
    DEFAULT_OUTPUT_ROOT / "gem_asset_phase_snapshot_2026-02.parquet"
)
GEM_REPOSITORY_URL = "https://github.com/GlobalEnergyMonitor/interim-maps"
GEM_LICENSE_URL = "https://globalenergymonitor.org/creative-commons-license"
USER_AGENT = "dc-climate-research-gem-vintage-builder/1.0"


@dataclass(frozen=True)
class ReleaseFile:
    tracker: str
    proxy_target_year: int
    release_label: str
    commit_sha: str
    url: str
    filename: str
    expected_rows: int


RELEASE_FILES = (
    ReleaseFile(
        tracker="solar",
        proxy_target_year=2023,
        release_label="2024-06",
        commit_sha="635131df55f9a240c927602391755a0afdd279d4",
        url=(
            "https://raw.githubusercontent.com/GlobalEnergyMonitor/interim-maps/"
            "635131df55f9a240c927602391755a0afdd279d4/trackers/solar/data.csv"
        ),
        filename="gem_solar_release_2024-06.csv",
        expected_rows=51_974,
    ),
    ReleaseFile(
        tracker="wind",
        proxy_target_year=2023,
        release_label="2024-06",
        commit_sha="635131df55f9a240c927602391755a0afdd279d4",
        url=(
            "https://raw.githubusercontent.com/GlobalEnergyMonitor/interim-maps/"
            "635131df55f9a240c927602391755a0afdd279d4/trackers/wind/data.csv"
        ),
        filename="gem_wind_release_2024-06.csv",
        expected_rows=29_930,
    ),
    ReleaseFile(
        tracker="solar",
        proxy_target_year=2024,
        release_label="2025-02",
        commit_sha="db543b1fd159fa172db6aa9ccdd16765858a881c",
        url=(
            "https://raw.githubusercontent.com/GlobalEnergyMonitor/interim-maps/"
            "db543b1fd159fa172db6aa9ccdd16765858a881c/trackers/solar/"
            "compilation_output/Solar-map-file-2025-02-04.csv"
        ),
        filename="gem_solar_release_2025-02.csv",
        expected_rows=75_689,
    ),
    ReleaseFile(
        tracker="wind",
        proxy_target_year=2024,
        release_label="2025-02",
        commit_sha="db543b1fd159fa172db6aa9ccdd16765858a881c",
        url=(
            "https://raw.githubusercontent.com/GlobalEnergyMonitor/interim-maps/"
            "db543b1fd159fa172db6aa9ccdd16765858a881c/trackers/wind/"
            "compilation_output/Wind-map-file-2025-02-04.csv"
        ),
        filename="gem_wind_release_2025-02.csv",
        expected_rows=31_454,
    ),
)

CURRENT_RELEASE_FILES = (
    ReleaseFile(
        tracker="solar",
        proxy_target_year=2025,
        release_label="2026-02",
        commit_sha="",
        url=(
            "https://publicgemdata.nyc3.cdn.digitaloceanspaces.com/"
            "interim_maps/gspt_map_2026-02.geojson"
        ),
        filename="gspt_map_2026-02.geojson",
        expected_rows=103_940,
    ),
    ReleaseFile(
        tracker="wind",
        proxy_target_year=2025,
        release_label="2026-02",
        commit_sha="",
        url=(
            "https://publicgemdata.nyc3.cdn.digitaloceanspaces.com/"
            "interim_maps/gwpt_map_2026-02.geojson"
        ),
        filename="gwpt_map_2026-02.geojson",
        expected_rows=35_089,
    ),
)
CURRENT_RAW_ROOT = DEFAULT_RAW_ROOT.parent / "2026-02"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-download immutable commit-pinned CSV files even if present.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def download(url: str, destination: Path, refresh: bool) -> bool:
    """Download atomically with bounded retries; return whether bytes changed."""
    if destination.exists() and destination.stat().st_size > 0 and not refresh:
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(1, 4):
        request = Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urlopen(request, timeout=180) as response, partial.open("wb") as out:
                shutil.copyfileobj(response, out, length=1024 * 1024)
            if partial.stat().st_size == 0:
                raise OSError(f"Downloaded zero bytes from {url}")
            partial.replace(destination)
            return True
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if partial.exists():
                partial.unlink()
            if attempt < 3:
                time.sleep(2**attempt)
    raise RuntimeError(f"Failed to download {url}: {last_error}")


def normalise_column_name(value: Any, position: int) -> str:
    text = str(value).strip().casefold()
    text = re.sub(r"[^0-9a-z]+", "_", text).strip("_")
    return text or f"unnamed_{position}"


def read_source_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        encoding="utf-8-sig",
        encoding_errors="replace",
        low_memory=False,
    )
    frame.columns = [
        normalise_column_name(column, position)
        for position, column in enumerate(frame.columns)
    ]
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"{path.name} has duplicate normalised columns: {duplicates}")
    return frame


def iter_geojson_features(
    path: Path, chunk_size: int = 1024 * 1024
) -> Iterator[dict[str, Any]]:
    """Stream a GeoJSON FeatureCollection without loading it all into memory."""
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


def parse_start_year_value(value: Any) -> int | None:
    if value is None or value == "":
        return None
    match = re.search(r"(?:19|20)\d{2}", str(value))
    return int(match.group()) if match else None


def numeric_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if pd.notna(number) else None


def load_current_geojson_assets(
    current_raw_root: Path,
) -> tuple[pd.DataFrame, str]:
    """Fallback parser matching the separately built current GEM snapshot."""
    rows: list[dict[str, Any]] = []
    for spec in CURRENT_RELEASE_FILES:
        path = current_raw_root / spec.filename
        if not path.exists():
            raise FileNotFoundError(
                f"Current snapshot parquet is absent and raw map is missing: {path}"
            )
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
                    "technology_family": spec.tracker,
                    "technology_detail": properties.get("tech-type"),
                    "country_area": properties.get("country-area1"),
                    "subnational": properties.get("subnational"),
                    "project_name": properties.get("name"),
                    "phase_name": properties.get("unit-name"),
                    "project_id": properties.get("project-id"),
                    "phase_id": properties.get("unit-id"),
                    "capacity_mw_reported": numeric_or_none(
                        properties.get("capacity")
                    ),
                    "capacity_basis": (
                        "mixed_reported_mw_not_uniform_mwac"
                        if spec.tracker == "solar"
                        else "nameplate_mw"
                    ),
                    "is_capacity_ac_comparable": spec.tracker == "wind",
                    "status": properties.get("status"),
                    "start_year_raw": properties.get("start-year"),
                    "start_year": parse_start_year_value(
                        properties.get("start-year")
                    ),
                    "latitude": numeric_or_none(latitude),
                    "longitude": numeric_or_none(longitude),
                    "location_accuracy": properties.get("location-accuracy"),
                    "owner": properties.get("owner"),
                    "operator": properties.get("operator"),
                    "source_record_url": properties.get("url"),
                    "source_dataset": (
                        "GEM Global Solar Power Tracker"
                        if spec.tracker == "solar"
                        else "GEM Global Wind Power Tracker"
                    ),
                    "source_release": "2026-02",
                    "source_snapshot_interpretation": (
                        "current_release_asset_phase"
                    ),
                    "license_note": (
                        "Mixed: GEM CC BY 4.0 default; some GSPT records supplied "
                        "by TransitionZero may be CC BY-NC 4.0, but the public map "
                        "does not expose the identifier needed to separate them."
                        if spec.tracker == "solar"
                        else "CC BY 4.0 under GEM default licensing."
                    ),
                }
            )
    return pd.DataFrame.from_records(rows), "raw_geojson_fallback"


def load_current_snapshot(
    snapshot_path: Path, current_raw_root: Path
) -> tuple[pd.DataFrame, str]:
    if snapshot_path.exists():
        return pd.read_parquet(snapshot_path), "processed_snapshot_parquet"
    return load_current_geojson_assets(current_raw_root)


def empty_string(index: pd.Index) -> pd.Series:
    return pd.Series(pd.NA, index=index, dtype="string")


def select(frame: pd.DataFrame, *aliases: str) -> pd.Series:
    for alias in aliases:
        if alias in frame.columns:
            return frame[alias]
    return empty_string(frame.index)


def clean_string(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().replace("", pd.NA)


def clean_number(series: pd.Series, dtype: str = "Float64") -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(dtype)


def build_status_base(status_raw: pd.Series) -> pd.Series:
    normalised = (
        clean_string(status_raw)
        .str.casefold()
        .str.replace(r"\s+", " ", regex=True)
    )
    canonical = normalised.str.extract(
        r"^(pre-construction|operating|construction|announced|shelved|"
        r"cancelled|mothballed|retired|decommissioned)\b",
        expand=False,
    )
    fallback = normalised.str.split(r"\s+-\s+", n=1, regex=True).str[0]
    return canonical.fillna(fallback).astype("string")


def harmonise(
    raw: pd.DataFrame,
    spec: ReleaseFile,
    source_hash: str,
    retrieved_at: str,
) -> pd.DataFrame:
    index = raw.index
    status_raw = clean_string(select(raw, "status"))
    legacy_identifier = select(raw, "unnamed_0", "unnamed_1")
    capacity = clean_number(select(raw, "capacity_mw"))

    if spec.tracker == "solar":
        rating_basis = "mixed_or_unspecified_not_uniform_mwac"
        capacity_note = (
            "GEM map-reported MW; solar rating is not uniformly MWac and must "
            "not be interpreted as directly aggregable AC supply capacity."
        )
    else:
        rating_basis = "unspecified_source_reported_mw"
        capacity_note = (
            "GEM map-reported MW; rating basis is retained as reported and no "
            "AC conversion or capacity inference has been applied."
        )

    out = pd.DataFrame(index=index)
    out["proxy_target_year"] = pd.Series(
        spec.proxy_target_year, index=index, dtype="Int16"
    )
    out["proxy_role"] = pd.Series(
        "next_release_proxy_not_exact_year_end", index=index, dtype="string"
    )
    out["temporal_interpretation"] = pd.Series(
        "following_release_proxy_not_exact_year_end", index=index, dtype="string"
    )
    out["tracker"] = pd.Series(spec.tracker, index=index, dtype="string")
    out["technology"] = pd.Series(spec.tracker, index=index, dtype="string")
    out["technology_detail"] = clean_string(
        select(raw, "technology_type", "installation_type")
    )
    out["gem_location_id"] = clean_string(select(raw, "gem_location_id"))
    out["gem_phase_id"] = clean_string(select(raw, "gem_phase_id"))
    out["stable_gem_phase_id_available"] = out["gem_phase_id"].notna().astype(
        "boolean"
    )
    out["legacy_source_identifier"] = clean_string(legacy_identifier)
    out["country_area"] = clean_string(select(raw, "country_area", "country"))
    out["state_province"] = clean_string(select(raw, "state_province"))
    out["major_area"] = clean_string(select(raw, "major_area_prefecture_district"))
    out["local_area"] = clean_string(select(raw, "local_area_taluk_county"))
    out["region"] = clean_string(select(raw, "region"))
    out["subregion"] = clean_string(select(raw, "subregion"))
    out["project_name"] = clean_string(select(raw, "project_name"))
    out["phase_name"] = clean_string(select(raw, "phase_name"))
    out["project_name_local"] = clean_string(
        select(raw, "project_name_in_local_language_script")
    )
    out["phase_name_local"] = clean_string(
        select(raw, "phase_name_in_local_language_script")
    )
    out["other_names"] = clean_string(select(raw, "other_name_s"))
    out["status_raw"] = status_raw
    out["status_base"] = build_status_base(status_raw)
    out["capacity_reported_mw"] = capacity
    out["capacity_unit"] = pd.Series("MW_as_reported", index=index, dtype="string")
    out["capacity_rating_basis"] = pd.Series(
        rating_basis, index=index, dtype="string"
    )
    out["safe_for_direct_ac_capacity_aggregation"] = pd.Series(
        False, index=index, dtype="boolean"
    )
    out["capacity_interpretation_note"] = pd.Series(
        capacity_note, index=index, dtype="string"
    )
    out["start_year"] = clean_number(select(raw, "start_year"), "Int16")
    out["retired_year"] = clean_number(select(raw, "retired_year"), "Int16")
    out["latitude"] = clean_number(select(raw, "lat"))
    out["longitude"] = clean_number(select(raw, "lng"))
    out["location_accuracy"] = clean_string(select(raw, "location_accuracy"))
    out["owner"] = clean_string(select(raw, "owner"))
    out["operator"] = clean_string(select(raw, "operator"))
    out["owner_name_local"] = clean_string(
        select(raw, "owner_name_in_local_language_script")
    )
    out["operator_name_local"] = clean_string(
        select(raw, "operator_name_in_local_language_script")
    )
    out["record_url"] = clean_string(select(raw, "url", "wiki_url"))
    out["source_row_number"] = pd.Series(
        range(2, len(raw) + 2), index=index, dtype="Int64"
    )
    out["source_dataset"] = pd.Series(
        f"GEM Global {spec.tracker.title()} Power Tracker map CSV",
        index=index,
        dtype="string",
    )
    out["source_release_label"] = pd.Series(
        spec.release_label, index=index, dtype="string"
    )
    out["source_commit_sha"] = pd.Series(
        spec.commit_sha, index=index, dtype="string"
    )
    out["source_repository_url"] = pd.Series(
        GEM_REPOSITORY_URL, index=index, dtype="string"
    )
    out["source_url"] = pd.Series(spec.url, index=index, dtype="string")
    out["source_file"] = pd.Series(spec.filename, index=index, dtype="string")
    out["source_file_sha256"] = pd.Series(
        source_hash, index=index, dtype="string"
    )
    out["source_retrieved_at_utc"] = pd.Series(
        retrieved_at, index=index, dtype="string"
    )
    out["source_license_url"] = pd.Series(
        GEM_LICENSE_URL, index=index, dtype="string"
    )
    out["source_license_note"] = pd.Series(
        "GEM default licence applies; source-specific record exceptions may apply.",
        index=index,
        dtype="string",
    )
    return out.reset_index(drop=True)


def harmonise_current_snapshot(
    raw: pd.DataFrame,
    spec: ReleaseFile,
    source_hash: str,
    retrieved_at: str,
) -> pd.DataFrame:
    """Map the prebuilt 2026-02 asset snapshot into the vintage schema."""
    required = {
        "technology_family",
        "technology_detail",
        "country_area",
        "subnational",
        "project_name",
        "phase_name",
        "project_id",
        "phase_id",
        "capacity_mw_reported",
        "capacity_basis",
        "is_capacity_ac_comparable",
        "status",
        "start_year",
        "latitude",
        "longitude",
        "location_accuracy",
        "owner",
        "operator",
        "source_record_url",
        "source_dataset",
        "license_note",
    }
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(
            "Current GEM snapshot lacks required columns: " + ", ".join(sorted(missing))
        )

    selected = raw.loc[
        clean_string(raw["technology_family"]).str.casefold().eq(spec.tracker)
    ].copy()
    selected = selected.reset_index(drop=True)
    index = selected.index
    status_raw = clean_string(selected["status"])

    if spec.tracker == "solar":
        capacity_note = (
            "GEM map-reported MW; solar rating is mixed AC/DC or unspecified, is "
            "not uniformly MWac, and must not be directly aggregated as AC supply."
        )
    else:
        capacity_note = (
            "GEM nameplate MW retained from the 2026-02 current map; no generation "
            "or weather-based conversion has been applied."
        )

    out = pd.DataFrame(index=index)
    out["proxy_target_year"] = pd.Series(2025, index=index, dtype="Int16")
    out["proxy_role"] = pd.Series(
        "next_release_proxy_not_exact_year_end", index=index, dtype="string"
    )
    out["temporal_interpretation"] = pd.Series(
        "following_release_proxy_not_exact_year_end", index=index, dtype="string"
    )
    out["tracker"] = pd.Series(spec.tracker, index=index, dtype="string")
    out["technology"] = pd.Series(spec.tracker, index=index, dtype="string")
    out["technology_detail"] = clean_string(selected["technology_detail"])
    out["gem_location_id"] = clean_string(selected["project_id"])
    out["gem_phase_id"] = clean_string(selected["phase_id"])
    out["stable_gem_phase_id_available"] = out["gem_phase_id"].notna().astype(
        "boolean"
    )
    out["legacy_source_identifier"] = empty_string(index)
    out["country_area"] = clean_string(selected["country_area"])
    out["state_province"] = clean_string(selected["subnational"])
    out["major_area"] = empty_string(index)
    out["local_area"] = empty_string(index)
    out["region"] = empty_string(index)
    out["subregion"] = empty_string(index)
    out["project_name"] = clean_string(selected["project_name"])
    out["phase_name"] = clean_string(selected["phase_name"])
    out["project_name_local"] = empty_string(index)
    out["phase_name_local"] = empty_string(index)
    out["other_names"] = empty_string(index)
    out["status_raw"] = status_raw
    out["status_base"] = build_status_base(status_raw)
    out["capacity_reported_mw"] = clean_number(
        selected["capacity_mw_reported"]
    )
    out["capacity_unit"] = pd.Series("MW_as_reported", index=index, dtype="string")
    out["capacity_rating_basis"] = clean_string(selected["capacity_basis"])
    out["safe_for_direct_ac_capacity_aggregation"] = selected[
        "is_capacity_ac_comparable"
    ].astype("boolean")
    out["capacity_interpretation_note"] = pd.Series(
        capacity_note, index=index, dtype="string"
    )
    out["start_year"] = clean_number(selected["start_year"], "Int16")
    out["retired_year"] = pd.Series(pd.NA, index=index, dtype="Int16")
    out["latitude"] = clean_number(selected["latitude"])
    out["longitude"] = clean_number(selected["longitude"])
    out["location_accuracy"] = clean_string(selected["location_accuracy"])
    out["owner"] = clean_string(selected["owner"])
    out["operator"] = clean_string(selected["operator"])
    out["owner_name_local"] = empty_string(index)
    out["operator_name_local"] = empty_string(index)
    out["record_url"] = clean_string(selected["source_record_url"])
    out["source_row_number"] = pd.Series(
        range(1, len(selected) + 1), index=index, dtype="Int64"
    )
    out["source_dataset"] = clean_string(selected["source_dataset"])
    out["source_release_label"] = pd.Series(
        spec.release_label, index=index, dtype="string"
    )
    out["source_commit_sha"] = empty_string(index)
    out["source_repository_url"] = pd.Series(
        GEM_REPOSITORY_URL, index=index, dtype="string"
    )
    out["source_url"] = pd.Series(spec.url, index=index, dtype="string")
    out["source_file"] = pd.Series(spec.filename, index=index, dtype="string")
    out["source_file_sha256"] = pd.Series(
        source_hash, index=index, dtype="string"
    )
    out["source_retrieved_at_utc"] = pd.Series(
        retrieved_at, index=index, dtype="string"
    )
    out["source_license_url"] = pd.Series(
        GEM_LICENSE_URL, index=index, dtype="string"
    )
    out["source_license_note"] = clean_string(selected["license_note"])
    return out.reset_index(drop=True)


def status_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["status_base"].value_counts(dropna=False)
    return {
        ("<missing>" if pd.isna(key) else str(key)): int(value)
        for key, value in counts.items()
    }


def validate_subset(frame: pd.DataFrame, spec: ReleaseFile) -> dict[str, Any]:
    phase_ids = frame["gem_phase_id"].dropna()
    operating = frame.loc[frame["status_base"].eq("operating")]
    invalid_coordinates = (
        frame["latitude"].notna()
        & frame["longitude"].notna()
        & (
            ~frame["latitude"].between(-90, 90)
            | ~frame["longitude"].between(-180, 180)
        )
    )
    return {
        "proxy_target_year": spec.proxy_target_year,
        "proxy_role": "next_release_proxy_not_exact_year_end",
        "source_release_label": spec.release_label,
        "tracker": spec.tracker,
        "rows": int(len(frame)),
        "expected_rows": spec.expected_rows,
        "row_count_matches_expected_release": len(frame) == spec.expected_rows,
        "countries_or_areas": int(frame["country_area"].nunique(dropna=True)),
        "status_base_counts": status_counts(frame),
        "missing_status": int(frame["status_raw"].isna().sum()),
        "missing_capacity_reported_mw": int(
            frame["capacity_reported_mw"].isna().sum()
        ),
        "zero_or_negative_capacity_rows": int(
            frame["capacity_reported_mw"].le(0).fillna(False).sum()
        ),
        "missing_latitude": int(frame["latitude"].isna().sum()),
        "missing_longitude": int(frame["longitude"].isna().sum()),
        "invalid_coordinate_pairs": int(invalid_coordinates.sum()),
        "missing_gem_phase_id": int(frame["gem_phase_id"].isna().sum()),
        "duplicate_nonmissing_gem_phase_id_rows": int(phase_ids.duplicated().sum()),
        "operating_rows": int(len(operating)),
        "operating_capacity_reported_mw_sum": float(
            operating["capacity_reported_mw"].sum(min_count=1)
        ),
        "operating_missing_start_year": int(operating["start_year"].isna().sum()),
        "solar_capacity_uniform_mwac": False if spec.tracker == "solar" else None,
    }


def build_manifest(
    raw_root: Path,
    file_records: list[dict[str, Any]],
    generated_at: str,
) -> dict[str, Any]:
    return {
        "dataset": "GEM tracker historical CSV and current-map proxy vintages",
        "generated_at_utc": generated_at,
        "official_repository": GEM_REPOSITORY_URL,
        "raw_root": relative_path(raw_root),
        "temporal_mapping": {
            "2023": {
                "source_release": "2024-06",
                "role": "next_release_proxy_not_exact_year_end",
                "exact_year_end_snapshot": False,
            },
            "2024": {
                "source_release": "2025-02",
                "role": "next_release_proxy_not_exact_year_end",
                "exact_year_end_snapshot": False,
            },
            "2025": {
                "source_release": "2026-02",
                "role": "next_release_proxy_not_exact_year_end",
                "exact_year_end_snapshot": False,
            },
        },
        "files": file_records,
        "licence_url": GEM_LICENSE_URL,
        "licence_note": (
            "GEM's default licence and attribution requirements apply; consult "
            "GEM metadata for source-specific exceptions before redistribution."
        ),
    }


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root.resolve()
    output_root = args.output_root.resolve()
    raw_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    generated_at = utc_now()

    frames: list[pd.DataFrame] = []
    source_records: list[dict[str, Any]] = []
    subset_validations: list[dict[str, Any]] = []
    for spec in RELEASE_FILES:
        path = raw_root / spec.filename
        downloaded = download(spec.url, path, args.refresh)
        source_hash = sha256(path)
        raw = read_source_csv(path)
        harmonised = harmonise(raw, spec, source_hash, generated_at)
        if len(harmonised) != spec.expected_rows:
            raise ValueError(
                f"{spec.filename}: expected {spec.expected_rows:,} rows from "
                f"pinned commit, found {len(harmonised):,}"
            )
        frames.append(harmonised)
        subset_validations.append(validate_subset(harmonised, spec))
        source_records.append(
            {
                "tracker": spec.tracker,
                "proxy_target_year": spec.proxy_target_year,
                "source_release_label": spec.release_label,
                "source_commit_sha": spec.commit_sha,
                "url": spec.url,
                "file": relative_path(path),
                "bytes": path.stat().st_size,
                "sha256": source_hash,
                "rows": int(len(raw)),
                "downloaded_or_refreshed_this_run": downloaded,
                "file_kind": "official_commit_pinned_csv",
            }
        )

    current_raw_root = raw_root.parent / "2026-02"
    current_snapshot_path = output_root / CURRENT_SNAPSHOT_PATH.name
    current_raw, current_input_mode = load_current_snapshot(
        current_snapshot_path, current_raw_root
    )
    if current_snapshot_path.exists():
        source_records.append(
            {
                "tracker": "solar_and_wind",
                "proxy_target_year": 2025,
                "source_release_label": "2026-02",
                "file": relative_path(current_snapshot_path),
                "bytes": current_snapshot_path.stat().st_size,
                "sha256": sha256(current_snapshot_path),
                "rows": int(len(current_raw)),
                "file_kind": "derived_current_snapshot_input",
                "input_mode": current_input_mode,
            }
        )

    for spec in CURRENT_RELEASE_FILES:
        raw_path = current_raw_root / spec.filename
        if not raw_path.exists():
            raise FileNotFoundError(
                f"Underlying official 2026-02 GeoJSON is required for provenance: {raw_path}"
            )
        source_hash = sha256(raw_path)
        harmonised = harmonise_current_snapshot(
            current_raw, spec, source_hash, generated_at
        )
        if len(harmonised) != spec.expected_rows:
            raise ValueError(
                f"{spec.filename}: expected {spec.expected_rows:,} rows from "
                f"2026-02 map, found {len(harmonised):,}"
            )
        frames.append(harmonised)
        subset_validations.append(validate_subset(harmonised, spec))
        source_records.append(
            {
                "tracker": spec.tracker,
                "proxy_target_year": 2025,
                "source_release_label": "2026-02",
                "source_commit_sha": None,
                "url": spec.url,
                "file": relative_path(raw_path),
                "bytes": raw_path.stat().st_size,
                "sha256": source_hash,
                "rows": int(len(harmonised)),
                "downloaded_or_refreshed_this_run": False,
                "file_kind": "official_current_map_geojson",
                "input_mode": current_input_mode,
            }
        )

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(
        [
            "proxy_target_year",
            "tracker",
            "country_area",
            "gem_phase_id",
            "source_row_number",
        ],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)

    output_path = output_root / OUTPUT_NAME
    combined.to_parquet(output_path, index=False, compression="zstd")

    manifest = build_manifest(raw_root, source_records, generated_at)
    manifest_path = raw_root / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    expected_total = sum(
        spec.expected_rows for spec in (*RELEASE_FILES, *CURRENT_RELEASE_FILES)
    )
    validation = {
        "dataset": "GEM historical tracker vintage asset-phase proxy snapshot",
        "generated_at_utc": generated_at,
        "output": {
            "file": relative_path(output_path),
            "bytes": output_path.stat().st_size,
            "sha256": sha256(output_path),
            "rows": int(len(combined)),
            "columns": int(len(combined.columns)),
            "column_names": combined.columns.tolist(),
        },
        "coverage": {
            "requested_target_years": [2023, 2024, 2025],
            "available_proxy_target_years": [2023, 2024, 2025],
            "pending_target_years": [],
            "2025_status": (
                "available from GEM 2026-02 next-release proxy; not exact year-end"
            ),
            "current_map_input_mode": current_input_mode,
        },
        "temporal_validity": {
            "2023": "2024-06 next-release proxy; not an exact 2023 year-end snapshot",
            "2024": "2025-02 next-release proxy; not an exact 2024 year-end snapshot",
            "2025": "2026-02 next-release proxy; not an exact 2025 year-end snapshot",
        },
        "row_count": {
            "observed": int(len(combined)),
            "expected_from_pinned_commits_and_current_map": expected_total,
            "matches": len(combined) == expected_total,
        },
        "subsets": subset_validations,
        "whole_table_checks": {
            "duplicate_source_row_keys": int(
                combined.duplicated(
                    ["source_file_sha256", "source_row_number"]
                ).sum()
            ),
            "proxy_roles": sorted(combined["proxy_role"].dropna().unique().tolist()),
            "direct_ac_capacity_aggregation_allowed_rows": int(
                combined["safe_for_direct_ac_capacity_aggregation"].fillna(False).sum()
            ),
        },
        "limitations": [
            "All three target-year labels are next-release proxies, not exact calendar-year-end snapshots.",
            "GEM tracker research is updated at different times by geography; vintage changes mix real-world changes with research updates.",
            "The 2024-06 legacy files lack GEM location/phase IDs and retired-year, so exact cross-vintage entity matching is not guaranteed.",
            "Solar capacity is map-reported MW with mixed or unspecified rating and is not uniformly MWac; do not use its sum as AC supply capacity.",
            "The 2025 proxy uses the 2026-02 current map and therefore includes post-year-end research/status updates.",
            "Current-status vintages cannot by themselves guarantee complete reconstruction of assets retired before a snapshot.",
            "GEM default licensing applies, but source-specific record exceptions may require more restrictive reuse handling.",
        ],
        "validation_pass": bool(
            len(combined) == expected_total
            and not combined.duplicated(
                ["source_file_sha256", "source_row_number"]
            ).any()
            and all(
                item["row_count_matches_expected_release"]
                for item in subset_validations
            )
        ),
    }
    validation_path = output_root / VALIDATION_NAME
    validation_path.write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "output": relative_path(output_path),
                "validation": relative_path(validation_path),
                "manifest": relative_path(manifest_path),
                "rows": len(combined),
                "columns": len(combined.columns),
                "validation_pass": validation["validation_pass"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
