#!/usr/bin/env python3
"""Fetch auditable 2023--2025 ERA5 weather for all DCWatch grid cells.

The downloader consumes ``era5_grid_catalog.csv`` created from the complete
logical-site inventory.  It is concurrent, resumable, and keeps the raw JSON,
normalized Parquet, request metadata, response checksum, and a run manifest.
No market sample is involved.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG = (
    ROOT
    / "data"
    / "24x7_cfe"
    / "processed"
    / "climate_coupling"
    / "era5_grid_catalog.csv"
)
DEFAULT_OUTPUT = (
    ROOT
    / "data"
    / "24x7_cfe"
    / "raw"
    / "openmeteo"
    / "era5_2023_2025_dcwatch_grid"
)
ENDPOINT = "https://archive-api.open-meteo.com/v1/archive"
VARIABLES = (
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "surface_pressure",
    "wind_speed_100m",
    "shortwave_radiation",
    "direct_normal_irradiance",
    "diffuse_radiation",
)
METHOD_VERSION = "openmeteo_era5_dcwatch_grid_2023_2025_v1"
RADIATION_NEGATIVE_CLIP_TOLERANCE_W_M2 = 5.0
PRINT_LOCK = Lock()


@dataclass(frozen=True)
class FetchResult:
    grid_id: str
    status: str
    rows: int
    response_bytes: int
    elapsed_seconds: float
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--grid-id", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--order",
        choices=("demand_priority", "grid_id"),
        default="demand_priority",
        help="Fetch high-demand cells first or use deterministic grid-ID order.",
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def expected_time_index(start_date: str, end_date: str) -> pd.DatetimeIndex:
    start = pd.Timestamp(start_date, tz="UTC")
    end_exclusive = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)
    return pd.date_range(start, end_exclusive, freq="h", inclusive="left")


def output_paths(output_root: Path, grid_id: str) -> dict[str, Path]:
    return {
        "raw": output_root / f"{grid_id}.json.gz",
        "parquet": output_root / f"{grid_id}.parquet",
        "metadata": output_root / f"{grid_id}.metadata.json",
    }


def is_complete(paths: dict[str, Path], expected_rows: int) -> bool:
    if not paths["parquet"].exists() or not paths["metadata"].exists():
        return False
    try:
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        if metadata.get("method_version") != METHOD_VERSION:
            return False
        if int(metadata.get("normalized_rows", -1)) != expected_rows:
            return False
        frame = pd.read_parquet(paths["parquet"], columns=["time"])
        return len(frame) == expected_rows
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def request_params(row: pd.Series, start_date: str, end_date: str) -> dict[str, Any]:
    return {
        "latitude": round(float(row["era5_latitude"]), 6),
        "longitude": round(float(row["era5_longitude"]), 6),
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(VARIABLES),
        "models": "era5",
        "timezone": "GMT",
        "wind_speed_unit": "ms",
        "cell_selection": "nearest",
        # Disable Open-Meteo's elevation adjustment so every facility mapped
        # to the same native ERA5 grid centre receives the same grid weather.
        "elevation": "nan",
    }


def request_with_retry(
    session: requests.Session,
    params: dict[str, Any],
    timeout: float,
    max_attempts: int,
) -> requests.Response:
    last_error = "unknown failure"
    for attempt in range(1, max_attempts + 1):
        try:
            response = session.get(ENDPOINT, params=params, timeout=timeout)
            if response.status_code == 200:
                return response
            last_error = f"HTTP {response.status_code}: {response.text[:300]}"
            if response.status_code not in {429, 500, 502, 503, 504}:
                break
            retry_after = response.headers.get("Retry-After", "")
            if retry_after.isdigit():
                delay = min(float(retry_after), 60.0)
            else:
                delay = min(60.0, 1.8 * (2 ** (attempt - 1)) + random.random())
        except requests.RequestException as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            delay = min(60.0, 1.8 * (2 ** (attempt - 1)) + random.random())
        if attempt < max_attempts:
            time.sleep(delay)
    raise RuntimeError(f"Open-Meteo request failed after {max_attempts} attempts: {last_error}")


def normalize_payload(
    payload: dict[str, Any],
    row: pd.Series,
    expected_times: pd.DatetimeIndex,
) -> pd.DataFrame:
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict) or "time" not in hourly:
        raise ValueError("Response lacks hourly time series")
    missing = [variable for variable in VARIABLES if variable not in hourly]
    if missing:
        raise ValueError(f"Response lacks required variables: {missing}")
    frame = pd.DataFrame({"time": hourly["time"], **{name: hourly[name] for name in VARIABLES}})
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    if not pd.DatetimeIndex(frame["time"]).equals(expected_times):
        raise ValueError(
            f"Timestamp coverage mismatch: {len(frame):,} rows, expected {len(expected_times):,}"
        )
    values = frame[list(VARIABLES)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        bad = int((~np.isfinite(values)).sum())
        raise ValueError(f"Weather response contains {bad} non-finite values")
    if not frame["relative_humidity_2m"].between(0.0, 100.0).all():
        raise ValueError("Relative humidity is outside 0--100%")
    if (frame["surface_pressure"] <= 0.0).any():
        raise ValueError("Surface pressure is non-positive")
    if (frame["wind_speed_100m"] < -1e-9).any():
        raise ValueError("100 m wind speed is negative")
    radiation_columns = [
        "shortwave_radiation",
        "direct_normal_irradiance",
        "diffuse_radiation",
    ]
    radiation_values = frame[radiation_columns].to_numpy(dtype=float)
    negative_mask = radiation_values < 0.0
    negative_count = int(negative_mask.sum())
    negative_min = float(radiation_values.min())
    if negative_min < -RADIATION_NEGATIVE_CLIP_TOLERANCE_W_M2:
        raise ValueError(
            "Radiation value is below the documented numerical clipping tolerance: "
            f"{negative_min} W/m2"
        )
    # ERA5/Open-Meteo occasionally returns isolated -1 or -2 W/m2 diffuse
    # values at night.  Preserve the raw response and audit statistics, then
    # clip only these tiny interpolation/rounding artefacts to the physical 0.
    if negative_count:
        frame[radiation_columns] = frame[radiation_columns].clip(lower=0.0)
    frame.attrs["radiation_negative_clipped_count"] = negative_count
    frame.attrs["radiation_negative_min_w_m2"] = negative_min
    if (frame["dew_point_2m"] > frame["temperature_2m"] + 1.0).any():
        count = int((frame["dew_point_2m"] > frame["temperature_2m"] + 1.0).sum())
        raise ValueError(f"Dew point exceeds dry-bulb temperature by >1 C in {count} hours")

    frame.insert(0, "era5_grid_id", str(row["era5_grid_id"]))
    frame.insert(1, "requested_latitude", float(row["era5_latitude"]))
    frame.insert(2, "requested_longitude", float(row["era5_longitude"]))
    frame.insert(3, "response_latitude", float(payload["latitude"]))
    frame.insert(4, "response_longitude", float(payload["longitude"]))
    frame.insert(5, "response_elevation_m", float(payload.get("elevation", np.nan)))
    return frame


def atomic_outputs(
    paths: dict[str, Path],
    response: requests.Response,
    frame: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    paths["parquet"].parent.mkdir(parents=True, exist_ok=True)
    temporary_raw = paths["raw"].with_suffix(paths["raw"].suffix + ".tmp")
    temporary_parquet = paths["parquet"].with_suffix(paths["parquet"].suffix + ".tmp")
    temporary_metadata = paths["metadata"].with_suffix(paths["metadata"].suffix + ".tmp")
    with gzip.open(temporary_raw, "wb", compresslevel=6) as stream:
        stream.write(response.content)
    frame.to_parquet(temporary_parquet, index=False, compression="zstd")
    temporary_metadata.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary_raw.replace(paths["raw"])
    temporary_parquet.replace(paths["parquet"])
    temporary_metadata.replace(paths["metadata"])


def fetch_one(
    row: pd.Series,
    output_root: Path,
    start_date: str,
    end_date: str,
    expected_times: pd.DatetimeIndex,
    timeout: float,
    max_attempts: int,
    force: bool,
) -> FetchResult:
    grid_id = str(row["era5_grid_id"])
    paths = output_paths(output_root, grid_id)
    if not force and is_complete(paths, len(expected_times)):
        return FetchResult(grid_id, "skipped_complete", len(expected_times), 0, 0.0)
    started = time.perf_counter()
    session = requests.Session()
    session.headers.update(
        {"User-Agent": "academic-24x7-cfe-climate-study/1.0 (Open-Meteo ERA5 retrieval)"}
    )
    params = request_params(row, start_date, end_date)
    try:
        response = request_with_retry(session, params, timeout, max_attempts)
        payload = response.json()
        frame = normalize_payload(payload, row, expected_times)
        metadata = {
            "method_version": METHOD_VERSION,
            "retrieved_at_utc": datetime.now(UTC).isoformat(),
            "era5_grid_id": grid_id,
            "endpoint": ENDPOINT,
            "request_url": response.url,
            "parameters": params,
            "http_status": response.status_code,
            "response_sha256": sha256_bytes(response.content),
            "response_bytes": len(response.content),
            "normalized_rows": len(frame),
            "start_time_utc": frame["time"].iloc[0].isoformat(),
            "end_time_utc": frame["time"].iloc[-1].isoformat(),
            "units": payload.get("hourly_units", {}),
            "source": "Open-Meteo Historical Weather API; model=ERA5",
            "radiation_negative_clip_tolerance_w_m2": (
                RADIATION_NEGATIVE_CLIP_TOLERANCE_W_M2
            ),
            "radiation_negative_clipped_count": int(
                frame.attrs.get("radiation_negative_clipped_count", 0)
            ),
            "radiation_negative_min_w_m2": float(
                frame.attrs.get("radiation_negative_min_w_m2", 0.0)
            ),
            "market_sample_used": False,
        }
        atomic_outputs(paths, response, frame, metadata)
        elapsed = time.perf_counter() - started
        return FetchResult(grid_id, "downloaded", len(frame), len(response.content), elapsed)
    except Exception as exc:  # worker must report failure without cancelling completed cells
        elapsed = time.perf_counter() - started
        return FetchResult(grid_id, "failed", 0, 0, elapsed, f"{type(exc).__name__}: {exc}")
    finally:
        session.close()


def main() -> None:
    args = parse_args()
    if args.workers <= 0 or args.max_attempts <= 0:
        raise ValueError("--workers and --max-attempts must be positive")
    catalog = pd.read_csv(args.grid_catalog)
    required = {"era5_grid_id", "era5_latitude", "era5_longitude"}
    if not required.issubset(catalog.columns):
        raise ValueError(f"Grid catalog lacks {sorted(required - set(catalog.columns))}")
    if catalog["era5_grid_id"].duplicated().any():
        raise ValueError("Grid catalog contains duplicate era5_grid_id values")
    if args.grid_id:
        requested = set(args.grid_id)
        missing = requested - set(catalog["era5_grid_id"])
        if missing:
            raise ValueError(f"Unknown --grid-id values: {sorted(missing)}")
        catalog = catalog[catalog["era5_grid_id"].isin(requested)].copy()
    if args.order == "demand_priority":
        if "site_mean_it_mw_p50_sum" not in catalog.columns:
            raise ValueError("Demand-priority order requires site_mean_it_mw_p50_sum")
        catalog = catalog.sort_values(
            ["site_mean_it_mw_p50_sum", "era5_grid_id"],
            ascending=[False, True],
            kind="stable",
        ).reset_index(drop=True)
    else:
        catalog = catalog.sort_values("era5_grid_id", kind="stable").reset_index(drop=True)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        catalog = catalog.head(args.limit).copy()
    if catalog.empty:
        raise ValueError("No ERA5 grid cells selected")

    expected_times = expected_time_index(args.start_date, args.end_date)
    args.output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    results: list[FetchResult] = []
    total = len(catalog)
    print(
        f"Fetching {total:,} full-inventory ERA5 grid cells for "
        f"{args.start_date}--{args.end_date} with {args.workers} workers."
    )
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                fetch_one,
                row,
                args.output_root,
                args.start_date,
                args.end_date,
                expected_times,
                args.timeout,
                args.max_attempts,
                args.force,
            ): str(row["era5_grid_id"])
            for _, row in catalog.iterrows()
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if completed == 1 or completed % 25 == 0 or completed == total or result.status == "failed":
                downloaded = sum(item.status == "downloaded" for item in results)
                skipped = sum(item.status == "skipped_complete" for item in results)
                failed = sum(item.status == "failed" for item in results)
                with PRINT_LOCK:
                    print(
                        f"Progress {completed:,}/{total:,}: downloaded={downloaded:,}, "
                        f"skipped={skipped:,}, failed={failed:,}"
                    )

    elapsed = time.perf_counter() - started
    result_frame = pd.DataFrame([item.__dict__ for item in results]).sort_values("grid_id")
    manifest_path = args.output_root / "download_manifest.csv"
    result_frame.to_csv(manifest_path, index=False)
    failures = result_frame[result_frame["status"].eq("failed")]
    summary = {
        "method_version": METHOD_VERSION,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "grid_catalog": str(args.grid_catalog),
        "selected_grid_cells": total,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "expected_hours_per_grid": len(expected_times),
        "variables": list(VARIABLES),
        "workers": args.workers,
        "fetch_order": args.order,
        "downloaded": int(result_frame["status"].eq("downloaded").sum()),
        "skipped_complete": int(result_frame["status"].eq("skipped_complete").sum()),
        "failed": int(len(failures)),
        "elapsed_seconds": elapsed,
        "response_bytes_downloaded": int(result_frame["response_bytes"].sum()),
        "market_sample_used": False,
        "manifest": str(manifest_path),
    }
    summary_path = args.output_root / "download_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"Finished in {elapsed / 60.0:.1f} min: "
        f"{total - len(failures):,}/{total:,} complete."
    )
    if not failures.empty:
        examples = failures[["grid_id", "error"]].head(20).to_dict("records")
        raise RuntimeError(
            f"{len(failures)} grid cells failed. Re-run to resume. Examples: {examples}"
        )


if __name__ == "__main__":
    main()
