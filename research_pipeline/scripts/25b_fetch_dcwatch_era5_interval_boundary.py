#!/usr/bin/env python3
"""Fetch the 2026-01-01 00:00 UTC ERA5 boundary for interval alignment.

The 2023--2025 coupled model labels every hourly energy interval by its start
time ``[t, t+1)``.  Open-Meteo's radiation variables at timestamp ``u`` are
means over the preceding hour, while temperature, humidity, pressure, and
wind are instantaneous at ``u``.  Consequently, the final 2025 interval
needs all variables at 2026-01-01 00:00 UTC.  This small batched retrieval
provides that right endpoint without redownloading the complete archive.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG = (
    ROOT / "data" / "24x7_cfe" / "processed" / "climate_coupling" / "era5_grid_catalog.csv"
)
DEFAULT_OUTPUT = (
    ROOT
    / "data"
    / "24x7_cfe"
    / "raw"
    / "openmeteo"
    / "era5_2023_2025_dcwatch_grid"
    / "boundaries"
)
ENDPOINT = "https://archive-api.open-meteo.com/v1/archive"
BOUNDARY_TIME = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
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
METHOD_VERSION = "openmeteo_era5_dcwatch_interval_boundary_2026_00_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


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
            delay = (
                min(float(retry_after), 60.0)
                if retry_after.isdigit()
                else min(60.0, 1.8 * (2 ** (attempt - 1)) + random.random())
            )
        except requests.RequestException as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            delay = min(60.0, 1.8 * (2 ** (attempt - 1)) + random.random())
        if attempt < max_attempts:
            time.sleep(delay)
    raise RuntimeError(f"Open-Meteo boundary request failed: {last_error}")


def request_params(batch: pd.DataFrame) -> dict[str, Any]:
    latitudes = [f"{value:.6f}" for value in batch["era5_latitude"].astype(float)]
    longitudes = [f"{value:.6f}" for value in batch["era5_longitude"].astype(float)]
    return {
        "latitude": ",".join(latitudes),
        "longitude": ",".join(longitudes),
        "elevation": ",".join(["nan"] * len(batch)),
        "start_date": "2026-01-01",
        "end_date": "2026-01-01",
        "hourly": ",".join(VARIABLES),
        "models": "era5",
        "timezone": "GMT",
        "wind_speed_unit": "ms",
        "cell_selection": "nearest",
    }


def normalize_location(payload: dict[str, Any], row: Any) -> dict[str, Any]:
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict) or "time" not in hourly:
        raise ValueError("Boundary response lacks hourly data")
    missing = [name for name in VARIABLES if name not in hourly]
    if missing:
        raise ValueError(f"Boundary response lacks variables: {missing}")
    times = pd.to_datetime(hourly["time"], utc=True)
    matches = np.flatnonzero(times == BOUNDARY_TIME)
    if len(matches) != 1:
        raise ValueError(f"Expected one {BOUNDARY_TIME} row, found {len(matches)}")
    index = int(matches[0])
    result: dict[str, Any] = {
        "era5_grid_id": str(row.era5_grid_id),
        "requested_latitude": float(row.era5_latitude),
        "requested_longitude": float(row.era5_longitude),
        "response_latitude": float(payload["latitude"]),
        "response_longitude": float(payload["longitude"]),
        "response_elevation_m": float(payload.get("elevation", np.nan)),
        "time": BOUNDARY_TIME,
    }
    for variable in VARIABLES:
        result[variable] = float(hourly[variable][index])
    values = np.asarray([result[name] for name in VARIABLES], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite boundary data for {row.era5_grid_id}")
    if abs(result["response_latitude"] - result["requested_latitude"]) > 0.251:
        raise ValueError(f"Latitude mismatch for {row.era5_grid_id}")
    longitude_delta = abs(
        ((result["response_longitude"] - result["requested_longitude"] + 180.0) % 360.0)
        - 180.0
    )
    if longitude_delta > 0.251:
        raise ValueError(f"Longitude mismatch for {row.era5_grid_id}")
    return result


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    output_path = args.output_root / "era5_interval_boundary_2026-01-01_00.parquet"
    metadata_path = args.output_root / "era5_interval_boundary_2026-01-01_00.metadata.json"
    catalog = pd.read_csv(
        args.grid_catalog,
        usecols=["era5_grid_id", "era5_latitude", "era5_longitude"],
    ).sort_values("era5_grid_id")
    if len(catalog) != 2_416 or catalog["era5_grid_id"].nunique() != 2_416:
        raise ValueError("Expected the complete 2,416-grid ERA5 catalog")
    if output_path.exists() and metadata_path.exists() and not args.force:
        existing = pd.read_parquet(output_path)
        if len(existing) == len(catalog) and existing["era5_grid_id"].nunique() == len(catalog):
            print(f"Boundary is already complete: {output_path}")
            return

    args.output_root.mkdir(parents=True, exist_ok=True)
    raw_root = args.output_root / "raw_batches"
    raw_root.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update(
        {"User-Agent": "academic-24x7-cfe-climate-study/1.0 (Open-Meteo ERA5 retrieval)"}
    )
    records: list[dict[str, Any]] = []
    requests_metadata: list[dict[str, Any]] = []
    for start in range(0, len(catalog), args.batch_size):
        batch = catalog.iloc[start : start + args.batch_size]
        batch_number = start // args.batch_size + 1
        params = request_params(batch)
        response = request_with_retry(session, params, args.timeout, args.max_attempts)
        payload = response.json()
        payloads = payload if isinstance(payload, list) else [payload]
        if len(payloads) != len(batch):
            raise ValueError(
                f"Batch {batch_number} returned {len(payloads)} locations for {len(batch)} requests"
            )
        raw_path = raw_root / f"batch_{batch_number:04d}.json.gz"
        raw_tmp = raw_path.with_suffix(raw_path.suffix + ".tmp")
        with gzip.open(raw_tmp, "wb", compresslevel=6) as stream:
            stream.write(response.content)
        raw_tmp.replace(raw_path)
        for payload_item, row in zip(payloads, batch.itertuples(index=False), strict=True):
            records.append(normalize_location(payload_item, row))
        requests_metadata.append(
            {
                "batch_number": batch_number,
                "grid_count": len(batch),
                "first_grid_id": str(batch.iloc[0]["era5_grid_id"]),
                "last_grid_id": str(batch.iloc[-1]["era5_grid_id"]),
                "request_url": response.url,
                "response_bytes": len(response.content),
                "response_sha256": sha256_bytes(response.content),
                "raw_path": str(raw_path),
            }
        )
        if batch_number == 1 or batch_number % 10 == 0 or start + len(batch) == len(catalog):
            print(f"Fetched boundary for {start + len(batch):,}/{len(catalog):,} grids.")

    frame = pd.DataFrame(records).sort_values("era5_grid_id").reset_index(drop=True)
    if frame["era5_grid_id"].tolist() != catalog["era5_grid_id"].tolist():
        raise ValueError("Boundary grid identifiers do not match the catalog")
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(output_path)
    metadata = {
        "method_version": METHOD_VERSION,
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "endpoint": ENDPOINT,
        "boundary_time_utc": BOUNDARY_TIME.isoformat(),
        "normalized_rows": len(frame),
        "variables": list(VARIABLES),
        "interval_use": (
            "Right endpoint for instantaneous variables and preceding-hour "
            "radiation mean for [2025-12-31 23:00, 2026-01-01 00:00)."
        ),
        "market_sample_used": False,
        "requests": requests_metadata,
    }
    metadata_tmp = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    metadata_tmp.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata_tmp.replace(metadata_path)
    print(f"Wrote {len(frame):,} boundary records to {output_path}")


if __name__ == "__main__":
    main()
