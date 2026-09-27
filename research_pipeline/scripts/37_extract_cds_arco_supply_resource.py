#!/usr/bin/env python3
"""Extract 1995--2014 supply-grid capacity factors from CDS ERA5 ARCO.

The CDS geo-chunked Zarr is designed for long time series at limited spatial
locations.  Each completed grid is stored as a compact NPZ of interval
capacity factors plus non-secret provenance metadata.  The extractor is
concurrent, resumable and never writes the CDS API key to a URL, manifest or
metadata file.

Authentication is read from the ``CDSAPI_KEY`` environment variable or the
``key:`` entry in ``~/.cdsapirc``.  Do not pass credentials on the command
line, where they could be exposed through process listings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import OrderedDict
from collections.abc import Iterator, MutableMapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock, local
from typing import Any
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = ROOT / "research_pipeline"
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from supply_resource_models import (  # noqa: E402
    CENTRAL_WIND_SCENARIO,
    MODEL_VERSION as RESOURCE_MODEL_VERSION,
    interval_capacity_factors,
)


DEFAULT_CATALOG = (
    ROOT
    / "data"
    / "24x7_cfe"
    / "processed"
    / "wind_solar_supply_20y_fixed_2025"
    / "supply_asset_grid_catalog_2025.csv"
)
IN_TREE_OUTPUT_ROOT = (
    ROOT
    / "data"
    / "24x7_cfe"
    / "processed"
    / "wind_solar_supply_20y_fixed_2025"
    / "resource_cf_1995_2014"
)
# Roughly 43,000 files land here.  Keeping them off a continuously synchronised
# cloud folder is a deployment choice, so the location stays overridable.
DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get("SUPPLY_RESOURCE_CF_ROOT") or IN_TREE_OUTPUT_ROOT
)
SOURCE_URL = (
    "https://arco.datastores.ecmwf.int/cadl-arco-geo-002/arco/"
    "reanalysis_era5_single_levels/sfc/geoChunked.zarr"
)
METHOD_VERSION = "cds_era5_arco_geo_fixed_supply_resource_1995_2014_v1"
START_ENDPOINT = pd.Timestamp("1995-01-01 00:00:00", tz="UTC")
END_ENDPOINT = pd.Timestamp("2015-01-01 00:00:00", tz="UTC")
EXPECTED_ENDPOINTS = pd.date_range(START_ENDPOINT, END_ENDPOINT, freq="h")
EXPECTED_ENDPOINT_COUNT = 175_321
EXPECTED_INTERVAL_COUNT = 175_320
SOLAR_VARIABLES = (
    "2m_temperature",
    "surface_solar_radiation_downwards",
    "total_sky_direct_solar_radiation_at_surface",
)
WIND_VARIABLES = (
    "2m_temperature",
    "surface_pressure",
    "100m_u_component_of_wind",
    "100m_v_component_of_wind",
)
VARIABLE_ALIASES = {
    "2m_temperature": ("2m_temperature", "t2m"),
    "surface_pressure": ("surface_pressure", "sp"),
    "100m_u_component_of_wind": ("100m_u_component_of_wind", "u100"),
    "100m_v_component_of_wind": ("100m_v_component_of_wind", "v100"),
    "surface_solar_radiation_downwards": (
        "surface_solar_radiation_downwards",
        "ssrd",
    ),
    "total_sky_direct_solar_radiation_at_surface": (
        "total_sky_direct_solar_radiation_at_surface",
        "fdir",
    ),
}
EXPECTED_UNITS = {
    "2m_temperature": {"k", "kelvin"},
    "surface_pressure": {"pa", "pascal", "pascals"},
    "100m_u_component_of_wind": {"ms-1", "m/s", "ms−1"},
    "100m_v_component_of_wind": {"ms-1", "m/s", "ms−1"},
    "surface_solar_radiation_downwards": {"jm-2", "j/m2", "jm−2"},
    "total_sky_direct_solar_radiation_at_surface": {"jm-2", "j/m2", "jm−2"},
}
THREAD_STATE = local()
PRINT_LOCK = Lock()
WARNED_LAYOUT = set()

# The geo-chunked store packs a 4x4 block of 0.25 degree cells and 67,584 hours
# into one chunk, so 1995--2014 spans three time chunks.  Grouping the supply
# grids by that block is what keeps every chunk from being fetched repeatedly;
# it is a scheduling optimisation only, because the authoritative point lookup
# stays `sel(..., method="nearest")` on the store's own axes.
GEO_GRID_STEP = 0.25
GEO_LATITUDE_ORIGIN = -90.0
GEO_LONGITUDE_ORIGIN = -180.0
GEO_LONGITUDE_COUNT = 1440
GEO_SPATIAL_CHUNK = 4


def supply_chunk_cell(latitude: float, longitude: float) -> tuple[int, int]:
    """Return the (latitude, longitude) Zarr chunk block holding a point."""

    latitude_index = int(round((latitude - GEO_LATITUDE_ORIGIN) / GEO_GRID_STEP))
    longitude_index = (
        int(round((longitude - GEO_LONGITUDE_ORIGIN) % 360.0 / GEO_GRID_STEP))
        % GEO_LONGITUDE_COUNT
    )
    return latitude_index // GEO_SPATIAL_CHUNK, longitude_index // GEO_SPATIAL_CHUNK


def warn_once(message: str) -> None:
    with PRINT_LOCK:
        if message not in WARNED_LAYOUT:
            WARNED_LAYOUT.add(message)
            print(f"WARNING: {message}", file=sys.stderr)


def check_geo_chunk_layout(dataset: Any) -> None:
    """Warn when the store layout no longer matches the grouping assumptions.

    A layout change only costs redundant downloads, never correctness, so this
    never aborts a run that is otherwise valid.
    """

    chunks = dataset["2m_temperature"].encoding.get("chunks")
    if chunks is not None and tuple(chunks[1:]) != (
        GEO_SPATIAL_CHUNK,
        GEO_SPATIAL_CHUNK,
    ):
        warn_once(
            f"ERA5 ARCO spatial chunking is {tuple(chunks[1:])}, not "
            f"({GEO_SPATIAL_CHUNK}, {GEO_SPATIAL_CHUNK}); chunk-locality "
            "ordering will fetch more data than planned."
        )
    latitude = dataset["latitude"].values
    longitude = dataset["longitude"].values
    if (
        abs(float(latitude[0]) - GEO_LATITUDE_ORIGIN) > 1e-6
        or abs(float(longitude[0]) - GEO_LONGITUDE_ORIGIN) > 1e-6
        or longitude.size != GEO_LONGITUDE_COUNT
    ):
        warn_once(
            f"ERA5 ARCO axes start at ({latitude[0]}, {longitude[0]}) with "
            f"{longitude.size} longitudes; chunk-locality ordering assumes "
            f"({GEO_LATITUDE_ORIGIN}, {GEO_LONGITUDE_ORIGIN}) and "
            f"{GEO_LONGITUDE_COUNT}."
        )


@dataclass(frozen=True)
class ExtractResult:
    grid_id: str
    status: str
    interval_count: int
    output_bytes: int
    elapsed_seconds: float
    error: str | None = None


class RequestsReadOnlyStore(MutableMapping[str, bytes]):
    """Minimal retrying Zarr mapping that keeps bearer credentials in memory."""

    def __init__(
        self,
        base_url: str,
        bearer_token: str,
        cache_items: int = 24,
        body_attempts: int = 5,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.body_attempts = max(1, body_attempts)
        self.session = requests.Session()
        self.cache: OrderedDict[str, bytes] = OrderedDict()
        # One 4x4 chunk block needs six variables x three time chunks, so the
        # cache must hold 18 chunks for a whole block to be read exactly once.
        self.cache_items = max(1, cache_items)
        if bearer_token:
            self.session.headers.update({"Authorization": f"Bearer {bearer_token}"})
        retry = Retry(
            total=6,
            connect=6,
            read=6,
            status=6,
            backoff_factor=0.75,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET", "HEAD"}),
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def __getitem__(self, key: str) -> bytes:
        normalized_key = str(key).lstrip("/")
        if normalized_key in self.cache:
            value = self.cache.pop(normalized_key)
            self.cache[normalized_key] = value
            return value
        encoded_key = quote(normalized_key, safe="/._-")
        url = f"{self.base_url}/{encoded_key}"
        # urllib3's Retry covers connect/read failures before the body starts,
        # but a chunked body truncated mid-transfer surfaces later as
        # ChunkedEncodingError, so multi-megabyte chunks need their own retry.
        for attempt in range(1, self.body_attempts + 1):
            try:
                response = self.session.get(url, timeout=(30.0, 300.0))
                if response.status_code == 404:
                    raise KeyError(key)
                response.raise_for_status()
                value = response.content
                break
            except (
                requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
            ):
                if attempt == self.body_attempts:
                    raise
                time.sleep(min(20.0, 1.5**attempt + random.random()))
        self.cache[normalized_key] = value
        while len(self.cache) > self.cache_items:
            self.cache.popitem(last=False)
        return value

    def __contains__(self, key: object) -> bool:
        try:
            self[str(key)]
            return True
        except KeyError:
            return False

    def __setitem__(self, key: str, value: bytes) -> None:
        raise PermissionError("ERA5 ARCO store is read-only")

    def __delitem__(self, key: str) -> None:
        raise PermissionError("ERA5 ARCO store is read-only")

    def __iter__(self) -> Iterator[str]:
        # Consolidated Zarr metadata supplies the key index, so remote listing
        # is intentionally disabled and never needed by this workflow.
        return iter(())

    def __len__(self) -> int:
        return 0

    def close(self) -> None:
        self.cache.clear()
        self.session.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--source-url", default=SOURCE_URL)
    parser.add_argument("--cdsapirc", type=Path, default=Path.home() / ".cdsapirc")
    parser.add_argument("--key-env", default="CDSAPI_KEY")
    parser.add_argument("--grid-id", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--order",
        choices=("capacity_cell", "capacity_priority", "grid_id"),
        default="capacity_cell",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--chunk-cache",
        type=int,
        default=24,
        help="Chunks held per worker; 18 covers one 4x4 block for all variables.",
    )
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_npz(arrays: dict[str, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


PLACEHOLDER_KEYS = {
    "paste-your-personal-access-token-here",
    "<personal-access-token>",
    "personal-access-token",
}


def reject_placeholder(value: str, source: str) -> str:
    if value.casefold() in PLACEHOLDER_KEYS:
        raise RuntimeError(
            f"{source} still holds the placeholder token. Log in to the "
            "Copernicus Climate Data Store, accept the ERA5 licence and replace "
            "the key: line with your personal access token."
        )
    return value


def resolve_cds_key(key_env: str, cdsapirc: Path) -> tuple[str, str]:
    environment_value = os.environ.get(key_env, "").strip()
    if environment_value:
        source = f"environment:{key_env}"
        return reject_placeholder(environment_value, source), source
    if cdsapirc.exists():
        for raw_line in cdsapirc.read_text(encoding="utf-8-sig").splitlines():
            line = raw_line.strip()
            if line.casefold().startswith("key:"):
                value = line.split(":", 1)[1].strip()
                if value:
                    source = f"file:{cdsapirc}"
                    return reject_placeholder(value, source), source
    raise RuntimeError(
        "CDS API key not found. Set CDSAPI_KEY or create ~/.cdsapirc after "
        "logging in to the Copernicus Climate Data Store and accepting the ERA5 licence."
    )


def import_arco_dependencies() -> Any:
    try:
        import xarray as xr
        import zarr  # noqa: F401
    except ImportError as exc:
        requirements = ROOT / "research_pipeline" / "requirements_supply_replay.txt"
        raise RuntimeError(
            f"Missing ERA5 ARCO dependency ({exc}). Install {requirements}."
        ) from exc
    return xr


def normalized_unit(value: Any) -> str:
    return (
        str(value)
        .strip()
        .casefold()
        .replace(" ", "")
        .replace("**", "")
        .replace("^", "")
        .replace("²", "2")
    )


def canonicalize_variables(dataset: Any) -> Any:
    """Accept both CDS descriptive names and common ERA5 short names."""

    rename: dict[str, str] = {}
    missing: list[str] = []
    for canonical, aliases in VARIABLE_ALIASES.items():
        actual = next((name for name in aliases if name in dataset.data_vars), None)
        if actual is None:
            missing.append(canonical)
        elif actual != canonical:
            rename[actual] = canonical
    if missing:
        dataset.close()
        raise ValueError(f"CDS ARCO dataset lacks variables: {sorted(missing)}")
    if rename:
        dataset = dataset.rename(rename)
    for name, allowed in EXPECTED_UNITS.items():
        unit = dataset[name].attrs.get("units")
        if unit is not None and normalized_unit(unit) not in allowed:
            dataset.close()
            raise ValueError(
                f"CDS ARCO variable {name} has unexpected units {unit!r}; "
                f"expected one of {sorted(allowed)}"
            )
    return dataset


def open_dataset(source_url: str, key: str, cache_items: int = 24) -> Any:
    xr = import_arco_dependencies()
    store = RequestsReadOnlyStore(source_url, key, cache_items=cache_items)
    try:
        # Avoid constructing a Dask chunk tuple for every hourly chunk in the
        # multi-decade archive.  Point selections are read directly from Zarr.
        dataset = xr.open_zarr(store, consolidated=True, chunks=None)
    except Exception:
        store.close()
        raise
    dataset.set_close(store.close)
    required_coordinates = {"time", "latitude", "longitude"}
    missing_coordinates = required_coordinates - set(dataset.coords)
    if missing_coordinates:
        dataset.close()
        raise ValueError(f"CDS ARCO dataset lacks coordinates: {sorted(missing_coordinates)}")
    dataset = canonicalize_variables(dataset)
    dataset.set_close(store.close)
    check_geo_chunk_layout(dataset)
    return dataset


def thread_dataset(source_url: str, key: str, cache_items: int = 24) -> Any:
    identity = (source_url, hashlib.sha256(key.encode("utf-8")).hexdigest())
    if getattr(THREAD_STATE, "identity", None) != identity:
        previous = getattr(THREAD_STATE, "dataset", None)
        if previous is not None:
            previous.close()
        THREAD_STATE.dataset = open_dataset(source_url, key, cache_items=cache_items)
        THREAD_STATE.identity = identity
    return THREAD_STATE.dataset


def drop_thread_dataset() -> None:
    """Close and forget this thread's dataset so a retry reconnects cleanly."""

    dataset = getattr(THREAD_STATE, "dataset", None)
    if dataset is not None:
        try:
            dataset.close()
        except Exception:  # a broken session must not mask the original error
            pass
    THREAD_STATE.dataset = None
    THREAD_STATE.identity = None


def output_paths(output_root: Path, grid_id: str) -> dict[str, Path]:
    return {
        "resource": output_root / f"{grid_id}.npz",
        "metadata": output_root / f"{grid_id}.metadata.json",
    }


def expected_array_names(row: pd.Series) -> set[str]:
    names: set[str] = set()
    if bool(row["has_solar_pv"]):
        names.add("pv_cf")
    if bool(row["has_wind"]):
        names.add("wind_cf")
    return names


def is_complete(
    paths: dict[str, Path], row: pd.Series, catalog_sha256: str
) -> bool:
    if not paths["resource"].exists() or not paths["metadata"].exists():
        return False
    try:
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        if (
            metadata.get("method_version") != METHOD_VERSION
            or metadata.get("resource_model_version") != RESOURCE_MODEL_VERSION
            or metadata.get("grid_catalog_sha256") != catalog_sha256
            or int(metadata.get("interval_count", -1)) != EXPECTED_INTERVAL_COUNT
            or metadata.get("output_sha256") != sha256_file(paths["resource"])
        ):
            return False
        with np.load(paths["resource"]) as archive:
            if set(archive.files) != expected_array_names(row):
                return False
            return all(
                archive[name].shape == (EXPECTED_INTERVAL_COUNT,)
                and np.isfinite(archive[name]).all()
                for name in archive.files
            )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def utc_time_index(values: Any) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(pd.to_datetime(values, utc=True))
    if not index.equals(EXPECTED_ENDPOINTS):
        raise ValueError(
            f"ERA5 endpoint mismatch: got {len(index):,}, expected "
            f"{EXPECTED_ENDPOINT_COUNT:,} continuous hourly endpoints"
        )
    return index


def load_point_arrays(dataset: Any, row: pd.Series) -> tuple[pd.DatetimeIndex, dict[str, np.ndarray], dict[str, float]]:
    requested_variables: set[str] = set()
    if bool(row["has_solar_pv"]):
        requested_variables.update(SOLAR_VARIABLES)
    if bool(row["has_wind"]):
        requested_variables.update(WIND_VARIABLES)
    latitude = float(row["era5_latitude"])
    longitude = float(row["era5_longitude"])
    point = (
        dataset[sorted(requested_variables)]
        .sel(latitude=latitude, longitude=longitude, method="nearest")
        .sel(time=slice(START_ENDPOINT.tz_localize(None), END_ENDPOINT.tz_localize(None)))
        .load()
    )
    times = utc_time_index(point["time"].values)
    response_coordinates = {
        "latitude": float(point["latitude"].values),
        "longitude": float(point["longitude"].values),
    }
    if abs(response_coordinates["latitude"] - latitude) > 0.126:
        raise ValueError("CDS ARCO latitude selection is farther than half a grid cell")
    longitude_error = abs(
        ((response_coordinates["longitude"] - longitude + 180.0) % 360.0) - 180.0
    )
    if longitude_error > 0.126:
        raise ValueError("CDS ARCO longitude selection is farther than half a grid cell")
    arrays = {
        name: np.asarray(point[name].values, dtype=float).reshape(-1)
        for name in requested_variables
    }
    if any(len(values) != EXPECTED_ENDPOINT_COUNT for values in arrays.values()):
        raise ValueError("CDS ARCO variable endpoint length mismatch")
    if not all(np.isfinite(values).all() for values in arrays.values()):
        raise ValueError("CDS ARCO point contains non-finite values")
    temperature = arrays["2m_temperature"]
    if temperature.min() < 150.0 or temperature.max() > 350.0:
        raise ValueError("CDS ARCO 2 m temperature is outside plausible Kelvin bounds")
    if "surface_pressure" in arrays:
        pressure = arrays["surface_pressure"]
        if pressure.min() < 20_000.0 or pressure.max() > 120_000.0:
            raise ValueError("CDS ARCO surface pressure is outside plausible Pa bounds")
    for name in ("100m_u_component_of_wind", "100m_v_component_of_wind"):
        if name in arrays and np.abs(arrays[name]).max() > 150.0:
            raise ValueError(f"CDS ARCO {name} is outside plausible m/s bounds")
    for name in (
        "surface_solar_radiation_downwards",
        "total_sky_direct_solar_radiation_at_surface",
    ):
        if name in arrays and arrays[name].max() > 10_000_000.0:
            raise ValueError(f"CDS ARCO {name} is outside plausible hourly J/m2 bounds")
    return times, arrays, response_coordinates


def extract_one(
    row: pd.Series,
    output_root: Path,
    source_url: str,
    key: str,
    catalog_sha256: str,
    max_attempts: int,
    force: bool,
    cache_items: int = 24,
) -> ExtractResult:
    grid_id = str(row["supply_era5_grid_id"])
    paths = output_paths(output_root, grid_id)
    if not force and is_complete(paths, row, catalog_sha256):
        return ExtractResult(grid_id, "skipped_complete", EXPECTED_INTERVAL_COUNT, 0, 0.0)
    started = time.perf_counter()
    last_error = "unknown failure"
    for attempt in range(1, max_attempts + 1):
        try:
            dataset = thread_dataset(source_url, key, cache_items=cache_items)
            times, weather, response_coordinates = load_point_arrays(dataset, row)
            kwargs: dict[str, Any] = {}
            if bool(row["has_solar_pv"]):
                kwargs.update(
                    {
                        "surface_solar_radiation_downwards_j_m2": weather[
                            "surface_solar_radiation_downwards"
                        ],
                        "total_sky_direct_solar_radiation_at_surface_j_m2": weather[
                            "total_sky_direct_solar_radiation_at_surface"
                        ],
                    }
                )
            if bool(row["has_wind"]):
                kwargs.update(
                    {
                        "surface_pressure_pa": weather["surface_pressure"],
                        "wind_u_100m_m_s": weather["100m_u_component_of_wind"],
                        "wind_v_100m_m_s": weather["100m_v_component_of_wind"],
                    }
                )
            resources, diagnostics = interval_capacity_factors(
                times,
                float(row["era5_latitude"]),
                float(row["era5_longitude"]),
                weather["2m_temperature"],
                wind_scenario=CENTRAL_WIND_SCENARIO,
                **kwargs,
            )
            atomic_npz(resources, paths["resource"])
            metadata = {
                "method_version": METHOD_VERSION,
                "resource_model_version": RESOURCE_MODEL_VERSION,
                "retrieved_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
                "supply_era5_grid_id": grid_id,
                "requested_latitude": float(row["era5_latitude"]),
                "requested_longitude": float(row["era5_longitude"]),
                "response_latitude": response_coordinates["latitude"],
                "response_longitude": response_coordinates["longitude"],
                "source": "Copernicus CDS ERA5 ARCO geo-chunked",
                "source_url": source_url,
                "source_authorization_recorded": False,
                "variables": sorted(weather),
                "endpoint_start_utc": START_ENDPOINT.isoformat(),
                "endpoint_end_utc": END_ENDPOINT.isoformat(),
                "endpoint_count": EXPECTED_ENDPOINT_COUNT,
                "interval_count": EXPECTED_INTERVAL_COUNT,
                "interval_definition": "left-labelled [t,t+1) UTC; final right endpoint retained",
                "radiation_timestamp_interpretation": "ERA5 one-hour accumulation timestamp is interval end",
                "wind_scenario": CENTRAL_WIND_SCENARIO,
                "grid_catalog_sha256": catalog_sha256,
                "output_arrays": {
                    name: {"dtype": str(values.dtype), "rows": len(values)}
                    for name, values in resources.items()
                },
                "diagnostics": diagnostics,
                "output_sha256": sha256_file(paths["resource"]),
                "is_historical_actual_generation": False,
                "data_center_demand_used": False,
            }
            atomic_json(metadata, paths["metadata"])
            elapsed = time.perf_counter() - started
            return ExtractResult(
                grid_id,
                "extracted",
                EXPECTED_INTERVAL_COUNT,
                paths["resource"].stat().st_size,
                elapsed,
            )
        except Exception as exc:  # keep completed grids resumable
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < max_attempts:
                drop_thread_dataset()
                time.sleep(min(60.0, 2.0**attempt + random.random()))
    return ExtractResult(
        grid_id,
        "failed",
        0,
        0,
        time.perf_counter() - started,
        last_error,
    )


def extract_cell(
    rows: list[pd.Series],
    output_root: Path,
    source_url: str,
    key: str,
    catalog_sha256: str,
    max_attempts: int,
    force: bool,
    cache_items: int,
) -> list[ExtractResult]:
    """Extract every supply grid sharing one Zarr chunk block on one thread.

    Consecutive points reuse the thread-local chunk cache, so the block's 18
    chunks are downloaded once instead of once per point.
    """

    return [
        extract_one(
            row,
            output_root,
            source_url,
            key,
            catalog_sha256,
            max_attempts,
            force,
            cache_items=cache_items,
        )
        for row in rows
    ]


def selected_catalog(args: argparse.Namespace) -> pd.DataFrame:
    catalog = pd.read_csv(args.grid_catalog)
    required = {
        "supply_era5_grid_id",
        "era5_latitude",
        "era5_longitude",
        "has_solar_pv",
        "has_wind",
        "allocated_on_grid_capacity_mw",
    }
    missing = required - set(catalog)
    if missing:
        raise ValueError(f"Supply grid catalog lacks columns: {sorted(missing)}")
    if catalog["supply_era5_grid_id"].duplicated().any():
        raise ValueError("Supply grid catalog has duplicate grid identifiers")
    if args.grid_id:
        requested = set(args.grid_id)
        unknown = requested - set(catalog["supply_era5_grid_id"])
        if unknown:
            raise ValueError(f"Unknown --grid-id values: {sorted(unknown)}")
        catalog = catalog[catalog["supply_era5_grid_id"].isin(requested)].copy()
    cells = [
        supply_chunk_cell(float(latitude), float(longitude))
        for latitude, longitude in zip(
            catalog["era5_latitude"], catalog["era5_longitude"], strict=True
        )
    ]
    catalog["chunk_lat_block"] = [cell[0] for cell in cells]
    catalog["chunk_lon_block"] = [cell[1] for cell in cells]
    if args.order == "capacity_cell":
        # Chunk blocks are visited highest-capacity first, so an interrupted
        # run still leaves the largest fleets finished, while every point in a
        # block is read from one warm chunk cache.
        block_capacity = catalog.groupby(
            ["chunk_lat_block", "chunk_lon_block"], sort=False
        )["allocated_on_grid_capacity_mw"].transform("sum")
        catalog = catalog.assign(_block_capacity=block_capacity).sort_values(
            [
                "_block_capacity",
                "chunk_lat_block",
                "chunk_lon_block",
                "allocated_on_grid_capacity_mw",
                "supply_era5_grid_id",
            ],
            ascending=[False, True, True, False, True],
            kind="stable",
        )
    elif args.order == "capacity_priority":
        catalog = catalog.sort_values(
            ["allocated_on_grid_capacity_mw", "supply_era5_grid_id"],
            ascending=[False, True],
            kind="stable",
        )
    else:
        catalog = catalog.sort_values("supply_era5_grid_id", kind="stable")
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        catalog = catalog.head(args.limit)
    if catalog.empty:
        raise ValueError("No supply grids selected")
    return catalog.reset_index(drop=True)


def build_task_groups(catalog: pd.DataFrame, order: str) -> list[list[pd.Series]]:
    """Split the ordered catalog into units of work, preserving that order."""

    if order != "capacity_cell":
        return [[row] for _, row in catalog.iterrows()]
    groups: list[list[pd.Series]] = []
    current_key: tuple[int, int] | None = None
    for _, row in catalog.iterrows():
        key = (int(row["chunk_lat_block"]), int(row["chunk_lon_block"]))
        if key != current_key:
            groups.append([])
            current_key = key
        groups[-1].append(row)
    return groups


def probe(source_url: str, key: str, key_source: str) -> None:
    dataset = open_dataset(source_url, key)
    try:
        payload = {
            "source_url": source_url,
            "authorization_available": True,
            "authorization_source": key_source,
            "authorization_value_recorded": False,
            "coordinates": list(dataset.coords),
            "required_variables_available": sorted(
                (set(SOLAR_VARIABLES) | set(WIND_VARIABLES)) & set(dataset.data_vars)
            ),
            "required_variable_units": {
                name: dataset[name].attrs.get("units")
                for name in sorted(set(SOLAR_VARIABLES) | set(WIND_VARIABLES))
            },
            "time_start": str(dataset["time"].values[0]),
            "time_end": str(dataset["time"].values[-1]),
            "time_length": int(dataset.sizes.get("time", 0)),
            # Chunk geometry decides whether 21,466 point series can be read
            # without re-fetching the same chunks, so it drives the run order.
            "required_variable_chunks": {
                name: (
                    list(dataset[name].encoding.get("chunks"))
                    if dataset[name].encoding.get("chunks") is not None
                    else None
                )
                for name in sorted(set(SOLAR_VARIABLES) | set(WIND_VARIABLES))
                if name in dataset.data_vars
            },
            "required_variable_dims": {
                name: list(dataset[name].dims)
                for name in sorted(set(SOLAR_VARIABLES) | set(WIND_VARIABLES))
                if name in dataset.data_vars
            },
            "latitude_step": float(
                abs(dataset["latitude"].values[1] - dataset["latitude"].values[0])
            ),
            "longitude_step": float(
                abs(dataset["longitude"].values[1] - dataset["longitude"].values[0])
            ),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    finally:
        dataset.close()


def main() -> None:
    args = parse_args()
    if args.workers <= 0 or args.max_attempts <= 0:
        raise ValueError("--workers and --max-attempts must be positive")
    key, key_source = resolve_cds_key(args.key_env, args.cdsapirc)
    if args.probe_only:
        probe(args.source_url, key, key_source)
        return
    catalog = selected_catalog(args)
    catalog_sha256 = sha256_file(args.grid_catalog)
    args.output_root.mkdir(parents=True, exist_ok=True)
    groups = build_task_groups(catalog, args.order)
    total = len(catalog)
    started = time.perf_counter()
    print(
        f"Extracting {total:,} supply-grid ERA5 ARCO time series in "
        f"{len(groups):,} chunk-locality groups with {args.workers} workers "
        f"and {args.chunk_cache} cached chunks each; "
        "credentials will not be recorded."
    )
    results: list[ExtractResult] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                extract_cell,
                group,
                args.output_root,
                args.source_url,
                key,
                catalog_sha256,
                args.max_attempts,
                args.force,
                args.chunk_cache,
            )
            for group in groups
        ]
        for future in as_completed(futures):
            group_results = future.result()
            results.extend(group_results)
            completed += len(group_results)
            if (
                completed == len(group_results)
                or completed % 250 < len(group_results)
                or completed == total
                or any(item.status == "failed" for item in group_results)
            ):
                extracted = sum(item.status == "extracted" for item in results)
                skipped = sum(item.status == "skipped_complete" for item in results)
                failed = sum(item.status == "failed" for item in results)
                elapsed = time.perf_counter() - started
                rate = completed / elapsed if elapsed > 0 else 0.0
                remaining = (total - completed) / rate / 3600.0 if rate > 0 else 0.0
                with PRINT_LOCK:
                    print(
                        f"Progress {completed:,}/{total:,}: extracted={extracted:,}, "
                        f"skipped={skipped:,}, failed={failed:,}, "
                        f"{rate * 3600:.0f} grids/h, ~{remaining:.1f} h left"
                    )
    result_frame = pd.DataFrame([asdict(item) for item in results]).sort_values("grid_id")
    manifest_path = args.output_root / "resource_manifest.csv"
    result_frame.to_csv(manifest_path, index=False)
    failures = result_frame[result_frame["status"].eq("failed")]
    summary = {
        "method_version": METHOD_VERSION,
        "resource_model_version": RESOURCE_MODEL_VERSION,
        "completed_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "source": "Copernicus CDS ERA5 ARCO geo-chunked",
        "source_url": args.source_url,
        "authorization_available": True,
        "authorization_source": key_source,
        "authorization_value_recorded": False,
        "grid_catalog": str(args.grid_catalog),
        "grid_catalog_sha256": catalog_sha256,
        "selected_grid_count": total,
        "expected_endpoint_count_per_grid": EXPECTED_ENDPOINT_COUNT,
        "expected_interval_count_per_grid": EXPECTED_INTERVAL_COUNT,
        "workers": args.workers,
        "order": args.order,
        "chunk_locality_groups": len(groups),
        "chunk_cache_items": args.chunk_cache,
        "extracted": int(result_frame["status"].eq("extracted").sum()),
        "skipped_complete": int(result_frame["status"].eq("skipped_complete").sum()),
        "failed": int(len(failures)),
        "elapsed_seconds": time.perf_counter() - started,
        "output_bytes_written": int(result_frame["output_bytes"].sum()),
        "manifest": str(manifest_path),
        "is_historical_actual_generation": False,
    }
    summary_path = args.output_root / "resource_summary.json"
    atomic_json(summary, summary_path)
    if not failures.empty:
        examples = failures[["grid_id", "error"]].head(20).to_dict("records")
        raise RuntimeError(
            f"{len(failures)} supply grids failed; rerun to resume. Examples: {examples}"
        )
    print(
        f"Completed {total:,} grids in {summary['elapsed_seconds'] / 3600.0:.2f} h; "
        f"summary={summary_path}"
    )


if __name__ == "__main__":
    main()
