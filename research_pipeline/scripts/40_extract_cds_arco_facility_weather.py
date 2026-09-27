#!/usr/bin/env python3
"""Extract 1995--2014 facility-grid weather from CDS ERA5 ARCO.

This is the demand-side twin of ``37_extract_cds_arco_supply_resource.py``.
The supply extractor stores only the derived capacity factors, so changing the
photovoltaic or wind model there forces a second 117 GB download.  This
extractor therefore stores the **raw** hourly endpoints of 2 m temperature,
2 m dew point and surface pressure, and derives nothing.  Wet bulb, relative
humidity, PUE and facility load are all computed downstream, where the model
can be revised without touching the network again.

Endpoints, not intervals, are stored: the facility demand chain averages the
Stull wet bulb at the two endpoints of each interval ``[t, t+1)``, so both
endpoints must survive.  1995-01-01 00:00 to 2015-01-01 00:00 UTC inclusive is
175,321 endpoints and 175,320 intervals.

Authentication is read from the ``CDSAPI_KEY`` environment variable or the
``key:`` entry in ``~/.cdsapirc``.  Credentials never reach a URL, manifest,
metadata file or log line.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "research_pipeline" / "scripts"

# The supply extractor is a validated script, not an importable module.  Load
# it by path under a real module name so its generic pieces -- the retrying
# read-only Zarr store, credential resolution, chunk-block arithmetic and the
# atomic writers -- are reused verbatim rather than copied and left to diverge.
_SPEC = importlib.util.spec_from_file_location(
    "arco_supply_extractor", SCRIPTS / "37_extract_cds_arco_supply_resource.py"
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError("Cannot load 37_extract_cds_arco_supply_resource.py")
_SUPPLY = importlib.util.module_from_spec(_SPEC)
sys.modules["arco_supply_extractor"] = _SUPPLY
_SPEC.loader.exec_module(_SUPPLY)

RequestsReadOnlyStore = _SUPPLY.RequestsReadOnlyStore
atomic_json = _SUPPLY.atomic_json
atomic_npz = _SUPPLY.atomic_npz
chunk_cell = _SUPPLY.supply_chunk_cell
import_arco_dependencies = _SUPPLY.import_arco_dependencies
normalized_unit = _SUPPLY.normalized_unit
resolve_cds_key = _SUPPLY.resolve_cds_key
sha256_file = _SUPPLY.sha256_file

DEFAULT_CATALOG = (
    ROOT
    / "data"
    / "24x7_cfe"
    / "processed"
    / "climate_coupling"
    / "era5_grid_catalog.csv"
)
IN_TREE_OUTPUT_ROOT = (
    ROOT
    / "data"
    / "24x7_cfe"
    / "processed"
    / "climate_coupling"
    / "facility_weather_1995_2014"
)
# Roughly 4,800 files and several gigabytes land here.  Keeping them off a
# continuously synchronised cloud folder is a deployment choice, so the
# location stays overridable, exactly as on the supply side.
DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get("FACILITY_WEATHER_ROOT") or IN_TREE_OUTPUT_ROOT
)
SOURCE_URL = _SUPPLY.SOURCE_URL
METHOD_VERSION = "cds_era5_arco_geo_facility_weather_1995_2014_v1"
START_ENDPOINT = pd.Timestamp("1995-01-01 00:00:00", tz="UTC")
END_ENDPOINT = pd.Timestamp("2015-01-01 00:00:00", tz="UTC")
EXPECTED_ENDPOINTS = pd.date_range(START_ENDPOINT, END_ENDPOINT, freq="h")
EXPECTED_ENDPOINT_COUNT = 175_321
EXPECTED_INTERVAL_COUNT = 175_320

DEMAND_VARIABLES = (
    "2m_temperature",
    "2m_dewpoint_temperature",
    "surface_pressure",
)
VARIABLE_ALIASES = {
    "2m_temperature": ("2m_temperature", "t2m"),
    "2m_dewpoint_temperature": ("2m_dewpoint_temperature", "d2m"),
    "surface_pressure": ("surface_pressure", "sp"),
}
EXPECTED_UNITS = {
    "2m_temperature": {"k", "kelvin"},
    "2m_dewpoint_temperature": {"k", "kelvin"},
    "surface_pressure": {"pa", "pascal", "pascals"},
}
# Stored array name per canonical variable.  Short names keep the NPZ readable
# next to the supply-side `pv_cf`/`wind_cf` convention.
ARRAY_NAMES = {
    "2m_temperature": "t2m_k",
    "2m_dewpoint_temperature": "d2m_k",
    "surface_pressure": "sp_pa",
}

# ERA5 can report a dew point marginally above the dry bulb in saturated grid
# cells.  A small excess is a known reanalysis artefact and is recorded as a
# diagnostic; a large one means the variables are mismatched and must fail.
DEWPOINT_EXCESS_TOLERANCE_K = 0.5

THREAD_STATE = _SUPPLY.THREAD_STATE
PRINT_LOCK = Lock()


@dataclass(frozen=True)
class ExtractResult:
    grid_id: str
    status: str
    endpoint_count: int
    output_bytes: int
    elapsed_seconds: float
    error: str | None = None


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
        choices=("demand_cell", "demand_priority", "grid_id"),
        default="demand_cell",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--chunk-cache",
        type=int,
        default=12,
        help="Chunks held per worker; 9 covers one 4x4 block for all variables.",
    )
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    return parser.parse_args()


def canonicalize_variables(dataset: Any) -> Any:
    """Accept both CDS descriptive names and ERA5 short names."""

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


def open_dataset(source_url: str, key: str, cache_items: int) -> Any:
    xr = import_arco_dependencies()
    store = RequestsReadOnlyStore(source_url, key, cache_items=cache_items)
    try:
        dataset = xr.open_zarr(store, consolidated=True, chunks=None)
    except Exception:
        store.close()
        raise
    dataset.set_close(store.close)
    missing_coordinates = {"time", "latitude", "longitude"} - set(dataset.coords)
    if missing_coordinates:
        dataset.close()
        raise ValueError(
            f"CDS ARCO dataset lacks coordinates: {sorted(missing_coordinates)}"
        )
    dataset = canonicalize_variables(dataset)
    dataset.set_close(store.close)
    _SUPPLY.check_geo_chunk_layout(dataset)
    return dataset


def thread_dataset(source_url: str, key: str, cache_items: int) -> Any:
    identity = (source_url, "facility_weather")
    if getattr(THREAD_STATE, "facility_identity", None) != identity:
        previous = getattr(THREAD_STATE, "facility_dataset", None)
        if previous is not None:
            previous.close()
        THREAD_STATE.facility_dataset = open_dataset(source_url, key, cache_items)
        THREAD_STATE.facility_identity = identity
    return THREAD_STATE.facility_dataset


def drop_thread_dataset() -> None:
    dataset = getattr(THREAD_STATE, "facility_dataset", None)
    if dataset is not None:
        try:
            dataset.close()
        except Exception:  # a broken session must not mask the original error
            pass
    THREAD_STATE.facility_dataset = None
    THREAD_STATE.facility_identity = None


def output_paths(output_root: Path, grid_id: str) -> dict[str, Path]:
    return {
        "weather": output_root / f"{grid_id}.npz",
        "metadata": output_root / f"{grid_id}.metadata.json",
    }


def is_complete(paths: dict[str, Path], catalog_sha256: str) -> bool:
    if not paths["weather"].exists() or not paths["metadata"].exists():
        return False
    try:
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        if (
            metadata.get("method_version") != METHOD_VERSION
            or metadata.get("grid_catalog_sha256") != catalog_sha256
            or int(metadata.get("endpoint_count", -1)) != EXPECTED_ENDPOINT_COUNT
            or metadata.get("output_sha256") != sha256_file(paths["weather"])
        ):
            return False
        with np.load(paths["weather"]) as archive:
            if set(archive.files) != set(ARRAY_NAMES.values()):
                return False
            return all(
                archive[name].shape == (EXPECTED_ENDPOINT_COUNT,)
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


def load_point_arrays(
    dataset: Any, row: pd.Series
) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, Any]]:
    latitude = float(row["era5_latitude"])
    longitude = float(row["era5_longitude"])
    point = (
        dataset[list(DEMAND_VARIABLES)]
        .sel(latitude=latitude, longitude=longitude, method="nearest")
        .sel(
            time=slice(
                START_ENDPOINT.tz_localize(None), END_ENDPOINT.tz_localize(None)
            )
        )
        .load()
    )
    utc_time_index(point["time"].values)
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
        raise ValueError(
            "CDS ARCO longitude selection is farther than half a grid cell"
        )

    arrays = {
        name: np.asarray(point[name].values, dtype=float).reshape(-1)
        for name in DEMAND_VARIABLES
    }
    if any(len(values) != EXPECTED_ENDPOINT_COUNT for values in arrays.values()):
        raise ValueError("CDS ARCO variable endpoint length mismatch")
    if not all(np.isfinite(values).all() for values in arrays.values()):
        raise ValueError("CDS ARCO point contains non-finite values")

    temperature = arrays["2m_temperature"]
    dewpoint = arrays["2m_dewpoint_temperature"]
    pressure = arrays["surface_pressure"]
    if temperature.min() < 150.0 or temperature.max() > 350.0:
        raise ValueError("CDS ARCO 2 m temperature is outside plausible Kelvin bounds")
    if dewpoint.min() < 150.0 or dewpoint.max() > 350.0:
        raise ValueError("CDS ARCO 2 m dew point is outside plausible Kelvin bounds")
    if pressure.min() < 20_000.0 or pressure.max() > 120_000.0:
        raise ValueError("CDS ARCO surface pressure is outside plausible Pa bounds")
    excess = dewpoint - temperature
    max_excess = float(excess.max())
    if max_excess > DEWPOINT_EXCESS_TOLERANCE_K:
        raise ValueError(
            f"CDS ARCO dew point exceeds dry bulb by {max_excess:.3f} K, beyond "
            f"the {DEWPOINT_EXCESS_TOLERANCE_K} K saturation artefact tolerance"
        )

    diagnostics = {
        "min_t2m_k": float(temperature.min()),
        "max_t2m_k": float(temperature.max()),
        "min_d2m_k": float(dewpoint.min()),
        "max_d2m_k": float(dewpoint.max()),
        "min_sp_pa": float(pressure.min()),
        "max_sp_pa": float(pressure.max()),
        "dewpoint_above_dry_bulb_hours": int((excess > 0.0).sum()),
        "max_dewpoint_excess_k": max_excess,
    }
    return arrays, response_coordinates, diagnostics


def extract_one(
    row: pd.Series,
    output_root: Path,
    source_url: str,
    key: str,
    catalog_sha256: str,
    max_attempts: int,
    force: bool,
    cache_items: int,
) -> ExtractResult:
    grid_id = str(row["era5_grid_id"])
    paths = output_paths(output_root, grid_id)
    if not force and is_complete(paths, catalog_sha256):
        return ExtractResult(
            grid_id, "skipped_complete", EXPECTED_ENDPOINT_COUNT, 0, 0.0
        )
    started = time.perf_counter()
    last_error = "unknown failure"
    for attempt in range(1, max_attempts + 1):
        try:
            dataset = thread_dataset(source_url, key, cache_items)
            weather, response_coordinates, diagnostics = load_point_arrays(
                dataset, row
            )
            stored = {
                ARRAY_NAMES[name]: values.astype(np.float32)
                for name, values in weather.items()
            }
            atomic_npz(stored, paths["weather"])
            metadata = {
                "method_version": METHOD_VERSION,
                "retrieved_at_utc": datetime.now(UTC)
                .replace(microsecond=0)
                .isoformat(),
                "era5_grid_id": grid_id,
                "requested_latitude": float(row["era5_latitude"]),
                "requested_longitude": float(row["era5_longitude"]),
                "response_latitude": response_coordinates["latitude"],
                "response_longitude": response_coordinates["longitude"],
                "source": "Copernicus CDS ERA5 ARCO geo-chunked",
                "source_url": source_url,
                "source_authorization_recorded": False,
                "variables": list(DEMAND_VARIABLES),
                "endpoint_start_utc": START_ENDPOINT.isoformat(),
                "endpoint_end_utc": END_ENDPOINT.isoformat(),
                "endpoint_count": EXPECTED_ENDPOINT_COUNT,
                "interval_count": EXPECTED_INTERVAL_COUNT,
                "endpoint_definition": (
                    "instantaneous ERA5 analysis values at each hourly endpoint; "
                    "interval [t,t+1) statistics are derived downstream"
                ),
                "derived_quantities_stored": False,
                "grid_catalog_sha256": catalog_sha256,
                "output_arrays": {
                    name: {"dtype": str(values.dtype), "rows": len(values)}
                    for name, values in stored.items()
                },
                "diagnostics": diagnostics,
                "output_sha256": sha256_file(paths["weather"]),
                "is_historical_actual_demand": False,
                "data_center_capacity_used": False,
            }
            atomic_json(metadata, paths["metadata"])
            return ExtractResult(
                grid_id,
                "extracted",
                EXPECTED_ENDPOINT_COUNT,
                paths["weather"].stat().st_size,
                time.perf_counter() - started,
            )
        except Exception as exc:  # keep completed grids resumable
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < max_attempts:
                drop_thread_dataset()
                time.sleep(min(60.0, 2.0**attempt + random.random()))
    return ExtractResult(
        grid_id, "failed", 0, 0, time.perf_counter() - started, last_error
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
    """Extract every facility grid sharing one Zarr chunk block on one thread."""

    return [
        extract_one(
            row,
            output_root,
            source_url,
            key,
            catalog_sha256,
            max_attempts,
            force,
            cache_items,
        )
        for row in rows
    ]


def selected_catalog(args: argparse.Namespace) -> pd.DataFrame:
    catalog = pd.read_csv(args.grid_catalog)
    required = {
        "era5_grid_id",
        "era5_latitude",
        "era5_longitude",
        "site_mean_it_mw_p50_sum",
    }
    missing = required - set(catalog)
    if missing:
        raise ValueError(f"Facility grid catalog lacks columns: {sorted(missing)}")
    if catalog["era5_grid_id"].duplicated().any():
        raise ValueError("Facility grid catalog has duplicate grid identifiers")
    if args.grid_id:
        requested = set(args.grid_id)
        unknown = requested - set(catalog["era5_grid_id"])
        if unknown:
            raise ValueError(f"Unknown --grid-id values: {sorted(unknown)}")
        catalog = catalog[catalog["era5_grid_id"].isin(requested)].copy()
    cells = [
        chunk_cell(float(latitude), float(longitude))
        for latitude, longitude in zip(
            catalog["era5_latitude"], catalog["era5_longitude"], strict=True
        )
    ]
    catalog["chunk_lat_block"] = [cell[0] for cell in cells]
    catalog["chunk_lon_block"] = [cell[1] for cell in cells]
    if args.order == "demand_cell":
        # Blocks are visited largest-demand first, so an interrupted run leaves
        # the biggest data centre clusters finished, while every point in a
        # block is read from one warm chunk cache.
        block_demand = catalog.groupby(
            ["chunk_lat_block", "chunk_lon_block"], sort=False
        )["site_mean_it_mw_p50_sum"].transform("sum")
        catalog = catalog.assign(_block_demand=block_demand).sort_values(
            [
                "_block_demand",
                "chunk_lat_block",
                "chunk_lon_block",
                "site_mean_it_mw_p50_sum",
                "era5_grid_id",
            ],
            ascending=[False, True, True, False, True],
            kind="stable",
        )
    elif args.order == "demand_priority":
        catalog = catalog.sort_values(
            ["site_mean_it_mw_p50_sum", "era5_grid_id"],
            ascending=[False, True],
            kind="stable",
        )
    else:
        catalog = catalog.sort_values("era5_grid_id", kind="stable")
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        catalog = catalog.head(args.limit)
    if catalog.empty:
        raise ValueError("No facility grids selected")
    return catalog.reset_index(drop=True)


def build_task_groups(catalog: pd.DataFrame, order: str) -> list[list[pd.Series]]:
    """Split the ordered catalog into units of work, preserving that order."""

    if order != "demand_cell":
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
    dataset = open_dataset(source_url, key, cache_items=4)
    try:
        payload = {
            "source_url": source_url,
            "authorization_available": True,
            "authorization_source": key_source,
            "authorization_value_recorded": False,
            "required_variables_available": sorted(
                set(DEMAND_VARIABLES) & set(dataset.data_vars)
            ),
            "required_variable_units": {
                name: dataset[name].attrs.get("units") for name in DEMAND_VARIABLES
            },
            "required_variable_chunks": {
                name: (
                    list(dataset[name].encoding.get("chunks"))
                    if dataset[name].encoding.get("chunks") is not None
                    else None
                )
                for name in DEMAND_VARIABLES
            },
            "time_start": str(dataset["time"].values[0]),
            "time_end": str(dataset["time"].values[-1]),
            "time_length": int(dataset.sizes.get("time", 0)),
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
        f"Extracting {total:,} facility-grid ERA5 ARCO time series in "
        f"{len(groups):,} chunk-locality groups with {args.workers} workers "
        f"and {args.chunk_cache} cached chunks each; "
        "raw endpoints only, credentials will not be recorded."
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
                or completed % 100 < len(group_results)
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
                        f"{rate * 3600:.0f} grids/h, ~{remaining:.2f} h left",
                        flush=True,
                    )
    result_frame = pd.DataFrame([asdict(item) for item in results]).sort_values(
        "grid_id"
    )
    manifest_path = args.output_root / "facility_weather_manifest.csv"
    result_frame.to_csv(manifest_path, index=False)
    failures = result_frame[result_frame["status"].eq("failed")]
    summary = {
        "method_version": METHOD_VERSION,
        "completed_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "source": "Copernicus CDS ERA5 ARCO geo-chunked",
        "source_url": args.source_url,
        "authorization_available": True,
        "authorization_source": key_source,
        "authorization_value_recorded": False,
        "grid_catalog": str(args.grid_catalog),
        "grid_catalog_sha256": catalog_sha256,
        "selected_grid_count": total,
        "variables": list(DEMAND_VARIABLES),
        "derived_quantities_stored": False,
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
        "is_historical_actual_demand": False,
    }
    summary_path = args.output_root / "facility_weather_summary.json"
    atomic_json(summary, summary_path)
    if not failures.empty:
        examples = failures[["grid_id", "error"]].head(20).to_dict("records")
        raise RuntimeError(
            f"{len(failures)} facility grids failed; rerun to resume. "
            f"Examples: {examples}"
        )
    print(
        f"Completed {total:,} grids in {summary['elapsed_seconds'] / 3600.0:.2f} h; "
        f"summary={summary_path}"
    )


if __name__ == "__main__":
    main()
