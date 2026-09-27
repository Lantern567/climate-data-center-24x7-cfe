# -*- coding: utf-8 -*-
"""Fetch NEX-GDDP-CMIP6 daily fields and extract the study's ERA5 cells.

## Why not the existing point downloader

`06_fetch_nex_gddp.py` asks NASA's NCSS to extract one point server-side.  That
is the right trade for a handful of markets and the wrong one here: the coupled
inventory needs 22,429 ERA5 cells (2,416 demand cells + 21,466 supply cells,
1,453 shared), and one request per cell per model per scenario per variable per
year runs to millions of requests.

A NEX-GDDP file already holds one global year of one variable at 0.25 deg, the
same grid the study uses, for about 233 MiB.  Downloading the file and slicing
every cell out of it locally costs one request per year-variable instead of
22,429, so the global route wins by four orders of magnitude at this scale.

## Disk discipline

The full design (5 models x [historical 20y + 3 scenarios x 40y] x 4 variables)
is ~0.62 TiB of source files, which does not fit.  Each file is therefore
downloaded, sliced and deleted before the next one starts, so peak usage is one
file plus the extracted points.  The extracted points are ~131 MiB per
model-scenario-year across all four variables.

## Variables

tas (K), hurs (%), sfcWind (m/s), rsds (W/m2) -- the four the QDM stage in
`06b_qdm_disaggregate_nex.py` consumes.  These are daily; the hourly structure
is restored downstream by borrowing an ERA5 analogue day, so this script never
tries to invent sub-daily detail.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "24x7_cfe" / "processed"
DEFAULT_OUTPUT = Path(r"D:\nex_gddp_points")
STAGING = Path(r"D:\nex_gddp_staging")

S3_PREFIX = "https://nex-gddp-cmip6.s3.us-west-2.amazonaws.com/NEX-GDDP-CMIP6"
VARIABLES = ("tas", "hurs", "sfcWind", "rsds")
# tasmax/tasmin are not part of the default pull.  They exist because the QDM
# delta built from daily-mean `tas` turns out to be almost flat across quantiles
# (global q95 - q05 = +0.055 K), so morphing on `tas` alone shifts hot hours and
# cold hours by the same amount and cannot produce any tail amplification.  The
# daily range carries the part of the warming signal that cooling load actually
# responds to, so it has to be fetched separately.
OPTIONAL_VARIABLES = ("tasmax", "tasmin")
ALL_VARIABLES = VARIABLES + OPTIONAL_VARIABLES
PERIODS = {"historical": (1995, 2014), "near": (2031, 2050), "mid": (2051, 2070)}
METHOD_VERSION = "nex_gddp_global_point_extract_v1"
# NEX-GDDP publishes a v1.1/v2.0 revision for some model-years; prefer the
# revised file when it exists, exactly as the archive intends.
VERSION_SUFFIXES = ("_v2.0", "_v1.1", "")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="ACCESS-CM2")
    p.add_argument("--member", default="r1i1p1f1")
    p.add_argument("--grid", default="auto")
    p.add_argument("--scenario", default="historical",
                   help="historical, ssp126, ssp245 or ssp585")
    p.add_argument("--period", default="historical", choices=sorted(PERIODS))
    p.add_argument("--variable", action="append", choices=ALL_VARIABLES)
    p.add_argument("--year", type=int, action="append")
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--staging", type=Path, default=STAGING)
    p.add_argument("--keep-source", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def study_cells() -> pd.DataFrame:
    demand = pd.read_csv(
        PROCESSED / "climate_coupling" / "era5_grid_catalog.csv",
        usecols=["era5_latitude", "era5_longitude"])
    supply = pd.read_parquet(
        PROCESSED / "wind_solar_supply_20y_fixed_2025"
        / "fixed_2025_supply_grid_allocation.parquet",
        columns=["era5_latitude", "era5_longitude"])
    cells = (pd.concat([demand, supply])
             .drop_duplicates()
             .sort_values(["era5_latitude", "era5_longitude"])
             .reset_index(drop=True))
    cells["cell_id"] = [f"c{i:06d}" for i in range(len(cells))]
    return cells


def source_url(model: str, scenario: str, member: str, grid: str,
               variable: str, year: int, suffix: str) -> str:
    return (f"{S3_PREFIX}/{model}/{scenario}/{member}/{variable}/"
            f"{variable}_day_{model}_{scenario}_{member}_{grid}_{year}{suffix}.nc")


# The grid label is a property of the model's native output, not a constant:
# GFDL-ESM4 publishes on `gr1` while ACCESS-CM2, MIROC6, MPI-ESM1-2-HR and
# MRI-ESM2-0 use `gn`.  Guessing wrong yields a 404 that looks exactly like a
# missing year, so it is read off the archive listing instead of hard-coded.
GRID_CANDIDATES = ("gn", "gr", "gr1")


def discover_grid_label(model: str, scenario: str, member: str,
                        variable: str, year: int) -> str | None:
    import xml.etree.ElementTree as ET
    prefix = f"NEX-GDDP-CMIP6/{model}/{scenario}/{member}/{variable}/"
    url = (f"https://nex-gddp-cmip6.s3.amazonaws.com/?list-type=2"
           f"&max-keys=1000&prefix={prefix}")
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            root = ET.fromstring(response.read())
    except Exception:
        return None
    namespace = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    for node in root.findall("s3:Contents", namespace):
        name = node.find("s3:Key", namespace).text.rsplit("/", 1)[-1]
        parts = name.removesuffix(".nc").split("_")
        if len(parts) >= 6 and parts[5] in GRID_CANDIDATES:
            return parts[5]
    return None


def download(url: str, target: Path, attempts: int = 4) -> int:
    """Fetch with an explicit completeness check.

    `shutil.copyfileobj` returns quietly when the connection drops mid-stream,
    leaving a short file that only fails later inside the HDF5 reader.  One such
    truncation (287 of 307 MiB) killed an entire overnight run, so the byte
    count is compared with Content-Length here and a short read is retried
    rather than being discovered downstream.
    """
    last = "unknown failure"
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "24x7-cfe-study"})
            with urllib.request.urlopen(request, timeout=900) as response:
                expected = response.headers.get("Content-Length")
                expected = int(expected) if expected else None
                with open(target, "wb") as handle:
                    shutil.copyfileobj(response, handle, length=1 << 22)
            size = target.stat().st_size
            if expected is not None and size != expected:
                raise OSError(f"truncated: got {size} of {expected} bytes")
            return size
        except Exception as error:
            last = f"{type(error).__name__}: {error}"
            target.unlink(missing_ok=True)
            if attempt < attempts:
                time.sleep(5 * attempt)
    raise OSError(f"download failed after {attempts} attempts: {last}")


def normalise_time(values: np.ndarray) -> pd.DatetimeIndex:
    """Coerce a NEX time axis to pandas dates whatever calendar it uses.

    Models do not share a calendar: ACCESS-CM2 publishes a standard one and
    decodes straight to datetime64, while GFDL-ESM4 uses `365_day` and decodes
    to cftime objects that `pd.to_datetime` refuses.  A 365-day year contains no
    29 February, so every date it produces is a legal Gregorian date and the ISO
    string round-trip is lossless.  Only the calendar *label* is dropped, and
    downstream needs nothing but the month.
    """
    try:
        return pd.to_datetime(values).normalize()
    except (TypeError, ValueError):
        return pd.to_datetime([f"{t.year:04d}-{t.month:02d}-{t.day:02d}"
                               for t in values])


def extract(path: Path, variable: str, cells: pd.DataFrame) -> pd.DataFrame:
    with xr.open_dataset(path, engine="h5netcdf") as dataset:
        array = dataset[variable]
        lon_name = "lon" if "lon" in array.coords else "longitude"
        lat_name = "lat" if "lat" in array.coords else "latitude"
        # NEX longitudes run 0..360; the study's ERA5 cells run -180..180.
        lon = cells.era5_longitude.to_numpy() % 360.0
        selected = array.sel(
            {lat_name: xr.DataArray(cells.era5_latitude.to_numpy(), dims="cell"),
             lon_name: xr.DataArray(lon, dims="cell")},
            method="nearest",
        ).load()
        values = np.asarray(selected.values, dtype=np.float32)
        times = normalise_time(selected["time"].values)
        units = str(array.attrs.get("units", ""))
    if values.shape[0] != len(times):
        values = values.T
    return pd.DataFrame({
        "date": np.repeat(times.to_numpy(), len(cells)),
        "cell_id": np.tile(cells.cell_id.to_numpy(), len(times)),
        variable: values.reshape(-1),
    }), units


def main() -> None:
    args = parse_args()
    variables = tuple(args.variable) if args.variable else VARIABLES
    start, stop = PERIODS[args.period]
    years = tuple(args.year) if args.year else tuple(range(start, stop + 1))
    cells = study_cells()
    if args.grid == "auto":
        discovered = discover_grid_label(args.model, args.scenario, args.member,
                                         variables[0], years[0])
        if discovered is None:
            raise RuntimeError(
                f"Cannot determine the grid label for {args.model}/{args.scenario}")
        args.grid = discovered
        print(f"grid label for {args.model}: {args.grid}", flush=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.staging.mkdir(parents=True, exist_ok=True)
    (args.output_root / "cells.csv").write_text(cells.to_csv(index=False), "utf-8")

    started = time.perf_counter()
    manifest: list[dict] = []
    for year in years:
        for variable in variables:
            destination = (args.output_root / args.model / args.scenario
                           / variable / f"{variable}_{year}.parquet")
            if destination.exists() and not args.force:
                manifest.append({"year": year, "variable": variable,
                                 "status": "skipped_complete"})
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = None
            for suffix in VERSION_SUFFIXES:
                url = source_url(args.model, args.scenario, args.member,
                                 args.grid, variable, year, suffix)
                staged = args.staging / Path(url).name
                try:
                    size = download(url, staged)
                    source = (url, staged, size)
                    break
                except Exception:
                    staged.unlink(missing_ok=True)
                    continue
            if source is None:
                manifest.append({"year": year, "variable": variable,
                                 "status": "unavailable"})
                print(f"  {year} {variable}: no file found", flush=True)
                continue
            url, staged, size = source
            try:
                frame, units = extract(staged, variable, cells)
                frame.to_parquet(destination, index=False, compression="zstd")
                manifest.append({
                    "year": year, "variable": variable, "status": "extracted",
                    "source_url": url, "source_bytes": size,
                    "units": units, "rows": int(len(frame)),
                })
                print(f"  {year} {variable}: {size/2**20:.0f} MiB -> "
                      f"{len(frame):,} rows ({units})", flush=True)
            except Exception as error:
                # One unreadable file must not abort a 160-file run; the year is
                # recorded as failed and picked up on the next resume.
                manifest.append({"year": year, "variable": variable,
                                 "status": "failed", "source_url": url,
                                 "error": f"{type(error).__name__}: {error}"})
                destination.unlink(missing_ok=True)
                print(f"  {year} {variable}: FAILED {type(error).__name__}",
                      flush=True)
            finally:
                if not args.keep_source:
                    staged.unlink(missing_ok=True)

    extracted = [m for m in manifest if m["status"] == "extracted"]
    payload = {
        "schema_version": "1.0",
        "method_version": METHOD_VERSION,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "model": args.model, "scenario": args.scenario, "period": args.period,
        "member": args.member, "grid_label": args.grid,
        "variables": list(variables),
        "years": [int(y) for y in years],
        "study_cells": int(len(cells)),
        "files_extracted": len(extracted),
        "files_unavailable": sum(1 for m in manifest if m["status"] == "unavailable"),
        "files_failed": sum(1 for m in manifest if m["status"] == "failed"),
        "files_skipped": sum(1 for m in manifest if m["status"] == "skipped_complete"),
        "source_bytes_streamed": sum(m.get("source_bytes", 0) for m in extracted),
        "source_files_retained": bool(args.keep_source),
        "elapsed_seconds": time.perf_counter() - started,
        "units_seen": sorted({m["units"] for m in extracted}),
    }
    name = f"manifest_{args.model}_{args.scenario}_{args.period}.json"
    (args.output_root / name).write_text(
        json.dumps({"summary": payload, "files": manifest}, indent=2,
                   ensure_ascii=False, allow_nan=False), "utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if payload["files_unavailable"] or payload["files_failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
