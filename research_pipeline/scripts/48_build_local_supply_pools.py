# -*- coding: utf-8 -*-
"""Capacity-weighted local supply pools for the S2 (local direct wire) scenario.

S2 lets a site buy from nearby projects over a private wire rather than from the
whole grid region.  Its supply shape is therefore the capacity-weighted mean
capacity factor of the real wind and solar fleet within a radius of the site:

    cf_pool,t = sum_j K_j f_j cf_j,t / sum_j K_j f_j        for  d(i,j) <= R

where K_j is the fleet capacity the fixed-2025 inventory placed in supply cell j
and f_j is that cell's PV bias-correction factor.  Wind carries no correction
(its capacity-adjusted validation ratio is 0.996).

This needs **no new download**: the twenty-year per-cell capacity factors for
all 21,466 supply cells are already on disk.  Only S1 -- generation at the
site's own cell, where there may be no fleet at all -- requires the separate
facility-resource extraction.

At R = 100 km, 2,257 of 2,416 facility cells (93.4%) have at least one supply
cell in range, with a median of 18.  Sites with an empty radius are written with
zero capacity and are excluded from S2 rather than silently falling back to a
wider pool, which would quietly turn S2 into S3.
"""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "24x7_cfe" / "processed"
COUPLING = PROCESSED / "climate_coupling"
FIXED = PROCESSED / "wind_solar_supply_20y_fixed_2025"
CORRECTED = FIXED / "replay_results_pv_corrected"
SUPPLY_CF_ROOT = Path(r"D:\era5_resource_cf_1995_2014")
DEFAULT_OUTPUT = Path(r"D:\facility_local_supply_pools")

METHOD_VERSION = "local_supply_pool_v1"
ENDPOINTS = 175_320
EARTH_RADIUS_KM = 6371.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--radius-km", type=float, default=100.0)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--supply-cf-root", type=Path, default=SUPPLY_CF_ROOT)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def haversine_km(lat1: np.ndarray, lon1: np.ndarray,
                 lat2: float, lon2: float) -> np.ndarray:
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dlat, dlon = p1 - p2, np.radians(lon1) - np.radians(lon2)
    a = np.sin(dlat / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2) ** 2
    return EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(a))


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    facilities = pd.read_csv(COUPLING / "era5_grid_catalog.csv")
    allocation = pd.read_parquet(
        FIXED / "fixed_2025_supply_grid_allocation.parquet",
        columns=["supply_era5_grid_id", "era5_latitude", "era5_longitude",
                 "technology", "allocated_on_grid_capacity_mw"],
    )
    fleet = (
        allocation.groupby(
            ["supply_era5_grid_id", "era5_latitude", "era5_longitude", "technology"]
        ).allocated_on_grid_capacity_mw.sum().unstack(fill_value=0.0).reset_index()
    )
    for column in ("solar_pv", "wind"):
        if column not in fleet.columns:
            fleet[column] = 0.0

    # PV correction is per supply cell; wind is untouched.
    grid_cf = pd.read_parquet(
        CORRECTED / "annual_grid_capacity_factor_1995_2014.parquet",
        columns=["supply_era5_grid_id", "pv_correction_factor"],
    ).drop_duplicates("supply_era5_grid_id").set_index("supply_era5_grid_id")
    fleet["pv_factor"] = (
        fleet.supply_era5_grid_id.map(grid_cf.pv_correction_factor).fillna(1.0)
    )
    fleet["pv_effective_mw"] = fleet.solar_pv * fleet.pv_factor

    # Which facility cells each supply cell serves.
    members: dict[str, list[int]] = {}
    neighbour_count = np.zeros(len(facilities), dtype=int)
    for position in range(len(facilities)):
        distance = haversine_km(
            fleet.era5_latitude.to_numpy(), fleet.era5_longitude.to_numpy(),
            float(facilities.era5_latitude.iloc[position]),
            float(facilities.era5_longitude.iloc[position]),
        )
        for supply_index in np.flatnonzero(distance <= args.radius_km):
            members.setdefault(
                str(fleet.supply_era5_grid_id.iloc[supply_index]), []
            ).append(position)
            neighbour_count[position] += 1

    pv_sum = np.zeros((len(facilities), ENDPOINTS), dtype=np.float32)
    wd_sum = np.zeros((len(facilities), ENDPOINTS), dtype=np.float32)
    pv_weight = np.zeros(len(facilities))
    wd_weight = np.zeros(len(facilities))
    fleet_by_id = fleet.set_index("supply_era5_grid_id")

    missing = 0
    for number, (grid_id, positions) in enumerate(members.items(), start=1):
        path = args.supply_cf_root / f"{grid_id}.npz"
        if not path.exists():
            missing += 1
            continue
        row = fleet_by_id.loc[grid_id]
        with np.load(path) as archive:
            pv = archive["pv_cf"].astype(np.float32) if "pv_cf" in archive else None
            wd = archive["wind_cf"].astype(np.float32) if "wind_cf" in archive else None
        pv_mw, wd_mw = float(row.pv_effective_mw), float(row.wind)
        index = np.asarray(positions)
        if pv is not None and pv_mw > 0:
            pv_sum[index] += pv[None, :] * pv_mw
            pv_weight[index] += pv_mw
        if wd is not None and wd_mw > 0:
            wd_sum[index] += wd[None, :] * wd_mw
            wd_weight[index] += wd_mw
        if number % 2000 == 0:
            print(f"  {number}/{len(members)} supply cells", flush=True)

    with np.errstate(invalid="ignore", divide="ignore"):
        pv_cf = np.where(pv_weight[:, None] > 0, pv_sum / np.maximum(pv_weight, 1e-9)[:, None], 0.0)
        wd_cf = np.where(wd_weight[:, None] > 0, wd_sum / np.maximum(wd_weight, 1e-9)[:, None], 0.0)

    # Write through a temporary name and rename: this file is 1.5 GB and takes
    # minutes to compress, and a reader that opens it mid-write gets a
    # BadZipFile rather than a clean "not ready yet".
    target = args.output_root / "local_supply_pool_cf.npz"
    # Must still end in .npz: np.savez_compressed appends the extension when it
    # is missing, so ".npz.partial" lands as ".npz.partial.npz" and the rename
    # below then fails on a file that does not exist.
    staging = args.output_root / "local_supply_pool_cf.partial.npz"
    np.savez_compressed(
        staging,
        era5_grid_id=facilities.era5_grid_id.to_numpy().astype("U32"),
        pv_cf=pv_cf.astype(np.float32),
        wind_cf=wd_cf.astype(np.float32),
        pv_capacity_mw=pv_weight,
        wind_capacity_mw=wd_weight,
        neighbour_count=neighbour_count,
    )
    staging.replace(target)
    served = int(((pv_weight > 0) | (wd_weight > 0)).sum())
    payload = {
        "schema_version": "1.0",
        "method_version": METHOD_VERSION,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "radius_km": args.radius_km,
        "facility_cells": int(len(facilities)),
        "facility_cells_with_local_fleet": served,
        "facility_cells_without_local_fleet": int(len(facilities) - served),
        "supply_cells_used": int(len(members)),
        "supply_cf_files_missing": missing,
        "median_neighbours": float(np.median(neighbour_count)),
        "pv_correction_applied": True,
        "wind_correction_applied": False,
        "hard_checks": {
            "cf_within_unit_interval": bool(
                pv_cf.min() >= 0 and pv_cf.max() <= 1.0 + 1e-6
                and wd_cf.min() >= 0 and wd_cf.max() <= 1.0 + 1e-6),
            "no_missing_supply_files": missing == 0,
            "endpoints_exact": bool(pv_cf.shape[1] == ENDPOINTS),
        },
    }
    payload["all_hard_checks_pass"] = all(payload["hard_checks"].values())
    (args.output_root / "local_supply_pool_validation.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), "utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
