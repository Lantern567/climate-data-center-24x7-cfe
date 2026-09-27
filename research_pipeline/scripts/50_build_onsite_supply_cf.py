# -*- coding: utf-8 -*-
"""On-site capacity factors at the data-centre cells, for the S1 scenario.

S1 asks what a site could cover using generation on its own land: rooftop and
campus wind or solar, at the facility's own coordinates.  That needs the wind
and solar resource *where the data centre is*, which is what script 45
extracted (ssrd/fdir/u100/v100 on the 2,416 facility cells).  Temperature and
pressure come from script 40's earlier extraction on the same cells and the same
175,321 endpoints, so nothing is re-downloaded.

Conversion goes through `supply_resource_models.interval_capacity_factors` --
the very function the supply-side replay used.  Reimplementing the tilt model,
the density correction or the endpoint-to-interval trapezoid here would let S1
drift away from S2--S5, and the whole point of the ladder is that only the
geographic boundary differs between scenarios.

A cell's on-site capacity factor is a *resource* property: it says what a
turbine or panel placed there would yield, regardless of whether any fleet
exists there today.  That is the correct semantics for S1 -- the site would be
building this generation itself.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
COUPLING = ROOT / "data" / "24x7_cfe" / "processed" / "climate_coupling"
RESOURCE_ROOT = Path(r"D:\era5_facility_resource_1995_2014")
WEATHER_ROOT = Path(r"D:\era5_facility_weather_1995_2014")
DEFAULT_OUTPUT = Path(r"D:\facility_onsite_supply_cf")

METHOD_VERSION = "onsite_supply_cf_v1"
ENDPOINTS = 175_321
INTERVALS = 175_320


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--resource-root", type=Path, default=RESOURCE_ROOT)
    p.add_argument("--weather-root", type=Path, default=WEATHER_ROOT)
    p.add_argument("--method-version", default=METHOD_VERSION)
    p.add_argument("--limit", type=int)
    return p.parse_args()


def load_models():
    spec = importlib.util.spec_from_file_location(
        "supply_resource_models", ROOT / "research_pipeline" / "supply_resource_models.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    models = load_models()

    catalog = pd.read_csv(COUPLING / "era5_grid_catalog.csv")
    if args.limit:
        catalog = catalog.head(args.limit)
    endpoints = pd.date_range("1995-01-01", "2015-01-01", freq="h", tz="UTC")
    if len(endpoints) != ENDPOINTS:
        raise ValueError("Endpoint window is not 175,321 hours")

    pv = np.zeros((len(catalog), INTERVALS), dtype=np.float32)
    wind = np.zeros((len(catalog), INTERVALS), dtype=np.float32)
    clipped_total = 0
    for position, row in enumerate(catalog.itertuples(index=False)):
        grid_id = str(row.era5_grid_id)
        with np.load(args.resource_root / f"{grid_id}.npz") as resource, \
             np.load(args.weather_root / f"{grid_id}.npz") as weather:
            outputs, diagnostics = models.interval_capacity_factors(
                endpoints,
                float(row.era5_latitude),
                float(row.era5_longitude),
                weather["t2m_k"],
                surface_pressure_pa=weather["sp_pa"],
                wind_u_100m_m_s=resource["u100_ms"],
                wind_v_100m_m_s=resource["v100_ms"],
                surface_solar_radiation_downwards_j_m2=resource["ssrd_j_m2"],
                total_sky_direct_solar_radiation_at_surface_j_m2=resource["fdir_j_m2"],
            )
        pv[position] = outputs["pv_cf"].astype(np.float32)
        wind[position] = outputs["wind_cf"].astype(np.float32)
        clipped_total += int(diagnostics.get("diffuse_clipped_hours", 0) or 0)
        if (position + 1) % 250 == 0:
            print(f"  {position + 1}/{len(catalog)} cells", flush=True)

    target = args.output_root / "onsite_supply_cf.npz"
    # The staging name must still end in .npz: np.savez_compressed silently
    # appends the extension when it is missing, so a ".npz.partial" name is
    # written as ".npz.partial.npz" and the rename then fails.
    staging = args.output_root / "onsite_supply_cf.partial.npz"
    np.savez_compressed(
        staging,
        era5_grid_id=catalog.era5_grid_id.to_numpy().astype("U32"),
        pv_cf=pv, wind_cf=wind,
        latitude=catalog.era5_latitude.to_numpy(),
        longitude=catalog.era5_longitude.to_numpy(),
    )
    staging.replace(target)

    annual_pv = pv.mean(axis=1)
    annual_wind = wind.mean(axis=1)
    payload = {
        "schema_version": "1.0",
        "method_version": args.method_version,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "cells": int(len(catalog)),
        "intervals": INTERVALS,
        "model_version": str(models.MODEL_VERSION),
        "wind_scenario": str(models.CENTRAL_WIND_SCENARIO),
        "diffuse_clipped_hours_total": clipped_total,
        "pv_cf_mean": float(annual_pv.mean()),
        "pv_cf_range": [float(annual_pv.min()), float(annual_pv.max())],
        "wind_cf_mean": float(annual_wind.mean()),
        "wind_cf_range": [float(annual_wind.min()), float(annual_wind.max())],
        "hard_checks": {
            "pv_within_unit_interval": bool(pv.min() >= 0 and pv.max() <= 1.0 + 1e-6),
            "wind_within_unit_interval": bool(wind.min() >= 0 and wind.max() <= 1.0 + 1e-6),
            "all_finite": bool(np.isfinite(pv).all() and np.isfinite(wind).all()),
            "interval_count_exact": bool(pv.shape[1] == INTERVALS),
            "every_cell_present": bool(len(catalog) == len(pv)),
            # A cell with zero solar resource over twenty years would mean the
            # radiation arrays never made it through.
            "no_dead_solar_cells": bool((annual_pv > 0.01).all()),
        },
    }
    payload["all_hard_checks_pass"] = all(payload["hard_checks"].values())
    (args.output_root / "onsite_supply_cf_validation.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), "utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
