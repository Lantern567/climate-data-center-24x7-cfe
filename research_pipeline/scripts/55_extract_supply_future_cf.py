# -*- coding: utf-8 -*-
"""Future-climate capacity factors for the 21,466 supply cells.

## The problem this solves

The supply-side replay stored `pv_cf`/`wind_cf` and discarded the raw fields it
derived them from.  Morphing needs raw weather, so the supply leg of the
counterfactual set -- (D_H,S_F) and (D_F,S_F) -- was unreachable: there was
nothing left to morph.  This is the documented cost of that storage decision.

Re-extracting the raw fields and keeping them would need ~90 GB, which does not
fit.  So the raw weather is morphed **in flight** and only the resulting
capacity factors are written, returning the footprint to the ~10 GB of the
historical store:

    ARCO endpoints -> QDM morph -> capacity factors -> disk

## Why this reuses script 37 rather than reimplementing it

Script 37 already owns the parts that are hard to get right: chunk-locality
ordering that cut the download from 478 GB to 117 GB, the retrying store, the
resumable manifest, and the coordinate checks.  The only thing that must change
is *what the weather is* before the capacity-factor model sees it.

Script 37 resolves `interval_capacity_factors` as a module global at call time,
so replacing that one symbol with a morphing wrapper turns the entire validated
pipeline into a future-climate pipeline without touching its logic.  Same
technique as script 45, and for the same reason: a copied pipeline is a pipeline
free to drift.

## What is morphed, and what is not

t2m, ssrd, fdir, u100 and v100 carry deltas.  Surface pressure does not -- NEX
publishes no such field and its climate signal is negligible for air density
beside the wind signal.  Wind components are scaled together so direction is
preserved; fdir takes the ssrd ratio so the direct/diffuse split is not
silently re-partitioned.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "research_pipeline" / "scripts"
POINTS = Path(os.environ.get("NEX_POINTS_ROOT") or r"D:\nex_gddp_points")
DELTAS = Path(os.environ.get("NEX_DELTAS_ROOT") or r"D:\nex_gddp_deltas")
DEFAULT_OUTPUT = Path(os.environ.get("SUPPLY_FUTURE_CF_ROOT")
                      or r"D:\era5_supply_future_cf_ssp245")

MODEL = os.environ.get("QDM_MODEL", "ACCESS-CM2")
SCENARIO = os.environ.get("QDM_SCENARIO", "ssp245")
QUANTILES = np.round(np.arange(0.05, 0.96, 0.05), 2)
QUANTILE_COLUMNS = [f"q{int(round(q * 100)):02d}" for q in QUANTILES]

_SPEC = importlib.util.spec_from_file_location(
    "arco_supply_extractor", SCRIPTS / "37_extract_cds_arco_supply_resource.py")
SUPPLY = importlib.util.module_from_spec(_SPEC)
sys.modules["arco_supply_extractor"] = SUPPLY
_SPEC.loader.exec_module(SUPPLY)

_REAL_CF = SUPPLY.interval_capacity_factors


def _load_deltas() -> dict[str, dict[tuple[str, int], np.ndarray]]:
    out: dict[str, dict[tuple[str, int], np.ndarray]] = {}
    for variable in ("tas", "sfcWind", "rsds"):
        frame = pd.read_parquet(DELTAS / f"delta_{MODEL}_{SCENARIO}_{variable}.parquet")
        out[variable] = {
            (str(r.cell_id), int(r.month)):
                np.array([getattr(r, c) for c in QUANTILE_COLUMNS], dtype=float)
            for r in frame.itertuples(index=False)
        }
    return out


def _cell_lookup() -> dict[tuple[float, float], str]:
    cells = pd.read_csv(POINTS / "cells.csv")
    return {(round(r.era5_latitude, 2), round(r.era5_longitude, 2)): str(r.cell_id)
            for r in cells.itertuples(index=False)}


DELTA_TABLES = _load_deltas()
CELL_LOOKUP = _cell_lookup()


def _shift(values: np.ndarray, month: np.ndarray, cell_id: str,
           variable: str, multiplicative: bool) -> np.ndarray:
    """Move each hour by the delta at its own quantile in its cell-month."""
    table = DELTA_TABLES[variable]
    out = np.asarray(values, dtype=np.float64).copy()
    for m in range(1, 13):
        key = (cell_id, m)
        if key not in table:
            continue
        mask = month == m
        if not mask.any():
            continue
        sample = out[mask]
        reference = np.quantile(sample, QUANTILES)
        factor = np.interp(sample, reference, table[key])
        out[mask] = sample * factor if multiplicative else sample + factor
    return out


def morphing_capacity_factors(times: Any, latitude: float, longitude: float,
                              temperature_k: np.ndarray, **kwargs: Any):
    """Drop-in replacement for script 37's capacity-factor call."""
    index = pd.DatetimeIndex(times)
    month = index.month.to_numpy()
    cell_id = CELL_LOOKUP.get((round(latitude, 2), round(longitude, 2)))
    if cell_id is None:
        raise KeyError(f"No QDM cell for supply point {latitude},{longitude}")

    temperature = _shift(temperature_k, month, cell_id, "tas", multiplicative=False)

    u = kwargs.get("wind_u_100m_m_s")
    v = kwargs.get("wind_v_100m_m_s")
    if u is not None and v is not None:
        speed = np.hypot(u, v)
        speed_f = _shift(speed, month, cell_id, "sfcWind", multiplicative=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            scale = np.where(speed > 1e-6, speed_f / np.maximum(speed, 1e-6), 1.0)
        kwargs["wind_u_100m_m_s"] = u * scale
        kwargs["wind_v_100m_m_s"] = v * scale

    ssrd = kwargs.get("surface_solar_radiation_downwards_j_m2")
    fdir = kwargs.get("total_sky_direct_solar_radiation_at_surface_j_m2")
    if ssrd is not None:
        ssrd_f = _shift(ssrd, month, cell_id, "rsds", multiplicative=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(ssrd > 1e-6, ssrd_f / np.maximum(ssrd, 1e-6), 1.0)
        kwargs["surface_solar_radiation_downwards_j_m2"] = np.maximum(ssrd_f, 0.0)
        if fdir is not None:
            kwargs["total_sky_direct_solar_radiation_at_surface_j_m2"] = np.maximum(
                fdir * ratio, 0.0)

    resources, diagnostics = _REAL_CF(times, latitude, longitude, temperature, **kwargs)
    diagnostics = dict(diagnostics)
    diagnostics.update({
        "qdm_model": MODEL, "qdm_scenario": SCENARIO, "qdm_cell_id": cell_id,
        "morphed_in_flight": True, "surface_pressure_morphed": False,
    })
    return resources, diagnostics


SUPPLY.interval_capacity_factors = morphing_capacity_factors
SUPPLY.METHOD_VERSION = f"supply_future_cf_{SCENARIO}_{MODEL}_v1"


if __name__ == "__main__":
    if not any(a.startswith("--output-root") for a in sys.argv[1:]):
        sys.argv.extend(["--output-root", str(DEFAULT_OUTPUT)])
    SUPPLY.main()
