# -*- coding: utf-8 -*-
"""Twenty-year ERA5 resource variables AT THE DATA CENTRE GRIDS (1995--2014).

`40_extract_cds_arco_facility_weather.py` pulled only `t2m`/`d2m`/`sp`, which is
exactly what PUE needs and nothing more.  The S1 (on-site) and S2 (local direct
wire) procurement scenarios additionally need the wind and solar resource **at
the facility's own location**, so this script adds `ssrd`/`fdir`/`u100`/`v100`
on the same 2,416 grids and the same 175,321 endpoints.

Why extract all 2,416 rather than only the 963 gaps: 1,453 of the data-centre
grids happen to coincide with a supply grid and already have twenty-year
capacity factors on disk.  Reusing those would mix two extraction batches and
two model versions in one product.  The marginal download is small, so the whole
set is taken in one pass and one version.

Raw endpoints are stored, never derived quantities -- the supply side stored
only `pv_cf`/`wind_cf` and consequently could not be re-modelled without a
117 GB re-download.  `t2m` and `sp` are NOT re-fetched; they already exist from
script 40 and are joined downstream.

Implementation: script 40 is loaded by path and its infrastructure (retrying
store, threaded dataset cache, chunk-locality ordering, atomic writers,
manifest) is reused as-is.  Only the variable set and the physical-bounds check
are resource-specific, so exactly those two are overridden.  Copying the file
would create a second definition free to drift.
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

_SPEC = importlib.util.spec_from_file_location(
    "facility_weather_extractor", SCRIPTS / "40_extract_cds_arco_facility_weather.py"
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError("Cannot load 40_extract_cds_arco_facility_weather.py")
W40 = importlib.util.module_from_spec(_SPEC)
sys.modules["facility_weather_extractor"] = W40
_SPEC.loader.exec_module(W40)

RESOURCE_VARIABLES = (
    "surface_solar_radiation_downwards",
    "total_sky_direct_solar_radiation_at_surface",
    "100m_u_component_of_wind",
    "100m_v_component_of_wind",
)
VARIABLE_ALIASES = {
    "surface_solar_radiation_downwards": (
        "surface_solar_radiation_downwards", "ssrd"),
    "total_sky_direct_solar_radiation_at_surface": (
        "total_sky_direct_solar_radiation_at_surface", "fdir"),
    "100m_u_component_of_wind": ("100m_u_component_of_wind", "u100"),
    "100m_v_component_of_wind": ("100m_v_component_of_wind", "v100"),
    # Renamed only to satisfy script 37's chunk-layout probe, which reads the
    # encoding of `2m_temperature`.  It is deliberately absent from
    # RESOURCE_VARIABLES, so no temperature data is transferred -- script 40
    # already holds it for these same grids and endpoints.
    "2m_temperature": ("2m_temperature", "t2m"),
}
EXPECTED_UNITS = {
    "surface_solar_radiation_downwards": {"jm-2", "j/m2", "jm−2"},
    "total_sky_direct_solar_radiation_at_surface": {"jm-2", "j/m2", "jm−2"},
    "100m_u_component_of_wind": {"ms-1", "m/s", "ms−1"},
    "100m_v_component_of_wind": {"ms-1", "m/s", "ms−1"},
    "2m_temperature": {"k", "kelvin"},
}
ARRAY_NAMES = {
    "surface_solar_radiation_downwards": "ssrd_j_m2",
    "total_sky_direct_solar_radiation_at_surface": "fdir_j_m2",
    "100m_u_component_of_wind": "u100_ms",
    "100m_v_component_of_wind": "v100_ms",
}
METHOD_VERSION = "cds_era5_arco_geo_facility_resource_1995_2014_v1"
DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get("FACILITY_RESOURCE_ROOT")
    or r"D:\era5_facility_resource_1995_2014"
)

# One hour of extraterrestrial normal irradiance is about 4.9e6 J/m2; the
# surface can never exceed it, so 6e6 is a generous ceiling that still catches a
# unit or accumulation-window error.
RADIATION_CEILING_J_M2 = 6.0e6
# ERA5 accumulations are physically non-negative but arrive slightly negative
# around zero from GRIB packing.  The first run rejected 1,426 of 2,416 cells on
# minima of about -1.2 J/m2 -- noise at 3e-7 of a typical hourly total, and each
# rejection burned four full re-downloads before giving up.  The supply-side
# extractor (script 37) does not screen this at all; it clips the diffuse
# residual downstream instead, which is the behaviour these two products must
# share.  -100 J/m2 still catches a sign or unit error by orders of magnitude.
RADIATION_FLOOR_J_M2 = -100.0
# Script 37 uses the same +/-150 m/s component limit for the supply grids.
WIND_COMPONENT_LIMIT_MS = 150.0
# fdir is the direct component on a horizontal plane, so it is bounded by the
# global horizontal total -- but the two accumulations are packed independently
# and produced by different steps of the radiation scheme, so small exceedances
# are a known artefact (observed up to 4.7e4 J/m2, about 1.2%).  A RELATIVE
# bound scales with the cell's own irradiance and still fails hard if the two
# variables were swapped, which would put the excess near 100%.
DIRECT_EXCESS_RELATIVE_TOLERANCE = 0.05
# Above the polar circle the sun never climbs far, and the direct/global split
# is numerically fragile at grazing incidence: ssrd itself is small, so a fixed
# absolute error in the partition becomes a large relative one.  The single cell
# that needed this (era5_g2401, 64.0N 145.75W, interior Alaska, one site of
# 0.94 MW) peaked at 5.8%.  The tolerance follows the mechanism -- latitude --
# rather than being loosened everywhere to admit one high-latitude cell.
# Downstream this is harmless anyway: the diffuse residual is clipped at zero.
HIGH_LATITUDE_DEG = 60.0
HIGH_LATITUDE_EXCESS_TOLERANCE = 0.10


def load_point_arrays(
    dataset: Any, row: pd.Series
) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, Any]]:
    """Resource-variable twin of script 40's demand-variable loader.

    Geometry, endpoint-count and coordinate checks are delegated to script 40 so
    the two products cannot disagree about which cell they sampled; only the
    physical bounds differ.
    """
    latitude = float(row["era5_latitude"])
    longitude = float(row["era5_longitude"])
    point = (
        dataset[list(RESOURCE_VARIABLES)]
        .sel(latitude=latitude, longitude=longitude, method="nearest")
        .sel(time=slice(W40.START_ENDPOINT.tz_localize(None),
                        W40.END_ENDPOINT.tz_localize(None)))
        .load()
    )
    W40.utc_time_index(point["time"].values)
    response = {
        "latitude": float(point["latitude"].values),
        "longitude": float(point["longitude"].values),
    }
    if abs(response["latitude"] - latitude) > 0.126:
        raise ValueError("CDS ARCO latitude selection is farther than half a grid cell")
    longitude_error = abs(
        ((response["longitude"] - longitude + 180.0) % 360.0) - 180.0
    )
    if longitude_error > 0.126:
        raise ValueError("CDS ARCO longitude selection is farther than half a grid cell")

    arrays = {
        name: np.asarray(point[name].values, dtype=float).reshape(-1)
        for name in RESOURCE_VARIABLES
    }
    if any(len(v) != W40.EXPECTED_ENDPOINT_COUNT for v in arrays.values()):
        raise ValueError("CDS ARCO variable endpoint length mismatch")
    if not all(np.isfinite(v).all() for v in arrays.values()):
        raise ValueError("CDS ARCO point contains non-finite values")

    ssrd = arrays["surface_solar_radiation_downwards"]
    fdir = arrays["total_sky_direct_solar_radiation_at_surface"]
    u100 = arrays["100m_u_component_of_wind"]
    v100 = arrays["100m_v_component_of_wind"]
    for name, values in (("ssrd", ssrd), ("fdir", fdir)):
        if values.min() < RADIATION_FLOOR_J_M2 or values.max() > RADIATION_CEILING_J_M2:
            raise ValueError(
                f"CDS ARCO {name} outside plausible accumulation bounds "
                f"[{values.min():.1f}, {values.max():.1f}] J/m2"
            )
    direct_excess = float((fdir - ssrd).max())
    tolerance = (HIGH_LATITUDE_EXCESS_TOLERANCE
                 if abs(latitude) >= HIGH_LATITUDE_DEG
                 else DIRECT_EXCESS_RELATIVE_TOLERANCE)
    excess_limit = tolerance * max(float(ssrd.max()), 1.0)
    if direct_excess > excess_limit:
        raise ValueError(
            f"CDS ARCO direct radiation exceeds global horizontal by "
            f"{direct_excess:.1f} J/m2 ({direct_excess / max(ssrd.max(), 1.0):.1%} "
            f"of peak) at latitude {latitude:.2f}, beyond the {tolerance:.0%} tolerance"
        )
    speed = np.hypot(u100, v100)
    if max(np.abs(u100).max(), np.abs(v100).max()) > WIND_COMPONENT_LIMIT_MS:
        raise ValueError("CDS ARCO 100 m wind component outside plausible bounds")

    diagnostics = {
        "max_ssrd_j_m2": float(ssrd.max()),
        "max_fdir_j_m2": float(fdir.max()),
        "min_ssrd_j_m2": float(ssrd.min()),
        "max_direct_excess_j_m2": direct_excess,
        "max_direct_excess_fraction_of_peak": direct_excess / max(float(ssrd.max()), 1.0),
        "hours_direct_exceeds_global": int((fdir > ssrd).sum()),
        "night_hours_zero_ssrd": int((ssrd <= 0.0).sum()),
        "max_wind_speed_100m_ms": float(speed.max()),
        "mean_wind_speed_100m_ms": float(speed.mean()),
    }
    return arrays, response, diagnostics


# Rebind the resource-specific pieces.  Script 40's functions resolve these as
# module globals at call time, so the reused infrastructure picks them up.
W40.DEMAND_VARIABLES = RESOURCE_VARIABLES
W40.VARIABLE_ALIASES = VARIABLE_ALIASES
W40.EXPECTED_UNITS = EXPECTED_UNITS
W40.ARRAY_NAMES = ARRAY_NAMES
W40.METHOD_VERSION = METHOD_VERSION
W40.load_point_arrays = load_point_arrays


if __name__ == "__main__":
    if not any(a.startswith("--output-root") for a in sys.argv[1:]):
        # parse_args baked the demand default in at import time, so the resource
        # destination has to be supplied explicitly or the two products collide.
        sys.argv.extend(["--output-root", str(DEFAULT_OUTPUT_ROOT)])
    W40.main()
