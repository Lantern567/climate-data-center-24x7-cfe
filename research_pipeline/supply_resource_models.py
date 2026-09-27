"""Versioned wind and PV resource-to-power functions for supply replay.

The equations intentionally reproduce the resource functions first used in
``scripts/26_compute_dcwatch_climate_demand_supply.py``.  Keeping this module
independent prevents the fixed-capacity replay from importing demand logic.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


MODEL_VERSION = "fixed_tilt_pv_and_nrel_composite_wind_v1"
WIND_AVAILABILITY = 0.97
WIND_OTHER_LOSS_FRACTION = 0.0
CENTRAL_WIND_SCENARIO = "iec_ii"
WIND_SCENARIOS = ("iec_i", "iec_ii", "iec_iii", "cubic_conservative")
_WIND_SPEED_NODES = np.arange(0.0, 26.0, 1.0)
WIND_POWER_CURVES: dict[str, dict[str, Any]] = {
    "iec_i": {
        "speed_nodes_m_s": _WIND_SPEED_NODES,
        "power_fraction": np.asarray(
            [
                0.0,
                0.0,
                0.0,
                0.0043,
                0.0323,
                0.0771,
                0.1426,
                0.2329,
                0.3528,
                0.5024,
                0.6732,
                0.8287,
                0.9264,
                0.9774,
                0.9946,
                0.9990,
                0.9999,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            dtype=float,
        ),
        "cut_out_m_s": 26.0,
    },
    "iec_ii": {
        "speed_nodes_m_s": _WIND_SPEED_NODES,
        "power_fraction": np.asarray(
            [
                0.0,
                0.0,
                0.0,
                0.0052,
                0.0423,
                0.1031,
                0.1909,
                0.3127,
                0.4731,
                0.6693,
                0.8554,
                0.9641,
                0.9942,
                0.9994,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            dtype=float,
        ),
        "cut_out_m_s": 26.0,
    },
    "iec_iii": {
        "speed_nodes_m_s": np.arange(0.0, 23.0, 1.0),
        "power_fraction": np.asarray(
            [
                0.0,
                0.0,
                0.0,
                0.0054,
                0.0530,
                0.1351,
                0.2508,
                0.4033,
                0.5952,
                0.7849,
                0.9178,
                0.9796,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            dtype=float,
        ),
        "cut_out_m_s": 23.0,
    },
}


def solar_geometry(
    times: pd.DatetimeIndex,
    latitude: float,
    longitude: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return cosine of zenith and equator-facing fixed-tilt incidence."""

    day = times.dayofyear.to_numpy(dtype=float)
    utc_hour = times.hour.to_numpy(dtype=float) + times.minute.to_numpy(dtype=float) / 60.0
    b_angle = 2.0 * np.pi * (day - 81.0) / 364.0
    equation_of_time_min = (
        9.87 * np.sin(2.0 * b_angle) - 7.53 * np.cos(b_angle) - 1.5 * np.sin(b_angle)
    )
    solar_time = utc_hour + longitude / 15.0 + equation_of_time_min / 60.0
    hour_angle = np.radians(15.0 * (solar_time - 12.0))
    declination = np.radians(23.45 * np.sin(np.radians(360.0 * (284.0 + day) / 365.0)))
    latitude_rad = np.radians(latitude)
    north = (
        np.cos(latitude_rad) * np.sin(declination)
        - np.sin(latitude_rad) * np.cos(declination) * np.cos(hour_angle)
    )
    upward = (
        np.sin(latitude_rad) * np.sin(declination)
        + np.cos(latitude_rad) * np.cos(declination) * np.cos(hour_angle)
    )
    tilt = np.radians(float(np.clip(abs(latitude), 10.0, 60.0)))
    equator_facing_north_component = -1.0 if latitude >= 0.0 else 1.0
    cosine_incidence = (
        upward * np.cos(tilt)
        + north * equator_facing_north_component * np.sin(tilt)
    )
    return upward, cosine_incidence


def fixed_tilt_pv_cf(
    times: pd.DatetimeIndex,
    latitude: float,
    longitude: float,
    ghi: np.ndarray,
    dni: np.ndarray,
    dhi: np.ndarray,
    air_temperature_c: np.ndarray,
) -> np.ndarray:
    """Isotropic transposition plus the registered PVWatts-style AC proxy."""

    _, cosine_incidence = solar_geometry(times, latitude, longitude)
    tilt = np.radians(float(np.clip(abs(latitude), 10.0, 60.0)))
    beam = np.maximum(dni, 0.0) * np.maximum(cosine_incidence, 0.0)
    sky = np.maximum(dhi, 0.0) * (1.0 + np.cos(tilt)) / 2.0
    ground = np.maximum(ghi, 0.0) * 0.20 * (1.0 - np.cos(tilt)) / 2.0
    poa = np.maximum(beam + sky + ground, 0.0)
    cell_temperature = np.asarray(air_temperature_c, dtype=float) + (45.0 - 20.0) / 800.0 * poa
    temperature_factor = np.clip(1.0 - 0.004 * (cell_temperature - 25.0), 0.0, 1.25)
    dc_per_ac = 1.34 * 0.86 * poa / 1_000.0 * temperature_factor
    return np.clip(dc_per_ac, 0.0, 1.0)


def density_corrected_wind_cf_cubic(
    wind_speed_100m: np.ndarray,
    air_temperature_c: np.ndarray,
    surface_pressure_hpa: np.ndarray,
) -> np.ndarray:
    density = np.asarray(surface_pressure_hpa, dtype=float) * 100.0 / (
        287.05 * (np.asarray(air_temperature_c, dtype=float) + 273.15)
    )
    equivalent_speed = np.maximum(wind_speed_100m, 0.0) * np.cbrt(density / 1.225)
    factor = np.zeros_like(equivalent_speed, dtype=float)
    ramp = (equivalent_speed >= 3.0) & (equivalent_speed < 12.0)
    factor[ramp] = (equivalent_speed[ramp] ** 3 - 3.0**3) / (12.0**3 - 3.0**3)
    rated = (equivalent_speed >= 12.0) & (equivalent_speed < 25.0)
    factor[rated] = 1.0
    return np.clip(
        factor * WIND_AVAILABILITY * (1.0 - WIND_OTHER_LOSS_FRACTION),
        0.0,
        WIND_AVAILABILITY,
    )


def density_corrected_wind_cf(
    wind_speed_100m: np.ndarray,
    air_temperature_c: np.ndarray,
    surface_pressure_hpa: np.ndarray,
    scenario: str = CENTRAL_WIND_SCENARIO,
) -> np.ndarray:
    """Apply air-density correction and a named NREL composite power curve."""

    if scenario == "cubic_conservative":
        return density_corrected_wind_cf_cubic(
            wind_speed_100m, air_temperature_c, surface_pressure_hpa
        )
    if scenario not in WIND_POWER_CURVES:
        raise KeyError(f"Unknown wind scenario: {scenario}")
    density = np.asarray(surface_pressure_hpa, dtype=float) * 100.0 / (
        287.05 * (np.asarray(air_temperature_c, dtype=float) + 273.15)
    )
    actual_speed = np.maximum(np.asarray(wind_speed_100m, dtype=float), 0.0)
    equivalent_speed = actual_speed * np.cbrt(density / 1.225)
    curve = WIND_POWER_CURVES[scenario]
    power = np.interp(
        equivalent_speed,
        curve["speed_nodes_m_s"],
        curve["power_fraction"],
        left=0.0,
        right=1.0,
    )
    power[actual_speed >= float(curve["cut_out_m_s"])] = 0.0
    return np.clip(
        power * WIND_AVAILABILITY * (1.0 - WIND_OTHER_LOSS_FRACTION),
        0.0,
        WIND_AVAILABILITY,
    )


def era5_radiation_components_w_m2(
    interval_midpoints: pd.DatetimeIndex,
    latitude: float,
    longitude: float,
    surface_solar_radiation_downwards_j_m2: np.ndarray,
    total_sky_direct_solar_radiation_at_surface_j_m2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Convert one-hour ERA5 accumulations to GHI, DNI and derived DHI.

    ERA5 ``fdir`` is direct radiation on a plane normal to the solar beam.
    ``ssrd`` is global horizontal radiation.  The residual diffuse horizontal
    component is clipped only where sub-hour solar geometry and hourly
    accumulation rounding would otherwise make it slightly negative.
    """

    ghi = np.maximum(np.asarray(surface_solar_radiation_downwards_j_m2, dtype=float), 0.0) / 3600.0
    dni = np.maximum(
        np.asarray(total_sky_direct_solar_radiation_at_surface_j_m2, dtype=float), 0.0
    ) / 3600.0
    cosine_zenith, _ = solar_geometry(interval_midpoints, latitude, longitude)
    cosine_zenith = np.maximum(cosine_zenith, 0.0)
    raw_dhi = ghi - dni * cosine_zenith
    clipped_count = int((raw_dhi < 0.0).sum())
    dhi = np.maximum(raw_dhi, 0.0)
    return ghi, dni, dhi, clipped_count


def interval_capacity_factors(
    endpoint_times: pd.DatetimeIndex,
    latitude: float,
    longitude: float,
    temperature_k: np.ndarray,
    *,
    surface_pressure_pa: np.ndarray | None = None,
    wind_u_100m_m_s: np.ndarray | None = None,
    wind_v_100m_m_s: np.ndarray | None = None,
    surface_solar_radiation_downwards_j_m2: np.ndarray | None = None,
    total_sky_direct_solar_radiation_at_surface_j_m2: np.ndarray | None = None,
    wind_scenario: str = CENTRAL_WIND_SCENARIO,
) -> tuple[dict[str, np.ndarray], dict[str, int | float | str]]:
    """Convert consecutive ERA5 endpoints into hourly interval CF arrays."""

    times = pd.DatetimeIndex(endpoint_times)
    if times.tz is None:
        times = times.tz_localize("UTC")
    else:
        times = times.tz_convert("UTC")
    if len(times) < 2 or not np.all(np.diff(times.asi8) == 3_600_000_000_000):
        raise ValueError("ERA5 endpoint times must be continuous hourly UTC values")
    temperature = np.asarray(temperature_k, dtype=float)
    if len(temperature) != len(times) or not np.isfinite(temperature).all():
        raise ValueError("Temperature endpoints are incomplete or non-finite")
    temperature_c = temperature - 273.15
    interval_temperature_c = 0.5 * (temperature_c[:-1] + temperature_c[1:])
    interval_midpoints = times[:-1] + pd.Timedelta(minutes=30)
    outputs: dict[str, np.ndarray] = {}
    diagnostics: dict[str, int | float | str] = {
        "model_version": MODEL_VERSION,
        "interval_count": len(times) - 1,
    }

    solar_inputs = (
        surface_solar_radiation_downwards_j_m2,
        total_sky_direct_solar_radiation_at_surface_j_m2,
    )
    if any(value is not None for value in solar_inputs):
        if any(value is None for value in solar_inputs):
            raise ValueError("Both ERA5 solar radiation arrays are required")
        ssrd = np.asarray(surface_solar_radiation_downwards_j_m2, dtype=float)
        fdir = np.asarray(total_sky_direct_solar_radiation_at_surface_j_m2, dtype=float)
        if len(ssrd) != len(times) or len(fdir) != len(times):
            raise ValueError("Solar radiation endpoint length mismatch")
        if not np.isfinite(ssrd).all() or not np.isfinite(fdir).all():
            raise ValueError("Solar radiation contains non-finite values")
        # The timestamp is the end of the accumulation interval, so [t,t+1)
        # uses the radiation reported at t+1.
        ghi, dni, dhi, clipped = era5_radiation_components_w_m2(
            interval_midpoints,
            latitude,
            longitude,
            ssrd[1:],
            fdir[1:],
        )
        pv = fixed_tilt_pv_cf(
            interval_midpoints,
            latitude,
            longitude,
            ghi,
            dni,
            dhi,
            interval_temperature_c,
        )
        outputs["pv_cf"] = pv.astype(np.float32)
        diagnostics.update(
            {
                "derived_dhi_negative_clipped_hours": clipped,
                "max_ghi_w_m2": float(np.max(ghi)),
                "max_dni_w_m2": float(np.max(dni)),
            }
        )

    wind_inputs = (surface_pressure_pa, wind_u_100m_m_s, wind_v_100m_m_s)
    if any(value is not None for value in wind_inputs):
        if any(value is None for value in wind_inputs):
            raise ValueError("Pressure and both 100 m wind components are required")
        pressure = np.asarray(surface_pressure_pa, dtype=float)
        u100 = np.asarray(wind_u_100m_m_s, dtype=float)
        v100 = np.asarray(wind_v_100m_m_s, dtype=float)
        if any(len(value) != len(times) for value in (pressure, u100, v100)):
            raise ValueError("Wind endpoint length mismatch")
        if not all(np.isfinite(value).all() for value in (pressure, u100, v100)):
            raise ValueError("Wind inputs contain non-finite values")
        if (pressure <= 0.0).any():
            raise ValueError("ERA5 surface pressure is non-positive")
        speed = np.hypot(u100, v100)
        left_cf = density_corrected_wind_cf(
            speed[:-1], temperature_c[:-1], pressure[:-1] / 100.0, wind_scenario
        )
        right_cf = density_corrected_wind_cf(
            speed[1:], temperature_c[1:], pressure[1:] / 100.0, wind_scenario
        )
        outputs["wind_cf"] = (0.5 * (left_cf + right_cf)).astype(np.float32)
        diagnostics.update(
            {
                "wind_scenario": wind_scenario,
                "max_wind_speed_100m_m_s": float(np.max(speed)),
            }
        )

    if not outputs:
        raise ValueError("At least one solar or wind resource must be requested")
    for name, values in outputs.items():
        upper = WIND_AVAILABILITY if name == "wind_cf" else 1.0
        if (values < 0.0).any() or (values > upper + 1e-7).any():
            raise ValueError(f"{name} is outside its physical model bounds")
    return outputs, diagnostics
