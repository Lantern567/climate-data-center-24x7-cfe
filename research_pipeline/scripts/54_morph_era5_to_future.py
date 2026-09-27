# -*- coding: utf-8 -*-
"""Apply QDM deltas to the ERA5 hourly record, producing a future-climate replay.

Script 53 produced, per cell and per calendar month, how each quantile of the
distribution moves between the model's historical and future periods.  This
applies those shifts to the real ERA5 hours:

    additive        x_future(t) = x_era5(t) + delta(p(t))
    multiplicative  x_future(t) = x_era5(t) * ratio(p(t))

where p(t) is the hour's own quantile within its cell-month ERA5 distribution.
Every hour keeps its position in the record, so heatwave timing, calm spells and
the co-occurrence between variables survive untouched -- only the distribution
moves.  The output slots straight into scripts 42, 47 and 50 with no changes.

## Variable-by-variable, and the honest caveats

  t2m   <- tas       additive in kelvin.
  RH    <- hurs      additive in percent.  ERA5 stores dew point, not RH, so the
                     pair is converted to RH, RH and temperature are morphed
                     independently, and the dew point is rebuilt from the morphed
                     pair.  Morphing d2m directly would let the dew point drift
                     above the dry bulb.
  u100  <- sfcWind   multiplicative, applied to both components so wind
                     DIRECTION is preserved.  NEX publishes 10 m wind while the
                     study uses 100 m; the ratio is assumed transferable because
                     both track the same large-scale flow.  This is an
                     assumption, and it is recorded as one.
  ssrd  <- rsds      multiplicative.  fdir takes the same ratio so the
                     direct/diffuse split is not silently re-partitioned.
  sp                 unchanged: NEX-GDDP publishes no surface-pressure field,
                     and its climate-change signal is negligible for air density
                     next to the wind-speed signal.

## Two temperature modes, and why the default under-reads the tail

`--temperature-mode quantile` (default, reproduces the published future replay)
    t_future(t) = t(t) + Delta_tas(p_hourly(t))

    Two things go wrong here, and they push in the same direction.  First, the
    `tas` delta is built from the model's *daily mean* distribution but is read
    at the hour's rank in the *hourly* distribution.  Those are different
    objects: the hourly spread contains the diurnal cycle, the daily-mean spread
    does not, so "the 95th percentile hour" (a summer afternoon) is handed the
    shift belonging to "the 95th percentile day" (a warm day).  Diurnal position
    and synoptic anomaly get conflated.  Second, and decisively, the `tas` delta
    is nearly flat in quantile -- global mean q95 - q05 is +0.055 K on a +1.74 K
    base -- so the morph degenerates into adding one constant per cell-month.
    Hot hours and cold hours warm by the same amount, the load curve translates
    without changing shape (measured: L1 shape distance 0.10%), and no tail
    amplification can appear no matter how the downstream model behaves.

`--temperature-mode shift-stretch`
    Belcher-style morphing, which separates the two scales instead of mixing
    them:

        t_future(t) = daily_mean(d) + Delta_tas(p_daily(daily_mean(d)))
                    + [t(t) - daily_mean(d)] * alpha

    The shift is now read at the daily-mean's own rank in the daily-mean
    distribution, which is the scale the delta was actually built on.  The
    stretch alpha = 1 + dDTR / DTR_era5 carries the change in diurnal range,
    taken from the tasmax and tasmin deltas.  Warming that lands
    disproportionately on the daily maximum is exactly the part cooling load
    responds to, and it is absent from the default mode by construction.

    Requires tasmax/tasmin deltas from script 53; DTR_era5 is measured from the
    ERA5 record itself rather than from the model, because it is the ERA5
    anomaly that is being stretched.

## Scope

Only the 2,416 demand cells can be morphed today.  The supply-side extraction
kept `pv_cf`/`wind_cf` and discarded the raw fields, so the 21,466 supply cells
have nothing to morph -- the direct cost of that earlier storage decision.  That
limits the counterfactual set to (D_H,S_H) and (D_F,S_H), i.e. the demand-side
effect only.  Recovering the supply leg means re-extracting raw weather for the
supply cells.
"""
from __future__ import annotations

import argparse
import os
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "24x7_cfe" / "processed"
POINTS = Path(os.environ.get("NEX_POINTS_ROOT") or r"D:\nex_gddp_points")
DELTAS = Path(os.environ.get("NEX_DELTAS_ROOT") or r"D:\nex_gddp_deltas")
WEATHER_ROOT = Path(r"D:\era5_facility_weather_1995_2014")
RESOURCE_ROOT = Path(r"D:\era5_facility_resource_1995_2014")
DEFAULT_OUTPUT = Path(r"D:\era5_facility_future_1995_2014")

METHOD_VERSION = "era5_qdm_morph_v1"
ENDPOINTS = 175_321
QUANTILES = np.round(np.arange(0.05, 0.96, 0.05), 2)
QUANTILE_COLUMNS = [f"q{int(round(q * 100)):02d}" for q in QUANTILES]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="ACCESS-CM2")
    p.add_argument("--scenario", default="ssp245")
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--limit", type=int)
    p.add_argument("--force", action="store_true")
    p.add_argument("--temperature-mode", default="quantile",
                   choices=("quantile", "shift-stretch"))
    p.add_argument("--alpha-clip", type=float, default=0.5,
                   help="Bound the diurnal stretch to 1 +/- this, so a cell with "
                        "a near-zero observed daily range cannot blow up.")
    args = p.parse_args()
    if args.output_root is None:
        args.output_root = (DEFAULT_OUTPUT if args.temperature_mode == "quantile"
                            else DEFAULT_OUTPUT.with_name(
                                DEFAULT_OUTPUT.name + "_shift_stretch"))
    return args


def saturation_vapour_pressure_hpa(temperature_c: np.ndarray) -> np.ndarray:
    return 6.112 * np.exp(17.67 * temperature_c / (temperature_c + 243.5))


def relative_humidity_percent(t_k: np.ndarray, d_k: np.ndarray) -> np.ndarray:
    return 100.0 * (saturation_vapour_pressure_hpa(d_k - 273.15)
                    / saturation_vapour_pressure_hpa(t_k - 273.15))


def dewpoint_from_rh(t_k: np.ndarray, rh_percent: np.ndarray) -> np.ndarray:
    """Invert the Magnus form used everywhere else in the pipeline."""
    rh = np.clip(rh_percent, 1e-3, 100.0)
    e = saturation_vapour_pressure_hpa(t_k - 273.15) * rh / 100.0
    ln = np.log(e / 6.112)
    return 243.5 * ln / (17.67 - ln) + 273.15


def day_index(endpoints: pd.DatetimeIndex) -> np.ndarray:
    """Which calendar day each endpoint belongs to, as a 0-based counter."""
    return (endpoints.normalize() - endpoints[0].normalize()).days.to_numpy()


def daily_mean_by_day(values: np.ndarray, days: np.ndarray) -> np.ndarray:
    counts = np.bincount(days)
    return np.bincount(days, weights=values) / np.maximum(counts, 1)


def diurnal_range_by_month(values: np.ndarray, days: np.ndarray,
                           day_month: np.ndarray) -> dict[int, float]:
    """Observed mean daily range of the ERA5 record, per calendar month.

    This is measured on ERA5 rather than on the model because it is the ERA5
    intraday anomaly that gets stretched; using the model's own range would
    import its diurnal-cycle bias into the stretch factor.
    """
    order = np.argsort(days, kind="stable")
    sorted_days, sorted_values = days[order], values[order]
    bounds = np.searchsorted(sorted_days, np.arange(sorted_days[-1] + 2))
    highs = np.maximum.reduceat(sorted_values, bounds[:-1])
    lows = np.minimum.reduceat(sorted_values, bounds[:-1])
    ranges = highs - lows
    return {m: float(np.nanmean(ranges[day_month == m])) if (day_month == m).any()
            else np.nan for m in range(1, 13)}


def shift_stretch_temperature(t2m: np.ndarray, endpoints: pd.DatetimeIndex,
                              tas_curve: dict[int, np.ndarray],
                              tasmax_curve: dict[int, np.ndarray],
                              tasmin_curve: dict[int, np.ndarray],
                              alpha_clip: float) -> tuple[np.ndarray, dict]:
    """Belcher-style morph: shift the daily mean, stretch the intraday anomaly.

    The shift is read at the daily mean's rank in the *daily-mean* distribution,
    matching the scale the NEX delta was built on.  The stretch carries the
    change in diurnal range implied by the tasmax and tasmin deltas.
    """
    days = day_index(endpoints)
    means = daily_mean_by_day(t2m, days)
    anomaly = t2m - means[days]
    day_month = pd.DatetimeIndex(
        endpoints[0].normalize() + pd.to_timedelta(np.arange(len(means)), "D")
    ).month.to_numpy()
    era5_dtr = diurnal_range_by_month(t2m, days, day_month)

    shifted_means = means.copy()
    alphas: dict[int, float] = {}
    for m in range(1, 13):
        mask = day_month == m
        if not mask.any():
            continue
        sample = means[mask]
        reference = np.quantile(sample, QUANTILES)
        shifted_means[mask] = sample + np.interp(sample, reference, tas_curve[m])
        # Mean change of the daily extremes; the quantile grid is uniform in
        # probability, so averaging it estimates the change in the mean.
        d_range = float(np.mean(tasmax_curve[m]) - np.mean(tasmin_curve[m]))
        base = era5_dtr.get(m, np.nan)
        alpha = 1.0 if not np.isfinite(base) or base <= 1e-6 else 1.0 + d_range / base
        alphas[m] = float(np.clip(alpha, 1.0 - alpha_clip, 1.0 + alpha_clip))

    stretch = np.array([alphas.get(m, 1.0) for m in range(1, 13)])[day_month - 1]
    out = shifted_means[days] + anomaly * stretch[days]
    return out, {"alpha_mean": float(np.mean(list(alphas.values()) or [1.0])),
                 "dtr_change_k": float(np.nanmean(
                     [(alphas.get(m, 1.0) - 1.0) * era5_dtr.get(m, np.nan)
                      for m in range(1, 13)]))}


def apply_delta(values: np.ndarray, month: np.ndarray,
                delta: dict[int, np.ndarray], multiplicative: bool) -> np.ndarray:
    """Shift each hour by the delta at its own quantile within its cell-month."""
    out = values.astype(np.float64).copy()
    for m in range(1, 13):
        mask = month == m
        if not mask.any():
            continue
        sample = values[mask]
        # Empirical quantile of every hour within this cell-month, then read the
        # delta curve at that quantile.  Interpolation keeps the mapping
        # monotone, so the ordering of hours is never reshuffled.
        reference = np.quantile(sample, QUANTILES)
        shift = np.interp(sample, reference, delta[m])
        out[mask] = sample * shift if multiplicative else sample + shift
    return out


def load_delta(model: str, scenario: str, variable: str) -> pd.DataFrame:
    path = DELTAS / f"delta_{model}_{scenario}_{variable}.parquet"
    frame = pd.read_parquet(path)
    return frame.set_index(["cell_id", "month"])[QUANTILE_COLUMNS]


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    cells = pd.read_csv(POINTS / "cells.csv")
    catalog = pd.read_csv(PROCESSED / "climate_coupling" / "era5_grid_catalog.csv")
    key = lambda lat, lon: (np.round(lat, 2), np.round(lon, 2))
    lookup = {key(r.era5_latitude, r.era5_longitude): r.cell_id
              for r in cells.itertuples(index=False)}
    catalog["cell_id"] = [lookup.get(key(r.era5_latitude, r.era5_longitude))
                          for r in catalog.itertuples(index=False)]
    missing = int(catalog.cell_id.isna().sum())
    if missing:
        raise ValueError(f"{missing} facility cells have no NEX delta cell")
    if args.limit:
        catalog = catalog.head(args.limit)

    wanted = ["tas", "hurs", "sfcWind", "rsds"]
    if args.temperature_mode == "shift-stretch":
        wanted += ["tasmax", "tasmin"]
    deltas = {v: load_delta(args.model, args.scenario, v) for v in wanted}
    endpoints = pd.date_range("1995-01-01", "2015-01-01", freq="h", tz="UTC")
    month = endpoints.month.to_numpy()

    diagnostics = {"warming_k": [], "rh_change": [], "wind_ratio": [],
                   "radiation_ratio": [], "dewpoint_above_dry_bulb": 0,
                   "diurnal_stretch": [], "diurnal_range_change_k": []}
    for position, row in enumerate(catalog.itertuples(index=False), start=1):
        grid_id, cell_id = str(row.era5_grid_id), str(row.cell_id)
        target = args.output_root / f"{grid_id}.npz"
        if target.exists() and not args.force:
            continue
        with np.load(WEATHER_ROOT / f"{grid_id}.npz") as w, \
             np.load(RESOURCE_ROOT / f"{grid_id}.npz") as r:
            t2m, d2m, sp = w["t2m_k"], w["d2m_k"], w["sp_pa"]
            ssrd, fdir = r["ssrd_j_m2"], r["fdir_j_m2"]
            u100, v100 = r["u100_ms"], r["v100_ms"]

        curve = {v: {m: deltas[v].loc[(cell_id, m)].to_numpy(dtype=float)
                     for m in range(1, 13)} for v in deltas}

        if args.temperature_mode == "shift-stretch":
            t2m_f, stretch_info = shift_stretch_temperature(
                t2m, endpoints, curve["tas"], curve["tasmax"], curve["tasmin"],
                args.alpha_clip)
            diagnostics["diurnal_stretch"].append(stretch_info["alpha_mean"])
            diagnostics["diurnal_range_change_k"].append(
                stretch_info["dtr_change_k"])
        else:
            t2m_f = apply_delta(t2m, month, curve["tas"], multiplicative=False)
        rh = relative_humidity_percent(t2m, d2m)
        rh_f = np.clip(apply_delta(rh, month, curve["hurs"], multiplicative=False),
                       1e-3, 100.0)
        d2m_f = dewpoint_from_rh(t2m_f, rh_f)

        speed = np.hypot(u100, v100)
        speed_f = apply_delta(speed, month, curve["sfcWind"], multiplicative=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            scale = np.where(speed > 1e-6, speed_f / np.maximum(speed, 1e-6), 1.0)
        u_f, v_f = u100 * scale, v100 * scale

        ssrd_f = apply_delta(ssrd, month, curve["rsds"], multiplicative=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            rad_scale = np.where(ssrd > 1e-6, ssrd_f / np.maximum(ssrd, 1e-6), 1.0)
        fdir_f = fdir * rad_scale

        excess = float((d2m_f - t2m_f).max())
        if excess > 0.5:
            diagnostics["dewpoint_above_dry_bulb"] += 1
        diagnostics["warming_k"].append(float((t2m_f - t2m).mean()))
        diagnostics["rh_change"].append(float((rh_f - rh).mean()))
        diagnostics["wind_ratio"].append(float(speed_f.mean() / max(speed.mean(), 1e-9)))
        diagnostics["radiation_ratio"].append(
            float(ssrd_f.mean() / max(ssrd.mean(), 1e-9)))

        staging = args.output_root / f"{grid_id}.partial.npz"
        np.savez_compressed(
            staging,
            t2m_k=t2m_f.astype(np.float32), d2m_k=d2m_f.astype(np.float32),
            sp_pa=sp.astype(np.float32),
            ssrd_j_m2=np.maximum(ssrd_f, 0.0).astype(np.float32),
            fdir_j_m2=np.maximum(fdir_f, 0.0).astype(np.float32),
            u100_ms=u_f.astype(np.float32), v100_ms=v_f.astype(np.float32),
        )
        staging.replace(target)
        if position % 200 == 0:
            print(f"  {position}/{len(catalog)} cells", flush=True)

    payload = {
        "schema_version": "1.0",
        "method_version": METHOD_VERSION,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "model": args.model, "scenario": args.scenario,
        "temperature_mode": args.temperature_mode,
        "mean_diurnal_stretch": (float(np.mean(diagnostics["diurnal_stretch"]))
                                 if diagnostics["diurnal_stretch"] else None),
        "mean_diurnal_range_change_k": (
            float(np.nanmean(diagnostics["diurnal_range_change_k"]))
            if diagnostics["diurnal_range_change_k"] else None),
        "cells": int(len(catalog)),
        "endpoints_per_cell": ENDPOINTS,
        "mean_warming_k": float(np.mean(diagnostics["warming_k"])),
        "mean_rh_change_percent": float(np.mean(diagnostics["rh_change"])),
        "mean_wind_ratio": float(np.mean(diagnostics["wind_ratio"])),
        "mean_radiation_ratio": float(np.mean(diagnostics["radiation_ratio"])),
        "cells_with_dewpoint_above_dry_bulb": diagnostics["dewpoint_above_dry_bulb"],
        "surface_pressure_unchanged": True,
        "wind_ratio_from_10m_applied_to_100m": True,
        "correlation_structure_assumed_fixed": True,
        "supply_cells_not_morphed_raw_weather_discarded": True,
        "hard_checks": {
            "warming_is_positive": float(np.mean(diagnostics["warming_k"])) > 0,
            "no_dewpoint_inversion": diagnostics["dewpoint_above_dry_bulb"] == 0,
            "wind_ratio_near_unity": abs(
                float(np.mean(diagnostics["wind_ratio"])) - 1.0) < 0.15,
        },
    }
    payload["all_hard_checks_pass"] = all(payload["hard_checks"].values())
    (args.output_root / f"morph_validation_{args.model}_{args.scenario}.json"
     ).write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                             allow_nan=False), "utf-8")
    print(json.dumps({k: v for k, v in payload.items()
                      if k not in ("schema_version", "completed_at_utc")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
