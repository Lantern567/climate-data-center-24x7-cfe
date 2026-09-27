# -*- coding: utf-8 -*-
"""Four-cell demand/supply climate decomposition (proposal section 6.6).

Runs the S3 hourly matching kernel over the two-by-two grid of historical and
future demand against historical and future supply, then splits the joint change
into a demand effect, a supply effect and an interaction term:

    delta_D  = Y(D_F, S_H) - Y(D_H, S_H)
    delta_S  = Y(D_H, S_F) - Y(D_H, S_H)
    delta_DS = Y(D_F, S_F) - Y(D_H, S_H) - delta_D - delta_S

The interaction term is the reason both sides had to be driven by the *same*
climate realisation.  Splicing an unrelated demand year onto a supply year would
force delta_DS toward zero by construction, so the design constraint in section
6.2 is what makes this number meaningful rather than an artefact.

`delta_DS` is not "the weather correlation".  It also carries the non-linearity
of the matching operator itself: `min(load, generation)` is concave, so even
perfectly independent shifts on the two sides do not add.  That is a feature --
it is exactly the optimisation non-linearity the proposal wants separated from
the single-sided effects -- but it forbids reading delta_DS as a pure physical
co-occurrence signal.

Portfolio mixes stay frozen at the historical optimum in all four cells.  Letting
each cell re-optimise would fold an adaptation decision into what is meant to be
a pure exposure measurement.

## Two sizing conventions, and why the default hides the demand effect

The matching kernel sizes the procured portfolio to an annual energy target and
then spends it through a normalised hourly shape.  *Which* annual demand sets
that target is a modelling choice, and it decides what `delta_D` can possibly
measure:

`--size-per-cell` (default, matches the published four-cell run)
    Each cell sizes its portfolio to its own annual demand.  Future demand rises
    0.73%, and the portfolio silently grows 0.73% with it, so a uniform load
    increase cancels exactly.  What survives is the change in the *shape* of
    demand -- which is 0.10% by L1 distance.  This measures exposure under
    perfect volume adaptation, and it is why delta_D came out near zero.

`--freeze-volume`
    All four cells size the portfolio to the *historical* annual demand.  This is
    the proposal's section 6.6 intent -- a portfolio bought under today's climate
    and then operated under tomorrow's -- so the demand level effect is retained
    instead of being divided out.

Neither is wrong; they answer different questions.  Report both.
"""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "24x7_cfe" / "processed"
CROSSWALK = PROCESSED / "climate_coupling" / "demand_supply_country_crosswalk"
SUPPLY_HIST = (PROCESSED / "wind_solar_supply_20y_fixed_2025"
               / "replay_results_pv_corrected")
DEFAULT_SUPPLY_FUTURE = (PROCESSED / "wind_solar_supply_20y_fixed_2025"
                         / "replay_results_future_ssp245")
DEMAND_HIST = Path(r"D:\facility_demand_replay_1995_2014")
DEMAND_FUTURE = Path(r"D:\facility_demand_future_1995_2014")
DEFAULT_OUTPUT = PROCESSED / "climate_counterfactuals"
DEFAULT_VALIDATION = ROOT / "output" / "24x7_cfe_facility" / "climate_counterfactuals"

METHOD_VERSION = "climate_counterfactual_v1"
DEMAND_COLUMN = "global_reconciled_allocation_facility_load_mw"
REFERENCE_YEAR = 2000
GAP_WINDOWS = (24, 168)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--supply-future", type=Path, default=DEFAULT_SUPPLY_FUTURE)
    # Must be given per model.  It was previously a module constant, and the
    # multi-model runner silently fed every model the ACCESS-CM2 demand replay,
    # producing identical delta_D for two models with different temperature
    # deltas -- the giveaway that caught it.
    p.add_argument("--demand-future", type=Path, default=DEMAND_FUTURE)
    p.add_argument("--year", type=int, action="append")
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--validation-root", type=Path, default=None)
    p.add_argument("--freeze-volume", action="store_true",
                   help="Size the portfolio to historical annual demand in all "
                        "four cells (proposal 6.6 'buy under today, run under "
                        "tomorrow'), instead of letting each cell size to its "
                        "own demand.")
    args = p.parse_args()
    suffix = "_frozen_volume" if args.freeze_volume else ""
    if args.output_root is None:
        args.output_root = DEFAULT_OUTPUT.with_name(DEFAULT_OUTPUT.name + suffix)
    if args.validation_root is None:
        args.validation_root = DEFAULT_VALIDATION.with_name(
            DEFAULT_VALIDATION.name + suffix)
    return args


def calendar(year: int) -> pd.DatetimeIndex:
    return pd.date_range(f"{year}-01-01", f"{year + 1}-01-01", freq="h",
                         inclusive="left", tz="UTC")


def country_cf(root: Path, year: int) -> tuple[pd.Index, np.ndarray, np.ndarray]:
    partition = root / "hourly_country_supply" / f"weather_year={year}"
    frame = pd.read_parquet(next(partition.glob("*.parquet")))
    annual = pd.read_parquet(root / "annual_country_supply_1995_2014.parquet")
    annual = annual[annual.weather_year.eq(year)].set_index("country_area_code")
    hours = len(calendar(year))
    frame = frame.sort_values(["country_area_code", "time_utc"])
    codes = pd.Index(frame.country_area_code.unique())
    pv = frame.solar_pv_generation_power_mw.to_numpy().reshape(len(codes), hours)
    wd = frame.wind_generation_power_mw.to_numpy().reshape(len(codes), hours)
    pv_cap = annual.loc[codes, "solar_pv_installed_capacity_mw"].to_numpy()
    wd_cap = annual.loc[codes, "wind_installed_capacity_mw"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        pv_cf = np.where(pv_cap[:, None] > 0, pv / np.maximum(pv_cap[:, None], 1e-9), 0.0)
        wd_cf = np.where(wd_cap[:, None] > 0, wd / np.maximum(wd_cap[:, None], 1e-9), 0.0)
    return codes, pv_cf, wd_cf


def load_demand(root: Path, year: int, sites: pd.DataFrame) -> np.ndarray:
    frame = pq.read_table(
        root / f"facility_hourly_demand_replay_{year}.parquet",
        columns=["logical_site_id", "time_utc", DEMAND_COLUMN]).to_pandas()
    frame = frame.sort_values(["logical_site_id", "time_utc"])
    hours = len(calendar(year))
    ids = pd.Index(frame.logical_site_id.unique())
    values = frame[DEMAND_COLUMN].to_numpy(dtype=np.float64).reshape(-1, hours)
    order = pd.Series(range(len(ids)), index=ids)
    return values[order.reindex(sites.logical_site_id).to_numpy()]


def portfolio_shape(pv_cf: np.ndarray, wd_cf: np.ndarray, mix: float) -> np.ndarray:
    e_pv, e_wd = pv_cf.sum(), wd_cf.sum()
    if e_pv <= 0 and e_wd <= 0:
        return np.zeros_like(pv_cf)
    if e_wd <= 0:
        mix = 0.0
    elif e_pv <= 0:
        mix = 1.0
    shape = np.zeros_like(pv_cf, dtype=float)
    if mix < 1.0:
        shape += (1.0 - mix) * pv_cf / e_pv
    if mix > 0.0:
        shape += mix * wd_cf / e_wd
    return shape


def evaluate(load: np.ndarray, shape: np.ndarray,
             sizing_load: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Match `load` against a portfolio whose annual energy equals `sizing_load`.

    `sizing_load` defaults to `load` itself -- the portfolio is bought for the
    demand it is about to serve.  Passing the historical load instead freezes the
    procurement volume, so a change in demand level is no longer cancelled by a
    matching change in procurement.
    """
    demand = load.sum(axis=1)
    procured = demand if sizing_load is None else sizing_load.sum(axis=1)
    generation = procured[:, None] * shape[None, :]
    matched = np.minimum(load, generation)
    unmatched = np.maximum(load - generation, 0.0)
    out = {"demand_mwh": demand,
           "matched_mwh": matched.sum(axis=1),
           "unmatched_mwh": unmatched.sum(axis=1)}
    for window in GAP_WINDOWS:
        kernel = np.ones(window)
        rolled = np.apply_along_axis(
            lambda row: np.convolve(row, kernel, mode="valid"), 1, unmatched)
        out[f"worst_{window}h_mwh"] = rolled.max(axis=1)
    return out


def main() -> None:
    args = parse_args()
    years = tuple(sorted(set(args.year))) if args.year else (REFERENCE_YEAR,)
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.validation_root.mkdir(parents=True, exist_ok=True)

    sites = pd.read_csv(CROSSWALK / "facility_to_era5_grid_with_country.csv",
                        low_memory=False)
    sites = sites[sites.supply_country_area_code.notna()].sort_values(
        "logical_site_id").reset_index(drop=True)
    # The frozen historical mixes: the same portfolios the ladder used, so all
    # four cells differ only in weather, never in the procurement decision.
    ladder = pd.read_parquet(
        PROCESSED / "cfe_geographic_scenarios" / "site_scenario_year_cfe.parquet",
        columns=["scenario", "weather_year", "logical_site_id", "wind_energy_share"])
    ladder = ladder[ladder.scenario.eq("S3") & ladder.weather_year.eq(REFERENCE_YEAR)]
    mixes = ladder.set_index("logical_site_id").wind_energy_share

    records = []
    for year in years:
        demand = {"H": load_demand(DEMAND_HIST, year, sites),
                  "F": load_demand(args.demand_future, year, sites)}
        supply = {"H": country_cf(SUPPLY_HIST, year),
                  "F": country_cf(args.supply_future, year)}
        groups = sites.groupby("supply_country_area_code").indices
        for d_label in ("H", "F"):
            for s_label in ("H", "F"):
                codes, pv_cf, wd_cf = supply[s_label]
                code_row = pd.Series(range(len(codes)), index=codes)
                totals: dict[str, float] = {}
                for code, positions in groups.items():
                    if code not in code_row.index:
                        continue
                    sub = demand[d_label][positions]
                    sizing = demand["H"][positions] if args.freeze_volume else None
                    site_mix = mixes.reindex(
                        sites.logical_site_id.iloc[positions]).to_numpy()
                    for mix in np.unique(site_mix[~np.isnan(site_mix)]):
                        which = site_mix == mix
                        shape = portfolio_shape(pv_cf[code_row[code]],
                                                wd_cf[code_row[code]], float(mix))
                        block = evaluate(sub[which], shape,
                                         None if sizing is None else sizing[which])
                        for key, values in block.items():
                            totals[key] = totals.get(key, 0.0) + float(values.sum())
                records.append({"weather_year": year, "demand": d_label,
                                "supply": s_label,
                                "cfe": totals["matched_mwh"] / totals["demand_mwh"],
                                **totals})
                print(f"  {year} D{d_label}S{s_label}: "
                      f"CFE {records[-1]['cfe']:.4f}", flush=True)

    frame = pd.DataFrame(records)
    frame.to_csv(args.output_root / "counterfactual_cells.csv", index=False)

    decomposition = []
    for year, block in frame.groupby("weather_year"):
        cell = {(r.demand, r.supply): r for r in block.itertuples(index=False)}
        for metric in ("cfe", "unmatched_mwh", "worst_24h_mwh", "worst_168h_mwh"):
            base = getattr(cell[("H", "H")], metric)
            d_only = getattr(cell[("F", "H")], metric) - base
            s_only = getattr(cell[("H", "F")], metric) - base
            joint = getattr(cell[("F", "F")], metric) - base
            decomposition.append({
                "weather_year": year, "metric": metric, "baseline": base,
                "delta_demand": d_only, "delta_supply": s_only,
                "delta_joint": joint,
                "delta_interaction": joint - d_only - s_only,
                "supply_over_demand": (abs(s_only) / abs(d_only)
                                       if d_only else float("nan")),
            })
    table = pd.DataFrame(decomposition)
    table.to_csv(args.output_root / "counterfactual_decomposition.csv", index=False)

    payload = {
        "schema_version": "1.0",
        "method_version": METHOD_VERSION,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "weather_years": [int(y) for y in years],
        "scenario": "S3 same-grid-region, alpha=1, no storage",
        "portfolio_mixes": "frozen at the historical optimum in all four cells",
        "sizing_convention": ("historical annual demand in all four cells"
                             if args.freeze_volume else
                             "each cell sized to its own annual demand"),
        "procurement_volume_frozen": bool(args.freeze_volume),
        "climate": "ACCESS-CM2 SSP2-4.5 near-term (2031-2050)",
        "single_model_single_scenario": True,
        "interaction_includes_operator_nonlinearity": True,
        "cells": frame.to_dict("records"),
        "decomposition": table.to_dict("records"),
    }
    (args.validation_root / "climate_counterfactuals.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), "utf-8")
    print("\n" + table[table.metric.eq("cfe")].to_string(index=False))


if __name__ == "__main__":
    main()
