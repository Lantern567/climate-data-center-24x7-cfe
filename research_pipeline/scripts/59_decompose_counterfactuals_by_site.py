# -*- coding: utf-8 -*-
"""Site-resolved four-cell demand/supply climate decomposition.

Script 56 evaluates the same two-by-two grid but collapses every cell to a
global scalar, so the published global change is the residual of offsetting
regional signals.  The matching kernel already produces per-site vectors --
`evaluate()` sums along the time axis, leaving one value per site -- and 56
simply adds them up.  This script keeps them.

Nothing else changes: same frozen S3 portfolio mixes, same sizing conventions,
same metrics.  The output is therefore exactly decomposable back into 56's
numbers, which is the first hard check run at the end.

Resolution caveat, which the manuscript has to carry: the demand leg is
genuinely per site (each facility sits on its own ERA5 cell), but the supply leg
is a country-level fleet capacity factor shared by every site in that country
with the same wind share.  Site-level maps of the supply effect therefore show
country blocks, not site variation.
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

METHOD_VERSION = "climate_counterfactual_site_v1"
DEMAND_COLUMN = "global_reconciled_allocation_facility_load_mw"
REFERENCE_YEAR = 2000
GAP_WINDOWS = (24, 168)
METRICS = ("demand_mwh", "matched_mwh", "unmatched_mwh",
           "worst_24h_mwh", "worst_168h_mwh")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--supply-future", type=Path, default=DEFAULT_SUPPLY_FUTURE)
    p.add_argument("--demand-future", type=Path, default=DEMAND_FUTURE)
    p.add_argument("--year", type=int, action="append")
    p.add_argument("--output-root", type=Path,
                   default=PROCESSED / "climate_counterfactuals_by_site")
    p.add_argument("--freeze-volume", action="store_true")
    p.add_argument("--label", default="ACCESS-CM2")
    return p.parse_args()


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
    demand = load.sum(axis=1)
    procured = demand if sizing_load is None else sizing_load.sum(axis=1)
    generation = procured[:, None] * shape[None, :]
    matched = np.minimum(load, generation)
    unmatched = np.maximum(load - generation, 0.0)
    out = {"demand_mwh": demand,
           "matched_mwh": matched.sum(axis=1),
           "unmatched_mwh": unmatched.sum(axis=1)}
    # Sliding-window sums by cumulative difference rather than np.convolve.  The
    # convolution is O(hours * window) per site and made the 20-year site-level
    # run a three-hour job; the cumsum form is O(hours) and reproduces it exactly.
    csum = np.cumsum(unmatched, axis=1)
    for window in GAP_WINDOWS:
        rolled = csum[:, window - 1:].copy()
        rolled[:, 1:] -= csum[:, :-window]
        out[f"worst_{window}h_mwh"] = rolled.max(axis=1)
    return out


def main() -> None:
    args = parse_args()
    years = tuple(sorted(set(args.year))) if args.year else (REFERENCE_YEAR,)
    args.output_root.mkdir(parents=True, exist_ok=True)

    sites = pd.read_csv(CROSSWALK / "facility_to_era5_grid_with_country.csv",
                        low_memory=False)
    sites = sites[sites.supply_country_area_code.notna()].sort_values(
        "logical_site_id").reset_index(drop=True)
    ladder = pd.read_parquet(
        PROCESSED / "cfe_geographic_scenarios" / "site_scenario_year_cfe.parquet",
        columns=["scenario", "weather_year", "logical_site_id", "wind_energy_share"])
    ladder = ladder[ladder.scenario.eq("S3") & ladder.weather_year.eq(REFERENCE_YEAR)]
    mixes = ladder.set_index("logical_site_id").wind_energy_share

    rows = []
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
                for code, positions in groups.items():
                    if code not in code_row.index:
                        continue
                    sub = demand[d_label][positions]
                    sizing = demand["H"][positions] if args.freeze_volume else None
                    site_ids = sites.logical_site_id.iloc[positions].to_numpy()
                    site_mix = mixes.reindex(site_ids).to_numpy()
                    for mix in np.unique(site_mix[~np.isnan(site_mix)]):
                        which = site_mix == mix
                        shape = portfolio_shape(pv_cf[code_row[code]],
                                                wd_cf[code_row[code]], float(mix))
                        block = evaluate(sub[which], shape,
                                         None if sizing is None else sizing[which])
                        frame = pd.DataFrame(block)
                        frame.insert(0, "logical_site_id", site_ids[which])
                        frame.insert(1, "supply_country_area_code", code)
                        frame.insert(2, "weather_year", year)
                        frame.insert(3, "demand", d_label)
                        frame.insert(4, "supply", s_label)
                        rows.append(frame)
        print(f"  {year} done", flush=True)

    panel = pd.concat(rows, ignore_index=True)
    panel.to_parquet(args.output_root / "site_cell_metrics.parquet", index=False)

    # Site-level decomposition, wide over the four cells.
    wide = panel.pivot_table(
        index=["logical_site_id", "supply_country_area_code", "weather_year"],
        columns=["demand", "supply"], values=list(METRICS))
    dec = []
    for metric in METRICS:
        base = wide[(metric, "H", "H")]
        dec.append(pd.DataFrame({
            "metric": metric,
            "baseline": base,
            "delta_demand": wide[(metric, "F", "H")] - base,
            "delta_supply": wide[(metric, "H", "F")] - base,
            "delta_joint": wide[(metric, "F", "F")] - base,
        }))
    site_dec = pd.concat(dec).reset_index()
    site_dec["delta_interaction"] = (site_dec.delta_joint
                                     - site_dec.delta_demand
                                     - site_dec.delta_supply)
    site_dec.to_parquet(args.output_root / "site_decomposition.parquet", index=False)

    # Hard check: re-aggregating must reproduce the global run bit for bit.
    glob = panel.groupby(["weather_year", "demand", "supply"])[list(METRICS)].sum()
    glob["cfe"] = glob.matched_mwh / glob.demand_mwh
    glob.reset_index().to_csv(args.output_root / "reaggregated_cells.csv", index=False)

    payload = {
        "schema_version": "1.0",
        "method_version": METHOD_VERSION,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "model": args.label,
        "weather_years": [int(y) for y in years],
        "sites": int(panel.logical_site_id.nunique()),
        "supply_regions": int(panel.supply_country_area_code.nunique()),
        "procurement_volume_frozen": bool(args.freeze_volume),
        "supply_effect_resolution": "country fleet capacity factor, not per site",
        "global_cfe_by_cell": glob.reset_index().to_dict("records"),
    }
    (args.output_root / "site_decomposition.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), "utf-8")
    print(glob[["cfe"]].round(6).to_string())


if __name__ == "__main__":
    main()
