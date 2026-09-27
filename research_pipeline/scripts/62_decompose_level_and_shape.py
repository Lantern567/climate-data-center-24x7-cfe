# -*- coding: utf-8 -*-
"""Result 3: separate the climate signal into level and shape, on both sides.

Script 56 runs the two-by-two grid of historical/future demand against
historical/future supply and reports a demand channel and a supply channel.  The
manuscript's Result 3 needs one level deeper, because the two channels turn out
to be different *kinds* of effect:

    demand level   the procured volume no longer matches the load it must serve
    demand shape   the hourly profile of load has changed
    supply level   annual wind and solar energy has changed
    supply shape   the hourly profile of generation has changed

The matching kernel makes one of these identically zero.  Writing
p_t = (1-w) cf_pv,t / E_pv + w cf_w,t / E_w with E the annual capacity-factor
sum, any constant rescaling cf -> a*cf leaves p_t unchanged, so a pure change in
generation *level* cannot reach the score at all.  That is an algebraic identity,
not an empirical finding, and cell `HL` below verifies it numerically rather than
asking the reader to take it on faith.

The demand side has no such invariance, but only because of the procurement
convention: if the buyer resizes the portfolio to future load, the level effect
cancels the same way; if the buyer carries yesterday's volume forward, it does
not.  Running both conventions in one pass is what turns "cooling dominates" into
the sharper statement the paper makes -- most of the near-term cooling penalty is
a procurement-volume mismatch, not a temporal-matching failure.

## Cells

    HH_vH   baseline                      Y(D_H, S_H)  procured at D_H
    FH_vH   demand total, frozen volume   Y(D_F, S_H)  procured at D_H
    FH_vF   demand shape only             Y(D_F, S_H)  procured at D_F
    HF_vH   supply total (= supply shape) Y(D_H, S_F)  procured at D_H
    HL_vH   supply level only             Y(D_H, S_L)  procured at D_H   -> == HH_vH
    FF_vH   joint, frozen volume          Y(D_F, S_F)  procured at D_H
    FF_vF   joint, adaptive volume        Y(D_F, S_F)  procured at D_F

S_L is the historical supply rescaled to future annual energy, per country and
per technology: the level change with the shape change removed.

Portfolio mixes are supplied from a design file so the same script serves the
old single-year-2000 design and the new twenty-year weather-robust design; the
comparison of the two is itself a manuscript result.
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
SUPPLY_ROOT = PROCESSED / "wind_solar_supply_20y_fixed_2025"
SUPPLY_HIST = SUPPLY_ROOT / "replay_results_pv_corrected"
DESIGN = PROCESSED / "weather_robust_design"
DEMAND_HIST = Path(r"D:\facility_demand_replay_1995_2014")
DEFAULT_VALIDATION = ROOT / "output" / "24x7_cfe_facility" / "level_shape_decomposition"

METHOD_VERSION = "level_shape_decomposition_v1"
DEMAND_COLUMN = "global_reconciled_allocation_facility_load_mw"
REPLAY_YEARS = tuple(range(1995, 2015))
REFERENCE_YEAR = 2000
GAP_WINDOWS = (24, 168)

CELLS = ("HH_vH", "FH_vH", "FH_vF", "HF_vH", "HL_vH", "FF_vH", "FF_vF")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="ACCESS-CM2")
    p.add_argument("--demand-future", type=Path,
                   default=Path(r"D:\facility_demand_future_1995_2014"))
    p.add_argument("--supply-future", type=Path,
                   default=SUPPLY_ROOT / "replay_results_future_ssp245")
    p.add_argument("--design-file", type=Path,
                   default=DESIGN / "site_portfolio_designs.csv")
    p.add_argument("--design-column", default="wind_share_robust_mean_20y")
    p.add_argument("--year", type=int, action="append")
    p.add_argument("--supply-only", action="store_true",
                   help="Skip every cell that needs a future demand replay. Used "
                        "for models whose hourly demand replay is no longer on "
                        "disk but whose future supply replay is.")
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--validation-root", type=Path, default=None)
    args = p.parse_args()
    if args.output_root is None:
        args.output_root = PROCESSED / f"level_shape_{args.model}"
    if args.validation_root is None:
        args.validation_root = DEFAULT_VALIDATION.with_name(
            f"level_shape_{args.model}")
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


def level_only_supply(hist: tuple[pd.Index, np.ndarray, np.ndarray],
                      future: tuple[pd.Index, np.ndarray, np.ndarray]
                      ) -> tuple[pd.Index, np.ndarray, np.ndarray]:
    """Historical shape carried at future annual energy, per country and technology."""
    codes, pv_h, wd_h = hist
    codes_f, pv_f, wd_f = future
    if not codes.equals(codes_f):
        raise SystemExit("historical and future supply carry different country sets")
    with np.errstate(divide="ignore", invalid="ignore"):
        pv_ratio = np.where(pv_h.sum(axis=1) > 0,
                            pv_f.sum(axis=1) / np.maximum(pv_h.sum(axis=1), 1e-30), 1.0)
        wd_ratio = np.where(wd_h.sum(axis=1) > 0,
                            wd_f.sum(axis=1) / np.maximum(wd_h.sum(axis=1), 1e-30), 1.0)
    return codes, pv_h * pv_ratio[:, None], wd_h * wd_ratio[:, None]


def portfolio_shape(pv_cf: np.ndarray, wd_cf: np.ndarray, mix: float) -> np.ndarray:
    e_pv, e_wd = pv_cf.sum(), wd_cf.sum()
    if e_pv <= 0 and e_wd <= 0:
        return np.zeros_like(pv_cf)
    if e_wd <= 0:
        mix = 0.0
    elif e_pv <= 0:
        mix = 1.0
    shape = np.zeros_like(pv_cf, dtype=np.float64)
    if mix < 1.0:
        shape += (1.0 - mix) * pv_cf / e_pv
    if mix > 0.0:
        shape += mix * wd_cf / e_wd
    return shape


def rolling_worst(unmatched: np.ndarray, window: int) -> np.ndarray:
    cumulative = np.empty((unmatched.shape[0], unmatched.shape[1] + 1), np.float64)
    cumulative[:, 0] = 0.0
    np.cumsum(unmatched, axis=1, dtype=np.float64, out=cumulative[:, 1:])
    return (cumulative[:, window:] - cumulative[:, :-window]).max(axis=1)


def site_block_order(handle: pq.ParquetFile, hours: int) -> pd.Index:
    order: list[str] = []
    counts: list[int] = []
    for group in range(handle.metadata.num_row_groups):
        ids = handle.read_row_group(group, columns=["logical_site_id"]) \
            .column("logical_site_id").to_numpy(zero_copy_only=False)
        edges = np.flatnonzero(ids[1:] != ids[:-1]) + 1
        starts = np.concatenate([[0], edges])
        lengths = np.diff(np.concatenate([starts, [len(ids)]]))
        for site, length in zip(ids[starts], lengths):
            if order and order[-1] == site:
                counts[-1] += int(length)
            else:
                order.append(str(site))
                counts.append(int(length))
    if set(counts) != {hours}:
        raise SystemExit(f"demand file does not carry exactly {hours} hours per site")
    return pd.Index(order)


def load_demand(root: Path, year: int, site_ids: pd.Index) -> np.ndarray:
    hours = len(calendar(year))
    handle = pq.ParquetFile(root / f"facility_hourly_demand_replay_{year}.parquet")
    present = site_block_order(handle, hours)
    values = handle.read(columns=[DEMAND_COLUMN]).column(DEMAND_COLUMN) \
        .to_numpy(zero_copy_only=False).astype(np.float32, copy=False).reshape(-1, hours)
    rows = pd.Series(range(len(present)), index=present).reindex(site_ids).to_numpy()
    if pd.isna(rows).any():
        raise SystemExit(f"{int(pd.isna(rows).sum())} sites absent from {year}")
    return np.ascontiguousarray(values[rows.astype(int)])


def longest_run(flag: np.ndarray) -> np.ndarray:
    """Longest consecutive run of True along axis 1, per row.

    Loops over hours with all facilities vectorised.  Script 47 loops over
    facilities in Python instead, which is two orders of magnitude slower at this
    inventory size; the two agree exactly.
    """
    best = np.zeros(flag.shape[0], dtype=np.int64)
    current = np.zeros(flag.shape[0], dtype=np.int64)
    for t in range(flag.shape[1]):
        current = np.where(flag[:, t], current + 1, 0)
        np.maximum(best, current, out=best)
    return best.astype(np.float64)


def evaluate(load: np.ndarray, shape: np.ndarray,
             procured: np.ndarray) -> dict[str, np.ndarray]:
    generation = procured[:, None] * shape[None, :]
    matched = np.minimum(load, generation).sum(axis=1, dtype=np.float64)
    unmatched = np.maximum(load - generation, 0.0)
    out = {"demand_mwh": load.sum(axis=1, dtype=np.float64),
           "matched_mwh": matched,
           "unmatched_mwh": unmatched.sum(axis=1, dtype=np.float64)}
    for window in GAP_WINDOWS:
        out[f"worst_{window}h_mwh"] = rolling_worst(unmatched, window)
    # The deepest-tail metric in the paper: how long the portfolio can stay more
    # than 30% short without interruption.
    out["longest_run_above_30pct"] = longest_run(
        unmatched > 0.30 * np.maximum(load, 1e-12))
    return out


def main() -> None:
    args = parse_args()
    years = tuple(sorted(set(args.year))) if args.year else REPLAY_YEARS
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.validation_root.mkdir(parents=True, exist_ok=True)

    design = pd.read_csv(args.design_file)
    if args.design_column not in design.columns:
        raise SystemExit(f"{args.design_column} not in {args.design_file}")
    mixes = design.set_index("logical_site_id")[args.design_column]

    sites = pd.read_csv(CROSSWALK / "facility_to_era5_grid_with_country.csv",
                        low_memory=False)
    sites = sites[sites.logical_site_id.isin(mixes.index)].sort_values(
        "logical_site_id").reset_index(drop=True)
    site_ids = pd.Index(sites.logical_site_id)
    site_mix = mixes.reindex(site_ids).to_numpy()
    groups = sites.groupby("supply_country_area_code").indices

    cells = ("HH_vH", "HF_vH", "HL_vH") if args.supply_only else CELLS
    records: list[dict] = []
    site_rows: list[pd.DataFrame] = []

    for year in years:
        started = datetime.now(UTC)
        demand = {"H": load_demand(DEMAND_HIST, year, site_ids)}
        if not args.supply_only:
            demand["F"] = load_demand(args.demand_future, year, site_ids)
        hist = country_cf(SUPPLY_HIST, year)
        future = country_cf(args.supply_future, year)
        supply = {"H": hist, "F": future, "L": level_only_supply(hist, future)}
        code_row = pd.Series(range(len(hist[0])), index=hist[0])

        per_cell: dict[str, dict[str, np.ndarray]] = {
            cell: {} for cell in cells}
        for cell in cells:
            # "FH_vF" -> future demand, historical supply, volume sized on future
            d_label, s_label, volume = cell[0], cell[1], cell[4]
            codes, pv_cf, wd_cf = supply[s_label]
            block: dict[str, np.ndarray] = {}
            for code, positions in groups.items():
                if code not in code_row.index:
                    continue
                row = code_row[code]
                sub = demand[d_label][positions]
                procured = demand[volume][positions].sum(axis=1, dtype=np.float64)
                pool_mix = site_mix[positions]
                for mix in np.unique(pool_mix[~np.isnan(pool_mix)]):
                    which = pool_mix == mix
                    shape = portfolio_shape(pv_cf[row], wd_cf[row], float(mix))
                    values = evaluate(sub[which], shape, procured[which])
                    for key, array in values.items():
                        store = block.setdefault(
                            key, np.full(len(site_ids), np.nan))
                        store[positions[which]] = array
            per_cell[cell] = block
            # Energies are additive across facilities; a longest uncovered run is
            # not, so it aggregates as a demand-weighted mean -- the same
            # estimator the manuscript uses for every non-additive quantity.
            totals = {k: float(np.nansum(v)) for k, v in block.items()
                      if k != "longest_run_above_30pct"}
            weight = block["demand_mwh"]
            totals["longest_run_above_30pct"] = float(
                np.nansum(block["longest_run_above_30pct"] * weight)
                / np.nansum(weight))
            records.append({"model": args.model, "weather_year": year, "cell": cell,
                            "cfe": totals["matched_mwh"] / totals["demand_mwh"],
                            **totals})
            print(f"  {year} {cell}: CFE {records[-1]['cfe']:.6f}", flush=True)

        frame = pd.DataFrame({"logical_site_id": site_ids,
                              "supply_country_area_code": sites.supply_country_area_code,
                              "continent": sites.continent,
                              "weather_year": year})
        for cell in cells:
            for key, array in per_cell[cell].items():
                frame[f"{cell}__{key}"] = array
        site_rows.append(frame)
        print(f"  {year} complete in "
              f"{(datetime.now(UTC) - started).total_seconds():.0f}s", flush=True)

    cell_frame = pd.DataFrame(records)
    cell_frame.to_csv(args.output_root / "cells.csv", index=False)
    site_panel = pd.concat(site_rows, ignore_index=True)
    site_panel.to_parquet(args.output_root / "site_cells.parquet", index=False)

    # ------------------------------------------------------------ channels ---
    metrics = ["cfe", "unmatched_mwh", "worst_24h_mwh", "worst_168h_mwh",
               "longest_run_above_30pct"]
    channel_rows = []
    for year, block in cell_frame.groupby("weather_year"):
        cell = block.set_index("cell")
        for metric in metrics:
            base = float(cell.at["HH_vH", metric])
            row = {"model": args.model, "weather_year": year, "metric": metric,
                   "baseline": base,
                   "supply_total": float(cell.at["HF_vH", metric]) - base,
                   "supply_level": float(cell.at["HL_vH", metric]) - base}
            row["supply_shape"] = row["supply_total"] - row["supply_level"]
            if not args.supply_only:
                row["demand_total_frozen_volume"] = float(cell.at["FH_vH", metric]) - base
                row["demand_shape"] = float(cell.at["FH_vF", metric]) - base
                row["demand_level"] = (row["demand_total_frozen_volume"]
                                       - row["demand_shape"])
                row["joint_frozen_volume"] = float(cell.at["FF_vH", metric]) - base
                row["joint_adaptive_volume"] = float(cell.at["FF_vF", metric]) - base
                row["interaction_frozen_volume"] = (
                    row["joint_frozen_volume"] - row["demand_total_frozen_volume"]
                    - row["supply_total"])
            channel_rows.append(row)
    channels = pd.DataFrame(channel_rows)
    channels.to_csv(args.output_root / "level_shape_channels.csv", index=False)

    level_residual = float(channels[channels.metric.eq("cfe")].supply_level.abs().max())
    payload = {
        "schema_version": "1.0",
        "method_version": METHOD_VERSION,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "model": args.model,
        "design_file": str(args.design_file),
        "design_column": args.design_column,
        "demand_weighted_wind_share": float(np.nansum(
            site_mix * design.set_index("logical_site_id")
            .mean_annual_demand_mwh.reindex(site_ids).to_numpy())
            / float(design.mean_annual_demand_mwh.sum())),
        "weather_years": [int(v) for v in years],
        "cells_evaluated": list(cells),
        "sites": int(len(site_ids)),
        "supply_only": bool(args.supply_only),
        "twenty_year_means_pp": {
            metric: {
                key: float(channels[channels.metric.eq(metric)][key].mean()
                           * (100 if metric == "cfe" else 1))
                for key in channels.columns
                if key not in ("model", "weather_year", "metric")}
            for metric in metrics},
        "hard_checks": {
            # The identity the paper leans on: a pure change in generation LEVEL
            # cannot move the score, because the portfolio shape is normalised by
            # each technology's own annual energy.
            "supply_level_effect_is_zero": bool(level_residual < 1e-12),
            "max_abs_supply_level_effect_on_cfe": level_residual,
            "cfe_within_unit_interval": bool(
                cell_frame.cfe.between(0, 1).all()),
            "no_missing_sites": bool(site_panel.filter(like="HH_vH__").notna()
                                     .all().all()),
        },
    }
    payload["all_hard_checks_pass"] = all(
        v for k, v in payload["hard_checks"].items() if isinstance(v, bool))
    (args.validation_root / "level_shape_decomposition.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), "utf-8")
    print(json.dumps({"twenty_year_means_pp": payload["twenty_year_means_pp"]["cfe"],
                      "hard_checks": payload["hard_checks"]}, indent=2))


if __name__ == "__main__":
    main()
