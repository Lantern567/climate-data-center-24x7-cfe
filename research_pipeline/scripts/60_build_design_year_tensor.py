# -*- coding: utf-8 -*-
"""Per-site CFE metrics for every (weather year, wind share) pair -- the object
that makes the design-year x evaluation-year experiment cheap.

## Why a tensor rather than 400 runs

Result 1 of the manuscript asks three questions that all look like separate
experiments:

    1. does the optimal procurement mix depend on which weather year designed it?
    2. does a mix that is optimal in year d stay good in year e?
    3. does designing across K years converge, and how fast?

Naively that is 20 designs x 20 evaluations x several design-set sizes, i.e.
hundreds of passes over 35 GB of hourly demand.  But the matching kernel only
ever sees the pair (evaluation year, mix): the *design* year enters solely
through which mix is selected.  So one pass that scores every mix on every year,

    C[year, mix, site]

answers all three by slicing.  The design-year optimum is argmax over the mix
axis at the design year; the 20x20 matrix is a gather; a K-year design is an
argmax of the mean (or of the minimum) over a subset of the year axis.  Nothing
downstream re-reads the demand.

## Scope

S3 (same grid region) only, because that is the boundary every climate result in
the paper uses.  The mix grid, the portfolio-shape normalisation and the tail
metrics are byte-for-byte the ones script 47 uses, so the diagonal of the 20x20
matrix must reproduce the published year-2000 ladder to floating point.  That is
asserted, not assumed -- see script 61.

## Numerics

Demand is carried as float32 (halves the 585 MB/year memory traffic) but every
reduction accumulates in float64: an 8,760-term float32 sum loses about four
digits, which is more than the 0.35-percentage-point signal this paper reports.
Rolling-window gaps use a float64 cumulative sum rather than np.convolve, which
is what script 47 uses and which costs O(n*w) instead of O(n).
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
SUPPLY = PROCESSED / "wind_solar_supply_20y_fixed_2025" / "replay_results_pv_corrected"
DEMAND_ROOT = Path(r"D:\facility_demand_replay_1995_2014")
DEFAULT_OUTPUT = PROCESSED / "weather_robust_design"
DEFAULT_VALIDATION = ROOT / "output" / "24x7_cfe_facility" / "weather_robust_design"

METHOD_VERSION = "design_year_tensor_v1"
REPLAY_YEARS = tuple(range(1995, 2015))
MIX_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 4)
GAP_WINDOWS = (24, 168)
DEMAND_COLUMN = "global_reconciled_allocation_facility_load_mw"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--year", type=int, action="append")
    p.add_argument("--demand-root", type=Path, default=DEMAND_ROOT)
    p.add_argument("--supply-root", type=Path, default=SUPPLY)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--validation-root", type=Path, default=DEFAULT_VALIDATION)
    p.add_argument("--tag", default="historical",
                   help="Name of the tensor written, e.g. 'historical' or "
                        "'future_ACCESS-CM2'.")
    return p.parse_args()


def calendar(year: int) -> pd.DatetimeIndex:
    return pd.date_range(f"{year}-01-01", f"{year + 1}-01-01", freq="h",
                         inclusive="left", tz="UTC")


def country_cf(root: Path, year: int) -> tuple[pd.Index, np.ndarray, np.ndarray]:
    """Country fleet capacity factors for one weather year (script 47 verbatim)."""
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


def portfolio_shape(pv_cf: np.ndarray, wd_cf: np.ndarray, mix: float) -> np.ndarray:
    """Unit-energy portfolio shape p_t.

    Both technologies are normalised by their OWN annual capacity-factor sum, so
    any constant rescaling of either series -- a bias correction, a capacity
    revision, a uniform climate-driven change in resource level -- cancels
    exactly.  Only the shape of generation can reach the score.  This is the
    analytic basis for the "supply level effect is identically zero" statement
    in the manuscript, and it is verified numerically by script 62.
    """
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
    """Largest uncovered energy in any rolling window, per site.

    Cumulative-sum differencing, not convolution: script 47's np.convolve costs
    O(hours * window) per site, which for the 168-hour window is two orders of
    magnitude more work than is needed and is why that script is slow.  The two
    agree to floating point.
    """
    cumulative = np.empty((unmatched.shape[0], unmatched.shape[1] + 1), np.float64)
    cumulative[:, 0] = 0.0
    np.cumsum(unmatched, axis=1, dtype=np.float64, out=cumulative[:, 1:])
    return (cumulative[:, window:] - cumulative[:, :-window]).max(axis=1)


def site_block_order(handle: pq.ParquetFile, hours: int) -> pd.Index:
    """Order in which sites appear in the file, without materialising 73M strings.

    The replay writes each site as one contiguous run of `hours` rows, already
    time-ordered, but a run may straddle a row-group boundary.  Scanning the id
    column row group by row group and run-length encoding it costs a few MB,
    whereas `sort_values` on the full 73-million-row frame needs ~4 GB and dies
    on a machine that is also holding the tensor.
    """
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
                counts[-1] += int(length)      # run continues across the boundary
            else:
                order.append(str(site))
                counts.append(int(length))
    if set(counts) != {hours}:
        bad = {s: c for s, c in zip(order, counts) if c != hours}
        raise SystemExit(f"{len(bad)} sites do not carry exactly {hours} hours")
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


def main() -> None:
    args = parse_args()
    years = tuple(sorted(set(args.year))) if args.year else REPLAY_YEARS
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.validation_root.mkdir(parents=True, exist_ok=True)

    sites = pd.read_csv(CROSSWALK / "facility_to_era5_grid_with_country.csv",
                        low_memory=False)
    sites = sites[sites.supply_country_area_code.notna()].sort_values(
        "logical_site_id").reset_index(drop=True)
    site_ids = pd.Index(sites.logical_site_id)
    n_site, n_mix, n_year = len(sites), len(MIX_GRID), len(years)
    groups = sites.groupby("supply_country_area_code").indices

    matched = np.full((n_year, n_mix, n_site), np.nan)
    worst = {w: np.full((n_year, n_mix, n_site), np.nan) for w in GAP_WINDOWS}
    demand_mwh = np.full((n_year, n_site), np.nan)
    covered_sites: set[int] = set()

    for y, year in enumerate(years):
        started = datetime.now(UTC)
        load = load_demand(args.demand_root, year, site_ids)
        codes, pv_country, wd_country = country_cf(args.supply_root, year)
        code_row = pd.Series(range(len(codes)), index=codes)
        demand_mwh[y] = load.sum(axis=1, dtype=np.float64)

        for code, positions in groups.items():
            if code not in code_row.index:
                continue          # country with data centres but no wind/solar fleet
            covered_sites.update(positions.tolist())
            sub = load[positions]
            scale = demand_mwh[y, positions]
            pv_cf, wd_cf = pv_country[code_row[code]], wd_country[code_row[code]]
            for m, mix in enumerate(MIX_GRID):
                shape = portfolio_shape(pv_cf, wd_cf, float(mix))
                generation = scale[:, None] * shape[None, :]
                matched[y, m, positions] = np.minimum(sub, generation).sum(
                    axis=1, dtype=np.float64)
                unmatched = np.maximum(sub - generation, 0.0)
                for window in GAP_WINDOWS:
                    worst[window][y, m, positions] = rolling_worst(unmatched, window)
                del generation, unmatched
        del load
        elapsed = (datetime.now(UTC) - started).total_seconds()
        served = np.nansum(matched[y, 0]) / np.nansum(demand_mwh[y])
        print(f"  {year} done in {elapsed:6.1f}s  (all-solar CFE {served:.4f})",
              flush=True)

    keep = np.array(sorted(covered_sites))
    payload_sites = sites.iloc[keep]
    # Atomic publish: a 100 MB savez takes long enough that a concurrent reader
    # can otherwise open a half-written zip (see the project note on write races).
    # np.savez appends ".npz" to a *path* whose suffix differs, so it is handed an
    # open file object instead -- otherwise the temporary lands on ".partial.npz"
    # and the rename below cannot find it.
    target = args.output_root / f"design_tensor_{args.tag}.npz"
    staging = target.with_suffix(".npz.partial")
    with staging.open("wb") as handle:
        np.savez_compressed(
            handle,
            years=np.asarray(years), mix_grid=MIX_GRID,
            logical_site_id=payload_sites.logical_site_id.to_numpy().astype(str),
            supply_country_area_code=payload_sites.supply_country_area_code.to_numpy().astype(str),
            continent=payload_sites.continent.to_numpy().astype(str),
            demand_mwh=demand_mwh[:, keep],
            matched_mwh=matched[:, :, keep],
            worst_24h_mwh=worst[24][:, :, keep],
            worst_168h_mwh=worst[168][:, :, keep],
        )
    staging.replace(target)

    weighted = (matched[:, :, keep].sum(axis=2)
                / demand_mwh[:, keep].sum(axis=1)[:, None])
    payload = {
        "schema_version": "1.0",
        "method_version": METHOD_VERSION,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "tag": args.tag,
        "scenario": "S3 same grid region, alpha=1, no storage",
        "weather_years": [int(v) for v in years],
        "mix_grid": [float(v) for v in MIX_GRID],
        "sites": int(len(keep)),
        "sites_dropped_no_supply_pool": int(len(sites) - len(keep)),
        "supply_pools": int(sum(1 for c in groups if c in set(
            pd.read_parquet(args.supply_root / "annual_country_supply_1995_2014.parquet")
            .country_area_code.unique()))),
        "demand_weighted_cfe_by_year_and_mix": {
            str(int(year)): {str(float(mix)): float(weighted[y, m])
                             for m, mix in enumerate(MIX_GRID)}
            for y, year in enumerate(years)},
        "hard_checks": {
            "no_missing_cells": bool(np.isfinite(matched[:, :, keep]).all()),
            "cfe_within_unit_interval": bool(
                (weighted >= 0).all() and (weighted <= 1 + 1e-9).all()),
            "matched_never_exceeds_demand": bool(
                (matched[:, :, keep] <= demand_mwh[:, keep][:, None, :] * (1 + 1e-9)).all()),
        },
    }
    payload["all_hard_checks_pass"] = all(payload["hard_checks"].values())
    (args.validation_root / f"design_tensor_{args.tag}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), "utf-8")
    print(json.dumps({k: payload[k] for k in
                      ("sites", "sites_dropped_no_supply_pool", "hard_checks",
                       "all_hard_checks_pass")}, indent=2))


if __name__ == "__main__":
    main()
