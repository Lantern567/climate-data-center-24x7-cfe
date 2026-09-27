# -*- coding: utf-8 -*-
"""Result 4: what the residual climate exposure costs, in firming rather than in
percentage points.

A change of 0.35 percentage points of hourly coverage is hard to act on.  The
same change expressed as "the extra firm clean capacity, or the extra storage,
that a buyer must hold to end up where they started" is a procurement decision.
This script converts the climate effect into three such quantities, all defined
against one target: **restore the site's historical hourly coverage**.

    A  firm capacity      phi, a constant-output carbon-free block sized as a
                          share of mean load, added to the frozen portfolio
    B  storage energy     E, in hours of mean load; unlimited power, round-trip
                          efficiency from the ATB battery row, store starts EMPTY
                          so the metric cannot be gamed by draining a free
                          reservoir over the year
    C  worst-week firm    the increase in the largest 168-hour uncovered energy,
       energy             read directly rather than simulated: that energy is by
                          definition what firm resources must deliver in the
                          worst week

A and B are the same requirement priced two ways -- energy versus flexibility --
and they answer the two halves of the manuscript's level/shape distinction.  C is
a tail statistic and needs no dispatch assumption at all, which is why it is
reported alongside: it is the one number here that is free of any storage or
firm-resource modelling choice.

Both A and B are solved per facility by vectorised bisection.  Coverage is
monotone non-decreasing in phi and in E, so bisection is exact to the bracket
width; the bracket is asserted to contain the root rather than assumed to.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def _load_geography_module():
    """Reuse script 66's procurement-pool builder rather than restating it.

    Result 4 has to be priced under the same five geographies Result 1 reports,
    and the S1/S2 grid pools, S4 continental pools and S5 global pool are already
    defined there.  The module name starts with a digit, so it is loaded by path.
    """
    path = Path(__file__).with_name("66_compute_geographic_climate_conditioning.py")
    spec = importlib.util.spec_from_file_location("geo66", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GEO = _load_geography_module()


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "24x7_cfe" / "processed"
CROSSWALK = PROCESSED / "climate_coupling" / "demand_supply_country_crosswalk"
SUPPLY_ROOT = PROCESSED / "wind_solar_supply_20y_fixed_2025"
SUPPLY_HIST = SUPPLY_ROOT / "replay_results_pv_corrected"
COSTS = PROCESSED / "costs" / "atb_2024_v3_costs_compact.csv"
DESIGN = PROCESSED / "weather_robust_design"
DEMAND_HIST = Path(r"D:\facility_demand_replay_1995_2014")

METHOD_VERSION = "firming_requirement_v1"
DEMAND_COLUMN = "global_reconciled_allocation_facility_load_mw"
REPLAY_YEARS = tuple(range(1995, 2015))
PHI_MAX = 1.0                 # a firm block equal to mean load: coverage -> 1
STORAGE_MAX_HOURS = 400.0     # ~17 days of mean load
BISECTION_STEPS = 34


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
    p.add_argument("--storage-steps", type=int, default=22)
    p.add_argument("--scenario", default="S3", choices=["S1", "S2", "S3", "S4", "S5"],
                   help="Procurement geography. S3 reproduces the published run.")
    p.add_argument("--mix-from-ladder", action="store_true",
                   help="Use the scenario ladder's frozen wind share even for S3, "
                        "so all five geographies share one mix convention.")
    p.add_argument("--skip-storage", action="store_true",
                   help="Solve firm capacity only. The residual-risk premium in "
                        "Result 4 never uses the storage leg, and its dispatch "
                        "loop dominates the runtime.")
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--validation-root", type=Path, default=None)
    args = p.parse_args()
    suffix = "" if args.scenario == "S3" else f"_{args.scenario}"
    if args.output_root is None:
        args.output_root = PROCESSED / f"firming_{args.model}{suffix}"
    if args.validation_root is None:
        args.validation_root = (ROOT / "output" / "24x7_cfe_facility"
                                / f"firming_{args.model}{suffix}")
    return args


def calendar(year: int) -> pd.DatetimeIndex:
    return pd.date_range(f"{year}-01-01", f"{year + 1}-01-01", freq="h",
                         inclusive="left", tz="UTC")


def round_trip_efficiency() -> float:
    table = pd.read_csv(COSTS)
    row = table[(table.technology_key == "battery_4h")
                & (table.scenario == "Moderate") & (table.year == 2030)].iloc[0]
    return float(row.round_trip_efficiency_fraction)


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


# --------------------------------------------------------------- requirements
def coverage_with_firm(load: np.ndarray, generation: np.ndarray,
                       phi: np.ndarray, mean_load: np.ndarray,
                       chunk: int = 1024) -> np.ndarray:
    """Coverage once a constant firm block of phi x mean load is added.

    Evaluated in facility chunks: the intermediate `generation + firm` for the
    whole inventory is a 585 MB float64 array, and the bisection would allocate
    two of them on every step.
    """
    out = np.empty(load.shape[0])
    for start in range(0, load.shape[0], chunk):
        stop = min(start + chunk, load.shape[0])
        block = generation[start:stop] + (phi[start:stop]
                                          * mean_load[start:stop])[:, None]
        np.minimum(block, load[start:stop], out=block)
        out[start:stop] = (block.sum(axis=1, dtype=np.float64)
                           / load[start:stop].sum(axis=1, dtype=np.float64))
    return out


def storage_recovery(surplus: np.ndarray, deficit: np.ndarray,
                     capacity: np.ndarray, efficiency: float) -> np.ndarray:
    """Energy a lossy, unlimited-power store recovers under greedy dispatch.

    Greedy is optimal for maximising served energy with a single store and no
    power limit, and starting empty is what forbids the metric from being gamed:
    a store initialised full would hand the portfolio free energy it never
    procured.

    The hour loop is the expensive part, so surplus and deficit are computed once
    by the caller and every facility is stepped together.  An earlier version
    called this inside the per-country, per-mix loop and re-derived surplus and
    deficit on every bisection step, which turned one weather year into roughly
    1.4e8 Python-level iterations and never finished.
    """
    leg = np.sqrt(efficiency)
    soc = np.zeros(surplus.shape[0])
    recovered = np.zeros(surplus.shape[0])
    for t in range(surplus.shape[1]):
        soc += np.minimum(surplus[:, t] * leg, capacity - soc)
        released = np.minimum(soc * leg, deficit[:, t])
        soc -= released / leg
        recovered += released
    return recovered


def bisect(evaluate, target: np.ndarray, high: np.ndarray,
           steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Smallest x in [0, high] with evaluate(x) >= target, per site."""
    low = np.zeros_like(high)
    reachable = evaluate(high) >= target - 1e-12
    for _ in range(steps):
        mid = 0.5 * (low + high)
        ok = evaluate(mid) >= target
        high = np.where(ok, mid, high)
        low = np.where(ok, low, mid)
    return high, reachable


def main() -> None:
    args = parse_args()
    years = tuple(sorted(set(args.year))) if args.year else REPLAY_YEARS
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.validation_root.mkdir(parents=True, exist_ok=True)
    efficiency = round_trip_efficiency()

    if args.scenario == "S3" and not args.mix_from_ladder:
        design = pd.read_csv(args.design_file)
        mixes = design.set_index("logical_site_id")[args.design_column]
    else:
        # Each procurement geography has its own optimised wind share; using the
        # S3 design outside S3 would price a portfolio nobody would have bought.
        ladder = pd.read_parquet(
            GEO.LADDER,
            columns=["scenario", "weather_year", "logical_site_id", "wind_energy_share"])
        ladder = ladder[ladder.scenario.eq(args.scenario)
                        & ladder.weather_year.eq(GEO.REFERENCE_YEAR)]
        mixes = ladder.drop_duplicates("logical_site_id").set_index(
            "logical_site_id")["wind_energy_share"]

    sites = pd.read_csv(CROSSWALK / "facility_to_era5_grid_with_country.csv",
                        low_memory=False)
    sites = sites[sites.logical_site_id.isin(mixes.index)].sort_values(
        "logical_site_id").reset_index(drop=True)
    sites["region"] = sites.continent.where(~sites.continent.isin(["Unknown", "Antarctica"]))
    country_to_region = (sites.dropna(subset=["region"])
                         .groupby("supply_country_area_code").region
                         .agg(lambda values: values.mode().iloc[0]))
    site_ids = pd.Index(sites.logical_site_id)
    site_mix = mixes.reindex(site_ids).to_numpy()

    large_pool = None
    if args.scenario == "S1":
        large_pool = GEO.load_npz_pair(GEO.ONSITE_H, GEO.ONSITE_F,
                                       "onsite_supply_cf.npz",
                                       GEO.DEFAULT_CACHE, "S1_onsite")
    elif args.scenario == "S2":
        large_pool = GEO.load_npz_pair(GEO.LOCAL_H, GEO.LOCAL_F,
                                       "local_supply_pool_cf.npz",
                                       GEO.DEFAULT_CACHE, "S2_local_100km")

    rows: list[pd.DataFrame] = []
    for year in years:
        started = datetime.now(UTC)
        hours = len(calendar(year))
        load_h = load_demand(DEMAND_HIST, year, site_ids)
        load_f = load_demand(args.demand_future, year, site_ids)
        groups, pools = GEO.pool_spec(args.scenario, year, sites,
                                      country_to_region, large_pool)

        n = len(site_ids)
        out = {name: np.full(n, np.nan) for name in
               ("coverage_historical", "coverage_future", "firm_share_of_mean_load",
                "storage_hours_of_mean_load", "worst_168h_historical_mwh",
                "worst_168h_future_mwh", "demand_historical_mwh",
                "demand_future_mwh", "firm_reachable", "storage_reachable")}

        # Pass one: assemble the two generation profiles for every facility.  The
        # portfolio shape is a property of (country, mix), so it is still built
        # per group, but the expensive per-facility work below is then done once
        # for the whole inventory rather than once per group.
        # Generation is carried in float64 so that the historical coverage this
        # script computes is bit-comparable with the four-cell run in script 62;
        # the loads stay float32 to keep the working set inside memory.
        gen_h = np.zeros((n, hours), dtype=np.float64)
        gen_f = np.zeros((n, hours), dtype=np.float64)
        covered = np.zeros(n, dtype=bool)
        for code, positions in groups.items():
            positions = np.asarray(positions)
            pv_h, wd_h, pv_f, wd_f = pools[code]
            pool_mix = site_mix[positions]
            for mix in np.unique(pool_mix[~np.isnan(pool_mix)]):
                which = positions[pool_mix == mix]
                # Frozen procurement volume: bought against historical demand,
                # operated against future demand -- the convention every headline
                # number in the manuscript uses.
                procured = load_h[which].sum(axis=1, dtype=np.float64)
                gen_h[which] = (procured[:, None]
                                * portfolio_shape(pv_h, wd_h,
                                                  float(mix))[None, :])
                gen_f[which] = (procured[:, None]
                                * portfolio_shape(pv_f, wd_f,
                                                  float(mix))[None, :])
                covered[which] = True

        demand_h = load_h.sum(axis=1, dtype=np.float64)
        demand_f = load_f.sum(axis=1, dtype=np.float64)
        mean_load_f = demand_f / hours
        target = np.minimum(load_h, gen_h).sum(axis=1, dtype=np.float64) / demand_h
        future = np.minimum(load_f, gen_f).sum(axis=1, dtype=np.float64) / demand_f

        phi, phi_ok = bisect(
            lambda x: coverage_with_firm(load_f, gen_f, x, mean_load_f),
            target, np.full(n, PHI_MAX), BISECTION_STEPS)

        # Surplus and deficit do not depend on the storage size, so they are
        # built once and reused by all bisection steps.  float32 is ample here:
        # they feed a dispatch whose answer is reported to two decimal places.
        if args.skip_storage:
            store = np.full(n, np.nan)
            store_ok = np.ones(n, dtype=bool)
        else:
            surplus = np.maximum(gen_f - load_f, 0.0).astype(np.float32)
            deficit = np.maximum(load_f - gen_f, 0.0).astype(np.float32)
            matched_f = np.minimum(load_f, gen_f).sum(axis=1, dtype=np.float64)
            store, store_ok = bisect(
                lambda x: (matched_f + storage_recovery(
                    surplus, deficit, x * mean_load_f, efficiency)) / demand_f,
                target, np.full(n, STORAGE_MAX_HOURS), args.storage_steps)
            del surplus, deficit

        worst = {}
        for label, gap in (("historical", np.maximum(load_h - gen_h, 0.0)),
                           ("future", np.maximum(load_f - gen_f, 0.0))):
            cumulative = np.empty((n, hours + 1), np.float64)
            cumulative[:, 0] = 0.0
            np.cumsum(gap, axis=1, dtype=np.float64, out=cumulative[:, 1:])
            worst[label] = (cumulative[:, 168:] - cumulative[:, :-168]).max(axis=1)
            del cumulative, gap

        keep = np.flatnonzero(covered)
        out["coverage_historical"][keep] = target[keep]
        out["coverage_future"][keep] = future[keep]
        out["firm_share_of_mean_load"][keep] = phi[keep]
        out["storage_hours_of_mean_load"][keep] = store[keep]
        out["worst_168h_historical_mwh"][keep] = worst["historical"][keep]
        out["worst_168h_future_mwh"][keep] = worst["future"][keep]
        out["demand_historical_mwh"][keep] = demand_h[keep]
        out["demand_future_mwh"][keep] = demand_f[keep]
        out["firm_reachable"][keep] = phi_ok[keep].astype(float)
        out["storage_reachable"][keep] = store_ok[keep].astype(float)
        del gen_h, gen_f, load_h, load_f

        frame = pd.DataFrame({"model": args.model, "weather_year": year,
                              "logical_site_id": site_ids,
                              "supply_country_area_code": sites.supply_country_area_code,
                              "continent": sites.continent, **out})
        rows.append(frame)
        print(f"  {year} done in {(datetime.now(UTC) - started).total_seconds():.0f}s "
              f"(mean firm {np.nanmean(out['firm_share_of_mean_load']) * 100:.3f}% "
              f"of mean load)", flush=True)

    panel = pd.concat(rows, ignore_index=True)
    panel.to_parquet(args.output_root / "site_firming.parquet", index=False)

    weight = panel.demand_future_mwh
    rows_summary = []
    for year, b in panel.groupby("weather_year"):
        w = b.demand_future_mwh
        rows_summary.append({
            "weather_year": int(year),
            "coverage_historical": (b.coverage_historical * w).sum() / w.sum(),
            "coverage_future": (b.coverage_future * w).sum() / w.sum(),
            "firm_share_of_mean_load": (b.firm_share_of_mean_load * w).sum() / w.sum(),
            "storage_hours_of_mean_load": (b.storage_hours_of_mean_load * w).sum() / w.sum(),
            "additional_firm_energy_twh": (
                b.firm_share_of_mean_load * b.demand_future_mwh).sum() / 1e6,
            # A store rated in hours of mean load: convert to energy with that
            # year's own hour count, which differs in leap years.
            "additional_storage_energy_gwh": (
                b.storage_hours_of_mean_load * b.demand_future_mwh
                / len(calendar(int(year)))).sum() / 1e3,
            "worst_168h_increase_mwh": (
                b.worst_168h_future_mwh - b.worst_168h_historical_mwh).sum(),
            "worst_168h_increase_pct": (
                b.worst_168h_future_mwh.sum()
                / b.worst_168h_historical_mwh.sum() - 1) * 100,
        })
    summary = pd.DataFrame(rows_summary).set_index("weather_year")
    summary.to_csv(args.output_root / "global_firming_by_year.csv")

    total_demand = float(panel.demand_future_mwh.sum())
    rows_region = []
    for code, b in panel.groupby("supply_country_area_code"):
        w = b.demand_future_mwh
        rows_region.append({
            "supply_country_area_code": code,
            "demand_share": w.sum() / total_demand,
            "firm_share_of_mean_load": (b.firm_share_of_mean_load * w).sum() / w.sum(),
            "storage_hours_of_mean_load": (b.storage_hours_of_mean_load * w).sum() / w.sum(),
            "coverage_change_pp": ((b.coverage_future - b.coverage_historical) * w).sum()
            / w.sum() * 100,
        })
    region = (pd.DataFrame(rows_region)
              .sort_values("firm_share_of_mean_load", ascending=False))
    region.to_csv(args.output_root / "region_firming.csv", index=False)
    del weight

    payload = {
        "schema_version": "1.0",
        "method_version": METHOD_VERSION,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "model": args.model,
        "scenario_geography": args.scenario,
        "mix_source": ("design file" if args.scenario == "S3"
                       else f"scenario ladder {args.scenario}"),
        "storage_solved": not args.skip_storage,
        "design_column": args.design_column,
        "weather_years": [int(v) for v in years],
        "sites": int(panel.logical_site_id.nunique()),
        "round_trip_efficiency": efficiency,
        "storage_starts_empty": True,
        "storage_power_unconstrained": True,
        "procurement_volume": "frozen at historical annual demand",
        "target": "restore each facility's historical hourly coverage",
        "twenty_year_means": {
            "firm_share_of_mean_load_pct": float(summary.firm_share_of_mean_load.mean() * 100),
            "storage_hours_of_mean_load": (
                None if args.skip_storage
                else float(summary.storage_hours_of_mean_load.mean())),
            "additional_firm_energy_twh": float(summary.additional_firm_energy_twh.mean()),
            "additional_storage_energy_gwh": (
                None if args.skip_storage
                else float(summary.additional_storage_energy_gwh.mean())),
            "worst_168h_increase_pct": float(summary.worst_168h_increase_pct.mean()),
            "coverage_gap_pp": float((summary.coverage_historical
                                      - summary.coverage_future).mean() * 100),
        },
        "hard_checks": {
            "firm_bracket_sufficient": bool(panel.firm_reachable.min() == 1.0),
            "storage_bracket_sufficient": bool(panel.storage_reachable.min() == 1.0),
            "no_missing_sites": bool(panel.firm_share_of_mean_load.notna().all()),
            "firm_non_negative": bool((panel.firm_share_of_mean_load >= 0).all()),
        },
    }
    payload["all_hard_checks_pass"] = all(payload["hard_checks"].values())
    (args.validation_root / "firming_requirement.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), "utf-8")
    print(json.dumps({"twenty_year_means": payload["twenty_year_means"],
                      "hard_checks": payload["hard_checks"]}, indent=2))


if __name__ == "__main__":
    main()
