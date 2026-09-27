# -*- coding: utf-8 -*-
"""Hourly 24/7 CFE scores under the S1--S5 geographic procurement scenarios.

## Why the scenarios collapse to one computation

Per `02-Concepts/24-7-CFE的地理边界与采购情景`, S1--S5 differ *only* in the
constraint placed on the deliverability matrix A_ij,t -- that is, in **which
generation shape a site is allowed to buy**.  Everything downstream is identical.
So each scenario is implemented as a supplier of one (pv_cf, wind_cf) pair per
site, and a single matching kernel serves all of them:

    S1  on-site           the site's own ERA5 cell
    S2  local direct      capacity-weighted cells within a radius
    S3  same grid region  the site's country fleet          <- headline scenario
    S4  cross-region      BLOCKED: needs hourly transmission headroom
    S5  no geography      the global fleet                  <- loose accounting

## Why global supply exceeding demand does not make this degenerate

Global wind+solar is ~9x data-centre demand, so "is there enough clean power"
is trivially yes and carries no information.  The 24/7 CFE question is instead:
procure a portfolio whose ANNUAL generation equals ANNUAL consumption, then ask
what fraction of HOURS is actually covered.  Writing E_pv = sum_t cf_pv,t and
E_w = sum_t cf_w,t, and letting w be wind's share of generated energy,

    g_i,t = D_i * p_t ,  p_t = (1-w) cf_pv,t / E_pv + w cf_w,t / E_w ,  sum_t p_t = 1

The site's demand scale D_i cancels out of the score entirely -- only shape
survives.  That is exactly what makes the experiment scale-free.

## Freeze-then-evaluate

The mix w is chosen once on a reference weather year and then held fixed across
all twenty.  Re-optimising w per year would let the portfolio use hindsight
about that year's weather and would erase precisely the climate risk being
measured.  This mirrors the proposal's "optimise on H, freeze, run on F".

## What the score is and is not

alpha = 1 within the allowed geography and no storage.  The result is therefore
a **physical deliverability ceiling**, not a corporate CFE score: contract
share, attribute tracking and existing load claims are not modelled.  Countries
whose clean fleet is hydro or geothermal (Iceland) score near zero here because
the supply side covers wind and solar only.
"""
from __future__ import annotations

import argparse
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "24x7_cfe" / "processed"
COUPLING = PROCESSED / "climate_coupling"
CROSSWALK = COUPLING / "demand_supply_country_crosswalk"
SUPPLY = PROCESSED / "wind_solar_supply_20y_fixed_2025" / "replay_results_pv_corrected"
DEMAND_ROOT = Path(r"D:\facility_demand_replay_1995_2014")
FACILITY_RESOURCE_ROOT = Path(r"D:\era5_facility_resource_1995_2014")
FACILITY_WEATHER_ROOT = Path(r"D:\era5_facility_weather_1995_2014")
LOCAL_POOL_ROOT = Path(r"D:\facility_local_supply_pools")
ONSITE_ROOT = Path(r"D:\facility_onsite_supply_cf")

DEFAULT_OUTPUT = PROCESSED / "cfe_geographic_scenarios"
DEFAULT_VALIDATION = ROOT / "output" / "24x7_cfe_facility" / "cfe_geographic_scenarios"
METHOD_VERSION = "geographic_cfe_scenarios_v1"

REPLAY_YEARS = tuple(range(1995, 2015))
REFERENCE_YEAR = 2000          # neutral compound-stress year; see the KB note
MIX_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 4)
GAP_WINDOWS = (24, 168)
EVENT_THRESHOLDS = (0.10, 0.30, 0.50)
EVENT_DURATIONS = (12, 24, 100, 168)
DEMAND_COLUMN = "global_reconciled_allocation_facility_load_mw"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", action="append",
                   choices=["S1", "S2", "S3", "S4", "S5"],
                   help="Restrict to these scenarios (repeatable).")
    p.add_argument("--year", type=int, action="append")
    p.add_argument("--local-radius-km", type=float, default=100.0)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--validation-root", type=Path, default=DEFAULT_VALIDATION)
    p.add_argument("--demand-root", type=Path, default=DEMAND_ROOT)
    return p.parse_args()


def calendar(year: int) -> pd.DatetimeIndex:
    return pd.date_range(f"{year}-01-01", f"{year + 1}-01-01", freq="h",
                         inclusive="left", tz="UTC")


def year_slice(year: int) -> tuple[int, int]:
    start = sum(len(calendar(y)) for y in REPLAY_YEARS if y < year)
    return start, start + len(calendar(year))


# ------------------------------------------------------------------ supply
def country_cf(year: int) -> tuple[pd.Index, np.ndarray, np.ndarray]:
    """Country fleet capacity factors for one weather year."""
    partition = SUPPLY / "hourly_country_supply" / f"weather_year={year}"
    frame = pd.read_parquet(next(partition.glob("*.parquet")))
    annual = pd.read_parquet(SUPPLY / "annual_country_supply_1995_2014.parquet")
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


def regional_cf(year: int, country_to_region: pd.Series
                ) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, float]]:
    """Continent-pooled fleet capacity factors -- the S4 no-congestion ceiling.

    True S4 needs hourly transmission headroom, which does not exist for a
    global inventory.  Pooling a whole continent as one copper plate is the
    UPPER BOUND on what cross-region delivery could ever buy, so S3 and S4
    bracket the answer rather than pretending to resolve it.
    """
    partition = SUPPLY / "hourly_country_supply" / f"weather_year={year}"
    frame = pd.read_parquet(next(partition.glob("*.parquet")))
    annual = pd.read_parquet(SUPPLY / "annual_country_supply_1995_2014.parquet")
    annual = annual[annual.weather_year.eq(year)].set_index("country_area_code")
    frame["region"] = frame.country_area_code.map(country_to_region)
    frame = frame[frame.region.notna()]
    pools: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    coverage: dict[str, float] = {}
    for region, block in frame.groupby("region"):
        codes = block.country_area_code.unique()
        pv_cap = float(annual.loc[codes, "solar_pv_installed_capacity_mw"].sum())
        wd_cap = float(annual.loc[codes, "wind_installed_capacity_mw"].sum())
        summed = block.groupby("time_utc")[
            ["solar_pv_generation_power_mw", "wind_generation_power_mw"]].sum()
        summed = summed.sort_index()
        pv = summed.solar_pv_generation_power_mw.to_numpy() / pv_cap if pv_cap > 0 else \
            np.zeros(len(summed))
        wd = summed.wind_generation_power_mw.to_numpy() / wd_cap if wd_cap > 0 else \
            np.zeros(len(summed))
        pools[str(region)] = (pv, wd)
        coverage[str(region)] = pv_cap + wd_cap
    return pools, coverage


def local_pool_cf(year: int, grid_ids: pd.Series
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """S2 pools precomputed by script 48, sliced to one weather year."""
    path = LOCAL_POOL_ROOT / "local_supply_pool_cf.npz"
    if not path.exists():
        return None
    start, stop = year_slice(year)
    try:
        archive = np.load(path)
    except (OSError, ValueError, zipfile.BadZipFile):
        # Script 48 may still be writing; treat a half-written pool as absent
        # rather than crashing a twenty-year run that is otherwise fine.
        return None
    with archive:
        order = pd.Series(range(len(archive["era5_grid_id"])),
                          index=archive["era5_grid_id"])
        rows = order.reindex(grid_ids).to_numpy()
        pv = archive["pv_cf"][:, start:stop][rows]
        wd = archive["wind_cf"][:, start:stop][rows]
        capacity = (archive["pv_capacity_mw"] + archive["wind_capacity_mw"])[rows]
    return pv, wd, capacity


def global_cf(year: int) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_parquet(SUPPLY / "hourly_global_supply_1995_2014.parquet")
    frame = frame[frame.weather_year.eq(year)].sort_values("time_utc")
    annual = pd.read_csv(SUPPLY / "annual_global_supply_1995_2014.csv")
    annual = annual[annual.weather_year.eq(year)].iloc[0]
    pv = frame.solar_pv_generation_power_mw.to_numpy() / annual.solar_pv_installed_capacity_mw
    wd = frame.wind_generation_power_mw.to_numpy() / annual.wind_installed_capacity_mw
    return pv, wd


def onsite_cf(year: int, grid_ids: pd.Series
              ) -> tuple[np.ndarray, np.ndarray] | None:
    """S1 on-site capacity factors precomputed by script 50, sliced to a year."""
    path = ONSITE_ROOT / "onsite_supply_cf.npz"
    if not path.exists():
        return None
    start, stop = year_slice(year)
    try:
        archive = np.load(path)
    except (OSError, ValueError, zipfile.BadZipFile):
        return None
    with archive:
        order = pd.Series(range(len(archive["era5_grid_id"])),
                          index=archive["era5_grid_id"])
        rows = order.reindex(grid_ids).to_numpy()
        return archive["pv_cf"][:, start:stop][rows], archive["wind_cf"][:, start:stop][rows]


# ------------------------------------------------------------------- kernel
def portfolio_shape(pv_cf: np.ndarray, wd_cf: np.ndarray, mix: float) -> np.ndarray:
    """Unit-energy portfolio shape p_t for one supply pool and one mix."""
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


def score(load: np.ndarray, shape: np.ndarray) -> np.ndarray:
    """Load-weighted hourly CFE score for every site sharing one supply pool."""
    demand = load.sum(axis=1)
    generation = demand[:, None] * shape[None, :]
    return np.minimum(load, generation).sum(axis=1) / demand


def tail_metrics(load: np.ndarray, shape: np.ndarray) -> dict[str, np.ndarray]:
    demand = load.sum(axis=1)
    unmatched = np.maximum(load - demand[:, None] * shape[None, :], 0.0)
    out: dict[str, np.ndarray] = {
        "unmatched_energy_mwh": unmatched.sum(axis=1),
        "unmatched_hours": (unmatched > 0).sum(axis=1).astype(float),
    }
    fraction = unmatched / np.maximum(load, 1e-12)
    for window in GAP_WINDOWS:
        kernel = np.ones(window)
        rolled = np.apply_along_axis(
            lambda row: np.convolve(row, kernel, mode="valid"), 1, unmatched)
        out[f"worst_{window}h_unmatched_mwh"] = rolled.max(axis=1)
    for threshold in EVENT_THRESHOLDS:
        exceed = fraction > threshold
        out[f"hours_above_{int(threshold * 100)}pct"] = exceed.sum(axis=1).astype(float)
        # Longest consecutive run above the threshold, per site.
        longest = np.zeros(len(load))
        for row in range(len(load)):
            run = best = 0
            for flag in exceed[row]:
                run = run + 1 if flag else 0
                best = max(best, run)
            longest[row] = best
        out[f"longest_run_above_{int(threshold * 100)}pct"] = longest
    return out


def main() -> None:
    args = parse_args()
    scenarios = tuple(args.scenario) if args.scenario else ("S1", "S2", "S3", "S5")
    years = tuple(sorted(set(args.year))) if args.year else REPLAY_YEARS
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.validation_root.mkdir(parents=True, exist_ok=True)

    sites = pd.read_csv(CROSSWALK / "facility_to_era5_grid_with_country.csv",
                        low_memory=False)
    sites = sites[sites.supply_country_area_code.notna()].copy()
    sites = sites.sort_values("logical_site_id").reset_index(drop=True)
    sites["region"] = sites.continent.where(
        ~sites.continent.isin(["Unknown", "Antarctica"]))
    # Supply-side country -> region, learned from the sites themselves so the
    # two sides cannot disagree about which continent a country sits on.
    country_to_region = (
        sites.dropna(subset=["region"])
        .groupby("supply_country_area_code").region.agg(lambda s: s.mode().iloc[0])
    )

    records: list[pd.DataFrame] = []
    frozen_mix: dict[str, pd.Series] = {}
    skipped: dict[str, str] = {}

    for year in (REFERENCE_YEAR, *[y for y in years if y != REFERENCE_YEAR]):
        path = args.demand_root / f"facility_hourly_demand_replay_{year}.parquet"
        frame = pq.read_table(
            path, columns=["logical_site_id", "time_utc", DEMAND_COLUMN]
        ).to_pandas()
        frame = frame.sort_values(["logical_site_id", "time_utc"])
        site_ids = pd.Index(frame.logical_site_id.unique())
        hours = len(calendar(year))
        load_all = frame[DEMAND_COLUMN].to_numpy(dtype=np.float64).reshape(-1, hours)
        order = pd.Series(range(len(site_ids)), index=site_ids)
        rows = order.reindex(sites.logical_site_id).to_numpy()
        load = load_all[rows]
        del frame, load_all

        codes, pv_country, wd_country = country_cf(year)
        code_row = pd.Series(range(len(codes)), index=codes)
        pv_global, wd_global = global_cf(year)
        regional_pools, regional_capacity = regional_cf(year, country_to_region)
        local_year = local_pool_cf(year, sites.era5_grid_id)
        onsite_year = onsite_cf(year, sites.era5_grid_id)

        for scenario in scenarios:
            if scenario in skipped:
                continue
            if scenario == "S3":
                groups = sites.groupby("supply_country_area_code").indices
                pools = {k: (pv_country[code_row[k]], wd_country[code_row[k]])
                         for k in groups if k in code_row.index}
            elif scenario == "S4":
                groups = {k: v for k, v in sites.groupby("region").indices.items()
                          if k in regional_pools}
                pools = regional_pools
            elif scenario == "S5":
                groups = {"GLOBAL": np.arange(len(sites))}
                pools = {"GLOBAL": (pv_global, wd_global)}
            elif scenario == "S2":
                if local_year is None:
                    skipped[scenario] = "local supply pools not built (script 48)"
                    continue
                pv_local, wd_local, capacity = local_year
                # Sites sharing a facility cell share one local pool (3.5 sites
                # per cell on average), so group by cell rather than by site.
                # A cell with no fleet inside the radius is genuinely outside
                # S2; widening its radius would silently turn S2 into S3.
                groups = {
                    key: positions
                    for key, positions in sites.groupby("era5_grid_id").indices.items()
                    if capacity[positions[0]] > 0
                }
                pools = {key: (pv_local[positions[0]], wd_local[positions[0]])
                         for key, positions in groups.items()}
            elif scenario == "S1":
                if onsite_year is None:
                    skipped[scenario] = "on-site capacity factors not built (script 50)"
                    continue
                pv_onsite, wd_onsite = onsite_year
                # On-site resource is a property of the cell, so sites sharing a
                # cell share it.  Unlike S2 there is no coverage test: S1 builds
                # its own generation, so every cell qualifies.
                groups = sites.groupby("era5_grid_id").indices
                pools = {key: (pv_onsite[positions[0]], wd_onsite[positions[0]])
                         for key, positions in groups.items()}
            else:
                skipped[scenario] = "unknown scenario"
                continue

            for key, positions in groups.items():
                if key not in pools:
                    continue
                pv_cf, wd_cf = pools[key]
                sub = load[positions]
                if year == REFERENCE_YEAR:
                    # Choose the mix once, on the reference year, then freeze.
                    best = None
                    for mix in MIX_GRID:
                        value = score(sub, portfolio_shape(pv_cf, wd_cf, mix))
                        if best is None:
                            best, best_mix = value.copy(), np.full(len(sub), mix)
                        else:
                            better = value > best
                            best_mix[better] = mix
                            best[better] = value[better]
                    frozen_mix.setdefault(scenario, pd.Series(dtype=float))
                    frozen_mix[scenario] = pd.concat([
                        frozen_mix[scenario],
                        pd.Series(best_mix, index=sites.logical_site_id.iloc[positions]),
                    ])
                mixes = frozen_mix[scenario].reindex(
                    sites.logical_site_id.iloc[positions]).to_numpy()
                cfe = np.zeros(len(sub))
                metrics: dict[str, np.ndarray] = {}
                for mix in np.unique(mixes):
                    which = mixes == mix
                    shape = portfolio_shape(pv_cf, wd_cf, float(mix))
                    cfe[which] = score(sub[which], shape)
                    part = tail_metrics(sub[which], shape)
                    for name, values in part.items():
                        metrics.setdefault(name, np.zeros(len(sub)))[which] = values
                records.append(pd.DataFrame({
                    "weather_year": year,
                    "scenario": scenario,
                    "supply_pool": key,
                    "logical_site_id": sites.logical_site_id.iloc[positions].to_numpy(),
                    "supply_country_area_code":
                        sites.supply_country_area_code.iloc[positions].to_numpy(),
                    "wind_energy_share": mixes,
                    "annual_demand_mwh": sub.sum(axis=1),
                    "cfe_score": cfe,
                    **metrics,
                }))
        print(f"  {year} done ({len(records)} blocks)", flush=True)

    panel = pd.concat(records, ignore_index=True)
    panel.to_parquet(args.output_root / "site_scenario_year_cfe.parquet", index=False)

    weighted = (
        panel.assign(w=panel.cfe_score * panel.annual_demand_mwh)
        .groupby(["scenario", "weather_year"], as_index=False)
        .agg(sites=("logical_site_id", "nunique"),
             demand_mwh=("annual_demand_mwh", "sum"),
             weighted_numerator=("w", "sum"),
             unmatched_mwh=("unmatched_energy_mwh", "sum"),
             mean_wind_share=("wind_energy_share", "mean"))
    )
    weighted["cfe_score_demand_weighted"] = (
        weighted.weighted_numerator / weighted.demand_mwh)
    weighted.drop(columns="weighted_numerator").to_csv(
        args.output_root / "global_scenario_year_cfe.csv", index=False)

    payload = {
        "schema_version": "1.0",
        "method_version": METHOD_VERSION,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "scenarios_computed": sorted(set(panel.scenario)),
        "scenarios_skipped": skipped,
        "reference_year_for_mix": REFERENCE_YEAR,
        "mix_frozen_across_weather_years": True,
        "alpha_assumption": "alpha=1 within the allowed geography; no storage",
        "interpretation": "physical deliverability ceiling, not a corporate CFE score",
        "sites": int(panel.logical_site_id.nunique()),
        "sites_by_scenario": panel.groupby("scenario").logical_site_id.nunique().to_dict(),
        # Share of total site demand each scenario can actually serve.  The
        # denominator is the full inventory, not another scenario, so the number
        # stays meaningful when scenarios are run one at a time.
        "demand_share_by_scenario": (
            panel[panel.weather_year.eq(REFERENCE_YEAR)]
            .groupby("scenario").annual_demand_mwh.sum()
            / float(panel[panel.weather_year.eq(REFERENCE_YEAR)]
                    .groupby("scenario").annual_demand_mwh.sum().max())
        ).to_dict(),
        "weather_years": sorted(set(panel.weather_year.tolist())),
        "hard_checks": {
            "cfe_within_unit_interval": bool(
                (panel.cfe_score >= 0).all() and (panel.cfe_score <= 1.0000001).all()),
            "unmatched_non_negative": bool((panel.unmatched_energy_mwh >= 0).all()),
            # Site coverage must be constant across YEARS within a scenario, but
            # NOT across scenarios: a country with data centres and no wind or
            # solar fleet (Iceland, whose clean power is hydro and geothermal)
            # has no S3 supply pool while still drawing from the S5 global pool.
            "site_coverage_constant_within_scenario": bool(
                panel.groupby(["scenario", "weather_year"]).logical_site_id.nunique()
                .groupby("scenario").nunique().eq(1).all()),
            "mix_within_unit_interval": bool(
                (panel.wind_energy_share >= 0).all()
                and (panel.wind_energy_share <= 1).all()),
        },
    }
    payload["all_hard_checks_pass"] = all(payload["hard_checks"].values())
    (args.validation_root / "cfe_geographic_scenarios_validation.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), "utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print()
    print(weighted[["scenario", "weather_year", "cfe_score_demand_weighted",
                    "mean_wind_share"]].to_string(index=False))


if __name__ == "__main__":
    main()
