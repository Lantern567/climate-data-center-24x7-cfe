# -*- coding: utf-8 -*-
"""Least-cost wind/solar/storage portfolios for a given 24/7 CFE target.

This is the step the geographic ladder could not answer.  Scripts 47--49 ask
"what does an annual-energy-matched portfolio actually cover hour by hour?".
Here the question is inverted to the one the proposal's section 6.5 poses:
**what does it cost to reach 90 / 95 / 99 / 100 percent hourly coverage, and
what capacity mix gets you there?**

Its output x_H* is also the object stage IV needs: the portfolio optimised on
historical weather, frozen, and then run under future climate to measure
maladaptation loss J_F(x_H*) - J_F(x_F*).  Without it that headline number
cannot be computed no matter how much CMIP6 data arrives.

## Formulation (one linear program per market)

Decision variables: PV capacity, wind capacity, and power/energy ratings for a
short-duration battery and a long-duration store; per hour, each store's charge,
discharge and state of charge, plus unserved load and curtailment.

    min  sum_tech annuity_tech * capacity_tech
    s.t. generation_t + discharge_t - charge_t + unserved_t - curtail_t = load_t
         soc_{t+1} = soc_t + eff_in * charge_t - discharge_t / eff_out
         soc_t <= energy_rating,  charge_t/discharge_t <= power_rating
         sum_t unserved_t <= (1 - target) * sum_t load_t

Storage is cyclic (soc at the end equals soc at the start) so a portfolio cannot
score by quietly draining a free reservoir over the horizon.

## Why both storage durations

The 24/7 CFE literature's sharpest result is that cost rises non-linearly as the
target approaches 100 percent, and that non-linearity lives in *long* events:
short batteries arbitrage day/night, long stores ride out multi-day wind-solar
droughts.  Section 3 of the ladder analysis already localised the risk there --
the twenty-year CV of the longest sub-30-percent run is 8-48 percent, far above
any mean metric -- so a 4-hour-only model would price the 100 percent target
wrongly by construction.

## Cost provenance

Wind, PV and battery costs come from the NREL ATB 2024 v3 table already in the
repository (Moderate scenario, 2030).  ATB quotes batteries as $/kW of a fixed
duration, which conflates power and energy; the 4-hour and 8-hour rows solve for
the split exactly:

    P + 4E = 1451.3     P + 8E = 2470.0   ->   E = 254.7 $/kWh,  P = 432.6 $/kW

Long-duration storage has no ATB row and is an explicit assumption, flagged in
the output rather than buried.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, hstack, vstack, eye, diags


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "24x7_cfe" / "processed"
COSTS = PROCESSED / "costs" / "atb_2024_v3_costs_compact.csv"
CROSSWALK = PROCESSED / "climate_coupling" / "demand_supply_country_crosswalk"
SUPPLY = PROCESSED / "wind_solar_supply_20y_fixed_2025" / "replay_results_pv_corrected"
DEMAND_ROOT = Path(r"D:\facility_demand_replay_1995_2014")
DEFAULT_OUTPUT = PROCESSED / "cfe_portfolio_optimisation"
DEFAULT_VALIDATION = ROOT / "output" / "24x7_cfe_facility" / "cfe_portfolio_optimisation"

METHOD_VERSION = "cfe_portfolio_lp_v1"
DISCOUNT_RATE = 0.07
TARGETS = (0.90, 0.95, 0.99, 1.00)
REFERENCE_YEAR = 2000
DEMAND_COLUMN = "global_reconciled_allocation_facility_load_mw"

# No ATB row exists for multi-day storage.  These are explicit, documented
# assumptions in the iron-air / flow-battery range, not fitted values.
LDS_POWER_USD_PER_KW = 1000.0
LDS_ENERGY_USD_PER_KWH = 20.0
LDS_ROUND_TRIP = 0.50
LDS_LIFE_YEARS = 25


@dataclass(frozen=True)
class Tech:
    name: str
    capex_power_usd_per_kw: float
    capex_energy_usd_per_kwh: float
    fixed_om_usd_per_kw_year: float
    life_years: int
    round_trip: float | None


def crf(rate: float, years: int) -> float:
    return rate * (1 + rate) ** years / ((1 + rate) ** years - 1)


def load_technologies() -> dict[str, Tech]:
    table = pd.read_csv(COSTS)
    pick = lambda key: table[(table.technology_key == key)
                             & (table.scenario == "Moderate")
                             & (table.year == 2030)].iloc[0]
    wind, pv = pick("landbased_wind"), pick("utility_pv")
    b4, b8 = pick("battery_4h"), pick("battery_8h")
    # Solve the power/energy split from the two published durations.
    energy = (b8.capex_2022usd_per_kw - b4.capex_2022usd_per_kw) / 4.0
    power = b4.capex_2022usd_per_kw - 4.0 * energy
    return {
        "wind": Tech("wind", float(wind.capex_2022usd_per_kw), 0.0,
                     float(wind.fixed_om_2022usd_per_kw_year),
                     int(wind.technical_life_years), None),
        "pv": Tech("pv", float(pv.capex_2022usd_per_kw), 0.0,
                   float(pv.fixed_om_2022usd_per_kw_year),
                   int(pv.technical_life_years), None),
        "battery": Tech("battery", float(power), float(energy),
                        float(b4.fixed_om_2022usd_per_kw_year),
                        int(b4.technical_life_years),
                        float(b4.round_trip_efficiency_fraction)),
        "lds": Tech("lds", LDS_POWER_USD_PER_KW, LDS_ENERGY_USD_PER_KWH,
                    0.0, LDS_LIFE_YEARS, LDS_ROUND_TRIP),
    }


def annuities(techs: dict[str, Tech]) -> dict[str, float]:
    """Annualised cost per MW (power) or MWh (energy), in 2022 USD."""
    out: dict[str, float] = {}
    for name, tech in techs.items():
        factor = crf(DISCOUNT_RATE, tech.life_years)
        out[f"{name}_power"] = (tech.capex_power_usd_per_kw * factor
                                + tech.fixed_om_usd_per_kw_year) * 1000.0
        out[f"{name}_energy"] = tech.capex_energy_usd_per_kwh * factor * 1000.0
    return out


def solve_market(load: np.ndarray, pv_cf: np.ndarray, wind_cf: np.ndarray,
                 target: float, cost: dict[str, float],
                 techs: dict[str, Tech]) -> dict | None:
    """One market, one target.  Returns None if the LP is infeasible."""
    n = len(load)
    stores = ("battery", "lds")
    # Column layout: [K_pv, K_wind, P_b, E_b, P_l, E_l, then per store
    # charge(n), discharge(n), soc(n), then unserved(n), curtail(n)]
    n_cap = 6
    per_store = 3 * n
    total = n_cap + len(stores) * per_store + 2 * n

    def block(offset: int) -> slice:
        return slice(offset, offset + n)

    charge = {s: n_cap + i * per_store for i, s in enumerate(stores)}
    discharge = {s: charge[s] + n for s in stores}
    soc = {s: charge[s] + 2 * n for s in stores}
    unserved = n_cap + len(stores) * per_store
    curtail = unserved + n

    objective = np.zeros(total)
    objective[0] = cost["pv_power"]
    objective[1] = cost["wind_power"]
    objective[2] = cost["battery_power"]
    objective[3] = cost["battery_energy"]
    objective[4] = cost["lds_power"]
    objective[5] = cost["lds_energy"]

    eye_n = eye(n, format="csr")
    zero_cap = csr_matrix((n, n_cap))

    # --- hourly balance -----------------------------------------------------
    cap_cols = csr_matrix(np.column_stack([pv_cf, wind_cf, np.zeros((n, 4))]))
    pieces = [cap_cols]
    for s in stores:
        pieces.extend([-eye_n, eye_n, csr_matrix((n, n))])   # -charge +discharge
    pieces.extend([eye_n, -eye_n])                            # +unserved -curtail
    balance = hstack(pieces, format="csr")

    # --- storage dynamics: soc_t - soc_{t-1} - eff*chg_t + dis_t/eff = 0 -----
    shift = diags([np.ones(n - 1)], [-1], shape=(n, n), format="csr")
    wrap = csr_matrix(([1.0], ([0], [n - 1])), shape=(n, n))   # cyclic closure
    dyn_rows, dyn_rhs = [], []
    for s in stores:
        eff = np.sqrt(techs[s].round_trip)
        cols = [zero_cap]
        for other in stores:
            if other == s:
                cols.extend([-eff * eye_n, eye_n / eff, eye_n - shift - wrap])
            else:
                cols.extend([csr_matrix((n, n))] * 3)
        cols.extend([csr_matrix((n, n)), csr_matrix((n, n))])
        dyn_rows.append(hstack(cols, format="csr"))
        dyn_rhs.append(np.zeros(n))

    # --- capacity limits: chg <= P, dis <= P, soc <= E ----------------------
    limit_rows = []
    for index, s in enumerate(stores):
        power_col, energy_col = 2 + 2 * index, 3 + 2 * index
        for var_offset, cap_col in ((charge[s], power_col),
                                    (discharge[s], power_col),
                                    (soc[s], energy_col)):
            cap = csr_matrix((-np.ones(n), (np.arange(n), np.full(n, cap_col))),
                             shape=(n, n_cap))
            body = csr_matrix((n, total - n_cap)).tolil()
            body[:, var_offset - n_cap: var_offset - n_cap + n] = eye_n
            limit_rows.append(hstack([cap, csr_matrix(body)], format="csr"))

    # --- CFE target ---------------------------------------------------------
    target_row = np.zeros(total)
    target_row[block(unserved)] = 1.0
    target_rhs = (1.0 - target) * load.sum()

    result = linprog(
        objective,
        A_ub=vstack(limit_rows + [csr_matrix(target_row)], format="csr"),
        b_ub=np.concatenate([np.zeros(n * 3 * len(stores)), [target_rhs]]),
        A_eq=vstack([balance] + dyn_rows, format="csr"),
        b_eq=np.concatenate([load] + dyn_rhs),
        bounds=[(0, None)] * total,
        method="highs",
    )
    if not result.success:
        return None
    x = result.x
    served = load.sum() - x[block(unserved)].sum()
    return {
        "annual_cost_usd": float(result.fun),
        "pv_mw": float(x[0]), "wind_mw": float(x[1]),
        "battery_mw": float(x[2]), "battery_mwh": float(x[3]),
        "lds_mw": float(x[4]), "lds_mwh": float(x[5]),
        "unserved_mwh": float(x[block(unserved)].sum()),
        "curtailed_mwh": float(x[block(curtail)].sum()),
        "achieved_cfe": float(served / load.sum()),
        "lcoe_usd_per_mwh": float(result.fun / served),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--country", action="append")
    p.add_argument("--year", type=int, default=REFERENCE_YEAR)
    p.add_argument("--target", type=float, action="append")
    p.add_argument("--min-demand-share", type=float, default=0.002)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--validation-root", type=Path, default=DEFAULT_VALIDATION)
    args = p.parse_args()
    targets = tuple(args.target) if args.target else TARGETS
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.validation_root.mkdir(parents=True, exist_ok=True)

    techs = load_technologies()
    cost = annuities(techs)

    sites = pd.read_csv(CROSSWALK / "facility_to_era5_grid_with_country.csv",
                        low_memory=False)
    sites = sites[sites.supply_country_area_code.notna()]
    demand = pd.read_parquet(
        DEMAND_ROOT / f"facility_hourly_demand_replay_{args.year}.parquet",
        columns=["logical_site_id", "time_utc", DEMAND_COLUMN])
    demand = demand.merge(sites[["logical_site_id", "supply_country_area_code"]],
                          on="logical_site_id")
    country_load = (demand.groupby(["supply_country_area_code", "time_utc"])
                    [DEMAND_COLUMN].sum().unstack(0))

    partition = SUPPLY / "hourly_country_supply" / f"weather_year={args.year}"
    supply = pd.read_parquet(next(partition.glob("*.parquet")))
    annual = pd.read_parquet(SUPPLY / "annual_country_supply_1995_2014.parquet")
    annual = annual[annual.weather_year.eq(args.year)].set_index("country_area_code")

    totals = country_load.sum()
    keep = totals[totals >= totals.sum() * args.min_demand_share].index
    if args.country:
        keep = [c for c in args.country if c in country_load.columns]

    records = []
    for code in keep:
        block = supply[supply.country_area_code.eq(code)].sort_values("time_utc")
        if block.empty or code not in annual.index:
            continue
        pv_cap = float(annual.at[code, "solar_pv_installed_capacity_mw"])
        wd_cap = float(annual.at[code, "wind_installed_capacity_mw"])
        pv_cf = (block.solar_pv_generation_power_mw.to_numpy() / pv_cap
                 if pv_cap > 0 else np.zeros(len(block)))
        wind_cf = (block.wind_generation_power_mw.to_numpy() / wd_cap
                   if wd_cap > 0 else np.zeros(len(block)))
        load = country_load[code].to_numpy()
        for target in targets:
            solution = solve_market(load, pv_cf, wind_cf, target, cost, techs)
            row = {"country_area_code": code, "weather_year": args.year,
                   "target": target, "annual_demand_mwh": float(load.sum()),
                   "feasible": solution is not None}
            if solution:
                row.update(solution)
            records.append(row)
            status = (f"CFE {solution['achieved_cfe']:.4f} "
                      f"LCOE {solution['lcoe_usd_per_mwh']:.1f} $/MWh"
                      if solution else "INFEASIBLE")
            print(f"  {code} target {target:.0%}: {status}", flush=True)

    frame = pd.DataFrame(records)
    frame.to_csv(args.output_root / f"portfolio_{args.year}.csv", index=False)
    payload = {
        "schema_version": "1.0",
        "method_version": METHOD_VERSION,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "weather_year": args.year,
        "targets": list(targets),
        "markets": int(frame.country_area_code.nunique()) if len(frame) else 0,
        "infeasible_cases": int((~frame.feasible).sum()) if len(frame) else 0,
        "discount_rate": DISCOUNT_RATE,
        "annuities_usd_per_mw_or_mwh_year": cost,
        "technologies": {k: asdict(v) for k, v in techs.items()},
        "long_duration_storage_is_assumption": True,
        "storage_is_cyclic": True,
        "interpretation": "x_H*: optimised on historical weather, to be frozen "
                          "and stress-tested under future climate",
    }
    (args.validation_root / f"portfolio_validation_{args.year}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), "utf-8")
    print(json.dumps({k: payload[k] for k in
                      ("markets", "infeasible_cases", "targets")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
