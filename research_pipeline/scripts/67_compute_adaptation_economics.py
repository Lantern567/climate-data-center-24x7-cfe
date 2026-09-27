# -*- coding: utf-8 -*-
"""Price the climate-restoration boundaries used in Results 3 and 4.

The calculation has two deliberately separate layers.

1. The global procurement--storage boundary is priced with the NREL ATB 2024
   Moderate 2030 wind, PV and lithium-ion battery rows. Renewable procurement
   is valued at the demand-weighted wind/PV energy mix used in the climate
   experiment. Battery power and energy costs are separated from the ATB 4 h
   and 8 h rows, then recombined at a 4 h power-to-energy ratio. The resulting
   values are annualised 2022 USD, not market prices.
2. Country resilience is expressed with a common clean-firm planning proxy:
   ATB Moderate 2030 hydrothermal-flash geothermal. The annual-only capacity
   restores historical annual hourly coverage. The resilience-inclusive lower
   bound is the larger of that capacity and the constant capacity needed to
   deliver the climate-induced increase in worst-168-h uncovered energy.

All costs use a 7% real discount rate, matching the portfolio-optimisation
module. Tail-restoration results remain lower-bound planning estimates because
the underlying event calculation assumes a full store at event onset and does
not enforce chronological storage power constraints.
"""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "24x7_cfe" / "processed"
ADAPT = PROCESSED / "climate_adaptation_ACCESS-CM2"
COSTS = PROCESSED / "costs" / "atb_2024_v3_costs_compact.csv"
DESIGN = PROCESSED / "weather_robust_design" / "site_portfolio_designs.csv"
FIRMING = PROCESSED / "firming_ACCESS-CM2" / "site_firming.parquet"
VALIDATION = ROOT / "output" / "24x7_cfe_facility" / "climate_adaptation_ACCESS-CM2"

DISCOUNT_RATE = 0.07
ATB_SCENARIO = "Moderate"
ATB_YEAR = 2030
BATTERY_DURATION_H = 4.0
WORST_WEEK_H = 168.0

# Delivered-cost basis for the five procurement geographies.  Storage sits at
# the load and pays nothing here; procured wind and solar energy has to travel,
# and pricing it at one ATB benchmark regardless of distance would make an
# 8,900 km global portfolio as cheap as an on-site one.  S1 is behind the
# meter.  S2 and S3 move energy across networks that already exist, so they pay
# an industrial transmission tariff rather than a new line.  S4 and S5 cross
# interconnections that do not exist today and are charged a new-build HVDC
# annuity over the demand-weighted mean procurement distance measured in
# tmp/procurement_distance.csv.
PROCUREMENT_DISTANCE_KM = {"S1": 0.0, "S2": 65.0, "S3": 1257.0,
                           "S4": 1864.0, "S5": 8917.0}
NETWORK_TARIFF_USD_PER_MWH = {"S1": 0.0, "S2": 5.0, "S3": 10.0}
HVDC_LINE_USD_PER_KW_KM = 0.43
HVDC_CONVERTER_USD_PER_KW = 160.0
HVDC_LINE_UTILISATION = 0.35
HVDC_LIFE_YEARS = 30

OUT = ADAPT
ARCHETYPES = ADAPT / "regional_archetypes.csv"
SCENARIO = "S3"
PRICE_TRANSMISSION = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", default="S3",
                   choices=["S1", "S2", "S3", "S4", "S5"])
    p.add_argument("--price-transmission", action="store_true",
                   help="Charge procured renewable energy for delivery. Off by "
                        "default so the published S3 run reproduces exactly.")
    p.add_argument("--adapt-root", type=Path, default=None)
    p.add_argument("--output-root", type=Path, default=None)
    # Country classification is deliberately held on the S3 demand base so
    # the same 38 major markets are compared under every geography.
    p.add_argument("--archetypes", type=Path,
                   default=PROCESSED / "climate_adaptation_ACCESS-CM2"
                           / "regional_archetypes.csv")
    p.add_argument("--firming", type=Path, default=None)
    p.add_argument("--validation-root", type=Path, default=None)
    return p.parse_args()


def transmission_adder(scenario: str, priced: bool) -> tuple[float, dict]:
    """Return the $/MWh delivery adder on procured renewable energy."""
    km = PROCUREMENT_DISTANCE_KM[scenario]
    if not priced:
        return 0.0, {"basis": "not_priced", "distance_km": km, "usd_per_mwh": 0.0}
    if scenario in NETWORK_TARIFF_USD_PER_MWH:
        value = NETWORK_TARIFF_USD_PER_MWH[scenario]
        basis = "behind_the_meter" if km == 0 else "existing_network_tariff"
        return value, {"basis": basis, "distance_km": km, "usd_per_mwh": value}
    capex_per_kw = HVDC_LINE_USD_PER_KW_KM * km + HVDC_CONVERTER_USD_PER_KW
    annuity_per_kw_year = capex_per_kw * crf(DISCOUNT_RATE, HVDC_LIFE_YEARS)
    value = annuity_per_kw_year / (8.76 * HVDC_LINE_UTILISATION)
    return value, {"basis": "new_build_hvdc", "distance_km": km,
                   "capex_usd_per_kw": capex_per_kw,
                   "annuity_usd_per_kw_year": annuity_per_kw_year,
                   "line_utilisation": HVDC_LINE_UTILISATION,
                   "usd_per_mwh": value}

TARGET_FIELDS = {
    "annual": "annual_coverage_restored_fraction",
    "24h": "worst_24h_restored_fraction",
    "168h": "worst_168h_restored_fraction",
}


def crf(rate: float, years: float) -> float:
    growth = (1.0 + rate) ** years
    return rate * growth / (growth - 1.0)


def pick(table: pd.DataFrame, technology: str) -> pd.Series:
    block = table[
        table.technology_key.eq(technology)
        & table.scenario.eq(ATB_SCENARIO)
        & table.year.eq(ATB_YEAR)
    ]
    if len(block) != 1:
        raise RuntimeError(f"Expected one ATB row for {technology}, found {len(block)}")
    return block.iloc[0]


def annual_power_cost(row: pd.Series) -> float:
    """Annualised 2022 USD per MW-year."""
    return float(
        (row.capex_2022usd_per_kw * crf(DISCOUNT_RATE, row.technical_life_years)
         + row.fixed_om_2022usd_per_kw_year) * 1000.0
    )


def renewable_lcoe(row: pd.Series) -> float:
    """Annualised 2022 USD per generated MWh."""
    return annual_power_cost(row) / (8760.0 * float(row.capacity_factor_fraction))


def restoration_intercept(surface: pd.DataFrame, field: str) -> float:
    zero = surface[np.isclose(surface.storage_hours_mean_load, 0.0)].sort_values(
        "additional_procurement_pct"
    )
    x = zero.additional_procurement_pct.to_numpy(dtype=float)
    value = zero[field].to_numpy(dtype=float)
    for index in range(len(x) - 1):
        if (value[index] - 1.0) * (value[index + 1] - 1.0) <= 0:
            if np.isclose(value[index + 1], value[index]):
                return float(x[index])
            return float(
                x[index]
                + (1.0 - value[index]) * (x[index + 1] - x[index])
                / (value[index + 1] - value[index])
            )
    raise RuntimeError(f"Restoration target not crossed for {field}")


def restoration_frontier(
    targets: pd.DataFrame, surface: pd.DataFrame, target: str, field: str
) -> pd.DataFrame:
    intercept = restoration_intercept(surface, field)
    block = targets[targets.target.eq(target)].copy()
    block = block[
        block.additional_procurement_pct.lt(intercept)
        & block.storage_hours_mean_load.gt(1e-8)
        & block.reachable_within_grid
    ]
    endpoint = pd.DataFrame({
        "target": [target],
        "additional_procurement_pct": [intercept],
        "storage_hours_mean_load": [0.0],
        "reachable_within_grid": [True],
    })
    return pd.concat([block, endpoint], ignore_index=True).sort_values(
        "additional_procurement_pct"
    )


def interpolate_crossing(
    x: np.ndarray, y: np.ndarray, historical: float, higher_is_better: bool
) -> float:
    reached = y >= historical if higher_is_better else y <= historical
    if reached[0]:
        return float(x[0])
    if not reached.any():
        return float("nan")
    index = int(np.flatnonzero(reached)[0])
    x0, x1 = x[index - 1], x[index]
    y0, y1 = y[index - 1], y[index]
    if np.isclose(y1, y0):
        return float(x1)
    return float(x0 + (historical - y0) * (x1 - x0) / (y1 - y0))


def price_weather_years(
    procurement_lcoe: float, battery_4h_annuity_per_mwh: float
) -> pd.DataFrame:
    surface = pd.read_csv(ADAPT / "adaptation_surface_by_year.csv")
    metric_spec = {
        "annual": ("coverage", "historical_coverage", True),
        "24h": ("worst_24h_mwh", "historical_worst_24h_mwh", False),
        "168h": ("worst_168h_mwh", "historical_worst_168h_mwh", False),
    }
    rows: list[dict] = []
    for weather_year, year_frame in surface.groupby("weather_year", observed=True):
        baseline = year_frame[
            np.isclose(year_frame.additional_procurement_pct, 0.0)
            & np.isclose(year_frame.storage_hours_mean_load, 0.0)
        ].iloc[0]
        hours = 8784.0 if int(weather_year) % 4 == 0 else 8760.0
        historical_demand_mwh = float(
            baseline.historical_uncovered_mwh / (1.0 - baseline.historical_coverage)
        )
        future_demand_mwh = float(baseline.uncovered_mwh / (1.0 - baseline.coverage))
        future_mean_load_mw = future_demand_mwh / hours

        for target, (metric, historical_field, higher_is_better) in metric_spec.items():
            historical = float(baseline[historical_field])
            frontier_rows: list[dict] = []
            for procurement, block in year_frame.groupby(
                "additional_procurement_pct", observed=True
            ):
                block = block.sort_values("storage_hours_mean_load")
                storage = interpolate_crossing(
                    block.storage_hours_mean_load.to_numpy(dtype=float),
                    block[metric].to_numpy(dtype=float),
                    historical,
                    higher_is_better,
                )
                if np.isfinite(storage):
                    frontier_rows.append({
                        "additional_procurement_pct": float(procurement),
                        "storage_hours_mean_load": storage,
                    })
            frontier = pd.DataFrame(frontier_rows).sort_values(
                "additional_procurement_pct"
            )
            if frontier.empty:
                continue
            reached_without_storage = frontier.storage_hours_mean_load.le(1e-8)
            if reached_without_storage.any():
                index = int(np.flatnonzero(reached_without_storage.to_numpy())[0])
                if index == 0:
                    procurement_intercept = 0.0
                else:
                    left = frontier.iloc[index - 1]
                    right = frontier.iloc[index]
                    procurement_intercept = float(
                        left.additional_procurement_pct
                        + (0.0 - left.storage_hours_mean_load)
                        * (right.additional_procurement_pct
                           - left.additional_procurement_pct)
                        / (right.storage_hours_mean_load
                           - left.storage_hours_mean_load)
                    )
                frontier = frontier[
                    frontier.additional_procurement_pct.lt(procurement_intercept)
                ]
                frontier = pd.concat([
                    frontier,
                    pd.DataFrame([{
                        "additional_procurement_pct": procurement_intercept,
                        "storage_hours_mean_load": 0.0,
                    }]),
                ], ignore_index=True)

            frontier["procurement_cost_billion_usd_per_year"] = (
                frontier.additional_procurement_pct / 100.0
                * historical_demand_mwh * procurement_lcoe / 1e9
            )
            frontier["storage_cost_billion_usd_per_year"] = (
                frontier.storage_hours_mean_load * future_mean_load_mw
                * battery_4h_annuity_per_mwh / 1e9
            )
            frontier["total_cost_billion_usd_per_year"] = (
                frontier.procurement_cost_billion_usd_per_year
                + frontier.storage_cost_billion_usd_per_year
            )
            optimum = frontier.loc[frontier.total_cost_billion_usd_per_year.idxmin()]
            storage_only = frontier.iloc[0]
            procurement_only = frontier.iloc[-1]
            rows.append({
                "weather_year": int(weather_year),
                "restoration_target": target,
                "least_cost_billion_usd_per_year": float(
                    optimum.total_cost_billion_usd_per_year
                ),
                "least_cost_procurement_pct": float(optimum.additional_procurement_pct),
                "least_cost_storage_hours_mean_load": float(
                    optimum.storage_hours_mean_load
                ),
                "procurement_only_pct": float(
                    procurement_only.additional_procurement_pct
                ),
                "procurement_only_cost_billion_usd_per_year": float(
                    procurement_only.total_cost_billion_usd_per_year
                ),
                "storage_only_hours_mean_load": float(
                    storage_only.storage_hours_mean_load
                ),
                "storage_only_cost_billion_usd_per_year": float(
                    storage_only.total_cost_billion_usd_per_year
                ),
            })
    result = pd.DataFrame(rows)
    annual = result[result.restoration_target.eq("annual")].set_index(
        "weather_year"
    ).least_cost_billion_usd_per_year
    result["resilience_premium_vs_annual_billion_usd_per_year"] = (
        result.least_cost_billion_usd_per_year
        - result.weather_year.map(annual)
    )
    return result


def price_global_frontiers(cost_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    design = pd.read_csv(DESIGN)
    surface = pd.read_csv(ADAPT / "adaptation_surface_mean.csv")
    targets = pd.read_csv(ADAPT / "storage_restoration_targets.csv")

    wind = pick(cost_table, "landbased_wind")
    pv = pick(cost_table, "utility_pv")
    battery_4h = pick(cost_table, "battery_4h")
    battery_8h = pick(cost_table, "battery_8h")

    wind_share = float(np.average(
        design.wind_share_robust_mean_20y,
        weights=design.mean_annual_demand_mwh,
    ))
    busbar_lcoe = (
        wind_share * renewable_lcoe(wind)
        + (1.0 - wind_share) * renewable_lcoe(pv)
    )
    delivery_adder, delivery_basis = transmission_adder(SCENARIO, PRICE_TRANSMISSION)
    procurement_lcoe = busbar_lcoe + delivery_adder

    # ATB fixed-duration rows obey P + duration * E. Solving the 4 h and
    # 8 h rows gives a common power and energy CAPEX split.
    battery_energy_capex = float(
        (battery_8h.capex_2022usd_per_kw - battery_4h.capex_2022usd_per_kw) / 4.0
    )
    battery_power_capex = float(
        battery_4h.capex_2022usd_per_kw - 4.0 * battery_energy_capex
    )
    battery_factor = crf(DISCOUNT_RATE, float(battery_4h.technical_life_years))
    battery_power_annuity = (
        battery_power_capex * battery_factor
        + float(battery_4h.fixed_om_2022usd_per_kw_year)
    ) * 1000.0
    battery_energy_annuity = battery_energy_capex * battery_factor * 1000.0
    battery_4h_annuity_per_mwh = (
        battery_energy_annuity + battery_power_annuity / BATTERY_DURATION_H
    )

    baseline = surface[
        np.isclose(surface.additional_procurement_pct, 0.0)
        & np.isclose(surface.storage_hours_mean_load, 0.0)
    ].iloc[0]
    future_demand_mwh = float(baseline.uncovered_mwh / (1.0 - baseline.coverage))
    future_mean_load_mw = future_demand_mwh / (365.25 * 24.0)
    historical_procurement_mwh = float(design.mean_annual_demand_mwh.sum())

    rows: list[pd.DataFrame] = []
    summaries: list[dict] = []
    for target, field in TARGET_FIELDS.items():
        frontier = restoration_frontier(targets, surface, target, field).copy()
        frontier["procurement_cost_billion_usd_per_year"] = (
            frontier.additional_procurement_pct / 100.0
            * historical_procurement_mwh * procurement_lcoe / 1e9
        )
        frontier["storage_cost_billion_usd_per_year"] = (
            frontier.storage_hours_mean_load * future_mean_load_mw
            * battery_4h_annuity_per_mwh / 1e9
        )
        frontier["total_cost_billion_usd_per_year"] = (
            frontier.procurement_cost_billion_usd_per_year
            + frontier.storage_cost_billion_usd_per_year
        )
        frontier["least_cost_point"] = False
        optimum_index = int(frontier.total_cost_billion_usd_per_year.idxmin())
        frontier.loc[optimum_index, "least_cost_point"] = True
        frontier.insert(0, "restoration_target", target)
        rows.append(frontier)

        storage_only = frontier.iloc[0]
        procurement_only = frontier.iloc[-1]
        optimum = frontier.loc[optimum_index]
        summaries.append({
            "restoration_target": target,
            "procurement_only_pct": float(procurement_only.additional_procurement_pct),
            "procurement_only_cost_billion_usd_per_year": float(
                procurement_only.total_cost_billion_usd_per_year
            ),
            "storage_only_hours_mean_load": float(storage_only.storage_hours_mean_load),
            "storage_only_cost_billion_usd_per_year": float(
                storage_only.total_cost_billion_usd_per_year
            ),
            "least_cost_procurement_pct": float(optimum.additional_procurement_pct),
            "least_cost_storage_hours_mean_load": float(optimum.storage_hours_mean_load),
            "least_cost_billion_usd_per_year": float(
                optimum.total_cost_billion_usd_per_year
            ),
        })

    frontier_frame = pd.concat(rows, ignore_index=True)
    summary = pd.DataFrame(summaries)
    annual_cost = float(summary.loc[
        summary.restoration_target.eq("annual"), "least_cost_billion_usd_per_year"
    ].iloc[0])
    summary["resilience_premium_vs_annual_billion_usd_per_year"] = (
        summary.least_cost_billion_usd_per_year - annual_cost
    )
    summary["cost_multiple_vs_annual"] = (
        summary.least_cost_billion_usd_per_year / annual_cost
    )

    weather_years = price_weather_years(procurement_lcoe, battery_4h_annuity_per_mwh)
    weather_years.to_csv(ADAPT / "adaptation_cost_by_year.csv", index=False)

    assumptions = {
        "dollar_year": 2022,
        "atb_scenario": ATB_SCENARIO,
        "atb_year": ATB_YEAR,
        "real_discount_rate": DISCOUNT_RATE,
        "global_wind_energy_share": wind_share,
        "renewable_procurement_lcoe_usd_per_mwh": procurement_lcoe,
        "renewable_busbar_lcoe_usd_per_mwh": busbar_lcoe,
        "procurement_geography": SCENARIO,
        "delivery_cost_basis": delivery_basis,
        "battery_power_capex_usd_per_kw": battery_power_capex,
        "battery_energy_capex_usd_per_kwh": battery_energy_capex,
        "battery_power_annuity_usd_per_mw_year": battery_power_annuity,
        "battery_energy_annuity_usd_per_mwh_year": battery_energy_annuity,
        "battery_duration_hours_used_for_pricing": BATTERY_DURATION_H,
        "battery_4h_annuity_usd_per_mwh_year": battery_4h_annuity_per_mwh,
        "historical_procurement_mwh": historical_procurement_mwh,
        "future_demand_mwh": future_demand_mwh,
        "future_mean_load_mw": future_mean_load_mw,
    }
    return frontier_frame, summary, assumptions


def price_country_resilience(cost_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    firming = pd.read_parquet(FIRMING)
    firming["hours_in_year"] = np.where(firming.weather_year.mod(4).eq(0), 8784, 8760)
    firming["mean_load_mw"] = firming.demand_future_mwh / firming.hours_in_year
    firming["annual_firm_capacity_mw"] = (
        firming.firm_share_of_mean_load * firming.mean_load_mw
    )
    firming["worst_168h_increase_mwh"] = (
        firming.worst_168h_future_mwh - firming.worst_168h_historical_mwh
    )

    country_year = firming.groupby(
        ["supply_country_area_code", "weather_year"], as_index=False
    ).agg(
        demand_future_mwh=("demand_future_mwh", "sum"),
        mean_load_mw=("mean_load_mw", "sum"),
        annual_firm_capacity_mw=("annual_firm_capacity_mw", "sum"),
        worst_168h_increase_mwh=("worst_168h_increase_mwh", "sum"),
    )
    country_year["worst_168h_firm_capacity_lower_bound_mw"] = (
        country_year.worst_168h_increase_mwh.clip(lower=0.0) / WORST_WEEK_H
    )
    country_year["resilience_firm_capacity_lower_bound_mw"] = np.maximum(
        country_year.annual_firm_capacity_mw,
        country_year.worst_168h_firm_capacity_lower_bound_mw,
    )
    country_year["resilience_premium_capacity_mw"] = (
        country_year.resilience_firm_capacity_lower_bound_mw
        - country_year.annual_firm_capacity_mw
    )

    country = country_year.groupby("supply_country_area_code", as_index=False).mean(
        numeric_only=True
    )
    archetypes = pd.read_csv(ARCHETYPES)
    country = country.merge(
        archetypes[["supply_country_area_code", "demand_share", "major_market"]],
        on="supply_country_area_code",
        how="left",
    )
    # Wider geographies reach countries the S3 classification never saw; they are
    # by construction not major markets.
    country["demand_share"] = country["demand_share"].fillna(0.0)
    country["major_market"] = country["major_market"].fillna(False).astype(bool)

    geothermal = pick(cost_table, "geothermal_hydroflash")
    firm_annuity = annual_power_cost(geothermal)
    for capacity_field, cost_field in (
        ("annual_firm_capacity_mw", "annual_only_cost_billion_usd_per_year"),
        ("resilience_firm_capacity_lower_bound_mw",
         "resilience_cost_lower_bound_billion_usd_per_year"),
        ("resilience_premium_capacity_mw",
         "resilience_premium_billion_usd_per_year"),
    ):
        country[cost_field] = country[capacity_field] * firm_annuity / 1e9

    country["annual_only_cost_usd_per_mean_mw_year"] = (
        country.annual_firm_capacity_mw / country.mean_load_mw * firm_annuity
    )
    country["resilience_cost_lower_bound_usd_per_mean_mw_year"] = (
        country.resilience_firm_capacity_lower_bound_mw
        / country.mean_load_mw * firm_annuity
    )
    country["resilience_premium_usd_per_mean_mw_year"] = (
        country.resilience_premium_capacity_mw / country.mean_load_mw * firm_annuity
    )
    country["resilience_cost_multiple"] = np.divide(
        country.resilience_cost_lower_bound_billion_usd_per_year,
        country.annual_only_cost_billion_usd_per_year,
        out=np.full(len(country), np.nan),
        where=country.annual_only_cost_billion_usd_per_year.gt(1e-12),
    )

    summaries = []
    for scope, block in (
        ("all_145_countries", country),
        ("major_38_countries", country[country.major_market]),
    ):
        annual = float(block.annual_only_cost_billion_usd_per_year.sum())
        resilience = float(
            block.resilience_cost_lower_bound_billion_usd_per_year.sum()
        )
        premium = float(block.resilience_premium_billion_usd_per_year.sum())
        summaries.append({
            "scope": scope,
            "countries": int(len(block)),
            "demand_share": float(block.demand_share.sum()),
            "annual_only_cost_billion_usd_per_year": annual,
            "resilience_cost_lower_bound_billion_usd_per_year": resilience,
            "resilience_premium_billion_usd_per_year": premium,
            "resilience_cost_multiple": resilience / annual,
            "resilience_premium_pct": premium / annual * 100.0,
        })
    summary = pd.DataFrame(summaries)
    assumptions = {
        "clean_firm_proxy": "NREL ATB 2024 Moderate 2030 hydrothermal-flash geothermal",
        "clean_firm_annuity_usd_per_mw_year": firm_annuity,
        "worst_week_hours": WORST_WEEK_H,
        "resilience_capacity_definition": (
            "max(annual-restoration firm capacity, positive worst-168-h "
            "uncovered-energy increase / 168 h)"
        ),
    }
    return country, summary, assumptions


def main() -> None:
    global SCENARIO, PRICE_TRANSMISSION, ADAPT, OUT, FIRMING, VALIDATION, ARCHETYPES
    args = parse_args()
    SCENARIO = args.scenario
    PRICE_TRANSMISSION = args.price_transmission
    if args.adapt_root is not None:
        ADAPT = args.adapt_root
    if args.firming is not None:
        FIRMING = args.firming
    if args.validation_root is not None:
        VALIDATION = args.validation_root
    ARCHETYPES = args.archetypes
    OUT = args.output_root if args.output_root is not None else ADAPT
    OUT.mkdir(parents=True, exist_ok=True)

    cost_table = pd.read_csv(COSTS)
    global_frontier, global_summary, global_assumptions = price_global_frontiers(
        cost_table
    )
    country_costs, country_summary, country_assumptions = price_country_resilience(
        cost_table
    )

    global_frontier.to_csv(OUT / "adaptation_cost_frontier.csv", index=False)
    global_summary.to_csv(OUT / "adaptation_cost_summary.csv", index=False)
    country_costs.to_csv(OUT / "country_resilience_costs.csv", index=False)
    country_summary.to_csv(OUT / "country_resilience_cost_summary.csv", index=False)

    major = country_costs[country_costs.major_market]
    payload = {
        "schema_version": "1.0",
        "method_version": "adaptation_economics_v1",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "global_cost_assumptions": global_assumptions,
        "country_cost_assumptions": country_assumptions,
        "global_restoration_summary": global_summary.to_dict(orient="records"),
        "country_resilience_summary": country_summary.to_dict(orient="records"),
        "hard_checks": {
            "major_country_count": int(len(major)),
            "major_country_demand_share_pct": float(major.demand_share.sum() * 100.0),
            "all_major_countries_have_positive_resilience_premium": bool(
                major.resilience_premium_billion_usd_per_year.gt(0).all()
            ),
            "least_cost_points_identified": int(global_frontier.least_cost_point.sum()),
        },
    }
    VALIDATION.mkdir(parents=True, exist_ok=True)
    (VALIDATION / "adaptation_economics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(global_summary.to_string(index=False))
    print(country_summary.to_string(index=False))
    print(json.dumps(payload["hard_checks"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
