# -*- coding: utf-8 -*-
"""Classify major data-centre markets by climate-risk mechanism.

The four mutually exclusive classes are intended for Result 4 rather than as a
universal climate taxonomy.  Major markets carry at least 0.2% of global data-
centre demand in the model.  A market is weakly exposed when its joint annual
coverage change is within +/-0.10 percentage points.  Outside that band,
adverse renewable timing defines compounding risk; a positive supply effect
that fully offsets cooling defines supply compensation; the remaining markets
are demand dominated.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "24x7_cfe" / "processed"
VALIDATION = ROOT / "output" / "24x7_cfe_facility" / "climate_adaptation_ACCESS-CM2"
MODEL = "ACCESS-CM2"
MAJOR_MARKET_SHARE = 0.002
WEAK_EXPOSURE_PP = 0.10


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    ok = values.notna() & weights.notna() & weights.gt(0)
    return float(np.average(values[ok], weights=weights[ok]))


def main() -> None:
    panel = pd.read_parquet(PROCESSED / "level_shape_ACCESS-CM2"
                            / "site_cells.parquet")
    weight = panel.HH_vH__demand_mwh

    def coverage(cell: str) -> pd.Series:
        return panel[f"{cell}__matched_mwh"] / panel[f"{cell}__demand_mwh"]

    mechanism = pd.DataFrame({
        "country": panel.supply_country_area_code,
        "weight": weight,
        "delta_demand_pp": (coverage("FH_vH") - coverage("HH_vH")) * 100,
        "delta_supply_pp": (coverage("HF_vH") - coverage("HH_vH")) * 100,
        "delta_joint_pp": (coverage("FF_vH") - coverage("HH_vH")) * 100,
        "worst_168h_historical_mwh": panel.HH_vH__worst_168h_mwh,
        "worst_168h_future_mwh": panel.FF_vH__worst_168h_mwh,
    })
    rows = []
    for country, block in mechanism.groupby("country", observed=True):
        rows.append({
            "supply_country_area_code": country,
            "demand_mwh": float(block.weight.sum()),
            "delta_demand_pp": weighted_mean(block.delta_demand_pp, block.weight),
            "delta_supply_pp": weighted_mean(block.delta_supply_pp, block.weight),
            "delta_joint_pp": weighted_mean(block.delta_joint_pp, block.weight),
            "worst_168h_change_pct": (
                block.worst_168h_future_mwh.sum()
                / block.worst_168h_historical_mwh.sum() - 1) * 100,
        })
    countries = pd.DataFrame(rows)
    countries["demand_share"] = countries.demand_mwh / countries.demand_mwh.sum()

    firm = pd.read_csv(PROCESSED / "firming_ACCESS-CM2" / "region_firming.csv")
    firm["firm_capacity_pct"] = firm.firm_share_of_mean_load * 100
    compound = pd.read_csv(PROCESSED / "climate_adaptation_ACCESS-CM2"
                           / "compound_country_summary.csv")
    countries = countries.merge(
        firm[["supply_country_area_code", "firm_capacity_pct"]],
        on="supply_country_area_code", how="left")
    countries = countries.merge(compound, on="supply_country_area_code", how="left",
                                suffixes=("", "_compound"))
    countries["major_market"] = countries.demand_share.ge(MAJOR_MARKET_SHARE)

    weak = countries.delta_joint_pp.abs().lt(WEAK_EXPOSURE_PP)
    compounding = (~weak & countries.delta_supply_pp.lt(0)
                   & countries.delta_joint_pp.lt(0))
    compensated = (~weak & countries.delta_supply_pp.gt(0)
                   & countries.delta_joint_pp.ge(0))
    countries["archetype"] = np.select(
        [weak, compounding, compensated],
        ["Weak exposure", "Compounding", "Supply-compensated"],
        default="Demand-dominated")
    countries = countries.sort_values("demand_share", ascending=False)
    countries.to_csv(PROCESSED / "climate_adaptation_ACCESS-CM2"
                     / "regional_archetypes.csv", index=False)

    major = countries[countries.major_market].copy()
    category_rows = []
    for archetype, block in major.groupby("archetype", observed=True):
        weights = block.demand_share
        category_rows.append({
            "archetype": archetype,
            "markets": int(len(block)),
            "demand_share_pct": float(weights.sum() * 100),
            "compound_probability_historical_pct": weighted_mean(
                block.compound_probability_historical * 100, weights),
            "compound_probability_future_pct": weighted_mean(
                block.compound_probability_future * 100, weights),
            "compound_probability_change_pp": weighted_mean(
                block.compound_probability_change_pp, weights),
            "max_compound_hours_168h_change": weighted_mean(
                block.max_compound_hours_168h_change, weights),
            "future_compound_deficit_concentration": weighted_mean(
                block.future_compound_deficit_concentration, weights),
        })
    category = pd.DataFrame(category_rows).sort_values("archetype")
    category.to_csv(PROCESSED / "climate_adaptation_ACCESS-CM2"
                    / "regional_archetype_summary.csv", index=False)

    checks = {
        "major_markets": int(len(major)),
        "cooling_effect_negative_markets": int(major.delta_demand_pp.lt(0).sum()),
        "adverse_supply_effect_markets": int(major.delta_supply_pp.lt(0).sum()),
        "archetypes_present": sorted(major.archetype.unique().tolist()),
    }
    payload = {
        "schema_version": "1.0",
        "method_version": "regional_climate_archetypes_v1",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "model": MODEL,
        "major_market_demand_share_threshold": MAJOR_MARKET_SHARE,
        "weak_exposure_threshold_pp": WEAK_EXPOSURE_PP,
        "checks": checks,
        "all_hard_checks_pass": bool(
            checks["major_markets"] == 38
            and checks["cooling_effect_negative_markets"] == 38
            and checks["adverse_supply_effect_markets"] == 15
            and len(checks["archetypes_present"]) == 4),
    }
    VALIDATION.mkdir(parents=True, exist_ok=True)
    (VALIDATION / "regional_archetypes.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), "utf-8")
    print(category.to_string(index=False))
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
