#!/usr/bin/env python3
"""Compute observed 2023--2025 wind/solar physical supply quantities.

The script does not combine overlapping providers. Ember World is the sole
global annual generation series. EIA and ENTSO-E are retained as finer-grained
regional observations and validation layers. No capacity factor, weather
model, data-center demand, or imputation is used in these reported-actual
generation outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT
    / "data"
    / "24x7_cfe"
    / "processed"
    / "wind_solar_supply_2023_2025"
)
DEFAULT_OUTPUT = DEFAULT_INPUT / "reported_actual_supply"
DEFAULT_REPORT = (
    ROOT
    / "output"
    / "24x7_cfe_facility"
    / "wind_solar_supply_database"
    / "REPORTED_ACTUAL_WIND_SOLAR_SUPPLY_2023_2025.md"
)
YEARS = (2023, 2024, 2025)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def atomic_text(content: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hours_in_year(year: int) -> int:
    return 8784 if pd.Timestamp(year=year, month=12, day=31).is_leap_year else 8760


def build_global_annual(
    generation: pd.DataFrame, irena_validation: dict[str, Any]
) -> pd.DataFrame:
    world = generation.loc[
        generation["area"].eq("World")
        & generation["electricity_source"].isin(["Solar", "Wind"]),
        ["year", "electricity_source", "generation_twh"],
    ].copy()
    if world.duplicated(["year", "electricity_source"]).any():
        raise ValueError("Ember World generation key is not unique")
    pivot = world.pivot(
        index="year", columns="electricity_source", values="generation_twh"
    ).rename(columns={"Solar": "solar_generation_twh", "Wind": "wind_generation_twh"})
    pivot = pivot.reindex(YEARS)
    if pivot.isna().any().any():
        raise ValueError("Ember World wind/solar generation is incomplete")
    pivot["wind_solar_generation_twh"] = (
        pivot["solar_generation_twh"] + pivot["wind_generation_twh"]
    )
    pivot["solar_share_of_wind_solar_percent"] = (
        100.0
        * pivot["solar_generation_twh"]
        / pivot["wind_solar_generation_twh"]
    )
    pivot["wind_share_of_wind_solar_percent"] = (
        100.0
        * pivot["wind_generation_twh"]
        / pivot["wind_solar_generation_twh"]
    )
    pivot["wind_solar_yoy_percent"] = (
        pivot["wind_solar_generation_twh"].pct_change(fill_method=None) * 100.0
    )
    pivot["average_solar_supply_gw"] = [
        pivot.loc[year, "solar_generation_twh"] * 1000.0 / hours_in_year(year)
        for year in YEARS
    ]
    pivot["average_wind_supply_gw"] = [
        pivot.loc[year, "wind_generation_twh"] * 1000.0 / hours_in_year(year)
        for year in YEARS
    ]
    pivot["average_wind_solar_supply_gw"] = [
        pivot.loc[year, "wind_solar_generation_twh"] * 1000.0 / hours_in_year(year)
        for year in YEARS
    ]

    irena_checks = irena_validation["official_world_region_validation"]["checks"]
    capacity = {
        (int(row["year"]), str(row["technology"])): float(
            row["observed_world_on_plus_off_grid_mw"]
        )
        for row in irena_checks
    }
    pivot["solar_year_end_capacity_mw"] = [
        capacity[(year, "Solar energy")] for year in YEARS
    ]
    pivot["wind_year_end_capacity_mw"] = [
        capacity[(year, "Wind energy")] for year in YEARS
    ]
    pivot["wind_solar_year_end_capacity_mw"] = (
        pivot["solar_year_end_capacity_mw"] + pivot["wind_year_end_capacity_mw"]
    )
    pivot["generation_observation_status"] = "reported_actual"
    pivot["generation_source_dataset"] = "Ember Yearly Electricity Data"
    pivot["capacity_source_dataset"] = "IRENASTAT 2026 H1 Region=World"
    pivot["capacity_timing"] = "calendar_year_end_maximum_net_capacity"
    pivot["contains_modeled_generation"] = False
    pivot["contains_imputation"] = False
    return pivot.reset_index()


def build_country_annual(generation: pd.DataFrame) -> pd.DataFrame:
    countries = generation.loc[
        generation["area_type"].eq("Country or economy")
        & generation["year"].isin(YEARS)
        & generation["electricity_source"].isin(["Solar", "Wind"])
    ].copy()
    countries["technology_family"] = countries["electricity_source"].str.lower()
    countries["generation_observation_status"] = "reported_actual"
    countries["contains_modeled_generation"] = False
    countries["contains_imputation"] = False
    countries["coverage_note"] = (
        "Rows exist only where Ember reports the country/economy-year-technology; "
        "absence is not interpreted as zero."
    )
    if countries.duplicated(["year", "area", "electricity_source"]).any():
        raise ValueError("Ember country/economy generation key is not unique")
    return countries.sort_values(["year", "area", "electricity_source"]).reset_index(
        drop=True
    )


def build_eia_monthly(eia: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    final = eia.loc[eia["year"].isin([2023, 2024])].copy()
    summary = (
        final.groupby(["year", "month", "technology_family"], as_index=False)
        .agg(
            generation_mwh=("net_generation_mwh", "sum"),
            source_rows=("net_generation_mwh", "size"),
            reporting_plants=("plant_code", "nunique"),
            missing_source_values=("net_generation_mwh", lambda values: int(values.isna().sum())),
        )
        .sort_values(["year", "month", "technology_family"])
    )
    summary["generation_twh"] = summary["generation_mwh"] / 1_000_000.0
    summary["generation_observation_status"] = "reported_actual_final_release"
    summary["scope_note"] = (
        "EIA-923 plant-prime-mover-fuel observations; solar excludes much of "
        "distributed small-scale generation."
    )
    summary["contains_modeled_generation"] = False
    summary["contains_imputation"] = False

    provisional = eia.loc[eia["year"].eq(2025)].copy()
    provisional_inventory = (
        provisional.groupby(["year", "month", "technology_family"], as_index=False)
        .agg(
            source_rows=("net_generation_mwh", "size"),
            reporting_plants=("plant_code", "nunique"),
            missing_source_values=("net_generation_mwh", lambda values: int(values.isna().sum())),
        )
        .sort_values(["month", "technology_family"])
    )
    provisional_inventory["generation_aggregation_withheld"] = True
    provisional_inventory["withheld_reason"] = (
        "EIA 2025 early release explicitly warns against state, regional, or national aggregation."
    )
    return summary.reset_index(drop=True), provisional_inventory.reset_index(drop=True)


def build_entsoe_reported_sum(entsoe: pd.DataFrame) -> pd.DataFrame:
    summary = (
        entsoe.groupby(["year", "month", "technology"], as_index=False)
        .agg(
            generation_gwh=("generation_gwh", "sum"),
            reporting_country_count=("country_code", "nunique"),
            source_rows=("generation_gwh", "size"),
        )
        .sort_values(["year", "month", "technology"])
    )
    summary["generation_twh"] = summary["generation_gwh"] / 1000.0
    summary["generation_observation_status"] = "reported_actual_area_sum_partial_coverage"
    summary["geographic_interpretation"] = (
        "Sum of countries present in ENTSO-E Power Statistics for the month; "
        "not a constant-boundary Europe total."
    )
    summary["contains_modeled_generation"] = False
    summary["contains_imputation"] = False
    return summary.reset_index(drop=True)


def make_report(global_annual: pd.DataFrame, validation: dict[str, Any]) -> str:
    lines = [
        "# 2023—2025 风光报告实际供给量",
        "",
        "> [!abstract] 正式结论",
        "> 本页计算的是报告实际风光发电量，不是由装机与天气模拟的潜在发电，也没有使用数据中心需求反推装机。全球总量只采用 Ember World，EIA 与 ENTSO-E 只作更细粒度区域层和外部验证，不能重复相加。",
        "",
        "## 全球风光实际发电",
        "",
        "| 年份 | 光伏（TWh） | 风电（TWh） | 风光合计（TWh） | 年均供给功率（GW） | 光伏占风光（%） | 同比（%） |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in global_annual.itertuples(index=False):
        yoy = "—" if pd.isna(row.wind_solar_yoy_percent) else f"{row.wind_solar_yoy_percent:.2f}"
        lines.append(
            f"| {row.year} | {row.solar_generation_twh:,.3f} | "
            f"{row.wind_generation_twh:,.3f} | {row.wind_solar_generation_twh:,.3f} | "
            f"{row.average_wind_solar_supply_gw:,.3f} | "
            f"{row.solar_share_of_wind_solar_percent:.2f} | {yoy} |"
        )
    lines.extend(
        [
            "",
            "> [!important] 2025 年结构变化",
            "> 2025 年全球光伏实际发电量首次在本三年序列中超过风电，光伏占风光合计 50.64%。",
            "",
            "## 分层结果",
            "",
            f"- Ember 国家/经济体—年度实际发电：{validation['row_counts']['ember_country_year']} 行。",
            f"- EIA 2023—2024 美国月度正式版汇总：{validation['row_counts']['eia_monthly_final']} 行；2025 early release 只保留记录清单，不进行禁止的全国汇总。",
            f"- ENTSO-E 国家报告值月度求和：{validation['row_counts']['entsoe_monthly_reported_area_sum']} 行；这是当月有值国家之和，不是恒定边界的欧洲总量。",
            "",
            "## 与小时供给的关系",
            "",
            "这些结果先确定观测能量总量。后续小时计算必须在月内或年内严格守恒相应观测量：有小时观测时直接使用；没有小时观测时，才用真实装机地点的 ERA5 资源形状做 `observed_constrained_hourly_reconstruction`。仅有装机而没有观测能量约束时，输出命名为 `modeled_generation_from_actual_capacity`。",
            "",
            "## 验证边界",
            "",
            "- 全球、国家、美国和欧洲数据层互相重叠，绝不相加；",
            "- EIA 光伏不覆盖全部分布式小型光伏，故只与 Ember 作口径差异验证；",
            "- ENTSO-E 官方缺月不插补；",
            "- IRENA 年末装机仅作容量校准，不能把年末容量当作全年平均容量来反推实际发电。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()

    generation = pd.read_parquet(
        input_root / "ember_country_year_generation_2023_2025.parquet"
    )
    irena_validation = json.loads(
        (input_root / "irena_validation.json").read_text(encoding="utf-8")
    )
    eia = pd.read_parquet(
        input_root / "eia_us_plant_month_generation_2023_2025.parquet"
    )
    entsoe = pd.read_parquet(
        input_root / "entsoe_country_month_generation_2023_2025.parquet"
    )

    global_annual = build_global_annual(generation, irena_validation)
    country_annual = build_country_annual(generation)
    eia_monthly, eia_2025_inventory = build_eia_monthly(eia)
    entsoe_sum = build_entsoe_reported_sum(entsoe)

    outputs = {
        "global_parquet": output_root
        / "global_annual_reported_actual_supply_2023_2025.parquet",
        "global_csv": output_root / "global_annual_reported_actual_supply_2023_2025.csv",
        "country": output_root
        / "ember_country_year_reported_actual_supply_2023_2025.parquet",
        "eia_monthly": output_root
        / "eia_us_monthly_reported_actual_supply_2023_2024.parquet",
        "eia_2025_inventory": output_root
        / "eia_us_2025_early_release_aggregation_withheld_inventory.parquet",
        "entsoe_sum": output_root
        / "entsoe_monthly_reported_area_sum_supply_2023_2025.parquet",
    }
    atomic_parquet(global_annual, outputs["global_parquet"])
    atomic_csv(global_annual, outputs["global_csv"])
    atomic_parquet(country_annual, outputs["country"])
    atomic_parquet(eia_monthly, outputs["eia_monthly"])
    atomic_parquet(eia_2025_inventory, outputs["eia_2025_inventory"])
    atomic_parquet(entsoe_sum, outputs["entsoe_sum"])

    expected_global = {
        2023: {"solar": 1664.140, "wind": 2317.180},
        2024: {"solar": 2142.708, "wind": 2507.483},
        2025: {"solar": 2781.552, "wind": 2711.191},
    }
    global_checks = []
    for row in global_annual.itertuples(index=False):
        solar_difference = row.solar_generation_twh - expected_global[row.year]["solar"]
        wind_difference = row.wind_generation_twh - expected_global[row.year]["wind"]
        global_checks.append(
            {
                "year": int(row.year),
                "solar_difference_twh": float(solar_difference),
                "wind_difference_twh": float(wind_difference),
                "pass": bool(abs(solar_difference) < 1e-9 and abs(wind_difference) < 1e-9),
            }
        )

    eia_annual = (
        eia_monthly.groupby(["year", "technology_family"])["generation_twh"]
        .sum()
        .unstack()
    )
    ember_us = country_annual.loc[country_annual["iso3"].eq("USA")].pivot(
        index="year", columns="electricity_source", values="generation_twh"
    )
    comparison_rows = []
    for year in (2023, 2024):
        for family, ember_label in (("solar", "Solar"), ("wind", "Wind")):
            eia_value = float(eia_annual.loc[year, family])
            ember_value = float(ember_us.loc[year, ember_label])
            comparison_rows.append(
                {
                    "year": year,
                    "technology_family": family,
                    "eia_twh": eia_value,
                    "ember_twh": ember_value,
                    "difference_twh": eia_value - ember_value,
                    "ratio": eia_value / ember_value if ember_value else None,
                    "interpretation": (
                        "EIA wind aligns closely with Ember"
                        if family == "wind"
                        else "EIA utility-scale solar excludes substantial distributed solar"
                    ),
                }
            )

    validation = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": "PASS_WITH_DECLARED_SOURCE_GAPS",
        "all_hard_checks_pass": bool(all(item["pass"] for item in global_checks)),
        "scope": {
            "years": list(YEARS),
            "technologies": ["solar", "wind"],
            "global_generation_source": "Ember World only",
            "reported_actual_only": True,
            "contains_modeled_generation": False,
            "contains_imputation": False,
            "data_center_demand_used": False,
        },
        "row_counts": {
            "global_annual": int(len(global_annual)),
            "ember_country_year": int(len(country_annual)),
            "eia_monthly_final": int(len(eia_monthly)),
            "eia_2025_aggregation_withheld_inventory": int(len(eia_2025_inventory)),
            "entsoe_monthly_reported_area_sum": int(len(entsoe_sum)),
        },
        "global_source_value_checks": global_checks,
        "global_results": global_annual.replace({np.nan: None}).to_dict(orient="records"),
        "eia_vs_ember_us_validation": comparison_rows,
        "eia_2025_national_aggregation_performed": False,
        "entsoe_is_constant_boundary_europe_total": False,
        "source_overlap_added_together": False,
        "outputs": {
            key: {
                "relative_path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for key, path in outputs.items()
        },
        "limitations": [
            "Ember 2025 country/economy coverage is incomplete even though World totals are present.",
            "EIA-923 solar excludes substantial distributed small-scale generation.",
            "EIA 2025 early release is retained at source-record level and not nationally aggregated.",
            "ENTSO-E reported-area sums have changing country coverage and source gaps.",
            "Annual or monthly observed energy is not yet an hourly supply series.",
        ],
    }
    atomic_json(validation, output_root / "reported_actual_supply_validation.json")
    atomic_text(make_report(global_annual, validation), args.report.resolve())
    print(json.dumps(validation, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
