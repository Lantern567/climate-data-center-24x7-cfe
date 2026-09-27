#!/usr/bin/env python3
"""Validate 2023--2025 DCWatch demand and unit-capacity resource outputs.

This report deliberately excludes the legacy portfolios whose wind/PV
nameplate capacity was back-solved from data-center demand.  Those portfolios
are diagnostics only and cannot be interpreted as installed supply,
generation, or 24/7 CFE performance.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
COUPLING = ROOT / "data" / "24x7_cfe" / "processed" / "climate_coupling"
DEFAULT_CATALOG = COUPLING / "era5_grid_catalog.csv"
DEFAULT_WEATHER = (
    ROOT
    / "data"
    / "24x7_cfe"
    / "raw"
    / "openmeteo"
    / "era5_2023_2025_dcwatch_grid"
)
DEFAULT_RESULTS = COUPLING / "historical_2023_2025"
DEFAULT_VALIDATION = (
    ROOT / "output" / "24x7_cfe_facility" / "climate_coupled_validation"
)
YEARS = (2023, 2024, 2025)
WIND_SCENARIOS = ("iec_i", "iec_ii", "iec_iii", "cubic_conservative")
CENTRAL_WIND_SCENARIO = "iec_ii"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--weather-root", type=Path, default=DEFAULT_WEATHER)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--validation-root", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument(
        "--weather-qc-step",
        type=int,
        default=10,
        help="Audit every nth catalog grid for variable-level proxy checks.",
    )
    return parser.parse_args()


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def atomic_text(content: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def solar_cosine_zenith(
    times: pd.DatetimeIndex, latitude: float, longitude: float
) -> np.ndarray:
    day = times.dayofyear.to_numpy(dtype=float)
    utc_hour = times.hour.to_numpy(dtype=float) + times.minute.to_numpy(dtype=float) / 60.0
    b_angle = 2.0 * np.pi * (day - 81.0) / 364.0
    equation_of_time_min = (
        9.87 * np.sin(2.0 * b_angle)
        - 7.53 * np.cos(b_angle)
        - 1.5 * np.sin(b_angle)
    )
    solar_time = utc_hour + longitude / 15.0 + equation_of_time_min / 60.0
    hour_angle = np.radians(15.0 * (solar_time - 12.0))
    declination = np.radians(
        23.45 * np.sin(np.radians(360.0 * (284.0 + day) / 365.0))
    )
    latitude_rad = np.radians(latitude)
    return np.maximum(
        np.sin(latitude_rad) * np.sin(declination)
        + np.cos(latitude_rad) * np.cos(declination) * np.cos(hour_angle),
        0.0,
    )


def magnus_rh(temperature_c: np.ndarray, dew_point_c: np.ndarray) -> np.ndarray:
    a = 17.625
    b = 243.04
    numerator = np.exp(a * dew_point_c / (b + dew_point_c))
    denominator = np.exp(a * temperature_c / (b + temperature_c))
    return np.clip(100.0 * numerator / denominator, 0.0, 100.0)


def weather_inventory_qc(
    catalog: pd.DataFrame, weather_root: Path, step: int
) -> dict[str, Any]:
    if step <= 0:
        raise ValueError("--weather-qc-step must be positive")
    expected_rows = 26_304
    missing_parquet: list[str] = []
    missing_raw: list[str] = []
    missing_metadata: list[str] = []
    bad_row_count: list[str] = []
    clipped_grid_count = 0
    clipped_value_count = 0
    clipped_min = 0.0
    for row in catalog.itertuples(index=False):
        grid_id = str(row.era5_grid_id)
        parquet_path = weather_root / f"{grid_id}.parquet"
        raw_path = weather_root / f"{grid_id}.json.gz"
        metadata_path = weather_root / f"{grid_id}.metadata.json"
        if not parquet_path.exists():
            missing_parquet.append(grid_id)
        elif pq.ParquetFile(parquet_path).metadata.num_rows != expected_rows:
            bad_row_count.append(grid_id)
        if not raw_path.exists():
            missing_raw.append(grid_id)
        if not metadata_path.exists():
            missing_metadata.append(grid_id)
        else:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            count = int(metadata.get("radiation_negative_clipped_count", 0))
            if count:
                clipped_grid_count += 1
                clipped_value_count += count
                clipped_min = min(
                    clipped_min,
                    float(metadata.get("radiation_negative_min_w_m2", 0.0)),
                )

    sampled = catalog.iloc[::step].copy()
    qc_rows = 0
    nonfinite_count = 0
    physical_violation_count = 0
    rh_abs_error_sum = 0.0
    radiation_error_endpoint_sum = 0.0
    radiation_error_midpoint_sum = 0.0
    horizontal_direct_sum = 0.0
    dni_above_solar_constant = 0
    for row in sampled.itertuples(index=False):
        frame = pd.read_parquet(
            weather_root / f"{row.era5_grid_id}.parquet",
            columns=[
                "time",
                "temperature_2m",
                "relative_humidity_2m",
                "dew_point_2m",
                "surface_pressure",
                "wind_speed_100m",
                "shortwave_radiation",
                "direct_normal_irradiance",
                "diffuse_radiation",
            ],
        )
        times = pd.DatetimeIndex(pd.to_datetime(frame["time"], utc=True))
        values = frame.iloc[:, 1:].to_numpy(dtype=float)
        nonfinite_count += int((~np.isfinite(values)).sum())
        physical_violation_count += int(
            (~frame["relative_humidity_2m"].between(0.0, 100.0)).sum()
            + (frame["surface_pressure"] <= 0.0).sum()
            + (frame["wind_speed_100m"] < 0.0).sum()
            + (frame[["shortwave_radiation", "direct_normal_irradiance", "diffuse_radiation"]] < 0.0)
            .to_numpy()
            .sum()
            + (frame["dew_point_2m"] > frame["temperature_2m"] + 1.0).sum()
        )
        temperature = frame["temperature_2m"].to_numpy(dtype=float)
        dew_point = frame["dew_point_2m"].to_numpy(dtype=float)
        humidity = frame["relative_humidity_2m"].to_numpy(dtype=float)
        rh_abs_error_sum += float(np.abs(magnus_rh(temperature, dew_point) - humidity).sum())
        horizontal_direct = np.maximum(
            frame["shortwave_radiation"].to_numpy(dtype=float)
            - frame["diffuse_radiation"].to_numpy(dtype=float),
            0.0,
        )
        dni = frame["direct_normal_irradiance"].to_numpy(dtype=float)
        endpoint_estimate = dni * solar_cosine_zenith(
            times, float(row.era5_latitude), float(row.era5_longitude)
        )
        midpoint_estimate = dni * solar_cosine_zenith(
            times - pd.Timedelta(minutes=30),
            float(row.era5_latitude),
            float(row.era5_longitude),
        )
        radiation_error_endpoint_sum += float(
            np.abs(horizontal_direct - endpoint_estimate).sum()
        )
        radiation_error_midpoint_sum += float(
            np.abs(horizontal_direct - midpoint_estimate).sum()
        )
        horizontal_direct_sum += float(horizontal_direct.sum())
        dni_above_solar_constant += int((dni > 1_361.0).sum())
        qc_rows += len(frame)

    return {
        "catalog_grids": int(len(catalog)),
        "expected_rows_per_grid": expected_rows,
        "expected_total_grid_hours": int(len(catalog) * expected_rows),
        "missing_parquet": missing_parquet,
        "missing_raw_json_gz": missing_raw,
        "missing_metadata": missing_metadata,
        "bad_parquet_row_count": bad_row_count,
        "inventory_complete": not any(
            [missing_parquet, missing_raw, missing_metadata, bad_row_count]
        ),
        "radiation_negative_clipped_grids": clipped_grid_count,
        "radiation_negative_clipped_values": clipped_value_count,
        "radiation_negative_min_raw_w_m2": clipped_min,
        "sample_step": step,
        "sampled_grids": int(len(sampled)),
        "sampled_grid_hours": qc_rows,
        "nonfinite_values": nonfinite_count,
        "physical_constraint_violations": physical_violation_count,
        "temperature_dewpoint_reconstructed_rh_mae_percentage_points": (
            rh_abs_error_sum / qc_rows
        ),
        "radiation_decomposition_endpoint_nmae": (
            radiation_error_endpoint_sum / horizontal_direct_sum
        ),
        "radiation_decomposition_preceding_hour_midpoint_nmae": (
            radiation_error_midpoint_sum / horizontal_direct_sum
        ),
        "dni_above_solar_constant_sample_hours": dni_above_solar_constant,
    }


def output_qc(results_root: Path) -> dict[str, Any]:
    expected_grid_rows = {2023: 2_416 * 8_760, 2024: 2_416 * 8_784, 2025: 2_416 * 8_760}
    grid_hour_rows = {
        year: pq.ParquetFile(
            results_root / f"grid_hourly_climate_demand_unit_resource_{year}.parquet"
        ).metadata.num_rows
        for year in YEARS
    }
    global_hour_path = results_root / "global_hourly_climate_demand_2023_2025.parquet"
    global_hour_rows = pq.ParquetFile(global_hour_path).metadata.num_rows
    annual = pd.read_csv(results_root / "grid_annual_climate_demand_unit_resource.csv")
    global_annual = pd.read_csv(
        results_root / "global_annual_climate_demand.csv"
    )
    return {
        "grid_hourly_rows": {str(year): int(rows) for year, rows in grid_hour_rows.items()},
        "grid_hourly_expected_rows": {
            str(year): int(rows) for year, rows in expected_grid_rows.items()
        },
        "grid_hourly_row_counts_passed": grid_hour_rows == expected_grid_rows,
        "global_hourly_rows": int(global_hour_rows),
        "global_hourly_expected_rows": 26_304,
        "global_hourly_row_count_passed": global_hour_rows == 26_304,
        "grid_annual_rows": int(len(annual)),
        "grid_annual_expected_rows": 2_416 * 3,
        "grid_annual_row_count_passed": len(annual) == 2_416 * 3,
        "global_it_reconciliation_max_abs_mw": float(
            global_annual["global_it_reconciliation_max_abs_mw"].abs().max()
        ),
    }


def result_summary(
    results_root: Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    annual = pd.read_csv(results_root / "grid_annual_climate_demand_unit_resource.csv")
    global_annual = pd.read_csv(
        results_root / "global_annual_climate_demand.csv"
    )
    central = global_annual.copy()
    demand_scenario_rows: list[dict[str, Any]] = []
    demand_scenario_columns = {
        "type_linked_proxy": "type_linked_load_weighted_pue",
        "all_advanced": "all_advanced_load_weighted_pue",
        "all_liquid": "all_liquid_load_weighted_pue",
        "all_conventional": "all_conventional_load_weighted_pue",
    }
    climate_variable_by_year = (
        annual.groupby("calendar_year", sort=True)[
            "type_linked_climate_variable_overhead_energy_mwh"
        ]
        .sum()
        .div(1_000_000.0)
    )
    for row in central.itertuples(index=False):
        for demand_scenario, pue_column in demand_scenario_columns.items():
            pue = float(getattr(row, pue_column))
            demand_scenario_rows.append(
                {
                    "calendar_year": int(row.calendar_year),
                    "demand_scenario": demand_scenario,
                    "it_energy_twh": float(row.d1_it_energy_twh),
                    "load_weighted_pue": pue,
                    "facility_energy_twh": float(row.d1_it_energy_twh * pue),
                    "facility_overhead_energy_twh": float(
                        row.d1_it_energy_twh * (pue - 1.0)
                    ),
                    "type_linked_climate_variable_overhead_energy_twh": (
                        float(climate_variable_by_year.loc[int(row.calendar_year)])
                        if demand_scenario == "type_linked_proxy"
                        else np.nan
                    ),
                }
            )
    demand_scenarios = pd.DataFrame(demand_scenario_rows)
    weighted_resource_rows: list[dict[str, Any]] = []
    for year, group in annual.groupby("calendar_year", sort=True):
        weights = group["d1_it_energy_mwh"].to_numpy(dtype=float)
        row: dict[str, Any] = {
            "calendar_year": int(year),
            "demand_weighted_pv_capacity_factor": float(
                np.average(group["pv_capacity_factor"], weights=weights)
            ),
        }
        for scenario in WIND_SCENARIOS:
            row[f"demand_weighted_wind_capacity_factor_{scenario}"] = float(
                np.average(group[f"wind_capacity_factor_{scenario}"], weights=weights)
            )
        weighted_resource_rows.append(row)
    weighted_resources = pd.DataFrame(weighted_resource_rows)
    summary = {
        "central_wind_scenario": CENTRAL_WIND_SCENARIO,
        "global_central_scenario_by_year": central.to_dict("records"),
        "demand_weighted_resource_capacity_factors": weighted_resource_rows,
        "grid_year_pue_plausibility_pass_rate": float(
            annual[
                [column for column in annual if column.endswith("_annual_pue_plausible")]
            ]
            .to_numpy(dtype=bool)
            .mean()
        ),
        "zero_wind_resource_grid_scenarios_in_2023": int(
            annual[annual["calendar_year"].eq(2023)][
                [
                    f"wind_resource_zero_in_design_year_{scenario}"
                    for scenario in WIND_SCENARIOS
                ]
            ]
            .to_numpy(dtype=bool)
            .sum()
        ),
        "global_demand_scenarios": demand_scenario_rows,
        "formal_supply_status": "actual_installed_capacity_not_loaded",
        "formal_supply_outputs_available": False,
        "hypothetical_demand_scaled_portfolios_excluded": True,
    }
    return summary, weighted_resources, demand_scenarios


def render_report(
    weather: dict[str, Any], outputs: dict[str, Any], summary: dict[str, Any]
) -> str:
    central = pd.DataFrame(summary["global_central_scenario_by_year"])
    resources = pd.DataFrame(summary["demand_weighted_resource_capacity_factors"])
    demand_scenarios = pd.DataFrame(summary["global_demand_scenarios"])
    lines = [
        "# 全量 DCWatch 气候需求与单位装机资源结果",
        "",
        f"> 生成时间：{datetime.now(UTC).isoformat()}。本报告使用 8,360 个逻辑站点、2,416 个 ERA5 网格，不使用 30 市场样本。",
        "",
        "> [!danger] 正式供给结果尚未形成",
        "> 由数据中心需求反推的风光装机、总发电量、匹配电量、缺口和 CFE 分数已全部排除。当前供给侧只报告无量纲容量因子，也就是每 1 MW 名义装机的气象出力系数。",
        "",
        "## 数据与内部验证",
        "",
        f"- 气象清单完整：{weather['inventory_complete']}；总网格小时 {weather['expected_total_grid_hours']:,}。",
        f"- 变量级抽查：{weather['sampled_grids']:,} 个网格、{weather['sampled_grid_hours']:,} 小时；非有限值 {weather['nonfinite_values']}，物理边界违例 {weather['physical_constraint_violations']}。",
        f"- 温度/露点反算 RH 的 MAE：{weather['temperature_dewpoint_reconstructed_rh_mae_percentage_points']:.3f} 个百分点。",
        f"- 辐射分解 nMAE：错误整点法 {weather['radiation_decomposition_endpoint_nmae']:.2%}；按前一小时中点法 {weather['radiation_decomposition_preceding_hour_midpoint_nmae']:.2%}。",
        f"- 网格小时输出行数通过：{outputs['grid_hourly_row_counts_passed']}；全球 IT 对账最大绝对误差 {outputs['global_it_reconciliation_max_abs_mw']:.3e} MW。",
        "",
        "## 全球需求与设施总负荷（中心情景）",
        "",
        "| 年份 | 平均 IT (GW) | IT (TWh) | 平均设施负荷 (GW) | 设施用电 (TWh) | 加权 PUE | 设施附加用电 (TWh) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in central.itertuples(index=False):
        lines.append(
            f"| {int(row.calendar_year)} | {row.d1_mean_it_gw:.3f} | {row.d1_it_energy_twh:.3f} | "
            f"{row.type_linked_mean_facility_load_gw:.3f} | {row.type_linked_facility_energy_twh:.3f} | "
            f"{row.type_linked_load_weighted_pue:.3f} | {row.type_linked_facility_overhead_energy_twh:.3f} |"
        )
    lines.extend(
        [
            "",
            "### 四套需求侧 PUE 情景",
            "",
            "| 年份 | 情景 | 加权 PUE | 设施用电 (TWh) | 设施附加用电 (TWh) |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for row in demand_scenarios.itertuples(index=False):
        lines.append(
            f"| {int(row.calendar_year)} | {row.demand_scenario} | {row.load_weighted_pue:.3f} | "
            f"{row.facility_energy_twh:.3f} | {row.facility_overhead_energy_twh:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 风光单位装机资源系数",
            "",
            "| 年份 | PV CF | IEC I 风 CF | IEC II 风 CF | IEC III 风 CF | 保守风 CF |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in resources.itertuples(index=False):
        lines.append(
            f"| {int(row.calendar_year)} | {row.demand_weighted_pv_capacity_factor:.3f} | "
            f"{row.demand_weighted_wind_capacity_factor_iec_i:.3f} | "
            f"{row.demand_weighted_wind_capacity_factor_iec_ii:.3f} | "
            f"{row.demand_weighted_wind_capacity_factor_iec_iii:.3f} | "
            f"{row.demand_weighted_wind_capacity_factor_cubic_conservative:.3f} |"
        )
    lines.extend(
        [
            "",
            "容量因子只回答天气条件下单位装机能发多少。对任意小时，1 MW 名义装机的模拟电量为 `CF × 1 MW × 1 h`；总发电量必须再乘逐电站、逐年份的真实有效装机，并考虑并网、退役、损耗和可用率。",
            "",
            "## 当前可用结果边界",
            "",
            "| 模块 | 状态 | 可解释为 | 不可解释为 |",
            "|---|---|---|---|",
            "| IT 需求功率与电量 | 已完成 | 条件重构的小时 MW、年度 TWh | 单站电表实测 |",
            "| PUE 与设施用电 | 已完成（工程情景） | 四类冷却情景的小时负荷与年度电量 | 全球设施级实证预测 |",
            "| 风光容量因子 | 已完成 | 单位装机资源可用性 | 实际发电量或装机量 |",
            "| 真实装机与总供给 | 未完成 | — | 不得由需求反推 |",
            "| 小时匹配与 CFE 分数 | 未完成 | — | 无真实装机/可交付性时不得计算 |",
            "",
            "- PUE 原型是工程情景，已有公开 HGB 对照未通过全球预测门槛，不能称作设施级实证预测。",
            "- 下一步须接入真实电站资产表及年度有效装机；优先使用小时实测发电，缺失时才计算“真实装机 × 气象容量因子”的模拟发电。",
            "- 数据中心可计入的 CFE 还需要电网可交付关系以及 PPA/自有资产的分配份额，不能把所在网区全部发电自动归给数据中心。",
            "- 三个年份使用共同的当前需求尺度，以隔离天气作用；它们不是包含真实行业增长的 2023–2025 历史需求序列。",
            "- 近期三年基线不能替代 1995–2014 长历史基准或未来 CMIP6 压力测试。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    catalog = pd.read_csv(args.catalog).sort_values("era5_grid_id").reset_index(drop=True)
    if len(catalog) != 2_416 or catalog["era5_grid_id"].nunique() != 2_416:
        raise ValueError("Expected the complete 2,416-grid catalog")
    weather = weather_inventory_qc(catalog, args.weather_root, args.weather_qc_step)
    outputs = output_qc(args.results_root)
    summary, weighted_resources, demand_scenarios = result_summary(args.results_root)
    payload = {
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "weather_inventory_and_proxy_qc": weather,
        "output_and_accounting_qc": outputs,
        "result_summary": summary,
        "market_sample_used": False,
    }
    args.validation_root.mkdir(parents=True, exist_ok=True)
    atomic_json(
        payload,
        args.validation_root / "demand_and_unit_resource_validation.json",
    )
    weighted_resources.to_csv(
        args.validation_root / "demand_weighted_resource_summary.csv", index=False
    )
    demand_scenarios.to_csv(
        args.validation_root / "global_demand_scenario_summary.csv", index=False
    )
    atomic_text(
        render_report(weather, outputs, summary),
        args.validation_root / "DEMAND_AND_UNIT_RESOURCE_REPORT.md",
    )
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
