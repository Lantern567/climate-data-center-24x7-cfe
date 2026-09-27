#!/usr/bin/env python3
"""Validate and document the assembled 2023--2025 wind/solar database."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = (
    ROOT
    / "data"
    / "24x7_cfe"
    / "processed"
    / "wind_solar_supply_2023_2025"
)
DEFAULT_REPORT_ROOT = (
    ROOT / "output" / "24x7_cfe_facility" / "wind_solar_supply_database"
)
YEARS = [2023, 2024, 2025]

REQUIRED_PARQUETS = {
    "gem_asset_phase_snapshot_2026-02.parquet": (
        "GEM",
        "全球项目/阶段",
        "2026-02 当前发布快照",
    ),
    "gem_historical_vintage_asset_snapshot_2023_2025.parquet": (
        "GEM",
        "全球项目/阶段",
        "下一次发布代理，不是精确年末",
    ),
    "gem_country_year_utility_scale_capacity_2023_2025.parquet": (
        "GEM",
        "国家/地区—年",
        "公用事业级、当前运行队列按投运年",
    ),
    "irena_country_year_capacity_2023_2025.parquet": (
        "IRENASTAT",
        "国家/地区—技术—并网方式—年",
        "年末实际装机 MW",
    ),
    "ember_monthly_capacity_2023_2025.parquet": (
        "Ember",
        "国家/地区—技术—月",
        "月度装机，含 AC/DC 明确口径",
    ),
    "ember_country_year_capacity_2023_2025.parquet": (
        "Ember",
        "国家/地区—技术—年",
        "12 月装机快照",
    ),
    "ember_country_year_generation_2023_2025.parquet": (
        "Ember",
        "国家/地区—技术—年",
        "实际发电 TWh",
    ),
    "eia_us_generator_year_capacity_2023_2025.parquet": (
        "EIA-860",
        "美国发电机—年",
        "逐发电机实际装机；2025 provisional",
    ),
    "eia_us_plant_month_generation_2023_2025.parquet": (
        "EIA-923",
        "美国电厂—技术—月",
        "实际净发电 MWh；2025 provisional",
    ),
    "entsoe_country_year_capacity_2023_2025.parquet": (
        "ENTSO-E",
        "欧洲国家—技术—年",
        "报告净装机 MW",
    ),
    "entsoe_country_month_generation_2023_2025.parquet": (
        "ENTSO-E",
        "欧洲国家—技术—月",
        "报告实际净发电 GWh",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_text(content: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def parquet_inventory(data_root: Path) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for filename, (source, granularity, interpretation) in REQUIRED_PARQUETS.items():
        path = data_root / filename
        if not path.exists():
            raise FileNotFoundError(path)
        parquet = pq.ParquetFile(path)
        inventory.append(
            {
                "file": filename,
                "source": source,
                "granularity": granularity,
                "interpretation": interpretation,
                "rows": parquet.metadata.num_rows,
                "columns": len(parquet.schema.names),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return inventory


def render_markdown(payload: dict[str, Any]) -> str:
    inventory = payload["inventory"]
    lines = [
        "# 2023—2025 风光真实资源数据库",
        "",
        f"> 统一验证状态：**{payload['status']}**。本数据库不使用数据中心需求反推装机，且不把气象模拟出力命名为实际发电。",
        "",
        "## 已取得的数据",
        "",
        "| 来源 | 粒度 | 行数 | 口径 | 文件 |",
        "|---|---|---:|---|---|",
    ]
    for item in inventory:
        lines.append(
            "| {source} | {granularity} | {rows:,} | {interpretation} | "
            "`{file}` |".format(**item)
        )
    lines.extend(
        [
            "",
            "## 全球年末装机校验",
            "",
            "IRENASTAT 的世界区域表（On-grid + Off-grid）逐值复核通过：",
            "",
            "| 年份 | 光伏（MW） | 风电（MW） |",
            "|---:|---:|---:|",
        ]
    )
    world = payload["key_results"]["irena_world_capacity_mw"]
    for year in YEARS:
        lines.append(
            f"| {year} | {world[str(year)]['solar']:,.2f} | "
            f"{world[str(year)]['wind']:,.2f} |"
        )
    lines.extend(
        [
            "",
            "## 实际发电与区域覆盖",
            "",
            f"- Ember 年度风光实际发电表：{payload['key_results']['ember_generation_rows']:,} 行。",
            f"- EIA-923 美国电厂—技术—月实际净发电：{payload['key_results']['eia_generation_rows']:,} 行。",
            f"- ENTSO-E 欧洲国家—技术—月实际净发电：{payload['key_results']['entsoe_generation_rows']:,} 行。",
            "- ENTSO-E 官方源有 14 条国家—年份—技术序列缺月；未补值。",
            "- EIA 2025 为 early release，仅保留逐记录用途并标记 provisional，不按州、地区或全国汇总。",
            "",
            "## 重要边界",
            "",
            "1. IRENASTAT 是年末真实装机主校准层；Country/area 含非国家聚合项，不能直接把所有行相加成全球值。",
            "2. GEM 提供项目坐标、状态和投运信息。2023、2024、2025 分别使用下一次发布作为代理，不是精确的 12 月 31 日资产快照。",
            "3. GEM 地图太阳能容量不是统一 MWac，不能直接作为 AC 装机求和；AC 总量使用 IRENASTAT、Ember 或 GEM MWac 汇总表。",
            "4. Ember 2025 国家年度发电覆盖不完整，只对有观测的国家使用；全球汇总值可作一致性校验。",
            "5. 当前得到的是电网物理装机和实际发电数据库。没有 PPA、直供或可交付份额时，不能据此宣称数据中心拥有这些电量，也不能直接计算企业 24/7 CFE 匹配率。",
            "",
            "## 下一计算接口",
            "",
            "- 需求端继续保持设施小时功率与需求电量；",
            "- 供给端优先使用实际发电观测；无小时观测处，才用真实有效装机乘同地点 ERA5 容量因子重建小时形状；",
            "- 所有重建结果命名为 `observed_constrained_hourly_reconstruction` 或 `modeled_generation_from_actual_capacity`，与 `reported_actual_generation` 分开。",
            "",
            "## 验证文件",
            "",
            "- `database_validation.json`：本报告的机器可读总验证；",
            "- `validation.json`：GEM、Ember、EIA 主流程验证；",
            "- `irena_validation.json`：IRENASTAT API 请求、哈希和世界总量校验；",
            "- `entsoe_validation.json`：欧洲源数据缺月与逐年对账；",
            "- `gem_historical_vintages_validation.json`：GEM 历史发布代理及局限。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    report_root = args.report_root.resolve()

    main_validation = read_json(data_root / "validation.json")
    irena = read_json(data_root / "irena_validation.json")
    entsoe = read_json(data_root / "entsoe_validation.json")
    gem_vintages = read_json(data_root / "gem_historical_vintages_validation.json")
    inventory = parquet_inventory(data_root)

    irena_checks = irena["official_world_region_validation"]["checks"]
    world: dict[str, dict[str, float]] = {str(year): {} for year in YEARS}
    for check in irena_checks:
        technology = "solar" if check["technology"] == "Solar energy" else "wind"
        world[str(check["year"])][technology] = check[
            "observed_world_on_plus_off_grid_mw"
        ]

    hard_checks = {
        "required_parquets_exist_and_readable": len(inventory) == len(REQUIRED_PARQUETS),
        "target_years_exact": main_validation["all_target_years_present"],
        "capacity_not_inferred_from_demand": not main_validation["boundary"][
            "demand_used_to_infer_capacity"
        ],
        "observed_and_modeled_generation_separated": main_validation["boundary"][
            "observed_generation_kept_separate_from_modeled_generation"
        ],
        "irena_world_checks_all_pass": irena["official_world_region_validation"][
            "all_pass"
        ],
        "entsoe_transform_hard_checks_pass": entsoe["hard_checks_passed"],
        "entsoe_contains_no_imputation": not entsoe["contains_imputed_values"],
        "gem_vintage_validation_pass": gem_vintages["validation_pass"],
        "gem_vintage_target_years_complete": gem_vintages["coverage"][
            "available_proxy_target_years"
        ]
        == YEARS,
        "gem_vintage_pending_years_empty": not gem_vintages["coverage"][
            "pending_target_years"
        ],
    }
    all_hard_checks_pass = all(hard_checks.values())

    inventory_by_file = {item["file"]: item for item in inventory}
    payload = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": (
            "PASS_WITH_DECLARED_SOURCE_GAPS" if all_hard_checks_pass else "FAIL"
        ),
        "all_hard_checks_pass": all_hard_checks_pass,
        "hard_checks": hard_checks,
        "scope": {
            "years": YEARS,
            "technologies": ["solar", "wind"],
            "data_roles": [
                "installed_capacity",
                "asset_inventory",
                "reported_actual_generation",
            ],
            "demand_derived_capacity": False,
        },
        "inventory": inventory,
        "key_results": {
            "irena_world_capacity_mw": world,
            "ember_generation_rows": inventory_by_file[
                "ember_country_year_generation_2023_2025.parquet"
            ]["rows"],
            "eia_generation_rows": inventory_by_file[
                "eia_us_plant_month_generation_2023_2025.parquet"
            ]["rows"],
            "entsoe_generation_rows": inventory_by_file[
                "entsoe_country_month_generation_2023_2025.parquet"
            ]["rows"],
            "gem_asset_phase_rows_2026_release": inventory_by_file[
                "gem_asset_phase_snapshot_2026-02.parquet"
            ]["rows"],
            "gem_historical_proxy_rows": inventory_by_file[
                "gem_historical_vintage_asset_snapshot_2023_2025.parquet"
            ]["rows"],
            "entsoe_incomplete_monthly_series": entsoe["generation_validation"][
                "incomplete_series_count"
            ],
        },
        "declared_source_gaps": [
            "ENTSO-E has 14 incomplete country-year-technology monthly series; no imputation.",
            "EIA 2025 is an early release and is not approved for aggregate totals.",
            "GEM historical vintages are next-release proxies, not exact year-end snapshots.",
            "GEM solar map capacity is not a uniform MWac series.",
            "Ember 2025 country-level annual generation coverage is incomplete.",
        ],
    }
    atomic_json(payload, data_root / "database_validation.json")
    atomic_text(
        render_markdown(payload),
        report_root / "WIND_SOLAR_DATABASE_2023_2025_REPORT.md",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
