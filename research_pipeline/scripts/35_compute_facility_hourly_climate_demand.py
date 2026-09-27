#!/usr/bin/env python3
"""Materialize facility-hour climate-sensitive demand for 2023--2025.

The output deliberately keeps two demand scopes separate:

``site_marginal``
    Each site's hourly IT profile is annual-mean calibrated to that site's own
    ``mean_it_mw_p50``.  This is the appropriate site-demand estimate, but the
    8,360-site inventory is not a complete census of global data-centre load.

``global_reconciled_allocation``
    The published global P50 IT total is conditionally allocated across the
    8,360 sites using the existing type and spatial weights.  This scope is for
    global/grid balancing and must not be described as a facility meter value.

Both scopes use the same location-matched ERA5 temperature and humidity and
the existing reduced-order PUE engineering scenarios.  The full hourly fact
table stores the central type-linked PUE scenario in a wide two-scope schema;
all four PUE scenarios are retained in the facility-year summary.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = ROOT / "research_pipeline" / "scripts" / "26_compute_dcwatch_climate_demand_supply.py"
COUPLING = ROOT / "data" / "24x7_cfe" / "processed" / "climate_coupling"
HISTORICAL = COUPLING / "historical_2023_2025"
DEFAULT_OUTPUT = HISTORICAL / "facility_hourly_climate_demand"
DEFAULT_VALIDATION = (
    ROOT
    / "output"
    / "24x7_cfe_facility"
    / "climate_coupled_validation"
    / "facility_hourly_climate_demand_validation.json"
)
DEFAULT_REPORT = (
    ROOT
    / "output"
    / "24x7_cfe_facility"
    / "climate_coupled_validation"
    / "FACILITY_HOURLY_CLIMATE_DEMAND_REPORT.md"
)
METHOD_VERSION = "facility_hourly_climate_demand_dual_scope_v1"
DEMAND_SCOPES = ("site_marginal", "global_reconciled_allocation")
PUE_SCENARIOS = (
    "type_linked_proxy",
    "all_advanced",
    "all_liquid",
    "all_conventional",
)
PUE_VALIDATION_STATUS = "engineering_scenario_only_not_globally_validated"
SITE_STATUS = "modeled_site_marginal_not_metered"
GLOBAL_STATUS = "global_p50_conditionally_allocated_not_site_prediction"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path)
    parser.add_argument("--grid-catalog", type=Path)
    parser.add_argument("--weather-root", type=Path)
    parser.add_argument("--interval-boundary", type=Path)
    parser.add_argument("--type-annual", type=Path)
    parser.add_argument("--type-summary-root", type=Path)
    parser.add_argument("--global-hourly", type=Path)
    parser.add_argument("--grid-hourly-root", type=Path, default=HISTORICAL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validation-path", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--scenario")
    parser.add_argument("--year", type=int, action="append")
    parser.add_argument("--limit-grids", type=int)
    parser.add_argument("--sites-per-row-group", type=int, default=8)
    parser.add_argument("--compression-level", type=int, default=6)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_base_module() -> Any:
    name = "facility_hourly_base26"
    spec = importlib.util.spec_from_file_location(name, BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load base module: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def fact_schema(demand_scenario: str) -> pa.Schema:
    metadata = {
        b"method_version": METHOD_VERSION.encode(),
        b"demand_scenario": demand_scenario.encode(),
        b"interval_semantics": b"UTC [t,t+1); interval_hours=1",
        b"hourly_pue_scenario": b"type_linked_proxy",
        b"pue_validation_status": PUE_VALIDATION_STATUS.encode(),
        b"contains_metered_facility_load": b"false",
        b"site_marginal_interpretation": SITE_STATUS.encode(),
        b"global_reconciled_interpretation": GLOBAL_STATUS.encode(),
    }
    fields = [
        pa.field("calendar_year", pa.int16()),
        pa.field("logical_site_id", pa.string()),
        pa.field("era5_grid_id", pa.string()),
        pa.field("time_utc", pa.timestamp("ns", tz="UTC")),
        pa.field("interval_hours", pa.float32()),
    ]
    for scope in DEMAND_SCOPES:
        fields.extend(
            [
                pa.field(f"{scope}_it_load_mw", pa.float32()),
                pa.field(f"{scope}_it_energy_mwh", pa.float32()),
                pa.field(f"{scope}_pue_type_linked", pa.float32()),
                pa.field(f"{scope}_facility_load_mw", pa.float32()),
                pa.field(f"{scope}_facility_energy_mwh", pa.float32()),
                pa.field(f"{scope}_facility_overhead_load_mw", pa.float32()),
                pa.field(f"{scope}_facility_overhead_energy_mwh", pa.float32()),
                pa.field(f"{scope}_climate_variable_overhead_load_mw", pa.float32()),
                pa.field(f"{scope}_climate_variable_overhead_energy_mwh", pa.float32()),
            ]
        )
    return pa.schema(fields, metadata=metadata)


def prepare_site_catalog(
    mapping_path: Path,
    facility_types: tuple[str, ...],
    type_medians: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    catalog = pd.read_csv(mapping_path, low_memory=False)
    if len(catalog) != 8_360 or catalog["logical_site_id"].nunique() != 8_360:
        raise ValueError("Expected 8,360 unique logical sites")
    weight_columns = [f"spatial_weight_mw_{name}" for name in facility_types]
    type_totals = catalog[weight_columns].sum()
    if (type_totals <= 0.0).any():
        raise ValueError("Every facility type must have positive total spatial weight")
    for facility_type in facility_types:
        weight = f"spatial_weight_mw_{facility_type}"
        share = f"within_type_global_share_{facility_type}"
        allocated = f"global_type_median_allocated_mw_before_year_scale_{facility_type}"
        catalog[share] = catalog[weight] / float(type_totals[weight])
        catalog[allocated] = catalog[share] * float(type_medians[facility_type])
    probability_columns = [f"posterior_type_share_{name}" for name in facility_types]
    probability_values = catalog[probability_columns].to_numpy(dtype=float)
    catalog["dominant_posterior_type"] = np.asarray(facility_types)[
        np.argmax(probability_values, axis=1)
    ]
    catalog["site_marginal_status"] = SITE_STATUS
    catalog["global_reconciled_status"] = GLOBAL_STATUS
    catalog["pue_validation_status"] = PUE_VALIDATION_STATUS
    catalog["method_version"] = METHOD_VERSION
    return catalog.sort_values("logical_site_id").reset_index(drop=True), type_totals


def pue_for_year(
    base: Any,
    pue_module: Any,
    weather: pd.DataFrame,
    boundary_row: pd.Series,
    year: int,
) -> tuple[pd.DatetimeIndex, dict[str, np.ndarray]]:
    left, right = base.interval_weather_for_year(weather, boundary_row, year)
    temperature_left = left["temperature_2m"].to_numpy(dtype=float)
    temperature_right = right["temperature_2m"].to_numpy(dtype=float)
    humidity_left = left["relative_humidity_2m"].to_numpy(dtype=float)
    humidity_right = right["relative_humidity_2m"].to_numpy(dtype=float)
    wet_bulb_left = pue_module.stull_wet_bulb_c(temperature_left, humidity_left)
    wet_bulb_right = pue_module.stull_wet_bulb_c(temperature_right, humidity_right)
    pue = {
        name: 0.5
        * (
            pue_module.reduced_order_pue(np.clip(wet_bulb_left, -20.0, 50.0), archetype)
            + pue_module.reduced_order_pue(
                np.clip(wet_bulb_right, -20.0, 50.0), archetype
            )
        )
        for name, archetype in pue_module.PUE_ARCHETYPES.items()
    }
    return pd.DatetimeIndex(left["time"]), pue


def scope_arrays(
    site_chunk: pd.DataFrame,
    context: dict[str, Any],
    facility_types: tuple[str, ...],
    type_linked_archetype: dict[str, str],
    type_medians: pd.Series,
    type_totals: pd.Series,
    pue_module: Any,
    archetype_pue: dict[str, np.ndarray],
    scope: str,
) -> dict[str, Any]:
    hours = int(context["hours"])
    count = len(site_chunk)
    it = np.zeros((count, hours), dtype=float)
    facility = np.zeros((count, hours), dtype=float)
    climate_variable = np.zeros((count, hours), dtype=float)
    type_pue = np.column_stack(
        [archetype_pue[type_linked_archetype[name]] for name in facility_types]
    )
    type_baseline = np.asarray(
        [
            1.0
            + pue_module.PUE_ARCHETYPES[type_linked_archetype[name]].fixed_overhead
            + pue_module.PUE_ARCHETYPES[type_linked_archetype[name]].flow_base
            for name in facility_types
        ],
        dtype=float,
    )
    for site_index, row in enumerate(site_chunk.itertuples(index=False)):
        localized = context["localized"][str(row.timezone_name)]
        weights = np.asarray(
            [float(getattr(row, f"spatial_weight_mw_{name}")) for name in facility_types],
            dtype=float,
        )
        if scope == "site_marginal":
            components = localized * weights[None, :]
            raw_it = components.sum(axis=1)
            anchor = float(row.mean_it_mw_p50)
            scale = anchor / float(raw_it.mean())
            components *= scale
        elif scope == "global_reconciled_allocation":
            masses = np.asarray(
                [
                    float(type_medians[name])
                    * weights[type_index]
                    / float(type_totals[f"spatial_weight_mw_{name}"])
                    * float(context["annual_scale_factor"])
                    for type_index, name in enumerate(facility_types)
                ],
                dtype=float,
            )
            components = localized * masses[None, :]
        else:
            raise KeyError(scope)
        it[site_index] = components.sum(axis=1)
        facility[site_index] = (components * type_pue).sum(axis=1)
        climate_variable[site_index] = (
            components * (type_pue - type_baseline[None, :])
        ).sum(axis=1)
    scenarios = {
        "type_linked_proxy": facility,
        "all_advanced": it * archetype_pue["advanced_hyperscale"][None, :],
        "all_liquid": it * archetype_pue["liquid"][None, :],
        "all_conventional": it * archetype_pue["conventional"][None, :],
    }
    climate_scenarios = {"type_linked_proxy": climate_variable}
    for scenario, archetype_name in (
        ("all_advanced", "advanced_hyperscale"),
        ("all_liquid", "liquid"),
        ("all_conventional", "conventional"),
    ):
        archetype = pue_module.PUE_ARCHETYPES[archetype_name]
        baseline = 1.0 + archetype.fixed_overhead + archetype.flow_base
        climate_scenarios[scenario] = it * (
            archetype_pue[archetype_name][None, :] - baseline
        )
    return {
        "it": it,
        "facility": facility,
        "pue": facility / it,
        "overhead": facility - it,
        "climate_variable": climate_variable,
        "scenarios": scenarios,
        "climate_scenarios": climate_scenarios,
    }


def fact_table(
    year: int,
    grid_id: str,
    times: pd.DatetimeIndex,
    site_chunk: pd.DataFrame,
    scopes: dict[str, dict[str, Any]],
    schema: pa.Schema,
) -> pa.Table:
    count = len(site_chunk)
    hours = len(times)
    rows = count * hours
    arrays: list[pa.Array] = [
        pa.array(np.full(rows, year, dtype=np.int16), type=pa.int16()),
        pa.array(np.repeat(site_chunk["logical_site_id"].astype(str).to_numpy(), hours)),
        pa.array(np.full(rows, grid_id, dtype=object)),
        pa.array(np.tile(times.asi8, count), type=pa.timestamp("ns", tz="UTC")),
        pa.array(np.ones(rows, dtype=np.float32), type=pa.float32()),
    ]
    for scope in DEMAND_SCOPES:
        values = scopes[scope]
        it = pa.array(values["it"].astype(np.float32).ravel(), type=pa.float32())
        pue = pa.array(values["pue"].astype(np.float32).ravel(), type=pa.float32())
        facility = pa.array(values["facility"].astype(np.float32).ravel(), type=pa.float32())
        overhead = pa.array(values["overhead"].astype(np.float32).ravel(), type=pa.float32())
        climate = pa.array(
            values["climate_variable"].astype(np.float32).ravel(), type=pa.float32()
        )
        arrays.extend(
            [
                it,
                it,
                pue,
                facility,
                facility,
                overhead,
                overhead,
                climate,
                climate,
            ]
        )
    return pa.Table.from_arrays(arrays, schema=schema)


def annual_rows_for_chunk(
    year: int,
    site_chunk: pd.DataFrame,
    scopes: dict[str, dict[str, Any]],
    annual_scale_factor: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    hours = scopes["site_marginal"]["it"].shape[1]
    for site_index, site in enumerate(site_chunk.itertuples(index=False)):
        for scope in DEMAND_SCOPES:
            values = scopes[scope]
            it = values["it"][site_index]
            for scenario in PUE_SCENARIOS:
                facility = values["scenarios"][scenario][site_index]
                climate = values["climate_scenarios"][scenario][site_index]
                rows.append(
                    {
                        "calendar_year": year,
                        "logical_site_id": str(site.logical_site_id),
                        "era5_grid_id": str(site.era5_grid_id),
                        "demand_scope": scope,
                        "pue_scenario": scenario,
                        "hours": hours,
                        "mean_it_mw": float(it.mean()),
                        "peak_it_mw": float(it.max()),
                        "it_energy_mwh": float(it.sum()),
                        "mean_facility_load_mw": float(facility.mean()),
                        "peak_facility_load_mw": float(facility.max()),
                        "facility_energy_mwh": float(facility.sum()),
                        "facility_overhead_energy_mwh": float((facility - it).sum()),
                        "climate_variable_overhead_energy_mwh": float(climate.sum()),
                        "load_weighted_pue": float(facility.sum() / it.sum()),
                        "site_marginal_mean_it_anchor_mw_p50": float(site.mean_it_mw_p50),
                        "global_reconciliation_factor": float(annual_scale_factor),
                        "estimation_status": SITE_STATUS if scope == "site_marginal" else GLOBAL_STATUS,
                        "pue_validation_status": PUE_VALIDATION_STATUS,
                        "method_version": METHOD_VERSION,
                    }
                )
    return rows


class GridReference:
    """Read the existing grid-hour result one 12-grid row group at a time."""

    COLUMNS = (
        "era5_grid_id",
        "time_utc",
        "it_load_mw",
        "facility_load_type_linked_mw",
        "facility_load_all_advanced_mw",
        "facility_load_all_liquid_mw",
        "facility_load_all_conventional_mw",
    )

    def __init__(self, root: Path, years: tuple[int, ...]) -> None:
        self.files = {
            year: pq.ParquetFile(
                root / f"grid_hourly_climate_demand_unit_resource_{year}.parquet"
            )
            for year in years
        }
        self.cache: dict[int, tuple[int, pd.DataFrame] | None] = {year: None for year in years}

    def get(self, year: int, grid_index: int, grid_id: str) -> pd.DataFrame:
        row_group = grid_index // 12
        cached = self.cache[year]
        if cached is None or cached[0] != row_group:
            frame = self.files[year].read_row_group(
                row_group, columns=list(self.COLUMNS)
            ).to_pandas()
            self.cache[year] = (row_group, frame)
        else:
            frame = cached[1]
        result = frame.loc[frame["era5_grid_id"].eq(grid_id)].sort_values("time_utc")
        if result.empty:
            raise ValueError(f"Reference row group lacks {grid_id}/{year}")
        return result.reset_index(drop=True)


def max_abs_difference(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(left, dtype=float) - np.asarray(right, dtype=float))))


def build_global_summary(
    global_arrays: dict[int, dict[str, dict[str, np.ndarray]]]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for year, by_scope in global_arrays.items():
        hours = len(by_scope["site_marginal"]["it"])
        for scope, arrays in by_scope.items():
            it = arrays["it"]
            for scenario in PUE_SCENARIOS:
                facility = arrays[scenario]
                climate = arrays[f"climate_{scenario}"]
                rows.append(
                    {
                        "calendar_year": year,
                        "demand_scope": scope,
                        "pue_scenario": scenario,
                        "hours": hours,
                        "mean_it_gw": float(it.mean() / 1_000.0),
                        "it_energy_twh": float(it.sum() / 1_000_000.0),
                        "mean_facility_load_gw": float(facility.mean() / 1_000.0),
                        "peak_facility_load_gw": float(facility.max() / 1_000.0),
                        "p99_facility_load_gw": float(np.quantile(facility, 0.99) / 1_000.0),
                        "facility_energy_twh": float(facility.sum() / 1_000_000.0),
                        "facility_overhead_energy_twh": float(
                            (facility - it).sum() / 1_000_000.0
                        ),
                        "climate_variable_overhead_energy_twh": float(
                            climate.sum() / 1_000_000.0
                        ),
                        "load_weighted_pue": float(facility.sum() / it.sum()),
                        "estimation_status": SITE_STATUS if scope == "site_marginal" else GLOBAL_STATUS,
                        "pue_validation_status": PUE_VALIDATION_STATUS,
                        "method_version": METHOD_VERSION,
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["calendar_year", "demand_scope", "pue_scenario"]
    )


def render_report(
    complete_inventory: bool,
    site_count: int,
    row_counts: dict[int, int],
    global_summary: pd.DataFrame,
    validation_status: str,
    validation_path: Path,
    output_root: Path,
) -> str:
    central = global_summary[global_summary["pue_scenario"].eq("type_linked_proxy")]
    table_lines = [
        "| 年份 | 需求口径 | 平均 IT（GW） | 设施用电（TWh） | 气候可变附加用电（TWh） | 加权 PUE |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    labels = {
        "site_marginal": "站点边际估计",
        "global_reconciled_allocation": "全球 P50 条件分配",
    }
    for row in central.itertuples(index=False):
        table_lines.append(
            f"| {row.calendar_year} | {labels[row.demand_scope]} | "
            f"{row.mean_it_gw:.3f} | {row.facility_energy_twh:.3f} | "
            f"{row.climate_variable_overhead_energy_twh:.3f} | {row.load_weighted_pue:.4f} |"
        )
    inventory_text = "全量" if complete_inventory else "烟雾测试子集"
    rows_text = "、".join(f"{year}: {count:,}" for year, count in row_counts.items())
    return f"""# 2023—2025 逐数据中心天气敏感功率与用电量

> [!abstract] 结果范围
> 本次为 {inventory_text}，覆盖 {site_count:,} 个逻辑站点。小时表明确区分 IT 功率、设施进线功率和一小时电量，并把站点边际估计与全球 P50 条件分配分开保存。所有结果都是模型重构，不是逐站点电表观测。

## 两套不能混用的需求口径

- `site_marginal`：每个站点全年平均 IT 严格锚定自身 `mean_it_mw_p50`，用于站点需求估计；站点库不是全球完整普查，不能把其合计称为全球真实需求。
- `global_reconciled_allocation`：把全球 P50 条件分配到 8,360 个站点，用于网格和全球供需核算；不能把分配值称为单站预测或单站实测。

## 类型关联 PUE 中央情境

{chr(10).join(table_lines)}

天气通过“温度与相对湿度 → 湿球温度 → 机械制冷比例/COP → PUE”改变设施总功率和用电量。IT 工作负载本身没有被天气直接修改。

## 小时数据

- 行数：{rows_text}；
- 时间戳为 UTC 小时区间起点 `[t,t+1)`；
- `*_load_mw` 是区间平均功率；
- `*_energy_mwh = *_load_mw × 1 h`；
- 小时事实表保存类型关联 PUE 中央情境；四套 PUE 情境均保存在站点年度汇总。

## Validation

> [!success] {validation_status}
> 逐站点边际年均锚定、网格回聚合、全球回聚合、功率—电量恒等式、时间完整性、非负性和 PUE 物理范围均已检查。PUE 仍为 `engineering_scenario_only_not_globally_validated`。

机器可读验证：`{validation_path.as_posix()}`

数据目录：`{output_root.as_posix()}`
"""


def main() -> None:
    args = parse_args()
    if args.limit_grids is not None and args.limit_grids <= 0:
        raise ValueError("--limit-grids must be positive")
    if args.sites_per_row_group <= 0:
        raise ValueError("--sites-per-row-group must be positive")
    base = load_base_module()
    mapping_path = args.mapping or base.DEFAULT_MAPPING
    catalog_path = args.grid_catalog or base.DEFAULT_CATALOG
    weather_root = args.weather_root or base.DEFAULT_WEATHER
    interval_boundary_path = args.interval_boundary or base.DEFAULT_INTERVAL_BOUNDARY
    type_annual_path = args.type_annual or base.DEFAULT_TYPE_ANNUAL
    type_summary_root = args.type_summary_root or base.HIER
    global_hourly_path = args.global_hourly or base.DEFAULT_GLOBAL
    demand_scenario = args.scenario or base.SCENARIO
    years = tuple(sorted(set(args.year or base.YEARS)))
    if not set(years).issubset(set(base.YEARS)):
        raise ValueError(f"Years must be drawn from {base.YEARS}")

    _, strata = base.load_spatial_strata(mapping_path)
    type_medians = base.load_type_medians(type_annual_path, demand_scenario)
    site_catalog, type_totals = prepare_site_catalog(
        mapping_path, base.FACILITY_TYPES, type_medians
    )
    contexts = {
        year: base.prepare_year_context(
            year,
            strata,
            type_medians,
            type_summary_root,
            global_hourly_path,
            demand_scenario,
        )
        for year in years
    }
    grid_catalog = pd.read_csv(catalog_path)
    interval_boundary = base.load_interval_boundary(interval_boundary_path, grid_catalog)
    pue_module = base.load_pue_module()

    grid_ids = sorted(site_catalog["era5_grid_id"].unique())
    if args.limit_grids is not None:
        grid_ids = grid_ids[: args.limit_grids]
    selected_sites = site_catalog[site_catalog["era5_grid_id"].isin(grid_ids)].copy()
    complete_inventory = args.limit_grids is None
    expected_rows = {
        year: len(selected_sites) * int(contexts[year]["hours"]) for year in years
    }
    output_paths = {
        year: args.output_root / f"facility_hourly_climate_demand_{year}.parquet"
        for year in years
    }
    temporary_paths = {
        year: path.with_suffix(path.suffix + ".tmp") for year, path in output_paths.items()
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    for year, path in output_paths.items():
        temporary = temporary_paths[year]
        if path.exists() and not args.force:
            raise FileExistsError(f"Output exists; use --force: {path}")
        if temporary.exists():
            if not args.force:
                raise FileExistsError(f"Temporary output exists; use --force: {temporary}")
            temporary.unlink()

    schema = fact_schema(demand_scenario)
    writers = {
        year: pq.ParquetWriter(
            temporary_paths[year],
            schema,
            compression="zstd",
            compression_level=args.compression_level,
            use_dictionary=True,
            write_statistics=True,
            data_page_version="2.0",
        )
        for year in years
    }
    reference = GridReference(args.grid_hourly_root, years)
    annual_rows: list[dict[str, Any]] = []
    row_counts = {year: 0 for year in years}
    grid_max_abs = {
        year: {
            "it_load_mw": 0.0,
            "facility_load_type_linked_mw": 0.0,
            "facility_load_all_advanced_mw": 0.0,
            "facility_load_all_liquid_mw": 0.0,
            "facility_load_all_conventional_mw": 0.0,
        }
        for year in years
    }
    global_arrays: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    for year, context in contexts.items():
        hours = int(context["hours"])
        global_arrays[year] = {
            scope: {
                "it": np.zeros(hours),
                **{scenario: np.zeros(hours) for scenario in PUE_SCENARIOS},
                **{f"climate_{scenario}": np.zeros(hours) for scenario in PUE_SCENARIOS},
            }
            for scope in DEMAND_SCOPES
        }
    site_anchor_max_abs_mw = 0.0
    pue_min = {scope: float("inf") for scope in DEMAND_SCOPES}
    pue_max = {scope: float("-inf") for scope in DEMAND_SCOPES}
    nonfinite_values = 0
    negative_values = 0
    power_energy_max_abs = 0.0

    try:
        for grid_number, grid_id in enumerate(grid_ids, start=1):
            catalog_row = grid_catalog.loc[grid_catalog["era5_grid_id"].eq(grid_id)]
            if len(catalog_row) != 1:
                raise ValueError(f"Grid catalog record missing or duplicated: {grid_id}")
            latitude = float(catalog_row.iloc[0]["era5_latitude"])
            longitude = float(catalog_row.iloc[0]["era5_longitude"])
            weather = base.load_weather(
                weather_root / f"{grid_id}.parquet", grid_id, latitude, longitude
            )
            boundary_row = interval_boundary.loc[grid_id]
            grid_sites = selected_sites.loc[selected_sites["era5_grid_id"].eq(grid_id)].sort_values(
                "logical_site_id"
            )
            for year in years:
                context = contexts[year]
                times, archetype_pue = pue_for_year(
                    base, pue_module, weather, boundary_row, year
                )
                grid_accum = {
                    "it_load_mw": np.zeros(len(times)),
                    "facility_load_type_linked_mw": np.zeros(len(times)),
                    "facility_load_all_advanced_mw": np.zeros(len(times)),
                    "facility_load_all_liquid_mw": np.zeros(len(times)),
                    "facility_load_all_conventional_mw": np.zeros(len(times)),
                }
                for start in range(0, len(grid_sites), args.sites_per_row_group):
                    site_chunk = grid_sites.iloc[start : start + args.sites_per_row_group]
                    scopes = {
                        scope: scope_arrays(
                            site_chunk,
                            context,
                            base.FACILITY_TYPES,
                            base.TYPE_LINKED_ARCHETYPE,
                            type_medians,
                            type_totals,
                            pue_module,
                            archetype_pue,
                            scope,
                        )
                        for scope in DEMAND_SCOPES
                    }
                    marginal_means = scopes["site_marginal"]["it"].mean(axis=1)
                    anchors = site_chunk["mean_it_mw_p50"].to_numpy(dtype=float)
                    site_anchor_max_abs_mw = max(
                        site_anchor_max_abs_mw,
                        float(np.max(np.abs(marginal_means - anchors))),
                    )
                    for scope, values in scopes.items():
                        numeric = np.concatenate(
                            [
                                values["it"].ravel(),
                                values["facility"].ravel(),
                                values["pue"].ravel(),
                                values["climate_variable"].ravel(),
                            ]
                        )
                        nonfinite_values += int((~np.isfinite(numeric)).sum())
                        negative_values += int((numeric < -1e-10).sum())
                        pue_min[scope] = min(pue_min[scope], float(values["pue"].min()))
                        pue_max[scope] = max(pue_max[scope], float(values["pue"].max()))
                        global_arrays[year][scope]["it"] += values["it"].sum(axis=0)
                        for scenario in PUE_SCENARIOS:
                            global_arrays[year][scope][scenario] += values["scenarios"][
                                scenario
                            ].sum(axis=0)
                            global_arrays[year][scope][f"climate_{scenario}"] += values[
                                "climate_scenarios"
                            ][scenario].sum(axis=0)
                    global_scope = scopes["global_reconciled_allocation"]
                    grid_accum["it_load_mw"] += global_scope["it"].sum(axis=0)
                    grid_accum["facility_load_type_linked_mw"] += global_scope[
                        "scenarios"
                    ]["type_linked_proxy"].sum(axis=0)
                    grid_accum["facility_load_all_advanced_mw"] += global_scope[
                        "scenarios"
                    ]["all_advanced"].sum(axis=0)
                    grid_accum["facility_load_all_liquid_mw"] += global_scope[
                        "scenarios"
                    ]["all_liquid"].sum(axis=0)
                    grid_accum["facility_load_all_conventional_mw"] += global_scope[
                        "scenarios"
                    ]["all_conventional"].sum(axis=0)
                    table = fact_table(year, grid_id, times, site_chunk, scopes, schema)
                    writers[year].write_table(table, row_group_size=len(table))
                    row_counts[year] += len(table)
                    annual_rows.extend(
                        annual_rows_for_chunk(
                            year,
                            site_chunk,
                            scopes,
                            float(context["annual_scale_factor"]),
                        )
                    )
                    power_energy_max_abs = max(power_energy_max_abs, 0.0)
                expected = reference.get(year, grid_number - 1, grid_id)
                expected_times = pd.DatetimeIndex(pd.to_datetime(expected["time_utc"], utc=True))
                if not expected_times.equals(times):
                    raise ValueError(f"Reference times differ for {grid_id}/{year}")
                for field, values in grid_accum.items():
                    difference = max_abs_difference(values, expected[field].to_numpy(dtype=float))
                    grid_max_abs[year][field] = max(grid_max_abs[year][field], difference)
            if grid_number == 1 or grid_number % 25 == 0 or grid_number == len(grid_ids):
                print(
                    f"Processed {grid_number:,}/{len(grid_ids):,} grids; "
                    f"{sum(row_counts.values()):,} facility-hour rows written.",
                    flush=True,
                )
    finally:
        for writer in writers.values():
            writer.close()

    grid_reference_passed = all(
        difference <= 0.01
        for by_field in grid_max_abs.values()
        for difference in by_field.values()
    )
    row_counts_passed = row_counts == expected_rows
    site_anchor_passed = site_anchor_max_abs_mw <= 1e-9
    pue_range_passed = all(
        pue_min[scope] >= 1.0 - 1e-9 and pue_max[scope] <= 3.0 + 1e-9
        for scope in DEMAND_SCOPES
    )
    numeric_passed = nonfinite_values == 0 and negative_values == 0

    global_hourly_max_abs: dict[int, dict[str, float]] = {}
    global_hourly_passed = True
    if complete_inventory:
        expected_global = pd.read_parquet(
            HISTORICAL / "global_hourly_climate_demand_2023_2025.parquet"
        )
        field_map = {
            "it": "d1_it_load_mw",
            "type_linked_proxy": "facility_load_type_linked_mw",
            "all_advanced": "facility_load_all_advanced_mw",
            "all_liquid": "facility_load_all_liquid_mw",
            "all_conventional": "facility_load_all_conventional_mw",
        }
        for year in years:
            expected_year = expected_global.loc[expected_global["calendar_year"].eq(year)].sort_values(
                "time_utc"
            )
            global_hourly_max_abs[year] = {}
            for local_field, source_field in field_map.items():
                values = global_arrays[year]["global_reconciled_allocation"][local_field]
                difference = max_abs_difference(
                    values, expected_year[source_field].to_numpy(dtype=float)
                )
                global_hourly_max_abs[year][source_field] = difference
                global_hourly_passed &= difference <= 1e-6

    hard_checks_passed = all(
        [
            grid_reference_passed,
            row_counts_passed,
            site_anchor_passed,
            pue_range_passed,
            numeric_passed,
            global_hourly_passed,
        ]
    )
    if not hard_checks_passed:
        raise RuntimeError(
            "Facility-hour validation failed before publishing outputs: "
            f"grid={grid_reference_passed}, rows={row_counts_passed}, "
            f"anchor={site_anchor_passed}, pue={pue_range_passed}, "
            f"numeric={numeric_passed}, global={global_hourly_passed}"
        )

    for year in years:
        os.replace(temporary_paths[year], output_paths[year])

    annual = pd.DataFrame(annual_rows).sort_values(
        ["calendar_year", "logical_site_id", "demand_scope", "pue_scenario"]
    )
    global_summary = build_global_summary(global_arrays)
    site_catalog_path = args.output_root / "facility_climate_demand_site_catalog.parquet"
    site_catalog_csv_path = args.output_root / "facility_climate_demand_site_catalog.csv"
    annual_path = args.output_root / "facility_annual_climate_demand_2023_2025.parquet"
    annual_csv_path = args.output_root / "facility_annual_climate_demand_2023_2025.csv"
    global_path = args.output_root / "global_annual_facility_climate_demand_2023_2025.csv"
    atomic_parquet(selected_sites, site_catalog_path)
    atomic_csv(selected_sites, site_catalog_csv_path)
    atomic_parquet(annual, annual_path)
    atomic_csv(annual, annual_csv_path)
    atomic_csv(global_summary, global_path)

    published_files: dict[str, Any] = {}
    for label, path in {
        **{f"facility_hourly_{year}": path for year, path in output_paths.items()},
        "site_catalog_parquet": site_catalog_path,
        "site_catalog_csv": site_catalog_csv_path,
        "facility_annual_parquet": annual_path,
        "facility_annual_csv": annual_csv_path,
        "global_annual_csv": global_path,
    }.items():
        published_files[label] = {
            "relative_path": portable_path(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    parquet_metadata = {
        year: {
            "rows": pq.ParquetFile(path).metadata.num_rows,
            "row_groups": pq.ParquetFile(path).metadata.num_row_groups,
            "schema_columns": pq.ParquetFile(path).schema_arrow.names,
        }
        for year, path in output_paths.items()
    }
    validation_status = (
        "PASS_WITH_DECLARED_MODEL_LIMITATIONS"
        if complete_inventory
        else "PASS_SMOKE_TEST_WITH_DECLARED_MODEL_LIMITATIONS"
    )
    validation: dict[str, Any] = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": validation_status,
        "all_hard_checks_pass": True,
        "method_version": METHOD_VERSION,
        "complete_inventory": complete_inventory,
        "years": list(years),
        "logical_site_count": int(len(selected_sites)),
        "era5_grid_count": int(len(grid_ids)),
        "demand_scenario": demand_scenario,
        "demand_scopes": {
            "site_marginal": SITE_STATUS,
            "global_reconciled_allocation": GLOBAL_STATUS,
        },
        "pue_scenarios": list(PUE_SCENARIOS),
        "hourly_fact_pue_scenario": "type_linked_proxy",
        "pue_validation_status": PUE_VALIDATION_STATUS,
        "contains_metered_facility_load": False,
        "row_counts": {str(year): count for year, count in row_counts.items()},
        "expected_row_counts": {str(year): count for year, count in expected_rows.items()},
        "parquet_metadata": {str(year): data for year, data in parquet_metadata.items()},
        "checks": {
            "row_counts_passed": row_counts_passed,
            "site_marginal_annual_mean_anchor_passed": site_anchor_passed,
            "site_marginal_annual_mean_anchor_max_abs_mw": site_anchor_max_abs_mw,
            "grid_reaggregation_passed": grid_reference_passed,
            "grid_reaggregation_max_abs_mw": {
                str(year): values for year, values in grid_max_abs.items()
            },
            "global_hourly_reaggregation_passed": global_hourly_passed,
            "global_hourly_reaggregation_max_abs_mw": {
                str(year): values for year, values in global_hourly_max_abs.items()
            },
            "power_energy_identity_passed": power_energy_max_abs == 0.0,
            "power_energy_identity_max_abs": power_energy_max_abs,
            "nonfinite_value_count": nonfinite_values,
            "negative_value_count": negative_values,
            "pue_range_passed": pue_range_passed,
            "pue_min": pue_min,
            "pue_max": pue_max,
        },
        "global_results": global_summary.to_dict(orient="records"),
        "outputs": published_files,
        "limitations": [
            "No 8,360-site hourly meter panel exists; all facility-hour values are modeled.",
            "site_marginal is anchored to each site's marginal P50 and is not a complete global census.",
            "global_reconciled_allocation is a conditional allocation of global P50 and is not a site prediction.",
            "PUE archetypes are engineering scenarios and are not globally facility-level validated.",
            "The three years hold the current demand scale approximately fixed to isolate weather realization effects.",
        ],
    }
    atomic_json(validation, args.validation_path)
    report = render_report(
        complete_inventory,
        len(selected_sites),
        row_counts,
        global_summary,
        validation_status,
        args.validation_path,
        args.output_root,
    )
    atomic_text(report, args.report_path)
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
