#!/usr/bin/env python3
"""Validate the fixed-2025 weather replay against reported actual generation.

The replay holds 2025 capacity constant across 1995--2014 weather, while the
reported observations cover 2023--2025 with a growing fleet.  The two therefore
cannot be compared year on year.  Two comparisons are defensible and are the
ones made here:

* annual capacity factor levels, which test whether the resource model produces
  physically plausible yield per installed megawatt, and
* normalised monthly shape, which is largely insensitive to fleet size and is
  the sharper test of the weather-to-capacity-factor chain.

Every difference is reported, never suppressed.  Curtailment, fleet vintage and
weather-year mismatch are recorded as interpretation limits rather than being
corrected away.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE_ROOT = (
    ROOT / "data" / "24x7_cfe" / "processed" / "wind_solar_supply_20y_fixed_2025"
)
REPLAY_ROOT = BASE_ROOT / "replay_results"
REPORTED_ROOT = (
    ROOT / "data" / "24x7_cfe" / "processed" / "wind_solar_supply_2023_2025"
)
DEFAULT_OUTPUT_ROOT = REPLAY_ROOT / "external_validation"
METHOD_VERSION = "fixed_2025_replay_reported_supply_comparison_v1"
REPLAY_YEARS = tuple(range(1995, 2015))
REPORTED_YEARS = (2023, 2024, 2025)

# ENTSO-E Power Statistics reports ISO 3166-1 alpha-2; the replay uses alpha-3.
ISO2_TO_ISO3 = {
    "AL": "ALB", "AT": "AUT", "BA": "BIH", "BE": "BEL", "BG": "BGR",
    "CH": "CHE", "CY": "CYP", "CZ": "CZE", "DE": "DEU", "DK": "DNK",
    "EE": "EST", "ES": "ESP", "FI": "FIN", "FR": "FRA", "GB": "GBR",
    "GE": "GEO", "GR": "GRC", "HR": "HRV", "HU": "HUN", "IE": "IRL",
    "IS": "ISL", "IT": "ITA", "LT": "LTU", "LU": "LUX", "LV": "LVA",
    "MD": "MDA", "ME": "MNE", "MK": "MKD", "MT": "MLT", "NL": "NLD",
    "NO": "NOR", "PL": "POL", "PT": "PRT", "RO": "ROU", "RS": "SRB",
    "SE": "SWE", "SI": "SVN", "SK": "SVK", "TR": "TUR", "UA": "UKR",
    "XK": "XKX",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-root", type=Path, default=REPLAY_ROOT)
    parser.add_argument("--reported-root", type=Path, default=REPORTED_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Refusing to serialise {type(value).__name__} into JSON evidence")


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_scalar),
        encoding="utf-8",
    )
    temporary.replace(path)


def finite(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def replay_country_annual(replay_root: Path) -> pd.DataFrame:
    frame = pd.read_parquet(replay_root / "annual_country_supply_1995_2014.parquet")
    if set(frame["weather_year"]) != set(REPLAY_YEARS):
        raise ValueError("Replay country table does not cover 1995--2014")
    return frame


def reported_country_capacity_factor(reported_root: Path) -> pd.DataFrame:
    """Reported capacity factor per country, technology and year.

    Generation comes from Ember, capacity from IRENA on-grid, matching the
    replay's on-grid basis.  A fleet that grows during the year makes year-end
    capacity too large a denominator, so a mid-year mean is also produced where
    the preceding year is available.
    """

    generation = pd.read_parquet(
        reported_root
        / "reported_actual_supply"
        / "ember_country_year_reported_actual_supply_2023_2025.parquet"
    )
    generation = generation[generation["iso3"].notna()].copy()
    generation["technology"] = generation["technology_family"].str.lower().map(
        {"solar": "solar_pv", "wind": "wind"}
    )
    generation = generation[generation["technology"].notna()]
    generation = (
        generation.groupby(["iso3", "year", "technology"], as_index=False)[
            "generation_twh"
        ]
        .sum()
        .rename(columns={"iso3": "country_area_code"})
    )

    capacity = pd.read_parquet(
        reported_root / "irena_country_year_capacity_2023_2025.parquet"
    )
    capacity = capacity[capacity["grid_connection"].eq("OnGrid")].copy()
    capacity["technology"] = capacity["technology"].map(
        {"Solar energy": "solar_pv", "Wind energy": "wind"}
    )
    capacity = capacity[capacity["technology"].notna()]
    capacity["installed_capacity_mw"] = pd.to_numeric(
        capacity["installed_capacity_mw"], errors="coerce"
    )
    capacity["year"] = capacity["year"].astype(int)
    capacity = capacity.groupby(
        ["country_area_code", "year", "technology"], as_index=False
    )["installed_capacity_mw"].sum()

    previous = capacity.copy()
    previous["year"] = previous["year"] + 1
    previous = previous.rename(
        columns={"installed_capacity_mw": "previous_year_end_capacity_mw"}
    )
    capacity = capacity.merge(
        previous, on=["country_area_code", "year", "technology"], how="left"
    )
    capacity["mid_year_capacity_mw"] = capacity[
        ["installed_capacity_mw", "previous_year_end_capacity_mw"]
    ].mean(axis=1, skipna=False)

    generation["year"] = generation["year"].astype(int)
    merged = generation.merge(
        capacity, on=["country_area_code", "year", "technology"], how="inner"
    )
    hours = merged["year"].map(
        lambda year: 8784.0 if year % 4 == 0 else 8760.0
    )
    merged["hours"] = hours
    merged["reported_cf_year_end_capacity"] = (
        merged["generation_twh"] * 1e6 / (merged["installed_capacity_mw"] * hours)
    )
    merged["reported_cf_mid_year_capacity"] = (
        merged["generation_twh"] * 1e6 / (merged["mid_year_capacity_mw"] * hours)
    )
    for column in (
        "reported_cf_year_end_capacity",
        "reported_cf_mid_year_capacity",
    ):
        merged.loc[~np.isfinite(merged[column]), column] = np.nan
        merged.loc[merged[column] <= 0.0, column] = np.nan
    return merged


def compare_annual_capacity_factor(
    replay: pd.DataFrame, reported: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    replay_long = []
    for technology, cf_column, capacity_column in (
        ("solar_pv", "solar_pv_capacity_factor", "solar_pv_installed_capacity_mw"),
        ("wind", "wind_capacity_factor", "wind_installed_capacity_mw"),
    ):
        subset = replay[replay[capacity_column] > 0.0]
        grouped = subset.groupby("country_area_code").agg(
            replay_cf_mean=(cf_column, "mean"),
            replay_cf_min=(cf_column, "min"),
            replay_cf_max=(cf_column, "max"),
            replay_capacity_mw=(capacity_column, "first"),
        )
        grouped["technology"] = technology
        replay_long.append(grouped.reset_index())
    replay_cf = pd.concat(replay_long, ignore_index=True)
    replay_cf = replay_cf[replay_cf["replay_cf_mean"].notna()]

    reported_cf = (
        reported.groupby(["country_area_code", "technology"], as_index=False)
        .agg(
            reported_cf_year_end=("reported_cf_year_end_capacity", "mean"),
            reported_cf_mid_year=("reported_cf_mid_year_capacity", "mean"),
            reported_generation_twh=("generation_twh", "mean"),
            reported_capacity_mw=("installed_capacity_mw", "last"),
        )
    )
    comparison = replay_cf.merge(
        reported_cf, on=["country_area_code", "technology"], how="inner"
    )
    comparison["reported_cf"] = comparison["reported_cf_mid_year"].fillna(
        comparison["reported_cf_year_end"]
    )
    comparison = comparison[comparison["reported_cf"].notna()]
    comparison["cf_ratio_replay_over_reported"] = (
        comparison["replay_cf_mean"] / comparison["reported_cf"]
    )
    comparison["cf_difference"] = comparison["replay_cf_mean"] - comparison["reported_cf"]

    summary: dict[str, Any] = {}
    for technology, group in comparison.groupby("technology"):
        weights = group["replay_capacity_mw"]
        ratio = group["cf_ratio_replay_over_reported"]
        summary[technology] = {
            "country_count": int(len(group)),
            "replay_capacity_weighted_cf": finite(
                np.average(group["replay_cf_mean"], weights=weights)
            ),
            "reported_capacity_weighted_cf": finite(
                np.average(group["reported_cf"], weights=weights)
            ),
            "capacity_weighted_ratio": finite(np.average(ratio, weights=weights)),
            "median_ratio": finite(ratio.median()),
            "ratio_p10": finite(ratio.quantile(0.10)),
            "ratio_p90": finite(ratio.quantile(0.90)),
            "spearman_rank_correlation": finite(
                group["replay_cf_mean"].corr(group["reported_cf"], method="spearman")
            ),
            "pearson_correlation": finite(
                group["replay_cf_mean"].corr(group["reported_cf"])
            ),
            "countries_within_25_percent": int(
                ((ratio >= 0.75) & (ratio <= 1.25)).sum()
            ),
            "countries_within_50_percent": int(
                ((ratio >= 0.50) & (ratio <= 1.50)).sum()
            ),
        }
    return comparison, summary


def replay_monthly(replay_root: Path, countries: set[str]) -> pd.DataFrame:
    """Monthly replay generation for selected countries, all weather years."""

    records = []
    for year in REPLAY_YEARS:
        path = (
            replay_root
            / "hourly_country_supply"
            / f"weather_year={year}"
            / "country_hourly_supply.parquet"
        )
        frame = pd.read_parquet(
            path,
            columns=[
                "time_utc",
                "country_area_code",
                "solar_pv_generation_energy_mwh",
                "wind_generation_energy_mwh",
            ],
        )
        frame = frame[frame["country_area_code"].isin(countries)]
        frame["month"] = pd.DatetimeIndex(frame["time_utc"]).month
        grouped = frame.groupby(["country_area_code", "month"], as_index=False)[
            ["solar_pv_generation_energy_mwh", "wind_generation_energy_mwh"]
        ].sum()
        grouped["weather_year"] = year
        records.append(grouped)
    return pd.concat(records, ignore_index=True)


def monthly_share(frame: pd.DataFrame, value_column: str) -> pd.Series:
    monthly = frame.groupby("month")[value_column].sum()
    total = monthly.sum()
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(f"Cannot normalise monthly shape for {value_column}")
    return monthly / total


def shape_metrics(replay: pd.Series, reported: pd.Series, label: str) -> dict[str, Any]:
    months = sorted(set(replay.index) & set(reported.index))
    if len(months) != 12:
        raise ValueError(f"{label} does not have twelve comparable months")
    a = replay.reindex(months).to_numpy(dtype=float)
    b = reported.reindex(months).to_numpy(dtype=float)
    return {
        "months": months,
        "replay_monthly_share_percent": [round(100.0 * v, 4) for v in a],
        "reported_monthly_share_percent": [round(100.0 * v, 4) for v in b],
        "pearson_correlation": finite(np.corrcoef(a, b)[0, 1]),
        "mean_absolute_deviation_percentage_points": finite(
            100.0 * np.mean(np.abs(a - b))
        ),
        "max_absolute_deviation_percentage_points": finite(
            100.0 * np.max(np.abs(a - b))
        ),
        "replay_peak_month": int(months[int(np.argmax(a))]),
        "reported_peak_month": int(months[int(np.argmax(b))]),
        "replay_trough_month": int(months[int(np.argmin(a))]),
        "reported_trough_month": int(months[int(np.argmin(b))]),
        "peak_month_matches": bool(
            months[int(np.argmax(a))] == months[int(np.argmax(b))]
        ),
        "trough_month_matches": bool(
            months[int(np.argmin(a))] == months[int(np.argmin(b))]
        ),
    }


def europe_shape(
    replay_root: Path, reported_root: Path
) -> tuple[dict[str, Any], pd.DataFrame]:
    entsoe = pd.read_parquet(
        reported_root / "entsoe_country_month_generation_2023_2025.parquet"
    )
    entsoe = entsoe[entsoe["series_complete_12_months"].fillna(False)].copy()
    entsoe["country_area_code"] = entsoe["country_code"].map(ISO2_TO_ISO3)
    entsoe = entsoe[entsoe["country_area_code"].notna()]
    entsoe["technology_family"] = np.where(
        entsoe["technology"].eq("Solar"), "solar_pv", "wind"
    )
    countries = set(entsoe["country_area_code"])

    replay = replay_monthly(replay_root, countries)
    replay_countries = set(replay["country_area_code"])
    shared = sorted(countries & replay_countries)
    entsoe = entsoe[entsoe["country_area_code"].isin(shared)]
    replay = replay[replay["country_area_code"].isin(shared)]

    result: dict[str, Any] = {
        "region": "ENTSO-E reporting countries with complete twelve-month series",
        "country_count": len(shared),
        "countries": shared,
        "reported_years": sorted(int(y) for y in entsoe["year"].unique()),
        "replay_years": list(REPLAY_YEARS),
    }
    for technology, replay_column in (
        ("solar_pv", "solar_pv_generation_energy_mwh"),
        ("wind", "wind_generation_energy_mwh"),
    ):
        reported_subset = entsoe[entsoe["technology_family"].eq(technology)]
        reported_share = monthly_share(
            reported_subset.rename(columns={"generation_gwh": "value"}), "value"
        )
        replay_share = monthly_share(replay, replay_column)
        result[technology] = shape_metrics(
            replay_share, reported_share, f"Europe {technology}"
        )
    return result, replay


def united_states_shape(replay_root: Path, reported_root: Path) -> dict[str, Any]:
    eia = pd.read_parquet(
        reported_root
        / "reported_actual_supply"
        / "eia_us_monthly_reported_actual_supply_2023_2024.parquet"
    )
    eia["technology"] = eia["technology_family"].str.lower().map(
        {"solar": "solar_pv", "wind": "wind"}
    )
    eia = eia[eia["technology"].notna()]
    replay = replay_monthly(replay_root, {"USA"})
    result: dict[str, Any] = {
        "region": "United States",
        "reported_source": "EIA-923 monthly plant generation",
        "reported_years": sorted(int(y) for y in eia["year"].unique()),
        "replay_years": list(REPLAY_YEARS),
        "reported_scope_note": (
            "EIA-923 excludes much small-scale distributed solar, so the solar "
            "shape reflects utility-scale plants."
        ),
    }
    for technology, replay_column in (
        ("solar_pv", "solar_pv_generation_energy_mwh"),
        ("wind", "wind_generation_energy_mwh"),
    ):
        reported_subset = eia[eia["technology"].eq(technology)]
        reported_share = monthly_share(
            reported_subset.rename(columns={"generation_mwh": "value"}), "value"
        )
        replay_share = monthly_share(replay, replay_column)
        result[technology] = shape_metrics(
            replay_share, reported_share, f"United States {technology}"
        )
    return result


def global_level_comparison(
    replay_root: Path, reported_root: Path
) -> dict[str, Any]:
    replay_global = pd.read_csv(replay_root / "annual_global_supply_1995_2014.csv")
    reported_global = pd.read_parquet(
        reported_root
        / "reported_actual_supply"
        / "global_annual_reported_actual_supply_2023_2025.parquet"
    ).sort_values("year")
    reported_global["previous_solar_capacity_mw"] = reported_global[
        "solar_year_end_capacity_mw"
    ].shift(1)
    reported_global["previous_wind_capacity_mw"] = reported_global[
        "wind_year_end_capacity_mw"
    ].shift(1)
    rows = []
    for row in reported_global.itertuples(index=False):
        hours = 8784.0 if int(row.year) % 4 == 0 else 8760.0
        for technology, generation, year_end, previous in (
            (
                "solar_pv",
                row.solar_generation_twh,
                row.solar_year_end_capacity_mw,
                row.previous_solar_capacity_mw,
            ),
            (
                "wind",
                row.wind_generation_twh,
                row.wind_year_end_capacity_mw,
                row.previous_wind_capacity_mw,
            ),
        ):
            mid_year = (
                np.nan
                if not np.isfinite(previous)
                else 0.5 * (float(previous) + float(year_end))
            )
            rows.append(
                {
                    "year": int(row.year),
                    "technology": technology,
                    "reported_generation_twh": float(generation),
                    "year_end_capacity_mw": float(year_end),
                    "mid_year_capacity_mw": finite(mid_year),
                    "reported_cf_year_end_capacity": float(generation)
                    * 1e6
                    / (float(year_end) * hours),
                    "reported_cf_mid_year_capacity": finite(
                        float(generation) * 1e6 / (mid_year * hours)
                    )
                    if np.isfinite(mid_year)
                    else None,
                }
            )
    reported_frame = pd.DataFrame(rows)
    result: dict[str, Any] = {
        "replay_years": list(REPLAY_YEARS),
        "reported_years": sorted(reported_frame["year"].unique()),
        "reported_detail": reported_frame.to_dict("records"),
    }
    for technology, cf_column in (
        ("solar_pv", "solar_pv_capacity_factor"),
        ("wind", "wind_capacity_factor"),
    ):
        subset = reported_frame[reported_frame["technology"].eq(technology)]
        mid_year_values = subset["reported_cf_mid_year_capacity"].dropna()
        replay_mean = float(replay_global[cf_column].mean())
        reported_mid = finite(mid_year_values.mean()) if len(mid_year_values) else None
        result[technology] = {
            "replay_capacity_factor_mean": replay_mean,
            "replay_capacity_factor_min": float(replay_global[cf_column].min()),
            "replay_capacity_factor_max": float(replay_global[cf_column].max()),
            "reported_capacity_factor_year_end_capacity_mean": finite(
                subset["reported_cf_year_end_capacity"].mean()
            ),
            "reported_capacity_factor_mid_year_capacity_mean": reported_mid,
            "ratio_replay_over_reported_mid_year": finite(replay_mean / reported_mid)
            if reported_mid
            else None,
        }
    return result


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    replay = replay_country_annual(args.replay_root)
    reported = reported_country_capacity_factor(args.reported_root)
    comparison, annual_summary = compare_annual_capacity_factor(replay, reported)
    comparison.to_parquet(
        args.output_root / "country_annual_capacity_factor_comparison.parquet",
        index=False,
    )
    comparison.sort_values(
        ["technology", "replay_capacity_mw"], ascending=[True, False]
    ).to_csv(
        args.output_root / "country_annual_capacity_factor_comparison.csv", index=False
    )
    print(
        "Annual capacity-factor comparison written for "
        f"{len(comparison):,} country-technology pairs"
    )

    europe, _ = europe_shape(args.replay_root, args.reported_root)
    print(f"Europe seasonal shape compared over {europe['country_count']} countries")
    united_states = united_states_shape(args.replay_root, args.reported_root)
    print("United States seasonal shape compared")
    global_levels = global_level_comparison(args.replay_root, args.reported_root)

    validation = {
        "schema_version": "1.0",
        "method_version": METHOD_VERSION,
        "completed_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "comparison_is_year_matched": False,
        "interpretation_limits": [
            "The replay uses fixed 2025 capacity under 1995--2014 weather; the observations cover 2023--2025 with a growing fleet, so no year-on-year identity is expected.",
            "Reported generation is net of curtailment and outages while the replay is unconstrained available energy, so the replay should sit at or above reported capacity factors.",
            "Reported fleets contain older turbines and modules than the 2025 fleet the replay represents.",
            "Normalised monthly shape is the sharper test because it is largely insensitive to fleet size and capacity vintage.",
            "IRENA year-end capacity overstates the mean operating fleet in a fast-growing year, so a mid-year mean capacity is reported alongside it.",
        ],
        "global_capacity_factor_levels": global_levels,
        "country_annual_capacity_factor": annual_summary,
        "seasonal_shape_europe": europe,
        "seasonal_shape_united_states": united_states,
    }
    atomic_json(validation, args.output_root / "reported_supply_comparison.json")
    print(f"Wrote {args.output_root / 'reported_supply_comparison.json'}")


if __name__ == "__main__":
    main()
