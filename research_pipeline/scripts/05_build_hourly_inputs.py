"""Convert hourly climate data into data-centre load and renewable resources.

This module is an independent, transparent reduced-order implementation.  It
does not import, copy, or translate code from the Lei--Masanet GitHub
repository.  The published work is used only to define plausible technology
archetypes and calibration ranges.

The main scenario is a constant 1 MW IT load in an advanced hyperscale
facility.  Colocation, conventional air-cooled, and liquid-cooled archetypes
are emitted as sensitivity cases.  Solar and wind outputs are resource
proxies, not site-specific engineering yield assessments.

Calibration anchors
-------------------
* Lei and Masanet (2022), "Climate- and Technology-Specific PUE and WUE
  Predictions for U.S. Data Centers using a Physics-Based Approach",
  Resources, Conservation and Recycling 182, 106323,
  doi:10.1016/j.resconrec.2022.106323.  The paper motivates hourly dependence
  on dry-bulb temperature and humidity and the use of cooling archetypes.
* Wang et al. (2026), "Global data-water symbiosis reduces AI
  infrastructure's carbon and water footprint", Environmental Science and
  Ecotechnology 31, 100702, doi:10.1016/j.ese.2026.100702.  Its supplementary
  assumptions report representative PUE levels of 1.76 (conventional),
  1.46--1.48 (cloud/colocation), and 1.14--1.22 (hyperscale).
* A local audit of public Epoch AI records found a median reported/derived PUE
  of about 1.30 across 74 sites.  This is used only as an external plausibility
  check because the records mix technologies and operating periods.

The parameters below were chosen before evaluating future climates so that
historical Dublin annual means reproduce the broad archetype anchors while
retaining an interpretable wet-bulb response.  They are scenario parameters,
not facility-level estimates.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_MARKETS = (
    WORKSPACE / "data" / "24x7_cfe" / "processed" / "market_sample.csv"
)
DEFAULT_ERA5_ROOT = (
    WORKSPACE / "data" / "24x7_cfe" / "raw" / "openmeteo" / "era5_1995_2014"
)
DEFAULT_FUTURE_ROOT = (
    WORKSPACE / "data" / "24x7_cfe" / "processed" / "future_hourly"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE / "data" / "24x7_cfe" / "processed" / "hourly_inputs"
)
DEFAULT_DIAGNOSTICS = (
    WORKSPACE / "output" / "24x7_cfe" / "diagnostics" / "hourly_inputs.csv"
)

CLIMATE_MODELS = [
    "FGOALS_f3_H",
    "HiRAM_SIT_HR",
    "MRI_AGCM3_2_S",
    "EC_Earth3P_HR",
    "MPI_ESM1_2_XR",
    "NICAM16_8S",
]

METHOD_VERSION = "reduced_order_pue_resources_v1"
IT_LOAD_MW = 1.0

REQUIRED_CLIMATE_COLUMNS = [
    "market_id",
    "time",
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "wind_speed_100m",
    "shortwave_radiation",
]


@dataclass(frozen=True)
class PUEArchetype:
    """Parameters for the reduced-order PUE equation.

    The equation separates non-cooling electrical overhead, continuously
    operating air/pump power, economizer-dependent fan/pump power, and
    compressor/heat-rejection power:

        m(Tw) = logistic((Tw - Tw_threshold) / transition_width)
        COP(Tw) = clip(COP_20 - COP_slope * (Tw - 20), COP_min, COP_max)
        PUE = 1 + fixed + flow_base + flow_variable*m
                  + compressor_scale*m/COP
                  + extreme_penalty*max(Tw-penalty_threshold, 0)/penalty_scale

    Here Tw is wet-bulb temperature in degrees Celsius.  ``m`` is a smooth
    mechanical-cooling fraction: zero approximates full economizer operation
    and one approximates full mechanical cooling.  The equation is deliberately
    parsimonious so every climate response is visible and auditable.
    """

    name: str
    wet_bulb_threshold_c: float
    transition_width_c: float
    fixed_overhead: float
    flow_base: float
    flow_variable: float
    compressor_scale: float
    cop_at_20c: float
    cop_slope_per_c: float
    cop_min: float
    cop_max: float
    extreme_penalty: float
    penalty_threshold_c: float
    penalty_scale_c: float
    plausible_annual_min: float
    plausible_annual_max: float


# The ordering intentionally encodes progressively less efficient cooling.
# ``advanced_hyperscale`` is the paper's main scenario; ``liquid`` represents
# a low-climate-sensitivity adaptation rather than today's fleet average.
PUE_ARCHETYPES = {
    "advanced_hyperscale": PUEArchetype(
        name="advanced_hyperscale",
        wet_bulb_threshold_c=15.0,
        transition_width_c=2.5,
        fixed_overhead=0.075,
        flow_base=0.040,
        flow_variable=0.035,
        compressor_scale=0.45,
        cop_at_20c=6.0,
        cop_slope_per_c=0.12,
        cop_min=3.0,
        cop_max=8.0,
        extreme_penalty=0.020,
        penalty_threshold_c=24.0,
        penalty_scale_c=12.0,
        plausible_annual_min=1.08,
        plausible_annual_max=1.35,
    ),
    "colocation": PUEArchetype(
        name="colocation",
        wet_bulb_threshold_c=10.0,
        transition_width_c=3.0,
        fixed_overhead=0.270,
        flow_base=0.100,
        flow_variable=0.070,
        compressor_scale=0.80,
        cop_at_20c=5.5,
        cop_slope_per_c=0.13,
        cop_min=2.8,
        cop_max=7.5,
        extreme_penalty=0.025,
        penalty_threshold_c=23.0,
        penalty_scale_c=10.0,
        plausible_annual_min=1.30,
        plausible_annual_max=1.75,
    ),
    "conventional": PUEArchetype(
        name="conventional",
        wet_bulb_threshold_c=5.0,
        transition_width_c=5.0,
        fixed_overhead=0.450,
        flow_base=0.120,
        flow_variable=0.100,
        compressor_scale=1.00,
        cop_at_20c=5.0,
        cop_slope_per_c=0.14,
        cop_min=2.5,
        cop_max=7.0,
        extreme_penalty=0.030,
        penalty_threshold_c=22.0,
        penalty_scale_c=10.0,
        plausible_annual_min=1.55,
        plausible_annual_max=2.15,
    ),
    "liquid": PUEArchetype(
        name="liquid",
        wet_bulb_threshold_c=22.0,
        transition_width_c=3.0,
        fixed_overhead=0.035,
        flow_base=0.025,
        flow_variable=0.020,
        compressor_scale=0.10,
        cop_at_20c=8.0,
        cop_slope_per_c=0.10,
        cop_min=4.0,
        cop_max=10.0,
        extreme_penalty=0.008,
        penalty_threshold_c=30.0,
        penalty_scale_c=15.0,
        plausible_annual_min=1.03,
        plausible_annual_max=1.18,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build hourly PUE, facility-load, solar-CF, and wind-CF inputs "
            "from ERA5 and bias-corrected future climate data."
        )
    )
    parser.add_argument("--markets", type=Path, default=DEFAULT_MARKETS)
    parser.add_argument("--era5-root", type=Path, default=DEFAULT_ERA5_ROOT)
    parser.add_argument("--future-root", type=Path, default=DEFAULT_FUTURE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--diagnostics", type=Path, default=DEFAULT_DIAGNOSTICS)
    parser.add_argument(
        "--market-id",
        action="append",
        help="Process only this market ID; repeat to select multiple markets.",
    )
    parser.add_argument(
        "--model",
        action="append",
        choices=CLIMATE_MODELS,
        help="Process only this future model; repeat to select multiple models.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip missing climate files instead of failing after the run.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite outputs even if they already use this method version.",
    )
    return parser.parse_args()


def stull_wet_bulb_c(
    dry_bulb_c: np.ndarray, relative_humidity_pct: np.ndarray
) -> np.ndarray:
    """Approximate wet-bulb temperature using Stull (2011).

    Citation: Stull, R. (2011), "Wet-Bulb Temperature from Relative Humidity
    and Air Temperature", Journal of Applied Meteorology and Climatology 50,
    2267--2269, doi:10.1175/JAMC-D-11-0143.1.

    The approximation is intended for ordinary near-surface conditions
    (approximately -20 to 50 degC and 5--99% RH).  RH is clipped to 5--99.9%
    only inside this approximation; the unmodified input RH remains in output.
    """

    temperature = np.asarray(dry_bulb_c, dtype=float)
    humidity = np.clip(np.asarray(relative_humidity_pct, dtype=float), 5.0, 99.9)
    return (
        temperature * np.arctan(0.151977 * np.sqrt(humidity + 8.313659))
        + np.arctan(temperature + humidity)
        - np.arctan(humidity - 1.676331)
        + 0.00391838
        * humidity**1.5
        * np.arctan(0.023101 * humidity)
        - 4.686035
    )


def reduced_order_pue(
    wet_bulb_c: np.ndarray, archetype: PUEArchetype
) -> np.ndarray:
    """Calculate hourly PUE for one technology archetype."""

    wet_bulb = np.asarray(wet_bulb_c, dtype=float)
    scaled = (
        wet_bulb - archetype.wet_bulb_threshold_c
    ) / archetype.transition_width_c
    scaled = np.clip(scaled, -60.0, 60.0)
    mechanical_fraction = 1.0 / (1.0 + np.exp(-scaled))
    cop = np.clip(
        archetype.cop_at_20c
        - archetype.cop_slope_per_c * (wet_bulb - 20.0),
        archetype.cop_min,
        archetype.cop_max,
    )
    extreme = (
        archetype.extreme_penalty
        * np.maximum(wet_bulb - archetype.penalty_threshold_c, 0.0)
        / archetype.penalty_scale_c
    )
    pue = (
        1.0
        + archetype.fixed_overhead
        + archetype.flow_base
        + archetype.flow_variable * mechanical_fraction
        + archetype.compressor_scale * mechanical_fraction / cop
        + extreme
    )
    return pue


def pv_capacity_factor(
    shortwave_radiation_w_m2: np.ndarray,
    air_temperature_c: np.ndarray,
    inverter_loading_ratio: float = 1.34,
    system_derate: float = 0.86,
    noct_c: float = 45.0,
    temperature_coefficient_per_c: float = -0.004,
) -> np.ndarray:
    """Convert surface shortwave radiation into an AC PV capacity-factor proxy.

    Global horizontal shortwave radiation is used as a plane-of-array proxy.
    Cell temperature follows the common NOCT approximation

        T_cell = T_air + (NOCT - 20) / 800 * G,

    and the DC response is linear in irradiance with a crystalline-silicon
    temperature coefficient.  The default 1.34 inverter loading ratio follows
    the NREL 2024 ATB utility-PV representative plant.  The 0.86 derate bundles
    non-temperature DC and AC losses.  Output is clipped at the AC nameplate.
    """

    radiation = np.maximum(
        np.asarray(shortwave_radiation_w_m2, dtype=float), 0.0
    )
    air_temperature = np.asarray(air_temperature_c, dtype=float)
    cell_temperature = air_temperature + (noct_c - 20.0) / 800.0 * radiation
    temperature_factor = 1.0 + temperature_coefficient_per_c * (
        cell_temperature - 25.0
    )
    temperature_factor = np.clip(temperature_factor, 0.0, 1.25)
    dc_per_ac = (
        inverter_loading_ratio
        * system_derate
        * radiation
        / 1000.0
        * temperature_factor
    )
    return np.clip(dc_per_ac, 0.0, 1.0)


def wind_capacity_factor(
    wind_speed_100m_m_s: np.ndarray,
    cut_in_m_s: float = 3.0,
    rated_m_s: float = 12.0,
    cut_out_m_s: float = 25.0,
) -> np.ndarray:
    """Apply a generic hub-height wind-turbine power curve.

    Power is zero below cut-in, increases with the normalized cubic wind-energy
    flux between cut-in and rated speed, remains at nameplate until cut-out,
    and is zero at and above cut-out.  ERA5/QDM already provides 100 m wind, so
    no additional shear extrapolation is applied.
    """

    speed = np.maximum(np.asarray(wind_speed_100m_m_s, dtype=float), 0.0)
    factor = np.zeros_like(speed)
    ramp = (speed >= cut_in_m_s) & (speed < rated_m_s)
    factor[ramp] = (
        speed[ramp] ** 3 - cut_in_m_s**3
    ) / (rated_m_s**3 - cut_in_m_s**3)
    rated = (speed >= rated_m_s) & (speed < cut_out_m_s)
    factor[rated] = 1.0
    return np.clip(factor, 0.0, 1.0)


def validate_climate(frame: pd.DataFrame, source: Path) -> pd.DataFrame:
    """Validate and normalize the common hourly climate schema."""

    missing = sorted(set(REQUIRED_CLIMATE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"{source}: missing required columns {missing}")

    result = frame.copy()
    result["time"] = pd.to_datetime(result["time"], utc=True)
    result = result.sort_values("time", kind="stable").reset_index(drop=True)

    if result["market_id"].nunique(dropna=False) != 1:
        raise ValueError(f"{source}: expected exactly one market_id")
    if result["time"].duplicated().any():
        raise ValueError(f"{source}: duplicate timestamps")
    if len(result) > 1:
        steps = result["time"].diff().iloc[1:]
        if not steps.eq(pd.Timedelta(hours=1)).all():
            bad = int((~steps.eq(pd.Timedelta(hours=1))).sum())
            raise ValueError(f"{source}: {bad} non-hourly time steps")

    numeric = [
        "temperature_2m",
        "relative_humidity_2m",
        "dew_point_2m",
        "wind_speed_100m",
        "shortwave_radiation",
    ]
    values = result[numeric].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{source}: non-finite values in climate variables")
    if not result["relative_humidity_2m"].between(0.0, 100.0).all():
        raise ValueError(f"{source}: relative humidity outside 0--100%")
    if (result["wind_speed_100m"] < -1e-9).any():
        raise ValueError(f"{source}: negative 100 m wind speed")
    if (result["shortwave_radiation"] < -1e-9).any():
        raise ValueError(f"{source}: negative shortwave radiation")
    return result


def build_hourly_inputs(
    climate: pd.DataFrame,
    market: pd.Series,
    period: str,
    climate_model: str,
) -> pd.DataFrame:
    """Build a common, publication-ready hourly input table."""

    temperature = climate["temperature_2m"].to_numpy(dtype=float)
    humidity = climate["relative_humidity_2m"].to_numpy(dtype=float)
    wet_bulb = stull_wet_bulb_c(temperature, humidity)

    output = pd.DataFrame(
        {
            "market_id": str(market["market_id"]),
            "market_name": str(market["market_name"]),
            "country_code": str(market["country_code"]),
            "macro_region": str(market["macro_region"]),
            "latitude": float(market["latitude"]),
            "longitude": float(market["longitude"]),
            "period": period,
            "climate_model": climate_model,
            "time": climate["time"].array,
            "temperature_2m": temperature,
            "relative_humidity_2m": humidity,
            "dew_point_2m": climate["dew_point_2m"].to_numpy(dtype=float),
            "wet_bulb_2m": wet_bulb,
            "wind_speed_100m": climate["wind_speed_100m"].to_numpy(dtype=float),
            "shortwave_radiation": climate[
                "shortwave_radiation"
            ].to_numpy(dtype=float),
            "it_load_mw": IT_LOAD_MW,
            "method_version": METHOD_VERSION,
        }
    )
    output["pv_capacity_factor"] = pv_capacity_factor(
        output["shortwave_radiation"].to_numpy(),
        output["temperature_2m"].to_numpy(),
    )
    output["wind_capacity_factor"] = wind_capacity_factor(
        output["wind_speed_100m"].to_numpy()
    )

    for name, archetype in PUE_ARCHETYPES.items():
        pue = reduced_order_pue(wet_bulb, archetype)
        output[f"pue_{name}"] = pue
        # PUE - 1 includes cooling plus power-distribution and other non-IT
        # overhead, so this is deliberately not labelled "cooling power".
        output[f"facility_overhead_mw_{name}"] = IT_LOAD_MW * (pue - 1.0)
        output[f"facility_load_mw_{name}"] = IT_LOAD_MW * pue

    return output


def diagnostic_row(frame: pd.DataFrame, source: Path) -> dict[str, object]:
    """Summarize one market-period-model output and its validation status."""

    diagnostic: dict[str, object] = {
        "market_id": frame["market_id"].iloc[0],
        "market_name": frame["market_name"].iloc[0],
        "period": frame["period"].iloc[0],
        "climate_model": frame["climate_model"].iloc[0],
        "source_file": str(source),
        "method_version": frame["method_version"].iloc[0],
        "n_hours": len(frame),
        "start_time": frame["time"].min().isoformat(),
        "end_time": frame["time"].max().isoformat(),
        "duplicate_timestamps": int(frame["time"].duplicated().sum()),
        "missing_values": int(frame.isna().sum().sum()),
        "wet_bulb_min_c": float(frame["wet_bulb_2m"].min()),
        "wet_bulb_max_c": float(frame["wet_bulb_2m"].max()),
        "pv_cf_mean": float(frame["pv_capacity_factor"].mean()),
        "pv_cf_min": float(frame["pv_capacity_factor"].min()),
        "pv_cf_max": float(frame["pv_capacity_factor"].max()),
        "wind_cf_mean": float(frame["wind_capacity_factor"].mean()),
        "wind_cf_min": float(frame["wind_capacity_factor"].min()),
        "wind_cf_max": float(frame["wind_capacity_factor"].max()),
    }

    pue_columns = []
    plausible = True
    for name, archetype in PUE_ARCHETYPES.items():
        column = f"pue_{name}"
        pue_columns.append(column)
        mean_pue = float(frame[column].mean())
        diagnostic[f"{column}_mean"] = mean_pue
        diagnostic[f"{column}_p01"] = float(frame[column].quantile(0.01))
        diagnostic[f"{column}_p99"] = float(frame[column].quantile(0.99))
        diagnostic[f"{column}_min"] = float(frame[column].min())
        diagnostic[f"{column}_max"] = float(frame[column].max())
        in_range = (
            archetype.plausible_annual_min
            <= mean_pue
            <= archetype.plausible_annual_max
        )
        diagnostic[f"{column}_annual_plausible"] = bool(in_range)
        plausible = plausible and in_range

    pue_values = frame[pue_columns].to_numpy(dtype=float)
    cf_values = frame[
        ["pv_capacity_factor", "wind_capacity_factor"]
    ].to_numpy(dtype=float)
    pue_physical = bool(
        np.isfinite(pue_values).all()
        and (pue_values >= 1.0).all()
        and (pue_values <= 3.0).all()
    )
    cf_physical = bool(
        np.isfinite(cf_values).all()
        and (cf_values >= 0.0).all()
        and (cf_values <= 1.0).all()
    )
    ordering = bool(
        (
            frame["pue_liquid"]
            <= frame["pue_advanced_hyperscale"]
        ).all()
        and (
            frame["pue_advanced_hyperscale"]
            <= frame["pue_colocation"]
        ).all()
        and (
            frame["pue_colocation"]
            <= frame["pue_conventional"]
        ).all()
    )
    complete = bool(
        diagnostic["duplicate_timestamps"] == 0
        and diagnostic["missing_values"] == 0
    )
    diagnostic["pue_physical_bounds_ok"] = pue_physical
    diagnostic["renewable_cf_bounds_ok"] = cf_physical
    diagnostic["archetype_ordering_ok"] = ordering
    diagnostic["annual_pue_plausible_ok"] = plausible
    diagnostic["diagnostic_pass"] = bool(
        pue_physical and cf_physical and ordering and complete and plausible
    )
    return diagnostic


def validate_derived(frame: pd.DataFrame, diagnostic: dict[str, object]) -> None:
    """Fail fast rather than silently persist an invalid derived table."""

    if not diagnostic["diagnostic_pass"]:
        failed = [
            name
            for name in [
                "pue_physical_bounds_ok",
                "renewable_cf_bounds_ok",
                "archetype_ordering_ok",
                "annual_pue_plausible_ok",
            ]
            if not diagnostic[name]
        ]
        if diagnostic["duplicate_timestamps"]:
            failed.append("duplicate_timestamps")
        if diagnostic["missing_values"]:
            failed.append("missing_values")
        raise ValueError(
            f"{frame['market_id'].iloc[0]} "
            f"{frame['period'].iloc[0]} "
            f"{frame['climate_model'].iloc[0]} failed: {failed}"
        )


def select_markets(
    market_table: pd.DataFrame,
    requested_ids: Iterable[str] | None,
    limit: int | None,
) -> pd.DataFrame:
    """Filter the preregistered market table without changing its order."""

    selected = market_table.copy()
    if requested_ids:
        requested = list(dict.fromkeys(requested_ids))
        available = set(selected["market_id"].astype(str))
        unknown = sorted(set(requested) - available)
        if unknown:
            raise ValueError(f"Unknown market IDs: {unknown}")
        rank = {market_id: index for index, market_id in enumerate(requested)}
        selected = selected[selected["market_id"].isin(requested)].copy()
        selected["_selection_order"] = selected["market_id"].map(rank)
        selected = selected.sort_values("_selection_order").drop(
            columns="_selection_order"
        )
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        selected = selected.head(limit)
    return selected.reset_index(drop=True)


def load_existing_output(path: Path) -> pd.DataFrame:
    """Load a cached output only when its method version matches."""

    frame = pd.read_parquet(path)
    versions = set(frame.get("method_version", pd.Series(dtype=str)).dropna())
    if versions != {METHOD_VERSION}:
        raise ValueError(
            f"{path} was produced by {versions or 'an unknown method'}; "
            "rerun with --force"
        )
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    return frame


def process_one(
    source: Path,
    destination: Path,
    market: pd.Series,
    period: str,
    climate_model: str,
    force: bool,
) -> dict[str, object]:
    """Build, validate, and persist one common hourly input file."""

    if destination.exists() and not force:
        derived = load_existing_output(destination)
    else:
        climate = validate_climate(pd.read_parquet(source), source)
        expected_market = str(market["market_id"])
        actual_market = str(climate["market_id"].iloc[0])
        if actual_market != expected_market:
            raise ValueError(
                f"{source}: market_id {actual_market!r} != {expected_market!r}"
            )
        derived = build_hourly_inputs(
            climate=climate,
            market=market,
            period=period,
            climate_model=climate_model,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        derived.to_parquet(
            destination, index=False, compression="zstd", engine="pyarrow"
        )

    diagnostic = diagnostic_row(derived, source)
    validate_derived(derived, diagnostic)
    print(
        f"ok {market['market_id']} {period} {climate_model}: "
        f"{len(derived):,} hours -> {destination}"
    )
    return diagnostic


def main() -> None:
    args = parse_args()
    markets = pd.read_csv(args.markets)
    required_market_columns = {
        "market_id",
        "market_name",
        "country_code",
        "macro_region",
        "latitude",
        "longitude",
    }
    missing_market_columns = sorted(required_market_columns - set(markets.columns))
    if missing_market_columns:
        raise ValueError(
            f"{args.markets}: missing market columns {missing_market_columns}"
        )
    markets = select_markets(markets, args.market_id, args.limit)
    models = args.model or CLIMATE_MODELS

    diagnostics: list[dict[str, object]] = []
    missing_sources: list[Path] = []
    for _, market in markets.iterrows():
        market_id = str(market["market_id"])

        historical_source = args.era5_root / f"{market_id}.parquet"
        historical_output = (
            args.output_root
            / "historical"
            / "era5_1995_2014"
            / f"{market_id}.parquet"
        )
        if historical_source.exists():
            diagnostics.append(
                process_one(
                    source=historical_source,
                    destination=historical_output,
                    market=market,
                    period="era5_1995_2014",
                    climate_model="ERA5",
                    force=args.force,
                )
            )
        else:
            missing_sources.append(historical_source)

        for model in models:
            future_source = args.future_root / market_id / f"{model}.parquet"
            future_output = (
                args.output_root
                / "future"
                / "highresmip_2031_2050"
                / market_id
                / f"{model}.parquet"
            )
            if future_source.exists():
                diagnostics.append(
                    process_one(
                        source=future_source,
                        destination=future_output,
                        market=market,
                        period="highresmip_2031_2050",
                        climate_model=model,
                        force=args.force,
                    )
                )
            else:
                missing_sources.append(future_source)

    if diagnostics:
        diagnostics_frame = pd.DataFrame(diagnostics).sort_values(
            ["market_id", "period", "climate_model"]
        )
        args.diagnostics.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_frame.to_csv(args.diagnostics, index=False)
        print(
            f"wrote {len(diagnostics_frame)} diagnostic rows -> "
            f"{args.diagnostics}"
        )
    if missing_sources:
        message = (
            f"{len(missing_sources)} climate inputs were missing; first: "
            f"{missing_sources[0]}"
        )
        if args.skip_missing:
            print(f"warning: {message}")
        else:
            raise FileNotFoundError(
                f"{message}. Use --skip-missing for an intentionally partial run."
            )
    if not diagnostics:
        raise RuntimeError("No hourly inputs were produced.")


if __name__ == "__main__":
    main()
