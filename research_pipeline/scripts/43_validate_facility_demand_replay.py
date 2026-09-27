# -*- coding: utf-8 -*-
"""Hard validation of the twenty-year facility hourly demand replay.

The replay is only usable if it is mechanically exact where it claims to be
exact and physically bounded where it claims to be a model.  This script asserts
both, plus the two properties that make the replay an experiment rather than a
projection: the demand scale must not move with the weather year, and the
weather must reach the facility only through PUE.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "research_pipeline" / "scripts"

_SPEC = importlib.util.spec_from_file_location(
    "demand_replay", SCRIPTS / "42_compute_facility_hourly_demand_replay.py"
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError("Cannot load 42_compute_facility_hourly_demand_replay.py")
REPLAY = importlib.util.module_from_spec(_SPEC)
sys.modules["demand_replay"] = REPLAY
_SPEC.loader.exec_module(REPLAY)

REPLAY_YEARS = REPLAY.REPLAY_YEARS
DEMAND_SCOPES = REPLAY.DEMAND_SCOPES
PUE_SCENARIOS = REPLAY.PUE_SCENARIOS
SITES = 8_360
GRIDS = 2_416

DEFAULT_OUTPUT = (
    ROOT / "output" / "24x7_cfe_facility" / "demand_replay_validation"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay-root", type=Path, default=REPLAY.DEFAULT_OUTPUT_ROOT)
    p.add_argument("--summary-root", type=Path, default=REPLAY.DEFAULT_SUMMARY_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--sample-grids", type=int, default=40,
                   help="Grids read at full hourly resolution for deep checks.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    run = json.loads((args.summary_root / "demand_replay_run_summary.json").read_text("utf-8"))
    annual = pd.read_parquet(args.summary_root / "facility_annual_demand_replay_1995_2014.parquet")
    checks: dict[str, bool] = {}
    detail: dict[str, Any] = {}

    # ---- 1. shape ----------------------------------------------------------
    expected_rows = {
        year: SITES * len(REPLAY.calendar(year)) for year in REPLAY_YEARS
    }
    actual_rows: dict[int, int] = {}
    for year in REPLAY_YEARS:
        path = args.replay_root / f"facility_hourly_demand_replay_{year}.parquet"
        actual_rows[year] = pq.ParquetFile(path).metadata.num_rows
    checks["every_weather_year_present"] = len(actual_rows) == len(REPLAY_YEARS)
    checks["row_counts_exact"] = actual_rows == expected_rows
    checks["total_rows_equal_sites_times_intervals"] = (
        sum(actual_rows.values()) == SITES * REPLAY.EXPECTED_INTERVALS
    )
    detail["total_rows"] = int(sum(actual_rows.values()))
    detail["leap_years_have_8784_hours"] = sorted(
        y for y in REPLAY_YEARS if actual_rows[y] // SITES == 8_784
    )

    # ---- 2. the experiment's defining property -----------------------------
    # A weather replay with a demand scale that drifts between years cannot
    # attribute anything to weather.  IT energy may differ ONLY by calendar
    # length, so mean IT power must be flat across the twenty years.
    central = annual[
        annual.pue_scenario.eq("type_linked_proxy")
        & annual.demand_scope.eq("global_reconciled_allocation")
    ]
    per_year = central.groupby("weather_year").agg(
        mean_it_gw=("mean_it_mw", lambda s: s.sum() / 1_000.0),
        it_twh=("it_energy_mwh", lambda s: s.sum() / 1e6),
    )
    # Exact flatness is unattainable: localising an annual unit-mean template
    # into 227 civil time zones does not preserve the annual mean to machine
    # precision in every individual year (DST transitions and sub-hour offsets
    # move a little mass across the year boundary).  The residual is a CALENDAR
    # artefact, not a weather one -- IT is built from templates and masses only,
    # so weather cannot enter this channel by construction.  What must hold is
    # that the residual is negligible against the weather signal it could
    # otherwise contaminate, so the criterion is signal separation.
    spread = float(per_year.mean_it_gw.max() - per_year.mean_it_gw.min())
    relative_spread = spread / float(per_year.mean_it_gw.mean())
    checks["mean_it_power_flat_across_weather_years"] = relative_spread < 1e-5
    detail["mean_it_gw"] = float(per_year.mean_it_gw.iloc[0])
    detail["mean_it_gw_spread_across_years"] = spread
    detail["mean_it_gw_relative_spread"] = relative_spread
    detail["it_twh_min"] = float(per_year.it_twh.min())
    detail["it_twh_max"] = float(per_year.it_twh.max())
    # The only admissible IT-energy difference is the leap day: 8784/8760.
    ratio = float(per_year.it_twh.max() / per_year.it_twh.min())
    checks["it_energy_varies_only_by_calendar_length"] = abs(ratio - 8_784 / 8_760) < 1e-6
    detail["it_energy_max_over_min"] = ratio

    # ---- 3. weather reaches the facility only through PUE ------------------
    facility = central.groupby("weather_year").facility_energy_mwh.sum() / 1e6
    pue = facility / per_year.it_twh
    facility_spread = float(facility.max() - facility.min()) / float(facility.mean())
    checks["weather_moves_facility_energy"] = facility_spread > 0.0
    # The whole experiment rests on this ratio: the demand-scale residual must be
    # orders below the weather signal, otherwise attribution is contaminated.
    checks["demand_scale_residual_negligible_vs_weather_signal"] = bool(
        relative_spread / facility_spread < 1e-3
    )
    detail["facility_twh_min"] = float(facility.min())
    detail["facility_twh_max"] = float(facility.max())
    detail["facility_relative_spread"] = facility_spread
    detail["demand_residual_over_weather_signal"] = relative_spread / facility_spread
    detail["load_weighted_pue_min"] = float(pue.min())
    detail["load_weighted_pue_max"] = float(pue.max())
    checks["load_weighted_pue_within_archetype_bounds"] = bool(
        pue.min() > 1.0 and pue.max() < 2.0
    )

    # ---- 4. scenario ordering ----------------------------------------------
    # The cooling envelope must keep its physical order in every weather year:
    # liquid < advanced < type-linked < conventional.
    by_scenario = (
        annual[annual.demand_scope.eq("global_reconciled_allocation")]
        .groupby(["weather_year", "pue_scenario"]).facility_energy_mwh.sum()
        .unstack() / 1e6
    )
    ordered = (
        (by_scenario["all_liquid"] < by_scenario["all_advanced"])
        & (by_scenario["all_advanced"] < by_scenario["type_linked_proxy"])
        & (by_scenario["type_linked_proxy"] < by_scenario["all_conventional"])
    )
    checks["cooling_envelope_ordered_every_year"] = bool(ordered.all())
    detail["scenario_twh_means"] = {
        column: float(by_scenario[column].mean()) for column in sorted(by_scenario.columns)
    }

    # ---- 5. scope separation ------------------------------------------------
    site_scope = annual[
        annual.pue_scenario.eq("type_linked_proxy") & annual.demand_scope.eq("site_marginal")
    ].groupby("weather_year").mean_it_mw.sum() / 1_000.0
    checks["site_marginal_scope_flat_across_years"] = bool(
        float(site_scope.max() - site_scope.min()) < 1e-6
    )
    checks["scopes_are_not_interchangeable"] = bool(
        abs(float(site_scope.iloc[0]) - detail["mean_it_gw"]) > 1.0
    )
    detail["site_marginal_mean_it_gw"] = float(site_scope.iloc[0])

    # ---- 6. deep hourly checks on a sample ---------------------------------
    sample_year = 2003
    path = args.replay_root / f"facility_hourly_demand_replay_{sample_year}.parquet"
    frame = pq.ParquetFile(path).read_row_groups(
        list(range(min(args.sample_grids, pq.ParquetFile(path).num_row_groups)))
    ).to_pandas()
    numeric = [c for c in frame.columns if c.endswith("_mw")]
    checks["sample_all_finite"] = bool(np.isfinite(frame[numeric].to_numpy()).all())
    checks["sample_it_and_facility_non_negative"] = bool(
        (frame[[c for c in numeric if "climate_variable" not in c]].to_numpy() >= 0.0).all()
    )
    for scope in DEMAND_SCOPES:
        ratio_sample = (
            frame[f"{scope}_facility_load_mw"] / frame[f"{scope}_it_load_mw"]
        )
        checks[f"sample_{scope}_pue_between_1_and_2"] = bool(
            ratio_sample.min() > 1.0 and ratio_sample.max() < 2.0
        )
        # Facility load must exceed IT load in every hour: overhead is positive.
        checks[f"sample_{scope}_facility_exceeds_it"] = bool(
            (frame[f"{scope}_facility_load_mw"] >= frame[f"{scope}_it_load_mw"]).all()
        )
    checks["sample_hours_per_site_exact"] = bool(
        frame.groupby("logical_site_id").size().eq(len(REPLAY.calendar(sample_year))).all()
    )
    checks["sample_timestamps_are_utc_hourly"] = bool(
        frame.time_utc.dt.tz is not None
        and frame.groupby("logical_site_id").time_utc.apply(
            lambda s: s.diff().dropna().eq(pd.Timedelta(hours=1)).all()
        ).all()
    )
    detail["sample_rows"] = int(len(frame))

    # ---- 7. inventory -------------------------------------------------------
    checks["site_and_grid_inventory_complete"] = bool(
        annual.logical_site_id.nunique() == SITES
        and annual.era5_grid_id.nunique() == GRIDS
    )
    checks["all_scopes_and_scenarios_present"] = bool(
        set(annual.demand_scope) == set(DEMAND_SCOPES)
        and set(annual.pue_scenario) == set(PUE_SCENARIOS)
    )
    checks["run_declares_fixed_demand_scale"] = bool(
        run.get("demand_scale_fixed_across_weather_years") is True
    )
    checks["run_declares_not_historical_actual"] = run.get("is_historical_actual_demand") is False
    checks["producer_reported_no_nonfinite_or_negative"] = bool(
        run["diagnostics"]["nonfinite"] == 0 and run["diagnostics"]["negative"] == 0
    )

    payload = {
        "schema_version": "1.0",
        "method_version": "facility_demand_replay_validation_v1",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "replay_root": str(args.replay_root),
        "hard_checks": checks,
        "all_hard_checks_pass": all(checks.values()),
        "detail": detail,
        "validation_status": "PASS" if all(checks.values()) else "FAIL",
    }
    out = args.output_root / "facility_demand_replay_validation.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), "utf-8")
    per_year.assign(load_weighted_pue=pue, facility_twh=facility).to_csv(
        args.output_root / "demand_replay_annual_check.csv"
    )
    failed = [k for k, v in checks.items() if not v]
    print(json.dumps({"status": payload["validation_status"], "failed": failed,
                      "detail": detail}, indent=2, ensure_ascii=False))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
