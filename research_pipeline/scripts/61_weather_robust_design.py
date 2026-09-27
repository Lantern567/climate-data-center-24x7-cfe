# -*- coding: utf-8 -*-
"""Result 1: is a portfolio designed on one weather year robust to weather?

Reads the (year, mix, site) tensor from script 60 and answers the three
questions that make up Result 1, none of which needs another pass over the
hourly demand:

  1.1  Does the *design decision* depend on the weather year?
       Per-site argmax over the mix axis, once per design year.  If procurement
       were insensitive to weather sampling, the twenty argmaxes would coincide.

  1.2  Does an in-sample optimum stay optimal out of sample?
       The 20x20 design-year x evaluation-year matrix, plus two derived
       quantities: optimism bias (a design's in-sample score minus its mean
       out-of-sample score) and regret (the gap to the portfolio that year e
       would have chosen with perfect hindsight).

  1.3  Does designing across K years converge?
       For each K, random design subsets are drawn, the mix is chosen on the
       subset, and the portfolio is scored on the years NOT in the subset.  The
       held-out evaluation is the point: an in-sample curve necessarily improves
       with K and would prove nothing about robustness.

Two design objectives are carried side by side throughout, because "robust" is
ambiguous and they do not select the same portfolio:

    mean     argmax_m  mean_y  coverage(y, m)      expected performance
    minimax  argmax_m  min_y   coverage(y, m)      worst-weather-year performance

The mean-objective twenty-year design is what every downstream climate script
uses as its baseline; the minimax design is reported alongside so the paper can
say what tail-oriented design would have chosen instead.
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
DESIGN = PROCESSED / "weather_robust_design"
LADDER = PROCESSED / "cfe_geographic_scenarios"
DEFAULT_VALIDATION = ROOT / "output" / "24x7_cfe_facility" / "weather_robust_design"

METHOD_VERSION = "weather_robust_design_v2"
# The held-out set is held at a FIXED size, and design sets are drawn from the
# remaining years.  An earlier version drew K design years and evaluated on all
# 20-K survivors, which silently made "worst held-out year" incomparable across
# K: a minimum over five years is mechanically deeper than a minimum over one,
# so the curve improved with K for a purely combinatorial reason.
HELD_OUT_YEARS = 5
SUBSET_SIZES = (1, 2, 3, 5, 8, 12, 15)
N_DRAWS = 200
SEED = 20260820
LADDER_REFERENCE_YEAR = 2000
# The near-term SSP2-4.5 shift in annual coverage, in percentage points, used
# only as a yardstick for "how many facilities carry a design risk larger than
# the climate signal".  Recomputed by script 62; kept as an argument so the two
# never silently disagree.
CLIMATE_SHIFT_PP = 0.35


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tensor", type=Path,
                   default=DESIGN / "design_tensor_historical.npz")
    p.add_argument("--output-root", type=Path, default=DESIGN)
    p.add_argument("--validation-root", type=Path, default=DEFAULT_VALIDATION)
    p.add_argument("--draws", type=int, default=N_DRAWS)
    p.add_argument("--climate-shift-pp", type=float, default=CLIMATE_SHIFT_PP)
    return p.parse_args()


# ------------------------------------------------------------------ helpers --
def weighted_coverage(matched: np.ndarray, demand: np.ndarray,
                      pick: np.ndarray) -> np.ndarray:
    """Demand-weighted global coverage in every year, for one mix choice per site.

    `pick` indexes the mix axis per site; the result is the ratio of summed
    matched energy to summed demand, which is the paper's headline convention
    (a mean of per-site ratios would answer a different question and is reported
    separately in the Methods).
    """
    site = np.arange(matched.shape[2])
    gathered = matched[:, pick, site]                       # (year, site)
    return gathered.sum(axis=1) / demand.sum(axis=1)


def weighted_metric(values: np.ndarray, pick: np.ndarray) -> np.ndarray:
    """Global total of an additive tail metric, for one mix choice per site."""
    site = np.arange(values.shape[2])
    return values[:, pick, site].sum(axis=1)


def argmax_mean(coverage: np.ndarray, years: np.ndarray) -> np.ndarray:
    """Per-site mix maximising mean coverage over the given year indices."""
    return coverage[years].mean(axis=0).argmax(axis=0)


def argmax_minimax(coverage: np.ndarray, years: np.ndarray) -> np.ndarray:
    """Per-site mix maximising the worst year's coverage over the given years."""
    return coverage[years].min(axis=0).argmax(axis=0)


OBJECTIVES = {"mean": argmax_mean, "minimax": argmax_minimax}


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.validation_root.mkdir(parents=True, exist_ok=True)

    with np.load(args.tensor, allow_pickle=False) as store:
        years = store["years"]
        mix_grid = store["mix_grid"]
        site_ids = store["logical_site_id"]
        country = store["supply_country_area_code"]
        continent = store["continent"]
        demand = store["demand_mwh"]                        # (year, site)
        matched = store["matched_mwh"]                      # (year, mix, site)
        worst_24 = store["worst_24h_mwh"]
        worst_168 = store["worst_168h_mwh"]

    n_year, n_mix, n_site = matched.shape
    coverage = matched / demand[:, None, :]                 # per-site annual coverage
    unmatched = demand[:, None, :] - matched
    site_weight = demand.mean(axis=0)

    # ---------------------------------------------------- 1.1 design decisions
    per_year_pick = np.stack([coverage[y].argmax(axis=0) for y in range(n_year)])
    per_year_mix = mix_grid[per_year_pick]                  # (design year, site)
    robust_pick = {name: fn(coverage, np.arange(n_year))
                   for name, fn in OBJECTIVES.items()}

    # How far does the design decision move when only the weather year changes?
    mix_spread = per_year_mix.max(axis=0) - per_year_mix.min(axis=0)
    disagree = mix_spread > 1e-9
    fleet_share = (per_year_mix * site_weight).sum(axis=1) / site_weight.sum()

    design_frame = pd.DataFrame({
        "design_year": years,
        "demand_weighted_wind_share": fleet_share,
        "median_site_wind_share": np.median(per_year_mix, axis=1),
        "sites_differing_from_20y_mean_design": [
            int((per_year_pick[y] != robust_pick["mean"]).sum()) for y in range(n_year)],
    })

    # ---------------------------------------- 1.2 design x evaluation matrix --
    designs: dict[str, np.ndarray] = {str(int(y)): per_year_pick[i]
                                      for i, y in enumerate(years)}
    designs["robust_mean_20y"] = robust_pick["mean"]
    designs["robust_minimax_20y"] = robust_pick["minimax"]

    matrix = {name: weighted_coverage(matched, demand, pick)
              for name, pick in designs.items()}
    matrix_frame = pd.DataFrame(matrix, index=pd.Index(years, name="evaluation_year")).T
    matrix_frame.index.name = "design"

    tail_matrix = {
        "uncovered_energy_mwh": {n: weighted_metric(unmatched, p) for n, p in designs.items()},
        "worst_24h_mwh": {n: weighted_metric(worst_24, p) for n, p in designs.items()},
        "worst_168h_mwh": {n: weighted_metric(worst_168, p) for n, p in designs.items()},
    }

    square = matrix_frame.loc[[str(int(y)) for y in years]].to_numpy()   # (design, eval)
    diagonal = np.diag(square)
    off_diagonal_mean = (square.sum(axis=1) - diagonal) / (n_year - 1)
    optimism = diagonal - off_diagonal_mean

    # Regret against the portfolio year e would have chosen with hindsight.
    oracle = np.array([square[e, e] for e in range(n_year)])
    regret = oracle[None, :] - square

    single_year_frame = pd.DataFrame({
        "design_year": years,
        "in_sample_coverage": diagonal,
        "out_of_sample_mean_coverage": off_diagonal_mean,
        "optimism_bias_pp": (diagonal - off_diagonal_mean) * 100,
        "out_of_sample_worst_year_coverage": np.array(
            [np.delete(square[d], d).min() for d in range(n_year)]),
        "mean_regret_pp": regret.mean(axis=1) * 100,
        "max_regret_pp": regret.max(axis=1) * 100,
    })

    # ---------------------------------- 1.2b the risk an individual buyer runs
    # The global aggregate averages 8,317 idiosyncratic design errors across 145
    # supply pools, so it measures diversification rather than robustness.  A
    # buyer designs and operates ONE facility, so the decision-relevant quantity
    # is the per-site regret distribution.
    cube = np.stack([coverage[:, per_year_pick[d], np.arange(n_site)]
                     for d in range(n_year)])              # (design, eval, site)
    site_oracle = np.stack([coverage[e, per_year_pick[e], np.arange(n_site)]
                            for e in range(n_year)])       # (eval, site)
    site_regret = (site_oracle[None] - cube) * 100
    off = ~np.eye(n_year, dtype=bool)
    flat = site_regret[off].reshape(-1, n_site)             # (design x eval, site)
    share = site_weight / site_weight.sum()
    site_mean_regret, site_max_regret = flat.mean(axis=0), flat.max(axis=0)
    exceeds = site_max_regret > args.climate_shift_pp

    # Does designing for annual coverage misprice the multi-day tail?  Compare
    # each design's worst-week gap against the mix that would have minimised it.
    gap = worst_168 / demand[:, None, :]
    tail_best = gap.min(axis=1)                             # (eval, site)
    def tail_excess(pick: np.ndarray) -> np.ndarray:
        achieved = gap[:, pick, np.arange(n_site)]
        return (achieved - tail_best) / np.maximum(tail_best, 1e-15) * 100
    tail_single = np.stack([tail_excess(per_year_pick[d]) for d in range(n_year)])
    tail_robust = tail_excess(robust_pick["mean"])

    site_frame = pd.DataFrame({
        "logical_site_id": site_ids,
        "supply_country_area_code": country,
        "continent": continent,
        "mean_annual_demand_mwh": site_weight,
        "mean_regret_pp": site_mean_regret,
        "max_regret_pp": site_max_regret,
        "wind_share_spread": mix_spread,
        "tail_excess_single_year_pct": tail_single.mean(axis=(0, 1)),
        "tail_excess_robust_pct": tail_robust.mean(axis=0),
    })
    site_frame.to_csv(args.output_root / "site_design_risk.csv", index=False)

    # ------------------------------------------- 1.3 held-out design curves --
    rng = np.random.default_rng(SEED)
    all_years = np.arange(n_year)
    curve_rows = []
    for k in SUBSET_SIZES:
        for objective, fn in OBJECTIVES.items():
            for _ in range(args.draws):
                held_out = rng.choice(all_years, size=HELD_OUT_YEARS, replace=False)
                pool = np.setdiff1d(all_years, held_out)
                draw = rng.choice(pool, size=k, replace=False)
                pick = fn(coverage, draw)
                scores = weighted_coverage(matched, demand, pick)[held_out]
                oracle_held = np.array([square[e, e] for e in held_out])
                curve_rows.append({
                    "design_years": int(k),
                    "objective": objective,
                    "mean_coverage": float(scores.mean()),
                    "worst_year_coverage": float(scores.min()),
                    "mean_regret_pp": float((oracle_held - scores).mean() * 100),
                    "worst_regret_pp": float((oracle_held - scores).max() * 100),
                    # Sampling and objective are different failures: more design
                    # years cure the first, but the design still optimises annual
                    # coverage, so the multi-day gap it leaves is unaffected.
                    "tail_excess_pct": float(
                        tail_excess(pick)[held_out].mean(axis=0) @ share),
                    "demand_weighted_wind_share": float(
                        (mix_grid[pick] * site_weight).sum() / site_weight.sum()),
                    "sites_differing_from_20y_design": int(
                        (pick != robust_pick[objective]).sum()),
                })
    curve = pd.DataFrame(curve_rows)
    measures = [c for c in curve.columns if c not in ("design_years", "objective")]
    grouped = curve.groupby(["design_years", "objective"])[measures]
    summary = pd.concat(
        {"mean": grouped.mean(), "median": grouped.median(),
         "p05": grouped.quantile(0.05), "p95": grouped.quantile(0.95)},
        axis=1)
    summary.columns = [f"{measure}_{statistic}" for statistic, measure in summary.columns]
    summary = summary.reset_index()

    # The full-record design has no held-out year by construction; it is the
    # reference the curve converges towards, reported separately rather than
    # silently scored in sample.
    full_record = []
    for objective, pick in robust_pick.items():
        scores = weighted_coverage(matched, demand, pick)
        full_record.append({
            "design_years": n_year, "objective": objective,
            "mean_coverage_in_sample": float(scores.mean()),
            "worst_year_coverage_in_sample": float(scores.min()),
            "mean_regret_pp_in_sample": float((oracle - scores).mean() * 100),
            "demand_weighted_wind_share": float(
                (mix_grid[pick] * site_weight).sum() / site_weight.sum()),
        })

    # ------------------------------------------------------------- outputs ---
    matrix_frame.to_csv(args.output_root / "design_by_evaluation_matrix.csv")
    design_frame.to_csv(args.output_root / "design_year_portfolios.csv", index=False)
    # The full design-year x site mix matrix, so a figure can show the decision
    # itself moving rather than only its demand-weighted average.
    np.savez_compressed(
        args.output_root / "design_year_site_wind_share.npz",
        years=years, logical_site_id=site_ids.astype(str),
        wind_share=per_year_mix, mean_annual_demand_mwh=site_weight,
        wind_share_robust_mean_20y=mix_grid[robust_pick["mean"]],
        wind_share_robust_minimax_20y=mix_grid[robust_pick["minimax"]])
    single_year_frame.to_csv(args.output_root / "single_year_design_risk.csv", index=False)
    curve.to_csv(args.output_root / "held_out_design_curve_draws.csv", index=False)
    summary.to_csv(args.output_root / "held_out_design_curve.csv", index=False)
    pd.DataFrame(
        {"logical_site_id": site_ids, "supply_country_area_code": country,
         "continent": continent, "mean_annual_demand_mwh": site_weight,
         "wind_share_design_2000": mix_grid[per_year_pick[list(years).index(
             LADDER_REFERENCE_YEAR)]],
         "wind_share_robust_mean_20y": mix_grid[robust_pick["mean"]],
         "wind_share_robust_minimax_20y": mix_grid[robust_pick["minimax"]]}
    ).to_csv(args.output_root / "site_portfolio_designs.csv", index=False)
    for name, block in tail_matrix.items():
        pd.DataFrame(block, index=pd.Index(years, name="evaluation_year")).T.to_csv(
            args.output_root / f"design_by_evaluation_{name}.csv")

    # ------------------------------------------------------------ validation -
    ladder = pd.read_parquet(LADDER / "site_scenario_year_cfe.parquet",
                             columns=["scenario", "weather_year", "logical_site_id",
                                      "wind_energy_share", "cfe_score",
                                      "annual_demand_mwh"])
    ladder = ladder[ladder.scenario.eq("S3")]
    published = ladder[ladder.weather_year.eq(LADDER_REFERENCE_YEAR)]
    published_mix = published.set_index("logical_site_id").wind_energy_share.reindex(site_ids)
    reproduced_mix = pd.Series(mix_grid[per_year_pick[list(years).index(
        LADDER_REFERENCE_YEAR)]], index=site_ids)
    published_global = (
        ladder.assign(w=ladder.cfe_score * ladder.annual_demand_mwh)
        .groupby("weather_year").apply(lambda b: b.w.sum() / b.annual_demand_mwh.sum(),
                                       include_groups=False))
    frozen_2000 = matrix_frame.loc[str(LADDER_REFERENCE_YEAR)]

    payload = {
        "schema_version": "1.0",
        "method_version": METHOD_VERSION,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "sites": int(n_site),
        "weather_years": [int(v) for v in years],
        "mix_grid_step": float(np.diff(mix_grid).mean()),
        "draws_per_subset_size": int(args.draws),
        "held_out_years_per_draw": HELD_OUT_YEARS,
        "seed": SEED,
        "result_1_1_design_depends_on_weather_year": {
            "sites_whose_optimal_mix_moves_across_weather_years": int(disagree.sum()),
            "share_of_sites": float(disagree.mean()),
            "share_of_demand": float(site_weight[disagree].sum() / site_weight.sum()),
            "mean_within_site_mix_spread": float(mix_spread.mean()),
            "max_within_site_mix_spread": float(mix_spread.max()),
            "demand_weighted_wind_share_min": float(fleet_share.min()),
            "demand_weighted_wind_share_max": float(fleet_share.max()),
            "demand_weighted_wind_share_spread": float(fleet_share.max() - fleet_share.min()),
        },
        "result_1_2_in_sample_is_not_out_of_sample": {
            # Optimism bias mixes two things: a design year that happens to be an
            # easy weather year raises the in-sample score without any
            # overfitting.  Regret is the clean overfitting measure and is
            # reported alongside; the text uses regret.
            "mean_optimism_bias_pp": float(optimism.mean() * 100),
            "max_optimism_bias_pp": float(optimism.max() * 100),
            "design_year_with_max_optimism": int(years[optimism.argmax()]),
            "aggregate_single_year_design_mean_regret_pp": float(regret.mean() * 100),
            "aggregate_single_year_design_max_regret_pp": float(regret.max() * 100),
            "best_single_year_design_by_out_of_sample_mean": int(
                years[off_diagonal_mean.argmax()]),
            "worst_single_year_design_by_out_of_sample_mean": int(
                years[off_diagonal_mean.argmin()]),
            "out_of_sample_mean_spread_pp": float(
                (off_diagonal_mean.max() - off_diagonal_mean.min()) * 100),
            "robust_mean_design_beats_n_single_year_designs": int(
                (matrix_frame.loc["robust_mean_20y"].mean() > square.mean(axis=1)).sum()),
            "robust_gain_over_2000_design_mean_pp": float(
                (matrix_frame.loc["robust_mean_20y"].mean()
                 - matrix_frame.loc[str(LADDER_REFERENCE_YEAR)].mean()) * 100),
        },
        "result_1_2b_the_risk_is_facility_level": {
            "demand_weighted_mean_regret_pp": float(site_mean_regret @ share),
            "demand_weighted_worst_case_regret_pp": float(site_max_regret @ share),
            "worst_case_regret_pp_percentiles": {
                str(q): float(np.percentile(site_max_regret, q))
                for q in (50, 75, 90, 95, 99, 100)},
            "climate_shift_yardstick_pp": float(args.climate_shift_pp),
            "share_of_facilities_worse_than_climate_shift": float(exceeds.mean()),
            "share_of_demand_worse_than_climate_shift": float(share[exceeds].sum()),
            # How much a global aggregate understates what one buyer carries.
            "facility_to_aggregate_ratio": float(
                (site_max_regret @ share) / (site_mean_regret @ share)),
        },
        "result_1_2c_multi_year_design_does_not_fix_the_objective": {
            "excess_worst_week_gap_single_year_design_pct": float(
                tail_single.mean(axis=(0, 1)) @ share),
            "excess_worst_week_gap_robust_design_pct": float(
                tail_robust.mean(axis=0) @ share),
        },
        "result_1_3_convergence": {
            "full_record_designs": full_record,
            "median_regret_pp_by_k": {
                objective: {int(k): float(
                    curve[(curve.design_years == k) & (curve.objective == objective)]
                    .mean_regret_pp.median()) for k in SUBSET_SIZES}
                for objective in OBJECTIVES},
            "worst_year_coverage_by_k": {
                objective: {int(k): float(
                    curve[(curve.design_years == k) & (curve.objective == objective)]
                    .worst_year_coverage.median()) for k in SUBSET_SIZES}
                for objective in OBJECTIVES},
        },
        "hard_checks": {
            "reproduces_published_2000_mix": bool(
                np.allclose(published_mix.to_numpy(), reproduced_mix.to_numpy(), atol=1e-9)),
            "reproduces_published_frozen_2000_coverage": bool(
                np.allclose(frozen_2000.to_numpy(),
                            published_global.reindex(years).to_numpy(), atol=2e-6)),
            "max_abs_coverage_difference_vs_published": float(np.max(np.abs(
                frozen_2000.to_numpy() - published_global.reindex(years).to_numpy()))),
            "robust_mean_design_is_weakly_better_on_mean": bool(
                matrix_frame.loc["robust_mean_20y"].mean() >= square.mean(axis=1).max() - 1e-12),
            "regret_non_negative": bool((regret >= -1e-12).all()),
        },
    }
    payload["all_hard_checks_pass"] = all(payload["hard_checks"].values())
    (args.validation_root / "weather_robust_design.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), "utf-8")
    print(json.dumps({k: v for k, v in payload.items()
                      if k.startswith("result") or k.startswith("hard")
                      or k == "all_hard_checks_pass"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
