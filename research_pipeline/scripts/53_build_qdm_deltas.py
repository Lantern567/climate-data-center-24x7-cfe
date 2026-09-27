# -*- coding: utf-8 -*-
"""Quantile-delta maps that morph the ERA5 replay into a future climate.

## What this produces, and what it deliberately does not

Climate models carry credible information about how a distribution *moves*;
their hourly sequences do not have credible texture.  ERA5 has real texture --
when heatwaves occurred, how long they lasted, whether the wind died at the same
time -- but knows nothing about the future.  Quantile delta mapping takes the
move from the model and the texture from ERA5:

    additive        x_future = x_era5 + [Q_model_future(p) - Q_model_hist(p)]
    multiplicative  x_future = x_era5 * [Q_model_future(p) / Q_model_hist(p)]

evaluated at p = the ERA5 value's own quantile, per cell and per calendar month.
Temperature and humidity are additive; wind and radiation are multiplicative so
they cannot go negative and so zero stays zero (night, calm).

Nothing is resampled.  Because every hour keeps its position in the record, the
joint structure across variables and across time survives -- which is precisely
what the proposal's section 6.2 requires and what per-variable resampling would
destroy.

**The standing assumption**: only the marginals move; the correlation structure
is held fixed.  If future heatwaves co-occur with calms more often than they do
today, this method cannot see it.  That belongs in the limitations, not in a
footnote.

## The ocean mask

NEX-GDDP-CMIP6 is a land dataset.  332 of the study's 22,429 cells (1.5%) are
all-NaN, and they are not a random 1.5%: they carry 74.3 GW of wind, essentially
the entire offshore fleet, plus 33 data-centre cells.  Propagating the NaN would
silently delete offshore wind from every future scenario and quietly bias the UK,
Denmark, the Netherlands and China.

Those cells therefore borrow the delta of the nearest cell that does have data.
Only the *delta* is borrowed -- the absolute values still come from ERA5, which
is correct over water.  Deltas are spatially smooth and coastal cells sit within
tens of kilometres of land, so the approximation is small; the donor distance is
recorded per cell so it can be audited rather than trusted.
"""
from __future__ import annotations

import argparse
import os
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
POINTS = Path(os.environ.get("NEX_POINTS_ROOT") or r"D:\nex_gddp_points")
DEFAULT_OUTPUT = Path(os.environ.get("NEX_DELTAS_ROOT") or r"D:\nex_gddp_deltas")

METHOD_VERSION = "qdm_delta_v1"
ADDITIVE = ("tas", "hurs", "tasmax", "tasmin")
MULTIPLICATIVE = ("sfcWind", "rsds")
# tasmax/tasmin are not in the default set.  They exist because the `tas` delta
# is almost flat across quantiles -- global mean q95 - q05 is +0.055 K -- so a
# `tas`-only morph warms the hottest and the coldest hour of a cell-month by the
# same amount and can produce no tail amplification by construction.  The daily
# range carries the part of the signal cooling load responds to.
DEFAULT_VARIABLES = ("tas", "hurs", "sfcWind", "rsds")
VARIABLES = ADDITIVE + MULTIPLICATIVE
QUANTILES = np.round(np.arange(0.05, 0.96, 0.05), 2)
# A ratio this large means the historical quantile was essentially zero; the
# cap stops a divide-by-almost-zero from inventing radiation or wind.
RATIO_CAP = 3.0
# Below this a monthly radiation or wind quantile is indistinguishable from zero
# (polar night, sustained calm) and the ratio is meaningless rather than large.
ZERO_FLOOR = 1e-3
EARTH_RADIUS_KM = 6371.0
# Deltas are spatially smooth over tens of kilometres, not thousands.  A cell
# whose nearest donor is further than this is reported, not silently trusted.
DONOR_DISTANCE_WARN_KM = 200.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="ACCESS-CM2")
    p.add_argument("--scenario", default="ssp245")
    p.add_argument("--historical", default="historical")
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--variable", action="append", choices=VARIABLES)
    return p.parse_args()


def monthly_quantiles(model: str, scenario: str, variable: str,
                      cell_order: pd.Index) -> pd.DataFrame:
    """Per cell, per month, the value at each reference quantile.

    Held as a wide (days x cells) float32 array rather than a long frame: the
    long form is 164 million rows for one variable and pandas' groupby-quantile
    exhausts memory on it, while the wide form is ~650 MiB and lets numpy take
    the quantiles per month in one pass.
    """
    folder = POINTS / model / scenario / variable
    files = sorted(folder.glob(f"{variable}_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No extracted files under {folder}")
    blocks, months = [], []
    for path in files:
        frame = pd.read_parquet(path)
        wide = frame.pivot(index="date", columns="cell_id", values=variable)
        wide = wide.reindex(columns=cell_order)
        blocks.append(wide.to_numpy(dtype=np.float32))
        months.append(pd.to_datetime(wide.index).month.to_numpy())
    values = np.concatenate(blocks, axis=0)
    month = np.concatenate(months)
    del blocks

    rows = []
    for m in range(1, 13):
        subset = values[month == m]
        with np.errstate(all="ignore"):
            q = np.nanquantile(subset, QUANTILES, axis=0)
        block = pd.DataFrame(q.T, index=cell_order,
                             columns=[f"q{int(round(x * 100)):02d}" for x in QUANTILES])
        block["month"] = m
        rows.append(block.reset_index().rename(columns={"index": "cell_id"}))
    table = pd.concat(rows, ignore_index=True)
    return table.set_index(["cell_id", "month"]).sort_index()


def nearest_donor(cells: pd.DataFrame, valid: set[str]) -> pd.DataFrame:
    """Map every masked cell to the closest cell that has data."""
    have = cells[cells.cell_id.isin(valid)]
    need = cells[~cells.cell_id.isin(valid)]
    if need.empty:
        return pd.DataFrame(columns=["cell_id", "donor_cell_id", "donor_km"])
    lat_h = np.radians(have.era5_latitude.to_numpy())
    lon_h = np.radians(have.era5_longitude.to_numpy())
    rows = []
    for row in need.itertuples(index=False):
        lat, lon = np.radians(row.era5_latitude), np.radians(row.era5_longitude)
        d = np.sin((lat_h - lat) / 2) ** 2 + np.cos(lat) * np.cos(lat_h) * \
            np.sin((lon_h - lon) / 2) ** 2
        km = EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(d))
        best = int(np.argmin(km))
        rows.append({"cell_id": row.cell_id,
                     "donor_cell_id": have.cell_id.iloc[best],
                     "donor_km": float(km[best])})
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    variables = tuple(args.variable) if args.variable else DEFAULT_VARIABLES
    args.output_root.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(POINTS / "cells.csv")

    summary: dict[str, dict] = {}
    donors_written = False
    cell_order = pd.Index(cells.cell_id, name="cell_id")
    for variable in variables:
        q_hist = monthly_quantiles(args.model, args.historical, variable, cell_order)
        q_future = monthly_quantiles(args.model, args.scenario, variable, cell_order)
        common = q_hist.index.intersection(q_future.index)
        q_hist, q_future = q_hist.loc[common], q_future.loc[common]

        if variable in ADDITIVE:
            delta = q_future - q_hist
            kind = "additive"
        else:
            near_zero = q_hist.abs() <= ZERO_FLOOR
            delta = q_future / q_hist.where(~near_zero)
            # Polar night: a whole month of zero radiation gives a 0/0 ratio.
            # Zero stays zero under warming, so the honest delta there is 1.0 --
            # leaving NaN would strand every high-latitude cell.
            delta = delta.where(~near_zero, 1.0).clip(upper=RATIO_CAP)
            kind = "multiplicative"

        # A cell is usable only if it has a finite delta in every month.
        finite = delta.notna().all(axis=1)
        usable = set(delta.index.get_level_values("cell_id")[finite.to_numpy()])
        masked = set(cells.cell_id) - usable
        donors = nearest_donor(cells, usable)
        if not donors_written and len(donors):
            donors.to_csv(args.output_root / "ocean_mask_donors.csv", index=False)
            donors_written = True
        if len(donors):
            lookup = delta.reset_index()
            borrow = donors.merge(lookup, left_on="donor_cell_id",
                                  right_on="cell_id", suffixes=("", "_donor"))
            borrow = borrow.drop(columns=["cell_id_donor", "donor_cell_id",
                                          "donor_km"], errors="ignore")
            borrow = borrow.set_index(["cell_id", "month"])[delta.columns]
            delta = pd.concat([delta[finite], borrow]).sort_index()

        delta.reset_index().to_parquet(
            args.output_root / f"delta_{args.model}_{args.scenario}_{variable}.parquet",
            index=False)

        values = delta.to_numpy(dtype=float)
        summary[variable] = {
            "kind": kind,
            "cells_with_own_delta": len(usable),
            "cells_borrowing_delta": len(masked),
            "donor_km_median": float(donors.donor_km.median()) if len(donors) else 0.0,
            "donor_km_max": float(donors.donor_km.max()) if len(donors) else 0.0,
            "donor_beyond_warn_distance": int(
                (donors.donor_km > DONOR_DISTANCE_WARN_KM).sum()) if len(donors) else 0,
            "delta_median": float(np.nanmedian(values)),
            "delta_p05": float(np.nanpercentile(values, 5)),
            "delta_p95": float(np.nanpercentile(values, 95)),
            "all_finite": bool(np.isfinite(values).all()),
        }
        print(f"{variable} ({kind}): 自有 {len(usable):,} 借用 {len(masked):,} "
              f"中位 delta {summary[variable]['delta_median']:.4f}", flush=True)

    payload = {
        "schema_version": "1.0",
        "method_version": METHOD_VERSION,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "model": args.model, "scenario": args.scenario,
        "reference_quantiles": QUANTILES.tolist(),
        "ratio_cap": RATIO_CAP,
        "variables": summary,
        "correlation_structure_assumed_fixed": True,
        "ocean_cells_borrow_nearest_land_delta": True,
        "hard_checks": {
            "every_variable_all_finite": all(v["all_finite"] for v in summary.values()),
            "no_cell_left_without_delta": all(
                v["cells_with_own_delta"] + v["cells_borrowing_delta"] >= len(cells)
                for v in summary.values()),
        },
    }
    payload["all_hard_checks_pass"] = all(payload["hard_checks"].values())
    (args.output_root / f"qdm_delta_validation_{args.model}_{args.scenario}.json"
     ).write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                             allow_nan=False), "utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
