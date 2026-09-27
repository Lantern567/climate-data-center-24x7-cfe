# -*- coding: utf-8 -*-
"""Join key between the demand-side site inventory and the supply-side areas.

The two sides were built from different sources and never shared a country
vocabulary: demand carries 151 long-form names (Data Center Watch lineage),
supply carries 198 IRENA `country_area` labels.  Only 117 names match verbatim,
and the 34 that do not hold **66.8% of demand** -- "United States" alone is
61.8%.  Nothing at country resolution can be computed until this is fixed, so
the crosswalk is a hard prerequisite for S3, not housekeeping.

Two jobs:

  1. Map every demand country name to ISO 3166-1 alpha-3, which is what the
     supply side already keys on (`country_area_code`).  Joining on codes rather
     than names means later vocabulary drift on either side cannot silently
     mis-join.
  2. Fill the 90 sites with no `country_for_aggregation` (0.27% of demand).

Name matching is done by hand rather than by fuzzy string distance because two
pairs are actively adversarial to it:

  * "Republic of the Congo" is COG; a substring match pulls COD (Congo DR),
    a different country with ~40x the population.
  * "Guinea" is GIN; substring matching offers GNQ (Equatorial Guinea) and
    GNB (Guinea-Bissau) and not the right answer.

For the 90 unlabelled sites the country is inferred from the site's IANA time
zone, using the modal country among the 8,270 sites that ARE labelled.  That
keeps the vocabulary internally consistent instead of importing an external
gazetteer.  Purity is recorded per site so a downstream analysis can drop the
ambiguous ones; zones with no labelled precedent fall back to an explicit
zone-to-country table.
"""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
COUPLING = ROOT / "data" / "24x7_cfe" / "processed" / "climate_coupling"
REPLAY = (
    ROOT / "data" / "24x7_cfe" / "processed"
    / "wind_solar_supply_20y_fixed_2025" / "replay_results_pv_corrected"
)
DEFAULT_OUTPUT = COUPLING / "demand_supply_country_crosswalk"
METHOD_VERSION = "demand_supply_country_crosswalk_v1"

# Demand-side name -> ISO 3166-1 alpha-3.  Only the names that do not match a
# supply `country_area` verbatim need an entry; the rest resolve by name.
MANUAL_ISO3 = {
    "United States": "USA",
    "United Kingdom": "GBR",
    "Russia": "RUS",
    "Turkey": "TUR",
    "South Korea": "KOR",
    "Taiwan": "TWN",
    "Iceland": "ISL",
    "Vietnam": "VNM",
    "United Arab Emirates": "ARE",
    "Hong Kong": "HKG",
    "Republic of Serbia": "SRB",
    "United Republic of Tanzania": "TZA",
    "Iran": "IRN",
    "Trinidad and Tobago": "TTO",
    "Moldova": "MDA",
    "Democratic Republic of the Congo": "COD",
    "Dominican Republic": "DOM",
    "Ivory Coast": "CIV",
    "New Caledonia": "NCL",
    "Isle of Man": "IMN",
    "Laos": "LAO",
    "Kyrgyzstan": "KGZ",
    "Equatorial Guinea": "GNQ",
    "Republic of the Congo": "COG",   # NOT COD
    "Guinea": "GIN",                  # NOT GNQ / GNB
    "Bosnia and Herzegovina": "BIH",
    "Northern Mariana Islands": "MNP",
    "Curaçao": "CUW",
    "Papua New Guinea": "PNG",
    "Syria": "SYR",
    "Brunei": "BRN",
    "Somaliland": "SOM",              # not an ISO entity; folded into Somalia
    "Saint Barthélemy": "BLM",
    "Antarctica": None,               # no grid, no IRENA capacity: excluded
}

# Time zones held by unlabelled sites with no labelled precedent anywhere.
ZONE_FALLBACK = {
    "Indian/Mayotte": "Mayotte",
    "Asia/Macau": "Macau",
    "Atlantic/Faroe": "Faroe Islands",
    "Arctic/Longyearbyen": "Norway",       # Svalbard, administered by Norway
    "Pacific/Nauru": "Nauru",
    "Atlantic/Cape_Verde": "Cape Verde",
    "Pacific/Funafuti": "Tuvalu",
    "Asia/Jayapura": "Indonesia",
}
FALLBACK_ISO3 = {
    "Mayotte": "MYT", "Macau": "MAC", "Faroe Islands": "FRO", "Norway": "NOR",
    "Nauru": "NRU", "Cape Verde": "CPV", "Tuvalu": "TUV", "Indonesia": "IDN",
}
MIN_ZONE_PURITY = 0.95


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mapping", type=Path, default=COUPLING / "facility_to_era5_grid.csv")
    p.add_argument("--replay-root", type=Path, default=REPLAY)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    sites = pd.read_csv(args.mapping, low_memory=False)
    partition = next((args.replay_root / "hourly_country_supply").glob("weather_year=*"))
    supply = (
        pd.read_parquet(next(partition.glob("*.parquet")),
                        columns=["country_area_code", "country_area"])
        .drop_duplicates().reset_index(drop=True)
    )
    name_to_code = dict(zip(supply.country_area, supply.country_area_code))
    valid_codes = set(supply.country_area_code)

    # ---- 1. fill the unlabelled sites --------------------------------------
    labelled = sites[sites.country_for_aggregation.notna()]
    zone_stats = labelled.groupby("timezone_name").country_for_aggregation.agg(
        modal=lambda s: s.mode().iloc[0],
        purity=lambda s: float(s.value_counts(normalize=True).iloc[0]),
        n="size",
    )
    missing = sites.country_for_aggregation.isna()
    sites["country_fill_method"] = pd.Series("observed", index=sites.index).where(
        ~missing, other=pd.NA
    )
    sites["country_fill_purity"] = pd.Series(1.0, index=sites.index).where(~missing)

    filled = zone_purity = 0
    for index in sites.index[missing]:
        zone = str(sites.at[index, "timezone_name"])
        if zone in zone_stats.index:
            sites.at[index, "country_for_aggregation"] = zone_stats.at[zone, "modal"]
            sites.at[index, "country_fill_purity"] = zone_stats.at[zone, "purity"]
            sites.at[index, "country_fill_method"] = "timezone_modal"
            filled += 1
            zone_purity += zone_stats.at[zone, "purity"] >= MIN_ZONE_PURITY
        elif zone in ZONE_FALLBACK:
            sites.at[index, "country_for_aggregation"] = ZONE_FALLBACK[zone]
            sites.at[index, "country_fill_purity"] = 1.0
            sites.at[index, "country_fill_method"] = "timezone_explicit_table"
            filled += 1
            zone_purity += 1
    still_missing = int(sites.country_for_aggregation.isna().sum())

    # ---- 2. resolve every demand country to ISO3 ---------------------------
    demand_names = sorted(sites.country_for_aggregation.dropna().unique())
    rows = []
    for name in demand_names:
        if name in name_to_code:
            code, how = name_to_code[name], "verbatim_name_match"
        elif name in MANUAL_ISO3:
            code, how = MANUAL_ISO3[name], "manual_iso3"
        elif name in FALLBACK_ISO3:
            code, how = FALLBACK_ISO3[name], "manual_iso3"
        else:
            code, how = None, "unresolved"
        rows.append({
            "demand_country_name": name,
            "supply_country_area_code": code,
            "resolution": how,
            "present_in_supply": bool(code in valid_codes) if code else False,
            "site_count": int((sites.country_for_aggregation == name).sum()),
            "site_mean_it_mw_p50": float(
                sites.loc[sites.country_for_aggregation == name, "mean_it_mw_p50"].sum()
            ),
        })
    crosswalk = pd.DataFrame(rows).sort_values("site_mean_it_mw_p50", ascending=False)

    total_mw = float(sites.mean_it_mw_p50.sum())
    joinable = crosswalk[crosswalk.present_in_supply]
    coverage = float(joinable.site_mean_it_mw_p50.sum() / total_mw)

    sites["supply_country_area_code"] = sites.country_for_aggregation.map(
        dict(zip(crosswalk.demand_country_name, crosswalk.supply_country_area_code))
    )
    sites.to_csv(args.output_root / "facility_to_era5_grid_with_country.csv", index=False)
    crosswalk.to_csv(args.output_root / "country_crosswalk.csv", index=False)

    payload = {
        "schema_version": "1.0",
        "method_version": METHOD_VERSION,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "demand_countries": len(crosswalk),
        "supply_areas": int(len(supply)),
        "sites_total": int(len(sites)),
        "sites_country_filled": filled,
        "sites_filled_at_or_above_purity_threshold": int(zone_purity),
        "sites_still_without_country": still_missing,
        "countries_joinable_to_supply": int(len(joinable)),
        "countries_not_in_supply": crosswalk.loc[
            ~crosswalk.present_in_supply, "demand_country_name"].tolist(),
        "demand_mw_coverage_of_joinable_countries": coverage,
        "demand_mw_excluded": total_mw - float(joinable.site_mean_it_mw_p50.sum()),
        "resolution_counts": crosswalk.resolution.value_counts().to_dict(),
        "hard_checks": {
            "every_site_has_country": still_missing == 0,
            "no_unresolved_names": bool((crosswalk.resolution != "unresolved").all()),
            "coverage_above_0p99": coverage > 0.99,
            "codes_are_valid_supply_areas": bool(
                joinable.supply_country_area_code.isin(valid_codes).all()),
        },
    }
    payload["all_hard_checks_pass"] = all(payload["hard_checks"].values())
    (args.output_root / "country_crosswalk_validation.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), "utf-8"
    )
    print(json.dumps({k: payload[k] for k in (
        "sites_country_filled", "sites_still_without_country",
        "countries_joinable_to_supply", "countries_not_in_supply",
        "demand_mw_coverage_of_joinable_countries", "resolution_counts",
        "hard_checks", "all_hard_checks_pass")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
