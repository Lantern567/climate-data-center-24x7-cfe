# -*- coding: utf-8 -*-
"""Aggregate the future-climate supply capacity factors to country/global hours.

Script 38 refuses to read a resource store whose metadata `method_version` is
not the one it produced -- a deliberate guard against silently mixing two
provenances into one aggregate.  That guard is correct and worth keeping, so
rather than relaxing it, this wrapper tells script 38 which provenance is
expected for this run and leaves every other check (interval counts, per-file
sha256, energy conservation, leap-year hours) untouched.

Same technique as scripts 45 and 55: load the validated script by path, rebind
the one constant that legitimately differs, and reuse the rest verbatim.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "research_pipeline" / "scripts"

MODEL = os.environ.get("QDM_MODEL", "ACCESS-CM2")
SCENARIO = os.environ.get("QDM_SCENARIO", "ssp245")
FUTURE_RESOURCE_VERSION = f"supply_future_cf_{SCENARIO}_{MODEL}_v1"
DEFAULT_RESOURCE = Path(rf"D:\era5_supply_future_cf_{SCENARIO}")
DEFAULT_OUTPUT = (ROOT / "data" / "24x7_cfe" / "processed"
                  / "wind_solar_supply_20y_fixed_2025"
                  / f"replay_results_future_{SCENARIO}_raw")

_SPEC = importlib.util.spec_from_file_location(
    "supply_replay", SCRIPTS / "38_compute_fixed_2025_supply_replay.py")
REPLAY = importlib.util.module_from_spec(_SPEC)
sys.modules["supply_replay"] = REPLAY
_SPEC.loader.exec_module(REPLAY)

REPLAY.RESOURCE_METHOD_VERSION = FUTURE_RESOURCE_VERSION
REPLAY.METHOD_VERSION = f"fixed_2025_supply_replay_future_{SCENARIO}_{MODEL}_v1"


if __name__ == "__main__":
    if not any(a.startswith("--resource-root") for a in sys.argv[1:]):
        sys.argv.extend(["--resource-root", str(DEFAULT_RESOURCE)])
    if not any(a.startswith("--output-root") for a in sys.argv[1:]):
        sys.argv.extend(["--output-root", str(DEFAULT_OUTPUT)])
    REPLAY.main()
