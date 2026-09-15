#!/usr/bin/env python3
# Copyright (C) 2026 WUI Mapping Project Authors
# SPDX-License-Identifier: GPL-3.0-only
# Repository version: filesystem paths are loaded through repo_config.py.
# Copy config/paths.example.json to config/paths.json before a new run.
# Raw input data and large intermediate files are stored outside GitHub.

"""Run the accepted Step47D kernel for one reporting unit at 500 m only."""
from __future__ import annotations

from repo_config import portable_path
import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(portable_path("project"))
KERNEL = ROOT / "scripts/47d_patch75_four_state_psz.py"
STATE_NAMES = {
    "AL":"Alabama","AZ":"Arizona","AR":"Arkansas","CA":"California",
    "CO":"Colorado","CT":"Connecticut","DE":"Delaware","DC":"DistrictofColumbia",
    "FL":"Florida","GA":"Georgia","ID":"Idaho","IL":"Illinois","IN":"Indiana",
    "IA":"Iowa","KS":"Kansas","KY":"Kentucky","LA":"Louisiana","ME":"Maine",
    "MD":"Maryland","MA":"Massachusetts","MI":"Michigan","MN":"Minnesota",
    "MS":"Mississippi","MO":"Missouri","MT":"Montana","NE":"Nebraska",
    "NV":"Nevada","NH":"NewHampshire","NJ":"NewJersey","NM":"NewMexico",
    "NY":"NewYork","NC":"NorthCarolina","ND":"NorthDakota","OH":"Ohio",
    "OK":"Oklahoma","OR":"Oregon","PA":"Pennsylvania","RI":"RhodeIsland",
    "SC":"SouthCarolina","SD":"SouthDakota","TN":"Tennessee","TX":"Texas",
    "UT":"Utah","VT":"Vermont","VA":"Virginia","WA":"Washington",
    "WV":"WestVirginia","WI":"Wisconsin","WY":"Wyoming",
}

def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--state",required=True,choices=sorted(STATE_NAMES))
    ap.add_argument("--output",required=True,type=Path)
    a=ap.parse_args()
    spec=importlib.util.spec_from_file_location("step49_kernel",KERNEL)
    if spec is None or spec.loader is None: raise RuntimeError(f"Cannot load {KERNEL}")
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    mod.STATE_CONFIG={a.state:{"name":STATE_NAMES[a.state]}}
    mod.RADII=(500,)
    sys.argv=[str(KERNEL),"--state",a.state,"--output",str(a.output)]
    return int(mod.main())

if __name__=="__main__":
    raise SystemExit(main())
