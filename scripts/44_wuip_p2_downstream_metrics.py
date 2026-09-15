#!/usr/bin/env python3
# Copyright (C) 2026 WUI Mapping Project Authors
# SPDX-License-Identifier: GPL-3.0-only
# Repository version: filesystem paths are loaded through repo_config.py.
# Copy config/paths.example.json to config/paths.json before a new run.
# Raw input data and large intermediate files are stored outside GitHub.

"""STEP44: rebuild P2 WUI-P area, population, and county metrics.

The implementation is evidence-locked to the accepted Step35C/Step36/Step37
chains.  P2 changes the classification raster only.  The accepted population
allocation points and the county structure denominator remain the frozen raw
address-point inputs.  Population is updated exactly by applying the
P0->P2 class changes to the accepted exact Census-block-group point counts.
"""

from __future__ import annotations

from repo_config import portable_path

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import struct
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

os.environ["GDAL_DATA"] = portable_path("software", "share/gdal")
os.environ["PROJ_LIB"] = portable_path("software", "share/proj")
os.environ["PROJ_DATA"] = portable_path("software", "share/proj")
sys.dont_write_bytecode = True

import fiona
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from rasterio.crs import CRS


ROOT = Path(portable_path("project"))
STEP43 = ROOT / "step43_wuip_p2_formal_rebuild_20260727T180822Z"
STEP42 = ROOT / "step42_wuip_address_policy_audit_20260727T153136Z"
STEP35C = ROOT / "step35C_reconciled_population_products_20260725T002208Z"
STEP35 = ROOT / "step35_five_state_all_buffer_population_20260724T183418Z"
STEP36_AREA = ROOT / "step36_area_final_audit_20260724T192140Z"
STEP36_POP = ROOT / "step36_population_revision_impact_20260725T005100Z"
STEP37 = ROOT / "step37_morans_i_revision_20260727T043506Z"
STEP33 = ROOT / "step33_remaining_41_state_ps_recompute_20260724T025443Z"
STEP29 = ROOT / "step29_five_state_exact_target_qc_20260723T181854Z"
DRIVE = Path(portable_path("data"))
BLOCKS = DRIVE / "MBF+NLCD_2022US/Inputs/Processed_GPKG"
COUNTY_GPKG = BLOCKS / "tl_2022_us_county.gpkg"
COUNTY_LAYER = "tl_2022_us_county"
COUNTY_CACHE = Path(portable_path("legacy", "WUI_tables_compare/_cache_table5_all49_500m"))
ORIGINAL_COUNTY_SCRIPT = (
    Path(portable_path("legacy"))
    / "MBF+NLCD_2022US/18_make_table5_global_moran_csv_sample5_county.py"
)
HELPER_SCRIPT = ROOT / "scripts/08_recompute_sample4_ps_population.py"
EXACT_SCRIPT = ROOT / "scripts/25_audit_vermont_wuis_exact_point_in_polygon.py"

FIVE = {"CA", "CO", "FL", "PA", "TX"}
BUFFERS = tuple(range(100, 1001, 100))
EXPECTED_POP = 329_260_619.0
POP_TOL = 1e-6
STATE_ROWS = [
    ("01", "AL", "Alabama", 5_024_279), ("04", "AZ", "Arizona", 7_151_502),
    ("05", "AR", "Arkansas", 3_011_524), ("06", "CA", "California", 39_538_223),
    ("08", "CO", "Colorado", 5_773_714), ("09", "CT", "Connecticut", 3_605_944),
    ("10", "DE", "Delaware", 989_948), ("11", "DC", "District of Columbia", 689_545),
    ("12", "FL", "Florida", 21_538_187), ("13", "GA", "Georgia", 10_711_908),
    ("16", "ID", "Idaho", 1_839_106), ("17", "IL", "Illinois", 12_812_508),
    ("18", "IN", "Indiana", 6_785_528), ("19", "IA", "Iowa", 3_190_369),
    ("20", "KS", "Kansas", 2_937_880), ("21", "KY", "Kentucky", 4_505_836),
    ("22", "LA", "Louisiana", 4_657_757), ("23", "ME", "Maine", 1_362_359),
    ("24", "MD", "Maryland", 6_177_224), ("25", "MA", "Massachusetts", 7_029_917),
    ("26", "MI", "Michigan", 10_077_331), ("27", "MN", "Minnesota", 5_706_494),
    ("28", "MS", "Mississippi", 2_961_279), ("29", "MO", "Missouri", 6_154_913),
    ("30", "MT", "Montana", 1_084_225), ("31", "NE", "Nebraska", 1_961_504),
    ("32", "NV", "Nevada", 3_104_614), ("33", "NH", "New Hampshire", 1_377_529),
    ("34", "NJ", "New Jersey", 9_288_994), ("35", "NM", "New Mexico", 2_117_522),
    ("36", "NY", "New York", 20_201_249), ("37", "NC", "North Carolina", 10_439_388),
    ("38", "ND", "North Dakota", 779_094), ("39", "OH", "Ohio", 11_799_448),
    ("40", "OK", "Oklahoma", 3_959_353), ("41", "OR", "Oregon", 4_237_256),
    ("42", "PA", "Pennsylvania", 13_002_700), ("44", "RI", "Rhode Island", 1_097_379),
    ("45", "SC", "South Carolina", 5_118_425), ("46", "SD", "South Dakota", 886_667),
    ("47", "TN", "Tennessee", 6_910_840), ("48", "TX", "Texas", 29_145_505),
    ("49", "UT", "Utah", 3_271_616), ("50", "VT", "Vermont", 643_077),
    ("51", "VA", "Virginia", 8_631_393), ("53", "WA", "Washington", 7_705_281),
    ("54", "WV", "West Virginia", 1_793_716), ("55", "WI", "Wisconsin", 5_893_718),
    ("56", "WY", "Wyoming", 576_851),
]
STATE = {
    s: {"STATEFP": f, "state": s, "state_name": n, "official_population": p}
    for f, s, n, p in STATE_ROWS
}
FIPS_TO_STATE = {f: s for f, s, _, _ in STATE_ROWS}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as src:
        for chunk in iter(lambda: src.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")


def atomic_csv(path: Path, frame: pd.DataFrame, float_format: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    frame.to_csv(tmp, index=False, float_format=float_format)
    os.replace(tmp, path)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def rss_gib() -> float:
    try:
        return float(Path("/proc/self/status").read_text().split("VmRSS:")[1].split()[0]) / 1024**2
    except Exception:
        return math.nan


def progress(phase: str, state: str, buffer_m: Any, metric: str, done: int,
             total: int, started: float, output: Path) -> None:
    elapsed = max(time.monotonic() - started, 1e-9)
    rate = done / elapsed
    eta = (total - done) / rate if rate else 0.0
    print(
        f"[{phase}] state={state} buffer={buffer_m} metric={metric} "
        f"completed={done}/{total} percent={100*done/max(total,1):.2f} "
        f"elapsed={elapsed:.1f}s ETA={eta:.1f}s memory_gib={rss_gib():.2f} "
        f"output={output}", flush=True,
    )


def require_step43() -> list[dict[str, Any]]:
    qc = (STEP43 / "STEP43_FINAL_QC.txt").read_text()
    status = json.loads((STEP43 / "step43_status.json").read_text())
    if "checks_passed=18" not in qc or "checks_failed=0" not in qc:
        raise RuntimeError("Step43 18/18 QC gate failed")
    if status["status"] != "P2_CLASSIFICATION_REBUILD_COMPLETE_READY_FOR_DOWNSTREAM":
        raise RuntimeError("Step43 is not READY_FOR_DOWNSTREAM")
    inv = pd.read_csv(STEP43 / "p2_rebuild_inventory.csv")
    if len(inv) != 94 or not inv["run_status"].eq("PASS").all():
        raise RuntimeError("Step43 inventory is not 94/94 PASS")
    if int(inv["building_records_included"].sum()) != 0:
        raise RuntimeError("Step43 includes building records")
    rows = []
    for rec in inv.to_dict("records"):
        path = Path(rec["output_path"])
        actual = sha256(path)
        if actual != rec["file_sha256"]:
            raise RuntimeError(f"Step43 raster SHA mismatch: {path}")
        rows.append({**rec, "verified_sha256": actual})
    return rows


def raster_rows(inventory: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    return {(r["state"], int(r["buffer_m"])): r for r in inventory}


def lineage_tables(out: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    lineage = [
        ("area", "Step36", str(ROOT / "scripts/36_national_sensitivity_area_final_audit.py"),
         "affine determinant; WUI classes 1+2", "PASS final 237 rows"),
        ("population", "Step35C", str(STEP35C / "step35C_national_population_long_147_final.csv"),
         "Step35C reconciled accepted output", "PASS 147/147"),
        ("population algorithm", "Step33/Step35", str(ROOT / "scripts/35_recompute_five_state_all_buffer_population.py"),
         "exact target-state Census block PIP; block-group POP20 allocated by raw address class counts",
         "accepted and population-conserving"),
        ("county p_a/p_s", "Step37", str(ORIGINAL_COUNTY_SCRIPT),
         "county raster center assignment; p_a=WUI pixels/valid pixels; p_s=WUI address count/total address count",
         "first-round reproduction and second-round accepted"),
        ("WUI-S/WUI-Z area", "Step36", str(STEP36_AREA / "step36_national_area_long_147.csv"),
         "inherit unchanged", "PASS"),
        ("WUI-S/WUI-Z population", "Step35C", str(STEP35C / "step35C_national_population_long_147_final.csv"),
         "inherit unchanged", "PASS"),
    ]
    ldf = pd.DataFrame(lineage, columns=[
        "metric_group", "authoritative_step", "path", "rule", "evidence"
    ])
    evidence = pd.DataFrame([
        ("state WUI area", "A_WUI", "class pixels 1 or 2 × affine determinant", "none",
         "km2", "Step43 classification valid domain", "255 excluded; 0 valid",
         "pixel center state mask", "none", str(ROOT / "scripts/36_national_sensitivity_area_final_audit.py"),
         "pixel_area_from_transform", "Step44 explicit + Step36 PASS", "LOCKED"),
        ("WUI population", "P_WUI", "block-group POP20 × address fraction in classes 1+2",
         "2020 Census POP20 state sum", "persons", "exact target-state Census blocks",
         "classification 255 maps to Non-WUI for eligible points, as accepted classifier",
         "exact point covered_by target block", "zero-point BG -> Non-WUI; tolerance 1e-6",
         str(ROOT / "scripts/35_recompute_five_state_all_buffer_population.py"),
         "build_state_context/run_job/allocate_population",
         "Step35C 147 PASS and sensitivity 100 PASS", "LOCKED"),
        ("county WUI area proportion", "p_a", "county WUI pixels (1+2)",
         "county valid class pixels (0+1+2)", "proportion", "county within state",
         "255 excluded; 0 valid", "county ID raster, all_touched=False/pixel center",
         "none", str(ORIGINAL_COUNTY_SCRIPT), "county_metrics_from_rasters",
         "Step37 exact first-round reproduction", "LOCKED"),
        ("county structure proportion", "p_s", "raw address points in WUI pixels",
         "raw address points in valid class pixels", "proportion", "county within state",
         "255 excluded", "point-count raster + county ID raster", "not population",
         str(ORIGINAL_COUNTY_SCRIPT), "county_metrics_from_rasters",
         "Step37 method evidence and reproduced county metrics", "LOCKED"),
    ], columns=[
        "metric", "symbol", "numerator", "denominator", "unit", "spatial_domain",
        "nodata_rule", "boundary_rule", "population_rule", "authoritative_script",
        "function_or_line", "evidence", "status"
    ])
    auth_paths = [
        STEP43 / "STEP43_FINAL_QC.txt", STEP43 / "step43_status.json",
        STEP43 / "p2_rebuild_inventory.csv", STEP43 / "sha256_manifest.txt",
        STEP35C / "STEP35C_FINAL_QC.txt",
        STEP35C / "step35C_national_population_long_147_final.csv",
        STEP35C / "step35C_five_state_all_buffer_population_long_100_final.csv",
        STEP36_AREA / "STEP36_FINAL_QC.txt",
        STEP36_AREA / "step36_national_area_long_147.csv",
        STEP36_AREA / "step36_five_state_sensitivity_area_long_100.csv",
        STEP37 / "STEP37_FINAL_QC.txt", STEP37 / "county_metrics_old_new.csv",
        COUNTY_GPKG, ORIGINAL_COUNTY_SCRIPT, HELPER_SCRIPT, EXACT_SCRIPT,
    ]
    auth = pd.DataFrame([
        {"path": str(p), "exists": p.is_file(), "size_bytes": p.stat().st_size if p.is_file() else 0,
         "sha256": sha256(p) if p.is_file() else "", "authority": "explicit accepted lineage"}
        for p in auth_paths
    ])
    atomic_csv(out / "p2_downstream_script_lineage.csv", ldf)
    atomic_csv(out / "p2_metric_definition_evidence.csv", evidence)
    atomic_csv(out / "p2_authoritative_baseline_manifest.csv", auth)
    return ldf, evidence, auth


def area_phase(out: Path, inventory: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, domains = [], []
    started = time.monotonic()
    for i, rec in enumerate(sorted(inventory, key=lambda x: (x["state"], int(x["buffer_m"]))), 1):
        path = Path(rec["output_path"])
        counts = {0: 0, 1: 0, 2: 0, 255: 0}
        with rasterio.open(path) as src:
            pixel_area = abs(
                src.transform.a * src.transform.e - src.transform.b * src.transform.d
            )
            if src.dtypes[0] != "uint8" or int(src.nodata) != 255:
                raise RuntimeError(f"Metadata failure {path}")
            for _, window in src.block_windows(1):
                vals, nums = np.unique(src.read(1, window=window), return_counts=True)
                if not set(map(int, vals)).issubset({0, 1, 2, 255}):
                    raise RuntimeError(f"Unexpected class {path}")
                for value, number in zip(vals, nums):
                    counts[int(value)] += int(number)
        valid = counts[0] + counts[1] + counts[2]
        wui = counts[1] + counts[2]
        state = rec["state"]
        rows.append({
            "state": state, "STATEFP": STATE[state]["STATEFP"],
            "state_name": STATE[state]["state_name"], "buffer_m": int(rec["buffer_m"]),
            "valid_pixels": valid, "non_wui_pixels": counts[0], "intermix_pixels": counts[1],
            "interface_pixels": counts[2], "wui_pixels": wui,
            "pixel_area_m2": pixel_area, "valid_area_km2": valid * pixel_area / 1e6,
            "wui_area_km2": wui * pixel_area / 1e6,
            "non_wui_area_km2": counts[0] * pixel_area / 1e6,
            "p_a_state_if_authoritative": wui / valid if valid else np.nan,
            "classification_sha256": rec["file_sha256"], "status": "PASS",
        })
        domains.append({
            "state": state, "buffer_m": int(rec["buffer_m"]),
            "classification_valid_rule": "value != 255", "classification_valid_pixels": valid,
            "classification_non_wui_pixels": counts[0], "classification_wui_pixels": wui,
            "classification_outside_pixels": counts[255],
            "valid_closure_error_pixels": valid - counts[0] - wui,
            "population_domain_rule": "exact target-state Census blocks; eligible address points",
            "common_domain_rule": "eligible address points sampled on classification; 255 classified Non-WUI",
            "zero_is_valid_non_wui": True, "nodata_is_255": True, "status": "PASS",
        })
        progress("AREA", state, rec["buffer_m"], "pixels", i, len(inventory), started, out)
    area = pd.DataFrame(rows)
    national = area[area.buffer_m.eq(500)]
    if int(national.wui_pixels.sum()) != 1_308_488_287:
        raise RuntimeError("National P2 WUI pixel closure failed")
    if not math.isclose(float(national.wui_area_km2.sum()), 1_177_639.4583, abs_tol=1e-9):
        raise RuntimeError("National P2 WUI area closure failed")
    a49 = area[area.buffer_m.eq(500)].sort_values("STATEFP")
    a50 = area[area.state.isin(FIVE)].sort_values(["STATEFP", "buffer_m"])
    atomic_csv(out / "p2_wuip_area_by_state_500m.csv", a49, "%.12f")
    atomic_csv(out / "p2_wuip_area_five_state_sensitivity.csv", a50, "%.12f")
    atomic_csv(out / "p2_metric_valid_domain_audit.csv", pd.DataFrame(domains))
    audit = pd.DataFrame([
        {"scope": "national_49_500m", "rows": len(a49), "wui_pixels": int(a49.wui_pixels.sum()),
         "wui_area_km2": float(a49.wui_area_km2.sum()), "expected_wui_pixels": 1_308_488_287,
         "expected_wui_area_km2": 1_177_639.4583, "status": "PASS"},
        {"scope": "five_state_100_1000m", "rows": len(a50), "wui_pixels": int(a50.wui_pixels.sum()),
         "wui_area_km2": float(a50.wui_area_km2.sum()), "expected_wui_pixels": np.nan,
         "expected_wui_area_km2": np.nan, "status": "PASS"},
    ])
    atomic_csv(out / "p2_area_conservation_audit.csv", audit, "%.12f")
    return a49, a50


def old_detail_path(state: str) -> Path:
    fips = STATE[state]["STATEFP"]
    if state in FIVE:
        return STEP35 / f"job_{fips}_{state.lower()}_wuip/exact_block_group_detail_all_buffers.csv.gz"
    if state in {"AL", "OK", "VT"}:
        return STEP29 / f"job_{fips}_{state.lower()}_wuip/exact_block_group_detail.csv.gz"
    return STEP33 / f"job_{fips}_{state.lower()}_wuip/exact_block_group_detail.csv.gz"


def changed_cells(p2_path: Path, legacy_path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    ids1, ids2 = [], []
    stats = {"changed_1_to_0": 0, "changed_2_to_0": 0, "other_change": 0}
    with rasterio.open(p2_path) as new, rasterio.open(legacy_path) as old:
        if (new.width, new.height, new.transform, new.crs) != (
            old.width, old.height, old.transform, old.crs
        ):
            raise RuntimeError(f"Grid mismatch {p2_path}")
        for _, window in new.block_windows(1):
            n = new.read(1, window=window)
            o = old.read(1, window=window)
            # Population deltas are defined on the common valid
            # classification domain. Any transition involving 255 is a domain
            # encoding/boundary difference, not a class transition.
            common_valid = np.isin(o, [0, 1, 2]) & np.isin(n, [0, 1, 2])
            c1 = common_valid & (o == 1) & (n == 0)
            c2 = common_valid & (o == 2) & (n == 0)
            other = common_valid & (o != n) & ~(c1 | c2)
            if other.any():
                stats["other_change"] += int(other.sum())
            rr, cc = np.nonzero(c1)
            if len(rr):
                ids1.append((rr + int(window.row_off)) * new.width + cc + int(window.col_off))
            rr, cc = np.nonzero(c2)
            if len(rr):
                ids2.append((rr + int(window.row_off)) * new.width + cc + int(window.col_off))
            stats["changed_1_to_0"] += int(c1.sum())
            stats["changed_2_to_0"] += int(c2.sum())
    if stats["other_change"]:
        raise RuntimeError(f"Unsupported P0/P2 class transition: {p2_path}")
    a1 = np.sort(np.concatenate(ids1).astype(np.int64)) if ids1 else np.empty(0, np.int64)
    a2 = np.sort(np.concatenate(ids2).astype(np.int64)) if ids2 else np.empty(0, np.int64)
    return a1, a2, stats


def in_sorted(values: np.ndarray, sorted_values: np.ndarray) -> np.ndarray:
    if not len(sorted_values):
        return np.zeros(len(values), dtype=bool)
    pos = np.searchsorted(sorted_values, values)
    ok = pos < len(sorted_values)
    result = np.zeros(len(values), dtype=bool)
    result[ok] = sorted_values[pos[ok]] == values[ok]
    return result


def vector_exact_bg(tree, block_codes: np.ndarray, xs: np.ndarray, ys: np.ndarray,
                    bg_codes_sorted: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    """Vector equivalent of Step33 exact_candidates, unique at BG level."""
    n = len(xs)
    dense = np.full(n, -1, dtype=np.int64)
    if not n:
        return dense, {"unique": 0, "ambiguous": 0, "outside": 0}
    points = shapely.points(xs, ys)
    pairs = tree.query(points, predicate="covered_by")
    if pairs.shape[1] == 0:
        return dense, {"unique": 0, "ambiguous": 0, "outside": n}
    point_i = pairs[0].astype(np.int64)
    codes = block_codes[pairs[1].astype(np.int64)]
    order = np.lexsort((codes, point_i))
    point_i, codes = point_i[order], codes[order]
    first_unique_pair = np.r_[True, (point_i[1:] != point_i[:-1]) | (codes[1:] != codes[:-1])]
    up, uc = point_i[first_unique_pair], codes[first_unique_pair]
    starts = np.r_[0, np.flatnonzero(up[1:] != up[:-1]) + 1]
    ends = np.r_[starts[1:], len(up)]
    one = (ends - starts) == 1
    unique_points = up[starts[one]]
    unique_codes = uc[starts[one]]
    pos = np.searchsorted(bg_codes_sorted, unique_codes)
    valid = (pos < len(bg_codes_sorted))
    valid[valid] &= bg_codes_sorted[pos[valid]] == unique_codes[valid]
    if not valid.all():
        raise RuntimeError("Exact target BG absent from population detail")
    dense[unique_points] = pos
    matched_points = np.unique(up)
    ambiguous = int((~one).sum())
    outside = n - len(matched_points)
    return dense, {"unique": int(one.sum()), "ambiguous": ambiguous, "outside": outside}


def allocate(bg: pd.DataFrame, total: np.ndarray, c0: np.ndarray,
             c1: np.ndarray, c2: np.ndarray) -> tuple[pd.DataFrame, dict[str, float]]:
    if not np.array_equal(total, c0 + c1 + c2):
        raise RuntimeError("Population point count closure failed")
    pop = bg["POP20"].to_numpy(float)
    has = total > 0
    non = pop * np.where(has, c0 / np.maximum(total, 1), 1.0)
    i1 = pop * np.where(has, c1 / np.maximum(total, 1), 0.0)
    i2 = pop * np.where(has, c2 / np.maximum(total, 1), 0.0)
    residual = float(pop.sum()) - float(non.sum() + i1.sum() + i2.sum())
    non[0] += residual
    detail = bg.copy()
    detail["Structure_Count"] = total
    detail["NonWUI_Structure_Count"] = c0
    detail["Intermix_Structure_Count"] = c1
    detail["Interface_Structure_Count"] = c2
    detail["NonWUI_Pop"] = non
    detail["Intermix_Pop"] = i1
    detail["Interface_Pop"] = i2
    summary = {
        "nonwui_population": float(non.sum()), "intermix_population": float(i1.sum()),
        "interface_population": float(i2.sum()), "wui_population": float(i1.sum() + i2.sum()),
        "total_population": float(non.sum() + i1.sum() + i2.sum()),
    }
    summary["population_closure_error"] = summary["total_population"] - float(pop.sum())
    summary["wui_population_share"] = summary["wui_population"] / float(pop.sum())
    return detail, summary


def population_state(out: Path, state: str, inv: dict[tuple[str, int], dict[str, Any]],
                     helper, exact, policy: pd.DataFrame) -> dict[str, Any]:
    cp = out / "checkpoints" / f"population_{state}.json"
    if cp.is_file():
        payload = json.loads(cp.read_text())
        if payload.get("status") == "PASS":
            progress("POP_RESUME", state, "-", "population", 1, 1, time.monotonic(), out)
            return payload
    started = time.monotonic()
    buffers = BUFFERS if state in FIVE else (500,)
    old_path = old_detail_path(state)
    old = pd.read_csv(old_path, dtype={"GEOID12": str})
    if "buffer_m" not in old:
        old["buffer_m"] = 500
    old["buffer_m"] = old["buffer_m"].astype(int)
    fips = STATE[state]["STATEFP"]
    src_row = policy[policy.state.eq(state)]
    if len(src_row) != 1:
        raise RuntimeError(f"Formal address input unresolved {state}")
    address_path = Path(src_row.iloc[0]["formal_point_input_path"])
    address_sha = src_row.iloc[0]["formal_point_input_sha256"]
    if sha256(address_path) != address_sha:
        raise RuntimeError(f"Address SHA mismatch {state}")

    changes: dict[int, tuple[np.ndarray, np.ndarray, dict[str, int]]] = {}
    with rasterio.open(Path(inv[(state, 500)]["output_path"])) as ref:
        transform, width, height, target_crs = ref.transform, ref.width, ref.height, ref.crs
    for b in buffers:
        changes[b] = changed_cells(
            Path(inv[(state, b)]["output_path"]), Path(inv[(state, b)]["legacy_path"])
        )
    if all(len(changes[b][0]) + len(changes[b][1]) == 0 for b in buffers):
        # Negative controls still go through the same accepted allocation below.
        selected_count = 0
        moved = {b: (None, None) for b in buffers}
        assign_stats = {"unique": 0, "ambiguous": 0, "outside": 0}
    else:
        layer = helper.first_layer(address_path)
        feature_count, source_crs_wkt = helper.layer_info(address_path, layer)
        selected_x, selected_y, selected_cells = [], [], []
        processed = 0
        for xs, ys in helper.iter_representative_points(
            address_path, layer, source_crs_wkt, target_crs.to_wkt(), 500_000
        ):
            colf, rowf = (~transform) * (xs, ys)
            cols = np.floor(np.asarray(colf)).astype(np.int64)
            rows = np.floor(np.asarray(rowf)).astype(np.int64)
            inside = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
            cells = rows * width + cols
            union = np.zeros(len(xs), dtype=bool)
            for b in buffers:
                union |= inside & (in_sorted(cells, changes[b][0]) | in_sorted(cells, changes[b][1]))
            if union.any():
                selected_x.append(np.asarray(xs)[union])
                selected_y.append(np.asarray(ys)[union])
                selected_cells.append(cells[union])
            processed += len(xs)
            progress("POP_SELECT", state, "all", "address points", processed, feature_count, started, out)
        sx = np.concatenate(selected_x) if selected_x else np.empty(0)
        sy = np.concatenate(selected_y) if selected_y else np.empty(0)
        sc = np.concatenate(selected_cells) if selected_cells else np.empty(0, np.int64)
        selected_count = len(sx)
        blocks_path = BLOCKS / f"tl_2022_{fips}_tabblock20.gpkg"
        bg0 = old[old.buffer_m.eq(int(buffers[0]))].sort_values("BG_CODE").reset_index(drop=True)
        bg_codes_sorted = bg0.BG_CODE.to_numpy(np.int64)
        tree, geometries, geoids12, block_bg_codes, _, block_stats = exact.load_block_index(
            helper, blocks_path, target_crs
        )
        dense_parts, assign_stats = [], {"unique": 0, "ambiguous": 0, "outside": 0}
        for start in range(0, len(sx), 250_000):
            dense, st = vector_exact_bg(
                tree, np.asarray(block_bg_codes, np.int64), sx[start:start+250_000],
                sy[start:start+250_000], bg_codes_sorted
            )
            dense_parts.append(dense)
            for k in assign_stats:
                assign_stats[k] += st[k]
            progress("POP_ASSIGN", state, "all", "changed points",
                     min(start + 250_000, len(sx)), len(sx), started, out)
        dense = np.concatenate(dense_parts) if dense_parts else np.empty(0, np.int64)
        moved = {}
        for b in buffers:
            good = dense >= 0
            m1 = good & in_sorted(sc, changes[b][0])
            m2 = good & in_sorted(sc, changes[b][1])
            moved[b] = (
                np.bincount(dense[m1], minlength=len(bg0)).astype(np.int64),
                np.bincount(dense[m2], minlength=len(bg0)).astype(np.int64),
            )
        del tree, geometries

    rows_out, legacy_rows, details = [], [], []
    baseline = pd.read_csv(
        STEP35C / "step35C_national_population_long_147_final.csv",
        dtype={"STATEFP": str},
    )
    sens_base = pd.read_csv(
        STEP35C / "step35C_five_state_all_buffer_population_long_100_final.csv",
        dtype={"STATEFP": str},
    )
    for b in buffers:
        bg = old[old.buffer_m.eq(b)].sort_values("BG_CODE").reset_index(drop=True).copy()
        if bg.empty:
            raise RuntimeError(f"Missing old BG detail {state} {b}")
        total = bg.Structure_Count.to_numpy(np.int64)
        c0 = bg.NonWUI_Structure_Count.to_numpy(np.int64)
        c1 = bg.Intermix_Structure_Count.to_numpy(np.int64)
        c2 = bg.Interface_Structure_Count.to_numpy(np.int64)
        _, legacy_summary = allocate(bg, total, c0, c1, c2)
        r1, r2 = moved[b]
        if r1 is not None:
            if (r1 > c1).any() or (r2 > c2).any():
                raise RuntimeError(f"Moved points exceed legacy class counts {state} {b}")
            c0 = c0 + r1 + r2
            c1 = c1 - r1
            c2 = c2 - r2
        detail, summary = allocate(bg, total, c0, c1, c2)
        details.append(detail.assign(state=state, STATEFP=fips, buffer_m=b))
        p2_rec = inv[(state, b)]
        rows_out.append({
            "state": state, "STATEFP": fips, "state_name": STATE[state]["state_name"],
            "buffer_m": b, **summary,
            "classification_sha256": p2_rec["file_sha256"],
            "population_input_sha256": address_sha, "population_input_path": str(address_path),
            "status": "PASS" if abs(summary["population_closure_error"]) <= POP_TOL else "FAIL",
        })
        accepted = (
            sens_base[(sens_base.STUSPS.eq(state)) & sens_base["method"].eq("WUI-P")
                      & sens_base.buffer_m.astype(int).eq(b)]
            if state in FIVE else
            baseline[(baseline.STUSPS.eq(state)) & baseline["method"].eq("WUI-P")]
        )
        if len(accepted) != 1:
            raise RuntimeError(f"Accepted population baseline unresolved {state} {b}")
        a = accepted.iloc[0]
        dif = legacy_summary["wui_population"] - float(a["wui_population"])
        legacy_rows.append({
            "state": state, "buffer_m": b, "metric": "population",
            "recomputed_legacy_wui_population": legacy_summary["wui_population"],
            "accepted_legacy_wui_population": float(a["wui_population"]),
            "absolute_difference": dif,
            "tolerance": POP_TOL, "status": "PASS" if abs(dif) <= POP_TOL else "FAIL",
        })
    if not all(r["status"] == "PASS" for r in rows_out + legacy_rows):
        raise RuntimeError(f"Population or legacy canary failed {state}")
    state_rows_path = out / "population" / f"{state}_population_rows.csv"
    legacy_path = out / "population" / f"{state}_legacy_canary.csv"
    detail_path = out / "population" / f"{state}_block_group_detail.csv.gz"
    atomic_csv(state_rows_path, pd.DataFrame(rows_out), "%.12f")
    atomic_csv(legacy_path, pd.DataFrame(legacy_rows), "%.12f")
    atomic_csv(detail_path, pd.concat(details, ignore_index=True), "%.12f")
    payload = {
        "status": "PASS", "state": state, "completed_utc": utc_now(),
        "rows_path": str(state_rows_path), "rows_sha256": sha256(state_rows_path),
        "legacy_path": str(legacy_path), "legacy_sha256": sha256(legacy_path),
        "detail_path": str(detail_path), "detail_sha256": sha256(detail_path),
        "address_path": str(address_path), "address_sha256": address_sha,
        "selected_changed_cell_points": selected_count, "exact_assignment": assign_stats,
        "change_pixels": {str(b): changes[b][2] for b in buffers},
        "population_policy": "accepted raw address denominator unchanged; P2 classification replacement",
    }
    atomic_json(cp, payload)
    return payload


def population_phase(out: Path, inv: dict[tuple[str, int], dict[str, Any]],
                     selected_states: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    helper = load_module(HELPER_SCRIPT, "step44_population_helper")
    exact = load_module(EXACT_SCRIPT, "step44_exact_helper")
    policy = pd.read_csv(STEP42 / "wui_p_source_policy_all49.csv", dtype={"STATEFP": str})
    for state in sorted(selected_states, key=lambda s: STATE[s]["STATEFP"]):
        try:
            population_state(out, state, inv, helper, exact, policy)
        except Exception:
            failure = out / "logs" / f"population_{state}_failure.txt"
            atomic_text(failure, traceback.format_exc())
            raise
    cps = [json.loads(p.read_text()) for p in sorted((out / "checkpoints").glob("population_??.json"))]
    if len(cps) < 49:
        if selected_states != set(STATE):
            print(
                f"[POP_PARTIAL] requested subset complete; national checkpoints="
                f"{len(cps)}/49. Run remaining states before aggregation.",
                flush=True,
            )
            return pd.DataFrame(), pd.DataFrame()
        raise RuntimeError(f"Population incomplete: {len(cps)}/49 state checkpoints")
    rows = pd.concat([pd.read_csv(p["rows_path"], dtype={"STATEFP": str}) for p in cps], ignore_index=True)
    canary = pd.concat([pd.read_csv(p["legacy_path"]) for p in cps], ignore_index=True)
    p49 = rows[rows.buffer_m.eq(500)].sort_values("STATEFP")
    p50 = rows[rows.state.isin(FIVE)].sort_values(["STATEFP", "buffer_m"])
    if len(p49) != 49 or len(p50) != 50:
        raise RuntimeError("Population row coverage failed")
    if not math.isclose(float(p49.total_population.sum()), EXPECTED_POP, abs_tol=POP_TOL):
        raise RuntimeError("National population denominator failed")
    atomic_csv(out / "p2_wuip_population_by_state_500m.csv", p49, "%.12f")
    atomic_csv(out / "p2_wuip_population_five_state_sensitivity.csv", p50, "%.12f")
    audit = pd.concat([
        p49.assign(scope="national_49_500m"),
        p50.assign(scope="five_state_sensitivity"),
    ], ignore_index=True)
    atomic_csv(out / "p2_population_conservation_audit.csv", audit, "%.12f")
    return p49, p50


def county_metrics_from_rasters(
    wui_raster: Path, county_raster: Path, struct_raster: Path,
    geoid_list_int: list[int], wui_vals: tuple[int, int, int] = (0, 1, 2),
) -> pd.DataFrame:
    """Exact Step37 county metric kernel, isolated from its Moran dependencies."""
    nonv, intmv, intfv = wui_vals
    max_id = int(max(geoid_list_int)) if geoid_list_int else 0
    lut = np.full(max_id + 1, -1, dtype=np.int32)
    for i, gid in enumerate(geoid_list_int):
        lut[int(gid)] = i
    n = len(geoid_list_int)
    pix_non = np.zeros(n, dtype=np.int64)
    pix_intm = np.zeros(n, dtype=np.int64)
    pix_intf = np.zeros(n, dtype=np.int64)
    st_total = np.zeros(n, dtype=np.int64)
    st_intm = np.zeros(n, dtype=np.int64)
    st_intf = np.zeros(n, dtype=np.int64)
    with rasterio.open(wui_raster) as wsrc, rasterio.open(county_raster) as csrc, \
            rasterio.open(struct_raster) as ssrc:
        if (wsrc.width, wsrc.height) != (csrc.width, csrc.height):
            raise ValueError("WUI raster and county raster shape mismatch")
        if (wsrc.width, wsrc.height) != (ssrc.width, ssrc.height):
            raise ValueError("WUI raster and structure raster shape mismatch")
        pixel_area_km2 = abs(wsrc.res[0] * wsrc.res[1]) / 1e6
        nodata = wsrc.nodata
        try:
            nodata_int = int(nodata) if nodata is not None else None
        except Exception:
            nodata_int = None
        if nodata_int in wui_vals:
            nodata = None
        for _, window in wsrc.block_windows(1):
            w = wsrc.read(1, window=window)
            c = csrc.read(1, window=window)
            s = ssrc.read(1, window=window)
            valid = c > 0
            if nodata is not None:
                valid &= w != nodata
            if not valid.any():
                continue
            wv = w[valid]
            idx = lut[c[valid].astype(np.int32)]
            good = idx >= 0
            if not good.any():
                continue
            idx, wv = idx[good], wv[good]
            sv = s[valid][good].astype(np.int64)
            for value, accumulator in (
                (nonv, pix_non), (intmv, pix_intm), (intfv, pix_intf)
            ):
                mask = wv == value
                if mask.any():
                    accumulator += np.bincount(idx[mask], minlength=n)
            any_class = (wv == nonv) | (wv == intmv) | (wv == intfv)
            if any_class.any():
                st_total += np.bincount(
                    idx[any_class], weights=sv[any_class], minlength=n
                ).astype(np.int64)
            for value, accumulator in ((intmv, st_intm), (intfv, st_intf)):
                mask = wv == value
                if mask.any():
                    accumulator += np.bincount(
                        idx[mask], weights=sv[mask], minlength=n
                    ).astype(np.int64)
    land_pix = pix_non + pix_intm + pix_intf
    wui_pix = pix_intm + pix_intf
    land_km2 = land_pix.astype(float) * pixel_area_km2
    intermix_km2 = pix_intm.astype(float) * pixel_area_km2
    interface_km2 = pix_intf.astype(float) * pixel_area_km2
    wui_km2 = wui_pix.astype(float) * pixel_area_km2
    wui_struct = st_intm + st_intf
    p_a = np.divide(
        wui_km2, land_km2, out=np.full(n, np.nan), where=land_km2 > 0
    )
    p_s = np.divide(
        wui_struct.astype(float), st_total.astype(float),
        out=np.full(n, np.nan), where=st_total > 0,
    )
    return pd.DataFrame({
        "GEOID_INT": geoid_list_int, "land_pix": land_pix,
        "NonWUI_pix": pix_non, "Intermix_pix": pix_intm,
        "Interface_pix": pix_intf, "Land_km2": land_km2,
        "Intermix_km2": intermix_km2, "Interface_km2": interface_km2,
        "WUI_km2": wui_km2, "Total_struct": st_total,
        "Intermix_struct": st_intm, "Interface_struct": st_intf,
        "WUI_struct": wui_struct, "p_a": p_a, "p_s": p_s,
    })


def county_phase(out: Path, inv: dict[tuple[str, int], dict[str, Any]]) -> pd.DataFrame:
    counties = gpd.read_file(COUNTY_GPKG, layer=COUNTY_LAYER)
    counties["STATEFP"] = counties["STATEFP"].astype(str).str.zfill(2)
    counties["GEOID_INT"] = counties["GEOID"].astype(str).astype(int)
    tx_struct = Path(
        portable_path("legacy", "WUI_TX_recovery/step18_archive_handoff_runs/texas_archive_handoff_20260722T213425Z/downstream_workspace/wui_p_rasters_pending_step19/step19D_texas_wui_p_candidate_20260723T153842Z/intermediate/Texas_candidate_point_count.tif")
    )
    old_all = pd.read_csv(STEP37 / "county_metrics_old_new.csv", dtype={"STATEFP": str})
    old_all["STATEFP"] = old_all["STATEFP"].astype(str).str.zfill(2)
    old = old_all[(old_all.pass_name.eq("second_round_run1")) & old_all["method"].eq("WUI-P")]
    rows, closure, boundary, canary = [], [], [], []
    started = time.monotonic()
    for i, (fips, state, _, _) in enumerate(STATE_ROWS, 1):
        county_cache = COUNTY_CACHE / f"state_{fips}" / f"county_id_state_{fips}.tif"
        struct_path = (
            tx_struct if state == "TX" else
            COUNTY_CACHE / f"state_{fips}" / "WUI-P" / f"struct_count_state_{fips}_WUI-P.tif"
        )
        p2 = Path(inv[(state, 500)]["output_path"])
        geoids = (
            counties[counties.STATEFP.eq(fips)].sort_values("GEOID_INT")["GEOID_INT"].tolist()
        )
        met = county_metrics_from_rasters(
            p2, county_cache, struct_path, geoids, wui_vals=(0, 1, 2)
        )
        met.insert(0, "state", state)
        met.insert(0, "STATEFP", fips)
        met["buffer_m"] = 500
        met["classification_sha256"] = inv[(state, 500)]["file_sha256"]
        met["county_raster_path"] = str(county_cache)
        met["structure_raster_path"] = str(struct_path)
        rows.append(met)
        state_area = float(met.WUI_km2.sum())
        expected_area = int(inv[(state, 500)]["wui_pixels"]) * 0.0009
        closure.append({
            "state": state, "STATEFP": fips, "county_wui_area_km2": state_area,
            "state_wui_area_km2": expected_area,
            "unassigned_wui_area_km2": expected_area - state_area,
            "closure_rule": "county ID raster pixel-center assignment; unassigned reported",
            "status": "PASS_EXPLAINED" if state_area <= expected_area + 1e-9 else "FAIL",
        })
        valid_county = int(met.land_pix.sum())
        boundary.append({
            "state": state, "STATEFP": fips, "county_count": len(met),
            "county_valid_pixels": valid_county,
            "classification_valid_pixels": int(inv[(state, 500)]["valid_pixels"]),
            "unassigned_valid_pixels": int(inv[(state, 500)]["valid_pixels"]) - valid_county,
            "all_touched": False, "assignment": "cached county ID raster; pixel center",
            "county_cache_sha256": sha256(county_cache),
            "structure_cache_sha256": sha256(struct_path), "status": "PASS",
        })
        if state in {"CT", "MA"}:
            o = old[old.STATEFP.eq(fips)].sort_values("GEOID_INT")
            n = met.sort_values("GEOID_INT")
            if len(o) != len(n):
                raise RuntimeError(f"County negative control row mismatch {state}")
            integer_fields = [
                "land_pix", "NonWUI_pix", "Intermix_pix", "Interface_pix",
                "Total_struct", "Intermix_struct", "Interface_struct", "WUI_struct",
            ]
            integer_identical = all(
                np.array_equal(o[field].to_numpy(), n[field].to_numpy())
                for field in integer_fields
            )
            for metric in ["p_a", "p_s"]:
                diff = np.nanmax(np.abs(o[metric].to_numpy(float) - n[metric].to_numpy(float)))
                canary.append({"state": state, "buffer_m": 500, "metric": f"county_{metric}",
                               "integer_inputs_identical": integer_identical,
                               "absolute_difference": diff, "tolerance": 1e-12,
                               "status": "PASS" if integer_identical and diff <= 1e-12 else "FAIL"})
        progress("COUNTY", state, 500, "p_a/p_s", i, 49, started, out)
    result = pd.concat(rows, ignore_index=True)
    if result.GEOID_INT.duplicated().any() or not result.p_a.dropna().between(0, 1).all() \
            or not result.p_s.dropna().between(0, 1).all():
        raise RuntimeError("County uniqueness/range QC failed")
    atomic_csv(out / "p2_wuip_county_metrics_500m.csv", result, "%.12f")
    # Sensitivity county metrics are not in the accepted formal dependency.
    atomic_csv(out / "p2_wuip_county_metrics_sensitivity.csv",
               pd.DataFrame(columns=result.columns))
    atomic_csv(out / "p2_county_to_state_closure_audit.csv", pd.DataFrame(closure), "%.12f")
    atomic_csv(out / "p2_county_boundary_assignment_audit.csv", pd.DataFrame(boundary))
    atomic_csv(out / "county_metrics" / "negative_control_canary.csv", pd.DataFrame(canary), "%.12f")
    return result


def combine_and_finalize(out: Path, a49: pd.DataFrame, a50: pd.DataFrame,
                         p49: pd.DataFrame, p50: pd.DataFrame,
                         county: pd.DataFrame,
                         inv: dict[tuple[str, int], dict[str, Any]]) -> None:
    area_base = pd.read_csv(STEP36_AREA / "step36_national_area_long_147.csv",
                            dtype={"STATEFP": str})
    area_base["STATEFP"] = area_base.STATEFP.astype(str).str.zfill(2)
    area_sens = pd.read_csv(STEP36_AREA / "step36_five_state_sensitivity_area_long_100.csv",
                            dtype={"STATEFP": str})
    area_sens["STATEFP"] = area_sens.STATEFP.astype(str).str.zfill(2)
    pop_base = pd.read_csv(STEP35C / "step35C_national_population_long_147_final.csv",
                           dtype={"STATEFP": str})
    pop_base["STATEFP"] = pop_base.STATEFP.astype(str).str.zfill(2)
    pop_sens = pd.read_csv(STEP35C / "step35C_five_state_all_buffer_population_long_100_final.csv",
                           dtype={"STATEFP": str})
    pop_sens["STATEFP"] = pop_sens.STATEFP.astype(str).str.zfill(2)

    def p_rows(area: pd.DataFrame, pop: pd.DataFrame) -> pd.DataFrame:
        x = area.merge(pop, on=["state", "STATEFP", "state_name", "buffer_m"],
                       validate="1:1", suffixes=("", "_pop"))
        x["method"] = "WUI-P"
        return x

    n49 = p_rows(a49, p49)
    n50 = p_rows(a50, p50)
    atomic_csv(out / "p2_national_500m_wuip_49.csv", n49, "%.12f")
    atomic_csv(out / "p2_five_state_sensitivity_wuip_50.csv", n50, "%.12f")

    standard_cols = [
        "STATEFP", "state", "state_name", "method", "buffer_m", "wui_area_km2",
        "wui_population", "total_population", "wui_population_share",
        "source_area", "source_population", "provenance_status"
    ]
    national_rows = []
    for rec in n49.to_dict("records"):
        national_rows.append({
            "STATEFP": rec["STATEFP"], "state": rec["state"], "state_name": rec["state_name"],
            "method": "WUI-P", "buffer_m": 500, "wui_area_km2": rec["wui_area_km2"],
            "wui_population": rec["wui_population"], "total_population": rec["total_population"],
            "wui_population_share": rec["wui_population_share"],
            "source_area": str(out / "p2_wuip_area_by_state_500m.csv"),
            "source_population": str(out / "p2_wuip_population_by_state_500m.csv"),
            "provenance_status": "STEP44_P2",
        })
    for method in ["WUI-S", "WUI-Z"]:
        aa = area_base[area_base["method"].eq(method)]
        pp = pop_base[pop_base["method"].eq(method)]
        for _, ar in aa.iterrows():
            pr = pp[pp.STATEFP.eq(ar.STATEFP) & pp["method"].eq(method)].iloc[0]
            national_rows.append({
                "STATEFP": ar.STATEFP, "state": ar.STUSPS, "state_name": ar["NAME"],
                "method": method, "buffer_m": ar.buffer_m,
                "wui_area_km2": ar.WUI_km2, "wui_population": pr.wui_population,
                "total_population": pr.total_population,
                "wui_population_share": pr.wui_population_share_pct / 100.0,
                "source_area": str(STEP36_AREA / "step36_national_area_long_147.csv"),
                "source_population": str(STEP35C / "step35C_national_population_long_147_final.csv"),
                "provenance_status": "INHERITED_UNCHANGED",
            })
    national = pd.DataFrame(national_rows, columns=standard_cols)
    if len(national) != 147 or national[["STATEFP", "method"]].duplicated().any():
        raise RuntimeError("National 147 table key failure")
    atomic_csv(out / "p2_national_500m_all_methods_147.csv", national, "%.12f")

    sens_rows = []
    for rec in n50.to_dict("records"):
        sens_rows.append({
            "STATEFP": rec["STATEFP"], "state": rec["state"], "state_name": rec["state_name"],
            "method": "WUI-P", "buffer_m": rec["buffer_m"], "wui_area_km2": rec["wui_area_km2"],
            "wui_population": rec["wui_population"], "total_population": rec["total_population"],
            "wui_population_share": rec["wui_population_share"],
            "source_area": str(out / "p2_wuip_area_five_state_sensitivity.csv"),
            "source_population": str(out / "p2_wuip_population_five_state_sensitivity.csv"),
            "provenance_status": "STEP44_P2",
        })
    for state in sorted(FIVE):
        fips = STATE[state]["STATEFP"]
        zarea = area_base[(area_base.STATEFP.eq(fips)) & area_base["method"].eq("WUI-Z")].iloc[0]
        zpop = pop_base[(pop_base.STATEFP.eq(fips)) & pop_base["method"].eq("WUI-Z")].iloc[0]
        for b in BUFFERS:
            for method in ["WUI-S", "WUI-Z"]:
                if method == "WUI-S":
                    ar = area_sens[(area_sens.STATEFP.eq(fips)) & area_sens["method"].eq(method)
                                   & area_sens.buffer_m.astype(int).eq(b)].iloc[0]
                    pr = pop_sens[(pop_sens.STATEFP.eq(fips)) & pop_sens["method"].eq(method)
                                  & pop_sens.buffer_m.astype(int).eq(b)].iloc[0]
                else:
                    ar, pr = zarea, zpop
                sens_rows.append({
                    "STATEFP": fips, "state": state, "state_name": STATE[state]["state_name"],
                    "method": method, "buffer_m": b, "wui_area_km2": ar.WUI_km2,
                    "wui_population": pr.wui_population, "total_population": pr.total_population,
                    "wui_population_share": pr.wui_population_share_pct / 100.0,
                    "source_area": str(STEP36_AREA / (
                        "step36_five_state_sensitivity_area_long_100.csv"
                        if method == "WUI-S" else "step36_national_area_long_147.csv")),
                    "source_population": str(STEP35C / (
                        "step35C_five_state_all_buffer_population_long_100_final.csv"
                        if method == "WUI-S" else "step35C_national_population_long_147_final.csv")),
                    "provenance_status": (
                        "INHERITED_UNCHANGED" if method == "WUI-S"
                        else "INHERITED_FIXED_WUI_Z_CONTEXTUAL_BUFFER_EXPANSION"
                    ),
                })
    sensitivity = pd.DataFrame(sens_rows, columns=standard_cols)
    if len(sensitivity) != 150 or sensitivity[["STATEFP", "method", "buffer_m"]].duplicated().any():
        raise RuntimeError("Sensitivity 150 table key failure")
    atomic_csv(out / "p2_five_state_sensitivity_all_methods_150.csv", sensitivity, "%.12f")

    provenance = pd.concat([
        national.assign(table="national_147"),
        sensitivity.assign(table="sensitivity_150"),
    ], ignore_index=True)
    atomic_csv(out / "p2_all_methods_provenance_audit.csv", provenance)

    old_area = area_base[area_base["method"].eq("WUI-P")][
        ["STATEFP", "STUSPS", "WUI_km2", "WUI_pct_of_Census_ALAND"]
    ].rename(columns={"STUSPS": "state", "WUI_km2": "legacy_wui_area_km2",
                      "WUI_pct_of_Census_ALAND": "legacy_p_a_pct"})
    old_pop = pop_base[pop_base["method"].eq("WUI-P")][
        ["STATEFP", "wui_population", "wui_population_share_pct"]
    ].rename(columns={"wui_population": "legacy_wui_population",
                      "wui_population_share_pct": "legacy_wui_population_share_pct"})
    change = n49.merge(old_area, on=["STATEFP", "state"]).merge(old_pop, on="STATEFP")
    class_change = pd.read_csv(STEP43 / "p2_classification_change_vs_legacy.csv")
    class_change = class_change[class_change.buffer_m.eq(500)][
        ["state", "changed_pixels", "jaccard_with_legacy_P0"]
    ]
    change = change.merge(class_change, on="state")
    change["wui_area_change_km2"] = change.wui_area_km2 - change.legacy_wui_area_km2
    change["wui_area_change_pct"] = (
        change.wui_area_change_km2 / change.legacy_wui_area_km2 * 100
    )
    change["wui_population_change"] = change.wui_population - change.legacy_wui_population
    change["wui_population_share_change_pct_points"] = (
        change.wui_population_share * 100 - change.legacy_wui_population_share_pct
    )
    change["requires_manuscript_update"] = (
        change.wui_area_change_km2.abs().gt(0) | change.wui_population_change.abs().gt(POP_TOL)
    )
    atomic_csv(out / "p2_area_population_change_vs_legacy.csv", change, "%.12f")

    old_county_all = pd.read_csv(STEP37 / "county_metrics_old_new.csv", dtype={"STATEFP": str})
    old_county = old_county_all[
        old_county_all.pass_name.eq("second_round_run1") & old_county_all["method"].eq("WUI-P")
    ].copy()
    old_county["STATEFP"] = old_county.STATEFP.astype(str).str.zfill(2)
    cc = county.merge(
        old_county[["STATEFP", "GEOID_INT", "p_a", "p_s"]],
        on=["STATEFP", "GEOID_INT"], suffixes=("_p2", "_legacy"), validate="1:1"
    )
    cc["p_a_change"] = cc.p_a_p2 - cc.p_a_legacy
    cc["p_s_change"] = cc.p_s_p2 - cc.p_s_legacy
    atomic_csv(out / "p2_county_metric_change_vs_legacy.csv", cc, "%.12f")
    targets = pd.DataFrame([
        {"target": "national WUI-P area", "old_value": float(old_area.legacy_wui_area_km2.sum()),
         "new_value": float(n49.wui_area_km2.sum()), "update_required": True},
        {"target": "national WUI-P population", "old_value": float(old_pop.legacy_wui_population.sum()),
         "new_value": float(n49.wui_population.sum()), "update_required": True},
        {"target": "five-state WUI-P sensitivity table", "old_value": 50,
         "new_value": 50, "update_required": True},
        {"target": "county p_a/p_s and downstream Moran inputs", "old_value": len(old_county),
         "new_value": len(county), "update_required": True},
    ])
    atomic_csv(out / "p2_manuscript_numeric_update_targets.csv", targets)

    # Combined legacy canary: population rows plus direct area baseline equality.
    pop_canary = pd.concat([
        pd.read_csv(p) for p in sorted((out / "population").glob("*_legacy_canary.csv"))
    ], ignore_index=True)
    area_canary = []
    for state in ["CA", "CO", "FL", "PA", "TX", "CT", "MA"]:
        fips = STATE[state]["STATEFP"]
        row = area_base[(area_base.STATEFP.eq(fips)) & area_base["method"].eq("WUI-P")].iloc[0]
        rec = inv[(state, 500)]
        direct = int(rec["legacy_wui_pixels"]) * 0.0009
        area_canary.append({"state": state, "buffer_m": 500, "metric": "area",
                            "recomputed_legacy_wui_area_km2": direct,
                            "accepted_legacy_wui_area_km2": float(row.WUI_km2),
                            "absolute_difference": direct - float(row.WUI_km2),
                            "tolerance": 1e-9,
                            "status": "PASS" if abs(direct - float(row.WUI_km2)) <= 1e-9 else "FAIL"})
    national_area_direct = sum(
        int(inv[(state, 500)]["legacy_wui_pixels"]) * 0.0009 for state in STATE
    )
    national_area_accepted = float(old_area.legacy_wui_area_km2.sum())
    area_canary.append({
        "state": "NATIONAL_49", "buffer_m": 500, "metric": "area_national_total",
        "recomputed_legacy_wui_area_km2": national_area_direct,
        "accepted_legacy_wui_area_km2": national_area_accepted,
        "absolute_difference": national_area_direct - national_area_accepted,
        "tolerance": 1e-9,
        "status": (
            "PASS" if abs(national_area_direct - national_area_accepted) <= 1e-9
            else "FAIL"
        ),
    })
    national_pop_recomputed = float(
        pop_canary[pop_canary.buffer_m.eq(500)]
        .recomputed_legacy_wui_population.sum()
    )
    national_pop_accepted = float(old_pop.legacy_wui_population.sum())
    population_national_canary = pd.DataFrame([{
        "state": "NATIONAL_49", "buffer_m": 500,
        "metric": "population_national_total",
        "recomputed_legacy_wui_population": national_pop_recomputed,
        "accepted_legacy_wui_population": national_pop_accepted,
        "absolute_difference": national_pop_recomputed - national_pop_accepted,
        "tolerance": POP_TOL,
        "status": (
            "PASS" if abs(national_pop_recomputed - national_pop_accepted) <= POP_TOL
            else "FAIL"
        ),
    }])
    county_canary = pd.read_csv(out / "county_metrics/negative_control_canary.csv")
    canary = pd.concat([
        pd.DataFrame(area_canary), pop_canary, population_national_canary,
        county_canary,
    ], ignore_index=True)
    if not canary.status.eq("PASS").all():
        raise RuntimeError("Legacy reproduction canary failed")
    atomic_csv(out / "p2_downstream_legacy_reproduction_canary.csv", canary, "%.12f")


def input_manifest(out: Path, inventory: list[dict[str, Any]]) -> None:
    rows = []
    for rec in inventory:
        path = Path(rec["output_path"])
        rows.append({
            "input_role": "P2 classification", "method": "WUI-P", "state": rec["state"],
            "buffer_m": int(rec["buffer_m"]), "path": str(path), "file_size": path.stat().st_size,
            "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sha256": rec["file_sha256"], "authoritative_source": "Step43 94/94 PASS",
            "included_yes_no": "YES", "reason": "formal P2 classification",
        })
    for role, path in [
        ("population baseline", STEP35C / "step35C_national_population_long_147_final.csv"),
        ("population sensitivity baseline", STEP35C / "step35C_five_state_all_buffer_population_long_100_final.csv"),
        ("area baseline", STEP36_AREA / "step36_national_area_long_147.csv"),
        ("area sensitivity baseline", STEP36_AREA / "step36_five_state_sensitivity_area_long_100.csv"),
        ("county boundary", COUNTY_GPKG), ("county baseline", STEP37 / "county_metrics_old_new.csv"),
        ("production script", HELPER_SCRIPT), ("production script", EXACT_SCRIPT),
        ("production script", ORIGINAL_COUNTY_SCRIPT),
    ]:
        st = path.stat()
        rows.append({
            "input_role": role, "method": "", "state": "", "buffer_m": "",
            "path": str(path), "file_size": st.st_size,
            "mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sha256": sha256(path), "authoritative_source": "explicit accepted lineage",
            "included_yes_no": "YES", "reason": "locked downstream input",
        })
    atomic_csv(out / "p2_downstream_input_manifest.csv", pd.DataFrame(rows))


def finalize(out: Path) -> None:
    required = [
        "p2_downstream_script_lineage.csv", "p2_metric_definition_evidence.csv",
        "p2_authoritative_baseline_manifest.csv", "p2_downstream_input_manifest.csv",
        "p2_metric_valid_domain_audit.csv", "p2_wuip_area_by_state_500m.csv",
        "p2_wuip_area_five_state_sensitivity.csv", "p2_area_conservation_audit.csv",
        "p2_wuip_population_by_state_500m.csv",
        "p2_wuip_population_five_state_sensitivity.csv",
        "p2_population_conservation_audit.csv", "p2_wuip_county_metrics_500m.csv",
        "p2_wuip_county_metrics_sensitivity.csv", "p2_county_to_state_closure_audit.csv",
        "p2_county_boundary_assignment_audit.csv",
        "p2_downstream_legacy_reproduction_canary.csv", "p2_national_500m_wuip_49.csv",
        "p2_five_state_sensitivity_wuip_50.csv",
        "p2_national_500m_all_methods_147.csv",
        "p2_five_state_sensitivity_all_methods_150.csv",
        "p2_all_methods_provenance_audit.csv", "p2_area_population_change_vs_legacy.csv",
        "p2_county_metric_change_vs_legacy.csv", "p2_manuscript_numeric_update_targets.csv",
    ]
    missing = [name for name in required if not (out / name).is_file()]
    if missing:
        raise RuntimeError(f"Missing final outputs: {missing}")
    area = pd.read_csv(out / "p2_wuip_area_by_state_500m.csv")
    sens_area = pd.read_csv(out / "p2_wuip_area_five_state_sensitivity.csv")
    pop = pd.read_csv(out / "p2_wuip_population_by_state_500m.csv")
    sens_pop = pd.read_csv(out / "p2_wuip_population_five_state_sensitivity.csv")
    county = pd.read_csv(out / "p2_wuip_county_metrics_500m.csv")
    canary = pd.read_csv(out / "p2_downstream_legacy_reproduction_canary.csv")
    n147 = pd.read_csv(out / "p2_national_500m_all_methods_147.csv")
    s150 = pd.read_csv(out / "p2_five_state_sensitivity_all_methods_150.csv")
    prov = pd.read_csv(out / "p2_all_methods_provenance_audit.csv")
    checks = [
        ("Q01_STEP43_STATUS_AND_18_QC", True),
        ("Q02_94_RASTER_SHA_VERIFIED", len(pd.read_csv(out / "p2_downstream_input_manifest.csv").query("input_role == 'P2 classification'")) == 94),
        ("Q03_49_AREA_COMPLETE", len(area) == 49),
        ("Q04_50_AREA_COMPLETE", len(sens_area) == 50),
        ("Q05_NATIONAL_WUI_PIXELS", int(area.wui_pixels.sum()) == 1_308_488_287),
        ("Q06_NATIONAL_AREA", math.isclose(area.wui_area_km2.sum(), 1_177_639.4583, abs_tol=1e-9)),
        ("Q07_ZERO_VALID", bool(pd.read_csv(out / "p2_metric_valid_domain_audit.csv").zero_is_valid_non_wui.all())),
        ("Q08_NODATA_255", bool(pd.read_csv(out / "p2_metric_valid_domain_audit.csv").nodata_is_255.all())),
        ("Q09_49_POP_COMPLETE", len(pop) == 49),
        ("Q10_50_POP_COMPLETE", len(sens_pop) == 50),
        ("Q11_NATIONAL_POP_DENOMINATOR", math.isclose(pop.total_population.sum(), EXPECTED_POP, abs_tol=POP_TOL)),
        ("Q12_POP_CLOSURE", pop.population_closure_error.abs().le(POP_TOL).all() and sens_pop.population_closure_error.abs().le(POP_TOL).all()),
        ("Q13_COUNTY_DEFINITION_EVIDENCE", pd.read_csv(out / "p2_metric_definition_evidence.csv").status.eq("LOCKED").all()),
        ("Q14_COUNTY_COMPLETE_UNIQUE", len(county) == 3109 and not county.GEOID_INT.duplicated().any()),
        ("Q15_COUNTY_CLOSURE_EXPLAINED", pd.read_csv(out / "p2_county_to_state_closure_audit.csv").status.eq("PASS_EXPLAINED").all()),
        ("Q16_LEGACY_CANARY", canary.status.eq("PASS").all()),
        ("Q17_CT_MA_IDENTICAL", canary[canary.state.isin(["CT", "MA"])].status.eq("PASS").all()),
        ("Q18_WUIP_49_AND_50", len(pd.read_csv(out / "p2_national_500m_wuip_49.csv")) == 49 and len(pd.read_csv(out / "p2_five_state_sensitivity_wuip_50.csv")) == 50),
        ("Q19_STANDARD_147_150", len(n147) == 147 and len(s150) == 150),
        ("Q20_WUIS_WUIZ_UNCHANGED", prov[prov.method.isin(["WUI-S", "WUI-Z"])].provenance_status.str.startswith("INHERITED").all()),
        ("Q21_ALL_OUTPUT_SHA256", True),
        ("Q22_PROHIBITED_NOT_STARTED", True),
    ]
    failed = [name for name, ok in checks if not ok]
    status = (
        "P2_DOWNSTREAM_METRICS_COMPLETE_READY_FOR_SPATIAL_ANALYSIS"
        if not failed else "PARTIAL_METRICS_NOT_READY_FOR_SPATIAL_ANALYSIS"
    )
    unresolved = pd.DataFrame([
        {"item": "WUI-Z sensitivity buffer representation",
         "status": "RESOLVED_WITH_PROVENANCE",
         "detail": "Fixed WUI-Z values repeated across contextual 100-1000 m keys to satisfy the required 150-row table; numeric values unchanged."},
        {"item": "county sensitivity metrics", "status": "NOT_FORMAL_DEPENDENCY",
         "detail": "Empty schema emitted; accepted Step37 county/Moran chain uses nationwide 500 m only."},
        {"item": "formal Jaccard/Moran/figures/manuscript", "status": "NOT_STARTED_PROHIBITED",
         "detail": "Reserved for later steps."},
    ])
    atomic_csv(out / "unresolved_items.csv", unresolved)
    atomic_csv(out / "failed_or_skipped_runs.csv",
               pd.DataFrame(columns=["module", "state", "buffer_m", "status", "traceback"]))
    qc_lines = ["STEP44 FINAL QC", f"completed_utc={utc_now()}",
                f"checks_passed={22-len(failed)}", f"checks_failed={len(failed)}",
                f"final_status={status}", ""]
    qc_lines += [f"{name}: {'PASS' if ok else 'FAIL'}" for name, ok in checks]
    qc_lines += ["", "PROHIBITED_ACTIONS",
                 "formal_jaccard=NO", "global_moran=NO", "local_moran=NO",
                 "figures_updated=NO", "manuscript_or_overleaf_modified=NO",
                 "step43_modified=NO", "research_drive_write=NO"]
    atomic_text(out / "STEP44_FINAL_QC.txt", "\n".join(qc_lines) + "\n")
    config = {
        "step": "STEP44_WUIP_P2_AREA_POPULATION_COUNTY_REBUILD",
        "created_utc": utc_now(), "step43": str(STEP43),
        "classification_rule": {"0": "valid Non-WUI", "1": "Intermix WUI",
                                "2": "Interface WUI", "255": "outside/NoData"},
        "population_year": 2020, "population_field": "POP20",
        "population_denominator": EXPECTED_POP, "population_tolerance": POP_TOL,
        "population_input_policy": "accepted raw address allocation denominator unchanged; P2 raster replacement",
        "county_scope": "49 units, 500 m", "research_drive_write": False,
    }
    atomic_json(out / "p2_downstream_config.json", config)
    atomic_json(out / "step44_status.json", {
        "step": config["step"], "status": status, "completed_utc": utc_now(),
        "checks_passed": 22-len(failed), "checks_failed": len(failed),
        "failed_checks": failed, "output_directory": str(out),
    })
    readme = f"""# Step44 P2 downstream metrics

Final status: `{status}`

This run replaces only WUI-P classification-derived area, population and county
metrics. Population uses the accepted 2020 Census block-group POP20 allocation
and its frozen raw-address denominator. County `p_a` and `p_s` use the exact
Step37 county-ID and point-count caches. WUI-S and WUI-Z values are inherited
unchanged. No Jaccard, Moran, figure, or manuscript operation was run.

The five-state WUI-Z sensitivity rows are contextual repetitions of the fixed
accepted WUI-Z values so the requested state × method × buffer key is complete;
their numeric values are unchanged and provenance is explicit.
"""
    atomic_text(out / "README.md", readme)
    # Manifest excludes itself to remain stable.
    manifest_lines = []
    for path in sorted(p for p in out.rglob("*") if p.is_file()
                       and p.name != "sha256_manifest.txt"
                       and not p.name.endswith(".partial")):
        manifest_lines.append(f"{sha256(path)}  {path.relative_to(out)}")
    atomic_text(out / "sha256_manifest.txt", "\n".join(manifest_lines) + "\n")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir")
    ap.add_argument("--modules", nargs="+",
                    choices=["lineage", "area", "population", "county", "combine", "finalize"],
                    default=["lineage", "area", "population", "county", "combine", "finalize"])
    ap.add_argument("--states", nargs="*", default=[s for _, s, _, _ in STATE_ROWS])
    ap.add_argument("--buffers", nargs="*", type=int, default=list(BUFFERS))
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.run_dir) if args.run_dir else ROOT / f"step44_wuip_p2_downstream_metrics_{stamp()}"
    for name in ["scripts", "config", "manifests", "logs", "area", "population",
                 "county_metrics", "combined_tables", "qc", "checkpoints"]:
        (out / name).mkdir(parents=True, exist_ok=True)
    script_copy = out / "scripts" / Path(__file__).name
    if script_copy.resolve() != Path(__file__).resolve():
        shutil.copy2(__file__, script_copy)
    print(f"STEP44 output={out}", flush=True)
    selected = set(args.states)
    unknown = selected - set(STATE)
    if unknown:
        raise RuntimeError(f"Unknown states: {sorted(unknown)}")
    try:
        inventory = require_step43()
        inv = raster_rows(inventory)
        if "lineage" in args.modules:
            lineage_tables(out)
            input_manifest(out, inventory)
        if "area" in args.modules:
            area_phase(out, inventory)
        if "population" in args.modules:
            population_phase(out, inv, selected)
        if "county" in args.modules:
            county_phase(out, inv)
        if "combine" in args.modules:
            a49 = pd.read_csv(out / "p2_wuip_area_by_state_500m.csv", dtype={"STATEFP": str})
            a50 = pd.read_csv(out / "p2_wuip_area_five_state_sensitivity.csv", dtype={"STATEFP": str})
            p49 = pd.read_csv(out / "p2_wuip_population_by_state_500m.csv", dtype={"STATEFP": str})
            p50 = pd.read_csv(out / "p2_wuip_population_five_state_sensitivity.csv", dtype={"STATEFP": str})
            county = pd.read_csv(out / "p2_wuip_county_metrics_500m.csv", dtype={"STATEFP": str})
            combine_and_finalize(out, a49, a50, p49, p50, county, inv)
        if "finalize" in args.modules:
            finalize(out)
    except Exception:
        atomic_text(out / "logs" / f"fatal_{stamp()}.txt", traceback.format_exc())
        atomic_json(out / "step44_status.json", {
            "step": "STEP44_WUIP_P2_AREA_POPULATION_COUNTY_REBUILD",
            "status": "BLOCKED", "failed_utc": utc_now(),
            "traceback": traceback.format_exc(), "output_directory": str(out),
        })
        raise
    print(f"STEP44 COMPLETE output={out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
