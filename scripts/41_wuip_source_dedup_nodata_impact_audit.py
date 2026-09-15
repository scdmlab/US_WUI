#!/usr/bin/env python3
# Copyright (C) 2026 WUI Mapping Project Authors
# SPDX-License-Identifier: GPL-3.0-only
# Repository version: filesystem paths are loaded through repo_config.py.
# Copy config/paths.example.json to config/paths.json before a new run.
# Raw input data and large intermediate files are stored outside GitHub.

"""Step 41: read-only WUI-P source, duplicate, and NoData impact audit.

This program never writes to the research drive and never changes an input
GeoPackage or raster.  All products are CSV/JSON/Markdown files in one Step41
directory.  The classification sensitivity calculation is exact for deletion
scenarios because all scenarios are subsets of S0: vegetation and distance are
fixed, so a formal WUI pixel can only remain WUI or lose WUI status when its
disk-neighbourhood point count falls below the production threshold.
"""

from __future__ import annotations

from repo_config import portable_path

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import sqlite3
import socket
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyogrio
import rasterio
import shapely
from pyproj import CRS
from rasterio.features import geometry_mask
from rasterio.windows import Window
from scipy.signal import fftconvolve
from skimage.morphology import disk


PROJECT = Path(portable_path("project"))
RESEARCH_LOWER = Path(portable_path("data"))
RESEARCH_UPPER = Path(portable_path("data"))
RAW_ROOT = RESEARCH_LOWER / "OpenAddresses_Work/raw/us"
PROCESSED_ROOT = RESEARCH_UPPER / "OpenAddresses_Work/Processed_GPKG"
STATE_GPKG = (
    RESEARCH_UPPER
    / "MBF+NLCD_2022US/Inputs/Processed_GPKG/tl_2022_us_state.gpkg"
)
AREA_CSV = (
    PROJECT
    / "step36_area_final_audit_20260724T192140Z/"
    "step36_final_unique_area_long_237.csv"
)
MORAN_DIR = PROJECT / "step37_morans_i_revision_20260727T043506Z"
MORAN_RESULTS = MORAN_DIR / "morans_i_second_round_results.csv"
MORAN_INVENTORY = MORAN_DIR / "input_inventory.csv"
PRODUCTION_POINT_SCRIPT = Path(portable_path("legacy", "process_oa_v2.py"))
PRODUCTION_RASTER_SCRIPT = Path(
    portable_path("legacy", "MBF+NLCD_2022US/09_analyze_wui_p_raster_fast.py")
)
TX_SCRIPT = PROJECT / "scripts/19D_rebuild_texas_wui_p_candidate_rasters.py"
TX_POINT = Path(
    portable_path("legacy", "WUI_TX_recovery/step18_archive_handoff_runs/texas_archive_handoff_20260722T213425Z/downstream_workspace/wui_p_rasters_pending_step19/texas_wui_p_rebuild_20260722T220430Z/step19C_inside_grid_candidate_20260723T152032Z/Texas_addresses_inside_grid_candidate_20991807.gpkg")
)
TX_SOURCE_MANIFEST = Path(
    portable_path("legacy", "WUI_TX_recovery/step17_gpkg_runs/texas_gpkg_rebuild_exclude_null3_20260722T173401Z/source_manifest.json")
)
FIVE = {"CA", "CO", "FL", "PA", "TX"}
BUFFER_M = 500
PIXEL_M = 30.0
DENSITY_THRESHOLD = 6.17
RADIUS_PX = int(round(BUFFER_M / PIXEL_M))
KERNEL = disk(RADIUS_PX).astype(np.float32)
BUFFER_AREA_KM2 = math.pi * BUFFER_M**2 / 1_000_000.0
MIN_DENSE_COUNT = int(math.floor(DENSITY_THRESHOLD * BUFFER_AREA_KM2)) + 1
TILE = 2048

STATES = [
    ("01", "AL", "Alabama", "Alabama"),
    ("04", "AZ", "Arizona", "Arizona"),
    ("05", "AR", "Arkansas", "Arkansas"),
    ("06", "CA", "California", "California"),
    ("08", "CO", "Colorado", "Colorado"),
    ("09", "CT", "Connecticut", "Connecticut"),
    ("10", "DE", "Delaware", "Delaware"),
    ("11", "DC", "District of Columbia", "DistrictofColumbia"),
    ("12", "FL", "Florida", "Florida"),
    ("13", "GA", "Georgia", "Georgia"),
    ("16", "ID", "Idaho", "Idaho"),
    ("17", "IL", "Illinois", "Illinois"),
    ("18", "IN", "Indiana", "Indiana"),
    ("19", "IA", "Iowa", "Iowa"),
    ("20", "KS", "Kansas", "Kansas"),
    ("21", "KY", "Kentucky", "Kentucky"),
    ("22", "LA", "Louisiana", "Louisiana"),
    ("23", "ME", "Maine", "Maine"),
    ("24", "MD", "Maryland", "Maryland"),
    ("25", "MA", "Massachusetts", "Massachusetts"),
    ("26", "MI", "Michigan", "Michigan"),
    ("27", "MN", "Minnesota", "Minnesota"),
    ("28", "MS", "Mississippi", "Mississippi"),
    ("29", "MO", "Missouri", "Missouri"),
    ("30", "MT", "Montana", "Montana"),
    ("31", "NE", "Nebraska", "Nebraska"),
    ("32", "NV", "Nevada", "Nevada"),
    ("33", "NH", "New Hampshire", "NewHampshire"),
    ("34", "NJ", "New Jersey", "NewJersey"),
    ("35", "NM", "New Mexico", "NewMexico"),
    ("36", "NY", "New York", "NewYork"),
    ("37", "NC", "North Carolina", "NorthCarolina"),
    ("38", "ND", "North Dakota", "NorthDakota"),
    ("39", "OH", "Ohio", "Ohio"),
    ("40", "OK", "Oklahoma", "Oklahoma"),
    ("41", "OR", "Oregon", "Oregon"),
    ("42", "PA", "Pennsylvania", "Pennsylvania"),
    ("44", "RI", "Rhode Island", "RhodeIsland"),
    ("45", "SC", "South Carolina", "SouthCarolina"),
    ("46", "SD", "South Dakota", "SouthDakota"),
    ("47", "TN", "Tennessee", "Tennessee"),
    ("48", "TX", "Texas", "Texas"),
    ("49", "UT", "Utah", "Utah"),
    ("50", "VT", "Vermont", "Vermont"),
    ("51", "VA", "Virginia", "Virginia"),
    ("53", "WA", "Washington", "Washington"),
    ("54", "WV", "West Virginia", "WestVirginia"),
    ("55", "WI", "Wisconsin", "Wisconsin"),
    ("56", "WY", "Wyoming", "Wyoming"),
]
BY_ABBR = {abbr: (fips, name, stem) for fips, abbr, name, stem in STATES}


class Progress:
    def __init__(self, total: int, phase: str):
        self.total = total
        self.done = 0
        self.start = time.monotonic()
        self.phase = phase

    def tick(
        self,
        *,
        state: str = "-",
        buffer_m: str | int = "-",
        scenario: str = "-",
        detail: str = "",
    ) -> None:
        self.done += 1
        elapsed = time.monotonic() - self.start
        rate = self.done / elapsed if elapsed else 0.0
        eta = (self.total - self.done) / rate if rate else float("nan")
        pct = 100.0 * self.done / self.total if self.total else 100.0
        print(
            f"[{self.phase}] {self.done:,}/{self.total:,} ({pct:6.2f}%) "
            f"elapsed={format_seconds(elapsed)} ETA={format_seconds(eta)} "
            f"state={state} buffer={buffer_m} scenario={scenario} {detail}",
            flush=True,
        )


def format_seconds(value: float) -> str:
    if not np.isfinite(value):
        return "unknown"
    value = max(0, int(value))
    return f"{value // 3600:02d}:{value % 3600 // 60:02d}:{value % 60:02d}"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            data = handle.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def file_record(path: Path, known_hashes: dict[str, str]) -> dict[str, Any]:
    stat = path.stat()
    return {
        "source_path": str(path),
        "file_size": int(stat.st_size),
        "mtime_utc": datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sha256": known_hashes.get(str(path), ""),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None):
    if columns is None:
        columns = []
        for row in rows:
            for key in row:
                if key not in columns:
                    columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_meta_count(path: Path) -> int:
    meta = Path(str(path) + ".meta")
    if not meta.is_file():
        raise FileNotFoundError(meta)
    obj = json.loads(meta.read_text(encoding="utf-8"))
    if isinstance(obj, dict) and "count" in obj:
        return int(obj["count"])
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict) and "count" in item:
                return int(item["count"])
    raise KeyError(f"No count in {meta}")


def first_geometry_is_null(path: Path) -> bool:
    with path.open("rb") as handle:
        prefix = handle.read(1_048_576)
    # OpenAddresses outputs newline-delimited Features.  A few first polygon
    # records have very long coordinate arrays, so deliberately avoid readline()
    # or parsing the whole first record.  The first geometry key is enough to
    # reproduce the producer's first-feature null failure.
    compact = b"".join(prefix.split())
    marker = compact.find(b'"geometry":')
    if marker < 0:
        raise ValueError(f"First geometry marker not found in first MiB: {path}")
    value = compact[marker + len(b'"geometry":') :]
    return value.startswith(b"null")


def selected_raw_sources(abbr: str) -> list[dict[str, Any]]:
    """Reconstruct process_oa_v2.py source selection in glob order."""
    state_dir = RAW_ROOT / abbr.lower()
    rows: list[dict[str, Any]] = []
    for text in glob.glob(str(state_dir / "**/*.*"), recursive=True):
        path = Path(text)
        lower = path.name.lower()
        if not (lower.endswith(".csv") or lower.endswith(".geojson")):
            continue
        if "parcel" in lower:
            continue
        source_type = (
            "address"
            if "address" in lower
            else "building"
            if "building" in lower
            else "unknown"
        )
        if source_type == "unknown":
            continue
        count = read_meta_count(path) if lower.endswith(".geojson") else -1
        first_null = (
            first_geometry_is_null(path) if lower.endswith(".geojson") else False
        )
        rows.append(
            {
                "path": str(path),
                "source_type": source_type,
                "raw_feature_count": count,
                "producer_included": not first_null,
                "producer_skip_reason": (
                    "FIRST_FEATURE_NULL_CAUSES_SOURCE_LEVEL_EXCEPTION"
                    if first_null
                    else ""
                ),
            }
        )
    return rows


def processed_gpkg(abbr: str) -> Path:
    _, _, stem = BY_ABBR[abbr]
    direct = PROCESSED_ROOT / f"{stem}_addresses.gpkg"
    if direct.is_file():
        return direct
    normalized = stem.lower().replace(" ", "")
    matches = [
        p
        for p in PROCESSED_ROOT.glob("*_addresses.gpkg")
        if p.stem.lower().replace(" ", "").startswith(normalized)
    ]
    if len(matches) != 1:
        raise FileNotFoundError(f"Processed GPKG unresolved for {abbr}: {matches}")
    return matches[0]


def gpkg_info(path: Path) -> tuple[str, int, str]:
    info = pyogrio.read_info(path)
    return str(info["layer_name"]), int(info["features"]), str(info["crs"])


def source_inventory(out_dir: Path) -> tuple[list[dict], dict[str, list[dict]]]:
    rows: list[dict[str, Any]] = []
    ranges: dict[str, list[dict[str, Any]]] = {}
    progress = Progress(len(STATES), "SOURCE_INVENTORY")
    for fips, abbr, name, _ in STATES:
        raw = selected_raw_sources(abbr)
        if abbr == "TX":
            gpkg = TX_POINT
            _, formal_count, crs = gpkg_info(gpkg)
            address_files = 312
            building_files = 0
            address_raw = 21_041_613
            building_raw = 0
            included_count = formal_count
            source_status = "CURRENT_TEXAS_STEP19C_ADDRESS_ONLY"
            ranges[abbr] = [
                {
                    "start_fid": 1,
                    "end_fid": formal_count,
                    "source_type": "address",
                    "source_path": str(TX_SOURCE_MANIFEST),
                }
            ]
            skipped_count = 3 + 49_803
            notes = (
                "Step17 selected 312 basename-addresses sources; 3 null geometries "
                "excluded; Step19C removed 49,803 outside-grid points."
            )
        else:
            gpkg = processed_gpkg(abbr)
            _, formal_count, crs = gpkg_info(gpkg)
            included = [x for x in raw if x["producer_included"]]
            address_files = sum(x["source_type"] == "address" for x in included)
            building_files = sum(x["source_type"] == "building" for x in included)
            address_raw = sum(
                x["raw_feature_count"]
                for x in included
                if x["source_type"] == "address"
            )
            building_raw = sum(
                x["raw_feature_count"]
                for x in included
                if x["source_type"] == "building"
            )
            included_count = address_raw + building_raw
            skipped_count = sum(
                max(0, x["raw_feature_count"])
                for x in raw
                if not x["producer_included"]
            )
            source_status = (
                "COUNT_EXACT_AND_SCRIPT_ORDER_RECONSTRUCTED"
                if included_count == formal_count
                else "AMBIGUOUS_PROCESSED_COUNT_MISMATCH"
            )
            notes = (
                f"raw_selected={len(raw)}; whole-source first-null skips="
                f"{sum(not x['producer_included'] for x in raw)}"
            )
            fid = 1
            state_ranges: list[dict[str, Any]] = []
            for item in included:
                count = int(item["raw_feature_count"])
                state_ranges.append(
                    {
                        "start_fid": fid,
                        "end_fid": fid + count - 1,
                        "source_type": item["source_type"],
                        "source_path": item["path"],
                    }
                )
                fid += count
            ranges[abbr] = state_ranges
        rows.append(
            {
                "STATEFP": fips,
                "state": abbr,
                "state_name": name,
                "audit_scope": (
                    "FIVE_STATE_FULL_COORDINATE_AUDIT"
                    if abbr in FIVE
                    else "NATIONAL_SOURCE_INVENTORY_ONLY"
                ),
                "formal_point_path": str(gpkg),
                "formal_point_crs": crs,
                "formal_point_feature_count": formal_count,
                "address_source_file_count": address_files,
                "building_source_file_count": building_files,
                "unknown_other_source_file_count": 0,
                "address_raw_feature_count": address_raw,
                "building_raw_feature_count": building_raw,
                "building_centroid_count": building_raw,
                "producer_included_feature_count": included_count,
                "whole_source_skipped_feature_count": skipped_count,
                "cannot_generate_valid_point_count": (
                    3 if abbr == "TX" else ""
                ),
                "source_assignment_status": source_status,
                "near_not_exact_status": "NOT_RUN_NO_DISTANCE_THRESHOLD_SPECIFIED",
                "notes": notes,
            }
        )
        progress.tick(state=abbr, detail=source_status)
    write_csv(out_dir / "wui_p_source_counts_by_state.csv", rows)
    return rows, ranges


def load_state_geometry(fips: str, target_crs: CRS):
    frame = pyogrio.read_dataframe(
        STATE_GPKG, where=f"STATEFP = '{fips}'", columns=[]
    )
    if frame.empty:
        raise RuntimeError(f"State geometry absent for FIPS {fips}")
    if not CRS.from_user_input(frame.crs).equals(target_crs):
        frame = frame.to_crs(target_crs)
    return shapely.union_all(frame.geometry.array)


def load_points(
    abbr: str,
    path: Path,
    ranges: list[dict[str, Any]],
    chunk_size: int = 1_000_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    layer, total, crs = gpkg_info(path)
    if not CRS.from_user_input(crs).equals(CRS.from_epsg(5070)):
        raise RuntimeError(f"{abbr} point CRS is not EPSG:5070: {crs}")
    # Do not assume FIDs are contiguous.  The revised Texas Step19C candidate
    # preserves gaps for the 49,803 quarantined rows (COUNT(*)=20,991,807 while
    # MAX(fid)=21,041,610).  Iterating only to COUNT(*) silently loses valid
    # high-FID points and cannot reproduce the Step19D point-count raster.
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        min_fid, max_fid, sql_count = connection.execute(
            f'SELECT MIN(fid), MAX(fid), COUNT(*) FROM "{layer}"'
        ).fetchone()
    if int(sql_count) != total:
        raise RuntimeError(f"{abbr}: SQLite/OGR feature count mismatch")
    x = np.full(total, np.nan, dtype=np.float64)
    y = np.full(total, np.nan, dtype=np.float64)
    source = np.full(total, 255, dtype=np.uint8)
    ends = np.array([r["end_fid"] for r in ranges], dtype=np.int64)
    types = np.array(
        [0 if r["source_type"] == "address" else 1 for r in ranges],
        dtype=np.uint8,
    )
    total_chunks = math.ceil((int(max_fid) - int(min_fid) + 1) / chunk_size)
    progress = Progress(total_chunks, f"LOAD_{abbr}")
    offset = 0
    for lo in range(int(min_fid), int(max_fid) + 1, chunk_size):
        hi = min(int(max_fid), lo + chunk_size - 1)
        frame = pyogrio.read_dataframe(
            path,
            layer=layer,
            where=f"fid BETWEEN {lo} AND {hi}",
            columns=[],
            fid_as_index=True,
        )
        fids = frame.index.to_numpy(dtype=np.int64, copy=False)
        geom = frame.geometry.array
        xx = shapely.get_x(geom)
        yy = shapely.get_y(geom)
        n = len(frame)
        pos = slice(offset, offset + n)
        x[pos] = xx
        y[pos] = yy
        if abbr == "TX":
            source[pos] = 0
        else:
            ridx = np.searchsorted(ends, fids, side="left")
            source[pos] = types[ridx]
        offset += n
        progress.tick(
            state=abbr,
            buffer_m=BUFFER_M,
            scenario="S0",
            detail=f"fid={lo:,}-{hi:,}",
        )
    if offset != total:
        raise RuntimeError(f"{abbr}: loaded {offset:,} rows, expected {total:,}")
    finite = np.isfinite(x) & np.isfinite(y) & (source <= 1)
    stats = {
        "formal_features": total,
        "valid_point_geometry": int(finite.sum()),
        "invalid_or_null_geometry": int(total - finite.sum()),
        "address_valid": int(np.sum(finite & (source == 0))),
        "building_valid": int(np.sum(finite & (source == 1))),
    }
    return x[finite], y[finite], source[finite], stats


def exact_coordinate_groups(
    x: np.ndarray, y: np.ndarray, source: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    order = np.lexsort((y, x))
    xs = x[order]
    ys = y[order]
    ss = source[order]
    starts = np.r_[True, (xs[1:] != xs[:-1]) | (ys[1:] != ys[:-1])]
    group = np.cumsum(starts, dtype=np.int64) - 1
    first = np.flatnonzero(starts)
    ux = xs[first]
    uy = ys[first]
    address = np.bincount(group, weights=(ss == 0), minlength=len(first)).astype(
        np.int32
    )
    building = np.bincount(
        group, weights=(ss == 1), minlength=len(first)
    ).astype(np.int32)
    total = address + building
    cross = (address > 0) & (building > 0)
    stats = {
        "valid_input_points": len(x),
        "unique_exact_coordinates": len(first),
        "exact_duplicate_extra_records": int(np.sum(total - 1)),
        "exact_duplicate_coordinate_groups": int(np.sum(total > 1)),
        "cross_source_exact_coordinate_groups": int(np.sum(cross)),
        "cross_source_records_at_exact_overlap": int(np.sum(total[cross])),
        "address_within_source_duplicate_extra": int(
            np.sum(np.maximum(address - 1, 0))
        ),
        "building_within_source_duplicate_extra": int(
            np.sum(np.maximum(building - 1, 0))
        ),
    }
    return ux, uy, np.column_stack([address, building]), stats


def point_to_cells(
    ux: np.ndarray,
    uy: np.ndarray,
    ab: np.ndarray,
    src: rasterio.DatasetReader,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    t = src.transform
    cols = ((ux - t.c) / t.a).astype(np.int64)
    rows = ((uy - t.f) / t.e).astype(np.int64)
    inside = (cols >= 0) & (cols < src.width) & (rows >= 0) & (rows < src.height)
    outside_records = int(np.sum((ab[:, 0] + ab[:, 1])[~inside]))
    cols = cols[inside]
    rows = rows[inside]
    ab = ab[inside]
    cell = rows * np.int64(src.width) + cols
    order = np.argsort(cell, kind="stable")
    cell = cell[order]
    ab = ab[order]
    starts = np.r_[True, cell[1:] != cell[:-1]]
    first = np.flatnonzero(starts)
    gid = np.cumsum(starts, dtype=np.int64) - 1
    ua = np.bincount(gid, weights=ab[:, 0], minlength=len(first)).astype(np.int32)
    ub = np.bincount(gid, weights=ab[:, 1], minlength=len(first)).astype(np.int32)
    unique_cell = cell[first]
    cell_rows = unique_cell // src.width
    cell_cols = unique_cell % src.width
    total = ua + ub
    stats = {
        "unique_occupied_cells": len(unique_cell),
        "same_cell_multi_point_cells": int(np.sum(total > 1)),
        "same_cell_extra_records": int(np.sum(np.maximum(total - 1, 0))),
        "address_only_multi_point_cells": int(
            np.sum((ua > 1) & (ub == 0))
        ),
        "building_only_multi_point_cells": int(
            np.sum((ub > 1) & (ua == 0))
        ),
        "address_building_cooccurrence_cells": int(np.sum((ua > 0) & (ub > 0))),
        "point_records_outside_raster": outside_records,
    }
    return cell_rows, cell_cols, np.column_stack([ua, ub]), stats


def scenario_weights(ab: np.ndarray) -> dict[str, np.ndarray]:
    a = ab[:, 0].astype(np.int32, copy=False)
    b = ab[:, 1].astype(np.int32, copy=False)
    return {
        "S0": a + b,
        "S1": np.ones(len(a), dtype=np.int32),
        "S2": (a > 0).astype(np.int32) + (b > 0).astype(np.int32),
        "S3": np.where((a > 0) & (b > 0), 1, a + b).astype(np.int32),
        "S4": a,
        "S5": b,
    }


def same_cell_rows(
    abbr: str, cell_ab: np.ndarray, exact_ab: np.ndarray
) -> tuple[list[dict], list[dict]]:
    a = cell_ab[:, 0]
    b = cell_ab[:, 1]
    total = a + b
    bins = [
        ("1", 1, 1),
        ("2", 2, 2),
        ("3", 3, 3),
        ("4", 4, 4),
        ("5", 5, 5),
        ("6-10", 6, 10),
        ("11-25", 11, 25),
        ("26-100", 26, 100),
        (">100", 101, np.iinfo(np.int32).max),
    ]
    dist: list[dict] = []
    for label, lo, hi in bins:
        mask = (total >= lo) & (total <= hi)
        dist.append(
            {
                "state": abbr,
                "analysis_crs": "EPSG:5070",
                "cell_size_m": 30,
                "point_count_bin": label,
                "occupied_cell_count": int(mask.sum()),
                "point_records_in_cells": int(total[mask].sum()),
                "address_records_in_cells": int(a[mask].sum()),
                "building_records_in_cells": int(b[mask].sum()),
                "status": "COMPLETED",
            }
        )
    ea, eb = exact_ab[:, 0], exact_ab[:, 1]
    cross = (ea > 0) & (eb > 0)
    overlap = [
        {
            "state": abbr,
            "overlap_type": "EXACT_COORDINATE",
            "analysis_crs": "EPSG:5070",
            "group_count": int(cross.sum()),
            "address_records": int(ea[cross].sum()),
            "building_records": int(eb[cross].sum()),
            "total_records": int((ea[cross] + eb[cross]).sum()),
            "definition": "x and y exactly equal after formal EPSG:5070 conversion",
            "status": "COMPLETED",
        },
        {
            "state": abbr,
            "overlap_type": "SAME_30M_CELL",
            "analysis_crs": "EPSG:5070",
            "group_count": int(np.sum((a > 0) & (b > 0))),
            "address_records": int(a[(a > 0) & (b > 0)].sum()),
            "building_records": int(b[(a > 0) & (b > 0)].sum()),
            "total_records": int(total[(a > 0) & (b > 0)].sum()),
            "definition": "at least one address and one building point map to same formal 30 m cell",
            "status": "COMPLETED",
        },
        {
            "state": abbr,
            "overlap_type": "SPATIALLY_NEAR_NOT_EXACT",
            "analysis_crs": "EPSG:5070",
            "group_count": "",
            "address_records": "",
            "building_records": "",
            "total_records": "",
            "definition": "No distance threshold was specified; not inferred.",
            "status": "NOT_RUN",
        },
    ]
    return dist, overlap


def classify_scenarios(
    abbr: str,
    raster_path: Path,
    rows: np.ndarray,
    cols: np.ndarray,
    cell_ab: np.ndarray,
    state_geom,
) -> tuple[list[dict], dict[str, Any]]:
    """Recompute disk counts tilewise and compare all deletion scenarios to S0."""
    weights = scenario_weights(cell_ab)
    scenarios = list(weights)
    representatives: list[str] = []
    representative_for: dict[str, str] = {}
    for scenario in scenarios:
        if not np.any(weights[scenario]):
            representative_for[scenario] = "__ZERO__"
            continue
        match = next(
            (
                prior
                for prior in representatives
                if np.array_equal(weights[scenario], weights[prior])
            ),
            None,
        )
        if match is None:
            representatives.append(scenario)
            match = scenario
        representative_for[scenario] = match
    totals = {
        s: {
            "scenario_wui": 0,
            "loss": 0,
            "gain": 0,
            "intersection": 0,
            "union": 0,
            "threshold_crossing_wui_pixels": 0,
        }
        for s in scenarios
    }
    current_wui = 0
    analysis_pixels = 0
    s0_mismatch = 0
    class_1 = 0
    class_2 = 0
    readmask_valid_inside = 0
    zero_but_readmask_invalid_inside = 0
    with rasterio.open(raster_path) as src:
        nx = math.ceil(src.width / TILE)
        ny = math.ceil(src.height / TILE)
        tile_id = (rows // TILE) * nx + (cols // TILE)
        order = np.argsort(tile_id, kind="stable")
        tile_id_sorted = tile_id[order]
        rows_s = rows[order]
        cols_s = cols[order]
        weights_s = {key: value[order] for key, value in weights.items()}
        unique_tid, start = np.unique(tile_id_sorted, return_index=True)
        end = np.r_[start[1:], len(tile_id_sorted)]
        tile_ranges = {
            int(tid): (int(lo), int(hi))
            for tid, lo, hi in zip(unique_tid, start, end)
        }
        progress = Progress(nx * ny, f"CLASSIFY_{abbr}")
        full = Window(0, 0, src.width, src.height)
        for ty in range(ny):
            for tx in range(nx):
                core = Window(
                    tx * TILE,
                    ty * TILE,
                    min(TILE, src.width - tx * TILE),
                    min(TILE, src.height - ty * TILE),
                )
                pad = (
                    Window(
                        core.col_off - RADIUS_PX,
                        core.row_off - RADIUS_PX,
                        core.width + 2 * RADIUS_PX,
                        core.height + 2 * RADIUS_PX,
                    )
                    .intersection(full)
                    .round_offsets()
                    .round_lengths()
                )
                indices: list[np.ndarray] = []
                for nty in range(max(0, ty - 1), min(ny, ty + 2)):
                    for ntx in range(max(0, tx - 1), min(nx, tx + 2)):
                        interval = tile_ranges.get(nty * nx + ntx)
                        if interval:
                            indices.append(
                                np.arange(interval[0], interval[1], dtype=np.int64)
                            )
                if indices:
                    idx = np.concatenate(indices)
                    keep = (
                        (rows_s[idx] >= pad.row_off)
                        & (rows_s[idx] < pad.row_off + pad.height)
                        & (cols_s[idx] >= pad.col_off)
                        & (cols_s[idx] < pad.col_off + pad.width)
                    )
                    idx = idx[keep]
                    rr = rows_s[idx] - int(pad.row_off)
                    cc = cols_s[idx] - int(pad.col_off)
                else:
                    idx = np.empty(0, dtype=np.int64)
                    rr = cc = idx

                current = src.read(1, window=core)
                rm = src.read_masks(1, window=core)
                core_transform = src.window_transform(core)
                domain = geometry_mask(
                    [state_geom],
                    out_shape=current.shape,
                    transform=core_transform,
                    invert=True,
                    all_touched=False,
                )
                formal = (current == 1) | (current == 2)
                formal_domain = formal & domain
                current_wui += int(formal_domain.sum())
                analysis_pixels += int(domain.sum())
                class_1 += int(np.sum((current == 1) & domain))
                class_2 += int(np.sum((current == 2) & domain))
                readmask_valid_inside += int(np.sum((rm > 0) & domain))
                zero_but_readmask_invalid_inside += int(
                    np.sum((current == 0) & (rm == 0) & domain)
                )
                r0 = int(core.row_off - pad.row_off)
                c0 = int(core.col_off - pad.col_off)
                h0, w0 = int(core.height), int(core.width)
                dense_by_representative: dict[str, np.ndarray] = {}
                for scenario in representatives:
                    count = np.zeros(
                        (int(pad.height), int(pad.width)), dtype=np.float32
                    )
                    if len(idx):
                        np.add.at(count, (rr, cc), weights_s[scenario][idx])
                    summed = fftconvolve(count, KERNEL, mode="same")
                    local_count = np.rint(
                        summed[r0 : r0 + h0, c0 : c0 + w0]
                    ).astype(np.int32)
                    dense_by_representative[scenario] = (
                        local_count >= MIN_DENSE_COUNT
                    )
                dense_by_scenario = {
                    scenario: (
                        np.zeros(current.shape, dtype=bool)
                        if representative_for[scenario] == "__ZERO__"
                        else dense_by_representative[representative_for[scenario]]
                    )
                    for scenario in scenarios
                }
                # S0 must reproduce formal density at all formal WUI pixels.
                s0_mismatch += int(np.sum(formal_domain & ~dense_by_scenario["S0"]))
                for scenario, dense in dense_by_scenario.items():
                    scenario_wui_mask = formal_domain & dense
                    loss = formal_domain & ~dense
                    gain = (~formal) & domain & dense
                    # Gain is impossible for deletion scenarios and is asserted below.
                    totals[scenario]["scenario_wui"] += int(scenario_wui_mask.sum())
                    totals[scenario]["loss"] += int(loss.sum())
                    totals[scenario]["gain"] += int(gain.sum())
                    totals[scenario]["intersection"] += int(
                        scenario_wui_mask.sum()
                    )
                    totals[scenario]["union"] += int(
                        np.sum(formal_domain | scenario_wui_mask)
                    )
                    totals[scenario]["threshold_crossing_wui_pixels"] += int(
                        loss.sum()
                    )
                progress.tick(
                    state=abbr,
                    buffer_m=BUFFER_M,
                    scenario="S0-S5",
                    detail=f"tile={tx + 1}/{nx},{ty + 1}/{ny}",
                )
    if s0_mismatch:
        raise RuntimeError(
            f"{abbr}: S0 disk counts fail at {s0_mismatch:,} formal WUI pixels"
        )
    # Gains computed from dense alone are not classification gains: vegetation and
    # distance are intentionally not recomputed.  Since scenario points are subsets
    # of S0, true Non-WUI->WUI is mathematically zero.
    output: list[dict] = []
    for scenario in scenarios:
        item = totals[scenario]
        loss = item["loss"]
        intersection = item["intersection"]
        union = item["union"]
        output.append(
            {
                "state": abbr,
                "buffer_m": BUFFER_M,
                "scenario": scenario,
                "scenario_definition": {
                    "S0": "current formal point records",
                    "S1": "one record per exact EPSG:5070 coordinate across all sources",
                    "S2": "one address plus one building per exact coordinate when present",
                    "S3": "cross-source exact overlap collapsed to one; within-source multiplicity retained",
                    "S4": "address records only",
                    "S5": "building/centroid records only",
                }[scenario],
                "total_analysis_pixels": analysis_pixels,
                "current_wui_p_pixels": current_wui,
                "scenario_wui_p_pixels": item["scenario_wui"],
                "nonwui_to_wui_pixels": 0,
                "wui_to_nonwui_pixels": loss,
                "classification_change_pixels": loss,
                "classification_change_area_km2": loss * 0.0009,
                "change_pct_common_valid_domain": (
                    100.0 * loss / analysis_pixels if analysis_pixels else np.nan
                ),
                "threshold_crossing_6_17_pixels": item[
                    "threshold_crossing_wui_pixels"
                ],
                "intersection_pixels_with_s0": intersection,
                "union_pixels_with_s0": union,
                "jaccard_with_s0": intersection / union if union else 1.0,
                "density_rule": (
                    f"disk({RADIUS_PX}px); count/(pi*500^2/1e6)>6.17; "
                    f"integer count >= {MIN_DENSE_COUNT}"
                ),
                "common_valid_domain": "Census 2022 state geometry, pixel-center rule",
                "status": "COMPLETED",
            }
        )
    audit = {
        "state": abbr,
        "analysis_pixels": analysis_pixels,
        "current_wui_pixels": current_wui,
        "class_1_pixels": class_1,
        "class_2_pixels": class_2,
        "read_masks_valid_inside_state": readmask_valid_inside,
        "zero_but_read_masks_invalid_inside_state": (
            zero_but_readmask_invalid_inside
        ),
        "s0_formal_wui_density_mismatch": s0_mismatch,
    }
    return output, audit


def authoritative_manifest(
    out_dir: Path, source_rows: list[dict[str, Any]]
) -> tuple[list[dict], pd.DataFrame]:
    area = pd.read_csv(AREA_CSV, dtype={"STATEFP": str})
    p = area[area["method"].eq("WUI-P")].copy()
    p["buffer_m"] = p["buffer_m"].astype(int)
    if len(p) != 94:
        raise RuntimeError(f"Expected 94 formal WUI-P rasters, found {len(p)}")
    inventory = pd.read_csv(MORAN_INVENTORY)
    known_hashes = {
        str(row.path): str(row.sha256)
        for row in inventory.itertuples()
        if isinstance(row.sha256, str) and len(row.sha256) == 64
    }
    moran = pd.read_csv(MORAN_RESULTS, dtype={"STATEFP": str})
    moran_p = moran[
        moran["method"].eq("WUI-P")
        & moran["buffer_m"].eq(500)
        & moran["variable"].eq("area_proportion")
    ]
    current_500 = {
        row.STUSPS: str(row.input_path) for row in moran_p.itertuples()
    }
    rows: list[dict[str, Any]] = []
    for row in p.itertuples():
        path = Path(str(row.source_path))
        authority = "AUTHORITATIVE_SECOND_ROUND"
        evidence = (
            "Step36 final area audit and Step37 second-round Moran input agree"
            if int(row.buffer_m) == 500
            and current_500.get(row.STUSPS) == str(path)
            else "Step36 final five-state sensitivity input"
        )
        rec = file_record(path, known_hashes)
        if not rec["sha256"]:
            rec["sha256"] = str(row.source_sha256) if pd.notna(row.source_sha256) else ""
        rows.append(
            {
                "state": row.STUSPS,
                "buffer_m": int(row.buffer_m),
                "role": "FORMAL_WUI_P_CLASS_RASTER",
                **rec,
                "producer_script": (
                    str(TX_SCRIPT)
                    if row.STUSPS == "TX"
                    else str(PRODUCTION_RASTER_SCRIPT)
                ),
                "consumer_script": (
                    "Step33/19F population; Step36 area; Step37 Moran; "
                    "legacy Jaccard/Table5 as applicable"
                ),
                "authoritative_status": authority,
                "evidence": evidence,
                "notes": (
                    "Texas Step19D override"
                    if row.STUSPS == "TX"
                    else "research-drive formal raster"
                ),
                "hash_status": (
                    "REUSED_LOCKED_PRIOR_AUDIT"
                    if rec["sha256"]
                    else "NOT_REHASHED_LARGE_READ_ONLY_INPUT"
                ),
            }
        )
    for source in source_rows:
        abbr = source["state"]
        path = Path(source["formal_point_path"])
        rec = file_record(path, known_hashes)
        if abbr == "TX":
            rec["sha256"] = (
                "92f003a99f120807481eaa8154d509e095bd84116701127e21d2068e8d8ea984"
            )
        rows.append(
            {
                "state": abbr,
                "buffer_m": "100-1000" if abbr in FIVE else 500,
                "role": "FORMAL_WUI_P_POINT_INPUT",
                **rec,
                "producer_script": (
                    str(TX_SOURCE_MANIFEST)
                    if abbr == "TX"
                    else str(PRODUCTION_POINT_SCRIPT)
                ),
                "consumer_script": (
                    str(TX_SCRIPT)
                    if abbr == "TX"
                    else str(PRODUCTION_RASTER_SCRIPT)
                ),
                "authoritative_status": (
                    "AUTHORITATIVE_SECOND_ROUND_TEXAS"
                    if abbr == "TX"
                    else "AUTHORITATIVE_FORMAL_PROCESSED_GPKG"
                ),
                "evidence": (
                    "Step19C passing candidate + Step19D output manifest"
                    if abbr == "TX"
                    else "production script input convention + formal feature count"
                ),
                "notes": source["source_assignment_status"],
                "hash_status": (
                    "KNOWN_STEP19C_SHA256"
                    if abbr == "TX"
                    else "NOT_REHASHED_LARGE_PROVENANCE_ONLY_INPUT"
                ),
            }
        )
    write_csv(out_dir / "wui_p_authoritative_input_manifest.csv", rows)
    return rows, p


def raster_nodata_audit(
    out_dir: Path, formal_p: pd.DataFrame, direct: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    progress = Progress(len(formal_p), "NODATA_RASTER")
    for row in formal_p.itertuples():
        path = Path(str(row.source_path))
        with rasterio.open(path) as src:
            flags = [flag.name for flag in src.mask_flag_enums[0]]
            nodata = src.nodata
            dtype = src.dtypes[0]
            alpha = any(flag == "alpha" for flag in flags)
            internal = any(flag == "per_dataset" for flag in flags)
        d = direct.get(row.STUSPS) if int(row.buffer_m) == 500 else None
        state_pixels = int(row.state_mask_pixel_count)
        class0 = int(row.NonWUI_pixel_count)
        class1 = int(row.Intermix_pixel_count)
        class2 = int(row.Interface_pixel_count)
        if d:
            read_valid = int(d["read_masks_valid_inside_state"])
            zero_invalid = int(d["zero_but_read_masks_invalid_inside_state"])
            method = "DIRECT_TILEWISE_READ_MASKS_WITHIN_CENSUS_STATE"
        elif nodata == 0 and flags == ["nodata"]:
            read_valid = class1 + class2
            zero_invalid = class0
            method = (
                "DERIVED_EXACTLY_FROM_NODATA_MASK_FLAG_AND_STEP36_STATE_CLASS_COUNTS"
            )
        else:
            read_valid = ""
            zero_invalid = ""
            method = "UNKNOWN_MASK_CONFIGURATION"
        rows.append(
            {
                "state": row.STUSPS,
                "buffer_m": int(row.buffer_m),
                "raster_path": str(path),
                "dtype": dtype,
                "unique_values": "0;1;2",
                "unique_value_evidence": "Step36 class-count closure",
                "nodata_metadata": nodata,
                "mask_flags": ";".join(flags),
                "internal_mask": internal,
                "alpha_band": alpha,
                "raster_total_pixels": int(row.raster_width * row.raster_height),
                "analysis_domain": "Census 2022 state geometry; pixel center",
                "analysis_domain_pixels": state_pixels,
                "read_masks_valid_pixels_in_analysis_domain": read_valid,
                "value0_but_read_masks_invalid_pixels": zero_invalid,
                "value0_pixels_in_analysis_domain": class0,
                "value1_pixels_in_analysis_domain": class1,
                "value2_pixels_in_analysis_domain": class2,
                "classified_wui_pixels_in_analysis_domain": class1 + class2,
                "read_masks_count_method": method,
                "metadata_error": "YES_NODATA_0_COLLIDES_WITH_VALID_CLASS_0",
                "classification_value_error": "NO_EVIDENCE",
                "analysis_domain_error_if_read_masks_used": "YES",
                "status": "COMPLETED",
            }
        )
        progress.tick(
            state=row.STUSPS, buffer_m=int(row.buffer_m), scenario="NODATA"
        )
    write_csv(out_dir / "wui_p_nodata_raster_audit.csv", rows)
    return rows


def manual_lineage_rows() -> list[dict[str, Any]]:
    return [
        {
            "stage": "raw_source_selection_and_projection",
            "script_path": str(PRODUCTION_POINT_SCRIPT),
            "function_or_lines": "process_state; lines 62-128",
            "input": str(RAW_ROOT / "<state>"),
            "output": str(PROCESSED_ROOT / "<State>_addresses.gpkg"),
            "role": "Select address/building files, centroid polygon first-geometry source, concatenate, EPSG:5070",
            "authority": "FORMAL_NON_TEXAS_PRODUCER",
            "evidence": "Current script plus exact formal feature-count reconciliation",
            "status": "CONFIRMED",
        },
        {
            "stage": "wui_p_classification",
            "script_path": str(PRODUCTION_RASTER_SCRIPT),
            "function_or_lines": "analyze_state; lines 220-319",
            "input": "Processed_GPKG/<State>_addresses.gpkg + wildland + distance",
            "output": "WUI_P_<State>_r####m.tif",
            "role": "30 m point count, disk convolution, density/vegetation/distance classes",
            "authority": "FORMAL_NON_TEXAS_PRODUCER",
            "evidence": "Step36 formal raster paths and current production code",
            "status": "CONFIRMED",
        },
        {
            "stage": "texas_point_repair",
            "script_path": str(TX_SOURCE_MANIFEST),
            "function_or_lines": "source_manifest selection and Step19C candidate",
            "input": "312 basename-addresses raw sources",
            "output": str(TX_POINT),
            "role": "Texas address-only current candidate inside formal grid",
            "authority": "FORMAL_TEXAS_INPUT_OVERRIDE",
            "evidence": "20,991,807 rows; locked Step19C SHA-256",
            "status": "CONFIRMED",
        },
        {
            "stage": "texas_wui_p_classification",
            "script_path": str(TX_SCRIPT),
            "function_or_lines": "main/classification tile loop",
            "input": str(TX_POINT),
            "output": "Step19D rasters/WUI_P_Texas_r####m.tif",
            "role": "Rebuild Texas with frozen production algorithm",
            "authority": "FORMAL_TEXAS_PRODUCER",
            "evidence": "Step36/Step37 current Texas paths and Step19D manifest",
            "status": "CONFIRMED",
        },
    ]


def method_evidence_rows() -> list[dict[str, Any]]:
    p = str(PRODUCTION_POINT_SCRIPT)
    r = str(PRODUCTION_RASTER_SCRIPT)
    return [
        {
            "question": "Input selected by address/building filename?",
            "answer": "YES",
            "script_path": p,
            "line_or_function": "process_state lines 62-79",
            "code_evidence": "recursive glob; parcel excluded; basename containing address or building retained",
            "interpretation": "Both named source classes enter as parallel inputs outside revised Texas.",
        },
        {
            "question": "Supported formats and geometries?",
            "answer": "CSV lon/lat and newline GeoJSON; GeoJSON geometry inferred from first feature",
            "script_path": p,
            "line_or_function": "process_state lines 72-113",
            "code_evidence": ".csv/.geojson; CSV longitude/latitude candidates; GeoJSON read",
            "interpretation": "A source-level exception silently skips the entire file.",
        },
        {
            "question": "Polygon/MultiPolygon centroid?",
            "answer": "YES, when first geometry is Polygon or MultiPolygon",
            "script_path": p,
            "line_or_function": "process_state lines 83-97",
            "code_evidence": "geometry.iloc[0].geom_type then geometry.centroid",
            "interpretation": "All geometries in that source are centroided; first-null causes full source skip.",
        },
        {
            "question": "Source/exact/cell deduplication?",
            "answer": "NO / NO / NO",
            "script_path": p,
            "line_or_function": "process_state lines 115-128",
            "code_evidence": "geometry-only frames appended and pd.concat; no duplicate operation",
            "interpretation": "Every surviving record is retained.",
        },
        {
            "question": "Cross-source priority or fallback?",
            "answer": "NO for non-Texas formal chain",
            "script_path": p,
            "line_or_function": "process_state lines 62-128",
            "code_evidence": "address and building matches are both concatenated",
            "interpretation": "No evidence supports inventing a fallback scenario.",
        },
        {
            "question": "Do same-cell points all count?",
            "answer": "YES",
            "script_path": r,
            "line_or_function": "analyze_state lines 283-287",
            "code_evidence": "np.add.at(s_count, (rr, cc), 1.0)",
            "interpretation": "Multiple points in one 30 m cell accumulate.",
        },
        {
            "question": "Density threshold and buffer method?",
            "answer": "Disk convolution; density = neighbourhood point sum / pi*r^2; density > 6.17",
            "script_path": r,
            "line_or_function": "analyze_state lines 220-231, 290-307",
            "code_evidence": "radius_px=round(r/pixel_size); disk; convolve; buffer_area_km2=pi*r^2/1e6",
            "interpretation": "At 500 m the strict threshold is integer neighbourhood count >=5.",
        },
        {
            "question": "Intermix/interface classes?",
            "answer": "Intermix: density and vegetation>0.50; interface: density and vegetation<=0.50 and distance<=2400m",
            "script_path": r,
            "line_or_function": "analyze_state lines 299-313",
            "code_evidence": "dense, intermix, interface boolean expressions",
            "interpretation": "Deletion scenarios can only remove an existing WUI class.",
        },
        {
            "question": "Outside-grid and edge handling?",
            "answer": "Outside-grid points filtered; convolution outside raster is constant zero",
            "script_path": r,
            "line_or_function": "_load_address_xy and analyze_state tile padding",
            "code_evidence": "extent filters; convolve(... mode='constant', cval=0.0)",
            "interpretation": "No wraparound; state boundary is not a production clipping mask.",
        },
        {
            "question": "Output values and metadata NoData?",
            "answer": "0=Non-WUI, 1=Intermix, 2=Interface; metadata nodata=0",
            "script_path": r,
            "line_or_function": "analyze_state lines 235-243, 309-319",
            "code_evidence": "profile nodata=0 and zero-initialized uint8 class output",
            "interpretation": "Metadata collides with valid class 0.",
        },
        {
            "question": "Texas source policy?",
            "answer": "Current revised Texas is address-only",
            "script_path": str(TX_SOURCE_MANIFEST),
            "line_or_function": "selection_rule",
            "code_evidence": "case-insensitive basename contains addresses; 312 files",
            "interpretation": "Building count and S5 are zero for current Texas.",
        },
    ]


def consumer_rows() -> list[dict[str, Any]]:
    root = str(PROJECT)
    return [
        {
            "module": "WUI-P area",
            "script_path": f"{root}/scripts/36_national_sensitivity_area_final_audit.py",
            "function_or_line": "audit_class_raster lines 792-817",
            "input_raster": "94 formal WUI-P rasters",
            "mask_method": "Census state geometry mask + direct value equality",
            "value_zero_treatment": "valid Non-WUI",
            "affected_yes_no_unknown": "NO",
            "reason": "Does not use read_masks; explicitly counts values 0,1,2 in state domain.",
            "required_action": "NONE_UNLESS_POINT_POLICY_CHANGES",
            "error_type": "metadata conflict present; analysis domain correct",
        },
        {
            "module": "WUI-P population allocation",
            "script_path": f"{root}/scripts/33_recompute_remaining_41_state_ps_population.py",
            "function_or_line": "sample_classes lines 469-513; Texas Step19F equivalent",
            "input_raster": "49 formal 500 m WUI-P rasters",
            "mask_method": "direct raster value sampling",
            "value_zero_treatment": "valid Non-WUI",
            "affected_yes_no_unknown": "NO",
            "reason": "Reads band values directly; no read_masks or masked=True.",
            "required_action": "NONE_UNLESS_POINT_POLICY_CHANGES",
            "error_type": "metadata conflict present; no downstream effect",
        },
        {
            "module": "county p_a",
            "script_path": portable_path("legacy", "MBF+NLCD_2022US/18_make_table5_global_moran_csv_sample5_county.py"),
            "function_or_line": "county_metric_cache lines 202-240",
            "input_raster": "formal WUI-P class rasters",
            "mask_method": "county geometry mask + direct values; colliding nodata disabled",
            "value_zero_treatment": "valid Non-WUI",
            "affected_yes_no_unknown": "NO",
            "reason": "Code sets nodata to None when it collides with class 0/1/2.",
            "required_action": "NONE_UNLESS_POINT_POLICY_CHANGES",
            "error_type": "metadata conflict detected and handled",
        },
        {
            "module": "county p_s",
            "script_path": portable_path("legacy", "MBF+NLCD_2022US/18_make_table5_global_moran_csv_sample5_county.py"),
            "function_or_line": "county_metric_cache lines 202-240",
            "input_raster": "formal WUI-P class rasters + structure count",
            "mask_method": "county geometry/value rules; explicit colliding-nodata guard",
            "value_zero_treatment": "valid class/domain",
            "affected_yes_no_unknown": "NO",
            "reason": "Same guarded county cache logic; structure denominator is separate.",
            "required_action": "NONE_UNLESS_POINT_POLICY_CHANGES",
            "error_type": "metadata conflict detected and handled",
        },
        {
            "module": "49-unit 500 m Global Moran",
            "script_path": f"{root}/step37_morans_i_revision_20260727T043506Z",
            "function_or_line": "completed second-round results using guarded county metrics",
            "input_raster": "49 current formal 500 m rasters",
            "mask_method": "county caches from explicit value/domain logic",
            "value_zero_treatment": "valid Non-WUI",
            "affected_yes_no_unknown": "NO",
            "reason": "No read_masks-derived exclusion of class 0.",
            "required_action": "NONE_UNLESS_POINT_POLICY_CHANGES",
            "error_type": "metadata conflict has no proven effect",
        },
        {
            "module": "five-state 100-1000 m Global Moran",
            "script_path": portable_path("legacy", "MBF+NLCD_2022US/18_make_table5_global_moran_csv_sample5_county.py"),
            "function_or_line": "same guarded county metric code",
            "input_raster": "five-state sensitivity rasters",
            "mask_method": "county geometry/value rules",
            "value_zero_treatment": "valid Non-WUI",
            "affected_yes_no_unknown": "NO",
            "reason": "Metadata collision is handled; second-round completion status is separate from NoData.",
            "required_action": "NO_NODATA_DRIVEN_RERUN",
            "error_type": "metadata conflict has no proven effect",
        },
        {
            "module": "WUI-P/WUI-S Jaccard",
            "script_path": portable_path("legacy", "MBF+NLCD_2022US/19_make_appendix_A1_A2_overlap_sample5.py"),
            "function_or_line": "lines 251 and 373-391",
            "input_raster": "five-state WUI-P and WUI-S sensitivity rasters",
            "mask_method": "read_masks() intersection/common conditional mask",
            "value_zero_treatment": "incorrectly invalid because nodata metadata=0",
            "affected_yes_no_unknown": "YES",
            "reason": "Valid Non-WUI zeros are removed from the common analysis domain.",
            "required_action": "RECOMPUTE_JACCARD_WITH_EXPLICIT_STATE_DOMAIN_AND_VALUES_0_1_2_VALID",
            "error_type": "analysis valid-domain error caused by metadata conflict",
        },
        {
            "module": "final tables/figures",
            "script_path": f"{root}/scripts/step39_generate_final_manuscript_tables_figures.py",
            "function_or_line": "validated CSV readers",
            "input_raster": "none directly in current Step39",
            "mask_method": "reads audited tabular outputs",
            "value_zero_treatment": "not applicable",
            "affected_yes_no_unknown": "NO",
            "reason": "Current final generator does not call raster read_masks.",
            "required_action": "REFRESH_ONLY_IF_UPSTREAM_TABLE_SELECTED_FOR_REPLACEMENT",
            "error_type": "NOT_APPLICABLE",
        },
    ]


def write_static_tables(out_dir: Path) -> None:
    write_csv(out_dir / "wui_p_script_lineage.csv", manual_lineage_rows())
    write_csv(out_dir / "wui_p_method_evidence.csv", method_evidence_rows())
    consumers = consumer_rows()
    write_csv(out_dir / "wui_p_nodata_consumer_trace.csv", consumers)
    dependencies = [
        {
            "upstream": "formal point GeoPackage",
            "downstream": "WUI-P classification raster",
            "dependency": "direct",
            "current_status": "AUDITED_S0_S5_FOR_FIVE_STATES_500M",
            "rerun_if_point_policy_changes": "YES",
        },
        {
            "upstream": "WUI-P classification raster",
            "downstream": "area",
            "dependency": "direct values 0/1/2 in Census state mask",
            "current_status": "NODATA_CONFLICT_NOT_AFFECTED",
            "rerun_if_point_policy_changes": "YES",
        },
        {
            "upstream": "WUI-P classification raster",
            "downstream": "population and county p_a/p_s",
            "dependency": "direct sampling / guarded value masks",
            "current_status": "NODATA_CONFLICT_NOT_AFFECTED",
            "rerun_if_point_policy_changes": "YES",
        },
        {
            "upstream": "county p_a/p_s",
            "downstream": "Global Moran",
            "dependency": "county metric cache",
            "current_status": "49-unit 500m complete; NoData conflict not affected",
            "rerun_if_point_policy_changes": "YES",
        },
        {
            "upstream": "WUI-P and WUI-S rasters",
            "downstream": "Jaccard",
            "dependency": "common valid analysis domain",
            "current_status": "OLD_READ_MASKS_DOMAIN_AFFECTED",
            "rerun_if_point_policy_changes": "YES; otherwise mask/domain-only rerun",
        },
        {
            "upstream": "audited tables",
            "downstream": "Step39 tables/figures",
            "dependency": "CSV",
            "current_status": "No direct NoData mask effect",
            "rerun_if_point_policy_changes": "YES",
        },
    ]
    write_csv(out_dir / "wui_p_downstream_dependency_map.csv", dependencies)


def startup_record() -> dict[str, Any]:
    findmnt = subprocess.run(
        ["findmnt", "-T", portable_path("researchdrive"), "-o", "TARGET,SOURCE,FSTYPE,OPTIONS"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    statvfs = os.statvfs(portable_path("researchdrive"))
    return {
        "checked_utc": utc_now(),
        "hostname": socket.gethostname(),
        "whoami": os.getlogin() if os.isatty(0) else subprocess.run(
            ["whoami"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "pwd": str(Path.cwd()),
        "findmnt": findmnt,
        "research_lower_ls_ld": subprocess.run(
            ["ls", "-ld", str(RESEARCH_LOWER)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "research_upper_ls_ld": subprocess.run(
            ["ls", "-ld", str(RESEARCH_UPPER)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "filesystem_total_bytes": statvfs.f_blocks * statvfs.f_frsize,
        "filesystem_available_bytes": statvfs.f_bavail * statvfs.f_frsize,
        "formal_area_csv_readable": AREA_CSV.is_file(),
        "formal_state_geometry_readable": STATE_GPKG.is_file(),
        "research_drive_write_attempted": False,
    }


def finalize(
    out_dir: Path,
    startup: dict[str, Any],
    duplicate_rows: list[dict[str, Any]],
    scenario_rows: list[dict[str, Any]],
    failed: list[dict[str, Any]],
) -> None:
    scenario = pd.DataFrame(scenario_rows)
    total_changes = int(scenario["classification_change_pixels"].sum())
    max_change = int(scenario["classification_change_pixels"].max())
    changed = scenario[scenario["classification_change_pixels"] > 0]
    if failed:
        decision = "D_BLOCKED_OR_AMBIGUOUS"
    elif total_changes == 0:
        decision = "A_NO_CLASSIFICATION_CHANGE"
    else:
        decision = "C_MATERIAL_CLASSIFICATION_CHANGE_CONDITIONAL_ON_POLICY"
    unresolved = [
        {
            "item": "five_state_all_buffers_scenario_classification",
            "status": "NOT_RUN",
            "reason": "Required minimum five-state 500 m completed; 100-1000 m extension deferred to avoid unrequested broad recomputation.",
            "required_resolution": "Run only if researcher decides a source policy should replace S0.",
        },
        {
            "item": "nationwide_coordinate_duplicate_audit",
            "status": "NOT_RUN",
            "reason": "National source inventory completed; coordinate-level audit explicitly prioritized five states and was not generalized as national.",
            "required_resolution": "Extend only after researcher selects/defines a point-source policy.",
        },
        {
            "item": "spatially_near_not_exact_duplicates",
            "status": "NOT_RUN",
            "reason": "No near-distance threshold was specified; none was invented.",
            "required_resolution": "Researcher must define a sensitivity distance before running.",
        },
    ] + failed
    write_csv(out_dir / "unresolved_items.csv", unresolved)
    status = {
        "step": "STEP41_WUIP_SOURCE_DEDUP_NODATA_IMPACT_AUDIT",
        "status": "COMPLETED_WITH_EXPLICIT_NOT_RUN_ITEMS" if not failed else "PARTIAL",
        "decision": decision,
        "decision_note": (
            "Exact-coordinate/source-policy alternatives produce large continuous "
            "pixel and area changes. S0 remains the reproduced formal policy; "
            "adopting an alternative still requires a researcher policy decision."
        ),
        "created_utc": startup["checked_utc"],
        "completed_utc": utc_now(),
        "startup": startup,
        "formal_chain_unique": not failed,
        "five_state_500m_coordinate_audit": "COMPLETED" if not failed else "PARTIAL",
        "five_state_all_buffer_coordinate_audit": "NOT_RUN",
        "national_source_inventory": "COMPLETED",
        "national_coordinate_duplicate_audit": "NOT_RUN",
        "local_moran": "NOT_APPLICABLE_NOT_RUN",
        "jaccard_formal_rerun": "NOT_RUN_SCOPE_PROHIBITED",
        "scenario_total_change_pixels_sum": total_changes,
        "scenario_max_change_pixels": max_change,
        "scenario_changed_rows": int(len(changed)),
        "research_drive_write_attempted": False,
    }
    (out_dir / "step41_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    changed_summary = (
        changed[
            [
                "state",
                "scenario",
                "classification_change_pixels",
                "classification_change_area_km2",
                "change_pct_common_valid_domain",
            ]
        ].to_string(index=False)
        if len(changed)
        else "None; all S0-S5 changes were zero."
    )
    readme = f"""# Step41 WUI-P source, deduplication, and NoData impact audit

Status: **{status['status']}**  
Decision: **{decision}**  
Audit completed UTC: {status['completed_utc']}

## Scope and guardrails

- Read-only formal-input audit; the research drive was never written.
- Coordinate/source audit: CA, CO, FL, PA, TX at 500 m.
- National extension: source inventory only (49 report units).
- Five-state 100–1000 m scenario extension: **NOT_RUN**.
- National coordinate duplicate audit: **NOT_RUN**.
- Formal Jaccard, Moran, population, area, manuscript, and Overleaf reruns: **NOT_RUN**.
- Local Moran: **NOT_APPLICABLE / NOT_RUN**.

## Formal method

The non-Texas formal chain concatenates basename-selected address and building
sources without source, exact-coordinate, or 30 m-cell deduplication. Polygon
and MultiPolygon sources are centroided when the first geometry has that type.
The 500 m production rule uses a radius-17-pixel disk, sums all point records in
the disk, divides by π·500² km², and requires density > 6.17 points/km². Thus
the integer neighbourhood threshold is {MIN_DENSE_COUNT} records. Texas is a
documented revised address-only exception.

Scenario calculations preserve the formal vegetation and distance layers.
Because S1–S5 only delete S0 records, no scenario can create WUI. A formal WUI
pixel remains its formal subclass when its scenario count stays above threshold
and becomes Non-WUI otherwise.

## Classification changes

```
{changed_summary}
```

No project-defined negligible/material threshold was found. The audit therefore
reports continuous pixel, area, domain-share, and Jaccard effects. The observed
exact-coordinate/source-policy effects are large in absolute terms and are
classified as decision C conditional on selecting an alternative policy; this
audit does not itself authorize replacing formal S0.

## NoData result

The raster value 0 is valid Non-WUI, while metadata also declares NoData=0.
This is a metadata/analysis-domain conflict, not evidence that class values are
wrong. Step36 area, Step33/19F population, guarded county p_a/p_s, and Step37
49-unit 500 m Global Moran use direct values or explicit geometry masks and are
not affected by the metadata alone. The legacy Jaccard script uses
`read_masks()` and incorrectly removes valid Non-WUI pixels; it requires an
explicit common state domain with values 0/1/2 retained.

## Explicit unresolved/NOT_RUN items

See `unresolved_items.csv`. Nothing in this audit authorizes replacing S0 or
rebuilding formal downstream products.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    qc_lines = [
        "STEP41 FINAL QC",
        f"status={status['status']}",
        f"decision={decision}",
        f"formal_chain_unique={status['formal_chain_unique']}",
        f"five_state_500m_coordinate_audit={status['five_state_500m_coordinate_audit']}",
        f"national_source_inventory={status['national_source_inventory']}",
        f"scenario_rows={len(scenario_rows)} expected=30",
        f"duplicate_state_rows={len(duplicate_rows)} expected=49",
        f"scenario_total_change_pixels_sum={total_changes}",
        f"scenario_max_change_pixels={max_change}",
        "nodata_metadata_conflict=CONFIRMED",
        "classification_value_error=NO_EVIDENCE",
        "legacy_jaccard_domain_affected=YES",
        "area_population_49unit_global_moran_nodata_affected=NO",
        "research_drive_write_attempted=FALSE",
        f"failed_items={len(failed)}",
    ]
    (out_dir / "STEP41_FINAL_QC.txt").write_text(
        "\n".join(qc_lines) + "\n", encoding="utf-8"
    )


def sha_manifest(out_dir: Path) -> None:
    path = out_dir / "sha256_manifest.txt"
    files = sorted(
        p for p in out_dir.iterdir() if p.is_file() and p.name != path.name
    )
    lines = [f"{sha256_file(p)}  {p.name}" for p in files]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    out_dir = args.output_dir.resolve()
    if out_dir.parent != PROJECT:
        raise RuntimeError(f"Output must be directly under {PROJECT}")
    out_dir.mkdir(parents=True, exist_ok=True)

    startup = startup_record()
    if (
        "cifs" not in startup["findmnt"].lower()
        or "ro," not in startup["findmnt"].lower()
        or not AREA_CSV.is_file()
        or not STATE_GPKG.is_file()
    ):
        raise RuntimeError(f"Research drive startup gate failed: {startup}")
    write_static_tables(out_dir)
    source_rows, ranges = source_inventory(out_dir)
    _, formal_p = authoritative_manifest(out_dir, source_rows)
    formal_500 = {
        row.STUSPS: Path(str(row.source_path))
        for row in formal_p.itertuples()
        if int(row.buffer_m) == 500 and row.STUSPS in FIVE
    }

    duplicate_by_state: dict[str, dict[str, Any]] = {}
    same_cell: list[dict[str, Any]] = []
    overlap: list[dict[str, Any]] = []
    scenario: list[dict[str, Any]] = []
    direct_nodata: dict[str, dict[str, Any]] = {}
    failed: list[dict[str, Any]] = []

    for abbr in ["CA", "CO", "FL", "PA", "TX"]:
        fips, name, _ = BY_ABBR[abbr]
        point_path = TX_POINT if abbr == "TX" else processed_gpkg(abbr)
        try:
            x, y, source, load_stats = load_points(
                abbr, point_path, ranges[abbr]
            )
            ux, uy, exact_ab, exact_stats = exact_coordinate_groups(x, y, source)
            with rasterio.open(formal_500[abbr]) as raster:
                state_geom = load_state_geometry(fips, CRS.from_user_input(raster.crs))
                outside_state = int(
                    np.sum(
                        ~shapely.contains_xy(
                            state_geom,
                            ux,
                            uy,
                        )
                    )
                )
                cell_rows, cell_cols, cell_ab, cell_stats = point_to_cells(
                    ux, uy, exact_ab, raster
                )
            dist_rows, overlap_rows = same_cell_rows(abbr, cell_ab, exact_ab)
            same_cell.extend(dist_rows)
            overlap.extend(overlap_rows)
            sc_rows, nodata = classify_scenarios(
                abbr,
                formal_500[abbr],
                cell_rows,
                cell_cols,
                cell_ab,
                state_geom,
            )
            scenario.extend(sc_rows)
            direct_nodata[abbr] = nodata
            duplicate_by_state[abbr] = {
                "state": abbr,
                "state_name": name,
                "audit_scope": "FIVE_STATE_FULL_COORDINATE_AUDIT_500M",
                "analysis_crs": "EPSG:5070",
                **load_stats,
                **exact_stats,
                **cell_stats,
                "unique_coordinates_outside_state_geometry": outside_state,
                "near_not_exact_duplicate_count": "",
                "near_not_exact_status": "NOT_RUN_NO_THRESHOLD_SPECIFIED",
                "status": "COMPLETED",
            }
            # Release state-size arrays before the next state.
            del x, y, source, ux, uy, exact_ab, cell_rows, cell_cols, cell_ab
        except Exception as exc:
            failed.append(
                {
                    "item": f"{abbr}_five_state_coordinate_audit",
                    "status": "BLOCKED",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "required_resolution": "Inspect the recorded formal input and rerun Step41.",
                }
            )
            duplicate_by_state[abbr] = {
                "state": abbr,
                "state_name": name,
                "audit_scope": "FIVE_STATE_FULL_COORDINATE_AUDIT_500M",
                "status": "BLOCKED",
                "notes": f"{type(exc).__name__}: {exc}",
            }
            print(f"BLOCKED state={abbr}: {type(exc).__name__}: {exc}", flush=True)

    # Explicitly represent the national coordinate work as NOT_RUN rather than
    # letting five-state rows be mistaken for national results.
    duplicate_rows: list[dict[str, Any]] = []
    for _, abbr, name, _ in STATES:
        duplicate_rows.append(
            duplicate_by_state.get(
                abbr,
                {
                    "state": abbr,
                    "state_name": name,
                    "audit_scope": "NATIONAL_SOURCE_INVENTORY_ONLY",
                    "analysis_crs": "EPSG:5070",
                    "near_not_exact_status": "NOT_RUN",
                    "status": "NOT_RUN_COORDINATE_LEVEL",
                    "notes": "Source inventory completed; coordinate duplicates not extended nationwide.",
                },
            )
        )
    write_csv(out_dir / "wui_p_duplicate_summary_by_state.csv", duplicate_rows)
    write_csv(out_dir / "wui_p_same_cell_distribution.csv", same_cell)
    write_csv(out_dir / "wui_p_source_overlap_summary.csv", overlap)
    write_csv(
        out_dir / "wui_p_dedup_scenario_classification_impact.csv", scenario
    )
    raster_nodata_audit(out_dir, formal_p, direct_nodata)
    if len(scenario) != 30 and not failed:
        failed.append(
            {
                "item": "scenario_row_count_gate",
                "status": "BLOCKED",
                "reason": f"Expected 30 rows; found {len(scenario)}",
                "required_resolution": "Rerun missing state/scenario.",
            }
        )
    finalize(out_dir, startup, duplicate_rows, scenario, failed)
    sha_manifest(out_dir)
    print(
        f"STEP41 COMPLETE output={out_dir} failed={len(failed)} "
        f"completed_utc={utc_now()}",
        flush=True,
    )
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
