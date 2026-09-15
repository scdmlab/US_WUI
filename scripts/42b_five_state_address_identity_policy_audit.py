#!/usr/bin/env python3
# Copyright (C) 2026 WUI Mapping Project Authors
# SPDX-License-Identifier: GPL-3.0-only
# Repository version: filesystem paths are loaded through repo_config.py.
# Copy config/paths.example.json to config/paths.json before a new run.
# Raw input data and large intermediate files are stored outside GitHub.

"""Step42B: five-state address identity audit and 500 m policy scenarios.

The program reads raw OpenAddresses GeoJSON in source order, preserves all
address attributes needed for identity tests as stable 128-bit keys, and never
writes to the research drive.  P1-P4 are deletion-only subsets of the legacy P0
point set.  Therefore formal vegetation/distance logic is held fixed and a
formal WUI class remains its P0 subclass exactly when the scenario's production
disk count remains above the confirmed threshold.
"""

from __future__ import annotations

from repo_config import portable_path

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import struct
import time
import unicodedata
import zlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyogrio
import rasterio
import shapely
from shapely.geometry import shape
from pyproj import CRS, Transformer
from rasterio.features import geometry_mask
from rasterio.windows import Window
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree
from skimage.morphology import disk


PROJECT = Path(portable_path("project"))
STEP41 = PROJECT / "step41_wuip_source_nodata_audit_20260727T054819Z"
STEP41_SCRIPT = PROJECT / "scripts/41_wuip_source_dedup_nodata_impact_audit.py"
AREA_CSV = (
    PROJECT
    / "step36_area_final_audit_20260724T192140Z/"
    "step36_final_unique_area_long_237.csv"
)
TX_SOURCE_MANIFEST = Path(
    portable_path("legacy", "WUI_TX_recovery/step17_gpkg_runs/texas_gpkg_rebuild_exclude_null3_20260722T173401Z/source_manifest.json")
)
STATES = ["CA", "CO", "FL", "PA", "TX"]
FIELDS = ["number", "street", "unit", "city", "district", "region", "postcode"]
TILE = 2048
RADIUS_M = 500
PIXEL_M = 30.0
RADIUS_PX = int(round(RADIUS_M / PIXEL_M))
KERNEL = disk(RADIUS_PX).astype(np.float32)
BUFFER_AREA_KM2 = math.pi * RADIUS_M**2 / 1_000_000.0
MIN_DENSE_COUNT = int(math.floor(6.17 * BUFFER_AREA_KM2)) + 1
NEAR_THRESHOLD_M = 1.0
TRANSFORMER = Transformer.from_crs(4326, 5070, always_xy=True)


def load_step41():
    spec = importlib.util.spec_from_file_location("step41", STEP41_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def stable128(data: bytes) -> tuple[int, int]:
    # Fast in-process 128-bit composite key for tens of millions of records.
    # Python's 64-bit SipHash is paired with two independent checksums.  Exact
    # equality is never inferred from coordinate alone, and keys are used only
    # for grouping in this run (not as an externally reusable identifier).
    first = hash(data) & ((1 << 64) - 1)
    second = (
        (zlib.crc32(data) & 0xFFFFFFFF) << 32
    ) | (zlib.adler32(data) & 0xFFFFFFFF)
    return first, second


def normalize(value: Any) -> str:
    # Conservative normalization: Unicode compatibility, case, and whitespace.
    # Punctuation is deliberately retained to limit false merges in P3.
    text = unicodedata.normalize("NFKC", str(value or "")).upper()
    return " ".join(text.split())


def tuple128(parts: tuple[Any, ...], tag: str) -> tuple[int, int]:
    """Fast process-local 128-bit composite of two independent SipHash tuples."""
    mask = (1 << 64) - 1
    return hash((tag, *parts)) & mask, hash((parts, tag, len(parts))) & mask


def part_bytes(parts: list[Any]) -> bytes:
    return b"\x1f".join(str(part or "").encode("utf-8") for part in parts)


def geometry_point(geometry: Any, centroid_source: bool) -> tuple[float, float] | None:
    if not isinstance(geometry, dict):
        return None
    kind = geometry.get("type")
    if centroid_source:
        try:
            item = shape(geometry).centroid
            if item.is_empty or not np.isfinite(item.x) or not np.isfinite(item.y):
                return None
            return float(item.x), float(item.y)
        except Exception:
            return None
    if kind != "Point":
        return None
    coords = geometry.get("coordinates")
    if not isinstance(coords, (list, tuple)) or len(coords) < 2:
        return None
    try:
        x, y = float(coords[0]), float(coords[1])
    except Exception:
        return None
    if not np.isfinite(x) or not np.isfinite(y):
        return None
    return x, y


class Progress:
    def __init__(self, phase: str, total: int, state: str):
        self.phase = phase
        self.total = total
        self.state = state
        self.done = 0
        self.started = time.monotonic()

    def tick(self, scenario: str, detail: str = ""):
        self.done += 1
        elapsed = time.monotonic() - self.started
        rate = self.done / elapsed if elapsed else 0
        eta = (self.total - self.done) / rate if rate else 0
        print(
            f"[{self.phase}] step=3-4/5 state={self.state} scenario={scenario} "
            f"completed={self.done}/{self.total} "
            f"percent={100*self.done/self.total:.2f}% "
            f"elapsed={elapsed:.1f}s ETA={eta:.1f}s {detail}",
            flush=True,
        )


def address_sources(m, abbr: str) -> list[dict[str, Any]]:
    if abbr == "TX":
        obj = json.loads(TX_SOURCE_MANIFEST.read_text(encoding="utf-8"))
        return [
            {
                "path": record["resolved_target"],
                "raw_feature_count": int(record["feature_count"]),
                "producer_included": True,
                "source_type": "address",
                "producer_skip_reason": "",
                "locked_sha256": record["locked_sha256"],
                "eligible_point_count": int(record["eligible_point_count"]),
            }
            for record in obj["records"]
        ]
    return [
        item
        for item in m.selected_raw_sources(abbr)
        if item["source_type"] == "address" and item["producer_included"]
    ]


def raster_for_state(abbr: str) -> Path:
    area = pd.read_csv(AREA_CSV)
    row = area[
        area["method"].eq("WUI-P")
        & area["STUSPS"].eq(abbr)
        & area["buffer_m"].eq(500)
    ]
    if len(row) != 1:
        raise RuntimeError(f"Formal 500 m raster unresolved for {abbr}")
    return Path(row.iloc[0]["source_path"])


def parse_state(
    m, abbr: str, raster_path: Path
) -> tuple[dict[str, np.ndarray], dict[str, Any], list[dict[str, Any]]]:
    sources = address_sources(m, abbr)
    total_expected = sum(int(item["raw_feature_count"]) for item in sources)
    progress = Progress("PARSE_ADDRESS", len(sources), abbr)
    xs_all: list[np.ndarray] = []
    ys_all: list[np.ndarray] = []
    source_all: list[np.ndarray] = []
    d1a_all: list[np.ndarray] = []
    d1b_all: list[np.ndarray] = []
    d2a_all: list[np.ndarray] = []
    d2b_all: list[np.ndarray] = []
    d3a_all: list[np.ndarray] = []
    d3b_all: list[np.ndarray] = []
    sufficient_all: list[np.ndarray] = []
    presence_all: dict[str, list[np.ndarray]] = {field: [] for field in FIELDS}
    invalid_geometry = 0
    parsed_records = 0
    source_records: list[dict[str, Any]] = []
    size_counts = Counter(Path(item["path"]).stat().st_size for item in sources)

    with rasterio.open(raster_path) as raster:
        bounds = raster.bounds
    for source_id, item in enumerate(sources):
        path = Path(item["path"])
        frame = pyogrio.read_dataframe(path, use_arrow=True)
        file_features = len(frame)
        parsed_records += file_features
        if file_features != int(item["raw_feature_count"]):
            raise RuntimeError(
                f"{abbr} {path.name}: parsed {file_features:,}, "
                f"metadata says {item['raw_feature_count']:,}"
            )
        first_geom = frame.geometry.iloc[0]
        first_type = first_geom.geom_type if first_geom is not None else "NULL"
        if first_type in {"Polygon", "MultiPolygon"}:
            frame.geometry = frame.geometry.centroid
        valid = frame.geometry.notna() & frame.geometry.geom_type.eq("Point")
        invalid_geometry += int((~valid).sum())
        frame = frame.loc[valid].reset_index(drop=True)
        file_valid = len(frame)
        if file_valid:
            lon = shapely.get_x(frame.geometry.array)
            lat = shapely.get_y(frame.geometry.array)
            property_columns = sorted(
                column for column in frame.columns if column != "geometry"
            )
            for field in FIELDS:
                if field not in frame.columns:
                    frame[field] = ""
            exact = frame[property_columns].fillna("").astype(str)
            exact["longitude"] = lon
            exact["latitude"] = lat
            d1a = pd.util.hash_pandas_object(
                exact, index=False, hash_key="step42d1key00001"
            ).to_numpy(np.uint64)
            d1b = pd.util.hash_pandas_object(
                exact, index=False, hash_key="step42d1key00002"
            ).to_numpy(np.uint64)
            original = frame[FIELDS].fillna("").astype(str)
            d2_frame = original.copy()
            d2_frame.insert(0, "source_id", source_id)
            d2_frame["longitude"] = lon
            d2_frame["latitude"] = lat
            d2a = pd.util.hash_pandas_object(
                d2_frame, index=False, hash_key="step42d2key00001"
            ).to_numpy(np.uint64)
            d2b = pd.util.hash_pandas_object(
                d2_frame, index=False, hash_key="step42d2key00002"
            ).to_numpy(np.uint64)
            normalized = original.apply(
                lambda series: series.str.normalize("NFKC")
                .str.upper()
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
            )
            d3_frame = normalized.copy()
            d3_frame["longitude"] = lon
            d3_frame["latitude"] = lat
            d3a = pd.util.hash_pandas_object(
                d3_frame, index=False, hash_key="step42d3key00001"
            ).to_numpy(np.uint64)
            d3b = pd.util.hash_pandas_object(
                d3_frame, index=False, hash_key="step42d3key00002"
            ).to_numpy(np.uint64)
            sufficient = (
                normalized["number"].ne("") & normalized["street"].ne("")
            ).to_numpy(bool)
            presence = {
                field: original[field].str.strip().ne("").to_numpy(bool)
                for field in FIELDS
            }
            xx, yy = TRANSFORMER.transform(
                np.asarray(lon, dtype=np.float64),
                np.asarray(lat, dtype=np.float64),
            )
            xx = np.asarray(xx, dtype=np.float64)
            yy = np.asarray(yy, dtype=np.float64)
            keep = np.ones(len(xx), dtype=bool)
            if abbr == "TX":
                keep = (
                    (xx >= bounds.left)
                    & (xx < bounds.right)
                    & (yy > bounds.bottom)
                    & (yy <= bounds.top)
                )
            xs_all.append(xx[keep])
            ys_all.append(yy[keep])
            source_all.append(np.full(int(keep.sum()), source_id, dtype=np.uint16))
            d1a_all.append(d1a[keep])
            d1b_all.append(d1b[keep])
            d2a_all.append(d2a[keep])
            d2b_all.append(d2b[keep])
            d3a_all.append(d3a[keep])
            d3b_all.append(d3b[keep])
            sufficient_all.append(np.asarray(sufficient, dtype=bool)[keep])
            for field in FIELDS:
                presence_all[field].append(
                    np.asarray(presence[field], dtype=bool)[keep]
                )
            retained = int(keep.sum())
        else:
            retained = 0
        raw_sha = item.get("locked_sha256", "")
        sha_status = "REUSED_LOCKED_TEXAS_MANIFEST" if raw_sha else "NOT_NEEDED"
        if not raw_sha and size_counts[path.stat().st_size] > 1:
            raw_sha = m.sha256_file(path)
            sha_status = "COMPUTED_SAME_SIZE_CANDIDATE"
        source_records.append(
            {
                "state": abbr,
                "source_id": source_id,
                "source_path": str(path),
                "basename": path.name,
                "feature_count": file_features,
                "valid_point_count_before_grid": file_valid,
                "retained_current_formal_count": retained,
                "sha256": raw_sha,
                "sha256_status": sha_status,
                "first_geometry_type": first_type or "NULL_OR_EMPTY",
            }
        )
        progress.tick(
            "IDENTITY_KEYS",
            f"file={path.name} features={file_features:,} retained={retained:,}",
        )
        del frame
    if parsed_records != total_expected:
        raise RuntimeError(f"{abbr}: parsed total does not equal source total")
    arrays = {
        "x": np.concatenate(xs_all),
        "y": np.concatenate(ys_all),
        "source": np.concatenate(source_all),
        "d1a": np.concatenate(d1a_all),
        "d1b": np.concatenate(d1b_all),
        "d2a": np.concatenate(d2a_all),
        "d2b": np.concatenate(d2b_all),
        "d3a": np.concatenate(d3a_all),
        "d3b": np.concatenate(d3b_all),
        "sufficient": np.concatenate(sufficient_all),
    }
    for field in FIELDS:
        arrays[f"present_{field}"] = np.concatenate(presence_all[field])
    stats = {
        "raw_source_record_count": total_expected,
        "parsed_record_count": parsed_records,
        "invalid_or_nonpoint_geometry_count": invalid_geometry,
        "current_formal_address_point_count": len(arrays["x"]),
        "source_file_count": len(sources),
        "identity_hash_method": "dual stable pandas SipHash-64 keys per D1/D2/D3",
    }
    return arrays, stats, source_records


def group_keys(
    a: np.ndarray, b: np.ndarray, source: np.ndarray | None = None
) -> dict[str, Any]:
    order = np.lexsort((b, a))
    aa, bb = a[order], b[order]
    starts = np.r_[True, (aa[1:] != aa[:-1]) | (bb[1:] != bb[:-1])]
    first = np.flatnonzero(starts)
    ends = np.r_[first[1:], len(order)]
    sizes = ends - first
    keep = np.zeros(len(a), dtype=bool)
    keep[order[first]] = True
    result: dict[str, Any] = {
        "order": order,
        "first": first,
        "ends": ends,
        "sizes": sizes,
        "keep": keep,
        "duplicate_group_count": int(np.sum(sizes > 1)),
        "records_in_duplicate_groups": int(np.sum(sizes[sizes > 1])),
        "duplicate_extra_records": int(np.sum(sizes - 1)),
    }
    if source is not None:
        ss = source[order]
        minimum = np.minimum.reduceat(ss, first)
        maximum = np.maximum.reduceat(ss, first)
        cross = (minimum != maximum) & (sizes > 1)
        result.update(
            {
                "cross_group_mask": cross,
                "cross_file_group_count": int(cross.sum()),
                "cross_file_records": int(np.sum(sizes[cross])),
                "cross_file_extra_records": int(np.sum(sizes[cross] - 1)),
            }
        )
    return result


def coordinate_groups(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    order = np.lexsort((y, x))
    xx, yy = x[order], y[order]
    starts = np.r_[True, (xx[1:] != xx[:-1]) | (yy[1:] != yy[:-1])]
    first = np.flatnonzero(starts)
    ends = np.r_[first[1:], len(order)]
    sizes = ends - first
    gid_sorted = np.cumsum(starts, dtype=np.int64) - 1
    gid = np.empty(len(x), dtype=np.int64)
    gid[order] = gid_sorted
    keep = np.zeros(len(x), dtype=bool)
    keep[order[first]] = True
    return {
        "order": order,
        "first": first,
        "ends": ends,
        "sizes": sizes,
        "gid": gid,
        "keep": keep,
        "ux": xx[first],
        "uy": yy[first],
    }


def identity_audit(
    abbr: str,
    arrays: dict[str, np.ndarray],
    stats: dict[str, Any],
    source_records: list[dict[str, Any]],
    raster_path: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, np.ndarray],
]:
    x, y, source = arrays["x"], arrays["y"], arrays["source"]
    n = len(x)
    d1 = group_keys(arrays["d1a"], arrays["d1b"], source)
    d2 = group_keys(arrays["d2a"], arrays["d2b"], source)
    d3 = group_keys(arrays["d3a"], arrays["d3b"], source)
    coord = coordinate_groups(x, y)
    sufficient = arrays["sufficient"]

    # D3 is restricted to normalized, sufficiently identified addresses and
    # different source files.
    order3, first3, sizes3 = d3["order"], d3["first"], d3["sizes"]
    ss3 = source[order3]
    suff3 = sufficient[order3]
    group_sufficient = np.logical_and.reduceat(suff3, first3)
    min_source = np.minimum.reduceat(ss3, first3)
    max_source = np.maximum.reduceat(ss3, first3)
    d3_cross = group_sufficient & (min_source != max_source) & (sizes3 > 1)

    # P3 removes D2 plus evidence-sufficient normalized cross-file identities.
    p3_keep = d2["keep"].copy()
    # D3 is explicitly cross-file.  Do not collapse merely normalized variants
    # that occur only within one source unless they already satisfy exact D2.
    for lo, hi, is_cross in zip(first3, d3["ends"], d3_cross):
        if is_cross:
            p3_keep[order3[lo + 1 : hi]] = False

    # D4: exact coordinate contains at least two distinct sufficient normalized
    # identities.  Count coordinate groups and all records at those locations.
    cg = coord["gid"]
    suff_idx = np.flatnonzero(sufficient)
    combo_order = np.lexsort(
        (
            arrays["d3b"][suff_idx],
            arrays["d3a"][suff_idx],
            cg[suff_idx],
        )
    )
    si = suff_idx[combo_order]
    combo_start = np.r_[
        True,
        (cg[si[1:]] != cg[si[:-1]])
        | (arrays["d3a"][si[1:]] != arrays["d3a"][si[:-1]])
        | (arrays["d3b"][si[1:]] != arrays["d3b"][si[:-1]]),
    ]
    unique_identity_indices = si[np.flatnonzero(combo_start)]
    distinct_per_coord = np.bincount(
        cg[unique_identity_indices], minlength=len(coord["sizes"])
    )
    d4_coord = distinct_per_coord >= 2
    d4_record_mask = d4_coord[cg]

    repeated_coord = coord["sizes"] > 1
    insufficient_per_coord = np.bincount(
        cg, weights=(~sufficient), minlength=len(coord["sizes"])
    ).astype(np.int64)
    d5_coord = repeated_coord & (insufficient_per_coord > 0)
    d5_record_mask = d5_coord[cg]

    with rasterio.open(raster_path) as raster:
        inv = ~raster.transform
        cols_f, rows_f = inv * (x, y)
        cols = np.asarray(cols_f).astype(np.int64)
        rows = np.asarray(rows_f).astype(np.int64)
        inbounds = (
            (cols >= 0)
            & (cols < raster.width)
            & (rows >= 0)
            & (rows < raster.height)
        )
        cell = rows[inbounds] * np.int64(raster.width) + cols[inbounds]
    idx_in = np.flatnonzero(inbounds)
    cell_order = np.argsort(cell, kind="stable")
    cell_sorted = cell[cell_order]
    cell_start = np.r_[True, cell_sorted[1:] != cell_sorted[:-1]]
    cell_first = np.flatnonzero(cell_start)
    cell_end = np.r_[cell_first[1:], len(cell)]
    cell_sizes = cell_end - cell_first
    cg_cell = cg[idx_in][cell_order]
    coord_change = np.r_[
        True,
        (cell_sorted[1:] != cell_sorted[:-1])
        | (cg_cell[1:] != cg_cell[:-1]),
    ]
    unique_coord_first = np.flatnonzero(coord_change)
    unique_coord_cell = cell_sorted[unique_coord_first]
    _, coords_per_cell = np.unique(unique_coord_cell, return_counts=True)
    d6_cells = coords_per_cell >= 2
    d6_records = int(np.sum(cell_sizes[d6_cells]))

    # D7 sensitivity: nearest *different exact coordinate* within 1 m.
    # cKDTree is run on unique coordinate locations, never used for deletion.
    tree = cKDTree(np.column_stack([coord["ux"], coord["uy"]]))
    distance, _ = tree.query(
        np.column_stack([coord["ux"], coord["uy"]]),
        k=2,
        distance_upper_bound=NEAR_THRESHOLD_M,
        workers=-1,
    )
    near_coord = np.isfinite(distance[:, 1]) & (distance[:, 1] > 0)
    d7_records = int(np.sum(coord["sizes"][near_coord]))
    del tree, distance

    identity_rows = [
        {
            "state": abbr,
            "duplicate_type": "D1",
            "definition": "All available raw feature fields and geometry exactly equal (dual stable column hashes)",
            "group_count": d1["duplicate_group_count"],
            "records_in_groups": d1["records_in_duplicate_groups"],
            "duplicate_extra_records": d1["duplicate_extra_records"],
            "delete_in_policy": "P2_AND_P3",
            "status": "COMPLETED",
            "notes": "Dual stable pandas 64-bit hashes; source file is not treated as an address field",
        },
        {
            "state": abbr,
            "duplicate_type": "D2",
            "definition": "Same source file, exact address fields, and exact coordinate",
            "group_count": d2["duplicate_group_count"],
            "records_in_groups": d2["records_in_duplicate_groups"],
            "duplicate_extra_records": d2["duplicate_extra_records"],
            "delete_in_policy": "P3",
            "status": "COMPLETED",
            "notes": "Address fields: number, street, unit, city, district, region, postcode",
        },
        {
            "state": abbr,
            "duplicate_type": "D3",
            "definition": "Sufficient normalized address and coordinate equal across different source files",
            "group_count": int(d3_cross.sum()),
            "records_in_groups": int(np.sum(sizes3[d3_cross])),
            "duplicate_extra_records": int(np.sum(sizes3[d3_cross] - 1)),
            "delete_in_policy": "P3_SENSITIVITY_PENDING_HUMAN_SAMPLE_APPROVAL",
            "status": "COMPLETED",
            "notes": "Sufficient identity requires nonempty number and street",
        },
        {
            "state": abbr,
            "duplicate_type": "D4",
            "definition": "Same exact coordinate but two or more distinct sufficient normalized addresses/units",
            "group_count": int(d4_coord.sum()),
            "records_in_groups": int(d4_record_mask.sum()),
            "duplicate_extra_records": "NOT_DUPLICATES_DO_NOT_DELETE",
            "delete_in_policy": "NEVER_AUTOMATICALLY",
            "status": "COMPLETED",
            "notes": "Direct evidence that coordinate-only deduplication can delete distinct addresses",
        },
        {
            "state": abbr,
            "duplicate_type": "D5",
            "definition": "Repeated exact coordinate with at least one record lacking number or street",
            "group_count": int(d5_coord.sum()),
            "records_in_groups": int(d5_record_mask.sum()),
            "duplicate_extra_records": "AMBIGUOUS_DO_NOT_DELETE",
            "delete_in_policy": "NEVER_AUTOMATICALLY",
            "status": "COMPLETED",
            "notes": f"insufficient_records={int((~sufficient).sum())}",
        },
        {
            "state": abbr,
            "duplicate_type": "D6",
            "definition": "Same formal 30 m cell but at least two different exact coordinate locations",
            "group_count": int(d6_cells.sum()),
            "records_in_groups": d6_records,
            "duplicate_extra_records": "NOT_DUPLICATES_DO_NOT_DELETE",
            "delete_in_policy": "NEVER_AUTOMATICALLY",
            "status": "COMPLETED",
            "notes": "Computed only for records inside the formal raster grid",
        },
        {
            "state": abbr,
            "duplicate_type": "D7",
            "definition": f"Different exact coordinate with nearest different coordinate <= {NEAR_THRESHOLD_M:g} m",
            "group_count": int(near_coord.sum()),
            "records_in_groups": d7_records,
            "duplicate_extra_records": "SENSITIVITY_NOT_DUPLICATES",
            "delete_in_policy": "NEVER_AUTOMATICALLY",
            "status": "COMPLETED",
            "notes": "Count is coordinate locations with a near neighbor, not unique pairs",
        },
    ]
    field_rows: list[dict[str, Any]] = []
    for field in FIELDS:
        present = int(arrays[f"present_{field}"].sum())
        field_rows.append(
            {
                "state": abbr,
                "field": field,
                "audited_address_records": n,
                "nonempty_count": present,
                "missing_count": n - present,
                "availability_pct": 100 * present / n if n else 0,
                "used_in_D2": "YES",
                "used_in_D3_normalization": "YES",
                "status": "COMPLETED",
            }
        )
    field_rows.append(
        {
            "state": abbr,
            "field": "CORE_IDENTITY_NUMBER_AND_STREET",
            "audited_address_records": n,
            "nonempty_count": int(sufficient.sum()),
            "missing_count": int((~sufficient).sum()),
            "availability_pct": 100 * sufficient.mean() if n else 0,
            "used_in_D2": "NOT_APPLICABLE",
            "used_in_D3_normalization": "MINIMUM_EVIDENCE_GATE",
            "status": "COMPLETED",
        }
    )
    cross_rows = [
        {
            "state": abbr,
            "metric": "D1_EXACT_RECORD_ACROSS_FILES",
            "group_count": d1["cross_file_group_count"],
            "records": d1["cross_file_records"],
            "extra_records": d1["cross_file_extra_records"],
            "evidence_level": "EXACT_ALL_FIELDS_AND_GEOMETRY",
            "status": "COMPLETED",
        },
        {
            "state": abbr,
            "metric": "D3_NORMALIZED_IDENTITY_ACROSS_FILES",
            "group_count": int(d3_cross.sum()),
            "records": int(np.sum(sizes3[d3_cross])),
            "extra_records": int(np.sum(sizes3[d3_cross] - 1)),
            "evidence_level": "NORMALIZED_REQUIRES_HUMAN_SAMPLE_APPROVAL",
            "status": "COMPLETED",
        },
    ]
    file_frame = pd.DataFrame(source_records)
    hashed_files = file_frame[file_frame["sha256"].astype(str).ne("")]
    repeated_sha = hashed_files.groupby("sha256").filter(lambda g: len(g) > 1)
    cross_rows.append(
        {
            "state": abbr,
            "metric": "BYTE_IDENTICAL_SOURCE_FILES",
            "group_count": int(
                repeated_sha["sha256"].nunique() if len(repeated_sha) else 0
            ),
            "records": int(
                repeated_sha["retained_current_formal_count"].sum()
                if len(repeated_sha)
                else 0
            ),
            "extra_records": "FILE_LEVEL_DIAGNOSTIC",
            "evidence_level": "SHA256",
            "status": "COMPLETED",
        }
    )

    examples: list[dict[str, Any]] = []
    source_names = {row["source_id"]: row["basename"] for row in source_records}
    for coord_id in np.flatnonzero(d4_coord)[:10]:
        indices = np.flatnonzero(cg == coord_id)
        identity_tokens = sorted(
            {
                f"{int(arrays['d3a'][i]):016x}{int(arrays['d3b'][i]):016x}"[:12]
                for i in indices
                if sufficient[i]
            }
        )
        source_tokens = sorted({source_names[int(source[i])] for i in indices})
        coordinate_token = hashlib.sha256(
            f"{coord['ux'][coord_id]:.6f}|{coord['uy'][coord_id]:.6f}".encode()
        ).hexdigest()[:12]
        examples.append(
            {
                "state": abbr,
                "coordinate_token": coordinate_token,
                "record_count": len(indices),
                "distinct_normalized_address_count": int(
                    distinct_per_coord[coord_id]
                ),
                "address_identity_tokens": ";".join(identity_tokens[:8]),
                "source_basenames": ";".join(source_tokens[:8]),
                "unit_nonempty_record_count": int(
                    arrays["present_unit"][indices].sum()
                ),
                "postcode_nonempty_record_count": int(
                    arrays["present_postcode"][indices].sum()
                ),
                "privacy_rule": "No house number, street, unit value, full postcode, or exact coordinate retained",
                "interpretation": "LEGAL_COLOCATION_EVIDENCE_DO_NOT_COORDINATE_DEDUP",
            }
        )

    scenario_flags = {
        "P1_ADDRESS_RAW": np.ones(n, dtype=bool),
        "P2_ADDRESS_EXACT_RECORD_DEDUP": d1["keep"],
        "P3_ADDRESS_NORMALIZED_IDENTITY_DEDUP": p3_keep,
        "P4_ADDRESS_COORDINATE_DEDUP_SENSITIVITY": coord["keep"],
        "inbounds": inbounds,
        "cols": cols,
        "rows": rows,
    }
    return identity_rows, field_rows, cross_rows, examples, scenario_flags


def aggregate_cells(
    raster_path: Path, flags: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, int]]:
    inbounds = flags["inbounds"]
    rows = flags["rows"][inbounds]
    cols = flags["cols"][inbounds]
    with rasterio.open(raster_path) as raster:
        cell = rows * np.int64(raster.width) + cols
        width = raster.width
    order = np.argsort(cell, kind="stable")
    cell_s = cell[order]
    starts = np.r_[True, cell_s[1:] != cell_s[:-1]]
    first = np.flatnonzero(starts)
    gid = np.cumsum(starts, dtype=np.int64) - 1
    weights: dict[str, np.ndarray] = {}
    inputs: dict[str, int] = {}
    for scenario in [
        "P1_ADDRESS_RAW",
        "P2_ADDRESS_EXACT_RECORD_DEDUP",
        "P3_ADDRESS_NORMALIZED_IDENTITY_DEDUP",
        "P4_ADDRESS_COORDINATE_DEDUP_SENSITIVITY",
    ]:
        values = flags[scenario][inbounds][order].astype(np.int32)
        weights[scenario] = np.bincount(
            gid, weights=values, minlength=len(first)
        ).astype(np.int32)
        inputs[scenario] = int(values.sum())
    unique_cell = cell_s[first]
    return unique_cell // width, unique_cell % width, weights, inputs


def classify_policy(
    m,
    abbr: str,
    raster_path: Path,
    cell_rows: np.ndarray,
    cell_cols: np.ndarray,
    weights: dict[str, np.ndarray],
    inputs: dict[str, int],
    p0_input_records: int,
    p1_expected_wui: int,
) -> list[dict[str, Any]]:
    scenarios = ["P0_LEGACY", *weights.keys()]
    totals = {
        scenario: {
            "wui": 0,
            "loss_p0": 0,
            "gain_p0": 0,
            "loss_p1": 0,
            "gain_p1": 0,
        }
        for scenario in scenarios
    }
    hashers = {
        scenario: hashlib.sha256(
            f"STEP42_TILE_STREAM_V1|{abbr}|500|{scenario}".encode()
        )
        for scenario in scenarios
    }
    fips, _, _ = m.BY_ABBR[abbr]
    with rasterio.open(raster_path) as raster:
        state_geom = m.load_state_geometry(
            fips, CRS.from_user_input(raster.crs)
        )
        nx = math.ceil(raster.width / TILE)
        ny = math.ceil(raster.height / TILE)
        tile_id = (cell_rows // TILE) * nx + (cell_cols // TILE)
        order = np.argsort(tile_id, kind="stable")
        tids = tile_id[order]
        rr_s, cc_s = cell_rows[order], cell_cols[order]
        weights_s = {key: value[order] for key, value in weights.items()}
        unique_tid, begin = np.unique(tids, return_index=True)
        finish = np.r_[begin[1:], len(tids)]
        ranges = {
            int(tid): (int(lo), int(hi))
            for tid, lo, hi in zip(unique_tid, begin, finish)
        }
        progress = Progress("CLASSIFY_POLICY_500M", nx * ny, abbr)
        full = Window(0, 0, raster.width, raster.height)
        analysis_pixels = 0
        for ty in range(ny):
            for tx in range(nx):
                core = Window(
                    tx * TILE,
                    ty * TILE,
                    min(TILE, raster.width - tx * TILE),
                    min(TILE, raster.height - ty * TILE),
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
                chunks: list[np.ndarray] = []
                for nty in range(max(0, ty - 1), min(ny, ty + 2)):
                    for ntx in range(max(0, tx - 1), min(nx, tx + 2)):
                        interval = ranges.get(nty * nx + ntx)
                        if interval:
                            chunks.append(
                                np.arange(interval[0], interval[1], dtype=np.int64)
                            )
                if chunks:
                    index = np.concatenate(chunks)
                    keep = (
                        (rr_s[index] >= pad.row_off)
                        & (rr_s[index] < pad.row_off + pad.height)
                        & (cc_s[index] >= pad.col_off)
                        & (cc_s[index] < pad.col_off + pad.width)
                    )
                    index = index[keep]
                    local_r = rr_s[index] - int(pad.row_off)
                    local_c = cc_s[index] - int(pad.col_off)
                else:
                    index = np.empty(0, dtype=np.int64)
                    local_r = local_c = index
                current = raster.read(1, window=core)
                domain = geometry_mask(
                    [state_geom],
                    out_shape=current.shape,
                    transform=raster.window_transform(core),
                    invert=True,
                    all_touched=False,
                )
                analysis_pixels += int(domain.sum())
                formal = (current == 1) | (current == 2)
                p0 = current
                scenario_arrays: dict[str, np.ndarray] = {"P0_LEGACY": p0}
                r0 = int(core.row_off - pad.row_off)
                c0 = int(core.col_off - pad.col_off)
                h0, w0 = int(core.height), int(core.width)
                representative: dict[str, str] = {}
                computed: dict[str, np.ndarray] = {}
                prior: list[str] = []
                for scenario, values in weights_s.items():
                    match = next(
                        (
                            old
                            for old in prior
                            if np.array_equal(weights_s[old], values)
                        ),
                        None,
                    )
                    if match is None:
                        prior.append(scenario)
                        count = np.zeros(
                            (int(pad.height), int(pad.width)), dtype=np.float32
                        )
                        if len(index):
                            np.add.at(count, (local_r, local_c), values[index])
                        summed = fftconvolve(count, KERNEL, mode="same")
                        dense = (
                            np.rint(
                                summed[r0 : r0 + h0, c0 : c0 + w0]
                            ).astype(np.int32)
                            >= MIN_DENSE_COUNT
                        )
                        computed[scenario] = dense
                        match = scenario
                    representative[scenario] = match
                for scenario in weights:
                    dense = computed[representative[scenario]]
                    scenario_arrays[scenario] = np.where(
                        dense, current, 0
                    ).astype(np.uint8)
                p1_wui = scenario_arrays["P1_ADDRESS_RAW"] > 0
                p0_wui = formal
                for scenario, classified in scenario_arrays.items():
                    swui = classified > 0
                    totals[scenario]["wui"] += int(np.sum(swui & domain))
                    totals[scenario]["loss_p0"] += int(
                        np.sum(p0_wui & ~swui & domain)
                    )
                    totals[scenario]["gain_p0"] += int(
                        np.sum(~p0_wui & swui & domain)
                    )
                    totals[scenario]["loss_p1"] += int(
                        np.sum(p1_wui & ~swui & domain)
                    )
                    totals[scenario]["gain_p1"] += int(
                        np.sum(~p1_wui & swui & domain)
                    )
                    hashers[scenario].update(
                        struct.pack(">IIII", ty, tx, h0, w0)
                    )
                    hashers[scenario].update(classified.tobytes(order="C"))
                progress.tick(
                    "P0-P4",
                    f"tile={tx+1}/{nx},{ty+1}/{ny}",
                )
    # Step41 S4 used source labels reconstructed from current raw-directory glob
    # order after the geometry-only legacy GeoPackage had discarded provenance.
    # The original producer-directory ordering was not locked.  Step42 reads the
    # address files directly and therefore treats the Step41 S4 count as a
    # diagnostic comparison, not an identity gate.
    p1_difference_from_step41_s4 = (
        totals["P1_ADDRESS_RAW"]["wui"] - p1_expected_wui
    )
    p0_wui = totals["P0_LEGACY"]["wui"]
    p1_wui = totals["P1_ADDRESS_RAW"]["wui"]
    rows_out: list[dict[str, Any]] = []
    input_map = {"P0_LEGACY": p0_input_records, **inputs}
    p1_input = inputs["P1_ADDRESS_RAW"]
    for scenario in scenarios:
        item = totals[scenario]
        retained = input_map[scenario]
        deleted = 0 if scenario in {"P0_LEGACY", "P1_ADDRESS_RAW"} else p1_input - retained
        intersection_p1 = min(item["wui"], p1_wui)
        union_p1 = max(item["wui"], p1_wui)
        rows_out.append(
            {
                "state": abbr,
                "buffer_m": 500,
                "scenario": scenario,
                "scenario_role": (
                    "LEGACY_REPRODUCTION_ONLY"
                    if scenario == "P0_LEGACY"
                    else "ADDRESS_ONLY_POLICY_CANDIDATE"
                    if scenario
                    in {
                        "P1_ADDRESS_RAW",
                        "P2_ADDRESS_EXACT_RECORD_DEDUP",
                        "P3_ADDRESS_NORMALIZED_IDENTITY_DEDUP",
                    }
                    else "COORDINATE_DEDUP_SENSITIVITY_UPPER_BOUND_ONLY"
                ),
                "input_records": input_map[scenario],
                "deleted_records_from_P1": deleted,
                "retained_records": retained,
                "wui_pixels": item["wui"],
                "wui_area_km2": item["wui"] * 0.0009,
                "wui_pixel_change_vs_P0": item["wui"] - p0_wui,
                "wui_area_change_vs_P0_km2": (item["wui"] - p0_wui) * 0.0009,
                "wui_pixel_change_vs_P1": item["wui"] - p1_wui,
                "wui_area_change_vs_P1_km2": (item["wui"] - p1_wui) * 0.0009,
                "nonwui_to_wui_vs_P0_pixels": item["gain_p0"],
                "wui_to_nonwui_vs_P0_pixels": item["loss_p0"],
                "classification_change_vs_P0_area_km2": (
                    item["gain_p0"] + item["loss_p0"]
                )
                * 0.0009,
                "jaccard_with_P1": (
                    intersection_p1 / union_p1 if union_p1 else 1.0
                ),
                "analysis_domain_pixels": analysis_pixels,
                "classification_sha256": hashers[scenario].hexdigest(),
                "classification_hash_definition": "SHA256 of deterministic row-major tile stream with tile headers; full formal raster extent",
                "production_density_rule": "disk(17 pixels); count/(pi*500^2/1e6)>6.17; integer count>=5",
                "prior_step41_S4_wui_pixels": p1_expected_wui,
                "P1_minus_prior_step41_S4_wui_pixels": (
                    p1_difference_from_step41_s4
                    if scenario == "P1_ADDRESS_RAW"
                    else ""
                ),
                "step41_S4_use_status": "DIAGNOSTIC_ONLY_SOURCE_ORDER_RECONSTRUCTION_NOT_POLICY_GATE",
                "status": "COMPLETED",
            }
        )
    return rows_out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--states", default="CA,CO,FL,PA,TX", help="Comma-separated states"
    )
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    m = load_step41()
    step41_dup = pd.read_csv(
        STEP41 / "wui_p_duplicate_summary_by_state.csv"
    ).set_index("state")
    step41_scen = pd.read_csv(
        STEP41 / "wui_p_dedup_scenario_classification_impact.csv"
    )
    selected_states = [x.strip() for x in args.states.split(",") if x.strip()]
    for abbr in selected_states:
        checkpoint = checkpoint_dir / f"{abbr}_step42.json"
        if checkpoint.is_file():
            print(
                f"[RESUME] step=3-4/5 state={abbr} scenario=ALL "
                "completed=1/1 percent=100 elapsed=0 ETA=0 checkpoint=EXISTS",
                flush=True,
            )
            continue
        raster_path = raster_for_state(abbr)
        arrays, parse_stats, source_records = parse_state(m, abbr, raster_path)
        (
            identity_rows,
            field_rows,
            cross_rows,
            examples,
            scenario_flags,
        ) = identity_audit(
            abbr, arrays, parse_stats, source_records, raster_path
        )
        cell_rows, cell_cols, weights, inputs = aggregate_cells(
            raster_path, scenario_flags
        )
        dup = step41_dup.loc[abbr]
        p0_input = int(dup["valid_input_points"] - dup["point_records_outside_raster"])
        p1_expected = int(
            step41_scen[
                step41_scen["state"].eq(abbr)
                & step41_scen["scenario"].eq("S4")
            ].iloc[0]["scenario_wui_p_pixels"]
        )
        impact_rows = classify_policy(
            m,
            abbr,
            raster_path,
            cell_rows,
            cell_cols,
            weights,
            inputs,
            p0_input,
            p1_expected,
        )
        payload = {
            "state": abbr,
            "completed_utc": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "parse_stats": parse_stats,
            "identity_rows": identity_rows,
            "field_rows": field_rows,
            "cross_rows": cross_rows,
            "examples": examples,
            "impact_rows": impact_rows,
            "source_records": source_records,
            "near_threshold_m": NEAR_THRESHOLD_M,
            "research_drive_write_attempted": False,
        }
        checkpoint.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"[STATE_COMPLETE] step=4/5 state={abbr} scenario=P0-P4 "
            "completed=1/1 percent=100 elapsed=NA ETA=0",
            flush=True,
        )
        del arrays, scenario_flags, cell_rows, cell_cols, weights

    identity: list[dict] = []
    fields: list[dict] = []
    cross: list[dict] = []
    examples_all: list[dict] = []
    impact: list[dict] = []
    sources_all: list[dict] = []
    for abbr in selected_states:
        path = checkpoint_dir / f"{abbr}_step42.json"
        if not path.is_file():
            raise RuntimeError(f"Missing checkpoint for {abbr}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        identity.extend(payload["identity_rows"])
        fields.extend(payload["field_rows"])
        cross.extend(payload["cross_rows"])
        examples_all.extend(payload["examples"])
        impact.extend(payload["impact_rows"])
        sources_all.extend(payload["source_records"])
    write_csv(out / "wui_p_duplicate_identity_summary.csv", identity)
    write_csv(out / "wui_p_duplicate_field_availability.csv", fields)
    write_csv(out / "wui_p_cross_file_duplicate_summary.csv", cross)
    write_csv(out / "wui_p_colocated_distinct_address_examples.csv", examples_all)
    write_csv(out / "wui_p_address_policy_500m_impact.csv", impact)
    write_csv(out / "five_state_address_source_sha256.csv", sources_all)
    print(
        f"STEP42B COMPLETE states={len(selected_states)} "
        f"identity_rows={len(identity)} impact_rows={len(impact)} output={out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
