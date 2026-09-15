#!/usr/bin/env python3
# Copyright (C) 2026 WUI Mapping Project Authors
# SPDX-License-Identifier: GPL-3.0-only
# Repository version: filesystem paths are loaded through repo_config.py.
# Copy config/paths.example.json to config/paths.json before a new run.
# Raw input data and large intermediate files are stored outside GitHub.

"""Step43 Phases B-D: deterministic P2 D1 audit and class-raster rebuild.

The implementation deliberately reuses Step42's D1 key construction and 500 m
tile classifier.  WUI-P classification is a deletion-only subset of each
same-buffer legacy P0 classification: vegetation/distance class is unchanged,
while the exact production disk-density gate is recomputed from address-only
D1-deduplicated points.  Outside the Census state valid domain the new raster
is 255 NoData; inside, 0 remains valid Non-WUI.
"""

from __future__ import annotations

from repo_config import portable_path

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import resource
import struct
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyogrio
import rasterio
import shapely
from pyproj import CRS, Transformer
from rasterio.features import geometry_mask
from rasterio.windows import Window
from scipy.signal import fftconvolve
from skimage.morphology import disk


PROJECT = Path(portable_path("project"))
STEP42 = PROJECT / "step42_wuip_address_policy_audit_20260727T153136Z"
STATE_GPKG = Path(
    portable_path("data", "MBF+NLCD_2022US/Inputs/Processed_GPKG/tl_2022_us_state.gpkg")
)
TILE = 2048
PIXEL_M = 30.0
DENSITY_THRESHOLD = 6.17
NODATA = 255
FIVE = ["CA", "CO", "FL", "PA", "TX"]
FIELDS = ["number", "street", "unit", "city", "district", "region", "postcode"]
TRANSFORMER = Transformer.from_crs(4326, 5070, always_xy=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def atomic_json(path: Path, payload: Any) -> None:
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    temp = path.with_suffix(path.suffix + ".partial")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def state_geometry(fips: str) -> Any:
    frame = pyogrio.read_dataframe(
        STATE_GPKG,
        where=f"STATEFP = '{fips}'",
        columns=["STATEFP"],
    )
    if frame.empty:
        raise RuntimeError(f"State geometry missing for {fips}")
    geometry = shapely.union_all(frame.geometry.array)
    source_crs = CRS.from_user_input(frame.crs)
    if not source_crs.equals(CRS.from_epsg(5070)):
        geometry = shapely.transform(
            geometry,
            Transformer.from_crs(source_crs, 5070, always_xy=True).transform,
            interleaved=False,
        )
    return geometry


def deterministic_source_sort(path: str) -> str:
    return os.path.normpath(os.path.abspath(path)).casefold()


def load_context(output: Path) -> dict[str, Any]:
    phase_a = output / "checkpoints/phaseA_complete.json"
    if not phase_a.is_file():
        raise RuntimeError("Phase A checkpoint is missing")
    status = json.loads(phase_a.read_text())
    if status.get("status") != "PHASE_A_PREPARED_FOR_CANARY":
        raise RuntimeError("Phase A is not ready")
    inputs = pd.read_csv(
        output / "manifests/p2_input_manifest_all49.csv",
        keep_default_na=False,
        dtype={"STATEFP": str},
    )
    included = inputs[
        inputs["included_yes_no"].eq("YES")
        & inputs["source_type"].eq("address")
    ].copy()
    if included["state"].nunique() != 49:
        raise RuntimeError("Included-address manifest does not cover 49 units")
    if (
        inputs[
            inputs["included_yes_no"].eq("YES")
            & inputs["source_type"].ne("address")
        ].shape[0]
        != 0
    ):
        raise RuntimeError("Non-address source is included")
    legacy = pd.read_csv(
        output / "manifests/p2_legacy_raster_manifest.csv",
        keep_default_na=False,
    )
    policy = pd.read_csv(
        STEP42 / "wui_p_source_policy_all49.csv",
        dtype={"STATEFP": str},
    )
    return {"inputs": inputs, "included": included, "legacy": legacy, "policy": policy}


def parse_and_dedup_state(
    output: Path,
    state: str,
    fips: str,
    input_rows: pd.DataFrame,
    reference_raster: Path,
) -> dict[str, Any]:
    checkpoint = output / "checkpoints" / f"dedup_{state}.json"
    sparse_path = output / "intermediate" / f"{state}_p2_sparse_cells.npz"
    if checkpoint.is_file():
        payload = json.loads(checkpoint.read_text())
        if (
            payload.get("status") == "PASS"
            and sparse_path.is_file()
            and sha256_file(sparse_path) == payload.get("sparse_sha256")
        ):
            print(
                f"[DEDUP_RESUME] phase=state state={state} buffer=- "
                f"completed=1/1 percent=100 elapsed=0 ETA=0 "
                f"memory_gib={rss_gib():.2f} output={sparse_path}",
                flush=True,
            )
            return payload
        raise RuntimeError(f"Invalid D1 checkpoint for {state}")

    rows = sorted(
        input_rows.to_dict("records"),
        key=lambda row: deterministic_source_sort(row["source_path"]),
    )
    total_raw = sum(int(row["record_count"]) for row in rows)
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    d1a_parts: list[np.ndarray] = []
    d1b_parts: list[np.ndarray] = []
    source_parts: list[np.ndarray] = []
    source_audit: list[dict[str, Any]] = []
    processed = 0
    started = time.monotonic()

    with rasterio.open(reference_raster) as reference:
        bounds = reference.bounds
        transform = reference.transform
        width, height = reference.width, reference.height

    for source_id, row in enumerate(rows):
        path = Path(row["source_path"])
        stat = path.stat()
        if (
            stat.st_size != int(row["file_size"])
            or datetime.fromtimestamp(stat.st_mtime, timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
            != row["mtime_utc"]
        ):
            raise RuntimeError(f"Frozen source size/mtime changed: {path}")
        frame = pyogrio.read_dataframe(path, use_arrow=True)
        raw_count = len(frame)
        if raw_count != int(row["record_count"]):
            raise RuntimeError(
                f"{state} {path.name}: raw count {raw_count:,} != "
                f"manifest {int(row['record_count']):,}"
            )
        original_position = np.arange(raw_count, dtype=np.int64)
        if raw_count:
            first_geometry = frame.geometry.iloc[0]
            first_type = (
                first_geometry.geom_type if first_geometry is not None else "NULL"
            )
        else:
            first_type = "EMPTY"
        if first_type in {"Polygon", "MultiPolygon"}:
            frame.geometry = frame.geometry.centroid
        valid = frame.geometry.notna() & frame.geometry.geom_type.eq("Point")
        invalid_count = int((~valid).sum())
        frame = frame.loc[valid].copy()
        valid_positions = original_position[valid.to_numpy()]
        if len(frame):
            lon = shapely.get_x(frame.geometry.array)
            lat = shapely.get_y(frame.geometry.array)
            property_columns = sorted(
                column for column in frame.columns if column != "geometry"
            )
            exact = frame[property_columns].fillna("").astype(str)
            exact["longitude"] = lon
            exact["latitude"] = lat
            # Exact reuse of Step42 D1 dual pandas row-hash implementation.
            d1a = pd.util.hash_pandas_object(
                exact, index=False, hash_key="step42d1key00001"
            ).to_numpy(np.uint64)
            d1b = pd.util.hash_pandas_object(
                exact, index=False, hash_key="step42d1key00002"
            ).to_numpy(np.uint64)
            xx, yy = TRANSFORMER.transform(
                np.asarray(lon, dtype=np.float64),
                np.asarray(lat, dtype=np.float64),
            )
            xx = np.asarray(xx, dtype=np.float64)
            yy = np.asarray(yy, dtype=np.float64)
            x_parts.append(xx)
            y_parts.append(yy)
            d1a_parts.append(d1a)
            d1b_parts.append(d1b)
            source_parts.append(
                np.full(len(frame), source_id, dtype=np.uint16)
            )
            outside_grid_before_dedup = int(
                np.sum(
                    (xx < bounds.left)
                    | (xx >= bounds.right)
                    | (yy <= bounds.bottom)
                    | (yy > bounds.top)
                )
            )
        else:
            valid_positions = np.empty(0, dtype=np.int64)
            outside_grid_before_dedup = 0
        source_audit.append(
            {
                "state": state,
                "source_id": source_id,
                "source_path": str(path),
                "source_sha256": row["sha256"],
                "raw_address_records": raw_count,
                "valid_address_records": len(frame),
                "invalid_geometry_records": invalid_count,
                "outside_grid_records_before_dedup": outside_grid_before_dedup,
                "d1_records_removed": 0,
                "retained_records": 0,
                "first_geometry_type": first_type,
                "property_fields_in_D1": ";".join(property_columns)
                if len(frame)
                else "",
                "original_position_min": int(valid_positions.min())
                if len(valid_positions)
                else "",
                "original_position_max": int(valid_positions.max())
                if len(valid_positions)
                else "",
                "status": "PENDING_GROUP_ASSIGNMENT",
            }
        )
        processed += raw_count
        elapsed = max(time.monotonic() - started, 1e-9)
        rate = processed / elapsed
        eta = (total_raw - processed) / rate if rate else 0
        print(
            f"[D1_LOAD] phase=state state={state} buffer=- "
            f"file={path.name} processed={processed:,}/{total_raw:,} "
            f"percent={100*processed/max(total_raw,1):.2f}% "
            f"elapsed={elapsed:.1f}s ETA={eta:.1f}s "
            f"memory_gib={rss_gib():.2f} output={sparse_path}",
            flush=True,
        )
        del frame

    x = np.concatenate(x_parts)
    y = np.concatenate(y_parts)
    d1a = np.concatenate(d1a_parts)
    d1b = np.concatenate(d1b_parts)
    source = np.concatenate(source_parts)
    valid_total = len(x)
    # Global input order is already normalized source path, then original row
    # order.  Include that order as the least-significant lexsort key so the
    # retained member is deterministically the first input record.
    input_order = np.arange(valid_total, dtype=np.int64)
    order = np.lexsort((input_order, d1b, d1a))
    aa, bb = d1a[order], d1b[order]
    starts = np.r_[True, (aa[1:] != aa[:-1]) | (bb[1:] != bb[:-1])]
    first = np.flatnonzero(starts)
    ends = np.r_[first[1:], len(order)]
    sizes = ends - first
    keep = np.zeros(valid_total, dtype=bool)
    keep[order[first]] = True
    duplicate_groups = int(np.sum(sizes > 1))
    records_removed = int(np.sum(sizes - 1))
    source_sorted = source[order]
    min_source = np.minimum.reduceat(source_sorted, first)
    max_source = np.maximum.reduceat(source_sorted, first)
    cross = (min_source != max_source) & (sizes > 1)
    cross_groups = int(cross.sum())
    cross_removed = int(np.sum(sizes[cross] - 1))
    removed_by_source = np.bincount(
        source[~keep], minlength=len(rows)
    ).astype(np.int64)
    retained_by_source = np.bincount(
        source[keep], minlength=len(rows)
    ).astype(np.int64)
    for index, audit in enumerate(source_audit):
        audit["d1_records_removed"] = int(removed_by_source[index])
        audit["retained_records"] = int(retained_by_source[index])
        audit["status"] = "PASS"

    retained_x = x[keep]
    retained_y = y[keep]
    inv = ~transform
    cols_f, rows_f = inv * (retained_x, retained_y)
    cols = np.asarray(cols_f).astype(np.int64)
    raster_rows = np.asarray(rows_f).astype(np.int64)
    inbounds = (
        (cols >= 0)
        & (cols < width)
        & (raster_rows >= 0)
        & (raster_rows < height)
    )
    outside_grid = int((~inbounds).sum())
    cols = cols[inbounds]
    raster_rows = raster_rows[inbounds]
    cells = raster_rows * np.int64(width) + cols
    unique_cells, counts = np.unique(cells, return_counts=True)
    sparse_rows = (unique_cells // width).astype(np.int32)
    sparse_cols = (unique_cells % width).astype(np.int32)
    if counts.max(initial=0) > np.iinfo(np.uint32).max:
        raise RuntimeError("Point count exceeds uint32")
    sparse_counts = counts.astype(np.uint32)
    temp_sparse = sparse_path.with_suffix(".npz.partial")
    with temp_sparse.open("wb") as handle:
        np.savez_compressed(
            handle,
            rows=sparse_rows,
            cols=sparse_cols,
            counts=sparse_counts,
            width=np.int64(width),
            height=np.int64(height),
        )
    os.replace(temp_sparse, sparse_path)
    sparse_sha = sha256_file(sparse_path)
    by_file_path = output / "dedup_audit" / f"{state}_p2_d1_by_file.csv"
    write_csv(by_file_path, source_audit)

    step42_identity = pd.read_csv(
        STEP42 / "wui_p_duplicate_identity_summary.csv"
    )
    five_rows = step42_identity[step42_identity["state"].eq(state)]
    refs: dict[str, Any] = {}
    for duplicate_type in ["D2", "D3", "D4", "D5"]:
        match = five_rows[five_rows["duplicate_type"].eq(duplicate_type)]
        if len(match) == 1:
            refs[duplicate_type] = {
                "status": "NOT_RECOMPUTED_REFERENCE_STEP42",
                "records_in_groups": int(match.iloc[0]["records_in_groups"]),
                "group_count": int(match.iloc[0]["group_count"]),
            }
        else:
            refs[duplicate_type] = {
                "status": "NOT_RECOMPUTED",
                "records_in_groups": "NOT_RECOMPUTED",
                "group_count": "NOT_RECOMPUTED",
            }
    payload = {
        "status": "PASS",
        "state": state,
        "completed_utc": utc_now(),
        "raw_address_records": total_raw,
        "valid_address_records": valid_total,
        "invalid_geometry_records": total_raw - valid_total,
        "d1_duplicate_groups": duplicate_groups,
        "d1_records_in_groups": int(np.sum(sizes[sizes > 1])),
        "d1_records_removed": records_removed,
        "d1_cross_file_groups": cross_groups,
        "d1_cross_file_records_removed": cross_removed,
        "retained_records": valid_total - records_removed,
        "retained_records_inside_grid": int(inbounds.sum()),
        "outside_domain_records": outside_grid,
        "outside_domain_definition": "outside formal reference raster grid",
        "d2_not_removed": refs["D2"],
        "d3_not_removed": refs["D3"],
        "d4_not_removed": refs["D4"],
        "d5_not_removed": refs["D5"],
        "d6_not_removed": "POLICY_FROZEN_NOT_RECOMPUTED",
        "d7_not_removed": "POLICY_FROZEN_NOT_RECOMPUTED",
        "source_order": "normalized absolute path casefold sort",
        "source_record_order": "original OGR row position",
        "retention_rule": "first record by source path then source row",
        "D1_implementation": (
            "Exact Step42 dual pandas row hashes over sorted available raw "
            "property columns plus exact longitude/latitude; null to empty "
            "string; astype(str); no address normalization; provenance source "
            "id and row position excluded from equality"
        ),
        "sparse_path": str(sparse_path),
        "sparse_sha256": sparse_sha,
        "sparse_cells": len(sparse_rows),
        "point_count_sum_inside_grid": int(sparse_counts.sum(dtype=np.uint64)),
        "by_file_path": str(by_file_path),
        "by_file_sha256": sha256_file(by_file_path),
        "peak_memory_gib": rss_gib(),
        "runtime_seconds": time.monotonic() - started,
        "building_records_included": 0,
    }
    atomic_json(checkpoint, payload)
    print(
        f"[D1_COMPLETE] phase=state state={state} buffer=- "
        f"processed={total_raw:,}/{total_raw:,} percent=100 "
        f"elapsed={payload['runtime_seconds']:.1f}s ETA=0 "
        f"memory_gib={rss_gib():.2f} removed={records_removed:,} "
        f"output={sparse_path}",
        flush=True,
    )
    del x, y, d1a, d1b, source, order, keep, retained_x, retained_y
    gc.collect()
    return payload


def sparse_tile_index(sparse: dict[str, np.ndarray]) -> dict[str, Any]:
    rows = sparse["rows"].astype(np.int64, copy=False)
    cols = sparse["cols"].astype(np.int64, copy=False)
    counts = sparse["counts"].astype(np.uint32, copy=False)
    width = int(sparse["width"])
    nx = math.ceil(width / TILE)
    tile_id = (rows // TILE) * nx + (cols // TILE)
    order = np.argsort(tile_id, kind="stable")
    tids = tile_id[order]
    unique, begin = np.unique(tids, return_index=True)
    finish = np.r_[begin[1:], len(tids)]
    ranges = {
        int(tid): (int(lo), int(hi))
        for tid, lo, hi in zip(unique, begin, finish)
    }
    return {
        "rows": rows[order],
        "cols": cols[order],
        "counts": counts[order],
        "nx": nx,
        "ranges": ranges,
    }


def checkpoint_valid(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    output = Path(payload.get("output_path", ""))
    if (
        payload.get("run_status") == "PASS"
        and output.is_file()
        and sha256_file(output) == payload.get("file_sha256")
    ):
        return payload
    raise RuntimeError(f"Invalid raster checkpoint: {path}")


def classify_state_buffers(
    output: Path,
    state: str,
    fips: str,
    buffers: list[int],
    legacy_rows: pd.DataFrame,
    dedup: dict[str, Any],
) -> list[dict[str, Any]]:
    pending: list[int] = []
    completed_payloads: list[dict[str, Any]] = []
    for buffer_m in buffers:
        checkpoint = (
            output / "checkpoints" / f"raster_{state}_{buffer_m:04d}m.json"
        )
        existing = checkpoint_valid(checkpoint)
        if existing:
            print(
                f"[RASTER_RESUME] phase=classification state={state} "
                f"buffer={buffer_m} completed=1/1 percent=100 elapsed=0 ETA=0 "
                f"memory_gib={rss_gib():.2f} output={existing['output_path']}",
                flush=True,
            )
            completed_payloads.append(existing)
        else:
            pending.append(buffer_m)
    if not pending:
        return completed_payloads

    sparse_path = Path(dedup["sparse_path"])
    if sha256_file(sparse_path) != dedup["sparse_sha256"]:
        raise RuntimeError(f"Sparse point artifact changed: {state}")
    with np.load(sparse_path) as sparse_npz:
        sparse = {key: sparse_npz[key] for key in sparse_npz.files}
    index = sparse_tile_index(sparse)
    legacy_paths: dict[int, Path] = {}
    for buffer_m in pending:
        match = legacy_rows[
            legacy_rows["state"].eq(state)
            & legacy_rows["buffer_m"].astype(int).eq(buffer_m)
        ]
        if len(match) != 1:
            raise RuntimeError(f"Legacy raster unresolved: {state} {buffer_m}")
        legacy_paths[buffer_m] = Path(match.iloc[0]["source_path"])
    first_path = legacy_paths[pending[0]]
    with rasterio.open(first_path) as ref:
        width, height = ref.width, ref.height
        transform, crs = ref.transform, ref.crs
        profile = ref.profile.copy()
    if width != int(sparse["width"]) or height != int(sparse["height"]):
        raise RuntimeError(f"Sparse/reference grid mismatch: {state}")
    for path in legacy_paths.values():
        with rasterio.open(path) as source:
            if (
                source.width != width
                or source.height != height
                or source.transform != transform
                or source.crs != crs
            ):
                raise RuntimeError(f"Legacy sensitivity grid mismatch: {path}")
    geometry = state_geometry(fips)
    kernels: dict[int, dict[str, Any]] = {}
    for buffer_m in pending:
        radius_px = max(int(round(buffer_m / PIXEL_M)), 1)
        area = math.pi * buffer_m**2 / 1_000_000.0
        kernels[buffer_m] = {
            "radius_px": radius_px,
            "kernel": disk(radius_px).astype(np.float32),
            "minimum_count": int(math.floor(DENSITY_THRESHOLD * area)) + 1,
            "area_km2": area,
        }
    max_radius = max(item["radius_px"] for item in kernels.values())
    profile.update(
        driver="GTiff",
        dtype="uint8",
        count=1,
        nodata=NODATA,
        compress="LZW",
        tiled=True,
        BIGTIFF="YES",
    )
    partials: dict[int, Path] = {}
    finals: dict[int, Path] = {}
    destinations: dict[int, Any] = {}
    legacy_sources: dict[int, Any] = {}
    stats: dict[int, dict[str, Any]] = {}
    for buffer_m in pending:
        root = (
            output / "rasters_500m"
            if buffer_m == 500
            else output / "rasters_sensitivity"
        )
        state_dir = root / state
        state_dir.mkdir(parents=True, exist_ok=True)
        final = state_dir / f"WUI_P_P2_{state}_r{buffer_m:04d}m.tif"
        partial = state_dir / f"WUI_P_P2_{state}_r{buffer_m:04d}m.partial.tif"
        if final.exists():
            raise RuntimeError(f"Uncheckpointed final output exists: {final}")
        if partial.exists():
            partial.unlink()
        finals[buffer_m] = final
        partials[buffer_m] = partial
        destinations[buffer_m] = rasterio.open(partial, "w", **profile)
        legacy_sources[buffer_m] = rasterio.open(legacy_paths[buffer_m])
        stats[buffer_m] = {
            "valid_pixels": 0,
            "non_wui_pixels": 0,
            "intermix_pixels": 0,
            "interface_pixels": 0,
            "out_of_domain_pixels": 0,
            "legacy_wui_pixels": 0,
            "nonwui_to_wui": 0,
            "wui_to_nonwui": 0,
            "changed_pixels": 0,
            "intersection": 0,
            "union": 0,
            "classification_hasher": hashlib.sha256(
                f"STEP43_CLASS_TILE_STREAM_V1|{state}|{buffer_m}".encode()
            ),
            "step42_hasher": (
                hashlib.sha256(
                    f"STEP42_TILE_STREAM_V1|{state}|500|"
                    "P2_ADDRESS_EXACT_RECORD_DEDUP".encode()
                )
                if buffer_m == 500 and state in FIVE
                else None
            ),
        }
    nx = math.ceil(width / TILE)
    ny = math.ceil(height / TILE)
    total_tiles = nx * ny
    full = Window(0, 0, width, height)
    started = time.monotonic()
    try:
        for ty in range(ny):
            for tx in range(nx):
                core = Window(
                    tx * TILE,
                    ty * TILE,
                    min(TILE, width - tx * TILE),
                    min(TILE, height - ty * TILE),
                )
                pad = (
                    Window(
                        core.col_off - max_radius,
                        core.row_off - max_radius,
                        core.width + 2 * max_radius,
                        core.height + 2 * max_radius,
                    )
                    .intersection(full)
                    .round_offsets()
                    .round_lengths()
                )
                chunks: list[np.ndarray] = []
                min_tx = max(0, int(pad.col_off) // TILE)
                max_tx = min(nx - 1, int(pad.col_off + pad.width - 1) // TILE)
                min_ty = max(0, int(pad.row_off) // TILE)
                max_ty = min(ny - 1, int(pad.row_off + pad.height - 1) // TILE)
                for near_ty in range(min_ty, max_ty + 1):
                    for near_tx in range(min_tx, max_tx + 1):
                        interval = index["ranges"].get(near_ty * nx + near_tx)
                        if interval:
                            chunks.append(
                                np.arange(interval[0], interval[1], dtype=np.int64)
                            )
                point_array = np.zeros(
                    (int(pad.height), int(pad.width)), dtype=np.float32
                )
                if chunks:
                    selected = np.concatenate(chunks)
                    rr = index["rows"][selected]
                    cc = index["cols"][selected]
                    inside = (
                        (rr >= pad.row_off)
                        & (rr < pad.row_off + pad.height)
                        & (cc >= pad.col_off)
                        & (cc < pad.col_off + pad.width)
                    )
                    selected = selected[inside]
                    rr = index["rows"][selected] - int(pad.row_off)
                    cc = index["cols"][selected] - int(pad.col_off)
                    point_array[rr, cc] = index["counts"][selected]
                r0 = int(core.row_off - pad.row_off)
                c0 = int(core.col_off - pad.col_off)
                h0, w0 = int(core.height), int(core.width)
                domain = geometry_mask(
                    [geometry],
                    out_shape=(h0, w0),
                    transform=rasterio.windows.transform(core, transform),
                    invert=True,
                    all_touched=False,
                )
                for buffer_m in pending:
                    item = kernels[buffer_m]
                    summed = fftconvolve(
                        point_array, item["kernel"], mode="same"
                    )
                    count_core = np.rint(
                        summed[r0 : r0 + h0, c0 : c0 + w0]
                    ).astype(np.int32)
                    dense = count_core >= item["minimum_count"]
                    legacy = legacy_sources[buffer_m].read(1, window=core)
                    if not np.isin(legacy, [0, 1, 2]).all():
                        raise RuntimeError(
                            f"Legacy class outside 0/1/2: {state} {buffer_m}"
                        )
                    candidate = np.where(dense, legacy, 0).astype(np.uint8)
                    out_array = candidate.copy()
                    out_array[~domain] = NODATA
                    destinations[buffer_m].write(out_array, 1, window=core)
                    new_wui = candidate > 0
                    old_wui = legacy > 0
                    st = stats[buffer_m]
                    st["valid_pixels"] += int(domain.sum())
                    st["non_wui_pixels"] += int(np.sum((candidate == 0) & domain))
                    st["intermix_pixels"] += int(np.sum((candidate == 1) & domain))
                    st["interface_pixels"] += int(np.sum((candidate == 2) & domain))
                    st["out_of_domain_pixels"] += int((~domain).sum())
                    st["legacy_wui_pixels"] += int(np.sum(old_wui & domain))
                    st["nonwui_to_wui"] += int(
                        np.sum(~old_wui & new_wui & domain)
                    )
                    st["wui_to_nonwui"] += int(
                        np.sum(old_wui & ~new_wui & domain)
                    )
                    st["changed_pixels"] += int(
                        np.sum((legacy != candidate) & domain)
                    )
                    st["intersection"] += int(
                        np.sum(old_wui & new_wui & domain)
                    )
                    st["union"] += int(np.sum((old_wui | new_wui) & domain))
                    header = struct.pack(">IIII", ty, tx, h0, w0)
                    st["classification_hasher"].update(header)
                    st["classification_hasher"].update(
                        candidate.tobytes(order="C")
                    )
                    if st["step42_hasher"] is not None:
                        st["step42_hasher"].update(header)
                        st["step42_hasher"].update(
                            candidate.tobytes(order="C")
                        )
                completed = ty * nx + tx + 1
                elapsed = max(time.monotonic() - started, 1e-9)
                eta = (total_tiles - completed) / (completed / elapsed)
                print(
                    f"[CLASSIFY] phase=classification state={state} "
                    f"buffer={','.join(map(str,pending))} "
                    f"tile={completed}/{total_tiles} "
                    f"percent={100*completed/total_tiles:.2f}% "
                    f"elapsed={elapsed:.1f}s ETA={eta:.1f}s "
                    f"memory_gib={rss_gib():.2f} "
                    f"output={finals[pending[0]].parent}",
                    flush=True,
                )
    finally:
        for destination in destinations.values():
            destination.close()
        for source in legacy_sources.values():
            source.close()

    payloads: list[dict[str, Any]] = []
    step42_impact = pd.read_csv(
        STEP42 / "wui_p_address_policy_500m_impact.csv"
    )
    for buffer_m in pending:
        st = stats[buffer_m]
        if st["valid_pixels"] != (
            st["non_wui_pixels"] + st["intermix_pixels"] + st["interface_pixels"]
        ):
            raise RuntimeError(f"Class conservation failed: {state} {buffer_m}")
        with rasterio.open(partials[buffer_m]) as source:
            if (
                source.nodata != NODATA
                or source.dtypes[0] != "uint8"
                or source.width != width
                or source.height != height
                or source.transform != transform
                or source.crs != crs
            ):
                raise RuntimeError(f"Output metadata failed: {state} {buffer_m}")
            # Check written values by blocks before atomic promotion.
            for _, window in source.block_windows(1):
                values = source.read(1, window=window)
                if not np.isin(values, [0, 1, 2, NODATA]).all():
                    raise RuntimeError(f"Output values failed: {state} {buffer_m}")
        os.replace(partials[buffer_m], finals[buffer_m])
        file_hash = sha256_file(finals[buffer_m])
        new_wui = st["intermix_pixels"] + st["interface_pixels"]
        jaccard = st["intersection"] / st["union"] if st["union"] else 1.0
        step42_hash = (
            st["step42_hasher"].hexdigest()
            if st["step42_hasher"] is not None
            else ""
        )
        canary_disagreement: Any = ""
        canary_jaccard: Any = ""
        canary_expected_hash = ""
        if buffer_m == 500 and state in FIVE:
            expected = step42_impact[
                step42_impact["state"].eq(state)
                & step42_impact["scenario"].eq(
                    "P2_ADDRESS_EXACT_RECORD_DEDUP"
                )
            ]
            if len(expected) != 1:
                raise RuntimeError(f"Step42 canary row missing: {state}")
            expected = expected.iloc[0]
            canary_expected_hash = expected["classification_sha256"]
            canary_disagreement = (
                0 if step42_hash == canary_expected_hash else "HASH_MISMATCH"
            )
            canary_jaccard = (
                1.0 if step42_hash == canary_expected_hash else ""
            )
            if new_wui != int(expected["wui_pixels"]):
                canary_disagreement = "WUI_COUNT_MISMATCH"
        payload = {
            "run_status": "PASS",
            "state": state,
            "STATEFP": fips,
            "buffer_m": buffer_m,
            "input_policy": "P2_ADDRESS_EXACT_RECORD_DEDUP",
            "raw_address_records": dedup["raw_address_records"],
            "valid_address_records": dedup["valid_address_records"],
            "d1_removed_records": dedup["d1_records_removed"],
            "retained_records": dedup["retained_records"],
            "retained_records_inside_grid": dedup[
                "retained_records_inside_grid"
            ],
            "building_records_included": 0,
            "output_path": str(finals[buffer_m]),
            "width": width,
            "height": height,
            "crs": crs.to_string(),
            "transform": [float(x) for x in transform[:6]],
            "dtype": "uint8",
            "nodata": NODATA,
            "valid_pixels": st["valid_pixels"],
            "non_wui_pixels": st["non_wui_pixels"],
            "intermix_pixels": st["intermix_pixels"],
            "interface_pixels": st["interface_pixels"],
            "wui_pixels": new_wui,
            "out_of_domain_pixels": st["out_of_domain_pixels"],
            "classification_sha256": st[
                "classification_hasher"
            ].hexdigest(),
            "classification_hash_definition": (
                "SHA256 deterministic 2048 tile stream of class values with "
                "out-of-domain normalized to 0"
            ),
            "file_sha256": file_hash,
            "runtime_seconds": time.monotonic() - started,
            "peak_memory_gib": rss_gib(),
            "legacy_path": str(legacy_paths[buffer_m]),
            "legacy_wui_pixels": st["legacy_wui_pixels"],
            "wui_pixel_difference": new_wui - st["legacy_wui_pixels"],
            "difference_area_km2": (
                new_wui - st["legacy_wui_pixels"]
            )
            * 0.0009,
            "non_wui_to_wui": st["nonwui_to_wui"],
            "wui_to_non_wui": st["wui_to_nonwui"],
            "changed_pixels": st["changed_pixels"],
            "jaccard_with_legacy_P0": jaccard,
            "common_valid_pixels": st["valid_pixels"],
            "step42_compatible_classification_sha256": step42_hash,
            "step42_expected_classification_sha256": canary_expected_hash,
            "classification_disagreement_pixels_vs_step42": canary_disagreement,
            "jaccard_with_step42_P2": canary_jaccard,
            "valid_domain_rule": (
                "2022 Census state geometry pixel-center mask; outside=255"
            ),
            "notes": (
                "500m canary exact hash gate"
                if buffer_m == 500 and state in FIVE
                else "formal P2 candidate; no downstream metrics computed"
            ),
        }
        checkpoint = (
            output / "checkpoints" / f"raster_{state}_{buffer_m:04d}m.json"
        )
        atomic_json(checkpoint, payload)
        payloads.append(payload)
        print(
            f"[RASTER_COMPLETE] phase=classification state={state} "
            f"buffer={buffer_m} completed=1/1 percent=100 "
            f"elapsed={payload['runtime_seconds']:.1f}s ETA=0 "
            f"memory_gib={rss_gib():.2f} output={finals[buffer_m]}",
            flush=True,
        )
    return completed_payloads + payloads


def collect_dedup(output: Path) -> None:
    state_rows: list[dict[str, Any]] = []
    file_rows: list[dict[str, Any]] = []
    for path in sorted((output / "checkpoints").glob("dedup_??.json")):
        payload = json.loads(path.read_text())
        state_rows.append(
            {
                key: value
                for key, value in payload.items()
                if key
                not in {
                    "d2_not_removed",
                    "d3_not_removed",
                    "d4_not_removed",
                    "d5_not_removed",
                }
            }
            | {
                "d2_not_removed": json.dumps(payload["d2_not_removed"]),
                "d3_not_removed": json.dumps(payload["d3_not_removed"]),
                "d4_not_removed": json.dumps(payload["d4_not_removed"]),
                "d5_not_removed": json.dumps(payload["d5_not_removed"]),
            }
        )
        file_rows.extend(
            pd.read_csv(payload["by_file_path"], keep_default_na=False).to_dict(
                "records"
            )
        )
    write_csv(output / "dedup_audit/p2_d1_audit_by_state.csv", state_rows)
    write_csv(output / "dedup_audit/p2_d1_audit_by_file.csv", file_rows)


def collect_rasters(output: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((output / "checkpoints").glob("raster_??_????m.json")):
        payload = checkpoint_valid(path)
        if payload:
            rows.append(payload)
    rows.sort(key=lambda row: (int(row["STATEFP"]), int(row["buffer_m"])))
    write_csv(output / "qc/p2_rebuild_inventory.csv", rows)
    write_csv(
        output / "qc/p2_classification_change_vs_legacy.csv",
        [
            {
                key: row[key]
                for key in [
                    "state",
                    "buffer_m",
                    "legacy_path",
                    "output_path",
                    "legacy_wui_pixels",
                    "wui_pixels",
                    "wui_pixel_difference",
                    "difference_area_km2",
                    "non_wui_to_wui",
                    "wui_to_non_wui",
                    "changed_pixels",
                    "jaccard_with_legacy_P0",
                    "common_valid_pixels",
                ]
            }
            for row in rows
        ],
    )
    write_csv(
        output / "qc/p2_raster_metadata_audit.csv",
        [
            {
                key: row[key]
                for key in [
                    "state",
                    "buffer_m",
                    "output_path",
                    "width",
                    "height",
                    "crs",
                    "transform",
                    "dtype",
                    "nodata",
                    "file_sha256",
                    "run_status",
                ]
            }
            for row in rows
        ],
    )
    write_csv(
        output / "qc/p2_valid_domain_audit.csv",
        [
            {
                key: row[key]
                for key in [
                    "state",
                    "buffer_m",
                    "valid_domain_rule",
                    "valid_pixels",
                    "non_wui_pixels",
                    "intermix_pixels",
                    "interface_pixels",
                    "wui_pixels",
                    "out_of_domain_pixels",
                    "common_valid_pixels",
                ]
            }
            for row in rows
        ],
    )
    return rows


def canary_gate(output: Path) -> None:
    rows = collect_rasters(output)
    canary = [
        row for row in rows if row["state"] in FIVE and row["buffer_m"] == 500
    ]
    write_csv(output / "qc/p2_canary_comparison_step42.csv", canary)
    passed = (
        len(canary) == 5
        and all(
            row["classification_disagreement_pixels_vs_step42"] == 0
            and float(row["jaccard_with_step42_P2"]) == 1.0
            and row["step42_compatible_classification_sha256"]
            == row["step42_expected_classification_sha256"]
            for row in canary
        )
    )
    payload = {
        "status": "PASS" if passed else "BLOCKED",
        "completed_utc": utc_now(),
        "canary_states": len(canary),
        "all_zero_disagreement": passed,
        "all_jaccard_one": passed,
    }
    atomic_json(output / "checkpoints/phaseB_canary_gate.json", payload)
    if not passed:
        raise RuntimeError(f"Step42 canary gate failed: {canary}")


def field_definition(output: Path) -> None:
    rows = [
        {
            "component": "raw_property_fields",
            "definition": (
                "All property columns available in each source, sorted by field "
                "name; values fillna('') then astype(str)"
            ),
            "participates_in_D1": "YES",
        },
        {
            "component": "geometry",
            "definition": (
                "Valid Point exact longitude and latitude appended as float "
                "columns; whole-source centroid conversion only when first "
                "geometry is Polygon/MultiPolygon, matching legacy loader"
            ),
            "participates_in_D1": "YES",
        },
        {
            "component": "source_path",
            "definition": "Normalized/casefold-sorted only for deterministic order",
            "participates_in_D1": "NO",
        },
        {
            "component": "source_id",
            "definition": "Loader provenance and audit attribution only",
            "participates_in_D1": "NO",
        },
        {
            "component": "original_record_position",
            "definition": "Deterministic within-source order and keep-first rule",
            "participates_in_D1": "NO",
        },
        {
            "component": "normalization",
            "definition": (
                "No address normalization, punctuation removal, case folding, "
                "coordinate rounding, cell grouping or proximity matching"
            ),
            "participates_in_D1": "NO",
        },
        {
            "component": "hash",
            "definition": (
                "Exact Step42 dual pandas hash_pandas_object keys "
                "step42d1key00001/00002"
            ),
            "participates_in_D1": "GROUP_KEY",
        },
        {
            "component": "scope",
            "definition": "Same-file and cross-file D1 groups both detected",
            "participates_in_D1": "YES",
        },
    ]
    write_csv(output / "dedup_audit/p2_d1_field_definition.csv", rows)


def run_phase(output: Path, phase: str) -> None:
    context = load_context(output)
    policy = context["policy"].sort_values("STATEFP", key=lambda s: s.astype(int))
    if phase == "canary":
        selected = policy[policy["state"].isin(FIVE)]
        buffers_by_state = {state: [500] for state in FIVE}
    elif phase == "national":
        gate = output / "checkpoints/phaseB_canary_gate.json"
        if not gate.is_file() or json.loads(gate.read_text()).get("status") != "PASS":
            raise RuntimeError("National phase prohibited before canary PASS")
        selected = policy
        buffers_by_state = {state: [500] for state in policy["state"]}
    elif phase == "sensitivity":
        national_rows = collect_rasters(output)
        completed500 = {
            row["state"] for row in national_rows if int(row["buffer_m"]) == 500
        }
        if len(completed500) != 49:
            raise RuntimeError("Sensitivity phase prohibited before 49-unit 500m completion")
        selected = policy[policy["state"].isin(FIVE)]
        buffers_by_state = {
            state: [100, 200, 300, 400, 600, 700, 800, 900, 1000]
            for state in FIVE
        }
    else:
        raise ValueError(phase)

    failures: list[dict[str, Any]] = []
    for row in selected.itertuples():
        state, fips = row.state, str(row.STATEFP).zfill(2)
        legacy500 = context["legacy"][
            context["legacy"]["state"].eq(state)
            & context["legacy"]["buffer_m"].astype(int).eq(500)
        ]
        if len(legacy500) != 1:
            raise RuntimeError(f"500m reference missing: {state}")
        try:
            dedup = parse_and_dedup_state(
                output,
                state,
                fips,
                context["included"][context["included"]["state"].eq(state)],
                Path(legacy500.iloc[0]["source_path"]),
            )
            classify_state_buffers(
                output,
                state,
                fips,
                buffers_by_state[state],
                context["legacy"],
                dedup,
            )
            collect_dedup(output)
            collect_rasters(output)
        except Exception as error:
            failure = {
                "phase": phase,
                "state": state,
                "buffer_m": ",".join(map(str, buffers_by_state[state])),
                "failed_utc": utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "status": "FAILED_NOT_SILENTLY_SKIPPED",
            }
            failures.append(failure)
            write_csv(output / "qc/p2_failed_or_skipped_runs.csv", failures)
            history_path = output / "qc/p2_failure_attempt_history.csv"
            history = (
                pd.read_csv(history_path, keep_default_na=False).to_dict("records")
                if history_path.is_file()
                else []
            )
            history.append(failure)
            write_csv(history_path, history)
            atomic_json(
                output / "checkpoints" / f"phase_{phase}_FAILED.json", failure
            )
            raise
    if phase == "canary":
        canary_gate(output)
    atomic_json(
        output / "checkpoints" / f"phase_{phase}_complete.json",
        {
            "status": "PASS",
            "phase": phase,
            "completed_utc": utc_now(),
            "states": len(selected),
            "failed": 0,
        },
    )
    write_csv(
        output / "qc/p2_failed_or_skipped_runs.csv",
        [
            {
                "phase": phase,
                "state": "",
                "buffer_m": "",
                "failed_utc": "",
                "error_type": "",
                "error": "",
                "traceback": "",
                "status": "NO_FAILED_OR_SKIPPED_RUNS",
            }
        ],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=["canary", "national", "sensitivity"], required=True
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    field_definition(output)
    run_phase(output, args.phase)
    print(
        f"[PHASE_COMPLETE] phase={args.phase} state=ALL buffer=ALL "
        f"completed=1/1 percent=100 elapsed=NA ETA=0 "
        f"memory_gib={rss_gib():.2f} output={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
