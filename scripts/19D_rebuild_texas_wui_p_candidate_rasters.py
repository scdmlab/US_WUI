#!/usr/bin/env python3
# Copyright (C) 2026 WUI Mapping Project Authors
# SPDX-License-Identifier: GPL-3.0-only
# -*- coding: utf-8 -*-
# Repository version: filesystem paths are loaded through repo_config.py.
# Copy config/paths.example.json to config/paths.json before a new run.
# Raw input data and large intermediate files are stored outside GitHub.

"""
Step 19D - rebuild the ten Texas WUI-P candidate rasters from the passing
Step-19C inside-grid GeoPackage.

This is a non-destructive candidate build.  It discovers the latest passing
Step 19C run, verifies its SHA-256 and row count, reuses the locked Step-18
Texas grid/wildland/distance inputs, writes all products below a new UTC-
stamped directory, compares them with the historical rasters, and never
replaces a historical file.
"""

from __future__ import annotations

from repo_config import portable_path

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import importlib
import json
import math
import os
import shutil
import sqlite3
import subprocess
import sys
import time

import numpy as np
import pyogrio
import rasterio
from pyproj import CRS
from rasterio.windows import Window
from scipy.ndimage import convolve
from skimage.morphology import disk


STEP18_RUN = Path(
    portable_path("legacy", "WUI_TX_recovery/step18_archive_handoff_runs/texas_archive_handoff_20260722T213425Z")
)
HANDOFF_PATH = STEP18_RUN / "downstream_workspace/downstream_handoff.json"
BASELINE_PATH = STEP18_RUN / "archive/historical_baseline_recheck.json"
OUTPUT_PARENT = (
    STEP18_RUN / "downstream_workspace/wui_p_rasters_pending_step19"
)

LAYER = "Texas_addresses"
EXPECTED_ROWS = 20_991_807
EXPECTED_CRS = CRS.from_epsg(5070)
RADII_M = list(range(100, 1001, 100))
DENSITY_THRESHOLD = 6.17
VEGETATION_THRESHOLD = 0.50
DISTANCE_THRESHOLD_M = 2400.0
TILE_SIZE = 2048
PADDING_PX = 120
REPORT_SECONDS = 20.0
VERDICT = "TEN_TEXAS_WUI_P_CANDIDATE_RASTERS_PASS_QC"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as source:
        return json.load(source)


def save_json(path: Path, payload) -> None:
    partial = path.with_name(path.name + ".partial")
    if path.exists() or partial.exists():
        raise FileExistsError(path)
    partial.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def sha256_file(path: Path, label: str = "") -> str:
    digest = hashlib.sha256()
    size = path.stat().st_size
    done = 0
    started = time.monotonic()
    last = started
    with path.open("rb") as source:
        while True:
            chunk = source.read(8 * 1024**2)
            if not chunk:
                break
            digest.update(chunk)
            done += len(chunk)
            now = time.monotonic()
            if label and (now - last >= REPORT_SECONDS or done == size):
                elapsed = max(now - started, 1e-9)
                rate = done / elapsed
                eta = (size - done) / rate if rate else 0.0
                print(
                    f"HASH {label}: {done:,}/{size:,} "
                    f"({100.0 * done / size:.1f}%) | "
                    f"elapsed={elapsed/60:.2f}m | ETA={eta/60:.2f}m",
                    flush=True,
                )
                last = now
    return digest.hexdigest()


def fingerprint(path: Path, label: str = "") -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "sha256": sha256_file(path, label),
    }


def assert_unchanged(path: Path, before: dict, label: str) -> dict:
    after = fingerprint(path, label)
    for field in ("size_bytes", "mtime_ns", "device", "inode", "sha256"):
        if after[field] != before[field]:
            raise RuntimeError(f"Protected input changed ({field}): {path}")
    return after


def find_latest_step19c() -> tuple[Path, dict]:
    candidates = []
    for summary_path in OUTPUT_PARENT.rglob("step19C_summary.json"):
        try:
            summary = load_json(summary_path)
        except Exception:
            continue
        if (
            summary.get("verdict")
            == "CANDIDATE_GPKG_AND_QUARANTINE_EXACTLY_CONSERVE_WORKING_POINTS"
            and summary.get("count_conserved") is True
            and int(summary.get("candidate_rows", -1)) == EXPECTED_ROWS
        ):
            candidate = Path(summary["candidate_gpkg"])
            if candidate.is_file():
                candidates.append((summary_path, summary))
    if not candidates:
        raise FileNotFoundError("No passing Step 19C summary/candidate was found")
    return max(candidates, key=lambda item: item[0].stat().st_mtime)


def reject_sqlite_sidecars(path: Path) -> None:
    sidecars = [
        Path(str(path) + suffix)
        for suffix in ("-wal", "-shm", "-journal")
        if Path(str(path) + suffix).exists()
    ]
    if sidecars:
        raise RuntimeError(f"Unexpected SQLite sidecars: {sidecars}")


def candidate_database_qc(path: Path) -> dict:
    reject_sqlite_sidecars(path)
    info = pyogrio.read_info(path, layer=LAYER)
    rows = int(info["features"])
    geometry = str(info["geometry_type"])
    crs = CRS.from_user_input(info["crs"])
    if rows != EXPECTED_ROWS:
        raise RuntimeError(f"Candidate rows changed: {rows:,}")
    if geometry.lower() not in {"point", "point z", "3d point"}:
        raise RuntimeError(f"Unexpected candidate geometry: {geometry}")
    if not crs.equals(EXPECTED_CRS):
        raise RuntimeError(f"Candidate CRS is not EPSG:5070: {crs}")
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        integrity = [row[0] for row in connection.execute("PRAGMA integrity_check")]
        foreign_keys = list(connection.execute("PRAGMA foreign_key_check"))
        sql_rows = int(
            connection.execute(f'SELECT COUNT(*) FROM "{LAYER}"').fetchone()[0]
        )
    if integrity != ["ok"] or foreign_keys or sql_rows != EXPECTED_ROWS:
        raise RuntimeError(
            f"Candidate GeoPackage QC failed: integrity={integrity[:3]}, "
            f"foreign_keys={len(foreign_keys)}, rows={sql_rows:,}"
        )
    reject_sqlite_sidecars(path)
    return {
        "rows": rows,
        "geometry_type": geometry,
        "crs": crs.to_string(),
        "bounds": [float(value) for value in info["total_bounds"]],
        "integrity_check": "ok",
        "foreign_key_check_rows": 0,
    }


def grid_record(path: Path) -> dict:
    with rasterio.open(path) as source:
        return {
            "path": str(path),
            "width": int(source.width),
            "height": int(source.height),
            "count": int(source.count),
            "dtype": source.dtypes[0],
            "crs": source.crs.to_string() if source.crs else None,
            "transform": [float(value) for value in source.transform[:6]],
            "bounds": [float(value) for value in source.bounds],
            "nodata": source.nodata,
            "compression": (
                source.compression.value.upper() if source.compression else None
            ),
        }


def signature(record: dict) -> tuple:
    return (
        int(record["width"]),
        int(record["height"]),
        str(record["crs"]),
        tuple(float(value) for value in record["transform"]),
        tuple(float(value) for value in record["bounds"]),
    )


def validate_grid(record: dict, expected: dict, label: str) -> None:
    expected_record = {
        "width": expected["width"],
        "height": expected["height"],
        "crs": expected["crs"],
        "transform": expected["transform"],
        "bounds": expected.get("bounds", record["bounds"]),
    }
    if signature(record) != signature(expected_record):
        raise RuntimeError(f"{label} does not match the locked Texas grid")


def run(command: list[str], label: str) -> None:
    print(f"\nSTART {label}", flush=True)
    print("COMMAND:", " ".join(command), flush=True)
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if completed.stdout:
        print(completed.stdout.rstrip(), flush=True)
    if completed.returncode:
        raise RuntimeError(f"{label} failed with exit {completed.returncode}")


def build_point_count(
    candidate: Path,
    reference: dict,
    point_partial: Path,
    point_final: Path,
) -> dict:
    left, bottom, right, top = reference["bounds"]
    command = [
        "gdal_rasterize", "-q", "-of", "GTiff",
        "-l", LAYER, "-burn", "1", "-add", "-init", "0",
        "-a_nodata", "0", "-ot", "UInt32",
        "-te", repr(left), repr(bottom), repr(right), repr(top),
        "-ts", str(reference["width"]), str(reference["height"]),
        "-a_srs", "EPSG:5070",
        "-co", "TILED=YES", "-co", "COMPRESS=LZW", "-co", "BIGTIFF=YES",
        str(candidate), str(point_partial),
    ]
    run(command, "candidate address-point rasterization")
    record = grid_record(point_partial)
    validate_grid(record, reference, "Point-count raster")
    if record["dtype"] != "uint32" or record["count"] != 1:
        raise RuntimeError(f"Unexpected point-count schema: {record}")
    os.replace(point_partial, point_final)
    return grid_record(point_final)


def iter_tiles(width: int, height: int):
    x_steps = math.ceil(width / TILE_SIZE)
    y_steps = math.ceil(height / TILE_SIZE)
    for tile_y in range(y_steps):
        for tile_x in range(x_steps):
            yield (
                Window(
                    tile_x * TILE_SIZE,
                    tile_y * TILE_SIZE,
                    min(TILE_SIZE, width - tile_x * TILE_SIZE),
                    min(TILE_SIZE, height - tile_y * TILE_SIZE),
                ),
                x_steps * y_steps,
            )


def rebuild(
    point_path: Path,
    wild_path: Path,
    distance_path: Path,
    grid: dict,
    staging: Path,
) -> tuple[dict, dict]:
    with rasterio.open(wild_path) as reference:
        profile = reference.profile.copy()
        pixel_size = abs(float(reference.transform.a))
    profile.update(
        driver="GTiff", dtype="uint8", count=1, nodata=0,
        compress="LZW", tiled=True, BIGTIFF="YES",
    )
    kernels = {}
    for radius in RADII_M:
        radius_px = max(int(round(radius / pixel_size)), 1)
        kernel = disk(radius_px).astype(np.float32)
        kernels[radius] = {
            "radius_px": radius_px,
            "kernel": kernel,
            "cells": int(kernel.sum()),
            "area_km2": math.pi * radius**2 / 1_000_000.0,
        }
    if PADDING_PX < max(item["radius_px"] for item in kernels.values()):
        raise RuntimeError("Tile padding is smaller than the largest kernel")

    partials = {
        radius: staging / f"WUI_P_Texas_r{radius:04d}m.partial.tif"
        for radius in RADII_M
    }
    destinations = {
        radius: rasterio.open(path, "w", **profile)
        for radius, path in partials.items()
    }
    point_sum = 0
    nonzero = 0
    maximum = 0
    started = time.monotonic()
    try:
        with (
            rasterio.open(point_path) as points,
            rasterio.open(wild_path) as wild_source,
            rasterio.open(distance_path) as distance_source,
        ):
            full = Window(0, 0, wild_source.width, wild_source.height)
            completed = 0
            for core, total_tiles in iter_tiles(wild_source.width, wild_source.height):
                pad = Window(
                    core.col_off - PADDING_PX,
                    core.row_off - PADDING_PX,
                    core.width + 2 * PADDING_PX,
                    core.height + 2 * PADDING_PX,
                ).intersection(full).round_offsets().round_lengths()
                point_array = points.read(1, window=pad).astype(np.float32)
                wild = wild_source.read(1, window=pad).astype(np.float32)
                distance = distance_source.read(1, window=pad).astype(np.float32)
                r0 = int(core.row_off - pad.row_off)
                c0 = int(core.col_off - pad.col_off)
                h0, w0 = int(core.height), int(core.width)
                point_core = point_array[r0:r0+h0, c0:c0+w0]
                point_sum += int(point_core.sum(dtype=np.float64))
                nonzero += int(np.count_nonzero(point_core))
                maximum = max(maximum, int(point_core.max(initial=0)))
                for radius, item in kernels.items():
                    structures = convolve(
                        point_array, item["kernel"], mode="constant", cval=0.0
                    )
                    vegetation_sum = convolve(
                        wild, item["kernel"], mode="constant", cval=0.0
                    )
                    dense = structures / item["area_km2"] > DENSITY_THRESHOLD
                    vegetation = vegetation_sum / float(item["cells"])
                    intermix = dense & (vegetation > VEGETATION_THRESHOLD)
                    interface = (
                        dense
                        & (vegetation <= VEGETATION_THRESHOLD)
                        & (distance <= DISTANCE_THRESHOLD_M)
                    )
                    output = np.zeros((h0, w0), dtype=np.uint8)
                    output[intermix[r0:r0+h0, c0:c0+w0]] = 1
                    output[interface[r0:r0+h0, c0:c0+w0]] = 2
                    destinations[radius].write(output, 1, window=core)
                completed += 1
                elapsed = max(time.monotonic() - started, 1e-9)
                eta = (total_tiles - completed) / (completed / elapsed)
                print(
                    f"TILES {completed}/{total_tiles} "
                    f"({100.0*completed/total_tiles:.1f}%) | "
                    f"elapsed={elapsed/60:.2f}m | ETA={eta/60:.2f}m",
                    flush=True,
                )
    finally:
        for destination in destinations.values():
            destination.close()
    kernel_qc = {
        str(radius): {
            "radius_px": item["radius_px"],
            "kernel_cells": item["cells"],
            "buffer_area_km2": item["area_km2"],
        }
        for radius, item in kernels.items()
    }
    return partials, {
        "rasterized_address_sum": point_sum,
        "nonzero_point_cells": nonzero,
        "maximum_addresses_per_cell": maximum,
        "kernel_parameters": kernel_qc,
    }


def compare_rasters(new_path: Path, old_path: Path, grid: dict, radius: int) -> dict:
    new_counts = np.zeros(3, dtype=np.int64)
    old_counts = np.zeros(3, dtype=np.int64)
    confusion = np.zeros((3, 3), dtype=np.int64)
    changed = 0
    total = 0
    with rasterio.open(new_path) as new, rasterio.open(old_path) as old:
        validate_grid(grid_record(new_path), grid, "New candidate raster")
        validate_grid(grid_record(old_path), grid, "Historical raster")
        windows = list(new.block_windows(1))
        for index, (_, window) in enumerate(windows, 1):
            a = new.read(1, window=window)
            b = old.read(1, window=window)
            if not np.isin(a, (0, 1, 2)).all():
                raise RuntimeError(f"Invalid new class at radius {radius}")
            if not np.isin(b, (0, 1, 2)).all():
                raise RuntimeError(f"Invalid historical class at radius {radius}")
            new_counts += np.bincount(a.ravel(), minlength=3)[:3]
            old_counts += np.bincount(b.ravel(), minlength=3)[:3]
            confusion += np.bincount(
                (b.astype(np.int16) * 3 + a.astype(np.int16)).ravel(),
                minlength=9,
            ).reshape(3, 3)
            changed += int(np.count_nonzero(a != b))
            total += int(a.size)
            if index == len(windows) or index % 100 == 0:
                print(
                    f"COMPARE r={radius}: {index}/{len(windows)} blocks",
                    flush=True,
                )
    if int(new_counts[1] + new_counts[2]) == 0:
        raise RuntimeError(f"Radius {radius} candidate has zero WUI pixels")
    pixel_area = abs(grid["transform"][0] * grid["transform"][4]) / 1_000_000
    return {
        "radius_m": radius,
        "historical_path": str(old_path),
        "new_class_counts_0_1_2": new_counts.tolist(),
        "historical_class_counts_0_1_2": old_counts.tolist(),
        "new_wui_area_km2": float((new_counts[1] + new_counts[2]) * pixel_area),
        "historical_wui_area_km2": float(
            (old_counts[1] + old_counts[2]) * pixel_area
        ),
        "changed_pixels": changed,
        "changed_percent_all_grid": 100.0 * changed / total,
        "exact_pixel_match": changed == 0,
        "confusion_old_rows_new_columns": confusion.tolist(),
    }


def main() -> None:
    started = time.monotonic()
    if os.environ.get("CONDA_DEFAULT_ENV") != "vscserver":
        raise RuntimeError("Run `conda activate vscserver` first")
    if shutil.which("gdal_rasterize") is None:
        raise RuntimeError("gdal_rasterize is not available")

    summary19c_path, summary19c = find_latest_step19c()
    candidate = Path(summary19c["candidate_gpkg"])
    if sha256_file(candidate, "Step 19C candidate") != summary19c["candidate_sha256"]:
        raise RuntimeError("Step 19C candidate SHA-256 mismatch")
    candidate_before = fingerprint(candidate)
    candidate_qc = candidate_database_qc(candidate)

    handoff = load_json(HANDOFF_PATH)
    baseline = load_json(BASELINE_PATH)
    if handoff.get("status") != "PASS" or baseline.get("status") != "PASS":
        raise RuntimeError("Step 18 handoff/baseline gate is not PASS")
    wild_path = Path(handoff["wildland_bin"]["path"])
    distance_path = Path(handoff["dist_to_largepatch"]["path"])
    grid = handoff["raster_grid_signature"]
    wild_record = grid_record(wild_path)
    distance_record = grid_record(distance_path)
    # Add explicit bounds because older handoff signatures may omit them.
    grid = {**grid, "bounds": wild_record["bounds"]}
    validate_grid(wild_record, grid, "Wildland raster")
    validate_grid(distance_record, grid, "Distance raster")
    if signature(wild_record) != signature(distance_record):
        raise RuntimeError("Wildland and distance grids differ")

    historical = {
        Path(record["path"]).name: Path(record["path"])
        for record in baseline["records"]
        if Path(record["path"]).name.startswith("WUI_P_Texas_r")
    }
    expected_names = [f"WUI_P_Texas_r{radius:04d}m.tif" for radius in RADII_M]
    if sorted(historical) != sorted(expected_names):
        raise RuntimeError("Historical ten-raster set is incomplete")

    protected = {
        "candidate": candidate_before,
        "wildland": fingerprint(wild_path, "wildland before"),
        "distance": fingerprint(distance_path, "distance before"),
        **{
            f"historical_{name}": fingerprint(path, f"{name} before")
            for name, path in historical.items()
        },
    }

    run_dir = OUTPUT_PARENT / f"step19D_texas_wui_p_candidate_{utc_stamp()}"
    staging = run_dir / "rasters.partial"
    final_dir = run_dir / "rasters"
    intermediate = run_dir / "intermediate"
    qc_dir = run_dir / "qc"
    for directory in (staging, intermediate, qc_dir):
        directory.mkdir(parents=True, exist_ok=False)

    pixel_count = int(grid["width"]) * int(grid["height"])
    minimum_free = int(pixel_count * 14 * 1.20 + 2 * 1024**3)
    free = shutil.disk_usage(OUTPUT_PARENT).free
    print(f"Free space: {free/1024**3:.2f} GiB", flush=True)
    print(f"Conservative minimum: {minimum_free/1024**3:.2f} GiB", flush=True)
    if free < minimum_free:
        raise RuntimeError("Insufficient free disk space")

    point_partial = intermediate / "Texas_candidate_point_count.partial.tif"
    point_final = intermediate / "Texas_candidate_point_count.tif"
    point_record = build_point_count(
        candidate, grid, point_partial, point_final
    )
    partials, build_qc = rebuild(
        point_final, wild_path, distance_path, grid, staging
    )
    if build_qc["rasterized_address_sum"] != EXPECTED_ROWS:
        raise RuntimeError(
            "Rasterized candidate address count is not conserved: "
            f"{build_qc['rasterized_address_sum']:,} != {EXPECTED_ROWS:,}"
        )

    comparisons = []
    for index, radius in enumerate(RADII_M, 1):
        name = expected_names[index - 1]
        comparisons.append(
            compare_rasters(partials[radius], historical[name], grid, radius)
        )
        os.replace(partials[radius], staging / name)
        print(f"RASTER QC {index}/10 PASS: {name}", flush=True)
    os.replace(staging, final_dir)

    outputs = []
    for index, name in enumerate(expected_names, 1):
        path = final_dir / name
        outputs.append(
            {
                "radius_m": RADII_M[index - 1],
                **fingerprint(path, f"output {index}/10"),
                "grid": grid_record(path),
            }
        )

    assert_unchanged(candidate, protected["candidate"], "candidate after")
    assert_unchanged(wild_path, protected["wildland"], "wildland after")
    assert_unchanged(distance_path, protected["distance"], "distance after")
    for name, path in historical.items():
        assert_unchanged(path, protected[f"historical_{name}"], f"{name} after")

    comparison_path = qc_dir / "step19D_historical_comparison.json"
    output_manifest_path = qc_dir / "step19D_output_manifest.json"
    save_json(comparison_path, {"status": "PASS", "records": comparisons})
    save_json(output_manifest_path, {"status": "PASS", "records": outputs})

    final_qc_path = run_dir / "step19D_summary.json"
    report_path = run_dir / "step19D_report.txt"
    payload = {
        "verdict": VERDICT,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "step19c_summary": str(summary19c_path),
        "candidate_gpkg": str(candidate),
        "candidate_rows": EXPECTED_ROWS,
        "candidate_database_qc": candidate_qc,
        "candidate_sha256": summary19c["candidate_sha256"],
        "point_count_raster": str(point_final),
        "point_count_grid": point_record,
        **build_qc,
        "radii_m": RADII_M,
        "output_count": len(outputs),
        "output_directory": str(final_dir),
        "comparison_manifest": str(comparison_path),
        "output_manifest": str(output_manifest_path),
        "candidate_gpkg_modified": False,
        "historical_rasters_modified": False,
        "researchdrive_write_performed": False,
        "elapsed_minutes": (time.monotonic() - started) / 60.0,
        "software": {
            "python": sys.version,
            "numpy": importlib.import_module("numpy").__version__,
            "rasterio": importlib.import_module("rasterio").__version__,
            "scipy": importlib.import_module("scipy").__version__,
            "skimage": importlib.import_module("skimage").__version__,
        },
    }
    save_json(final_qc_path, payload)
    changed = {item["radius_m"]: item["changed_pixels"] for item in comparisons}
    report = "\n".join(
        [
            "STEP 19D TEXAS WUI-P CANDIDATE RASTER REBUILD",
            "=" * 120,
            f"VERDICT: {VERDICT}",
            "",
            "COUNT AND OUTPUT QC",
            "-" * 120,
            f"Candidate GPKG rows: {EXPECTED_ROWS:,}",
            f"Addresses rasterized inside grid: {build_qc['rasterized_address_sum']:,}",
            f"Candidate rasters created: {len(outputs)}",
            f"Radii: {','.join(map(str, RADII_M))}",
            f"Changed pixels by radius: {json.dumps(changed, sort_keys=True)}",
            "",
            "SAFETY STATUS",
            "-" * 120,
            "Step 19C candidate GPKG modified: NO",
            "Historical rasters modified: NO",
            "ResearchDrive write performed: NO",
            "Historical rasters replaced: NO",
            "",
            "OUTPUT FILES",
            "-" * 120,
            str(final_dir),
            str(comparison_path),
            str(output_manifest_path),
            str(final_qc_path),
            str(report_path),
            "",
            f"Total elapsed minutes: {payload['elapsed_minutes']:.2f}",
        ]
    ) + "\n"
    report_path.write_text(report, encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()