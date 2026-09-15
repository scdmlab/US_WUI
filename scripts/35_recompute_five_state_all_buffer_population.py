#!/usr/bin/env python3
# Copyright (C) 2026 WUI Mapping Project Authors
# SPDX-License-Identifier: GPL-3.0-only
# -*- coding: utf-8 -*-
# Repository version: filesystem paths are loaded through repo_config.py.
# Copy config/paths.example.json to config/paths.json before a new run.
# Raw input data and large intermediate files are stored outside GitHub.

"""
STEP 35 - Recompute and audit five-state all-buffer WUI-P/WUI-S population.

Scope
-----
California, Colorado, Florida, Pennsylvania and Texas
WUI-P and WUI-S
100, 200, ..., 1000 m

The final output contains exactly:

    5 states x 2 methods x 10 buffers = 100 rows

This is not a reformat of the historical fast-raster Table 4 CSV.  It
independently applies the population method validated by Steps 08, 25-31,
33, 19F and 34:

1. aggregate 2020 Census block POP20 to block groups;
2. create one representative point per address/building feature;
3. assign each point to an exact target-state Census block;
4. sample all ten WUI class rasters while processing the points only once;
5. allocate each block group's POP20 according to its eligible point counts
   in Non-WUI, Intermix and Interface;
6. assign zero-point block-group population to Non-WUI;
7. require population conservation for every state-method-buffer row; and
8. require every recomputed 500 m row to equal the final Step 34 result.

Texas WUI-P is never read from the historical ResearchDrive source.  Its
locked Step 19C address candidate and all ten independently validated Step
19D/19E WUI-P rasters are required and hash-checked.

The script is resumable at the state-method level.  Reuse the same --run-dir
after an interruption; completed checkpoints whose input signatures have not
changed are skipped.

Required sibling scripts
------------------------
08_recompute_sample4_ps_population.py
25_audit_vermont_wuis_exact_point_in_polygon.py
19F_recompute_texas_wui_p_wui_s_population.py

Dependencies
------------
Python: fiona, numpy, pandas, rasterio, shapely
Command: ogr2ogr
"""

from __future__ import annotations

from repo_config import portable_path

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import traceback
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.windows import Window
from shapely.geometry import Point


PROJECT_ROOT = Path(portable_path("project"))
TABLES_ROOT = Path(portable_path("legacy", "WUI_tables_compare"))

DEFAULT_BLOCKS_DIR = Path(
    portable_path("data", "MBF+NLCD_2022US/Inputs/Processed_GPKG")
)
DEFAULT_OA_DIR = Path(
    portable_path("data", "OpenAddresses_Work/Processed_GPKG")
)
DEFAULT_MBF_DIR = DEFAULT_BLOCKS_DIR
DEFAULT_WUIP_DIR = Path(
    portable_path("data", "WUI_P_Paper_Raster")
)
DEFAULT_WUIS_DIR = Path(
    portable_path("data", "WUI_S_Paper")
)
DEFAULT_OLD_PAPER_CSV = (
    TABLES_ROOT
    / "CONUS_table4_sample5_pop_fast_raster_v1"
    / "table4_population_state_buffer_SAMPLE5_100_1000_FINAL.csv"
)

HELPER_NAME = "08_recompute_sample4_ps_population.py"
EXACT_NAME = "25_audit_vermont_wuis_exact_point_in_polygon.py"
TX_NAME = "19F_recompute_texas_wui_p_wui_s_population.py"

BUFFERS = tuple(range(100, 1001, 100))
METHODS = ("WUI-P", "WUI-S")
POP_TOLERANCE = 1e-6
SHARE_TOLERANCE = 1e-8
EXPECTED_ROWS = 100
EXPECTED_JOBS = 10
PASS = "PASS"
FINAL_PASS_VERDICT = (
    "FIVE_STATE_ALL_BUFFER_POPULATION_RECOMPUTED_AND_QC_PASS"
)
FINAL_REVIEW_VERDICT = (
    "FIVE_STATE_ALL_BUFFER_POPULATION_INCOMPLETE_OR_REVIEW"
)

STATE_ROWS = (
    ("06", "CA", "California", 39_538_223),
    ("08", "CO", "Colorado", 5_773_714),
    ("12", "FL", "Florida", 21_538_187),
    ("42", "PA", "Pennsylvania", 13_002_700),
    ("48", "TX", "Texas", 29_145_505),
)
STATE_BY_FIPS = {
    row[0]: {
        "STATEFP": row[0],
        "STUSPS": row[1],
        "state_name": row[2],
        "file_token": row[2],
        "official_population": row[3],
    }
    for row in STATE_ROWS
}
STATE_ORDER = {row[0]: index for index, row in enumerate(STATE_ROWS)}
METHOD_ORDER = {method: index for index, method in enumerate(METHODS)}


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def hms(seconds: float) -> str:
    value = max(0, int(seconds))
    hours, remainder = divmod(value, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def load_module(path: Path, name: str):
    if not path.is_file():
        raise FileNotFoundError(f"Required sibling script is missing: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Python module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def atomic_write_text(path: Path, text: str) -> None:
    partial = path.with_name(path.name + ".partial")
    partial.write_text(text, encoding="utf-8")
    os.replace(partial, path)


def atomic_write_json(path: Path, payload: object) -> None:
    atomic_write_text(
        path,
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            default=str,
        )
        + "\n",
    )


def atomic_write_csv(
    frame: pd.DataFrame,
    path: Path,
    *,
    float_format: str | None = None,
) -> None:
    partial = path.with_name(path.name + ".partial")
    frame.to_csv(partial, index=False, float_format=float_format)
    os.replace(partial, path)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as source:
        payload = json.load(source)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return payload


def require_file(path: Path, label: str) -> Path:
    result = path.expanduser().resolve()
    if not result.is_file() or result.stat().st_size <= 0:
        raise FileNotFoundError(f"{label} not found or empty: {result}")
    return result


def require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    label: str,
) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise RuntimeError(f"{label} is missing required columns: {missing}")


def normalize_fips(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(2)
    )


def close(left: object, right: object, tolerance: float) -> bool:
    try:
        a = float(left)
        b = float(right)
    except (TypeError, ValueError):
        return False
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tolerance


def file_signature(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def same_signature(left: dict, right: dict) -> bool:
    return all(
        left.get(field) == right.get(field)
        for field in ("path", "size", "mtime_ns")
    )


def sha256_file(
    path: Path,
    *,
    label: str = "",
    report_seconds: float = 30.0,
) -> str:
    digest = hashlib.sha256()
    size = path.stat().st_size
    done = 0
    started = time.monotonic()
    last_report = started
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(chunk)
            done += len(chunk)
            now = time.monotonic()
            if label and (now - last_report >= report_seconds or done == size):
                elapsed = max(now - started, 1e-9)
                rate = done / elapsed
                eta = (size - done) / rate if rate else 0.0
                print(
                    f"[HASH] {label}: {done:,}/{size:,} "
                    f"({100.0 * done / size:.2f}%) | "
                    f"elapsed={hms(elapsed)} | ETA={hms(eta)}",
                    flush=True,
                )
                last_report = now
    return digest.hexdigest()


def normalized_crs(helper, value: object, label: str) -> CRS:
    crs = CRS.from_user_input(value)
    normalized, repaired = helper.normalized_sampling_crs(crs, label)
    if repaired:
        print(f"[CRS] Repaired known legacy EPSG:5070 metadata: {label}")
    return normalized


def sample_wui_by_tiles(
    wui: rasterio.io.DatasetReader,
    xs: np.ndarray,
    ys: np.ndarray,
    tile_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    inv = ~wui.transform
    cols_f, rows_f = inv * (xs, ys)
    cols = np.floor(cols_f).astype(np.int64)
    rows = np.floor(rows_f).astype(np.int64)
    inbounds = (
        (rows >= 0)
        & (rows < wui.height)
        & (cols >= 0)
        & (cols < wui.width)
    )
    values = np.zeros(len(xs), dtype=np.int16)
    valid = np.flatnonzero(inbounds)
    if not len(valid):
        return values, inbounds

    valid_rows = rows[valid]
    valid_cols = cols[valid]
    tile_columns = (wui.width + tile_size - 1) // tile_size
    keys = (valid_rows // tile_size) * tile_columns + (
        valid_cols // tile_size
    )
    order = np.argsort(keys, kind="stable")
    sorted_indices = valid[order]
    sorted_keys = keys[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_keys)) + 1]
    ends = np.r_[starts[1:], len(sorted_indices)]

    for start, end in zip(starts, ends):
        indices = sorted_indices[start:end]
        row0 = int((rows[indices[0]] // tile_size) * tile_size)
        col0 = int((cols[indices[0]] // tile_size) * tile_size)
        height = min(tile_size, wui.height - row0)
        width = min(tile_size, wui.width - col0)
        array = wui.read(1, window=Window(col0, row0, width, height))
        local_rows = rows[indices] - row0
        local_cols = cols[indices] - col0
        values[indices] = array[local_rows, local_cols].astype(
            np.int16, copy=False
        )
    return values, inbounds


def find_step34_dir(project_root: Path, requested: str) -> Path:
    if requested:
        candidates = [Path(requested).expanduser().resolve()]
    else:
        candidates = sorted(
            (
                path.resolve()
                for path in project_root.glob(
                    "step34_national_147_population_*"
                )
                if path.is_dir()
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    required = (
        "VERDICT: NATIONAL_147_POPULATION_RESULTS_MERGED_AND_QC_PASS",
        "FINAL_ROWS: 147",
        "PASS_ROW_QC: 147/147",
        "NONPASS_ROWS_OR_GATES: 0",
    )
    rejected: list[str] = []
    for directory in candidates:
        qc = directory / "STEP34_FINAL_QC.txt"
        table = directory / "step34_national_population_long_147.csv"
        if not qc.is_file() or not table.is_file():
            rejected.append(f"{directory}: missing final QC or population table")
            continue
        text = qc.read_text(encoding="utf-8", errors="replace")
        missing = [token for token in required if token not in text]
        if missing:
            rejected.append(f"{directory}: final QC lacks {missing}")
            continue
        return directory
    detail = "\n- ".join(rejected[:20]) if rejected else "no candidates"
    raise FileNotFoundError(
        "No passing Step 34 national population directory found:\n- " + detail
    )


def load_step34_five_state_500m(step34_dir: Path) -> tuple[pd.DataFrame, Path]:
    path = require_file(
        step34_dir / "step34_national_population_long_147.csv",
        "Step 34 population table",
    )
    frame = pd.read_csv(path, dtype={"STATEFP": str})
    required = {
        "STATEFP",
        "STUSPS",
        "state_name",
        "method",
        "nonwui_population",
        "intermix_population",
        "interface_population",
        "wui_population",
        "total_population",
        "wui_population_share_pct",
        "source_qc",
    }
    require_columns(frame, required, str(path))
    frame = frame.copy()
    frame["STATEFP"] = normalize_fips(frame["STATEFP"])
    wanted = frame[
        frame["STATEFP"].isin(STATE_BY_FIPS)
        & frame["method"].isin(METHODS)
    ].copy()
    expected_keys = {
        (statefp, method)
        for statefp in STATE_BY_FIPS
        for method in METHODS
    }
    actual_keys = set(zip(wanted["STATEFP"], wanted["method"]))
    if len(wanted) != 10 or actual_keys != expected_keys:
        raise RuntimeError(
            "Step 34 must contain exactly ten five-state P/S 500 m rows; "
            f"rows={len(wanted)}, missing={sorted(expected_keys - actual_keys)}, "
            f"extra={sorted(actual_keys - expected_keys)}"
        )
    if not wanted["source_qc"].astype(str).str.strip().eq(PASS).all():
        raise RuntimeError("A five-state Step 34 P/S source_qc is not PASS")
    if "buffer_m" in wanted.columns:
        buffers = pd.to_numeric(wanted["buffer_m"], errors="coerce")
        if not buffers.eq(500).all():
            raise RuntimeError("A five-state Step 34 P/S row is not 500 m")

    numeric = [
        "nonwui_population",
        "intermix_population",
        "interface_population",
        "wui_population",
        "total_population",
        "wui_population_share_pct",
    ]
    for column in numeric:
        wanted[column] = pd.to_numeric(wanted[column], errors="coerce")
    if wanted[numeric].isna().any().any():
        raise RuntimeError("Step 34 five-state P/S rows have nonnumeric results")
    return wanted.reset_index(drop=True), path


def validate_texas_all_buffer_chain(
    tx_module,
    step19e_dir: str,
) -> dict:
    directory = tx_module.find_step19e_dir(step19e_dir or None)
    chain = tx_module.validate_step19e_chain(directory)
    radius = pd.read_csv(chain["step19e_radius_path"])
    require_columns(
        radius,
        {"radius_m", "candidate_path"},
        str(chain["step19e_radius_path"]),
    )
    radius["radius_m"] = pd.to_numeric(radius["radius_m"], errors="coerce")
    if radius["radius_m"].isna().any():
        raise RuntimeError("Step 19E radius audit contains nonnumeric radius")
    radius["radius_m"] = radius["radius_m"].astype(int)
    if radius["radius_m"].duplicated().any():
        raise RuntimeError("Step 19E radius audit contains duplicate radii")
    if set(radius["radius_m"]) != set(BUFFERS):
        raise RuntimeError(
            "Step 19E radius coverage is not exactly 100-1000 m: "
            f"{sorted(radius['radius_m'].tolist())}"
        )

    manifest = load_json(chain["step19d_output_manifest_path"])
    records = manifest.get("records", [])
    if not isinstance(records, list):
        raise RuntimeError("Step 19D output manifest records are invalid")
    by_radius: dict[int, dict] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        try:
            buffer_m = int(record.get("radius_m"))
        except (TypeError, ValueError):
            continue
        if buffer_m in by_radius:
            raise RuntimeError(
                f"Step 19D output manifest duplicates radius {buffer_m}"
            )
        by_radius[buffer_m] = record
    if set(by_radius) != set(BUFFERS):
        raise RuntimeError(
            "Step 19D output manifest is not exactly the ten expected radii"
        )

    raster_paths: dict[int, Path] = {}
    raster_hashes: dict[int, str] = {}
    for number, buffer_m in enumerate(BUFFERS, 1):
        audited_rows = radius[radius["radius_m"] == buffer_m]
        if len(audited_rows) != 1:
            raise RuntimeError(
                f"Step 19E must have one row for {buffer_m} m"
            )
        audited = require_file(
            Path(str(audited_rows.iloc[0]["candidate_path"])),
            f"Texas audited WUI-P {buffer_m} m raster",
        )
        record = by_radius[buffer_m]
        manifest_path = require_file(
            Path(str(record.get("path", ""))),
            f"Texas manifest WUI-P {buffer_m} m raster",
        )
        if audited != manifest_path:
            raise RuntimeError(
                f"Texas {buffer_m} m Step 19D/19E paths differ"
            )
        expected_size = int(record.get("size_bytes", -1))
        expected_hash = str(record.get("sha256", ""))
        if audited.stat().st_size != expected_size or len(expected_hash) != 64:
            raise RuntimeError(
                f"Texas {buffer_m} m manifest size/hash metadata is invalid"
            )
        if buffer_m == 500:
            current_hash = chain["candidate_wuip_500m_sha256"]
        else:
            current_hash = sha256_file(
                audited,
                label=f"Texas WUI-P {buffer_m} m [{number}/10]",
            )
        if current_hash != expected_hash:
            raise RuntimeError(
                f"Texas WUI-P {buffer_m} m SHA-256 changed after Step 19D/19E"
            )
        raster_paths[buffer_m] = audited
        raster_hashes[buffer_m] = current_hash
    return {
        **chain,
        "raster_paths": raster_paths,
        "raster_hashes": raster_hashes,
        "all_ten_rasters_hash_verified": True,
    }


def generic_paths(
    args: argparse.Namespace,
    statefp: str,
    method: str,
) -> tuple[Path, Path, dict[int, Path]]:
    state = STATE_BY_FIPS[statefp]
    token = state["file_token"]
    blocks = Path(args.blocks_dir) / f"tl_2022_{statefp}_tabblock20.gpkg"
    if method == "WUI-P":
        structures = Path(args.oa_dir) / f"{token}_addresses.gpkg"
        rasters = {
            buffer_m: (
                Path(args.wuip_dir)
                / token
                / f"WUI_P_{token}_r{buffer_m:04d}m.tif"
            )
            for buffer_m in BUFFERS
        }
    elif method == "WUI-S":
        structures = Path(args.mbf_dir) / f"{token}.gpkg"
        rasters = {
            buffer_m: (
                Path(args.wuis_dir)
                / token
                / f"WUI_S_{token}_r{buffer_m:04d}m.tif"
            )
            for buffer_m in BUFFERS
        }
    else:
        raise RuntimeError(f"Unsupported method: {method}")
    return blocks, structures, rasters


def build_jobs(
    args: argparse.Namespace,
    helper,
    tx_chain: dict,
    step34_path: Path,
) -> tuple[dict[tuple[str, str], dict], int]:
    jobs: dict[tuple[str, str], dict] = {}
    total_features = 0
    state_crs: dict[str, CRS] = {}
    for state_number, (statefp, state) in enumerate(
        STATE_BY_FIPS.items(), 1
    ):
        for method_number, method in enumerate(METHODS, 1):
            blocks, structures, rasters = generic_paths(
                args, statefp, method
            )
            if statefp == "48" and method == "WUI-P":
                structures = Path(tx_chain["candidate_gpkg"])
                rasters = dict(tx_chain["raster_paths"])

            blocks = require_file(blocks, f"{state['STUSPS']} Census blocks")
            structures = require_file(
                structures,
                f"{state['STUSPS']} {method} structure source",
            )
            rasters = {
                buffer_m: require_file(
                    path,
                    f"{state['STUSPS']} {method} {buffer_m} m WUI raster",
                )
                for buffer_m, path in rasters.items()
            }
            if set(rasters) != set(BUFFERS):
                raise RuntimeError(
                    f"{state['STUSPS']} {method}: incomplete raster buffers"
                )

            layer = helper.first_layer(structures)
            feature_count, source_crs_wkt = helper.layer_info(
                structures, layer
            )
            if feature_count <= 0 or not source_crs_wkt:
                raise RuntimeError(
                    f"{state['STUSPS']} {method}: invalid structure source"
                )

            raster_meta: dict[int, dict] = {}
            common_grid: tuple | None = None
            common_crs: CRS | None = None
            for buffer_m in BUFFERS:
                with rasterio.open(rasters[buffer_m]) as source:
                    if (
                        source.count != 1
                        or source.crs is None
                        or source.width <= 0
                        or source.height <= 0
                    ):
                        raise RuntimeError(
                            f"{state['STUSPS']} {method} {buffer_m} m "
                            "raster has invalid band, CRS or dimensions"
                        )
                    crs = normalized_crs(
                        helper,
                        source.crs,
                        f"{state['STUSPS']} {method} {buffer_m} m raster",
                    )
                    grid = (
                        int(source.width),
                        int(source.height),
                        tuple(source.transform),
                    )
                    if common_grid is None:
                        common_grid = grid
                        common_crs = crs
                    elif grid != common_grid or crs != common_crs:
                        raise RuntimeError(
                            f"{state['STUSPS']} {method}: ten WUI rasters "
                            "do not share one grid and CRS"
                        )
                    raster_meta[buffer_m] = {
                        "width": int(source.width),
                        "height": int(source.height),
                        "dtype": str(source.dtypes[0]),
                        "nodata": source.nodata,
                        "crs": crs.to_wkt(),
                        "transform": list(source.transform),
                    }
            assert common_crs is not None
            if statefp in state_crs and state_crs[statefp] != common_crs:
                raise RuntimeError(
                    f"{state['STUSPS']}: WUI-P and WUI-S CRS differ"
                )
            state_crs[statefp] = common_crs

            signature_paths = {
                "blocks": blocks,
                "structures": structures,
                "step34_population_table": step34_path,
                **{
                    f"wui_{buffer_m}m": rasters[buffer_m]
                    for buffer_m in BUFFERS
                },
            }
            if statefp == "48" and method == "WUI-P":
                signature_paths.update(
                    {
                        "step19e_summary": tx_chain[
                            "step19e_summary_path"
                        ],
                        "step19e_radius_audit": tx_chain[
                            "step19e_radius_path"
                        ],
                        "step19d_summary": tx_chain[
                            "step19d_summary_path"
                        ],
                        "step19d_output_manifest": tx_chain[
                            "step19d_output_manifest_path"
                        ],
                    }
                )
            signatures = {
                label: file_signature(path)
                for label, path in signature_paths.items()
            }
            jobs[(statefp, method)] = {
                "STATEFP": statefp,
                "STUSPS": state["STUSPS"],
                "state_name": state["state_name"],
                "official_population": state["official_population"],
                "method": method,
                "blocks": blocks,
                "structures": structures,
                "structures_layer": layer,
                "source_crs_wkt": source_crs_wkt,
                "feature_count": int(feature_count),
                "rasters": rasters,
                "raster_meta": raster_meta,
                "target_crs": common_crs,
                "input_signatures": signatures,
            }
            total_features += int(feature_count)
            index = (
                (state_number - 1) * len(METHODS) + method_number
            )
            print(
                f"[PREFLIGHT {index:2d}/{EXPECTED_JOBS}] "
                f"{state['STUSPS']} {method}: "
                f"features={feature_count:,}; rasters=10/10",
                flush=True,
            )
    return jobs, total_features


def build_state_context(helper, exact, statefp: str, jobs: dict) -> dict:
    state = STATE_BY_FIPS[statefp]
    state_jobs = [jobs[(statefp, method)] for method in METHODS]
    blocks = state_jobs[0]["blocks"]
    if any(job["blocks"] != blocks for job in state_jobs):
        raise RuntimeError(f"{state['STUSPS']}: P/S block sources differ")
    target_crs = state_jobs[0]["target_crs"]
    if any(job["target_crs"] != target_crs for job in state_jobs):
        raise RuntimeError(f"{state['STUSPS']}: P/S target CRS differs")

    bg, _, population_field = helper.load_block_group_population(
        blocks, helper.first_layer(blocks)
    )
    official = float(bg["POP20"].sum())
    if official != float(state["official_population"]):
        raise RuntimeError(
            f"{state['STUSPS']} block POP20={official:.0f}; "
            f"expected={state['official_population']:.0f}"
        )
    (
        tree,
        geometries,
        geoids12,
        _block_bg_codes,
        id_to_index,
        block_stats,
    ) = exact.load_block_index(helper, blocks, target_crs)
    return {
        "bg": bg,
        "population_field": population_field,
        "target_crs": target_crs,
        "tree": tree,
        "geometries": geometries,
        "geoids12": geoids12,
        "id_to_index": id_to_index,
        "block_stats": block_stats,
    }


def job_slug(statefp: str, stusps: str, method: str) -> str:
    return f"{statefp}_{stusps.lower()}_{method.lower().replace('-', '')}"


def prepare_job_directory(
    run_dir: Path,
    slug: str,
) -> tuple[Path, Path]:
    final_dir = run_dir / f"job_{slug}"
    partial_dir = run_dir / f"job_{slug}.partial"
    if final_dir.exists() and not final_dir.is_dir():
        raise RuntimeError(f"Checkpoint is not a directory: {final_dir}")
    if partial_dir.exists():
        abandoned = run_dir / f"job_{slug}.abandoned_{utc_stamp()}"
        os.replace(partial_dir, abandoned)
        print(f"[RESUME] Preserved incomplete job: {abandoned}", flush=True)
    return final_dir, partial_dir


def read_checkpoint(final_dir: Path, signatures: dict) -> dict | None:
    path = final_dir / "job_summary.json"
    if not final_dir.exists():
        return None
    if not final_dir.is_dir() or not path.is_file():
        raise RuntimeError(
            f"Existing checkpoint directory is incomplete or invalid: "
            f"{final_dir}"
        )
    payload = load_json(path)
    saved = payload.get("input_signatures")
    if not isinstance(saved, dict) or set(saved) != set(signatures):
        raise RuntimeError(f"Checkpoint signature set changed: {final_dir}")
    changed = [
        label
        for label in signatures
        if not same_signature(saved[label], signatures[label])
    ]
    if changed:
        raise RuntimeError(
            f"Checkpoint inputs changed for {final_dir}: {changed}"
        )
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != 10:
        raise RuntimeError(f"Checkpoint does not contain ten rows: {final_dir}")
    return payload


def global_progress(
    *,
    started: float,
    completed_features: int,
    current_features: int,
    total_features: int,
    completed_jobs: int,
    label: str,
) -> None:
    done = completed_features + current_features
    elapsed = max(time.monotonic() - started, 1e-9)
    rate = done / elapsed
    eta = (total_features - done) / rate if rate else 0.0
    pct = 100.0 * done / total_features if total_features else 100.0
    print(
        f"[GLOBAL] {label} | jobs={completed_jobs}/{EXPECTED_JOBS} complete | "
        f"features={done:,}/{total_features:,} ({pct:.2f}%) | "
        f"elapsed={hms(elapsed)} | ETA={hms(eta)}",
        flush=True,
    )


def run_job(
    *,
    helper,
    exact,
    job: dict,
    state_context: dict,
    step34_lookup: dict,
    partial_dir: Path,
    chunk_size: int,
    tile_size: int,
    progress_every: int,
    global_started: float,
    completed_features: int,
    total_features: int,
    completed_jobs: int,
) -> dict:
    partial_dir.mkdir()
    statefp = job["STATEFP"]
    stusps = job["STUSPS"]
    method = job["method"]
    feature_count = int(job["feature_count"])
    bg = state_context["bg"]
    bg_codes = bg["BG_CODE"].to_numpy(dtype=np.int64)
    total = np.zeros(len(bg), dtype=np.int64)
    class_counts = {
        buffer_m: {
            0: np.zeros(len(bg), dtype=np.int64),
            1: np.zeros(len(bg), dtype=np.int64),
            2: np.zeros(len(bg), dtype=np.int64),
        }
        for buffer_m in BUFFERS
    }
    common_stats = {
        **state_context["block_stats"],
        "structure_features_expected": feature_count,
        "points_processed": 0,
        "points_exact_unique_block_group": 0,
        "points_exact_ambiguous_block_group": 0,
        "points_outside_target_state_block_coverage": 0,
    }
    buffer_stats = {
        buffer_m: {
            "points_in_raster_bounds": 0,
            "target_state_eligible_points_outside_wui_raster": 0,
            "target_state_eligible_nodata_255": 0,
            "unexpected_wui_values": 0,
        }
        for buffer_m in BUFFERS
    }
    exception_examples: list[dict] = []
    max_exception_examples = 1000
    point_sequence = 0
    started = time.monotonic()
    next_report = progress_every

    source_crs = normalized_crs(
        helper,
        CRS.from_wkt(job["source_crs_wkt"]),
        f"{stusps} {method} structures",
    )
    target_crs = state_context["target_crs"]
    action = (
        "no coordinate transform"
        if source_crs == target_crs
        else f"transform {source_crs.to_string()} -> {target_crs.to_string()}"
    )
    print(f"[CRS] {stusps} {method}: {action}", flush=True)

    with ExitStack() as stack:
        raster_sources = {
            buffer_m: stack.enter_context(
                rasterio.open(job["rasters"][buffer_m])
            )
            for buffer_m in BUFFERS
        }
        for xs, ys in helper.iter_representative_points(
            job["structures"],
            job["structures_layer"],
            job["source_crs_wkt"],
            target_crs.to_wkt(),
            chunk_size,
        ):
            count = len(xs)
            dense = np.full(count, -1, dtype=np.int64)
            status = np.zeros(count, dtype=np.int8)
            # status: 0 outside target state, 1 unique BG, 2 ambiguous BG
            for local_index, (x, y) in enumerate(zip(xs, ys)):
                point_sequence += 1
                candidates = exact.exact_candidates(
                    state_context["tree"],
                    state_context["geometries"],
                    Point(float(x), float(y)),
                    state_context["id_to_index"],
                )
                candidate_geoids = sorted(
                    {
                        state_context["geoids12"][index]
                        for index in candidates
                    }
                )
                if len(candidate_geoids) == 1:
                    geoid12 = candidate_geoids[0]
                    if not geoid12.startswith(statefp):
                        raise RuntimeError(
                            f"{stusps} target block has foreign GEOID {geoid12}"
                        )
                    code = int(helper.bg_code_from_geoid12(geoid12))
                    position = int(np.searchsorted(bg_codes, code))
                    if (
                        position >= len(bg_codes)
                        or int(bg_codes[position]) != code
                    ):
                        raise RuntimeError(
                            f"{stusps} exact BG absent from POP20: {geoid12}"
                        )
                    dense[local_index] = position
                    status[local_index] = 1
                elif len(candidate_geoids) > 1:
                    status[local_index] = 2
                    if len(exception_examples) < max_exception_examples:
                        exception_examples.append(
                            {
                                "point_sequence": point_sequence,
                                "x": float(x),
                                "y": float(y),
                                "exception": "AMBIGUOUS_TARGET_STATE_BG",
                                "detail": ";".join(candidate_geoids),
                            }
                        )
                elif len(exception_examples) < max_exception_examples:
                    exception_examples.append(
                        {
                            "point_sequence": point_sequence,
                            "x": float(x),
                            "y": float(y),
                            "exception": "OUTSIDE_TARGET_STATE_BLOCK_COVERAGE",
                            "detail": "",
                        }
                    )

            unique = status == 1
            ambiguous = status == 2
            eligible = status > 0
            common_stats["points_processed"] += count
            common_stats["points_exact_unique_block_group"] += int(
                unique.sum()
            )
            common_stats["points_exact_ambiguous_block_group"] += int(
                ambiguous.sum()
            )
            common_stats[
                "points_outside_target_state_block_coverage"
            ] += int((status == 0).sum())
            total += np.bincount(
                dense[unique], minlength=len(total)
            ).astype(np.int64, copy=False)

            for buffer_m in BUFFERS:
                raw_values, inbounds = sample_wui_by_tiles(
                    raster_sources[buffer_m], xs, ys, tile_size
                )
                classes, unexpected = exact.classify_wui(
                    raw_values, inbounds
                )
                stats = buffer_stats[buffer_m]
                stats["points_in_raster_bounds"] += int(inbounds.sum())
                stats[
                    "target_state_eligible_points_outside_wui_raster"
                ] += int((eligible & ~inbounds).sum())
                stats["target_state_eligible_nodata_255"] += int(
                    (eligible & inbounds & (raw_values == 255)).sum()
                )
                stats["unexpected_wui_values"] += int(unexpected)
                for wui_class in (0, 1, 2):
                    selected = unique & (classes == wui_class)
                    class_counts[buffer_m][wui_class] += np.bincount(
                        dense[selected], minlength=len(total)
                    ).astype(np.int64, copy=False)

            if common_stats["points_processed"] >= next_report:
                elapsed = max(time.monotonic() - started, 1e-9)
                rate = common_stats["points_processed"] / elapsed
                eta = (
                    max(
                        feature_count - common_stats["points_processed"],
                        0,
                    )
                    / rate
                    if rate
                    else 0.0
                )
                pct = (
                    100.0
                    * common_stats["points_processed"]
                    / feature_count
                    if feature_count
                    else 100.0
                )
                print(
                    f"[{stusps} {method}] "
                    f"{common_stats['points_processed']:,}/{feature_count:,} "
                    f"({pct:.2f}%) | elapsed={hms(elapsed)} | "
                    f"ETA={hms(eta)}",
                    flush=True,
                )
                global_progress(
                    started=global_started,
                    completed_features=completed_features,
                    current_features=common_stats["points_processed"],
                    total_features=total_features,
                    completed_jobs=completed_jobs,
                    label=f"{stusps} {method}",
                )
                while next_report <= common_stats["points_processed"]:
                    next_report += progress_every

    common_stats["source_features_without_valid_representative_point"] = (
        feature_count - common_stats["points_processed"]
    )
    common_stats["representative_point_coverage_pct"] = (
        100.0 * common_stats["points_processed"] / feature_count
        if feature_count
        else 100.0
    )
    accounted = (
        common_stats["points_exact_unique_block_group"]
        + common_stats["points_exact_ambiguous_block_group"]
        + common_stats["points_outside_target_state_block_coverage"]
    )
    eligible_count = (
        common_stats["points_exact_unique_block_group"]
        + common_stats["points_exact_ambiguous_block_group"]
    )
    unique_count = common_stats["points_exact_unique_block_group"]
    unique_pct = (
        100.0 * unique_count / eligible_count
        if eligible_count
        else 100.0
    )
    common_gates = {
        "all_processed_points_accounted_for": (
            accounted == common_stats["points_processed"]
        ),
        "all_source_features_accounted_for": (
            common_stats["points_processed"]
            + common_stats[
                "source_features_without_valid_representative_point"
            ]
            == feature_count
        ),
        "representative_point_coverage_at_least_99pct": (
            common_stats["representative_point_coverage_pct"] >= 99.0
        ),
        "no_exact_target_state_bg_ambiguity": (
            common_stats["points_exact_ambiguous_block_group"] == 0
        ),
        "target_state_eligible_unique_bg_at_least_99pct": (
            unique_pct >= 99.0
        ),
        "official_population_matches_census": (
            float(bg["POP20"].sum())
            == float(job["official_population"])
        ),
    }

    expected500 = step34_lookup[(statefp, method)]
    rows: list[dict] = []
    details: list[pd.DataFrame] = []
    row_gate_payload: dict[str, dict] = {}
    for buffer_m in BUFFERS:
        c0 = class_counts[buffer_m][0]
        c1 = class_counts[buffer_m][1]
        c2 = class_counts[buffer_m][2]
        detail, population = helper.allocate_population(
            bg, total, c0, c1, c2
        )
        detail.insert(0, "buffer_m", buffer_m)
        detail.insert(0, "method", method)
        detail.insert(0, "state_name", job["state_name"])
        detail.insert(0, "STUSPS", stusps)
        detail.insert(0, "STATEFP", statefp)
        details.append(detail)

        gates = {
            **common_gates,
            "all_target_state_eligible_points_in_wui_raster_bounds": (
                buffer_stats[buffer_m][
                    "target_state_eligible_points_outside_wui_raster"
                ]
                == 0
            ),
            "no_unexpected_wui_values": (
                buffer_stats[buffer_m]["unexpected_wui_values"] == 0
            ),
            "structure_class_counts_close": bool(
                np.array_equal(total, c0 + c1 + c2)
            ),
            "population_allocation_conserved": (
                abs(float(population["allocation_residual"]))
                <= POP_TOLERANCE
            ),
        }
        if buffer_m == 500:
            gates.update(
                {
                    "step34_500m_nonwui_matches": close(
                        population["nonwui_population"],
                        expected500["nonwui_population"],
                        POP_TOLERANCE,
                    ),
                    "step34_500m_intermix_matches": close(
                        population["intermix_population"],
                        expected500["intermix_population"],
                        POP_TOLERANCE,
                    ),
                    "step34_500m_interface_matches": close(
                        population["interface_population"],
                        expected500["interface_population"],
                        POP_TOLERANCE,
                    ),
                    "step34_500m_wui_matches": close(
                        population["wui_population"],
                        expected500["wui_population"],
                        POP_TOLERANCE,
                    ),
                    "step34_500m_total_matches": close(
                        population["allocated_population"],
                        expected500["total_population"],
                        POP_TOLERANCE,
                    ),
                    "step34_500m_share_matches": close(
                        population["wui_population_share_pct"],
                        expected500["wui_population_share_pct"],
                        SHARE_TOLERANCE,
                    ),
                }
            )
        verdict = PASS if all(gates.values()) else "REVIEW"
        failed = ";".join(
            name for name, passed in gates.items() if not passed
        )
        row = {
            "STATEFP": statefp,
            "STUSPS": stusps,
            "state_name": job["state_name"],
            "method": method,
            "buffer_m": buffer_m,
            "official_population": int(job["official_population"]),
            "allocation_unit": "2020 Census block group",
            "population_field": state_context["population_field"],
            "blocks_source": str(job["blocks"]),
            "structures_source": str(job["structures"]),
            "wui_raster": str(job["rasters"][buffer_m]),
            **common_stats,
            "target_state_eligible_points": eligible_count,
            "target_state_eligible_unique_bg_matches": unique_count,
            "target_state_eligible_unique_bg_pct": unique_pct,
            **buffer_stats[buffer_m],
            "zero_structure_bg_count": population[
                "zero_structure_bg_count"
            ],
            "zero_structure_bg_population": population[
                "zero_structure_bg_population"
            ],
            "nonwui_population": population["nonwui_population"],
            "intermix_population": population["intermix_population"],
            "interface_population": population["interface_population"],
            "wui_population": population["wui_population"],
            "total_population": population["allocated_population"],
            "wui_population_share_pct": population[
                "wui_population_share_pct"
            ],
            "population_residual": population["allocation_residual"],
            "step34_500m_reference_checked": buffer_m == 500,
            "source_step": (
                "STEP35_EXACT_TARGET_STATE_ALL_BUFFER_RECOMPUTATION"
            ),
            "source_qc": verdict,
            "failed_gates": failed,
        }
        rows.append(row)
        row_gate_payload[str(buffer_m)] = gates

    detail_all = pd.concat(details, ignore_index=True)
    detail_all.to_csv(
        partial_dir / "exact_block_group_detail_all_buffers.csv.gz",
        index=False,
        compression="gzip",
        float_format="%.12f",
    )
    if exception_examples:
        pd.DataFrame(exception_examples).to_csv(
            partial_dir / "assignment_exception_examples.csv",
            index=False,
            float_format="%.9f",
        )
    atomic_write_csv(
        pd.DataFrame(rows),
        partial_dir / "job_population_rows_10.csv",
        float_format="%.12f",
    )
    job_verdict = (
        PASS if all(row["source_qc"] == PASS for row in rows) else "REVIEW"
    )
    payload = {
        "job_verdict": job_verdict,
        "STATEFP": statefp,
        "STUSPS": stusps,
        "method": method,
        "buffers": list(BUFFERS),
        "rows": rows,
        "row_gates": row_gate_payload,
        "input_signatures": job["input_signatures"],
        "method_rule": (
            "Within each 2020 Census block group, allocate POP20 in "
            "proportion to exact target-state address/building point counts "
            "by WUI class; zero-point block groups are Non-WUI."
        ),
        "step34_rule": (
            "The recomputed 500 m row must match the final passing Step 34 "
            "row within population tolerance 1e-6 and share tolerance 1e-8."
        ),
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_write_json(partial_dir / "job_summary.json", payload)
    return payload


def verify_inputs_unchanged(jobs: dict) -> tuple[bool, list[str]]:
    changed: list[str] = []
    seen: set[tuple[str, str]] = set()
    for job in jobs.values():
        for label, before in job["input_signatures"].items():
            key = (label, str(before["path"]))
            if key in seen:
                continue
            seen.add(key)
            path = Path(str(before["path"]))
            if not path.is_file():
                changed.append(f"{label}: missing {path}")
                continue
            after = file_signature(path)
            if not same_signature(before, after):
                changed.append(f"{label}: changed {path}")
    return not changed, changed


def old_population_comparison(
    final: pd.DataFrame,
    old_path: Path,
    output_path: Path,
) -> str:
    if not old_path.is_file():
        return f"NOT_AVAILABLE: {old_path}"
    try:
        old = pd.read_csv(old_path, dtype={"STATEFP": str})
        require_columns(
            old,
            {"STATEFP", "STUSPS", "method", "buffer_m"},
            str(old_path),
        )
        old = old.copy()
        old["STATEFP"] = normalize_fips(old["STATEFP"])
        old["buffer_m"] = pd.to_numeric(
            old["buffer_m"], errors="coerce"
        )
        old = old[
            old["STATEFP"].isin(STATE_BY_FIPS)
            & old["method"].isin(METHODS)
            & old["buffer_m"].isin(BUFFERS)
        ].copy()
        candidates = {
            "nonwui_population": (
                "nonwui_population",
                "NonWUI_Pop",
                "NonWUI_Pop_2020",
            ),
            "intermix_population": (
                "intermix_population",
                "Intermix_Pop",
                "Intermix_Pop_2020",
            ),
            "interface_population": (
                "interface_population",
                "Interface_Pop",
                "Interface_Pop_2020",
            ),
            "wui_population": (
                "wui_population",
                "WUI_Pop",
                "WUI_Pop_2020",
            ),
            "total_population": (
                "total_population",
                "Total_Pop",
                "Total_Pop_2020",
            ),
            "wui_population_share_pct": (
                "wui_population_share_pct",
                "WUI_Pop_Pct",
                "WUI_Population_Percent",
                "Pct_Pop_WUI",
            ),
        }
        selected: dict[str, str] = {}
        for canonical, options in candidates.items():
            match = next((name for name in options if name in old), None)
            if match:
                selected[canonical] = match
        if not selected:
            return "SCHEMA_NOT_RECOGNIZED"

        keys = ["STATEFP", "STUSPS", "method", "buffer_m"]
        old_keep = old[keys + list(selected.values())].copy()
        old_keep = old_keep.rename(
            columns={
                source: f"old_{canonical}"
                for canonical, source in selected.items()
            }
        )
        if old_keep.duplicated(keys).any():
            return "DUPLICATE_OLD_KEYS"
        current_keep = final[
            keys + list(selected.keys())
        ].copy()
        comparison = current_keep.merge(
            old_keep, on=keys, how="outer", indicator=True
        )
        for canonical in selected:
            comparison[f"delta_{canonical}"] = (
                pd.to_numeric(
                    comparison[canonical], errors="coerce"
                )
                - pd.to_numeric(
                    comparison[f"old_{canonical}"], errors="coerce"
                )
            )
        atomic_write_csv(
            comparison,
            output_path,
            float_format="%.12f",
        )
        matched = int(comparison["_merge"].eq("both").sum())
        return f"WRITTEN_MATCHED_ROWS_{matched}"
    except Exception as exc:
        return f"COMPARISON_ERROR: {type(exc).__name__}: {exc}"


def build_final_report(
    *,
    verdict: str,
    overall_gates: dict[str, bool],
    final: pd.DataFrame,
    row_qc: pd.DataFrame,
    step34_dir: Path,
    step19e_dir: Path,
    run_dir: Path,
    source_unchanged: bool,
    changed_inputs: list[str],
    old_comparison_status: str,
    elapsed: float,
) -> str:
    unique_count = len(
        final.drop_duplicates(["STATEFP", "method", "buffer_m"])
    )
    pass_rows = int(row_qc["source_qc"].eq(PASS).sum())
    pass_jobs = int(
        (
            row_qc.groupby(["STATEFP", "method"])["source_qc"]
            .apply(lambda values: len(values) == 10 and values.eq(PASS).all())
        ).sum()
    )
    closure_pass = int(
        row_qc["population_residual"].abs().le(POP_TOLERANCE).sum()
    )
    rows500 = row_qc[row_qc["buffer_m"] == 500]
    step34_pass = int(
        (
            rows500["source_qc"].eq(PASS)
            & rows500["step34_500m_reference_checked"].astype(bool)
        ).sum()
    )
    lines = [
        "STEP 35 - FIVE-STATE ALL-BUFFER WUI-P/WUI-S POPULATION",
        "=" * 116,
        f"VERDICT: {verdict}",
        (
            "FIVE_STATE_ALL_BUFFER_POPULATION_STATUS: "
            + (
                "FINAL_100_ROWS_INCLUDED_AND_VALIDATED"
                if verdict == FINAL_PASS_VERDICT
                else "REVIEW_REQUIRED"
            )
        ),
        f"FINAL_ROWS: {len(final)}/{EXPECTED_ROWS}",
        (
            "UNIQUE_STATE_METHOD_BUFFER_KEYS: "
            f"{unique_count}/{EXPECTED_ROWS}"
        ),
        f"PASS_ROW_QC: {pass_rows}/{EXPECTED_ROWS}",
        f"PASS_STATE_METHOD_JOBS: {pass_jobs}/{EXPECTED_JOBS}",
        f"POPULATION_CLOSURE_PASS: {closure_pass}/{EXPECTED_ROWS}",
        f"STEP34_500M_EXACT_MATCH_PASS: {step34_pass}/10",
        (
            "FAILED_OVERALL_GATES: "
            f"{sum(not passed for passed in overall_gates.values())}"
        ),
        "STATES: CA, CO, FL, PA, TX",
        "METHODS: WUI-P, WUI-S",
        "BUFFERS_M: 100,200,300,400,500,600,700,800,900,1000",
        "EXPECTED_DESIGN: 5 states x 2 methods x 10 buffers = 100 rows",
        "POPULATION_FIELD: POP20",
        "ALLOCATION_UNIT: 2020 Census block group",
        "TARGET_STATE_RULE: exact Census-block point-in-polygon",
        "ZERO_POINT_BLOCK_GROUP_POLICY: assign population to Non-WUI",
        "TEXAS_WUI_P_SOURCE: locked Step 19C candidate GPKG",
        "TEXAS_WUI_P_RASTERS: ten Step 19D/19E hash-verified candidates",
        f"SOURCE_STEP34_DIR: {step34_dir}",
        f"SOURCE_STEP19E_DIR: {step19e_dir}",
        f"SOURCE_DATA_MODIFIED: {'NO' if source_unchanged else 'YES'}",
        f"OLD_FAST_TABLE_COMPARISON: {old_comparison_status}",
        f"OUTPUT_DIR: {run_dir}",
        f"ELAPSED: {hms(elapsed)}",
        "",
        "INTERPRETATION",
        "-" * 116,
        (
            "This step validates the five-state 100-1000 m population "
            "sensitivity dataset only. It does not validate WUI area. "
            "WUI-Z has no buffer radius and is intentionally excluded."
        ),
        (
            "The historical fast-raster population table is comparison-only "
            "and cannot make this workflow pass."
        ),
    ]
    if changed_inputs:
        lines.extend(["", "CHANGED INPUTS", "-" * 116])
        lines.extend(f"- {item}" for item in changed_inputs)
    failed_overall = [
        name for name, passed in overall_gates.items() if not passed
    ]
    if failed_overall:
        lines.extend(["", "FAILED OVERALL GATES", "-" * 116])
        lines.extend(f"- {name}" for name in failed_overall)
    failed = row_qc[row_qc["source_qc"] != PASS]
    if not failed.empty:
        lines.extend(["", "NONPASS ROWS", "-" * 116])
        for row in failed.itertuples(index=False):
            lines.append(
                f"- {row.STUSPS} {row.method} {row.buffer_m}m: "
                f"{row.failed_gates}"
            )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "STEP 35: recompute and audit CA/CO/FL/PA/TX WUI-P/WUI-S "
            "population at 100-1000 m."
        )
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument(
        "--step34-dir",
        default="",
        help=(
            "Passing Step 34 directory. Default: newest passing "
            "step34_national_147_population_* directory."
        ),
    )
    parser.add_argument(
        "--step19e-dir",
        default="",
        help=(
            "Passing Texas Step 19E directory. Default: newest passing "
            "directory found by Step 19F."
        ),
    )
    parser.add_argument("--blocks-dir", default=str(DEFAULT_BLOCKS_DIR))
    parser.add_argument("--oa-dir", default=str(DEFAULT_OA_DIR))
    parser.add_argument("--mbf-dir", default=str(DEFAULT_MBF_DIR))
    parser.add_argument("--wuip-dir", default=str(DEFAULT_WUIP_DIR))
    parser.add_argument("--wuis-dir", default=str(DEFAULT_WUIS_DIR))
    parser.add_argument(
        "--old-paper-csv",
        default=str(DEFAULT_OLD_PAPER_CSV),
        help="Historical five-state all-buffer CSV; comparison only.",
    )
    parser.add_argument(
        "--run-dir",
        default="",
        help=(
            "Output/checkpoint directory. Default: a new timestamped "
            "directory under project-root. Reuse the same path to resume."
        ),
    )
    parser.add_argument("--chunk-size", type=int, default=200_000)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--progress-every", type=int, default=500_000)
    args = parser.parse_args()
    if args.chunk_size <= 0 or args.tile_size <= 0:
        parser.error("--chunk-size and --tile-size must be positive")
    if args.progress_every <= 0:
        parser.error("--progress-every must be positive")
    return args


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    project_root = Path(args.project_root).expanduser().resolve()
    script_dir = Path(__file__).resolve().parent
    helper = load_module(script_dir / HELPER_NAME, "step35_helper")
    exact = load_module(script_dir / EXACT_NAME, "step35_exact")
    tx_module = load_module(script_dir / TX_NAME, "step35_tx")

    step34_dir = find_step34_dir(project_root, args.step34_dir)
    step34, step34_path = load_step34_five_state_500m(step34_dir)
    step34_lookup = {
        (row.STATEFP, row.method): row._asdict()
        for row in step34.itertuples(index=False)
    }
    run_dir = (
        Path(args.run_dir).expanduser().resolve()
        if args.run_dir
        else (
            project_root
            / f"step35_five_state_all_buffer_population_{utc_stamp()}"
        )
    )
    if str(run_dir).startswith(portable_path("researchdrive")):
        raise RuntimeError("Step 35 output cannot be written to ResearchDrive")
    run_dir.mkdir(parents=True, exist_ok=True)
    failure_path = run_dir / "STEP35_RUNTIME_FAILURE.json"

    print("=" * 116)
    print(
        "STEP 35: FIVE-STATE ALL-BUFFER POPULATION RECOMPUTATION AND QC"
    )
    print(f"Step 34 : {step34_dir}")
    print(f"Output  : {run_dir}")
    print("States  : CA, CO, FL, PA, TX")
    print("Methods : WUI-P, WUI-S")
    print("Buffers : 100-1000 m by 100 m")
    print("Rows    : 5 x 2 x 10 = 100")
    print("Writes  : new Step 35 outputs only; all sources read-only")
    print("=" * 116)

    try:
        print("[1/8] Validating final Step 34 five-state 500 m references")
        print("[2/8] Validating and hashing Texas Step 19C/19D/19E chain")
        tx_chain = validate_texas_all_buffer_chain(
            tx_module, args.step19e_dir
        )
        print("[3/8] Preflighting ten state-method jobs and 100 rasters")
        jobs, total_features = build_jobs(
            args, helper, tx_chain, step34_path
        )
        print(
            f"[PREFLIGHT] Total representative features across ten jobs: "
            f"{total_features:,}",
            flush=True,
        )

        print("[4/8] Recomputing population with resumable checkpoints")
        payloads: list[dict] = []
        completed_features = 0
        completed_jobs = 0
        for statefp, state in STATE_BY_FIPS.items():
            pending = False
            for method in METHODS:
                slug = job_slug(statefp, state["STUSPS"], method)
                final_dir, _partial_dir = prepare_job_directory(
                    run_dir, slug
                )
                if read_checkpoint(
                    final_dir, jobs[(statefp, method)]["input_signatures"]
                ) is None:
                    pending = True
                    break
            state_context = (
                build_state_context(helper, exact, statefp, jobs)
                if pending
                else None
            )

            for method in METHODS:
                job = jobs[(statefp, method)]
                slug = job_slug(statefp, state["STUSPS"], method)
                final_dir, partial_dir = prepare_job_directory(
                    run_dir, slug
                )
                payload = read_checkpoint(
                    final_dir, job["input_signatures"]
                )
                if payload is not None:
                    print(
                        f"[RESUME] {state['STUSPS']} {method}: "
                        f"{payload['job_verdict']}",
                        flush=True,
                    )
                else:
                    assert state_context is not None
                    print(
                        f"[JOB {completed_jobs + 1}/{EXPECTED_JOBS}] "
                        f"{state['STUSPS']} {method} starting",
                        flush=True,
                    )
                    payload = run_job(
                        helper=helper,
                        exact=exact,
                        job=job,
                        state_context=state_context,
                        step34_lookup=step34_lookup,
                        partial_dir=partial_dir,
                        chunk_size=args.chunk_size,
                        tile_size=args.tile_size,
                        progress_every=args.progress_every,
                        global_started=started,
                        completed_features=completed_features,
                        total_features=total_features,
                        completed_jobs=completed_jobs,
                    )
                    os.replace(partial_dir, final_dir)
                    print(
                        f"[JOB {completed_jobs + 1}/{EXPECTED_JOBS}] "
                        f"{state['STUSPS']} {method}: "
                        f"{payload['job_verdict']}",
                        flush=True,
                    )
                payloads.append(payload)
                completed_features += int(job["feature_count"])
                completed_jobs += 1
                global_progress(
                    started=started,
                    completed_features=completed_features,
                    current_features=0,
                    total_features=total_features,
                    completed_jobs=completed_jobs,
                    label=f"{state['STUSPS']} {method} complete",
                )

        print("[5/8] Merging and auditing all 100 rows")
        rows = [
            row
            for payload in payloads
            for row in payload.get("rows", [])
        ]
        final = pd.DataFrame(rows)
        if final.empty:
            raise RuntimeError("No Step 35 population rows were produced")
        final["STATEFP"] = normalize_fips(final["STATEFP"])
        final["buffer_m"] = pd.to_numeric(
            final["buffer_m"], errors="coerce"
        ).astype("Int64")
        final["_state_order"] = final["STATEFP"].map(STATE_ORDER)
        final["_method_order"] = final["method"].map(METHOD_ORDER)
        final = (
            final.sort_values(
                ["_state_order", "_method_order", "buffer_m"]
            )
            .drop(columns=["_state_order", "_method_order"])
            .reset_index(drop=True)
        )
        expected_keys = {
            (statefp, method, buffer_m)
            for statefp in STATE_BY_FIPS
            for method in METHODS
            for buffer_m in BUFFERS
        }
        actual_keys = set(
            zip(final["STATEFP"], final["method"], final["buffer_m"])
        )
        complete = len(final) == EXPECTED_ROWS and actual_keys == expected_keys
        unique = not final.duplicated(
            ["STATEFP", "method", "buffer_m"]
        ).any()
        rows_pass = final["source_qc"].eq(PASS).all()

        numeric = [
            "nonwui_population",
            "intermix_population",
            "interface_population",
            "wui_population",
            "total_population",
            "wui_population_share_pct",
            "population_residual",
        ]
        for column in numeric:
            final[column] = pd.to_numeric(final[column], errors="coerce")
        numeric_valid = not final[numeric].isna().any().any()
        closure = bool(
            numeric_valid
            and final["population_residual"]
            .abs()
            .le(POP_TOLERANCE)
            .all()
        )
        source_unchanged, changed_inputs = verify_inputs_unchanged(jobs)

        output_table = (
            run_dir / "step35_five_state_all_buffer_population_long_100.csv"
        )
        row_qc_path = (
            run_dir
            / "step35_five_state_all_buffer_population_row_qc_100.csv"
        )
        paper_path = (
            run_dir
            / "step35_five_state_all_buffer_population_paper_values_100.csv"
        )
        atomic_write_csv(final, output_table, float_format="%.12f")
        row_qc_columns = [
            "STATEFP",
            "STUSPS",
            "state_name",
            "method",
            "buffer_m",
            "official_population",
            "nonwui_population",
            "intermix_population",
            "interface_population",
            "wui_population",
            "total_population",
            "wui_population_share_pct",
            "population_residual",
            "step34_500m_reference_checked",
            "source_qc",
            "failed_gates",
        ]
        row_qc = final[row_qc_columns].copy()
        atomic_write_csv(row_qc, row_qc_path, float_format="%.12f")
        paper = final[
            [
                "STATEFP",
                "STUSPS",
                "state_name",
                "method",
                "buffer_m",
                "nonwui_population",
                "intermix_population",
                "interface_population",
                "wui_population",
                "total_population",
                "wui_population_share_pct",
            ]
        ].copy()
        for column in (
            "nonwui_population",
            "intermix_population",
            "interface_population",
            "wui_population",
            "total_population",
        ):
            paper[column] = paper[column].round().astype("Int64")
        paper["wui_population_share_pct"] = paper[
            "wui_population_share_pct"
        ].round(2)
        atomic_write_csv(paper, paper_path, float_format="%.2f")

        print("[6/8] Writing comparison-only old-versus-final table")
        comparison_path = (
            run_dir
            / "step35_old_fast_vs_final_population_difference_100.csv"
        )
        old_status = old_population_comparison(
            final,
            Path(args.old_paper_csv).expanduser(),
            comparison_path,
        )

        overall_gates = {
            "exact_100_rows": complete,
            "unique_100_keys": unique,
            "all_100_rows_pass": rows_pass,
            "all_100_rows_numeric": numeric_valid,
            "all_100_rows_population_conserved": closure,
            "all_inputs_unchanged": source_unchanged,
            "ten_texas_wuip_rasters_hash_verified": bool(
                tx_chain["all_ten_rasters_hash_verified"]
            ),
            "ten_state_method_jobs_present": len(payloads) == EXPECTED_JOBS,
            "all_state_method_jobs_pass": all(
                payload.get("job_verdict") == PASS
                for payload in payloads
            ),
        }
        verdict = (
            FINAL_PASS_VERDICT
            if all(overall_gates.values())
            else FINAL_REVIEW_VERDICT
        )

        print("[7/8] Writing final QC and provenance manifest")
        report = build_final_report(
            verdict=verdict,
            overall_gates=overall_gates,
            final=final,
            row_qc=row_qc,
            step34_dir=step34_dir,
            step19e_dir=tx_chain["step19e_dir"],
            run_dir=run_dir,
            source_unchanged=source_unchanged,
            changed_inputs=changed_inputs,
            old_comparison_status=old_status,
            elapsed=time.monotonic() - started,
        )
        qc_path = run_dir / "STEP35_FINAL_QC.txt"
        atomic_write_text(qc_path, report)
        manifest_path = run_dir / "step35_manifest.json"
        output_files = [output_table, row_qc_path, paper_path, qc_path]
        if comparison_path.is_file():
            output_files.append(comparison_path)
        manifest = {
            "step": 35,
            "created_utc": utc_now(),
            "verdict": verdict,
            "overall_gates": overall_gates,
            "scope": {
                "states": list(STATE_BY_FIPS),
                "methods": list(METHODS),
                "buffers_m": list(BUFFERS),
                "expected_rows": EXPECTED_ROWS,
            },
            "population_method": (
                "Exact target-state Census-block point assignment followed "
                "by block-group POP20 allocation proportional to WUI-class "
                "point counts; zero-point block groups assigned Non-WUI."
            ),
            "source_step34_dir": str(step34_dir),
            "source_step34_table": str(step34_path),
            "source_step19e_dir": str(tx_chain["step19e_dir"]),
            "texas_candidate_gpkg": str(tx_chain["candidate_gpkg"]),
            "texas_wuip_raster_hashes": {
                str(key): value
                for key, value in tx_chain["raster_hashes"].items()
            },
            "source_data_modified": not source_unchanged,
            "changed_inputs": changed_inputs,
            "old_fast_table_comparison_status": old_status,
            "outputs": {
                path.name: {
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in output_files
            },
            "job_checkpoints": [
                {
                    "STATEFP": payload["STATEFP"],
                    "STUSPS": payload["STUSPS"],
                    "method": payload["method"],
                    "job_verdict": payload["job_verdict"],
                }
                for payload in payloads
            ],
        }
        atomic_write_json(manifest_path, manifest)

        print("[8/8] Final readback complete")
        readback = pd.read_csv(output_table, dtype={"STATEFP": str})
        if len(readback) != EXPECTED_ROWS:
            raise RuntimeError("Final output readback is not 100 rows")
        if sha256_file(output_table) != manifest["outputs"][
            output_table.name
        ]["sha256"]:
            raise RuntimeError("Final output SHA-256 readback mismatch")

        pass_rows = int(row_qc["source_qc"].eq(PASS).sum())
        nonpass = (
            EXPECTED_ROWS
            - pass_rows
            + sum(not passed for passed in overall_gates.values())
        )
        print()
        print("=" * 116)
        print(f"VERDICT: {verdict}")
        print(
            "FIVE_STATE_ALL_BUFFER_POPULATION_STATUS: "
            + (
                "FINAL_100_ROWS_INCLUDED_AND_VALIDATED"
                if verdict == FINAL_PASS_VERDICT
                else "REVIEW_REQUIRED"
            )
        )
        print(f"FINAL_ROWS: {len(final)}/{EXPECTED_ROWS}")
        print(
            "UNIQUE_STATE_METHOD_BUFFER_KEYS: "
            f"{len(actual_keys)}/{EXPECTED_ROWS}"
        )
        print(f"PASS_ROW_QC: {pass_rows}/{EXPECTED_ROWS}")
        print(f"NONPASS_ROWS_OR_GATES: {nonpass}")
        print(f"Final QC: {qc_path}")
        print(f"Population table: {output_table}")
        print(f"Run directory: {run_dir}")
        print("=" * 116)
        return 0 if verdict == FINAL_PASS_VERDICT else 2
    except Exception as exc:
        failure = {
            "created_utc": utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "run_dir": str(run_dir),
        }
        atomic_write_json(failure_path, failure)
        print()
        print("=" * 116, file=sys.stderr)
        print(
            f"STEP 35 RUNTIME FAILURE: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        print(f"Failure record: {failure_path}", file=sys.stderr)
        print(
            "Completed job checkpoints were preserved. Correct the reported "
            "problem and rerun with the same --run-dir to resume.",
            file=sys.stderr,
        )
        print("=" * 116, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())