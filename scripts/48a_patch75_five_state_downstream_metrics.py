#!/usr/bin/env python3
# Copyright (C) 2026 WUI Mapping Project Authors
# SPDX-License-Identifier: GPL-3.0-only
# Repository version: filesystem paths are loaded through repo_config.py.
# Copy config/paths.example.json to config/paths.json before a new run.
# Raw input data and large intermediate files are stored outside GitHub.

"""STEP48A: strict-patch (>75%) five-state downstream metric rebuild.

This is an isolated candidate rebuild.  It never modifies Steps 43--47.
WUI-P population starts from the accepted Step44 P2 block-group allocation;
WUI-S population starts from the accepted Step35 allocation.  Only verified
class 1/2 -> class 0 transitions are applied.  WUI-Z population is joined
directly from POP20 at the Census-block level.
"""

from __future__ import annotations

from repo_config import portable_path

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fiona
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely


ROOT = Path(portable_path("project"))
DRIVE = Path(portable_path("data"))
STEP35 = ROOT / "step35_five_state_all_buffer_population_20260724T183418Z"
STEP44 = ROOT / "step44_wuip_p2_downstream_metrics_20260727T224419Z"
STEP43 = ROOT / "step43_wuip_p2_formal_rebuild_20260727T180822Z"
STEP47_CO500 = ROOT / "step47_patch75_silvis_colorado_canary_20260729T034216Z"
STEP47_COOTHER = ROOT / "step47b_patch75_colorado_remaining_radii_20260729T041412Z"
STEP47_COZ = ROOT / "step47c_patch75_colorado_wuiz_20260729T144953Z"
STEP47_FOUR = ROOT / "step47d_patch75_four_state_psz_20260729T150314Z"
COUNTY_CACHE = Path(portable_path("legacy", "WUI_tables_compare/_cache_table5_all49_500m"))
COUNTY_GPKG = DRIVE / "MBF+NLCD_2022US/Inputs/Processed_GPKG/tl_2022_us_county.gpkg"
BLOCKS_DIR = DRIVE / "MBF+NLCD_2022US/Inputs/Processed_GPKG"
HELPER_PATH = ROOT / "scripts/08_recompute_sample4_ps_population.py"
EXACT_PATH = ROOT / "scripts/25_audit_vermont_wuis_exact_point_in_polygon.py"
METRIC_PATH = ROOT / "scripts/44_wuip_p2_downstream_metrics.py"
OLD35_PATH = ROOT / "scripts/35_recompute_five_state_all_buffer_population.py"

RADII = tuple(range(100, 1001, 100))
METHODS = ("WUI-P", "WUI-S")
STATE = {
    "CA": ("06", "California", 39_538_223),
    "CO": ("08", "Colorado", 5_773_714),
    "FL": ("12", "Florida", 21_538_187),
    "PA": ("42", "Pennsylvania", 13_002_700),
    "TX": ("48", "Texas", 29_145_505),
}


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256(path: Path, chunk: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as src:
        while True:
            data = src.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def atomic_csv(path: Path, frame: pd.DataFrame, float_format: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    frame.to_csv(tmp, index=False, float_format=float_format)
    os.replace(tmp, path)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def progress(phase: str, state: str, method: str, done: int, total: int,
             started: float) -> None:
    elapsed = max(time.monotonic() - started, 1e-9)
    rate = done / elapsed
    eta = (total - done) / rate if rate and total else 0.0
    pct = 100.0 * done / total if total else 100.0
    print(
        f"[{phase}] state={state} method={method} {done:,}/{total:,} "
        f"({pct:.2f}%) elapsed={elapsed/60:.1f}m ETA={eta/60:.1f}m",
        flush=True,
    )


def strict_raster(state: str, method: str, radius: int) -> Path:
    token = "P" if method == "WUI-P" else "S"
    if state == "CO":
        if radius == 500:
            return STEP47_CO500 / f"WUI_{token}_CO_r0500m_SILVIS_GT75_CANARY.tif"
        return (
            STEP47_COOTHER / f"rasters/{method}/"
            f"WUI_{token}_CO_r{radius:04d}m_SILVIS_GT75.tif"
        )
    return (
        STEP47_FOUR / state / f"rasters/{method}/"
        f"WUI_{token}_{state}_r{radius:04d}m_SILVIS_GT75.tif"
    )


def strict_z_raster(state: str) -> Path:
    if state == "CO":
        return STEP47_COZ / "WUI_Z_CO_SILVIS_GT75_class.tif"
    return STEP47_FOUR / state / f"WUI_Z_{state}_SILVIS_GT75_class.tif"


def strict_z_vector(state: str) -> Path:
    name = STATE[state][1]
    if state == "CO":
        return STEP47_COZ / f"WUI_Z_Paper_{name}_SILVIS_GT75.gpkg"
    return STEP47_FOUR / state / f"WUI_Z_Paper_{name}_SILVIS_GT75.gpkg"


def baseline_raster(state: str, method: str, radius: int) -> Path:
    if method == "WUI-P":
        group = "rasters_500m" if radius == 500 else "rasters_sensitivity"
        return STEP43 / group / state / f"WUI_P_P2_{state}_r{radius:04d}m.tif"
    fips = STATE[state][0]
    rows = pd.read_csv(
        STEP35 / f"job_{fips}_{state.lower()}_wuis/job_population_rows_10.csv"
    )
    return Path(rows.loc[rows.buffer_m.eq(radius), "wui_raster"].iloc[0])


def baseline_detail(state: str, method: str) -> pd.DataFrame:
    fips = STATE[state][0]
    if method == "WUI-P":
        path = STEP44 / f"population/{state}_block_group_detail.csv.gz"
    else:
        path = (
            STEP35
            / f"job_{fips}_{state.lower()}_wuis/exact_block_group_detail_all_buffers.csv.gz"
        )
    # Step44 files retain .gz for lineage but are plain CSV.
    try:
        frame = pd.read_csv(path, dtype={"GEOID12": str})
    except Exception:
        frame = pd.read_csv(path, compression=None, dtype={"GEOID12": str})
    frame["GEOID12"] = frame["GEOID12"].astype(str).str.zfill(12)
    return frame


def structure_source(state: str, method: str) -> Path:
    fips = STATE[state][0]
    if method == "WUI-P":
        cp = json.loads((STEP44 / f"checkpoints/population_{state}.json").read_text())
        return Path(cp["address_path"])
    rows = pd.read_csv(
        STEP35 / f"job_{fips}_{state.lower()}_wuis/job_population_rows_10.csv"
    )
    return Path(rows.structures_source.iloc[0])


def changed_cells(
    new_path: Path, old_path: Path
) -> tuple[dict[tuple[int, int], np.ndarray], dict[str, int]]:
    pairs = [(old, new) for old in (0, 1, 2) for new in (0, 1, 2) if old != new]
    found: dict[tuple[int, int], list[np.ndarray]] = {pair: [] for pair in pairs}
    stats = {f"changed_{old}_to_{new}": 0 for old, new in pairs}
    stats["domain_difference_pixels"] = 0
    with rasterio.open(new_path) as new, rasterio.open(old_path) as old:
        # Step47 outputs preserve the historical LOCAL_CS label for the same
        # EPSG:5070 numerical grid.  Compare the grid explicitly; the accepted
        # baseline supplies the normalized sampling CRS.
        if (
            new.width != old.width
            or new.height != old.height
            or not new.transform.almost_equals(old.transform)
        ):
            raise RuntimeError(f"Grid mismatch: {new_path} vs {old_path}")
        for _, window in new.block_windows(1):
            n = new.read(1, window=window)
            o = old.read(1, window=window)
            valid = np.isin(n, [0, 1, 2]) & np.isin(o, [0, 1, 2])
            stats["domain_difference_pixels"] += int(
                (np.isin(n, [0, 1, 2]) ^ np.isin(o, [0, 1, 2])).sum()
            )
            for old_cls, new_cls in pairs:
                mask = valid & (o == old_cls) & (n == new_cls)
                stats[f"changed_{old_cls}_to_{new_cls}"] += int(mask.sum())
                rr, cc = np.nonzero(mask)
                if len(rr):
                    found[(old_cls, new_cls)].append(
                        (rr + int(window.row_off)) * new.width
                        + cc + int(window.col_off)
                    )
    return {
        pair: (
            np.sort(np.concatenate(parts).astype(np.int64))
            if parts else np.empty(0, np.int64)
        )
        for pair, parts in found.items()
    }, stats


def changed_cells_sparse(
    new_path: Path,
    old_path: Path,
    rows: np.ndarray,
    cols: np.ndarray,
    old35,
    chunk_size: int = 1_000_000,
) -> tuple[dict[tuple[int, int], np.ndarray], dict[str, int]]:
    """Audit transitions only where the frozen P2 address count is non-zero."""
    pairs = [(old, new) for old in (0, 1, 2) for new in (0, 1, 2) if old != new]
    found: dict[tuple[int, int], list[np.ndarray]] = {pair: [] for pair in pairs}
    stats = {f"changed_{old}_to_{new}": 0 for old, new in pairs}
    stats["domain_difference_pixels"] = 0
    stats["transition_audit_cells"] = int(len(rows))
    with rasterio.open(new_path) as new, rasterio.open(old_path) as old:
        if (
            new.width != old.width
            or new.height != old.height
            or not new.transform.almost_equals(old.transform)
        ):
            raise RuntimeError(f"Grid mismatch: {new_path} vs {old_path}")
        transform = new.transform
        for start in range(0, len(rows), chunk_size):
            rr = rows[start:start + chunk_size].astype(np.int64, copy=False)
            cc = cols[start:start + chunk_size].astype(np.int64, copy=False)
            xs = transform.c + transform.a * (cc.astype(float) + 0.5)
            ys = transform.f + transform.e * (rr.astype(float) + 0.5)
            nv, nin = old35.sample_wui_by_tiles(new, xs, ys, 512)
            ov, oin = old35.sample_wui_by_tiles(old, xs, ys, 512)
            nvalid = nin & np.isin(nv, [0, 1, 2])
            ovalid = oin & np.isin(ov, [0, 1, 2])
            stats["domain_difference_pixels"] += int((nvalid ^ ovalid).sum())
            valid = nvalid & ovalid
            ids = rr * new.width + cc
            for old_cls, new_cls in pairs:
                mask = valid & (ov == old_cls) & (nv == new_cls)
                stats[f"changed_{old_cls}_to_{new_cls}"] += int(mask.sum())
                if mask.any():
                    found[(old_cls, new_cls)].append(ids[mask])
    return {
        pair: (
            np.sort(np.concatenate(parts).astype(np.int64))
            if parts else np.empty(0, np.int64)
        )
        for pair, parts in found.items()
    }, stats


def in_sorted(values: np.ndarray, sorted_values: np.ndarray) -> np.ndarray:
    if not len(sorted_values):
        return np.zeros(len(values), dtype=bool)
    pos = np.searchsorted(sorted_values, values)
    ok = pos < len(sorted_values)
    result = np.zeros(len(values), dtype=bool)
    result[ok] = sorted_values[pos[ok]] == values[ok]
    return result


def vector_exact_bg(tree, block_codes: np.ndarray, xs: np.ndarray, ys: np.ndarray,
                    bg_codes: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    dense = np.full(len(xs), -1, dtype=np.int64)
    if not len(xs):
        return dense, {"unique": 0, "ambiguous": 0, "outside": 0}
    pairs = tree.query(shapely.points(xs, ys), predicate="covered_by")
    if pairs.shape[1] == 0:
        return dense, {"unique": 0, "ambiguous": 0, "outside": len(xs)}
    pi = pairs[0].astype(np.int64)
    codes = block_codes[pairs[1].astype(np.int64)]
    order = np.lexsort((codes, pi))
    pi, codes = pi[order], codes[order]
    first = np.r_[True, (pi[1:] != pi[:-1]) | (codes[1:] != codes[:-1])]
    up, uc = pi[first], codes[first]
    starts = np.r_[0, np.flatnonzero(up[1:] != up[:-1]) + 1]
    ends = np.r_[starts[1:], len(up)]
    one = ends - starts == 1
    points, unique_codes = up[starts[one]], uc[starts[one]]
    pos = np.searchsorted(bg_codes, unique_codes)
    valid = pos < len(bg_codes)
    valid[valid] &= bg_codes[pos[valid]] == unique_codes[valid]
    if not valid.all():
        raise RuntimeError("Exact target BG absent from baseline detail")
    dense[points] = pos
    return dense, {
        "unique": int(one.sum()),
        "ambiguous": int((~one).sum()),
        "outside": int(len(xs) - len(np.unique(up))),
    }


def allocate(bg: pd.DataFrame, total: np.ndarray, c0: np.ndarray,
             c1: np.ndarray, c2: np.ndarray) -> tuple[pd.DataFrame, dict[str, float]]:
    if (c0 < 0).any() or (c1 < 0).any() or (c2 < 0).any():
        raise RuntimeError("Negative class point count after strict transition")
    if not np.array_equal(total, c0 + c1 + c2):
        raise RuntimeError("Block-group point-count closure failed")
    pop = bg.POP20.to_numpy(float)
    has = total > 0
    non = pop * np.where(has, c0 / np.maximum(total, 1), 1.0)
    intermix = pop * np.where(has, c1 / np.maximum(total, 1), 0.0)
    interface = pop * np.where(has, c2 / np.maximum(total, 1), 0.0)
    residual = float(pop.sum() - non.sum() - intermix.sum() - interface.sum())
    non[0] += residual
    detail = bg.copy()
    detail["Structure_Count"] = total
    detail["NonWUI_Structure_Count"] = c0
    detail["Intermix_Structure_Count"] = c1
    detail["Interface_Structure_Count"] = c2
    detail["NonWUI_Pop"] = non
    detail["Intermix_Pop"] = intermix
    detail["Interface_Pop"] = interface
    summary = {
        "nonwui_population": float(non.sum()),
        "intermix_population": float(intermix.sum()),
        "interface_population": float(interface.sum()),
        "wui_population": float(intermix.sum() + interface.sum()),
        "total_population": float(non.sum() + intermix.sum() + interface.sum()),
    }
    summary["population_closure_error"] = summary["total_population"] - float(pop.sum())
    summary["wui_population_share_pct"] = (
        100.0 * summary["wui_population"] / summary["total_population"]
    )
    return detail, summary


def population_ps(
    out: Path, state: str, method: str, helper, exact, old35
) -> list[dict[str, Any]]:
    checkpoint = out / f"checkpoints/population_{state}_{method.replace('-', '')}.json"
    if checkpoint.is_file():
        saved = json.loads(checkpoint.read_text())
        if saved.get("status") == "PASS":
            return saved["rows"]
    started = time.monotonic()
    old = baseline_detail(state, method)
    src = structure_source(state, method)
    strict_changes: dict[int, dict[tuple[int, int], np.ndarray]] = {}
    transition_stats: dict[int, dict[str, int]] = {}
    with rasterio.open(strict_raster(state, method, 500)) as ref, rasterio.open(
        baseline_raster(state, method, 500)
    ) as accepted_ref:
        transform, width, height = ref.transform, ref.width, ref.height
        crs = accepted_ref.crs
    sparse = np.load(STEP43 / f"intermediate/{state}_p2_sparse_cells.npz")
    sparse_rows = sparse["rows"].astype(np.int32)
    sparse_cols = sparse["cols"].astype(np.int32)
    for radius in RADII:
        changes, stats = changed_cells_sparse(
            strict_raster(state, method, radius),
            baseline_raster(state, method, radius),
            sparse_rows,
            sparse_cols,
            old35,
        )
        if stats["domain_difference_pixels"] != 0:
            raise RuntimeError(
                f"Valid-domain mismatch: {state}/{method}/{radius}; {stats}"
            )
        strict_changes[radius] = changes
        transition_stats[radius] = stats

    layer = helper.first_layer(src)
    feature_count, source_crs = helper.layer_info(src, layer)
    selected_x: list[np.ndarray] = []
    selected_y: list[np.ndarray] = []
    selected_cell: list[np.ndarray] = []
    processed = 0
    for xs, ys in helper.iter_representative_points(
        src, layer, source_crs, crs.to_wkt(), 500_000
    ):
        colf, rowf = (~transform) * (xs, ys)
        cols = np.floor(np.asarray(colf)).astype(np.int64)
        rows = np.floor(np.asarray(rowf)).astype(np.int64)
        inside = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
        cells = rows * width + cols
        use = np.zeros(len(xs), dtype=bool)
        for radius in RADII:
            for changed in strict_changes[radius].values():
                use |= inside & in_sorted(cells, changed)
        if use.any():
            selected_x.append(np.asarray(xs)[use])
            selected_y.append(np.asarray(ys)[use])
            selected_cell.append(cells[use])
        processed += len(xs)
        progress("POP_SELECT", state, method, processed, feature_count, started)
    sx = np.concatenate(selected_x) if selected_x else np.empty(0)
    sy = np.concatenate(selected_y) if selected_y else np.empty(0)
    sc = np.concatenate(selected_cell) if selected_cell else np.empty(0, np.int64)

    fips = STATE[state][0]
    blocks = BLOCKS_DIR / f"tl_2022_{fips}_tabblock20.gpkg"
    bg0 = old[old.buffer_m.eq(RADII[0])].sort_values("BG_CODE").reset_index(drop=True)
    bg_codes = bg0.BG_CODE.to_numpy(np.int64)
    tree, geometries, _, block_codes, _, _ = exact.load_block_index(helper, blocks, crs)
    dense_parts: list[np.ndarray] = []
    assignment = {"unique": 0, "ambiguous": 0, "outside": 0}
    for start in range(0, len(sx), 250_000):
        dense, stats = vector_exact_bg(
            tree, np.asarray(block_codes, np.int64),
            sx[start:start + 250_000], sy[start:start + 250_000], bg_codes,
        )
        dense_parts.append(dense)
        for key in assignment:
            assignment[key] += stats[key]
        progress(
            "POP_ASSIGN", state, method, min(start + 250_000, len(sx)), len(sx), started
        )
    dense = np.concatenate(dense_parts) if dense_parts else np.empty(0, np.int64)
    del tree, geometries

    output_rows: list[dict[str, Any]] = []
    details: list[pd.DataFrame] = []
    for radius in RADII:
        bg = old[old.buffer_m.eq(radius)].sort_values("BG_CODE").reset_index(drop=True)
        total = bg.Structure_Count.to_numpy(np.int64)
        c0 = bg.NonWUI_Structure_Count.to_numpy(np.int64)
        c1 = bg.Intermix_Structure_Count.to_numpy(np.int64)
        c2 = bg.Interface_Structure_Count.to_numpy(np.int64)
        good = dense >= 0
        class_counts = [c0.copy(), c1.copy(), c2.copy()]
        moved: dict[tuple[int, int], int] = {}
        for (old_cls, new_cls), cell_ids in strict_changes[radius].items():
            selected = good & in_sorted(sc, cell_ids)
            delta = np.bincount(dense[selected], minlength=len(bg)).astype(np.int64)
            class_counts[old_cls] -= delta
            class_counts[new_cls] += delta
            moved[(old_cls, new_cls)] = int(selected.sum())
        detail, summary = allocate(bg, total, *class_counts)
        detail["buffer_m"] = radius
        detail["method"] = method
        detail["state"] = state
        lead = ["state", "method", "buffer_m"]
        detail = detail[lead + [c for c in detail.columns if c not in lead]]
        details.append(detail)
        output_rows.append({
            "state": state,
            "STATEFP": fips,
            "state_name": STATE[state][1],
            "method": method,
            "buffer_m": radius,
            **summary,
            "selected_changed_cell_points": int(len(sx)),
            **{
                f"moved_{old_cls}_to_{new_cls}_points": moved[(old_cls, new_cls)]
                for old_cls, new_cls in moved
            },
            **transition_stats[radius],
            "classification_path": str(strict_raster(state, method, radius)),
            "classification_sha256": sha256(strict_raster(state, method, radius)),
            "population_source": str(src),
            "status": (
                "PASS"
                if abs(summary["population_closure_error"]) <= 1e-6
                and round(summary["total_population"]) == STATE[state][2]
                else "FAIL"
            ),
        })
    atomic_csv(
        out / f"population/{state}_{method.replace('-', '')}_block_group_detail.csv",
        pd.concat(details, ignore_index=True),
        "%.12f",
    )
    atomic_csv(
        out / f"population/{state}_{method.replace('-', '')}_population_rows.csv",
        pd.DataFrame(output_rows),
        "%.12f",
    )
    if not all(row["status"] == "PASS" for row in output_rows):
        raise RuntimeError(f"Population closure failed: {state}/{method}")
    atomic_json(checkpoint, {
        "status": "PASS",
        "completed_utc": now(),
        "state": state,
        "method": method,
        "assignment": assignment,
        "rows": output_rows,
    })
    return output_rows


def population_full(
    out: Path, state: str, method: str, helper, exact, old35
) -> list[dict[str, Any]]:
    """Full accepted allocation for products whose historic NoData domain differs."""
    checkpoint = out / f"checkpoints/population_{state}_{method.replace('-', '')}.json"
    if checkpoint.is_file():
        saved = json.loads(checkpoint.read_text())
        if saved.get("status") == "PASS":
            return saved["rows"]
    started = time.monotonic()
    src = structure_source(state, method)
    fips, name, official = STATE[state]
    baseline = baseline_detail(state, method)
    bg = (
        baseline[baseline.buffer_m.eq(RADII[0])]
        .sort_values("BG_CODE")
        .reset_index(drop=True)[["GEOID12", "POP20", "BG_CODE"]]
    )
    bg_codes = bg.BG_CODE.to_numpy(np.int64)
    counts = {
        radius: [np.zeros(len(bg), dtype=np.int64) for _ in range(3)]
        for radius in RADII
    }
    layer = helper.first_layer(src)
    feature_count, source_crs = helper.layer_info(src, layer)
    with rasterio.open(baseline_raster(state, method, 500)) as accepted:
        target_crs = accepted.crs
    blocks = BLOCKS_DIR / f"tl_2022_{fips}_tabblock20.gpkg"
    tree, geometries, _, block_codes, _, _ = exact.load_block_index(
        helper, blocks, target_crs
    )
    assignment = {"unique": 0, "ambiguous": 0, "outside": 0}
    valid_target_nodata = {radius: 0 for radius in RADII}
    processed = 0
    from contextlib import ExitStack
    with ExitStack() as stack:
        rasters = {
            radius: stack.enter_context(rasterio.open(strict_raster(state, method, radius)))
            for radius in RADII
        }
        for xs, ys in helper.iter_representative_points(
            src, layer, source_crs, target_crs.to_wkt(), 250_000
        ):
            dense, stats = vector_exact_bg(
                tree, np.asarray(block_codes, np.int64),
                np.asarray(xs), np.asarray(ys), bg_codes,
            )
            for key in assignment:
                assignment[key] += stats[key]
            unique = dense >= 0
            for radius in RADII:
                raw, inbounds = old35.sample_wui_by_tiles(
                    rasters[radius], np.asarray(xs), np.asarray(ys), 512
                )
                classes, unexpected = exact.classify_wui(raw, inbounds)
                if unexpected:
                    raise RuntimeError(
                        f"Unexpected WUI codes {state}/{method}/{radius}: {unexpected}"
                    )
                valid_target_nodata[radius] += int(
                    (unique & (~inbounds | (raw == 255))).sum()
                )
                for cls in (0, 1, 2):
                    selected = unique & (classes == cls)
                    counts[radius][cls] += np.bincount(
                        dense[selected], minlength=len(bg)
                    ).astype(np.int64)
            processed += len(xs)
            progress("POP_FULL", state, method, processed, feature_count, started)
    del tree, geometries
    rows: list[dict[str, Any]] = []
    details: list[pd.DataFrame] = []
    for radius in RADII:
        classified_total = counts[radius][0] + counts[radius][1] + counts[radius][2]
        detail, summary = allocate(bg, classified_total, *counts[radius])
        detail["state"] = state
        detail["method"] = method
        detail["buffer_m"] = radius
        lead = ["state", "method", "buffer_m"]
        details.append(detail[lead + [c for c in detail.columns if c not in lead]])
        rows.append({
            "state": state, "STATEFP": fips, "state_name": name,
            "method": method, "buffer_m": radius, **summary,
            "points_processed": processed,
            "points_exact_unique_block_group": assignment["unique"],
            "points_exact_ambiguous_block_group": assignment["ambiguous"],
            "points_outside_target_state_block_coverage": assignment["outside"],
            "target_state_eligible_nodata_255": valid_target_nodata[radius],
            "classified_structure_denominator": int(classified_total.sum()),
            "nodata_points_excluded_from_denominator": valid_target_nodata[radius],
            "classification_path": str(strict_raster(state, method, radius)),
            "classification_sha256": sha256(strict_raster(state, method, radius)),
            "population_source": str(src),
            "status": (
                "PASS"
                if abs(summary["population_closure_error"]) <= 1e-6
                and round(summary["total_population"]) == official
                else "FAIL"
            ),
        })
    atomic_csv(
        out / f"population/{state}_{method.replace('-', '')}_block_group_detail.csv",
        pd.concat(details, ignore_index=True), "%.12f",
    )
    atomic_csv(
        out / f"population/{state}_{method.replace('-', '')}_population_rows.csv",
        pd.DataFrame(rows), "%.12f",
    )
    if not all(row["status"] == "PASS" for row in rows):
        raise RuntimeError(f"Full population closure failed: {state}/{method}")
    atomic_json(checkpoint, {
        "status": "PASS", "completed_utc": now(), "state": state,
        "method": method, "assignment": assignment, "rows": rows,
        "population_algorithm": "FULL_EXACT_BG_RECLASSIFICATION",
    })
    return rows


def population_z(state: str) -> tuple[dict[str, Any], pd.DataFrame]:
    zpath = strict_z_vector(state)
    layer = fiona.listlayers(zpath)[0]
    fips, name, official = STATE[state]
    blocks = BLOCKS_DIR / f"tl_2022_{fips}_tabblock20.gpkg"
    blayer = fiona.listlayers(blocks)[0]
    with sqlite3.connect(str(zpath)) as con:
        z = pd.read_sql_query(
            f'SELECT GEOID20, WUI_Code, HU_Census FROM "{layer}"', con
        )
    with sqlite3.connect(str(blocks)) as con:
        b = pd.read_sql_query(
            f'SELECT GEOID20, POP20 FROM "{blayer}"', con
        )
    for frame in (z, b):
        frame["GEOID20"] = frame.GEOID20.astype(str).str.replace(
            r"\\.0$", "", regex=True
        ).str.zfill(15)
    if z.GEOID20.duplicated().any() or b.GEOID20.duplicated().any():
        raise RuntimeError(f"Duplicate block GEOID in WUI-Z join: {state}")
    joined = z.merge(b, on="GEOID20", how="outer", indicator=True)
    if not joined._merge.eq("both").all():
        raise RuntimeError(f"Non-exact WUI-Z/POP20 block join: {state}")
    joined["WUI_Code"] = joined.WUI_Code.astype(int)
    rows = []
    for cls, label in [(0, "Non-WUI"), (1, "Intermix"), (2, "Interface")]:
        part = joined[joined.WUI_Code.eq(cls)]
        rows.append({
            "state": state,
            "STATEFP": fips,
            "state_name": name,
            "method": "WUI-Z",
            "buffer_m": 0,
            "class_code": cls,
            "class_label": label,
            "block_count": len(part),
            "population": float(part.POP20.sum()),
            "housing_units": float(part.HU_Census.sum()),
        })
    r = pd.DataFrame(rows)
    summary = {
        "state": state,
        "STATEFP": fips,
        "state_name": name,
        "method": "WUI-Z",
        "buffer_m": 0,
        "nonwui_population": float(r.loc[r.class_code.eq(0), "population"].iloc[0]),
        "intermix_population": float(r.loc[r.class_code.eq(1), "population"].iloc[0]),
        "interface_population": float(r.loc[r.class_code.eq(2), "population"].iloc[0]),
        "wui_population": float(r.loc[r.class_code.isin([1, 2]), "population"].sum()),
        "total_population": float(r.population.sum()),
        "population_closure_error": float(r.population.sum() - official),
        "wui_population_share_pct": 100.0 * float(
            r.loc[r.class_code.isin([1, 2]), "population"].sum()
        ) / official,
        "classification_path": str(strict_z_raster(state)),
        "classification_sha256": sha256(strict_z_raster(state)),
        "status": "PASS" if round(r.population.sum()) == official else "FAIL",
    }
    return summary, r


def p_count_raster(state: str) -> Path:
    return COUNTY_CACHE / f"state_{STATE[state][0]}/WUI-P/struct_count_state_{STATE[state][0]}_WUI-P.tif"


def s_count_raster(state: str) -> Path:
    if state == "CO":
        # Canary created the strict count source.
        candidate = STEP47_CO500 / "WUI_S_CO_centroid_count_30m.tif"
        if candidate.exists():
            return candidate
        return COUNTY_CACHE / f"state_08/WUI-S/struct_count_state_08_WUI-S.tif"
    return STEP47_FOUR / state / f"WUI_S_{state}_centroid_count_30m.tif"


def county_metrics(out: Path, metric_mod) -> pd.DataFrame:
    counties = gpd.read_file(COUNTY_GPKG, layer="tl_2022_us_county")
    counties["STATEFP"] = counties.STATEFP.astype(str).str.zfill(2)
    counties["GEOID_INT"] = counties.GEOID.astype(str).astype(int)
    all_rows: list[pd.DataFrame] = []
    started = time.monotonic()
    done, total = 0, 105
    for state, (fips, name, _) in STATE.items():
        part = counties[counties.STATEFP.eq(fips)].copy().sort_values("GEOID_INT")
        ids = part.GEOID_INT.astype(int).tolist()
        county_raster = COUNTY_CACHE / f"state_{fips}/county_id_state_{fips}.tif"
        for method in METHODS:
            counts = p_count_raster(state) if method == "WUI-P" else s_count_raster(state)
            for radius in RADII:
                met = metric_mod.county_metrics_from_rasters(
                    strict_raster(state, method, radius), county_raster, counts, ids
                )
                met.insert(0, "buffer_m", radius)
                met.insert(0, "method", method)
                met.insert(0, "state_name", name)
                met.insert(0, "STATEFP", fips)
                met.insert(0, "state", state)
                all_rows.append(met)
                done += 1
                progress("COUNTY", state, method, done, total, started)
        # WUI-Z p_a is formal; p_s is deliberately not assigned.
        met = metric_mod.county_metrics_from_rasters(
            strict_z_raster(state), county_raster, p_count_raster(state), ids
        )
        met["Total_struct"] = np.nan
        met["Intermix_struct"] = np.nan
        met["Interface_struct"] = np.nan
        met["WUI_struct"] = np.nan
        met["p_s"] = np.nan
        met.insert(0, "buffer_m", 0)
        met.insert(0, "method", "WUI-Z")
        met.insert(0, "state_name", name)
        met.insert(0, "STATEFP", fips)
        met.insert(0, "state", state)
        all_rows.append(met)
        done += 1
        progress("COUNTY", state, "WUI-Z", done, total, started)
    result = pd.concat(all_rows, ignore_index=True)
    atomic_csv(out / "county_metrics/patch75_five_state_county_metrics.csv", result, "%.12f")
    return result


def area_table() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for state, (fips, name, _) in STATE.items():
        for method in METHODS:
            for radius in RADII:
                path = strict_raster(state, method, radius)
                with rasterio.open(path) as src:
                    counts = {0: 0, 1: 0, 2: 0}
                    for _, window in src.block_windows(1):
                        a = src.read(1, window=window)
                        for cls in counts:
                            counts[cls] += int((a == cls).sum())
                    pixel = abs(src.res[0] * src.res[1]) / 1e6
                rows.append({
                    "state": state, "STATEFP": fips, "state_name": name,
                    "method": method, "buffer_m": radius,
                    "nonwui_area_km2": counts[0] * pixel,
                    "intermix_area_km2": counts[1] * pixel,
                    "interface_area_km2": counts[2] * pixel,
                    "wui_area_km2": (counts[1] + counts[2]) * pixel,
                    "valid_area_km2": sum(counts.values()) * pixel,
                    "classification_path": str(path),
                    "classification_sha256": sha256(path),
                })
        path = strict_z_raster(state)
        with rasterio.open(path) as src:
            counts = {cls: 0 for cls in (0, 1, 2)}
            for _, window in src.block_windows(1):
                a = src.read(1, window=window)
                for cls in counts:
                    counts[cls] += int((a == cls).sum())
            pixel = abs(src.res[0] * src.res[1]) / 1e6
        rows.append({
            "state": state, "STATEFP": fips, "state_name": name,
            "method": "WUI-Z", "buffer_m": 0,
            "nonwui_area_km2": counts[0] * pixel,
            "intermix_area_km2": counts[1] * pixel,
            "interface_area_km2": counts[2] * pixel,
            "wui_area_km2": (counts[1] + counts[2]) * pixel,
            "valid_area_km2": sum(counts.values()) * pixel,
            "classification_path": str(path),
            "classification_sha256": sha256(path),
        })
    return pd.DataFrame(rows)


def write_manifest(out: Path) -> None:
    rows = []
    for path in sorted(p for p in out.rglob("*") if p.is_file() and p.name != "sha256_manifest.txt"):
        rows.append(f"{sha256(path)}  {path.relative_to(out)}")
    (out / "sha256_manifest.txt").write_text("\n".join(rows) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--canary-state", choices=sorted(STATE), default="")
    args = ap.parse_args()
    out = (args.output or ROOT / f"step48_patch75_five_state_downstream_{stamp()}").resolve()
    resuming = out.exists()
    out.mkdir(parents=True, exist_ok=True)
    for sub in ("population", "county_metrics", "checkpoints", "logs", "tables", "figures"):
        (out / sub).mkdir(exist_ok=True)
    atomic_json(out / "step48_status.json", {
        "step": "STEP48_PATCH75_FIVE_STATE_DOWNSTREAM_REBUILD",
        "phase": "STEP48A",
        "status": "RUNNING",
        "created_utc": now(),
        "resumed_existing_output": resuming,
        "output_directory": str(out),
        "definition": "qualifying patch Veg_Percent >75%; area >=5 km2; distance <=2.4 km",
        "formal_overwrite": False,
    })
    helper = load_module(HELPER_PATH, "step48_helper")
    exact = load_module(EXACT_PATH, "step48_exact")
    metric_mod = load_module(METRIC_PATH, "step48_metric")
    old35 = load_module(OLD35_PATH, "step48_old35")
    selected = [args.canary_state] if args.canary_state else list(STATE)
    pop_rows: list[dict[str, Any]] = []
    z_class_rows: list[pd.DataFrame] = []
    for state in selected:
        for method in METHODS:
            if method == "WUI-P":
                pop_rows.extend(
                    population_ps(out, state, method, helper, exact, old35)
                )
            else:
                pop_rows.extend(
                    population_full(out, state, method, helper, exact, old35)
                )
        z_summary, z_rows = population_z(state)
        if z_summary["status"] != "PASS":
            raise RuntimeError(f"WUI-Z population closure failed: {state}")
        pop_rows.append(z_summary)
        z_class_rows.append(z_rows)
    atomic_csv(out / "patch75_five_state_population.csv", pd.DataFrame(pop_rows), "%.12f")
    atomic_csv(out / "population/patch75_wuiz_class_population.csv",
               pd.concat(z_class_rows, ignore_index=True), "%.12f")

    if args.canary_state:
        # Canary validates population first; the full run creates county tables.
        status = "CANARY_POPULATION_COMPLETE"
        atomic_json(out / "step48_status.json", {
            "step": "STEP48_PATCH75_FIVE_STATE_DOWNSTREAM_REBUILD",
            "phase": "STEP48A_CANARY",
            "status": status,
            "completed_utc": now(),
            "output_directory": str(out),
            "state": args.canary_state,
        })
        write_manifest(out)
        print(f"OUTPUT_DIR={out}", flush=True)
        return 0

    area = area_table()
    atomic_csv(out / "patch75_five_state_area.csv", area, "%.12f")
    county = county_metrics(out, metric_mod)
    checks = [
        ("population_rows_105", len(pop_rows) == 105, len(pop_rows)),
        ("area_rows_105", len(area) == 105, len(area)),
        ("county_metric_combinations_105",
         county[["state", "method", "buffer_m"]].drop_duplicates().shape[0] == 105,
         county[["state", "method", "buffer_m"]].drop_duplicates().shape[0]),
        ("population_all_pass", pd.DataFrame(pop_rows).status.eq("PASS").all(), ""),
        ("county_pa_range", county.p_a.dropna().between(0, 1).all(), ""),
        ("county_ps_range", county.p_s.dropna().between(0, 1).all(), ""),
        ("wuiz_ps_explicit_na",
         county.loc[county.method.eq("WUI-Z"), "p_s"].isna().all(), ""),
    ]
    qc = pd.DataFrame(checks, columns=["check", "passed", "detail"])
    atomic_csv(out / "STEP48A_QC.csv", qc)
    if not qc.passed.all():
        raise RuntimeError(f"Step48A QC failed: {qc.loc[~qc.passed].to_dict('records')}")
    (out / "README.md").write_text(
        "# Step48 strict-patch five-state downstream rebuild\n\n"
        "This isolated candidate run uses the strict `Veg_Percent >75%` qualifying-"
        "patch definition. WUI-P/WUI-S population is updated from accepted block-"
        "group allocation details only for verified WUI-to-Non-WUI transitions. "
        "WUI-Z population is joined directly from Census block POP20. County `p_a` "
        "is produced for P/S/Z; county `p_s` is produced only for P/S because the "
        "accepted Moran method never defined a WUI-Z structure denominator.\n"
    )
    atomic_json(out / "step48_status.json", {
        "step": "STEP48_PATCH75_FIVE_STATE_DOWNSTREAM_REBUILD",
        "phase": "STEP48A",
        "status": "PATCH75_FIVE_STATE_METRICS_COMPLETE_READY_FOR_MORAN",
        "completed_utc": now(),
        "output_directory": str(out),
        "population_rows": len(pop_rows),
        "area_rows": len(area),
        "county_rows": len(county),
        "qc_passed": int(qc.passed.sum()),
        "qc_failed": int((~qc.passed).sum()),
        "formal_overwrite": False,
    })
    write_manifest(out)
    print(f"OUTPUT_DIR={out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
