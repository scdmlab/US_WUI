#!/usr/bin/env python3
# Copyright (C) 2026 WUI Mapping Project Authors
# SPDX-License-Identifier: GPL-3.0-only
# -*- coding: utf-8 -*-
# Repository version: filesystem paths are loaded through repo_config.py.
# Copy config/paths.example.json to config/paths.json before a new run.
# Raw input data and large intermediate files are stored outside GitHub.

"""
STEP 08 - Recompute CA/CO/FL/TX WUI-P and WUI-S population at 500 m.

Why this script exists
----------------------
The earlier fast population raster did not conserve official 2020 Census
population in the four remaining sample states.  This script replaces that result
with a structure-based dasymetric allocation consistent with the reference
paper:

1. Aggregate block POP20 to 2020 Census block groups (first 12 GEOID digits).
2. Convert every address/building feature to one representative point.
3. Assign each point to a block group and to WUI class 0/1/2.
4. Within each block group, allocate POP20 in proportion to its point counts
   by WUI class.
5. Assign the population of block groups with no usable points to Non-WUI.

This guarantees that Non-WUI + Intermix + Interface equals the official POP20
state total for every state/method.  Existing source datasets are never
modified.  New caches and QC outputs are written only below --work-root.

Default scope
-------------
California (06), Colorado (08), Florida (12), and Texas (48), WUI-P and WUI-S, 500 m.

Dependencies
------------
Python: numpy, pandas, rasterio, fiona
Commands: gdal_rasterize, ogr2ogr
"""

from __future__ import annotations

from repo_config import portable_path

import argparse
import csv
import math
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import fiona
import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform as transform_xy
from rasterio.windows import Window


DEFAULT_WORK_ROOT = Path(portable_path("project"))
DEFAULT_BLOCKS_DIR = Path(
    portable_path("data", "MBF+NLCD_2022US/Inputs/Processed_GPKG")
)
DEFAULT_OA_DIR = Path(
    portable_path("data", "OpenAddresses_Work/Processed_GPKG")
)
DEFAULT_MBF_DIR = DEFAULT_BLOCKS_DIR
DEFAULT_WUIP_DIR = Path(portable_path("data", "WUI_P_Paper_Raster"))
DEFAULT_WUIS_DIR = Path(portable_path("data", "WUI_S_Paper"))

CLASS_LABEL = {0: "Non-WUI", 1: "Intermix", 2: "Interface"}
CONUS_ALBERS = CRS.from_epsg(5070)
INT32_MAX = int(np.iinfo(np.int32).max)
UINT32_MODULUS = 1 << 32


@dataclass(frozen=True)
class StateSpec:
    statefp: str
    stusps: str
    name: str


STATES = (
    StateSpec("06", "CA", "California"),
    StateSpec("08", "CO", "Colorado"),
    StateSpec("12", "FL", "Florida"),
    StateSpec("48", "TX", "Texas"),
)

OFFICIAL_2020_POPULATION = {
    "06": 39_538_223.0,
    "08": 5_773_714.0,
    "12": 21_538_187.0,
    "48": 29_145_505.0,
}


def hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def progress(done: int, total: int, started: float, label: str) -> None:
    elapsed = time.time() - started
    pct = 100.0 * done / total if total else 100.0
    rate = done / elapsed if elapsed > 0 else 0.0
    eta = (total - done) / rate if rate > 0 else 0.0
    print(
        f"[{done:,}/{total:,} {pct:6.2f}% | elapsed {hms(elapsed)} | "
        f"ETA {hms(eta)}] {label}",
        flush=True,
    )


def quote_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def require_command(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required command is not available: {name}")


def first_layer(path: Path) -> str:
    layers = fiona.listlayers(str(path))
    if not layers:
        raise RuntimeError(f"No layers found: {path}")
    return layers[0]


def layer_info(path: Path, layer: str) -> Tuple[int, str]:
    with fiona.open(str(path), layer=layer) as src:
        return len(src), src.crs_wkt or ""


def normalized_sampling_crs(crs: CRS, label: str) -> Tuple[CRS, bool]:
    """Return a transformable CRS, repairing the known legacy 5070 encoding.

    Some WUI GeoTIFFs written by the server's older GDAL stack expose
    ``NAD83 / Conus Albers`` as an EngineeringCRS.  Such a CRS retains the
    projected x/y grid but has no geodetic base or conversion, so PROJ cannot
    construct a coordinate operation to it.  Only that exact known encoding is
    repaired; any other EngineeringCRS remains a hard error.
    """
    text = crs.to_wkt()
    folded = " ".join(text.lower().split())
    if "engineeringcrs" in folded:
        if "nad83 / conus albers" not in folded:
            raise RuntimeError(
                f"{label} uses an unsupported EngineeringCRS; refusing to "
                "guess its coordinate system"
            )
        return CONUS_ALBERS, True
    return crs, False


def geometry_column(path: Path, layer: str) -> str:
    with sqlite3.connect(str(path)) as con:
        row = con.execute(
            "SELECT column_name FROM gpkg_geometry_columns WHERE table_name = ?",
            (layer,),
        ).fetchone()
    if not row:
        raise RuntimeError(f"Cannot resolve geometry column for {path}:{layer}")
    return str(row[0])


def schema_fields(path: Path, layer: str) -> List[str]:
    with fiona.open(str(path), layer=layer) as src:
        return list(src.schema["properties"].keys())


def pick_field(fields: Sequence[str], candidates: Sequence[str], label: str) -> str:
    lookup = {x.lower(): x for x in fields}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    raise RuntimeError(
        f"Cannot find {label}; tried {list(candidates)}; available={list(fields)}"
    )


def bg_code_from_geoid12(geoid12: str) -> int:
    """Collision-checked signed Int32 code for a state-local block group.

    The natural county(3)+tract(6)+block-group(1) decimal code can exceed
    signed Int32 in states whose county FIPS code is 215 or higher (including
    Texas).  Preserve all 32 low-order bits and represent them as signed Int32;
    ``load_block_group_population`` separately rejects the unlikely possibility
    of a within-state collision after folding.
    """
    value = str(geoid12).strip().zfill(12)
    if len(value) != 12 or not value.isdigit():
        raise ValueError(f"Invalid block-group GEOID: {geoid12!r}")
    county = int(value[2:5])
    tract = int(value[5:11])
    block_group = int(value[11])
    raw_code = county * 10_000_000 + tract * 10 + block_group
    unsigned_code = raw_code % UINT32_MODULUS
    code = (
        unsigned_code
        if unsigned_code <= INT32_MAX
        else unsigned_code - UINT32_MODULUS
    )
    if code == 0:
        raise ValueError(f"BG code collides with NoData zero: {value} -> {raw_code}")
    return code


def load_block_group_population(
    blocks_path: Path, layer: str
) -> Tuple[pd.DataFrame, str, str]:
    fields = schema_fields(blocks_path, layer)
    geoid_field = pick_field(fields, ["GEOID20", "GEOID"], "block GEOID")
    pop_field = pick_field(
        fields, ["POP20", "P0010001", "POPULATION20", "population"], "POP20"
    )
    sql = f"""
        SELECT SUBSTR({quote_ident(geoid_field)}, 1, 12) AS GEOID12,
               SUM(CAST({quote_ident(pop_field)} AS REAL)) AS POP20
        FROM {quote_ident(layer)}
        GROUP BY SUBSTR({quote_ident(geoid_field)}, 1, 12)
        ORDER BY GEOID12
    """
    with sqlite3.connect(str(blocks_path)) as con:
        df = pd.read_sql_query(sql, con)
    if df.empty:
        raise RuntimeError(f"No block-group population rows read from {blocks_path}")
    df["GEOID12"] = df["GEOID12"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(12)
    df["POP20"] = pd.to_numeric(df["POP20"], errors="coerce")
    if df["GEOID12"].duplicated().any() or df["POP20"].isna().any():
        raise RuntimeError(f"Invalid/duplicate block-group population in {blocks_path}")
    df["BG_CODE"] = df["GEOID12"].map(bg_code_from_geoid12).astype(np.int64)
    if df["BG_CODE"].duplicated().any():
        raise RuntimeError(f"BG_CODE collision in {blocks_path}")
    df = df.sort_values("BG_CODE").reset_index(drop=True)
    return df, geoid_field, pop_field


def valid_cached_bg_raster(path: Path, template: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with rasterio.open(path) as cached, rasterio.open(template) as source:
            if cached.crs is None or source.crs is None:
                return False
            cached_crs, _ = normalized_sampling_crs(
                cached.crs, f"Cached block-group raster {path}"
            )
            source_crs, _ = normalized_sampling_crs(
                source.crs, f"Template raster {template}"
            )
            return (
                cached.width == source.width
                and cached.height == source.height
                and cached.transform.almost_equals(source.transform)
                and cached_crs == source_crs
                and cached.count == 1
                and cached.dtypes[0] == "int32"
            )
    except Exception:
        return False


def build_bg_raster(
    blocks_path: Path,
    blocks_layer: str,
    geoid_field: str,
    template: Path,
    out_path: Path,
) -> None:
    """Rasterize block-group codes on the exact P/S grid without touching source data."""
    if valid_cached_bg_raster(out_path, template):
        print(f"[CACHE] Reusing aligned block-group raster: {out_path}", flush=True)
        return
    if out_path.exists():
        out_path.unlink()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(template) as src:
        if src.crs is None:
            raise RuntimeError(f"Template raster has no CRS: {template}")
        b = src.bounds
        width, height = src.width, src.height
        output_crs, repaired = normalized_sampling_crs(
            src.crs, f"Template raster {template}"
        )
        # An authority code is more portable across the server's mixed
        # GDAL/PROJ versions than serializing a newer WKT representation.
        crs_wkt = "EPSG:5070" if output_crs == CONUS_ALBERS else output_crs.to_wkt()
        if repaired:
            print(
                "[CRS] Template stores NAD83 / Conus Albers as an "
                "EngineeringCRS; assigning canonical EPSG:5070 to the "
                "block-group cache",
                flush=True,
            )

    g = quote_ident(geoid_field)
    lyr = quote_ident(blocks_layer)
    # For a 15-digit block GEOID: state 1-2, county 3-5, tract 6-11,
    # block-group digit 12. Fold the natural decimal code into signed Int32
    # exactly as bg_code_from_geoid12 does. This avoids overflow for Texas while
    # retaining compact 4-byte cache rasters and reserving zero for NoData.
    raw_code_sql = (
        f"(CAST(SUBSTR({g},3,3) AS INTEGER) * 10000000 + "
        f"CAST(SUBSTR({g},6,6) AS INTEGER) * 10 + "
        f"CAST(SUBSTR({g},12,1) AS INTEGER))"
    )
    unsigned_code_sql = f"(({raw_code_sql}) % {UINT32_MODULUS})"
    signed_code_sql = (
        f"(CASE WHEN {unsigned_code_sql} > {INT32_MAX} "
        f"THEN {unsigned_code_sql} - {UINT32_MODULUS} "
        f"ELSE {unsigned_code_sql} END)"
    )
    sql = f"SELECT *, {signed_code_sql} AS BG_CODE FROM {lyr}"
    cmd = [
        "gdal_rasterize",
        "-q",
        "-of", "GTiff",
        "-ot", "Int32",
        "-init", "0",
        "-a_nodata", "0",
        "-a_srs", crs_wkt,
        "-te", str(b.left), str(b.bottom), str(b.right), str(b.top),
        "-ts", str(width), str(height),
        "-co", "TILED=YES",
        "-co", "COMPRESS=LZW",
        "-co", "BIGTIFF=IF_SAFER",
        "-dialect", "SQLITE",
        "-sql", sql,
        "-a", "BG_CODE",
        str(blocks_path),
        str(out_path),
    ]
    print(f"[BUILD] Rasterizing block groups: {out_path}", flush=True)
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            "gdal_rasterize failed:\n" + (result.stderr or result.stdout)[-6000:]
        )
    if not valid_cached_bg_raster(out_path, template):
        raise RuntimeError(f"Block-group raster failed alignment validation: {out_path}")


def representative_point_command(
    source: Path, layer: str
) -> List[str]:
    geom_col = geometry_column(source, layer)
    sql = (
        f"SELECT ST_X(ST_PointOnSurface({quote_ident(geom_col)})) AS X, "
        f"ST_Y(ST_PointOnSurface({quote_ident(geom_col)})) AS Y "
        f"FROM {quote_ident(layer)} WHERE {quote_ident(geom_col)} IS NOT NULL"
    )
    return [
        "ogr2ogr",
        "-f", "CSV",
        "/vsistdout/",
        str(source),
        "-dialect", "SQLITE",
        "-sql", sql,
    ]


def find_xy_columns(fieldnames: Sequence[str] | None) -> Tuple[str, str]:
    if not fieldnames:
        raise RuntimeError("ogr2ogr CSV output has no header")
    lookup = {str(x).strip().lower(): str(x) for x in fieldnames}
    x_names = ["x", "xcoord", "longitude", "lon"]
    y_names = ["y", "ycoord", "latitude", "lat"]
    x_col = next((lookup[x] for x in x_names if x in lookup), None)
    y_col = next((lookup[y] for y in y_names if y in lookup), None)
    if not x_col or not y_col:
        raise RuntimeError(f"Cannot identify X/Y columns in ogr2ogr output: {fieldnames}")
    return x_col, y_col


def iter_representative_points(
    source: Path,
    layer: str,
    source_crs_wkt: str,
    target_crs_wkt: str,
    chunk_size: int,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Stream one representative point per input feature through ogr2ogr CSV."""
    source_crs = CRS.from_wkt(source_crs_wkt)
    target_crs = CRS.from_wkt(target_crs_wkt)
    needs_transform = source_crs != target_crs
    cmd = representative_point_command(source, layer)
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1024 * 1024,
    )
    assert process.stdout is not None
    reader = csv.DictReader(process.stdout)
    x_col, y_col = find_xy_columns(reader.fieldnames)
    xs: List[float] = []
    ys: List[float] = []
    invalid_rows = 0
    for row in reader:
        try:
            x = float(row[x_col])
            y = float(row[y_col])
            if not (math.isfinite(x) and math.isfinite(y)):
                raise ValueError
        except (TypeError, ValueError, KeyError):
            invalid_rows += 1
            continue
        xs.append(x)
        ys.append(y)
        if len(xs) >= chunk_size:
            if needs_transform:
                xs, ys = transform_xy(source_crs, target_crs, xs, ys)
            yield np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)
            xs, ys = [], []
    if xs:
        if needs_transform:
            xs, ys = transform_xy(source_crs, target_crs, xs, ys)
        yield np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)
    process.stdout.close()
    stderr = process.stderr.read() if process.stderr is not None else ""
    returncode = process.wait()
    if returncode != 0:
        raise RuntimeError(f"ogr2ogr representative-point stream failed:\n{stderr[-6000:]}")
    if invalid_rows:
        print(f"[WARN] Skipped {invalid_rows:,} invalid coordinate rows from {source}")


def sample_aligned_rasters_by_tiles(
    wui: rasterio.io.DatasetReader,
    bg: rasterio.io.DatasetReader,
    xs: np.ndarray,
    ys: np.ndarray,
    tile_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample aligned rasters with one read per occupied tile, not per point."""
    if (wui.width, wui.height, wui.transform) != (bg.width, bg.height, bg.transform):
        raise RuntimeError("WUI and block-group rasters are not aligned")
    inv = ~wui.transform
    cols_f, rows_f = inv * (xs, ys)
    cols = np.floor(cols_f).astype(np.int64)
    rows = np.floor(rows_f).astype(np.int64)
    inbounds = (
        (rows >= 0) & (rows < wui.height) & (cols >= 0) & (cols < wui.width)
    )
    wui_values = np.zeros(len(xs), dtype=np.int16)
    bg_values = np.zeros(len(xs), dtype=np.int64)
    valid_idx = np.flatnonzero(inbounds)
    if not len(valid_idx):
        return wui_values, bg_values, inbounds

    vr = rows[valid_idx]
    vc = cols[valid_idx]
    tile_cols = (wui.width + tile_size - 1) // tile_size
    keys = (vr // tile_size) * tile_cols + (vc // tile_size)
    order = np.argsort(keys, kind="stable")
    sorted_idx = valid_idx[order]
    sorted_keys = keys[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_keys)) + 1]
    ends = np.r_[starts[1:], len(sorted_idx)]

    for start, end in zip(starts, ends):
        idx = sorted_idx[start:end]
        r0 = int((rows[idx[0]] // tile_size) * tile_size)
        c0 = int((cols[idx[0]] // tile_size) * tile_size)
        height = min(tile_size, wui.height - r0)
        width = min(tile_size, wui.width - c0)
        window = Window(c0, r0, width, height)
        wa = wui.read(1, window=window)
        ba = bg.read(1, window=window)
        local_r = rows[idx] - r0
        local_c = cols[idx] - c0
        wui_values[idx] = wa[local_r, local_c].astype(np.int16, copy=False)
        bg_values[idx] = ba[local_r, local_c].astype(np.int64, copy=False)
    return wui_values, bg_values, inbounds


def aggregate_structure_counts(
    source_path: Path,
    source_layer: str,
    source_crs_wkt: str,
    wui_path: Path,
    bg_raster_path: Path,
    bg_codes: np.ndarray,
    chunk_size: int,
    tile_size: int,
    expected_features: int,
    progress_every: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    total = np.zeros(len(bg_codes), dtype=np.int64)
    c0 = np.zeros(len(bg_codes), dtype=np.int64)
    c1 = np.zeros(len(bg_codes), dtype=np.int64)
    c2 = np.zeros(len(bg_codes), dtype=np.int64)
    stats = {
        "points_processed": 0,
        "points_in_raster_bounds": 0,
        "points_matched_block_group": 0,
        "points_unmatched_block_group": 0,
        "unexpected_wui_values": 0,
    }
    started = time.time()
    next_report = progress_every

    with rasterio.open(wui_path) as wui, rasterio.open(bg_raster_path) as bg:
        if wui.crs is None:
            raise RuntimeError(f"WUI raster has no CRS: {wui_path}")
        source_crs = CRS.from_wkt(source_crs_wkt)
        target_crs, repaired = normalized_sampling_crs(
            wui.crs, f"WUI raster {wui_path}"
        )
        source_epsg = source_crs.to_epsg()
        target_epsg = target_crs.to_epsg()
        source_label = f"EPSG:{source_epsg}" if source_epsg else str(source_crs)
        target_label = f"EPSG:{target_epsg}" if target_epsg else str(target_crs)
        action = (
            "no coordinate transform"
            if source_crs == target_crs
            else "transform coordinates"
        )
        suffix = "; repaired legacy EngineeringCRS metadata" if repaired else ""
        print(
            f"[CRS] source={source_label}; sampling grid={target_label}; "
            f"{action}{suffix}",
            flush=True,
        )
        target_crs_wkt = target_crs.to_wkt()
        for xs, ys in iter_representative_points(
            source_path, source_layer, source_crs_wkt, target_crs_wkt, chunk_size
        ):
            wv, bv, inbounds = sample_aligned_rasters_by_tiles(
                wui, bg, xs, ys, tile_size
            )
            n = len(xs)
            stats["points_processed"] += n
            stats["points_in_raster_bounds"] += int(inbounds.sum())

            # Outside raster, declared nodata (including P's conflicting 0),
            # or any unexpected value is interpreted as Non-WUI. Values 1/2
            # remain Intermix/Interface.
            cls = np.zeros(n, dtype=np.int8)
            cls[wv == 1] = 1
            cls[wv == 2] = 2
            unexpected = inbounds & ~np.isin(wv, [0, 1, 2, 255])
            stats["unexpected_wui_values"] += int(unexpected.sum())

            idx = np.searchsorted(bg_codes, bv)
            # Zero alone is NoData. Valid signed Int32 block-group codes may be
            # negative after folding high Texas county/tract identifiers.
            matched = (bv != 0) & (idx < len(bg_codes))
            matched_positions = np.flatnonzero(matched)
            if len(matched_positions):
                matched[matched_positions] &= bg_codes[idx[matched_positions]] == bv[matched_positions]
            stats["points_matched_block_group"] += int(matched.sum())
            stats["points_unmatched_block_group"] += int((~matched).sum())

            dense = idx[matched]
            classes = cls[matched]
            total += np.bincount(dense, minlength=len(total))
            c0 += np.bincount(dense[classes == 0], minlength=len(total))
            c1 += np.bincount(dense[classes == 1], minlength=len(total))
            c2 += np.bincount(dense[classes == 2], minlength=len(total))

            if stats["points_processed"] >= next_report:
                progress(
                    stats["points_processed"], expected_features, started,
                    f"{source_path.name}: representative structures sampled",
                )
                next_report += progress_every
    progress(
        stats["points_processed"], expected_features, started,
        f"{source_path.name}: representative structures sampled",
    )
    return total, c0, c1, c2, stats


def allocate_population(
    bg_df: pd.DataFrame,
    total: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
    c2: np.ndarray,
) -> Tuple[pd.DataFrame, dict]:
    out = bg_df.copy()
    out["Structure_Count"] = total
    out["NonWUI_Structure_Count"] = c0
    out["Intermix_Structure_Count"] = c1
    out["Interface_Structure_Count"] = c2
    if not np.array_equal(total, c0 + c1 + c2):
        raise RuntimeError("Structure class counts do not sum to structure total")

    has = total > 0
    frac0 = np.where(has, c0 / np.maximum(total, 1), 1.0)
    frac1 = np.where(has, c1 / np.maximum(total, 1), 0.0)
    frac2 = np.where(has, c2 / np.maximum(total, 1), 0.0)
    pop = out["POP20"].to_numpy(dtype=np.float64)
    non = pop * frac0
    intermix = pop * frac1
    interface = pop * frac2
    official = float(pop.sum())
    # Remove only floating-point roundoff; do not conceal a logical residual.
    residual = official - float(non.sum() + intermix.sum() + interface.sum())
    non[0] += residual

    out["NonWUI_Fraction"] = frac0
    out["Intermix_Fraction"] = frac1
    out["Interface_Fraction"] = frac2
    out["NonWUI_Pop"] = non
    out["Intermix_Pop"] = intermix
    out["Interface_Pop"] = interface

    summary = {
        "official_population": official,
        "zero_structure_bg_count": int((~has).sum()),
        "zero_structure_bg_population": float(pop[~has].sum()),
        "nonwui_population": float(non.sum()),
        "intermix_population": float(intermix.sum()),
        "interface_population": float(interface.sum()),
    }
    summary["wui_population"] = (
        summary["intermix_population"] + summary["interface_population"]
    )
    summary["allocated_population"] = (
        summary["nonwui_population"] + summary["wui_population"]
    )
    summary["allocation_residual"] = (
        summary["allocated_population"] - summary["official_population"]
    )
    summary["wui_population_share_pct"] = (
        summary["wui_population"] / summary["official_population"] * 100.0
        if summary["official_population"] else 0.0
    )
    summary["verdict"] = (
        "PASS" if abs(summary["allocation_residual"]) <= 1e-6 else "FAIL"
    )
    return out, summary


def resolve_inputs(args: argparse.Namespace, state: StateSpec, method: str) -> dict:
    blocks = Path(args.blocks_dir) / f"tl_2022_{state.statefp}_tabblock20.gpkg"
    if method == "WUI-P":
        structures = Path(args.oa_dir) / f"{state.name}_addresses.gpkg"
        wui = Path(args.wuip_dir) / state.name / f"WUI_P_{state.name}_r0500m.tif"
    else:
        structures = Path(args.mbf_dir) / f"{state.name}.gpkg"
        wui = Path(args.wuis_dir) / state.name / f"WUI_S_{state.name}_r0500m.tif"
    require_file(blocks, "Census blocks")
    require_file(structures, f"{method} structure source")
    require_file(wui, f"{method} 500 m raster")
    return {"blocks": blocks, "structures": structures, "wui": wui}



def validate_step07_gate(work_root: Path, states: Sequence[StateSpec]) -> None:
    """Require the formal Step 07 POP20/WUI-Z gate to have passed."""
    path = work_root / "qc" / "07_sample4_population_wuiz_qc.csv"
    require_file(path, "Step 07 QC table")
    df = pd.read_csv(path, dtype={"STATEFP": str})
    required = {
        "STATEFP", "official_2020_population", "source_population",
        "source_minus_official", "population_allocation_residual", "verdict",
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"Step 07 QC table lacks required columns: {sorted(missing)}"
        )
    df["STATEFP"] = df["STATEFP"].astype(str).str.zfill(2)
    problems: List[str] = []
    for state in states:
        rows = df[df["STATEFP"] == state.statefp]
        if len(rows) != 1:
            problems.append(
                f"{state.name}: expected one Step 07 row, found {len(rows)}"
            )
            continue
        row = rows.iloc[0]
        expected = OFFICIAL_2020_POPULATION[state.statefp]
        checks = {
            "verdict": str(row["verdict"]).strip().upper() == "PASS",
            "official population": float(row["official_2020_population"]) == expected,
            "source population": float(row["source_population"]) == expected,
            "source-minus-official": abs(float(row["source_minus_official"])) <= 1e-9,
            "allocation residual": abs(float(row["population_allocation_residual"])) <= 1e-9,
        }
        failed = [label for label, ok in checks.items() if not ok]
        if failed:
            problems.append(f"{state.name}: failed {', '.join(failed)}")
    if problems:
        raise RuntimeError(
            "Step 07 prerequisite did not pass:\n- " + "\n- ".join(problems)
        )
    print(
        f"[GATE] Step 07 PASS confirmed for {len(states)} selected states",
        flush=True,
    )

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="STEP 08: recompute CA/CO/FL/TX WUI-P/S population using POP20."
    )
    ap.add_argument("--work-root", default=str(DEFAULT_WORK_ROOT))
    ap.add_argument("--blocks-dir", default=str(DEFAULT_BLOCKS_DIR))
    ap.add_argument("--oa-dir", default=str(DEFAULT_OA_DIR))
    ap.add_argument("--mbf-dir", default=str(DEFAULT_MBF_DIR))
    ap.add_argument("--wuip-dir", default=str(DEFAULT_WUIP_DIR))
    ap.add_argument("--wuis-dir", default=str(DEFAULT_WUIS_DIR))
    ap.add_argument("--chunk-size", type=int, default=200_000)
    ap.add_argument("--tile-size", type=int, default=512)
    ap.add_argument("--progress-every", type=int, default=1_000_000)
    ap.add_argument(
        "--resume", action="store_true",
        help="Reuse previously completed PASS jobs from STEP 08 outputs",
    )
    ap.add_argument(
        "--states", nargs="*", default=["06", "08", "12", "48"],
        help="STATEFP subset; default: 06 08 12 48",
    )
    ap.add_argument(
        "--methods", nargs="*", default=["WUI-P", "WUI-S"],
        choices=["WUI-P", "WUI-S"],
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    require_command("gdal_rasterize")
    require_command("ogr2ogr")
    work_root = Path(args.work_root)
    qc_dir = work_root / "qc"
    cache_dir = work_root / "cache" / "08_sample4_ps_population"
    qc_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    wanted_states = {str(x).zfill(2) for x in args.states}
    states = [x for x in STATES if x.statefp in wanted_states]
    unknown = wanted_states - {x.statefp for x in states}
    if unknown:
        raise RuntimeError(
            "This focused script supports only state FIPS 06/08/12/48; "
            f"got {sorted(unknown)}"
        )

    jobs = [(state, method) for state in states for method in args.methods]
    if not jobs:
        raise RuntimeError("No jobs selected")

    validate_step07_gate(work_root, states)

    print("[PREFLIGHT] Validating all selected input paths before computation", flush=True)
    input_map: Dict[Tuple[str, str], dict] = {}
    for state, method in jobs:
        input_map[(state.statefp, method)] = resolve_inputs(args, state, method)
    print(f"[PREFLIGHT] PASS: {len(jobs)} state-method jobs are ready", flush=True)

    summary_checkpoint = qc_dir / "08_sample4_ps_population_summary.csv"
    detail_checkpoint = qc_dir / "08_sample4_ps_population_by_block_group.csv"
    previous_summary = pd.DataFrame()
    previous_detail = pd.DataFrame()
    if args.resume and summary_checkpoint.is_file() and detail_checkpoint.is_file():
        previous_summary = pd.read_csv(summary_checkpoint, dtype={"STATEFP": str})
        previous_detail = pd.read_csv(detail_checkpoint, dtype={"STATEFP": str})
        previous_summary["STATEFP"] = previous_summary["STATEFP"].str.zfill(2)
        previous_detail["STATEFP"] = previous_detail["STATEFP"].str.zfill(2)
        print(
            f"[RESUME] Loaded {len(previous_summary)} prior summary rows and "
            f"{len(previous_detail):,} block-group rows",
            flush=True,
        )

    all_summaries: List[dict] = []
    all_bg: List[pd.DataFrame] = []
    job_started = time.time()

    for job_no, (state, method) in enumerate(jobs, 1):
        print("\n" + "=" * 100, flush=True)
        print(
            f"[JOB {job_no}/{len(jobs)}] {state.name} | {method} | 500 m",
            flush=True,
        )
        if not previous_summary.empty:
            old = previous_summary[
                (previous_summary["STATEFP"] == state.statefp)
                & (previous_summary["method"] == method)
                & (previous_summary["verdict"] == "PASS")
            ]
            old_detail = previous_detail[
                (previous_detail["STATEFP"] == state.statefp)
                & (previous_detail["method"] == method)
            ]
            if len(old) == 1 and not old_detail.empty:
                all_summaries.append(old.iloc[0].to_dict())
                all_bg.append(old_detail.copy())
                print(
                    f"[RESUME] Reusing completed PASS job with "
                    f"{len(old_detail):,} block-group rows",
                    flush=True,
                )
                continue
        inputs = input_map[(state.statefp, method)]
        blocks_layer = first_layer(inputs["blocks"])
        structures_layer = first_layer(inputs["structures"])
        feature_count, source_crs = layer_info(inputs["structures"], structures_layer)
        if not source_crs:
            raise RuntimeError(f"Structure source has no CRS: {inputs['structures']}")

        bg_df, geoid_field, pop_field = load_block_group_population(
            inputs["blocks"], blocks_layer
        )
        official = float(bg_df["POP20"].sum())
        expected_official = OFFICIAL_2020_POPULATION[state.statefp]
        if official != expected_official:
            raise RuntimeError(
                f"{state.name} POP20 total mismatch before allocation: "
                f"source={official:.0f}, expected={expected_official:.0f}"
            )
        print(
            f"[INPUT] block groups={len(bg_df):,}; {pop_field} total={official:,.0f}; "
            f"structure features={feature_count:,}",
            flush=True,
        )

        method_slug = method.lower().replace("-", "")
        bg_raster = cache_dir / f"bg_code_{state.statefp}_{method_slug}_aligned_500m.tif"
        build_bg_raster(
            inputs["blocks"], blocks_layer, geoid_field, inputs["wui"], bg_raster
        )
        counts = aggregate_structure_counts(
            inputs["structures"], structures_layer, source_crs, inputs["wui"], bg_raster,
            bg_df["BG_CODE"].to_numpy(dtype=np.int64),
            args.chunk_size, args.tile_size, feature_count, args.progress_every,
        )
        total, c0, c1, c2, point_stats = counts
        detail, summary = allocate_population(bg_df, total, c0, c1, c2)
        detail.insert(0, "method", method)
        detail.insert(0, "state_name", state.name)
        detail.insert(0, "STUSPS", state.stusps)
        detail.insert(0, "STATEFP", state.statefp)
        all_bg.append(detail)

        row = {
            "STATEFP": state.statefp,
            "STUSPS": state.stusps,
            "state_name": state.name,
            "method": method,
            "radius_m": 500,
            "allocation_unit": "2020 Census block group",
            "population_field": pop_field,
            "blocks_source": str(inputs["blocks"]),
            "structures_source": str(inputs["structures"]),
            "wui_raster": str(inputs["wui"]),
            "structure_features_expected": feature_count,
            **point_stats,
            "block_group_count": len(bg_df),
            **summary,
        }
        row["representative_point_coverage_pct"] = (
            row["points_processed"] / feature_count * 100.0 if feature_count else 100.0
        )
        row["block_group_match_pct"] = (
            row["points_matched_block_group"] / row["points_processed"] * 100.0
            if row["points_processed"] else 0.0
        )
        row["population_conservation_pass"] = (
            abs(row["allocation_residual"]) <= 1e-6
        )
        row["representative_point_coverage_pass"] = (
            row["representative_point_coverage_pct"] >= 99.0
        )
        row["block_group_match_pass"] = row["block_group_match_pct"] >= 99.0
        row["unexpected_wui_values_pass"] = row["unexpected_wui_values"] == 0
        row["verdict"] = (
            "PASS"
            if (
                row["population_conservation_pass"]
                and row["representative_point_coverage_pass"]
                and row["block_group_match_pass"]
                and row["unexpected_wui_values_pass"]
            )
            else "REVIEW"
        )
        summary["verdict"] = row["verdict"]
        all_summaries.append(row)
        print(
            f"[RESULT] {state.name} {method}: Non-WUI={summary['nonwui_population']:,.3f}; "
            f"Intermix={summary['intermix_population']:,.3f}; "
            f"Interface={summary['interface_population']:,.3f}; "
            f"WUI share={summary['wui_population_share_pct']:.4f}%; "
            f"residual={summary['allocation_residual']:.9f}; {summary['verdict']}",
            flush=True,
        )

        # Checkpoint after every job so a server interruption loses no finished work.
        pd.DataFrame(all_summaries).to_csv(
            summary_checkpoint, index=False, float_format="%.10f"
        )
        pd.concat(all_bg, ignore_index=True).to_csv(
            detail_checkpoint,
            index=False,
            float_format="%.12f",
        )

    summary_df = pd.DataFrame(all_summaries)
    detail_df = pd.concat(all_bg, ignore_index=True)
    summary_path = summary_checkpoint
    detail_path = detail_checkpoint
    report_path = qc_dir / "08_sample4_ps_population_qc_report.txt"
    summary_df.to_csv(summary_path, index=False, float_format="%.10f")
    detail_df.to_csv(detail_path, index=False, float_format="%.12f")

    passed = bool((summary_df["verdict"] == "PASS").all())
    lines = [
        "STEP 08 - CA/CO/FL/TX WUI-P/WUI-S 500 m population recomputation",
        "=" * 96,
        f"Generated: {time.ctime()}",
        "Source datasets modified: NO",
        "Population field: POP20",
        "Allocation: structure-based dasymetric allocation within 2020 Census block groups",
        "Zero-structure block groups: assigned to Non-WUI",
        f"Final verdict: {'PASS' if passed else 'FAIL'}",
        "",
    ]
    for _, row in summary_df.iterrows():
        lines.extend(
            [
                f"{row['state_name']} | {row['method']} | 500 m",
                "-" * 96,
                f"official_population: {row['official_population']:.0f}",
                f"allocated_population: {row['allocated_population']:.10f}",
                f"allocation_residual: {row['allocation_residual']:.10f}",
                f"Non-WUI population: {row['nonwui_population']:.10f}",
                f"Intermix population: {row['intermix_population']:.10f}",
                f"Interface population: {row['interface_population']:.10f}",
                f"WUI population: {row['wui_population']:.10f}",
                f"WUI population share: {row['wui_population_share_pct']:.10f}%",
                f"zero-structure BG count: {int(row['zero_structure_bg_count'])}",
                f"zero-structure BG population: {row['zero_structure_bg_population']:.10f}",
                f"structure points processed: {int(row['points_processed'])}",
                f"structure points matched to BG: {int(row['points_matched_block_group'])}",
                f"structure points unmatched to BG: {int(row['points_unmatched_block_group'])}",
                f"representative-point coverage: {row['representative_point_coverage_pct']:.10f}%",
                f"block-group match: {row['block_group_match_pct']:.10f}%",
                f"unexpected WUI values: {int(row['unexpected_wui_values'])}",
                f"verdict: {row['verdict']}",
                "",
            ]
        )
    lines.extend(
        [
            "PASS requirements:",
            "- Step 07 POP20/WUI-Z QC is PASS for every selected state.",
            "- Every job uses the validated POP20 block source for its own state.",
            "- One representative point is generated per usable address/building feature.",
            "- At least 99% of source features yield representative points.",
            "- At least 99% of representative points match a state Census block group.",
            "- No unexpected WUI raster values occur.",
            "- Population is allocated within Census block groups, not spread uniformly over land.",
            "- Zero-structure block-group population is retained as Non-WUI.",
            "- Non-WUI + Intermix + Interface equals official state POP20 within 1e-6 person.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n" + "=" * 100)
    print(f"[WROTE] {summary_path}")
    print(f"[WROTE] {detail_path}")
    print(f"[WROTE] {report_path}")
    print(f"[TOTAL ELAPSED] {hms(time.time() - job_started)}")
    print(f"[FINAL] {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 2


if __name__ == "__main__":
    sys.exit(main())