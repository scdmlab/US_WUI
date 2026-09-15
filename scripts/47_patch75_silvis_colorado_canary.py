#!/usr/bin/env python3
# Copyright (C) 2026 WUI Mapping Project Authors
# SPDX-License-Identifier: GPL-3.0-only
# Repository version: filesystem paths are loaded through repo_config.py.
# Copy config/paths.example.json to config/paths.json before a new run.
# Raw input data and large intermediate files are stored outside GitHub.

"""Colorado canary for a SILVIS-style >75% large-patch definition.

This script is diagnostic only.  It does not modify any Step43--46 input or
output.  The canary compares:

Q0_CURRENT_BINARY_PATCH
    Existing >=5 km2 eight-connected component of binary NLCD wildland cells.
Q1_SILVIS_BLOCK_GT75_PATCH
    Census blocks with Veg_Percent > 75 are dissolved; contiguous dissolved
    components with area >=5 km2 are retained and buffered by 2,400 m.

Both scenarios use a 500 m circular raster neighborhood, D > 6.17, intermix
V >= 50%, and interface V < 50% with distance <=2,400 m.
"""

from __future__ import annotations

from repo_config import portable_path

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from scipy.signal import fftconvolve
from shapely import get_parts, union_all
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from skimage.morphology import disk


ROOT = Path(portable_path("project"))
WUIP = (
    ROOT
    / "step43_wuip_p2_formal_rebuild_20260727T180822Z"
    / "rasters_500m/CO/WUI_P_P2_CO_r0500m.tif"
)
WUIP_SPARSE = (
    ROOT
    / "step43_wuip_p2_formal_rebuild_20260727T180822Z"
    / "intermediate/CO_p2_sparse_cells.npz"
)
WUIS = Path(
    portable_path("data", "WUI_S_Paper/Colorado/WUI_S_Colorado_r0500m.tif")
)
WILD = Path(
    portable_path("data", "WUI_S_Paper/Colorado/Colorado_wildland_bin.tif")
)
CURRENT_LARGE_PATCH = Path(
    portable_path("data", "WUI_S_Paper/Colorado/Colorado_wildland_largepatch.tif")
)
CURRENT_DISTANCE = Path(
    portable_path("data", "WUI_S_Paper/Colorado/Colorado_dist_to_largepatch.tif")
)
BLOCKS = Path(
    portable_path("data", "WUI_Z_Results/WUI_Z_Paper_Colorado.gpkg")
)
BLOCK_LAYER = "WUI_Z_Paper_Colorado"
MBF_CENTROIDS = Path(
    portable_path("data", "mbf_work/centroids_5070/MBF_Colorado_centroids_5070.gpkg")
)
MBF_LAYER = "centroids"

PIXEL_M = 30.0
RADIUS_M = 500
RADIUS_PX = int(round(RADIUS_M / PIXEL_M))
DENSITY_THRESHOLD = 6.17
MINIMUM_COUNT = int(
    math.floor(DENSITY_THRESHOLD * math.pi * RADIUS_M**2 / 1_000_000.0)
) + 1
PATCH_THRESHOLD = 75.0
PATCH_AREA_M2 = 5_000_000.0
DISTANCE_M = 2_400.0
NODATA = 255


def now_utc() -> str:
    import datetime as dt

    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def progress(stage: str, done: int, total: int, started: float) -> None:
    elapsed = time.perf_counter() - started
    pct = 100.0 * done / total if total else 100.0
    eta = elapsed * (total - done) / done if done else float("nan")
    print(
        f"[{stage}] completed={done}/{total} percent={pct:.2f} "
        f"elapsed_s={elapsed:.1f} ETA_s={eta:.1f}",
        flush=True,
    )


def polygon_parts(geom) -> list[Polygon]:
    """Flatten polygonal parts from a union result."""
    result: list[Polygon] = []
    stack = [geom]
    while stack:
        item = stack.pop()
        if item is None or item.is_empty:
            continue
        if isinstance(item, Polygon):
            result.append(item)
        elif isinstance(item, (MultiPolygon, GeometryCollection)):
            stack.extend(list(get_parts(item)))
    return result


def write_raster(path: Path, array: np.ndarray, ref, nodata: int = 0) -> None:
    profile = ref.profile.copy()
    profile.update(
        driver="GTiff",
        dtype="uint8",
        count=1,
        nodata=nodata,
        compress="LZW",
        tiled=True,
        BIGTIFF="YES",
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(np.uint8), 1)


def build_wuis_count_raster(output: Path, ref) -> tuple[Path, dict]:
    started = time.perf_counter()
    left, bottom, right, top = ref.bounds
    command = [
        portable_path("software", "bin/gdal_rasterize"),
        "-l",
        MBF_LAYER,
        "-burn",
        "1",
        "-add",
        "-init",
        "0",
        "-a_nodata",
        "0",
        "-ot",
        "UInt32",
        "-of",
        "GTiff",
        "-te",
        str(left),
        str(bottom),
        str(right),
        str(top),
        "-ts",
        str(ref.width),
        str(ref.height),
        "-co",
        "COMPRESS=LZW",
        "-co",
        "TILED=YES",
        "-co",
        "BIGTIFF=YES",
        str(MBF_CENTROIDS),
        str(output),
    ]
    subprocess.run(command, check=True)
    count_sum = 0
    sparse_cells = 0
    with rasterio.open(output) as src:
        if (
            src.width != ref.width
            or src.height != ref.height
            or src.transform != ref.transform
        ):
            raise RuntimeError("WUI-S count raster grid mismatch")
        for _, window in src.block_windows(1):
            array = src.read(1, window=window)
            count_sum += int(array.sum(dtype=np.uint64))
            sparse_cells += int((array > 0).sum())
    meta = {
        "source_building_centroids": str(MBF_CENTROIDS),
        "count_raster": str(output),
        "sparse_cells": sparse_cells,
        "count_sum": count_sum,
        "runtime_seconds": time.perf_counter() - started,
    }
    return output, meta


def build_patch75_buffer(output: Path, ref) -> tuple[np.ndarray, dict]:
    started = time.perf_counter()
    blocks_all = gpd.read_file(
        BLOCKS,
        layer=BLOCK_LAYER,
        engine="fiona",
        columns=["Veg_Percent"],
    )
    if blocks_all.crs is None or blocks_all.crs.to_epsg() != 5070:
        blocks_all = blocks_all.to_crs("EPSG:5070")
    veg = blocks_all["Veg_Percent"].to_numpy(dtype=float)
    exact_75 = int(np.isclose(veg, PATCH_THRESHOLD, rtol=0, atol=1e-12).sum())
    selected = blocks_all.loc[veg > PATCH_THRESHOLD, ["geometry"]].copy()
    selected_area = float(selected.geometry.area.sum())
    del blocks_all
    t_read = time.perf_counter() - started
    print(
        f"[PATCH75] blocks_selected={len(selected)} exact_75={exact_75} "
        f"read_s={t_read:.1f}",
        flush=True,
    )

    union_started = time.perf_counter()
    dissolved = union_all(selected.geometry.to_numpy())
    parts = polygon_parts(dissolved)
    qualifying = [geom for geom in parts if geom.area >= PATCH_AREA_M2]
    dissolved_area = float(sum(geom.area for geom in parts))
    qualifying_area = float(sum(geom.area for geom in qualifying))
    t_union = time.perf_counter() - union_started
    print(
        f"[PATCH75] dissolved_components={len(parts)} "
        f"qualifying_components={len(qualifying)} union_s={t_union:.1f}",
        flush=True,
    )

    buffer_started = time.perf_counter()
    buffered = [geom.buffer(DISTANCE_M) for geom in qualifying]
    buffer_mask = rasterize(
        [(geom, 1) for geom in buffered],
        out_shape=(ref.height, ref.width),
        transform=ref.transform,
        fill=0,
        all_touched=False,
        dtype="uint8",
    )
    write_raster(output, buffer_mask, ref, nodata=0)
    t_buffer = time.perf_counter() - buffer_started
    meta = {
        "source_blocks": str(BLOCKS),
        "source_block_count": int(len(selected) + (len(veg) - len(selected))),
        "selected_gt75_blocks": int(len(selected)),
        "exact_75_blocks_excluded": exact_75,
        "selected_block_area_km2": selected_area / 1_000_000.0,
        "dissolved_component_count": int(len(parts)),
        "dissolved_area_km2": dissolved_area / 1_000_000.0,
        "qualifying_component_count": int(len(qualifying)),
        "qualifying_component_area_km2": qualifying_area / 1_000_000.0,
        "buffer_mask_pixels": int(buffer_mask.sum()),
        "buffer_mask_area_km2": float(buffer_mask.sum()) * 0.0009,
        "read_runtime_seconds": t_read,
        "union_runtime_seconds": t_union,
        "buffer_rasterize_runtime_seconds": t_buffer,
        "total_runtime_seconds": time.perf_counter() - started,
    }
    return buffer_mask, meta


def tile_sparse(
    sparse_rows: np.ndarray,
    sparse_cols: np.ndarray,
    sparse_counts: np.ndarray,
    pad: Window,
) -> np.ndarray:
    row0 = int(pad.row_off)
    col0 = int(pad.col_off)
    row1 = row0 + int(pad.height)
    col1 = col0 + int(pad.width)
    selected = (
        (sparse_rows >= row0)
        & (sparse_rows < row1)
        & (sparse_cols >= col0)
        & (sparse_cols < col1)
    )
    array = np.zeros((int(pad.height), int(pad.width)), dtype=np.float32)
    if selected.any():
        rr = sparse_rows[selected] - row0
        cc = sparse_cols[selected] - col0
        array[rr, cc] = sparse_counts[selected]
    return array


def iter_windows(width: int, height: int, tile: int = 2048):
    for row in range(0, height, tile):
        for col in range(0, width, tile):
            yield Window(
                col,
                row,
                min(tile, width - col),
                min(tile, height - row),
            )


def padded_window(core: Window, width: int, height: int, pad: int) -> Window:
    col0 = max(0, int(core.col_off) - pad)
    row0 = max(0, int(core.row_off) - pad)
    col1 = min(width, int(core.col_off + core.width) + pad)
    row1 = min(height, int(core.row_off + core.height) + pad)
    return Window(col0, row0, col1 - col0, row1 - row0)


def init_stats() -> dict:
    return {
        "valid_pixels": 0,
        "old_non_wui": 0,
        "old_intermix": 0,
        "old_interface": 0,
        "new_non_wui": 0,
        "new_intermix": 0,
        "new_interface": 0,
        "changed_pixels": 0,
        "nonwui_to_interface": 0,
        "interface_to_nonwui": 0,
        "intermix_changed": 0,
        "old_wui": 0,
        "new_wui": 0,
        "intersection": 0,
        "union": 0,
        "baseline_mismatch_pixels": 0,
    }


def classify(
    output: Path,
    method: str,
    sparse_rows: np.ndarray | None,
    sparse_cols: np.ndarray | None,
    sparse_counts: np.ndarray | None,
    count_raster: Path | None,
    new_buffer: np.ndarray | Path,
    ref,
    kernel: np.ndarray,
    radius_px: int = RADIUS_PX,
    minimum_count: int = MINIMUM_COUNT,
    official_path_override: Path | None = None,
) -> dict:
    started = time.perf_counter()
    profile = ref.profile.copy()
    profile.update(
        driver="GTiff",
        dtype="uint8",
        count=1,
        nodata=NODATA,
        compress="LZW",
        tiled=True,
        BIGTIFF="YES",
    )
    official_path = (
        official_path_override
        if official_path_override is not None
        else (WUIP if method == "WUI-P" else WUIS)
    )
    stats = init_stats()
    windows = list(iter_windows(ref.width, ref.height))
    count_source = rasterio.open(count_raster) if count_raster else None
    patch_context = (
        rasterio.open(new_buffer)
        if isinstance(new_buffer, (str, Path))
        else nullcontext(None)
    )
    with (
        rasterio.open(WILD) as wild_src,
        rasterio.open(CURRENT_DISTANCE) as distance_src,
        rasterio.open(official_path) as official_src,
        patch_context as patch_source,
        rasterio.open(output, "w", **profile) as dst,
    ):
        for i, core in enumerate(windows, 1):
            pad = padded_window(
                core, ref.width, ref.height, radius_px
            )
            wild_pad = wild_src.read(1, window=pad).astype(np.float32)
            if count_source is not None:
                points = count_source.read(1, window=pad).astype(np.float32)
            else:
                if (
                    sparse_rows is None
                    or sparse_cols is None
                    or sparse_counts is None
                ):
                    raise RuntimeError("Missing sparse point-count inputs")
                points = tile_sparse(
                    sparse_rows, sparse_cols, sparse_counts, pad
                )
            count_sum = np.rint(
                fftconvolve(points, kernel, mode="same")
            ).astype(np.int32)
            wild_sum = np.rint(
                fftconvolve(wild_pad, kernel, mode="same")
            ).astype(np.int32)
            r0 = int(core.row_off - pad.row_off)
            c0 = int(core.col_off - pad.col_off)
            h = int(core.height)
            w = int(core.width)
            counts = count_sum[r0 : r0 + h, c0 : c0 + w]
            vegetation = wild_sum[r0 : r0 + h, c0 : c0 + w]
            dense = counts >= minimum_count
            # Kernel sum is odd; exactly 50% cannot occur for binary wildland.
            twice_vegetation = vegetation * 2
            intermix = dense & (twice_vegetation >= int(kernel.sum()))
            low_vegetation = dense & (
                twice_vegetation < int(kernel.sum())
            )
            distance = distance_src.read(1, window=core)
            old_interface = low_vegetation & (distance <= DISTANCE_M)
            if patch_source is not None:
                new_patch = patch_source.read(1, window=core) == 1
            else:
                new_patch = (
                    new_buffer[
                        int(core.row_off) : int(
                            core.row_off + core.height
                        ),
                        int(core.col_off) : int(
                            core.col_off + core.width
                        ),
                    ]
                    == 1
                )
            new_interface = low_vegetation & new_patch
            old = np.zeros((h, w), dtype=np.uint8)
            new = np.zeros((h, w), dtype=np.uint8)
            old[intermix] = 1
            old[old_interface] = 2
            new[intermix] = 1
            new[new_interface] = 2
            official = official_src.read(1, window=core)
            # Use the frozen WUI-P state mask for both methods because some
            # historical WUI-S rasters encoded exterior pixels as zero.
            domain = ref.read(1, window=core) != NODATA
            out = new.copy()
            out[~domain] = NODATA
            dst.write(out, 1, window=core)

            old_wui = old > 0
            new_wui = new > 0
            stats["valid_pixels"] += int(domain.sum())
            stats["old_non_wui"] += int(((old == 0) & domain).sum())
            stats["old_intermix"] += int(((old == 1) & domain).sum())
            stats["old_interface"] += int(((old == 2) & domain).sum())
            stats["new_non_wui"] += int(((new == 0) & domain).sum())
            stats["new_intermix"] += int(((new == 1) & domain).sum())
            stats["new_interface"] += int(((new == 2) & domain).sum())
            stats["changed_pixels"] += int(((old != new) & domain).sum())
            stats["nonwui_to_interface"] += int(
                ((old == 0) & (new == 2) & domain).sum()
            )
            stats["interface_to_nonwui"] += int(
                ((old == 2) & (new == 0) & domain).sum()
            )
            stats["intermix_changed"] += int(
                (((old == 1) ^ (new == 1)) & domain).sum()
            )
            stats["old_wui"] += int((old_wui & domain).sum())
            stats["new_wui"] += int((new_wui & domain).sum())
            stats["intersection"] += int(
                (old_wui & new_wui & domain).sum()
            )
            stats["union"] += int(
                ((old_wui | new_wui) & domain).sum()
            )
            stats["baseline_mismatch_pixels"] += int(
                ((old != official) & domain).sum()
            )
            if i == 1 or i % 10 == 0 or i == len(windows):
                progress(f"CLASSIFY_{method}", i, len(windows), started)
    if count_source is not None:
        count_source.close()
    stats["runtime_seconds"] = time.perf_counter() - started
    stats["jaccard_new_vs_old"] = (
        stats["intersection"] / stats["union"]
        if stats["union"]
        else 1.0
    )
    stats["old_wui_area_km2"] = stats["old_wui"] * 0.0009
    stats["new_wui_area_km2"] = stats["new_wui"] * 0.0009
    stats["change_area_km2"] = stats["changed_pixels"] * 0.0009
    stats["net_wui_area_change_km2"] = (
        stats["new_wui"] - stats["old_wui"]
    ) * 0.0009
    stats["net_wui_change_percent"] = (
        100.0
        * (stats["new_wui"] - stats["old_wui"])
        / stats["old_wui"]
        if stats["old_wui"]
        else 0.0
    )
    return stats


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    created_utc = now_utc()

    required = [
        WUIP,
        WUIP_SPARSE,
        WUIS,
        WILD,
        CURRENT_LARGE_PATCH,
        CURRENT_DISTANCE,
        BLOCKS,
        MBF_CENTROIDS,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing inputs: {missing}")

    with rasterio.open(WUIP) as ref:
        patch_path = output / "CO_silvis_block_gt75_buffer_2400m.tif"
        new_buffer, patch_meta = build_patch75_buffer(
            patch_path, ref
        )

        # Patch-zone comparison is restricted to the formal state domain.
        old_patch_pixels = 0
        new_patch_pixels = 0
        patch_intersection = 0
        patch_union = 0
        current_large_patch_pixels = 0
        with (
            rasterio.open(CURRENT_DISTANCE) as old_dist,
            rasterio.open(CURRENT_LARGE_PATCH) as old_large,
        ):
            for core in iter_windows(ref.width, ref.height):
                official = ref.read(1, window=core)
                domain = official != NODATA
                old = old_dist.read(1, window=core) <= DISTANCE_M
                large = old_large.read(1, window=core) == 1
                row0, col0 = int(core.row_off), int(core.col_off)
                row1 = row0 + int(core.height)
                col1 = col0 + int(core.width)
                new = new_buffer[row0:row1, col0:col1] == 1
                old_patch_pixels += int((old & domain).sum())
                new_patch_pixels += int((new & domain).sum())
                patch_intersection += int((old & new & domain).sum())
                patch_union += int(((old | new) & domain).sum())
                current_large_patch_pixels += int((large & domain).sum())
        patch_meta.update(
            {
                "current_binary_large_patch_pixels": current_large_patch_pixels,
                "current_binary_large_patch_area_km2": (
                    current_large_patch_pixels * 0.0009
                ),
                "current_buffer_pixels_in_domain": old_patch_pixels,
                "new_buffer_pixels_in_domain": new_patch_pixels,
                "buffer_intersection_pixels": patch_intersection,
                "buffer_union_pixels": patch_union,
                "buffer_jaccard": (
                    patch_intersection / patch_union
                    if patch_union
                    else 1.0
                ),
            }
        )

        p = np.load(WUIP_SPARSE)
        p_rows = p["rows"].astype(np.int32)
        p_cols = p["cols"].astype(np.int32)
        p_counts = p["counts"].astype(np.uint32)
        s_count_raster, s_meta = build_wuis_count_raster(
            output / "WUI_S_CO_centroid_count_30m.tif", ref
        )
        kernel = disk(RADIUS_PX).astype(np.float32)
        if int(kernel.sum()) % 2 != 1:
            raise RuntimeError("Expected odd circular-kernel cell count")

        p_out = output / "WUI_P_CO_r0500m_SILVIS_GT75_CANARY.tif"
        s_out = output / "WUI_S_CO_r0500m_SILVIS_GT75_CANARY.tif"
        p_stats = classify(
            p_out,
            "WUI-P",
            p_rows,
            p_cols,
            p_counts,
            None,
            new_buffer,
            ref,
            kernel,
        )
        s_stats = classify(
            s_out,
            "WUI-S",
            None,
            None,
            None,
            s_count_raster,
            new_buffer,
            ref,
            kernel,
        )

    summary_rows = []
    for method, stats in (("WUI-P", p_stats), ("WUI-S", s_stats)):
        summary_rows.append(
            {
                "state": "CO",
                "method": method,
                "radius_m": RADIUS_M,
                "density_rule": "D>6.17",
                "minimum_integer_count": MINIMUM_COUNT,
                "intermix_rule": "V>=50%",
                "interface_rule": "V<50% and distance<=2400m",
                "new_patch_rule": (
                    "Census blocks Veg_Percent>75%; dissolve contiguous; "
                    "component area>=5km2"
                ),
                **stats,
            }
        )
    write_csv(output / "patch75_classification_impact.csv", summary_rows)
    write_csv(output / "patch75_layer_summary.csv", [patch_meta])
    write_csv(output / "wuis_source_summary.csv", [s_meta])

    baseline_pass = all(
        row["baseline_mismatch_pixels"] == 0 for row in (p_stats, s_stats)
    )
    elapsed = time.perf_counter() - started
    status = {
        "step": "STEP47_PATCH75_SILVIS_COLORADO_CANARY",
        "status": (
            "COLORADO_PATCH75_CANARY_COMPLETE"
            if baseline_pass
            else "BASELINE_REPRODUCTION_MISMATCH_REVIEW_REQUIRED"
        ),
        "created_utc": created_utc,
        "completed_utc": now_utc(),
        "state": "CO",
        "radius_m": RADIUS_M,
        "baseline_reproduction_pass": baseline_pass,
        "p_baseline_mismatch_pixels": p_stats[
            "baseline_mismatch_pixels"
        ],
        "s_baseline_mismatch_pixels": s_stats[
            "baseline_mismatch_pixels"
        ],
        "elapsed_seconds": elapsed,
        "environment_modified": False,
        "upstream_modified": False,
        "output_directory": str(output),
    }
    (output / "step47_status.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )

    readme = f"""# Step47 Colorado >75% patch canary

This diagnostic canary compares the frozen binary-wildland patch construction
with a SILVIS-style block-based patch construction for Colorado at 500 m.

## Fixed rules

- development density: `D > 6.17` (minimum integer count {MINIMUM_COUNT});
- intermix: local wildland vegetation `V >= 50%`;
- interface: `V < 50%` and distance `<= 2,400 m`;
- circular neighborhood radius: 500 m;
- qualifying patch: Census blocks with `Veg_Percent > 75%`, dissolved by
  spatial contiguity, retaining dissolved components with area `>= 5 km2`.

The official SILVIS production metadata uses `>=75%`; this canary uses the
strict `>75%` rule requested for the manuscript equation.  Blocks exactly at
75% are counted separately in `patch75_layer_summary.csv`.

## Safety

No Step43--46 raster or source dataset was modified.  The output rasters are
diagnostic only.

## Baseline gate

The old-patch reconstruction must equal the frozen WUI-P and WUI-S 500 m
classification inside the formal Colorado domain before the impact estimate
is accepted.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")

    qc_lines = [
        "STEP47 PATCH75 COLORADO CANARY QC",
        f"completed_utc={status['completed_utc']}",
        f"baseline_reproduction_pass={baseline_pass}",
        f"p_baseline_mismatch_pixels={p_stats['baseline_mismatch_pixels']}",
        f"s_baseline_mismatch_pixels={s_stats['baseline_mismatch_pixels']}",
        f"kernel_cells={int(kernel.sum())}",
        f"kernel_cells_odd={int(kernel.sum()) % 2 == 1}",
        f"elapsed_seconds={elapsed:.3f}",
        f"final_status={status['status']}",
    ]
    (output / "STEP47_FINAL_QC.txt").write_text(
        "\n".join(qc_lines) + "\n", encoding="utf-8"
    )

    manifest_files = sorted(
        path
        for path in output.iterdir()
        if path.is_file() and path.name != "sha256_manifest.txt"
    )
    with (output / "sha256_manifest.txt").open(
        "w", encoding="utf-8"
    ) as f:
        for path in manifest_files:
            f.write(f"{sha256(path)}  {path.name}\n")
    print(json.dumps(status, indent=2), flush=True)
    return 0 if baseline_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
