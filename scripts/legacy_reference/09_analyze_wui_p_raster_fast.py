#!/usr/bin/env python3
# Copyright (C) 2026 WUI Mapping Project Authors
# SPDX-License-Identifier: GPL-3.0-only
# -*- coding: utf-8 -*-
# Repository version: filesystem paths are loaded through repo_config.py.
# Copy config/paths.example.json to config/paths.json before a new run.
# Raw input data and large intermediate files are stored outside GitHub.

"""Legacy WUI-P raster script retained only for provenance.

This script is not the final address-only P2 workflow. It remains in the
repository because Step43 recorded it as part of the earlier production
lineage.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from repo_config import portable_path

import os
import math
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from scipy.ndimage import convolve
from skimage.morphology import disk
from tqdm import tqdm

# ============================================================
# Input and output locations
# ============================================================
# State-level OpenAddresses point files
ADDR_DIR = portable_path("legacy", "OpenAddresses_Work/Processed_GPKG")

# Existing WUI-S intermediate rasters (wildland_bin and dist_to_largepatch)
WUI_S_ROOT = portable_path("data", "WUI_S_Paper")

# WUI-P raster output directory
OUT_ROOT = portable_path("legacy", "WUI_P_Paper_Raster")
os.makedirs(OUT_ROOT, exist_ok=True)

TARGET_CRS = "EPSG:5070"

# ============================================================
# Classification thresholds used by this legacy script
# ============================================================
DENSITY_THRESHOLD = 6.17      # The rule is strictly greater than 6.17 units/km²
VEG_THRESHOLD = 0.50          # More than 50% vegetation gives intermix WUI
DIST_THRESHOLD_M = 2400.0     # within 2.4 km of >= 5 km^2 patch

# ============================================================
# Run 100–1000 m for five focus states and 500 m for other units
# ============================================================
SAMPLED_STATES = {
    "06": "California",
    "08": "Colorado",
    "12": "Florida",
    "42": "Pennsylvania",
    "48": "Texas",
}
SAMPLED_RADII_M = list(range(100, 1001, 100))
OTHER_RADII_M = [500]

# CONUS reporting units, excluding Alaska and Hawaii
STATE_FIPS = {
    '01': 'Alabama', '04': 'Arizona', '05': 'Arkansas', '06': 'California',
    '08': 'Colorado', '09': 'Connecticut', '10': 'Delaware', '11': 'DistrictofColumbia',
    '12': 'Florida', '13': 'Georgia', '16': 'Idaho', '17': 'Illinois',
    '18': 'Indiana', '19': 'Iowa', '20': 'Kansas', '21': 'Kentucky',
    '22': 'Louisiana', '23': 'Maine', '24': 'Maryland', '25': 'Massachusetts',
    '26': 'Michigan', '27': 'Minnesota', '28': 'Mississippi', '29': 'Missouri',
    '30': 'Montana', '31': 'Nebraska', '32': 'Nevada', '33': 'NewHampshire',
    '34': 'NewJersey', '35': 'NewMexico', '36': 'NewYork', '37': 'NorthCarolina',
    '38': 'NorthDakota', '39': 'Ohio', '40': 'Oklahoma', '41': 'Oregon',
    '42': 'Pennsylvania', '44': 'RhodeIsland', '45': 'SouthCarolina', '46': 'SouthDakota',
    '47': 'Tennessee', '48': 'Texas', '49': 'Utah', '50': 'Vermont',
    '51': 'Virginia', '53': 'Washington', '54': 'WestVirginia', '55': 'Wisconsin',
    '56': 'Wyoming'
}

# ============================================================
# Raster-tile parameters
# ============================================================
TILE_SIZE = 2048
# Padding covers the maximum neighborhood radius and 2.4 km distance rule.
# At 30 m resolution, 120 pixels are approximately 3.6 km.
PADDING_PX = 120


# ============================================================
# Helper functions
# ============================================================
def _find_wui_s_inputs(state_name: str):
    """Locate the WUI-S wildland mask and patch-distance rasters."""
    state_dir = os.path.join(WUI_S_ROOT, state_name)
    wild_bin = os.path.join(state_dir, f"{state_name}_wildland_bin.tif")
    dist_tif = os.path.join(state_dir, f"{state_name}_dist_to_largepatch.tif")

    if not os.path.exists(wild_bin):
        raise FileNotFoundError(f"wildland_bin not found: {wild_bin}")
    if not os.path.exists(dist_tif):
        raise FileNotFoundError(f"dist_to_largepatch not found: {dist_tif}")

    return wild_bin, dist_tif


def _find_addr_gpkg(state_name: str):
    """Locate one state-level OpenAddresses point file."""
    p = os.path.join(ADDR_DIR, f"{state_name}_addresses.gpkg")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Address GPKG not found: {p}")
    return p


def _load_address_xy(state_name: str):
    """Read address points, project to EPSG:5070, and return x/y pairs."""
    addr_path = _find_addr_gpkg(state_name)
    gdf = gpd.read_file(addr_path, engine="pyogrio")[["geometry"]]
    gdf = gdf[gdf.geometry.notna()]
    if gdf.crs is None or gdf.crs.to_string() != TARGET_CRS:
        gdf = gdf.to_crs(TARGET_CRS)

    # Keep Point geometries only.
    gdf = gdf[gdf.geometry.geom_type == "Point"]
    x = gdf.geometry.x.to_numpy(dtype=np.float64)
    y = gdf.geometry.y.to_numpy(dtype=np.float64)
    return np.column_stack([x, y])


def _build_point_tile_index(pts_xy, transform, width, height, tile_size):
    """
    Convert points to raster rows and columns and build a tile-to-slice index.
    Each tile then reads only nearby points instead of scanning the full file.
    """
    if pts_xy.size == 0:
        return None

    inv = ~transform
    cols_f, rows_f = inv * (pts_xy[:, 0], pts_xy[:, 1])
    cols = cols_f.astype(np.int64)
    rows = rows_f.astype(np.int64)

    ok = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
    cols = cols[ok]
    rows = rows[ok]

    x_steps = int(math.ceil(width / tile_size))

    tile_c = cols // tile_size
    tile_r = rows // tile_size
    tile_id = tile_r * x_steps + tile_c

    sort_idx = np.argsort(tile_id, kind="mergesort")
    tile_id_s = tile_id[sort_idx]
    cols_s = cols[sort_idx]
    rows_s = rows[sort_idx]

    uniq, start_idx, counts = np.unique(tile_id_s, return_index=True, return_counts=True)
    tile_ranges = {int(tid): (int(st), int(st + cnt)) for tid, st, cnt in zip(uniq, start_idx, counts)}

    return cols_s, rows_s, tile_ranges, x_steps


def _points_in_pad(cols_s, rows_s, tile_ranges, x_steps, tile_size, pad: Window):
    """Return local point rows and columns inside the padded window."""
    if cols_s is None:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    pad = pad.round_offsets().round_lengths()

    cmin = int(pad.col_off)
    rmin = int(pad.row_off)
    cmax = cmin + int(pad.width)
    rmax = rmin + int(pad.height)

    tc_min = max(0, cmin // tile_size)
    tr_min = max(0, rmin // tile_size)
    tc_max = max(0, (cmax - 1) // tile_size)
    tr_max = max(0, (rmax - 1) // tile_size)

    cs_list = []
    rs_list = []
    for tr in range(tr_min, tr_max + 1):
        for tc in range(tc_min, tc_max + 1):
            tid = tr * x_steps + tc
            if tid not in tile_ranges:
                continue
            st, en = tile_ranges[tid]
            cs_list.append(cols_s[st:en])
            rs_list.append(rows_s[st:en])

    if not cs_list:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    cols = np.concatenate(cs_list)
    rows = np.concatenate(rs_list)

    ok = (cols >= cmin) & (cols < cmax) & (rows >= rmin) & (rows < rmax)
    cols = cols[ok] - cmin
    rows = rows[ok] - rmin
    return rows.astype(np.int64), cols.astype(np.int64)


def _run_wui_p_raster_for_state(state_name: str, fips: str, radii_m):
    print("\n==============================")
    print(f"🌲 WUI-P raster (paper-like) | {state_name} ({fips}) | radii={radii_m}")
    print("==============================")

    out_state_dir = os.path.join(OUT_ROOT, state_name)
    os.makedirs(out_state_dir, exist_ok=True)

    wild_bin, dist_tif = _find_wui_s_inputs(state_name)

    # Read the address points once.
    print("   📍 load OpenAddresses points ...")
    pts_xy = _load_address_xy(state_name)
    print(f"   📌 points: {len(pts_xy):,}")

    # Open the wildland and distance rasters, which must share a grid.
    with rasterio.open(wild_bin) as wsrc, rasterio.open(dist_tif) as dsrc:
        if wsrc.crs is None or wsrc.crs.to_string() != TARGET_CRS:
            raise ValueError("wildland_bin must be EPSG:5070")
        if (dsrc.transform != wsrc.transform) or (dsrc.width != wsrc.width) or (dsrc.height != wsrc.height):
            raise ValueError("dist raster grid must match wildland_bin grid")

        pixel_size = float(abs(wsrc.transform.a))  # ~30m
        print(f"   🧱 grid: {wsrc.width} x {wsrc.height}, pixel={pixel_size}m")

        # Build the point index once to avoid repeated full-file scans.
        idx_pack = _build_point_tile_index(pts_xy, wsrc.transform, wsrc.width, wsrc.height, TILE_SIZE)
        if idx_pack is None:
            print("   ⚠️ no points in raster extent -> all Non-WUI")
            return
        cols_s, rows_s, tile_ranges, x_steps = idx_pack
        y_steps = int(math.ceil(wsrc.height / TILE_SIZE))
        total_tiles = x_steps * y_steps

        # Precompute the kernel and area for each radius.
        kernel_pack = {}
        for r in radii_m:
            radius_px = int(round(r / pixel_size))
            radius_px = max(radius_px, 1)
            kern = disk(radius_px).astype(np.float32)
            kern_sum = float(kern.sum())
            buffer_area_km2 = (math.pi * (r ** 2)) / 1_000_000.0
            kernel_pack[r] = (kern, kern_sum, buffer_area_km2)
            print(f"   🔧 r={r}m -> {radius_px}px kernel, cells={int(kern_sum)}")

        # Create one output raster for each radius.
        base_profile = wsrc.profile.copy()
        base_profile.update(
            dtype="uint8",
            count=1,
            nodata=0,          # 0 = Non-WUI
            compress="LZW",
            tiled=True,
            BIGTIFF="YES"
        )

        out_dsts = {}
        for r in radii_m:
            out_tif = os.path.join(out_state_dir, f"WUI_P_{state_name}_r{r:04d}m.tif")
            if os.path.exists(out_tif):
                print(f"   ⏩ exists, skip: {os.path.basename(out_tif)}")
                out_dsts[r] = None
            else:
                out_dsts[r] = rasterio.open(out_tif, "w", **base_profile)

        if all(v is None for v in out_dsts.values()):
            print("   ✅ all outputs already exist.")
            return

        full = Window(0, 0, wsrc.width, wsrc.height)

        pbar = tqdm(total=total_tiles, desc=f"   Tiles {state_name}", ncols=95)
        for ty in range(y_steps):
            for tx in range(x_steps):
                pbar.update(1)

                col_off = tx * TILE_SIZE
                row_off = ty * TILE_SIZE
                width = min(TILE_SIZE, wsrc.width - col_off)
                height = min(TILE_SIZE, wsrc.height - row_off)

                core = Window(col_off, row_off, width, height)

                pad = Window(
                    col_off - PADDING_PX,
                    row_off - PADDING_PX,
                    width + 2 * PADDING_PX,
                    height + 2 * PADDING_PX
                ).intersection(full).round_offsets().round_lengths()

                # Read the binary wildland mask and distance in meters.
                wild = wsrc.read(1, window=pad).astype(np.float32)
                dist = dsrc.read(1, window=pad).astype(np.float32)

                # Rasterize points inside the padded window as counts.
                rr, cc = _points_in_pad(cols_s, rows_s, tile_ranges, x_steps, TILE_SIZE, pad)
                s_count = np.zeros((int(pad.height), int(pad.width)), dtype=np.float32)
                if rr.size > 0:
                    np.add.at(s_count, (rr, cc), 1.0)

                # Locate the core window inside the padding.
                r0 = int(core.row_off - pad.row_off)
                c0 = int(core.col_off - pad.col_off)
                h0 = int(core.height)
                w0 = int(core.width)

                # Write one classification for each radius.
                for r in radii_m:
                    dst = out_dsts.get(r)
                    if dst is None:
                        continue

                    kern, kern_sum, buffer_area_km2 = kernel_pack[r]

                    s_sum = convolve(s_count, kern, mode="constant", cval=0.0)
                    wild_sum = convolve(wild, kern, mode="constant", cval=0.0)

                    density = s_sum / buffer_area_km2          # structures/km^2
                    veg_prop = wild_sum / kern_sum             # 0~1

                    # Apply the strict density > 6.17 rule.
                    dense = density > DENSITY_THRESHOLD
                    intermix = dense & (veg_prop > VEG_THRESHOLD)
                    interface = dense & (veg_prop <= VEG_THRESHOLD) & (dist <= DIST_THRESHOLD_M)

                    # 0=Non-WUI, 1=Intermix, 2=Interface
                    wui_core = np.zeros((h0, w0), dtype=np.uint8)
                    wui_core[intermix[r0:r0+h0, c0:c0+w0]] = 1
                    wui_core[interface[r0:r0+h0, c0:c0+w0]] = 2

                    dst.write(wui_core, 1, window=core)

        pbar.close()

        for r, dst in out_dsts.items():
            if dst is not None:
                dst.close()

    print(f"   ✅ finished {state_name}")


def main():
    # Optional debugging filter for selected state FIPS codes.
    RUN_ONLY = None  # Example: {"17", "55"}; None runs every unit.

    items = list(STATE_FIPS.items())
    if RUN_ONLY is not None:
        items = [it for it in items if it[0] in RUN_ONLY]

    for fips, name in items:
        # Use the full radius series for focus states and 500 m elsewhere.
        radii = SAMPLED_RADII_M if fips in SAMPLED_STATES else OTHER_RADII_M
        try:
            _run_wui_p_raster_for_state(name, fips, radii)
        except Exception as e:
            print(f"   ❌ {name} failed: {e}")
            continue

    print("\n🎉 All done.")
    print(f"Outputs in: {OUT_ROOT}")


if __name__ == "__main__":
    main()
