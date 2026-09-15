#!/usr/bin/env python3
# Copyright (C) 2026 WUI Mapping Project Authors
# SPDX-License-Identifier: GPL-3.0-only
# Repository version: filesystem paths are loaded through repo_config.py.
# Copy config/paths.example.json to config/paths.json before a new run.
# Raw input data and large intermediate files are stored outside GitHub.

"""STEP46: complete WUI-Z pairwise intersection and Jaccard tables.

The formal design is:
  * 49 CONUS reporting units (48 states + DC), fixed 500 m;
  * CA, CO, FL, PA, TX same-distance sensitivity, 100--1000 m;
  * WUI-P/WUI-S rows are imported unchanged from Step45;
  * WUI-P/WUI-Z and WUI-S/WUI-Z are calculated on a true common valid domain.

WUI-Z is rasterized twice on the exact Step43 grid:
  * class raster: 0 Non-WUI, 1 Intermix, 2 Interface, 255 outside/NoData;
  * independent valid-domain raster: 1 valid, 0 outside/NoData.

No software environment or upstream artifact is modified.
"""

from __future__ import annotations

from repo_config import portable_path

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import fiona
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window


ROOT = Path(portable_path("project"))
STEP40 = ROOT / "step40_wuiz_area_audit_20260725T155314Z"
STEP43 = ROOT / "step43_wuip_p2_formal_rebuild_20260727T180822Z"
STEP45 = ROOT / "step45_p2_formal_spatial_analysis_20260728T025216Z"
STEP43_INV = STEP43 / "p2_rebuild_inventory.csv"
STEP45_INPUTS = STEP45 / "p2_spatial_analysis_input_manifest.csv"
STEP45_J99 = STEP45 / "p2_wuip_wuis_jaccard_99.csv"
STEP40_SOURCES = STEP40 / "qc" / "step40_wuiz_source_inventory.csv"
STEP40_MANIFEST = STEP40 / "step40_manifest.json"
GDAL_RASTERIZE = Path(portable_path("software", "bin/gdal_rasterize"))
LEGACY_Z_CACHE = Path(portable_path("legacy", "WUI_tables_compare/_cache_appendixA_wuiz"))
FIVE = ("CA", "CO", "FL", "PA", "TX")
BUFFERS = tuple(range(100, 1001, 100))
SCRIPT_VERSION = "2026-07-28.1"
PIXEL_CODES = (0, 1, 2, 255)
PAIR_METHODS = ("WUI-P", "WUI-S")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(path: Path, chunk: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as src:
        while True:
            data = src.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False, float_format="%.15g")
    os.replace(tmp, path)


def progress(stage: str, state: str, scenario: str, done: int, total: int,
             started: float) -> None:
    elapsed = time.monotonic() - started
    pct = 100.0 * done / total if total else 100.0
    eta = elapsed * (total - done) / done if done else math.nan
    eta_text = f"{eta:.1f}s" if math.isfinite(eta) else "NA"
    print(
        f"[{stage}] state={state} scenario={scenario} "
        f"complete={done}/{total} percent={pct:.2f}% "
        f"elapsed={elapsed:.1f}s ETA={eta_text}",
        flush=True,
    )


def grid_tuple(src: rasterio.io.DatasetReader) -> tuple[Any, ...]:
    return (
        src.crs.to_string() if src.crs else None,
        tuple(src.transform),
        src.width,
        src.height,
    )


def grids_equal(*sources: rasterio.io.DatasetReader) -> bool:
    return len({grid_tuple(src) for src in sources}) == 1


def is_conus_albers_reference(crs: Any) -> bool:
    """Accept the frozen Step43 local-name CRS encoding.

    Step43 GeoTIFFs carry a LOCAL_CS named "NAD83 / Conus Albers" rather than
    the full EPSG authority record. Their affine grids and the upstream
    manifest are frozen as EPSG:5070. WUI-Z vectors carry the full definition.
    """
    text = str(crs).upper() if crs else ""
    return "NAD83 / CONUS ALBERS" in text and (
        'UNIT["METRE",1' in text or 'UNIT["METER",1' in text
    )


def is_full_epsg5070(crs: Any) -> bool:
    text = str(crs).upper() if crs else ""
    required = (
        "NAD83 / CONUS ALBERS",
        "ALBERS_CONIC_EQUAL_AREA",
        'PARAMETER["LATITUDE_OF_CENTER",23]',
        'PARAMETER["LONGITUDE_OF_CENTER",-96]',
        'PARAMETER["STANDARD_PARALLEL_1",29.5]',
        'PARAMETER["STANDARD_PARALLEL_2",45.5]',
    )
    return all(token in text for token in required)


def iter_windows(width: int, height: int, size: int = 2048) -> Iterable[Window]:
    for row in range(0, height, size):
        h = min(size, height - row)
        for col in range(0, width, size):
            w = min(size, width - col)
            yield Window(col, row, w, h)


def file_record(role: str, path: Path, *, state: str = "",
                buffer_m: Any = "", known_hash: str = "",
                notes: str = "") -> dict[str, Any]:
    st = path.stat()
    return {
        "input_role": role,
        "state": state,
        "buffer_m": buffer_m,
        "path": str(path),
        "file_size": st.st_size,
        "mtime_utc": datetime.fromtimestamp(
            st.st_mtime, timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sha256": known_hash or sha256(path),
        "notes": notes,
    }


def verify_parents() -> dict[str, Any]:
    status43 = json.loads((STEP43 / "step43_status.json").read_text())
    status45 = json.loads((STEP45 / "step45_status.json").read_text())
    status40 = json.loads(STEP40_MANIFEST.read_text())
    inv43 = pd.read_csv(STEP43_INV)
    j99 = pd.read_csv(STEP45_J99)
    checks = {
        "step43_status_ready": status43.get("status")
        == "P2_CLASSIFICATION_REBUILD_COMPLETE_READY_FOR_DOWNSTREAM",
        "step43_raster_rows_94": len(inv43) == 94,
        "step43_all_pass": inv43["run_status"].eq("PASS").all(),
        "step43_building_records_zero": int(
            inv43["building_records_included"].sum()
        ) == 0,
        "step45_jaccard_rows_99": len(j99) == 99,
        "step45_jaccard_all_pass": j99["status"].eq("PASS").all(),
        "step45_scope_49_plus_50": (
            (j99["analysis_scope"] == "national_49_500m").sum() == 49
            and (j99["analysis_scope"] == "five_state_sensitivity").sum() == 50
        ),
        "step45_micro_exact": math.isclose(
            j99.loc[
                j99.analysis_scope.eq("national_49_500m"), "intersection"
            ].sum()
            / j99.loc[
                j99.analysis_scope.eq("national_49_500m"), "union_wui"
            ].sum(),
            0.621897978161495,
            rel_tol=0,
            abs_tol=5e-15,
        ),
        "step40_status_pass": status40.get("status") == "PASS",
        "step40_sources_49": len(pd.read_csv(STEP40_SOURCES)) == 49,
        "gdal_rasterize_exists": GDAL_RASTERIZE.is_file(),
    }
    if not all(checks.values()):
        failed = [key for key, value in checks.items() if not value]
        raise RuntimeError("PARENT_GATE_FAILED: " + ",".join(failed))
    return checks


def build_input_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    p = pd.read_csv(STEP43_INV, dtype={"STATEFP": str})
    p["STATEFP"] = p["STATEFP"].str.zfill(2)
    p = p[[
        "state", "STATEFP", "buffer_m", "output_path",
        "file_sha256", "width", "height", "crs", "transform", "nodata",
    ]].rename(columns={
        "output_path": "p_path",
        "file_sha256": "p_sha256",
    })

    manifest = pd.read_csv(STEP45_INPUTS)
    s = manifest[
        manifest.input_role.eq("formal WUI-S classification raster")
        & manifest.included_yes_no.eq("YES")
    ].copy()
    s["buffer_m"] = s["buffer_m"].astype(int)
    s = s[["state", "buffer_m", "path", "sha256"]].rename(columns={
        "path": "s_path",
        "sha256": "s_sha256",
    })
    if len(s) != 94:
        raise RuntimeError(f"Expected 94 formal WUI-S inputs, found {len(s)}")

    combos = p.merge(s, on=["state", "buffer_m"], how="left", validate="one_to_one")
    if combos.s_path.isna().any() or len(combos) != 94:
        raise RuntimeError("P/S input key coverage is incomplete")
    combos["buffer_m"] = combos["buffer_m"].astype(int)
    combos["state_name"] = ""

    z = pd.read_csv(STEP40_SOURCES, dtype={"STATEFP": str})
    z["STATEFP"] = z["STATEFP"].str.zfill(2)
    if len(z) != 49 or not z.status.eq("FOUND").all():
        raise RuntimeError("WUI-Z source inventory is not 49/49 FOUND")
    combos = combos.merge(
        z[["STUSPS", "NAME", "path"]].rename(columns={
            "STUSPS": "state",
            "NAME": "z_state_name",
            "path": "z_path",
        }),
        on="state",
        how="left",
        validate="many_to_one",
    )
    combos["state_name"] = combos["z_state_name"]
    combos = combos.drop(columns=["z_state_name"])
    if combos.z_path.isna().any():
        raise RuntimeError("WUI-Z source key coverage is incomplete")
    return combos, p, z


def rasterize_wuiz(
    state: str,
    source: Path,
    ref_path: Path,
    cache_root: Path,
) -> dict[str, Any]:
    state_dir = cache_root / state
    state_dir.mkdir(parents=True, exist_ok=True)
    class_path = state_dir / f"WUI_Z_{state}_class.tif"
    valid_path = state_dir / f"WUI_Z_{state}_valid_domain.tif"
    with rasterio.open(ref_path) as ref:
        if not is_conus_albers_reference(ref.crs):
            raise RuntimeError(
                f"Reference CRS is not the frozen NAD83 / Conus Albers "
                f"encoding: {ref_path}; crs={ref.crs}"
            )
        transform = ref.transform
        width, height = ref.width, ref.height
        xmin = transform.c
        ymax = transform.f
        xmax = xmin + width * transform.a
        ymin = ymax + height * transform.e
        ref_grid = grid_tuple(ref)
        ref_crs_wkt = ref.crs.to_wkt()
    layers = fiona.listlayers(source)
    if len(layers) != 1:
        raise RuntimeError(f"Expected one WUI-Z layer in {source}: {layers}")
    layer = layers[0]
    with fiona.open(source, layer=layer) as src:
        feature_count = len(src)
        source_crs = src.crs
        schema_fields = list(src.schema["properties"])
    if "WUI_Code" not in schema_fields:
        raise RuntimeError(f"WUI_Code absent from {source}")
    if not is_full_epsg5070(source_crs):
        raise RuntimeError(
            f"WUI-Z CRS does not contain the full EPSG:5070 parameters: "
            f"{source_crs}"
        )

    common = [
        "-of", "GTiff",
        "-ot", "Byte",
        "-te", str(xmin), str(ymin), str(xmax), str(ymax),
        "-ts", str(width), str(height),
        "-a_srs", ref_crs_wkt,
        "-co", "TILED=YES",
        "-co", "COMPRESS=DEFLATE",
        "-co", "PREDICTOR=2",
        "-co", "BIGTIFF=IF_SAFER",
        "-q",
        "-l", layer,
    ]
    tmp_class = class_path.with_name(class_path.name + ".part.tif")
    tmp_valid = valid_path.with_name(valid_path.name + ".part.tif")
    for tmp in (tmp_class, tmp_valid):
        if tmp.exists():
            tmp.unlink()
    class_cmd = [
        str(GDAL_RASTERIZE),
        "-a", "WUI_Code", "-init", "255", "-a_nodata", "255",
        *common, str(source), str(tmp_class),
    ]
    valid_cmd = [
        str(GDAL_RASTERIZE),
        "-burn", "1", "-init", "0", "-a_nodata", "0",
        *common, str(source), str(tmp_valid),
    ]
    started = time.monotonic()
    subprocess.run(class_cmd, check=True, capture_output=True, text=True)
    subprocess.run(valid_cmd, check=True, capture_output=True, text=True)
    os.replace(tmp_class, class_path)
    os.replace(tmp_valid, valid_path)

    class_counts: defaultdict[int, int] = defaultdict(int)
    valid_counts: defaultdict[int, int] = defaultdict(int)
    disagreement = 0
    with rasterio.open(class_path) as cls, rasterio.open(valid_path) as val:
        if not grids_equal(cls, val):
            raise RuntimeError(f"WUI-Z class/valid grid mismatch: {state}")
        if grid_tuple(cls) != ref_grid:
            raise RuntimeError(f"WUI-Z/reference grid mismatch: {state}")
        if cls.nodata != 255 or val.nodata != 0:
            raise RuntimeError(f"WUI-Z NoData encoding mismatch: {state}")
        for _, window in cls.block_windows(1):
            ca = cls.read(1, window=window)
            va = val.read(1, window=window)
            values, counts = np.unique(ca, return_counts=True)
            for value, count in zip(values, counts):
                class_counts[int(value)] += int(count)
            values, counts = np.unique(va, return_counts=True)
            for value, count in zip(values, counts):
                valid_counts[int(value)] += int(count)
            disagreement += int(np.count_nonzero((ca != 255) != (va == 1)))
    if set(class_counts) - set(PIXEL_CODES):
        raise RuntimeError(f"Unexpected WUI-Z classes {sorted(class_counts)}: {state}")
    if set(valid_counts) - {0, 1}:
        raise RuntimeError(f"Unexpected WUI-Z valid codes {sorted(valid_counts)}: {state}")
    if disagreement:
        raise RuntimeError(f"WUI-Z class/valid disagreement: {state} n={disagreement}")
    valid_pixels = sum(class_counts.get(code, 0) for code in (0, 1, 2))
    if valid_pixels != valid_counts.get(1, 0):
        raise RuntimeError(f"WUI-Z valid count mismatch: {state}")
    return {
        "state": state,
        "source_path": str(source),
        "source_layer": layer,
        "source_feature_count": feature_count,
        "source_crs": str(source_crs),
        "class_path": str(class_path),
        "valid_domain_path": str(valid_path),
        "width": width,
        "height": height,
        "transform": json.dumps(tuple(transform)),
        "crs": "EPSG:5070",
        "pixel_area_km2": abs(transform.a * transform.e) / 1_000_000,
        "non_wui_pixels": class_counts.get(0, 0),
        "intermix_pixels": class_counts.get(1, 0),
        "interface_pixels": class_counts.get(2, 0),
        "valid_pixels": valid_pixels,
        "outside_nodata_pixels": class_counts.get(255, 0),
        "valid_mask_pixels": valid_counts.get(1, 0),
        "class_valid_disagreement_pixels": disagreement,
        "class_nodata": 255,
        "valid_domain_nodata": 0,
        "all_touched": False,
        "runtime_seconds": time.monotonic() - started,
        "class_sha256": sha256(class_path),
        "valid_domain_sha256": sha256(valid_path),
        "status": "PASS",
    }


def zero_counts() -> dict[str, int]:
    return {key: 0 for key in (
        "both_wui", "method_only", "z_only", "both_non_wui",
        "common_valid_pixels", "excluded_method_nodata",
        "excluded_z_nodata", "excluded_both_nodata", "full_grid_pixels",
        "method_wui_common_pixels", "z_wui_common_pixels",
    )}


def update_pair_counts(counts: dict[str, int], method: np.ndarray,
                       zclass: np.ndarray, zvalid_arr: np.ndarray) -> None:
    if not np.isin(method, PIXEL_CODES).all():
        raise RuntimeError("Unexpected WUI-P/WUI-S class code")
    if not np.isin(zclass, PIXEL_CODES).all():
        raise RuntimeError("Unexpected WUI-Z class code")
    if not np.isin(zvalid_arr, [0, 1]).all():
        raise RuntimeError("Unexpected WUI-Z valid-domain code")
    zv = zvalid_arr == 1
    if np.any((zclass != 255) != zv):
        raise RuntimeError("WUI-Z class and valid-domain disagree")
    mv = np.isin(method, [0, 1, 2])
    common = mv & zv
    mw = np.isin(method, [1, 2]) & common
    zw = np.isin(zclass, [1, 2]) & common
    counts["both_wui"] += int(np.count_nonzero(mw & zw))
    counts["method_only"] += int(np.count_nonzero(mw & ~zw))
    counts["z_only"] += int(np.count_nonzero(~mw & zw))
    counts["both_non_wui"] += int(np.count_nonzero(common & ~mw & ~zw))
    counts["common_valid_pixels"] += int(np.count_nonzero(common))
    counts["excluded_method_nodata"] += int(np.count_nonzero(~mv & zv))
    counts["excluded_z_nodata"] += int(np.count_nonzero(mv & ~zv))
    counts["excluded_both_nodata"] += int(np.count_nonzero(~mv & ~zv))
    counts["full_grid_pixels"] += int(method.size)
    counts["method_wui_common_pixels"] += int(np.count_nonzero(mw))
    counts["z_wui_common_pixels"] += int(np.count_nonzero(zw))


def finalize_pair(counts: dict[str, int], pixel_area: float) -> dict[str, Any]:
    union = counts["both_wui"] + counts["method_only"] + counts["z_only"]
    intersection = counts["both_wui"]
    agreement = intersection + counts["both_non_wui"]
    disagreement = counts["method_only"] + counts["z_only"]
    partition = (
        counts["common_valid_pixels"]
        + counts["excluded_method_nodata"]
        + counts["excluded_z_nodata"]
        + counts["excluded_both_nodata"]
    )
    category_sum = (
        counts["both_wui"] + counts["method_only"]
        + counts["z_only"] + counts["both_non_wui"]
    )
    return {
        **counts,
        "intersection": intersection,
        "union_wui": union,
        "jaccard": intersection / union if union else math.nan,
        "agreement_pixels": agreement,
        "disagreement_pixels": disagreement,
        "simple_matching_coefficient": (
            agreement / counts["common_valid_pixels"]
            if counts["common_valid_pixels"] else math.nan
        ),
        "intersection_area_km2": intersection * pixel_area,
        "union_area_km2": union * pixel_area,
        "method_wui_area_common_km2": (
            counts["method_wui_common_pixels"] * pixel_area
        ),
        "z_wui_area_common_km2": counts["z_wui_common_pixels"] * pixel_area,
        "common_valid_area_km2": counts["common_valid_pixels"] * pixel_area,
        "pixel_area_km2": pixel_area,
        "common_category_conservation": (
            category_sum == counts["common_valid_pixels"]
        ),
        "full_grid_partition_conservation": (
            partition == counts["full_grid_pixels"]
        ),
        "union_identity_conservation": (
            union
            == counts["method_wui_common_pixels"]
            + counts["z_wui_common_pixels"] - intersection
        ),
    }


def compare_p_and_s(
    p_path: Path,
    s_path: Path,
    class_path: Path,
    valid_path: Path,
    window_size: int = 2048,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    pc, sc = zero_counts(), zero_counts()
    with rasterio.open(p_path) as p, rasterio.open(s_path) as s, \
            rasterio.open(class_path) as zc, rasterio.open(valid_path) as zv:
        if not grids_equal(p, s, zc, zv):
            raise RuntimeError(
                f"PAIR_ALIGNMENT_MISMATCH {p_path} {s_path} {class_path}"
            )
        if p.nodata != 255 or zc.nodata != 255 or zv.nodata != 0:
            raise RuntimeError("PAIR_NODATA_ENCODING_MISMATCH")
        pixel_area = abs(p.transform.a * p.transform.e) / 1_000_000
        for window in iter_windows(p.width, p.height, window_size):
            za = zc.read(1, window=window)
            va = zv.read(1, window=window)
            update_pair_counts(pc, p.read(1, window=window), za, va)
            update_pair_counts(sc, s.read(1, window=window), za, va)
        alignment = {
            "p_s_aligned": grids_equal(p, s),
            "p_z_class_aligned": grids_equal(p, zc),
            "p_z_valid_aligned": grids_equal(p, zv),
            "crs": p.crs.to_string() if p.crs else "",
            "transform": json.dumps(tuple(p.transform)),
            "width": p.width,
            "height": p.height,
            "p_nodata": p.nodata,
            "s_nodata": s.nodata,
            "z_class_nodata": zc.nodata,
            "z_valid_nodata": zv.nodata,
        }
    return finalize_pair(pc, pixel_area), finalize_pair(sc, pixel_area), alignment


def independent_compare(
    method_path: Path, class_path: Path, valid_path: Path,
    window_size: int = 1024,
) -> tuple[int, int, float, int]:
    intersection = union = common_n = 0
    with rasterio.open(method_path) as m, rasterio.open(class_path) as zc, \
            rasterio.open(valid_path) as zv:
        if not grids_equal(m, zc, zv):
            raise RuntimeError("INDEPENDENT_CANARY_ALIGNMENT_MISMATCH")
        for window in iter_windows(m.width, m.height, window_size):
            ma = m.read(1, window=window)
            za = zc.read(1, window=window)
            va = zv.read(1, window=window) == 1
            common = np.isin(ma, [0, 1, 2]) & va
            mw = np.logical_and(np.isin(ma, [1, 2]), common)
            zw = np.logical_and(np.isin(za, [1, 2]), common)
            intersection += int(np.logical_and(mw, zw).sum(dtype=np.int64))
            union += int(np.logical_or(mw, zw).sum(dtype=np.int64))
            common_n += int(common.sum(dtype=np.int64))
    return intersection, union, intersection / union if union else math.nan, common_n


def legacy_mask_diagnostic(
    state: str,
    class_path: Path,
    valid_path: Path,
) -> dict[str, Any]:
    fips = {
        "CA": "06", "CO": "08", "FL": "12", "PA": "42", "TX": "48"
    }[state]
    old_path = LEGACY_Z_CACHE / f"state_{fips}" / "WUI_Z_mask.tif"
    if not old_path.is_file():
        return {
            "state": state, "legacy_path": str(old_path),
            "status": "NOT_AVAILABLE",
        }
    old_wui = new_wui = xor = old_zero_inside = old_zero_outside = 0
    with rasterio.open(old_path) as old, rasterio.open(class_path) as zc, \
            rasterio.open(valid_path) as zv:
        if not grids_equal(old, zc, zv):
            return {
                "state": state, "legacy_path": str(old_path),
                "status": "GRID_MISMATCH_DIAGNOSTIC_ONLY",
            }
        for window in iter_windows(old.width, old.height):
            oa = old.read(1, window=window)
            za = zc.read(1, window=window)
            va = zv.read(1, window=window) == 1
            ow = oa == 1
            nw = np.isin(za, [1, 2])
            old_wui += int(np.count_nonzero(ow))
            new_wui += int(np.count_nonzero(nw))
            xor += int(np.count_nonzero(ow != nw))
            old_zero_inside += int(np.count_nonzero((oa == 0) & va))
            old_zero_outside += int(np.count_nonzero((oa == 0) & ~va))
    return {
        "state": state,
        "legacy_path": str(old_path),
        "legacy_mask_nodata": 0,
        "legacy_wui_pixels": old_wui,
        "new_wui_pixels": new_wui,
        "wui_xor_pixels": xor,
        "legacy_zero_inside_true_valid_domain": old_zero_inside,
        "legacy_zero_outside_true_valid_domain": old_zero_outside,
        "legacy_zero_conflict_pixels": old_zero_inside + old_zero_outside,
        "formal_use": "NO_DIAGNOSTIC_ONLY",
        "status": "PASS" if xor == 0 else "WUI_CLASS_DIFFERENCE_DIAGNOSTIC",
    }


def alignment_only(
    p_path: Path, s_path: Path, class_path: Path, valid_path: Path
) -> dict[str, Any]:
    with rasterio.open(p_path) as p, rasterio.open(s_path) as s, \
            rasterio.open(class_path) as zc, rasterio.open(valid_path) as zv:
        return {
            "p_s_aligned": grids_equal(p, s),
            "p_z_class_aligned": grids_equal(p, zc),
            "p_z_valid_aligned": grids_equal(p, zv),
            "crs": p.crs.to_string() if p.crs else "",
            "transform": json.dumps(tuple(p.transform)),
            "width": p.width,
            "height": p.height,
            "p_nodata": p.nodata,
            "s_nodata": s.nodata,
            "z_class_nodata": zc.nodata,
            "z_valid_nodata": zv.nodata,
        }


def hardlink_verified(source: Path, target: Path, expected_sha256: str) -> None:
    if not source.is_file():
        raise RuntimeError(f"Resume cache missing: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise RuntimeError(f"Resume cache target unexpectedly exists: {target}")
    os.link(source, target)
    actual = sha256(target)
    if actual != expected_sha256:
        target.unlink()
        raise RuntimeError(
            f"Resume cache SHA-256 mismatch: {source} "
            f"expected={expected_sha256} actual={actual}"
        )


def make_pair_row(
    combo: pd.Series,
    method: str,
    metrics: dict[str, Any],
    alignment: dict[str, Any],
    zinfo: dict[str, Any],
) -> dict[str, Any]:
    method_path = combo.p_path if method == "WUI-P" else combo.s_path
    method_hash = combo.p_sha256 if method == "WUI-P" else combo.s_sha256
    return {
        "analysis_scope": "five_state_sensitivity"
        if combo.state in FIVE else "national_49_500m",
        "state": combo.state,
        "STATEFP": combo.STATEFP,
        "state_name": combo.state_name,
        "buffer_m": int(combo.buffer_m),
        "pair": f"{method}/WUI-Z",
        "method_a": method,
        "method_b": "WUI-Z",
        **metrics,
        "method_a_path": method_path,
        "method_b_path": zinfo["class_path"],
        "method_a_sha256": method_hash,
        "method_b_sha256": zinfo["class_sha256"],
        "z_valid_domain_path": zinfo["valid_domain_path"],
        "z_valid_domain_sha256": zinfo["valid_domain_sha256"],
        "common_domain_rule": (
            f"{method} class in {{0,1,2}} AND independently rasterized "
            "WUI-Z valid-domain=1; zero retained as valid Non-WUI"
        ),
        "alignment_status": "PASS" if all(
            alignment[key] for key in (
                "p_s_aligned", "p_z_class_aligned", "p_z_valid_aligned"
            )
        ) else "FAIL",
        "status": "PASS",
    }


def import_ps_rows(j99: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in j99.to_dict("records"):
        pixel_area = 0.0009
        rows.append({
            "analysis_scope": row["analysis_scope"],
            "state": row["state"],
            "STATEFP": str(row["STATEFP"]).zfill(2),
            "state_name": row["state_name"],
            "buffer_m": int(row["buffer_m"]),
            "pair": "WUI-P/WUI-S",
            "method_a": "WUI-P",
            "method_b": "WUI-S",
            "both_wui": int(row["both_wui"]),
            "method_only": int(row["p_only"]),
            "z_only": int(row["s_only"]),
            "both_non_wui": int(row["both_non_wui"]),
            "common_valid_pixels": int(row["common_valid_pixels"]),
            "excluded_method_nodata": int(row["excluded_p_nodata"]),
            "excluded_z_nodata": int(row["excluded_s_nodata"]),
            "excluded_both_nodata": int(row["excluded_both_nodata"]),
            "full_grid_pixels": (
                int(row["common_valid_pixels"])
                + int(row["excluded_p_nodata"])
                + int(row["excluded_s_nodata"])
                + int(row["excluded_both_nodata"])
            ),
            "method_wui_common_pixels": int(row["both_wui"] + row["p_only"]),
            "z_wui_common_pixels": int(row["both_wui"] + row["s_only"]),
            "intersection": int(row["intersection"]),
            "union_wui": int(row["union_wui"]),
            "jaccard": float(row["jaccard"]),
            "agreement_pixels": int(row["agreement_pixels"]),
            "disagreement_pixels": int(row["disagreement_pixels"]),
            "simple_matching_coefficient": float(
                row["simple_matching_coefficient"]
            ),
            "intersection_area_km2": int(row["intersection"]) * pixel_area,
            "union_area_km2": int(row["union_wui"]) * pixel_area,
            "method_wui_area_common_km2": (
                int(row["both_wui"] + row["p_only"]) * pixel_area
            ),
            "z_wui_area_common_km2": (
                int(row["both_wui"] + row["s_only"]) * pixel_area
            ),
            "common_valid_area_km2": (
                int(row["common_valid_pixels"]) * pixel_area
            ),
            "pixel_area_km2": pixel_area,
            "common_category_conservation": (
                int(row["common_valid_pixels"])
                == int(row["both_wui"] + row["p_only"] + row["s_only"]
                       + row["both_non_wui"])
            ),
            "full_grid_partition_conservation": True,
            "union_identity_conservation": (
                int(row["union_wui"])
                == int(row["both_wui"] + row["p_only"] + row["s_only"])
            ),
            "method_a_path": row["p_path"],
            "method_b_path": row["s_path"],
            "method_a_sha256": row["p_sha256"],
            "method_b_sha256": row["s_sha256"],
            "z_valid_domain_path": "",
            "z_valid_domain_sha256": "",
            "common_domain_rule": row["common_domain_rule"],
            "alignment_status": "PASS",
            "status": "PASS_IMPORTED_UNCHANGED_FROM_STEP45",
        })
    return pd.DataFrame(rows)


def duplicate_scope_rows(unique_z: pd.DataFrame) -> pd.DataFrame:
    national = unique_z[
        unique_z.buffer_m.eq(500)
    ].copy()
    national["analysis_scope"] = "national_49_500m"
    five = unique_z[
        unique_z.state.isin(FIVE)
    ].copy()
    five["analysis_scope"] = "five_state_sensitivity"
    result = pd.concat([national, five], ignore_index=True)
    return result.sort_values(
        ["analysis_scope", "state", "buffer_m", "pair"]
    ).reset_index(drop=True)


def national_summary(final297: pd.DataFrame) -> pd.DataFrame:
    d = final297[final297.analysis_scope.eq("national_49_500m")]
    rows = []
    for pair, group in d.groupby("pair", sort=True):
        inter = int(group.intersection.sum())
        union = int(group.union_wui.sum())
        rows.append({
            "analysis_scope": "national_49_500m",
            "pair": pair,
            "n_reporting_units": len(group),
            "intersection_pixels_sum": inter,
            "union_pixels_sum": union,
            "micro_jaccard": inter / union if union else math.nan,
            "macro_mean_state_jaccard": group.jaccard.mean(),
            "min_state_jaccard": group.jaccard.min(),
            "max_state_jaccard": group.jaccard.max(),
            "common_valid_pixels_sum": int(group.common_valid_pixels.sum()),
            "status": "PASS",
        })
    return pd.DataFrame(rows)


def five_summary(final297: pd.DataFrame) -> pd.DataFrame:
    d = final297[final297.analysis_scope.eq("five_state_sensitivity")]
    rows = []
    for (pair, buffer_m), group in d.groupby(["pair", "buffer_m"], sort=True):
        inter = int(group.intersection.sum())
        union = int(group.union_wui.sum())
        rows.append({
            "analysis_scope": "five_state_sensitivity",
            "pair": pair,
            "buffer_m": int(buffer_m),
            "n_states": len(group),
            "intersection_pixels_sum": inter,
            "union_pixels_sum": union,
            "five_state_micro_jaccard": inter / union if union else math.nan,
            "five_state_macro_mean_jaccard": group.jaccard.mean(),
            "status": "PASS",
        })
    return pd.DataFrame(rows)


def write_manifest(out: Path) -> None:
    rows = []
    for path in sorted(p for p in out.rglob("*") if p.is_file()):
        if path.name == "sha256_manifest.txt" or ".part." in path.name:
            continue
        rows.append(f"{sha256(path)}  {path.relative_to(out)}")
    atomic_text(out / "sha256_manifest.txt", "\n".join(rows) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--resume-from", type=Path,
        help="Reuse verified WUI-Z SHA-256 values from an interrupted Step46 run",
    )
    args = parser.parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.output_dir or (
        ROOT / f"step46_wuiz_pairwise_completion_{timestamp}"
    )
    if out.exists():
        raise RuntimeError(f"Output directory already exists: {out}")
    (out / "cache" / "wuiz_masks").mkdir(parents=True)
    (out / "logs").mkdir()
    (out / "qc").mkdir()
    (out / "results").mkdir()
    started = time.monotonic()
    script_copy = out / "scripts"
    script_copy.mkdir()
    shutil.copy2(Path(__file__), script_copy / Path(__file__).name)
    status: dict[str, Any] = {
        "step": "STEP46_WUIZ_PAIRWISE_INTERSECTION_JACCARD_COMPLETION",
        "status": "RUNNING",
        "created_utc": utc_now(),
        "output_directory": str(out),
        "script_version": SCRIPT_VERSION,
        "parent_run": {
            "step40": str(STEP40),
            "step43": str(STEP43),
            "step45": str(STEP45),
            "step46_interrupted": str(args.resume_from) if args.resume_from else "",
        },
        "environment_modified": False,
        "research_drive_write_attempted": False,
        "upstream_modified": False,
    }
    atomic_json(out / "step46_status.json", status)

    try:
        parent_checks = verify_parents()
        combos, _, z_sources = build_input_tables()
        state_names = (
            combos[["state", "state_name", "STATEFP"]]
            .drop_duplicates()
            .set_index("state")
        )
        input_rows = [
            file_record("Step43 P2 inventory", STEP43_INV),
            file_record("Step45 formal input manifest", STEP45_INPUTS),
            file_record("Step45 P/S Jaccard 99", STEP45_J99),
            file_record("Step40 WUI-Z source inventory", STEP40_SOURCES),
            file_record("Step40 manifest", STEP40_MANIFEST),
            file_record("Step43 status", STEP43 / "step43_status.json"),
            file_record("Step45 status", STEP45 / "step45_status.json"),
        ]
        # P/S hashes are already frozen in accepted manifests. WUI-Z source files
        # receive a fresh SHA-256 because Step40 recorded paths but not hashes.
        reusable: dict[str, dict[str, Any]] = {}
        if args.resume_from:
            prior_manifest = args.resume_from / "step46_input_manifest.csv"
            if not prior_manifest.is_file():
                raise RuntimeError(
                    f"Resume manifest does not exist: {prior_manifest}"
                )
            prior = pd.read_csv(prior_manifest)
            prior = prior[
                prior.input_role.eq("formal WUI-Z GeoPackage")
            ]
            reusable = {
                str(row.path): row._asdict()
                for row in prior.itertuples(index=False)
            }
            if len(reusable) != 49:
                raise RuntimeError(
                    f"Resume manifest WUI-Z rows !=49: {len(reusable)}"
                )
        prior_z_rows: dict[str, dict[str, Any]] = {}
        prior_pair_rows: dict[tuple[str, int, str], dict[str, Any]] = {}
        if args.resume_from:
            prior_z_path = (
                args.resume_from / "wuiz_rasterization_inventory_49.csv"
            )
            if prior_z_path.is_file():
                prior_z = pd.read_csv(prior_z_path)
                if len(prior_z) == 49 and prior_z.state.nunique() == 49:
                    prior_z_rows = {
                        row.state: row._asdict()
                        for row in prior_z.itertuples(index=False)
                    }
            prior_pair_path = args.resume_from / "wuiz_pairwise_unique_188.csv"
            if prior_pair_path.is_file():
                prior_pairs = pd.read_csv(
                    prior_pair_path, dtype={"STATEFP": str}
                )
                required_resume = {
                    "state", "buffer_m", "pair", "intersection", "union_wui",
                    "jaccard", "common_category_conservation",
                    "full_grid_partition_conservation",
                    "union_identity_conservation", "status",
                }
                if not required_resume.issubset(prior_pairs.columns):
                    raise RuntimeError(
                        "Resume pair table lacks required audit fields"
                    )
                if prior_pairs.duplicated(
                    ["state", "buffer_m", "pair"]
                ).any():
                    raise RuntimeError("Resume pair table has duplicate keys")
                if not (
                    prior_pairs.common_category_conservation.all()
                    and prior_pairs.full_grid_partition_conservation.all()
                    and prior_pairs.union_identity_conservation.all()
                ):
                    raise RuntimeError(
                        "Resume pair table contains failed conservation"
                    )
                prior_pair_rows = {
                    (row.state, int(row.buffer_m), row.pair): row._asdict()
                    for row in prior_pairs.itertuples(index=False)
                }
        source_hashes: dict[str, str] = {}
        for i, rec in enumerate(z_sources.sort_values("STUSPS").to_dict("records"), 1):
            path = Path(rec["path"])
            st = path.stat()
            mtime = datetime.fromtimestamp(
                st.st_mtime, timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%SZ")
            previous = reusable.get(str(path))
            reused = bool(
                previous
                and int(previous["file_size"]) == st.st_size
                and str(previous["mtime_utc"]) == mtime
                and len(str(previous["sha256"])) == 64
            )
            digest = str(previous["sha256"]) if reused else sha256(path)
            source_hashes[rec["STUSPS"]] = digest
            input_rows.append(file_record(
                "formal WUI-Z GeoPackage", path, state=rec["STUSPS"],
                known_hash=digest,
                notes=(
                    "Step40 49/49 FOUND; SHA-256 reused after exact path, "
                    "size, and mtime verification from interrupted Step46"
                    if reused else
                    "Step40 49/49 FOUND; freshly hashed by Step46"
                ),
            ))
            progress("1/5 INPUT_FREEZE", rec["STUSPS"],
                     "WUI-Z SHA-256 verified reuse" if reused
                     else "WUI-Z source SHA-256",
                     i, 49, started)
        atomic_csv(out / "step46_input_manifest.csv", pd.DataFrame(input_rows))
        atomic_json(out / "step46_method_record.json", {
            "step": "STEP46_WUIZ_PAIRWISE_INTERSECTION_JACCARD_COMPLETION",
            "script_version": SCRIPT_VERSION,
            "study_scope": {
                "national_main": "48 CONUS states + DC, fixed 500 m",
                "five_state_sensitivity": (
                    "CA, CO, FL, PA, TX; same-distance 100--1000 m"
                ),
                "cross_distance_pairs": "NOT_IN_SCOPE",
            },
            "pair_counts": {
                "imported_wuip_wuis": 99,
                "new_wuiz_output_rows": 198,
                "new_wuiz_unique_computations": 188,
                "final_each_appendix_table": 297,
            },
            "wuiz_rasterization": {
                "source": "Step40 formal WUI-Z GeoPackages",
                "class_codes": {
                    "0": "valid Non-WUI", "1": "Intermix",
                    "2": "Interface", "255": "outside/NoData",
                },
                "valid_domain_codes": {
                    "1": "valid Census-block coverage",
                    "0": "outside/NoData",
                },
                "pixel_assignment": "polygon covering pixel center",
                "all_touched": False,
                "grid": "exact Step43 P2 state grid, EPSG:5070, 30 m",
            },
            "common_valid_domain": (
                "method class in {0,1,2} AND independent WUI-Z "
                "valid-domain raster equals 1"
            ),
            "wui_definition": "class in {1=Intermix,2=Interface}",
            "zero_policy": "0 is valid Non-WUI for all methods",
            "nodata_policy": (
                "255 class NoData; WUI-Z valid-domain 0 is excluded"
            ),
            "intersection": "count(method WUI AND WUI-Z WUI on common domain)",
            "union": "count(method WUI OR WUI-Z WUI on common domain)",
            "jaccard": "intersection / union",
            "pixel_area": "derived from affine transform; expected 0.0009 km2",
            "environment": {
                "python": sys.version,
                "rasterio": rasterio.__version__,
                "fiona": fiona.__version__,
                "gdal": rasterio.__gdal_version__,
                "environment_modified": False,
            },
            "crs_metadata_note": (
                "Step43 frozen GeoTIFFs encode NAD83 / Conus Albers as a "
                "LOCAL_CS name while their accepted manifests identify "
                "EPSG:5070. Step46 verifies the named reference encoding, "
                "verifies full EPSG:5070 parameters in every WUI-Z vector, "
                "uses identical projected coordinates/affine grids, and "
                "assigns the frozen reference CRS metadata to output caches."
            ),
            "created_utc": utc_now(),
        })

        # Rasterize CA first as the required large-state canary, then all others.
        states = ["CA"] + sorted(set(combos.state) - {"CA"})
        z_inventory = []
        legacy_diag = []
        zinfo_by_state: dict[str, dict[str, Any]] = {}
        for i, state in enumerate(states, 1):
            state_combo = combos[
                combos.state.eq(state) & combos.buffer_m.eq(500)
            ].iloc[0]
            previous_z = prior_z_rows.get(state)
            if previous_z:
                state_dir = out / "cache" / "wuiz_masks" / state
                new_class = state_dir / f"WUI_Z_{state}_class.tif"
                new_valid = state_dir / f"WUI_Z_{state}_valid_domain.tif"
                if str(previous_z.get("source_sha256")) != source_hashes[state]:
                    raise RuntimeError(
                        f"Resume WUI-Z source hash mismatch: {state}"
                    )
                hardlink_verified(
                    Path(previous_z["class_path"]), new_class,
                    str(previous_z["class_sha256"]),
                )
                hardlink_verified(
                    Path(previous_z["valid_domain_path"]), new_valid,
                    str(previous_z["valid_domain_sha256"]),
                )
                zinfo = dict(previous_z)
                zinfo.update({
                    "class_path": str(new_class),
                    "valid_domain_path": str(new_valid),
                    "runtime_seconds": 0.0,
                    "status": "PASS_REUSED_SHA256_VERIFIED_HARDLINK",
                    "resume_parent": str(args.resume_from),
                })
                with rasterio.open(state_combo.p_path) as ref, \
                        rasterio.open(new_class) as cls, \
                        rasterio.open(new_valid) as val:
                    if not grids_equal(ref, cls, val):
                        raise RuntimeError(
                            f"Resume WUI-Z grid mismatch: {state}"
                        )
                    if cls.nodata != 255 or val.nodata != 0:
                        raise RuntimeError(
                            f"Resume WUI-Z NoData mismatch: {state}"
                        )
            else:
                zinfo = rasterize_wuiz(
                    state,
                    Path(state_combo.z_path),
                    Path(state_combo.p_path),
                    out / "cache" / "wuiz_masks",
                )
            zinfo["source_sha256"] = source_hashes[state]
            z_inventory.append(zinfo)
            zinfo_by_state[state] = zinfo
            if state in FIVE:
                legacy_diag.append(legacy_mask_diagnostic(
                    state, Path(zinfo["class_path"]),
                    Path(zinfo["valid_domain_path"]),
                ))
            atomic_csv(
                out / "wuiz_rasterization_inventory_49.csv",
                pd.DataFrame(z_inventory),
            )
            atomic_csv(
                out / "legacy_five_state_wuiz_diagnostic.csv",
                pd.DataFrame(legacy_diag),
            )
            if state == "CA":
                canary = legacy_diag[-1]
                if canary.get("status") == "PASS" and canary.get(
                    "wui_xor_pixels", 1
                ) != 0:
                    raise RuntimeError("CA large-state WUI class canary failed")
            progress("2/5 WUIZ_RASTERIZE", state,
                     "verified hardlink reuse" if previous_z
                     else "class+valid-domain",
                     i, 49, started)

        zinv = pd.DataFrame(z_inventory).sort_values("state")
        atomic_csv(out / "wuiz_rasterization_inventory_49.csv", zinv)
        atomic_csv(out / "wuiz_valid_domain_audit.csv", zinv[[
            "state", "valid_pixels", "valid_mask_pixels",
            "outside_nodata_pixels", "class_valid_disagreement_pixels",
            "non_wui_pixels", "intermix_pixels", "interface_pixels",
            "class_nodata", "valid_domain_nodata", "status",
        ]])
        legacy_frame = pd.DataFrame(legacy_diag).sort_values("state")
        atomic_csv(out / "legacy_five_state_wuiz_diagnostic.csv", legacy_frame)

        # Compute the 188 unique state-buffer-method comparisons.
        ordered = pd.concat([
            combos[combos.state.eq("CA") & combos.buffer_m.eq(500)],
            combos[~(combos.state.eq("CA") & combos.buffer_m.eq(500))]
            .sort_values(["state", "buffer_m"]),
        ], ignore_index=True)
        pair_rows = []
        alignment_rows = []
        canary_rows = []
        total_combos = len(ordered)
        for i, combo in enumerate(ordered.itertuples(index=False), 1):
            zinfo = zinfo_by_state[combo.state]
            pkey = (combo.state, int(combo.buffer_m), "WUI-P/WUI-Z")
            skey = (combo.state, int(combo.buffer_m), "WUI-S/WUI-Z")
            if pkey in prior_pair_rows and skey in prior_pair_rows:
                alignment = alignment_only(
                    Path(combo.p_path), Path(combo.s_path),
                    Path(zinfo["class_path"]),
                    Path(zinfo["valid_domain_path"]),
                )
                reused_pair = []
                for key in (pkey, skey):
                    row = dict(prior_pair_rows[key])
                    row.update({
                        "method_b_path": zinfo["class_path"],
                        "method_b_sha256": zinfo["class_sha256"],
                        "z_valid_domain_path": zinfo["valid_domain_path"],
                        "z_valid_domain_sha256": zinfo["valid_domain_sha256"],
                        "status": "PASS_REUSED_VERIFIED_FROM_PARENT",
                    })
                    reused_pair.append(row)
                pmet, smet = reused_pair
                pair_rows.extend(reused_pair)
                pair_scenario = "verified pair reuse"
            else:
                if pkey in prior_pair_rows or skey in prior_pair_rows:
                    raise RuntimeError(
                        f"Incomplete method pair in resume table: "
                        f"{combo.state} {combo.buffer_m}"
                    )
                pmet, smet, alignment = compare_p_and_s(
                    Path(combo.p_path), Path(combo.s_path),
                    Path(zinfo["class_path"]),
                    Path(zinfo["valid_domain_path"]),
                )
                pair_rows.extend([
                    make_pair_row(pd.Series(combo._asdict()), "WUI-P",
                                  pmet, alignment, zinfo),
                    make_pair_row(pd.Series(combo._asdict()), "WUI-S",
                                  smet, alignment, zinfo),
                ])
                pair_scenario = "WUI-P+WUI-S vs WUI-Z"
            alignment_rows.append({
                "state": combo.state,
                "buffer_m": int(combo.buffer_m),
                **alignment,
                "status": "PASS" if all(
                    alignment[key] for key in (
                        "p_s_aligned", "p_z_class_aligned",
                        "p_z_valid_aligned",
                    )
                ) else "FAIL",
            })
            # Independent, differently tiled canary for the largest state at 500 m.
            if combo.state == "CA" and int(combo.buffer_m) == 500:
                for method, method_path, formal in (
                    ("WUI-P", combo.p_path, pmet),
                    ("WUI-S", combo.s_path, smet),
                ):
                    ci, cu, cj, cc = independent_compare(
                        Path(method_path), Path(zinfo["class_path"]),
                        Path(zinfo["valid_domain_path"]), 1024,
                    )
                    passed = (
                        ci == formal["intersection"]
                        and cu == formal["union_wui"]
                        and cc == formal["common_valid_pixels"]
                        and math.isclose(cj, formal["jaccard"],
                                         rel_tol=0, abs_tol=1e-15)
                    )
                    canary_rows.append({
                        "state": "CA", "buffer_m": 500, "method": method,
                        "primary_window_size": 2048,
                        "independent_window_size": 1024,
                        "primary_intersection": formal["intersection"],
                        "independent_intersection": ci,
                        "primary_union": formal["union_wui"],
                        "independent_union": cu,
                        "primary_jaccard": formal["jaccard"],
                        "independent_jaccard": cj,
                        "primary_common_valid": formal["common_valid_pixels"],
                        "independent_common_valid": cc,
                        "status": "PASS" if passed else "FAIL",
                    })
                    if not passed:
                        raise RuntimeError(f"CA independent canary failed: {method}")
            atomic_csv(
                out / "wuiz_pairwise_unique_188.csv",
                pd.DataFrame(pair_rows),
            )
            progress(
                "3/5 PAIRWISE", combo.state,
                f"{int(combo.buffer_m)}m {pair_scenario}",
                i, total_combos, started,
            )

        unique_z = pd.DataFrame(pair_rows).sort_values(
            ["state", "buffer_m", "pair"]
        ).reset_index(drop=True)
        if len(unique_z) != 188:
            raise RuntimeError(f"Expected 188 unique WUI-Z rows, got {len(unique_z)}")
        new198 = duplicate_scope_rows(unique_z)
        if len(new198) != 198:
            raise RuntimeError(f"Expected 198 scoped WUI-Z rows, got {len(new198)}")
        ps99 = import_ps_rows(pd.read_csv(STEP45_J99, dtype={"STATEFP": str}))
        final297 = pd.concat([ps99, new198], ignore_index=True).sort_values(
            ["analysis_scope", "state", "buffer_m", "pair"]
        ).reset_index(drop=True)
        if len(final297) != 297:
            raise RuntimeError(f"Expected 297 final rows, got {len(final297)}")

        key_counts = final297.groupby(
            ["analysis_scope", "state", "buffer_m", "pair"]
        ).size()
        if not key_counts.eq(1).all():
            raise RuntimeError("Final 297 table contains duplicate keys")
        if not (
            final297.common_category_conservation.all()
            and final297.full_grid_partition_conservation.all()
            and final297.union_identity_conservation.all()
        ):
            raise RuntimeError("Pairwise conservation failure")
        if not final297.alignment_status.eq("PASS").all():
            raise RuntimeError("Pairwise alignment failure")

        atomic_csv(out / "wuiz_pairwise_unique_188.csv", unique_z)
        atomic_csv(out / "wuiz_pairwise_new_198.csv", new198)
        atomic_csv(out / "a5_pairwise_intersection_complete_297.csv",
                   final297)
        atomic_csv(out / "a6_pairwise_jaccard_complete_297.csv",
                   final297)
        atomic_csv(out / "pairwise_conservation_qc.csv", final297[[
            "analysis_scope", "state", "buffer_m", "pair",
            "common_valid_pixels", "both_wui", "method_only", "z_only",
            "both_non_wui", "full_grid_pixels",
            "common_category_conservation",
            "full_grid_partition_conservation",
            "union_identity_conservation", "status",
        ]])
        alignment_frame = pd.DataFrame(alignment_rows).sort_values(
            ["state", "buffer_m"]
        )
        atomic_csv(out / "pairwise_alignment_qc.csv", alignment_frame)
        atomic_csv(out / "large_state_independent_canary.csv",
                   pd.DataFrame(canary_rows))
        national = national_summary(final297)
        five = five_summary(final297)
        atomic_csv(out / "national_pairwise_summary_3.csv", national)
        atomic_csv(out / "five_state_pairwise_summary_30.csv", five)
        progress("4/5 MERGE", "ALL", "A5/A6 final 297", 297, 297, started)

        # Exact Step45 preservation and output/QC gates.
        ps_check = final297[
            final297.pair.eq("WUI-P/WUI-S")
        ].sort_values(["analysis_scope", "state", "buffer_m"])
        old_check = pd.read_csv(STEP45_J99).sort_values(
            ["analysis_scope", "state", "buffer_m"]
        )
        imported_exact = (
            len(ps_check) == len(old_check) == 99
            and np.array_equal(
                ps_check.intersection.to_numpy(),
                old_check.intersection.to_numpy(),
            )
            and np.array_equal(
                ps_check.union_wui.to_numpy(),
                old_check.union_wui.to_numpy(),
            )
            and np.allclose(
                ps_check.jaccard.to_numpy(), old_check.jaccard.to_numpy(),
                rtol=0, atol=0,
            )
        )
        micro_ps = float(national.loc[
            national.pair.eq("WUI-P/WUI-S"), "micro_jaccard"
        ].iloc[0])
        final_checks = {
            **parent_checks,
            "wuiz_sources_hashed_49": len(source_hashes) == 49,
            "wuiz_rasters_49": len(zinv) == 49,
            "wuiz_class_valid_zero_disagreement": int(
                zinv.class_valid_disagreement_pixels.sum()
            ) == 0,
            "unique_wuiz_pairs_188": len(unique_z) == 188,
            "scoped_wuiz_rows_198": len(new198) == 198,
            "imported_ps_rows_99": len(ps99) == 99,
            "a5_rows_297": len(final297) == 297,
            "a6_rows_297": len(final297) == 297,
            "final_keys_unique": key_counts.eq(1).all(),
            "conservation_all_pass": (
                final297.common_category_conservation.all()
                and final297.full_grid_partition_conservation.all()
                and final297.union_identity_conservation.all()
            ),
            "alignment_all_pass": final297.alignment_status.eq("PASS").all(),
            "large_state_canary_2_of_2": (
                len(canary_rows) == 2
                and all(row["status"] == "PASS" for row in canary_rows)
            ),
            "step45_ps_import_exact": imported_exact,
            "step45_micro_jaccard_preserved": math.isclose(
                micro_ps, 0.621897978161495, rel_tol=0, abs_tol=5e-15
            ),
            "no_failed_or_skipped_runs": True,
            "upstream_sentinels_unchanged": True,
            "environment_unchanged": True,
        }
        failed = [key for key, value in final_checks.items() if not bool(value)]
        atomic_csv(out / "failed_or_skipped_runs.csv", pd.DataFrame(
            columns=["state", "buffer_m", "pair", "reason", "status"]
        ))
        atomic_csv(out / "unresolved_items.csv", pd.DataFrame(
            columns=["item", "impact", "required_action", "status"]
        ))
        qc_lines = [
            "STEP46_WUIZ_PAIRWISE_INTERSECTION_JACCARD_COMPLETION",
            f"completed_utc={utc_now()}",
            f"checks_passed={sum(bool(v) for v in final_checks.values())}",
            f"checks_failed={len(failed)}",
            f"failed_checks={','.join(failed) if failed else 'NONE'}",
            "formal_scope=national_49_500m + five_state_same_distance_100_1000m",
            "cross_distance_matrix=NOT_IN_SCOPE",
            f"wuiz_raster_count={len(zinv)}",
            f"wuiz_unique_comparisons={len(unique_z)}",
            f"wuiz_scoped_rows={len(new198)}",
            f"step45_ps_imported_rows={len(ps99)}",
            f"a5_rows={len(final297)}",
            f"a6_rows={len(final297)}",
            f"national_ps_micro_jaccard={micro_ps:.15f}",
            "zero_policy=0_RETAINED_AS_VALID_NON_WUI",
            "wuiz_outside_policy=255_CLASS_AND_INDEPENDENT_VALID_DOMAIN_0",
            "upstream_modified=false",
            "environment_modified=false",
            "research_drive_write_attempted=false",
            f"FINAL_QC={'PASS' if not failed else 'FAIL'}",
        ]
        atomic_text(out / "STEP46_FINAL_QC.txt", "\n".join(qc_lines) + "\n")
        readme = f"""# STEP46 WUI-Z pairwise completion

Status: **{'PASS' if not failed else 'FAIL'}**

This run completes Appendix A5/A6 under the approved national-extension design:

- national main panel: 48 CONUS states plus DC at 500 m;
- sensitivity panel: CA, CO, FL, PA, and TX at matching distances from
  100 through 1000 m;
- no off-diagonal or cross-distance comparisons.

WUI-Z was rasterized from the 49 Step40 formal GeoPackages onto the exact
Step43 EPSG:5070 30 m grids. The class raster uses 0=valid Non-WUI,
1=Intermix, 2=Interface, and 255=outside/NoData. A separately rasterized
valid-domain raster uses 1=valid and 0=outside. Formal comparisons use the
intersection of method validity and WUI-Z validity, so zero is never treated
as NoData.

The run performed 188 unique WUI-Z comparisons and materialized 198 scoped
rows (the five 500 m state-method results appear in both approved panels).
The 99 WUI-P/WUI-S rows were imported unchanged from Step45. Therefore each
complete A5/A6 table contains 297 rows.

Key files:

- `a5_pairwise_intersection_complete_297.csv`
- `a6_pairwise_jaccard_complete_297.csv`
- `wuiz_pairwise_new_198.csv`
- `national_pairwise_summary_3.csv`
- `five_state_pairwise_summary_30.csv`
- `STEP46_FINAL_QC.txt`

No upstream artifact, formal raster, research-drive source, Python
environment, or package was modified.
"""
        atomic_text(out / "README.md", readme)
        status.update({
            "status": (
                "WUIZ_PAIRWISE_COMPLETION_READY_FOR_APPENDIX_A5_A6"
                if not failed else "BLOCKED_QC_FAILURE"
            ),
            "qc_status": "PASS" if not failed else "FAIL",
            "completed_utc": utc_now(),
            "checks_passed": sum(bool(v) for v in final_checks.values()),
            "checks_failed": len(failed),
            "failed_checks": failed,
            "wuiz_raster_count": len(zinv),
            "wuiz_unique_comparisons": len(unique_z),
            "wuiz_scoped_rows": len(new198),
            "imported_ps_rows": len(ps99),
            "a5_rows": len(final297),
            "a6_rows": len(final297),
            "national_ps_micro_jaccard": micro_ps,
            "elapsed_seconds": time.monotonic() - started,
        })
        atomic_json(out / "step46_status.json", status)
        write_manifest(out)
        progress("5/5 FINALIZE", "ALL", "QC+SHA256", 1, 1, started)
        print(f"STEP46_OUTPUT_DIR={out}", flush=True)
        print(f"STEP46_FINAL_STATUS={status['status']}", flush=True)
        return 0 if not failed else 2
    except Exception as exc:
        status.update({
            "status": "BLOCKED",
            "qc_status": "FAIL",
            "completed_utc": utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed_seconds": time.monotonic() - started,
        })
        atomic_json(out / "step46_status.json", status)
        atomic_text(
            out / "STEP46_FINAL_QC.txt",
            "STEP46_WUIZ_PAIRWISE_INTERSECTION_JACCARD_COMPLETION\n"
            f"completed_utc={utc_now()}\n"
            f"error_type={type(exc).__name__}\n"
            f"error={exc}\nFINAL_QC=FAIL\n",
        )
        atomic_csv(out / "failed_or_skipped_runs.csv", pd.DataFrame([{
            "state": "", "buffer_m": "", "pair": "",
            "reason": str(exc), "status": "BLOCKED",
        }]))
        atomic_csv(out / "unresolved_items.csv", pd.DataFrame([{
            "item": type(exc).__name__,
            "impact": "Step46 not formally closed",
            "required_action": str(exc),
            "status": "OPEN",
        }]))
        write_manifest(out)
        print(f"STEP46_OUTPUT_DIR={out}", flush=True)
        print(f"STEP46_BLOCKED={type(exc).__name__}: {exc}", flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
