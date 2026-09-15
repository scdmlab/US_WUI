#!/usr/bin/env python3
# Copyright (C) 2026 WUI Mapping Project Authors
# SPDX-License-Identifier: GPL-3.0-only
# Repository version: filesystem paths are loaded through repo_config.py.
# Copy config/paths.example.json to config/paths.json before a new run.
# Raw input data and large intermediate files are stored outside GitHub.

"""Reclassify Colorado WUI-Z with a strict block-based >75% patch rule.

This is a diagnostic continuation of Step47/47B. It never modifies the
formal WUI-Z GeoPackage or any Step40--47B artifact.

Fixed WUI-Z rules
-----------------
* development density: Census housing density > 6.17 units/km2;
* intermix: block wildland vegetation >= 50%;
* interface: block wildland vegetation < 50%, with the block intersecting
  the exact <=2,400 m buffer of a qualifying large wildland patch;
* qualifying patch: contiguous Census blocks with Veg_Percent >75%,
  dissolved into components, retaining components with area >=5 km2.
"""

from __future__ import annotations

from repo_config import portable_path

import csv
import datetime as dt
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.windows import Window
from shapely import get_parts, union_all
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon


ROOT = Path(portable_path("project"))
SOURCE_GPKG = Path(
    portable_path("data", "WUI_Z_Results/WUI_Z_Paper_Colorado.gpkg")
)
SOURCE_LAYER = "WUI_Z_Paper_Colorado"
FORMAL_Z_CLASS = (
    ROOT
    / "step46_wuiz_pairwise_completion_20260728T203108Z"
    / "cache/wuiz_masks/CO/WUI_Z_CO_class.tif"
)
FORMAL_Z_VALID = (
    ROOT
    / "step46_wuiz_pairwise_completion_20260728T203108Z"
    / "cache/wuiz_masks/CO/WUI_Z_CO_valid_domain.tif"
)
STEP47 = (
    ROOT / "step47_patch75_silvis_colorado_canary_20260729T034216Z"
)
STEP47B = (
    ROOT
    / "step47b_patch75_colorado_remaining_radii_20260729T041412Z"
)
REFERENCE_GRID = (
    ROOT
    / "step43_wuip_p2_formal_rebuild_20260727T180822Z"
    / "rasters_500m/CO/WUI_P_P2_CO_r0500m.tif"
)
GDAL_RASTERIZE = Path(
    portable_path("software", "bin/gdal_rasterize")
)

DENSITY_THRESHOLD = 6.17
INTERMIX_THRESHOLD = 50.0
PATCH_THRESHOLD = 75.0
PATCH_AREA_M2 = 5_000_000.0
DISTANCE_M = 2_400.0
PIXEL_AREA_KM2 = 0.0009
NODATA = 255
CLASS_LABELS = {0: "Non-WUI", 1: "Intermix", 2: "Interface"}
RADII = tuple(range(100, 1001, 100))


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(chunk):
            digest.update(data)
    return digest.hexdigest()


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def polygon_parts(geometry) -> list[Polygon]:
    result: list[Polygon] = []
    stack = [geometry]
    while stack:
        item = stack.pop()
        if item is None or item.is_empty:
            continue
        if isinstance(item, Polygon):
            result.append(item)
        elif isinstance(item, (MultiPolygon, GeometryCollection)):
            stack.extend(list(get_parts(item)))
    return result


def iter_windows(width: int, height: int, tile: int = 2048):
    for row in range(0, height, tile):
        for col in range(0, width, tile):
            yield Window(
                col,
                row,
                min(tile, width - col),
                min(tile, height - row),
            )


def grids_equal(*datasets) -> bool:
    first = datasets[0]
    return all(
        ds.width == first.width
        and ds.height == first.height
        and ds.transform == first.transform
        for ds in datasets[1:]
    )


def method_path(method: str, radius: int) -> Path:
    if radius == 500:
        return STEP47 / f"WUI_{method}_CO_r0500m_SILVIS_GT75_CANARY.tif"
    return (
        STEP47B
        / "rasters"
        / f"WUI-{method}"
        / f"WUI_{method}_CO_r{radius:04d}m_SILVIS_GT75.tif"
    )


def rasterize_field(
    source: Path,
    layer: str,
    output: Path,
    reference,
    *,
    field: str | None = None,
    burn: int | None = None,
    init: int,
    nodata: int,
) -> None:
    command = [
        str(GDAL_RASTERIZE),
        "-l",
        layer,
    ]
    if field is not None:
        command += ["-a", field]
    elif burn is not None:
        command += ["-burn", str(burn)]
    else:
        raise ValueError("field or burn is required")
    left, bottom, right, top = reference.bounds
    command += [
        "-init",
        str(init),
        "-a_nodata",
        str(nodata),
        "-ot",
        "Byte",
        "-of",
        "GTiff",
        "-te",
        str(left),
        str(bottom),
        str(right),
        str(top),
        "-ts",
        str(reference.width),
        str(reference.height),
        "-a_srs",
        reference.crs.to_wkt(),
        "-co",
        "COMPRESS=DEFLATE",
        "-co",
        "TILED=YES",
        "-co",
        "PREDICTOR=2",
        "-co",
        "BIGTIFF=IF_SAFER",
        str(source),
        str(output),
    ]
    subprocess.run(command, check=True)


def class_counts(path: Path) -> dict[int, int]:
    counts = {0: 0, 1: 0, 2: 0, 255: 0}
    with rasterio.open(path) as source:
        for _, window in source.block_windows(1):
            values, frequencies = np.unique(
                source.read(1, window=window), return_counts=True
            )
            for value, frequency in zip(values, frequencies):
                value_int = int(value)
                if value_int not in counts:
                    raise RuntimeError(
                        f"Unexpected class {value_int} in {path}"
                    )
                counts[value_int] += int(frequency)
    return counts


def compare_old_new(
    old_path: Path, new_path: Path, valid_path: Path
) -> dict[str, Any]:
    transitions = {(i, j): 0 for i in range(3) for j in range(3)}
    valid_pixels = baseline_outside = 0
    with (
        rasterio.open(old_path) as old,
        rasterio.open(new_path) as new,
        rasterio.open(valid_path) as valid,
    ):
        if not grids_equal(old, new, valid):
            raise RuntimeError("Old/new WUI-Z grids differ")
        for window in iter_windows(old.width, old.height):
            old_array = old.read(1, window=window)
            new_array = new.read(1, window=window)
            domain = valid.read(1, window=window) == 1
            valid_pixels += int(domain.sum())
            baseline_outside += int(
                np.count_nonzero((new_array != 255) & ~domain)
            )
            for old_code in range(3):
                for new_code in range(3):
                    transitions[(old_code, new_code)] += int(
                        np.count_nonzero(
                            (old_array == old_code)
                            & (new_array == new_code)
                            & domain
                        )
                    )
    unchanged = sum(transitions[(code, code)] for code in range(3))
    old_wui = sum(
        value
        for (old_code, _), value in transitions.items()
        if old_code in (1, 2)
    )
    new_wui = sum(
        value
        for (_, new_code), value in transitions.items()
        if new_code in (1, 2)
    )
    intersection = sum(
        value
        for (old_code, new_code), value in transitions.items()
        if old_code in (1, 2) and new_code in (1, 2)
    )
    union = old_wui + new_wui - intersection
    return {
        "valid_pixels": valid_pixels,
        "unchanged_pixels": unchanged,
        "changed_pixels": valid_pixels - unchanged,
        "old_wui_pixels": old_wui,
        "new_wui_pixels": new_wui,
        "intersection_pixels": intersection,
        "union_pixels": union,
        "jaccard_new_vs_old": intersection / union if union else 1.0,
        "old_wui_area_km2": old_wui * PIXEL_AREA_KM2,
        "new_wui_area_km2": new_wui * PIXEL_AREA_KM2,
        "net_wui_area_change_km2": (new_wui - old_wui)
        * PIXEL_AREA_KM2,
        "net_wui_change_percent": (
            100.0 * (new_wui - old_wui) / old_wui if old_wui else 0.0
        ),
        "changed_area_km2": (valid_pixels - unchanged)
        * PIXEL_AREA_KM2,
        "outside_non_nodata_pixels": baseline_outside,
        "transitions": transitions,
    }


def pairwise(method_file: Path, z_file: Path, valid_file: Path) -> dict:
    intersection = union = common = method_wui = z_wui = 0
    with (
        rasterio.open(method_file) as method,
        rasterio.open(z_file) as zclass,
        rasterio.open(valid_file) as zvalid,
    ):
        if not grids_equal(method, zclass, zvalid):
            raise RuntimeError(f"Pairwise grid mismatch: {method_file}")
        for window in iter_windows(method.width, method.height):
            method_array = method.read(1, window=window)
            z_array = zclass.read(1, window=window)
            domain = (
                np.isin(method_array, [0, 1, 2])
                & (zvalid.read(1, window=window) == 1)
            )
            method_mask = np.isin(method_array, [1, 2]) & domain
            z_mask = np.isin(z_array, [1, 2]) & domain
            intersection += int(np.count_nonzero(method_mask & z_mask))
            union += int(np.count_nonzero(method_mask | z_mask))
            common += int(np.count_nonzero(domain))
            method_wui += int(np.count_nonzero(method_mask))
            z_wui += int(np.count_nonzero(z_mask))
    return {
        "common_valid_pixels": common,
        "intersection_pixels": intersection,
        "union_pixels": union,
        "jaccard": intersection / union if union else math.nan,
        "method_wui_pixels": method_wui,
        "z_wui_pixels": z_wui,
        "intersection_area_km2": intersection * PIXEL_AREA_KM2,
        "union_area_km2": union * PIXEL_AREA_KM2,
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    created_utc = now_utc()

    required = [
        SOURCE_GPKG,
        FORMAL_Z_CLASS,
        FORMAL_Z_VALID,
        REFERENCE_GRID,
        GDAL_RASTERIZE,
        *[
            method_path(method, radius)
            for method in ("P", "S")
            for radius in RADII
        ],
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing inputs: {missing}")

    print("[1/7] Reading formal Colorado WUI-Z blocks", flush=True)
    blocks = gpd.read_file(
        SOURCE_GPKG,
        layer=SOURCE_LAYER,
        engine="fiona",
    )
    required_fields = {
        "GEOID20",
        "ALAND20",
        "HU_Census",
        "Housing_Density",
        "Veg_Percent",
        "WUI_Code",
        "WUI_Label",
        "geometry",
    }
    if missing_fields := sorted(required_fields - set(blocks.columns)):
        raise RuntimeError(f"Missing block fields: {missing_fields}")
    if blocks.crs is None or blocks.crs.to_epsg() != 5070:
        raise RuntimeError(f"Unexpected WUI-Z CRS: {blocks.crs}")
    if blocks["GEOID20"].duplicated().any():
        raise RuntimeError("Duplicate GEOID20 in formal Colorado WUI-Z")
    if not blocks["WUI_Code"].isin([0, 1, 2]).all():
        raise RuntimeError("Unexpected formal WUI_Code")

    old_code = blocks["WUI_Code"].to_numpy(dtype=np.uint8)
    vegetation = blocks["Veg_Percent"].to_numpy(dtype=float)
    density = blocks["Housing_Density"].to_numpy(dtype=float)
    exact_75 = int(
        np.isclose(vegetation, PATCH_THRESHOLD, rtol=0, atol=1e-12).sum()
    )
    exact_50 = int(
        np.isclose(
            vegetation, INTERMIX_THRESHOLD, rtol=0, atol=1e-12
        ).sum()
    )
    exact_617 = int(
        np.isclose(
            density, DENSITY_THRESHOLD, rtol=0, atol=1e-12
        ).sum()
    )

    print("[2/7] Building strict >75% qualifying block patches", flush=True)
    patch_started = time.perf_counter()
    selected = blocks.loc[
        vegetation > PATCH_THRESHOLD, ["geometry"]
    ].copy()
    selected_area_m2 = float(selected.geometry.area.sum())
    dissolved = union_all(selected.geometry.to_numpy())
    components = polygon_parts(dissolved)
    qualifying = [
        geometry
        for geometry in components
        if geometry.area >= PATCH_AREA_M2
    ]
    qualifying_area_m2 = float(sum(g.area for g in qualifying))
    buffer_union = union_all(
        [geometry.buffer(DISTANCE_M) for geometry in qualifying]
    )
    patch_runtime = time.perf_counter() - patch_started
    patch_summary = {
        "source_block_count": int(len(blocks)),
        "selected_gt75_blocks": int(len(selected)),
        "exact_75_blocks_excluded": exact_75,
        "selected_gt75_area_km2": selected_area_m2 / 1_000_000.0,
        "dissolved_component_count": int(len(components)),
        "qualifying_component_count": int(len(qualifying)),
        "qualifying_component_area_km2": qualifying_area_m2
        / 1_000_000.0,
        "distance_m": DISTANCE_M,
        "patch_runtime_seconds": patch_runtime,
    }

    print("[3/7] Reclassifying WUI-Z blocks", flush=True)
    dense = density > DENSITY_THRESHOLD
    intermix = dense & (vegetation >= INTERMIX_THRESHOLD)
    potential_interface = dense & (vegetation < INTERMIX_THRESHOLD)
    candidate_index = blocks.index[potential_interface]
    interface_hits = blocks.loc[
        candidate_index, "geometry"
    ].intersects(buffer_union)
    interface_index = candidate_index[interface_hits.to_numpy()]

    new_code = np.zeros(len(blocks), dtype=np.uint8)
    new_code[intermix] = 1
    new_code[interface_index.to_numpy(dtype=int)] = 2
    blocks["WUI_Code"] = new_code.astype(np.int64)
    blocks["WUI_Label"] = [
        CLASS_LABELS[int(code)] for code in new_code
    ]

    if int(np.count_nonzero((old_code == 1) != (new_code == 1))) != 0:
        raise RuntimeError("Intermix changed although only patch rule changed")
    if len(blocks) != sum(int(np.count_nonzero(new_code == c)) for c in range(3)):
        raise RuntimeError("Block class conservation failed")

    output_gpkg = output / "WUI_Z_Paper_Colorado_SILVIS_GT75.gpkg"
    output_layer = "WUI_Z_Paper_Colorado_SILVIS_GT75"
    blocks.to_file(
        output_gpkg,
        layer=output_layer,
        driver="GPKG",
        engine="fiona",
        index=False,
    )

    print("[4/7] Rasterizing old and new WUI-Z on the frozen grid", flush=True)
    old_reconstructed = output / "WUI_Z_CO_old_reconstructed.tif"
    new_class = output / "WUI_Z_CO_SILVIS_GT75_class.tif"
    new_valid = output / "WUI_Z_CO_SILVIS_GT75_valid_domain.tif"
    with rasterio.open(REFERENCE_GRID) as reference:
        rasterize_field(
            SOURCE_GPKG,
            SOURCE_LAYER,
            old_reconstructed,
            reference,
            field="WUI_Code",
            init=NODATA,
            nodata=NODATA,
        )
        rasterize_field(
            output_gpkg,
            output_layer,
            new_class,
            reference,
            field="WUI_Code",
            init=NODATA,
            nodata=NODATA,
        )
        rasterize_field(
            output_gpkg,
            output_layer,
            new_valid,
            reference,
            burn=1,
            init=0,
            nodata=0,
        )

    baseline_mismatch = 0
    valid_mismatch = 0
    with (
        rasterio.open(old_reconstructed) as reconstructed,
        rasterio.open(FORMAL_Z_CLASS) as formal,
        rasterio.open(new_valid) as candidate_valid,
        rasterio.open(FORMAL_Z_VALID) as formal_valid,
    ):
        if not grids_equal(
            reconstructed, formal, candidate_valid, formal_valid
        ):
            raise RuntimeError("Frozen WUI-Z raster grids differ")
        for window in iter_windows(formal.width, formal.height):
            baseline_mismatch += int(
                np.count_nonzero(
                    reconstructed.read(1, window=window)
                    != formal.read(1, window=window)
                )
            )
            valid_mismatch += int(
                np.count_nonzero(
                    candidate_valid.read(1, window=window)
                    != formal_valid.read(1, window=window)
                )
            )
    if baseline_mismatch != 0 or valid_mismatch != 0:
        raise RuntimeError(
            "WUI-Z raster baseline gate failed: "
            f"class={baseline_mismatch}, valid={valid_mismatch}"
        )

    print("[5/7] Computing vector and raster impacts", flush=True)
    block_rows: list[dict[str, Any]] = []
    area_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    aland = blocks["ALAND20"].to_numpy(dtype=float)
    geometry_area = blocks.geometry.area.to_numpy(dtype=float)
    for scenario, codes in (("OLD_FORMAL", old_code), ("NEW_GT75", new_code)):
        for code in range(3):
            selected_code = codes == code
            area_rows.append(
                {
                    "state": "CO",
                    "scenario": scenario,
                    "class_code": code,
                    "class_label": CLASS_LABELS[code],
                    "block_count": int(selected_code.sum()),
                    "aland20_area_km2": float(aland[selected_code].sum())
                    / 1_000_000.0,
                    "whole_geometry_area_km2": float(
                        geometry_area[selected_code].sum()
                    )
                    / 1_000_000.0,
                }
            )
    for old_value in range(3):
        for new_value in range(3):
            selected_transition = (
                (old_code == old_value) & (new_code == new_value)
            )
            transition_rows.append(
                {
                    "state": "CO",
                    "old_code": old_value,
                    "old_label": CLASS_LABELS[old_value],
                    "new_code": new_value,
                    "new_label": CLASS_LABELS[new_value],
                    "block_count": int(selected_transition.sum()),
                    "aland20_area_km2": float(
                        aland[selected_transition].sum()
                    )
                    / 1_000_000.0,
                    "whole_geometry_area_km2": float(
                        geometry_area[selected_transition].sum()
                    )
                    / 1_000_000.0,
                }
            )

    old_raster_counts = class_counts(FORMAL_Z_CLASS)
    new_raster_counts = class_counts(new_class)
    raster_impact = compare_old_new(
        FORMAL_Z_CLASS, new_class, new_valid
    )
    for scenario, counts in (
        ("OLD_FORMAL", old_raster_counts),
        ("NEW_GT75", new_raster_counts),
    ):
        for code in range(3):
            area_rows.append(
                {
                    "state": "CO",
                    "scenario": scenario,
                    "class_code": code,
                    "class_label": CLASS_LABELS[code],
                    "raster_pixels": counts[code],
                    "raster_area_km2": counts[code] * PIXEL_AREA_KM2,
                }
            )

    block_changed = old_code != new_code
    old_block_wui = np.isin(old_code, [1, 2])
    new_block_wui = np.isin(new_code, [1, 2])
    block_rows.append(
        {
            "state": "CO",
            "old_block_count": int(len(blocks)),
            "new_block_count": int(len(blocks)),
            "changed_blocks": int(block_changed.sum()),
            "old_wui_blocks": int(old_block_wui.sum()),
            "new_wui_blocks": int(new_block_wui.sum()),
            "old_intermix_blocks": int((old_code == 1).sum()),
            "new_intermix_blocks": int((new_code == 1).sum()),
            "old_interface_blocks": int((old_code == 2).sum()),
            "new_interface_blocks": int((new_code == 2).sum()),
            "nonwui_to_interface_blocks": int(
                ((old_code == 0) & (new_code == 2)).sum()
            ),
            "interface_to_nonwui_blocks": int(
                ((old_code == 2) & (new_code == 0)).sum()
            ),
            "intermix_changed_blocks": int(
                np.count_nonzero((old_code == 1) != (new_code == 1))
            ),
            "exact_75_blocks": exact_75,
            "exact_50_blocks": exact_50,
            "exact_6_17_density_blocks": exact_617,
            "block_class_conservation": (
                len(blocks)
                == sum(int(np.count_nonzero(new_code == c)) for c in range(3))
            ),
            "old_raster_reconstruction_mismatch_pixels": baseline_mismatch,
            "valid_domain_mismatch_pixels": valid_mismatch,
            **{
                key: value
                for key, value in raster_impact.items()
                if key != "transitions"
            },
        }
    )
    for (old_value, new_value), pixel_count in raster_impact[
        "transitions"
    ].items():
        transition_rows.append(
            {
                "state": "CO",
                "old_code": old_value,
                "old_label": CLASS_LABELS[old_value],
                "new_code": new_value,
                "new_label": CLASS_LABELS[new_value],
                "raster_pixels": pixel_count,
                "raster_area_km2": pixel_count * PIXEL_AREA_KM2,
            }
        )

    print("[6/7] Computing new P/Z and S/Z Jaccard by radius", flush=True)
    pair_rows: list[dict[str, Any]] = []
    for radius in RADII:
        for method in ("P", "S"):
            result = pairwise(
                method_path(method, radius), new_class, new_valid
            )
            pair_rows.append(
                {
                    "state": "CO",
                    "method_pair": f"WUI-{method}/WUI-Z",
                    "radius_m": radius,
                    "patch_rule": (
                        "Census blocks Veg_Percent>75%; dissolve contiguous; "
                        "component area>=5km2; distance<=2400m"
                    ),
                    **result,
                }
            )

    write_csv(output / "colorado_wuiz_gt75_impact.csv", block_rows)
    write_csv(output / "colorado_wuiz_class_area.csv", area_rows)
    write_csv(output / "colorado_wuiz_transition_matrix.csv", transition_rows)
    write_csv(output / "colorado_wuiz_pairwise_100_1000m.csv", pair_rows)
    (output / "colorado_patch75_summary.json").write_text(
        json.dumps(patch_summary, indent=2) + "\n"
    )

    print("[7/7] Final QC and manifests", flush=True)
    completed_utc = now_utc()
    elapsed = time.perf_counter() - started
    outputs_to_hash = sorted(
        path
        for path in output.iterdir()
        if path.is_file()
        and path.name
        not in {"sha256_manifest.txt", "step47c_status.json"}
    )
    input_rows = []
    for path in [
        SOURCE_GPKG,
        FORMAL_Z_CLASS,
        FORMAL_Z_VALID,
        REFERENCE_GRID,
        *[
            method_path(method, radius)
            for method in ("P", "S")
            for radius in RADII
        ],
    ]:
        input_rows.append(
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    write_csv(output / "input_manifest.csv", input_rows)

    status = {
        "step": "STEP47C_PATCH75_COLORADO_WUIZ",
        "status": "COLORADO_WUIZ_GT75_COMPLETE",
        "created_utc": created_utc,
        "completed_utc": completed_utc,
        "state": "CO",
        "source_wuiz": str(SOURCE_GPKG),
        "output_wuiz": str(output_gpkg),
        "output_class_raster": str(new_class),
        "strict_patch_rule": (
            "Veg_Percent>75%; contiguous dissolved component area>=5km2; "
            "distance<=2400m"
        ),
        "density_rule": "Housing_Density>6.17",
        "intermix_rule": "Veg_Percent>=50%",
        "interface_rule": (
            "Veg_Percent<50% and block intersects <=2400m patch buffer"
        ),
        "baseline_mismatch_pixels": baseline_mismatch,
        "valid_domain_mismatch_pixels": valid_mismatch,
        "block_class_conservation": block_rows[0][
            "block_class_conservation"
        ],
        "intermix_changed_blocks": block_rows[0][
            "intermix_changed_blocks"
        ],
        "pairwise_rows": len(pair_rows),
        "elapsed_seconds": elapsed,
        "upstream_modified": False,
        "environment_modified": False,
        "output_directory": str(output),
    }
    (output / "step47c_status.json").write_text(
        json.dumps(status, indent=2) + "\n"
    )
    qc = [
        "STEP47C COLORADO WUI-Z >75% PATCH QC",
        f"completed_utc={completed_utc}",
        f"source_blocks={len(blocks)}",
        f"changed_blocks={int(block_changed.sum())}",
        f"baseline_mismatch_pixels={baseline_mismatch}",
        f"valid_domain_mismatch_pixels={valid_mismatch}",
        "intermix_changed_blocks=0",
        "block_class_conservation=PASS",
        f"pairwise_rows={len(pair_rows)}",
        f"elapsed_seconds={elapsed:.3f}",
        "upstream_modified=NO",
        "environment_modified=NO",
        "final_status=COLORADO_WUIZ_GT75_COMPLETE",
    ]
    (output / "STEP47C_FINAL_QC.txt").write_text("\n".join(qc) + "\n")
    readme = """# Step47C Colorado WUI-Z strict >75% patch test

This diagnostic run reclassifies the fixed Colorado Census-block WUI-Z product
using the strict qualifying-patch rule tested in Step47 and Step47B.

## Rules

- housing density `>6.17` units/km2;
- Intermix: block `Veg_Percent >=50%`;
- Interface: block `Veg_Percent <50%` and exact intersection with the
  `<=2,400 m` buffer of a qualifying patch;
- qualifying patch: contiguous Census blocks with `Veg_Percent >75%`,
  dissolved and retained when component area is `>=5 km2`.

The formal Step40/46 WUI-Z input is read-only. A reconstruction of the old
WUI-Z raster must match the frozen Step46 raster exactly, and the new valid
domain must match the frozen valid-domain raster exactly.
"""
    (output / "README.md").write_text(readme)

    outputs_to_hash = sorted(
        path
        for path in output.iterdir()
        if path.is_file() and path.name != "sha256_manifest.txt"
    )
    manifest_lines = [
        f"{sha256(path)}  {path.relative_to(output)}"
        for path in outputs_to_hash
    ]
    (output / "sha256_manifest.txt").write_text(
        "\n".join(manifest_lines) + "\n"
    )
    print(json.dumps(status, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
