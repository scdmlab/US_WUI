#!/usr/bin/env python3
# Copyright (C) 2026 WUI Mapping Project Authors
# SPDX-License-Identifier: GPL-3.0-only
# Repository version: filesystem paths are loaded through repo_config.py.
# Copy config/paths.example.json to config/paths.json before a new run.
# Raw input data and large intermediate files are stored outside GitHub.

"""Strict >75% WUI-P/WUI-S/WUI-Z rebuild for CA, FL, PA, or TX.

Each invocation processes exactly one state and writes an independent,
diagnostic output directory. Formal Step40--46 products are read-only.
"""

from __future__ import annotations

from repo_config import portable_path

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.windows import Window
from shapely import get_parts, union_all
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from skimage.morphology import disk


ROOT = Path(portable_path("project"))
STEP43 = ROOT / "step43_wuip_p2_formal_rebuild_20260727T180822Z"
STEP46 = ROOT / "step46_wuiz_pairwise_completion_20260728T203108Z"
BASE_SCRIPT = ROOT / "scripts/47_patch75_silvis_colorado_canary.py"
Z_SCRIPT = ROOT / "scripts/47c_patch75_colorado_wuiz.py"
WUIS_BASE = Path(portable_path("data", "WUI_S_Paper"))
WUIZ_BASE = Path(portable_path("data", "WUI_Z_Results"))
CENTROID_BASE = Path(
    portable_path("data", "mbf_work/centroids_5070")
)

STATE_CONFIG = {
    "CA": {"name": "California"},
    "FL": {"name": "Florida"},
    "PA": {"name": "Pennsylvania"},
    "TX": {"name": "Texas"},
}
RADII = tuple(range(100, 1001, 100))
DENSITY_THRESHOLD = 6.17
INTERMIX_THRESHOLD = 50.0
PATCH_THRESHOLD = 75.0
PATCH_AREA_M2 = 5_000_000.0
DISTANCE_M = 2_400.0
PIXEL_AREA_KM2 = 0.0009
NODATA = 255
LABELS = {0: "Non-WUI", 1: "Intermix", 2: "Interface"}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def now_utc() -> str:
    import datetime as dt

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
    parts: list[Polygon] = []
    stack = [geometry]
    while stack:
        item = stack.pop()
        if item is None or item.is_empty:
            continue
        if isinstance(item, Polygon):
            parts.append(item)
        elif isinstance(item, (MultiPolygon, GeometryCollection)):
            stack.extend(list(get_parts(item)))
    return parts


def paths(state: str) -> dict[str, Path | str]:
    name = STATE_CONFIG[state]["name"]
    return {
        "name": name,
        "p_sparse": STEP43 / f"intermediate/{state}_p2_sparse_cells.npz",
        "wuis_dir": WUIS_BASE / name,
        "wild": WUIS_BASE / name / f"{name}_wildland_bin.tif",
        "distance": WUIS_BASE / name / f"{name}_dist_to_largepatch.tif",
        "centroids": (
            CENTROID_BASE / f"MBF_{name}_centroids_5070.gpkg"
        ),
        "z_source": WUIZ_BASE / f"WUI_Z_Paper_{name}.gpkg",
        "z_layer": f"WUI_Z_Paper_{name}",
        "z_formal_class": (
            STEP46 / f"cache/wuiz_masks/{state}/WUI_Z_{state}_class.tif"
        ),
        "z_formal_valid": (
            STEP46
            / f"cache/wuiz_masks/{state}/WUI_Z_{state}_valid_domain.tif"
        ),
        "reference": (
            STEP43
            / f"rasters_500m/{state}/"
            f"WUI_P_P2_{state}_r0500m.tif"
        ),
    }


def official_p(state: str, radius: int) -> Path:
    raster_group = (
        "rasters_500m" if radius == 500 else "rasters_sensitivity"
    )
    return (
        STEP43
        / f"{raster_group}/{state}/"
        f"WUI_P_P2_{state}_r{radius:04d}m.tif"
    )


def official_s(name: str, radius: int) -> Path:
    return (
        WUIS_BASE / name / f"WUI_S_{name}_r{radius:04d}m.tif"
    )


def binary_pair(path_a: Path, path_b: Path) -> dict[str, Any]:
    valid = intersection = union = 0
    with rasterio.open(path_a) as a, rasterio.open(path_b) as b:
        if (
            a.width != b.width
            or a.height != b.height
            or a.transform != b.transform
        ):
            raise RuntimeError(f"Pair grid mismatch: {path_a} {path_b}")
        for _, window in a.block_windows(1):
            aa = a.read(1, window=window)
            bb = b.read(1, window=window)
            domain = np.isin(aa, [0, 1, 2]) & np.isin(
                bb, [0, 1, 2]
            )
            aw = np.isin(aa, [1, 2]) & domain
            bw = np.isin(bb, [1, 2]) & domain
            valid += int(np.count_nonzero(domain))
            intersection += int(np.count_nonzero(aw & bw))
            union += int(np.count_nonzero(aw | bw))
    return {
        "common_valid_pixels": valid,
        "intersection_pixels": intersection,
        "union_pixels": union,
        "jaccard": intersection / union if union else math.nan,
        "intersection_area_km2": intersection * PIXEL_AREA_KM2,
        "union_area_km2": union * PIXEL_AREA_KM2,
    }


def build_patch_and_z(
    state: str,
    cfg: dict[str, Path | str],
    output: Path,
    reference,
    zmod,
) -> tuple[Path, Path, Path, dict, dict, list[dict], list[dict]]:
    name = str(cfg["name"])
    z_source = Path(cfg["z_source"])
    z_layer = str(cfg["z_layer"])
    print(f"[{state} 1/8] Reading formal WUI-Z blocks", flush=True)
    blocks = gpd.read_file(z_source, layer=z_layer, engine="fiona")
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
    missing_fields = sorted(required_fields - set(blocks.columns))
    if missing_fields:
        raise RuntimeError(f"Missing WUI-Z fields: {missing_fields}")
    if blocks.crs is None or blocks.crs.to_epsg() != 5070:
        raise RuntimeError(f"Unexpected {state} WUI-Z CRS: {blocks.crs}")
    if blocks.GEOID20.duplicated().any():
        raise RuntimeError(f"Duplicate GEOID20: {state}")
    old_code = blocks.WUI_Code.to_numpy(dtype=np.uint8)
    if not np.isin(old_code, [0, 1, 2]).all():
        raise RuntimeError(f"Unexpected old WUI-Z code: {state}")
    vegetation = blocks.Veg_Percent.to_numpy(dtype=float)
    density = blocks.Housing_Density.to_numpy(dtype=float)

    print(f"[{state} 2/8] Building strict >75% block patches", flush=True)
    patch_started = time.perf_counter()
    selected = blocks.loc[
        vegetation > PATCH_THRESHOLD, ["geometry"]
    ].copy()
    dissolved = union_all(selected.geometry.to_numpy())
    components = polygon_parts(dissolved)
    qualifying = [
        geometry
        for geometry in components
        if geometry.area >= PATCH_AREA_M2
    ]
    buffer_union = union_all(
        [geometry.buffer(DISTANCE_M) for geometry in qualifying]
    )
    exact_75 = int(
        np.isclose(
            vegetation, PATCH_THRESHOLD, rtol=0, atol=1e-12
        ).sum()
    )
    patch_vector = output / f"{state}_GT75_patch_buffer.gpkg"
    patch_layer = f"{state}_GT75_buffer_2400m"
    gpd.GeoDataFrame(
        {"patch_rule": ["GT75_AREA_GE_5KM2_BUFFER_2400M"]},
        geometry=[buffer_union],
        crs=blocks.crs,
    ).to_file(
        patch_vector,
        layer=patch_layer,
        driver="GPKG",
        engine="fiona",
        index=False,
    )
    patch_raster = output / f"{state}_GT75_patch_buffer_30m.tif"
    zmod.rasterize_field(
        patch_vector,
        patch_layer,
        patch_raster,
        reference,
        burn=1,
        init=0,
        nodata=0,
    )
    patch_pixels = 0
    with rasterio.open(patch_raster) as patch:
        if (
            patch.width != reference.width
            or patch.height != reference.height
            or patch.transform != reference.transform
        ):
            raise RuntimeError(f"Patch raster grid mismatch: {state}")
        for _, window in patch.block_windows(1):
            values = patch.read(1, window=window)
            if not np.isin(values, [0, 1]).all():
                raise RuntimeError(f"Patch raster code mismatch: {state}")
            patch_pixels += int(np.count_nonzero(values == 1))
    patch_summary = {
        "state": state,
        "state_name": name,
        "source_block_count": len(blocks),
        "selected_gt75_blocks": len(selected),
        "exact_75_blocks_excluded": exact_75,
        "selected_gt75_area_km2": float(selected.geometry.area.sum())
        / 1_000_000.0,
        "dissolved_component_count": len(components),
        "qualifying_component_count": len(qualifying),
        "qualifying_component_area_km2": float(
            sum(geometry.area for geometry in qualifying)
        )
        / 1_000_000.0,
        "patch_buffer_pixels": patch_pixels,
        "patch_buffer_area_km2": patch_pixels * PIXEL_AREA_KM2,
        "patch_runtime_seconds": time.perf_counter() - patch_started,
    }

    print(f"[{state} 3/8] Reclassifying fixed WUI-Z blocks", flush=True)
    dense = density > DENSITY_THRESHOLD
    intermix = dense & (vegetation >= INTERMIX_THRESHOLD)
    potential_interface = dense & (vegetation < INTERMIX_THRESHOLD)
    candidate_index = blocks.index[potential_interface]
    hits = blocks.loc[candidate_index, "geometry"].intersects(buffer_union)
    interface_index = candidate_index[hits.to_numpy()]
    new_code = np.zeros(len(blocks), dtype=np.uint8)
    new_code[intermix] = 1
    positions = blocks.index.get_indexer(interface_index)
    if np.any(positions < 0):
        raise RuntimeError(f"Interface index lookup failed: {state}")
    new_code[positions] = 2
    intermix_changes = int(
        np.count_nonzero((old_code == 1) != (new_code == 1))
    )
    if intermix_changes:
        raise RuntimeError(
            f"Intermix changed under patch-only update: {state}"
        )
    blocks["WUI_Code"] = new_code.astype(np.int64)
    blocks["WUI_Label"] = [LABELS[int(code)] for code in new_code]
    z_output = output / f"WUI_Z_Paper_{name}_SILVIS_GT75.gpkg"
    z_output_layer = f"WUI_Z_Paper_{name}_SILVIS_GT75"
    blocks.to_file(
        z_output,
        layer=z_output_layer,
        driver="GPKG",
        engine="fiona",
        index=False,
    )

    old_reconstructed = output / f"WUI_Z_{state}_old_reconstructed.tif"
    new_z_class = output / f"WUI_Z_{state}_SILVIS_GT75_class.tif"
    new_z_valid = output / f"WUI_Z_{state}_SILVIS_GT75_valid_domain.tif"
    zmod.rasterize_field(
        z_source,
        z_layer,
        old_reconstructed,
        reference,
        field="WUI_Code",
        init=NODATA,
        nodata=NODATA,
    )
    zmod.rasterize_field(
        z_output,
        z_output_layer,
        new_z_class,
        reference,
        field="WUI_Code",
        init=NODATA,
        nodata=NODATA,
    )
    zmod.rasterize_field(
        z_output,
        z_output_layer,
        new_z_valid,
        reference,
        burn=1,
        init=0,
        nodata=0,
    )

    baseline_mismatch = valid_mismatch = 0
    with (
        rasterio.open(old_reconstructed) as reconstructed,
        rasterio.open(Path(cfg["z_formal_class"])) as formal,
        rasterio.open(new_z_valid) as new_valid,
        rasterio.open(Path(cfg["z_formal_valid"])) as formal_valid,
    ):
        if not zmod.grids_equal(
            reconstructed, formal, new_valid, formal_valid
        ):
            raise RuntimeError(f"WUI-Z frozen grid mismatch: {state}")
        for window in zmod.iter_windows(formal.width, formal.height):
            baseline_mismatch += int(
                np.count_nonzero(
                    reconstructed.read(1, window=window)
                    != formal.read(1, window=window)
                )
            )
            valid_mismatch += int(
                np.count_nonzero(
                    new_valid.read(1, window=window)
                    != formal_valid.read(1, window=window)
                )
            )
    if baseline_mismatch or valid_mismatch:
        raise RuntimeError(
            f"WUI-Z baseline failed {state}: "
            f"class={baseline_mismatch} valid={valid_mismatch}"
        )

    raster_impact = zmod.compare_old_new(
        Path(cfg["z_formal_class"]), new_z_class, new_z_valid
    )
    aland = blocks.ALAND20.to_numpy(dtype=float)
    geom_area = blocks.geometry.area.to_numpy(dtype=float)
    area_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    for scenario, codes in (
        ("OLD_FORMAL", old_code),
        ("NEW_GT75", new_code),
    ):
        for code in range(3):
            chosen = codes == code
            area_rows.append(
                {
                    "state": state,
                    "scenario": scenario,
                    "class_code": code,
                    "class_label": LABELS[code],
                    "block_count": int(chosen.sum()),
                    "aland20_area_km2": float(aland[chosen].sum())
                    / 1_000_000.0,
                    "whole_geometry_area_km2": float(
                        geom_area[chosen].sum()
                    )
                    / 1_000_000.0,
                }
            )
    for old_value in range(3):
        for new_value in range(3):
            chosen = (old_code == old_value) & (new_code == new_value)
            transition_rows.append(
                {
                    "state": state,
                    "basis": "CENSUS_BLOCK",
                    "old_code": old_value,
                    "old_label": LABELS[old_value],
                    "new_code": new_value,
                    "new_label": LABELS[new_value],
                    "count": int(chosen.sum()),
                    "aland20_area_km2": float(aland[chosen].sum())
                    / 1_000_000.0,
                    "whole_geometry_area_km2": float(
                        geom_area[chosen].sum()
                    )
                    / 1_000_000.0,
                }
            )
    for (old_value, new_value), count in raster_impact[
        "transitions"
    ].items():
        transition_rows.append(
            {
                "state": state,
                "basis": "30M_RASTER",
                "old_code": old_value,
                "old_label": LABELS[old_value],
                "new_code": new_value,
                "new_label": LABELS[new_value],
                "count": count,
                "raster_area_km2": count * PIXEL_AREA_KM2,
            }
        )

    block_impact = {
        "state": state,
        "state_name": name,
        "block_count": len(blocks),
        "changed_blocks": int(np.count_nonzero(old_code != new_code)),
        "old_wui_blocks": int(np.isin(old_code, [1, 2]).sum()),
        "new_wui_blocks": int(np.isin(new_code, [1, 2]).sum()),
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
        "intermix_changed_blocks": intermix_changes,
        "exact_75_blocks": exact_75,
        "exact_50_blocks": int(
            np.isclose(
                vegetation, INTERMIX_THRESHOLD, rtol=0, atol=1e-12
            ).sum()
        ),
        "exact_6_17_density_blocks": int(
            np.isclose(
                density, DENSITY_THRESHOLD, rtol=0, atol=1e-12
            ).sum()
        ),
        "baseline_mismatch_pixels": baseline_mismatch,
        "valid_domain_mismatch_pixels": valid_mismatch,
        **{
            key: value
            for key, value in raster_impact.items()
            if key != "transitions"
        },
    }
    del blocks, selected, dissolved, components, qualifying, buffer_union
    return (
        patch_raster,
        new_z_class,
        new_z_valid,
        patch_summary,
        block_impact,
        area_rows,
        transition_rows,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True, choices=sorted(STATE_CONFIG))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    state = args.state
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "rasters/WUI-P").mkdir(parents=True)
    (output / "rasters/WUI-S").mkdir(parents=True)
    started = time.perf_counter()
    created_utc = now_utc()
    cfg = paths(state)
    name = str(cfg["name"])
    base = load_module(BASE_SCRIPT, "step47_base")
    zmod = load_module(Z_SCRIPT, "step47_z")

    required = [
        BASE_SCRIPT,
        Z_SCRIPT,
        Path(cfg["p_sparse"]),
        Path(cfg["wild"]),
        Path(cfg["distance"]),
        Path(cfg["centroids"]),
        Path(cfg["z_source"]),
        Path(cfg["z_formal_class"]),
        Path(cfg["z_formal_valid"]),
        Path(cfg["reference"]),
        *[official_p(state, radius) for radius in RADII],
        *[official_s(name, radius) for radius in RADII],
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {state} inputs: {missing}")

    base.WILD = Path(cfg["wild"])
    base.CURRENT_DISTANCE = Path(cfg["distance"])
    base.MBF_CENTROIDS = Path(cfg["centroids"])
    base.MBF_LAYER = "centroids"

    with rasterio.open(Path(cfg["reference"])) as reference:
        (
            patch_raster,
            new_z_class,
            new_z_valid,
            patch_summary,
            z_impact,
            z_area_rows,
            z_transition_rows,
        ) = build_patch_and_z(
            state, cfg, output, reference, zmod
        )

        print(
            f"[{state} 4/8] Rasterizing WUI-S building centroids",
            flush=True,
        )
        s_count, s_count_meta = base.build_wuis_count_raster(
            output / f"WUI_S_{state}_centroid_count_30m.tif",
            reference,
        )
        sparse = np.load(Path(cfg["p_sparse"]))
        p_rows = sparse["rows"].astype(np.int32)
        p_cols = sparse["cols"].astype(np.int32)
        p_counts = sparse["counts"].astype(np.uint32)

        print(
            (
                f"[{state} 5/8] Generating {len(RADII)}-radius "
                "WUI-P/WUI-S"
            ),
            flush=True,
        )
        impact_rows: list[dict[str, Any]] = []
        output_paths: dict[tuple[str, int], Path] = {}
        total = 2 * len(RADII)
        done = 0
        for radius in RADII:
            radius_px = max(int(round(radius / 30.0)), 1)
            kernel = disk(radius_px).astype(np.float32)
            minimum_count = (
                int(
                    math.floor(
                        DENSITY_THRESHOLD
                        * math.pi
                        * radius**2
                        / 1_000_000.0
                    )
                )
                + 1
            )
            for method in ("WUI-P", "WUI-S"):
                scenario_started = time.perf_counter()
                method_code = method.replace("-", "_")
                out = (
                    output
                    / "rasters"
                    / method
                    / f"{method_code}_{state}_r{radius:04d}m_SILVIS_GT75.tif"
                )
                if method == "WUI-P":
                    stats = base.classify(
                        out,
                        method,
                        p_rows,
                        p_cols,
                        p_counts,
                        None,
                        patch_raster,
                        reference,
                        kernel,
                        radius_px=radius_px,
                        minimum_count=minimum_count,
                        official_path_override=official_p(state, radius),
                    )
                else:
                    stats = base.classify(
                        out,
                        method,
                        None,
                        None,
                        None,
                        s_count,
                        patch_raster,
                        reference,
                        kernel,
                        radius_px=radius_px,
                        minimum_count=minimum_count,
                        official_path_override=official_s(name, radius),
                    )
                done += 1
                elapsed = time.perf_counter() - started
                eta = elapsed * (total - done) / done
                print(
                    f"[{state}] radius={radius} method={method} "
                    f"completed={done}/{total} "
                    f"percent={100*done/total:.2f} "
                    f"elapsed_s={elapsed:.1f} ETA_s={eta:.1f}",
                    flush=True,
                )
                output_paths[(method, radius)] = out
                impact_rows.append(
                    {
                        "state": state,
                        "method": method,
                        "radius_m": radius,
                        "radius_px": radius_px,
                        "kernel_cells": int(kernel.sum()),
                        "minimum_integer_count": minimum_count,
                        "density_rule": "D>6.17",
                        "intermix_rule": "V>=50%",
                        "interface_rule": "V<50% and distance<=2400m",
                        "patch_rule": (
                            "blocks Veg_Percent>75%; dissolve contiguous; "
                            "component area>=5km2"
                        ),
                        **stats,
                        "classification_sha256": sha256(out),
                        "wall_runtime_seconds": (
                            time.perf_counter() - scenario_started
                        ),
                        "output_path": str(out),
                    }
                )

    print(f"[{state} 6/8] Computing P/S/Z pairwise comparisons", flush=True)
    pair_rows: list[dict[str, Any]] = []
    for radius in RADII:
        p_path = output_paths[("WUI-P", radius)]
        s_path = output_paths[("WUI-S", radius)]
        pair_rows.append(
            {
                "state": state,
                "radius_m": radius,
                "method_pair": "WUI-P/WUI-S",
                "z_scenario": "NOT_APPLICABLE",
                **binary_pair(p_path, s_path),
            }
        )
        for method, method_file in (
            ("WUI-P", p_path),
            ("WUI-S", s_path),
        ):
            for z_scenario, z_class, z_valid in (
                (
                    "OLD_FORMAL_Z",
                    Path(cfg["z_formal_class"]),
                    Path(cfg["z_formal_valid"]),
                ),
                ("NEW_GT75_Z", new_z_class, new_z_valid),
            ):
                pair_rows.append(
                    {
                        "state": state,
                        "radius_m": radius,
                        "method_pair": f"{method}/WUI-Z",
                        "z_scenario": z_scenario,
                        **zmod.pairwise(method_file, z_class, z_valid),
                    }
                )

    write_csv(output / f"{state}_psz_gt75_impact.csv", impact_rows)
    write_csv(output / f"{state}_wuiz_gt75_impact.csv", [z_impact])
    write_csv(output / f"{state}_wuiz_class_area.csv", z_area_rows)
    write_csv(
        output / f"{state}_wuiz_transition_matrix.csv",
        z_transition_rows,
    )
    write_csv(output / f"{state}_psz_pairwise.csv", pair_rows)
    write_csv(output / f"{state}_patch75_summary.csv", [patch_summary])
    (output / f"{state}_wuis_count_metadata.json").write_text(
        json.dumps(s_count_meta, indent=2) + "\n"
    )

    print(f"[{state} 7/8] Running final QC", flush=True)
    baseline_failures = [
        row
        for row in impact_rows
        if int(row["baseline_mismatch_pixels"]) != 0
    ]
    conservation_failures = [
        row
        for row in impact_rows
        if sum(
            int(row[key])
            for key in (
                "new_non_wui",
                "new_intermix",
                "new_interface",
            )
        )
        != int(row["valid_pixels"])
    ]
    output_rasters = sorted((output / "rasters").rglob("*.tif"))
    expected_method_rasters = 2 * len(RADII)
    if len(output_rasters) != expected_method_rasters:
        raise RuntimeError(
            f"Expected {expected_method_rasters} P/S rasters for {state}, "
            f"got {len(output_rasters)}"
        )
    final_status = (
        "STATE_PSZ_GT75_COMPLETE"
        if (
            not baseline_failures
            and not conservation_failures
            and int(z_impact["baseline_mismatch_pixels"]) == 0
            and int(z_impact["valid_domain_mismatch_pixels"]) == 0
            and int(z_impact["intermix_changed_blocks"]) == 0
        )
        else "QC_REVIEW_REQUIRED"
    )
    elapsed = time.perf_counter() - started
    status = {
        "step": "STEP47D_PATCH75_FOUR_STATE_PSZ",
        "status": final_status,
        "state": state,
        "state_name": name,
        "created_utc": created_utc,
        "completed_utc": now_utc(),
        "radii_m": RADII,
        "wui_p_raster_count": len(RADII),
        "wui_s_raster_count": len(RADII),
        "wui_z_vector_count": 1,
        "wui_z_class_raster_count": 1,
        "baseline_failure_count": len(baseline_failures),
        "class_conservation_failure_count": len(
            conservation_failures
        ),
        "wuiz_baseline_mismatch_pixels": int(
            z_impact["baseline_mismatch_pixels"]
        ),
        "wuiz_valid_domain_mismatch_pixels": int(
            z_impact["valid_domain_mismatch_pixels"]
        ),
        "wuiz_intermix_changed_blocks": int(
            z_impact["intermix_changed_blocks"]
        ),
        "pairwise_rows": len(pair_rows),
        "elapsed_seconds": elapsed,
        "upstream_modified": False,
        "environment_modified": False,
        "output_directory": str(output),
    }
    (output / "step47d_state_status.json").write_text(
        json.dumps(status, indent=2) + "\n"
    )
    qc_lines = [
        f"STEP47D {state} STRICT GT75 P/S/Z QC",
        f"completed_utc={status['completed_utc']}",
        f"p_rasters={len(RADII)}",
        f"s_rasters={len(RADII)}",
        f"baseline_failure_count={len(baseline_failures)}",
        (
            "class_conservation_failure_count="
            f"{len(conservation_failures)}"
        ),
        (
            "wuiz_baseline_mismatch_pixels="
            f"{z_impact['baseline_mismatch_pixels']}"
        ),
        (
            "wuiz_valid_domain_mismatch_pixels="
            f"{z_impact['valid_domain_mismatch_pixels']}"
        ),
        (
            "wuiz_intermix_changed_blocks="
            f"{z_impact['intermix_changed_blocks']}"
        ),
        f"pairwise_rows={len(pair_rows)}",
        f"elapsed_seconds={elapsed:.3f}",
        "upstream_modified=NO",
        "environment_modified=NO",
        f"final_status={final_status}",
    ]
    (output / "STEP47D_STATE_FINAL_QC.txt").write_text(
        "\n".join(qc_lines) + "\n"
    )
    radius_count = len(RADII)
    readme = f"""# Step47D {state} strict >75% WUI-P/WUI-S/WUI-Z

Diagnostic rebuild for {name}. The run generated {radius_count} WUI-P
raster(s), {radius_count} WUI-S raster(s), and one fixed Census-block
WUI-Z product.

Rules: D>6.17; Intermix V>=50%; Interface V<50% and distance<=2,400 m;
qualifying patch uses contiguous Census blocks with Veg_Percent>75% and
component area>=5 km2.

No formal Step40--46 product was modified.
"""
    (output / "README.md").write_text(readme)

    print(f"[{state} 8/8] Hashing output artifacts", flush=True)
    input_rows = []
    for path in required:
        input_rows.append(
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    write_csv(output / "input_manifest.csv", input_rows)
    artifacts = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "sha256_manifest.txt"
    )
    with (output / "sha256_manifest.txt").open(
        "w", encoding="utf-8"
    ) as stream:
        for artifact in artifacts:
            stream.write(
                f"{sha256(artifact)}  {artifact.relative_to(output)}\n"
            )
    print(json.dumps(status, indent=2), flush=True)
    return 0 if final_status == "STATE_PSZ_GT75_COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
