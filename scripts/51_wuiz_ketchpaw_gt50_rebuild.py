#!/usr/bin/env python3
# Copyright (C) 2026 WUI Mapping Project Authors
# SPDX-License-Identifier: GPL-3.0-only
# Repository version: filesystem paths are loaded through repo_config.py.
# Copy config/paths.example.json to config/paths.json before a new run.
# Raw input data and large intermediate files are stored outside GitHub.

"""STEP51: audit and rebuild WUI-Z with Ketchpaw >50% boundary.

The strict qualifying-patch rule remains unchanged:
Veg_Percent >75%, contiguous component area >=5 km2, and distance <=2.4 km.
Only the local vegetation equality boundary changes relative to Step49:

* Intermix: density >6.17 and Veg_Percent >50%;
* Interface: density >6.17, Veg_Percent <=50%, and block geometry
  intersects the exact 2.4-km qualifying-patch buffer.

Existing Step47/49/50 products are read-only. WUI-P and WUI-S are never
rebuilt by this program.
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from osgeo import ogr
from rasterio.windows import Window
from shapely import from_wkb, get_parts, union_all
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.prepared import prep

ROOT = Path(portable_path("project"))
STEP47_COZ = ROOT / "step47c_patch75_colorado_wuiz_20260729T144953Z"
STEP47_FOUR = ROOT / "step47d_patch75_four_state_psz_20260729T150314Z"
STEP49 = ROOT / "step49_patch75_remaining44_500m_20260729T231000Z"
NATIONAL49 = ROOT / "step49_patch75_national49_500m_20260730T032000Z"
INVENTORY_PATH = NATIONAL49 / "patch75_national49_500m_product_inventory.csv"
GDAL_RASTERIZE = Path(
    portable_path("software", "bin/gdal_rasterize")
)
FIVE = {"CA", "CO", "FL", "PA", "TX"}
PIXEL_AREA_KM2 = 0.0009
NODATA = 255
CLASS_LABELS = {0: "Non-WUI", 1: "Intermix", 2: "Interface"}
PROTOCOL = (
    "DENSITY_GT_6_17;INTERMIX_LOCAL_V_GT_50_PERCENT;"
    "INTERFACE_LOCAL_V_LE_50_PERCENT_AND_DISTANCE_LE_2400M;"
    "PATCH_VEGETATION_FRACTION_GT_75_PERCENT;PATCH_AREA_GE_5_KM2"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(partial, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    frame.to_csv(partial, index=False, float_format="%.12f")
    os.replace(partial, path)


def inventory() -> pd.DataFrame:
    frame = pd.read_csv(INVENTORY_PATH, dtype={"state": str})
    if len(frame) != 49 or frame.state.nunique() != 49:
        raise RuntimeError("Frozen national49 inventory gate failed")
    return frame.sort_values("state").set_index("state")


INVENTORY = inventory()
STATES = INVENTORY.index.tolist()


def state_base(state: str) -> Path:
    if state == "CO":
        return STEP47_COZ
    if state in FIVE:
        return STEP47_FOUR / state
    return STEP49 / "states" / state


def z_vector(state: str) -> Path:
    candidates = sorted(state_base(state).glob("WUI_Z_Paper_*_SILVIS_GT75.gpkg"))
    if len(candidates) != 1:
        raise RuntimeError(f"WUI-Z vector resolution failed for {state}: {candidates}")
    return candidates[0]


def only_layer(path: Path) -> str:
    source = ogr.Open(str(path), 0)
    if source is None or source.GetLayerCount() != 1:
        raise RuntimeError(f"Expected one layer in {path}")
    name = source.GetLayer(0).GetName()
    source = None
    return name


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


def buffer_union(state: str, vector: Path):
    """Return the exact vector <=2.4-km qualifying-patch buffer."""
    if state != "CO":
        candidate = state_base(state) / f"{state}_GT75_patch_buffer.gpkg"
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        source = ogr.Open(str(candidate), 0)
        layer = source.GetLayer(0)
        if layer.GetFeatureCount() != 1:
            raise RuntimeError(f"Expected one patch-buffer feature: {candidate}")
        feature = layer.GetNextFeature()
        geometry = feature.GetGeometryRef()
        # DC has no qualifying >75% component; its deliberately empty
        # one-row patch artifact therefore carries a null geometry.
        value = (
            GeometryCollection()
            if geometry is None
            else from_wkb(bytes(geometry.ExportToWkb()))
        )
        feature = None
        source = None
        return value, str(candidate), False

    # Step47C retained the exact buffer raster but not its vector geometry.
    # Reconstruct the vector using the identical frozen algorithm.
    blocks = gpd.read_file(
        vector,
        layer=only_layer(vector),
        columns=["Veg_Percent"],
        engine="fiona",
    )
    selected = blocks.loc[blocks.Veg_Percent.to_numpy(float) > 75.0, "geometry"]
    dissolved = union_all(selected.to_numpy())
    qualifying = [part for part in polygon_parts(dissolved) if part.area >= 5_000_000.0]
    value = union_all([part.buffer(2400.0) for part in qualifying])
    return value, str(vector) + "#RECONSTRUCTED_IDENTICAL_PATCH_BUFFER", True


def exact50_records(state: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    vector = z_vector(state)
    layer_name = only_layer(vector)
    patch, patch_source, reconstructed = buffer_union(state, vector)
    prepared = prep(patch)
    source = ogr.Open(str(vector), 0)
    layer = source.GetLayerByName(layer_name)
    total_blocks = int(layer.GetFeatureCount())
    layer.SetAttributeFilter('ABS("Veg_Percent" - 50.0) <= 1e-12')
    records: list[dict[str, Any]] = []
    for feature in layer:
        density = float(feature.GetField("Housing_Density"))
        dense = density > 6.17
        current_code = int(feature.GetField("WUI_Code"))
        geometry = from_wkb(bytes(feature.GetGeometryRef().ExportToWkb()))
        in_buffer = bool(prepared.intersects(geometry)) if dense else False
        new_code = (2 if in_buffer else 0) if dense else current_code
        records.append({
            "fid": int(feature.GetFID()),
            "geoid20": str(feature.GetField("GEOID20")),
            "housing_density": density,
            "aland20_m2": int(feature.GetField("ALAND20") or 0),
            "geometry_area_m2": float(geometry.area),
            "dense_gt_6_17": dense,
            "intersects_patch_buffer": in_buffer,
            "current_code": current_code,
            "new_code": new_code,
        })
    layer.SetAttributeFilter(None)
    source = None
    dense_rows = [row for row in records if row["dense_gt_6_17"]]
    current_mismatch = sum(row["current_code"] != 1 for row in dense_rows)
    summary = {
        "state": state,
        "source_vector": str(vector),
        "source_layer": layer_name,
        "patch_buffer_source": patch_source,
        "patch_buffer_reconstructed": reconstructed,
        "total_blocks": total_blocks,
        "exact_50_blocks": len(records),
        "exact_50_dense_blocks": len(dense_rows),
        "exact_50_dense_in_buffer_blocks": sum(
            row["intersects_patch_buffer"] for row in dense_rows
        ),
        "exact_50_dense_outside_buffer_blocks": sum(
            not row["intersects_patch_buffer"] for row in dense_rows
        ),
        "expected_intermix_to_interface_blocks": sum(
            row["new_code"] == 2 for row in dense_rows
        ),
        "expected_intermix_to_nonwui_blocks": sum(
            row["new_code"] == 0 for row in dense_rows
        ),
        "affected_aland20_km2": sum(row["aland20_m2"] for row in dense_rows) / 1e6,
        "affected_geometry_km2": sum(row["geometry_area_m2"] for row in dense_rows) / 1e6,
        "current_dense_exact50_not_intermix": current_mismatch,
        "audit_status": "PASS" if current_mismatch == 0 else "CURRENT_INPUT_MISMATCH",
    }
    return records, summary


def audit_worker(output: Path, state: str) -> None:
    path = output / "audit" / "states" / f"{state}.json"
    if path.exists() and json.loads(path.read_text()).get("audit_status") == "PASS":
        print(f"[STEP51 AUDIT] {state} resumed PASS", flush=True)
        return
    started = time.monotonic()
    _, summary = exact50_records(state)
    summary["elapsed_seconds"] = time.monotonic() - started
    summary["completed_utc"] = utc_now()
    atomic_json(path, summary)
    if summary["audit_status"] != "PASS":
        raise RuntimeError(f"Audit failed for {state}: {summary}")
    print(
        f"[STEP51 AUDIT] {state} PASS exact50={summary['exact_50_blocks']} "
        f"dense={summary['exact_50_dense_blocks']} elapsed="
        f"{summary['elapsed_seconds']:.1f}s",
        flush=True,
    )


def iter_windows(width: int, height: int, tile: int = 2048):
    for row in range(0, height, tile):
        for col in range(0, width, tile):
            yield Window(col, row, min(tile, width-col), min(tile, height-row))


def rasterize(vector: Path, layer: str, output: Path, reference: Path) -> None:
    with rasterio.open(reference) as ref:
        left, bottom, right, top = ref.bounds
        command = [
            str(GDAL_RASTERIZE), "-l", layer, "-a", "WUI_Code",
            "-init", str(NODATA), "-a_nodata", str(NODATA), "-ot", "Byte",
            "-of", "GTiff", "-te", str(left), str(bottom), str(right), str(top),
            "-ts", str(ref.width), str(ref.height), "-a_srs", ref.crs.to_wkt(),
            "-co", "COMPRESS=DEFLATE", "-co", "TILED=YES",
            "-co", "PREDICTOR=2", "-co", "BIGTIFF=IF_SAFER",
            str(vector), str(output),
        ]
    subprocess.run(command, check=True)


def compare_rasters(old_path: Path, new_path: Path, valid_path: Path) -> dict[str, Any]:
    transitions = {(old, new): 0 for old in range(3) for new in range(3)}
    valid_pixels = 0
    with rasterio.open(old_path) as old, rasterio.open(new_path) as new, rasterio.open(valid_path) as valid:
        if not (
            old.width == new.width == valid.width
            and old.height == new.height == valid.height
            and old.transform == new.transform == valid.transform
        ):
            raise RuntimeError("WUI-Z raster grid mismatch")
        for window in iter_windows(old.width, old.height):
            a = old.read(1, window=window)
            b = new.read(1, window=window)
            domain = valid.read(1, window=window) == 1
            valid_pixels += int(domain.sum())
            for old_code in range(3):
                for new_code in range(3):
                    transitions[(old_code, new_code)] += int(
                        np.count_nonzero((a == old_code) & (b == new_code) & domain)
                    )
    changed = sum(value for key, value in transitions.items() if key[0] != key[1])
    old_wui = sum(v for (a, _), v in transitions.items() if a in (1, 2))
    new_wui = sum(v for (_, b), v in transitions.items() if b in (1, 2))
    unexpected = sum(
        value for key, value in transitions.items()
        if key[0] != key[1] and key not in {(1, 0), (1, 2)}
    )
    return {
        "valid_pixels": valid_pixels,
        "changed_pixels": changed,
        "intermix_to_nonwui_pixels": transitions[(1, 0)],
        "intermix_to_interface_pixels": transitions[(1, 2)],
        "unexpected_transition_pixels": unexpected,
        "old_wui_pixels": old_wui,
        "new_wui_pixels": new_wui,
        "old_wui_area_km2": old_wui * PIXEL_AREA_KM2,
        "new_wui_area_km2": new_wui * PIXEL_AREA_KM2,
        "net_wui_area_change_km2": (new_wui - old_wui) * PIXEL_AREA_KM2,
        "changed_area_km2": changed * PIXEL_AREA_KM2,
        "transitions": {f"{a}_to_{b}": v for (a, b), v in transitions.items()},
    }


def rebuild_worker(output: Path, state: str) -> None:
    state_out = output / "states" / state
    checkpoint = state_out / "state_status.json"
    if checkpoint.exists() and json.loads(checkpoint.read_text()).get("status") == "PASS":
        print(f"[STEP51 REBUILD] {state} resumed PASS", flush=True)
        return
    state_out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    records, audit = exact50_records(state)
    if audit["audit_status"] != "PASS":
        raise RuntimeError(f"Audit gate failed for {state}")
    source_vector = Path(audit["source_vector"])
    source_layer = str(audit["source_layer"])
    token = source_vector.stem.removeprefix("WUI_Z_Paper_").removesuffix("_SILVIS_GT75")
    output_vector = state_out / f"WUI_Z_Paper_{token}_KETCHPAW_GT50.gpkg"
    shutil.copy2(source_vector, output_vector)
    changes = [
        (int(row["new_code"]), CLASS_LABELS[int(row["new_code"])], int(row["fid"]))
        for row in records if row["dense_gt_6_17"]
    ]
    # Use the GDAL GeoPackage driver for attribute updates so the standard
    # GeoPackage geometry triggers have their required spatial functions.
    update_source = ogr.Open(str(output_vector), 1)
    if update_source is None:
        raise RuntimeError(f"Cannot open copied GeoPackage for update: {output_vector}")
    update_layer = update_source.GetLayerByName(source_layer)
    for new_code, new_label, fid in changes:
        feature = update_layer.GetFeature(fid)
        if feature is None:
            raise RuntimeError(f"Missing copied feature {state}/{fid}")
        feature.SetField("WUI_Code", new_code)
        feature.SetField("WUI_Label", new_label)
        if update_layer.SetFeature(feature) != 0:
            raise RuntimeError(f"Failed to update copied feature {state}/{fid}")
        feature = None
    update_source = None
    # Verify the vector update before rasterization.
    verify = ogr.Open(str(output_vector), 0)
    verify_layer = verify.GetLayerByName(source_layer)
    verify_layer.SetAttributeFilter(
        'ABS("Veg_Percent" - 50.0) <= 1e-12 AND "Housing_Density" > 6.17'
    )
    observed = {0: 0, 1: 0, 2: 0}
    for feature in verify_layer:
        observed[int(feature.GetField("WUI_Code"))] += 1
    verify = None
    expected = {
        0: audit["expected_intermix_to_nonwui_blocks"],
        1: 0,
        2: audit["expected_intermix_to_interface_blocks"],
    }
    if observed != expected:
        raise RuntimeError(f"Vector update verification failed {state}: {observed} != {expected}")

    old_class = Path(INVENTORY.loc[state, "wui_z_path"])
    old_valid = Path(INVENTORY.loc[state, "wui_z_valid_path"])
    output_class = state_out / f"WUI_Z_{state}_KETCHPAW_GT50_class.tif"
    output_valid = state_out / f"WUI_Z_{state}_KETCHPAW_GT50_valid_domain.tif"
    rasterize(output_vector, source_layer, output_class, old_class)
    shutil.copy2(old_valid, output_valid)
    impact = compare_rasters(old_class, output_class, output_valid)
    if impact["unexpected_transition_pixels"] != 0:
        raise RuntimeError(f"Unexpected raster transition for {state}")
    status = {
        "state": state,
        "status": "PASS",
        "protocol": PROTOCOL,
        "source_vector": str(source_vector),
        "source_vector_sha256": sha256(source_vector),
        "output_vector": str(output_vector),
        "output_vector_sha256": sha256(output_vector),
        "source_class": str(old_class),
        "source_class_sha256": sha256(old_class),
        "output_class": str(output_class),
        "output_class_sha256": sha256(output_class),
        "output_valid": str(output_valid),
        "output_valid_sha256": sha256(output_valid),
        "audit": audit,
        "vector_codes_at_dense_exact50": observed,
        "raster_impact": impact,
        "elapsed_seconds": time.monotonic() - started,
        "completed_utc": utc_now(),
        "source_modified": False,
        "environment_modified": False,
    }
    atomic_json(checkpoint, status)
    print(
        f"[STEP51 REBUILD] {state} PASS changed_pixels={impact['changed_pixels']} "
        f"elapsed={status['elapsed_seconds']/60:.1f}m",
        flush=True,
    )


def run_subprocess(output: Path, mode: str, state: str) -> dict[str, str]:
    log = output / "logs" / f"{mode}_{state}.log"
    env = os.environ.copy()
    env["PROJ_DATA"] = portable_path("software", "share/proj")
    env["GDAL_DATA"] = portable_path("software", "share/gdal")
    with log.open("w") as stream:
        completed = subprocess.run(
            [
                sys.executable, str(Path(__file__).resolve()),
                "--output", str(output), "--mode", mode,
                "--worker-state", state,
            ],
            stdout=stream, stderr=subprocess.STDOUT, text=True, env=env,
        )
    if completed.returncode:
        raise RuntimeError(f"{mode} failed for {state}; see {log}")
    return {"state": state, "status": "PASS", "log": str(log)}


def aggregate_audit(output: Path) -> None:
    rows = [
        json.loads((output / "audit" / "states" / f"{state}.json").read_text())
        for state in STATES
    ]
    frame = pd.DataFrame(rows)
    atomic_csv(output / "audit" / "wuiz_exact50_boundary_audit_49.csv", frame)
    totals = {
        key: int(frame[key].sum())
        for key in [
            "total_blocks", "exact_50_blocks", "exact_50_dense_blocks",
            "exact_50_dense_in_buffer_blocks", "exact_50_dense_outside_buffer_blocks",
            "expected_intermix_to_interface_blocks",
            "expected_intermix_to_nonwui_blocks",
            "current_dense_exact50_not_intermix",
        ]
    }
    totals.update({
        "state_units": len(frame),
        "states_with_dense_exact50": int((frame.exact_50_dense_blocks > 0).sum()),
        "states_with_expected_wui_loss": int(
            (frame.expected_intermix_to_nonwui_blocks > 0).sum()
        ),
        "audit_status": "PASS" if frame.audit_status.eq("PASS").all() else "FAIL",
        "protocol": PROTOCOL,
        "completed_utc": utc_now(),
    })
    atomic_json(output / "audit" / "audit_status.json", totals)
    if totals["audit_status"] != "PASS":
        raise RuntimeError("National exact50 audit failed")


def aggregate_rebuild(output: Path) -> None:
    rows = [
        json.loads((output / "states" / state / "state_status.json").read_text())
        for state in STATES
    ]
    inventory_rows = []
    impact_rows = []
    for row in rows:
        impact = dict(row["raster_impact"])
        impact.pop("transitions", None)
        impact_rows.append({"state": row["state"], **impact})
        inventory_rows.append({
            "state": row["state"],
            "wui_p_path": INVENTORY.loc[row["state"], "wui_p_path"],
            "wui_s_path": INVENTORY.loc[row["state"], "wui_s_path"],
            "wui_z_path": row["output_class"],
            "wui_z_sha256": row["output_class_sha256"],
            "wui_z_valid_path": row["output_valid"],
            "wui_z_valid_sha256": row["output_valid_sha256"],
            "wui_z_vector_path": row["output_vector"],
            "wui_z_vector_sha256": row["output_vector_sha256"],
            "protocol": PROTOCOL,
        })
    impact = pd.DataFrame(impact_rows)
    products = pd.DataFrame(inventory_rows)
    atomic_csv(output / "wuiz_ketchpaw_gt50_raster_impact_49.csv", impact)
    atomic_csv(output / "wuiz_ketchpaw_gt50_product_inventory_49.csv", products)
    total_changed = int(impact.changed_pixels.sum())
    total_unexpected = int(impact.unexpected_transition_pixels.sum())
    status = {
        "step": "STEP51_WUIZ_KETCHPAW_GT50_REBUILD",
        "status": "WUIZ_KETCHPAW_GT50_REBUILD_COMPLETE" if total_unexpected == 0 else "QC_REVIEW_REQUIRED",
        "parent_run": str(NATIONAL49),
        "protocol": PROTOCOL,
        "state_units": len(rows),
        "state_pass": sum(row["status"] == "PASS" for row in rows),
        "changed_pixels": total_changed,
        "changed_area_km2": float(impact.changed_area_km2.sum()),
        "intermix_to_interface_pixels": int(impact.intermix_to_interface_pixels.sum()),
        "intermix_to_nonwui_pixels": int(impact.intermix_to_nonwui_pixels.sum()),
        "net_wui_area_change_km2": float(impact.net_wui_area_change_km2.sum()),
        "unexpected_transition_pixels": total_unexpected,
        "wui_p_rebuilt": False,
        "wui_s_rebuilt": False,
        "environment_modified": False,
        "completed_utc": utc_now(),
    }
    atomic_json(output / "step51_rebuild_status.json", status)
    qc = [
        "STEP51 WUI-Z KETCHPAW GT50 REBUILD QC",
        *[f"{key}={value}" for key, value in status.items()],
    ]
    (output / "STEP51_REBUILD_QC.txt").write_text("\n".join(qc) + "\n")
    if status["status"] != "WUIZ_KETCHPAW_GT50_REBUILD_COMPLETE":
        raise RuntimeError("Step51 rebuild QC failed")


def run_master(output: Path, mode: str, states: list[str], workers: int) -> None:
    started = time.monotonic()
    completed: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_subprocess, output, mode, state): state for state in states
        }
        for index, future in enumerate(as_completed(futures), 1):
            state = futures[future]
            completed.append(future.result())
            elapsed = time.monotonic() - started
            eta = elapsed * (len(states) - index) / index
            print(
                f"[STEP51 {mode.upper()}] state={state} completed={index}/{len(states)} "
                f"percent={100*index/len(states):.2f} elapsed={elapsed/60:.1f}m "
                f"ETA={eta/60:.1f}m",
                flush=True,
            )
    atomic_csv(output / f"{mode}_state_status.csv", pd.DataFrame(completed))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=["audit", "rebuild"])
    parser.add_argument("--worker-state", choices=STATES, default="")
    parser.add_argument("--canary-state", choices=STATES, default="")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for folder in ["audit/states", "states", "logs"]:
        (output / folder).mkdir(parents=True, exist_ok=True)
    if args.worker_state:
        if args.mode == "audit":
            audit_worker(output, args.worker_state)
        else:
            rebuild_worker(output, args.worker_state)
        return
    states = [args.canary_state] if args.canary_state else STATES
    run_master(output, args.mode, states, args.workers)
    if args.canary_state:
        atomic_json(output / f"{args.mode}_canary_status.json", {
            "mode": args.mode, "state": args.canary_state,
            "status": "PASS", "completed_utc": utc_now(),
        })
        return
    if args.mode == "audit":
        aggregate_audit(output)
    else:
        aggregate_rebuild(output)


if __name__ == "__main__":
    main()
