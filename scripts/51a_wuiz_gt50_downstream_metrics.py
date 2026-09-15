#!/usr/bin/env python3
# Copyright (C) 2026 WUI Mapping Project Authors
# SPDX-License-Identifier: GPL-3.0-only
# Repository version: filesystem paths are loaded through repo_config.py.
# Copy config/paths.example.json to config/paths.json before a new run.
# Raw input data and large intermediate files are stored outside GitHub.

"""STEP51A: update national metrics after the WUI-Z >50% rebuild.

Frozen WUI-P/WUI-S rows are inherited byte-for-value from Step50A. Only
WUI-Z population, raster area, and county area proportions are recomputed.
"""
from __future__ import annotations

from repo_config import portable_path

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

ROOT = Path(portable_path("project"))
STEP51 = ROOT / "step51_wuiz_ketchpaw_gt50_20260731T215644Z"
PARENT = ROOT / "step50a_patch75_national49_metrics_20260730T033000Z"
BASE_PATH = ROOT / "scripts/50a_patch75_national49_metrics.py"
PROTOCOL = (
    "DENSITY_GT_6_17;INTERMIX_LOCAL_V_GT_50_PERCENT;"
    "INTERFACE_LOCAL_V_LE_50_PERCENT_AND_DISTANCE_LE_2400M;"
    "PATCH_VEGETATION_FRACTION_GT_75_PERCENT;PATCH_AREA_GE_5_KM2"
)


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    frame.to_csv(partial, index=False, float_format="%.12f")
    os.replace(partial, path)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(partial, path)


INVENTORY = pd.read_csv(
    STEP51 / "wuiz_ketchpaw_gt50_product_inventory_49.csv"
).set_index("state")
STATES = sorted(INVENTORY.index.tolist())


def configure_base():
    base = load(BASE_PATH, "step51a_base")
    base.INVENTORY = INVENTORY

    def strict_z_vector(state: str) -> Path:
        return Path(INVENTORY.loc[state, "wui_z_vector_path"])

    base.strict_z_vector = strict_z_vector
    return base


def z_area_row(state: str, base) -> pd.DataFrame:
    path = Path(INVENTORY.loc[state, "wui_z_path"])
    counts = {0: 0, 1: 0, 2: 0}
    with rasterio.open(path) as source:
        pixel_area = abs(source.res[0] * source.res[1]) / 1e6
        for _, window in source.block_windows(1):
            values = source.read(1, window=window)
            for code in counts:
                counts[code] += int(np.count_nonzero(values == code))
    return pd.DataFrame([{
        "state": state,
        "STATEFP": base.STATE[state][0],
        "state_name": base.STATE[state][1],
        "method": "WUI-Z",
        "buffer_m": 0,
        "nonwui_area_km2": counts[0] * pixel_area,
        "intermix_area_km2": counts[1] * pixel_area,
        "interface_area_km2": counts[2] * pixel_area,
        "wui_area_km2": (counts[1] + counts[2]) * pixel_area,
        "valid_area_km2": sum(counts.values()) * pixel_area,
        "classification_path": str(path),
        "classification_sha256": sha256(path),
    }])


def z_county_rows(state: str, base, metric_mod) -> pd.DataFrame:
    fips, name, _ = base.STATE[state]
    counties = gpd.read_file(base.COUNTY_GPKG, layer="tl_2022_us_county")
    counties["STATEFP"] = counties.STATEFP.astype(str).str.zfill(2)
    counties["GEOID_INT"] = counties.GEOID.astype(str).astype(int)
    ids = (
        counties[counties.STATEFP.eq(fips)]
        .sort_values("GEOID_INT").GEOID_INT.astype(int).tolist()
    )
    county_raster = base.COUNTY_CACHE / f"state_{fips}/county_id_state_{fips}.tif"
    class_path = Path(INVENTORY.loc[state, "wui_z_path"])
    count_path = base.p_count_raster(state)
    result = metric_mod.county_metrics_from_rasters(
        class_path, county_raster, count_path, ids
    )
    for column in [
        "Total_struct", "Intermix_struct", "Interface_struct", "WUI_struct", "p_s"
    ]:
        result[column] = np.nan
    result.insert(0, "buffer_m", 0)
    result.insert(0, "method", "WUI-Z")
    result.insert(0, "state_name", name)
    result.insert(0, "STATEFP", fips)
    result.insert(0, "state", state)
    result["classification_sha256"] = sha256(class_path)
    result["classification_path"] = str(class_path)
    result["county_raster_path"] = str(county_raster)
    result["structure_raster_path"] = str(count_path)
    return result


def worker(output: Path, state: str) -> None:
    checkpoint = output / "checkpoints" / f"state_{state}.json"
    if checkpoint.exists() and json.loads(checkpoint.read_text()).get("status") == "PASS":
        print(f"[STEP51A] {state} resumed PASS", flush=True)
        return
    started = time.monotonic()
    base = configure_base()
    kernel = base.configure_kernel()
    metric_mod = load(base.METRIC_PATH, f"step51a_metric_{state}")
    z_summary, z_classes = kernel.population_z(state)
    if z_summary["status"] != "PASS":
        raise RuntimeError(f"WUI-Z population closure failed: {state}")
    z_area = z_area_row(state, base)
    z_county = z_county_rows(state, base, metric_mod)

    parent_population = pd.read_csv(PARENT / "population" / f"{state}_population_3.csv")
    parent_area = pd.read_csv(PARENT / "area" / f"{state}_area_3.csv")
    parent_county = pd.read_csv(
        PARENT / "county_metrics" / f"{state}_county_metrics.csv",
        dtype={"STATEFP": str},
    )
    ps_population = parent_population[parent_population.method.isin(["WUI-P", "WUI-S"])]
    ps_area = parent_area[parent_area.method.isin(["WUI-P", "WUI-S"])]
    ps_county = parent_county[parent_county.method.isin(["WUI-P", "WUI-S"])]
    population = pd.concat([ps_population, pd.DataFrame([z_summary])], ignore_index=True, sort=False)
    area = pd.concat([ps_area, z_area], ignore_index=True, sort=False)
    county = pd.concat([ps_county, z_county], ignore_index=True, sort=False)
    atomic_csv(output / "population" / f"{state}_population_3.csv", population)
    atomic_csv(output / "population" / f"{state}_wuiz_class_population.csv", z_classes)
    atomic_csv(output / "area" / f"{state}_area_3.csv", area)
    atomic_csv(output / "county_metrics" / f"{state}_county_metrics.csv", county)
    passed = (
        len(population) == 3 and len(area) == 3
        and county[["method", "buffer_m"]].drop_duplicates().shape[0] == 3
        and county.p_a.dropna().between(0, 1).all()
        and county.loc[county.method.eq("WUI-Z"), "p_s"].isna().all()
        and abs(float(z_summary["population_closure_error"])) <= 1e-6
    )
    status = {
        "state": state, "status": "PASS" if passed else "FAIL",
        "elapsed_seconds": time.monotonic() - started,
        "completed_utc": now(), "protocol": PROTOCOL,
        "p_s_rows_inherited": True, "environment_modified": False,
    }
    atomic_json(checkpoint, status)
    if not passed:
        raise RuntimeError(f"Step51A state QC failed: {state}")
    print(f"[STEP51A] {state} PASS elapsed={status['elapsed_seconds']:.1f}s", flush=True)


def run_one(output: Path, state: str) -> dict:
    log = output / "logs" / f"{state}.log"
    environment = os.environ.copy()
    environment["PROJ_DATA"] = portable_path("software", "share/proj")
    environment["GDAL_DATA"] = portable_path("software", "share/gdal")
    with log.open("w") as stream:
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--output", str(output),
             "--worker-state", state],
            stdout=stream, stderr=subprocess.STDOUT, text=True, env=environment,
        )
    if completed.returncode:
        raise RuntimeError(f"Step51A failed {state}; see {log}")
    return {"state": state, "status": "PASS", "log": str(log)}


def aggregate(output: Path) -> None:
    checkpoints = [
        json.loads((output / "checkpoints" / f"state_{state}.json").read_text())
        for state in STATES
    ]
    population = pd.concat([
        pd.read_csv(output / "population" / f"{state}_population_3.csv")
        for state in STATES
    ], ignore_index=True)
    z_classes = pd.concat([
        pd.read_csv(output / "population" / f"{state}_wuiz_class_population.csv")
        for state in STATES
    ], ignore_index=True)
    area = pd.concat([
        pd.read_csv(output / "area" / f"{state}_area_3.csv") for state in STATES
    ], ignore_index=True)
    county = pd.concat([
        pd.read_csv(output / "county_metrics" / f"{state}_county_metrics.csv",
                    dtype={"STATEFP": str}) for state in STATES
    ], ignore_index=True)
    atomic_csv(output / "ketchpaw_gt50_national49_population_147.csv", population)
    atomic_csv(output / "population" / "ketchpaw_gt50_wuiz_class_population_147.csv", z_classes)
    atomic_csv(output / "ketchpaw_gt50_national49_area_147.csv", area)
    atomic_csv(output / "county_metrics" / "ketchpaw_gt50_national49_county_metrics.csv", county)
    summary = population.groupby("method", as_index=False)[
        ["wui_population", "total_population"]
    ].sum()
    summary["wui_population_share_pct"] = 100 * summary.wui_population / summary.total_population
    summary = summary.merge(area.groupby("method", as_index=False).wui_area_km2.sum(), on="method")
    atomic_csv(output / "ketchpaw_gt50_national49_area_population_summary.csv", summary)

    parent_population = pd.read_csv(PARENT / "patch75_national49_population_147.csv")
    parent_area = pd.read_csv(PARENT / "patch75_national49_area_147.csv")
    parent_county = pd.read_csv(
        PARENT / "county_metrics/patch75_national49_county_metrics.csv",
        dtype={"STATEFP": str},
    )
    def equivalent(left: pd.DataFrame, right: pd.DataFrame, keys: list[str]) -> bool:
        left = left.sort_values(keys).reset_index(drop=True)
        right = right.sort_values(keys).reset_index(drop=True)
        if left.shape != right.shape or list(left.columns) != list(right.columns):
            return False
        for column in left.columns:
            if pd.api.types.is_numeric_dtype(left[column]) and pd.api.types.is_numeric_dtype(right[column]):
                if not np.allclose(
                    pd.to_numeric(left[column], errors="coerce").to_numpy(float),
                    pd.to_numeric(right[column], errors="coerce").to_numpy(float),
                    rtol=0, atol=5e-10, equal_nan=True,
                ):
                    return False
            elif not left[column].fillna("").astype(str).equals(
                right[column].fillna("").astype(str)
            ):
                return False
        return True

    inheritance = {
        "population_ps_exact": equivalent(
            population[population.method.ne("WUI-Z")],
            parent_population[parent_population.method.ne("WUI-Z")],
            ["state", "method", "buffer_m"],
        ),
        "area_ps_exact": equivalent(
            area[area.method.ne("WUI-Z")],
            parent_area[parent_area.method.ne("WUI-Z")],
            ["state", "method", "buffer_m"],
        ),
        "county_ps_exact": equivalent(
            county[county.method.ne("WUI-Z")],
            parent_county[parent_county.method.ne("WUI-Z")],
            ["state", "method", "buffer_m", "GEOID_INT"],
        ),
    }
    qc = {
        "state_units": len(checkpoints),
        "state_pass": sum(row["status"] == "PASS" for row in checkpoints),
        "population_rows": len(population), "area_rows": len(area),
        "county_rows": len(county), "county_unique_fips": int(county.GEOID_INT.nunique()),
        "population_closure_max_abs": float(population.population_closure_error.abs().max()),
        **inheritance,
    }
    passed = (
        qc["state_units"] == qc["state_pass"] == 49
        and qc["population_rows"] == qc["area_rows"] == 147
        and qc["county_rows"] == 9327 and qc["county_unique_fips"] == 3109
        and qc["population_closure_max_abs"] <= 1e-5
        and all(inheritance.values())
    )
    status = {
        "step": "STEP51A_WUIZ_GT50_DOWNSTREAM_METRICS",
        "status": "WUIZ_GT50_METRICS_COMPLETE_READY_FOR_PAIRWISE_MORAN" if passed else "QC_REVIEW_REQUIRED",
        "parent_metrics_run": str(PARENT), "parent_wuiz_run": str(STEP51),
        "protocol": PROTOCOL, "environment_modified": False,
        "completed_utc": now(), "qc": qc,
    }
    atomic_json(output / "step51a_status.json", status)
    (output / "STEP51A_FINAL_QC.txt").write_text(
        "\n".join(["STEP51A WUI-Z GT50 DOWNSTREAM METRICS QC"]
                  + [f"{key}={value}" for key, value in qc.items()]
                  + [f"final_status={status['status']}"]) + "\n"
    )
    if not passed:
        raise RuntimeError("Step51A aggregate QC failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--worker-state", choices=STATES, default="")
    parser.add_argument("--canary-state", choices=STATES, default="")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for folder in ["population", "area", "county_metrics", "checkpoints", "logs"]:
        (output / folder).mkdir(exist_ok=True)
    if args.worker_state:
        worker(output, args.worker_state)
        return
    states = [args.canary_state] if args.canary_state else STATES
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, output, state): state for state in states}
        completed = []
        for index, future in enumerate(as_completed(futures), 1):
            completed.append(future.result())
            elapsed = time.monotonic() - started
            eta = elapsed * (len(states)-index) / index
            print(f"[STEP51A] completed={index}/{len(states)} percent={100*index/len(states):.1f} "
                  f"elapsed={elapsed/60:.1f}m ETA={eta/60:.1f}m", flush=True)
    atomic_csv(output / "step51a_state_status.csv", pd.DataFrame(completed))
    if args.canary_state:
        atomic_json(output / "step51a_canary_status.json", {
            "state": args.canary_state, "status": "PASS", "completed_utc": now()
        })
        return
    aggregate(output)


if __name__ == "__main__":
    main()
