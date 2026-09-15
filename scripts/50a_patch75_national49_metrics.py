#!/usr/bin/env python3
# Copyright (C) 2026 WUI Mapping Project Authors
# SPDX-License-Identifier: GPL-3.0-only
# Repository version: filesystem paths are loaded through repo_config.py.
# Copy config/paths.example.json to config/paths.json before a new run.
# Raw input data and large intermediate files are stored outside GitHub.

"""STEP50A: national strict-patch 500 m population and county metrics."""
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
STEP49 = ROOT / "step49_patch75_remaining44_500m_20260729T231000Z"
NATIONAL49 = ROOT / "step49_patch75_national49_500m_20260730T032000Z"
STEP47_FOUR = ROOT / "step47d_patch75_four_state_psz_20260729T150314Z"
STEP47_CO = ROOT / "step47_patch75_silvis_colorado_canary_20260729T034216Z"
STEP47_COZ = ROOT / "step47c_patch75_colorado_wuiz_20260729T144953Z"
STEP44 = ROOT / "step44_wuip_p2_downstream_metrics_20260727T224419Z"
STEP43 = ROOT / "step43_wuip_p2_formal_rebuild_20260727T180822Z"
STEP35 = ROOT / "step35_five_state_all_buffer_population_20260724T183418Z"
STEP33 = ROOT / "step33_remaining_41_state_ps_recompute_20260724T025443Z"
STEP29 = ROOT / "step29_five_state_exact_target_qc_20260723T181854Z"
COUNTY_CACHE = Path(portable_path("legacy", "WUI_tables_compare/_cache_table5_all49_500m"))
DRIVE = Path(portable_path("data"))
COUNTY_GPKG = (
    DRIVE / "MBF+NLCD_2022US/Inputs/Processed_GPKG/tl_2022_us_county.gpkg"
)
KERNEL_PATH = ROOT / "scripts/48a_patch75_five_state_downstream_metrics.py"
HELPER_PATH = ROOT / "scripts/08_recompute_sample4_ps_population.py"
EXACT_PATH = ROOT / "scripts/25_audit_vermont_wuis_exact_point_in_polygon.py"
METRIC_PATH = ROOT / "scripts/44_wuip_p2_downstream_metrics.py"
OLD35_PATH = ROOT / "scripts/35_recompute_five_state_all_buffer_population.py"
FIVE = {"CA", "CO", "FL", "PA", "TX"}
STEP29_STATES = {"AL", "OK", "VT"}
TOKENS = {
    "AL":"Alabama","AZ":"Arizona","AR":"Arkansas","CA":"California",
    "CO":"Colorado","CT":"Connecticut","DE":"Delaware",
    "DC":"DistrictofColumbia","FL":"Florida","GA":"Georgia","ID":"Idaho",
    "IL":"Illinois","IN":"Indiana","IA":"Iowa","KS":"Kansas",
    "KY":"Kentucky","LA":"Louisiana","ME":"Maine","MD":"Maryland",
    "MA":"Massachusetts","MI":"Michigan","MN":"Minnesota",
    "MS":"Mississippi","MO":"Missouri","MT":"Montana","NE":"Nebraska",
    "NV":"Nevada","NH":"NewHampshire","NJ":"NewJersey",
    "NM":"NewMexico","NY":"NewYork","NC":"NorthCarolina",
    "ND":"NorthDakota","OH":"Ohio","OK":"Oklahoma","OR":"Oregon",
    "PA":"Pennsylvania","RI":"RhodeIsland","SC":"SouthCarolina",
    "SD":"SouthDakota","TN":"Tennessee","TX":"Texas","UT":"Utah",
    "VT":"Vermont","VA":"Virginia","WA":"Washington",
    "WV":"WestVirginia","WI":"Wisconsin","WY":"Wyoming",
}


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
    tmp = path.with_name(path.name + ".partial")
    frame.to_csv(tmp, index=False, float_format="%.12f")
    os.replace(tmp, path)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(tmp, path)


def state_table() -> tuple[list[str], dict[str, tuple[str, str, int]]]:
    national = pd.read_csv(
        STEP44 / "p2_national_500m_all_methods_147.csv",
        dtype={"STATEFP": str},
    )
    p = national[national.method.eq("WUI-P")].copy()
    p["STATEFP"] = p.STATEFP.astype(str).str.zfill(2)
    p = p.sort_values("STATEFP")
    states = p.state.tolist()
    mapping = {
        row.state: (
            str(row.STATEFP).zfill(2),
            str(row.state_name),
            int(round(float(row.total_population))),
        )
        for row in p.itertuples(index=False)
    }
    if len(states) != 49 or set(states) != set(TOKENS):
        raise RuntimeError("49-state metadata gate failed")
    return states, mapping


STATES, STATE = state_table()
INVENTORY = pd.read_csv(
    NATIONAL49 / "patch75_national49_500m_product_inventory.csv"
).set_index("state")


def strict_raster(state: str, method: str, radius: int) -> Path:
    if radius != 500:
        raise ValueError("Step50A is fixed to 500 m")
    return Path(INVENTORY.loc[state, "wui_p_path" if method == "WUI-P" else "wui_s_path"])


def strict_z_raster(state: str) -> Path:
    return Path(INVENTORY.loc[state, "wui_z_path"])


def state_base(state: str) -> Path:
    if state == "CO":
        return STEP47_COZ
    if state in FIVE:
        return STEP47_FOUR / state
    return STEP49 / "states" / state


def strict_z_vector(state: str) -> Path:
    candidates = sorted(state_base(state).glob("WUI_Z_Paper_*_SILVIS_GT75.gpkg"))
    if len(candidates) != 1:
        raise RuntimeError(f"WUI-Z vector resolution failed {state}: {candidates}")
    return candidates[0]


def baseline_raster(state: str, method: str, radius: int) -> Path:
    if radius != 500:
        raise ValueError("Step50A is fixed to 500 m")
    if method == "WUI-P":
        return STEP43 / f"rasters_500m/{state}/WUI_P_P2_{state}_r0500m.tif"
    token = TOKENS[state]
    return DRIVE / f"WUI_S_Paper/{token}/WUI_S_{token}_r0500m.tif"


def read_csv_flexible(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype={"GEOID12": str})
    except Exception:
        return pd.read_csv(path, compression=None, dtype={"GEOID12": str})


def baseline_detail(state: str, method: str) -> pd.DataFrame:
    fips = STATE[state][0]
    if method == "WUI-P":
        path = STEP44 / f"population/{state}_block_group_detail.csv.gz"
    elif state in FIVE:
        path = (
            STEP35 / f"job_{fips}_{state.lower()}_wuis/"
            "exact_block_group_detail_all_buffers.csv.gz"
        )
    elif state in STEP29_STATES:
        path = (
            STEP29 / f"job_{fips}_{state.lower()}_wuis/"
            "exact_block_group_detail.csv.gz"
        )
    else:
        path = (
            STEP33 / f"job_{fips}_{state.lower()}_wuis/"
            "exact_block_group_detail.csv.gz"
        )
    frame = read_csv_flexible(path)
    frame["GEOID12"] = frame.GEOID12.astype(str).str.zfill(12)
    if "buffer_m" not in frame:
        frame["buffer_m"] = 500
    return frame[frame.buffer_m.eq(500)].copy()


def structure_source(state: str, method: str) -> Path:
    if method == "WUI-P":
        checkpoint = json.loads(
            (STEP44 / f"checkpoints/population_{state}.json").read_text()
        )
        return Path(checkpoint["address_path"])
    return (
        DRIVE / "mbf_work/centroids_5070"
        / f"MBF_{TOKENS[state]}_centroids_5070.gpkg"
    )


def p_count_raster(state: str) -> Path:
    fips = STATE[state][0]
    if state == "TX":
        return Path(
            portable_path("legacy", "WUI_TX_recovery/step18_archive_handoff_runs/texas_archive_handoff_20260722T213425Z/downstream_workspace/wui_p_rasters_pending_step19/step19D_texas_wui_p_candidate_20260723T153842Z/intermediate/Texas_candidate_point_count.tif")
        )
    return (
        COUNTY_CACHE / f"state_{fips}/WUI-P/"
        f"struct_count_state_{fips}_WUI-P.tif"
    )


def s_count_raster(state: str) -> Path:
    if state == "CO":
        return STEP47_CO / "WUI_S_CO_centroid_count_30m.tif"
    return state_base(state) / f"WUI_S_{state}_centroid_count_30m.tif"


def configure_kernel():
    kernel = load(KERNEL_PATH, "step50a_kernel")
    kernel.RADII = (500,)
    kernel.METHODS = ("WUI-P", "WUI-S")
    kernel.STATE = STATE
    kernel.strict_raster = strict_raster
    kernel.strict_z_raster = strict_z_raster
    kernel.strict_z_vector = strict_z_vector
    kernel.baseline_raster = baseline_raster
    kernel.baseline_detail = baseline_detail
    kernel.structure_source = structure_source
    kernel.p_count_raster = p_count_raster
    kernel.s_count_raster = s_count_raster
    return kernel


def raster_area_rows(state: str) -> pd.DataFrame:
    rows = []
    for method, path in [
        ("WUI-P", strict_raster(state, "WUI-P", 500)),
        ("WUI-S", strict_raster(state, "WUI-S", 500)),
        ("WUI-Z", strict_z_raster(state)),
    ]:
        counts = {0: 0, 1: 0, 2: 0}
        with rasterio.open(path) as src:
            pixel_area = abs(src.res[0] * src.res[1]) / 1e6
            for _, window in src.block_windows(1):
                data = src.read(1, window=window)
                for value in counts:
                    counts[value] += int(np.count_nonzero(data == value))
        rows.append({
            "state": state,
            "STATEFP": STATE[state][0],
            "state_name": STATE[state][1],
            "method": method,
            "buffer_m": 0 if method == "WUI-Z" else 500,
            "nonwui_area_km2": counts[0] * pixel_area,
            "intermix_area_km2": counts[1] * pixel_area,
            "interface_area_km2": counts[2] * pixel_area,
            "wui_area_km2": (counts[1] + counts[2]) * pixel_area,
            "valid_area_km2": sum(counts.values()) * pixel_area,
            "classification_path": str(path),
            "classification_sha256": sha256(path),
        })
    return pd.DataFrame(rows)


def county_rows(state: str, metric_mod) -> pd.DataFrame:
    fips, name, _ = STATE[state]
    counties = gpd.read_file(COUNTY_GPKG, layer="tl_2022_us_county")
    counties["STATEFP"] = counties.STATEFP.astype(str).str.zfill(2)
    counties["GEOID_INT"] = counties.GEOID.astype(str).astype(int)
    ids = (
        counties[counties.STATEFP.eq(fips)]
        .sort_values("GEOID_INT").GEOID_INT.astype(int).tolist()
    )
    county_raster = COUNTY_CACHE / f"state_{fips}/county_id_state_{fips}.tif"
    frames = []
    for method, path, count_path in [
        ("WUI-P", strict_raster(state, "WUI-P", 500), p_count_raster(state)),
        ("WUI-S", strict_raster(state, "WUI-S", 500), s_count_raster(state)),
        ("WUI-Z", strict_z_raster(state), p_count_raster(state)),
    ]:
        result = metric_mod.county_metrics_from_rasters(
            path, county_raster, count_path, ids
        )
        if method == "WUI-Z":
            for col in [
                "Total_struct", "Intermix_struct", "Interface_struct",
                "WUI_struct", "p_s",
            ]:
                result[col] = np.nan
        result.insert(0, "buffer_m", 0 if method == "WUI-Z" else 500)
        result.insert(0, "method", method)
        result.insert(0, "state_name", name)
        result.insert(0, "STATEFP", fips)
        result.insert(0, "state", state)
        result["classification_sha256"] = sha256(path)
        result["classification_path"] = str(path)
        result["county_raster_path"] = str(county_raster)
        result["structure_raster_path"] = str(count_path)
        frames.append(result)
    return pd.concat(frames, ignore_index=True)


def worker(output: Path, state: str) -> None:
    checkpoint = output / "checkpoints" / f"state_{state}.json"
    if checkpoint.exists():
        saved = json.loads(checkpoint.read_text())
        if saved.get("status") == "PASS":
            print(f"[STEP50A WORKER] {state} resumed PASS", flush=True)
            return
    kernel = configure_kernel()
    helper = load(HELPER_PATH, f"step50_helper_{state}")
    exact = load(EXACT_PATH, f"step50_exact_{state}")
    metric_mod = load(METRIC_PATH, f"step50_metric_{state}")
    old35 = load(OLD35_PATH, f"step50_old35_{state}")
    started = time.monotonic()
    pop_rows = []
    pop_rows.extend(
        kernel.population_ps(output, state, "WUI-P", helper, exact, old35)
    )
    pop_rows.extend(
        kernel.population_full(output, state, "WUI-S", helper, exact, old35)
    )
    z_summary, z_classes = kernel.population_z(state)
    if z_summary["status"] != "PASS":
        raise RuntimeError(f"WUI-Z population closure failed: {state}")
    pop_rows.append(z_summary)
    atomic_csv(output / "population" / f"{state}_population_3.csv", pd.DataFrame(pop_rows))
    atomic_csv(
        output / "population" / f"{state}_wuiz_class_population.csv",
        z_classes,
    )
    area = raster_area_rows(state)
    county = county_rows(state, metric_mod)
    atomic_csv(output / "area" / f"{state}_area_3.csv", area)
    atomic_csv(output / "county_metrics" / f"{state}_county_metrics.csv", county)
    passed = (
        len(pop_rows) == 3
        and all(row["status"] == "PASS" for row in pop_rows)
        and len(area) == 3
        and county[["method", "buffer_m"]].drop_duplicates().shape[0] == 3
        and county.p_a.dropna().between(0, 1).all()
        and county.p_s.dropna().between(0, 1).all()
        and county.loc[county.method.eq("WUI-Z"), "p_s"].isna().all()
    )
    atomic_json(checkpoint, {
        "state": state,
        "status": "PASS" if passed else "FAIL",
        "completed_utc": now(),
        "elapsed_seconds": time.monotonic() - started,
        "population_rows": len(pop_rows),
        "area_rows": len(area),
        "county_rows": len(county),
    })
    if not passed:
        raise RuntimeError(f"Step50A state QC failed: {state}")
    print(
        f"[STEP50A WORKER] {state} PASS elapsed="
        f"{(time.monotonic()-started)/60:.1f}m",
        flush=True,
    )


def run_one(output: Path, state: str) -> dict:
    log = output / "logs" / f"{state}.log"
    env = os.environ.copy()
    env["PROJ_DATA"] = portable_path("software", "share/proj")
    env["GDAL_DATA"] = portable_path("software", "share/gdal")
    with log.open("w") as stream:
        cp = subprocess.run(
            [
                sys.executable, str(Path(__file__).resolve()),
                "--output", str(output), "--worker-state", state,
            ],
            stdout=stream, stderr=subprocess.STDOUT, env=env, text=True,
        )
    if cp.returncode:
        raise RuntimeError(f"{state} failed; see {log}")
    return {"state": state, "status": "PASS", "log": str(log)}


def aggregate(output: Path) -> None:
    checkpoints = [
        json.loads((output / "checkpoints" / f"state_{state}.json").read_text())
        for state in STATES
    ]
    if not all(row["status"] == "PASS" for row in checkpoints):
        raise RuntimeError("Not all state checkpoints passed")
    population = pd.concat(
        [
            pd.read_csv(output / "population" / f"{state}_population_3.csv")
            for state in STATES
        ],
        ignore_index=True,
    )
    z_class = pd.concat(
        [
            pd.read_csv(
                output / "population" / f"{state}_wuiz_class_population.csv"
            )
            for state in STATES
        ],
        ignore_index=True,
    )
    area = pd.concat(
        [
            pd.read_csv(output / "area" / f"{state}_area_3.csv")
            for state in STATES
        ],
        ignore_index=True,
    )
    county = pd.concat(
        [
            pd.read_csv(
                output / "county_metrics" / f"{state}_county_metrics.csv",
                dtype={"STATEFP": str},
            )
            for state in STATES
        ],
        ignore_index=True,
    )
    atomic_csv(output / "patch75_national49_population_147.csv", population)
    atomic_csv(output / "population" / "patch75_wuiz_class_population_147.csv", z_class)
    atomic_csv(output / "patch75_national49_area_147.csv", area)
    atomic_csv(output / "county_metrics" / "patch75_national49_county_metrics.csv", county)
    summary = (
        population.groupby("method", as_index=False)
        [["wui_population", "total_population"]].sum()
    )
    summary["wui_population_share_pct"] = (
        100 * summary.wui_population / summary.total_population
    )
    summary = summary.merge(
        area.groupby("method", as_index=False).wui_area_km2.sum(),
        on="method",
    )
    atomic_csv(output / "patch75_national49_area_population_summary.csv", summary)
    qc = {
        "state_checkpoints": len(checkpoints),
        "state_checkpoint_failures": sum(row["status"] != "PASS" for row in checkpoints),
        "population_rows": len(population),
        "population_state_units": int(population.state.nunique()),
        "population_method_rows": population.groupby("method").size().to_dict(),
        "population_closure_max_abs": float(
            population.population_closure_error.abs().max()
        ),
        "area_rows": len(area),
        "county_rows": len(county),
        "county_unique_fips": int(county.GEOID_INT.nunique()),
        "county_combinations": int(
            county[["state", "method", "buffer_m"]].drop_duplicates().shape[0]
        ),
        "county_pa_in_range": bool(county.p_a.dropna().between(0, 1).all()),
        "county_ps_in_range": bool(county.p_s.dropna().between(0, 1).all()),
        "wuiz_ps_all_na": bool(
            county.loc[county.method.eq("WUI-Z"), "p_s"].isna().all()
        ),
    }
    passed = (
        qc["state_checkpoints"] == 49
        and qc["state_checkpoint_failures"] == 0
        and qc["population_rows"] == 147
        and qc["population_state_units"] == 49
        and qc["area_rows"] == 147
        and qc["county_rows"] == 9327
        and qc["county_unique_fips"] == 3109
        and qc["county_combinations"] == 147
        and qc["county_pa_in_range"]
        and qc["county_ps_in_range"]
        and qc["wuiz_ps_all_na"]
        and qc["population_closure_max_abs"] <= 1e-5
    )
    status = {
        "step": "STEP50A_PATCH75_NATIONAL49_METRICS",
        "status": (
            "PATCH75_NATIONAL49_METRICS_COMPLETE_READY_FOR_MORAN"
            if passed else "QC_REVIEW_REQUIRED"
        ),
        "completed_utc": now(),
        "buffer_m": 500,
        "upstream_modified": False,
        "environment_modified": False,
        "qc": qc,
    }
    atomic_json(output / "step50a_status.json", status)
    (output / "STEP50A_FINAL_QC.txt").write_text(
        "\n".join(
            ["STEP50A PATCH75 NATIONAL49 METRICS QC"]
            + [f"{key}={value}" for key, value in qc.items()]
            + [f"final_status={status['status']}"]
        ) + "\n"
    )
    (output / "README.md").write_text(
        "# Step50A national strict-patch downstream metrics\n\n"
        "Population, area, and county metrics for strict qualifying-patch "
        "WUI-P, WUI-S, and fixed WUI-Z across 48 states plus DC at the "
        "standardized 500 m P/S setting. WUI-Z population is joined from "
        "2020 Census block POP20; WUI-Z p_s remains explicitly undefined.\n"
    )
    files = sorted(
        p for p in output.rglob("*")
        if p.is_file() and p.name != "sha256_manifest.txt"
    )
    (output / "sha256_manifest.txt").write_text(
        "\n".join(f"{sha256(path)}  {path.relative_to(output)}" for path in files)
        + "\n"
    )
    print(json.dumps(status), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--worker-state", choices=STATES, default="")
    parser.add_argument("--canary-state", choices=STATES, default="")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for sub in ["population", "area", "county_metrics", "checkpoints", "logs"]:
        (output / sub).mkdir(exist_ok=True)
    if args.worker_state:
        worker(output, args.worker_state)
        return
    chosen = [args.canary_state] if args.canary_state else STATES
    started = time.monotonic()
    completed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, output, state): state for state in chosen}
        for index, future in enumerate(as_completed(futures), 1):
            state = futures[future]
            completed.append(future.result())
            elapsed = time.monotonic() - started
            eta = elapsed * (len(chosen) - index) / index
            print(
                f"[STEP50A] {state} {index}/{len(chosen)} "
                f"{100*index/len(chosen):.1f}% elapsed={elapsed/60:.1f}m "
                f"ETA={eta/60:.1f}m",
                flush=True,
            )
    atomic_csv(output / "step50a_state_status.csv", pd.DataFrame(completed))
    if args.canary_state:
        atomic_json(output / "step50a_canary_status.json", {
            "status": "CANARY_COMPLETE",
            "state": args.canary_state,
            "completed_utc": now(),
        })
        return
    aggregate(output)


if __name__ == "__main__":
    main()
