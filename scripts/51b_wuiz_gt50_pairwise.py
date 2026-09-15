#!/usr/bin/env python3
# Copyright (C) 2026 WUI Mapping Project Authors
# SPDX-License-Identifier: GPL-3.0-only
# Repository version: filesystem paths are loaded through repo_config.py.
# Copy config/paths.example.json to config/paths.json before a new run.
# Raw input data and large intermediate files are stored outside GitHub.

"""STEP51B: recompute all pairwise comparisons affected by WUI-Z >50%."""
from __future__ import annotations

from repo_config import portable_path

import argparse
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

ROOT = Path(portable_path("project"))
STEP51 = ROOT / "step51_wuiz_ketchpaw_gt50_20260731T215644Z"
STEP49 = ROOT / "step49_patch75_national49_500m_20260730T032000Z"
STEP48 = ROOT / "step48_patch75_five_state_downstream_20260729T202500Z"
OLD_TABLES = STEP48 / "step48b_moran_figures_tables/tables"
FIVE = {"CA", "CO", "FL", "PA", "TX"}
PIXEL_AREA_KM2 = 0.0009
PROTOCOL = "KETCHPAW_INTERMIX_GT50_INTERFACE_LE50"

INVENTORY = pd.read_csv(
    STEP51 / "wuiz_ketchpaw_gt50_product_inventory_49.csv"
).set_index("state")
STATES = sorted(INVENTORY.index.tolist())
OLD_NATIONAL = pd.read_csv(STEP49 / "patch75_national49_500m_pairwise.csv")
OLD_A1 = pd.read_csv(OLD_TABLES / "Appendix_A1_patch75_area_candidate.csv")
OLD_A5 = pd.read_csv(OLD_TABLES / "Appendix_A5_patch75_intersection_candidate.csv")
OLD_A6 = pd.read_csv(OLD_TABLES / "Appendix_A6_patch75_jaccard_candidate.csv")
OLD_FIVE = OLD_A5.merge(
    OLD_A6, on=["state", "radius_m", "method_pair", "intersection_area_km2"]
    if "intersection_area_km2" in OLD_A6.columns else
    ["state", "radius_m", "method_pair"], how="inner"
)


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    frame.to_csv(partial, index=False, float_format="%.12f")
    os.replace(partial, path)


def atomic_json(path: Path, value: dict) -> None:
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(partial, path)


def windows(width: int, height: int, tile: int = 2048):
    for row in range(0, height, tile):
        for col in range(0, width, tile):
            yield Window(col, row, min(tile, width-col), min(tile, height-row))


def compare(method_path: Path, z_path: Path, valid_path: Path) -> dict:
    common = intersection = union = method_wui = z_wui = 0
    with rasterio.open(method_path) as method, rasterio.open(z_path) as z, rasterio.open(valid_path) as valid:
        if not (
            method.width == z.width == valid.width
            and method.height == z.height == valid.height
            and method.transform == z.transform == valid.transform
        ):
            raise RuntimeError(f"Pairwise grid mismatch: {method_path}")
        for window in windows(method.width, method.height):
            a = method.read(1, window=window)
            b = z.read(1, window=window)
            domain = np.isin(a, [0, 1, 2]) & (valid.read(1, window=window) == 1)
            aw = np.isin(a, [1, 2]) & domain
            bw = np.isin(b, [1, 2]) & domain
            common += int(domain.sum())
            intersection += int(np.count_nonzero(aw & bw))
            union += int(np.count_nonzero(aw | bw))
            method_wui += int(aw.sum())
            z_wui += int(bw.sum())
    return {
        "common_valid_pixels": common,
        "intersection_pixels": intersection,
        "union_pixels": union,
        "jaccard": intersection / union if union else math.nan,
        "intersection_area_km2": intersection * PIXEL_AREA_KM2,
        "union_area_km2": union * PIXEL_AREA_KM2,
        "method_wui_pixels": method_wui,
        "z_wui_pixels": z_wui,
    }


def sensitivity_path(state: str, method: str, radius: int) -> Path:
    hit = OLD_A1[
        OLD_A1.state.eq(state) & OLD_A1.method.eq(method) & OLD_A1.buffer_m.eq(radius)
    ]
    if len(hit) != 1:
        raise RuntimeError(f"Sensitivity raster resolution failed {state}/{method}/{radius}")
    return Path(hit.iloc[0].classification_path)


def worker(output: Path, state: str) -> None:
    checkpoint = output / "states" / f"{state}.json"
    if checkpoint.exists() and json.loads(checkpoint.read_text()).get("status") == "PASS":
        print(f"[STEP51B] {state} resumed PASS", flush=True)
        return
    started = time.monotonic()
    z_path = Path(INVENTORY.loc[state, "wui_z_path"])
    valid_path = Path(INVENTORY.loc[state, "wui_z_valid_path"])
    rows = []
    # National standardized 500-m panel: inherit P/S and recompute P/Z and S/Z.
    ps = OLD_NATIONAL[
        OLD_NATIONAL.state.eq(state) & OLD_NATIONAL.method_pair.eq("WUI-P/WUI-S")
    ]
    if len(ps) != 1:
        raise RuntimeError(f"Missing national P/S row {state}")
    inherited = ps.iloc[0].to_dict()
    inherited["source_policy"] = "INHERITED_UNCHANGED_P_S"
    rows.append(inherited)
    for method, column in [("WUI-P", "wui_p_path"), ("WUI-S", "wui_s_path")]:
        rows.append({
            "state": state, "radius_m": 500, "method_pair": f"{method}/WUI-Z",
            "z_scenario": "KETCHPAW_GT50", "source_policy": "RECOMPUTED_NEW_Z",
            **compare(Path(INVENTORY.loc[state, column]), z_path, valid_path),
        })
    atomic_csv(output / "states" / f"{state}_national.csv", pd.DataFrame(rows))

    sensitivity_rows = []
    if state in FIVE:
        for radius in range(100, 1001, 100):
            old_ps = OLD_FIVE[
                OLD_FIVE.state.eq(state) & OLD_FIVE.radius_m.eq(radius)
                & OLD_FIVE.method_pair.eq("WUI-P/WUI-S")
            ]
            if len(old_ps) != 1:
                raise RuntimeError(f"Missing sensitivity P/S row {state}/{radius}")
            inherited_ps = old_ps.iloc[0].to_dict()
            inherited_ps["source_policy"] = "INHERITED_UNCHANGED_P_S"
            sensitivity_rows.append(inherited_ps)
            for method in ["WUI-P", "WUI-S"]:
                sensitivity_rows.append({
                    "state": state, "radius_m": radius,
                    "method_pair": f"{method}/WUI-Z",
                    "source_policy": "RECOMPUTED_NEW_Z",
                    **compare(sensitivity_path(state, method, radius), z_path, valid_path),
                })
        atomic_csv(
            output / "states" / f"{state}_sensitivity.csv",
            pd.DataFrame(sensitivity_rows),
        )
    status = {
        "state": state, "status": "PASS", "national_rows": len(rows),
        "sensitivity_rows": len(sensitivity_rows),
        "elapsed_seconds": time.monotonic() - started,
        "completed_utc": now(), "protocol": PROTOCOL,
    }
    atomic_json(checkpoint, status)
    print(f"[STEP51B] {state} PASS elapsed={status['elapsed_seconds']:.1f}s", flush=True)


def run_one(output: Path, state: str) -> dict:
    log = output / "logs" / f"{state}.log"
    with log.open("w") as stream:
        cp = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--output", str(output),
             "--worker-state", state], stdout=stream, stderr=subprocess.STDOUT, text=True,
        )
    if cp.returncode:
        raise RuntimeError(f"Step51B failed {state}; see {log}")
    return {"state": state, "status": "PASS", "log": str(log)}


def aggregate(output: Path) -> None:
    statuses = [json.loads((output / "states" / f"{s}.json").read_text()) for s in STATES]
    national = pd.concat([
        pd.read_csv(output / "states" / f"{s}_national.csv") for s in STATES
    ], ignore_index=True, sort=False)
    sensitivity = pd.concat([
        pd.read_csv(output / "states" / f"{s}_sensitivity.csv") for s in sorted(FIVE)
    ], ignore_index=True, sort=False)
    national = national.sort_values(["state", "method_pair"]).reset_index(drop=True)
    sensitivity = sensitivity.sort_values(["state", "radius_m", "method_pair"]).reset_index(drop=True)
    atomic_csv(output / "ketchpaw_gt50_national49_500m_pairwise_147.csv", national)
    atomic_csv(output / "ketchpaw_gt50_five_state_sensitivity_pairwise_150.csv", sensitivity)
    micro = national.groupby("method_pair", as_index=False)[
        ["common_valid_pixels", "intersection_pixels", "union_pixels"]
    ].sum()
    micro["micro_jaccard"] = micro.intersection_pixels / micro.union_pixels
    micro["intersection_area_km2"] = micro.intersection_pixels * PIXEL_AREA_KM2
    micro["union_area_km2"] = micro.union_pixels * PIXEL_AREA_KM2
    atomic_csv(output / "ketchpaw_gt50_national_micro_jaccard.csv", micro)
    inherited_national = national[national.source_policy.eq("INHERITED_UNCHANGED_P_S")]
    inherited_five = sensitivity[sensitivity.source_policy.eq("INHERITED_UNCHANGED_P_S")]
    passed = (
        len(statuses) == sum(s["status"] == "PASS" for s in statuses) == 49
        and len(national) == 147 and len(sensitivity) == 150
        and len(inherited_national) == 49 and len(inherited_five) == 50
        and national.jaccard.dropna().between(0, 1).all()
        and sensitivity.jaccard.dropna().between(0, 1).all()
    )
    status = {
        "step": "STEP51B_WUIZ_GT50_PAIRWISE",
        "status": "WUIZ_GT50_PAIRWISE_COMPLETE_READY_FOR_MORAN_APPENDIX" if passed else "QC_REVIEW_REQUIRED",
        "state_units": len(statuses), "national_rows": len(national),
        "sensitivity_rows": len(sensitivity),
        "national_ps_rows_inherited": len(inherited_national),
        "sensitivity_ps_rows_inherited": len(inherited_five),
        "protocol": PROTOCOL, "environment_modified": False,
        "completed_utc": now(), "micro_jaccard": micro.to_dict("records"),
    }
    atomic_json(output / "step51b_status.json", status)
    if not passed:
        raise RuntimeError("Step51B aggregate QC failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--worker-state", choices=STATES, default="")
    parser.add_argument("--canary-state", choices=STATES, default="")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for folder in ["states", "logs"]:
        (output / folder).mkdir(exist_ok=True)
    if args.worker_state:
        worker(output, args.worker_state)
        return
    states = [args.canary_state] if args.canary_state else STATES
    started = time.monotonic()
    completed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, output, state): state for state in states}
        for index, future in enumerate(as_completed(futures), 1):
            completed.append(future.result())
            elapsed = time.monotonic() - started
            eta = elapsed * (len(states)-index) / index
            print(f"[STEP51B] completed={index}/{len(states)} percent={100*index/len(states):.1f} "
                  f"elapsed={elapsed/60:.1f}m ETA={eta/60:.1f}m", flush=True)
    atomic_csv(output / "step51b_state_status.csv", pd.DataFrame(completed))
    if args.canary_state:
        atomic_json(output / "step51b_canary_status.json", {
            "state": args.canary_state, "status": "PASS", "completed_utc": now()
        })
        return
    aggregate(output)


if __name__ == "__main__":
    main()
