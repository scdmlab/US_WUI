#!/usr/bin/env python3
# Copyright (C) 2026 WUI Mapping Project Authors
# SPDX-License-Identifier: GPL-3.0-only
# Repository version: filesystem paths are loaded through repo_config.py.
# Copy config/paths.example.json to config/paths.json before a new run.
# Raw input data and large intermediate files are stored outside GitHub.

"""Generate Colorado >75% patch canary rasters for the nine non-500 m radii.

Diagnostic only: no Step43--46 file is modified.
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

import numpy as np
import rasterio
from skimage.morphology import disk


ROOT = Path(portable_path("project"))
PARENT = (
    ROOT / "step47_patch75_silvis_colorado_canary_20260729T034216Z"
)
BASE_SCRIPT = ROOT / "scripts/47_patch75_silvis_colorado_canary.py"
STEP43 = (
    ROOT / "step43_wuip_p2_formal_rebuild_20260727T180822Z"
)
WUIS_ROOT = Path(
    portable_path("data", "WUI_S_Paper/Colorado")
)
RADII = [100, 200, 300, 400, 600, 700, 800, 900, 1000]
NODATA = 255


def load_base():
    spec = importlib.util.spec_from_file_location("step47_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def now_utc() -> str:
    import datetime as dt

    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def official_p(radius: int) -> Path:
    return (
        STEP43
        / "rasters_sensitivity/CO"
        / f"WUI_P_P2_CO_r{radius:04d}m.tif"
    )


def official_s(radius: int) -> Path:
    return WUIS_ROOT / f"WUI_S_Colorado_r{radius:04d}m.tif"


def pairwise(path_p: Path, path_s: Path) -> dict:
    valid = intersection = union = 0
    with rasterio.open(path_p) as p, rasterio.open(path_s) as s:
        if (
            p.width != s.width
            or p.height != s.height
            or p.transform != s.transform
        ):
            raise RuntimeError("P/S output grid mismatch")
        for _, window in p.block_windows(1):
            a = p.read(1, window=window)
            b = s.read(1, window=window)
            domain = (a != NODATA) & (b != NODATA)
            aw = (a > 0) & (a < NODATA)
            bw = (b > 0) & (b < NODATA)
            valid += int(domain.sum())
            intersection += int((aw & bw & domain).sum())
            union += int(((aw | bw) & domain).sum())
    return {
        "valid_pixels": valid,
        "intersection_pixels": intersection,
        "union_pixels": union,
        "jaccard": intersection / union if union else 1.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "rasters/WUI-P").mkdir(parents=True)
    (output / "rasters/WUI-S").mkdir(parents=True)
    started = time.perf_counter()
    created = now_utc()

    base = load_base()
    required = [
        BASE_SCRIPT,
        PARENT / "CO_silvis_block_gt75_buffer_2400m.tif",
        PARENT / "WUI_S_CO_centroid_count_30m.tif",
        STEP43 / "intermediate/CO_p2_sparse_cells.npz",
        base.WILD,
        base.CURRENT_DISTANCE,
    ]
    required.extend(official_p(r) for r in RADII)
    required.extend(official_s(r) for r in RADII)
    missing = [str(path) for path in required if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing inputs: {missing}")

    parent_hashes = {
        str(path): sha256(Path(path))
        for path in required
        if Path(path).is_file()
    }
    with rasterio.open(base.WUIP) as ref, rasterio.open(
        PARENT / "CO_silvis_block_gt75_buffer_2400m.tif"
    ) as patch:
        if (
            patch.width != ref.width
            or patch.height != ref.height
            or patch.transform != ref.transform
        ):
            raise RuntimeError("Parent patch grid mismatch")
        new_buffer = patch.read(1)
        p = np.load(STEP43 / "intermediate/CO_p2_sparse_cells.npz")
        p_rows = p["rows"].astype(np.int32)
        p_cols = p["cols"].astype(np.int32)
        p_counts = p["counts"].astype(np.uint32)
        s_count = PARENT / "WUI_S_CO_centroid_count_30m.tif"

        rows: list[dict] = []
        output_paths: dict[tuple[str, int], Path] = {}
        total = len(RADII) * 2
        done = 0
        for radius in RADII:
            radius_px = max(int(round(radius / 30.0)), 1)
            kernel = disk(radius_px).astype(np.float32)
            area_km2 = math.pi * radius**2 / 1_000_000.0
            minimum_count = int(math.floor(6.17 * area_km2)) + 1
            if int(kernel.sum()) % 2 != 1:
                raise RuntimeError(f"Even kernel cell count at {radius} m")
            for method in ("WUI-P", "WUI-S"):
                method_started = time.perf_counter()
                out = (
                    output
                    / "rasters"
                    / method
                    / f"{method.replace('-', '_')}_CO_r{radius:04d}m_SILVIS_GT75.tif"
                )
                if method == "WUI-P":
                    stats = base.classify(
                        out,
                        method,
                        p_rows,
                        p_cols,
                        p_counts,
                        None,
                        new_buffer,
                        ref,
                        kernel,
                        radius_px=radius_px,
                        minimum_count=minimum_count,
                        official_path_override=official_p(radius),
                    )
                else:
                    stats = base.classify(
                        out,
                        method,
                        None,
                        None,
                        None,
                        s_count,
                        new_buffer,
                        ref,
                        kernel,
                        radius_px=radius_px,
                        minimum_count=minimum_count,
                        official_path_override=official_s(radius),
                    )
                done += 1
                elapsed = time.perf_counter() - started
                eta = elapsed * (total - done) / done
                print(
                    f"[STEP47B] radius={radius} method={method} "
                    f"completed={done}/{total} "
                    f"percent={100*done/total:.2f} "
                    f"elapsed_s={elapsed:.1f} ETA_s={eta:.1f}",
                    flush=True,
                )
                output_paths[(method, radius)] = out
                rows.append(
                    {
                        "state": "CO",
                        "method": method,
                        "radius_m": radius,
                        "radius_px": radius_px,
                        "kernel_cells": int(kernel.sum()),
                        "minimum_integer_count": minimum_count,
                        "density_rule": "D>6.17",
                        "intermix_rule": "V>=50%",
                        "interface_rule": "V<50% and distance<=2400m",
                        "patch_rule": (
                            "Census blocks Veg_Percent>75%; dissolve "
                            "contiguous; component area>=5km2"
                        ),
                        **stats,
                        "classification_sha256": sha256(out),
                        "wall_runtime_seconds": (
                            time.perf_counter() - method_started
                        ),
                        "output_path": str(out),
                    }
                )

    pair_rows = []
    for radius in RADII:
        values = pairwise(
            output_paths[("WUI-P", radius)],
            output_paths[("WUI-S", radius)],
        )
        pair_rows.append(
            {
                "state": "CO",
                "radius_m": radius,
                "scenario": "SILVIS_BLOCK_GT75_PATCH",
                **values,
            }
        )

    write_csv(output / "colorado_patch75_remaining_radii_impact.csv", rows)
    write_csv(output / "colorado_patch75_ps_jaccard_9.csv", pair_rows)
    baseline_failures = [
        row
        for row in rows
        if int(row["baseline_mismatch_pixels"]) != 0
    ]
    class_conservation_failures = [
        row
        for row in rows
        if (
            int(row["new_non_wui"])
            + int(row["new_intermix"])
            + int(row["new_interface"])
            != int(row["valid_pixels"])
        )
    ]
    output_rasters = sorted((output / "rasters").rglob("*.tif"))
    status = {
        "step": "STEP47B_PATCH75_COLORADO_REMAINING_RADII",
        "status": (
            "COLORADO_18_RASTERS_COMPLETE"
            if not baseline_failures and not class_conservation_failures
            else "QC_REVIEW_REQUIRED"
        ),
        "created_utc": created,
        "completed_utc": now_utc(),
        "parent_run": str(PARENT),
        "state": "CO",
        "radii_m": RADII,
        "raster_count": len(output_rasters),
        "wui_p_raster_count": sum(
            "WUI-P" in str(path) for path in output_rasters
        ),
        "wui_s_raster_count": sum(
            "WUI-S" in str(path) for path in output_rasters
        ),
        "baseline_failure_count": len(baseline_failures),
        "class_conservation_failure_count": len(
            class_conservation_failures
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "environment_modified": False,
        "upstream_modified": False,
        "output_directory": str(output),
    }
    (output / "step47b_status.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    (output / "input_sha256.json").write_text(
        json.dumps(parent_hashes, indent=2), encoding="utf-8"
    )
    qc = [
        "STEP47B COLORADO REMAINING RADII QC",
        f"completed_utc={status['completed_utc']}",
        f"raster_count={len(output_rasters)}",
        f"baseline_failure_count={len(baseline_failures)}",
        (
            "class_conservation_failure_count="
            f"{len(class_conservation_failures)}"
        ),
        f"elapsed_seconds={status['elapsed_seconds']:.3f}",
        f"final_status={status['status']}",
    ]
    (output / "STEP47B_FINAL_QC.txt").write_text(
        "\n".join(qc) + "\n", encoding="utf-8"
    )
    readme = """# Step47B Colorado remaining-radius patch75 canary

Generated diagnostic WUI-P and WUI-S rasters at 100, 200, 300, 400, 600,
700, 800, 900, and 1,000 m. The validated Step47 Colorado >75% qualifying
patch and 2,400 m mask are reused. The 500 m canary remains in the parent run.

No WUI-Z, population, Moran, or formal manuscript table was recomputed.
No Step43--46 file was modified.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")

    files = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "sha256_manifest.txt"
    )
    with (output / "sha256_manifest.txt").open(
        "w", encoding="utf-8"
    ) as f:
        for path in files:
            f.write(f"{sha256(path)}  {path.relative_to(output)}\n")
    print(json.dumps(status, indent=2), flush=True)
    return 0 if status["status"] == "COLORADO_18_RASTERS_COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
