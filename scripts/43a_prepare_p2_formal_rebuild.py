#!/usr/bin/env python3
# Copyright (C) 2026 WUI Mapping Project Authors
# SPDX-License-Identifier: GPL-3.0-only
# Repository version: filesystem paths are loaded through repo_config.py.
# Copy config/paths.example.json to config/paths.json before a new run.
# Raw input data and large intermediate files are stored outside GitHub.

"""Step43 Phase A: freeze policy, code, inputs, environment, and capacity.

This phase is read-only with respect to every research/legacy input.  It hashes
the complete Step42 source inventory (included addresses and explicitly
excluded buildings), locks the 98 wildland/distance reference rasters, verifies
the 94 legacy class rasters, and creates the production configuration.  It does
not create a classification raster.
"""

from __future__ import annotations

from repo_config import portable_path

import argparse
import concurrent.futures
import csv
import hashlib
import importlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyogrio
import rasterio
from pyproj import CRS


PROJECT = Path(portable_path("project"))
STEP41 = PROJECT / "step41_wuip_source_nodata_audit_20260727T054819Z"
STEP42 = PROJECT / "step42_wuip_address_policy_audit_20260727T153136Z"
STEP36 = PROJECT / "step36_area_final_audit_20260724T192140Z"
STATE_GPKG = Path(
    portable_path("data", "MBF+NLCD_2022US/Inputs/Processed_GPKG/tl_2022_us_state.gpkg")
)
WUIS_ROOT = Path(portable_path("data", "WUI_S_Paper"))
PRODUCTION_SOURCES = [
    PROJECT / "scripts/42b_five_state_address_identity_policy_audit.py",
    PROJECT / "scripts/41_wuip_source_dedup_nodata_impact_audit.py",
    Path(portable_path("legacy", "MBF+NLCD_2022US/09_analyze_wui_p_raster_fast.py")),
    PROJECT / "scripts/43a_prepare_p2_formal_rebuild.py",
    PROJECT / "scripts/43a_repair_step42_canary_input_freeze.py",
    PROJECT / "scripts/43b_run_p2_formal_rebuild.py",
    PROJECT / "scripts/43c_finalize_p2_formal_rebuild.py",
]
REQUIRED42 = [
    "recommended_wui_p_policy.md",
    "wui_p_source_policy_all49.csv",
    "wui_p_source_files_all49.csv",
    "rebuild_scope_by_state.csv",
    "wui_p_duplicate_identity_summary.csv",
    "wui_p_address_policy_500m_impact.csv",
    "STEP42_FINAL_QC.txt",
    "step42_status.json",
    "sha256_manifest.txt",
]
FIVE = {"CA", "CO", "FL", "PA", "TX"}
MIN_FREE_BYTES = 60 * 1024**3


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    temp = path.with_suffix(path.suffix + ".partial")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def command_text(command: list[str]) -> str:
    result = subprocess.run(
        command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    if result.returncode:
        raise RuntimeError(f"Command failed: {command}\n{result.stdout}")
    return result.stdout.strip()


def check_step42() -> dict[str, Any]:
    missing = [name for name in REQUIRED42 if not (STEP42 / name).is_file()]
    if missing:
        raise RuntimeError(f"Step42 required files missing: {missing}")
    status = json.loads((STEP42 / "step42_status.json").read_text())
    if status.get("status") != "ADDRESS_POLICY_READY_FOR_REBUILD":
        raise RuntimeError(f"Step42 policy is not ready: {status.get('status')}")
    qc = (STEP42 / "STEP42_FINAL_QC.txt").read_text()
    if "qc_status=PASS" not in qc or "checks_failed=0" not in qc:
        raise RuntimeError("Step42 final QC is not PASS")
    policy = (STEP42 / "recommended_wui_p_policy.md").read_text()
    if "P2_ADDRESS_EXACT_RECORD_DEDUP" not in policy:
        raise RuntimeError("Step42 frozen policy is not P2")
    # Verify the complete Step42 seal.
    result = subprocess.run(
        ["sha256sum", "-c", "sha256_manifest.txt"],
        cwd=STEP42,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode:
        raise RuntimeError(f"Step42 SHA manifest failed:\n{result.stdout}")
    return status


def step42_canary_sources() -> dict[tuple[str, str], dict[str, Any]]:
    """Return the exact five-state files recorded by Step42 execution."""
    sources: dict[tuple[str, str], dict[str, Any]] = {}
    for state in sorted(FIVE):
        checkpoint = STEP42 / "checkpoints" / f"{state}_step42.json"
        payload = json.loads(checkpoint.read_text())
        for row in payload["source_records"]:
            key = (state, row["basename"])
            if key in sources:
                raise RuntimeError(f"Duplicate Step42 canary source key: {key}")
            sources[key] = row
    return sources


def raster_record(path: Path) -> dict[str, Any]:
    with rasterio.open(path) as source:
        return {
            "path": str(path),
            "width": int(source.width),
            "height": int(source.height),
            "crs": source.crs.to_string() if source.crs else "",
            "transform": json.dumps([float(x) for x in source.transform[:6]]),
            "dtype": source.dtypes[0],
            "nodata": source.nodata,
            "file_size": path.stat().st_size,
            "mtime_utc": datetime.fromtimestamp(
                path.stat().st_mtime, timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }


def hash_records(
    rows: list[dict[str, Any]], label: str, workers: int = 3
) -> list[dict[str, Any]]:
    total_bytes = sum(int(row["file_size"]) for row in rows)
    completed_bytes = 0
    completed = 0
    started = time.monotonic()
    result: dict[int, str] = {}

    def task(index_row: tuple[int, dict[str, Any]]) -> tuple[int, str]:
        index, row = index_row
        return index, sha256_file(Path(row["source_path"]))

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(task, item): item
            for item in enumerate(rows)
        }
        for future in concurrent.futures.as_completed(futures):
            index, digest = future.result()
            result[index] = digest
            completed += 1
            completed_bytes += int(rows[index]["file_size"])
            elapsed = max(time.monotonic() - started, 1e-9)
            rate = completed_bytes / elapsed
            eta = (total_bytes - completed_bytes) / rate if rate else 0
            print(
                f"[HASH_{label}] phase=A state={rows[index].get('state','ALL')} "
                f"file={Path(rows[index]['source_path']).name} "
                f"completed={completed}/{len(rows)} "
                f"percent={100*completed_bytes/max(total_bytes,1):.2f}% "
                f"bytes={completed_bytes:,}/{total_bytes:,} "
                f"elapsed={elapsed:.1f}s ETA={eta:.1f}s",
                flush=True,
            )
    for index, row in enumerate(rows):
        row["sha256"] = result[index]
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hash-workers", type=int, default=3)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    started = time.monotonic()
    print("[PHASE_A] step=1/8 startup checks", flush=True)

    mount = command_text(["findmnt", "-no", "TARGET,SOURCE,FSTYPE,OPTIONS", portable_path("researchdrive")])
    if "cifs" not in mount or "ro" not in mount.split()[-1].split(","):
        raise RuntimeError(f"Research drive is not a read-only CIFS mount: {mount}")
    step42_status = check_step42()
    for name in REQUIRED42:
        if not (STEP42 / name).is_file():
            raise RuntimeError(name)
    if not STEP41.is_dir() or not (STEP41 / "STEP41_FINAL_QC.txt").is_file():
        raise RuntimeError("Step41 is incomplete")
    if not STATE_GPKG.is_file():
        raise FileNotFoundError(STATE_GPKG)
    free = shutil.disk_usage(output).free
    if free < MIN_FREE_BYTES:
        raise RuntimeError(
            f"Insufficient local free space: {free/2**30:.2f} GiB; "
            f"required >= {MIN_FREE_BYTES/2**30:.2f} GiB"
        )
    print("[PHASE_A] step=2/8 policy and capacity gates PASS", flush=True)

    source = pd.read_csv(
        STEP42 / "wui_p_source_files_all49.csv",
        keep_default_na=False,
        dtype={"STATEFP": str},
    )
    if len(source) != 2538 or source["state"].nunique() != 49:
        raise RuntimeError("Step42 source inventory cardinality changed")
    if not source["source_type"].isin(["address", "building"]).all():
        raise RuntimeError("Unknown source type exists")
    included_flag = (
        source["current_formal_wuip_included"].astype(str).str.lower().eq("true")
        & source["source_type"].eq("address")
    )
    if source.loc[included_flag].groupby("state").size().size != 49:
        raise RuntimeError("Not all 49 units have included address files")
    manifest_rows: list[dict[str, Any]] = []
    canary_sources = step42_canary_sources()
    for row, include in zip(source.to_dict("records"), included_flag):
        path = Path(row["source_path"])
        authority = ""
        if include and row["state"] in FIVE:
            canary = canary_sources.get((row["state"], path.name))
            if canary is None:
                raise RuntimeError(
                    f"Step42 canary source missing: {row['state']} {path.name}"
                )
            canary_path = Path(canary["source_path"])
            if "/replacements/" in canary_path.as_posix():
                path = canary_path
                row["raw_feature_count"] = int(canary["feature_count"])
                authority = "STEP42_CANARY_LOCKED_SOURCE_OVERRIDE"
        if not path.is_file():
            raise FileNotFoundError(path)
        stat = path.stat()
        if include:
            reason = "P2_ADDRESS_SOURCE_INCLUDED"
            authority = authority or "STEP42_CURRENT_ADDRESS_SOURCE"
        elif row["source_type"] == "building":
            reason = "BUILDING_RESERVED_FOR_WUI_S_EXCLUDED_FROM_WUI_P"
            authority = "STEP42_BUILDING_EXCLUSION"
        else:
            reason = (
                str(row.get("producer_skip_reason", "")).strip()
                or "STEP42_NONCURRENT_ADDRESS_SOURCE_EXCLUDED"
            )
            authority = "STEP42_EXPLICIT_ADDRESS_EXCLUSION"
        manifest_rows.append(
            {
                "state": row["state"],
                "source_type": row["source_type"],
                "source_path": str(path),
                "file_size": int(stat.st_size),
                "mtime_utc": datetime.fromtimestamp(
                    stat.st_mtime, timezone.utc
                ).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "sha256": "",
                "record_count": int(row["raw_feature_count"]),
                "included_yes_no": "YES" if include else "NO",
                "exclusion_reason": "" if include else reason,
                "inclusion_reason": reason if include else "",
                "step42_authoritative_status": authority,
            }
        )
    print(
        f"[PHASE_A] step=3/8 source typing PASS rows={len(manifest_rows)} "
        f"included_address={sum(r['included_yes_no']=='YES' for r in manifest_rows)} "
        f"included_building={sum(r['included_yes_no']=='YES' and r['source_type']=='building' for r in manifest_rows)}",
        flush=True,
    )

    # Hash all source rows, including excluded building files, to prove the
    # exclusion set is frozen rather than inferred later.
    manifest_rows = hash_records(
        manifest_rows, "POINT_SOURCES", workers=args.hash_workers
    )
    write_csv(output / "manifests/p2_input_manifest_all49.csv", manifest_rows)
    write_csv(
        output / "manifests/p2_excluded_building_manifest.csv",
        [row for row in manifest_rows if row["source_type"] == "building"],
    )
    print("[PHASE_A] step=4/8 point-source hashes frozen", flush=True)

    state_policy = pd.read_csv(STEP42 / "wui_p_source_policy_all49.csv")
    reference_rows: list[dict[str, Any]] = []
    for row in state_policy.itertuples():
        stem = (
            "DistrictofColumbia"
            if row.state == "DC"
            else str(row.state_name).replace(" ", "")
        )
        for role, suffix in [
            ("WILDLAND_BIN", f"{stem}_wildland_bin.tif"),
            ("DIST_TO_LARGEPATCH", f"{stem}_dist_to_largepatch.tif"),
        ]:
            path = WUIS_ROOT / stem / suffix
            if not path.is_file():
                raise FileNotFoundError(path)
            record = raster_record(path)
            if not CRS.from_user_input(record["crs"]).equals(CRS.from_epsg(5070)):
                raise RuntimeError(f"Reference raster is not EPSG:5070: {path}")
            reference_rows.append(
                {
                    "state": row.state,
                    "role": role,
                    "source_path": str(path),
                    **{k: v for k, v in record.items() if k != "path"},
                    "sha256": "",
                }
            )
    # Grid-pair checks occur before the expensive hashes.
    refs = pd.DataFrame(reference_rows)
    for state, group in refs.groupby("state"):
        if len(group) != 2:
            raise RuntimeError(f"Reference pair unresolved: {state}")
        signatures = group[["width", "height", "crs", "transform"]].drop_duplicates()
        if len(signatures) != 1:
            raise RuntimeError(f"Reference grid mismatch: {state}")
    reference_rows = hash_records(
        reference_rows, "REFERENCE_RASTERS", workers=args.hash_workers
    )
    write_csv(
        output / "manifests/p2_reference_raster_manifest.csv",
        reference_rows,
    )
    print("[PHASE_A] step=5/8 reference rasters frozen", flush=True)

    legacy = pd.read_csv(
        STEP41 / "wui_p_authoritative_input_manifest.csv",
        keep_default_na=False,
    )
    legacy = legacy[legacy["role"].eq("FORMAL_WUI_P_CLASS_RASTER")].copy()
    if len(legacy) != 94:
        raise RuntimeError(f"Legacy raster count is not 94: {len(legacy)}")
    legacy_rows: list[dict[str, Any]] = []
    raw_output_bytes = 0
    existing_output_bytes = 0
    for row in legacy.itertuples():
        path = Path(row.source_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        record = raster_record(path)
        if sha256_file(path) != row.sha256:
            raise RuntimeError(f"Legacy raster hash changed: {path}")
        raw_output_bytes += int(record["width"]) * int(record["height"])
        existing_output_bytes += path.stat().st_size
        legacy_rows.append(
            {
                "state": row.state,
                "buffer_m": int(row.buffer_m),
                "source_path": str(path),
                **{k: v for k, v in record.items() if k != "path"},
                "sha256": row.sha256,
                "step41_authoritative_status": row.authoritative_status,
            }
        )
    write_csv(output / "manifests/p2_legacy_raster_manifest.csv", legacy_rows)
    print("[PHASE_A] step=6/8 legacy 94-raster seal verified", flush=True)

    script_rows: list[dict[str, Any]] = []
    for path in PRODUCTION_SOURCES:
        if not path.is_file():
            raise FileNotFoundError(path)
        frozen = output / "scripts" / path.name
        shutil.copy2(path, frozen)
        script_rows.append(
            {
                "source_path": str(path),
                "frozen_path": str(frozen),
                "file_size": path.stat().st_size,
                "sha256": sha256_file(path),
                "frozen_sha256": sha256_file(frozen),
                "role": (
                    "STEP43_PRODUCTION"
                    if path.name.startswith("43")
                    else "METHOD_AUTHORITY_REFERENCE"
                ),
            }
        )
    write_csv(output / "manifests/p2_script_manifest.csv", script_rows)

    policy_hashes = {
        name: sha256_file(STEP42 / name)
        for name in REQUIRED42
    }
    environment = {
        "created_utc": utc_now(),
        "hostname": socket.gethostname(),
        "whoami": command_text(["whoami"]),
        "pwd": str(PROJECT),
        "python": sys.version,
        "platform": platform.platform(),
        "mount": mount,
        "local_free_bytes_at_start": free,
        "memory_total_bytes": int(
            os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        ),
        "dependencies": {
            name: importlib.import_module(name).__version__
            for name in [
                "numpy",
                "pandas",
                "pyogrio",
                "rasterio",
                "pyproj",
                "scipy",
                "skimage",
                "shapely",
            ]
        },
        "state_boundary": {
            "path": str(STATE_GPKG),
            "size": STATE_GPKG.stat().st_size,
            "sha256": sha256_file(STATE_GPKG),
        },
    }
    atomic_json(output / "config/environment.json", environment)
    policy_payload = {
        "policy": "P2_ADDRESS_EXACT_RECORD_DEDUP",
        "source_policy": "ADDRESS_ONLY",
        "remove": ["D1"],
        "do_not_remove": ["D2", "D3", "D4", "D5", "D6", "D7"],
        "step42_directory": str(STEP42),
        "step42_status": step42_status["status"],
        "step42_policy_file_sha256": policy_hashes[
            "recommended_wui_p_policy.md"
        ],
        "step42_500m_impact_sha256": policy_hashes[
            "wui_p_address_policy_500m_impact.csv"
        ],
        "all_step42_required_hashes": policy_hashes,
        "frozen_utc": utc_now(),
    }
    atomic_json(output / "config/p2_policy_frozen.json", policy_payload)
    config = {
        "analysis_crs": "EPSG:5070",
        "pixel_size_m": 30.0,
        "tile_size": 2048,
        "density_threshold_points_per_km2_strictly_greater_than": 6.17,
        "buffers_m": list(range(100, 1001, 100)),
        "national_500m_units": 49,
        "sensitivity_states": sorted(FIVE),
        "expected_unique_rasters": 94,
        "point_kernel": "skimage.morphology.disk(round(buffer_m/30))",
        "density_convolution": "scipy.signal.fftconvolve; rint integer count",
        "classification_gate": (
            "legacy P0 class at same grid/buffer; mathematically exact deletion-only "
            "reclassification because P2 point set is a subset and vegetation/distance "
            "rules are unchanged"
        ),
        "class_values": {"non_wui": 0, "intermix": 1, "interface": 2},
        "dtype": "uint8",
        "nodata": 255,
        "valid_domain": (
            "2022 Census state geometry, pixel-center mask "
            f"from {STATE_GPKG}"
        ),
        "compression": "LZW",
        "tiled": True,
        "bigtiff": "YES",
        "source_order": "normalized absolute source path sorted",
        "record_order": "original OGR row position within source",
        "D1_keep_rule": "first record in source-path then source-row order",
        "atomic_output": "*.partial.tif then os.replace after QC",
        "raw_output_bytes_worst_case": raw_output_bytes,
        "legacy_compressed_bytes_observed": existing_output_bytes,
        "local_free_bytes_at_start": free,
        "peak_memory_estimate_bytes": 18 * 1024**3,
        "runtime_estimate_hours": [3.0, 7.0],
        "temporary_space_estimate_bytes": 12 * 1024**3,
        "output_space_estimate_bytes": max(
            2 * existing_output_bytes, 2 * 1024**3
        ),
    }
    atomic_json(output / "config/p2_production_config.json", config)
    print("[PHASE_A] step=7/8 code, policy and config frozen", flush=True)

    # Small-record dry-run: use five records per first included address file.
    dry_rows = []
    for state, group in pd.DataFrame(manifest_rows).query(
        "included_yes_no == 'YES'"
    ).groupby("state", sort=True):
        path = Path(sorted(group["source_path"])[0])
        frame = pyogrio.read_dataframe(path, max_features=5, use_arrow=True)
        if len(frame) == 0:
            raise RuntimeError(f"Dry-run source is empty: {path}")
        dry_rows.append(
            {
                "state": state,
                "source_path": str(path),
                "records_read": len(frame),
                "geometry_types": ";".join(
                    sorted(set(frame.geometry.geom_type.dropna().astype(str)))
                ),
                "building_included": "NO",
                "dry_run_status": "PASS",
            }
        )
    write_csv(output / "qc/phaseA_dry_run.csv", dry_rows)
    startup = {
        "status": "PHASE_A_PREPARED_FOR_CANARY",
        "completed_utc": utc_now(),
        "output_directory": str(output),
        "input_manifest_rows": len(manifest_rows),
        "included_address_files": sum(
            row["included_yes_no"] == "YES" for row in manifest_rows
        ),
        "included_building_files": sum(
            row["included_yes_no"] == "YES"
            and row["source_type"] == "building"
            for row in manifest_rows
        ),
        "unknown_files": sum(
            row["source_type"] == "unknown" for row in manifest_rows
        ),
        "reference_rasters": len(reference_rows),
        "legacy_rasters": len(legacy_rows),
        "dry_run_states": len(dry_rows),
        "elapsed_seconds": time.monotonic() - started,
        "research_drive_write_attempted": False,
        "sudo_used": False,
    }
    atomic_json(output / "checkpoints/phaseA_complete.json", startup)
    print(
        f"[PHASE_A] step=8/8 COMPLETE elapsed={time.monotonic()-started:.1f}s "
        f"output={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
