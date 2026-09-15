#!/usr/bin/env python3
# Copyright (C) 2026 WUI Mapping Project Authors
# SPDX-License-Identifier: GPL-3.0-only
# Repository version: filesystem paths are loaded through repo_config.py.
# Copy config/paths.example.json to config/paths.json before a new run.
# Raw input data and large intermediate files are stored outside GitHub.

"""Amend an existing Step43 freeze with exact Step42 canary provenance."""

from __future__ import annotations

from repo_config import portable_path

import argparse
import csv
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT = Path(portable_path("project"))
STEP42 = PROJECT / "step42_wuip_address_policy_audit_20260727T153136Z"
FIVE = {"CA", "CO", "FL", "PA", "TX"}
PRODUCTION_SOURCES = [
    PROJECT / "scripts/42b_five_state_address_identity_policy_audit.py",
    PROJECT / "scripts/41_wuip_source_dedup_nodata_impact_audit.py",
    Path(portable_path("legacy", "MBF+NLCD_2022US/09_analyze_wui_p_raster_fast.py")),
    PROJECT / "scripts/43a_prepare_p2_formal_rebuild.py",
    PROJECT / "scripts/43a_repair_step42_canary_input_freeze.py",
    PROJECT / "scripts/43b_run_p2_formal_rebuild.py",
    PROJECT / "scripts/43c_finalize_p2_formal_rebuild.py",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = list(rows[0])
    temp = path.with_suffix(path.suffix + ".partial")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def atomic_json(path: Path, payload: Any) -> None:
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    manifest_path = output / "manifests/p2_input_manifest_all49.csv"
    manifest = pd.read_csv(manifest_path, keep_default_na=False)

    canary: dict[tuple[str, str], dict[str, Any]] = {}
    for state in sorted(FIVE):
        checkpoint = json.loads(
            (STEP42 / "checkpoints" / f"{state}_step42.json").read_text()
        )
        for row in checkpoint["source_records"]:
            canary[(state, row["basename"])] = row

    changes: list[dict[str, Any]] = []
    for index, row in manifest.iterrows():
        if row["included_yes_no"] != "YES" or row["state"] not in FIVE:
            continue
        old_path = Path(row["source_path"])
        authoritative = canary.get((row["state"], old_path.name))
        if authoritative is None:
            raise RuntimeError(
                f"Missing Step42 canary provenance: {row['state']} {old_path.name}"
            )
        new_path = Path(authoritative["source_path"])
        if "/replacements/" not in new_path.as_posix():
            continue
        if not new_path.is_file():
            raise FileNotFoundError(new_path)
        stat = new_path.stat()
        actual_hash = sha256_file(new_path)
        expected_hash = authoritative.get("sha256", "")
        if expected_hash and actual_hash != expected_hash:
            raise RuntimeError(f"Step42 locked SHA changed: {new_path}")
        changes.append(
            {
                "state": row["state"],
                "basename": old_path.name,
                "old_source_path": str(old_path),
                "old_sha256": row["sha256"],
                "old_record_count": int(row["record_count"]),
                "new_source_path": str(new_path),
                "new_sha256": actual_hash,
                "new_record_count": int(authoritative["feature_count"]),
                "reason": "STEP42_CANARY_EXECUTION_PROVENANCE_OVERRIDES_CROSSWALK_PATH",
            }
        )
        manifest.at[index, "source_path"] = str(new_path)
        manifest.at[index, "file_size"] = int(stat.st_size)
        manifest.at[index, "mtime_utc"] = datetime.fromtimestamp(
            stat.st_mtime, timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        manifest.at[index, "sha256"] = actual_hash
        manifest.at[index, "record_count"] = int(authoritative["feature_count"])
        manifest.at[
            index, "step42_authoritative_status"
        ] = "STEP42_CANARY_LOCKED_SOURCE_OVERRIDE"

    expected = {
        ("TX", "kerr-addresses-county.geojson"),
        ("TX", "mclennan-addresses-county.geojson"),
        ("TX", "midland-addresses-county.geojson"),
    }
    observed = {(row["state"], row["basename"]) for row in changes}
    if observed != expected:
        raise RuntimeError(
            f"Expected exactly three TX source overrides; observed={observed}"
        )

    write_csv(manifest_path, manifest.to_dict("records"))
    write_csv(
        output / "manifests/p2_step42_canary_source_override_audit.csv",
        changes,
    )

    script_rows = []
    for source in PRODUCTION_SOURCES:
        destination = output / "scripts" / source.name
        shutil.copy2(source, destination)
        script_rows.append(
            {
                "role": "PRODUCTION_OR_PROVENANCE",
                "source_path": str(source),
                "frozen_path": str(destination),
                "sha256": sha256_file(destination),
            }
        )
    write_csv(output / "manifests/p2_script_manifest.csv", script_rows)

    phase_a = json.loads(
        (output / "checkpoints/phaseA_complete.json").read_text()
    )
    phase_a["input_authority_amended_utc"] = utc_now()
    phase_a["step42_canary_source_overrides"] = len(changes)
    phase_a["step42_canary_override_states"] = ["TX"]
    atomic_json(output / "checkpoints/phaseA_complete.json", phase_a)
    atomic_json(
        output / "checkpoints/phaseA_input_authority_amendment.json",
        {
            "status": "PASS",
            "completed_utc": utc_now(),
            "reason": (
                "Step42 all-49 crosswalk listed logical research-drive paths, "
                "while the TX execution checkpoint used three locked recovery "
                "files; exact canary provenance is authoritative."
            ),
            "changes": changes,
            "other_manifest_rows_changed": 0,
            "rasters_changed": 0,
        },
    )
    failed_checkpoint = output / "checkpoints/phase_canary_FAILED.json"
    if failed_checkpoint.is_file():
        failure = json.loads(failed_checkpoint.read_text())
        failure["resolution"] = (
            "PENDING_RETRY_AFTER_STEP42_CANARY_SOURCE_OVERRIDE"
        )
        failure["root_cause"] = (
            "STEP42_CROSSWALK_EXECUTION_PROVENANCE_PATH_MISMATCH"
        )
        write_csv(
            output / "qc/p2_failure_attempt_history.csv",
            [failure],
        )
    print(
        f"[PHASE_A_AMENDMENT] overrides={len(changes)} states=TX "
        f"output={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
