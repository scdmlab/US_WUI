#!/usr/bin/env python3
# Copyright (C) 2026 WUI Mapping Project Authors
# SPDX-License-Identifier: GPL-3.0-only
# Repository version: filesystem paths are loaded through repo_config.py.
# Copy config/paths.example.json to config/paths.json before a new run.
# Raw input data and large intermediate files are stored outside GitHub.

"""Step42A: manuscript-method crosswalk and all-49 raw source inventory."""

from __future__ import annotations

from repo_config import portable_path

import argparse
import csv
import importlib.util
import json
import os
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT = Path(portable_path("project"))
STEP41 = PROJECT / "step41_wuip_source_nodata_audit_20260727T054819Z"
STEP41_SCRIPT = PROJECT / "scripts/41_wuip_source_dedup_nodata_impact_audit.py"
MANUSCRIPT = Path(
    portable_path("attachments", "422107a3-c115-42e6-aec7-dc9136829560/WUI_Generation_Hongbo (6).pdf")
)


def load_step41():
    spec = importlib.util.spec_from_file_location("step41", STEP41_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict]):
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def progress(done: int, total: int, started: float, state: str, detail: str):
    elapsed = time.monotonic() - started
    rate = done / elapsed if elapsed else 0.0
    eta = (total - done) / rate if rate else 0.0
    print(
        f"[STEP42_SOURCE_INVENTORY] step=2/5 state={state} scenario=SOURCE_POLICY "
        f"completed={done}/{total} percent={100*done/total:.2f}% "
        f"elapsed={elapsed:.1f}s ETA={eta:.1f}s {detail}",
        flush=True,
    )


def manuscript_crosswalk(m) -> list[dict]:
    production = Path(portable_path("legacy", "process_oa_v2.py"))
    raster = Path(
        portable_path("legacy", "MBF+NLCD_2022US/09_analyze_wui_p_raster_fast.py")
    )
    rows = [
        {
            "document": str(MANUSCRIPT),
            "section": "Abstract",
            "line_or_page": "PDF p.1",
            "method": "WUI-P",
            "declared_source": "OpenAddresses data",
            "actual_source": "Non-Texas legacy formal inputs mix address and building; Texas is address-only",
            "consistent_yes_no": "NO",
            "required_revision": "REBUILD_WUI_P_FROM_ADDRESS_ONLY; manuscript source wording itself can remain",
            "evidence": "Abstract states CONUS WUI-P uses OpenAddresses data.",
            "finding": "METHOD_INPUT_MISMATCH",
        },
        {
            "document": str(MANUSCRIPT),
            "section": "Introduction",
            "line_or_page": "PDF pp.2-3",
            "method": "WUI-P",
            "declared_source": "geocoded OpenAddresses records/address points",
            "actual_source": "Legacy producer selects filenames containing address OR building",
            "consistent_yes_no": "NO",
            "required_revision": "REBUILD_NON_ADDRESS_ONLY_STATES",
            "evidence": "Text explicitly contrasts WUI-P OpenAddresses records with WUI-S footprint-derived centroids.",
            "finding": "METHOD_INPUT_MISMATCH",
        },
        {
            "document": str(MANUSCRIPT),
            "section": "3.2 Data Sources and Table 1",
            "line_or_page": "PDF pp.8-9",
            "method": "WUI-P",
            "declared_source": "OpenAddresses vector address points (2025 snapshot)",
            "actual_source": "Legacy formal chain includes building-named raw sources in non-Texas point GeoPackages",
            "consistent_yes_no": "NO",
            "required_revision": "REBUILD_WUI_P_ADDRESS_ONLY",
            "evidence": "Table assigns OpenAddresses address points to WUI-P and Microsoft footprints/centroids to WUI-S.",
            "finding": "METHOD_INPUT_MISMATCH",
        },
        {
            "document": str(MANUSCRIPT),
            "section": "3.5 WUI-S",
            "line_or_page": "PDF p.11",
            "method": "WUI-S",
            "declared_source": "Microsoft Building Footprint polygons converted to centroids",
            "actual_source": "Separate formal WUI-S chain",
            "consistent_yes_no": "YES",
            "required_revision": "NONE_FOR_SOURCE_POLICY",
            "evidence": "Section 3.5 explicitly assigns building footprints to WUI-S.",
            "finding": "CONSISTENT",
        },
        {
            "document": str(MANUSCRIPT),
            "section": "3.6 WUI-P",
            "line_or_page": "PDF pp.11-12",
            "method": "WUI-P",
            "declared_source": "OpenAddresses points; Nr(p) counts neighboring address points",
            "actual_source": "Four sampled non-Texas formal inputs are mixed; Texas revised input is address-only",
            "consistent_yes_no": "NO",
            "required_revision": "REBUILD_NON_ADDRESS_ONLY_STATES",
            "evidence": "Section says 'Instead of building footprints, WUI-P used OpenAddresses points.'",
            "finding": "METHOD_INPUT_MISMATCH",
        },
        {
            "document": str(MANUSCRIPT),
            "section": "entire manuscript",
            "line_or_page": "56 PDF pages searched",
            "method": "WUI-P deduplication",
            "declared_source": "No deduplication policy declared",
            "actual_source": "Legacy producer concatenates surviving records without deduplication",
            "consistent_yes_no": "YES_WITH_MISSING_METHOD_DETAIL",
            "required_revision": "After policy selection, disclose retained-record and deduplication rule",
            "evidence": "No 'dedup' occurrence; 'duplicate' occurrences do not declare an address-record policy.",
            "finding": "METHOD_DETAIL_MISSING",
        },
        {
            "document": str(PROJECT / "WORKFLOW_AUTHORITY.md"),
            "section": "method authority",
            "line_or_page": "lines 1-20",
            "method": "WUI-P/WUI-S",
            "declared_source": "Defines buffer/fixed-output authority but not point-source separation",
            "actual_source": "Step41 establishes mixed non-Texas legacy chain",
            "consistent_yes_no": "NO",
            "required_revision": "Future workflow update after Step42 policy approval; do not edit in this audit",
            "evidence": "Workflow authority does not document address/building separation.",
            "finding": "WORKFLOW_POLICY_GAP",
        },
        {
            "document": str(PROJECT / "step41_wuip_source_nodata_audit_20260727T054819Z/README.md"),
            "section": "Formal method",
            "line_or_page": "current Step41 README",
            "method": "Legacy WUI-P",
            "declared_source": "Documents concatenated address and building sources; Texas address-only",
            "actual_source": "Matches audited formal legacy chain",
            "consistent_yes_no": "YES",
            "required_revision": "Retain as legacy provenance; do not promote P0 as new policy",
            "evidence": "Step41 explicitly records source mixing and no deduplication.",
            "finding": "LEGACY_CHAIN_CORRECTLY_DOCUMENTED",
        },
        {
            "document": str(production),
            "section": "process_state",
            "line_or_page": "lines 62-128",
            "method": "Legacy WUI-P input preparation",
            "declared_source": "Comment says addresses or buildings ('we only want houses')",
            "actual_source": "Both basename classes concatenated",
            "consistent_yes_no": "NO_VS_MANUSCRIPT",
            "required_revision": "A future address-only producer must not select building files",
            "evidence": "Filename filter retains address OR building and pd.concat performs no dedup.",
            "finding": "METHOD_INPUT_MISMATCH",
        },
        {
            "document": str(raster),
            "section": "_load_address_xy / analyze_state",
            "line_or_page": "lines 92-112, 189-319",
            "method": "Legacy WUI-P classification",
            "declared_source": "File named <State>_addresses.gpkg",
            "actual_source": "Filename conceals mixed provenance outside Texas",
            "consistent_yes_no": "NO_VS_MANUSCRIPT",
            "required_revision": "Point input manifest must explicitly enforce address-only",
            "evidence": "Classifier cannot distinguish source type after geometry-only concatenation.",
            "finding": "PROVENANCE_LOSS",
        },
    ]
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    m = load_step41()

    manifest = pd.read_csv(
        STEP41 / "wui_p_authoritative_input_manifest.csv", keep_default_na=False
    )
    point_hash = {
        row.state: row.sha256
        for row in manifest[
            manifest["role"].eq("FORMAL_WUI_P_POINT_INPUT")
        ].itertuples()
    }
    aggregate41 = pd.read_csv(
        STEP41 / "wui_p_source_counts_by_state.csv", keep_default_na=False
    ).set_index("state")
    file_rows: list[dict] = []
    policy_rows: list[dict] = []
    started = time.monotonic()
    for done, (fips, abbr, name, _) in enumerate(m.STATES, 1):
        selected = m.selected_raw_sources(abbr)
        included = [x for x in selected if x["producer_included"]]
        address = [x for x in included if x["source_type"] == "address"]
        building = [x for x in included if x["source_type"] == "building"]
        if abbr == "TX":
            formal_address_files = 312
            formal_building_files = 0
            formal_address_records = 20_991_807
            formal_building_records = 0
            policy = "ADDRESS_ONLY"
        else:
            formal_address_files = len(address)
            formal_building_files = len(building)
            formal_address_records = sum(x["raw_feature_count"] for x in address)
            formal_building_records = sum(x["raw_feature_count"] for x in building)
            policy = (
                "MIXED_ADDRESS_BUILDING"
                if formal_address_files and formal_building_files
                else "ADDRESS_ONLY"
                if formal_address_files
                else "BUILDING_ONLY"
                if formal_building_files
                else "UNKNOWN"
            )
        for item in selected:
            path = Path(item["path"])
            current = item["producer_included"]
            if abbr == "TX":
                current = current and item["source_type"] == "address"
                role = (
                    "CURRENT_TEXAS_ADDRESS_SOURCE"
                    if current
                    else "LEGACY_NOT_CURRENT_WUIP"
                )
            else:
                role = (
                    "CURRENT_LEGACY_WUIP_SOURCE"
                    if current
                    else "PRODUCER_SKIPPED_WHOLE_SOURCE"
                )
            stat = path.stat()
            file_rows.append(
                {
                    "STATEFP": fips,
                    "state": abbr,
                    "state_name": name,
                    "source_type": item["source_type"],
                    "source_path": str(path),
                    "basename": path.name,
                    "raw_feature_count": item["raw_feature_count"],
                    "producer_included": item["producer_included"],
                    "current_formal_wuip_included": current,
                    "current_role": role,
                    "producer_skip_reason": item["producer_skip_reason"],
                    "file_size_bytes": stat.st_size,
                    "mtime_utc": datetime.fromtimestamp(
                        stat.st_mtime, tz=timezone.utc
                    ).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "formal_point_input_sha256": point_hash.get(abbr, ""),
                    "identity_audit_scope": (
                        "FULL_FIVE_STATE" if abbr in m.FIVE else "FILE_INVENTORY_ONLY"
                    ),
                }
            )
        raw = aggregate41.loc[abbr]
        policy_rows.append(
            {
                "STATEFP": fips,
                "state": abbr,
                "state_name": name,
                "current_policy": policy,
                "address_file_count": formal_address_files,
                "building_file_count": formal_building_files,
                "unknown_file_count": 0,
                "address_record_count": formal_address_records,
                "building_record_count": formal_building_records,
                "current_formal_point_feature_count": int(
                    raw["formal_point_feature_count"]
                ),
                "raw_record_count_current_sources": (
                    formal_address_records + formal_building_records
                ),
                "requires_formal_wuip_input_change": (
                    "NO" if policy == "ADDRESS_ONLY" else "YES"
                ),
                "target_policy": "ADDRESS_ONLY",
                "formal_point_input_path": raw["formal_point_path"],
                "formal_point_input_sha256": point_hash.get(abbr, ""),
                "evidence": raw["source_assignment_status"],
                "notes": (
                    "Texas Step19C current input; 49,803 outside-grid points removed"
                    if abbr == "TX"
                    else "Formal legacy source selection reconstructed exactly"
                ),
            }
        )
        progress(done, len(m.STATES), started, abbr, policy)

    write_csv(out / "wui_p_manuscript_method_crosswalk.csv", manuscript_crosswalk(m))
    write_csv(out / "wui_p_source_policy_all49.csv", policy_rows)
    write_csv(out / "wui_p_source_files_all49.csv", file_rows)
    startup = {
        "checked_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hostname": socket.gethostname(),
        "whoami": subprocess.run(
            ["whoami"], capture_output=True, text=True, check=True
        ).stdout.strip(),
        "pwd": str(Path.cwd()),
        "findmnt": subprocess.run(
            ["findmnt", "-T", portable_path("researchdrive"), "-o", "TARGET,SOURCE,FSTYPE,OPTIONS"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        "research_drive_write_attempted": False,
        "step41_manifest_rows": len(manifest),
        "step41_manifest_sha256_complete": bool(
            manifest["sha256"].str.len().eq(64).all()
        ),
        "source_file_inventory_rows": len(file_rows),
    }
    (out / "step42_startup_inventory.json").write_text(
        json.dumps(startup, indent=2), encoding="utf-8"
    )
    print(
        f"STEP42A COMPLETE states={len(policy_rows)} files={len(file_rows)} "
        f"output={out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
