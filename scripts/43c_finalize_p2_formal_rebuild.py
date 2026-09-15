#!/usr/bin/env python3
# Copyright (C) 2026 WUI Mapping Project Authors
# SPDX-License-Identifier: GPL-3.0-only
# Repository version: filesystem paths are loaded through repo_config.py.
# Copy config/paths.example.json to config/paths.json before a new run.
# Raw input data and large intermediate files are stored outside GitHub.

"""Step43 final consolidation, integrity verification, and release gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import rasterio


FIVE = {"CA", "CO", "FL", "PA", "TX"}
EXPECTED_BUFFERS = set(range(100, 1001, 100))
REQUIRED_ROOT = [
    "README.md",
    "STEP43_FINAL_QC.txt",
    "step43_status.json",
    "p2_policy_frozen.json",
    "p2_production_config.json",
    "p2_script_manifest.csv",
    "p2_input_manifest_all49.csv",
    "p2_excluded_building_manifest.csv",
    "p2_d1_audit_by_state.csv",
    "p2_d1_audit_by_file.csv",
    "p2_d1_field_definition.csv",
    "p2_rebuild_inventory.csv",
    "p2_canary_comparison_step42.csv",
    "p2_classification_change_vs_legacy.csv",
    "p2_raster_metadata_audit.csv",
    "p2_valid_domain_audit.csv",
    "p2_failed_or_skipped_runs.csv",
    "unresolved_items.csv",
    "sha256_manifest.txt",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(
        path,
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
    )


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


def copy_release_files(output: Path) -> None:
    mapping = {
        "config/p2_policy_frozen.json": "p2_policy_frozen.json",
        "config/p2_production_config.json": "p2_production_config.json",
        "manifests/p2_script_manifest.csv": "p2_script_manifest.csv",
        "manifests/p2_input_manifest_all49.csv": "p2_input_manifest_all49.csv",
        "manifests/p2_excluded_building_manifest.csv": "p2_excluded_building_manifest.csv",
        "dedup_audit/p2_d1_audit_by_state.csv": "p2_d1_audit_by_state.csv",
        "dedup_audit/p2_d1_audit_by_file.csv": "p2_d1_audit_by_file.csv",
        "dedup_audit/p2_d1_field_definition.csv": "p2_d1_field_definition.csv",
        "qc/p2_rebuild_inventory.csv": "p2_rebuild_inventory.csv",
        "qc/p2_canary_comparison_step42.csv": "p2_canary_comparison_step42.csv",
        "qc/p2_classification_change_vs_legacy.csv": "p2_classification_change_vs_legacy.csv",
        "qc/p2_raster_metadata_audit.csv": "p2_raster_metadata_audit.csv",
        "qc/p2_valid_domain_audit.csv": "p2_valid_domain_audit.csv",
        "qc/p2_failed_or_skipped_runs.csv": "p2_failed_or_skipped_runs.csv",
    }
    for source, target in mapping.items():
        src = output / source
        if not src.is_file():
            raise FileNotFoundError(src)
        shutil.copy2(src, output / target)


def validate_outputs(output: Path) -> tuple[list[str], list[str], pd.DataFrame]:
    inventory = pd.read_csv(output / "p2_rebuild_inventory.csv")
    dedup = pd.read_csv(output / "p2_d1_audit_by_state.csv")
    inputs = pd.read_csv(output / "p2_input_manifest_all49.csv")
    canary = pd.read_csv(output / "p2_canary_comparison_step42.csv")
    failed = pd.read_csv(output / "p2_failed_or_skipped_runs.csv")
    legacy = pd.read_csv(
        output / "manifests/p2_legacy_raster_manifest.csv"
    )
    passed: list[str] = []
    failures: list[str] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        text = f"{name}: {'PASS' if condition else 'FAIL'}"
        if detail:
            text += f" — {detail}"
        (passed if condition else failures).append(text)

    rows500 = inventory[inventory["buffer_m"].eq(500)]
    check(
        "Q01_49_UNITS_500M",
        len(rows500) == 49 and rows500["state"].nunique() == 49,
        f"rows={len(rows500)} states={rows500['state'].nunique()}",
    )
    buffer_sets = (
        inventory[inventory["state"].isin(FIVE)]
        .groupby("state")["buffer_m"]
        .apply(lambda values: set(map(int, values)))
    )
    check(
        "Q02_FIVE_STATES_TEN_BUFFERS",
        len(buffer_sets) == 5
        and all(values == EXPECTED_BUFFERS for values in buffer_sets),
    )
    check(
        "Q03_UNIQUE_RASTER_COUNT_94",
        len(inventory) == 94
        and inventory[["state", "buffer_m"]].drop_duplicates().shape[0] == 94,
        f"rows={len(inventory)}",
    )
    included = inputs[inputs["included_yes_no"].eq("YES")]
    check(
        "Q04_ALL_INCLUDED_INPUTS_ADDRESS",
        len(included) > 0 and included["source_type"].eq("address").all(),
    )
    check(
        "Q05_BUILDING_INCLUDED_ZERO",
        not (
            inputs["included_yes_no"].eq("YES")
            & inputs["source_type"].eq("building")
        ).any()
        and inventory["building_records_included"].astype(int).eq(0).all(),
    )
    check(
        "Q06_UNKNOWN_INPUT_ZERO",
        not inputs["source_type"].eq("unknown").any(),
    )
    check(
        "Q07_D1_AUDIT_ALL49",
        len(dedup) == 49
        and dedup["state"].nunique() == 49
        and dedup["status"].eq("PASS").all(),
    )
    policy = json.loads((output / "p2_policy_frozen.json").read_text())
    check(
        "Q08_D2_D7_NOT_DELETED",
        policy.get("remove") == ["D1"]
        and set(policy.get("do_not_remove", []))
        == {"D2", "D3", "D4", "D5", "D6", "D7"},
    )
    check(
        "Q09_CANARY_ZERO_DISAGREEMENT",
        len(canary) == 5
        and pd.to_numeric(
            canary["classification_disagreement_pixels_vs_step42"],
            errors="coerce",
        )
        .eq(0)
        .all(),
    )
    check(
        "Q10_CANARY_JACCARD_ONE",
        len(canary) == 5
        and pd.to_numeric(canary["jaccard_with_step42_P2"]).eq(1.0).all(),
    )
    check(
        "Q11_ZERO_IS_VALID_NON_WUI",
        inventory["non_wui_pixels"].astype(int).gt(0).all()
        and inventory["nodata"].astype(int).ne(0).all(),
    )
    check(
        "Q12_NODATA_255",
        inventory["nodata"].astype(int).eq(255).all(),
    )
    conservation = (
        inventory["valid_pixels"].astype(int)
        == inventory["non_wui_pixels"].astype(int)
        + inventory["intermix_pixels"].astype(int)
        + inventory["interface_pixels"].astype(int)
    )
    check("Q13_VALID_CLASS_CONSERVATION", conservation.all())
    grid_conservation = (
        inventory["valid_pixels"].astype(int)
        + inventory["out_of_domain_pixels"].astype(int)
        == inventory["width"].astype(int) * inventory["height"].astype(int)
    )
    same_domain = (
        inventory.groupby("state")["valid_pixels"].nunique().eq(1).all()
        and inventory.groupby("state")["out_of_domain_pixels"].nunique().eq(1).all()
    )
    check(
        "Q14_VALID_DOMAIN_STATE_MASK",
        grid_conservation.all() and same_domain,
    )
    check(
        "Q15_REPEAT_CLASSIFICATION_HASH",
        len(canary) == 5
        and (
            canary["step42_compatible_classification_sha256"]
            == canary["step42_expected_classification_sha256"]
        ).all(),
        "independent Step42/Step43 P2 runs match for five canaries",
    )
    output_hashes_ok = True
    for row in inventory.itertuples():
        path = Path(row.output_path)
        output_hashes_ok &= path.is_file()
        if path.is_file():
            output_hashes_ok &= sha256_file(path) == row.file_sha256
    check("Q16_ALL_OUTPUT_SHA256", output_hashes_ok)
    legacy_unchanged = True
    for row in legacy.itertuples():
        path = Path(row.source_path)
        legacy_unchanged &= path.is_file()
        if path.is_file():
            legacy_unchanged &= sha256_file(path) == row.sha256
    check("Q17_NO_LEGACY_OVERWRITE", legacy_unchanged)
    real_failures = failed[
        ~failed["status"].eq("NO_FAILED_OR_SKIPPED_RUNS")
    ]
    prohibited_names = [
        "population",
        "moran",
        "jaccard_wui_p_wui_s",
        "county",
        "overleaf",
    ]
    prohibited_absent = not any(
        any(token in path.name.lower() for token in prohibited_names)
        for path in output.rglob("*")
        if path.is_file()
    )
    check(
        "Q18_NO_FAILURE_AND_NO_DOWNSTREAM",
        real_failures.empty and prohibited_absent,
    )
    return passed, failures, inventory


def write_readme(
    output: Path,
    inventory: pd.DataFrame,
    status: str,
    checks_passed: int,
    checks_failed: int,
) -> None:
    dedup = pd.read_csv(output / "p2_d1_audit_by_state.csv")
    change = pd.read_csv(output / "p2_classification_change_vs_legacy.csv")
    national = change[change["buffer_m"].eq(500)]
    text = f"""# Step43 WUI-P P2 formal classification rebuild

- Final status: `{status}`
- Policy: `P2_ADDRESS_EXACT_RECORD_DEDUP`
- Candidate rasters: {len(inventory)}
- National 500 m units: {inventory[inventory['buffer_m'].eq(500)]['state'].nunique()}
- Five-state sensitivity buffers: 100–1000 m
- Final QC: {checks_passed} passed, {checks_failed} failed

## Frozen method

WUI-P includes address points only. D1 uses the exact Step42 dual pandas
row-hash implementation over all available raw property fields plus exact
longitude/latitude. Nulls become empty strings and values use `astype(str)`;
there is no address normalization, coordinate rounding, cell deduplication, or
proximity deduplication. Source path and original row position determine the
stable keep-first order but are not D1 equality fields. D2–D7 are retained.

The density gate uses the unchanged 30 m EPSG:5070 grid,
`disk(round(buffer/30))`, and strict density >6.17 points/km². Because P2 is a
deletion-only subset of each same-buffer P0 input, the unchanged legacy
vegetation/distance class is retained exactly where the recomputed P2 density
gate passes. This is mathematically equivalent to rerunning the unchanged
vegetation/distance rules and is the Step42 canary-validated production path.

New rasters use uint8 values 0=valid Non-WUI, 1=Intermix, 2=Interface and
255=NoData outside the 2022 Census state pixel-center domain.

## Main totals

- Raw address records audited: {int(dedup['raw_address_records'].sum()):,}
- Valid address records: {int(dedup['valid_address_records'].sum()):,}
- D1 records removed: {int(dedup['d1_records_removed'].sum()):,}
- Retained address records: {int(dedup['retained_records'].sum()):,}
- National 500 m legacy WUI pixels: {int(national['legacy_wui_pixels'].sum()):,}
- National 500 m P2 WUI pixels: {int(national['wui_pixels'].sum()):,}
- National 500 m pixel difference: {int(national['wui_pixel_difference'].sum()):,}
- National 500 m difference area: {float(national['difference_area_km2'].sum()):,.4f} km²

## Scope boundary

No population, area-summary, county metric, WUI-P/WUI-S Jaccard, Global Moran,
Local Moran, WUI-S/WUI-Z, figure, manuscript, or Overleaf update was run.
"""
    atomic_text(output / "README.md", text)


def manifest(output: Path) -> None:
    rows = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name == "sha256_manifest.txt":
            continue
        rows.append(f"{sha256_file(path)}  {path.relative_to(output)}")
    atomic_text(output / "sha256_manifest.txt", "\n".join(rows) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    started = time.monotonic()
    copy_release_files(output)
    passed, failed, inventory = validate_outputs(output)

    change = pd.read_csv(output / "p2_classification_change_vs_legacy.csv")
    national = change[change["buffer_m"].eq(500)]
    write_csv(
        output / "p2_legacy_change_national_500m.csv",
        [
            {
                "buffer_m": 500,
                "units": len(national),
                "legacy_wui_pixels": int(national["legacy_wui_pixels"].sum()),
                "p2_wui_pixels": int(national["wui_pixels"].sum()),
                "wui_pixel_difference": int(
                    national["wui_pixel_difference"].sum()
                ),
                "difference_area_km2": float(
                    national["difference_area_km2"].sum()
                ),
                "non_wui_to_wui": int(national["non_wui_to_wui"].sum()),
                "wui_to_non_wui": int(national["wui_to_non_wui"].sum()),
                "changed_pixels": int(national["changed_pixels"].sum()),
                "intersection": int(national["wui_pixels"].sum()),
                "union": int(national["legacy_wui_pixels"].sum()),
                "jaccard": (
                    float(national["wui_pixels"].sum())
                    / float(national["legacy_wui_pixels"].sum())
                ),
                "common_valid_pixels": int(
                    national["common_valid_pixels"].sum()
                ),
                "interpretation": "P2_vs_legacy_P0_impact_audit_not_WUIP_WUIS_JACCARD",
            }
        ],
    )
    unresolved = [
        {
            "item_id": "U01",
            "status": "DEFERRED_BY_SCOPE",
            "topic": "DOWNSTREAM_RECOMPUTATION",
            "required_action": (
                "Run population, area summary, county metrics, formal WUI-P/WUI-S "
                "Jaccard and Moran only in later authorized steps"
            ),
            "impact": "Step43 classification candidates are not paper values yet",
        },
        {
            "item_id": "U02",
            "status": "NOT_REQUIRED_FOR_P2",
            "topic": "NATIONAL_D2_D7_DIAGNOSTICS",
            "required_action": (
                "D2-D7 remain retained by policy; extend diagnostics nationally "
                "only if separately requested"
            ),
            "impact": "No effect on P2 completion",
        },
    ]
    write_csv(output / "unresolved_items.csv", unresolved)
    status = (
        "P2_CLASSIFICATION_REBUILD_COMPLETE_READY_FOR_DOWNSTREAM"
        if not failed
        else "PARTIAL_REBUILD_NOT_READY_FOR_DOWNSTREAM"
    )
    write_readme(output, inventory, status, len(passed), len(failed))
    qc = [
        "STEP43 FINAL QC",
        f"completed_utc={utc_now()}",
        f"qc_status={'PASS' if not failed else 'FAIL'}",
        f"final_status={status}",
        f"checks_passed={len(passed)}",
        f"checks_failed={len(failed)}",
        "",
        *passed,
        *failed,
        "",
        "PROHIBITED_ACTIONS",
        "research_drive_write=NO",
        "sudo_used=NO",
        "legacy_overwrite=NO",
        "population_recompute=NO",
        "area_summary_recompute=NO",
        "county_metric_recompute=NO",
        "formal_wuip_wuis_jaccard=NO",
        "global_moran=NO",
        "local_moran=NO",
        "manuscript_or_overleaf_edit=NO",
    ]
    atomic_text(output / "STEP43_FINAL_QC.txt", "\n".join(qc) + "\n")
    status_payload = {
        "step": "STEP43_WUIP_P2_FORMAL_CLASSIFICATION_REBUILD",
        "status": status,
        "qc_status": "PASS" if not failed else "FAIL",
        "completed_utc": utc_now(),
        "output_directory": str(output),
        "policy": "P2_ADDRESS_EXACT_RECORD_DEDUP",
        "raster_count": len(inventory),
        "national_500m_units": int(
            inventory[inventory["buffer_m"].eq(500)]["state"].nunique()
        ),
        "dedup_states": int(
            pd.read_csv(output / "p2_d1_audit_by_state.csv")["state"].nunique()
        ),
        "building_records_included": int(
            inventory["building_records_included"].sum()
        ),
        "nodata": 255,
        "checks_passed": len(passed),
        "checks_failed": len(failed),
        "research_drive_write_attempted": False,
        "sudo_used": False,
        "downstream_recomputation_started": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_json(output / "step43_status.json", status_payload)
    pre_manifest = [
        name for name in REQUIRED_ROOT if name != "sha256_manifest.txt"
    ]
    missing = [
        name
        for name in pre_manifest
        if not (output / name).is_file()
        or (output / name).stat().st_size == 0
    ]
    if missing:
        raise RuntimeError(f"Required release files missing/empty: {missing}")
    manifest(output)
    if failed:
        raise RuntimeError(f"Final Step43 QC failed: {failed}")
    print(
        f"[FINALIZE] completed=1/1 percent=100 elapsed={time.monotonic()-started:.1f}s "
        f"ETA=0 status={status} output={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
