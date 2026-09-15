#!/usr/bin/env python3
# Copyright (C) 2026 WUI Mapping Project Authors
# SPDX-License-Identifier: GPL-3.0-only
# Repository version: filesystem paths are loaded through repo_config.py.
# Copy config/paths.example.json to config/paths.json before a new run.
# Raw input data and large intermediate files are stored outside GitHub.

"""Finalize Step42 address policy audit after all five checkpoints exist.

This program performs no raster classification.  It consolidates the completed
read-only audits, applies the explicit Step42 decision rules, writes the
rebuild-scope table and human-readable reports, and validates the deliverable.
"""

from __future__ import annotations

from repo_config import portable_path

import argparse
import csv
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT = Path(portable_path("project"))
STEP41 = PROJECT / "step41_wuip_source_nodata_audit_20260727T054819Z"
STATES = ["CA", "CO", "FL", "PA", "TX"]
SCENARIOS = [
    "P0_LEGACY",
    "P1_ADDRESS_RAW",
    "P2_ADDRESS_EXACT_RECORD_DEDUP",
    "P3_ADDRESS_NORMALIZED_IDENTITY_DEDUP",
    "P4_ADDRESS_COORDINATE_DEDUP_SENSITIVITY",
]
REQUIRED = [
    "README.md",
    "STEP42_FINAL_QC.txt",
    "step42_status.json",
    "wui_p_manuscript_method_crosswalk.csv",
    "wui_p_source_policy_all49.csv",
    "wui_p_source_files_all49.csv",
    "wui_p_duplicate_identity_summary.csv",
    "wui_p_duplicate_field_availability.csv",
    "wui_p_cross_file_duplicate_summary.csv",
    "wui_p_colocated_distinct_address_examples.csv",
    "wui_p_address_policy_500m_impact.csv",
    "recommended_wui_p_policy.md",
    "rebuild_scope_by_state.csv",
    "unresolved_items.csv",
    "sha256_manifest.txt",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def progress(step: int, total: int, label: str, started: float) -> None:
    elapsed = time.monotonic() - started
    done = step / total
    eta = elapsed * (1 - done) / done if done else 0
    print(
        f"[FINALIZE] step={step}/{total} state=ALL49 scenario={label} "
        f"completed={step}/{total} percent={100*done:.2f}% "
        f"elapsed={elapsed:.1f}s ETA={eta:.1f}s",
        flush=True,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fmt_int(value: Any) -> str:
    return f"{int(value):,}"


def fmt_float(value: Any, decimals: int = 4) -> str:
    return f"{float(value):,.{decimals}f}"


def checkpoint_payloads(output: Path) -> list[dict[str, Any]]:
    payloads = []
    for state in STATES:
        path = output / "checkpoints" / f"{state}_step42.json"
        if not path.is_file():
            raise RuntimeError(f"Missing completed five-state checkpoint: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("state") != state:
            raise RuntimeError(f"Checkpoint state mismatch: {path}")
        payloads.append(payload)
    return payloads


def repair_and_consolidate(output: Path, payloads: list[dict[str, Any]]) -> None:
    """Consolidate checkpoints and correct the pre-patch blank-SHA diagnostic.

    PA was completed immediately before the file-SHA grouping code gained its
    explicit nonempty-SHA gate.  Identity and classification results were not
    affected.  Recomputing the file diagnostic directly from the checkpoint is
    deterministic and avoids rerunning PA.
    """
    identity: list[dict[str, Any]] = []
    fields: list[dict[str, Any]] = []
    cross: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    impacts: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for payload in payloads:
        identity.extend(payload["identity_rows"])
        fields.extend(payload["field_rows"])
        examples.extend(payload["examples"])
        p1_inside = next(
            row["input_records"]
            for row in payload["impact_rows"]
            if row["scenario"] == "P1_ADDRESS_RAW"
        )
        valid_before_grid = int(
            payload["parse_stats"]["current_formal_address_point_count"]
        )
        for row in payload["impact_rows"]:
            enriched = dict(row)
            enriched.update(
                {
                    "input_record_domain": (
                        "VALID_SCENARIO_POINTS_INSIDE_FORMAL_RASTER_GRID"
                    ),
                    "raw_address_records_before_geometry_filter": int(
                        payload["parse_stats"]["raw_source_record_count"]
                    ),
                    "valid_address_records_before_grid": valid_before_grid,
                    "P1_address_points_outside_grid": valid_before_grid
                    - int(p1_inside),
                }
            )
            impacts.append(enriched)
        sources.extend(payload["source_records"])
        state_cross = [
            row
            for row in payload["cross_rows"]
            if row["metric"] != "BYTE_IDENTICAL_SOURCE_FILES"
        ]
        source_frame = pd.DataFrame(payload["source_records"])
        hashed = source_frame[
            source_frame["sha256"].fillna("").astype(str).str.strip().ne("")
        ]
        repeated = (
            hashed.groupby("sha256", dropna=False).filter(lambda group: len(group) > 1)
            if len(hashed)
            else hashed
        )
        state_cross.append(
            {
                "state": payload["state"],
                "metric": "BYTE_IDENTICAL_SOURCE_FILES",
                "group_count": int(repeated["sha256"].nunique()) if len(repeated) else 0,
                "records": (
                    int(repeated["retained_current_formal_count"].sum())
                    if len(repeated)
                    else 0
                ),
                "extra_records": "FILE_LEVEL_DIAGNOSTIC",
                "evidence_level": (
                    "SHA256_NONEMPTY_ONLY; hashes computed for locked Texas files "
                    "or same-size collision candidates"
                ),
                "status": "COMPLETED",
            }
        )
        cross.extend(state_cross)
    write_csv(output / "wui_p_duplicate_identity_summary.csv", identity)
    write_csv(output / "wui_p_duplicate_field_availability.csv", fields)
    write_csv(output / "wui_p_cross_file_duplicate_summary.csv", cross)
    write_csv(output / "wui_p_colocated_distinct_address_examples.csv", examples)
    write_csv(output / "wui_p_address_policy_500m_impact.csv", impacts)
    write_csv(output / "five_state_address_source_sha256.csv", sources)

    # Enrich the all-49 file inventory with direct raw-read counts for the five
    # states.  This is essential for Texas, where "raw source records",
    # "valid points", and the Step19C inside-grid formal count are deliberately
    # different quantities.
    source_files = pd.read_csv(
        output / "wui_p_source_files_all49.csv",
        keep_default_na=False,
        dtype={"STATEFP": str},
    )
    direct = {
        (row["state"], row["basename"]): row
        for row in sources
    }
    audited_raw: list[Any] = []
    valid_before_grid: list[Any] = []
    retained_formal: list[Any] = []
    count_evidence: list[str] = []
    for row in source_files.to_dict("records"):
        audit = direct.get((row["state"], row["basename"]))
        if audit is not None and row["source_type"] == "address":
            # Replace metadata-derived raw count with the direct OGR feature
            # count.  Basenames are unique within each audited state.
            row_raw = int(audit["feature_count"])
            audited_raw.append(row_raw)
            valid_before_grid.append(int(audit["valid_point_count_before_grid"]))
            retained_formal.append(int(audit["retained_current_formal_count"]))
            count_evidence.append("DIRECT_RAW_READ_STEP42")
        else:
            row_raw = int(row["raw_feature_count"])
            audited_raw.append("")
            valid_before_grid.append("")
            retained_formal.append("")
            count_evidence.append("STEP41_PRODUCER_METADATA")
        row["raw_feature_count"] = row_raw
    source_files["raw_feature_count"] = [
        (
            int(direct[(row.state, row.basename)]["feature_count"])
            if (row.state, row.basename) in direct and row.source_type == "address"
            else int(row.raw_feature_count)
        )
        for row in source_files.itertuples()
    ]
    source_files["audited_raw_feature_count"] = audited_raw
    source_files["valid_point_count_before_grid_if_audited"] = valid_before_grid
    source_files["retained_formal_count_if_audited"] = retained_formal
    source_files["record_count_evidence"] = count_evidence
    source_files.to_csv(
        output / "wui_p_source_files_all49.csv", index=False, lineterminator="\n"
    )

    policy = pd.read_csv(
        output / "wui_p_source_policy_all49.csv",
        keep_default_na=False,
        dtype={"STATEFP": str},
    )
    included = source_files[
        source_files["current_formal_wuip_included"]
        .astype(str)
        .str.lower()
        .eq("true")
    ]
    raw_rollup = (
        included.groupby(["state", "source_type"])["raw_feature_count"]
        .sum()
        .to_dict()
    )
    direct_valid_rollup = {
        payload["state"]: sum(
            int(record["valid_point_count_before_grid"])
            for record in payload["source_records"]
        )
        for payload in payloads
    }
    parse_by_state = {payload["state"]: payload["parse_stats"] for payload in payloads}
    raw_address: list[int] = []
    valid_address: list[Any] = []
    formal_address: list[int] = []
    raw_total: list[int] = []
    scope: list[str] = []
    notes: list[str] = []
    for row in policy.to_dict("records"):
        state = row["state"]
        address_raw = int(raw_rollup.get((state, "address"), 0))
        building_raw = int(raw_rollup.get((state, "building"), 0))
        raw_address.append(address_raw)
        raw_total.append(address_raw + building_raw)
        formal_address.append(int(row["current_formal_point_feature_count"]))
        if state in parse_by_state:
            valid_address.append(int(direct_valid_rollup[state]))
            scope.append("DIRECT_RAW_VALID_AND_FORMAL_COUNTS_AVAILABLE")
        else:
            valid_address.append("")
            scope.append("RAW_FILE_COUNT_AND_FORMAL_FEATURE_COUNT_AVAILABLE")
        row_notes = str(row.get("notes", "")).strip()
        notes.append(
            (row_notes + "; " if row_notes else "")
            + "raw/address-valid/formal-grid counts are separate fields"
        )
    policy["address_record_count"] = raw_address
    policy["raw_address_record_count_before_geometry_grid"] = raw_address
    policy["valid_address_point_count_before_grid_if_audited"] = valid_address
    policy["formal_point_feature_count"] = policy[
        "current_formal_point_feature_count"
    ].astype(int)
    policy["raw_record_count_current_sources"] = raw_total
    policy["record_count_scope"] = scope
    policy["notes"] = notes
    policy.to_csv(
        output / "wui_p_source_policy_all49.csv", index=False, lineterminator="\n"
    )


def build_rebuild_scope(output: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    source = pd.read_csv(output / "wui_p_source_policy_all49.csv", dtype={"STATEFP": str})
    identity = pd.read_csv(output / "wui_p_duplicate_identity_summary.csv")
    d1 = identity[identity["duplicate_type"].eq("D1")].set_index("state")
    rows: list[dict[str, Any]] = []
    for record in source.sort_values("STATEFP", key=lambda s: s.astype(int)).to_dict("records"):
        state = record["state"]
        mixed = record["current_policy"] == "MIXED_ADDRESS_BUILDING"
        audited = state in STATES
        d1_extra = int(d1.loc[state, "duplicate_extra_records"]) if audited else ""
        rows.append(
            {
                "STATEFP": str(record["STATEFP"]).zfill(2),
                "state": state,
                "state_name": record["state_name"],
                "legacy_source_policy": record["current_policy"],
                "address_only_source_change_required": "YES" if mixed else "NO",
                "five_state_identity_audit_completed": "YES" if audited else "NO",
                "D1_exact_extra_records_if_audited": d1_extra,
                "recommended_formal_policy": "P2_ADDRESS_EXACT_RECORD_DEDUP",
                "formal_policy_changed_from_legacy": "YES",
                "rebuild_under_recommended_P2": "YES",
                "rebuild_reason": (
                    "REMOVE_BUILDING_AND_APPLY_D1_EXACT_RECORD_DEDUP"
                    if mixed
                    else "APPLY_UNIFORM_D1_EXACT_RECORD_DEDUP_POLICY"
                ),
                "P1_address_only_minimum_rebuild_scope": "YES" if mixed else "NO",
                "identity_scope_note": (
                    "D1-D7_AND_500M_IMPACT_COMPLETED"
                    if audited
                    else "FILE_INVENTORY_ONLY; RUN_D1_D7_WHEN_REBUILDING"
                ),
                "do_not_delete_original_building_files": "YES",
            }
        )
    write_csv(output / "rebuild_scope_by_state.csv", rows)
    result = pd.DataFrame(rows)
    counts = {
        "all49": len(result),
        "mixed": int(result["address_only_source_change_required"].eq("YES").sum()),
        "address_only": int(result["address_only_source_change_required"].eq("NO").sum()),
        "p2_rebuild": int(result["rebuild_under_recommended_P2"].eq("YES").sum()),
        "p1_minimum_rebuild": int(
            result["P1_address_only_minimum_rebuild_scope"].eq("YES").sum()
        ),
    }
    return result, counts


def aggregate_metrics(output: Path) -> dict[str, Any]:
    identity = pd.read_csv(output / "wui_p_duplicate_identity_summary.csv")
    impact = pd.read_csv(output / "wui_p_address_policy_500m_impact.csv")
    field = pd.read_csv(output / "wui_p_duplicate_field_availability.csv")
    by_type: dict[str, dict[str, int]] = {}
    for duplicate_type, group in identity.groupby("duplicate_type"):
        by_type[duplicate_type] = {
            "group_count": int(group["group_count"].sum()),
            "records_in_groups": int(group["records_in_groups"].sum()),
        }
        numeric_extra = pd.to_numeric(group["duplicate_extra_records"], errors="coerce")
        by_type[duplicate_type]["duplicate_extra_records"] = int(numeric_extra.sum())
    scenarios: dict[str, dict[str, Any]] = {}
    for scenario, group in impact.groupby("scenario"):
        scenarios[scenario] = {
            "states": int(group["state"].nunique()),
            "input_records": int(group["input_records"].sum()),
            "deleted_records_from_P1": int(group["deleted_records_from_P1"].sum()),
            "retained_records": int(group["retained_records"].sum()),
            "wui_pixels": int(group["wui_pixels"].sum()),
            "wui_area_km2": float(group["wui_area_km2"].sum()),
            "wui_pixel_change_vs_P1": int(group["wui_pixel_change_vs_P1"].sum()),
            "wui_area_change_vs_P1_km2": float(
                group["wui_area_change_vs_P1_km2"].sum()
            ),
            "min_jaccard_with_P1": float(group["jaccard_with_P1"].min()),
        }
    core = field[field["field"].eq("CORE_IDENTITY_NUMBER_AND_STREET")]
    return {
        "by_type": by_type,
        "scenarios": scenarios,
        "core_identity_min_availability_pct": float(core["availability_pct"].min()),
        "core_identity_max_availability_pct": float(core["availability_pct"].max()),
    }


def write_policy(
    output: Path, counts: dict[str, int], metrics: dict[str, Any]
) -> None:
    d = metrics["by_type"]
    s = metrics["scenarios"]
    p2 = s["P2_ADDRESS_EXACT_RECORD_DEDUP"]
    p3 = s["P3_ADDRESS_NORMALIZED_IDENTITY_DEDUP"]
    p4 = s["P4_ADDRESS_COORDINATE_DEDUP_SENSITIVITY"]
    content = f"""# Recommended formal WUI-P point policy

## Decision

Use **P2_ADDRESS_EXACT_RECORD_DEDUP** for the new formal WUI-P input:

1. select address-point files only;
2. retain every valid address record by default;
3. remove only D1 records whose complete available raw attributes and geometry are exactly identical, retaining one deterministic representative;
4. do not merge WUI-S building footprints or centroids into WUI-P;
5. preserve D4, D5, D6 and D7; exact coordinate, common 30 m cell, or spatial proximity alone is never a deletion rule;
6. retain the original address and building source files unchanged.

P0 is rejected because it conflicts with the manuscript's address-only definition. P4 is rejected because D4 directly demonstrates distinct address identities at common coordinates. P3 is not recommended for the formal rebuild in this round: its D2/D3 normalization component has not received the required record-level human false-merge review.

## Five-state evidence

- D1 exact duplicate extras removed by P2: {fmt_int(d['D1']['duplicate_extra_records'])}.
- D4 colocated-distinct records that coordinate-only deletion could damage: {fmt_int(d['D4']['records_in_groups'])} records in {fmt_int(d['D4']['group_count'])} coordinate groups.
- P2 versus P1 classification change: {fmt_int(p2['wui_pixel_change_vs_P1'])} WUI pixels ({fmt_float(p2['wui_area_change_vs_P1_km2'])} km²); minimum state Jaccard with P1 = {p2['min_jaccard_with_P1']:.8f}.
- P3 versus P1 classification change: {fmt_int(p3['wui_pixel_change_vs_P1'])} WUI pixels ({fmt_float(p3['wui_area_change_vs_P1_km2'])} km²); minimum state Jaccard with P1 = {p3['min_jaccard_with_P1']:.8f}.
- P4 versus P1 classification change: {fmt_int(p4['wui_pixel_change_vs_P1'])} WUI pixels ({fmt_float(p4['wui_area_change_vs_P1_km2'])} km²); minimum state Jaccard with P1 = {p4['min_jaccard_with_P1']:.8f}.

These are temporary 500 m policy-stability results, not formal revised paper estimates.

## Rebuild scope

- Source-type correction alone (P1) has a minimum scope of {counts['p1_minimum_rebuild']} mixed-source units.
- The recommended P2 rule is a new uniform point-identity policy. For a reproducible national product, process all {counts['p2_rebuild']} units through the address-only P2 producer. This includes the {counts['address_only']} legacy address-only units: they need no building removal, but they must pass the same D1 rule and receive a new provenance manifest/classification hash.
- D1-D7 and 500 m classification impacts are complete for CA, CO, FL, PA and TX. The other 44 units must receive the same D1 audit during rebuild; D2-D7 remain audit diagnostics, not deletion rules.

## Explicit non-decisions

P3 can be reconsidered only after a privacy-preserving stratified human review of normalized matches establishes a tolerable false-merge rate. No Step42 result authorizes formal nationwide raster, population, area, county, Jaccard, Moran, or Local Moran computation.
"""
    atomic_text(output / "recommended_wui_p_policy.md", content)


def write_unresolved(output: Path) -> None:
    rows = [
        {
            "item_id": "U01",
            "topic": "P3_NORMALIZED_IDENTITY",
            "status": "OPEN_NOT_BLOCKING_P2",
            "required_action": (
                "If P3 is reconsidered, conduct privacy-preserving stratified "
                "human review of D2/D3 matches and quantify false merges"
            ),
            "owner": "RESEARCHER",
            "impact": "P3 is not authorized; P2 remains implementable",
        },
        {
            "item_id": "U02",
            "topic": "NATIONAL_IDENTITY_DIAGNOSTICS",
            "status": "OPEN_SCHEDULE_WITH_REBUILD",
            "required_action": (
                "Run D1 for the remaining 44 units before/during P2 production; "
                "D2-D7 may be extended as diagnostics when compute permits"
            ),
            "owner": "NEXT_REBUILD_STEP",
            "impact": "Five-state counts must not be extrapolated to other units",
        },
        {
            "item_id": "U03",
            "topic": "STEP41_SOURCE_SPECIFIC_SCENARIO_PROVENANCE",
            "status": "OPEN_LEGACY_DOCUMENTATION_CORRECTION",
            "required_action": (
                "Do not use Step41 S4/S5 as source-policy truth: source identity "
                "was reconstructed from current directory order after the legacy "
                "geometry-only GeoPackage discarded record provenance"
            ),
            "owner": "NEXT_DOCUMENTATION_STEP",
            "impact": (
                "Step41 P0/S0 mixed classification and exact all-record coordinate "
                "audit remain usable; source-specific S4/S5 are diagnostic only"
            ),
        },
        {
            "item_id": "U04",
            "topic": "MANUSCRIPT_METHOD_WORDING",
            "status": "OPEN_EDIT_PROHIBITED_IN_STEP42",
            "required_action": (
                "In a later manuscript-edit step, explicitly state address-only "
                "WUI-P, P2 exact-record rule, and keep MBF/Google building data in WUI-S"
            ),
            "owner": "FUTURE_MANUSCRIPT_STEP",
            "impact": "Current manuscript/actual legacy input mismatch remains disclosed",
        },
        {
            "item_id": "U05",
            "topic": "FORMAL_DOWNSTREAM_REBUILD",
            "status": "NOT_STARTED_BY_DESIGN",
            "required_action": (
                "After policy approval, create immutable address-only P2 manifests "
                "and rebuild formal outputs in a new step"
            ),
            "owner": "FUTURE_REBUILD_STEP",
            "impact": "No formal paper statistic changes in Step42",
        },
    ]
    write_csv(output / "unresolved_items.csv", rows)


def write_readme(
    output: Path,
    counts: dict[str, int],
    metrics: dict[str, Any],
    status: str,
) -> None:
    d = metrics["by_type"]
    s = metrics["scenarios"]
    lines = [
        "# Step42 WUI-P address policy and duplicate-identity audit",
        "",
        f"- Completed UTC: {utc_now()}",
        f"- Final status: `{status}`",
        f"- Formal Step41 input: `{STEP41}`",
        "- Scope: policy audit only; no formal raster or downstream paper metric was rebuilt.",
        "",
        "## Main findings",
        "",
        "The manuscript defines WUI-P from OpenAddresses/address points and assigns "
        "building footprints/centroids to WUI-S. The audited legacy producer instead "
        "mixed address and building inputs outside ten address-only units. This is a "
        "`METHOD_INPUT_MISMATCH`.",
        "",
        f"Across 48 states plus DC, {counts['mixed']} units are mixed address+building "
        f"and {counts['address_only']} are address-only; no unit is building-only or unknown. "
        "These classifications were established state by state from the formal producer "
        "chain and file inventory, not extrapolated from the five sampled states.",
        "",
        "CA, CO, FL, PA and TX were directly read from raw address files with address "
        "attributes. D1-D7 were kept distinct; D4-D7 were never interpreted as deletions.",
        "",
        "Five-state totals:",
        "",
        "| Type | Groups/locations | Records in groups | Extra records if defined |",
        "|---|---:|---:|---:|",
    ]
    for key in ["D1", "D2", "D3", "D4", "D5", "D6", "D7"]:
        row = d[key]
        lines.append(
            f"| {key} | {fmt_int(row['group_count'])} | "
            f"{fmt_int(row['records_in_groups'])} | "
            f"{fmt_int(row['duplicate_extra_records']) if key in {'D1','D2','D3'} else 'not a deletion count'} |"
        )
    lines.extend(
        [
            "",
        "Temporary 500 m policy results:",
        "",
        "| Scenario | Deleted from P1 | WUI pixel change vs P1 | Area change vs P1 (km²) | Minimum state Jaccard with P1 |",
        "|---|---:|---:|---:|---:|",
        ]
    )
    for scenario in SCENARIOS:
        row = s[scenario]
        lines.append(
            f"| {scenario} | {fmt_int(row['deleted_records_from_P1'])} | "
            f"{fmt_int(row['wui_pixel_change_vs_P1'])} | "
            f"{fmt_float(row['wui_area_change_vs_P1_km2'])} | "
            f"{row['min_jaccard_with_P1']:.8f} |"
        )
    lines.extend(
        [
            "",
            "`input_records`, `retained_records`, and `deleted_records_from_P1` in "
            "the 500 m impact table refer to valid scenario points inside the formal "
            "raster grid. Raw, valid-before-grid, and P1-outside-grid counts are retained "
            "in separate columns.",
            "",
            "## Policy conclusion",
            "",
            "Recommend `P2_ADDRESS_EXACT_RECORD_DEDUP`: address-only WUI-P and deletion "
            "only of D1 complete exact-record duplicates. P0 is method-inconsistent; "
            "P4 would destroy documented colocated distinct identities; P3 awaits human "
            "false-merge review.",
            "",
            f"P1 source correction alone requires {counts['mixed']} units. Applying the "
            "recommended P2 rule uniformly changes the production policy for all 49 units, "
            "so the next formal rebuild should process all 49 with immutable address-only "
            "manifests. The remaining 44 units receive D1 audit during that rebuild; their "
            "duplicate scale is not inferred from the five-state results.",
            "",
            "## Critical provenance caveat",
            "",
            "A direct raw-address P1 reconstruction exposed that Step41 S4/S5 source labels "
            "depended on reconstructed current directory order after the legacy GeoPackage "
            "discarded source provenance. Therefore Step41 S4/S5 are diagnostic only. This "
            "does not alter the Step41 P0/S0 total mixed classification or its all-record "
            "coordinate-dedup audit.",
            "",
            "## Output guide",
            "",
            "- `wui_p_manuscript_method_crosswalk.csv`: manuscript/workflow/code claims versus audited inputs.",
            "- `wui_p_source_policy_all49.csv`, `wui_p_source_files_all49.csv`: state and file source inventory.",
            "- `wui_p_duplicate_identity_summary.csv`: D1-D7 counts for the five states.",
            "- `wui_p_duplicate_field_availability.csv`: identity-field completeness.",
            "- `wui_p_cross_file_duplicate_summary.csv`: cross-file and byte-identity evidence.",
            "- `wui_p_colocated_distinct_address_examples.csv`: privacy-safe D4 examples.",
            "- `wui_p_address_policy_500m_impact.csv`: P0-P4 temporary 500 m impacts.",
            "- `recommended_wui_p_policy.md`: formal policy recommendation and boundaries.",
            "- `rebuild_scope_by_state.csv`: all-49 rebuild reasons and scope.",
            "- `unresolved_items.csv`: work deliberately deferred or requiring review.",
            "- `STEP42_FINAL_QC.txt`, `step42_status.json`, `sha256_manifest.txt`: validation and provenance.",
            "",
            "Original address/building files, Step41, formal rasters, and manuscript files "
            "were not modified. No research-drive write was attempted.",
        ]
    )
    atomic_text(output / "README.md", "\n".join(lines) + "\n")


def validate(
    output: Path, counts: dict[str, int], status: str
) -> tuple[list[str], list[str]]:
    passed: list[str] = []
    failed: list[str] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        text = f"{name}: {'PASS' if condition else 'FAIL'}"
        if detail:
            text += f" — {detail}"
        (passed if condition else failed).append(text)

    policy = pd.read_csv(output / "wui_p_source_policy_all49.csv")
    files = pd.read_csv(output / "wui_p_source_files_all49.csv")
    identity = pd.read_csv(output / "wui_p_duplicate_identity_summary.csv")
    fields = pd.read_csv(output / "wui_p_duplicate_field_availability.csv")
    impact = pd.read_csv(output / "wui_p_address_policy_500m_impact.csv")
    examples = pd.read_csv(output / "wui_p_colocated_distinct_address_examples.csv")
    crosswalk = pd.read_csv(output / "wui_p_manuscript_method_crosswalk.csv")
    rebuild = pd.read_csv(output / "rebuild_scope_by_state.csv")

    check("Q01_ALL49_UNIQUE", len(policy) == 49 and policy["state"].nunique() == 49)
    check(
        "Q02_SOURCE_POLICY_DISTRIBUTION",
        counts["mixed"] == 39 and counts["address_only"] == 10,
        f"mixed={counts['mixed']}; address_only={counts['address_only']}",
    )
    check(
        "Q03_NO_UNKNOWN_SOURCE_FILES",
        int(policy["unknown_file_count"].sum()) == 0
        and files["source_type"].isin(["address", "building"]).all(),
    )
    check(
        "Q04_FIVE_STATE_D1_D7",
        len(identity) == 35
        and identity.groupby("state")["duplicate_type"].nunique().eq(7).all(),
    )
    check(
        "Q05_FIVE_STATE_FIELDS",
        len(fields) == 40 and fields.groupby("state")["field"].nunique().eq(8).all(),
    )
    check(
        "Q06_FIVE_STATE_P0_P4",
        len(impact) == 25
        and impact.groupby("state")["scenario"].nunique().eq(5).all(),
    )
    check(
        "Q07_CLASSIFICATION_HASHES",
        impact["classification_sha256"].astype(str).str.fullmatch(r"[0-9a-f]{64}").all(),
    )
    p2_in_grid_deleted = int(
        impact[impact["scenario"].eq("P2_ADDRESS_EXACT_RECORD_DEDUP")][
            "deleted_records_from_P1"
        ].sum()
    )
    d1_all_valid_extra = int(
        identity[identity["duplicate_type"].eq("D1")][
            "duplicate_extra_records"
        ].astype(int).sum()
    )
    check(
        "Q08_P2_IN_GRID_DELETE_NOT_GREATER_THAN_ALL_VALID_D1",
        p2_in_grid_deleted <= d1_all_valid_extra,
        f"in_grid={p2_in_grid_deleted}; all_valid_D1={d1_all_valid_extra}",
    )
    check(
        "Q09_D4_PRIVACY_SAFE_EXAMPLES",
        len(examples) >= 5
        and {"coordinate_token", "address_identity_tokens", "privacy_rule"}.issubset(
            examples.columns
        ),
    )
    check(
        "Q10_METHOD_MISMATCH_FLAGGED",
        crosswalk["finding"].astype(str).eq("METHOD_INPUT_MISMATCH").any(),
    )
    check(
        "Q11_P0_NOT_RECOMMENDED",
        "P2_ADDRESS_EXACT_RECORD_DEDUP"
        in (output / "recommended_wui_p_policy.md").read_text(encoding="utf-8")
        and "P0 is rejected"
        in (output / "recommended_wui_p_policy.md").read_text(encoding="utf-8"),
    )
    check(
        "Q12_P4_NOT_RECOMMENDED",
        "P4 is rejected"
        in (output / "recommended_wui_p_policy.md").read_text(encoding="utf-8"),
    )
    check(
        "Q13_REBUILD_SCOPE_ALL49",
        len(rebuild) == 49
        and rebuild["rebuild_under_recommended_P2"].eq("YES").all(),
    )
    check(
        "Q14_STEP41_INPUT_EXISTS_UNCHANGED_BY_DESIGN",
        STEP41.is_dir(),
        "Step42 contains no Step41 write path and run is read-only for research drive",
    )
    check(
        "Q15_ALLOWED_FINAL_STATUS",
        status
        in {
            "ADDRESS_POLICY_READY_FOR_REBUILD",
            "ADDRESS_DEDUP_POLICY_REQUIRES_RESEARCHER_DECISION",
            "BLOCKED_OR_AMBIGUOUS",
        },
    )
    included = files[
        files["current_formal_wuip_included"]
        .astype(str)
        .str.lower()
        .eq("true")
    ]
    rollup = (
        included.groupby(["state", "source_type"])
        .agg(files=("source_path", "size"), records=("raw_feature_count", "sum"))
        .to_dict()
    )
    rollup_ok = True
    for row in policy.itertuples():
        for source_type in ["address", "building"]:
            observed_files = int(
                rollup["files"].get((row.state, source_type), 0)
            )
            observed_records = int(
                rollup["records"].get((row.state, source_type), 0)
            )
            expected_files = int(getattr(row, f"{source_type}_file_count"))
            expected_records = int(getattr(row, f"{source_type}_record_count"))
            rollup_ok &= observed_files == expected_files
            rollup_ok &= observed_records == expected_records
    check(
        "Q16_SOURCE_FILE_TO_STATE_RAW_COUNT_ROLLUP",
        rollup_ok,
        "address/building file counts and raw records reconcile for all 49 units",
    )
    tx = policy[policy["state"].eq("TX")].iloc[0]
    check(
        "Q17_TEXAS_RAW_VALID_FORMAL_COUNT_SEPARATION",
        int(tx["raw_address_record_count_before_geometry_grid"]) == 21_041_613
        and int(tx["valid_address_point_count_before_grid_if_audited"])
        == 21_041_610
        and int(tx["formal_point_feature_count"]) == 20_991_807,
        "raw=21,041,613; valid=21,041,610; formal-inside-grid=20,991,807",
    )
    return passed, failed


def manifest(output: Path) -> None:
    rows = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name == "sha256_manifest.txt":
            continue
        rows.append(f"{sha256_file(path)}  {path.relative_to(output)}")
    atomic_text(output / "sha256_manifest.txt", "\n".join(rows) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    started = time.monotonic()
    total = 8

    payloads = checkpoint_payloads(output)
    repair_and_consolidate(output, payloads)
    progress(1, total, "CONSOLIDATE_CHECKPOINTS", started)

    _, counts = build_rebuild_scope(output)
    progress(2, total, "REBUILD_SCOPE", started)

    metrics = aggregate_metrics(output)
    progress(3, total, "AGGREGATE_EVIDENCE", started)

    status = "ADDRESS_POLICY_READY_FOR_REBUILD"
    write_policy(output, counts, metrics)
    progress(4, total, "POLICY_DECISION", started)

    write_unresolved(output)
    progress(5, total, "UNRESOLVED_ITEMS", started)

    write_readme(output, counts, metrics, status)
    progress(6, total, "README", started)

    passed, failed = validate(output, counts, status)
    qc_status = "PASS" if not failed else "FAIL"
    qc = [
        "STEP42 FINAL QC",
        f"completed_utc={utc_now()}",
        f"qc_status={qc_status}",
        f"final_status={status if not failed else 'BLOCKED_OR_AMBIGUOUS'}",
        f"checks_passed={len(passed)}",
        f"checks_failed={len(failed)}",
        "",
        *passed,
        *failed,
        "",
        "PROHIBITED_ACTIONS",
        "formal_raster_overwrite=NO",
        "raw_address_or_building_delete=NO",
        "research_drive_write_attempted=NO",
        "formal_population_area_county_jaccard_moran_recompute=NO",
        "manuscript_or_overleaf_edit=NO",
        "local_moran=NO",
    ]
    atomic_text(output / "STEP42_FINAL_QC.txt", "\n".join(qc) + "\n")
    final_status = status if not failed else "BLOCKED_OR_AMBIGUOUS"
    status_payload = {
        "step": "STEP42_WUIP_ADDRESS_POLICY_AND_DUPLICATE_IDENTITY_AUDIT",
        "status": final_status,
        "qc_status": qc_status,
        "completed_utc": utc_now(),
        "output_directory": str(output),
        "step41_formal_input": str(STEP41),
        "recommended_policy": "P2_ADDRESS_EXACT_RECORD_DEDUP",
        "manuscript_actual_input_consistency": "METHOD_INPUT_MISMATCH",
        "source_policy_distribution": {
            "mixed_address_building": counts["mixed"],
            "address_only": counts["address_only"],
            "building_only": 0,
            "unknown": 0,
        },
        "five_state_identity_and_500m_audit": STATES,
        "minimum_rebuild_scope_if_P1": counts["p1_minimum_rebuild"],
        "recommended_uniform_P2_rebuild_scope": counts["p2_rebuild"],
        "aggregate_metrics": metrics,
        "prohibited_actions_confirmed_not_performed": True,
        "research_drive_write_attempted": False,
        "qc_checks_passed": len(passed),
        "qc_checks_failed": len(failed),
    }
    atomic_text(
        output / "step42_status.json",
        json.dumps(status_payload, indent=2, ensure_ascii=False) + "\n",
    )
    progress(7, total, "QC_AND_STATUS", started)

    pre_manifest_required = [
        name for name in REQUIRED if name != "sha256_manifest.txt"
    ]
    missing = [
        name for name in pre_manifest_required if not (output / name).is_file()
    ]
    empty = [
        name
        for name in pre_manifest_required
        if (output / name).is_file() and (output / name).stat().st_size == 0
    ]
    if missing or empty:
        raise RuntimeError(f"Required output check failed: missing={missing}; empty={empty}")
    manifest(output)
    if not (output / "sha256_manifest.txt").is_file() or not (
        output / "sha256_manifest.txt"
    ).stat().st_size:
        raise RuntimeError("sha256_manifest.txt was not created")
    progress(8, total, "SHA256_MANIFEST", started)
    if failed:
        raise RuntimeError(f"Final QC failed: {failed}")
    print(
        f"[DONE] status={final_status} output={output} "
        f"elapsed={time.monotonic()-started:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
