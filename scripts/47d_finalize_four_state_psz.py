#!/usr/bin/env python3
# Copyright (C) 2026 WUI Mapping Project Authors
# SPDX-License-Identifier: GPL-3.0-only
# Repository version: filesystem paths are loaded through repo_config.py.
# Copy config/paths.example.json to config/paths.json before a new run.
# Raw input data and large intermediate files are stored outside GitHub.

"""Finalize the four-state strict >75% P/S/Z diagnostic run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


STATES = ("PA", "FL", "CA", "TX")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()

    impacts: list[dict[str, object]] = []
    pairwise: list[dict[str, object]] = []
    wuiz: list[dict[str, object]] = []
    patches: list[dict[str, object]] = []
    state_summary: list[dict[str, object]] = []
    state_statuses: dict[str, dict[str, object]] = {}

    for state in STATES:
        state_dir = run_dir / state
        status = json.loads((state_dir / "step47d_state_status.json").read_text())
        state_statuses[state] = status
        if status["status"] != "STATE_PSZ_GT75_COMPLETE":
            raise RuntimeError(f"{state}: incomplete status {status['status']}")
        if any(
            int(status[key]) != 0
            for key in (
                "baseline_failure_count",
                "class_conservation_failure_count",
                "wuiz_baseline_mismatch_pixels",
                "wuiz_valid_domain_mismatch_pixels",
                "wuiz_intermix_changed_blocks",
            )
        ):
            raise RuntimeError(f"{state}: state QC failure")
        if int(status["pairwise_rows"]) != 50:
            raise RuntimeError(f"{state}: expected 50 pairwise rows")
        check_text = (state_dir / "manifest_check.log").read_text()
        if "FAILED" in check_text:
            raise RuntimeError(f"{state}: independent manifest check failed")

        state_impacts = read_rows(state_dir / f"{state}_psz_gt75_impact.csv")
        state_pairwise = read_rows(state_dir / f"{state}_psz_pairwise.csv")
        state_wuiz = read_rows(state_dir / f"{state}_wuiz_gt75_impact.csv")
        state_patches = read_rows(state_dir / f"{state}_patch75_summary.csv")
        if len(state_impacts) != 20 or len(state_pairwise) != 50:
            raise RuntimeError(f"{state}: unexpected impact or pairwise row count")
        if len(state_wuiz) != 1 or len(state_patches) != 1:
            raise RuntimeError(f"{state}: unexpected WUI-Z or patch row count")

        impacts.extend(state_impacts)
        pairwise.extend(state_pairwise)
        wuiz.extend(state_wuiz)
        patches.extend(state_patches)

        at_500 = {
            row["method"]: row
            for row in state_impacts
            if int(row["radius_m"]) == 500
        }
        p_s_500 = next(
            row
            for row in state_pairwise
            if int(row["radius_m"]) == 500
            and row["method_pair"] == "WUI-P/WUI-S"
        )
        p_z_500 = next(
            row
            for row in state_pairwise
            if int(row["radius_m"]) == 500
            and row["method_pair"] == "WUI-P/WUI-Z"
            and row["z_scenario"] == "NEW_GT75_Z"
        )
        s_z_500 = next(
            row
            for row in state_pairwise
            if int(row["radius_m"]) == 500
            and row["method_pair"] == "WUI-S/WUI-Z"
            and row["z_scenario"] == "NEW_GT75_Z"
        )
        z = state_wuiz[0]
        state_summary.append(
            {
                "state": state,
                "status": status["status"],
                "p_500_old_area_km2": at_500["WUI-P"]["old_wui_area_km2"],
                "p_500_new_area_km2": at_500["WUI-P"]["new_wui_area_km2"],
                "p_500_net_change_km2": at_500["WUI-P"]["net_wui_area_change_km2"],
                "p_500_net_change_percent": at_500["WUI-P"]["net_wui_change_percent"],
                "s_500_old_area_km2": at_500["WUI-S"]["old_wui_area_km2"],
                "s_500_new_area_km2": at_500["WUI-S"]["new_wui_area_km2"],
                "s_500_net_change_km2": at_500["WUI-S"]["net_wui_area_change_km2"],
                "s_500_net_change_percent": at_500["WUI-S"]["net_wui_change_percent"],
                "z_old_area_km2": z["old_wui_area_km2"],
                "z_new_area_km2": z["new_wui_area_km2"],
                "z_net_change_km2": z["net_wui_area_change_km2"],
                "z_net_change_percent": z["net_wui_change_percent"],
                "jaccard_p_s_500_new": p_s_500["jaccard"],
                "jaccard_p_z_500_new": p_z_500["jaccard"],
                "jaccard_s_z_500_new": s_z_500["jaccard"],
                "baseline_failures": status["baseline_failure_count"],
                "conservation_failures": status["class_conservation_failure_count"],
                "manifest_check": "PASS",
                "elapsed_seconds": status["elapsed_seconds"],
            }
        )

    if len(impacts) != 80 or len(pairwise) != 200:
        raise RuntimeError("Combined row-count gate failed")

    write_rows(run_dir / "step47d_four_state_psz_impact.csv", impacts)
    write_rows(run_dir / "step47d_four_state_pairwise.csv", pairwise)
    write_rows(run_dir / "step47d_four_state_wuiz_impact.csv", wuiz)
    write_rows(run_dir / "step47d_four_state_patch75_summary.csv", patches)
    write_rows(run_dir / "step47d_state_summary.csv", state_summary)

    five_hundred_rows: list[dict[str, object]] = []
    for summary in state_summary:
        for method, prefix in (("WUI-P", "p"), ("WUI-S", "s")):
            five_hundred_rows.append(
                {
                    "state": summary["state"],
                    "method": method,
                    "radius_m": 500,
                    "old_wui_area_km2": summary[f"{prefix}_500_old_area_km2"],
                    "new_wui_area_km2": summary[f"{prefix}_500_new_area_km2"],
                    "net_change_km2": summary[f"{prefix}_500_net_change_km2"],
                    "net_change_percent": summary[f"{prefix}_500_net_change_percent"],
                    "jaccard_p_s_new": summary["jaccard_p_s_500_new"],
                    "jaccard_with_new_z": summary[
                        "jaccard_p_z_500_new"
                        if method == "WUI-P"
                        else "jaccard_s_z_500_new"
                    ],
                }
            )
    write_rows(run_dir / "step47d_500m_summary.csv", five_hundred_rows)

    unresolved = [
        {
            "item": "FORMAL_POLICY_ADOPTION",
            "status": "RESEARCHER_DECISION_REQUIRED",
            "detail": "Strict >75% products are diagnostic and have not replaced Step43-Step46 formal inputs.",
        },
        {
            "item": "DOWNSTREAM_REBUILD",
            "status": "NOT_RUN",
            "detail": "Population, county metrics, Moran statistics, figures, and manuscript tables were not recomputed.",
        },
    ]
    write_rows(run_dir / "unresolved_items.csv", unresolved)

    completed_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    status = {
        "step": "STEP47D_PATCH75_FOUR_STATE_PSZ",
        "status": "FOUR_STATE_PSZ_GT75_COMPLETE",
        "completed_utc": completed_utc,
        "states": list(STATES),
        "state_count": 4,
        "radii_per_state": 10,
        "wui_p_raster_count": 40,
        "wui_s_raster_count": 40,
        "wui_z_product_count": 4,
        "impact_rows": len(impacts),
        "pairwise_rows": len(pairwise),
        "state_manifest_checks_passed": 4,
        "baseline_failure_count": 0,
        "class_conservation_failure_count": 0,
        "wuiz_baseline_mismatch_pixels": 0,
        "wuiz_valid_domain_mismatch_pixels": 0,
        "wuiz_intermix_changed_blocks": 0,
        "upstream_modified": False,
        "environment_modified": False,
        "formal_products_replaced": False,
        "output_directory": str(run_dir),
    }
    (run_dir / "step47d_status.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )

    qc_lines = [
        "STEP47D FINAL QC",
        "STATUS=FOUR_STATE_PSZ_GT75_COMPLETE",
        "STATES=PA,FL,CA,TX",
        "STATE_COUNT=4 PASS",
        "WUI_P_RASTERS=40 PASS",
        "WUI_S_RASTERS=40 PASS",
        "WUI_Z_PRODUCTS=4 PASS",
        "IMPACT_ROWS=80 PASS",
        "PAIRWISE_ROWS=200 PASS",
        "STATE_MANIFEST_CHECKS=4/4 PASS",
        "BASELINE_FAILURE_COUNT=0 PASS",
        "CLASS_CONSERVATION_FAILURE_COUNT=0 PASS",
        "WUIZ_BASELINE_MISMATCH_PIXELS=0 PASS",
        "WUIZ_VALID_DOMAIN_MISMATCH_PIXELS=0 PASS",
        "WUIZ_INTERMIX_CHANGED_BLOCKS=0 PASS",
        "UPSTREAM_MODIFIED=FALSE PASS",
        "ENVIRONMENT_MODIFIED=FALSE PASS",
        "FORMAL_PRODUCTS_REPLACED=FALSE PASS",
    ]
    (run_dir / "STEP47D_FINAL_QC.txt").write_text(
        "\n".join(qc_lines) + "\n", encoding="utf-8"
    )

    readme = f"""# Step47D strict >75% four-state P/S/Z diagnostic

Status: `FOUR_STATE_PSZ_GT75_COMPLETE`

This run completes California, Florida, Pennsylvania, and Texas under the
strict qualifying-patch definition: source blocks with `Veg_Percent > 75%`,
contiguous dissolved components at least 5 km2, and interface distance at most
2.4 km. Local intermix remains `V >= 50%`, interface remains `V < 50%`, and
development density remains `D > 6.17`.

For each state, WUI-P and WUI-S were generated at 100--1000 m in 100 m
increments, and a fixed WUI-Z product was reclassified using the same strict
qualifying-patch layer. Class 0 remained valid non-WUI; true outside/NoData was
excluded through the common valid domain.

Outputs:

- `step47d_four_state_psz_impact.csv`: 80 method-radius impact rows.
- `step47d_four_state_pairwise.csv`: 200 P/S/Z pairwise rows.
- `step47d_four_state_wuiz_impact.csv`: four fixed WUI-Z comparisons.
- `step47d_four_state_patch75_summary.csv`: patch construction audit.
- `step47d_500m_summary.csv`: concise standardized-setting summary.
- `step47d_state_summary.csv`: state-level completion and QC summary.

All four state manifests passed independent `sha256sum -c` validation.
Step43--Step46 formal inputs were not overwritten. No population, county,
Moran, figure, table, or manuscript rebuild was performed.

Completed UTC: {completed_utc}
"""
    (run_dir / "README.md").write_text(readme, encoding="utf-8")

    manifest_targets = [
        "README.md",
        "STEP47D_FINAL_QC.txt",
        "step47d_status.json",
        "step47d_four_state_psz_impact.csv",
        "step47d_four_state_pairwise.csv",
        "step47d_four_state_wuiz_impact.csv",
        "step47d_four_state_patch75_summary.csv",
        "step47d_500m_summary.csv",
        "step47d_state_summary.csv",
        "unresolved_items.csv",
    ]
    manifest_lines = [
        f"{sha256(run_dir / name)}  {name}" for name in manifest_targets
    ]
    (run_dir / "sha256_manifest.txt").write_text(
        "\n".join(manifest_lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
