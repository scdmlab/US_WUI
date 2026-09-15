#!/usr/bin/env python3
# Copyright (C) 2026 WUI Mapping Project Authors
# SPDX-License-Identifier: GPL-3.0-only
# Repository version: filesystem paths are loaded through repo_config.py.
# Copy config/paths.example.json to config/paths.json before a new run.
# Raw input data and large intermediate files are stored outside GitHub.

"""STEP51C: formal Moran and Appendix rebuild for Ketchpaw >50% WUI-Z."""
from __future__ import annotations

from repo_config import portable_path

import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(portable_path("project"))
STEP51 = ROOT / "step51_wuiz_ketchpaw_gt50_20260731T215644Z"
METRICS = STEP51 / "downstream_metrics"
PAIRWISE = STEP51 / "pairwise"
OLD48 = ROOT / "step48_patch75_five_state_downstream_20260729T202500Z"
BASE_PATH = ROOT / "scripts/50b_patch75_national49_moran_appendix.py"
FIVE = {"CA", "CO", "FL", "PA", "TX"}
PROTOCOL = (
    "PATCH_VEGETATION_FRACTION_GT_75_PERCENT;PATCH_AREA_GE_5_KM2;"
    "INTERMIX_LOCAL_V_GT_50_PERCENT;"
    "INTERFACE_LOCAL_V_LE_50_PERCENT_AND_DISTANCE_LE_2400M"
)


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    destination.symlink_to(source.resolve())


def prepare_compatibility_inputs() -> tuple[Path, Path, Path]:
    compat = STEP51 / "compatibility_inputs"
    metrics = compat / "metrics"
    pairwise = compat / "pairwise"
    focus = compat / "focus"
    tables = focus / "step48b_moran_figures_tables/tables"
    tables.mkdir(parents=True, exist_ok=True)
    link(
        METRICS / "ketchpaw_gt50_national49_area_147.csv",
        metrics / "patch75_national49_area_147.csv",
    )
    link(
        METRICS / "ketchpaw_gt50_national49_population_147.csv",
        metrics / "patch75_national49_population_147.csv",
    )
    link(
        METRICS / "county_metrics/ketchpaw_gt50_national49_county_metrics.csv",
        metrics / "county_metrics/patch75_national49_county_metrics.csv",
    )
    link(
        PAIRWISE / "ketchpaw_gt50_national49_500m_pairwise_147.csv",
        pairwise / "patch75_national49_500m_pairwise.csv",
    )

    old_tables = OLD48 / "step48b_moran_figures_tables/tables"
    new_area = pd.read_csv(METRICS / "ketchpaw_gt50_national49_area_147.csv")
    new_population = pd.read_csv(
        METRICS / "ketchpaw_gt50_national49_population_147.csv"
    )
    sensitivity = pd.read_csv(
        PAIRWISE / "ketchpaw_gt50_five_state_sensitivity_pairwise_150.csv"
    )

    a1 = pd.read_csv(old_tables / "Appendix_A1_patch75_area_candidate.csv")
    a1 = pd.concat([
        a1[a1.method.ne("WUI-Z")],
        new_area[new_area.method.eq("WUI-Z") & new_area.state.isin(FIVE)],
    ], ignore_index=True, sort=False).sort_values(["state", "method", "buffer_m"])
    a1.to_csv(tables / "Appendix_A1_patch75_area_candidate.csv", index=False)
    shutil.copy2(
        old_tables / "Appendix_A2_patch75_structure_counts_candidate.csv",
        tables / "Appendix_A2_patch75_structure_counts_candidate.csv",
    )
    a3 = pd.read_csv(old_tables / "Appendix_A3_patch75_population_candidate.csv")
    a3 = pd.concat([
        a3[a3.method.ne("WUI-Z")],
        new_population[
            new_population.method.eq("WUI-Z") & new_population.state.isin(FIVE)
        ],
    ], ignore_index=True, sort=False).sort_values(["state", "method", "buffer_m"])
    a3.to_csv(tables / "Appendix_A3_patch75_population_candidate.csv", index=False)
    # A4 is initially copied and is replaced with new WUI-Z Moran rows after
    # the national formal run finishes.
    shutil.copy2(
        old_tables / "Appendix_A4_patch75_global_moran_candidate.csv",
        tables / "Appendix_A4_patch75_global_moran_candidate.csv",
    )
    sensitivity[[
        "state", "radius_m", "method_pair", "intersection_pixels",
        "intersection_area_km2",
    ]].to_csv(tables / "Appendix_A5_patch75_intersection_candidate.csv", index=False)
    sensitivity[[
        "state", "radius_m", "method_pair", "common_valid_pixels",
        "union_pixels", "jaccard",
    ]].to_csv(tables / "Appendix_A6_patch75_jaccard_candidate.csv", index=False)
    return metrics, pairwise, focus


def postprocess(base, output: Path, focus: Path) -> None:
    tables = output / "appendix_tables"
    old_a4 = pd.read_csv(
        OLD48 / "step48b_moran_figures_tables/tables/"
        "Appendix_A4_patch75_global_moran_candidate.csv"
    )
    current_global = pd.read_csv(
        output / "moran/patch75_national49_global_moran_run1.csv"
    )
    focus_a4 = pd.concat([
        old_a4[old_a4.method.ne("WUI-Z")],
        current_global[
            current_global.method.eq("WUI-Z") & current_global.state.isin(FIVE)
        ],
    ], ignore_index=True, sort=False).sort_values(
        ["state", "method", "buffer_m", "variable"]
    )
    focus_a4_path = (
        focus / "step48b_moran_figures_tables/tables/"
        "Appendix_A4_patch75_global_moran_candidate.csv"
    )
    focus_a4.to_csv(focus_a4_path, index=False)
    national_a4 = current_global.copy()
    focus_panel = focus_a4.copy()
    focus_panel.insert(0, "analysis_panel", "five_state_radius_sensitivity")
    national_panel = national_a4.copy()
    national_panel.insert(0, "analysis_panel", "national49_standardized_500m")
    combined_a4 = pd.concat([focus_panel, national_panel], ignore_index=True, sort=False)
    combined_a4.to_csv(
        tables / "Appendix_A4_patch75_full_design_candidate.csv", index=False
    )

    status_path = output / "step50b_status.json"
    status = json.loads(status_path.read_text())
    status["step"] = "STEP51C_WUIZ_GT50_FORMAL_MORAN_APPENDIX"
    status["status"] = "KETCHPAW_GT50_NATIONAL49_DOWNSTREAM_COMPLETE"
    status["strict_patch_protocol"] = PROTOCOL
    status["parent_wuiz_run"] = str(STEP51)
    status["parent_metrics_run"] = str(METRICS)
    status["pairwise_run"] = str(PAIRWISE)
    status["five_state_a4_z_rows_replaced"] = 5
    status["full_design_rows"] = {
        f"A{number}": len(pd.read_csv(
            tables / f"Appendix_A{number}_patch75_full_design_candidate.csv"
        )) for number in range(1, 7)
    }
    base.atomic_text(status_path, json.dumps(status, indent=2) + "\n")
    qc_lines = [
        "STEP51C KETCHPAW GT50 NATIONAL49 FINAL QC",
        f"status={status['status']}",
        f"combinations={status['combination_count']}",
        f"eligible={status['eligible_count']}",
        f"excluded_or_undefined={status['excluded_or_undefined_count']}",
        f"global_two_run_exact={status['global_two_run_exact']}",
        f"local_two_run_exact={status['local_two_run_exact']}",
        f"local_array_hash_two_run_exact={status['local_array_hash_two_run_exact']}",
        f"full_design_rows={status['full_design_rows']}",
        "frozen_weights_rebuilt=False",
        "environment_modified=False",
    ]
    base.atomic_text(output / "STEP51C_FINAL_QC.txt", "\n".join(qc_lines) + "\n")
    count, errors = base.write_manifest(output)
    status["sha256_manifest_entries"] = count
    status["sha256_manifest_errors"] = errors
    if errors:
        raise RuntimeError(f"Step51C manifest failures: {errors}")
    base.atomic_text(status_path, json.dumps(status, indent=2) + "\n")
    base.write_manifest(output)


def main() -> None:
    output = STEP51 / "moran_appendix"
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite existing output: {output}")
    metrics, pairwise, focus = prepare_compatibility_inputs()
    base = load(BASE_PATH, "step51c_base")
    base.STEP48 = focus
    base.STEP49 = pairwise
    saved = sys.argv[:]
    try:
        sys.argv = [str(BASE_PATH), "--metrics-dir", str(metrics), "--output", str(output)]
        base.main()
    finally:
        sys.argv = saved
    postprocess(base, output, focus)


if __name__ == "__main__":
    main()
