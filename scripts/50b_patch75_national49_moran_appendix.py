#!/usr/bin/env python3
# Copyright (C) 2026 WUI Mapping Project Authors
# SPDX-License-Identifier: GPL-3.0-only
# Repository version: filesystem paths are loaded through repo_config.py.
# Copy config/paths.example.json to config/paths.json before a new run.
# Raw input data and large intermediate files are stored outside GitHub.

"""STEP50B: strict-patch national Moran, figures, and Appendix A candidates.

Uses the 196 frozen Step45 county graphs without rebuilding weights. WUI-Z
area proportions use the matching WUI-P area graph because WUI-Z has no
structure-proportion variable. Formal inference follows approved Option A:
only state-method-variable combinations with n >= 30 and nonzero variance.
"""
from __future__ import annotations

from repo_config import portable_path

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from esda import Moran, Moran_Local
from libpysal.weights import W

ROOT = Path(portable_path("project"))
STEP45 = ROOT / "step45_p2_formal_spatial_analysis_20260728T025216Z"
STEP48 = ROOT / "step48_patch75_five_state_downstream_20260729T202500Z"
STEP49 = ROOT / "step49_patch75_national49_500m_20260730T032000Z"
MOD_PATH = ROOT / "scripts/45c_p2_formal_moran_py310_compatibility.py"
SEED = 20260728
PERMUTATIONS = 999
ALPHA = 0.05
EXPECTED_COMBINATIONS = 245


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as src:
        for chunk in iter(lambda: src.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(value, encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    frame.to_csv(tmp, index=False, float_format="%.12f")
    os.replace(tmp, path)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def progress(stage: str, done: int, total: int, started: float, detail: str) -> None:
    elapsed = time.monotonic() - started
    eta = elapsed * max(total - done, 0) / max(done, 1)
    print(
        f"[STEP50B] stage={stage} detail={detail} completed={done}/{total} "
        f"percent={100*done/max(total,1):.2f} elapsed={elapsed/60:.2f}m "
        f"ETA={eta/60:.2f}m",
        flush=True,
    )


def canonical_weights_hash(ids: list[int], neighbors: dict[int, list[int]]) -> str:
    payload = [[int(i), [int(x) for x in sorted(neighbors[i])]] for i in ids]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def load_combinations(metrics: pd.DataFrame, mod) -> list[dict[str, Any]]:
    manifest = pd.read_csv(STEP45 / "p2_spatial_weights_manifest.csv")
    edges = pd.read_csv(STEP45 / "p2_spatial_weights_neighbors.csv")
    names = mod.county_names()
    specs: list[tuple[str, str, str, str, str]] = []
    states = sorted(metrics["state"].unique())
    if len(states) != 49:
        raise RuntimeError(f"Expected 49 state units, found {len(states)}")
    for state in states:
        specs.extend([
            (state, "WUI-P", "p_a", "WUI-P", "area_proportion"),
            (state, "WUI-P", "p_s", "WUI-P", "structure_proportion"),
            (state, "WUI-S", "p_a", "WUI-S", "area_proportion"),
            (state, "WUI-S", "p_s", "WUI-S", "structure_proportion"),
            (state, "WUI-Z", "p_a", "WUI-P", "area_proportion"),
        ])
    combinations: list[dict[str, Any]] = []
    for state, method, variable, weight_method, frozen_variable in specs:
        hit = manifest[
            manifest["state"].eq(state)
            & manifest["method"].eq(weight_method)
            & manifest["variable"].eq(frozen_variable)
        ]
        if len(hit) != 1:
            raise RuntimeError(
                f"Frozen graph record count !=1: {state}/{weight_method}/{frozen_variable}"
            )
        rec = hit.iloc[0]
        ids = [int(x) for x in str(rec["id_order"]).split("|") if x]
        edge_part = edges[
            edges["state"].eq(state)
            & edges["method"].eq(weight_method)
            & edges["variable"].eq(frozen_variable)
        ]
        neighbors = {i: [] for i in ids}
        for row in edge_part.itertuples(index=False):
            neighbors[int(row.fips)].append(int(row.neighbor_fips))
        for key in neighbors:
            neighbors[key] = sorted(neighbors[key])
        digest = canonical_weights_hash(ids, neighbors)
        if digest != str(rec["weights_sha256"]):
            raise RuntimeError(f"Frozen weight hash mismatch: {state}/{method}/{variable}")
        part = metrics[
            metrics["state"].eq(state) & metrics["method"].eq(method)
        ][["GEOID_INT", variable]].copy()
        part = part[np.isfinite(part[variable])].sort_values("GEOID_INT")
        if part["GEOID_INT"].astype(int).tolist() != ids:
            raise RuntimeError(f"FIPS order mismatch: {state}/{method}/{variable}")
        w = W(neighbors, id_order=ids, silence_warnings=True)
        w.transform = "r"
        combinations.append({
            "analysis_scope": "county_within_state_patch75_500m",
            "state": state,
            "method": method,
            "buffer_m": 500,
            "variable": variable,
            "frozen_variable": frozen_variable,
            "weight_method": weight_method,
            "ids": ids,
            "county": [names.get(i, "") for i in ids],
            "neighbors": neighbors,
            "w": w,
            "y": part[variable].to_numpy(dtype=np.float64),
            "islands": np.asarray([not neighbors[i] for i in ids], dtype=bool),
            "weights_sha256": digest,
        })
    if len(combinations) != EXPECTED_COMBINATIONS:
        raise RuntimeError(f"Expected {EXPECTED_COMBINATIONS} combinations")
    return combinations


def eligibility(combinations: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for combo in combinations:
        n = len(combo["ids"])
        variance = float(np.var(combo["y"]))
        s0 = float(combo["w"].s0)
        if n == 1 or s0 == 0:
            status = "UNDEFINED_N1_OR_S0_ZERO"
            reason = f"n={n}; row-standardized S0={s0}"
        elif not math.isfinite(variance) or variance <= 0:
            status = "UNDEFINED_ZERO_VARIANCE"
            reason = f"new strict metric variance={variance}"
        elif n < 30:
            status = "EXCLUDED_N_LT_30"
            reason = f"Option A excludes n={n}<30"
        else:
            status = "ELIGIBLE_FORMAL_N_GE_30"
            reason = f"Option A passes n={n}>=30"
        rows.append({
            "analysis_scope": combo["analysis_scope"],
            "state": combo["state"],
            "method": combo["method"],
            "buffer_m": 500,
            "variable": combo["variable"],
            "frozen_weight_method": combo["weight_method"],
            "frozen_weight_variable": combo["frozen_variable"],
            "feature_count": n,
            "n_islands": int(combo["islands"].sum()),
            "row_standardized_S0": s0,
            "variable_variance": variance,
            "weights_sha256": combo["weights_sha256"],
            "eligibility_status": status,
            "reason": reason,
        })
    return pd.DataFrame(rows)


def global_pass(combinations: list[dict[str, Any]], gate: pd.DataFrame,
                run: str) -> pd.DataFrame:
    lookup = {
        (r.state, r.method, r.variable): r.eligibility_status
        for r in gate.itertuples(index=False)
    }
    rows = []
    started = time.monotonic()
    for idx, combo in enumerate(combinations, 1):
        status = lookup[(combo["state"], combo["method"], combo["variable"])]
        row = {
            "analysis_scope": combo["analysis_scope"],
            "state": combo["state"],
            "method": combo["method"],
            "buffer_m": 500,
            "variable": combo["variable"],
            "frozen_weight_method": combo["weight_method"],
            "frozen_weight_variable": combo["frozen_variable"],
            "n_units": len(combo["ids"]),
            "n_islands": int(combo["islands"].sum()),
            "mean": float(np.mean(combo["y"])),
            "std": float(np.std(combo["y"])),
            "alpha": ALPHA,
            "weights_sha256": combo["weights_sha256"],
            "weight_transformation": "r",
            "permutations": 0,
            "random_seed": np.nan,
            "run": run,
            "eligibility_status": status,
        }
        if status == "ELIGIBLE_FORMAL_N_GE_30":
            mi = Moran(
                combo["y"], combo["w"], transformation="r",
                permutations=0, two_tailed=True,
            )
            significant = bool(mi.p_norm < ALPHA)
            row.update({
                "moran_i": float(mi.I),
                "expected_i": float(mi.EI),
                "variance_norm": float(mi.VI_norm),
                "z_norm": float(mi.z_norm),
                "p_norm": float(mi.p_norm),
                "significant_p_norm": significant,
                "direction": (
                    "POSITIVE_SIGNIFICANT" if significant and mi.I > mi.EI
                    else "NEGATIVE_SIGNIFICANT" if significant
                    else "NOT_SIGNIFICANT"
                ),
                "status": "FORMAL_COMPLETE",
            })
        else:
            row.update({
                "moran_i": np.nan, "expected_i": np.nan,
                "variance_norm": np.nan, "z_norm": np.nan, "p_norm": np.nan,
                "significant_p_norm": np.nan, "direction": "UNDEFINED",
                "status": (
                    "UNDEFINED" if status.startswith("UNDEFINED")
                    else "EXCLUDED_N_LT_30"
                ),
            })
        rows.append(row)
        progress("GLOBAL_" + run, idx, len(combinations), started,
                 f"{combo['state']}/{combo['method']}/{combo['variable']}")
    return pd.DataFrame(rows)


def local_pass(combinations: list[dict[str, Any]], mod, out: Path,
               run: str, save_simulations: bool):
    rows: list[dict[str, Any]] = []
    families: list[dict[str, Any]] = []
    hashes: list[dict[str, Any]] = []
    started = time.monotonic()
    for idx, combo in enumerate(combinations, 1):
        arrays = mod.run_local_library(combo)
        combo_rows, family = mod.local_rows_for_combo(combo, arrays)
        for row in combo_rows:
            row["analysis_scope"] = combo["analysis_scope"]
            row["frozen_weight_method"] = combo["weight_method"]
        family["analysis_scope"] = combo["analysis_scope"]
        family["frozen_weight_method"] = combo["weight_method"]
        rows.extend(combo_rows)
        families.append(family)
        rec = {
            "run": run,
            "state": combo["state"],
            "method": combo["method"],
            "variable": combo["variable"],
            "combination_id": mod.combo_id(
                combo["state"], combo["method"], combo["frozen_variable"]
            ),
        }
        for field in [
            "Is", "sim", "rlisas", "p_sim_directed_legacy",
            "p_sim_two_sided", "z_sim",
        ]:
            rec[field + "_sha256"] = mod.array_sha256(arrays[field])
        rec["simulation_artifact"] = ""
        rec["simulation_artifact_sha256"] = ""
        if save_simulations:
            path, digest = mod.save_simulation_artifact(out, combo, arrays)
            rec["simulation_artifact"] = path
            rec["simulation_artifact_sha256"] = digest
        hashes.append(rec)
        progress("LOCAL_" + run, idx, len(combinations), started,
                 f"{combo['state']}/{combo['method']}/{combo['variable']}")
    return pd.DataFrame(rows), pd.DataFrame(families), pd.DataFrame(hashes)


def excluded_placeholders(combinations: list[dict[str, Any]],
                          gate: pd.DataFrame, mod) -> pd.DataFrame:
    # The imported implementation produces the approved explicit placeholders.
    frame = mod.excluded_local_placeholders(combinations, gate.rename(columns={
        "frozen_weight_variable": "frozen_weight_variable",
    }))
    frame["analysis_scope"] = "county_within_state_patch75_500m"
    return frame


def exact_frame_equal(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    try:
        pd.testing.assert_frame_equal(
            a.reset_index(drop=True), b.reset_index(drop=True),
            check_exact=True, check_dtype=True,
        )
        return True
    except AssertionError:
        return False


def appendix_and_figures(out: Path, metrics_dir: Path, metrics: pd.DataFrame,
                         global_results: pd.DataFrame,
                         local_results: pd.DataFrame) -> dict[str, Any]:
    area = pd.read_csv(metrics_dir / "patch75_national49_area_147.csv")
    population = pd.read_csv(metrics_dir / "patch75_national49_population_147.csv")
    pairwise = pd.read_csv(STEP49 / "patch75_national49_500m_pairwise.csv")
    structures = (
        metrics[metrics["method"].isin(["WUI-P", "WUI-S"])]
        .groupby(["state", "method", "buffer_m"], as_index=False)[
            ["Total_struct", "Intermix_struct", "Interface_struct", "WUI_struct"]
        ].sum()
    )
    tables = out / "appendix_tables"
    atomic_csv(tables / "Appendix_A1_patch75_national49_area.csv", area)
    atomic_csv(tables / "Appendix_A2_patch75_national49_structure_counts.csv", structures)
    atomic_csv(tables / "Appendix_A3_patch75_national49_population.csv", population)
    atomic_csv(tables / "Appendix_A4_patch75_national49_global_moran.csv", global_results)
    atomic_csv(
        tables / "Appendix_A5_patch75_national49_intersection.csv",
        pairwise[[
            "state", "radius_m", "method_pair", "common_valid_pixels",
            "intersection_pixels", "intersection_area_km2",
        ]],
    )
    atomic_csv(
        tables / "Appendix_A6_patch75_national49_jaccard.csv",
        pairwise[[
            "state", "radius_m", "method_pair", "common_valid_pixels",
            "union_pixels", "union_area_km2", "jaccard",
        ]],
    )
    cluster_counts = (
        local_results.groupby(
            ["state", "method", "buffer_m", "variable",
             "formal_fdr_cluster_type"], dropna=False
        ).size().rename("county_count").reset_index()
    )
    atomic_csv(tables / "patch75_national49_local_cluster_counts.csv", cluster_counts)

    # Full-design tables retain the five-state sensitivity panel and add the
    # national standardized-500m panel; scope labels make intentional overlap explicit.
    old_tables = STEP48 / "step48b_moran_figures_tables" / "tables"
    for number, national in [
        ("A1", area), ("A2", structures), ("A3", population),
        ("A4", global_results),
        ("A5", pairwise[[
            "state", "radius_m", "method_pair", "common_valid_pixels",
            "intersection_pixels", "intersection_area_km2",
        ]]),
        ("A6", pairwise[[
            "state", "radius_m", "method_pair", "common_valid_pixels",
            "union_pixels", "union_area_km2", "jaccard",
        ]]),
    ]:
        previous = sorted(old_tables.glob(f"Appendix_{number}_patch75_*candidate.csv"))
        if len(previous) != 1:
            raise RuntimeError(f"Missing unique Step48 Appendix {number} source")
        focus = pd.read_csv(previous[0])
        focus.insert(0, "analysis_panel", "five_state_radius_sensitivity")
        national = national.copy()
        national.insert(0, "analysis_panel", "national49_standardized_500m")
        combined = pd.concat([focus, national], ignore_index=True, sort=False)
        atomic_csv(tables / f"Appendix_{number}_patch75_full_design_candidate.csv", combined)

    pop_summary = (
        population.groupby("method", as_index=False)[
            ["nonwui_population", "intermix_population", "interface_population",
             "wui_population", "total_population"]
        ].sum()
    )
    pop_summary["wui_population_share_pct"] = (
        100 * pop_summary["wui_population"] / pop_summary["total_population"]
    )
    area_summary = (
        area.groupby("method", as_index=False)[
            ["nonwui_area_km2", "intermix_area_km2", "interface_area_km2",
             "wui_area_km2", "valid_area_km2"]
        ].sum()
    )
    atomic_csv(tables / "patch75_national49_population_summary.csv", pop_summary)
    atomic_csv(tables / "patch75_national49_area_summary.csv", area_summary)

    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"WUI-P": "#2b8cbe", "WUI-S": "#41ab5d", "WUI-Z": "#e34a33"}
    for frame, value, ylabel, filename in [
        (area_summary, "wui_area_km2", "WUI area (km²)", "national_wui_area_by_method.png"),
        (pop_summary, "wui_population", "WUI population", "national_wui_population_by_method.png"),
    ]:
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        ax.bar(frame["method"], frame[value],
               color=[colors[x] for x in frame["method"]])
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(figures / filename, dpi=300)
        plt.close(fig)

    state_area = area.pivot(index="state", columns="method", values="wui_area_km2")
    state_area = state_area.sort_values("WUI-P", ascending=False)
    fig, ax = plt.subplots(figsize=(15, 6))
    x = np.arange(len(state_area))
    width = 0.26
    for offset, method in zip([-1, 0, 1], ["WUI-P", "WUI-S", "WUI-Z"]):
        ax.bar(x + offset * width, state_area[method], width,
               label=method, color=colors[method])
    ax.set_xticks(x)
    ax.set_xticklabels(state_area.index, rotation=90)
    ax.set_ylabel("WUI area (km²)")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "state_wui_area_comparison_49units.png", dpi=300)
    plt.close(fig)

    eligible_global = global_results[global_results["status"].eq("FORMAL_COMPLETE")]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for method in ["WUI-P", "WUI-S", "WUI-Z"]:
        values = eligible_global.loc[
            eligible_global["method"].eq(method), "moran_i"
        ].dropna()
        ax.hist(values, bins=16, alpha=0.48, label=method, color=colors[method])
    ax.set_xlabel("Global Moran's I")
    ax.set_ylabel("Eligible combinations")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "global_moran_distribution.png", dpi=300)
    plt.close(fig)

    return {
        "national_area_rows": len(area),
        "national_population_rows": len(population),
        "national_structure_rows": len(structures),
        "pairwise_rows": len(pairwise),
        "local_cluster_count_rows": len(cluster_counts),
        "national_area_summary": area_summary.to_dict("records"),
        "national_population_summary": pop_summary.to_dict("records"),
    }


def write_manifest(out: Path) -> tuple[int, list[str]]:
    files = sorted(
        p for p in out.rglob("*")
        if p.is_file() and p.name != "sha256_manifest.txt"
    )
    atomic_text(
        out / "sha256_manifest.txt",
        "\n".join(f"{sha256(p)}  {p.relative_to(out)}" for p in files) + "\n",
    )
    errors = []
    for line in (out / "sha256_manifest.txt").read_text().splitlines():
        digest, rel = line.split(None, 1)
        path = out / rel.strip()
        if not path.is_file() or sha256(path) != digest:
            errors.append(rel.strip())
    return len(files), errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    metrics_dir = args.metrics_dir.resolve()
    out = (args.output or ROOT / f"step50b_patch75_national49_moran_appendix_{stamp()}").resolve()
    out.mkdir(parents=True, exist_ok=False)
    for folder in ["moran", "local_moran/simulations_run1", "appendix_tables",
                   "figures", "qc"]:
        (out / folder).mkdir(parents=True, exist_ok=True)

    mod = load_module(MOD_PATH, "step50b_compatibility")
    required = {
        "transformation", "permutations", "geoda_quads", "n_jobs",
        "keep_simulations", "seed", "island_weight",
    }
    signature = inspect.signature(Moran_Local)
    if not required.issubset(signature.parameters):
        raise RuntimeError(f"Frozen Moran_Local signature gate failed: {signature}")
    metrics = pd.read_csv(metrics_dir / "county_metrics/patch75_national49_county_metrics.csv")
    if len(metrics) != 9327:
        raise RuntimeError(f"Expected 9,327 county-method rows, got {len(metrics)}")
    combinations = load_combinations(metrics, mod)
    gate = eligibility(combinations)
    atomic_csv(out / "moran/patch75_national49_moran_eligibility.csv", gate)
    eligible_keys = set(
        zip(
            gate.loc[gate["eligibility_status"].eq("ELIGIBLE_FORMAL_N_GE_30"), "state"],
            gate.loc[gate["eligibility_status"].eq("ELIGIBLE_FORMAL_N_GE_30"), "method"],
            gate.loc[gate["eligibility_status"].eq("ELIGIBLE_FORMAL_N_GE_30"), "variable"],
        )
    )
    eligible = [
        c for c in combinations
        if (c["state"], c["method"], c["variable"]) in eligible_keys
    ]
    excluded = [
        c for c in combinations
        if (c["state"], c["method"], c["variable"]) not in eligible_keys
    ]
    if not eligible:
        raise RuntimeError("No eligible Moran combinations")

    smoke, smoke_seconds = mod.smoke_tests(eligible[0])
    atomic_csv(out / "qc/patch75_moran_smoke_tests.csv", smoke)
    if not bool(smoke["passed"].all()):
        raise RuntimeError("Moran smoke test failed")

    global1 = global_pass(combinations, gate, "run1")
    global2 = global_pass(combinations, gate, "run2")
    global_equal = exact_frame_equal(
        global1.drop(columns=["run"]), global2.drop(columns=["run"])
    )
    atomic_csv(out / "moran/patch75_national49_global_moran_run1.csv", global1)
    atomic_csv(out / "moran/patch75_national49_global_moran_run2.csv", global2)
    if not global_equal:
        raise RuntimeError("Global Moran two-run reproduction failed")

    local1, families1, hashes1 = local_pass(eligible, mod, out, "run1", True)
    local2, families2, hashes2 = local_pass(eligible, mod, out, "run2", False)
    local_equal = exact_frame_equal(local1, local2)
    family_equal = exact_frame_equal(families1, families2)
    hash_fields = [
        c for c in hashes1.columns if c.endswith("_sha256")
        and not c.startswith("simulation_artifact")
    ]
    hash_equal = bool(
        (hashes1[hash_fields].to_numpy() == hashes2[hash_fields].to_numpy()).all()
    )
    if not (local_equal and family_equal and hash_equal):
        raise RuntimeError("Local Moran two-run reproduction failed")
    placeholders = excluded_placeholders(excluded, gate, mod)
    local_all = pd.concat([local1, placeholders], ignore_index=True, sort=False)
    atomic_csv(out / "moran/patch75_national49_local_moran.csv", local_all)
    atomic_csv(out / "moran/patch75_national49_local_fdr_family_audit.csv", families1)
    atomic_csv(out / "moran/patch75_national49_local_reproducibility_hashes_run1.csv", hashes1)
    atomic_csv(out / "moran/patch75_national49_local_reproducibility_hashes_run2.csv", hashes2)

    appendix = appendix_and_figures(out, metrics_dir, metrics, global1, local_all)
    eligible_global = global1[global1["status"].eq("FORMAL_COMPLETE")]
    formal_local = local1[~local1["is_island"]]
    cluster_counts = formal_local["formal_fdr_cluster_type"].value_counts().to_dict()
    status = {
        "parent_metrics_run": str(metrics_dir),
        "strict_patch_protocol": (
            "PATCH_VEGETATION_FRACTION_GT_75_PERCENT;"
            "PATCH_AREA_GE_5_KM2;INTERMIX_LOCAL_V_GE_50_PERCENT;"
            "INTERFACE_LOCAL_V_LT_50_PERCENT_AND_DISTANCE_LE_2400M"
        ),
        "environment_protocol": "PYTHON310_ESDA270_FROZEN_COMPATIBILITY_PROTOCOL",
        "environment_modified": False,
        "weight_graphs_rebuilt": False,
        "combination_count": len(combinations),
        "eligible_count": len(eligible),
        "excluded_or_undefined_count": len(excluded),
        "global_formal_count": len(eligible_global),
        "global_positive_significant_count": int(
            eligible_global["direction"].eq("POSITIVE_SIGNIFICANT").sum()
        ),
        "global_not_significant_count": int(
            eligible_global["direction"].eq("NOT_SIGNIFICANT").sum()
        ),
        "local_formal_counties": len(formal_local),
        "local_raw_significant_count": int(formal_local["raw_significant"].sum()),
        "local_fdr_significant_count": int(formal_local["fdr_significant"].sum()),
        "local_fdr_cluster_counts": cluster_counts,
        "smoke_test_all_pass": bool(smoke["passed"].all()),
        "smoke_test_seconds": smoke_seconds,
        "global_two_run_exact": global_equal,
        "local_two_run_exact": local_equal,
        "local_family_two_run_exact": family_equal,
        "local_array_hash_two_run_exact": hash_equal,
        "appendix": appendix,
        "software": {
            "python": sys.version.split()[0],
            "esda": __import__("esda").__version__,
            "libpysal": __import__("libpysal").__version__,
            "Moran_Local_signature": str(signature),
        },
        "completed_utc": utc_now(),
        "status": "PATCH75_NATIONAL49_DOWNSTREAM_COMPLETE",
    }
    atomic_text(out / "step50b_status.json", json.dumps(status, indent=2) + "\n")
    qc_lines = [
        "STEP50B PATCH75 NATIONAL49 FINAL QC",
        f"status={status['status']}",
        f"combinations={len(combinations)}",
        f"eligible={len(eligible)}",
        f"excluded_or_undefined={len(excluded)}",
        f"global_two_run_exact={global_equal}",
        f"local_two_run_exact={local_equal}",
        f"local_array_hash_two_run_exact={hash_equal}",
        f"smoke_test_all_pass={status['smoke_test_all_pass']}",
        "frozen_weights_rebuilt=False",
        "environment_modified=False",
    ]
    atomic_text(out / "STEP50B_FINAL_QC.txt", "\n".join(qc_lines) + "\n")
    manifest_count, manifest_errors = write_manifest(out)
    if manifest_errors:
        raise RuntimeError(f"SHA manifest verification failures: {manifest_errors}")
    status["sha256_manifest_entries"] = manifest_count
    status["sha256_manifest_errors"] = manifest_errors
    atomic_text(out / "step50b_status.json", json.dumps(status, indent=2) + "\n")
    # Rewrite manifest after final status update.
    manifest_count, manifest_errors = write_manifest(out)
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
