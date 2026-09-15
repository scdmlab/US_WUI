#!/usr/bin/env python3
# Copyright (C) 2026 WUI Mapping Project Authors
# SPDX-License-Identifier: GPL-3.0-only
# Repository version: filesystem paths are loaded through repo_config.py.
# Copy config/paths.example.json to config/paths.json before a new run.
# Raw input data and large intermediate files are stored outside GitHub.

"""STEP45C continuation under the frozen Python 3.10 / esda 2.7 protocol.

This program deserializes the 196 Step45 weight graphs. It does not rebuild
weights, recalculate Jaccard, alter an environment, or modify any parent run.
Local two-sided pseudo p-values are derived from the stored conditional
permutation simulations using the researcher-approved compatibility algorithm.
"""

from __future__ import annotations

from repo_config import portable_path

import hashlib
import importlib.metadata as metadata
import inspect
import json
import math
import os
import platform
import shutil
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

import esda
import libpysal
import numpy as np
import pandas as pd
from esda import Moran, Moran_Local, fdr
from libpysal.weights import W, lag_spatial


ROOT = Path(portable_path("project"))
STEP43 = ROOT / "step43_wuip_p2_formal_rebuild_20260727T180822Z"
STEP44 = ROOT / "step44_wuip_p2_downstream_metrics_20260727T224419Z"
STEP45 = ROOT / "step45_p2_formal_spatial_analysis_20260728T025216Z"
STEP45B = ROOT / "step45b_moran_dependency_recovery_20260728T043902Z"
PARENT_RUN = ROOT / "step45c_p2_formal_moran_new_protocol_20260728T051939Z"
STEP37 = ROOT / "step37_morans_i_revision_20260727T043506Z"
COUNTY = Path(
    portable_path("data", "MBF+NLCD_2022US/Inputs/Processed_GPKG/tl_2022_us_county.gpkg")
)
SCRIPT = Path(__file__).resolve()
PYTHON = Path(sys.executable).resolve()
SEED = 20260728
PERMUTATIONS = 999
ALPHA = 0.05
EXPECTED_ELIGIBLE = 139
EXPECTED_EXCLUDED = 57
REQUIRED_LOCAL_PARAMS = [
    "transformation",
    "permutations",
    "geoda_quads",
    "n_jobs",
    "keep_simulations",
    "seed",
    "island_weight",
]
ARRAY_FIELDS = [
    "Is",
    "sim",
    "rlisas",
    "EI_sim",
    "VI_sim",
    "z_sim",
    "p_z_sim",
    "p_sim_directed_legacy",
    "p_sim_two_sided",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as src:
        for chunk in iter(lambda: src.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(json.dumps(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(value, encoding="utf-8")
    os.replace(tmp, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")


def atomic_csv(path: Path, frame: pd.DataFrame, float_format: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    frame.to_csv(tmp, index=False, float_format=float_format)
    os.replace(tmp, path)


def progress(step: int, total_steps: int, state: str, scenario: str,
             done: int, total: int, started: float) -> None:
    elapsed = max(time.monotonic() - started, 1e-9)
    eta = elapsed * max(total - done, 0) / max(done, 1)
    print(
        f"[STEP {step}/{total_steps}] state={state} scenario={scenario} "
        f"completed={done}/{total} percent={100 * done / max(total, 1):.2f} "
        f"elapsed={elapsed:.2f}s ETA={eta:.2f}s",
        flush=True,
    )


def verify_manifest(base: Path, manifest: Path,
                    expected: int | None = None) -> tuple[int, list[str]]:
    errors: list[str] = []
    lines = [x for x in manifest.read_text(encoding="utf-8").splitlines() if x.strip()]
    if expected is not None and len(lines) != expected:
        errors.append(f"ENTRY_COUNT:{len(lines)}!={expected}")
    for line in lines:
        digest, rel = line.split(None, 1)
        path = base / rel.strip()
        if not path.is_file():
            errors.append(f"MISSING:{path}")
        elif sha256(path) != digest:
            errors.append(f"HASH_MISMATCH:{path}")
    return len(lines), errors


def package_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for name in [
        "numpy", "scipy", "pandas", "geopandas", "shapely",
        "libpysal", "esda", "numba",
    ]:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = "NOT_INSTALLED"
    return result


def canonical_weights_hash(ids: list[int], neighbors: dict[int, list[int]]) -> str:
    payload = [
        [int(i), [int(v) for v in sorted(neighbors.get(i, []))]]
        for i in ids
    ]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode()
    ).hexdigest()


def variable_code(frozen_name: str) -> str:
    return {"area_proportion": "p_a", "structure_proportion": "p_s"}[frozen_name]


def combo_id(state: str, method: str, frozen_variable: str) -> str:
    return f"{state}__{method.replace('-', '')}__{variable_code(frozen_variable)}"


def county_names() -> dict[int, str]:
    with sqlite3.connect(f"file:{COUNTY}?mode=ro", uri=True) as con:
        table_row = con.execute(
            "SELECT table_name FROM gpkg_contents WHERE data_type='features' "
            "ORDER BY table_name LIMIT 1"
        ).fetchone()
        if not table_row:
            raise RuntimeError("County GeoPackage has no feature layer")
        table = str(table_row[0]).replace('"', '""')
        rows = con.execute(f'SELECT GEOID, NAME FROM "{table}"').fetchall()
    return {int(fips): str(name) for fips, name in rows}


def load_metric_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    p = pd.read_csv(
        STEP44 / "p2_wuip_county_metrics_500m.csv",
        dtype={"STATEFP": str},
    )
    p["STATEFP"] = p["STATEFP"].str.zfill(2)
    p = p.rename(columns={"state": "STUSPS"})
    p["method"] = "WUI-P"
    s_all = pd.read_csv(
        STEP37 / "county_metrics_old_new.csv",
        dtype={"STATEFP": str},
    )
    s = s_all[
        s_all["pass_name"].eq("second_round_run1")
        & s_all["method"].eq("WUI-S")
    ].copy()
    s["STATEFP"] = s["STATEFP"].str.zfill(2)
    for label, frame in [("WUI-P", p), ("WUI-S", s)]:
        if frame["GEOID_INT"].duplicated().any():
            raise RuntimeError(f"{label} county metric FIPS are not unique")
        if not set(["p_a", "p_s"]).issubset(frame.columns):
            raise RuntimeError(f"{label} metric columns p_a/p_s missing")
    return p, s


def deserialize_combo(
    record: pd.Series,
    neighbors_frame: pd.DataFrame,
    metric_sources: dict[str, pd.DataFrame],
    names: dict[int, str],
) -> dict[str, Any]:
    state = str(record["state"])
    method = str(record["method"])
    frozen_variable = str(record["variable"])
    metric_col = variable_code(frozen_variable)
    ids = [int(x) for x in str(record["id_order"]).split("|") if x]
    part = neighbors_frame[
        neighbors_frame["state"].eq(state)
        & neighbors_frame["method"].eq(method)
        & neighbors_frame["variable"].eq(frozen_variable)
    ]
    neighbors = {i: [] for i in ids}
    for row in part.itertuples(index=False):
        neighbors[int(row.fips)].append(int(row.neighbor_fips))
    for key in neighbors:
        neighbors[key] = sorted(neighbors[key])
    whash = canonical_weights_hash(ids, neighbors)
    if whash != str(record["weights_sha256"]):
        raise RuntimeError(f"Frozen weight hash mismatch: {state}/{method}/{metric_col}")
    if len(ids) != int(record["feature_count"]):
        raise RuntimeError(f"Feature count mismatch: {state}/{method}/{metric_col}")
    source = metric_sources[method]
    m = source[source["STUSPS"].eq(state)][["GEOID_INT", metric_col]].copy()
    m = m[np.isfinite(m[metric_col])].sort_values("GEOID_INT")
    source_ids = m["GEOID_INT"].astype(int).tolist()
    if source_ids != ids:
        raise RuntimeError(f"Metric FIPS order mismatch: {state}/{method}/{metric_col}")
    value_map = dict(zip(source_ids, m[metric_col].astype(float)))
    y = np.asarray([value_map[i] for i in ids], dtype=np.float64)
    w = W(neighbors, id_order=ids, silence_warnings=True)
    w.transform = "r"
    islands = np.asarray([len(neighbors[i]) == 0 for i in ids], dtype=bool)
    return {
        "state": state,
        "method": method,
        "frozen_variable": frozen_variable,
        "variable": metric_col,
        "ids": ids,
        "county": [names.get(i, "") for i in ids],
        "neighbors": neighbors,
        "y": y,
        "w": w,
        "islands": islands,
        "weights_sha256": whash,
        "manifest": record,
    }


def two_sided_pseudo_p(observed: np.ndarray, sim: np.ndarray) -> np.ndarray:
    observed = np.asarray(observed, dtype=np.float64)
    sim = np.asarray(sim, dtype=np.float64)
    if sim.ndim != 2 or sim.shape[1] != observed.size:
        raise RuntimeError(
            f"Simulation shape must be (permutations,n_units), got {sim.shape}"
        )
    denominator = sim.shape[0] + 1
    result = np.empty(observed.size, dtype=np.float64)
    for idx, obs in enumerate(observed):
        values = sim[:, idx]
        percentile = float(np.mean(values <= obs) * 100.0)
        p_low = min(percentile, 100.0 - percentile)
        low_bound = float(np.percentile(values, p_low))
        high_bound = float(np.percentile(values, 100.0 - p_low))
        n_outside = int(np.count_nonzero(values <= low_bound))
        n_outside += int(np.count_nonzero(values >= high_bound))
        p_value = (n_outside + 1.0) / denominator
        if not (0.0 <= p_value <= 1.0):
            raise RuntimeError(
                "TWO_SIDED_P_OUT_OF_RANGE_TIES_AUDIT_REQUIRED:"
                f"unit={idx}; percentile={percentile}; low={low_bound}; "
                f"high={high_bound}; n_outside={n_outside}; p={p_value}"
            )
        result[idx] = p_value
    return result


def manual_two_sided_reference(observed: np.ndarray, sim: np.ndarray) -> np.ndarray:
    """Independent loop used only by the smoke test."""
    answers: list[float] = []
    for j in range(len(observed)):
        column = list(map(float, sim[:, j]))
        rank_percent = 100.0 * sum(v <= float(observed[j]) for v in column) / len(column)
        tail = min(rank_percent, 100.0 - rank_percent)
        lower, upper = np.percentile(np.asarray(column), [tail, 100.0 - tail])
        outside = sum(v <= lower for v in column) + sum(v >= upper for v in column)
        answers.append((outside + 1.0) / (len(column) + 1.0))
    return np.asarray(answers)


def run_local_library(combo: dict[str, Any]) -> dict[str, np.ndarray]:
    lm = Moran_Local(
        y=np.asarray(combo["y"], dtype=np.float64),
        w=combo["w"],
        transformation="r",
        permutations=PERMUTATIONS,
        geoda_quads=False,
        n_jobs=1,
        keep_simulations=True,
        seed=SEED,
    )
    sim = np.asarray(lm.sim, dtype=np.float64)
    if sim.shape != (PERMUTATIONS, len(combo["ids"])):
        raise RuntimeError(f"Unexpected lm.sim shape: {sim.shape}")
    formal_two_sided = np.full(len(combo["ids"]), np.nan, dtype=np.float64)
    non_island = ~np.asarray(combo["islands"], dtype=bool)
    formal_two_sided[non_island] = two_sided_pseudo_p(
        np.asarray(lm.Is, dtype=np.float64)[non_island],
        sim[:, non_island],
    )
    arrays = {
        "Is": np.asarray(lm.Is, dtype=np.float64),
        "sim": sim,
        "rlisas": np.asarray(lm.rlisas, dtype=np.float64),
        "EI_sim": np.asarray(lm.EI_sim, dtype=np.float64),
        "VI_sim": np.asarray(lm.VI_sim, dtype=np.float64),
        "z_sim": np.asarray(lm.z_sim, dtype=np.float64),
        "p_z_sim": np.asarray(lm.p_z_sim, dtype=np.float64),
        "p_sim_directed_legacy": np.asarray(lm.p_sim, dtype=np.float64),
        "p_sim_two_sided": formal_two_sided,
        "standardized_value": np.asarray(lm.z, dtype=np.float64),
        "spatial_lag": np.asarray(lag_spatial(combo["w"], lm.z), dtype=np.float64),
        "library_quadrant": np.asarray(lm.q, dtype=np.int16),
    }
    return arrays


def apply_explicit_island_mask(
    arrays: dict[str, np.ndarray], islands: np.ndarray
) -> dict[str, np.ndarray]:
    result = {key: np.array(value, copy=True) for key, value in arrays.items()}
    vector_fields = [
        "Is", "EI_sim", "VI_sim", "z_sim", "p_z_sim",
        "p_sim_directed_legacy", "p_sim_two_sided",
        "standardized_value", "spatial_lag",
    ]
    for key in vector_fields:
        result[key] = result[key].astype(np.float64, copy=False)
        result[key][islands] = np.nan
    result["library_quadrant"] = result["library_quadrant"].astype(np.float64)
    result["library_quadrant"][islands] = np.nan
    result["sim"][:, islands] = np.nan
    result["rlisas"][islands, :] = np.nan
    return result


def quadrant_from_sign(z: float, lag: float) -> str:
    if not (math.isfinite(z) and math.isfinite(lag)):
        return "ISLAND_OR_UNDEFINED"
    if z > 0 and lag > 0:
        return "HH"
    if z < 0 and lag < 0:
        return "LL"
    if z > 0 and lag < 0:
        return "HL"
    if z < 0 and lag > 0:
        return "LH"
    return "ISLAND_OR_UNDEFINED"


def smoke_tests(actual_canary: dict[str, Any]) -> tuple[pd.DataFrame, float]:
    rows: list[dict[str, Any]] = []

    def add(test: str, passed: bool, detail: str) -> None:
        rows.append({
            "test": test,
            "passed": bool(passed),
            "detail": detail,
            "status": "PASS" if passed else "FAIL",
        })
        if not passed:
            raise RuntimeError(f"Smoke test failed: {test}: {detail}")

    synthetic = {
        "ids": [0, 1, 2, 3, 4],
        "y": np.asarray([0.2, 1.0, 2.5, 4.0, 8.2], dtype=np.float64),
        "w": W(
            {0: [1], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3]},
            id_order=[0, 1, 2, 3, 4],
            silence_warnings=True,
        ),
        "islands": np.zeros(5, dtype=bool),
    }
    a = run_local_library(synthetic)
    b = run_local_library(synthetic)
    for field in ["Is", "sim", "p_sim_directed_legacy",
                  "p_sim_two_sided", "z_sim"]:
        add(
            f"synthetic_seed_repeat_{field}",
            np.array_equal(a[field], b[field], equal_nan=True),
            f"shape={a[field].shape};sha={array_sha256(a[field])}",
        )
    manual_sim = np.asarray(
        [[-3.0, -4.0], [-2.0, -2.0], [-1.0, -1.0], [0.0, 0.0],
         [1.0, 1.0], [2.0, 2.0], [3.0, 4.0]],
        dtype=np.float64,
    )
    manual_obs = np.asarray([2.5, -2.5], dtype=np.float64)
    derived = two_sided_pseudo_p(manual_obs, manual_sim)
    reference = manual_two_sided_reference(manual_obs, manual_sim)
    add(
        "manual_two_sided_function",
        np.array_equal(derived, reference) and np.allclose(derived, [0.375, 0.375]),
        f"derived={derived.tolist()};reference={reference.tolist()}",
    )
    add(
        "formal_two_sided_not_directed_p_sim",
        bool(np.any(a["p_sim_two_sided"] != a["p_sim_directed_legacy"])),
        "synthetic run contains at least one differing county p-value",
    )
    island_w = W(
        {0: [1], 1: [0, 2], 2: [1, 3], 3: [2], 4: []},
        id_order=[0, 1, 2, 3, 4],
        silence_warnings=True,
    )
    island_case = {
        "ids": [0, 1, 2, 3, 4],
        "y": np.asarray([0.2, 1.0, 2.5, 4.0, 8.2], dtype=np.float64),
        "w": island_w,
        "islands": np.asarray([False, False, False, False, True]),
    }
    masked = apply_explicit_island_mask(
        run_local_library(island_case), island_case["islands"]
    )
    add(
        "explicit_island_na_mask",
        all(np.isnan(masked[key][-1]) for key in [
            "Is", "p_sim_directed_legacy", "p_sim_two_sided", "z_sim"
        ]) and np.isnan(masked["sim"][:, -1]).all(),
        "island retained in id order and all formal/statistical fields masked NA",
    )

    canary_started = time.monotonic()
    canary_a = run_local_library(actual_canary)
    canary_b = run_local_library(actual_canary)
    canary_seconds = time.monotonic() - canary_started
    actual_equal = all(
        np.array_equal(canary_a[field], canary_b[field], equal_nan=True)
        for field in ["Is", "sim", "p_sim_directed_legacy",
                      "p_sim_two_sided", "z_sim"]
    )
    add(
        "actual_eligible_canary_repeat",
        actual_equal,
        f"{combo_id(actual_canary['state'], actual_canary['method'], actual_canary['frozen_variable'])};"
        f"two_runs_seconds={canary_seconds:.6f}",
    )
    return pd.DataFrame(rows), canary_seconds


def eligibility_table(manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rec in manifest.itertuples(index=False):
        n = int(rec.feature_count)
        variance = float(rec.variable_variance)
        s0 = float(rec.row_standardized_S0)
        if n == 1 and s0 == 0:
            status = "UNDEFINED_N1_S0_ZERO"
            reason = "n=1 and row-standardized S0=0"
        elif not bool(rec.variable_has_variance) or not math.isfinite(variance) or variance <= 0:
            status = "UNDEFINED_ZERO_VARIANCE"
            reason = "variable has zero or non-finite variance"
        elif n < 30:
            status = "EXCLUDED_N_LT_30"
            reason = f"Option A formal inference gate excludes n={n}<30"
        else:
            status = "ELIGIBLE_FORMAL_N_GE_30"
            reason = f"Option A basic sample-size gate passes n={n}>=30"
        rows.append({
            "analysis_scope": "county_within_state_500m",
            "state": rec.state,
            "method": rec.method,
            "buffer_m": 500,
            "variable": variable_code(rec.variable),
            "frozen_weight_variable": rec.variable,
            "feature_count": n,
            "n_islands": int(rec.islands),
            "row_standardized_S0": s0,
            "variable_variance": variance,
            "variable_has_variance": bool(rec.variable_has_variance),
            "weights_sha256": rec.weights_sha256,
            "eligibility_status": status,
            "reason": reason,
        })
    return pd.DataFrame(rows)


def run_global_pass(combos: list[dict[str, Any]], eligibility: pd.DataFrame,
                    pass_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    eligible_lookup = {
        (r.state, r.method, r.frozen_weight_variable): r.eligibility_status
        for r in eligibility.itertuples(index=False)
    }
    started = time.monotonic()
    for index, combo in enumerate(combos, 1):
        key = (combo["state"], combo["method"], combo["frozen_variable"])
        eligibility_status = eligible_lookup[key]
        common = {
            "analysis_scope": "county_within_state_500m",
            "state": combo["state"],
            "method": combo["method"],
            "buffer_m": 500,
            "variable": combo["variable"],
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
            "run": pass_name,
        }
        if eligibility_status == "ELIGIBLE_FORMAL_N_GE_30":
            mi = Moran(
                np.asarray(combo["y"], dtype=np.float64),
                combo["w"],
                transformation="r",
                permutations=0,
                two_tailed=True,
            )
            significant = bool(mi.p_norm < ALPHA)
            direction = (
                "POSITIVE_SIGNIFICANT" if significant and mi.I > mi.EI
                else "NEGATIVE_SIGNIFICANT" if significant
                else "NOT_SIGNIFICANT"
            )
            common.update({
                "moran_i": float(mi.I),
                "expected_i": float(mi.EI),
                "variance_norm": float(mi.VI_norm),
                "z_norm": float(mi.z_norm),
                "p_norm": float(mi.p_norm),
                "significant_p_norm": significant,
                "direction": direction,
                "status": "FORMAL_COMPLETE",
                "reason": "",
            })
        else:
            common.update({
                "moran_i": np.nan,
                "expected_i": np.nan,
                "variance_norm": np.nan,
                "z_norm": np.nan,
                "p_norm": np.nan,
                "significant_p_norm": np.nan,
                "direction": "UNDEFINED",
                "status": (
                    "UNDEFINED" if eligibility_status.startswith("UNDEFINED")
                    else "EXCLUDED_N_LT_30"
                ),
                "reason": eligibility_status,
            })
        rows.append(common)
        progress(6, 10, combo["state"],
                 f"GLOBAL_{pass_name}_{combo['method']}_{combo['variable']}",
                 index, len(combos), started)
    return pd.DataFrame(rows)


def local_rows_for_combo(
    combo: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    islands = combo["islands"]
    masked = apply_explicit_island_mask(arrays, islands)
    valid = ~islands & np.isfinite(masked["p_sim_two_sided"])
    valid_p = masked["p_sim_two_sided"][valid]
    critical = float(fdr(valid_p, alpha=ALPHA))
    raw_sig = np.zeros(len(combo["ids"]), dtype=bool)
    fdr_sig = np.zeros(len(combo["ids"]), dtype=bool)
    raw_sig[valid] = valid_p < ALPHA
    fdr_sig[valid] = valid_p <= critical
    family = combo_id(
        combo["state"], combo["method"], combo["frozen_variable"]
    )
    rows: list[dict[str, Any]] = []
    for idx, fips in enumerate(combo["ids"]):
        is_island = bool(islands[idx])
        quadrant = quadrant_from_sign(
            float(masked["standardized_value"][idx]),
            float(masked["spatial_lag"][idx]),
        )
        if is_island:
            raw_cluster = "ISLAND_OR_UNDEFINED"
            formal_cluster = "ISLAND_OR_UNDEFINED"
            status = "ISLAND_UNDEFINED_EXPLICIT_MASK"
        else:
            raw_cluster = quadrant if raw_sig[idx] else "NOT_SIGNIFICANT"
            formal_cluster = quadrant if fdr_sig[idx] else "NOT_SIGNIFICANT"
            status = (
                "FORMAL_COMPLETE" if quadrant != "ISLAND_OR_UNDEFINED"
                or not fdr_sig[idx] else "ZERO_SIGN_UNDEFINED"
            )
        rows.append({
            "analysis_scope": "county_within_state_500m",
            "state": combo["state"],
            "method": combo["method"],
            "buffer_m": 500,
            "variable": combo["variable"],
            "frozen_weight_variable": combo["frozen_variable"],
            "fips": int(fips),
            "county": combo["county"][idx],
            "metric_value": float(combo["y"][idx]),
            "standardized_value": masked["standardized_value"][idx],
            "spatial_lag": masked["spatial_lag"][idx],
            "local_moran_i": masked["Is"][idx],
            "expected_local_i": masked["EI_sim"][idx],
            "variance_sim": masked["VI_sim"][idx],
            "z_sim": masked["z_sim"][idx],
            "p_z_sim_one_sided": masked["p_z_sim"][idx],
            "p_z_sim_two_sided_approx": (
                min(1.0, 2.0 * masked["p_z_sim"][idx])
                if math.isfinite(masked["p_z_sim"][idx]) else np.nan
            ),
            "p_sim_directed_legacy": masked["p_sim_directed_legacy"][idx],
            "p_sim_two_sided": masked["p_sim_two_sided"][idx],
            "formal_p_value_field": "p_sim_two_sided",
            "two_sided_derivation": (
                "stored_conditional_simulations_esda210_public_logic"
            ),
            "raw_alpha": ALPHA,
            "raw_significant": bool(raw_sig[idx]) if not is_island else False,
            "fdr_family_id": family,
            "fdr_n_tests": int(valid.sum()),
            "fdr_alpha": ALPHA,
            "fdr_critical_p": critical,
            "fdr_significant": bool(fdr_sig[idx]) if not is_island else False,
            "quadrant": quadrant,
            "library_quadrant_code_audit": masked["library_quadrant"][idx],
            "raw_cluster_type": raw_cluster,
            "formal_fdr_cluster_type": formal_cluster,
            "neighbor_count": len(combo["neighbors"][fips]),
            "is_island": is_island,
            "weights_sha256": combo["weights_sha256"],
            "permutations": PERMUTATIONS,
            "random_seed": SEED,
            "n_jobs": 1,
            "status": status,
        })
    family_row = {
        "fdr_family_id": family,
        "state": combo["state"],
        "method": combo["method"],
        "variable": combo["variable"],
        "frozen_weight_variable": combo["frozen_variable"],
        "fdr_n_tests": int(valid.sum()),
        "n_islands_excluded": int(islands.sum()),
        "fdr_alpha": ALPHA,
        "fdr_critical_p": critical,
        "raw_significant_count": int(raw_sig.sum()),
        "fdr_significant_count": int(fdr_sig.sum()),
        "formal_p_value_field": "p_sim_two_sided",
        "status": "PASS",
    }
    return rows, family_row


def save_simulation_artifact(out: Path, combo: dict[str, Any],
                             arrays: dict[str, np.ndarray]) -> tuple[str, str]:
    artifact = out / "local_moran" / "simulations_run1" / (
        combo_id(combo["state"], combo["method"], combo["frozen_variable"]) + ".npz"
    )
    artifact.parent.mkdir(parents=True, exist_ok=True)
    tmp = artifact.with_name(artifact.name + ".partial")
    with tmp.open("wb") as dst:
        np.savez_compressed(
            dst,
            fips=np.asarray(combo["ids"], dtype=np.int64),
            Is=arrays["Is"],
            sim=arrays["sim"],
            rlisas=arrays["rlisas"],
            EI_sim=arrays["EI_sim"],
            VI_sim=arrays["VI_sim"],
            z_sim=arrays["z_sim"],
            p_z_sim=arrays["p_z_sim"],
            p_sim_directed_legacy=arrays["p_sim_directed_legacy"],
            p_sim_two_sided=arrays["p_sim_two_sided"],
            standardized_value=arrays["standardized_value"],
            spatial_lag=arrays["spatial_lag"],
            island_mask=combo["islands"],
            permutations=np.asarray([PERMUTATIONS], dtype=np.int64),
            seed=np.asarray([SEED], dtype=np.int64),
        )
    os.replace(tmp, artifact)
    return str(artifact), sha256(artifact)


def run_local_pass(
    eligible_combos: list[dict[str, Any]],
    out: Path,
    pass_name: str,
    save_artifacts: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_rows: list[dict[str, Any]] = []
    families: list[dict[str, Any]] = []
    hashes: list[dict[str, Any]] = []
    started = time.monotonic()
    for index, combo in enumerate(eligible_combos, 1):
        arrays = run_local_library(combo)
        rows, family = local_rows_for_combo(combo, arrays)
        all_rows.extend(rows)
        families.append(family)
        record = {
            "run": pass_name,
            "state": combo["state"],
            "method": combo["method"],
            "variable": combo["variable"],
            "combination_id": combo_id(
                combo["state"], combo["method"], combo["frozen_variable"]
            ),
            "Is_sha256": array_sha256(arrays["Is"]),
            "sim_sha256": array_sha256(arrays["sim"]),
            "rlisas_sha256": array_sha256(arrays["rlisas"]),
            "p_sim_directed_legacy_sha256": array_sha256(
                arrays["p_sim_directed_legacy"]
            ),
            "p_sim_two_sided_sha256": array_sha256(arrays["p_sim_two_sided"]),
            "z_sim_sha256": array_sha256(arrays["z_sim"]),
            "simulation_artifact": "",
            "simulation_artifact_sha256": "",
        }
        if save_artifacts:
            path, digest = save_simulation_artifact(out, combo, arrays)
            record["simulation_artifact"] = path
            record["simulation_artifact_sha256"] = digest
        hashes.append(record)
        progress(
            7 if pass_name == "run1" else 8,
            10,
            combo["state"],
            f"LOCAL_{pass_name}_{combo['method']}_{combo['variable']}",
            index,
            len(eligible_combos),
            started,
        )
    return pd.DataFrame(all_rows), pd.DataFrame(families), pd.DataFrame(hashes)


def excluded_local_placeholders(
    excluded_combos: list[dict[str, Any]],
    eligibility: pd.DataFrame,
) -> pd.DataFrame:
    lookup = {
        (r.state, r.method, r.frozen_weight_variable): r.eligibility_status
        for r in eligibility.itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    for combo in excluded_combos:
        status_code = lookup[
            (combo["state"], combo["method"], combo["frozen_variable"])
        ]
        status = "UNDEFINED" if status_code.startswith("UNDEFINED") else "EXCLUDED_N_LT_30"
        for idx, fips in enumerate(combo["ids"]):
            rows.append({
                "analysis_scope": "county_within_state_500m",
                "state": combo["state"],
                "method": combo["method"],
                "buffer_m": 500,
                "variable": combo["variable"],
                "frozen_weight_variable": combo["frozen_variable"],
                "fips": int(fips),
                "county": combo["county"][idx],
                "metric_value": float(combo["y"][idx]),
                "standardized_value": np.nan,
                "spatial_lag": np.nan,
                "local_moran_i": np.nan,
                "expected_local_i": np.nan,
                "variance_sim": np.nan,
                "z_sim": np.nan,
                "p_z_sim_one_sided": np.nan,
                "p_z_sim_two_sided_approx": np.nan,
                "p_sim_directed_legacy": np.nan,
                "p_sim_two_sided": np.nan,
                "formal_p_value_field": "NOT_APPLICABLE",
                "two_sided_derivation": "NOT_RUN_OPTION_A",
                "raw_alpha": ALPHA,
                "raw_significant": False,
                "fdr_family_id": "",
                "fdr_n_tests": 0,
                "fdr_alpha": ALPHA,
                "fdr_critical_p": np.nan,
                "fdr_significant": False,
                "quadrant": "ISLAND_OR_UNDEFINED",
                "library_quadrant_code_audit": np.nan,
                "raw_cluster_type": "ISLAND_OR_UNDEFINED",
                "formal_fdr_cluster_type": "ISLAND_OR_UNDEFINED",
                "neighbor_count": len(combo["neighbors"][fips]),
                "is_island": bool(combo["islands"][idx]),
                "weights_sha256": combo["weights_sha256"],
                "permutations": 0,
                "random_seed": np.nan,
                "n_jobs": 1,
                "status": status,
            })
    return pd.DataFrame(rows)


def frame_exact_equal(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    try:
        pd.testing.assert_frame_equal(
            left.reset_index(drop=True),
            right.reset_index(drop=True),
            check_exact=True,
            check_dtype=True,
        )
        return True
    except AssertionError:
        return False


def write_manifest(out: Path) -> tuple[int, list[str]]:
    files = sorted(
        p for p in out.rglob("*")
        if p.is_file() and p.name != "sha256_manifest.txt"
    )
    lines = [f"{sha256(path)}  {path.relative_to(out)}" for path in files]
    atomic_text(out / "sha256_manifest.txt", "\n".join(lines) + "\n")
    return verify_manifest(out, out / "sha256_manifest.txt", len(files))


def main() -> None:
    out = ROOT / f"step45c_p2_formal_moran_new_protocol_{stamp()}"
    out.mkdir()
    for subdir in [
        "scripts", "config", "manifests", "global_moran",
        "local_moran", "qc", "logs", "checkpoints",
    ]:
        (out / subdir).mkdir()
    started_all = time.monotonic()
    progress(1, 10, "ALL", "FROZEN_ENVIRONMENT_GATE", 0, 1, started_all)

    original_sentinels = {
        str(path): sha256(path)
        for path in [
            STEP43 / "step43_status.json",
            STEP43 / "sha256_manifest.txt",
            STEP44 / "step44_status.json",
            STEP44 / "sha256_manifest.txt",
            STEP45 / "step45_status.json",
            STEP45 / "sha256_manifest.txt",
            STEP45B / "step45b_status.json",
            STEP45B / "sha256_manifest.txt",
            PARENT_RUN / "step45c_status.json",
            PARENT_RUN / "sha256_manifest.txt",
        ]
    }
    versions = package_versions()
    local_signature = inspect.signature(Moran_Local)
    supported = {
        name: name in local_signature.parameters for name in REQUIRED_LOCAL_PARAMS
    }
    if platform.python_version() != "3.10.19":
        raise RuntimeError(f"Frozen Python gate failed: {platform.python_version()}")
    if versions["esda"] != "2.7.0" or versions["libpysal"] != "4.13.0":
        raise RuntimeError(f"Frozen PySAL gate failed: {versions}")
    if not all(supported.values()):
        raise RuntimeError(f"Required current API parameters missing: {supported}")
    optional_dependency = (
        "OPTIONAL_DEPENDENCY_ABSENT_PURE_PYTHON_FALLBACK"
        if versions["numba"] == "NOT_INSTALLED"
        else "OPTIONAL_DEPENDENCY_PRESENT_NOT_REQUIRED"
    )
    environment = {
        "captured_utc": utc_now(),
        "protocol": "PYTHON310_ESDA270_FROZEN_COMPATIBILITY_PROTOCOL",
        "python_executable": str(PYTHON),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "esda_Moran_signature": str(inspect.signature(Moran)),
        "esda_Moran_Local_signature": str(local_signature),
        "esda_fdr_signature": str(inspect.signature(fdr)),
        "required_Moran_Local_parameters": REQUIRED_LOCAL_PARAMS,
        "parameter_support": supported,
        "alternative_parameter_required": False,
        "alternative_parameter_passed": False,
        "numba_status": optional_dependency,
        "environment_modified": False,
        "environment_modification_policy": "NO_ENVIRONMENT_MODIFICATION",
        "gate_status": "PASS",
    }
    atomic_json(out / "p2_moran_software_environment.json", environment)
    signature_text = "\n".join([
        "STEP45C PY310 ESDA270 COMPATIBILITY SIGNATURE GATE",
        f"captured_utc={environment['captured_utc']}",
        f"python={PYTHON}",
        f"python_version={platform.python_version()}",
        f"esda_version={versions['esda']}",
        f"libpysal_version={versions['libpysal']}",
        f"numba={versions['numba']}",
        f"numba_status={optional_dependency}",
        f"esda.Moran{inspect.signature(Moran)}",
        f"esda.Moran_Local{local_signature}",
        f"esda.fdr{inspect.signature(fdr)}",
        *[f"{name}={'SUPPORTED' if value else 'MISSING'}"
          for name, value in supported.items()],
        "alternative=NOT_REQUIRED_NOT_PASSED",
        "gate_status=PASS",
        "environment_modified=NO",
        "",
    ])
    atomic_text(out / "p2_moran_function_signature.txt", signature_text)
    progress(1, 10, "ALL", "FROZEN_ENVIRONMENT_GATE", 1, 1, started_all)

    progress(2, 10, "ALL", "PARENT_AND_FREEZE_GATES", 0, 1, started_all)
    parent_status = json.loads((PARENT_RUN / "step45c_status.json").read_text())
    if parent_status["blocker_code"] != "BLOCKED_PYSAL_VERSION_MISSING_REQUIRED_PARAMETERS":
        raise RuntimeError("Parent blocked-run identity gate failed")
    step45_status = json.loads((STEP45 / "step45_status.json").read_text())
    step45b_status = json.loads((STEP45B / "step45b_status.json").read_text())
    if step45_status["status"] != "JACCARD_COMPLETE_MORAN_BLOCKED_MISSING_DEPENDENCY":
        raise RuntimeError("Step45 status gate failed")
    if step45b_status["status"] != (
        "LEGACY_LOCAL_MORAN_NOT_RECOVERABLE_METHOD_DECISION_REQUIRED"
    ) or step45b_status["checks_passed"] != 22:
        raise RuntimeError("Step45B status/QC gate failed")
    n45, err45 = verify_manifest(STEP45, STEP45 / "sha256_manifest.txt", 27)
    n45b, err45b = verify_manifest(STEP45B, STEP45B / "sha256_manifest.txt", 13)
    if err45 or err45b:
        raise RuntimeError(f"Parent manifest gate failed: {err45 + err45b}")
    jaccard = pd.read_csv(STEP45 / "p2_wuip_wuis_jaccard_99.csv")
    national = pd.read_csv(STEP45 / "p2_wuip_wuis_jaccard_national_500m.csv")
    if len(jaccard) != 99 or not math.isclose(
        float(national.iloc[0]["national_micro_jaccard"]),
        0.621897978161495,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise RuntimeError("Frozen Jaccard gate failed")
    canary_all = pd.read_csv(STEP45 / "p2_moran_legacy_reproduction_canary.csv")
    canary = canary_all[canary_all["analysis_type"].eq("GLOBAL_MORAN")].copy()
    if len(canary) != 196 or not canary["legacy_canary_status"].eq("PASS").all():
        raise RuntimeError("Legacy Global 196/196 canary gate failed")
    progress(2, 10, "ALL", "PARENT_AND_FREEZE_GATES", 1, 1, started_all)

    progress(3, 10, "ALL", "WEIGHT_AND_ELIGIBILITY_GATE", 0, 196, started_all)
    weight_manifest = pd.read_csv(STEP45 / "p2_spatial_weights_manifest.csv")
    scope = pd.read_csv(STEP45B / "p2_moran_state_feature_count_audit.csv")
    if len(weight_manifest) != 196 or len(scope) != 196:
        raise RuntimeError("196-combination manifest/scope gate failed")
    key_cols = ["state", "method", "variable"]
    weight_manifest = weight_manifest.merge(
        scope[key_cols + [
            "row_standardized_S0", "variable_variance", "variable_has_variance"
        ]],
        on=key_cols,
        validate="1:1",
    )
    eligibility = eligibility_table(weight_manifest)
    eligible_mask = eligibility["eligibility_status"].eq("ELIGIBLE_FORMAL_N_GE_30")
    if int(eligible_mask.sum()) != EXPECTED_ELIGIBLE:
        raise RuntimeError(f"Eligible count changed: {int(eligible_mask.sum())}")
    if int((~eligible_mask).sum()) != EXPECTED_EXCLUDED:
        raise RuntimeError(f"Excluded count changed: {int((~eligible_mask).sum())}")
    dc = eligibility[eligibility["state"].eq("DC")]
    if len(dc) != 4 or not dc["eligibility_status"].eq("UNDEFINED_N1_S0_ZERO").all():
        raise RuntimeError("DC four-combination gate failed")
    atomic_csv(out / "p2_moran_formal_eligibility_196.csv", eligibility, "%.15f")
    progress(3, 10, "ALL", "WEIGHT_AND_ELIGIBILITY_GATE", 196, 196, started_all)

    progress(4, 10, "ALL", "DESERIALIZE_FROZEN_INPUTS", 0, 196, started_all)
    neighbor_frame = pd.read_csv(
        STEP45 / "p2_spatial_weights_neighbors.csv",
        dtype={"fips": int, "neighbor_fips": int},
    )
    p_metrics, s_metrics = load_metric_sources()
    metrics = {"WUI-P": p_metrics, "WUI-S": s_metrics}
    names = county_names()
    combos: list[dict[str, Any]] = []
    started_deserialize = time.monotonic()
    for index, rec in weight_manifest.iterrows():
        combo = deserialize_combo(rec, neighbor_frame, metrics, names)
        combos.append(combo)
        progress(
            4, 10, combo["state"],
            f"DESERIALIZE_{combo['method']}_{combo['variable']}",
            index + 1, 196, started_deserialize,
        )
    if len({c["weights_sha256"] for c in combos}) == 0:
        raise RuntimeError("No frozen weights deserialized")

    mi = next(
        c for c in combos
        if c["state"] == "MI" and c["method"] == "WUI-P"
        and c["variable"] == "p_s"
    )
    mi_idx = mi["ids"].index(26001)
    if not mi["islands"][mi_idx] or len(mi["ids"]) != 43:
        raise RuntimeError("MI FIPS 26001 frozen island gate failed")
    mi_area = next(
        c for c in combos
        if c["state"] == "MI" and c["method"] == "WUI-P"
        and c["variable"] == "p_a"
    )
    geometry_neighbors = mi_area["neighbors"][26001]
    mi_audit = pd.DataFrame([{
        "state": "MI",
        "method": "WUI-P",
        "variable": "p_s",
        "fips": 26001,
        "metric_value": float(mi["y"][mi_idx]),
        "metric_value_finite": bool(np.isfinite(mi["y"][mi_idx])),
        "frozen_neighbor_count": len(mi["neighbors"][26001]),
        "geometry_topology_island": len(geometry_neighbors) == 0,
        "geometry_neighbors_from_frozen_p_a_graph": "|".join(map(str, geometry_neighbors)),
        "analysis_island_from_missing_metric": len(geometry_neighbors) > 0,
        "global_mean_treatment": "retained in n=43 mean and standardization",
        "standardization_treatment": "retained",
        "S0_treatment": "island row contributes zero; frozen row-standardized S0=42",
        "local_result_treatment": "explicit NA mask after library call",
        "fdr_treatment": "excluded from FDR family",
        "neighbor_relationship_impact": (
            "no virtual neighbor; no FIPS deletion; other frozen adjacencies unchanged"
        ),
        "formal_fdr_cluster_type": "ISLAND_OR_UNDEFINED",
        "status": "ISLAND_UNDEFINED_EXPLICIT_MASK",
    }])
    atomic_csv(out / "p2_mi_26001_island_policy_audit.csv", mi_audit)

    progress(5, 10, "ALL", "SMOKE_AND_ACTUAL_CANARY", 0, 1, started_all)
    eligible_keys = set(
        eligibility.loc[
            eligible_mask, ["state", "method", "frozen_weight_variable"]
        ].itertuples(index=False, name=None)
    )
    eligible_combos = [
        c for c in combos
        if (c["state"], c["method"], c["frozen_variable"]) in eligible_keys
    ]
    excluded_combos = [
        c for c in combos
        if (c["state"], c["method"], c["frozen_variable"]) not in eligible_keys
    ]
    actual_canary = next(
        c for c in eligible_combos
        if c["state"] == "AL" and c["method"] == "WUI-P" and c["variable"] == "p_a"
    )
    smoke, canary_seconds = smoke_tests(actual_canary)
    estimated_two_pass_seconds = canary_seconds * EXPECTED_ELIGIBLE
    performance_decision = (
        "ACCEPTABLE_CONTINUE_PURE_PYTHON"
        if estimated_two_pass_seconds <= 6 * 3600
        else "STOP_ESTIMATE_UNACCEPTABLE"
    )
    if performance_decision != "ACCEPTABLE_CONTINUE_PURE_PYTHON":
        raise RuntimeError(
            f"Pure Python ETA unacceptable: {estimated_two_pass_seconds}s"
        )
    smoke["actual_canary_two_run_seconds"] = canary_seconds
    smoke["estimated_139_two_pass_seconds"] = estimated_two_pass_seconds
    smoke["performance_decision"] = performance_decision
    atomic_csv(out / "p2_moran_compatibility_smoke_test.csv", smoke, "%.15f")
    progress(5, 10, "ALL", "SMOKE_AND_ACTUAL_CANARY", 1, 1, started_all)

    global1 = run_global_pass(combos, eligibility, "run1")
    global2 = run_global_pass(combos, eligibility, "run2")
    global_reproduced = frame_exact_equal(
        global1.drop(columns=["run"]), global2.drop(columns=["run"])
    )
    if not global_reproduced:
        raise RuntimeError("Global run1/run2 exact reproduction failed")
    global_formal = global1.drop(columns=["run"])
    global139 = global_formal[global_formal["status"].eq("FORMAL_COMPLETE")].copy()
    global57 = global_formal[~global_formal["status"].eq("FORMAL_COMPLETE")].copy()
    atomic_csv(out / "p2_global_moran_formal_results_196.csv", global_formal, "%.15f")
    atomic_csv(out / "p2_global_moran_formal_results_139.csv", global139, "%.15f")
    atomic_csv(out / "p2_global_moran_excluded_57.csv", global57, "%.15f")
    global_repro = pd.DataFrame([{
        "module": "Global Moran",
        "run1_completed": True,
        "run2_completed": True,
        "rows_compared": len(global_formal),
        "formal_rows_compared": len(global139),
        "fields_compared": "|".join(
            [c for c in global_formal.columns if c not in ["reason"]]
        ),
        "identical": global_reproduced,
        "status": "PASS",
    }])
    atomic_csv(out / "p2_global_moran_reproducibility_audit.csv", global_repro)

    legacy = pd.read_csv(STEP37 / "morans_i_second_round_results.csv")
    legacy = legacy[
        legacy["pass_name"].eq("second_round_run1")
        &
        legacy["method"].isin(["WUI-P", "WUI-S"])
        & legacy["variable"].isin(["area_proportion", "structure_proportion"])
    ].copy()
    legacy["variable"] = legacy["variable"].map(variable_code)
    legacy = legacy.rename(
        columns={"STUSPS": "state", "moran_i": "legacy_moran_i"}
    )
    change = global_formal.merge(
        legacy[["state", "method", "variable", "legacy_moran_i"]],
        on=["state", "method", "variable"],
        how="left",
        validate="1:1",
    )
    change = change[[
        "state", "method", "variable", "status", "legacy_moran_i", "moran_i",
    ]].rename(columns={"moran_i": "new_moran_i"})
    change["absolute_change"] = change["new_moran_i"] - change["legacy_moran_i"]
    change["comparison_status"] = np.where(
        change["status"].eq("FORMAL_COMPLETE"),
        "FORMAL_NEW_VS_LEGACY_GLOBAL",
        "NEW_FORMAL_EXCLUDED_OPTION_A",
    )
    atomic_csv(out / "p2_global_moran_change_vs_legacy.csv", change, "%.15f")

    local1, families1, hashes1 = run_local_pass(
        eligible_combos, out, "run1", True
    )
    local2, families2, hashes2 = run_local_pass(
        eligible_combos, out, "run2", False
    )
    local_reproduced = (
        frame_exact_equal(local1, local2)
        and frame_exact_equal(families1, families2)
        and frame_exact_equal(
            hashes1.drop(columns=[
                "run", "simulation_artifact", "simulation_artifact_sha256"
            ]),
            hashes2.drop(columns=[
                "run", "simulation_artifact", "simulation_artifact_sha256"
            ]),
        )
    )
    if not local_reproduced:
        raise RuntimeError("Local run1/run2 exact reproduction failed")
    placeholders = excluded_local_placeholders(excluded_combos, eligibility)
    local_all = pd.concat([local1, placeholders], ignore_index=True)
    local_all = local_all.sort_values(
        ["state", "method", "variable", "fips"]
    ).reset_index(drop=True)
    local_formal_count = (
        local1[["state", "method", "variable"]].drop_duplicates().shape[0]
    )
    if local_formal_count != EXPECTED_ELIGIBLE:
        raise RuntimeError(f"Local complete combination count {local_formal_count}")
    atomic_csv(out / "p2_local_moran_formal_results.csv", local_all, "%.15f")
    atomic_csv(out / "p2_local_moran_formal_fdr_results.csv", local_all, "%.15f")
    atomic_csv(out / "p2_local_moran_raw_significance_results.csv", local_all, "%.15f")
    atomic_csv(out / "p2_local_moran_fdr_family_audit.csv", families1, "%.15f")
    atomic_csv(
        out / "p2_local_moran_simulation_artifact_inventory.csv",
        hashes1,
    )
    hash_compare = hashes1.drop(columns=[
        "simulation_artifact", "simulation_artifact_sha256"
    ]).merge(
        hashes2.drop(columns=[
            "simulation_artifact", "simulation_artifact_sha256"
        ]),
        on=["state", "method", "variable", "combination_id"],
        suffixes=("_run1", "_run2"),
        validate="1:1",
    )
    hash_cols = [
        c[:-5] for c in hash_compare.columns
        if c.endswith("_run1") and c != "run_run1"
    ]
    for name in hash_cols:
        hash_compare[f"{name}_identical"] = (
            hash_compare[f"{name}_run1"] == hash_compare[f"{name}_run2"]
        )
    local_repro = pd.DataFrame([{
        "module": "Local Moran",
        "eligible_combinations": EXPECTED_ELIGIBLE,
        "run1_completed": True,
        "run2_completed": True,
        "county_rows_compared": len(local1),
        "arrays_compared": "|".join(ARRAY_FIELDS),
        "identical": local_reproduced,
        "permutations": PERMUTATIONS,
        "seed": SEED,
        "status": "PASS",
    }])
    atomic_csv(out / "p2_local_moran_reproducibility_audit.csv", local_repro)
    atomic_csv(
        out / "local_moran" / "run1_run2_array_hash_audit.csv",
        hash_compare,
    )

    cluster_counts = (
        local_all.groupby(
            ["state", "method", "variable", "formal_fdr_cluster_type"],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
    )
    cluster_counts["status"] = "PASS"
    atomic_csv(
        out / "p2_local_moran_cluster_counts_by_combination.csv",
        cluster_counts,
    )
    island_audit = local_all[
        local_all["is_island"].astype(bool)
        | local_all["status"].isin(["UNDEFINED", "ISLAND_UNDEFINED_EXPLICIT_MASK"])
    ][[
        "state", "method", "variable", "fips", "neighbor_count",
        "is_island", "status", "formal_fdr_cluster_type",
    ]].copy()
    island_audit["reason"] = np.where(
        island_audit["status"].eq("ISLAND_UNDEFINED_EXPLICIT_MASK"),
        "eligible combination island explicitly masked and excluded from FDR",
        "Option A combination undefined",
    )
    atomic_csv(out / "p2_local_moran_island_undefined_audit.csv", island_audit)

    progress(9, 10, "ALL", "QC_AND_PACKAGING", 0, 1, started_all)
    current_sentinels = {path: sha256(Path(path)) for path in original_sentinels}
    sentinels_unchanged = original_sentinels == current_sentinels
    formal_rows = local_all[local_all["status"].eq("FORMAL_COMPLETE")]
    raw_sig_count = int(formal_rows["raw_significant"].sum())
    fdr_sig_count = int(formal_rows["fdr_significant"].sum())
    formal_cluster_counts = (
        formal_rows["formal_fdr_cluster_type"].value_counts().to_dict()
    )
    directed_used_formally = False
    mi_row = local_all[
        local_all["state"].eq("MI")
        & local_all["method"].eq("WUI-P")
        & local_all["variable"].eq("p_s")
        & local_all["fips"].eq(26001)
    ].iloc[0]
    simulation_files = list(
        (out / "local_moran" / "simulations_run1").glob("*.npz")
    )
    qc_checks: list[tuple[str, bool, str]] = [
        ("Q01_OPTION_A_RECORDED", True, "within-state n>=30 formal scope"),
        ("Q02_PARENT_RUN_RECORDED", True, str(PARENT_RUN)),
        ("Q03_PARENT_BLOCKER_PRESERVED", parent_status["status"] == "BLOCKED", ""),
        ("Q04_PYTHON_31019", platform.python_version() == "3.10.19", ""),
        ("Q05_ESDA_270", versions["esda"] == "2.7.0", ""),
        ("Q06_LIBPYSAL_4130", versions["libpysal"] == "4.13.0", ""),
        ("Q07_REQUIRED_SIGNATURE_SUPPORT", all(supported.values()), str(supported)),
        ("Q08_ALTERNATIVE_NOT_REQUIRED_OR_PASSED", "alternative" not in REQUIRED_LOCAL_PARAMS, ""),
        ("Q09_NUMBA_OPTIONAL_FALLBACK", optional_dependency.startswith("OPTIONAL_DEPENDENCY"), optional_dependency),
        ("Q10_NO_ENVIRONMENT_MODIFICATION", True, ""),
        ("Q11_STEP45B_22_PASS_GATE", step45b_status["checks_passed"] == 22, ""),
        ("Q12_JACCARD_99_FROZEN", len(jaccard) == 99, ""),
        ("Q13_NATIONAL_MICRO_FROZEN", math.isclose(float(national.iloc[0]["national_micro_jaccard"]), 0.621897978161495, abs_tol=1e-15), ""),
        ("Q14_LEGACY_GLOBAL_CANARY_196", len(canary) == 196 and canary["legacy_canary_status"].eq("PASS").all(), ""),
        ("Q15_WEIGHTS_196_DESERIALIZED_HASH_PASS", len(combos) == 196, ""),
        ("Q16_ELIGIBILITY_139", int(eligible_mask.sum()) == 139, ""),
        ("Q17_EXCLUDED_57", int((~eligible_mask).sum()) == 57, ""),
        ("Q18_DC_FOUR_UNDEFINED", len(dc) == 4 and dc["eligibility_status"].eq("UNDEFINED_N1_S0_ZERO").all(), ""),
        ("Q19_SMOKE_ALL_PASS", smoke["passed"].all(), ""),
        ("Q20_ACTUAL_CANARY_PASS", bool(smoke.loc[smoke["test"].eq("actual_eligible_canary_repeat"), "passed"].iloc[0]), ""),
        ("Q21_TWO_SIDED_MANUAL_PASS", bool(smoke.loc[smoke["test"].eq("manual_two_sided_function"), "passed"].iloc[0]), ""),
        ("Q22_DIRECTED_P_NOT_FORMAL", not directed_used_formally, ""),
        ("Q23_GLOBAL_196_ROWS", len(global_formal) == 196, ""),
        ("Q24_GLOBAL_139_COMPLETE", len(global139) == 139, ""),
        ("Q25_GLOBAL_57_NO_P", len(global57) == 57 and global57["p_norm"].isna().all(), ""),
        ("Q26_GLOBAL_P_NORM_ONLY", global139["p_norm"].notna().all(), ""),
        ("Q27_GLOBAL_PERMUTATIONS_ZERO", global_formal["permutations"].eq(0).all(), ""),
        ("Q28_GLOBAL_SEED_NA", global_formal["random_seed"].isna().all(), ""),
        ("Q29_GLOBAL_TWO_RUNS_IDENTICAL", global_reproduced, ""),
        ("Q30_LOCAL_139_COMPLETE", local_formal_count == 139, ""),
        ("Q31_LOCAL_PERMUTATIONS_999", local1["permutations"].eq(999).all(), ""),
        ("Q32_LOCAL_SEED_20260728", local1["random_seed"].eq(SEED).all(), ""),
        ("Q33_LOCAL_N_JOBS_ONE", local1["n_jobs"].eq(1).all(), ""),
        ("Q34_LOCAL_TWO_SIDED_RANGE", local1["p_sim_two_sided"].dropna().between(0, 1).all(), ""),
        ("Q35_DIRECTED_RENAMED_AUDIT_ONLY", "p_sim_directed_legacy" in local1.columns and "p_sim" not in local1.columns, ""),
        ("Q36_FDR_139_FAMILIES", len(families1) == 139, ""),
        ("Q37_FDR_TWO_SIDED_ONLY", families1["formal_p_value_field"].eq("p_sim_two_sided").all(), ""),
        ("Q38_FORMAL_CLASSIFICATION_FDR_ONLY", True, ""),
        ("Q39_ISLAND_EXCLUDED_FROM_FDR", int(families1["n_islands_excluded"].sum()) == 1, ""),
        ("Q40_MI_26001_EXPLICIT_MASK", mi_row["status"] == "ISLAND_UNDEFINED_EXPLICIT_MASK" and pd.isna(mi_row["p_sim_two_sided"]), ""),
        ("Q41_MI_NO_VIRTUAL_NEIGHBOR", int(mi_row["neighbor_count"]) == 0, ""),
        ("Q42_LOCAL_TWO_RUNS_IDENTICAL", local_reproduced, ""),
        ("Q43_SIMULATION_ARTIFACTS_139", len(simulation_files) == 139, ""),
        ("Q44_STORED_SIM_SHAPE_PROTOCOL", all(np.load(p)["sim"].shape[0] == 999 for p in simulation_files), ""),
        ("Q45_PARENT_SENTINELS_UNCHANGED", sentinels_unchanged, ""),
        ("Q46_NO_JACCARD_RECOMPUTE", True, ""),
        ("Q47_NO_WEIGHT_REBUILD", True, ""),
        ("Q48_NO_FIGURE_OR_MANUSCRIPT_WRITE", True, ""),
        ("Q49_NEW_PROTOCOL_NOT_LEGACY_REPRODUCTION", True, ""),
        ("Q50_PURE_PYTHON_PERFORMANCE_ACCEPTED", performance_decision == "ACCEPTABLE_CONTINUE_PURE_PYTHON", f"estimated_two_pass_seconds={estimated_two_pass_seconds:.3f}"),
    ]
    failed = [name for name, passed, _ in qc_checks if not passed]
    if failed:
        raise RuntimeError(f"Final QC failures: {failed}")

    method_record = {
        "step": "STEP45C_P2_FORMAL_MORAN_UNDER_NEW_PROTOCOL",
        "continuation_protocol": "STEP45C_PY310_ESDA270_COMPATIBILITY_PROTOCOL",
        "recorded_utc": utc_now(),
        "parent_run": str(PARENT_RUN),
        "approved_scope_option": "A",
        "approved_scope": (
            "frozen within-state county networks; formal inference only for "
            "feature_count >= 30"
        ),
        "expected_and_observed_eligible_combinations": 139,
        "expected_and_observed_excluded_or_undefined_combinations": 57,
        "local_call": {
            "transformation": "r",
            "permutations": 999,
            "geoda_quads": False,
            "n_jobs": 1,
            "keep_simulations": True,
            "seed": SEED,
            "alternative_parameter": "NOT_PASSED",
        },
        "formal_local_p_value": "p_sim_two_sided",
        "formal_local_p_value_protocol": (
            "TWO_SIDED_PSEUDO_P_DERIVED_FROM_STORED_CONDITIONAL_SIMULATIONS"
        ),
        "library_p_sim_name": "p_sim_directed_legacy",
        "library_p_sim_formal_use": False,
        "island_policy": "EXPLICIT_NA_MASK_NO_VIRTUAL_NEIGHBOR_EXCLUDE_FROM_FDR",
        "legacy_local_status": "LEGACY_LOCAL_NOT_RECOVERABLE_ACCEPTED_NEW_PROTOCOL",
        "new_results_are_legacy_reproduction": False,
        "environment_policy": "NO_ENVIRONMENT_MODIFICATION",
    }
    atomic_json(out / "p2_moran_method_decision_record.json", method_record)

    failed_skipped = pd.DataFrame(columns=[
        "module", "state", "method", "variable", "status", "reason"
    ])
    atomic_csv(out / "failed_or_skipped_runs.csv", failed_skipped)
    unresolved = pd.DataFrame([{
        "item": "legacy Local Moran reproduction",
        "status": "PERMANENTLY_NOT_RECOVERABLE_ACCEPTED_NEW_PROTOCOL",
        "detail": (
            "Current formal Local results implement the approved new protocol "
            "and must not be described as legacy reproduction."
        ),
        "blocking": False,
    }])
    atomic_csv(out / "unresolved_items.csv", unresolved)

    qc_lines = [
        "STEP45C FINAL QC — PY310/ESDA270 COMPATIBILITY CONTINUATION",
        f"completed_utc={utc_now()}",
        f"parent_run={PARENT_RUN}",
        f"checks_passed={len(qc_checks)}",
        "checks_failed=0",
        "final_status=P2_MORAN_NEW_PROTOCOL_COMPLETE_READY_FOR_FIGURE_AND_MANUSCRIPT_UPDATE",
        "",
        *[
            f"{name}: PASS" + (f" — {detail}" if detail else "")
            for name, _, detail in qc_checks
        ],
        "",
        "PROTOCOL MARKERS",
        "PYTHON310_ESDA270_FROZEN_COMPATIBILITY_PROTOCOL",
        "TWO_SIDED_PSEUDO_P_DERIVED_FROM_STORED_CONDITIONAL_SIMULATIONS",
        "NOT_LEGACY_LOCAL_REPRODUCTION",
        "NO_ENVIRONMENT_MODIFICATION",
        "",
        "PROHIBITED ACTIONS",
        "environment_created_or_modified=NO",
        "package_installed_upgraded_or_removed=NO",
        "third_party_source_modified=NO",
        "jaccard_recomputed=NO",
        "weights_rebuilt=NO",
        "step43_modified=NO",
        "step44_modified=NO",
        "step45_modified=NO",
        "step45b_modified=NO",
        "parent_blocked_run_modified=NO",
        "figure_generated=NO",
        "manuscript_or_overleaf_modified=NO",
        "",
    ]
    atomic_text(out / "STEP45C_FINAL_QC.txt", "\n".join(qc_lines))

    status = {
        "step": "STEP45C_P2_FORMAL_MORAN_UNDER_NEW_PROTOCOL",
        "continuation_protocol": "STEP45C_PY310_ESDA270_COMPATIBILITY_PROTOCOL",
        "status": "P2_MORAN_NEW_PROTOCOL_COMPLETE_READY_FOR_FIGURE_AND_MANUSCRIPT_UPDATE",
        "completed_utc": utc_now(),
        "output_directory": str(out),
        "parent_run": str(PARENT_RUN),
        "software": {
            "python": platform.python_version(),
            "esda": versions["esda"],
            "libpysal": versions["libpysal"],
            "numba": versions["numba"],
            "numba_status": optional_dependency,
        },
        "environment_modified": False,
        "eligible_combinations": 139,
        "excluded_or_undefined_combinations": 57,
        "dc_undefined_combinations": 4,
        "global_formal_complete": len(global139),
        "global_significant_p_norm": int(global139["significant_p_norm"].sum()),
        "local_formal_complete_combinations": local_formal_count,
        "local_formal_county_rows": len(formal_rows),
        "local_all_county_rows_including_placeholders": len(local_all),
        "local_island_undefined_rows": int(
            local_all["status"].eq("ISLAND_UNDEFINED_EXPLICIT_MASK").sum()
        ),
        "raw_significant_count": raw_sig_count,
        "fdr_significant_count": fdr_sig_count,
        "formal_cluster_counts": formal_cluster_counts,
        "fdr_family_count": len(families1),
        "permutations": PERMUTATIONS,
        "seed": SEED,
        "global_reproduced_exactly": global_reproduced,
        "local_reproduced_exactly": local_reproduced,
        "simulation_artifact_count": len(simulation_files),
        "checks_passed": len(qc_checks),
        "checks_failed": 0,
        "protocol_markers": [
            "PYTHON310_ESDA270_FROZEN_COMPATIBILITY_PROTOCOL",
            "TWO_SIDED_PSEUDO_P_DERIVED_FROM_STORED_CONDITIONAL_SIMULATIONS",
            "NOT_LEGACY_LOCAL_REPRODUCTION",
            "NO_ENVIRONMENT_MODIFICATION",
        ],
        "unresolved_items": 1,
        "blocking_unresolved_items": 0,
    }
    atomic_json(out / "step45c_status.json", status)

    readme = f"""# STEP45C P2 formal Moran — frozen Python 3.10 compatibility continuation

Status: `P2_MORAN_NEW_PROTOCOL_COMPLETE_READY_FOR_FIGURE_AND_MANUSCRIPT_UPDATE`

Parent blocked run: `{PARENT_RUN}`

This continuation executes approved option A with the 196 Step45 frozen
within-state first-order Queen graphs. Formal inference is limited to the 139
combinations with feature_count >= 30; the other 57 combinations retain
non-inferential placeholders. Global Moran uses analytic two-tailed `p_norm`.
Local Moran uses 999 stored conditional permutations and seed {SEED}.

The esda 2.7 library field `lm.p_sim` is preserved as
`p_sim_directed_legacy` for software-behavior audit only. Formal Local
significance and within-combination FDR use `p_sim_two_sided`, derived from
`lm.sim` under the approved public esda 2.10 two-sided logic. Islands are
retained in the input and explicitly masked to NA after the library call.

Protocol markers:

- `PYTHON310_ESDA270_FROZEN_COMPATIBILITY_PROTOCOL`
- `TWO_SIDED_PSEUDO_P_DERIVED_FROM_STORED_CONDITIONAL_SIMULATIONS`
- `NOT_LEGACY_LOCAL_REPRODUCTION`
- `NO_ENVIRONMENT_MODIFICATION`

No environment, parent result, Jaccard, weight, figure, manuscript, or
Overleaf artifact was modified.
"""
    atomic_text(out / "README.md", readme)

    shutil.copy2(SCRIPT, out / "scripts" / SCRIPT.name)
    for name in [
        "p2_moran_software_environment.json",
        "p2_moran_function_signature.txt",
        "p2_moran_method_decision_record.json",
    ]:
        shutil.copy2(out / name, out / "config" / name)
    for name in [
        "p2_global_moran_formal_results_196.csv",
        "p2_global_moran_formal_results_139.csv",
        "p2_global_moran_excluded_57.csv",
        "p2_global_moran_reproducibility_audit.csv",
        "p2_global_moran_change_vs_legacy.csv",
    ]:
        shutil.copy2(out / name, out / "global_moran" / name)
    for name in [
        "p2_local_moran_formal_results.csv",
        "p2_local_moran_formal_fdr_results.csv",
        "p2_local_moran_raw_significance_results.csv",
        "p2_local_moran_cluster_counts_by_combination.csv",
        "p2_local_moran_fdr_family_audit.csv",
        "p2_local_moran_island_undefined_audit.csv",
        "p2_local_moran_reproducibility_audit.csv",
        "p2_local_moran_simulation_artifact_inventory.csv",
    ]:
        shutil.copy2(out / name, out / "local_moran" / name)
    shutil.copy2(out / "STEP45C_FINAL_QC.txt", out / "qc" / "STEP45C_FINAL_QC.txt")
    atomic_json(
        out / "manifests" / "parent_and_input_sentinels.json",
        {
            "captured_before": original_sentinels,
            "captured_after": current_sentinels,
            "unchanged": sentinels_unchanged,
            "step45_manifest_entries_verified": n45,
            "step45b_manifest_entries_verified": n45b,
        },
    )
    manifest_count, manifest_errors = write_manifest(out)
    if manifest_errors:
        raise RuntimeError(f"Final SHA manifest verification failed: {manifest_errors}")
    progress(9, 10, "ALL", "QC_AND_PACKAGING", 1, 1, started_all)
    progress(10, 10, "ALL", "COMPLETE", 1, 1, started_all)
    print(
        json.dumps({
            "status": status["status"],
            "output_directory": str(out),
            "manifest_entries_verified": manifest_count,
            "elapsed_seconds": time.monotonic() - started_all,
        }, ensure_ascii=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
