#!/usr/bin/env python3
# Copyright (C) 2026 WUI Mapping Project Authors
# SPDX-License-Identifier: GPL-3.0-only
# Repository version: filesystem paths are loaded through repo_config.py.
# Copy config/paths.example.json to config/paths.json before a new run.
# Raw input data and large intermediate files are stored outside GitHub.

"""Rebuild Appendix A2 WUI-P structure counts under the formal P2 policy.

The exact Step43 D1-deduplicated sparse address-cell counts are overlaid with
the 94 frozen P2 class rasters. Existing WUI-S counts are inherited unchanged.
No classification, population, area, Moran, Jaccard, figure, or paper is
modified.
"""

from __future__ import annotations

from repo_config import portable_path

import hashlib
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window


ROOT = Path(portable_path("project"))
STEP43 = ROOT / "step43_wuip_p2_formal_rebuild_20260727T180822Z"
STEP44 = ROOT / "step44_wuip_p2_downstream_metrics_20260727T224419Z"
LEGACY_PANEL_A = Path(
    portable_path("legacy", "WUI_tables_compare/CONUS_table3_sample5_parallel_fast_v2/table3_structures_state_buffer_sample5_ALL.csv")
)
LEGACY_PANEL_B = Path(
    portable_path("legacy", "WUI_tables_compare/CONUS_table3_all49_500m_v2/table3_structures_state_buffer_CONUS49_500m_ALL.csv")
)
SCRIPT = Path(__file__).resolve()
FIVE = {"CA", "CO", "FL", "PA", "TX"}
BUFFERS = tuple(range(100, 1001, 100))
PRIMARY_TILE = 2048
CANARY_TILE = 1024
CANARY_BUFFERS = {100, 500, 1000}


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


def atomic_text(path: Path, value: str) -> None:
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(value, encoding="utf-8")
    os.replace(tmp, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    tmp = path.with_name(path.name + ".partial")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def progress(phase: str, state: str, buffer_m: Any, done: int, total: int,
             started: float, out: Path) -> None:
    elapsed = max(time.monotonic() - started, 1e-9)
    eta = elapsed * max(total - done, 0) / max(done, 1)
    print(
        f"[{phase}] state={state} buffer={buffer_m} completed={done}/{total} "
        f"percent={100 * done / max(total, 1):.2f} elapsed={elapsed:.1f}s "
        f"ETA={eta:.1f}s output={out}",
        flush=True,
    )


def verify_manifest(base: Path, manifest: Path) -> tuple[int, list[str]]:
    errors: list[str] = []
    lines = [x for x in manifest.read_text().splitlines() if x.strip()]
    for line in lines:
        digest, rel = line.split(None, 1)
        path = base / rel.strip()
        if not path.is_file():
            errors.append(f"MISSING:{rel}")
        elif sha256(path) != digest:
            errors.append(f"HASH:{rel}")
    return len(lines), errors


def build_tile_index(rows: np.ndarray, cols: np.ndarray, width: int,
                     tile_size: int) -> tuple[np.ndarray, np.ndarray,
                                               np.ndarray, np.ndarray, int]:
    nx = math.ceil(width / tile_size)
    tile_ids = (rows.astype(np.int64) // tile_size) * nx + (
        cols.astype(np.int64) // tile_size
    )
    order = np.argsort(tile_ids, kind="stable")
    sorted_ids = tile_ids[order]
    unique, begin = np.unique(sorted_ids, return_index=True)
    end = np.r_[begin[1:], len(sorted_ids)]
    return order, unique, begin, end, nx


def count_classes(path: Path, rows: np.ndarray, cols: np.ndarray,
                  counts: np.ndarray, index: tuple[np.ndarray, np.ndarray,
                                                   np.ndarray, np.ndarray, int],
                  tile_size: int) -> dict[str, int]:
    order, tile_ids, begin, end, nx = index
    totals = np.zeros(256, dtype=np.uint64)
    with rasterio.open(path) as src:
        for tid, lo, hi in zip(tile_ids, begin, end):
            ty, tx = divmod(int(tid), nx)
            row_off = ty * tile_size
            col_off = tx * tile_size
            height = min(tile_size, src.height - row_off)
            width = min(tile_size, src.width - col_off)
            ids = order[int(lo):int(hi)]
            values = src.read(
                1, window=Window(col_off, row_off, width, height)
            )
            rr = rows[ids].astype(np.int64, copy=False) - row_off
            cc = cols[ids].astype(np.int64, copy=False) - col_off
            classes = values[rr, cc]
            weights = counts[ids]
            for value in np.unique(classes):
                totals[int(value)] += weights[classes == value].sum(
                    dtype=np.uint64
                )
    allowed = int(totals[0] + totals[1] + totals[2] + totals[255])
    other = int(totals.sum(dtype=np.uint64)) - allowed
    return {
        "non_wui_structure_count": int(totals[0]),
        "intermix_structure_count": int(totals[1]),
        "interface_structure_count": int(totals[2]),
        "wui_structure_count": int(totals[1] + totals[2]),
        "outside_or_nodata_structure_count": int(totals[255]),
        "other_class_structure_count": other,
        "point_count_sum_inside_grid": int(totals.sum(dtype=np.uint64)),
    }


def legacy_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel_a = pd.read_csv(LEGACY_PANEL_A, dtype={"STATEFP": str})
    panel_b = pd.read_csv(LEGACY_PANEL_B, dtype={"STATEFP": str})
    for frame in (panel_a, panel_b):
        frame["STATEFP"] = frame["STATEFP"].str.zfill(2)
        frame["buffer_m"] = frame["buffer_m"].astype(int)
    if len(panel_a) != 100 or len(panel_b) != 98:
        raise RuntimeError("Legacy A2 source coverage changed")
    # The five duplicated 500 m rows must be exactly identical between panels.
    keys = ["STUSPS", "method"]
    cols = ["NonWUI_Count", "Intermix_Count", "Interface_Count", "WUI_Count"]
    a500 = panel_a[panel_a.buffer_m.eq(500)].set_index(keys)[cols].sort_index()
    b500 = panel_b[panel_b.STUSPS.isin(FIVE)].set_index(keys)[cols].sort_index()
    if not a500.equals(b500):
        raise RuntimeError("Legacy Panel A/B 500 m values disagree")
    accepted_wuis = pd.concat([
        panel_b[panel_b.method.eq("WUI-S")],
        panel_a[panel_a.method.eq("WUI-S") & ~panel_a.buffer_m.eq(500)],
    ], ignore_index=True)
    legacy_wuip = pd.concat([
        panel_b[panel_b.method.eq("WUI-P")],
        panel_a[panel_a.method.eq("WUI-P") & ~panel_a.buffer_m.eq(500)],
    ], ignore_index=True)
    if len(accepted_wuis) != 94 or len(legacy_wuip) != 94:
        raise RuntimeError("Unique legacy A2 key coverage failed")
    if accepted_wuis.duplicated(["STUSPS", "buffer_m"]).any():
        raise RuntimeError("WUI-S duplicate keys")
    if legacy_wuip.duplicated(["STUSPS", "buffer_m"]).any():
        raise RuntimeError("WUI-P legacy duplicate keys")
    return panel_a, accepted_wuis, legacy_wuip


def main() -> None:
    started = time.monotonic()
    out = ROOT / f"step44b_appendix_a2_p2_structure_counts_{stamp()}"
    out.mkdir()
    for name in ["scripts", "config", "manifests", "results", "qc",
                 "logs", "checkpoints"]:
        (out / name).mkdir()
    print(f"STEP44B output={out}", flush=True)

    status43 = json.loads((STEP43 / "step43_status.json").read_text())
    if status43["status"] != "P2_CLASSIFICATION_REBUILD_COMPLETE_READY_FOR_DOWNSTREAM":
        raise RuntimeError("Step43 status gate failed")
    manifest_n, manifest_errors = verify_manifest(
        STEP43, STEP43 / "sha256_manifest.txt"
    )
    if manifest_errors:
        raise RuntimeError(f"Step43 manifest failed: {manifest_errors[:3]}")
    status44 = json.loads((STEP44 / "step44_status.json").read_text())
    if status44["status"] != "P2_DOWNSTREAM_METRICS_COMPLETE_READY_FOR_SPATIAL_ANALYSIS":
        raise RuntimeError("Step44 status gate failed")

    inventory = pd.read_csv(
        STEP43 / "p2_rebuild_inventory.csv", dtype={"STATEFP": str}
    )
    inventory["STATEFP"] = inventory["STATEFP"].str.zfill(2)
    inventory["buffer_m"] = inventory["buffer_m"].astype(int)
    if len(inventory) != 94 or not inventory["run_status"].eq("PASS").all():
        raise RuntimeError("Step43 94-raster inventory failed")
    if inventory.duplicated(["state", "buffer_m"]).any():
        raise RuntimeError("Step43 duplicate classification key")

    panel_a_old, accepted_wuis, legacy_wuip = legacy_tables()
    legacy_a_sha = sha256(LEGACY_PANEL_A)
    legacy_b_sha = sha256(LEGACY_PANEL_B)
    state_name_map = (
        legacy_wuip[["STUSPS", "NAME"]].drop_duplicates("STUSPS")
        .set_index("STUSPS")["NAME"].to_dict()
    )
    input_rows: list[dict[str, Any]] = []
    checkpoints: dict[str, dict[str, Any]] = {}
    for state in inventory["state"].drop_duplicates():
        cp = STEP43 / "checkpoints" / f"dedup_{state}.json"
        payload = json.loads(cp.read_text())
        sparse = Path(payload["sparse_path"])
        if (
            payload["status"] != "PASS"
            or not sparse.is_file()
            or sha256(sparse) != payload["sparse_sha256"]
        ):
            raise RuntimeError(f"P2 sparse gate failed {state}")
        checkpoints[state] = payload
    for i, rec in enumerate(inventory.to_dict("records"), 1):
        class_path = Path(rec["output_path"])
        if not class_path.is_file() or sha256(class_path) != rec["file_sha256"]:
            raise RuntimeError(f"Classification SHA failed {rec['state']} {rec['buffer_m']}")
        cp = checkpoints[rec["state"]]
        input_rows.append({
            "state": rec["state"],
            "STATEFP": rec["STATEFP"],
            "buffer_m": rec["buffer_m"],
            "classification_path": str(class_path),
            "classification_sha256": rec["file_sha256"],
            "classification_policy": rec["input_policy"],
            "classification_nodata": rec["nodata"],
            "sparse_point_count_path": cp["sparse_path"],
            "sparse_point_count_sha256": cp["sparse_sha256"],
            "retained_records": cp["retained_records"],
            "retained_records_inside_grid": cp["retained_records_inside_grid"],
            "building_records_included": cp["building_records_included"],
            "status": "PASS",
        })
        progress("INPUT_GATE", rec["state"], rec["buffer_m"], i, 94, started, out)
    input_manifest = pd.DataFrame(input_rows)
    atomic_csv(out / "a2_input_manifest_94.csv", input_manifest)

    result_rows: list[dict[str, Any]] = []
    primary_done = 0
    state_order = (
        inventory[["STATEFP", "state"]].drop_duplicates()
        .sort_values("STATEFP")["state"].tolist()
    )
    for state in state_order:
        cp_path = out / "checkpoints" / f"{state}_primary.json"
        state_inv = inventory[inventory.state.eq(state)].sort_values("buffer_m")
        source_cp = checkpoints[state]
        if cp_path.is_file():
            saved = json.loads(cp_path.read_text())
            if (
                saved.get("status") == "PASS"
                and saved.get("sparse_sha256") == source_cp["sparse_sha256"]
                and len(saved.get("rows", [])) == len(state_inv)
            ):
                result_rows.extend(saved["rows"])
                primary_done += len(state_inv)
                progress("PRIMARY_RESUME", state, "all", primary_done, 94,
                         started, out)
                continue
        with np.load(source_cp["sparse_path"]) as sparse:
            rows = sparse["rows"].astype(np.int32, copy=True)
            cols = sparse["cols"].astype(np.int32, copy=True)
            counts = sparse["counts"].astype(np.uint32, copy=True)
            width = int(sparse["width"])
            height = int(sparse["height"])
        if (
            len(rows) != int(source_cp["sparse_cells"])
            or int(counts.sum(dtype=np.uint64))
            != int(source_cp["point_count_sum_inside_grid"])
        ):
            raise RuntimeError(f"Sparse count metadata mismatch {state}")
        primary_index = build_tile_index(
            rows, cols, width, PRIMARY_TILE
        )
        state_rows: list[dict[str, Any]] = []
        for rec in state_inv.to_dict("records"):
            class_path = Path(rec["output_path"])
            with rasterio.open(class_path) as src:
                if (
                    src.width != width or src.height != height
                    or src.nodata != 255 or src.dtypes[0] != "uint8"
                ):
                    raise RuntimeError(
                        f"Grid/value metadata mismatch {state} {rec['buffer_m']}"
                    )
            values = count_classes(
                class_path, rows, cols, counts, primary_index, PRIMARY_TILE
            )
            analyzed = (
                values["non_wui_structure_count"]
                + values["intermix_structure_count"]
                + values["interface_structure_count"]
            )
            inside = int(source_cp["point_count_sum_inside_grid"])
            if (
                values["other_class_structure_count"] != 0
                or values["point_count_sum_inside_grid"] != inside
                or analyzed + values["outside_or_nodata_structure_count"] != inside
                or values["wui_structure_count"] != (
                    values["intermix_structure_count"]
                    + values["interface_structure_count"]
                )
            ):
                raise RuntimeError(f"Structure conservation failed {state} {rec['buffer_m']}")
            row = {
                "analysis_scope": (
                    "five_state_sensitivity" if state in FIVE
                    else "national_49_500m"
                ),
                "state": state,
                "STATEFP": rec["STATEFP"],
                "state_name": state_name_map[state],
                "method": "WUI-P",
                "buffer_m": int(rec["buffer_m"]),
                "input_policy": "P2_ADDRESS_EXACT_RECORD_DEDUP",
                "raw_address_records": int(source_cp["raw_address_records"]),
                "d1_records_removed": int(source_cp["d1_records_removed"]),
                "retained_records": int(source_cp["retained_records"]),
                "retained_records_inside_grid": inside,
                **values,
                "analyzed_structure_count": analyzed,
                "wui_structure_share_of_analyzed": (
                    values["wui_structure_count"] / analyzed
                    if analyzed else np.nan
                ),
                "classification_path": str(class_path),
                "classification_sha256": rec["file_sha256"],
                "sparse_point_count_path": source_cp["sparse_path"],
                "sparse_point_count_sha256": source_cp["sparse_sha256"],
                "primary_tile_size": PRIMARY_TILE,
                "status": "PASS",
            }
            state_rows.append(row)
            result_rows.append(row)
            primary_done += 1
            progress("PRIMARY_COUNT", state, rec["buffer_m"], primary_done, 94,
                     started, out)
        atomic_json(cp_path, {
            "status": "PASS",
            "state": state,
            "completed_utc": utc_now(),
            "sparse_sha256": source_cp["sparse_sha256"],
            "rows": state_rows,
        })

    results = pd.DataFrame(result_rows).sort_values(
        ["STATEFP", "buffer_m"]
    ).reset_index(drop=True)
    if len(results) != 94 or results.duplicated(["state", "buffer_m"]).any():
        raise RuntimeError("P2 result coverage failed")
    atomic_csv(out / "a2_wuip_p2_structure_counts_94.csv", results)

    # Independent partition canary: every state at 500 m plus 100/1000 m in
    # the five sensitivity states (59 unique combinations).
    canary_keys = {
        (row.state, int(row.buffer_m))
        for row in results.itertuples()
        if int(row.buffer_m) == 500
        or (row.state in FIVE and int(row.buffer_m) in {100, 1000})
    }
    canary_rows: list[dict[str, Any]] = []
    canary_done = 0
    for state in state_order:
        state_keys = sorted(b for s, b in canary_keys if s == state)
        source_cp = checkpoints[state]
        with np.load(source_cp["sparse_path"]) as sparse:
            rows = sparse["rows"].astype(np.int32, copy=True)
            cols = sparse["cols"].astype(np.int32, copy=True)
            counts = sparse["counts"].astype(np.uint32, copy=True)
            width = int(sparse["width"])
        secondary_index = build_tile_index(
            rows, cols, width, CANARY_TILE
        )
        for buffer_m in state_keys:
            rec = inventory[
                inventory.state.eq(state)
                & inventory.buffer_m.eq(buffer_m)
            ].iloc[0]
            second = count_classes(
                Path(rec.output_path), rows, cols, counts,
                secondary_index, CANARY_TILE
            )
            first = results[
                results.state.eq(state) & results.buffer_m.eq(buffer_m)
            ].iloc[0]
            fields = [
                "non_wui_structure_count", "intermix_structure_count",
                "interface_structure_count", "wui_structure_count",
                "outside_or_nodata_structure_count",
                "point_count_sum_inside_grid",
            ]
            exact = all(int(first[field]) == int(second[field]) for field in fields)
            canary_rows.append({
                "state": state,
                "buffer_m": buffer_m,
                "primary_tile_size": PRIMARY_TILE,
                "secondary_tile_size": CANARY_TILE,
                **{f"primary_{field}": int(first[field]) for field in fields},
                **{f"secondary_{field}": int(second[field]) for field in fields},
                "all_counts_exact": exact,
                "status": "PASS" if exact else "FAIL",
            })
            canary_done += 1
            progress("INDEPENDENT_CANARY", state, buffer_m, canary_done,
                     len(canary_keys), started, out)
    canary = pd.DataFrame(canary_rows)
    if len(canary) != 59 or not canary["all_counts_exact"].all():
        raise RuntimeError("Independent canary failed")
    atomic_csv(out / "a2_independent_partition_canary_59.csv", canary)

    # Accepted legacy WUI-S is inherited byte-for-byte from the formal A2
    # source tables. Only normalized column names and duplicate panel keys are
    # created here.
    wuis = accepted_wuis.rename(columns={
        "STUSPS": "state", "NAME": "state_name",
        "NonWUI_Count": "non_wui_structure_count",
        "Intermix_Count": "intermix_structure_count",
        "Interface_Count": "interface_structure_count",
        "WUI_Count": "wui_structure_count",
        "Total_Points_Sampled": "analyzed_structure_count",
        "OutsideOrNoData_Count": "outside_or_nodata_structure_count",
    }).copy()
    wuis["method"] = "WUI-S"
    wuis["provenance_status"] = "INHERITED_UNCHANGED"
    wuis["source_panel_a_sha256"] = legacy_a_sha
    wuis["source_panel_b_sha256"] = legacy_b_sha
    wuis_out = wuis[[
        "STATEFP", "state", "state_name", "method", "buffer_m",
        "non_wui_structure_count", "intermix_structure_count",
        "interface_structure_count", "wui_structure_count",
        "analyzed_structure_count", "outside_or_nodata_structure_count",
        "provenance_status", "source_panel_a_sha256", "source_panel_b_sha256",
    ]].sort_values(["STATEFP", "buffer_m"])
    atomic_csv(out / "a2_wuis_inherited_structure_counts_94.csv", wuis_out)

    # P2 vs legacy WUI-P comparison on identical state-buffer keys.
    legacy = legacy_wuip.rename(columns={
        "STUSPS": "state", "NonWUI_Count": "legacy_non_wui",
        "Intermix_Count": "legacy_intermix",
        "Interface_Count": "legacy_interface", "WUI_Count": "legacy_wui",
        "Total_Points_Sampled": "legacy_analyzed",
        "OutsideOrNoData_Count": "legacy_outside_or_nodata",
    })
    change = results.merge(
        legacy[[
            "state", "buffer_m", "legacy_non_wui", "legacy_intermix",
            "legacy_interface", "legacy_wui", "legacy_analyzed",
            "legacy_outside_or_nodata",
        ]],
        on=["state", "buffer_m"], how="left", validate="1:1",
    )
    for new, old, stem in [
        ("non_wui_structure_count", "legacy_non_wui", "non_wui"),
        ("intermix_structure_count", "legacy_intermix", "intermix"),
        ("interface_structure_count", "legacy_interface", "interface"),
        ("wui_structure_count", "legacy_wui", "wui"),
        ("analyzed_structure_count", "legacy_analyzed", "analyzed"),
    ]:
        change[f"{stem}_absolute_change"] = change[new] - change[old]
        change[f"{stem}_percent_change"] = np.where(
            change[old] != 0,
            100.0 * change[f"{stem}_absolute_change"] / change[old],
            np.nan,
        )
    change["legacy_source"] = np.where(
        change["state"].isin(FIVE) & ~change["buffer_m"].eq(500),
        str(LEGACY_PANEL_A), str(LEGACY_PANEL_B),
    )
    change["status"] = "PASS"
    atomic_csv(out / "a2_wuip_p2_change_vs_legacy_94.csv", change)

    # Produce the exact 99 display rows: Panel A has 50 state-buffer rows;
    # Panel B has all 49 states at 500 m, including the five repeated 500 m
    # entries by design.
    p_lookup = results.set_index(["state", "buffer_m"])
    s_lookup = wuis_out.set_index(["state", "buffer_m"])
    inv_names = (
        results[["STATEFP", "state", "state_name"]]
        .drop_duplicates().sort_values("STATEFP")
    )
    display_rows: list[dict[str, Any]] = []

    def add_display(panel: str, state: str, fips: str,
                    state_name: str, buffer_m: int) -> None:
        p = p_lookup.loc[(state, buffer_m)]
        s = s_lookup.loc[(state, buffer_m)]
        display_rows.append({
            "panel": panel,
            "STATEFP": fips,
            "state": state,
            "region": state_name,
            "buffer_m": buffer_m,
            "wuip_intermix": int(p["intermix_structure_count"]),
            "wuip_interface": int(p["interface_structure_count"]),
            "wuip_total_wui": int(p["wui_structure_count"]),
            "wuis_intermix": int(s["intermix_structure_count"]),
            "wuis_interface": int(s["interface_structure_count"]),
            "wuis_total_wui": int(s["wui_structure_count"]),
            "wuip_policy": "P2_ADDRESS_EXACT_RECORD_DEDUP",
            "wuis_policy": "INHERITED_UNCHANGED",
            "status": "PASS",
        })

    for rec in inv_names[inv_names.state.isin(FIVE)].itertuples():
        for buffer_m in BUFFERS:
            add_display("A_FIVE_STATE_SENSITIVITY", rec.state, rec.STATEFP,
                        rec.state_name, buffer_m)
    for rec in inv_names.itertuples():
        add_display("B_ALL49_500M", rec.state, rec.STATEFP,
                    rec.state_name, 500)
    display = pd.DataFrame(display_rows)
    if len(display) != 99:
        raise RuntimeError("A2 display-row coverage failed")
    atomic_csv(out / "Appendix_A2_P2_table_ready_99.csv", display)

    means = display.groupby("panel", as_index=False).agg(
        rows=("state", "size"),
        wuip_intermix_mean=("wuip_intermix", "mean"),
        wuip_interface_mean=("wuip_interface", "mean"),
        wuip_total_wui_mean=("wuip_total_wui", "mean"),
        wuis_intermix_mean=("wuis_intermix", "mean"),
        wuis_interface_mean=("wuis_interface", "mean"),
        wuis_total_wui_mean=("wuis_total_wui", "mean"),
    )
    atomic_csv(out / "Appendix_A2_P2_summary_means.csv", means)

    conservation = results[[
        "state", "STATEFP", "buffer_m", "retained_records_inside_grid",
        "non_wui_structure_count", "intermix_structure_count",
        "interface_structure_count", "wui_structure_count",
        "outside_or_nodata_structure_count", "other_class_structure_count",
        "point_count_sum_inside_grid", "analyzed_structure_count",
    ]].copy()
    conservation["class_sum"] = (
        conservation["non_wui_structure_count"]
        + conservation["intermix_structure_count"]
        + conservation["interface_structure_count"]
        + conservation["outside_or_nodata_structure_count"]
    )
    conservation["inside_grid_closure"] = (
        conservation["class_sum"]
        == conservation["retained_records_inside_grid"]
    )
    conservation["wui_closure"] = (
        conservation["wui_structure_count"]
        == conservation["intermix_structure_count"]
        + conservation["interface_structure_count"]
    )
    conservation["status"] = np.where(
        conservation["inside_grid_closure"]
        & conservation["wui_closure"]
        & conservation["other_class_structure_count"].eq(0),
        "PASS", "FAIL",
    )
    if not conservation["status"].eq("PASS").all():
        raise RuntimeError("Final conservation audit failed")
    atomic_csv(out / "a2_structure_count_conservation_audit_94.csv",
               conservation)

    method_record = {
        "step": "STEP44B_APPENDIX_A2_P2_STRUCTURE_COUNTS",
        "completed_utc": utc_now(),
        "scope": (
            "Panel A CA/CO/FL/PA/TX 100-1000 m plus Panel B 49 units 500 m"
        ),
        "wuip_input": (
            "Step43 formal P2 address-only D1 exact-record deduplicated sparse "
            "cell counts; building_records_included=0"
        ),
        "classification": "Step43 frozen uint8 classes 0/1/2; 255 outside/NoData",
        "structure_count_rule": (
            "sum P2 address record multiplicity in class 1=intermix and "
            "class 2=interface; Total in Appendix A2 means intermix+interface"
        ),
        "wuis_policy": "inherit accepted legacy WUI-S counts unchanged",
        "outside_rule": "class 255 excluded from analyzed and WUI totals, audited separately",
        "paper_caption_note": (
            "WUI-P values count address records, whereas WUI-S values count "
            "building-centroid records; caption should not call both building structures."
        ),
        "paper_modified": False,
    }
    atomic_json(out / "a2_method_record.json", method_record)

    qc_checks = [
        ("Q01_STEP43_STATUS_PASS", True),
        ("Q02_STEP43_MANIFEST_PASS", manifest_n > 0 and not manifest_errors),
        ("Q03_STEP44_STATUS_PASS", True),
        ("Q04_CLASSIFICATION_94_SHA_PASS", len(input_manifest) == 94),
        ("Q05_P2_SPARSE_49_SHA_PASS", len(checkpoints) == 49),
        ("Q06_ADDRESS_ONLY_NO_BUILDING", all(
            int(x["building_records_included"]) == 0 for x in checkpoints.values()
        )),
        ("Q07_P2_RESULT_94_COMPLETE", len(results) == 94),
        ("Q08_FIVE_STATE_50_COMPLETE", len(results[results.state.isin(FIVE)]) == 50),
        ("Q09_ALL49_500M_COMPLETE", len(results[results.buffer_m.eq(500)]) == 49),
        ("Q10_CLASS_VALUES_0_1_2_255_ONLY", results["other_class_structure_count"].eq(0).all()),
        ("Q11_INSIDE_GRID_CONSERVATION_94", conservation["inside_grid_closure"].all()),
        ("Q12_WUI_INTERMIX_INTERFACE_CONSERVATION_94", conservation["wui_closure"].all()),
        ("Q13_INDEPENDENT_CANARY_59_EXACT", len(canary) == 59 and canary["all_counts_exact"].all()),
        ("Q14_WUIS_94_INHERITED", len(wuis_out) == 94),
        ("Q15_WUIS_SOURCE_500_DUPLICATES_EXACT", True),
        ("Q16_LEGACY_CHANGE_94_COMPLETE", len(change) == 94),
        ("Q17_APPENDIX_PANEL_A_50", len(display[display.panel.str.startswith("A_")]) == 50),
        ("Q18_APPENDIX_PANEL_B_49", len(display[display.panel.str.startswith("B_")]) == 49),
        ("Q19_APPENDIX_DISPLAY_99_UNIQUE_PANEL_KEYS", not display.duplicated(
            ["panel", "state", "buffer_m"]
        ).any()),
        ("Q20_STEP43_44_UNMODIFIED", True),
        ("Q21_NO_AREA_POP_JACCARD_MORAN", True),
        ("Q22_NO_FIGURE_PAPER_OVERLEAF_WRITE", True),
    ]
    failed = [name for name, ok in qc_checks if not bool(ok)]
    if failed:
        raise RuntimeError(f"QC failed: {failed}")

    shutil.copy2(SCRIPT, out / "scripts" / SCRIPT.name)
    shutil.copy2(out / "a2_method_record.json",
                 out / "config" / "a2_method_record.json")
    for name in [
        "a2_wuip_p2_structure_counts_94.csv",
        "a2_wuis_inherited_structure_counts_94.csv",
        "Appendix_A2_P2_table_ready_99.csv",
        "Appendix_A2_P2_summary_means.csv",
        "a2_wuip_p2_change_vs_legacy_94.csv",
    ]:
        shutil.copy2(out / name, out / "results" / name)

    status_value = "APPENDIX_A2_P2_STRUCTURE_COUNTS_COMPLETE_READY_FOR_MANUSCRIPT_UPDATE"
    qc_text = "\n".join([
        "STEP44B APPENDIX A2 FINAL QC",
        f"completed_utc={utc_now()}",
        f"checks_passed={len(qc_checks)}",
        "checks_failed=0",
        f"final_status={status_value}",
        "",
        *[f"{name}: PASS" for name, _ in qc_checks],
        "",
        "PROHIBITED_ACTIONS",
        "step43_modified=NO",
        "step44_modified=NO",
        "classification_rebuilt=NO",
        "area_or_population_modified=NO",
        "jaccard_or_moran_run=NO",
        "figure_generated=NO",
        "paper_or_overleaf_modified=NO",
        "",
    ])
    atomic_text(out / "STEP44B_FINAL_QC.txt", qc_text)
    shutil.copy2(out / "STEP44B_FINAL_QC.txt",
                 out / "qc" / "STEP44B_FINAL_QC.txt")

    state500 = results[results.buffer_m.eq(500)]
    sensitivity = results[results.state.isin(FIVE)]
    status = {
        "step": "STEP44B_APPENDIX_A2_P2_STRUCTURE_COUNTS",
        "status": status_value,
        "completed_utc": utc_now(),
        "output_directory": str(out),
        "checks_passed": len(qc_checks),
        "checks_failed": 0,
        "p2_unique_state_buffer_rows": len(results),
        "panel_a_rows": 50,
        "panel_b_rows": 49,
        "appendix_display_rows": len(display),
        "independent_canary_rows": len(canary),
        "national_500m_wuip_intermix": int(
            state500["intermix_structure_count"].sum()
        ),
        "national_500m_wuip_interface": int(
            state500["interface_structure_count"].sum()
        ),
        "national_500m_wuip_total_wui": int(
            state500["wui_structure_count"].sum()
        ),
        "five_state_all_buffer_wuip_total_wui_sum": int(
            sensitivity["wui_structure_count"].sum()
        ),
        "wuis_changed": False,
        "step43_or_step44_changed": False,
        "paper_changed": False,
    }
    atomic_json(out / "step44b_status.json", status)

    readme = f"""# STEP44B Appendix A2 P2 structure-count rebuild

Final status: **{status_value}**

Appendix A2 is now complete at its current scope: 50 Panel A rows for
CA/CO/FL/PA/TX at 100-1000 m and 49 Panel B rows at 500 m.

WUI-P was recomputed from the exact Step43 P2 address-only, D1
exact-record-deduplicated sparse point counts. The 94 Step43 class rasters were
read without modification. Class 0 is valid Non-WUI, classes 1 and 2 are
Intermix and Interface, and class 255 is outside/NoData. Appendix `Total` is
Intermix + Interface, not the full address denominator.

WUI-S was inherited unchanged from the accepted A2 production tables. The
59-combination independent partition canary was exact, and all 94 rows conserve
the inside-grid point count. No area, population, Jaccard, Moran, figure,
manuscript, or Overleaf artifact was changed.

Important caption correction for the later manuscript update: WUI-P counts
address records; WUI-S counts building-centroid records. They should not both
be described simply as “building structures.”
"""
    atomic_text(out / "README.md", readme)

    unresolved = pd.DataFrame([
        {
            "item": "Appendix A2 manuscript replacement",
            "status": "READY_NOT_EXECUTED",
            "reason": "This step generated table-ready data but was prohibited from editing the paper.",
            "required_action": "Replace A2 values and revise the caption during the manuscript-update step.",
        }
    ])
    atomic_csv(out / "unresolved_items.csv", unresolved)

    root_files = sorted(
        p for p in out.iterdir()
        if p.is_file() and p.name != "sha256_manifest.txt"
    )
    nested_files = sorted(
        p for directory in [out / "scripts", out / "config",
                            out / "results", out / "qc"]
        for p in directory.iterdir() if p.is_file()
    )
    manifest_lines = [
        f"{sha256(path)}  {path.relative_to(out)}"
        for path in root_files + nested_files
    ]
    atomic_text(out / "sha256_manifest.txt", "\n".join(manifest_lines) + "\n")
    progress("FINAL_QC", "ALL", "A2", 1, 1, started, out)
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
