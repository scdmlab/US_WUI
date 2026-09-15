#!/usr/bin/env python3
# Copyright (C) 2026 WUI Mapping Project Authors
# SPDX-License-Identifier: GPL-3.0-only
# -*- coding: utf-8 -*-
# Repository version: filesystem paths are loaded through repo_config.py.
# Copy config/paths.example.json to config/paths.json before a new run.
# Raw input data and large intermediate files are stored outside GitHub.

"""
Step 25 - exact point-in-polygon audit for the Vermont WUI-S Step 24 review.

This script reprocesses every Vermont building representative point. It keeps
the Step 24 WUI-class lookup, but replaces the 30 m block-group lookup raster
with exact point-versus-2020-Census-block geometry tests. It then:

1. classifies each point as uniquely assigned, boundary-ambiguous, or unmatched;
2. compares the exact block-group assignment with the Step 24 raster assignment;
3. recomputes the paper-style block-group population allocation from exact,
   uniquely assigned points; and
4. writes a new audit package without changing Step 24, Step 23, source GPKGs,
   WUI rasters, or any legacy/national result.

Boundary rule
-------------
The exact test uses polygon ``covers(point)`` / point ``covered_by(polygon)``,
so polygon-boundary points are included. If a point is covered by blocks from
more than one block group, it is reported as ambiguous and is not silently
assigned by feature order.
"""

from __future__ import annotations

from repo_config import portable_path

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import fiona
import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from shapely.geometry import Point, shape
from shapely.strtree import STRtree


PROJECT_ROOT = Path(portable_path("project"))
DEFAULT_BLOCKS = Path(
    portable_path("data", "MBF+NLCD_2022US/Inputs/Processed_GPKG/tl_2022_50_tabblock20.gpkg")
)
HELPER_NAME = "08_recompute_sample4_ps_population.py"
STATEFP = "50"
STUSPS = "VT"
STATE_NAME = "Vermont"
METHOD = "WUI-S"
OFFICIAL_POPULATION = 643_077.0
PASS_VERDICT = "VERMONT_WUIS_EXACT_POINT_IN_POLYGON_AUDIT_PASS"
REVIEW_VERDICT = "VERMONT_WUIS_EXACT_POINT_IN_POLYGON_AUDIT_REVIEW"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    partial = path.with_name(path.name + ".partial")
    if path.exists() or partial.exists():
        raise FileExistsError(path)
    partial.write_text(text, encoding="utf-8")
    os.replace(partial, path)


def atomic_write_json(path: Path, payload: dict) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
    )


def load_helper(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Required sibling helper is missing: {path}")
    spec = importlib.util.spec_from_file_location("step08_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 25: exact Vermont WUI-S point-in-polygon audit"
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--step24-run", default=None)
    parser.add_argument("--blocks", default=str(DEFAULT_BLOCKS))
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--progress-every", type=int, default=25_000)
    return parser.parse_args()


def find_step24_run(project_root: Path, explicit: str | None) -> tuple[Path, pd.DataFrame]:
    candidates = (
        [Path(explicit).resolve()]
        if explicit
        else sorted(
            project_root.glob("step24_point_vs_pixel_method_*"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    )
    rejected = []
    for run_dir in candidates:
        summary_path = run_dir / "step24_state_method_summary.csv"
        detail_path = run_dir / "step24_block_group_detail.csv.gz"
        if not summary_path.is_file() or not detail_path.is_file():
            rejected.append(f"{run_dir}: missing summary/detail checkpoint")
            continue
        summary = pd.read_csv(summary_path, dtype={"STATEFP": str})
        summary["STATEFP"] = summary["STATEFP"].astype(str).str.zfill(2)
        if len(summary) != 10:
            rejected.append(f"{run_dir}: expected 10 rows, found {len(summary)}")
            continue
        rows = summary[
            (summary["STATEFP"] == STATEFP) & (summary["method"] == METHOD)
        ]
        if len(rows) != 1:
            rejected.append(f"{run_dir}: missing unique Vermont WUI-S row")
            continue
        return run_dir, rows.reset_index(drop=True)
    raise FileNotFoundError(
        "No complete Step 24 run was found.\n- " + "\n- ".join(rejected[:20])
    )


def find_bg_cache(run_dir: Path) -> Path:
    exact = run_dir / "cache" / "bg_code_50_wuis_aligned_500m.tif"
    if exact.is_file():
        return exact
    matches = list((run_dir / "cache").glob("bg_code_50_wuis_*.tif"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one Vermont WUI-S block-group cache, found {len(matches)}"
        )
    return matches[0]


def normalized_crs(helper, value, label: str) -> CRS:
    crs = CRS.from_user_input(value)
    normalized, repaired = helper.normalized_sampling_crs(crs, label)
    if repaired:
        print(f"[CRS] Repaired known legacy EPSG:5070 metadata for {label}")
    return normalized


def load_block_index(
    helper,
    blocks_path: Path,
    target_crs: CRS,
) -> tuple[STRtree, list, list[str], list[int], dict]:
    layer = helper.first_layer(blocks_path)
    fields = helper.schema_fields(blocks_path, layer)
    geoid_field = helper.pick_field(fields, ["GEOID20", "GEOID"], "block GEOID")
    geometries = []
    geoids12: list[str] = []
    repaired_invalid = 0
    skipped_empty = 0

    print(f"[INDEX] Loading exact Census block geometries: {blocks_path}", flush=True)
    with fiona.open(str(blocks_path), layer=layer) as source:
        source_crs = normalized_crs(
            helper, source.crs_wkt or source.crs, "Census blocks"
        )
        if source_crs != target_crs:
            raise RuntimeError(
                f"Census blocks CRS {source_crs} differs from point CRS {target_crs}; "
                "refusing an implicit geometry transform"
            )
        for feature in source:
            if feature["geometry"] is None:
                skipped_empty += 1
                continue
            geom = shape(feature["geometry"])
            if geom.is_empty:
                skipped_empty += 1
                continue
            if not geom.is_valid:
                fixed = geom.buffer(0)
                if fixed.is_empty or not fixed.is_valid:
                    raise RuntimeError(
                        f"Unrepairable Census block geometry at feature {feature.id}"
                    )
                geom = fixed
                repaired_invalid += 1
            geoid = str(feature["properties"][geoid_field]).replace(".0", "").zfill(15)
            geoid12 = geoid[:12]
            if len(geoid12) != 12 or not geoid12.isdigit():
                raise RuntimeError(f"Invalid block GEOID: {geoid!r}")
            geometries.append(geom)
            geoids12.append(geoid12)

    if not geometries:
        raise RuntimeError("No usable Census block geometries were loaded")
    bg_codes = [helper.bg_code_from_geoid12(value) for value in geoids12]
    tree = STRtree(geometries)
    id_to_index = {id(geom): index for index, geom in enumerate(geometries)}
    stats = {
        "block_geometry_count": len(geometries),
        "invalid_block_geometries_repaired": repaired_invalid,
        "empty_block_geometries_skipped": skipped_empty,
    }
    print(
        f"[INDEX] Blocks={len(geometries):,}; repaired={repaired_invalid:,}; "
        f"empty skipped={skipped_empty:,}",
        flush=True,
    )
    return tree, geometries, geoids12, bg_codes, id_to_index, stats


def exact_candidates(
    tree: STRtree,
    geometries: list,
    point: Point,
    id_to_index: dict,
) -> list[int]:
    """Return indices of polygons that cover point (Shapely 1.x or 2.x)."""
    try:
        result = tree.query(point, predicate="covered_by")
        return [int(value) for value in np.asarray(result).tolist()]
    except TypeError:
        candidates = tree.query(point)
        indices = []
        for geom in candidates:
            index = id_to_index.get(id(geom))
            if index is None:
                raise RuntimeError("STRtree returned an unrecognized geometry object")
            if geom.covers(point):
                indices.append(index)
        return indices


def classify_wui(values: np.ndarray, inbounds: np.ndarray) -> tuple[np.ndarray, int]:
    classes = np.zeros(len(values), dtype=np.int8)
    classes[values == 1] = 1
    classes[values == 2] = 2
    unexpected = inbounds & ~np.isin(values, [0, 1, 2, 255])
    return classes, int(unexpected.sum())


def run_audit(
    helper,
    row: pd.Series,
    blocks_path: Path,
    bg_cache: Path,
    audit_path: Path,
    chunk_size: int,
    tile_size: int,
    progress_every: int,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    structures = Path(row["structures_source"])
    wui_path = Path(row["wui_raster"])
    for label, path in (
        ("Vermont WUI-S source", structures),
        ("Vermont WUI-S raster", wui_path),
        ("Step 24 block-group cache", bg_cache),
        ("Census blocks", blocks_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    structures_layer = helper.first_layer(structures)
    feature_count, source_crs_wkt = helper.layer_info(structures, structures_layer)
    if not source_crs_wkt:
        raise RuntimeError("Vermont WUI-S source has no CRS")

    with rasterio.open(wui_path) as wui:
        if wui.crs is None:
            raise RuntimeError("Vermont WUI-S raster has no CRS")
        target_crs = normalized_crs(helper, wui.crs, "WUI raster")
    source_crs = normalized_crs(helper, CRS.from_wkt(source_crs_wkt), "WUI-S source")
    if source_crs != target_crs:
        raise RuntimeError(
            "This exact audit requires the already observed common EPSG:5070 CRS"
        )

    (
        tree,
        block_geometries,
        block_geoids12,
        block_bg_codes,
        id_to_index,
        block_stats,
    ) = load_block_index(helper, blocks_path, target_crs)

    bg, _, _ = helper.load_block_group_population(
        blocks_path, helper.first_layer(blocks_path)
    )
    bg_codes_sorted = bg["BG_CODE"].to_numpy(dtype=np.int64)
    total = np.zeros(len(bg), dtype=np.int64)
    c0 = np.zeros(len(bg), dtype=np.int64)
    c1 = np.zeros(len(bg), dtype=np.int64)
    c2 = np.zeros(len(bg), dtype=np.int64)

    stats = {
        **block_stats,
        "structure_features_expected": int(feature_count),
        "points_processed": 0,
        "points_in_raster_bounds": 0,
        "points_exact_unique_block_group": 0,
        "points_exact_ambiguous_block_group": 0,
        "points_exact_unmatched_block_group": 0,
        "raster_bg_zero_or_unknown": 0,
        "raster_and_exact_same_bg": 0,
        "raster_and_exact_different_bg": 0,
        "raster_missing_exact_recovered": 0,
        "unexpected_wui_values": 0,
    }
    fieldnames = [
        "point_sequence",
        "x",
        "y",
        "in_wui_raster_bounds",
        "wui_class",
        "raster_bg_code",
        "exact_status",
        "exact_geoid12",
        "exact_bg_code",
        "exact_candidate_bg_count",
        "raster_exact_relation",
    ]
    partial = audit_path.with_name(audit_path.name + ".partial")
    if audit_path.exists() or partial.exists():
        raise FileExistsError(audit_path)

    started = time.monotonic()
    next_progress = progress_every
    point_sequence = 0
    with (
        rasterio.open(wui_path) as wui,
        rasterio.open(bg_cache) as bg_raster,
        gzip.open(partial, "wt", encoding="utf-8", newline="") as stream,
    ):
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        target_crs_wkt = target_crs.to_wkt()
        for xs, ys in helper.iter_representative_points(
            structures,
            structures_layer,
            source_crs_wkt,
            target_crs_wkt,
            chunk_size,
        ):
            wui_values, raster_bg_values, inbounds = (
                helper.sample_aligned_rasters_by_tiles(
                    wui, bg_raster, xs, ys, tile_size
                )
            )
            classes, unexpected = classify_wui(wui_values, inbounds)
            stats["unexpected_wui_values"] += unexpected
            stats["points_in_raster_bounds"] += int(inbounds.sum())

            rows_to_write = []
            for local_index, (x, y) in enumerate(zip(xs, ys)):
                point_sequence += 1
                point = Point(float(x), float(y))
                polygon_indices = exact_candidates(
                    tree, block_geometries, point, id_to_index
                )
                candidate_geoids = sorted(
                    {block_geoids12[index] for index in polygon_indices}
                )
                raster_code = int(raster_bg_values[local_index])
                if raster_code == 0 or raster_code not in bg_codes_sorted:
                    stats["raster_bg_zero_or_unknown"] += 1

                if len(candidate_geoids) == 1:
                    exact_status = "UNIQUE"
                    exact_geoid = candidate_geoids[0]
                    exact_code = int(helper.bg_code_from_geoid12(exact_geoid))
                    stats["points_exact_unique_block_group"] += 1

                    dense = int(np.searchsorted(bg_codes_sorted, exact_code))
                    if dense >= len(bg_codes_sorted) or bg_codes_sorted[dense] != exact_code:
                        raise RuntimeError(
                            f"Exact BG code absent from population table: {exact_geoid}"
                        )
                    cls = int(classes[local_index])
                    total[dense] += 1
                    if cls == 0:
                        c0[dense] += 1
                    elif cls == 1:
                        c1[dense] += 1
                    elif cls == 2:
                        c2[dense] += 1
                    else:
                        raise RuntimeError(f"Unexpected normalized WUI class: {cls}")

                    if raster_code == exact_code:
                        relation = "SAME_BG"
                        stats["raster_and_exact_same_bg"] += 1
                    elif raster_code == 0 or raster_code not in bg_codes_sorted:
                        relation = "RASTER_MISSING_EXACT_RECOVERED"
                        stats["raster_missing_exact_recovered"] += 1
                    else:
                        relation = "DIFFERENT_BG"
                        stats["raster_and_exact_different_bg"] += 1
                elif len(candidate_geoids) > 1:
                    exact_status = "AMBIGUOUS_BOUNDARY"
                    exact_geoid = ";".join(candidate_geoids)
                    exact_code = ""
                    relation = "EXACT_AMBIGUOUS"
                    stats["points_exact_ambiguous_block_group"] += 1
                else:
                    exact_status = "NO_COVERING_BLOCK"
                    exact_geoid = ""
                    exact_code = ""
                    relation = "EXACT_UNMATCHED"
                    stats["points_exact_unmatched_block_group"] += 1

                rows_to_write.append(
                    {
                        "point_sequence": point_sequence,
                        "x": f"{float(x):.9f}",
                        "y": f"{float(y):.9f}",
                        "in_wui_raster_bounds": bool(inbounds[local_index]),
                        "wui_class": int(classes[local_index]),
                        "raster_bg_code": raster_code,
                        "exact_status": exact_status,
                        "exact_geoid12": exact_geoid,
                        "exact_bg_code": exact_code,
                        "exact_candidate_bg_count": len(candidate_geoids),
                        "raster_exact_relation": relation,
                    }
                )
            writer.writerows(rows_to_write)
            stats["points_processed"] += len(xs)
            if stats["points_processed"] >= next_progress:
                elapsed = max(time.monotonic() - started, 1e-9)
                rate = stats["points_processed"] / elapsed
                eta = (feature_count - stats["points_processed"]) / max(rate, 1e-9)
                print(
                    f"[{stats['points_processed']:,}/{feature_count:,} "
                    f"{100*stats['points_processed']/feature_count:.2f}% | "
                    f"elapsed {elapsed/60:.2f}m | ETA {eta/60:.2f}m] "
                    "exact point-in-polygon audit",
                    flush=True,
                )
                while next_progress <= stats["points_processed"]:
                    next_progress += progress_every

    os.replace(partial, audit_path)
    if stats["points_processed"] != feature_count:
        raise RuntimeError(
            f"Processed {stats['points_processed']:,} of {feature_count:,} features"
        )
    stats["representative_point_coverage_pct"] = (
        100.0 * stats["points_processed"] / feature_count if feature_count else 100.0
    )
    stats["exact_block_group_match_pct"] = (
        100.0 * stats["points_exact_unique_block_group"] / stats["points_processed"]
        if stats["points_processed"]
        else 0.0
    )
    stats["elapsed_minutes"] = (time.monotonic() - started) / 60.0
    return stats, bg, total, c0, c1, c2


def main() -> None:
    overall_started = time.monotonic()
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    helper_path = Path(__file__).resolve().parent / HELPER_NAME
    helper = load_helper(helper_path)
    step24_run, rows = find_step24_run(project_root, args.step24_run)
    source_row = rows.iloc[0]
    blocks_path = Path(args.blocks).resolve()
    bg_cache = find_bg_cache(step24_run)

    run_dir = project_root / f"step25_vt_wuis_exact_pip_audit_{utc_stamp()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    audit_path = run_dir / "step25_point_assignment_audit.csv.gz"
    detail_path = run_dir / "step25_exact_block_group_detail.csv.gz"
    summary_path = run_dir / "step25_summary.csv"
    json_path = run_dir / "step25_summary.json"
    report_path = run_dir / "step25_report.txt"

    print("STEP 25 PREFLIGHT", flush=True)
    print(f"Source Step 24 run: {step24_run}", flush=True)
    print(f"Structures: {source_row['structures_source']}", flush=True)
    print(f"Census blocks: {blocks_path}", flush=True)
    print(f"WUI raster: {source_row['wui_raster']}", flush=True)
    print(f"Step 24 BG cache (comparison only): {bg_cache}", flush=True)
    print("PREFLIGHT PASS: all required paths resolved", flush=True)

    stats, bg, total, c0, c1, c2 = run_audit(
        helper,
        source_row,
        blocks_path,
        bg_cache,
        audit_path,
        args.chunk_size,
        args.tile_size,
        args.progress_every,
    )
    exact_detail, exact_population = helper.allocate_population(
        bg, total, c0, c1, c2
    )
    exact_detail.insert(0, "method", METHOD)
    exact_detail.insert(0, "state_name", STATE_NAME)
    exact_detail.insert(0, "STUSPS", STUSPS)
    exact_detail.insert(0, "STATEFP", STATEFP)
    exact_detail.to_csv(
        detail_path,
        index=False,
        compression="gzip",
        float_format="%.12f",
    )

    gates = {
        "point_coverage_at_least_99pct": (
            stats["representative_point_coverage_pct"] >= 99.0
        ),
        "exact_block_group_match_at_least_99pct": (
            stats["exact_block_group_match_pct"] >= 99.0
        ),
        "no_unexpected_wui_values": stats["unexpected_wui_values"] == 0,
        "population_allocation_conserved": (
            abs(exact_population["allocation_residual"]) <= 1e-6
        ),
        "official_population_matches": (
            exact_population["official_population"] == OFFICIAL_POPULATION
        ),
    }
    all_pass = all(gates.values())
    verdict = PASS_VERDICT if all_pass else REVIEW_VERDICT
    conclusion = (
        "The Step 24 Vermont WUI-S review was caused by the rasterized "
        "block-group lookup if exact assignment passes the 99% gate."
        if all_pass
        else "Exact point-in-polygon assignment still requires review."
    )
    summary = {
        "STATEFP": STATEFP,
        "STUSPS": STUSPS,
        "method": METHOD,
        "source_step24_run": str(step24_run),
        "step24_block_group_match_pct": float(source_row["block_group_match_pct"]),
        **stats,
        "official_population": OFFICIAL_POPULATION,
        "exact_nonwui_population": exact_population["nonwui_population"],
        "exact_intermix_population": exact_population["intermix_population"],
        "exact_interface_population": exact_population["interface_population"],
        "exact_wui_population": exact_population["wui_population"],
        "exact_wui_share_pct": exact_population["wui_population_share_pct"],
        "exact_allocation_residual": exact_population["allocation_residual"],
        "step24_point_wui_population": float(
            source_row["point_method_wui_population"]
        ),
        "exact_minus_step24_point_wui_population": (
            exact_population["wui_population"]
            - float(source_row["point_method_wui_population"])
        ),
        "step23_wui_population": float(source_row["step23_wui_population"]),
        "exact_minus_step23_wui_population": (
            exact_population["wui_population"]
            - float(source_row["step23_wui_population"])
        ),
        "verdict": verdict,
    }
    pd.DataFrame([summary]).to_csv(
        summary_path, index=False, float_format="%.10f"
    )
    payload = {
        "verdict": verdict,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "conclusion": conclusion,
        "gates": gates,
        "summary": summary,
        "paper_method": (
            "2020 Census block-group population allocated equally across "
            "exactly assigned representative building points"
        ),
        "boundary_policy": (
            "covers/covered_by includes polygon boundaries; points covered by "
            "more than one block group remain explicitly ambiguous"
        ),
        "source_step24_modified": False,
        "step23_candidate_tiffs_modified": False,
        "legacy_caches_modified": False,
        "national_rerun_started": False,
        "national_replacement_authorized": False,
        "outputs": {
            "point_audit_csv_gz": str(audit_path),
            "exact_block_group_detail_csv_gz": str(detail_path),
            "summary_csv": str(summary_path),
            "summary_json": str(json_path),
            "report": str(report_path),
        },
        "helper_script": str(helper_path),
        "helper_sha256": sha256_file(helper_path),
        "elapsed_minutes": (time.monotonic() - overall_started) / 60.0,
    }
    atomic_write_json(json_path, payload)

    gate_lines = [
        f"{name}: {'PASS' if passed else 'FAIL'}"
        for name, passed in gates.items()
    ]
    report = "\n".join(
        [
            "STEP 25 VERMONT WUI-S EXACT POINT-IN-POLYGON AUDIT",
            "=" * 120,
            f"VERDICT: {verdict}",
            "",
            "EXACT ASSIGNMENT QC",
            "-" * 120,
            f"Points expected: {stats['structure_features_expected']:,}",
            f"Points processed: {stats['points_processed']:,}",
            (
                "Step 24 raster BG match: "
                f"{float(source_row['block_group_match_pct']):.6f}%"
            ),
            (
                "Exact unique BG match: "
                f"{stats['exact_block_group_match_pct']:.6f}%"
            ),
            (
                "Exact unique / ambiguous / unmatched: "
                f"{stats['points_exact_unique_block_group']:,} / "
                f"{stats['points_exact_ambiguous_block_group']:,} / "
                f"{stats['points_exact_unmatched_block_group']:,}"
            ),
            (
                "Raster missing but exact recovered: "
                f"{stats['raster_missing_exact_recovered']:,}"
            ),
            (
                "Raster and exact different BG: "
                f"{stats['raster_and_exact_different_bg']:,}"
            ),
            "",
            "QC GATES",
            "-" * 120,
            *gate_lines,
            "",
            "POPULATION COMPARISON",
            "-" * 120,
            (
                f"Exact WUI population: "
                f"{exact_population['wui_population']:,.6f} "
                f"({exact_population['wui_population_share_pct']:.6f}%)"
            ),
            (
                f"Step 24 raster-BG point population: "
                f"{float(source_row['point_method_wui_population']):,.6f}"
            ),
            (
                f"Exact minus Step 24: "
                f"{summary['exact_minus_step24_point_wui_population']:,.6f}"
            ),
            (
                f"Step 23 block-pixel population: "
                f"{float(source_row['step23_wui_population']):,.6f}"
            ),
            (
                f"Exact minus Step 23: "
                f"{summary['exact_minus_step23_wui_population']:,.6f}"
            ),
            "",
            "METHOD CONCLUSION",
            "-" * 120,
            conclusion,
            "This audit does not authorize national replacement or a 49-state rerun.",
            "",
            "SAFETY STATUS",
            "-" * 120,
            "Source Step 24 checkpoint modified: NO",
            "Step 23 candidate TIFFs modified: NO",
            "Legacy caches modified: NO",
            "49-state rerun started: NO",
            "National replacement authorized: NO",
            "",
            "OUTPUT FILES",
            "-" * 120,
            str(audit_path),
            str(detail_path),
            str(summary_path),
            str(json_path),
            str(report_path),
            "",
            f"Elapsed minutes: {payload['elapsed_minutes']:.2f}",
        ]
    ) + "\n"
    atomic_write_text(report_path, report)
    print("\n" + report, flush=True)


if __name__ == "__main__":
    main()