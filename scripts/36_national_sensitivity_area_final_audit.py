#!/usr/bin/env python3
# Copyright (C) 2026 WUI Mapping Project Authors
# SPDX-License-Identifier: GPL-3.0-only
# -*- coding: utf-8 -*-
# Repository version: filesystem paths are loaded through repo_config.py.
# Copy config/paths.example.json to config/paths.json before a new run.
# Raw input data and large intermediate files are stored outside GitHub.

"""
STEP 36 - NATIONAL/SENSITIVITY AREA FINAL AUDIT

Recompute and independently audit the final WUI areas used by the paper.

Scope
-----
National main analysis:
    48 conterminous states + District of Columbia
    WUI-P and WUI-S at 500 m
    WUI-Z with no buffer
    49 x 3 = 147 rows

Five-state sensitivity analysis:
    California, Colorado, Florida, Pennsylvania and Texas
    WUI-P and WUI-S at 100, 200, ..., 1000 m
    5 x 2 x 10 = 100 rows

The ten five-state 500 m P/S rows occur in both analysis scopes.  The final
deduplicated deliverable therefore contains:

    147 + 100 - 10 = 237 unique state-method-buffer rows

Area definitions
----------------
WUI-P / WUI-S:
    Intermix and Interface area are recomputed directly from final class
    rasters, within the official state boundary, using the determinant of
    each raster's affine transform.  Pixel area is never hard-coded.

    Class 0 in the historical rasters covers the classified state domain and
    can include water.  It is retained as Raster_Class0_km2 for audit.
    Paper-ready NonWUI_km2 is instead Census ALAND minus WUI area, so the
    denominator is consistently land area for all three methods.

WUI-Z:
    Areas are aggregated from Census-block ALAND20/ALAND by WUI class.  This
    avoids polygon-area and water-area ambiguity.

Texas WUI-P:
    The historical ResearchDrive WUI-P rasters are never accepted.  Step 36
    requires the passing Step 19D/19E chain, validates all ten candidate
    rasters through the Step 35/19F validators, and uses those exact paths.

Safety and restart behavior
---------------------------
All upstream sources are opened read-only.  Step 36 writes only to a new
run directory.  Work is checkpointed by state-method for P/S and by state for
WUI-Z.  Reusing the same --run-dir resumes completed jobs when their input
file signatures are unchanged.

Dependencies
------------
Python 3.10+: numpy, pandas, rasterio, fiona
Sibling scripts: 35_recompute_five_state_all_buffer_population.py
                 19F_recompute_texas_wui_p_wui_s_population.py
"""

from __future__ import annotations

from repo_config import portable_path

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import math
import os
import re
import sqlite3
import sys
import time
import traceback
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

try:
    import fiona
except ImportError:  # pragma: no cover - checked on the server
    fiona = None

try:
    from shapely.geometry import mapping as shapely_mapping
    from shapely.geometry import shape as shapely_shape
except ImportError:  # pragma: no cover - checked on the server
    shapely_mapping = None
    shapely_shape = None

try:
    import rasterio
    from rasterio.crs import CRS
    from rasterio.features import geometry_mask, geometry_window
    from rasterio.windows import Window
    from rasterio.windows import transform as window_transform
except ImportError:  # pragma: no cover - checked on the server
    rasterio = None
    CRS = None
    geometry_mask = None
    geometry_window = None
    Window = None
    window_transform = None


PROJECT_ROOT = Path(portable_path("project"))
TABLES_ROOT = Path(portable_path("legacy", "WUI_tables_compare"))

DEFAULT_STATE_GPKG = Path(
    portable_path("data", "MBF+NLCD_2022US/Inputs/Processed_GPKG/tl_2022_us_state.gpkg")
)
DEFAULT_WUIP_DIR = Path(
    portable_path("data", "WUI_P_Paper_Raster")
)
DEFAULT_WUIS_DIR = Path(
    portable_path("data", "WUI_S_Paper")
)
DEFAULT_WUIZ_CANDIDATES = (
    Path(portable_path("legacy", "WUI_Z_Results")),
    Path(portable_path("data", "WUI_Z_Results")),
    Path(portable_path("legacy", "WUI_Z_Paper")),
)
DEFAULT_OLD_NATIONAL = (
    TABLES_ROOT
    / "CONUS_table2_sample5_parallel_fast"
    / "table2_area_state_buffer_CONUS49_500m.csv"
)
DEFAULT_OLD_SENSITIVITY = (
    TABLES_ROOT
    / "CONUS_table2_sample5_parallel_fast"
    / "table2_area_state_buffer_sample5.csv"
)

STEP35_NAME = "35_recompute_five_state_all_buffer_population.py"
STEP19F_NAME = "19F_recompute_texas_wui_p_wui_s_population.py"

BUFFERS = tuple(range(100, 1001, 100))
METHODS_PS = ("WUI-P", "WUI-S")
METHODS = ("WUI-P", "WUI-S", "WUI-Z")
VALID_CODES = (0, 1, 2)
SAMPLE_FIPS = {"06", "08", "12", "42", "48"}
EXPECTED_FIPS = {
    "01", "04", "05", "06", "08", "09", "10", "11", "12", "13",
    "16", "17", "18", "19", "20", "21", "22", "23", "24", "25",
    "26", "27", "28", "29", "30", "31", "32", "33", "34", "35",
    "36", "37", "38", "39", "40", "41", "42", "44", "45", "46",
    "47", "48", "49", "50", "51", "53", "54", "55", "56",
}
EXPECTED_NATIONAL_ROWS = 147
EXPECTED_SENSITIVITY_ROWS = 100
EXPECTED_FINAL_ROWS = 237
EXPECTED_JOBS = 147

METHOD_ORDER = {"WUI-P": 0, "WUI-S": 1, "WUI-Z": 2}
PASS = "PASS"
FAIL = "FAIL"
FINAL_PASS_VERDICT = "NATIONAL_SENSITIVITY_AREA_FINAL_AUDIT_PASS"
FINAL_FAIL_VERDICT = "NATIONAL_SENSITIVITY_AREA_FINAL_AUDIT_FAIL_OR_REVIEW"

REFERENCE_URLS = {
    "rasterio_georeferencing":
        "https://rasterio.readthedocs.io/en/stable/topics/georeferencing.html",
    "census_gazetteer_layout":
        "https://www.census.gov/programs-surveys/geography/"
        "technical-documentation/records-layout/gaz-record-layouts.html",
    "usgs_nlcd":
        "https://www.usgs.gov/centers/eros/science/"
        "national-land-cover-database",
}


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def hms(seconds: float) -> str:
    value = max(0, int(seconds))
    hours, remainder = divmod(value, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def require_dependencies() -> None:
    missing = []
    if rasterio is None:
        missing.append("rasterio")
    if fiona is None:
        missing.append("fiona")
    if shapely_shape is None:
        missing.append("shapely")
    if missing:
        raise RuntimeError(
            "Missing required Python dependencies: " + ", ".join(missing)
            + ". Activate the vscserver conda environment."
        )


def atomic_write_text(path: Path, text: str) -> None:
    partial = path.with_name(path.name + ".partial")
    partial.write_text(text, encoding="utf-8")
    os.replace(partial, path)


def json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if value is pd.NA:
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"Cannot JSON-encode {type(value).__name__}")


def atomic_write_json(path: Path, payload: object) -> None:
    atomic_write_text(
        path,
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            default=json_ready,
        )
        + "\n",
    )


def atomic_write_csv(
    frame: pd.DataFrame,
    path: Path,
    *,
    float_format: str | None = None,
) -> None:
    partial = path.with_name(path.name + ".partial")
    frame.to_csv(partial, index=False, float_format=float_format)
    os.replace(partial, path)


def require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise FileNotFoundError(f"{label} not found or empty: {resolved}")
    return resolved


def require_dir(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def file_signature(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def signature_weight(signatures: dict[str, dict]) -> int:
    return max(
        1,
        sum(int(item.get("size_bytes", 0)) for item in signatures.values()),
    )


def load_module(path: Path, name: str):
    path = require_file(path, f"Required sibling script {path.name}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path, chunk_size: int = 8 * 1024**2) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def norm_token(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def state_token(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", name)


def normalize_fips(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(2)
    )


def require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    label: str,
) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise RuntimeError(f"{label} is missing columns: {missing}")


def close(
    left: object,
    right: object,
    *,
    abs_tol: float,
    rel_tol: float = 0.0,
) -> bool:
    try:
        a = float(left)
        b = float(right)
    except (TypeError, ValueError):
        return False
    return (
        math.isfinite(a)
        and math.isfinite(b)
        and math.isclose(a, b, abs_tol=abs_tol, rel_tol=rel_tol)
    )


def safe_pct(numerator: float, denominator: float) -> float:
    if not math.isfinite(denominator) or denominator == 0:
        return float("nan")
    return numerator / denominator * 100.0


def pick_case_insensitive(
    fields: Sequence[str],
    candidates: Sequence[str],
    label: str,
) -> str:
    lookup = {str(field).lower(): str(field) for field in fields}
    for candidate in candidates:
        found = lookup.get(candidate.lower())
        if found is not None:
            return found
    raise RuntimeError(
        f"Cannot find {label}; tried {list(candidates)}; "
        f"available={list(fields)}"
    )


def sql_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def crs_is_equal_area_metre(crs) -> tuple[bool, str]:
    if crs is None:
        return False, "missing CRS"
    epsg = crs.to_epsg() if hasattr(crs, "to_epsg") else None
    wkt = crs.to_wkt() if hasattr(crs, "to_wkt") else str(crs)
    projected = bool(getattr(crs, "is_projected", False))
    unit = str(getattr(crs, "linear_units", "")).lower()
    equal_area = (
        epsg == 5070
        or "albers" in wkt.lower()
        and "equal" in wkt.lower()
        and "area" in wkt.lower()
    )
    metre = unit in {"metre", "meter", "metres", "meters"}
    ok = projected and equal_area and metre
    return ok, f"epsg={epsg}; projected={projected}; unit={unit}"


def affine_pixel_area_m2(transform) -> float:
    area = abs(
        float(transform.a) * float(transform.e)
        - float(transform.b) * float(transform.d)
    )
    if not math.isfinite(area) or area <= 0:
        raise RuntimeError(f"Invalid affine pixel area: {area}")
    return area


def intersect_windows(left, right):
    col0 = max(int(left.col_off), int(right.col_off))
    row0 = max(int(left.row_off), int(right.row_off))
    col1 = min(
        int(left.col_off + left.width),
        int(right.col_off + right.width),
    )
    row1 = min(
        int(left.row_off + left.height),
        int(right.row_off + right.height),
    )
    if col1 <= col0 or row1 <= row0:
        return None
    return Window(col0, row0, col1 - col0, row1 - row0)


def iter_tiles(window, tile_size: int):
    col_start = int(window.col_off)
    row_start = int(window.row_off)
    col_stop = int(window.col_off + window.width)
    row_stop = int(window.row_off + window.height)
    for row in range(row_start, row_stop, tile_size):
        height = min(tile_size, row_stop - row)
        for col in range(col_start, col_stop, tile_size):
            width = min(tile_size, col_stop - col)
            yield Window(col, row, width, height)


def load_states(state_gpkg: Path) -> tuple[dict[str, dict], str]:
    state_gpkg = require_file(state_gpkg, "CONUS state boundary GPKG")
    with fiona.open(state_gpkg) as source:
        fields = list(source.schema["properties"])
        statefp_field = pick_case_insensitive(
            fields, ("STATEFP", "STATEFP20"), "state FIPS"
        )
        stusps_field = pick_case_insensitive(
            fields, ("STUSPS", "STUSPS20"), "state abbreviation"
        )
        name_field = pick_case_insensitive(
            fields, ("NAME", "NAME20"), "state name"
        )
        aland_field = pick_case_insensitive(
            fields, ("ALAND", "ALAND20"), "Census land area"
        )
        awater_field = pick_case_insensitive(
            fields, ("AWATER", "AWATER20"), "Census water area"
        )
        state_crs = CRS.from_wkt(source.crs_wkt) if source.crs_wkt else None
        crs_ok, crs_note = crs_is_equal_area_metre(state_crs)
        if not crs_ok:
            raise RuntimeError(
                "State boundary must use an equal-area metre CRS; " + crs_note
            )

        states: dict[str, dict] = {}
        for feature in source:
            props = feature["properties"]
            statefp = str(props[statefp_field]).strip().zfill(2)
            if statefp not in EXPECTED_FIPS:
                continue
            if statefp in states:
                raise RuntimeError(f"Duplicate state boundary FIPS: {statefp}")
            geometry = feature["geometry"]
            if geometry is None:
                raise RuntimeError(f"State {statefp} has null geometry")
            geometry_object = shapely_shape(dict(geometry))
            if geometry_object.is_empty:
                raise RuntimeError(f"State {statefp} has empty geometry")
            if not geometry_object.is_valid:
                geometry_object = geometry_object.buffer(0)
            if geometry_object.is_empty or not geometry_object.is_valid:
                raise RuntimeError(
                    f"State {statefp} geometry could not be repaired"
                )
            # Convert to plain, process-safe JSON values.
            geometry_plain = json.loads(
                json.dumps(shapely_mapping(geometry_object))
            )
            aland_m2 = float(props[aland_field])
            awater_m2 = float(props[awater_field])
            if (
                not math.isfinite(aland_m2)
                or not math.isfinite(awater_m2)
                or aland_m2 <= 0
                or awater_m2 < 0
            ):
                raise RuntimeError(
                    f"Invalid ALAND/AWATER for state {statefp}"
                )
            states[statefp] = {
                "STATEFP": statefp,
                "STUSPS": str(props[stusps_field]),
                "NAME": str(props[name_field]),
                "geometry": geometry_plain,
                "Census_ALAND_km2": aland_m2 / 1_000_000.0,
                "Census_AWATER_km2": awater_m2 / 1_000_000.0,
                "Census_total_area_km2": (
                    aland_m2 + awater_m2
                ) / 1_000_000.0,
            }

    if set(states) != EXPECTED_FIPS:
        raise RuntimeError(
            "State boundary coverage is not exactly Lower48+DC; "
            f"missing={sorted(EXPECTED_FIPS-set(states))}; "
            f"extra={sorted(set(states)-EXPECTED_FIPS)}"
        )
    return states, crs_note


def index_ps_rasters(root: Path, method: str) -> dict[tuple[str, int], Path]:
    root = require_dir(root, f"{method} raster root")
    letter = "P" if method == "WUI-P" else "S"
    pattern = re.compile(
        rf"^WUI_{letter}_(.+)_r(\d{{4}})m\.tif$",
        flags=re.IGNORECASE,
    )
    index: dict[tuple[str, int], Path] = {}
    for path in root.glob("*/*.tif"):
        match = pattern.match(path.name)
        if not match:
            continue
        key = (norm_token(match.group(1)), int(match.group(2)))
        if key in index:
            raise RuntimeError(
                f"Ambiguous {method} raster key {key}: "
                f"{index[key]} and {path}"
            )
        index[key] = path.resolve()
    if not index:
        raise RuntimeError(f"No {method} rasters indexed under {root}")
    return index


def resolve_ps_raster(
    index: dict[tuple[str, int], Path],
    state_name: str,
    method: str,
    buffer_m: int,
) -> Path:
    key = (norm_token(state_name), int(buffer_m))
    if key not in index:
        raise FileNotFoundError(
            f"No {method} {buffer_m} m raster for {state_name}; key={key}"
        )
    return require_file(index[key], f"{state_name} {method} {buffer_m} m")


def resolve_wuiz_dir(requested: str | None) -> Path:
    if requested:
        return require_dir(Path(requested), "WUI-Z directory")
    ranked = []
    for candidate in DEFAULT_WUIZ_CANDIDATES:
        count = len(list(candidate.glob("WUI_Z_Paper_*.gpkg"))) \
            if candidate.is_dir() else 0
        ranked.append((count, candidate))
    ranked.sort(key=lambda item: item[0], reverse=True)
    if not ranked or ranked[0][0] == 0:
        raise FileNotFoundError(
            "No WUI_Z_Paper_*.gpkg files found. Pass --wuiz-dir."
        )
    chosen = ranked[0][1].resolve()
    print(
        f"[INFO] Auto-selected WUI-Z directory: {chosen} "
        f"({ranked[0][0]} files)",
        flush=True,
    )
    return chosen


def index_wuiz(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in root.glob("WUI_Z_Paper_*.gpkg"):
        token = re.sub(
            r"^WUI_Z_Paper_", "", path.stem, flags=re.IGNORECASE
        )
        key = norm_token(token)
        if key in index:
            raise RuntimeError(
                f"Ambiguous WUI-Z state key {key}: {index[key]} and {path}"
            )
        index[key] = path.resolve()
    return index


def validate_texas_chain(
    scripts_dir: Path,
    step19e_dir: str | None,
) -> dict:
    step35 = load_module(
        scripts_dir / STEP35_NAME,
        "_step36_import_step35",
    )
    step19f = load_module(
        scripts_dir / STEP19F_NAME,
        "_step36_import_step19f",
    )
    chain = step35.validate_texas_all_buffer_chain(
        step19f,
        step19e_dir or "",
    )
    expected = {int(key) for key in chain["raster_paths"]}
    if expected != set(BUFFERS):
        raise RuntimeError(
            "Texas audited WUI-P chain does not contain all ten buffers"
        )
    return chain


def build_jobs(
    args: argparse.Namespace,
    states: dict[str, dict],
    tx_chain: dict,
    state_gpkg: Path,
    wuiz_dir: Path,
) -> list[dict]:
    p_index = index_ps_rasters(Path(args.wuip_dir), "WUI-P")
    s_index = index_ps_rasters(Path(args.wuis_dir), "WUI-S")
    z_index = index_wuiz(wuiz_dir)
    jobs: list[dict] = []

    for statefp in sorted(states):
        state = states[statefp]
        for method, index in (("WUI-P", p_index), ("WUI-S", s_index)):
            buffers = BUFFERS if statefp in SAMPLE_FIPS else (500,)
            rasters: dict[int, Path] = {}
            for buffer_m in buffers:
                if statefp == "48" and method == "WUI-P":
                    raster_path = require_file(
                        Path(tx_chain["raster_paths"][buffer_m]),
                        f"Audited Texas WUI-P {buffer_m} m",
                    )
                else:
                    raster_path = resolve_ps_raster(
                        index, state["NAME"], method, buffer_m
                    )
                rasters[buffer_m] = raster_path
            signatures = {
                "state_boundary": file_signature(state_gpkg),
                **{
                    f"raster_{buffer_m}m": file_signature(path)
                    for buffer_m, path in rasters.items()
                },
            }
            jobs.append(
                {
                    "job_key": f"ps_{statefp}_{method.replace('-', '').lower()}",
                    "job_type": "PS",
                    **state,
                    "method": method,
                    "rasters": {
                        str(key): str(value)
                        for key, value in rasters.items()
                    },
                    "input_signatures": signatures,
                    "weight": signature_weight(signatures),
                    "tile_size": int(args.tile_size),
                    "area_abs_tol_km2": float(args.area_abs_tol_km2),
                    "area_rel_tol": float(args.area_rel_tol),
                    "pixel_area_expected_m2": float(
                        args.pixel_area_expected_m2
                    ),
                    "pixel_area_tol_m2": float(args.pixel_area_tol_m2),
                    "texas_hashes": (
                        {
                            str(key): value
                            for key, value in tx_chain[
                                "raster_hashes"
                            ].items()
                        }
                        if statefp == "48" and method == "WUI-P"
                        else {}
                    ),
                }
            )

        z_key = norm_token(state["NAME"])
        if z_key not in z_index:
            raise FileNotFoundError(
                f"No WUI-Z GPKG for {state['NAME']}; key={z_key}"
            )
        z_path = require_file(
            z_index[z_key], f"{state['NAME']} WUI-Z GPKG"
        )
        z_signatures = {
            "state_boundary": file_signature(state_gpkg),
            "wuiz_gpkg": file_signature(z_path),
        }
        jobs.append(
            {
                "job_key": f"wuiz_{statefp}",
                "job_type": "WUIZ",
                **state,
                "wuiz_gpkg": str(z_path),
                "input_signatures": z_signatures,
                "weight": signature_weight(z_signatures),
                "area_abs_tol_km2": float(args.area_abs_tol_km2),
                "area_rel_tol": float(args.area_rel_tol),
            }
        )

    if len(jobs) != EXPECTED_JOBS:
        raise RuntimeError(
            f"Expected {EXPECTED_JOBS} jobs; built {len(jobs)}"
        )
    return jobs


def scan_ps_group(job: dict) -> dict:
    if rasterio is None:
        raise RuntimeError("rasterio is unavailable in worker")
    started = time.monotonic()
    buffers = sorted(int(key) for key in job["rasters"])
    raster_paths = {
        buffer_m: Path(job["rasters"][str(buffer_m)])
        for buffer_m in buffers
    }
    counts = {
        buffer_m: np.zeros(3, dtype=np.int64)
        for buffer_m in buffers
    }
    invalid = {buffer_m: 0 for buffer_m in buffers}
    state_mask_pixels = 0

    with ExitStack() as stack:
        sources = {
            buffer_m: stack.enter_context(rasterio.open(path))
            for buffer_m, path in raster_paths.items()
        }
        reference = sources[buffers[0]]
        if (
            reference.count != 1
            or reference.width <= 0
            or reference.height <= 0
            or reference.crs is None
        ):
            raise RuntimeError(
                f"Invalid reference raster: {raster_paths[buffers[0]]}"
            )
        crs_ok, crs_note = crs_is_equal_area_metre(reference.crs)
        if not crs_ok:
            raise RuntimeError(
                f"{job['STUSPS']} {job['method']} CRS is not "
                f"equal-area metres: {crs_note}"
            )
        grid = (
            int(reference.width),
            int(reference.height),
            tuple(float(x) for x in reference.transform),
            reference.crs.to_wkt(),
        )
        for buffer_m, source in sources.items():
            candidate_grid = (
                int(source.width),
                int(source.height),
                tuple(float(x) for x in source.transform),
                source.crs.to_wkt() if source.crs else "",
            )
            if source.count != 1 or candidate_grid != grid:
                raise RuntimeError(
                    f"{job['STUSPS']} {job['method']} rasters do not "
                    f"share one band/grid/CRS; buffer={buffer_m}"
                )

        pixel_area_m2 = affine_pixel_area_m2(reference.transform)
        pixel_area_gate = close(
            pixel_area_m2,
            job["pixel_area_expected_m2"],
            abs_tol=job["pixel_area_tol_m2"],
        )
        full = Window(0, 0, reference.width, reference.height)
        try:
            work = geometry_window(
                reference,
                [job["geometry"]],
                pad_x=0,
                pad_y=0,
                north_up=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Cannot derive state window for {job['STUSPS']}: {exc}"
            ) from exc
        work = intersect_windows(work, full)
        if work is None:
            raise RuntimeError(
                f"State geometry does not overlap raster: {job['STUSPS']}"
            )

        for tile in iter_tiles(work, int(job["tile_size"])):
            shape = (int(tile.height), int(tile.width))
            inside = geometry_mask(
                [job["geometry"]],
                out_shape=shape,
                transform=window_transform(tile, reference.transform),
                invert=True,
                all_touched=False,
            )
            inside_n = int(inside.sum())
            if inside_n == 0:
                continue
            state_mask_pixels += inside_n
            for buffer_m, source in sources.items():
                values = source.read(1, window=tile)[inside]
                valid = (
                    (values == 0)
                    | (values == 1)
                    | (values == 2)
                )
                invalid[buffer_m] += int((~valid).sum())
                if valid.any():
                    counts[buffer_m] += np.bincount(
                        values[valid].astype(np.int64, copy=False),
                        minlength=3,
                    )[:3]

        rows = []
        pixel_km2 = pixel_area_m2 / 1_000_000.0
        census_land = float(job["Census_ALAND_km2"])
        census_water = float(job["Census_AWATER_km2"])
        census_total = float(job["Census_total_area_km2"])
        external_tol = max(
            float(job["area_abs_tol_km2"]),
            census_total * float(job["area_rel_tol"]),
        )
        state_mask_area = state_mask_pixels * pixel_km2

        for buffer_m in buffers:
            class_counts = counts[buffer_m]
            class0 = int(class_counts[0])
            intermix_count = int(class_counts[1])
            interface_count = int(class_counts[2])
            classified_count = int(class_counts.sum())
            invalid_count = int(invalid[buffer_m])
            class0_area = class0 * pixel_km2
            intermix_area = intermix_count * pixel_km2
            interface_area = interface_count * pixel_km2
            wui_area = intermix_area + interface_area
            domain_area = classified_count * pixel_km2
            invalid_area = invalid_count * pixel_km2
            nonwui_land = census_land - wui_area
            class_closure = (
                class0 + intermix_count + interface_count
                == classified_count
            )
            wui_closure = close(
                wui_area,
                intermix_area + interface_area,
                abs_tol=1e-10,
            )
            domain_external_gate = (
                abs(domain_area - census_total) <= external_tol
            )
            state_mask_external_gate = (
                abs(state_mask_area - census_total) <= external_tol
            )
            invalid_gate = invalid_area <= external_tol
            wui_land_gate = wui_area <= census_land + external_tol
            nonnegative_gate = nonwui_land >= -external_tol
            row_pass = all(
                (
                    crs_ok,
                    pixel_area_gate,
                    class_closure,
                    wui_closure,
                    domain_external_gate,
                    state_mask_external_gate,
                    invalid_gate,
                    wui_land_gate,
                    nonnegative_gate,
                )
            )
            rows.append(
                {
                    "STATEFP": job["STATEFP"],
                    "STUSPS": job["STUSPS"],
                    "NAME": job["NAME"],
                    "method": job["method"],
                    "buffer_m": int(buffer_m),
                    "area_basis": (
                        "FINAL_CLASS_RASTER_AFFINE_WITHIN_STATE; "
                        "NONWUI_LAND=CENSUS_ALAND-WUI"
                    ),
                    "NonWUI_km2": max(0.0, nonwui_land),
                    "Intermix_km2": intermix_area,
                    "Interface_km2": interface_area,
                    "WUI_km2": wui_area,
                    "WUI_pct_of_Census_ALAND": safe_pct(
                        wui_area, census_land
                    ),
                    "Raster_Class0_km2": class0_area,
                    "Raster_classified_domain_km2": domain_area,
                    "Raster_state_mask_area_km2": state_mask_area,
                    "Raster_invalid_inside_state_km2": invalid_area,
                    "Census_ALAND_km2": census_land,
                    "Census_AWATER_km2": census_water,
                    "Census_total_area_km2": census_total,
                    "classified_domain_minus_Census_total_km2":
                        domain_area - census_total,
                    "state_mask_minus_Census_total_km2":
                        state_mask_area - census_total,
                    "pixel_area_m2": pixel_area_m2,
                    "pixel_width": float(reference.transform.a),
                    "pixel_height": float(reference.transform.e),
                    "raster_width": int(reference.width),
                    "raster_height": int(reference.height),
                    "raster_crs": reference.crs.to_string(),
                    "raster_nodata": reference.nodata,
                    "NonWUI_pixel_count": class0,
                    "Intermix_pixel_count": intermix_count,
                    "Interface_pixel_count": interface_count,
                    "classified_pixel_count": classified_count,
                    "invalid_inside_state_pixel_count": invalid_count,
                    "state_mask_pixel_count": state_mask_pixels,
                    "class_area_closure": class_closure,
                    "wui_area_closure": wui_closure,
                    "equal_area_metre_crs": crs_ok,
                    "pixel_area_expected_gate": pixel_area_gate,
                    "classified_domain_external_gate":
                        domain_external_gate,
                    "state_mask_external_gate":
                        state_mask_external_gate,
                    "invalid_area_gate": invalid_gate,
                    "wui_not_above_Census_ALAND": wui_land_gate,
                    "nonwui_land_nonnegative": nonnegative_gate,
                    "row_qc": PASS if row_pass else FAIL,
                    "source_path": str(raster_paths[buffer_m]),
                    "source_size_bytes":
                        raster_paths[buffer_m].stat().st_size,
                    "source_sha256": job.get(
                        "texas_hashes", {}
                    ).get(str(buffer_m), ""),
                    "source_provenance": (
                        "STEP19D_STEP19E_AUDITED_TEXAS_WUIP"
                        if job["STATEFP"] == "48"
                        and job["method"] == "WUI-P"
                        else "FINAL_RESEARCHDRIVE_PS_RASTER"
                    ),
                }
            )

    return {
        "job_key": job["job_key"],
        "job_type": job["job_type"],
        "input_signatures": job["input_signatures"],
        "elapsed_seconds": time.monotonic() - started,
        "rows": rows,
    }


def sqlite_feature_layer(connection: sqlite3.Connection) -> str:
    rows = connection.execute(
        "SELECT table_name FROM gpkg_contents "
        "WHERE data_type='features' ORDER BY table_name"
    ).fetchall()
    if len(rows) != 1:
        raise RuntimeError(
            f"Expected exactly one feature layer in WUI-Z GPKG; got {rows}"
        )
    return str(rows[0][0])


def scan_wuiz(job: dict) -> dict:
    started = time.monotonic()
    path = Path(job["wuiz_gpkg"])
    uri = "file:" + str(path) + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        layer = sqlite_feature_layer(connection)
        info = connection.execute(
            f"PRAGMA table_info({sql_ident(layer)})"
        ).fetchall()
        fields = [str(row[1]) for row in info]
        code_field = pick_case_insensitive(
            fields, ("WUI_Code", "WUI_CODE", "wui_code"), "WUI-Z code"
        )
        aland_field = pick_case_insensitive(
            fields, ("ALAND20", "ALAND"), "WUI-Z ALAND"
        )
        geoid_field = pick_case_insensitive(
            fields, ("GEOID20", "GEOID"), "WUI-Z GEOID"
        )
        table = sql_ident(layer)
        code = sql_ident(code_field)
        aland = sql_ident(aland_field)
        geoid = sql_ident(geoid_field)

        (
            total,
            distinct_geoid,
            null_geoid,
            null_aland,
            invalid_code,
            invalid_row_union,
        ) = (
            connection.execute(
                f"""
                SELECT
                    COUNT(*),
                    COUNT(DISTINCT {geoid}),
                    SUM(CASE WHEN {geoid} IS NULL
                                  OR TRIM(CAST({geoid} AS TEXT))=''
                             THEN 1 ELSE 0 END),
                    SUM(CASE WHEN {aland} IS NULL
                                  OR typeof({aland}) NOT IN ('integer','real')
                                  OR CAST({aland} AS REAL) < 0
                             THEN 1 ELSE 0 END),
                    SUM(CASE WHEN {code} IS NULL
                                  OR typeof({code}) NOT IN ('integer','real')
                                  OR {code} NOT IN (0,1,2)
                             THEN 1 ELSE 0 END),
                    SUM(CASE WHEN {aland} IS NULL
                                  OR typeof({aland}) NOT IN ('integer','real')
                                  OR CAST({aland} AS REAL) < 0
                                  OR {code} IS NULL
                                  OR typeof({code}) NOT IN ('integer','real')
                                  OR {code} NOT IN (0,1,2)
                             THEN 1 ELSE 0 END)
                FROM {table}
                """
            ).fetchone()
        )
        group_rows = connection.execute(
            f"""
            SELECT CAST({code} AS INTEGER),
                   COUNT(*),
                   SUM(CAST({aland} AS REAL))
            FROM {table}
            WHERE {code} IS NOT NULL
              AND typeof({code}) IN ('integer','real')
              AND {code} IN (0,1,2)
              AND {aland} IS NOT NULL
              AND typeof({aland}) IN ('integer','real')
              AND CAST({aland} AS REAL) >= 0
            GROUP BY CAST({code} AS INTEGER)
            ORDER BY CAST({code} AS INTEGER)
            """
        ).fetchall()
    finally:
        connection.close()

    total = int(total or 0)
    distinct_geoid = int(distinct_geoid or 0)
    null_geoid = int(null_geoid or 0)
    null_aland = int(null_aland or 0)
    invalid_code = int(invalid_code or 0)
    invalid_row_union = int(invalid_row_union or 0)
    duplicate_geoid_extra_rows = max(
        0, (total - null_geoid) - distinct_geoid
    )
    by_code = {
        int(row[0]): {
            "count": int(row[1] or 0),
            "area_m2": float(row[2] or 0.0),
        }
        for row in group_rows
    }
    class0 = by_code.get(0, {"count": 0, "area_m2": 0.0})
    intermix = by_code.get(1, {"count": 0, "area_m2": 0.0})
    interface = by_code.get(2, {"count": 0, "area_m2": 0.0})
    nonwui_area = class0["area_m2"] / 1_000_000.0
    intermix_area = intermix["area_m2"] / 1_000_000.0
    interface_area = interface["area_m2"] / 1_000_000.0
    wui_area = intermix_area + interface_area
    domain_area = nonwui_area + wui_area
    valid_feature_count = (
        class0["count"] + intermix["count"] + interface["count"]
    )
    census_land = float(job["Census_ALAND_km2"])
    census_water = float(job["Census_AWATER_km2"])
    census_total = float(job["Census_total_area_km2"])
    external_tol = max(
        float(job["area_abs_tol_km2"]),
        census_land * float(job["area_rel_tol"]),
    )
    class_closure = close(
        domain_area,
        nonwui_area + intermix_area + interface_area,
        abs_tol=1e-10,
    )
    wui_closure = close(
        wui_area,
        intermix_area + interface_area,
        abs_tol=1e-10,
    )
    feature_closure = valid_feature_count + invalid_row_union == total
    geoid_gate = null_geoid == 0 and duplicate_geoid_extra_rows == 0
    code_gate = invalid_code == 0
    aland_gate = null_aland == 0
    domain_gate = abs(domain_area - census_land) <= external_tol
    wui_land_gate = wui_area <= census_land + external_tol
    row_pass = all(
        (
            class_closure,
            wui_closure,
            feature_closure,
            geoid_gate,
            code_gate,
            aland_gate,
            domain_gate,
            wui_land_gate,
        )
    )
    row = {
        "STATEFP": job["STATEFP"],
        "STUSPS": job["STUSPS"],
        "NAME": job["NAME"],
        "method": "WUI-Z",
        "buffer_m": pd.NA,
        "area_basis": "CENSUS_BLOCK_ALAND_BY_WUI_CODE",
        "NonWUI_km2": nonwui_area,
        "Intermix_km2": intermix_area,
        "Interface_km2": interface_area,
        "WUI_km2": wui_area,
        "WUI_pct_of_Census_ALAND": safe_pct(wui_area, census_land),
        "Raster_Class0_km2": nonwui_area,
        "Raster_classified_domain_km2": domain_area,
        "Raster_state_mask_area_km2": pd.NA,
        "Raster_invalid_inside_state_km2": pd.NA,
        "Census_ALAND_km2": census_land,
        "Census_AWATER_km2": census_water,
        "Census_total_area_km2": census_total,
        "classified_domain_minus_Census_total_km2":
            domain_area - census_land,
        "state_mask_minus_Census_total_km2": pd.NA,
        "pixel_area_m2": pd.NA,
        "pixel_width": pd.NA,
        "pixel_height": pd.NA,
        "raster_width": pd.NA,
        "raster_height": pd.NA,
        "raster_crs": "",
        "raster_nodata": pd.NA,
        "NonWUI_pixel_count": class0["count"],
        "Intermix_pixel_count": intermix["count"],
        "Interface_pixel_count": interface["count"],
        "classified_pixel_count": valid_feature_count,
        "invalid_inside_state_pixel_count": invalid_code,
        "state_mask_pixel_count": total,
        "class_area_closure": class_closure,
        "wui_area_closure": wui_closure,
        "equal_area_metre_crs": pd.NA,
        "pixel_area_expected_gate": pd.NA,
        "classified_domain_external_gate": domain_gate,
        "state_mask_external_gate": pd.NA,
        "invalid_area_gate": code_gate and aland_gate,
        "wui_not_above_Census_ALAND": wui_land_gate,
        "nonwui_land_nonnegative": nonwui_area >= 0,
        "row_qc": PASS if row_pass else FAIL,
        "source_path": str(path),
        "source_size_bytes": path.stat().st_size,
        "source_sha256": "",
        "source_provenance": "WUIZ_BLOCK_ALAND_READ_ONLY_SQLITE",
        "WUIZ_feature_count": total,
        "WUIZ_valid_feature_count": valid_feature_count,
        "WUIZ_null_GEOID": null_geoid,
        "WUIZ_duplicate_GEOID_extra_rows": duplicate_geoid_extra_rows,
        "WUIZ_null_or_negative_ALAND": null_aland,
        "WUIZ_invalid_code": invalid_code,
        "WUIZ_invalid_row_union": invalid_row_union,
        "WUIZ_layer": layer,
    }
    return {
        "job_key": job["job_key"],
        "job_type": job["job_type"],
        "input_signatures": job["input_signatures"],
        "elapsed_seconds": time.monotonic() - started,
        "rows": [row],
    }


def run_worker(job: dict) -> dict:
    if job["job_type"] == "PS":
        return scan_ps_group(job)
    if job["job_type"] == "WUIZ":
        return scan_wuiz(job)
    raise RuntimeError(f"Unknown job type: {job['job_type']}")


def checkpoint_path(checkpoint_dir: Path, job: dict) -> Path:
    return checkpoint_dir / f"{job['job_key']}.json"


def load_checkpoint(path: Path, job: dict) -> dict | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if (
        payload.get("status") != PASS
        or payload.get("job_key") != job["job_key"]
        or payload.get("input_signatures") != job["input_signatures"]
        or not isinstance(payload.get("rows"), list)
    ):
        return None
    return payload


def write_checkpoint(path: Path, result: dict) -> None:
    atomic_write_json(
        path,
        {
            "status": PASS,
            "completed_utc": utc_now(),
            **result,
        },
    )


def sort_area(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["_method_order"] = result["method"].map(METHOD_ORDER)
    result["_buffer_order"] = pd.to_numeric(
        result["buffer_m"], errors="coerce"
    ).fillna(-1)
    result = result.sort_values(
        ["STATEFP", "_method_order", "_buffer_order"]
    )
    return result.drop(
        columns=["_method_order", "_buffer_order"]
    ).reset_index(drop=True)


def validate_design(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = rows.copy()
    rows["STATEFP"] = normalize_fips(rows["STATEFP"])
    rows["buffer_m"] = pd.to_numeric(rows["buffer_m"], errors="coerce")
    if set(rows["STATEFP"]) != EXPECTED_FIPS:
        raise RuntimeError("Recomputed row state coverage is incomplete")

    national = rows[
        (
            rows["method"].isin(METHODS_PS)
            & rows["buffer_m"].eq(500)
        )
        | rows["method"].eq("WUI-Z")
    ].copy()
    sensitivity = rows[
        rows["STATEFP"].isin(SAMPLE_FIPS)
        & rows["method"].isin(METHODS_PS)
        & rows["buffer_m"].isin(BUFFERS)
    ].copy()
    final_unique = pd.concat(
        [
            national,
            sensitivity[~sensitivity["buffer_m"].eq(500)],
        ],
        ignore_index=True,
    )
    national = sort_area(national)
    sensitivity = sort_area(sensitivity)
    final_unique = sort_area(final_unique)

    if len(national) != EXPECTED_NATIONAL_ROWS:
        raise RuntimeError(
            f"National rows={len(national)}; expected={EXPECTED_NATIONAL_ROWS}"
        )
    if len(sensitivity) != EXPECTED_SENSITIVITY_ROWS:
        raise RuntimeError(
            f"Sensitivity rows={len(sensitivity)}; "
            f"expected={EXPECTED_SENSITIVITY_ROWS}"
        )
    if len(final_unique) != EXPECTED_FINAL_ROWS:
        raise RuntimeError(
            f"Final rows={len(final_unique)}; expected={EXPECTED_FINAL_ROWS}"
        )
    if national.assign(
        _buffer=national["buffer_m"].fillna(-1)
    ).duplicated(["STATEFP", "method", "_buffer"]).any():
        raise RuntimeError("Duplicate national state-method-buffer key")
    if sensitivity.duplicated(
        ["STATEFP", "method", "buffer_m"]
    ).any():
        raise RuntimeError("Duplicate sensitivity state-method-buffer key")
    if final_unique.assign(
        _buffer=final_unique["buffer_m"].fillna(-1)
    ).duplicated(["STATEFP", "method", "_buffer"]).any():
        raise RuntimeError("Duplicate final state-method-buffer key")
    return national, sensitivity, final_unique


def build_method_totals(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHODS:
        group = frame[frame["method"] == method]
        if group.empty:
            continue
        census_land = float(group["Census_ALAND_km2"].sum())
        wui = float(group["WUI_km2"].sum())
        rows.append(
            {
                "method": method,
                "state_count": int(group["STATEFP"].nunique()),
                "NonWUI_km2": float(group["NonWUI_km2"].sum()),
                "Intermix_km2": float(group["Intermix_km2"].sum()),
                "Interface_km2": float(group["Interface_km2"].sum()),
                "WUI_km2": wui,
                "Census_ALAND_km2": census_land,
                "WUI_pct_of_Census_ALAND": safe_pct(wui, census_land),
                "all_rows_pass": bool(group["row_qc"].eq(PASS).all()),
            }
        )
    result = pd.DataFrame(rows)
    result["_order"] = result["method"].map(METHOD_ORDER)
    return result.sort_values("_order").drop(
        columns="_order"
    ).reset_index(drop=True)


def build_sensitivity_totals(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (buffer_m, method), group in frame.groupby(
        ["buffer_m", "method"], sort=True
    ):
        census_land = float(group["Census_ALAND_km2"].sum())
        wui = float(group["WUI_km2"].sum())
        rows.append(
            {
                "buffer_m": int(buffer_m),
                "method": method,
                "state_count": int(group["STATEFP"].nunique()),
                "NonWUI_km2": float(group["NonWUI_km2"].sum()),
                "Intermix_km2": float(group["Intermix_km2"].sum()),
                "Interface_km2": float(group["Interface_km2"].sum()),
                "WUI_km2": wui,
                "Census_ALAND_km2": census_land,
                "WUI_pct_of_Census_ALAND": safe_pct(wui, census_land),
                "all_rows_pass": bool(group["row_qc"].eq(PASS).all()),
            }
        )
    result = pd.DataFrame(rows)
    result["_order"] = result["method"].map(METHOD_ORDER)
    return result.sort_values(
        ["buffer_m", "_order"]
    ).drop(columns="_order").reset_index(drop=True)


def load_old_area(path: Path, scope: str) -> pd.DataFrame:
    path = require_file(path, f"Old {scope} area table")
    frame = pd.read_csv(path, dtype={"STATEFP": str})
    require_columns(
        frame,
        {
            "STATEFP", "STUSPS", "NAME", "buffer_m", "method",
            "NonWUI_km2", "Intermix_km2", "Interface_km2",
            "WUI_km2", "Land_km2",
        },
        str(path),
    )
    frame["STATEFP"] = normalize_fips(frame["STATEFP"])
    numeric = (
        "buffer_m", "NonWUI_km2", "Intermix_km2",
        "Interface_km2", "WUI_km2", "Land_km2",
    )
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[list(numeric)].isna().any().any():
        raise RuntimeError(f"Old area table has nonnumeric values: {path}")
    frame["comparison_scope"] = scope
    frame["old_source_csv"] = str(path)
    return frame


def build_old_new_comparison(
    ps_rows: pd.DataFrame,
    old_national: Path,
    old_sensitivity: Path,
    tolerance_km2: float,
) -> pd.DataFrame:
    old = pd.concat(
        [
            load_old_area(old_national, "NATIONAL_500M"),
            load_old_area(old_sensitivity, "FIVE_STATE_SENSITIVITY"),
        ],
        ignore_index=True,
    )
    current = ps_rows[
        [
            "STATEFP", "STUSPS", "NAME", "buffer_m", "method",
            "Raster_Class0_km2", "Intermix_km2", "Interface_km2",
            "WUI_km2", "Raster_classified_domain_km2",
            "source_path", "source_provenance",
        ]
    ].copy()
    joined = old.merge(
        current,
        on=["STATEFP", "STUSPS", "NAME", "buffer_m", "method"],
        how="left",
        validate="many_to_one",
        suffixes=("_old", "_new"),
    )
    if joined["WUI_km2_new"].isna().any():
        missing = joined.loc[
            joined["WUI_km2_new"].isna(),
            ["comparison_scope", "STATEFP", "method", "buffer_m"],
        ]
        raise RuntimeError(
            "Old comparison rows are absent from current recomputation:\n"
            + missing.to_string(index=False)
        )
    mapping = {
        "NonWUI": ("NonWUI_km2", "Raster_Class0_km2"),
        "Intermix": ("Intermix_km2_old", "Intermix_km2_new"),
        "Interface": ("Interface_km2_old", "Interface_km2_new"),
        "WUI": ("WUI_km2_old", "WUI_km2_new"),
        "Domain": ("Land_km2", "Raster_classified_domain_km2"),
    }
    for label, (old_column, new_column) in mapping.items():
        joined[f"{label}_diff_km2"] = (
            joined[new_column] - joined[old_column]
        )
    joined["expected_source_change"] = (
        joined["STATEFP"].eq("48")
        & joined["method"].eq("WUI-P")
    )
    diff_columns = [
        f"{label}_diff_km2" for label in mapping
    ]
    joined["max_abs_diff_km2"] = joined[diff_columns].abs().max(axis=1)
    joined["old_match_or_explained"] = (
        joined["expected_source_change"]
        | joined["max_abs_diff_km2"].le(float(tolerance_km2))
    )
    return joined.sort_values(
        ["comparison_scope", "STATEFP", "method", "buffer_m"]
    ).reset_index(drop=True)


def build_cross_gates(
    all_rows: pd.DataFrame,
    national: pd.DataFrame,
    sensitivity: pd.DataFrame,
    final_unique: pd.DataFrame,
    comparison: pd.DataFrame,
    tx_chain: dict,
    domain_pair_tol_km2: float,
) -> pd.DataFrame:
    gates = []

    def add(name: str, passed: bool, value: object, expected: object):
        gates.append(
            {
                "gate": name,
                "status": PASS if bool(passed) else FAIL,
                "value": value,
                "expected": expected,
            }
        )

    add(
        "national_rows",
        len(national) == EXPECTED_NATIONAL_ROWS,
        len(national),
        EXPECTED_NATIONAL_ROWS,
    )
    add(
        "sensitivity_rows",
        len(sensitivity) == EXPECTED_SENSITIVITY_ROWS,
        len(sensitivity),
        EXPECTED_SENSITIVITY_ROWS,
    )
    add(
        "final_unique_rows",
        len(final_unique) == EXPECTED_FINAL_ROWS,
        len(final_unique),
        EXPECTED_FINAL_ROWS,
    )
    add(
        "all_final_rows_pass",
        final_unique["row_qc"].eq(PASS).all(),
        int(final_unique["row_qc"].eq(PASS).sum()),
        EXPECTED_FINAL_ROWS,
    )

    ps = all_rows[all_rows["method"].isin(METHODS_PS)].copy()
    domain_spread = (
        ps.groupby(["STATEFP", "buffer_m"])[
            "Raster_classified_domain_km2"
        ]
        .agg(lambda values: float(values.max() - values.min()))
    )
    max_domain_spread = float(domain_spread.max()) \
        if len(domain_spread) else float("nan")
    add(
        "P_S_same_state_domain",
        bool((domain_spread <= domain_pair_tol_km2).all()),
        max_domain_spread,
        f"<= {domain_pair_tol_km2} km2",
    )

    overlap_n = national[
        national["STATEFP"].isin(SAMPLE_FIPS)
        & national["method"].isin(METHODS_PS)
    ].copy()
    overlap_s = sensitivity[
        sensitivity["buffer_m"].eq(500)
    ].copy()
    compare_fields = (
        "NonWUI_km2", "Intermix_km2", "Interface_km2", "WUI_km2",
        "Raster_Class0_km2", "Raster_classified_domain_km2",
        "source_path",
    )
    overlap = overlap_n.merge(
        overlap_s,
        on=["STATEFP", "method", "buffer_m"],
        how="outer",
        validate="one_to_one",
        suffixes=("_national", "_sensitivity"),
        indicator=True,
    )
    overlap_ok = len(overlap) == 10 and overlap["_merge"].eq("both").all()
    if overlap_ok:
        for field in compare_fields:
            left = overlap[f"{field}_national"]
            right = overlap[f"{field}_sensitivity"]
            if field == "source_path":
                overlap_ok = overlap_ok and left.astype(str).eq(
                    right.astype(str)
                ).all()
            else:
                overlap_ok = overlap_ok and np.isclose(
                    pd.to_numeric(left),
                    pd.to_numeric(right),
                    atol=1e-10,
                    rtol=0,
                ).all()
    add(
        "five_state_500m_equals_national_500m",
        overlap_ok,
        len(overlap),
        "10 exact P/S rows",
    )

    texas = ps[
        ps["STATEFP"].eq("48") & ps["method"].eq("WUI-P")
    ]
    expected_paths = {
        int(buffer_m): str(Path(path).resolve())
        for buffer_m, path in tx_chain["raster_paths"].items()
    }
    texas_paths_ok = len(texas) == 10
    if texas_paths_ok:
        for _, row in texas.iterrows():
            texas_paths_ok = texas_paths_ok and (
                str(Path(row["source_path"]).resolve())
                == expected_paths[int(row["buffer_m"])]
            )
    add(
        "texas_wuip_uses_step19d_step19e",
        texas_paths_ok,
        len(texas),
        "10 audited raster paths",
    )
    add(
        "texas_wuip_all_ten_hash_verified",
        bool(tx_chain.get("all_ten_rasters_hash_verified")),
        bool(tx_chain.get("all_ten_rasters_hash_verified")),
        True,
    )

    unexpected_old = int((~comparison["old_match_or_explained"]).sum())
    add(
        "old_new_unexpected_differences",
        unexpected_old == 0,
        unexpected_old,
        0,
    )

    z = all_rows[all_rows["method"].eq("WUI-Z")]
    add(
        "wuiz_49_rows",
        len(z) == 49,
        len(z),
        49,
    )
    add(
        "wuiz_geoid_code_area_integrity",
        z["row_qc"].eq(PASS).all(),
        int(z["row_qc"].eq(PASS).sum()),
        49,
    )
    return pd.DataFrame(gates)


def paper_values(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "STATEFP", "STUSPS", "NAME", "method", "buffer_m",
        "NonWUI_km2", "Intermix_km2", "Interface_km2",
        "WUI_km2", "Census_ALAND_km2",
        "WUI_pct_of_Census_ALAND",
    ]
    result = frame[columns].copy()
    numeric = [
        "NonWUI_km2", "Intermix_km2", "Interface_km2",
        "WUI_km2", "Census_ALAND_km2",
        "WUI_pct_of_Census_ALAND",
    ]
    result[numeric] = result[numeric].round(2)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step 36: recompute and audit national 500 m and five-state "
            "100-1000 m WUI area."
        )
    )
    parser.add_argument(
        "--project-root",
        default=str(PROJECT_ROOT),
    )
    parser.add_argument(
        "--scripts-dir",
        default=None,
        help="Default: directory containing this Step 36 script",
    )
    parser.add_argument(
        "--state-gpkg",
        default=str(DEFAULT_STATE_GPKG),
    )
    parser.add_argument(
        "--wuip-dir",
        default=str(DEFAULT_WUIP_DIR),
    )
    parser.add_argument(
        "--wuis-dir",
        default=str(DEFAULT_WUIS_DIR),
    )
    parser.add_argument(
        "--wuiz-dir",
        default=None,
        help="If omitted, common WUI-Z directories are searched",
    )
    parser.add_argument(
        "--step19e-dir",
        default=None,
        help="If omitted, Step 19F auto-selects the latest passing Step 19E",
    )
    parser.add_argument(
        "--old-national-area-csv",
        default=str(DEFAULT_OLD_NATIONAL),
    )
    parser.add_argument(
        "--old-sensitivity-area-csv",
        default=str(DEFAULT_OLD_SENSITIVITY),
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="New output directory; reuse the same value to resume",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Parallel read-only workers (default 2)",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=2048,
        help="Raster scan tile width/height (default 2048)",
    )
    parser.add_argument(
        "--area-abs-tol-km2",
        type=float,
        default=10.0,
        help="Minimum state-domain external-check tolerance",
    )
    parser.add_argument(
        "--area-rel-tol",
        type=float,
        default=5e-4,
        help="Relative state-domain external-check tolerance (default 0.05%%)",
    )
    parser.add_argument(
        "--pixel-area-expected-m2",
        type=float,
        default=900.0,
        help="QC expectation only; actual affine area is used in calculations",
    )
    parser.add_argument(
        "--pixel-area-tol-m2",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--domain-pair-tol-km2",
        type=float,
        default=0.001,
        help="Maximum P/S classified-domain difference per state/buffer",
    )
    parser.add_argument(
        "--old-comparison-tol-km2",
        type=float,
        default=0.001,
        help="Tolerance for unchanged-source old/new comparison",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 1 <= args.max_workers <= 8:
        raise ValueError("--max-workers must be between 1 and 8")
    if not 256 <= args.tile_size <= 8192:
        raise ValueError("--tile-size must be between 256 and 8192")
    for name in (
        "area_abs_tol_km2", "area_rel_tol", "pixel_area_expected_m2",
        "pixel_area_tol_m2", "domain_pair_tol_km2",
        "old_comparison_tol_km2",
    ):
        if float(getattr(args, name)) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be nonnegative")


def main() -> int:
    args = parse_args()
    validate_args(args)
    require_dependencies()
    project_root = require_dir(
        Path(args.project_root), "WUI revision project root"
    )
    scripts_dir = (
        Path(args.scripts_dir).expanduser().resolve()
        if args.scripts_dir
        else Path(__file__).resolve().parent
    )
    state_gpkg = require_file(
        Path(args.state_gpkg), "CONUS state boundary GPKG"
    )
    wuiz_dir = resolve_wuiz_dir(args.wuiz_dir)
    run_dir = (
        Path(args.run_dir).expanduser().resolve()
        if args.run_dir
        else project_root / f"step36_area_final_audit_{utc_stamp()}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    print("=" * 116, flush=True)
    print("STEP 36: NATIONAL/SENSITIVITY AREA FINAL AUDIT", flush=True)
    print(f"State boundary : {state_gpkg}", flush=True)
    print(f"WUI-P root    : {Path(args.wuip_dir).resolve()}", flush=True)
    print(f"WUI-S root    : {Path(args.wuis_dir).resolve()}", flush=True)
    print(f"WUI-Z root    : {wuiz_dir}", flush=True)
    print(f"Output        : {run_dir}", flush=True)
    print(
        "Scope         : National 147 rows + five-state sensitivity "
        "100 rows - 10 overlapping 500 m rows = 237 unique",
        flush=True,
    )
    print(
        f"Parallelism   : {args.max_workers} read-only worker(s); "
        f"tile={args.tile_size} x {args.tile_size}",
        flush=True,
    )
    print(
        "Writes        : new Step 36 outputs/checkpoints only; "
        "all upstream sources read-only",
        flush=True,
    )
    print("=" * 116, flush=True)

    try:
        print("[1/9] Validating final Texas Step 19D/19E chain", flush=True)
        tx_chain = validate_texas_chain(
            scripts_dir, args.step19e_dir
        )

        print("[2/9] Loading official CONUS state boundary areas", flush=True)
        states, state_crs_note = load_states(state_gpkg)

        print("[3/9] Indexing and preflighting all final P/S/Z sources", flush=True)
        jobs = build_jobs(
            args, states, tx_chain, state_gpkg, wuiz_dir
        )
        print(
            f"[PREFLIGHT] jobs={len(jobs)}/{EXPECTED_JOBS}; "
            f"design rows={EXPECTED_FINAL_ROWS} unique",
            flush=True,
        )

        print("[4/9] Recomputing areas with resumable checkpoints", flush=True)
        results: list[dict] = []
        pending: list[dict] = []
        completed_jobs = 0
        completed_rows = 0
        completed_weight = 0
        total_weight = sum(int(job["weight"]) for job in jobs)

        for job in jobs:
            saved = load_checkpoint(
                checkpoint_path(checkpoint_dir, job), job
            )
            if saved is None:
                pending.append(job)
            else:
                results.append(saved)
                completed_jobs += 1
                completed_rows += len(saved["rows"])
                completed_weight += int(job["weight"])

        if completed_jobs:
            print(
                f"[RESUME] valid checkpoints={completed_jobs}; "
                f"rows recovered={completed_rows}; pending={len(pending)}",
                flush=True,
            )

        failures = []
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.max_workers
        ) as executor:
            future_map = {
                executor.submit(run_worker, job): job
                for job in pending
            }
            for future in concurrent.futures.as_completed(future_map):
                job = future_map[future]
                try:
                    result = future.result()
                    write_checkpoint(
                        checkpoint_path(checkpoint_dir, job),
                        result,
                    )
                    results.append(result)
                    completed_jobs += 1
                    row_count = len(result["rows"])
                    completed_rows += row_count
                    completed_weight += int(job["weight"])
                    elapsed = time.monotonic() - started
                    rate = completed_weight / max(elapsed, 1e-9)
                    eta = (
                        (total_weight - completed_weight) / rate
                        if rate > 0 else 0.0
                    )
                    pct = 100.0 * completed_weight / total_weight
                    label = (
                        f"{job['STUSPS']} {job.get('method', 'WUI-Z')}"
                    )
                    print(
                        f"[AREA {completed_jobs:3d}/{EXPECTED_JOBS} | "
                        f"rows={completed_rows:3d}/{EXPECTED_FINAL_ROWS} | "
                        f"weighted={pct:6.2f}% | "
                        f"elapsed={hms(elapsed)} | ETA={hms(eta)}] "
                        f"{label} PASS ({row_count} row(s))",
                        flush=True,
                    )
                except Exception as exc:
                    failures.append(
                        {
                            "job_key": job["job_key"],
                            "STATEFP": job["STATEFP"],
                            "STUSPS": job["STUSPS"],
                            "method": job.get("method", "WUI-Z"),
                            "error": repr(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )
                    print(
                        f"[ERROR] {job['job_key']}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )

        if failures:
            atomic_write_json(
                run_dir / "step36_worker_failures.json",
                failures,
            )
            raise RuntimeError(
                f"{len(failures)} area jobs failed; see "
                "step36_worker_failures.json"
            )
        if len(results) != EXPECTED_JOBS:
            raise RuntimeError(
                f"Completed results={len(results)}; expected={EXPECTED_JOBS}"
            )

        print("[5/9] Building national, sensitivity and unique tables", flush=True)
        all_rows = pd.DataFrame(
            [row for result in results for row in result["rows"]]
        )
        national, sensitivity, final_unique = validate_design(all_rows)

        print("[6/9] Comparing current areas with historical paper tables", flush=True)
        ps_unique = all_rows[
            all_rows["method"].isin(METHODS_PS)
        ].copy()
        comparison = build_old_new_comparison(
            ps_unique,
            Path(args.old_national_area_csv),
            Path(args.old_sensitivity_area_csv),
            args.old_comparison_tol_km2,
        )

        print("[7/9] Running cross-scope and provenance gates", flush=True)
        gates = build_cross_gates(
            all_rows,
            national,
            sensitivity,
            final_unique,
            comparison,
            tx_chain,
            args.domain_pair_tol_km2,
        )
        national_totals = build_method_totals(national)
        sensitivity_totals = build_sensitivity_totals(sensitivity)
        nonpass_rows = int((final_unique["row_qc"] != PASS).sum())
        nonpass_gates = int((gates["status"] != PASS).sum())
        nonpass = nonpass_rows + nonpass_gates
        verdict = (
            FINAL_PASS_VERDICT if nonpass == 0 else FINAL_FAIL_VERDICT
        )

        print("[8/9] Writing final tables and provenance records", flush=True)
        output_paths = {
            "national": run_dir
            / "step36_national_area_long_147.csv",
            "sensitivity": run_dir
            / "step36_five_state_sensitivity_area_long_100.csv",
            "final_unique": run_dir
            / "step36_final_unique_area_long_237.csv",
            "national_paper": run_dir
            / "step36_national_area_paper_values_147.csv",
            "sensitivity_paper": run_dir
            / "step36_sensitivity_area_paper_values_100.csv",
            "national_totals": run_dir
            / "step36_national_method_totals_3.csv",
            "sensitivity_totals": run_dir
            / "step36_sensitivity_method_buffer_totals_20.csv",
            "comparison": run_dir
            / "step36_old_new_area_comparison.csv",
            "gates": run_dir
            / "step36_cross_gates.csv",
        }
        atomic_write_csv(
            national, output_paths["national"], float_format="%.10f"
        )
        atomic_write_csv(
            sensitivity,
            output_paths["sensitivity"],
            float_format="%.10f",
        )
        atomic_write_csv(
            final_unique,
            output_paths["final_unique"],
            float_format="%.10f",
        )
        atomic_write_csv(
            paper_values(national),
            output_paths["national_paper"],
            float_format="%.2f",
        )
        atomic_write_csv(
            paper_values(sensitivity),
            output_paths["sensitivity_paper"],
            float_format="%.2f",
        )
        atomic_write_csv(
            national_totals,
            output_paths["national_totals"],
            float_format="%.10f",
        )
        atomic_write_csv(
            sensitivity_totals,
            output_paths["sensitivity_totals"],
            float_format="%.10f",
        )
        atomic_write_csv(
            comparison,
            output_paths["comparison"],
            float_format="%.10f",
        )
        atomic_write_csv(gates, output_paths["gates"])

        manifest = {
            "step": 36,
            "created_utc": utc_now(),
            "verdict": verdict,
            "scope": {
                "national_rows": len(national),
                "sensitivity_rows": len(sensitivity),
                "final_unique_rows": len(final_unique),
                "states": sorted(EXPECTED_FIPS),
                "sample_states": sorted(SAMPLE_FIPS),
                "buffers_m": list(BUFFERS),
            },
            "area_definition": {
                "P_S_WUI":
                    "class 1/2 pixel counts within state boundary times "
                    "absolute affine determinant",
                "P_S_NonWUI":
                    "Census ALAND minus recomputed WUI area",
                "P_S_Raster_Class0":
                    "raw class 0 retained separately; can include water",
                "WUI_Z": "sum Census block ALAND20/ALAND by WUI code",
                "pixel_area_hardcoded": False,
            },
            "inputs": {
                "state_gpkg": file_signature(state_gpkg),
                "state_crs": state_crs_note,
                "wuip_dir": str(Path(args.wuip_dir).resolve()),
                "wuis_dir": str(Path(args.wuis_dir).resolve()),
                "wuiz_dir": str(wuiz_dir),
                "old_national_area_csv":
                    file_signature(Path(args.old_national_area_csv)),
                "old_sensitivity_area_csv":
                    file_signature(Path(args.old_sensitivity_area_csv)),
                "texas_step19e_dir": str(
                    tx_chain.get("step19e_dir", "")
                ),
                "texas_all_ten_hash_verified": bool(
                    tx_chain.get("all_ten_rasters_hash_verified")
                ),
            },
            "parameters": {
                "max_workers": args.max_workers,
                "tile_size": args.tile_size,
                "area_abs_tol_km2": args.area_abs_tol_km2,
                "area_rel_tol": args.area_rel_tol,
                "pixel_area_expected_m2": args.pixel_area_expected_m2,
                "pixel_area_tol_m2": args.pixel_area_tol_m2,
                "domain_pair_tol_km2": args.domain_pair_tol_km2,
                "old_comparison_tol_km2":
                    args.old_comparison_tol_km2,
            },
            "outputs": {
                key: str(value) for key, value in output_paths.items()
            },
            "reference_urls": REFERENCE_URLS,
        }
        atomic_write_json(
            run_dir / "step36_input_output_manifest.json",
            manifest,
        )

        print("[9/9] Final readback and QC", flush=True)
        readback_national = pd.read_csv(output_paths["national"])
        readback_sensitivity = pd.read_csv(output_paths["sensitivity"])
        readback_final = pd.read_csv(output_paths["final_unique"])
        readback_gates = pd.read_csv(output_paths["gates"])
        readback_ok = (
            len(readback_national) == EXPECTED_NATIONAL_ROWS
            and len(readback_sensitivity) == EXPECTED_SENSITIVITY_ROWS
            and len(readback_final) == EXPECTED_FINAL_ROWS
            and readback_gates["status"].eq(PASS).all()
            and readback_final["row_qc"].eq(PASS).all()
        )
        if not readback_ok:
            verdict = FINAL_FAIL_VERDICT
            nonpass = max(nonpass, 1)

        final_qc = "\n".join(
            [
                "STEP 36: NATIONAL/SENSITIVITY AREA FINAL AUDIT",
                "=" * 72,
                f"VERDICT: {verdict}",
                (
                    "AREA_STATUS: FINAL_237_UNIQUE_ROWS_INCLUDED_AND_VALIDATED"
                    if verdict == FINAL_PASS_VERDICT
                    else "AREA_STATUS: INCOMPLETE_OR_REVIEW"
                ),
                f"NATIONAL_ROWS: {len(national)}/{EXPECTED_NATIONAL_ROWS}",
                (
                    f"SENSITIVITY_ROWS: {len(sensitivity)}/"
                    f"{EXPECTED_SENSITIVITY_ROWS}"
                ),
                (
                    f"FINAL_UNIQUE_ROWS: {len(final_unique)}/"
                    f"{EXPECTED_FINAL_ROWS}"
                ),
                (
                    "UNIQUE_NATIONAL_STATE_METHOD_KEYS: "
                    f"{len(national)}/{EXPECTED_NATIONAL_ROWS}"
                ),
                (
                    "UNIQUE_SENSITIVITY_STATE_METHOD_BUFFER_KEYS: "
                    f"{len(sensitivity)}/{EXPECTED_SENSITIVITY_ROWS}"
                ),
                (
                    "PASS_ROW_QC: "
                    f"{int(final_unique['row_qc'].eq(PASS).sum())}/"
                    f"{EXPECTED_FINAL_ROWS}"
                ),
                f"PASS_CROSS_GATES: {int(gates['status'].eq(PASS).sum())}/{len(gates)}",
                f"NONPASS_ROWS_OR_GATES: {nonpass}",
                "PIXEL_AREA_CALCULATION: AFFINE_DETERMINANT_NOT_HARDCODED",
                "P_S_NONWUI_DENOMINATOR: CENSUS_ALAND_MINUS_WUI",
                "WUIZ_AREA_BASIS: CENSUS_BLOCK_ALAND",
                "TEXAS_WUIP_SOURCE: STEP19D_STEP19E_AUDITED",
                "UPSTREAM_SOURCES_MODIFIED: NO",
                f"RUN_DIRECTORY: {run_dir}",
                f"COMPLETED_UTC: {utc_now()}",
                "",
            ]
        )
        final_qc_path = run_dir / "STEP36_FINAL_QC.txt"
        atomic_write_text(final_qc_path, final_qc)

        print("\n" + "=" * 116, flush=True)
        print(f"VERDICT: {verdict}", flush=True)
        print(
            "AREA_STATUS: "
            + (
                "FINAL_237_UNIQUE_ROWS_INCLUDED_AND_VALIDATED"
                if verdict == FINAL_PASS_VERDICT
                else "INCOMPLETE_OR_REVIEW"
            ),
            flush=True,
        )
        print(
            f"NATIONAL_ROWS: {len(national)}/{EXPECTED_NATIONAL_ROWS}",
            flush=True,
        )
        print(
            f"SENSITIVITY_ROWS: {len(sensitivity)}/"
            f"{EXPECTED_SENSITIVITY_ROWS}",
            flush=True,
        )
        print(
            f"FINAL_UNIQUE_ROWS: {len(final_unique)}/"
            f"{EXPECTED_FINAL_ROWS}",
            flush=True,
        )
        print(
            f"PASS_ROW_QC: "
            f"{int(final_unique['row_qc'].eq(PASS).sum())}/"
            f"{EXPECTED_FINAL_ROWS}",
            flush=True,
        )
        print(f"NONPASS_ROWS_OR_GATES: {nonpass}", flush=True)
        print(f"Final QC: {final_qc_path}", flush=True)
        print(f"National table: {output_paths['national']}", flush=True)
        print(
            f"Sensitivity table: {output_paths['sensitivity']}",
            flush=True,
        )
        print(f"Run directory: {run_dir}", flush=True)
        print("=" * 116, flush=True)
        return 0 if verdict == FINAL_PASS_VERDICT else 2

    except Exception as exc:
        failure_path = run_dir / "STEP36_FAILURE.txt"
        atomic_write_text(
            failure_path,
            "\n".join(
                [
                    "STEP 36 FAILURE",
                    "=" * 72,
                    f"VERDICT: {FINAL_FAIL_VERDICT}",
                    f"ERROR: {repr(exc)}",
                    "",
                    traceback.format_exc(),
                ]
            )
            + "\n",
        )
        print(f"\n[FAIL] {exc}", file=sys.stderr, flush=True)
        print(f"Failure record: {failure_path}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())