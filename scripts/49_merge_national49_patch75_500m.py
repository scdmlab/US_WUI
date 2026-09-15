#!/usr/bin/env python3
# Copyright (C) 2026 WUI Mapping Project Authors
# SPDX-License-Identifier: GPL-3.0-only
# Repository version: filesystem paths are loaded through repo_config.py.
# Copy config/paths.example.json to config/paths.json before a new run.
# Raw input data and large intermediate files are stored outside GitHub.

"""Merge strict-patch 500 m P/S/Z outputs into a national 49-unit inventory."""
from __future__ import annotations

from repo_config import portable_path

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

ROOT = Path(portable_path("project"))
RUN44 = ROOT / "step49_patch75_remaining44_500m_20260729T231000Z"
FOUR = ROOT / "step47d_patch75_four_state_psz_20260729T150314Z"
CO_PS = ROOT / "step47_patch75_silvis_colorado_canary_20260729T034216Z"
CO_Z = ROOT / "step47c_patch75_colorado_wuiz_20260729T144953Z"
FIVE = {"CA", "CO", "FL", "PA", "TX"}
STATES = [
    "AL", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "ID",
    "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN",
    "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND",
    "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT",
    "VA", "WA", "WV", "WI", "WY",
]


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paths_for(state: str) -> dict[str, Path]:
    if state == "CO":
        return {
            "wui_p": CO_PS / "WUI_P_CO_r0500m_SILVIS_GT75_CANARY.tif",
            "wui_s": CO_PS / "WUI_S_CO_r0500m_SILVIS_GT75_CANARY.tif",
            "wui_z": CO_Z / "WUI_Z_CO_SILVIS_GT75_class.tif",
            "wui_z_valid": CO_Z / "WUI_Z_CO_SILVIS_GT75_valid_domain.tif",
        }
    if state in FIVE:
        base = FOUR / state
    else:
        base = RUN44 / "states" / state
    return {
        "wui_p": (
            base / "rasters" / "WUI-P"
            / f"WUI_P_{state}_r0500m_SILVIS_GT75.tif"
        ),
        "wui_s": (
            base / "rasters" / "WUI-S"
            / f"WUI_S_{state}_r0500m_SILVIS_GT75.tif"
        ),
        "wui_z": base / f"WUI_Z_{state}_SILVIS_GT75_class.tif",
        "wui_z_valid": (
            base / f"WUI_Z_{state}_SILVIS_GT75_valid_domain.tif"
        ),
    }


def raster_signature(path: Path) -> tuple:
    with rasterio.open(path) as src:
        return (
            src.width,
            src.height,
            str(src.crs),
            tuple(src.transform),
        )


def common_valid(path_a: Path, path_b: Path) -> int:
    total = 0
    with rasterio.open(path_a) as a, rasterio.open(path_b) as b:
        if raster_signature(path_a) != raster_signature(path_b):
            raise RuntimeError(f"Grid mismatch: {path_a} vs {path_b}")
        for _, window in a.block_windows(1):
            aa = a.read(1, window=window)
            bb = b.read(1, window=window)
            total += int(np.count_nonzero((aa != 255) & (bb != 255)))
    return total


def strict_pairs_44() -> pd.DataFrame:
    data = pd.read_csv(RUN44 / "step49_patch75_500m_pairwise.csv")
    keep = (
        (data.method_pair.eq("WUI-P/WUI-S") & data.z_scenario.eq("NOT_APPLICABLE"))
        | (
            data.method_pair.isin(["WUI-P/WUI-Z", "WUI-S/WUI-Z"])
            & data.z_scenario.eq("NEW_GT75_Z")
        )
    )
    return data.loc[keep].copy()


def strict_pairs_four() -> pd.DataFrame:
    frames = []
    for state in sorted(FIVE - {"CO"}):
        data = pd.read_csv(FOUR / state / f"{state}_psz_pairwise.csv")
        keep = (
            data.radius_m.eq(500)
            & (
                (
                    data.method_pair.eq("WUI-P/WUI-S")
                    & data.z_scenario.eq("NOT_APPLICABLE")
                )
                | (
                    data.method_pair.isin(["WUI-P/WUI-Z", "WUI-S/WUI-Z"])
                    & data.z_scenario.eq("NEW_GT75_Z")
                )
            )
        )
        frames.append(data.loc[keep])
    return pd.concat(frames, ignore_index=True)


def strict_pairs_co() -> pd.DataFrame:
    paths = paths_for("CO")
    ps = pd.read_csv(CO_PS / "patch75_pairwise_ps_impact.csv")
    ps = ps.loc[ps.scenario.eq("Q1_SILVIS_BLOCK_GT75_PATCH")].iloc[0]
    common = common_valid(paths["wui_p"], paths["wui_s"])
    rows = [{
        "state": "CO",
        "radius_m": 500,
        "method_pair": "WUI-P/WUI-S",
        "z_scenario": "NOT_APPLICABLE",
        "common_valid_pixels": common,
        "intersection_pixels": int(ps.intersection_pixels),
        "union_pixels": int(ps.union_pixels),
        "jaccard": float(ps.jaccard),
        "intersection_area_km2": int(ps.intersection_pixels) * 0.0009,
        "union_area_km2": int(ps.union_pixels) * 0.0009,
        "method_wui_pixels": np.nan,
        "z_wui_pixels": np.nan,
    }]
    pz = pd.read_csv(CO_Z / "colorado_wuiz_pairwise_100_1000m.csv")
    for row in pz.loc[pz.radius_m.eq(500)].to_dict("records"):
        rows.append({
            "state": "CO",
            "radius_m": 500,
            "method_pair": row["method_pair"],
            "z_scenario": "NEW_GT75_Z",
            "common_valid_pixels": int(row["common_valid_pixels"]),
            "intersection_pixels": int(row["intersection_pixels"]),
            "union_pixels": int(row["union_pixels"]),
            "jaccard": float(row["jaccard"]),
            "intersection_area_km2": float(row["intersection_area_km2"]),
            "union_area_km2": float(row["union_area_km2"]),
            "method_wui_pixels": int(row["method_wui_pixels"]),
            "z_wui_pixels": int(row["z_wui_pixels"]),
        })
    return pd.DataFrame(rows)


def impact_rows() -> pd.DataFrame:
    frames = [pd.read_csv(RUN44 / "step49_patch75_500m_impact.csv")]
    for state in sorted(FIVE - {"CO"}):
        data = pd.read_csv(FOUR / state / f"{state}_psz_gt75_impact.csv")
        frames.append(data.loc[data.radius_m.eq(500)])
    co = pd.read_csv(CO_PS / "patch75_classification_impact.csv")
    frames.append(co.assign(state="CO", radius_m=500))
    data = pd.concat(frames, ignore_index=True, sort=False)
    return data


def z_impact_rows() -> pd.DataFrame:
    frames = []
    for state in STATES:
        if state == "CO":
            path = CO_Z / "colorado_wuiz_gt75_impact.csv"
        elif state in FIVE:
            path = FOUR / state / f"{state}_wuiz_gt75_impact.csv"
        else:
            path = (
                RUN44 / "states" / state
                / f"{state}_wuiz_gt75_impact.csv"
            )
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True, sort=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    inventory = []
    for index, state in enumerate(STATES, 1):
        paths = paths_for(state)
        missing = [str(path) for path in paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError({"state": state, "missing": missing})
        signatures = {key: raster_signature(path) for key, path in paths.items()}
        grid_match = len(set(signatures.values())) == 1
        row = {
            "state": state,
            "buffer_m": 500,
            "grid_match_psz": grid_match,
            "source_run": (
                "STEP47_CO" if state == "CO"
                else "STEP47D_FOUR" if state in FIVE
                else "STEP49_REMAINING44"
            ),
        }
        for key, path in paths.items():
            row[f"{key}_path"] = str(path)
            row[f"{key}_bytes"] = path.stat().st_size
            row[f"{key}_sha256"] = sha256(path)
        inventory.append(row)
        print(
            f"[NATIONAL49 INVENTORY] {state} {index}/49 "
            f"{100 * index / 49:.1f}%",
            flush=True,
        )
    inventory_df = pd.DataFrame(inventory)
    inventory_df.to_csv(
        output / "patch75_national49_500m_product_inventory.csv",
        index=False,
    )

    pairs = pd.concat(
        [strict_pairs_44(), strict_pairs_four(), strict_pairs_co()],
        ignore_index=True,
        sort=False,
    )
    pairs = pairs.sort_values(["state", "method_pair"]).reset_index(drop=True)
    pairs.to_csv(
        output / "patch75_national49_500m_pairwise.csv",
        index=False,
    )
    micro = (
        pairs.groupby("method_pair", as_index=False)
        .agg(
            units=("state", "nunique"),
            intersection_pixels=("intersection_pixels", "sum"),
            union_pixels=("union_pixels", "sum"),
            common_valid_pixels=("common_valid_pixels", "sum"),
        )
    )
    micro["micro_jaccard"] = (
        micro.intersection_pixels / micro.union_pixels
    )
    micro["intersection_area_km2"] = micro.intersection_pixels * 0.0009
    micro["union_area_km2"] = micro.union_pixels * 0.0009
    micro.to_csv(
        output / "patch75_national49_500m_micro_jaccard.csv",
        index=False,
    )

    impacts = impact_rows()
    impacts.to_csv(
        output / "patch75_national49_500m_ps_impact.csv",
        index=False,
    )
    z_impacts = z_impact_rows()
    z_impacts.to_csv(
        output / "patch75_national49_wuiz_impact.csv",
        index=False,
    )
    summary_rows = []
    for method, data in impacts.groupby("method"):
        old_pixels = int(data.old_wui.sum())
        new_pixels = int(data.new_wui.sum())
        summary_rows.append({
            "method": method,
            "old_wui_pixels": old_pixels,
            "new_wui_pixels": new_pixels,
            "old_wui_area_km2": old_pixels * 0.0009,
            "new_wui_area_km2": new_pixels * 0.0009,
            "net_wui_area_change_km2": (new_pixels - old_pixels) * 0.0009,
            "net_wui_change_percent": (
                100 * (new_pixels - old_pixels) / old_pixels
            ),
        })
    old_z = int(z_impacts.old_wui_pixels.sum())
    new_z = int(z_impacts.new_wui_pixels.sum())
    summary_rows.append({
        "method": "WUI-Z",
        "old_wui_pixels": old_z,
        "new_wui_pixels": new_z,
        "old_wui_area_km2": old_z * 0.0009,
        "new_wui_area_km2": new_z * 0.0009,
        "net_wui_area_change_km2": (new_z - old_z) * 0.0009,
        "net_wui_change_percent": 100 * (new_z - old_z) / old_z,
    })
    pd.DataFrame(summary_rows).to_csv(
        output / "patch75_national49_area_change_summary.csv",
        index=False,
    )

    duplicate_pairs = int(
        pairs.duplicated(["state", "radius_m", "method_pair"]).sum()
    )
    qc = {
        "inventory_rows": len(inventory_df),
        "unique_state_units": int(inventory_df.state.nunique()),
        "grid_match_failures": int((~inventory_df.grid_match_psz).sum()),
        "pairwise_rows": len(pairs),
        "pairwise_unique_states": int(pairs.state.nunique()),
        "pairwise_duplicate_keys": duplicate_pairs,
        "pairwise_rows_per_state_min": int(pairs.groupby("state").size().min()),
        "pairwise_rows_per_state_max": int(pairs.groupby("state").size().max()),
        "micro_pair_rows": len(micro),
        "impact_rows": len(impacts),
        "impact_unique_states": int(impacts.state.nunique()),
        "wuiz_impact_rows": len(z_impacts),
        "wuiz_impact_unique_states": int(z_impacts.state.nunique()),
    }
    passed = (
        qc["inventory_rows"] == 49
        and qc["unique_state_units"] == 49
        and qc["grid_match_failures"] == 0
        and qc["pairwise_rows"] == 147
        and qc["pairwise_unique_states"] == 49
        and qc["pairwise_duplicate_keys"] == 0
        and qc["pairwise_rows_per_state_min"] == 3
        and qc["pairwise_rows_per_state_max"] == 3
        and qc["micro_pair_rows"] == 3
        and qc["impact_rows"] == 98
        and qc["impact_unique_states"] == 49
        and qc["wuiz_impact_rows"] == 49
        and qc["wuiz_impact_unique_states"] == 49
    )
    status = {
        "step": "STEP49_PATCH75_NATIONAL49_500M_MERGE",
        "status": (
            "PATCH75_NATIONAL49_PSZ_500M_COMPLETE"
            if passed else "QC_REVIEW_REQUIRED"
        ),
        "completed_utc": now(),
        "method": (
            "D>6.17; local V>=50 intermix; local V<50 and distance<=2400m "
            "interface; qualifying dissolved Census-block patch "
            "Veg_Percent>75 and area>=5km2"
        ),
        "buffer_m": 500,
        "state_level_reporting_units": 49,
        "upstream_modified": False,
        "environment_modified": False,
        "qc": qc,
    }
    (output / "step49_national49_status.json").write_text(
        json.dumps(status, indent=2) + "\n"
    )
    (output / "STEP49_NATIONAL49_FINAL_QC.txt").write_text(
        "\n".join(
            ["STEP49 PATCH75 NATIONAL49 500M FINAL QC"]
            + [f"{key}={value}" for key, value in qc.items()]
            + [
                "upstream_modified=NO",
                "environment_modified=NO",
                f"final_status={status['status']}",
            ]
        )
        + "\n"
    )
    (output / "README.md").write_text(
        "# Step49 national strict-patch 500 m WUI-P/WUI-S/WUI-Z\n\n"
        "This directory inventories the unified strict qualifying-patch "
        "500 m products for the 48 conterminous states and the District "
        "of Columbia. It merges the five previously completed focus "
        "states with the 44 Step49 rebuilds without copying or modifying "
        "the source rasters.\n\n"
        "The pairwise table contains exactly three strict comparisons per "
        "unit: WUI-P/WUI-S, WUI-P/WUI-Z, and WUI-S/WUI-Z. The micro table "
        "sums intersections and unions over all 49 units before division.\n"
    )
    files = sorted(
        path for path in output.iterdir()
        if path.is_file() and path.name != "sha256_manifest.txt"
    )
    (output / "sha256_manifest.txt").write_text(
        "\n".join(
            f"{sha256(path)}  {path.name}" for path in files
        )
        + "\n"
    )
    print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
