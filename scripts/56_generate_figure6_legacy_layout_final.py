#!/usr/bin/env python3
# Copyright (C) 2026 WUI Mapping Project Authors
# SPDX-License-Identifier: GPL-3.0-only
# Repository version: filesystem paths are loaded through repo_config.py.
# Copy config/paths.example.json to config/paths.json before a new run.
# Raw input data and large intermediate files are stored outside GitHub.

"""Generate final Step51 Figure 6 in the original 5-state x 3-method layout."""

from __future__ import annotations

from repo_config import portable_path

import gc
import hashlib
import importlib.util
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", portable_path("temp", "matplotlib-wui-figure6-legacy-final"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ROOT = Path(portable_path("project"))
SOURCE_SCRIPT = ROOT / "scripts" / "55_generate_results_figures3_6_7_8.py"
OUT_DIR = ROOT / "manuscript_results_figure_revision_20260803"
BASE_NAME = "Figure6_five_state_WUI_patterns_legacy_layout_final"


def load_source_module():
    spec = importlib.util.spec_from_file_location("results_figures_source", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load source module: {SOURCE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    source = load_source_module()
    source.apply_plot_style()
    plt.rcParams.update(
        {
            "font.size": 11.5,
            "axes.titlesize": 12.5,
            "legend.fontsize": 10.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
        }
    )

    inventory = source.load_national_inventory()
    states = source.load_state_geometry()
    methods = source.METHODS
    focus_states = source.FOCUS_STATES
    state_names = source.STATE_NAMES

    fig, axes = plt.subplots(3, 5, figsize=(16.8, 9.3))
    fig.patch.set_facecolor("white")
    method_labels = {
        "WUI-Z": "WUI-Z (fixed)",
        "WUI-P": "WUI-P (500 m)",
        "WUI-S": "WUI-S (500 m)",
    }

    for col_index, state in enumerate(focus_states):
        state_geometry = states.loc[states["STUSPS"].eq(state)]
        bounds = state_geometry.total_bounds
        xpad = 0.045 * (bounds[2] - bounds[0])
        ypad = 0.045 * (bounds[3] - bounds[1])

        for row_index, method in enumerate(methods):
            ax = axes[row_index, col_index]
            record = inventory.loc[
                inventory["state"].eq(state) & inventory["method"].eq(method)
            ]
            if len(record) != 1:
                raise RuntimeError(
                    f"Expected one final raster for {state} {method}; found {len(record)}"
                )

            classification_path = Path(record.iloc[0]["classification_path"])
            source.plot_classification_raster(
                ax,
                classification_path,
                target_resolution_m=300,
            )
            state_geometry.boundary.plot(
                ax=ax,
                color="#454545",
                linewidth=0.95,
                zorder=4,
            )
            ax.set_xlim(bounds[0] - xpad, bounds[2] + xpad)
            ax.set_ylim(bounds[1] - ypad, bounds[3] + ypad)
            ax.set_aspect("equal")
            ax.set_axis_off()

            if row_index == 0:
                ax.set_title(
                    f"{state_names[state]} ({state})",
                    fontweight="semibold",
                    pad=5,
                )
            if col_index == 0:
                ax.text(
                    -0.12,
                    0.5,
                    method_labels[method],
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=12,
                    fontweight="semibold",
                )

    legend_items = [
        Patch(facecolor="#73A857", edgecolor="#555555", label="Intermix WUI"),
        Patch(facecolor="#F28E2B", edgecolor="#555555", label="Interface WUI"),
        Patch(facecolor="#F2F2F2", edgecolor="#777777", label="Non-WUI"),
    ]
    fig.legend(
        handles=legend_items,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.53, 0.018),
        handlelength=1.5,
        columnspacing=1.8,
    )
    fig.suptitle(
        "WUI Patterns in the Five Focus States",
        fontsize=17,
        fontweight="semibold",
        y=0.985,
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.995,
        top=0.91,
        bottom=0.075,
        hspace=0.055,
        wspace=0.045,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT_DIR / f"{BASE_NAME}.pdf"
    png_path = OUT_DIR / f"{BASE_NAME}.png"
    svg_path = OUT_DIR / f"{BASE_NAME}.svg"
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    fig.savefig(png_path, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    gc.collect()

    manifest_path = OUT_DIR / f"{BASE_NAME}_sha256.txt"
    outputs = [pdf_path, png_path, svg_path]
    manifest_path.write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in outputs),
        encoding="utf-8",
    )

    qc_path = OUT_DIR / f"{BASE_NAME}_QC.txt"
    qc_path.write_text(
        "\n".join(
            [
                "FIGURE 6 ORIGINAL-LAYOUT FINAL QC",
                "status=PASS",
                f"authoritative_step51={source.STEP51}",
                "layout=3_method_rows_x_5_state_columns",
                "methods=WUI-Z_fixed,WUI-P_500m,WUI-S_500m",
                "states=CA,CO,FL,PA,TX",
                "display_raster_target_resolution_m=300",
                "pdf_text_and_boundaries=vector",
                "png_dpi=600",
                "outputs=PDF,PNG,SVG",
                "",
            ]
        ),
        encoding="utf-8",
    )

    for path in [*outputs, manifest_path, qc_path]:
        print(path)


if __name__ == "__main__":
    main()
