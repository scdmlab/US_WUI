#!/usr/bin/env python3
# Copyright (C) 2026 WUI Mapping Project Authors
# SPDX-License-Identifier: GPL-3.0-only
# Repository version: filesystem paths are loaded through repo_config.py.
# Copy config/paths.example.json to config/paths.json before a new run.
# Raw input data and large intermediate files are stored outside GitHub.

"""Generate revised Results Figures 3, 6, 7, and 8 from Step51 data."""

from __future__ import annotations

from repo_config import portable_path

import hashlib
import gc
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", portable_path("temp", "matplotlib-wui-results-figures-3678"))

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from rasterio.enums import Resampling


ROOT = Path(portable_path("project"))
STEP51 = ROOT / "step51_wuiz_ketchpaw_gt50_20260731T215644Z"
TABLE_DIR = STEP51 / "moran_appendix" / "appendix_tables"
AREA_CSV = TABLE_DIR / "Appendix_A1_patch75_full_design_candidate.csv"
POP_CSV = TABLE_DIR / "Appendix_A3_patch75_full_design_candidate.csv"
STATE_GPKG = Path(
    portable_path("data", "MBF+NLCD_2022US/Inputs/Processed_GPKG/tl_2022_us_state.gpkg")
)
OUT_DIR = ROOT / "manuscript_results_figure_revision_20260803"

NATIONAL_PANEL = "national49_standardized_500m"
SENSITIVITY_PANEL = "five_state_radius_sensitivity"
METHODS = ["WUI-Z", "WUI-P", "WUI-S"]
FOCUS_STATES = ["CA", "CO", "FL", "PA", "TX"]
STATE_NAMES = {
    "CA": "California",
    "CO": "Colorado",
    "FL": "Florida",
    "PA": "Pennsylvania",
    "TX": "Texas",
}

CONUS_STATES = {
    "AL", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA",
    "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA",
    "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM",
    "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD",
    "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
}

WUI_CMAP = ListedColormap(["#F2F2F2", "#73A857", "#F28E2B"])
WUI_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], WUI_CMAP.N)

METHOD_COLORS = {
    "WUI-P": "#3568A8",
    "WUI-S": "#E17C05",
    "WUI-Z": "#666666",
}
METHOD_MARKERS = {"WUI-P": "o", "WUI-S": "s", "WUI-Z": None}
METHOD_LINESTYLES = {"WUI-P": "-", "WUI-S": "-", "WUI-Z": ":"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 11.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_national_inventory() -> pd.DataFrame:
    data = pd.read_csv(AREA_CSV)
    data = data.loc[data["analysis_panel"].eq(NATIONAL_PANEL)].copy()
    expected = len(CONUS_STATES) * len(METHODS)
    if len(data) != expected:
        raise RuntimeError(f"Expected {expected} national raster rows, found {len(data)}")
    if set(data["state"]) != CONUS_STATES:
        raise RuntimeError("National raster inventory does not contain the expected 49 units")
    if set(data["method"]) != set(METHODS):
        raise RuntimeError("National raster inventory has unexpected methods")
    data["classification_path"] = data["classification_path"].map(Path)
    missing = [str(path) for path in data["classification_path"] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing classification rasters: {missing[:5]}")
    return data.sort_values(["method", "state"]).reset_index(drop=True)


def load_state_geometry() -> gpd.GeoDataFrame:
    states = gpd.read_file(STATE_GPKG, engine="fiona")
    states = states.loc[states["STUSPS"].isin(CONUS_STATES)].copy()
    states = states.to_crs("EPSG:5070")
    if len(states) != 49:
        raise RuntimeError(f"Expected 49 state geometries, found {len(states)}")
    return states[["STUSPS", "NAME", "geometry"]].sort_values("STUSPS")


def read_classification_for_display(path: Path, target_resolution_m: float):
    with rasterio.open(path) as src:
        scale_x = max(1.0, target_resolution_m / abs(src.res[0]))
        scale_y = max(1.0, target_resolution_m / abs(src.res[1]))
        out_width = max(1, int(math.ceil(src.width / scale_x)))
        out_height = max(1, int(math.ceil(src.height / scale_y)))
        data = src.read(
            1,
            out_shape=(out_height, out_width),
            resampling=Resampling.nearest,
        )
        values = set(np.unique(data).tolist())
        if not values.issubset({0, 1, 2, 255}):
            raise RuntimeError(f"Unexpected raster classes in {path}: {sorted(values)}")
        data = np.ma.masked_where(data == 255, data)
        bounds = src.bounds
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    return data, extent


def plot_classification_raster(
    ax: plt.Axes,
    path: Path,
    *,
    target_resolution_m: float,
    zorder: int = 1,
) -> None:
    data, extent = read_classification_for_display(path, target_resolution_m)
    ax.imshow(
        data,
        cmap=WUI_CMAP,
        norm=WUI_NORM,
        extent=extent,
        origin="upper",
        interpolation="nearest",
        zorder=zorder,
    )


def add_scale_bar(ax: plt.Axes, length_km: int = 1000) -> None:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    length = length_km * 1000
    x1 = xmax - 0.06 * (xmax - xmin)
    x0 = x1 - length
    y = ymin + 0.07 * (ymax - ymin)
    tick = 0.012 * (ymax - ymin)
    ax.plot([x0, x1], [y, y], color="#222222", linewidth=2.0, zorder=6)
    for x in (x0, x0 + length / 2, x1):
        ax.plot([x, x], [y - tick, y + tick], color="#222222", linewidth=1.5, zorder=6)
    for x, label in zip((x0, x0 + length / 2, x1), ("0", f"{length_km // 2}", f"{length_km:,} km")):
        ax.text(x, y - 2.2 * tick, label, ha="center", va="top", fontsize=8.5)


def add_scale_bar_beside_panel(
    fig: plt.Figure,
    ax: plt.Axes,
    *,
    length_km: int = 1000,
) -> None:
    """Place an accurately scaled bar in the whitespace beside a map panel."""
    fig.canvas.draw()
    xmin, xmax = ax.get_xlim()
    panel_box = ax.get_position()
    bar_width = panel_box.width * (length_km * 1000) / (xmax - xmin)
    x0 = min(0.79, 0.975 - bar_width)
    x1 = x0 + bar_width
    y = panel_box.y0 + 0.11 * panel_box.height
    tick = 0.008

    line_options = {
        "color": "#222222",
        "linewidth": 1.7,
        "transform": fig.transFigure,
        "clip_on": False,
        "zorder": 20,
    }
    ax.plot([x0, x1], [y, y], **line_options)
    for x in (x0, (x0 + x1) / 2, x1):
        ax.plot([x, x], [y - tick, y + tick], **line_options)
    for x, label in zip(
        (x0, (x0 + x1) / 2, x1),
        ("0", f"{length_km // 2}", f"{length_km:,} km"),
    ):
        ax.text(
            x,
            y - 1.8 * tick,
            label,
            transform=fig.transFigure,
            ha="center",
            va="top",
            fontsize=8.5,
            clip_on=False,
            zorder=20,
        )


def save_figure(fig: plt.Figure, base_name: str) -> list[Path]:
    base = OUT_DIR / base_name
    outputs = [base.with_suffix(".pdf"), base.with_suffix(".png"), base.with_suffix(".svg")]
    fig.savefig(outputs[0], bbox_inches="tight", facecolor="white")
    fig.savefig(outputs[1], dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(outputs[2], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    gc.collect()
    return outputs


def draw_figure3(inventory: pd.DataFrame, states: gpd.GeoDataFrame) -> list[Path]:
    apply_plot_style()
    # A compact one-page vertical layout.  The right-side whitespace is used
    # for the legend and scale bar so neither element covers the third map.
    fig, axes = plt.subplots(3, 1, figsize=(10.8, 12.2))
    fig.patch.set_facecolor("white")
    bounds = states.total_bounds
    xpad = 0.025 * (bounds[2] - bounds[0])
    ypad = 0.03 * (bounds[3] - bounds[1])
    titles = {
        "WUI-Z": "A  WUI-Z (fixed)",
        "WUI-P": "B  WUI-P (500 m)",
        "WUI-S": "C  WUI-S (500 m)",
    }

    for ax, method in zip(axes, METHODS):
        panel = inventory.loc[inventory["method"].eq(method)]
        for row in panel.itertuples():
            plot_classification_raster(
                ax,
                Path(row.classification_path),
                target_resolution_m=900,
            )
        states.boundary.plot(ax=ax, color="#555555", linewidth=0.32, zorder=4)
        ax.set_xlim(bounds[0] - xpad, bounds[2] + xpad)
        ax.set_ylim(bounds[1] - ypad, bounds[3] + ypad)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(titles[method], loc="left", fontweight="semibold", pad=4)

    legend = [
        Patch(facecolor="#73A857", edgecolor="#555555", label="Intermix WUI"),
        Patch(facecolor="#F28E2B", edgecolor="#555555", label="Interface WUI"),
        Patch(facecolor="#F2F2F2", edgecolor="#777777", label="Non-WUI"),
    ]
    fig.legend(
        legend,
        [item.get_label() for item in legend],
        loc="center left",
        ncol=1,
        frameon=False,
        bbox_to_anchor=(0.785, 0.205),
    )
    fig.suptitle(
        "National WUI Patterns Across Three Mapping Methods",
        fontsize=15,
        fontweight="semibold",
        y=0.986,
    )
    fig.subplots_adjust(left=0.015, right=0.775, top=0.935, bottom=0.018, hspace=0.002)
    add_scale_bar_beside_panel(fig, axes[-1], length_km=1000)
    return save_figure(fig, "Figure3_CONUS_WUI_patterns_final")


def draw_figure3_2x2_alternative(
    inventory: pd.DataFrame,
    states: gpd.GeoDataFrame,
) -> list[Path]:
    """Draw a compact 2 x 2 alternative with a separate legend/scale panel."""
    apply_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.7))
    fig.patch.set_facecolor("white")
    bounds = states.total_bounds
    xpad = 0.025 * (bounds[2] - bounds[0])
    ypad = 0.03 * (bounds[3] - bounds[1])
    titles = {
        "WUI-Z": "A  WUI-Z (fixed)",
        "WUI-P": "B  WUI-P (500 m)",
        "WUI-S": "C  WUI-S (500 m)",
    }
    map_axes = [axes[0, 0], axes[0, 1], axes[1, 0]]

    for ax, method in zip(map_axes, METHODS):
        panel = inventory.loc[inventory["method"].eq(method)]
        for row in panel.itertuples():
            plot_classification_raster(
                ax,
                Path(row.classification_path),
                target_resolution_m=900,
            )
        states.boundary.plot(ax=ax, color="#555555", linewidth=0.32, zorder=4)
        ax.set_xlim(bounds[0] - xpad, bounds[2] + xpad)
        ax.set_ylim(bounds[1] - ypad, bounds[3] + ypad)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(titles[method], loc="left", fontweight="semibold", pad=4)

    reference_ax = map_axes[-1]
    key_ax = axes[1, 1]
    key_ax.set_axis_off()
    key_ax.text(
        0.53,
        0.54,
        "Legend and scale",
        transform=key_ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12,
        fontweight="semibold",
    )
    legend = [
        Patch(facecolor="#73A857", edgecolor="#555555", label="Intermix WUI"),
        Patch(facecolor="#F28E2B", edgecolor="#555555", label="Interface WUI"),
        Patch(facecolor="#F2F2F2", edgecolor="#777777", label="Non-WUI"),
    ]
    key_ax.legend(
        legend,
        [item.get_label() for item in legend],
        loc="upper left",
        bbox_to_anchor=(0.53, 0.50),
        frameon=False,
        fontsize=11,
        handlelength=1.55,
        handleheight=0.75,
        handletextpad=0.45,
        labelspacing=0.22,
        borderaxespad=0.0,
    )

    xmin, xmax = reference_ax.get_xlim()
    length_km = 1000
    bar_fraction = (length_km * 1000) / (xmax - xmin)
    x0 = 0.53
    x1 = x0 + bar_fraction
    y = 0.22
    tick = 0.035
    key_ax.plot(
        [x0, x1],
        [y, y],
        transform=key_ax.transAxes,
        color="#222222",
        linewidth=2.0,
        clip_on=False,
    )
    for x in (x0, (x0 + x1) / 2, x1):
        key_ax.plot(
            [x, x],
            [y - tick, y + tick],
            transform=key_ax.transAxes,
            color="#222222",
            linewidth=1.7,
            clip_on=False,
        )
    for x, label in zip(
        (x0, (x0 + x1) / 2, x1),
        ("0", "500", "1,000 km"),
    ):
        key_ax.text(
            x,
            y - 1.7 * tick,
            label,
            transform=key_ax.transAxes,
            ha="center",
            va="top",
            fontsize=10,
        )

    fig.suptitle(
        "National WUI Patterns Across Three Mapping Methods",
        fontsize=16,
        fontweight="semibold",
        y=0.985,
    )
    fig.subplots_adjust(
        left=0.015,
        right=0.985,
        top=0.91,
        bottom=0.025,
        hspace=0.08,
        wspace=0.035,
    )
    return save_figure(fig, "Figure3_CONUS_WUI_patterns_2x2_alternative")


def draw_figure6(inventory: pd.DataFrame, states: gpd.GeoDataFrame) -> list[Path]:
    apply_plot_style()
    fig, axes = plt.subplots(5, 3, figsize=(11.6, 15.2))
    fig.patch.set_facecolor("white")
    method_titles = ["WUI-Z (fixed)", "WUI-P (500 m)", "WUI-S (500 m)"]
    panel_index = 0

    for row_index, state in enumerate(FOCUS_STATES):
        state_geometry = states.loc[states["STUSPS"].eq(state)]
        bounds = state_geometry.total_bounds
        xpad = 0.06 * (bounds[2] - bounds[0])
        ypad = 0.06 * (bounds[3] - bounds[1])
        for col_index, method in enumerate(METHODS):
            ax = axes[row_index, col_index]
            record = inventory.loc[
                inventory["state"].eq(state) & inventory["method"].eq(method)
            ]
            if len(record) != 1:
                raise RuntimeError(f"Expected one {state} {method} raster, found {len(record)}")
            path = Path(record.iloc[0]["classification_path"])
            plot_classification_raster(ax, path, target_resolution_m=450)
            state_geometry.boundary.plot(ax=ax, color="#4A4A4A", linewidth=0.85, zorder=4)
            ax.set_xlim(bounds[0] - xpad, bounds[2] + xpad)
            ax.set_ylim(bounds[1] - ypad, bounds[3] + ypad)
            ax.set_aspect("equal")
            ax.set_axis_off()
            ax.text(
                0.015,
                0.98,
                chr(65 + panel_index),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=10.5,
                fontweight="semibold",
                path_effects=[path_effects.withStroke(linewidth=2.5, foreground="white")],
            )
            if row_index == 0:
                ax.set_title(method_titles[col_index], fontweight="semibold", pad=5)
            if col_index == 0:
                ax.text(
                    -0.07,
                    0.5,
                    f"{STATE_NAMES[state]} ({state})",
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    rotation=90,
                    fontsize=11,
                    fontweight="semibold",
                )
            panel_index += 1

    legend = [
        Patch(facecolor="#73A857", edgecolor="#555555", label="Intermix WUI"),
        Patch(facecolor="#F28E2B", edgecolor="#555555", label="Interface WUI"),
        Patch(facecolor="#F2F2F2", edgecolor="#777777", label="Non-WUI"),
    ]
    fig.legend(legend, [item.get_label() for item in legend], loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.012))
    fig.suptitle(
        "WUI Patterns in the Five Focus States",
        fontsize=16,
        fontweight="semibold",
        y=0.992,
    )
    fig.subplots_adjust(left=0.09, right=0.985, top=0.925, bottom=0.045, hspace=0.035, wspace=0.03)
    return save_figure(fig, "Figure6_five_state_WUI_patterns_final")


def build_sensitivity_data(source_csv: Path, value_column: str) -> pd.DataFrame:
    data = pd.read_csv(source_csv)
    data = data.loc[data["analysis_panel"].eq(SENSITIVITY_PANEL)].copy()
    data = data.loc[data["state"].isin(FOCUS_STATES)].copy()
    data = data[["state", "method", "buffer_m", value_column]]
    ps = data.loc[data["method"].isin(["WUI-P", "WUI-S"])].copy()
    z = data.loc[data["method"].eq("WUI-Z")].copy()
    if len(ps) != 100 or len(z) != 5:
        raise RuntimeError(f"Unexpected sensitivity rows for {value_column}: PS={len(ps)}, Z={len(z)}")

    radii = np.arange(100, 1001, 100)
    expanded_z = pd.concat(
        [
            z.assign(buffer_m=radius)
            for radius in radii
        ],
        ignore_index=True,
    )
    expanded = pd.concat([ps, expanded_z], ignore_index=True)
    mean_data = (
        expanded.groupby(["method", "buffer_m"], as_index=False)[value_column]
        .mean()
        .assign(state="Five-state mean")
    )
    combined = pd.concat([expanded, mean_data], ignore_index=True)
    state_order = FOCUS_STATES + ["Five-state mean"]
    combined["state"] = pd.Categorical(combined["state"], state_order, ordered=True)
    combined["method"] = pd.Categorical(combined["method"], METHODS, ordered=True)
    return combined.sort_values(["state", "method", "buffer_m"]).reset_index(drop=True)


def draw_sensitivity_figure(
    data: pd.DataFrame,
    *,
    value_column: str,
    y_label: str,
    title: str,
    base_name: str,
    percent_axis: bool,
) -> list[Path]:
    apply_plot_style()
    states = FOCUS_STATES + ["Five-state mean"]
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.6), sharex=True)
    fig.patch.set_facecolor("white")
    handles = []
    labels = []

    for panel_index, (ax, state) in enumerate(zip(axes.flat, states)):
        panel = data.loc[data["state"].eq(state)]
        all_values = panel[value_column].to_numpy(dtype=float)
        ymin = float(np.nanmin(all_values))
        ymax = float(np.nanmax(all_values))
        padding = max((ymax - ymin) * 0.16, 1.5 if percent_axis else ymax * 0.06)
        ax.set_ylim(max(0, ymin - padding), ymax + padding)

        for method in ["WUI-P", "WUI-S", "WUI-Z"]:
            line_data = panel.loc[panel["method"].eq(method)]
            line, = ax.plot(
                line_data["buffer_m"],
                line_data[value_column],
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                markersize=4.8 if method != "WUI-Z" else 0,
                linewidth=2.0,
                linestyle=METHOD_LINESTYLES[method],
                label=method,
            )
            if panel_index == 0:
                handles.append(line)
                labels.append(method)

        ax.axvline(500, color="#8A8A8A", linewidth=1.0, linestyle="--", zorder=0)
        panel_title = STATE_NAMES.get(state, state)
        ax.set_title(
            f"{chr(65 + panel_index)}  {panel_title}",
            loc="left",
            fontweight="semibold",
        )
        ax.set_xlim(80, 1020)
        ax.set_xticks(np.arange(100, 1001, 100))
        ax.grid(color="#D9D9D9", linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#4A4A4A")
        ax.spines["bottom"].set_color("#4A4A4A")
        ax.tick_params(axis="x", rotation=45)
        if percent_axis:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1f}%"))
        else:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))

        if state == "Five-state mean":
            offsets = (
                {"WUI-P": (10, -16), "WUI-S": (10, 12), "WUI-Z": (10, 8)}
                if percent_axis
                else {"WUI-P": (10, 12), "WUI-S": (10, -18), "WUI-Z": (10, 8)}
            )
            for method in ("WUI-P", "WUI-S", "WUI-Z"):
                row = panel.loc[
                    panel["method"].eq(method) & panel["buffer_m"].eq(500)
                ]
                value = float(row[value_column].iloc[0])
                label = f"{value:.2f}%" if percent_axis else f"{value:,.0f}"
                ax.annotate(
                    label,
                    xy=(500, value),
                    xytext=offsets[method],
                    textcoords="offset points",
                    fontsize=8.3,
                    fontweight="semibold",
                    color=METHOD_COLORS[method],
                    path_effects=[path_effects.withStroke(linewidth=2.2, foreground="white")],
                )

    axes[0, 0].set_ylabel(y_label)
    axes[1, 0].set_ylabel(y_label)
    for ax in axes[1, :]:
        ax.set_xlabel("Neighborhood radius (m)")

    fig.suptitle(title, fontsize=16, fontweight="semibold", y=0.98)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=3, frameon=False)
    fig.text(
        0.5,
        0.018,
        "The dashed vertical line marks the common 500 m setting. WUI-Z is fixed and shown as a horizontal reference.",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#444444",
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.88, bottom=0.13, hspace=0.24, wspace=0.18)
    return save_figure(fig, base_name)


def write_final_qc() -> tuple[Path, Path]:
    stems = [
        "Figure3_CONUS_WUI_patterns_final",
        "Figure6_five_state_WUI_patterns_final",
        "Figure7_population_sensitivity_5states_final",
        "Figure8_area_sensitivity_5states_final",
    ]
    figure_files = [OUT_DIR / f"{stem}.{suffix}" for stem in stems for suffix in ("pdf", "png", "svg")]
    source_files = [
        OUT_DIR / "Figures3_6_classification_inventory.csv",
        OUT_DIR / "Figure7_population_sensitivity_5states_source.csv",
        OUT_DIR / "Figure8_area_sensitivity_5states_source.csv",
    ]
    missing = [str(path) for path in [*figure_files, *source_files] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing final figure files: {missing}")

    inventory = pd.read_csv(source_files[0])
    population = pd.read_csv(source_files[1])
    area = pd.read_csv(source_files[2])
    if len(inventory) != 147 or inventory.duplicated(["state", "method"]).any():
        raise RuntimeError("Figure 3/6 inventory did not pass the 49 x 3 uniqueness check")
    if inventory["classification_sha256"].isna().any():
        raise RuntimeError("Figure 3/6 inventory contains a missing classification hash")
    if len(population) != 180 or len(area) != 180:
        raise RuntimeError("Figure 7/8 source tables did not pass the 6 panels x 3 methods x 10 radii check")

    def mean_500(data: pd.DataFrame, value: str) -> dict[str, float]:
        rows = data.loc[
            data["state"].eq("Five-state mean") & data["buffer_m"].eq(500)
        ]
        return {str(row.method): float(getattr(row, value)) for row in rows.itertuples()}

    pop500 = mean_500(population, "wui_population_share_pct")
    area500 = mean_500(area, "wui_area_km2")
    manifest = OUT_DIR / "Figures3_6_7_8_sha256_manifest.txt"
    manifest.write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in [*figure_files, *source_files]),
        encoding="utf-8",
    )
    qc = OUT_DIR / "FIGURES3_6_7_8_FINAL_QC.txt"
    qc.write_text(
        "\n".join(
            [
                "FIGURES 3, 6, 7, AND 8 FINAL QC",
                "status=PASS",
                f"authoritative_step51={STEP51}",
                "figure3_layout=3_vertical_national_maps",
                "figure6_layout=5_states_x_3_methods",
                "figure7_layout=5_states_plus_five_state_mean",
                "figure8_layout=5_states_plus_five_state_mean",
                "classification_inventory_rows=147",
                "classification_inventory_expected=49_units_x_3_methods",
                "figure7_source_rows=180",
                "figure8_source_rows=180",
                "sensitivity_expected=6_panels_x_3_methods_x_10_radii",
                "WUI_Z=fixed_reference",
                "WUI_P_WUI_S_common_setting_m=500",
                f"five_state_mean_population_500m_WUI_P_pct={pop500['WUI-P']:.12f}",
                f"five_state_mean_population_500m_WUI_S_pct={pop500['WUI-S']:.12f}",
                f"five_state_mean_population_fixed_WUI_Z_pct={pop500['WUI-Z']:.12f}",
                f"five_state_mean_area_500m_WUI_P_km2={area500['WUI-P']:.12f}",
                f"five_state_mean_area_500m_WUI_S_km2={area500['WUI-S']:.12f}",
                f"five_state_mean_area_fixed_WUI_Z_km2={area500['WUI-Z']:.12f}",
                "output_formats=PDF,PNG,SVG",
                "result=READY_FOR_OVERLEAF",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return manifest, qc


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    inventory = load_national_inventory()
    inventory_source = OUT_DIR / "Figures3_6_classification_inventory.csv"
    export_inventory = inventory.copy()
    export_inventory["classification_path"] = export_inventory["classification_path"].astype(str)
    export_inventory.to_csv(inventory_source, index=False)
    states = load_state_geometry()

    figure3_outputs = draw_figure3(inventory, states)
    figure6_outputs = draw_figure6(inventory, states)

    population = build_sensitivity_data(POP_CSV, "wui_population_share_pct")
    population_source = OUT_DIR / "Figure7_population_sensitivity_5states_source.csv"
    population.to_csv(population_source, index=False, float_format="%.12f")
    figure7_outputs = draw_sensitivity_figure(
        population,
        value_column="wui_population_share_pct",
        y_label="Population in WUI (%)",
        title="Sensitivity of Population Share in WUI to Neighborhood Radius",
        base_name="Figure7_population_sensitivity_5states_final",
        percent_axis=True,
    )

    area = build_sensitivity_data(AREA_CSV, "wui_area_km2")
    area_source = OUT_DIR / "Figure8_area_sensitivity_5states_source.csv"
    area.to_csv(area_source, index=False, float_format="%.12f")
    figure8_outputs = draw_sensitivity_figure(
        area,
        value_column="wui_area_km2",
        y_label=r"Total WUI area (km$^2$)",
        title="Sensitivity of Total WUI Area to Neighborhood Radius",
        base_name="Figure8_area_sensitivity_5states_final",
        percent_axis=False,
    )

    _ = (
        figure3_outputs,
        figure6_outputs,
        figure7_outputs,
        figure8_outputs,
        inventory_source,
        population_source,
        area_source,
    )
    write_final_qc()


if __name__ == "__main__":
    main()
