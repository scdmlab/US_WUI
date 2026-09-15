#!/usr/bin/env python3
# Copyright (C) 2026 WUI Mapping Project Authors
# SPDX-License-Identifier: GPL-3.0-only
# Repository version: filesystem paths are loaded through repo_config.py.
# Copy config/paths.example.json to config/paths.json before a new run.
# Raw input data and large intermediate files are stored outside GitHub.

"""Generate revised Results Figures 5 and 9 from the frozen Step51 data."""

from __future__ import annotations

from repo_config import portable_path

import hashlib
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", portable_path("temp", "matplotlib-wui-results-figures"))

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter


ROOT = Path(portable_path("project"))
STEP51 = ROOT / "step51_wuiz_ketchpaw_gt50_20260731T215644Z"
TABLE_DIR = STEP51 / "moran_appendix" / "appendix_tables"
AREA_CSV = TABLE_DIR / "Appendix_A1_patch75_full_design_candidate.csv"
POP_CSV = TABLE_DIR / "Appendix_A3_patch75_full_design_candidate.csv"
JACCARD_CSV = TABLE_DIR / "Appendix_A6_patch75_full_design_candidate.csv"
STATE_GPKG = Path(
    portable_path("data", "MBF+NLCD_2022US/Inputs/Processed_GPKG/tl_2022_us_state.gpkg")
)
OUT_DIR = ROOT / "manuscript_results_figure_revision_20260803"

NATIONAL_PANEL = "national49_standardized_500m"
SENSITIVITY_PANEL = "five_state_radius_sensitivity"
METHODS = ["WUI-Z", "WUI-P", "WUI-S"]
FOCUS_STATES = ["CA", "CO", "FL", "PA", "TX"]
PAIR_ORDER = ["WUI-P/WUI-S", "WUI-P/WUI-Z", "WUI-S/WUI-Z"]

DIVISION_BY_STATE = {
    "CT": "New England",
    "ME": "New England",
    "MA": "New England",
    "NH": "New England",
    "RI": "New England",
    "VT": "New England",
    "NJ": "Middle Atlantic",
    "NY": "Middle Atlantic",
    "PA": "Middle Atlantic",
    "IN": "East North Central",
    "IL": "East North Central",
    "MI": "East North Central",
    "OH": "East North Central",
    "WI": "East North Central",
    "IA": "West North Central",
    "KS": "West North Central",
    "MN": "West North Central",
    "MO": "West North Central",
    "NE": "West North Central",
    "ND": "West North Central",
    "SD": "West North Central",
    "DE": "South Atlantic",
    "DC": "South Atlantic",
    "FL": "South Atlantic",
    "GA": "South Atlantic",
    "MD": "South Atlantic",
    "NC": "South Atlantic",
    "SC": "South Atlantic",
    "VA": "South Atlantic",
    "WV": "South Atlantic",
    "AL": "East South Central",
    "KY": "East South Central",
    "MS": "East South Central",
    "TN": "East South Central",
    "AR": "West South Central",
    "LA": "West South Central",
    "OK": "West South Central",
    "TX": "West South Central",
    "AZ": "Mountain",
    "CO": "Mountain",
    "ID": "Mountain",
    "MT": "Mountain",
    "NV": "Mountain",
    "NM": "Mountain",
    "UT": "Mountain",
    "WY": "Mountain",
    "CA": "Pacific",
    "OR": "Pacific",
    "WA": "Pacific",
}

DIVISION_ORDER = [
    "New England",
    "Middle Atlantic",
    "East North Central",
    "West North Central",
    "South Atlantic",
    "East South Central",
    "West South Central",
    "Mountain",
    "Pacific",
]

DIVISION_ABBR = {
    "New England": "NE",
    "Middle Atlantic": "MA",
    "East North Central": "ENC",
    "West North Central": "WNC",
    "South Atlantic": "SA",
    "East South Central": "ESC",
    "West South Central": "WSC",
    "Mountain": "MTN",
    "Pacific": "PAC",
}

PAIR_COLORS = {
    "WUI-P/WUI-S": "#3568A8",
    "WUI-P/WUI-Z": "#E17C05",
    "WUI-S/WUI-Z": "#4C9F70",
}

PAIR_MARKERS = {
    "WUI-P/WUI-S": "o",
    "WUI-P/WUI-Z": "s",
    "WUI-S/WUI-Z": "^",
}


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


def build_division_summary() -> pd.DataFrame:
    area = pd.read_csv(AREA_CSV)
    population = pd.read_csv(POP_CSV)
    area = area.loc[area["analysis_panel"].eq(NATIONAL_PANEL)].copy()
    population = population.loc[
        population["analysis_panel"].eq(NATIONAL_PANEL)
    ].copy()

    for frame, label in ((area, "area"), (population, "population")):
        frame["division"] = frame["state"].map(DIVISION_BY_STATE)
        if frame["division"].isna().any():
            missing = sorted(frame.loc[frame["division"].isna(), "state"].unique())
            raise RuntimeError(f"Missing division assignments in {label}: {missing}")
        counts = frame.groupby("method").size().to_dict()
        if counts != {method: 49 for method in METHODS}:
            raise RuntimeError(f"Unexpected {label} counts: {counts}")

    area_summary = (
        area.groupby(["division", "method"], as_index=False)
        .agg(mean_wui_area_km2=("wui_area_km2", "mean"), n_units=("state", "size"))
    )
    pop_summary = (
        population.groupby(["division", "method"], as_index=False)
        .agg(
            mean_population_share_pct=("wui_population_share_pct", "mean"),
            n_units_population=("state", "size"),
        )
    )
    summary = area_summary.merge(
        pop_summary,
        on=["division", "method"],
        validate="one_to_one",
    )
    if not np.array_equal(summary["n_units"], summary["n_units_population"]):
        raise RuntimeError("Area and population division counts differ")
    summary = summary.drop(columns="n_units_population")
    summary["division"] = pd.Categorical(
        summary["division"], DIVISION_ORDER, ordered=True
    )
    summary["method"] = pd.Categorical(summary["method"], METHODS, ordered=True)
    return summary.sort_values(["division", "method"]).reset_index(drop=True)


def load_division_geometry() -> gpd.GeoDataFrame:
    # Use Fiona explicitly because the frozen analysis environment contains a
    # pyogrio/GDAL metadata mismatch. This reads the existing file without any
    # environment or package changes.
    states = gpd.read_file(STATE_GPKG, engine="fiona")
    states = states.loc[states["STUSPS"].isin(DIVISION_BY_STATE)].copy()
    states["division"] = states["STUSPS"].map(DIVISION_BY_STATE)
    states = states.to_crs("EPSG:5070")
    divisions = states[["division", "geometry"]].dissolve(by="division").reset_index()
    divisions["division"] = pd.Categorical(
        divisions["division"], DIVISION_ORDER, ordered=True
    )
    return divisions.sort_values("division").reset_index(drop=True)


def nice_ceiling(value: float, step: float) -> float:
    return math.ceil(value / step) * step


def plot_division_panel(
    ax: plt.Axes,
    geometry: gpd.GeoDataFrame,
    summary: pd.DataFrame,
    *,
    method: str,
    value_column: str,
    cmap: str,
    norm: Normalize,
    title: str,
    value_format,
) -> None:
    values = summary.loc[summary["method"].eq(method), ["division", value_column]]
    panel = geometry.merge(values, on="division", validate="one_to_one")
    panel.plot(
        ax=ax,
        column=value_column,
        cmap=cmap,
        norm=norm,
        edgecolor="#2C2C2C",
        linewidth=0.65,
    )
    ax.set_title(title, loc="left", fontweight="semibold", pad=5)
    ax.set_axis_off()
    ax.set_aspect("equal")

    for row in panel.itertuples():
        point = row.geometry.representative_point()
        label = f"{DIVISION_ABBR[str(row.division)]}\n{value_format(getattr(row, value_column))}"
        text = ax.text(
            point.x,
            point.y,
            label,
            ha="center",
            va="center",
            fontsize=7.4,
            fontweight="semibold",
            color="#1D1D1D",
            linespacing=0.9,
        )
        text.set_path_effects(
            [path_effects.withStroke(linewidth=2.2, foreground="white", alpha=0.9)]
        )


def draw_figure5(summary: pd.DataFrame, geometry: gpd.GeoDataFrame) -> list[Path]:
    apply_plot_style()
    pop_max = nice_ceiling(summary["mean_population_share_pct"].max(), 10)
    area_max = nice_ceiling(summary["mean_wui_area_km2"].max(), 5_000)
    pop_norm = Normalize(vmin=0, vmax=pop_max)
    area_norm = Normalize(vmin=0, vmax=area_max)

    fig, axes = plt.subplots(2, 3, figsize=(15.8, 9.4))
    fig.patch.set_facecolor("white")
    panel_letters = ["A", "B", "C", "D", "E", "F"]
    for index, method in enumerate(METHODS):
        plot_division_panel(
            axes[0, index],
            geometry,
            summary,
            method=method,
            value_column="mean_population_share_pct",
            cmap="YlOrRd",
            norm=pop_norm,
            title=f"{panel_letters[index]}  {method} population share",
            value_format=lambda value: f"{value:.1f}%",
        )
        plot_division_panel(
            axes[1, index],
            geometry,
            summary,
            method=method,
            value_column="mean_wui_area_km2",
            cmap="Blues",
            norm=area_norm,
            title=f"{panel_letters[index + 3]}  {method} WUI area",
            value_format=lambda value: f"{value:,.0f}",
        )

    fig.suptitle(
        "Mean WUI Population Share and Area by U.S. Census Divisions",
        fontsize=16,
        fontweight="semibold",
        y=0.975,
    )
    fig.subplots_adjust(left=0.025, right=0.985, top=0.91, bottom=0.11, hspace=0.04, wspace=0.03)

    pop_cax = fig.add_axes([0.09, 0.050, 0.35, 0.018])
    area_cax = fig.add_axes([0.56, 0.050, 0.35, 0.018])
    pop_cb = fig.colorbar(
        ScalarMappable(norm=pop_norm, cmap="YlOrRd"),
        cax=pop_cax,
        orientation="horizontal",
    )
    pop_cb.set_label("Mean population share in WUI (%)", labelpad=3)
    area_cb = fig.colorbar(
        ScalarMappable(norm=area_norm, cmap="Blues"),
        cax=area_cax,
        orientation="horizontal",
    )
    area_cb.set_label(r"Mean WUI area (km$^2$)", labelpad=3)
    area_cb.ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))

    base = OUT_DIR / "Figure5_Census_division_population_area_final"
    outputs = [base.with_suffix(".pdf"), base.with_suffix(".png"), base.with_suffix(".svg")]
    fig.savefig(outputs[0], bbox_inches="tight", facecolor="white")
    fig.savefig(outputs[1], dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(outputs[2], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return outputs


def build_jaccard_data() -> pd.DataFrame:
    data = pd.read_csv(JACCARD_CSV)
    data = data.loc[data["analysis_panel"].eq(SENSITIVITY_PANEL)].copy()
    data = data.loc[data["state"].isin(FOCUS_STATES)].copy()
    data = data[["state", "radius_m", "method_pair", "jaccard"]]

    expected_rows = len(FOCUS_STATES) * 10 * len(PAIR_ORDER)
    if len(data) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} Jaccard rows, found {len(data)}")
    if data["jaccard"].isna().any() or not data["jaccard"].between(0, 1).all():
        raise RuntimeError("Jaccard values are missing or outside [0, 1]")

    mean_data = (
        data.groupby(["radius_m", "method_pair"], as_index=False)["jaccard"]
        .mean()
        .assign(state="Five-state mean")
    )
    combined = pd.concat([data, mean_data], ignore_index=True)
    state_order = FOCUS_STATES + ["Five-state mean"]
    combined["state"] = pd.Categorical(combined["state"], state_order, ordered=True)
    combined["method_pair"] = pd.Categorical(
        combined["method_pair"], PAIR_ORDER, ordered=True
    )
    return combined.sort_values(["state", "method_pair", "radius_m"]).reset_index(drop=True)


def draw_figure9(data: pd.DataFrame) -> list[Path]:
    apply_plot_style()
    states = FOCUS_STATES + ["Five-state mean"]
    state_names = {
        "CA": "California",
        "CO": "Colorado",
        "FL": "Florida",
        "PA": "Pennsylvania",
        "TX": "Texas",
        "Five-state mean": "Five-state mean",
    }
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.6), sharex=True, sharey=True)
    fig.patch.set_facecolor("white")

    handles = []
    labels = []
    for index, (ax, state) in enumerate(zip(axes.flat, states)):
        panel = data.loc[data["state"].eq(state)]
        for pair in PAIR_ORDER:
            line_data = panel.loc[panel["method_pair"].eq(pair)]
            line, = ax.plot(
                line_data["radius_m"],
                line_data["jaccard"],
                color=PAIR_COLORS[pair],
                marker=PAIR_MARKERS[pair],
                markersize=4.8,
                linewidth=2.0,
                label=pair,
            )
            if index == 0:
                handles.append(line)
                labels.append(pair)

        ax.axvline(500, color="#767676", linewidth=1.0, linestyle="--", zorder=0)
        ax.set_title(
            f"{chr(65 + index)}  {state_names[state]}",
            loc="left",
            fontweight="semibold",
        )
        ax.set_xlim(80, 1020)
        ax.set_ylim(0.20, 0.84)
        ax.set_xticks(np.arange(100, 1001, 100))
        ax.set_yticks(np.arange(0.2, 0.81, 0.1))
        ax.grid(color="#D9D9D9", linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#4A4A4A")
        ax.spines["bottom"].set_color("#4A4A4A")
        ax.tick_params(axis="x", rotation=45)

        if state == "Five-state mean":
            for pair, dy in zip(PAIR_ORDER, [0.018, -0.032, 0.018]):
                row = panel.loc[
                    panel["method_pair"].eq(pair) & panel["radius_m"].eq(500)
                ]
                value = float(row["jaccard"].iloc[0])
                ax.annotate(
                    f"{value:.3f}",
                    xy=(500, value),
                    xytext=(515, value + dy),
                    fontsize=8.5,
                    fontweight="semibold",
                    color=PAIR_COLORS[pair],
                    path_effects=[
                        path_effects.withStroke(linewidth=2.2, foreground="white")
                    ],
                )

    axes[0, 0].set_ylabel("Jaccard similarity")
    axes[1, 0].set_ylabel("Jaccard similarity")
    for ax in axes[1, :]:
        ax.set_xlabel("Neighborhood radius (m)")

    fig.suptitle(
        "Pairwise WUI Agreement Across Neighborhood Radii",
        fontsize=16,
        fontweight="semibold",
        y=0.98,
    )
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=3,
        frameon=False,
    )
    fig.text(
        0.5,
        0.018,
        "The dashed line marks the standardized 500 m setting. WUI-Z is fixed; radius varies only for WUI-P and WUI-S.",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#444444",
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.88, bottom=0.13, hspace=0.24, wspace=0.10)

    base = OUT_DIR / "Figure9_Jaccard_sensitivity_5states_final"
    outputs = [base.with_suffix(".pdf"), base.with_suffix(".png"), base.with_suffix(".svg")]
    fig.savefig(outputs[0], bbox_inches="tight", facecolor="white")
    fig.savefig(outputs[1], dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(outputs[2], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return outputs


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    division_summary = build_division_summary()
    division_geometry = load_division_geometry()
    division_source = OUT_DIR / "Figure5_Census_division_population_area_source.csv"
    division_summary.to_csv(division_source, index=False, float_format="%.12f")
    figure5_outputs = draw_figure5(division_summary, division_geometry)

    jaccard_data = build_jaccard_data()
    jaccard_source = OUT_DIR / "Figure9_Jaccard_sensitivity_5states_source.csv"
    jaccard_data.to_csv(jaccard_source, index=False, float_format="%.12f")
    figure9_outputs = draw_figure9(jaccard_data)

    outputs = [division_source, *figure5_outputs, jaccard_source, *figure9_outputs]
    manifest = OUT_DIR / "Figures5_9_sha256_manifest.txt"
    manifest.write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in outputs),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
