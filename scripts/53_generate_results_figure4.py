#!/usr/bin/env python3
# Copyright (C) 2026 WUI Mapping Project Authors
# SPDX-License-Identifier: GPL-3.0-only
# Repository version: filesystem paths are loaded through repo_config.py.
# Copy config/paths.example.json to config/paths.json before a new run.
# Raw input data and large intermediate files are stored outside GitHub.

"""Generate the revised Results Figure 4 from the frozen Step51 tables."""

from __future__ import annotations

from repo_config import portable_path

import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter


ROOT = Path(portable_path("project"))
STEP51 = ROOT / "step51_wuiz_ketchpaw_gt50_20260731T215644Z"
TABLE_DIR = STEP51 / "moran_appendix" / "appendix_tables"
AREA_CSV = TABLE_DIR / "Appendix_A1_patch75_full_design_candidate.csv"
POP_CSV = TABLE_DIR / "Appendix_A3_patch75_full_design_candidate.csv"
OUT_DIR = ROOT / "manuscript_results_figure_revision_20260803"

METHODS = ["WUI-Z", "WUI-P", "WUI-S"]
PANEL = "national49_standardized_500m"

COLORS = {
    "population": "#7B61B9",
    "intermix": "#4E79A7",
    "interface": "#F28E2B",
    "total": "#59A14F",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_means() -> pd.DataFrame:
    area = pd.read_csv(AREA_CSV)
    population = pd.read_csv(POP_CSV)

    area = area.loc[area["analysis_panel"].eq(PANEL)].copy()
    population = population.loc[population["analysis_panel"].eq(PANEL)].copy()

    expected = {(method, 49) for method in METHODS}
    area_counts = set(area.groupby("method").size().items())
    population_counts = set(population.groupby("method").size().items())
    if area_counts != expected or population_counts != expected:
        raise RuntimeError(
            "Expected exactly 49 rows per method in both Step51 tables; "
            f"area={area_counts}, population={population_counts}"
        )

    area_means = (
        area.groupby("method", as_index=False)[
            ["intermix_area_km2", "interface_area_km2", "wui_area_km2"]
        ]
        .mean()
    )
    population_means = (
        population.groupby("method", as_index=False)["wui_population_share_pct"]
        .mean()
    )
    summary = area_means.merge(population_means, on="method", validate="one_to_one")
    summary["method"] = pd.Categorical(summary["method"], METHODS, ordered=True)
    summary = summary.sort_values("method").reset_index(drop=True)

    if not np.allclose(
        summary["intermix_area_km2"] + summary["interface_area_km2"],
        summary["wui_area_km2"],
        rtol=0,
        atol=1e-6,
    ):
        raise RuntimeError("Mean intermix and interface areas do not close to total WUI area")

    return summary


def add_bar_labels(ax: plt.Axes, bars, *, fmt, offset: float) -> None:
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            fmt(value),
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="semibold",
            color="#202020",
        )


def draw(summary: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, (ax_pop, ax_area) = plt.subplots(
        1,
        2,
        figsize=(12.6, 5.9),
        gridspec_kw={"width_ratios": [0.82, 1.65], "wspace": 0.23},
    )
    fig.patch.set_facecolor("white")

    x = np.arange(len(METHODS))
    pop = summary["wui_population_share_pct"].to_numpy()
    pop_bars = ax_pop.bar(
        x,
        pop,
        width=0.62,
        color=COLORS["population"],
        edgecolor="white",
        linewidth=0.8,
    )
    ax_pop.set_title("A  Mean population share", loc="left", fontweight="semibold")
    ax_pop.set_ylabel("Population in WUI (%)")
    ax_pop.set_xticks(x, METHODS, fontweight="semibold")
    ax_pop.set_ylim(0, 42)
    ax_pop.set_yticks(np.arange(0, 41, 10))
    add_bar_labels(ax_pop, pop_bars, fmt=lambda v: f"{v:.2f}%", offset=0.75)

    width = 0.24
    intermix = summary["intermix_area_km2"].to_numpy()
    interface = summary["interface_area_km2"].to_numpy()
    total = summary["wui_area_km2"].to_numpy()

    intermix_bars = ax_area.bar(
        x - width,
        intermix,
        width,
        label="Intermix",
        color=COLORS["intermix"],
        edgecolor="white",
        linewidth=0.8,
    )
    interface_bars = ax_area.bar(
        x,
        interface,
        width,
        label="Interface",
        color=COLORS["interface"],
        edgecolor="white",
        linewidth=0.8,
    )
    total_bars = ax_area.bar(
        x + width,
        total,
        width,
        label="Total WUI",
        color=COLORS["total"],
        edgecolor="white",
        linewidth=0.8,
    )
    ax_area.set_title("B  Mean WUI area", loc="left", fontweight="semibold")
    ax_area.set_ylabel(r"WUI area (km$^2$)")
    ax_area.set_xticks(x, METHODS, fontweight="semibold")
    ax_area.set_ylim(0, 26_000)
    ax_area.set_yticks(np.arange(0, 25_001, 5_000))
    ax_area.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax_area.legend(
        loc="upper left",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.0, 1.0),
        borderaxespad=0.15,
    )
    add_bar_labels(ax_area, intermix_bars, fmt=lambda v: f"{v:,.0f}", offset=420)
    add_bar_labels(ax_area, interface_bars, fmt=lambda v: f"{v:,.0f}", offset=420)
    add_bar_labels(ax_area, total_bars, fmt=lambda v: f"{v:,.0f}", offset=420)

    for ax in (ax_pop, ax_area):
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#4A4A4A")
        ax.spines["bottom"].set_color("#4A4A4A")
        ax.tick_params(axis="x", pad=7)

    fig.suptitle(
        "Mean WUI Population Share and Area Across 49 Analysis Units",
        fontsize=16,
        fontweight="semibold",
        y=0.975,
    )
    fig.text(
        0.5,
        0.025,
        "WUI-P and WUI-S use the standardized 500 m setting; WUI-Z is fixed. "
        "Values are unweighted means across 48 states and the District of Columbia.",
        ha="center",
        va="bottom",
        fontsize=9.7,
        color="#444444",
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.86, bottom=0.16)

    base = OUT_DIR / "Figure4_CONUS49_mean_summary_final"
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".png"), dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = load_means()
    source_csv = OUT_DIR / "Figure4_CONUS49_mean_summary_source.csv"
    summary.to_csv(source_csv, index=False, float_format="%.12f")
    draw(summary)

    manifest = OUT_DIR / "Figure4_sha256_manifest.txt"
    outputs = [
        source_csv,
        OUT_DIR / "Figure4_CONUS49_mean_summary_final.pdf",
        OUT_DIR / "Figure4_CONUS49_mean_summary_final.png",
        OUT_DIR / "Figure4_CONUS49_mean_summary_final.svg",
    ]
    manifest.write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in outputs),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
