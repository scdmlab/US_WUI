#!/usr/bin/env python3
# Copyright (C) 2026 WUI Mapping Project Authors
# SPDX-License-Identifier: GPL-3.0-only
# Repository version: filesystem paths are loaded through repo_config.py.
# Copy config/paths.example.json to config/paths.json before a new run.
# Raw input data and large intermediate files are stored outside GitHub.

"""Create manuscript-ready replacements for the two Methodology figures.

The script is deliberately non-destructive: it writes PNG, PDF, and SVG
versions to a caller-supplied output directory and never edits Overleaf files.
"""

from __future__ import annotations

from repo_config import portable_path

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon


STATE_GPKG = Path(
    portable_path("data", "MBF+NLCD_2022US/Inputs/Processed_GPKG/tl_2022_us_state.gpkg")
)

REGION_NAMES = {
    "1": "Northeast",
    "2": "Midwest",
    "3": "South",
    "4": "West",
}

DIVISION_NAMES = {
    "1": "New England",
    "2": "Middle Atlantic",
    "3": "East North Central",
    "4": "West North Central",
    "5": "South Atlantic",
    "6": "East South Central",
    "7": "West South Central",
    "8": "Mountain",
    "9": "Pacific",
}

REGION_COLORS = {
    "1": "#CFE8F3",
    "2": "#DDECCF",
    "3": "#F6DFC8",
    "4": "#DDD8EE",
}

FOCUS = {"CA", "CO", "TX", "FL", "PA"}
FOCUS_COLOR = "#D55E00"
INK = "#263238"
MUTED = "#65727A"
LINE = "#65727A"


def save_all(fig: plt.Figure, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(output / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def study_area(output: Path) -> None:
    states = gpd.read_file(STATE_GPKG, engine="fiona")
    states["STATEFP"] = states["STATEFP"].astype(str).str.zfill(2)
    states["REGION"] = states["REGION"].astype(str)
    states["DIVISION"] = states["DIVISION"].astype(str)
    states = states[~states["STUSPS"].isin(["AK", "HI", "PR", "AS", "GU", "MP", "VI"])].copy()
    if len(states) != 49 or set(states["STUSPS"]) & FOCUS != FOCUS:
        raise RuntimeError("Expected 48 conterminous states plus DC and all five focus states")

    divisions = states.dissolve(by="DIVISION", as_index=False)
    regions = states.dissolve(by="REGION", as_index=False)

    fig = plt.figure(figsize=(14.0, 7.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[4.6, 1.35], wspace=0.02)
    ax = fig.add_subplot(gs[0, 0])
    side = fig.add_subplot(gs[0, 1])

    for region, part in states.groupby("REGION"):
        part.plot(
            ax=ax,
            facecolor=REGION_COLORS[region],
            edgecolor="white",
            linewidth=0.45,
            zorder=1,
        )
    divisions.boundary.plot(ax=ax, color=INK, linewidth=1.15, zorder=3)
    regions.boundary.plot(ax=ax, color=INK, linewidth=2.0, zorder=4)
    states.boundary.plot(ax=ax, color="#6F7B82", linewidth=0.38, zorder=2)

    focus = states[states["STUSPS"].isin(FOCUS)]
    focus.plot(
        ax=ax,
        facecolor=FOCUS_COLOR,
        edgecolor=INK,
        linewidth=1.25,
        zorder=5,
    )

    for row in focus.itertuples(index=False):
        point = row.geometry.representative_point()
        ax.text(
            point.x,
            point.y,
            row.STUSPS,
            ha="center",
            va="center",
            fontsize=10.5,
            fontweight="bold",
            color="white",
            zorder=6,
        )

    dc = states.loc[states["STUSPS"].eq("DC")].iloc[0]
    dc_point = dc.geometry.representative_point()
    ax.scatter([dc_point.x], [dc_point.y], s=34, color=INK, zorder=7)
    ax.annotate(
        "DC",
        xy=(dc_point.x, dc_point.y),
        xytext=(18, -8),
        textcoords="offset points",
        fontsize=9,
        fontweight="bold",
        color=INK,
        arrowprops={"arrowstyle": "-", "color": INK, "lw": 0.8},
        zorder=7,
    )

    region_offsets = {"1": (0, 150000), "2": (0, 0), "3": (120000, -80000), "4": (-100000, 100000)}
    for row in regions.itertuples(index=False):
        p = row.geometry.representative_point()
        dx, dy = region_offsets[row.REGION]
        ax.text(
            p.x + dx,
            p.y + dy,
            REGION_NAMES[row.REGION],
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color=INK,
            alpha=0.78,
            zorder=6,
        )

    ax.set_axis_off()
    ax.margins(0.01)

    side.set_axis_off()
    side.set_xlim(0, 1)
    side.set_ylim(0, 1)
    side.text(0.02, 0.97, "U.S. Census divisions", fontsize=12, fontweight="bold", color=INK, va="top")
    y = 0.91
    grouping = [
        ("Northeast", ["New England", "Middle Atlantic"]),
        ("Midwest", ["East North Central", "West North Central"]),
        ("South", ["South Atlantic", "East South Central", "West South Central"]),
        ("West", ["Mountain", "Pacific"]),
    ]
    focus_divisions = {"Middle Atlantic", "South Atlantic", "West South Central", "Mountain", "Pacific"}
    for region_name, names in grouping:
        code = next(k for k, value in REGION_NAMES.items() if value == region_name)
        side.add_patch(FancyBboxPatch((0.02, y - 0.021), 0.035, 0.035, boxstyle="round,pad=0.002", facecolor=REGION_COLORS[code], edgecolor=INK, linewidth=0.6))
        side.text(0.075, y, region_name, fontsize=10.3, fontweight="bold", color=INK, va="center")
        y -= 0.049
        for name in names:
            marker = "●" if name in focus_divisions else "•"
            color = FOCUS_COLOR if name in focus_divisions else MUTED
            side.text(0.08, y, marker, fontsize=10, color=color, va="center")
            side.text(0.135, y, name, fontsize=9.3, color=INK, va="center")
            y -= 0.045
        y -= 0.018

    side.plot([0.02, 0.96], [0.22, 0.22], color="#B0B7BB", lw=0.8)
    side.scatter([0.04], [0.17], s=85, color=FOCUS_COLOR, edgecolor=INK, linewidth=0.7)
    side.text(0.11, 0.17, "Five sensitivity states", fontsize=9.5, color=INK, va="center")
    side.text(0.02, 0.115, "CA · CO · TX · FL · PA", fontsize=10.3, fontweight="bold", color=INK)
    side.text(0.02, 0.065, "National analysis: 48 states + DC", fontsize=9.2, color=MUTED)
    side.text(0.02, 0.025, "No Midwestern focus state", fontsize=9.2, color=MUTED)

    save_all(fig, output, "target_state_revised")


def rounded_box(ax, xy, width, height, text, *, face="#F3F5F6", edge=LINE,
                fontsize=10.0, weight="normal", text_color=INK, radius=0.15, lw=1.2):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle=f"round,pad=0.03,rounding_size={radius}",
        facecolor=face, edgecolor=edge, linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, color=text_color, linespacing=1.25)
    return patch


def diamond(ax, center, width, height, text, *, face="#FFFFFF", edge=LINE, fontsize=10.2):
    cx, cy = center
    points = [(cx, cy + height / 2), (cx + width / 2, cy),
              (cx, cy - height / 2), (cx - width / 2, cy)]
    patch = Polygon(points, closed=True, facecolor=face, edgecolor=edge, linewidth=1.3)
    ax.add_patch(patch)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=INK, linespacing=1.2)
    return patch


def arrow(ax, start, end, *, label=None, label_xy=None, color=LINE, lw=1.35):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12,
                                linewidth=lw, color=color, shrinkA=1, shrinkB=1))
    if label:
        x, y = label_xy if label_xy is not None else ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        ax.text(x, y, label, fontsize=10.8, fontweight="bold", color=INK,
                ha="center", va="center", bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0})


def workflow(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(14.0, 8.4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9.2)
    ax.axis("off")

    # Method-specific inputs.
    rounded_box(ax, (0.35, 7.45), 3.9, 1.15,
                "WUI-Z · Census-block product\n2020 housing-unit density\nBlock-level wildland vegetation",
                face="#E8EEF5", edge="#557A95", fontsize=10.1, weight="bold")
    rounded_box(ax, (5.05, 7.45), 3.9, 1.15,
                "WUI-S · Footprint-derived product\nBuilding polygons → centroids\n30 m additive count raster",
                face="#E6F1E8", edge="#598B65", fontsize=10.1, weight="bold")
    rounded_box(ax, (9.75, 7.45), 3.9, 1.15,
                "WUI-P · Address-point product\nOpenAddresses → P2 exact-record dedup\n30 m additive count raster",
                face="#F7EADB", edge="#B06A36", fontsize=10.1, weight="bold")

    rounded_box(ax, (3.75, 5.90), 5.30, 1.00,
                "Shared classification layers\n$D_m$: method-specific development proxy\n$V_{m,local}$: local wildland vegetation",
                face="#F4F5F6", edge=LINE, fontsize=9.5, weight="bold")
    arrow(ax, (2.30, 7.43), (4.55, 6.84))
    arrow(ax, (7.00, 7.43), (6.40, 6.84))
    arrow(ax, (11.70, 7.43), (8.25, 6.84))

    # Shared decision tree.
    diamond(ax, (2.75, 4.35), 2.55, 1.12, "$D_m > 6.17$\nunits km$^{-2}$?")
    arrow(ax, (5.45, 5.89), (3.20, 4.86))

    diamond(ax, (6.45, 4.35), 2.55, 1.12, "$V_{m,local} > 50\%$?")
    arrow(ax, (4.03, 4.35), (5.16, 4.35), label="Yes", label_xy=(4.59, 4.57))

    rounded_box(ax, (0.50, 2.22), 2.55, 0.82, "Non-WUI\nclass 0", face="#E4E7E9", edge="#717A80", fontsize=10.7, weight="bold")
    arrow(ax, (2.75, 3.78), (1.78, 3.07), label="No", label_xy=(2.12, 3.53))

    rounded_box(ax, (4.35, 2.22), 2.55, 0.82, "Intermix WUI\nclass 1", face="#CFE8D2", edge="#4D7E57", fontsize=10.7, weight="bold")
    arrow(ax, (6.45, 3.78), (5.63, 3.07), label="Yes", label_xy=(6.13, 3.50))

    diamond(ax, (10.15, 4.35), 2.65, 1.12, "$d_Q \leq 2.4$ km?")
    arrow(ax, (7.73, 4.35), (8.82, 4.35), label="No ($\leq 50\%$)", label_xy=(8.28, 4.65))

    rounded_box(ax, (8.05, 2.22), 2.55, 0.82, "Interface WUI\nclass 2", face="#F5D5B8", edge="#A96331", fontsize=10.7, weight="bold")
    arrow(ax, (10.15, 3.78), (9.33, 3.07), label="Yes", label_xy=(9.84, 3.50))

    rounded_box(ax, (11.10, 2.22), 2.40, 0.82, "Non-WUI\nclass 0", face="#E4E7E9", edge="#717A80", fontsize=10.7, weight="bold")
    arrow(ax, (11.48, 4.35), (12.30, 3.07), label="No", label_xy=(11.96, 3.76))

    rounded_box(ax, (9.45, 5.88), 4.10, 1.00,
                "Qualifying patch $Q$\nsource blocks: wildland vegetation $>75\%$\ndissolved contiguous area $\geq5$ km$^2$",
                face="#EEF3E3", edge="#768C4D", fontsize=9.15, weight="bold")
    arrow(ax, (11.05, 5.87), (10.55, 4.90), color="#768C4D")

    rounded_box(ax, (0.45, 0.62), 13.1, 0.95,
                "WUI-P/WUI-S sensitivity: circular radii 100–1000 m  ·  standardized national setting: 500 m\n"
                "WUI-Z is fixed and radius-independent  ·  outside valid reporting-unit domain: NoData 255",
                face="#F7F8F8", edge="#AAB1B5", fontsize=9.7)
    ax.text(7.0, 0.22,
            "The common numerical density threshold is applied to non-equivalent development proxies.",
            ha="center", va="center", fontsize=9.2, color=MUTED, style="italic")

    save_all(fig, output, "workflow_revised")


def workflow_original_style(output: Path) -> None:
    """Rebuild the classification diagram in the visual language of the original.

    This version deliberately stays close to the manuscript figure: a navy input
    panel, white variable boxes, pale-yellow decisions, and grey/green/orange
    outcomes.  Unlike the original, every branch is explicit and logically correct.
    """
    navy = "#102F50"
    cream = "#FFF3C4"
    cream_edge = "#8A762C"
    grey = "#B8C4CF"
    green = "#A5D86E"
    orange = "#FFB23C"
    dark = "#1A1A1A"
    arrow_color = "#333333"

    fig, ax = plt.subplots(figsize=(16.0, 7.2))
    ax.set_xlim(0, 16.2)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    # Original-style input card.
    rounded_box(ax, (0.20, 2.18), 1.82, 3.92, "", face="#FFFFFF",
                edge="#4B4B4B", radius=0.24, lw=1.4)
    ax.add_patch(plt.Rectangle((0.20, 5.35), 1.82, 0.75,
                               facecolor=navy, edgecolor=navy))
    ax.text(1.11, 5.72, "INPUT", ha="center", va="center",
            fontsize=21, fontweight="bold", color="white")
    # A restrained spatial-layer icon that remains legible after reduction.
    ax.add_patch(plt.Rectangle((0.48, 4.15), 0.68, 0.72,
                               facecolor="#D8E8F4", edgecolor="#9EB4C4", lw=0.8))
    ax.add_patch(plt.Rectangle((1.03, 4.15), 0.68, 0.72,
                               facecolor="#DCECCF", edgecolor="#A4BC91", lw=0.8))
    ax.plot([0.56, 0.74, 0.91, 1.07], [4.31, 4.64, 4.37, 4.72],
            color="#6792B1", lw=1.3)
    ax.plot([1.10, 1.26, 1.43, 1.64], [4.31, 4.59, 4.39, 4.68],
            color="#6C9B55", lw=1.3)
    ax.text(1.11, 3.34, "Spatial\nlayers", ha="center", va="center",
            fontsize=14.2, fontweight="bold", color=dark, linespacing=1.15)

    # Three operational variables, as in the original figure.
    rounded_box(ax, (2.43, 5.24), 2.56, 1.35,
                "Development\ndensity, $D$\n(units/km$^2$)",
                face="#FAFAFA", edge="#555555", fontsize=13.2,
                weight="bold", radius=0.20, lw=1.2)
    rounded_box(ax, (2.43, 3.67), 2.56, 1.35,
                "Local wildland\nvegetation, $V_{local}$\n(%)",
                face="#FAFAFA", edge="#555555", fontsize=12.8,
                weight="bold", radius=0.20, lw=1.2)
    rounded_box(ax, (2.43, 2.10), 2.56, 1.35,
                "Euclidean distance\nto qualifying\npatch $Q$ (km)",
                face="#FAFAFA", edge="#555555", fontsize=12.4,
                weight="bold", radius=0.20, lw=1.2)

    # Input connectors.
    for yy in (5.92, 4.35, 2.78):
        arrow(ax, (2.03, 4.10), (2.40, yy), color=arrow_color, lw=1.25)

    # Correct shared decision tree.
    diamond(ax, (6.30, 4.55), 2.40, 2.12,
            "Development\ndensity\n$D>6.17$?", face=cream,
            edge=cream_edge, fontsize=12.8)
    diamond(ax, (9.55, 4.55), 2.40, 2.12,
            "Local wildland\nvegetation\n$V_{local}>50\%$?", face=cream,
            edge=cream_edge, fontsize=12.3)
    diamond(ax, (12.75, 4.55), 2.40, 2.12,
            "Distance to\nqualifying patch\n$d_Q\leq2.4$ km?", face=cream,
            edge=cream_edge, fontsize=12.1)

    arrow(ax, (4.99, 4.55), (5.08, 4.55), color=arrow_color, lw=1.35)
    arrow(ax, (7.52, 4.55), (8.33, 4.55), label="YES",
          label_xy=(7.92, 4.88), color=arrow_color, lw=1.35)
    arrow(ax, (10.77, 4.55), (11.53, 4.55), label="NO",
          label_xy=(11.15, 4.88), color=arrow_color, lw=1.35)

    # Outcomes. The branch direction is now the inverse of the incorrect original.
    rounded_box(ax, (5.00, 0.82), 2.60, 1.08, "NON-WUI",
                face=grey, edge="#4F5962", fontsize=15.0,
                weight="bold", radius=0.16, lw=1.25)
    rounded_box(ax, (8.25, 0.82), 2.60, 1.08, "INTERMIX\nWUI",
                face=green, edge="#4E683B", fontsize=15.0,
                weight="bold", radius=0.16, lw=1.25)
    rounded_box(ax, (11.45, 0.82), 2.60, 1.08, "INTERFACE\nWUI",
                face=orange, edge="#7D5B2D", fontsize=15.0,
                weight="bold", radius=0.16, lw=1.25)
    rounded_box(ax, (14.32, 4.02), 1.58, 1.06, "NON-WUI",
                face=grey, edge="#4F5962", fontsize=13.4,
                weight="bold", radius=0.16, lw=1.25)

    arrow(ax, (6.30, 3.47), (6.30, 1.93), label="NO",
          label_xy=(6.58, 2.72), color=arrow_color, lw=1.35)
    arrow(ax, (9.55, 3.47), (9.55, 1.93), label="YES",
          label_xy=(9.84, 2.72), color=arrow_color, lw=1.35)
    arrow(ax, (12.75, 3.47), (12.75, 1.93), label="YES",
          label_xy=(13.04, 2.72), color=arrow_color, lw=1.35)
    arrow(ax, (13.97, 4.55), (14.29, 4.55), label="NO",
          label_xy=(14.13, 4.88), color=arrow_color, lw=1.35)

    # Strict qualifying-patch definition, visually separated from the tree.
    rounded_box(ax, (10.12, 5.92), 5.26, 1.12,
                "Qualifying patch $Q$\nWildland vegetation $>75\%$ of valid land\nContiguous area $\geq5$ km$^2$",
                face="#EEF4DF", edge="#73864B", fontsize=11.2,
                weight="bold", radius=0.12, lw=1.15)
    arrow(ax, (12.75, 5.89), (12.75, 5.63), color="#73864B", lw=1.15)

    save_all(fig, output, "workflow_corrected_original_style_v3")


def workflow_method_specific_final(output: Path) -> None:
    """Final manuscript classification diagram with method-specific D clarified."""
    navy = "#102F50"
    cream = "#FFF3C4"
    cream_edge = "#8A762C"
    grey = "#B8C4CF"
    green = "#A5D86E"
    orange = "#FFB23C"
    dark = "#1A1A1A"
    arrow_color = "#333333"

    fig, ax = plt.subplots(figsize=(16.4, 7.7))
    ax.set_xlim(0, 16.4)
    ax.set_ylim(0, 7.45)
    ax.axis("off")

    # Input card.
    rounded_box(ax, (0.20, 2.35), 1.82, 3.92, "", face="#FFFFFF",
                edge="#4B4B4B", radius=0.24, lw=1.4)
    ax.add_patch(plt.Rectangle((0.20, 5.52), 1.82, 0.75,
                               facecolor=navy, edgecolor=navy))
    ax.text(1.11, 5.89, "INPUT", ha="center", va="center",
            fontsize=21, fontweight="bold", color="white")
    ax.add_patch(plt.Rectangle((0.48, 4.32), 0.68, 0.72,
                               facecolor="#D8E8F4", edgecolor="#9EB4C4", lw=0.8))
    ax.add_patch(plt.Rectangle((1.03, 4.32), 0.68, 0.72,
                               facecolor="#DCECCF", edgecolor="#A4BC91", lw=0.8))
    ax.plot([0.56, 0.74, 0.91, 1.07], [4.48, 4.81, 4.54, 4.89],
            color="#6792B1", lw=1.3)
    ax.plot([1.10, 1.26, 1.43, 1.64], [4.48, 4.76, 4.56, 4.85],
            color="#6C9B55", lw=1.3)
    ax.text(1.11, 3.51, "Spatial\nlayers", ha="center", va="center",
            fontsize=14.2, fontweight="bold", color=dark, linespacing=1.15)

    # Operational inputs. D is explicitly identified as a method-specific proxy.
    rounded_box(ax, (2.38, 5.30), 2.78, 1.48,
                "Method-specific\ndevelopment-density\nproxy, $D$\n(units/km$^2$)",
                face="#FAFAFA", edge="#555555", fontsize=11.7,
                weight="bold", radius=0.20, lw=1.2)
    rounded_box(ax, (2.38, 3.78), 2.78, 1.28,
                "Local wildland\nvegetation, $V_{local}$\n(%)",
                face="#FAFAFA", edge="#555555", fontsize=12.6,
                weight="bold", radius=0.20, lw=1.2)
    rounded_box(ax, (2.38, 2.28), 2.78, 1.28,
                "Euclidean distance\nto qualifying\npatch $Q$ (km)",
                face="#FAFAFA", edge="#555555", fontsize=12.2,
                weight="bold", radius=0.20, lw=1.2)
    for yy in (6.04, 4.42, 2.92):
        arrow(ax, (2.03, 4.27), (2.35, yy), color=arrow_color, lw=1.25)

    # Shared classification decisions.
    diamond(ax, (6.77, 4.62), 2.42, 2.12,
            "Development-\ndensity proxy\n$D>6.17$?", face=cream,
            edge=cream_edge, fontsize=12.4)
    diamond(ax, (9.73, 4.62), 2.42, 2.12,
            "Local wildland\nvegetation\n$V_{local}>50\%$?", face=cream,
            edge=cream_edge, fontsize=12.2)
    diamond(ax, (12.98, 4.62), 2.42, 2.12,
            "Distance to\nqualifying patch\n$d_Q\leq2.4$ km?", face=cream,
            edge=cream_edge, fontsize=12.0)

    arrow(ax, (5.16, 6.04), (5.54, 4.62), color=arrow_color, lw=1.35)
    arrow(ax, (7.99, 4.62), (8.50, 4.62), label="YES",
          label_xy=(8.25, 4.95), color=arrow_color, lw=1.35)
    arrow(ax, (10.95, 4.62), (11.75, 4.62), label="NO",
          label_xy=(11.35, 4.95), color=arrow_color, lw=1.35)

    # Outcomes.
    rounded_box(ax, (5.47, 1.28), 2.60, 1.08, "NON-WUI",
                face=grey, edge="#4F5962", fontsize=15.0,
                weight="bold", radius=0.16, lw=1.25)
    rounded_box(ax, (8.43, 1.28), 2.60, 1.08, "INTERMIX\nWUI",
                face=green, edge="#4E683B", fontsize=15.0,
                weight="bold", radius=0.16, lw=1.25)
    rounded_box(ax, (11.68, 1.28), 2.60, 1.08, "INTERFACE\nWUI",
                face=orange, edge="#7D5B2D", fontsize=15.0,
                weight="bold", radius=0.16, lw=1.25)
    rounded_box(ax, (14.55, 4.09), 1.58, 1.06, "NON-WUI",
                face=grey, edge="#4F5962", fontsize=13.4,
                weight="bold", radius=0.16, lw=1.25)

    arrow(ax, (6.77, 3.54), (6.77, 2.39), label="NO",
          label_xy=(7.05, 3.00), color=arrow_color, lw=1.35)
    arrow(ax, (9.73, 3.54), (9.73, 2.39), label="YES",
          label_xy=(10.02, 3.00), color=arrow_color, lw=1.35)
    arrow(ax, (12.98, 3.54), (12.98, 2.39), label="YES",
          label_xy=(13.27, 3.00), color=arrow_color, lw=1.35)
    arrow(ax, (14.20, 4.62), (14.52, 4.62), label="NO",
          label_xy=(14.36, 4.95), color=arrow_color, lw=1.35)

    rounded_box(ax, (10.25, 6.03), 5.30, 1.08,
                "Qualifying patch $Q$\nWildland vegetation $>75\%$ of valid land\nContiguous area $\geq5$ km$^2$",
                face="#EEF4DF", edge="#73864B", fontsize=11.1,
                weight="bold", radius=0.12, lw=1.15)
    arrow(ax, (12.98, 6.00), (12.98, 5.70), color="#73864B", lw=1.15)

    # Compact note makes the proxy distinction explicit without crowding the tree.
    rounded_box(
        ax,
        (0.42, 0.10),
        15.55,
        0.72,
        "$D$ is method-specific: WUI-Z = 2020 Census housing units; "
        "WUI-S = building-footprint centroids; WUI-P = retained address records.\n"
        "Spatial unit: Census block for WUI-Z; circular neighborhood on the 30 m raster grid "
        "for WUI-P/WUI-S (100--1000 m sensitivity; 500 m national setting).",
        face="#F7F8F8",
        edge="#AAB1B5",
        fontsize=9.1,
        radius=0.10,
        lw=0.9,
    )

    save_all(fig, output, "Figure2_WUI_classification_workflow_final")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    study_area(output)
    workflow(output)
    workflow_original_style(output)


if __name__ == "__main__":
    main()
