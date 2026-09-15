#!/usr/bin/env python3
# Copyright (C) 2026 WUI Mapping Project Authors
# SPDX-License-Identifier: GPL-3.0-only
# Repository version: filesystem paths are loaded through repo_config.py.
# Copy config/paths.example.json to config/paths.json before a new run.
# Raw input data and large intermediate files are stored outside GitHub.

"""Generate Appendix A (Tables A1--A6) from the frozen Step51 CSV files."""

from __future__ import annotations

import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "step51_wuiz_ketchpaw_gt50_20260731T215644Z"
    / "moran_appendix"
    / "appendix_tables"
)
OUTDIR = ROOT / "manuscript_appendix_revision_20260803"

FILES = {
    "A1": "Appendix_A1_patch75_full_design_candidate.csv",
    "A2": "Appendix_A2_patch75_full_design_candidate.csv",
    "A3": "Appendix_A3_patch75_full_design_candidate.csv",
    "A4": "Appendix_A4_patch75_full_design_candidate.csv",
    "A5": "Appendix_A5_patch75_full_design_candidate.csv",
    "A6": "Appendix_A6_patch75_full_design_candidate.csv",
}

FOCUS_ORDER = {s: i for i, s in enumerate(["CA", "CO", "FL", "PA", "TX"])}
METHOD_ORDER = {"WUI-Z": 0, "WUI-P": 1, "WUI-S": 2}
PAIR_ORDER = {"WUI-P/WUI-S": 0, "WUI-P/WUI-Z": 1, "WUI-S/WUI-Z": 2}
VAR_ORDER = {"p_a": 0, "p_s": 1}


def read_rows(name: str) -> list[dict[str, str]]:
    path = SOURCE / FILES[name]
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in (None, "") else math.nan


def integer(value: float) -> str:
    if math.isnan(value):
        return "--"
    return f"{round(value):,}"


def decimal(value: float, places: int = 1) -> str:
    if math.isnan(value):
        return "--"
    return f"{value:,.{places}f}"


def probability(value: float) -> str:
    if math.isnan(value):
        return "--"
    if value < 0.001:
        return "$<0.001$"
    return f"{value:.3f}"


def setting(row: dict[str, str], key: str = "buffer_m") -> str:
    if row.get("method") == "WUI-Z":
        return "Fixed"
    return str(round(num(row, key)))


def nowrap(value: str) -> str:
    """Prevent short identifiers such as WUI-P and CONUS from wrapping."""
    return f"\\mbox{{{value}}}"


def panel_sort(rows: list[dict[str, str]], *, pair: bool = False, moran: bool = False):
    def key(row: dict[str, str]):
        panel = row["analysis_panel"]
        state = row["state"]
        state_key = FOCUS_ORDER.get(state, 99) if panel.startswith("five_state") else state
        radius_key = num(row, "radius_m") if pair else num(row, "buffer_m")
        if row.get("method") == "WUI-Z":
            radius_key = -1
        method_key = PAIR_ORDER.get(row.get("method_pair", ""), METHOD_ORDER.get(row.get("method", ""), 9))
        variable_key = VAR_ORDER.get(row.get("variable", ""), 0) if moran else 0
        return (state_key, method_key, radius_key, variable_key)

    return sorted(rows, key=key)


def table_start(caption: str, label: str, columns: str, header: str, *, size: str = "footnotesize") -> list[str]:
    ncols = header.count("&") + 1
    return [
        "\\begingroup",
        f"\\{size}",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\renewcommand{\\arraystretch}{1.05}",
        f"\\begin{{longtable}}{{@{{}}{columns}@{{}}}}",
        f"\\caption{{{caption}}} \\\\",
        f"\\label{{{label}}} \\\\",
        "\\toprule",
        header + " \\\\",
        "\\midrule",
        "\\endfirsthead",
        f"\\multicolumn{{{ncols}}}{{c}}{{{{\\bfseries \\tablename\\ \\thetable{{}} -- continued}}}} \\\\",
        "\\toprule",
        header + " \\\\",
        "\\midrule",
        "\\endhead",
        "\\midrule",
        f"\\multicolumn{{{ncols}}}{{r}}{{{{Continued on next page}}}} \\\\",
        "\\bottomrule",
        "\\endfoot",
        "\\bottomrule",
        "\\endlastfoot",
    ]


def panel_heading(title: str, ncols: int) -> list[str]:
    return [f"\\multicolumn{{{ncols}}}{{l}}{{\\textbf{{\\textit{{{title}}}}}}} \\\\", "\\midrule"]


def table_end() -> list[str]:
    return ["\\end{longtable}", "\\endgroup", ""]


def a1() -> list[str]:
    rows = read_rows("A1")
    out = table_start(
        "\\textbf{WUI area by method and analysis setting.} Panel A reports the five focus states across the tested radii. WUI-Z is fixed and does not use a circular radius. Panel B reports all 49 state-level units at the common 500~m setting for WUI-P and WUI-S. Areas are in km$^2$.",
        "tab:a_wui_area_breakdown",
        "L{1.4cm}L{1.6cm}C{1.6cm}R{2.55cm}R{2.55cm}R{2.55cm}R{2.55cm}",
        "\\textbf{State} & \\textbf{Method} & \\textbf{Setting (m)} & \\textbf{Non-WUI} & \\textbf{Intermix} & \\textbf{Interface} & \\textbf{Total WUI}",
    )
    for panel, title in [
        ("five_state_radius_sensitivity", "Panel A: Five-state radius sensitivity"),
        ("national49_standardized_500m", "Panel B: National standardized comparison"),
    ]:
        out += panel_heading(title, 7)
        subset = panel_sort([r for r in rows if r["analysis_panel"] == panel])
        for r in subset:
            out.append(
                f"{nowrap(r['state'])} & {nowrap(r['method'])} & {setting(r)} & "
                f"{decimal(num(r, 'nonwui_area_km2'))} & {decimal(num(r, 'intermix_area_km2'))} & "
                f"{decimal(num(r, 'interface_area_km2'))} & {decimal(num(r, 'wui_area_km2'))} \\\\")
        if panel.startswith("national49"):
            out.append("\\midrule")
            for method in ["WUI-P", "WUI-S", "WUI-Z"]:
                x = [r for r in subset if r["method"] == method]
                out.append(
                    f"{nowrap('CONUS')} & {nowrap(method)} & {'Fixed' if method == 'WUI-Z' else '500'} & "
                    f"{decimal(sum(num(r, 'nonwui_area_km2') for r in x))} & "
                    f"{decimal(sum(num(r, 'intermix_area_km2') for r in x))} & "
                    f"{decimal(sum(num(r, 'interface_area_km2') for r in x))} & "
                    f"{decimal(sum(num(r, 'wui_area_km2') for r in x))} \\\\")
        out.append("\\midrule")
    return out + table_end()


def a2() -> list[str]:
    rows = read_rows("A2")
    out = table_start(
        "\\textbf{Source features inside mapped WUI.} WUI-P counts retained OpenAddresses records, whereas WUI-S counts building-footprint centroids. These two feature types are reported separately and should not be read as the same physical unit.",
        "tab:a_wui_structure_breakdown",
        "L{1.4cm}L{1.6cm}C{1.6cm}R{2.55cm}R{2.55cm}R{2.55cm}R{2.55cm}",
        "\\textbf{State} & \\textbf{Method} & \\textbf{Radius (m)} & \\textbf{All features} & \\textbf{Intermix} & \\textbf{Interface} & \\textbf{In WUI}",
    )
    for panel, title in [
        ("five_state_radius_sensitivity", "Panel A: Five-state radius sensitivity"),
        ("national49_standardized_500m", "Panel B: National standardized comparison"),
    ]:
        out += panel_heading(title, 7)
        subset = panel_sort([r for r in rows if r["analysis_panel"] == panel])
        for r in subset:
            out.append(
                f"{nowrap(r['state'])} & {nowrap(r['method'])} & {setting(r)} & {integer(num(r, 'Total_struct'))} & "
                f"{integer(num(r, 'Intermix_struct'))} & {integer(num(r, 'Interface_struct'))} & "
                f"{integer(num(r, 'WUI_struct'))} \\\\")
        if panel.startswith("national49"):
            out.append("\\midrule")
            for method in ["WUI-P", "WUI-S"]:
                x = [r for r in subset if r["method"] == method]
                out.append(
                    f"{nowrap('CONUS')} & {nowrap(method)} & 500 & {integer(sum(num(r, 'Total_struct') for r in x))} & "
                    f"{integer(sum(num(r, 'Intermix_struct') for r in x))} & "
                    f"{integer(sum(num(r, 'Interface_struct') for r in x))} & "
                    f"{integer(sum(num(r, 'WUI_struct') for r in x))} \\\\")
        out.append("\\midrule")
    return out + table_end()


def a3() -> list[str]:
    rows = read_rows("A3")
    out = table_start(
        "\\textbf{Estimated resident population by WUI class (2020).} Panel A reports the five focus states, and Panel B reports the 49 state-level units. Population was allocated using 2020 Census data.",
        "tab:a_wui_population_2020",
        "L{1.4cm}L{1.6cm}C{1.4cm}R{2.2cm}R{2.2cm}R{2.2cm}R{2.25cm}R{1.1cm}",
        "\\textbf{State} & \\textbf{Method} & \\textbf{Setting (m)} & \\textbf{Non-WUI} & \\textbf{Intermix} & \\textbf{Interface} & \\textbf{Total WUI} & \\textbf{WUI (\\%)}",
        size="footnotesize",
    )
    for panel, title in [
        ("five_state_radius_sensitivity", "Panel A: Five-state radius sensitivity"),
        ("national49_standardized_500m", "Panel B: National standardized comparison"),
    ]:
        out += panel_heading(title, 8)
        subset = panel_sort([r for r in rows if r["analysis_panel"] == panel])
        for r in subset:
            out.append(
                f"{nowrap(r['state'])} & {nowrap(r['method'])} & {setting(r)} & {integer(num(r, 'nonwui_population'))} & "
                f"{integer(num(r, 'intermix_population'))} & {integer(num(r, 'interface_population'))} & "
                f"{integer(num(r, 'wui_population'))} & {decimal(num(r, 'wui_population_share_pct'), 2)} \\\\")
        if panel.startswith("national49"):
            out.append("\\midrule")
            for method in ["WUI-P", "WUI-S", "WUI-Z"]:
                x = [r for r in subset if r["method"] == method]
                non = sum(num(r, "nonwui_population") for r in x)
                intermix = sum(num(r, "intermix_population") for r in x)
                interface = sum(num(r, "interface_population") for r in x)
                wui = intermix + interface
                total = sum(num(r, "total_population") for r in x)
                out.append(
                    f"{nowrap('CONUS')} & {nowrap(method)} & {'Fixed' if method == 'WUI-Z' else '500'} & {integer(non)} & "
                    f"{integer(intermix)} & {integer(interface)} & {integer(wui)} & {decimal(100*wui/total, 2)} \\\\")
        out.append("\\midrule")
    return out + table_end()


def status_label(row: dict[str, str]) -> str:
    status = row.get("status", "")
    return {
        "FORMAL_COMPLETE": "Formal",
        "EXCLUDED_N_LT_30": "$n<30$",
        "UNDEFINED": "Undefined",
    }.get(status, status.replace("_", "\\_"))


def a4() -> list[str]:
    rows = read_rows("A4")
    out = table_start(
        "\\textbf{Global Moran's $I$ results.} The variables are county-level WUI area share ($p_a$) and source-feature share ($p_s$). Formal national inference was limited to state--method--variable combinations with at least 30 counties or county-equivalent units. Smaller networks are listed as $n<30$, and the District of Columbia is undefined.",
        "tab:a_morans_i",
        "L{1.5cm}L{1.7cm}C{1.6cm}C{1.6cm}R{1.1cm}R{1.5cm}R{1.5cm}R{1.6cm}L{2.4cm}",
        "\\textbf{State} & \\textbf{Method} & \\textbf{Setting} & \\textbf{Variable} & $n$ & $I$ & $z$ & $p$ & \\textbf{Status}",
        size="footnotesize",
    )
    for panel, title in [
        ("five_state_radius_sensitivity", "Panel A: Five-state radius sensitivity"),
        ("national49_standardized_500m", "Panel B: National standardized comparison"),
    ]:
        out += panel_heading(title, 9)
        subset = panel_sort([r for r in rows if r["analysis_panel"] == panel], moran=True)
        for r in subset:
            out.append(
                f"{nowrap(r['state'])} & {nowrap(r['method'])} & {setting(r)} & ${r['variable']}$ & "
                f"{integer(num(r, 'n_units'))} & {decimal(num(r, 'moran_i'), 4)} & "
                f"{decimal(num(r, 'z_norm'), 3)} & {probability(num(r, 'p_norm'))} & {status_label(r)} \\\\")
        out.append("\\midrule")
    return out + table_end()


def a5() -> list[str]:
    rows = read_rows("A5")
    out = table_start(
        "\\textbf{Pairwise WUI intersection.} Intermix and interface were combined as binary WUI before comparison. Areas were calculated only where both products had valid data.",
        "tab:a_spatial_agreement",
        "L{1.6cm}C{2.2cm}L{3.4cm}R{4.4cm}R{4.0cm}",
        "\\textbf{State} & \\textbf{Radius (m)} & \\textbf{Method pair} & \\textbf{Intersection pixels} & \\textbf{Intersection (km$^2$)}",
    )
    for panel, title in [
        ("five_state_radius_sensitivity", "Panel A: Five-state radius sensitivity"),
        ("national49_standardized_500m", "Panel B: National standardized comparison"),
    ]:
        out += panel_heading(title, 5)
        subset = panel_sort([r for r in rows if r["analysis_panel"] == panel], pair=True)
        for r in subset:
            out.append(
                f"{nowrap(r['state'])} & {round(num(r, 'radius_m'))} & {nowrap(r['method_pair'])} & "
                f"{integer(num(r, 'intersection_pixels'))} & {decimal(num(r, 'intersection_area_km2'))} \\\\")
        if panel.startswith("national49"):
            out.append("\\midrule")
            for pair in ["WUI-P/WUI-S", "WUI-P/WUI-Z", "WUI-S/WUI-Z"]:
                x = [r for r in subset if r["method_pair"] == pair]
                out.append(
                    f"{nowrap('CONUS')} & 500 & {nowrap(pair)} & {integer(sum(num(r, 'intersection_pixels') for r in x))} & "
                    f"{decimal(sum(num(r, 'intersection_area_km2') for r in x))} \\\\")
        out.append("\\midrule")
    return out + table_end()


def a6() -> list[str]:
    rows = read_rows("A6")
    out = table_start(
        "\\textbf{Pairwise Jaccard agreement.} Class 0 was retained as valid non-WUI, while true NoData was excluded. National micro Jaccard values were calculated after summing intersections and unions across the 49 reporting units.",
        "tab:a_jaccard_similarity",
        "L{1.6cm}C{2.0cm}L{3.3cm}R{3.7cm}R{3.4cm}R{1.2cm}",
        "\\textbf{State} & \\textbf{Radius (m)} & \\textbf{Method pair} & \\textbf{Common valid pixels} & \\textbf{Union (km$^2$)} & $J$",
        size="footnotesize",
    )
    for panel, title in [
        ("five_state_radius_sensitivity", "Panel A: Five-state radius sensitivity"),
        ("national49_standardized_500m", "Panel B: National standardized comparison"),
    ]:
        out += panel_heading(title, 6)
        subset = panel_sort([r for r in rows if r["analysis_panel"] == panel], pair=True)
        for r in subset:
            out.append(
                f"{nowrap(r['state'])} & {round(num(r, 'radius_m'))} & {nowrap(r['method_pair'])} & "
                f"{integer(num(r, 'common_valid_pixels'))} & {decimal(num(r, 'union_area_km2'))} & "
                f"{decimal(num(r, 'jaccard'), 4)} \\\\")
        if panel.startswith("national49"):
            out.append("\\midrule")
            a5_rows = read_rows("A5")
            a5_national = [r for r in a5_rows if r["analysis_panel"] == panel]
            for pair in ["WUI-P/WUI-S", "WUI-P/WUI-Z", "WUI-S/WUI-Z"]:
                x = [r for r in subset if r["method_pair"] == pair]
                ix = [r for r in a5_national if r["method_pair"] == pair]
                intersection = sum(num(r, "intersection_pixels") for r in ix)
                union = sum(num(r, "union_pixels") for r in x)
                out.append(
                    f"{nowrap('CONUS')} & 500 & {nowrap(pair)} & {integer(sum(num(r, 'common_valid_pixels') for r in x))} & "
                    f"{decimal(sum(num(r, 'union_area_km2') for r in x))} & {decimal(intersection/union, 4)} \\\\")
        out.append("\\midrule")
    return out + table_end()


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    content = [
        "\\appendix",
        "\\section*{Appendix A. Supplementary Tables}",
        "\\addcontentsline{toc}{section}{Appendix A. Supplementary Tables}",
        "",
        "\\renewcommand{\\thetable}{A\\arabic{table}}",
        "\\setcounter{table}{0}",
        "\\renewcommand{\\thefigure}{A\\arabic{figure}}",
        "\\setcounter{figure}{0}",
        "\\newcolumntype{L}[1]{>{\\raggedright\\arraybackslash}p{#1}}",
        "\\newcolumntype{C}[1]{>{\\centering\\arraybackslash}p{#1}}",
        "\\newcolumntype{R}[1]{>{\\raggedleft\\arraybackslash}p{#1}}",
        "",
    ]
    for fn in [a1, a2, a3, a4, a5, a6]:
        content.extend(fn())
    output = OUTDIR / "appendix_A_step51_final_wide_v2.tex"
    output.write_text("\n".join(content) + "\n", encoding="utf-8")

    source_lines = ["table,source_csv,rows"]
    for table, filename in FILES.items():
        source_lines.append(f"{table},{SOURCE / filename},{len(read_rows(table))}")
    (OUTDIR / "appendix_data_sources.csv").write_text("\n".join(source_lines) + "\n", encoding="utf-8")

    readme = """# Appendix A Step51 update

This directory contains a manuscript-ready Appendix A generated from the final
Step51 WUI-Z (>50% intermix / <=50% interface) downstream outputs. WUI-P and
WUI-S values come from the frozen strict-patch national rebuild inherited by
Step51. The reference list was not changed.

Use `appendix_A_step51_final_wide_v2.tex` to replace the existing appendix file in
Overleaf. The source CSV for every table is recorded in
`appendix_data_sources.csv`.
"""
    (OUTDIR / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
