"""Baseline scoring-function comparison figure for dissertation Sec. 4.4.2,
restricted to the 29-target test set of the current (146-target, seed-5)
GATE split -- reproducing the style of Fig. 3.4 (per-target rho) and Fig. 3.7
(aggregate rho and AUC vs DockQ0) so Chapters 3 and 4 read against each other
directly.

Two output modes, controlled by INCLUDE_GATE below:
  * INCLUDE_GATE = False (default): the eight baselines only. This is the
    version currently in the dissertation (Fig. 4.9) -- a standalone
    baseline-difficulty figure for the test set, independent of GATE.
  * INCLUDE_GATE = True: adds a ninth GATE series once a finished run is
    wired up in gate_loader.py (GATE_PREDICTIONS_PATH). Writes to a
    separate "_with_gate" filename so it never silently overwrites the
    baseline-only figure already cited in the text.

Usage:
    cd gate_run/ch3_comparison
    ../../.venv/bin/python build_comparison_figure.py
"""
from __future__ import annotations
import json
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from load_targets import load_test_set, targets_in_84
from metrics import METHODS, DOCKQ0_GRID, build_summary, target_auc_curve
from gate_loader import load_gate_tables, merge_gate_column

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLIT_PATH = REPO_ROOT / "target_splits.json"
THESIS_FIG_DIR = Path.home() / "Desktop" / "phd_thesis" / "chapter_4_figs"

INCLUDE_GATE = False
OUT_STEM = "Fig4_6_baseline_comparison" if not INCLUDE_GATE else "Fig4_6_baseline_comparison_with_gate"

STYLE = {  # (color, marker) -- first six match Figs. 3.4/3.7 exactly
    "ZRank2":           ("midnightblue", "^"),
    "ITScorePP":        ("turquoise",    "o"),
    "Rosetta":          ("forestgreen",  "s"),
    "PyDock":           ("darkorange",   "h"),
    "VoroMQA":          ("darkorchid",   "*"),
    "Deeprank-GNN-ESM": ("tomato",       "D"),
    "GNN-DOVE":         ("saddlebrown",  "p"),
    "SVR":              ("crimson",      ">"),
    "GATE":             ("black",        "o"),
}


def thesis_style() -> None:
    mpl.rcParams.update({
        "text.usetex": shutil.which("latex") is not None or Path("/Library/TeX/texbin/latex").exists(),
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
        "xtick.labelsize": 8, "ytick.labelsize": 9, "legend.fontsize": 8,
        "axes.spines.top": True, "axes.spines.right": True,
        "savefig.dpi": 300,
    })
    if mpl.rcParams["text.usetex"]:
        import os
        os.environ["PATH"] = "/Library/TeX/texbin:" + os.environ.get("PATH", "")


def panel_label(ax, letter: str, x: float = -0.11, y: float = 1.05) -> None:
    text = rf"\textbf{{({letter})}}" if mpl.rcParams["text.usetex"] else f"({letter})"
    ax.text(x, y, text, transform=ax.transAxes, ha="left", va="bottom", fontsize=12)


def panel_pertarget(ax, summary: pd.DataFrame, order: list[str], methods: list[str]) -> None:
    x = np.arange(len(order))
    for method in methods:
        color, marker = STYLE[method]
        sub = summary[summary.method == method].set_index("target").reindex(order)
        rho = sub["rho"].to_numpy(dtype=float)
        finite = np.isfinite(rho)
        ax.scatter(x[finite], rho[finite], s=26, marker=marker,
                   facecolors="none" if method != "GATE" else color,
                   edgecolors=color, linewidths=1.0, label=method, zorder=3)

    pivot = summary.pivot_table(index="target", columns="method", values="rho").reindex(order)
    mean_rho = pivot[[m for m in methods if m != "GATE"]].mean(axis=1, skipna=True)
    std_rho = pivot[[m for m in methods if m != "GATE"]].std(axis=1, skipna=True)
    ax.errorbar(x, mean_rho, yerr=std_rho, color="black", lw=1.2, marker=".", ms=4,
               zorder=4, label="Average")
    ax.set_xticks(x, order, rotation=90, fontsize=6.5)
    ax.set_ylabel(r"$|\rho|$")
    ax.set_xlabel("Test target")
    ax.set_ylim(0, 1.05)
    panel_label(ax, "a")


def panel_aggregate_rho(ax, summary: pd.DataFrame, methods: list[str]) -> list[str]:
    agg = summary.groupby("method")["rho"].agg(["mean", "std", "count"]).reindex(methods)
    order = agg["mean"].dropna().sort_values().index.tolist()
    order += [m for m in methods if m not in order]
    x = np.arange(len(order))
    for i, method in enumerate(order):
        color, marker = STYLE[method]
        row = agg.loc[method]
        if not np.isfinite(row["mean"]):
            continue
        ax.errorbar(i, row["mean"], yerr=row["std"], color=color, marker=marker, ms=9,
                   markerfacecolor=color if method == "GATE" else "none",
                   markeredgecolor=color, mew=1.3, lw=1.4, capsize=0)
        ax.text(i, row["mean"] + row["std"] + 0.02, f"{row['mean']:.2f}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x, order, rotation=35, ha="right", fontsize=7.5)
    ax.set_ylabel(r"$\langle|\rho|\rangle_t$")
    ax.set_ylim(0, 1.05)
    panel_label(ax, "b")
    return order


def panel_auc_curve(ax, tables: dict[str, pd.DataFrame], methods: list[str]) -> None:
    for method in methods:
        color, marker = STYLE[method]
        curves = [target_auc_curve(df, method) for df in tables.values() if df[method].notna().sum() >= 10]
        if not curves:
            continue
        mean_curve = np.nanmean(np.vstack(curves), axis=0)
        ax.plot(DOCKQ0_GRID, mean_curve, color=color, marker=marker, ms=4,
               mfc=color if method == "GATE" else "none", mec=color, lw=1.2, label=method)
    ax.axhline(0.5, color="0.7", lw=0.7, ls=":")
    ax.set_xlabel(r"${\rm DockQ}_0$")
    ax.set_ylabel(r"$\langle{\rm AUC}\rangle_t$")
    ax.set_xlim(0.19, 0.81)
    panel_label(ax, "c")


def main() -> None:
    split = json.loads(SPLIT_PATH.read_text())
    test_paths = split["splits"]["test"]["paths"]
    test_targets = sorted(p.split("/")[-1].replace(".hdf5", "") for p in test_paths)
    in84, bm55 = targets_in_84(test_targets)
    print(f"Test set: {len(test_targets)} targets ({len(in84)} in the 84-PDB set, {len(bm55)} from BM5.5)")

    tables = load_test_set(test_targets)
    methods = list(METHODS)
    if INCLUDE_GATE:
        gate_tables = load_gate_tables(test_targets)
        tables = merge_gate_column(tables, gate_tables)
        methods = methods + ["GATE"]
        print(f"GATE predictions: {'LOADED (' + str(len(gate_tables)) + ' targets)' if gate_tables else 'not available -- placeholder only'}")

    summary = build_summary(tables, methods=methods)
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)
    summary.to_csv(out_dir / f"{OUT_STEM}_per_target_summary.csv", index=False)

    thesis_style()
    fig = plt.figure(figsize=(8.0, 7.3))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.85, wspace=0.32,
                          left=0.08, right=0.98, top=0.90, bottom=0.16)
    ax_top = fig.add_subplot(gs[0, :])
    ax_agg = fig.add_subplot(gs[1, 0])
    ax_auc = fig.add_subplot(gs[1, 1])

    baseline_methods = [m for m in methods if m != "GATE"]
    mean_rho_for_order = (summary[summary.method.isin(baseline_methods)]
                          .pivot_table(index="target", columns="method", values="rho")
                          .mean(axis=1, skipna=True).sort_values())
    order = mean_rho_for_order.index.tolist()

    panel_pertarget(ax_top, summary, order, methods)
    panel_aggregate_rho(ax_agg, summary, methods)
    panel_auc_curve(ax_auc, tables, methods)

    handles, labels = ax_top.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=5, fontsize=7.5, frameon=False)

    pdf_path = THESIS_FIG_DIR / f"{OUT_STEM}.pdf"
    png_path = THESIS_FIG_DIR / f"{OUT_STEM}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {pdf_path}")
    print(f"Wrote {out_dir / (OUT_STEM + '_per_target_summary.csv')}")

    print(f"\nAggregate |rho| over the {len(test_targets)}-target test set:")
    agg = summary.groupby("method")["rho"].agg(["mean", "std", "count"])
    print(agg.reindex(methods).to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
