"""Per-target decoy tables (DockQ + every baseline score) for the 29-target
test set of gate_run/../target_splits.json (seed 5, 146-target cohort).

16 of the 29 test targets are in the main 84-PDB dataset and are loaded from
the already-validated, pre-balanced per-decoy CSVs. The remaining 13 are
ZDOCK Benchmark-5.5 targets with no equivalent pre-balanced file; those are
parsed directly from the raw 'scores_sampled_<target>' score files with the
validated parsers in raw_score_parsers.py.

IMPORTANT ASYMMETRY (documented, not hidden): the 84-set decoys are rebalanced
to ~51% DockQ >= 0.23 (the final published pipeline); the BM5.5 'sampled' sets
are uniform-in-DockQ-bin but NOT separately rebalanced by class, and come out
~60-75% positive. Both are far better than raw docking output (~5% positive),
but they are not bit-identical procedures. See README.md.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

from raw_score_parsers import (
    PARSERS, parse_dockq, parse_deeprank_csv, parse_dove, parse_svr_csv, base_key,
)

SUPERSAMPLING_DIR = Path("/Users/jakesumner/Desktop/PPI Project_2/supersampling")
BALANCED_DIR = SUPERSAMPLING_DIR / "all_supersampled_balanced_datasets"
SVR_84_DIR = SUPERSAMPLING_DIR / "svr_2_score_results_no_overlap"
BM55_DIR = SUPERSAMPLING_DIR / "bm_55_scoring"
SVR_BM55_DIR = SUPERSAMPLING_DIR / "svr_2_score_results_bm55_test"
CORR_84_CSV = SUPERSAMPLING_DIR / "final_manuscript_datasets" / "supersampled_correlation_dataset_final_manuscript.csv"

# Column-name map: balanced-dataset column -> our canonical method name.
BALANCED_COLUMNS = {
    "ZRank2": "ZRank", "Rosetta": "Rosetta", "VoroMQA": "VoroMQA", "PyDock": "PyDock",
    "ITScorePP": "itscorepp", "Deeprank-GNN-ESM": "Deeprank_gnn_esm", "GNN-DOVE": "Dove_5",
}


def targets_in_84(pdb_ids: list[str]) -> tuple[list[str], list[str]]:
    df = pd.read_csv(CORR_84_CSV)
    in84 = set(df["pdb"].str.lower())
    return [t for t in pdb_ids if t in in84], [t for t in pdb_ids if t not in in84]


def load_84_target(pdb: str) -> pd.DataFrame:
    """DockQ + every baseline score for one 84-set target, one row per decoy."""
    path = BALANCED_DIR / f"{pdb}_supersampled_balanced_scores.csv"
    df = pd.read_csv(path)
    out = pd.DataFrame({"decoy": df["Decoy"], "DockQ": df["DockQ"]})
    for our_name, col in BALANCED_COLUMNS.items():
        out[our_name] = df[col]

    svr_path = SVR_84_DIR / f"svr_loo_rbf_{pdb}_results.csv"
    if svr_path.is_file():
        svr = parse_svr_csv(svr_path)
        out["SVR"] = out["decoy"].map(svr)
    else:
        out["SVR"] = np.nan
    return out


def load_bm55_target(pdb: str) -> pd.DataFrame:
    """DockQ + every baseline score for one BM5.5 target, one row per decoy.

    GNN-DOVE was never scored on the BM5.5 set (no dove_5 files exist in any
    scores_sampled_<target> directory); that column is present but all-NaN.
    """
    d = BM55_DIR / f"scores_sampled_{pdb}"
    dockq = parse_dockq(d / f"dockq_scores_{pdb}_sampled.txt")

    scores: dict[str, dict[str, float]] = {}
    filenames = {
        "ZRank2": f"zrank_files_sampled_{pdb}.txt.zr.out",
        "Rosetta": f"rosetta_scores_sampled_{pdb}.txt",
        "VoroMQA": f"voromqa_scores_{pdb}_sampled.txt",
        "PyDock": f"pydock_scores_{pdb}_sampled.txt",
        "ITScorePP": f"{pdb}_itscorepp_sampled.txt",
    }
    for method, fname in filenames.items():
        fpath = d / fname
        scores[method] = PARSERS[method](fpath) if fpath.is_file() else {}

    deeprank_path = d / f"sampled_{pdb}_deeprank.csv"
    scores["Deeprank-GNN-ESM"] = parse_deeprank_csv(deeprank_path) if deeprank_path.is_file() else {}

    rows = []
    for key, dq in dockq.items():
        row = {"decoy": key, "DockQ": dq}
        for method, d_scores in scores.items():
            row[method] = d_scores.get(key, np.nan)
        rows.append(row)
    out = pd.DataFrame(rows)
    out["GNN-DOVE"] = np.nan

    svr_path = SVR_BM55_DIR / f"bm_55_svr_results_{pdb}.csv"
    if svr_path.is_file():
        svr = parse_svr_csv(svr_path)
        out["SVR"] = out["decoy"].map(lambda k: svr.get(base_key(k), np.nan))
    else:
        out["SVR"] = np.nan
    return out


def load_test_set(pdb_ids: list[str]) -> dict[str, pd.DataFrame]:
    in84, bm55 = targets_in_84(pdb_ids)
    tables = {}
    for pdb in in84:
        tables[pdb] = load_84_target(pdb)
    for pdb in bm55:
        tables[pdb] = load_bm55_target(pdb)
    missing_order = [t for t in pdb_ids if t not in tables]
    if missing_order:
        raise RuntimeError(f"Failed to build tables for: {missing_order}")
    return {t: tables[t] for t in pdb_ids}  # preserve caller order
