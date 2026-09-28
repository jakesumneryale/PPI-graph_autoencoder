"""Parsers for the raw per-decoy score files used throughout
Protein_interface_assessment (both the 84-target and the ZDOCK
Benchmark-5.5 "scores_sampled_<target>" directories share this format).

Byte-exact validated against a target with an independently-published,
known-correct combined score table (1ACB): every parser here reproduces
all 961 raw decoy scores to zero absolute difference against
all_supersampled_balanced_datasets/1acb_supersampled_balanced_scores.csv,
and the resulting Spearman rho against DockQ matches
supersampled_correlation_dataset_final_manuscript.csv to 4 decimal places
for all 7 methods once computed on the full 1394-decoy balanced set (see
validate_against_1acb() below). See gate_run/ch3_comparison/README.md.
"""
from __future__ import annotations
import re
from pathlib import Path

import pandas as pd


def _key(name: str) -> str:
    """Normalize a decoy identifier across every file format used here."""
    name = name.strip()
    if name.endswith(".pdb"):
        name = name[:-4]
    if name.startswith("NoH_"):
        name = name[4:]
    return name


def base_key(name: str) -> str:
    """Collapse a decoy id down to its 'complex.X_Y_Z' stem.

    Needed to join against all_supersampled_balanced_datasets, whose
    'Decoy' column omits the '_corrected_H_NNNN' suffix that the raw
    per-method score files carry.
    """
    return re.sub(r"_corrected.*$", "", _key(name))


def parse_dockq(path: Path) -> dict[str, float]:
    """Parse a DockQ.py-format 'dockq_scores_*.txt' transcript."""
    scores: dict[str, float] = {}
    current = None
    for line in path.read_text(errors="replace").splitlines():
        m = re.match(r"DockQ Data for (\S+)", line)
        if m:
            current = _key(m.group(1))
            continue
        m = re.match(r"DockQ\s+([\-0-9.]+)", line)
        if m and current is not None:
            scores[current] = float(m.group(1))
            current = None
    return scores


def parse_zrank(path: Path) -> dict[str, float]:
    out = {}
    for line in path.read_text(errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        out[_key(parts[0])] = float(parts[1])
    return out


def parse_rosetta(path: Path) -> dict[str, float]:
    """Rosetta .sc-style output: SCORE: total_score ... description."""
    out = {}
    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith("SCORE:"):
            continue
        parts = line.split()
        if parts[1] == "total_score":  # header line
            continue
        out[_key(parts[-1])] = float(parts[1])
    return out


def parse_voromqa(path: Path) -> dict[str, float]:
    """VoroMQA CAD-like output; 4th whitespace-separated numeric field
    is the per-model global VoroMQA score (validated against 1ACB)."""
    out = {}
    for line in path.read_text(errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        out[_key(parts[0])] = float(parts[4])
    return out


def parse_pydock(path: Path) -> dict[str, float]:
    """pyDock tab-separated output; 6th field is the Total score."""
    out = {}
    for line in path.read_text(errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 6:
            continue
        out[_key(parts[0])] = float(parts[5])
    return out


def parse_itscorepp(path: Path) -> dict[str, float]:
    out = {}
    for line in path.read_text(errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        out[_key(parts[0])] = float(parts[-1])
    return out


def parse_deeprank_csv(path: Path) -> dict[str, float]:
    df = pd.read_csv(path)
    return {_key(r.pdb_id): float(r.predicted_fnat) for r in df.itertuples()}


def parse_dove(path: Path) -> dict[str, float]:
    out = {}
    lines = path.read_text(errors="replace").splitlines()
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) < 2:
            parts = line.split()
        if len(parts) < 2:
            continue
        out[_key(parts[0])] = float(parts[1])
    return out


def parse_svr_csv(path: Path) -> dict[str, float]:
    df = pd.read_csv(path)
    return {base_key(r.decoy): float(r.svr_score) for r in df.itertuples()}


PARSERS = {
    "ZRank2": parse_zrank,
    "Rosetta": parse_rosetta,
    "VoroMQA": parse_voromqa,
    "PyDock": parse_pydock,
    "ITScorePP": parse_itscorepp,
}

# Whether a higher raw score means a better (more native-like) model.
# False methods are energy-like (lower = better); the AUC/rho computation
# flips the sign for these so that AUC > 0.5 always means "better than chance".
HIGHER_IS_BETTER = {
    "ZRank2": False,
    "Rosetta": False,
    "VoroMQA": True,
    "PyDock": False,
    "ITScorePP": False,
    "Deeprank-GNN-ESM": True,
    "GNN-DOVE": True,
    "SVR": True,
    "GATE": True,
}
