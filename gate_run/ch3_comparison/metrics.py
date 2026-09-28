"""Per-target rho / AUC(DockQ0) from a decoy table, for every baseline method."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from raw_score_parsers import HIGHER_IS_BETTER

METHODS = ["ZRank2", "ITScorePP", "Rosetta", "PyDock", "VoroMQA", "Deeprank-GNN-ESM", "GNN-DOVE", "SVR"]
DOCKQ0_GRID = np.round(np.arange(0.20, 0.81, 0.02), 2)


def target_rho(df: pd.DataFrame, method: str) -> float | None:
    """|Spearman rho| between method score and DockQ. None if data are missing."""
    sub = df[[method, "DockQ"]].dropna()
    if len(sub) < 10 or sub[method].std() == 0:
        return None
    return abs(spearmanr(sub[method], sub["DockQ"])[0])


def target_auc(df: pd.DataFrame, method: str, dockq0: float = 0.23) -> float | None:
    sub = df[[method, "DockQ"]].dropna()
    if len(sub) < 10:
        return None
    label = (sub["DockQ"] >= dockq0).astype(int)
    if label.nunique() < 2:
        return None
    x = sub[method].to_numpy()
    if not HIGHER_IS_BETTER[method]:
        x = -x
    return roc_auc_score(label, x)


def target_auc_curve(df: pd.DataFrame, method: str) -> np.ndarray:
    return np.array([target_auc(df, method, d0) for d0 in DOCKQ0_GRID], dtype=float)


def build_summary(tables: dict[str, pd.DataFrame], methods: list[str] = METHODS) -> pd.DataFrame:
    """One row per (target, method): rho, auc@0.23, n_decoys, frac_positive."""
    rows = []
    for target, df in tables.items():
        n_pos = int((df["DockQ"] >= 0.23).sum())
        for method in methods:
            rows.append({
                "target": target,
                "method": method,
                "rho": target_rho(df, method),
                "auc": target_auc(df, method),
                "n_decoys": len(df),
                "n_scored": int(df[method].notna().sum()),
                "frac_positive": n_pos / len(df),
            })
    return pd.DataFrame(rows)
