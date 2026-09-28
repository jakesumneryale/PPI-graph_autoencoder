"""Loads GATE's own test_predictions.csv into the same per-target decoy-table
shape the baseline loaders use, so it drops into metrics.py unchanged.

Point GATE_PREDICTIONS_PATH at the finished seed-replicate run once the
cluster job completes:
    gate_run/seed_replicates/gat4_residual_full_seed<N>/test_predictions.csv
(that run trains on exactly the 105/12/29 split used throughout this module,
seed=5, so its test set *is* the 29 targets loaded here -- no leakage check
needed beyond confirming the target list matches, which load_gate_table does).
"""
from __future__ import annotations
from pathlib import Path

import pandas as pd

# Set this once the cluster run finishes and has been rsynced locally.
GATE_PREDICTIONS_PATH: Path | None = None


def load_gate_tables(pdb_ids: list[str], predictions_path: Path | None = None) -> dict[str, pd.DataFrame]:
    """Returns {target: DataFrame[decoy, DockQ, GATE]} or {} if not yet available."""
    path = predictions_path or GATE_PREDICTIONS_PATH
    if path is None or not Path(path).is_file():
        return {}

    raw = pd.read_csv(path)
    have = set(raw["target_id"].unique())
    missing = set(pdb_ids) - have
    if missing:
        raise RuntimeError(
            f"GATE predictions at {path} are missing target(s) {sorted(missing)}; "
            "this run's test split does not match the 29-target split used here."
        )

    tables = {}
    for pdb in pdb_ids:
        sub = raw[raw["target_id"] == pdb]
        tables[pdb] = pd.DataFrame({
            "decoy": sub["graph_name"].to_numpy(),
            "DockQ": sub["true_target"].to_numpy(),
            "GATE": sub["predicted_target"].to_numpy(),
        })
    return tables


def merge_gate_column(tables: dict[str, pd.DataFrame], gate_tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Left-joins a 'GATE' column onto each baseline table by decoy id."""
    if not gate_tables:
        for df in tables.values():
            df["GATE"] = float("nan")
        return tables
    merged = {}
    for target, df in tables.items():
        g = gate_tables[target][["decoy", "GATE"]]
        merged[target] = df.merge(g, on="decoy", how="left")
    return merged
