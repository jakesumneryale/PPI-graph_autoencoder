#!/usr/bin/env bash
# Copy the dataset-size comparison results (small files only: predictions, metrics, histories, logs; no checkpoints)
# from Bouchet into gate_run/dataset_size_comparison/<run>/gate_rank_seed<seed>/, the layout
# build_dataset_size_comparison_notebook.py reads. Needs an open ssh connection to "bouchet" (log in first; the
# ControlMaster socket in ~/.ssh/config is reused). Rerun as runs finish, then rebuild the notebook.
#
#   bash gate_run/pull_dataset_size_results.sh && python gate_run/build_dataset_size_comparison_notebook.py
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
DEST="$HERE/dataset_size_comparison"
mkdir -p "$DEST"
ssh -o BatchMode=yes bouchet 'cd "$(readlink -f ~/project_pi_co54)/jas485/PPI-graph_autoencoder/gate_run/dataset_size_comparison" &&
  tar czf - --ignore-failed-read --exclude="*.pt" */logs/train_*.out \
    */gate_rank_seed*/{epoch10/*.json,epoch10/*.csv,*.json,*.csv,*.jsonl,run_config.toml} 2>/dev/null' | tar xzf - -C "$DEST"
find "$DEST" -mindepth 3 -maxdepth 3 -name loss_history.csv | sort | while read -r h; do
  echo "$(echo "$h" | sed "s#$DEST/##; s#/loss_history.csv##"): $(( $(wc -l < "$h") - 1 )) epochs$( [[ -f $(dirname "$h")/test_predictions.csv ]] && echo ', final test done')"
done
