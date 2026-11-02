#!/usr/bin/env bash
# Copy the ranking-loss sweep results (predictions, metrics, histories, logs; no checkpoints) from Bouchet into
# gate_run/rank_sweep_expanded/<arm>/gate_rank_seed<seed>/, the layout build_rank_sweep_expanded_notebook.py reads.
# Needs an open ssh connection to "bouchet" (log in first; the ControlMaster socket in ~/.ssh/config is reused).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
DEST="$HERE/rank_sweep_expanded"
mkdir -p "$DEST"
ssh -o BatchMode=yes bouchet 'cd "$(readlink -f ~/project_pi_co54)/jas485/PPI-graph_autoencoder/gate_run/rank_sweep_expanded" &&
  tar czf - --ignore-failed-read --exclude="*.pt" */logs/train_*.out \
    */gate_rank_seed*/{*.json,*.csv,*.jsonl,run_config.toml} 2>/dev/null' | tar xzf - -C "$DEST"
find "$DEST" -mindepth 3 -maxdepth 3 -name loss_history.csv | sort | while read -r h; do
  echo "$(echo "$h" | sed "s#$DEST/##; s#/loss_history.csv##"): $(( $(wc -l < "$h") - 1 )) epochs$( [[ -f $(dirname "$h")/test_predictions.csv ]] && echo ', final test done')"
done
