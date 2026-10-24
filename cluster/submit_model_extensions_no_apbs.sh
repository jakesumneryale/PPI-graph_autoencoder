#!/usr/bin/env bash
# Default: one seed, 11 distinct configurations; full mode uses three seeds.
# Start from the original cluster 10% subset, not the APBS-filtered transfer bundle.
set -euo pipefail
export APBS_ENABLED=0
export PROJECT_DIR="${PROJECT_DIR:-/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder}"
export PRIOR_SPLIT="${PRIOR_SPLIT:-$PROJECT_DIR/gate_run/seed_replicates/shared_target_splits.json}"
if [[ ! -f "$PRIOR_SPLIT" ]]; then
  echo "Shared seed-replicate split missing: $PRIOR_SPLIT. Set PRIOR_SPLIT to its cluster location." >&2
  exit 2
fi
bash "$PROJECT_DIR/cluster/submit_model_extensions.sh" "${1:-quick}"
