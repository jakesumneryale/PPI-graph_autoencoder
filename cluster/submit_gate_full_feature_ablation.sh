#!/usr/bin/env bash
# Add interface-node degree to the full graph files, then release the six-job
# GPU ablation only if every graph was updated or already valid.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder}"
cd "$PROJECT_DIR"

PREPROCESS_JOB_ID=$(sbatch --parsable cluster/add_interface_node_degree_full.slurm)
echo "Submitted full-data interface-degree job: $PREPROCESS_JOB_ID"

ARRAY_JOB_ID=$(sbatch \
  --parsable \
  --dependency="afterok:$PREPROCESS_JOB_ID" \
  cluster/train_gate_full_feature_ablation.slurm)
echo "Submitted dependent six-run GATE array: $ARRAY_JOB_ID"
echo "The array starts only after full-data preprocessing succeeds."
