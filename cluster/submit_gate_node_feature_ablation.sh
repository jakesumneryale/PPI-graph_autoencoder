#!/usr/bin/env bash
# Submit interface-node-degree preprocessing, then release the GPU array only
# if every HDF5 model was updated or already contained a valid feature.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder}"
cd "$PROJECT_DIR"

PREPROCESS_JOB_ID=$(sbatch --parsable cluster/add_interface_node_degree.slurm)
echo "Submitted interface-node-degree preprocessing job: $PREPROCESS_JOB_ID"

ARRAY_JOB_ID=$(sbatch \
  --parsable \
  --dependency="afterok:$PREPROCESS_JOB_ID" \
  cluster/train_gate_node_feature_ablation.slurm)
echo "Submitted dependent GATE ablation array: $ARRAY_JOB_ID"
echo "The array will start only after preprocessing completes successfully."
