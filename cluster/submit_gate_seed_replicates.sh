#!/usr/bin/env bash
# Write one shared target split, then release three seed replicates against it.
# The array starts only if the split manifest was written successfully.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder}"
cd "$PROJECT_DIR"

PREPARE_JOB_ID=$(sbatch --parsable cluster/prepare_seed_replicate_splits.slurm)
echo "Submitted split-preparation job: $PREPARE_JOB_ID"

ARRAY_JOB_ID=$(sbatch \
  --parsable \
  --dependency="afterok:$PREPARE_JOB_ID" \
  cluster/train_gate_seed_replicates.slurm)
echo "Submitted dependent three-seed array: $ARRAY_JOB_ID"
echo "Seeds 7, 17, 27 share the split written by job $PREPARE_JOB_ID."
echo "Set CONFIG=base to run the no-Voronoi configuration instead."
