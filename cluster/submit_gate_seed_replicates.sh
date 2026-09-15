#!/usr/bin/env bash
# One command for the whole experiment: commit the Voronoi missing mask, verify
# the graphs and write a shared split, then run six GPU jobs (base vs full x
# three seeds) against that split.
#
#   stage 1  commit_voronoi_missing_mask.slurm     array 1-146, CPU
#   stage 2  prepare_seed_replicate_splits.slurm   single job,  CPU
#   stage 3  train_gate_seed_replicates.slurm      array 1-6,   GPU
#
# Each stage is chained with afterok, so the GPU jobs only start once every
# graph carries voronoi_contact_missing and the shared split exists. Nothing is
# recomputed: the mask is copied from checkpoints that already contain it.
#
# Safe to rerun. Committed targets are skipped via markers, and an existing
# split manifest is reused rather than rewritten, so a partial run resumes.
#
# Options:
#   SKIP_MASK=1   assume the mask is already committed and skip stage 1
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder}"
cd "$PROJECT_DIR"

if [[ "${SKIP_MASK:-0}" == "1" ]]; then
  echo "SKIP_MASK=1: not submitting the mask-commit stage."
  PREPARE_JOB_ID=$(sbatch --parsable cluster/prepare_seed_replicate_splits.slurm)
  echo "Submitted verify + split-preparation job: $PREPARE_JOB_ID"
else
  MASK_JOB_ID=$(sbatch --parsable cluster/commit_voronoi_missing_mask.slurm)
  echo "Submitted mask-commit array (146 targets): $MASK_JOB_ID"

  PREPARE_JOB_ID=$(sbatch \
    --parsable \
    --dependency="afterok:$MASK_JOB_ID" \
    cluster/prepare_seed_replicate_splits.slurm)
  echo "Submitted verify + split-preparation job: $PREPARE_JOB_ID (after $MASK_JOB_ID)"
fi

ARRAY_JOB_ID=$(sbatch \
  --parsable \
  --dependency="afterok:$PREPARE_JOB_ID" \
  cluster/train_gate_seed_replicates.slurm)
echo "Submitted six-run GPU array: $ARRAY_JOB_ID (after $PREPARE_JOB_ID)"

cat <<SUMMARY

Queued. The GPU array starts only after the mask is committed and verified.

  tasks 1-3  base  seeds 7, 17, 27   (no Voronoi, no interface-node degree)
  tasks 4-6  full  seeds 7, 17, 27   (Voronoi area as an input-only feature)

All six share one target split and one set of normalisation statistics.
Results:  $PROJECT_DIR/gate_run/seed_replicates/gat4_residual_<config>_seed<seed>/

Watch with:  squeue -u "\$USER"
SUMMARY
