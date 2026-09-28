#!/usr/bin/env bash
# One command for the whole triaged ablation sweep on the 10% subset.
#
#   stage 0  build_10pct_subset.slurm                    single job,  CPU
#   stage 1  train_gate_feature_ladder_10pct.slurm        array 1-12,  GPU  (Priority 1)
#   stage 2  analyze_attention_10pct.slurm                single job,  GPU  (Priority 2)
#   stage 3  train_gate_leave_one_out_10pct.slurm         array 1-9,   GPU  (Priority 3)
#   stage 3  train_gate_architecture_ablation_10pct.slurm array 1-6,   GPU  (Priority 4b)
#   stage 3  train_gate_recon_weight_zero_10pct.slurm     array 1-3,   GPU  (Priority 4a)
#
# Priority order, per Jake: (1) feature ladder is the one that would actually
# go in a figure; (2) attention interpretability next; (3) and (4) after
# that, roughly tied, submitted with increasing --nice so they queue behind
# (1) and (2) under contention without blocking on them (they read the same
# subset and split but ask independent questions).
#
# Stage 2 depends on stage 1 finishing (it reads one of stage 1's
# checkpoints). Stage 3's three jobs depend only on stage 1's task 1 having
# written the shared split manifest, not on the full ladder array completing
# -- submitted with --dependency=afterany:<ladder array>_1 so they can start
# as soon as that single task is done, and --nice so they yield GPUs to the
# ladder and the attention analysis if both are runnable at once.
#
# Safe to rerun: build_10pct_subset.slurm's output already exists check is in
# make_subset_dataset.py's caller semantics (it just rewrites, deterministic
# given the same seed), and every training job's --output-dir is per-run, so
# nothing overwrites a completed result silently.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder}"
cd "$PROJECT_DIR"

SUBSET_JOB_ID=$(sbatch --parsable cluster/build_10pct_subset.slurm)
echo "Stage 0 (build 10% subset):        $SUBSET_JOB_ID"

LADDER_JOB_ID=$(sbatch --parsable --dependency="afterok:$SUBSET_JOB_ID" cluster/train_gate_feature_ladder_10pct.slurm)
echo "Stage 1 (feature ladder, PRIORITY 1): $LADDER_JOB_ID  [array 1-12]"

ATTENTION_JOB_ID=$(sbatch --parsable --dependency="afterok:$LADDER_JOB_ID" cluster/analyze_attention_10pct.slurm)
echo "Stage 2 (attention analysis, PRIORITY 2): $ATTENTION_JOB_ID  [depends on stage 1 finishing]"

# Only the split-manifest write (ladder array task 1) needs to have happened
# for stages 3's jobs to be able to start; afterany on that one task index
# lets them begin without waiting for the whole 12-task ladder array.
LOO_JOB_ID=$(sbatch --parsable --dependency="afterany:${LADDER_JOB_ID}_1" cluster/train_gate_leave_one_out_10pct.slurm)
echo "Stage 3 (leave-one-out, PRIORITY 3):   $LOO_JOB_ID  [array 1-9, nice=200]"

ARCH_JOB_ID=$(sbatch --parsable --dependency="afterany:${LADDER_JOB_ID}_1" cluster/train_gate_architecture_ablation_10pct.slurm)
echo "Stage 3 (gat vs egnn, PRIORITY 4b):    $ARCH_JOB_ID  [array 1-6, nice=250]"

RECON0_JOB_ID=$(sbatch --parsable --dependency="afterany:${LADDER_JOB_ID}_1" cluster/train_gate_recon_weight_zero_10pct.slurm)
echo "Stage 3 (reconstruction weight=0, PRIORITY 4a): $RECON0_JOB_ID  [array 1-3, nice=300]"

cat <<SUMMARY

Queued. Priority 1 (feature ladder) and Priority 2 (attention analysis) run
un-niced and start as soon as their dependency is satisfied. Priorities 3
and 4 start as soon as the split manifest exists (after ladder task 1) but
queue behind 1 and 2 if GPUs are contended.

Results:
  gate_run/feature_ladder_10pct/gat4_residual_<rung>_seed<seed>/
  gate_run/feature_ladder_10pct/attention_analysis/
  gate_run/leave_one_out_10pct/gat4_residual_minus_<feature>_seed<seed>/
  gate_run/architecture_ablation_10pct/<gat|egnn>_seed<seed>/
  gate_run/recon_weight_zero_10pct/gat4_residual_recon0_seed<seed>/

NOT included (no code change made, or genuinely blocked -- see each slurm
script's header for why):
  - "+electrostatics" rung of the feature ladder: V_ae is not committed as a
    per-decoy feature anywhere (Table 4.1: planned). This is a data-
    generation project (APBS at ~150k-model scale), not a training run.
  - "With/without edge conditioning" (Priority 4): needs a GATConv variant
    with edge_dim disabled; not wired up in GATE_model.py.
  - Correlation of attention against |Delta phi_r| (Priority 2): same
    blocker as the electrostatics rung.

Watch with:  squeue -u "\$USER"
SUMMARY
