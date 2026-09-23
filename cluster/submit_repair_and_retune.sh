#!/usr/bin/env bash
# Repair interface_node_degree, re-audit, rebuild the cohort on a chosen target split,
# rebuild the ESM2 sidecars, then run the ESM2 tuning sweep. Each stage waits for the
# previous one to succeed, so a failed repair or audit stops the pipeline before any GPU
# time is spent.
set -euo pipefail
export PROJECT_DIR="${PROJECT_DIR:-/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder}"
export DATA_DIR="${DATA_DIR:-/nfs/roberts/project/pi_co54/jas485/ppi_processed_graphs}"
export RSASA_DIR="${RSASA_DIR:-/nfs/roberts/project/pi_co54/jas485/rsasa_i_graph_data}"
export EXPERIMENT_DIR="${EXPERIMENT_DIR:-$PROJECT_DIR/gate_run/full_pooling_repaired}"
export AUDIT_DIR="${AUDIT_DIR:-$PROJECT_DIR/gate_run/full_pooling_audit_repaired}"
export REPAIR_REPORTS="${REPAIR_REPORTS:-$PROJECT_DIR/gate_run/interface_degree_repair}"
export ESM_ROOT="${ESM_ROOT:-/nfs/roberts/pi/pi_co54/nb685/scratch_backup/SS_embeds}"
export ESM_SIDECARS="${ESM_SIDECARS:-$PROJECT_DIR/full_pooling_esm_sidecars_repaired}"
export PRIOR_SPLIT="${PRIOR_SPLIT:-$PROJECT_DIR/target_splits.json}"
export SOURCE_MATRIX="${SOURCE_MATRIX:-$PROJECT_DIR/gate_run/dockq_followup_sweep/matrix.json}"
# The repaired cohort should reach ~188,466 eligible models. The floor is deliberately
# below that so a small, explained shortfall does not block the pipeline, and deliberately
# far above the pre-repair 150,938 so a silently failed repair does.
export MIN_ELIGIBLE_MODELS="${MIN_ELIGIBLE_MODELS:-185000}"
export TUNING_DIR="${TUNING_DIR:-$PROJECT_DIR/gate_run/full_esm_tuning}"
export STAGE_TUNE="${STAGE_TUNE:-screen}"
TUNE_ARMS="${TUNE_ARMS:-}"
cd "$PROJECT_DIR"

for required in "$DATA_DIR" "$RSASA_DIR" "$ESM_ROOT"; do
 [[ -d "$required" ]] || { printf 'Missing required directory: %s\n' "$required" >&2; exit 2; }
done
[[ -f "$PRIOR_SPLIT" ]] || { printf 'Missing target split: %s\n' "$PRIOR_SPLIT" >&2; exit 2; }
[[ -f "$SOURCE_MATRIX" ]] || { printf 'Missing source matrix: %s\n' "$SOURCE_MATRIX" >&2; exit 2; }
for fresh in "$EXPERIMENT_DIR" "$TUNING_DIR"; do
 [[ ! -e "$fresh" ]] || { printf 'Choose a fresh directory; %s exists.\n' "$fresh" >&2; exit 2; }
done

# The repair writes into DATA_DIR. Every cached audit keyed to the old file signatures
# becomes stale, by design, and completed experiments can no longer be rebuilt from those
# graphs unaltered. Require that to be acknowledged rather than discovered.
if [[ "${CONFIRM_SOURCE_MUTATION:-}" != "yes" ]]; then
 printf 'This pipeline writes interface_node_degree into the graphs in:\n  %s\n' "$DATA_DIR" >&2
 printf 'Existing audits under gate_run/full_pooling_audit become stale and the completed\n' >&2
 printf 'pooling and ESM2 experiments can no longer be rebuilt from unaltered sources.\n' >&2
 printf 'Back up or snapshot DATA_DIR first, then re-run with CONFIRM_SOURCE_MUTATION=yes.\n' >&2
 printf 'To see what would change without writing anything: REPAIR_DRY_RUN=1.\n' >&2
 exit 2
fi

module load miniconda
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate py311_env
source "$PROJECT_DIR/venv/bin/activate"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

mkdir -p "$EXPERIMENT_DIR/logs" "$AUDIT_DIR" "$REPAIR_REPORTS" "$ESM_SIDECARS"
python full_pooling_validation.py inventory --data "$DATA_DIR" --output "$EXPERIMENT_DIR/paths.json"
export TARGET_COUNT
TARGET_COUNT=$(python -c 'import json,os; print(len(json.load(open(os.environ["EXPERIMENT_DIR"]+"/paths.json"))))')
printf 'Inventory: %s targets.\n' "$TARGET_COUNT"

L="$EXPERIMENT_DIR/logs"
REPAIR=$(sbatch --parsable --array="1-$TARGET_COUNT" --export=ALL,STAGE=repair \
 --output="$L/repair_%A_%a.out" cluster/repair_cohort.slurm); REPAIR=${REPAIR%%;*}
SUMMARY=$(sbatch --parsable --dependency="afterok:$REPAIR" --export=ALL,STAGE=repair_summary \
 --output="$L/repair_summary_%j.out" cluster/repair_cohort.slurm); SUMMARY=${SUMMARY%%;*}
AUDIT=$(sbatch --parsable --dependency="afterok:$SUMMARY" --array="1-$TARGET_COUNT" --export=ALL,STAGE=audit \
 --output="$L/audit_%A_%a.out" cluster/repair_cohort.slurm); AUDIT=${AUDIT%%;*}
BUILD=$(sbatch --parsable --dependency="afterok:$AUDIT" --export=ALL,STAGE=build \
 --output="$L/build_%j.out" cluster/repair_cohort.slurm); BUILD=${BUILD%%;*}
export FULL_PARENT_DIR="$EXPERIMENT_DIR"
PREPARE=$(sbatch --parsable --dependency="afterok:$BUILD" --array="1-$TARGET_COUNT" --export=ALL,STAGE=prepare \
 --output="$L/prepare_%A_%a.out" cluster/full_esm_preflight.slurm); PREPARE=${PREPARE%%;*}
TUNE=$(sbatch --parsable --dependency="afterok:$PREPARE" --export=ALL \
 --output="$L/tune_launcher_%j.out" --cpus-per-task=1 --mem=8G --time=01:00:00 \
 --wrap="cd $PROJECT_DIR && source \$(conda info --base)/etc/profile.d/conda.sh && conda activate py311_env && source $PROJECT_DIR/venv/bin/activate && FULL_PARENT_DIR=$EXPERIMENT_DIR EXPERIMENT_DIR=$TUNING_DIR ESM_SIDECARS=$ESM_SIDECARS SPLIT_MANIFEST=$EXPERIMENT_DIR/target_splits.json DATA_DIR=$DATA_DIR RSASA_DIR=$RSASA_DIR STAGE=$STAGE_TUNE ARMS='$TUNE_ARMS' bash cluster/submit_full_esm_tuning.sh")
TUNE=${TUNE%%;*}

cat <<EOF
Repair array      : $REPAIR   (writes interface_node_degree; $TARGET_COUNT tasks)
Repair summary    : $SUMMARY  (gate: aborts the pipeline if any target failed)
Re-audit array    : $AUDIT    (recomputes degree from contacts and checks agreement)
Cohort build      : $BUILD    (split from $PRIOR_SPLIT, counts recomputed, min $MIN_ELIGIBLE_MODELS)
ESM sidecars      : $PREPARE  (rebuilt for the expanded cohort)
Tuning submission : $TUNE     (submits the $STAGE_TUNE sweep into $TUNING_DIR)

Repaired cohort   : $EXPERIMENT_DIR
Fresh audit       : $AUDIT_DIR
Repair reports    : $REPAIR_REPORTS
Expected eligible : ~188,466 of 189,710 raw (1,243 models lack node_features entirely and
                    one lacks rSASA; neither is repairable by this pipeline).
EOF
