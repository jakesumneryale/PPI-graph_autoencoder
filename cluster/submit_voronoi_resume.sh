#!/usr/bin/env bash
# Finish the Voronoi contact areas the first array could not reach within its time limit, then
# run audit -> build -> the 12 full-dataset GATE runs.
# Usage: bash cluster/submit_voronoi_resume.sh [ORIGINAL_VORONOI_ARRAY_JOB_ID|none]
#   Give the original array ID if it is still running (the resume waits for it to end so two jobs
#   never write one checkpoint); use "none" or omit it if nothing is running.
#   TARGETS="1ggp 1itb ..."   targets to continue (default: the 19 uniformly sampled targets)
#   TIME_LIMIT=2-00:00:00     per-task limit (must not exceed the partition's MaxTime)
#   CHAIN=0                   submit only the resume array
set -euo pipefail
ORIGINAL_JOB="${1:-none}"
export PROJECT_DIR="${PROJECT_DIR:-/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder}"
TARGETS="${TARGETS:-1c3a 1ggp 1itb 1kfu 1pdk 1spp 1tfo 2d5r 2dvw 2hla 2hsm 2hth 2oul 2pe6 2wmp 3gfk 3k9p 3ona 3qc8}"
TIME_LIMIT="${TIME_LIMIT:-2-00:00:00}"
CHAIN="${CHAIN:-1}"
cd "$PROJECT_DIR"

INDICES=$(for t in $TARGETS; do
  line=$(grep -n -x "$t" cluster/targets.txt | cut -d: -f1)
  [[ -n "$line" ]] || { echo "Target $t is not in cluster/targets.txt" >&2; exit 2; }
  echo "$line"
done | paste -sd,)

WAIT=()
[[ "$ORIGINAL_JOB" == none ]] || WAIT=(--dependency="afterany:$ORIGINAL_JOB")
if squeue -u "$USER" -h -o "%j" | grep -q '^voronoi_contact_area$' && [[ "$ORIGINAL_JOB" == none ]]; then
  echo "A voronoi_contact_area job is still running; pass its array ID so the resume waits for it." >&2; exit 2
fi
RESUME=$(sbatch --parsable "${WAIT[@]}" --time="$TIME_LIMIT" --array="$INDICES" \
  cluster/dispatch_voronoi_resume.slurm); RESUME=${RESUME%%;*}
printf 'Voronoi resume: %s (waiting on: %s; array %s; limit %s)\n' "$RESUME" "$ORIGINAL_JOB" "$INDICES" "$TIME_LIMIT"
[[ "$CHAIN" == 1 ]] || exit 0

export EXPERIMENT_DIR="${EXPERIMENT_DIR:-$PROJECT_DIR/gate_run/full_pooling_repaired}"
export AUDIT_DIR="${AUDIT_DIR:-$PROJECT_DIR/gate_run/full_pooling_audit_repaired}"
export RSASA_DIR="${RSASA_DIR:-/nfs/roberts/project/pi_co54/jas485/rsasa_i_graph_data}"
export PRIOR_SPLIT="${PRIOR_SPLIT:-$PROJECT_DIR/target_splits.json}"
export SOURCE_MATRIX="${SOURCE_MATRIX:-$PROJECT_DIR/gate_run/dockq_followup_sweep/matrix.json}"
export MIN_ELIGIBLE_MODELS="${MIN_ELIGIBLE_MODELS:-165000}"
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate py311_env
source "$PROJECT_DIR/venv/bin/activate"
COUNT=$(python -c 'import json,os; print(len(json.load(open(os.environ["EXPERIMENT_DIR"]+"/paths.json"))))')
AUDIT=$(sbatch --parsable --dependency="afterany:$RESUME" --array="1-$COUNT" --export=ALL,STAGE=audit \
  --output="$EXPERIMENT_DIR/logs/audit_resume_%A_%a.out" cluster/repair_cohort.slurm); AUDIT=${AUDIT%%;*}
BUILD=$(sbatch --parsable --dependency="afterok:$AUDIT" --export=ALL,STAGE=build \
  --output="$EXPERIMENT_DIR/logs/build_resume_%j.out" cluster/repair_cohort.slurm); BUILD=${BUILD%%;*}
TRAIN=$(sbatch --parsable --array=1-12 --dependency="afterok:$BUILD" --export=ALL \
  --output="$EXPERIMENT_DIR/logs/train_%A_%a.out" cluster/train_model_extensions.slurm)
printf 'Audit: %s\nBuild: %s\n12 GATE runs: %s\n' "$AUDIT" "$BUILD" "$TRAIN"
