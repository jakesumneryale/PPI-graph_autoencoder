#!/usr/bin/env bash
# ESM2 tuning sweep on the full cohort. Builds the matrix locally (JSON only, no GPU),
# then submits one GPU array task per run. Reuses the parent cohort, model lists and ESM
# sidecars; nothing is re-audited and no source data is touched.
set -euo pipefail
export PROJECT_DIR="${PROJECT_DIR:-/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder}"
export FULL_PARENT_DIR="${FULL_PARENT_DIR:-$PROJECT_DIR/gate_run/full_pooling_validation}"
export EXPERIMENT_DIR="${EXPERIMENT_DIR:-$PROJECT_DIR/gate_run/full_esm_tuning}"
export ESM_SIDECARS="${ESM_SIDECARS:-$PROJECT_DIR/full_pooling_esm_sidecars}"
export DATA_DIR="${DATA_DIR:-/nfs/roberts/project/pi_co54/jas485/ppi_processed_graphs}"
export RSASA_DIR="${RSASA_DIR:-/nfs/roberts/project/pi_co54/jas485/rsasa_i_graph_data}"
# Point SPLIT_MANIFEST at the stratified-difficulty manifest once it exists. It must cover
# the same targets that have ESM sidecars; the builder checks this and refuses otherwise.
export SPLIT_MANIFEST="${SPLIT_MANIFEST:-$FULL_PARENT_DIR/target_splits.json}"
export STAGE="${STAGE:-screen}"
ARMS="${ARMS:-}"
cd "$PROJECT_DIR"

for required in "$FULL_PARENT_DIR/matrix.json" "$SPLIT_MANIFEST"; do
 [[ -f "$required" ]] || { printf 'Missing required input: %s\n' "$required" >&2; exit 2; }
done
[[ -d "$ESM_SIDECARS" ]] || { printf 'Missing ESM sidecars: %s\n' "$ESM_SIDECARS" >&2
 echo 'Build them first with cluster/submit_full_esm_validation.sh, or set ESM_SIDECARS.' >&2; exit 2; }
[[ ! -e "$EXPERIMENT_DIR" ]] || { echo 'Choose a fresh EXPERIMENT_DIR; existing results are never overwritten.' >&2; exit 2; }
case "$STAGE" in screen|confirm) ;; *) echo 'STAGE must be screen or confirm.' >&2; exit 2;; esac
if [[ "$STAGE" == confirm && -z "$ARMS" ]]; then
 echo 'STAGE=confirm without ARMS would run every arm at three seeds.' >&2
 echo 'Set ARMS to the arms the screening stage selected, e.g. ARMS="baseline lambda0.1 esm_dim16".' >&2
 exit 2
fi

module load miniconda
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate py311_env
source "$PROJECT_DIR/venv/bin/activate"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

mkdir -p "$EXPERIMENT_DIR/logs"
# shellcheck disable=SC2086
python full_esm_tuning.py build \
 --parent "$FULL_PARENT_DIR" --output "$EXPERIMENT_DIR" --esm-sidecars "$ESM_SIDECARS" \
 --split-manifest "$SPLIT_MANIFEST" --data "$DATA_DIR" --rsasa "$RSASA_DIR" \
 --stage "$STAGE" ${ARMS:+--arms $ARMS}

COUNT=$(python -c 'import json,os; print(len(json.load(open(os.environ["EXPERIMENT_DIR"]+"/matrix.json"))["runs"]))')
TRAIN=$(sbatch --parsable --array="1-$COUNT" --export=ALL \
 --output="$EXPERIMENT_DIR/logs/train_%A_%a.out" cluster/train_model_extensions.slurm)
printf '%s GPU runs submitted: %s\nResults and logs: %s\n' "$COUNT" "$TRAIN" "$EXPERIMENT_DIR"
printf 'Each run logs per-epoch validation and test DockQ RMSE and writes test_predictions.csv\n'
printf 'from the checkpoint selected on validation DockQ MSE.\n'
printf 'Check completion with: python full_esm_tuning.py verify --output %s\n' "$EXPERIMENT_DIR"
