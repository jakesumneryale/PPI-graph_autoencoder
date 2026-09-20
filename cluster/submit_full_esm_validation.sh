#!/usr/bin/env bash
set -euo pipefail
export PROJECT_DIR="${PROJECT_DIR:-/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder}"
export FULL_PARENT_DIR="${FULL_PARENT_DIR:-$PROJECT_DIR/gate_run/full_pooling_validation}"
export EXPERIMENT_DIR="${EXPERIMENT_DIR:-$PROJECT_DIR/gate_run/full_pooling_esm_validation}"
export ESM_ROOT="${ESM_ROOT:-/nfs/roberts/pi/pi_co54/nb685/scratch_backup/SS_embeds}"
export ESM_SIDECARS="${ESM_SIDECARS:-$PROJECT_DIR/full_pooling_esm_sidecars}"
cd "$PROJECT_DIR"
[[ -f "$FULL_PARENT_DIR/matrix.json" && -f "$FULL_PARENT_DIR/paths.json" ]] || {
 echo 'Wait for the full_pooling_validation matrix/build job to finish; its GPU jobs need not finish.' >&2; exit 2;
}
[[ ! -e "$EXPERIMENT_DIR" ]] || { echo 'Choose a fresh EXPERIMENT_DIR.' >&2; exit 2; }
module load miniconda
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate py311_env
source "$PROJECT_DIR/venv/bin/activate"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
COUNT=$(python -c 'import json,os; print(len(json.load(open(os.environ["FULL_PARENT_DIR"]+"/paths.json"))))')
mkdir -p "$EXPERIMENT_DIR/logs" "$ESM_SIDECARS"
PREP=$(sbatch --parsable --array="1-$COUNT" --export=ALL,STAGE=prepare --output="$EXPERIMENT_DIR/logs/prepare_%A_%a.out" cluster/full_esm_preflight.slurm)
PREP_ID=${PREP%%;*}
BUILD=$(sbatch --parsable --dependency="afterok:$PREP_ID" --export=ALL,STAGE=build --output="$EXPERIMENT_DIR/logs/build_%j.out" cluster/full_esm_preflight.slurm)
BUILD_ID=${BUILD%%;*}
TRAIN=$(sbatch --parsable --array=1-6 --dependency="afterok:$BUILD_ID" --export=ALL --output="$EXPERIMENT_DIR/logs/train_%A_%a.out" cluster/train_model_extensions.slurm)
printf 'ESM preparation: %s\nMatrix validation: %s\nSix ESM GPU runs: %s\nResults and logs: %s\n' "$PREP" "$BUILD" "$TRAIN" "$EXPERIMENT_DIR"
