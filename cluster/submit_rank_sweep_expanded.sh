#!/usr/bin/env bash
# Ranking-loss sweep on the expanded dataset: the rank model (configs/gate_rank.toml) x seeds 7/17/27, 10 epochs
# (cosine over 10), on exactly the data, split and model lists of the dataset-size comparison's expanded arm.
#   control            rank weight 0.1, 4 targets x 4 decoys per batch (the current recipe at 10 epochs)
#   rank0.3            rank weight 0.3
#   rank1.0            rank weight 1.0
#   rank0.3_8decoys    rank weight 0.3, 2 targets x 8 decoys per batch (same batch size 16)
# 12 GPU runs. Choose the winner on VALIDATION (this fold's test set has been seen); confirm it on another fold.
# See RANK_SWEEP_EXPANDED.md.
#
# Run from the repository on a Bouchet login node (after the dataset-size comparison's train stage has staged
# gate_run/dataset_size_comparison/{data_expanded,staged,model_lists}):
#   DRY_RUN=1 bash cluster/submit_rank_sweep_expanded.sh
#   bash cluster/submit_rank_sweep_expanded.sh
#   RUNS="rank1.0" bash cluster/submit_rank_sweep_expanded.sh    # a subset; existing run directories are refused
set -euo pipefail
export PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}"
SRC="$PROJECT_DIR/gate_run/dataset_size_comparison"
OUT_ROOT="${OUT_ROOT:-$PROJECT_DIR/gate_run/rank_sweep_expanded}"
CONFIG=configs/gate_rank.toml
cd "$PROJECT_DIR"

DATA="$SRC/data_expanded"; MANIFEST="$SRC/staged/target_splits_expanded.json"; LISTS="$SRC/model_lists"
[[ -d $DATA && -f $MANIFEST && -d $LISTS ]] || {
  echo "Missing $DATA, $MANIFEST or $LISTS; run the dataset-size comparison's train stage first." >&2; exit 2; }
n_data=$(find "$DATA" -maxdepth 1 -name '*.hdf5' | wc -l)
broken=$(find "$DATA" -maxdepth 1 -name '*.hdf5' -xtype l | wc -l)
(( n_data == 350 && broken == 0 )) || { echo "$DATA: $n_data graph links ($broken broken); expected 350." >&2; exit 2; }

module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate py311_env
source "$PROJECT_DIR/venv/bin/activate"
python train_gate.py --config "$CONFIG" --print-config > /dev/null

# Same validation interval as the expanded arm (~4 validations per epoch, 25.7k steps/epoch). 10 epochs took
# ~33 h in the expanded arm; 3 days leaves room for validation and the final test evaluation.
COMMON=(--data "$DATA" --model-list-dir "$LISTS" --split-manifest "$MANIFEST" --epochs 10 --val-every-steps 6000)
declare -A RUN_ARGS=(
  [control]="--rank-loss-weight 0.1 --targets-per-batch 4"
  [rank0.3]="--rank-loss-weight 0.3 --targets-per-batch 4"
  [rank1.0]="--rank-loss-weight 1.0 --targets-per-batch 4"
  [rank0.3_8decoys]="--rank-loss-weight 0.3 --targets-per-batch 2"
)
for run in ${RUNS:-control rank0.3 rank1.0 rank0.3_8decoys}; do
  [[ -n ${RUN_ARGS[$run]:-} ]] || { echo "Unknown run: $run" >&2; exit 2; }
  out="$OUT_ROOT/$run"
  [[ -e $out ]] && { echo "Refusing to reuse $out; move it away first." >&2; exit 2; }
  # shellcheck disable=SC2206
  targs=("$CONFIG" "$out" "${COMMON[@]}" ${RUN_ARGS[$run]})
  python train_gate.py --config "$CONFIG" "${targs[@]:2}" --output-dir /tmp/unused --print-config > /dev/null
  args=(--parsable --export=ALL --array=1-3 --time=3-00:00:00 --job-name="gate_ranksweep_$run"
        --output="$out/logs/train_%A_%a.out" --error="$out/logs/train_%A_%a.err")
  if [[ -n ${DRY_RUN:-} ]]; then echo "sbatch ${args[*]} cluster/train_gate_config.slurm ${targs[*]}"; continue; fi
  mkdir -p "$out/logs"
  echo "$run: $(sbatch "${args[@]}" cluster/train_gate_config.slurm "${targs[@]}") -> $out/gate_rank_seed{7,17,27}"
done
