#!/usr/bin/env bash
# Submit the 27-run cause ablation (9 configurations x 3 seeds, full data).
# Run from the cluster login node after syncing the code:
#   bash cluster/submit_cause_ablation_full.sh
# Set OUTPUT_ROOT to use a fresh experiment directory; existing ones are refused.
set -euo pipefail
export PROJECT_DIR="${PROJECT_DIR:-/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/gate_run/cause_ablation_full}"
cd "$PROJECT_DIR"

if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "Experiment directory already exists: $OUTPUT_ROOT -- choose a fresh OUTPUT_ROOT." >&2
  exit 2
fi
for run in A_control_seed7 A_control_seed17 A_control_seed27; do
  [[ -f "gate_run/anti_memorization_full/$run/validation_steps.csv" ]] || {
    echo "Reference control run missing: gate_run/anti_memorization_full/$run" >&2; exit 2; }
done

module load miniconda
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate py311_env
source "$PROJECT_DIR/venv/bin/activate"

# Cheap login-node check that the synced trainer has the new option; no data is read.
python train_gate.py --help | grep -q -- "--lr-schedule-epochs" || {
  echo "train_gate.py on the cluster lacks --lr-schedule-epochs; sync the updated code first." >&2; exit 2; }

mkdir -p "$OUTPUT_ROOT/logs"
JOB=$(sbatch --parsable --export=ALL \
  --output="$OUTPUT_ROOT/logs/train_%A_%a.out" --error="$OUTPUT_ROOT/logs/train_%A_%a.err" \
  cluster/train_gate_cause_ablation_full.slurm)
printf 'Submitted 27 GPU runs: %s\nResults: %s\nLogs: %s/logs\n' "$JOB" "$OUTPUT_ROOT" "$OUTPUT_ROOT"
