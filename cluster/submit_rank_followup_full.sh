#!/usr/bin/env bash
# Submit the 12-run ranking-loss follow-up (4 configurations x 3 seeds, 20 epochs,
# one test-set evaluation per run). Run from the cluster login node after syncing:
#   bash cluster/submit_rank_followup_full.sh
# Set OUTPUT_ROOT to use a fresh experiment directory; existing ones are refused.
set -euo pipefail
export PROJECT_DIR="${PROJECT_DIR:-/gpfs/boyd/pi/pi_co54/jas485/PPI-graph_autoencoder}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/gate_run/rank_followup_full}"
cd "$PROJECT_DIR"

if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "Experiment directory already exists: $OUTPUT_ROOT -- choose a fresh OUTPUT_ROOT." >&2
  exit 2
fi

module load miniconda
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate py311_env
source "$PROJECT_DIR/venv/bin/activate"

# Cheap login-node checks that the synced code has the ranking loss and the
# end-of-training test export (which writes a "test" block to prediction_metrics.json).
python train_gate.py --help | grep -q -- "--rank-loss-weight" || {
  echo "train_gate.py on the cluster lacks --rank-loss-weight; sync the updated code first." >&2; exit 2; }
grep -q 'report\["test"\]' train_gate.py || {
  echo "train_gate.py on the cluster predates the test-set metrics export; sync the updated code first." >&2; exit 2; }

mkdir -p "$OUTPUT_ROOT/logs"
JOB=$(sbatch --parsable --export=ALL \
  --output="$OUTPUT_ROOT/logs/train_%A_%a.out" --error="$OUTPUT_ROOT/logs/train_%A_%a.err" \
  cluster/train_gate_rank_followup_full.slurm)
printf 'Submitted 12 GPU runs: %s\nResults: %s\nLogs: %s/logs\n' "$JOB" "$OUTPUT_ROOT" "$OUTPUT_ROOT"
