#!/usr/bin/env bash
#SBATCH --job-name=gate_attention_analysis_10pct
#SBATCH --partition=pi_co54
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10G
#SBATCH --time=02:00:00
#SBATCH --nice=100
#SBATCH --output=/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder/voronoi_run/logs/%x_%j.out
#SBATCH --error=/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder/voronoi_run/logs/%x_%j.err

# PRIORITY 2 (second-most important -- attention interpretability). Runs
# gate_run/analyze_attention_weights.py on the richest completed checkpoint
# from the Priority-1 feature ladder (the "voronoi" rung, seed 7): attention
# weight on interface versus intra-chain edges, and correlation with Voronoi
# contact area. Burial (rSASA_i) is skipped for this specific checkpoint
# choice since the voronoi rung's feature list is
# aa_type,chain,interface_nodes,interface_node_degree,rsasa_i,voronoi -- it
# DOES have rsasa_i, so burial correlation runs too. |Delta phi_r|
# correlation is never available (see the script's own docstring).
#
# Submitted with --dependency=afterok on the feature-ladder array job by
# submit_ablation_sweep.sh; do not run standalone before that array's seed-7
# voronoi-rung task (array index 10) has actually produced a checkpoint.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/gate_run/subset_10pct_hdf5}"
CHECKPOINT="${CHECKPOINT:-$PROJECT_DIR/gate_run/feature_ladder_10pct/gat4_residual_voronoi_seed7/gate_model.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/gate_run/feature_ladder_10pct/attention_analysis}"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "No checkpoint at $CHECKPOINT -- the feature-ladder job may not have finished." >&2
  exit 2
fi

module load miniconda
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate py311_env
source "$PROJECT_DIR/venv/bin/activate"
cd "$PROJECT_DIR"

echo "Checkpoint: $CHECKPOINT"
echo "Output: $OUTPUT_DIR"

python gate_run/analyze_attention_weights.py \
  --checkpoint "$CHECKPOINT" \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --split test \
  --max-graphs 2000 \
  --device cuda
