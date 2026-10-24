#!/usr/bin/env bash
#SBATCH --job-name=gate_config
#SBATCH --partition=pi_co54
#SBATCH --array=1-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%A_%a.out

# Train GATE from a TOML config, one array task per seed (task 1/2/3 -> seed 7/17/27).
#
#   sbatch cluster/train_gate_config.slurm configs/gate_rank.toml gate_run/my_run
#   sbatch cluster/train_gate_config.slurm configs/gate_rank.toml gate_run/rank_capri --capri-head-weight 0.2
#
# Extra arguments after the output root are passed to train_gate.py and override the config. Each run writes
# <output_root>/<config name>_seed<seed>/run_config.toml with every resolved option.
#
# CPU: main process (one PyTorch/BLAS thread) + (cpus-per-task - 1) spawn loader workers; --num-workers and
# --cpu-threads below override whatever the config says so they always match the allocation.

set -euo pipefail
CONFIG=${1:?usage: sbatch cluster/train_gate_config.slurm <config.toml> <output_root> [train_gate.py args]}
OUTPUT_ROOT=${2:?usage: sbatch cluster/train_gate_config.slurm <config.toml> <output_root> [train_gate.py args]}
shift 2
PROJECT_DIR="${PROJECT_DIR:-/gpfs/boyd/pi/pi_co54/jas485/PPI-graph_autoencoder}"
SEEDS=(7 17 27)
SEED="${SEEDS[$(( ${SLURM_ARRAY_TASK_ID:-1} - 1 ))]}"
LOADER_WORKERS=$(( ${SLURM_CPUS_PER_TASK:-8} - 1 ))
(( LOADER_WORKERS >= 1 )) || { echo "Need at least 2 CPUs per task." >&2; exit 2; }

cd "$PROJECT_DIR"
[[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG" >&2; exit 2; }
OUTPUT_DIR="$OUTPUT_ROOT/$(basename "$CONFIG" .toml)_seed${SEED}"
[[ -e "$OUTPUT_DIR/gate_model.pt" ]] && { echo "Refusing to overwrite existing run: $OUTPUT_DIR" >&2; exit 2; }

module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate py311_env
source "$PROJECT_DIR/venv/bin/activate"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE
mkdir -p "$OUTPUT_DIR"
nvidia-smi || true

python train_gate.py --config "$CONFIG" --seed "$SEED" --output-dir "$OUTPUT_DIR" \
  --num-workers "$LOADER_WORKERS" --cpu-threads 1 "$@"
