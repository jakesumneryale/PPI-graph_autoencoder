#!/usr/bin/env bash
#SBATCH --job-name=gate_build_10pct
#SBATCH --partition=pi_co54
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10G
#SBATCH --time=06:00:00
#SBATCH --output=/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder/voronoi_run/logs/%x_%j.out
#SBATCH --error=/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder/voronoi_run/logs/%x_%j.err

# Builds one shared 10% stratified subset of the full graph dataset, used by
# every ablation job in the triaged sweep (cluster/submit_ablation_sweep.sh).
# Every ablation reads this SAME subset, so results are comparable to each
# other; run once, reuse everywhere.
#
# Subsampling is per-target and DockQ-bin-stratified (make_subset_dataset.py),
# not a naive first-N truncation: every target keeps ~10% of its models, and
# within each target the DockQ-bin balance of the full set is preserved. The
# target-level split itself is untouched -- all 146 targets remain, just with
# fewer models each -- so the existing shared_target_splits.json / 105-12-29
# split (Sec. 4.3.1) still applies to the subset directly.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder}"
DATA_DIR="${DATA_DIR:-${GRAPH_DATA_DIR:-/nfs/roberts/project/pi_co54/jas485/ppi_processed_graphs}}"
TARGETS_FILE="${TARGETS_FILE:-$PROJECT_DIR/cluster/targets.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/gate_run/subset_10pct_hdf5}"
FRACTION="${FRACTION:-0.10}"
SEED="${SEED:-20250909}"

module load miniconda
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate py311_env
source "$PROJECT_DIR/venv/bin/activate"
cd "$PROJECT_DIR"

echo "Data dir:   $DATA_DIR"
echo "Targets:    $TARGETS_FILE"
echo "Output dir: $OUTPUT_DIR"
echo "Fraction:   $FRACTION   Seed: $SEED"

python make_subset_dataset.py \
  --data-dir "$DATA_DIR" \
  --targets-file "$TARGETS_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --fraction "$FRACTION" \
  --seed "$SEED" \
  --allow-missing-targets
