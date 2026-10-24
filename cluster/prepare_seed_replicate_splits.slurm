#!/usr/bin/env bash
#SBATCH --job-name=gate_prepare_splits
#SBATCH --partition=pi_co54
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10G
#SBATCH --time=04:00:00
#SBATCH --output=/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder/voronoi_run/logs/%x_%j.out
#SBATCH --error=/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder/voronoi_run/logs/%x_%j.err

# Verifies the committed edge features, then writes ONE train/val/test target
# split that all six runs reuse.
#
# This has to happen before the array starts. --seed drives both model
# initialisation and the split, so three replicates left to create their own
# manifests would each get a different split and the comparison would measure
# split variance rather than seed variance.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder}"
DATA_DIR="${DATA_DIR:-${GRAPH_DATA_DIR:-/nfs/roberts/project/pi_co54/jas485/ppi_processed_graphs}}"
MODEL_LIST_DIR="${MODEL_LIST_DIR:-$PROJECT_DIR/voronoi_dataset_audit/successful_models}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/gate_run/seed_replicates}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-$OUTPUT_ROOT/shared_target_splits.json}"

# Stage 3 feature set: the split only depends on which targets have data, but
# using the real feature list makes this fail here rather than inside the array
# if voronoi_contact_missing has not been committed yet.
NODE_FEATURES="${NODE_FEATURES:-aa_type,chain,interface_nodes,rsasa_i,interface_node_degree}"
EDGE_FEATURES="${EDGE_FEATURES:-interface_edges,ca_dist,voronoi_contact_area,voronoi_contact_missing}"

module load miniconda
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate py311_env
source "$PROJECT_DIR/venv/bin/activate"
cd "$PROJECT_DIR"

mkdir -p "$OUTPUT_ROOT"


echo "=== Preflight: selected training models must carry the required edge features ==="
python verify_edge_features.py \
  --data "$DATA_DIR" \
  --edge-features "$EDGE_FEATURES" \
  --model-list-dir "$MODEL_LIST_DIR"

if [[ -f "$SPLIT_MANIFEST" ]]; then
  echo "Split manifest already exists, reusing: $SPLIT_MANIFEST"
  exit 0
fi

echo "Writing shared split manifest: $SPLIT_MANIFEST"
python train_gate.py \
  --cluster \
  --data "$DATA_DIR" \
  --model-list-dir "$MODEL_LIST_DIR" \
  --device cpu \
  --num-workers 4 \
  --node-features "$NODE_FEATURES" \
  --edge-features "$EDGE_FEATURES" \
  --seed 7 \
  --split-manifest "$SPLIT_MANIFEST" \
  --output-dir "$OUTPUT_ROOT" \
  --prepare-splits-only
