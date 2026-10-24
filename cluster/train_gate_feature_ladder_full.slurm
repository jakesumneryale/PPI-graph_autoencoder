#!/usr/bin/env bash
#SBATCH --job-name=gate_feature_ladder_full
#SBATCH --partition=pi_co54
#SBATCH --array=1-15
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10G
#SBATCH --time=6-23:00:00
#SBATCH --output=/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder/voronoi_run/logs/%x_%A_%a.out
#SBATCH --error=/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder/voronoi_run/logs/%x_%A_%a.err

# Full-dataset counterpart of cluster/train_gate_feature_ladder_10pct.slurm: the
# same 5 cumulative rungs (core -> +interface degree -> +rSASA_i_node ->
# +Voronoi contact area -> +electrostatics), 3 seeds/rung, but trained on the
# full repaired cohort instead of the 10% subset.
#
# Trains against the full_pooling_repaired cohort (~188,466 eligible models:
# interface_node_degree-repaired, Voronoi-audited -- see
# cluster/submit_repair_and_retune.sh), NOT the plain
# voronoi_dataset_audit/successful_models pool cluster/train_gate_seed_replicates.slurm
# uses. Reuses that cohort's EXISTING target_splits.json and model_lists rather
# than deriving a new split: target-level split assignment doesn't depend on
# which features a given rung requests (only which decoys within a target end
# up eligible does, and that's re-resolved per rung from model_lists + whatever
# feature datasets actually exist), so reusing it keeps this run's target
# assignment identical to, and comparable with, other work against this cohort.
#
# Requires add_rsasa_i_node.py and add_edge_ves.py already committed against
# DATA_DIR (see cluster/add_rsasa_i_node.slurm, cluster/add_edge_ves.slurm) --
# this reads the full graph directory directly, not a subset copy, so those
# commits are the only thing that puts rsasa_i_node/v_es where this script can
# see them. A decoy missing either dataset is silently excluded from whichever
# rung requests it (ProteinGraphHDF5Dataset's normal behavior), not fatal.
#
# Hyperparameters (epochs, warmup-steps, walltime) match the full-dataset
# precedent in cluster/train_gate_seed_replicates.slurm, not the 10%-scaled
# values in train_gate_feature_ladder_10pct.slurm. The walltime budget is
# directly validated on this exact cohort: the model_extensions/full_pooling_repaired
# run reached epoch 42/50 comfortably inside the same 6-23:00:00 window before
# failing on an unrelated NFS file-locking issue (see HDF5_USE_FILE_LOCKING
# below), so 50 epochs is known to fit.
#
# Before submitting the real 15-task array, preflight that the split manifest
# still loads cleanly against the STRICTEST (electrostatics) feature superset --
# a fast, CPU-only, --prepare-splits-only dry run is much cheaper than
# discovering a target/model-list mismatch after burning real GPU-days:
#   python train_gate.py --cluster --data "$DATA_DIR" \
#     --model-list-dir "$MODEL_LIST_DIR" --split-manifest "$SPLIT_MANIFEST" \
#     --node-features "aa_type,chain,interface_nodes,interface_node_degree,rsasa_i_node" \
#     --edge-features "interface_edges,ca_dist,voronoi_contact_area,voronoi_contact_missing,v_es" \
#     --device cpu --num-workers 4 --output-dir /tmp/ladder_full_preflight \
#     --prepare-splits-only

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder}"
DATA_DIR="${DATA_DIR:-/nfs/roberts/project/pi_co54/jas485/ppi_processed_graphs}"
MODEL_LIST_DIR="${MODEL_LIST_DIR:-$PROJECT_DIR/gate_run/full_pooling_repaired/model_lists}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-$PROJECT_DIR/gate_run/full_pooling_repaired/target_splits.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/gate_run/feature_ladder_full}"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "This script must be submitted as a SLURM array job." >&2
  exit 2
fi
for required in "$DATA_DIR" "$MODEL_LIST_DIR"; do
  if [[ ! -d "$required" ]]; then
    echo "Required directory not found: $required" >&2
    exit 2
  fi
done
if [[ ! -f "$SPLIT_MANIFEST" ]]; then
  echo "Split manifest not found: $SPLIT_MANIFEST -- run cluster/submit_repair_and_retune.sh's build stage first." >&2
  exit 2
fi

RUNGS=(core degree rsasa voronoi electrostatics)
SEEDS=(7 17 27)
RUNG_INDEX=$(( (SLURM_ARRAY_TASK_ID - 1) / 3 ))
SEED_INDEX=$(( (SLURM_ARRAY_TASK_ID - 1) % 3 ))
RUNG="${RUNGS[$RUNG_INDEX]}"
SEED="${SEEDS[$SEED_INDEX]}"

case "$RUNG" in
  core)
    NODE_FEATURES="aa_type,chain,interface_nodes"
    EDGE_FEATURES="interface_edges,ca_dist"
    EDGE_RECON_FEATURES="interface_edges,ca_dist"
    EXTRA_ARGS=()
    ;;
  degree)
    NODE_FEATURES="aa_type,chain,interface_nodes,interface_node_degree"
    EDGE_FEATURES="interface_edges,ca_dist"
    EDGE_RECON_FEATURES="interface_edges,ca_dist"
    EXTRA_ARGS=()
    ;;
  rsasa)
    NODE_FEATURES="aa_type,chain,interface_nodes,interface_node_degree,rsasa_i_node"
    EDGE_FEATURES="interface_edges,ca_dist"
    EDGE_RECON_FEATURES="interface_edges,ca_dist"
    EXTRA_ARGS=()
    ;;
  voronoi)
    NODE_FEATURES="aa_type,chain,interface_nodes,interface_node_degree,rsasa_i_node"
    EDGE_FEATURES="interface_edges,ca_dist,voronoi_contact_area,voronoi_contact_missing"
    EDGE_RECON_FEATURES="interface_edges,ca_dist"
    EXTRA_ARGS=(--edge-feature-transforms "voronoi_contact_area=log1p" --standardize-edge-features "voronoi_contact_area")
    ;;
  electrostatics)
    NODE_FEATURES="aa_type,chain,interface_nodes,interface_node_degree,rsasa_i_node"
    EDGE_FEATURES="interface_edges,ca_dist,voronoi_contact_area,voronoi_contact_missing,v_es"
    EDGE_RECON_FEATURES="interface_edges,ca_dist"
    # v_es is signed (unlike voronoi_contact_area), so it is z-scored but not log1p'd --
    # apply_feature_transform's log1p rejects negative values.
    EXTRA_ARGS=(--edge-feature-transforms "voronoi_contact_area=log1p" --standardize-edge-features "voronoi_contact_area,v_es")
    ;;
  *)
    echo "Unknown rung: $RUNG" >&2; exit 2 ;;
esac

RUN_LABEL="gat4_residual_${RUNG}_seed${SEED}"
OUTPUT_DIR="$OUTPUT_ROOT/$RUN_LABEL"

module load miniconda
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate py311_env
source "$PROJECT_DIR/venv/bin/activate"
cd "$PROJECT_DIR"

# 15 array tasks x 8 DataLoader workers = up to 120 concurrent read-only HDF5
# opens against the same NFS-mounted DATA_DIR over multi-day walltime. HDF5's
# default file locking applies even to read-only opens and can spuriously fail
# under this much sustained concurrency (BlockingIOError, errno 11) -- this is
# exactly what crashed the model_extensions/full_pooling_repaired run at epoch
# 42. Safe to disable here because every DataLoader access to these files is
# read-only. Do NOT do this for a process that writes (add_rsasa_i_node.py,
# add_edge_ves.py), where locking is what turns a real concurrent-write
# collision into a clean failure instead of a corrupted file.
export HDF5_USE_FILE_LOCKING=FALSE

mkdir -p "$OUTPUT_DIR"

echo "Array task: $SLURM_ARRAY_TASK_ID  (rung=$RUNG seed=$SEED)"
echo "Data dir: $DATA_DIR"
echo "Model list dir: $MODEL_LIST_DIR"
echo "Split manifest: $SPLIT_MANIFEST"
echo "Node features: $NODE_FEATURES"
echo "Edge features: $EDGE_FEATURES"
echo "Output: $OUTPUT_DIR"
nvidia-smi

python train_gate.py \
  --cluster \
  --data "$DATA_DIR" \
  --model-list-dir "$MODEL_LIST_DIR" \
  --device cuda \
  --epochs 50 \
  --batch-size 16 \
  --lr 3e-4 \
  --lr-schedule cosine \
  --warmup-steps 1000 \
  --num-workers 8 \
  --worker-start-method spawn \
  --persistent-workers \
  --node-features "$NODE_FEATURES" \
  --edge-features "$EDGE_FEATURES" \
  --edge-recon-features "$EDGE_RECON_FEATURES" \
  "${EXTRA_ARGS[@]}" \
  --edge-stats-seed 7 \
  --loss-weight-mode shares \
  --node-share 0.15 \
  --edge-attr-share 0.15 \
  --edge-presence-share 0.10 \
  --target-share 0.60 \
  --checkpoint-metric target_mse \
  --seed "$SEED" \
  --split-manifest "$SPLIT_MANIFEST" \
  --output-dir "$OUTPUT_DIR" \
  --residual_connections
