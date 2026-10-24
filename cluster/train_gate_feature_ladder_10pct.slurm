#!/usr/bin/env bash
#SBATCH --job-name=gate_feature_ladder_10pct
#SBATCH --partition=pi_co54
#SBATCH --array=1-15
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10G
#SBATCH --time=12:00:00
#SBATCH --output=/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder/voronoi_run/logs/%x_%A_%a.out
#SBATCH --error=/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder/voronoi_run/logs/%x_%A_%a.err

# PRIORITY 1 (highest -- the one that would actually go in a figure).
# Cumulative feature ladder: core -> +interface degree -> +rSASA_i ->
# +Voronoi contact area -> +electrostatics, >=3 seeds per rung, on the 10%
# subsampled dataset built by build_10pct_subset.slurm.
#
# The "rsasa" rung (and every rung after it) uses rsasa_i_node, a genuine
# per-residue rSASA_i (column rSASA_i_complex, committed by
# add_rsasa_i_node.py), not the older rsasa_i graph-level average. That
# average was a stopgap and is superseded here -- this rerun corrects it, it
# does not add a parallel rung.
#
# The ladder in dissertation.tex also names a fifth rung, "+electrostatics".
# It was previously blocked because the only committed electrostatics feature
# was V_ae, an APBS-derived quantity computed once per bound native structure
# (84 targets, Fig. 4.3/Table 4.1), never at decoy scale. The "electrostatics"
# rung here instead uses v_es (committed by add_edge_ves.py): a screened,
# per-decoy electrostatic edge feature computed from PQR partial charges
# (nb_scripts/create_graph_edge_ves.py), a DIFFERENT quantity from V_ae. It
# fills the same ladder slot conceptually but should be described as v_es, not
# as the previously-planned V_ae, in any write-up of this run.
#
# Both add_rsasa_i_node.py and add_edge_ves.py must be run against the full
# dataset (cluster/add_rsasa_i_node.slurm, cluster/add_edge_ves.slurm) and
# build_10pct_subset.slurm re-run afterward, before this script is submitted --
# the subset is a verbatim copy of whatever's already committed upstream.
#
# Same target-level split as every other run this chapter uses (105/12/29,
# seed 5): split_dataset_by_target only depends on the sorted target
# filenames and the seed, not the directory they live in, so re-deriving the
# split from the 10% subset directory (with --seed 5) reproduces the exact
# same train/val/test target assignment as the full-dataset runs. See
# gate_run/subset_10pct_hdf5/shared_target_splits.json after stage 1 runs.
#
# Uses the same loss-share rebalancing, cosine LR schedule, and
# target_mse-selected checkpoints already validated in
# cluster/train_gate_seed_replicates.slurm; only warmup-steps is scaled down
# to match the ~10x shorter epochs here.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/gate_run/subset_10pct_hdf5}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/gate_run/feature_ladder_10pct}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-$OUTPUT_ROOT/shared_target_splits.json}"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "This script must be submitted as a SLURM array job." >&2
  exit 2
fi
if [[ ! -d "$DATA_DIR" ]]; then
  echo "10% subset not found at $DATA_DIR -- run build_10pct_subset.slurm first." >&2
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

# 15 array tasks x --num-workers 8 = up to 120 DataLoader worker processes,
# each opening HDF5 files read-only from the same NFS-mounted subset
# directory. HDF5's default file locking applies even to read-only opens and
# can spuriously fail under this much concurrent access on NFS
# (BlockingIOError: "unable to lock file", errno 11 -- seen mid-training on
# the unrelated model_extensions/full_pooling_repaired run). Safe to disable
# here because every DataLoader access to these files is read-only --
# concurrent readers can never corrupt a file. Do NOT do this for a process
# that writes (e.g. add_rsasa_i_node.py/add_edge_ves.py), where locking is the
# thing that turns a real concurrent-write collision into a clean failure
# instead of a corrupted file.
export HDF5_USE_FILE_LOCKING=FALSE

mkdir -p "$OUTPUT_DIR"

# Task 1 writes the shared split; every other task waits for it rather than
# racing to create their own (all reading the same subset directory, so
# whichever task gets there first regenerates the identical manifest, but
# only one write should happen to avoid a torn file under concurrency).
if [[ ! -f "$SPLIT_MANIFEST" ]]; then
  if [[ "$SLURM_ARRAY_TASK_ID" == "1" ]]; then
    echo "Task 1: writing shared split manifest for the 10% subset."
    python train_gate.py \
      --cluster \
      --data "$DATA_DIR" \
      --device cpu --num-workers 4 \
      --node-features "aa_type,chain,interface_nodes,interface_node_degree,rsasa_i_node" \
      --edge-features "interface_edges,ca_dist,voronoi_contact_area,voronoi_contact_missing,v_es" \
      --seed 5 \
      --split-manifest "$SPLIT_MANIFEST" \
      --output-dir "$OUTPUT_ROOT" \
      --prepare-splits-only
  else
    echo "Waiting for task 1 to write the shared split manifest..."
    for _ in $(seq 1 60); do
      [[ -f "$SPLIT_MANIFEST" ]] && break
      sleep 10
    done
    if [[ ! -f "$SPLIT_MANIFEST" ]]; then
      echo "Split manifest never appeared at $SPLIT_MANIFEST" >&2
      exit 2
    fi
  fi
fi

echo "Array task: $SLURM_ARRAY_TASK_ID  (rung=$RUNG seed=$SEED)"
echo "Node features: $NODE_FEATURES"
echo "Edge features: $EDGE_FEATURES"
echo "Output: $OUTPUT_DIR"
nvidia-smi

python train_gate.py \
  --cluster \
  --data "$DATA_DIR" \
  --device cuda \
  --epochs 50 \
  --batch-size 16 \
  --lr 3e-4 \
  --lr-schedule cosine \
  --warmup-steps 200 \
  --num-workers 8 \
  --worker-start-method spawn \
  --persistent-workers \
  --node-features "$NODE_FEATURES" \
  --edge-features "$EDGE_FEATURES" \
  --edge-recon-features "$EDGE_RECON_FEATURES" \
  "${EXTRA_ARGS[@]}" \
  --edge-stats-seed 5 \
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
