#!/usr/bin/env bash
#SBATCH --job-name=gate_architecture_10pct
#SBATCH --partition=pi_co54
#SBATCH --array=1-6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10G
#SBATCH --time=12:00:00
#SBATCH --nice=250
#SBATCH --output=/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder/voronoi_run/logs/%x_%A_%a.out
#SBATCH --error=/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder/voronoi_run/logs/%x_%A_%a.err

# PRIORITY 4b. Architecture ablation: attention (GATConv, --architecture gat)
# versus a non-attention, E(n)-equivariant message-passing encoder
# (--architecture egnn, EGNN_model.py), 3 seeds each, on the same 10% subset
# and split as the rest of this sweep. "gat" here duplicates the feature-
# ladder's "voronoi" rung exactly (same features, same everything but the
# encoder), so only the egnn side is new work; gat's 3 seeds are re-run here
# too rather than reused, so the comparison is apples-to-apples under
# identical launch conditions (same job, same day, same node pool).
#
# This is NOT the "plain graph convolution" comparison the dissertation TODO
# literally names (that would be GCNConv, which is not wired up anywhere in
# this codebase); it is the closest already-implemented alternative to GAT's
# attention, and it happens to bear directly on Sec. 4.2.2's argument for
# skipping equivariant architectures -- this run is a direct empirical check
# of that argument, not just an ablation for its own sake. "With/without edge
# conditioning" (GATConv without edge_dim) is still not available without a
# small code change to GATE_model.py; not attempted here.
#
# EGNN requires committed per-model 3D coordinates (a top-level "pos" dataset,
# shape [N,3], in every model group -- see protein_hdf5_dataset.py). The
# preflight below checks a few models before committing the array to 12
# GPU-hours it cannot use.
#
# --nice=250: between leave-one-out (200) and recon-weight-zero (300).

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/gate_run/subset_10pct_hdf5}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/gate_run/architecture_ablation_10pct}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-$PROJECT_DIR/gate_run/feature_ladder_10pct/shared_target_splits.json}"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "This script must be submitted as a SLURM array job." >&2
  exit 2
fi
if [[ ! -d "$DATA_DIR" ]]; then
  echo "10% subset not found at $DATA_DIR -- run build_10pct_subset.slurm first." >&2
  exit 2
fi
if [[ ! -f "$SPLIT_MANIFEST" ]]; then
  echo "Shared split manifest not found at $SPLIT_MANIFEST." >&2
  echo "Run train_gate_feature_ladder_10pct.slurm (task 1) first, or set SPLIT_MANIFEST." >&2
  exit 2
fi

module load miniconda
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate py311_env
source "$PROJECT_DIR/venv/bin/activate"
cd "$PROJECT_DIR"

if [[ "$SLURM_ARRAY_TASK_ID" == "1" ]]; then
  echo "Preflight: checking that committed 'pos' coordinates exist for EGNN..."
  python - "$DATA_DIR" <<'PYEOF'
import sys, h5py, glob, itertools
data_dir = sys.argv[1]
files = sorted(glob.glob(f"{data_dir}/*.hdf5"))[:3]
if not files:
    sys.exit(f"No HDF5 files found in {data_dir}")
missing = []
for f in files:
    with h5py.File(f, "r") as h:
        groups = [k for k in h if not k.startswith("__")][:5]
        for g in groups:
            if "pos" not in h[g]:
                missing.append(f"{f}:{g}")
if missing:
    print(f"FAIL: 'pos' dataset missing on {len(missing)} checked model(s), e.g. {missing[:3]}", file=sys.stderr)
    print("EGNN cannot run on this data yet; see the architecture-ablation slurm script header.", file=sys.stderr)
    sys.exit(1)
print(f"OK: 'pos' present on all {sum(len([k for k in h5py.File(f)][:5]) for f in files)} checked models across {len(files)} files.")
PYEOF
fi

RUNGS=(gat egnn)
SEEDS=(7 17 27)
RUNG_INDEX=$(( (SLURM_ARRAY_TASK_ID - 1) / 3 ))
SEED_INDEX=$(( (SLURM_ARRAY_TASK_ID - 1) % 3 ))
ARCHITECTURE="${RUNGS[$RUNG_INDEX]}"
SEED="${SEEDS[$SEED_INDEX]}"

RUN_LABEL="${ARCHITECTURE}_seed${SEED}"
OUTPUT_DIR="$OUTPUT_ROOT/$RUN_LABEL"
mkdir -p "$OUTPUT_DIR"

echo "Array task: $SLURM_ARRAY_TASK_ID  (architecture=$ARCHITECTURE seed=$SEED)"
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
  --architecture "$ARCHITECTURE" \
  --node-features "aa_type,chain,interface_nodes,interface_node_degree,rsasa_i" \
  --edge-features "interface_edges,ca_dist,voronoi_contact_area,voronoi_contact_missing" \
  --edge-recon-features "interface_edges,ca_dist" \
  --edge-feature-transforms "voronoi_contact_area=log1p" \
  --standardize-edge-features "voronoi_contact_area" \
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
