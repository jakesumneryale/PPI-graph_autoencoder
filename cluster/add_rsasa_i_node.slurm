#!/usr/bin/env bash
#SBATCH --job-name=rsasa_i_node
#SBATCH --partition=pi_co54
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder/voronoi_run/logs/%x_%j.out
#SBATCH --error=/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder/voronoi_run/logs/%x_%j.err

# One-time ingestion of per-residue rSASA_i (column rSASA_i_complex) into every
# target's full graph HDF5, as node feature "rsasa_i_node". Single serial job
# over the whole dataset, matching add_interface_node_degree_full.slurm's
# pattern (1 CPU -- this script is a single-threaded CSV-parse + h5py-write
# loop with no internal parallelism to justify more, per AGENTS.md).
#
# Run this BEFORE re-building the 10% subset (build_10pct_subset.slurm): the
# subset is a verbatim copy of whatever's already committed in the full
# dataset, so the new feature only reaches the subset if it's committed here
# first and the subset is rebuilt afterward.
#
# rsasa_i_node is distinct from the existing optional "rsasa_i" (a per-decoy
# graph-level average, loaded from *_avg_rSASA_i.csv). This script commits a
# genuine per-node HDF5 feature and does not touch that older mechanism.
#
# Do not run this at the same time as add_edge_ves.slurm against the same
# DATA_DIR: both open the same per-target HDF5 files r+, and two processes
# opening the same file at once raises BlockingIOError (HDF5's own file
# locking), even though they write unrelated datasets. The script retries
# transient lock contention with backoff, but submit these sequentially
# (e.g. --dependency=afterany) rather than relying on the retry to paper over
# a full-run collision.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder}"
DATA_DIR="${DATA_DIR:-/home/jas485/project_pi_co54/jas485/ppi_processed_graphs}"
REPORT="${REPORT:-$PROJECT_DIR/gate_run/rsasa_i_node_summary.csv}"

module load miniconda
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate py311_env
source "$PROJECT_DIR/venv/bin/activate"
cd "$PROJECT_DIR"

python add_rsasa_i_node.py \
  "$DATA_DIR" \
  --cluster \
  --report "$REPORT"
