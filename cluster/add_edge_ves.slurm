#!/usr/bin/env bash
#SBATCH --job-name=edge_v_es
#SBATCH --partition=pi_co54
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder/voronoi_run/logs/%x_%j.out
#SBATCH --error=/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder/voronoi_run/logs/%x_%j.err

# One-time ingestion of per-edge electrostatic potential energy (Edge_Ves)
# into every target's full graph HDF5, as edge feature "v_es". Single serial
# job over the whole dataset, matching add_interface_node_degree_full.slurm's
# pattern (1 CPU -- this script is a single-threaded CSV-parse + h5py-write
# loop with no internal parallelism to justify more, per AGENTS.md).
#
# Run this BEFORE re-building the 10% subset (build_10pct_subset.slurm): the
# subset is a verbatim copy of whatever's already committed in the full
# dataset, so the new feature only reaches the subset if it's committed here
# first and the subset is rebuilt afterward.
#
# v_es is a screened, decoy-scale electrostatic term computed from PQR partial
# charges (nb_scripts/create_graph_edge_ves.py). It is NOT the APBS-derived
# V_ae named as a still-blocked, planned rung elsewhere in this project --
# don't conflate the two when writing up results.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder}"
DATA_DIR="${DATA_DIR:-/home/jas485/project_pi_co54/jas485/ppi_processed_graphs}"
REPORT="${REPORT:-$PROJECT_DIR/gate_run/v_es_summary.csv}"

module load miniconda
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate py311_env
source "$PROJECT_DIR/venv/bin/activate"
cd "$PROJECT_DIR"

python add_edge_ves.py \
  "$DATA_DIR" \
  --cluster \
  --report "$REPORT"
