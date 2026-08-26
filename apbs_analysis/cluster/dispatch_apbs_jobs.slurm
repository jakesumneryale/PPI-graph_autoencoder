#!/usr/bin/env bash
#SBATCH --job-name=apbs_surface_potential
#SBATCH --partition=pi_co54
#SBATCH --array=1-146%24
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@120
#SBATCH --output=/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder/apbs_run/logs/%x_%A_%a.out
#SBATCH --error=/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder/apbs_run/logs/%x_%A_%a.err
#
# 146 targets, one per array task, same layout as the Voronoi array.
#
# Sizing rationale. APBS memory is set by the fine grid, not the molecule:
# --memory-ceiling-mb (default 4000) caps the grid at roughly 4 GiB and
# coarsens it rather than exceed that. Measured locally on Apple silicon in
# exactly this configuration: 12 s wall for a 2,080-atom complex and 56 s /
# 3.45 GiB peak for a 14,022-atom one, both dominated by APBS itself. Two
# models run concurrently per task (APBS's solver is single-threaded), so two
# cores and 2 x 3.45 GiB of solver memory; 16 GiB leaves comfortable headroom.
# Far cheaper than the Voronoi array's 64 GiB because nothing here builds a
# mesh. At ~23 s per model for a median-sized target, 1,400 models takes about
# 5.5 h at two workers -- comfortably inside the 24 h window even derating for
# slower cluster cores. %24 keeps twenty-four targets running at once: 48 cores
# and 384 GiB, still under the Voronoi array's 512 GiB aggregate budget, and
# the whole 146-target sweep then lands in roughly two days rather than four.
# Drop it back to %12 if the partition is busy.
#
# Before submitting: run apbs_analysis/cluster/setup_env.sh once to build the
# conda environment (APBS has no PyPI wheel, so this cannot reuse the Voronoi
# venv). Then validate on a single target with `sbatch --array=1` before
# launching the full array.
#
# Results are written to
#   /nfs/roberts/pi/pi_co54/jas485/ppi_gnn_data_store/apbs_model_data/
# as <target>_apbs_surface.hdf5 plus <target>_apbs_summary.csv. The training
# graph HDF5 files are never opened. To resubmit only incomplete targets, use
# apbs_analysis/cluster/resubmit_failed.sh.

set -euo pipefail
bash /nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder/apbs_analysis/cluster/run_target.sh
