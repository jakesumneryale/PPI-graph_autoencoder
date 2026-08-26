#!/usr/bin/env bash
#SBATCH --job-name=apbs_surface_potential
#SBATCH --partition=day
#SBATCH --array=1-146%24
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
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
# 5.5 h at two workers, or 8-11 h derating for slower cluster cores.
#
# --time is deliberately set *below* what a full target needs in the worst
# case. This runs on the shared `day` partition, where a 12 h request backfills
# into gaps a 24 h request cannot reach, and the whole pipeline is resumable:
# a task that runs out of time has already committed every model it finished,
# withholds its completion marker, and resubmit_failed.sh restarts it from
# exactly where it stopped. Trading a possible second pass for a much shorter
# queue wait is the right side of that bargain.
#
# Roughly 1,800 median-sized models fit in 12 h at two workers derated 1.75x,
# so most 1,400-model targets finish in one pass and the rest need one more.
# Run resubmit_failed.sh until it reports "All available targets complete."
#
# %24 keeps twenty-four targets running at once (48 cores, 384 GiB). On a
# shared partition that throttle is about queue courtesy and per-user limits,
# not a private memory budget: if jobs sit pending, lower it rather than raise
# it. Check `scontrol show partition day` for the partition's MaxTime before
# the first submission -- sbatch rejects at submit time if --time exceeds it.
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
