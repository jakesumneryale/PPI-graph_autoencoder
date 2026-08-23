#!/usr/bin/env bash
#SBATCH --job-name=voronoi_contact_area
#SBATCH --partition=pi_co54
#SBATCH --array=1-146%8
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@120
#SBATCH --output=/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder/voronoi_run/logs/%x_%A_%a.out
#SBATCH --error=/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder/voronoi_run/logs/%x_%A_%a.err
#
# 146 targets, one per array task. A measured 304-residue 1acb model peaked at
# 10.21 GiB RSS. Linear scaling to the 1,500-residue theoretical maximum is
# 50.4 GiB, so each task requests 64 GiB for headroom. The manifold geometry
# operation used about three CPU cores in measurement, so each task requests
# four. The %8 throttle preserves the previous aggregate memory budget of
# roughly 512 GiB while avoiding 64 simultaneous memory-heavy jobs. Tasks
# remain independent and need no cross-node communication, so the scheduler
# can place and backfill them wherever they fit.
#
# cluster/targets.txt ships with the 146 current target names (one per
# line, line N = array task N). Before submitting the full array: run
# cluster/setup_env.sh once; it loads miniconda and uses py311_env to build the
# project venv. Each worker recreates that interpreter context and then
# activates the venv. If the target set ever changes, rerun
# cluster/list_targets.py and update --array above to match the new count.
# Test with `sbatch --array=1` first to validate the environment on one
# target before launching the full array.
#
# A target whose PDB directory or graph HDF5 is absent is logged and cleanly
# skipped without affecting other array tasks. To resubmit only incomplete,
# available targets after a run, use cluster/resubmit_failed.sh instead of
# resubmitting this whole array.

set -euo pipefail
bash /nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder/cluster/run_target.sh
