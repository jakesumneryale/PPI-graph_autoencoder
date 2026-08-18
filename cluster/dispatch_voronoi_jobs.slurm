#!/usr/bin/env bash
#SBATCH --job-name=voronoi_contact_area
#SBATCH --partition=pi_co54
#SBATCH --array=1-146%64
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@120
#SBATCH --output=/nfs/roberts/project/pi_co54/jas485/ppi_autoencoder_project/voronoi_run/logs/%x_%A_%a.out
#SBATCH --error=/nfs/roberts/project/pi_co54/jas485/ppi_autoencoder_project/voronoi_run/logs/%x_%A_%a.err
#
# 146 targets, one per array task. The %64 throttle caps concurrency at 64
# simultaneous 1-CPU/8GB tasks -- the same aggregate budget as "4 nodes x 16
# CPUs @ 8GB/CPU" -- without pinning to 4 specific nodes or reserving them
# exclusively, so the remaining cores/GPUs on whichever nodes pi_co54 lands
# these on stay free for other jobs. Each task is independent (no
# cross-node communication needed), so letting the scheduler backfill array
# tasks wherever there's room is more schedule-friendly than a fixed
# multi-node reservation.
#
# cluster/targets.txt ships with the 146 current target names (one per
# line, line N = array task N). Before submitting the full array: run
# cluster/setup_env.sh once. If the target set ever changes, rerun
# cluster/list_targets.py and update --array above to match the new count.
# Test with `sbatch --array=1` first to validate the environment on one
# target before launching the full array.
#
# To resubmit only incomplete/failed targets after a run, use
# cluster/resubmit_failed.sh instead of resubmitting this whole array.

set -euo pipefail
bash /nfs/roberts/project/pi_co54/jas485/ppi_autoencoder_project/cluster/run_target.sh
