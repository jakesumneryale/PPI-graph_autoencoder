#!/usr/bin/env bash
#SBATCH --job-name=voronoi_resume
#SBATCH --partition=pi_co54
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@180
#SBATCH --output=/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder/voronoi_run/logs/%x_%A_%a.out
#SBATCH --error=/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder/voronoi_run/logs/%x_%A_%a.err
#
# Continuation of dispatch_voronoi_jobs.slurm with a longer limit. run_target.sh already
# skips every model whose checkpoint group is complete (only missing models are computed) and
# commits only what is not yet in the graph, so this never redoes finished work. Submit it
# through cluster/submit_voronoi_resume.sh, which holds it until the first array has ended so two
# jobs never write the same checkpoint HDF5.
set -uo pipefail
PROJECT_DIR="${PROJECT_DIR:-/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder}"

# Slurm's B: signal reaches only this shell. Forward it to the running Python so it stops after
# the current model and flushes, rather than being killed mid-write at the time limit.
bash "$PROJECT_DIR/cluster/run_target.sh" &
child=$!
trap 'pkill -TERM -P "$child" 2>/dev/null || true' TERM
wait "$child"
status=$?
while (( status > 128 )) && kill -0 "$child" 2>/dev/null; do
  wait "$child"; status=$?
done
exit "$status"
