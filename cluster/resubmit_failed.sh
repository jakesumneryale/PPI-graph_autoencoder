#!/usr/bin/env bash
# Resubmit only the targets that don't yet have a completion marker --
# covers targets that never ran, timed out mid-run, or hit a model error.
# Already-computed models are skipped automatically by the compute step,
# so this never redoes finished work.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder}"
GRAPH_DATA_DIR="${GRAPH_DATA_DIR:-/nfs/roberts/project/pi_co54/jas485/ppi_processed_graphs}"
PDB_BASE_DIR="${PDB_BASE_DIR:-/nfs/roberts/project/pi_co54/jas485/uniformly_sampled_target_data}"
RUN_DIR="$PROJECT_DIR/voronoi_run"
TARGETS_FILE="$PROJECT_DIR/cluster/targets.txt"
COMMIT_MARKER_DIR="$RUN_DIR/committed"

if [[ ! -f "$TARGETS_FILE" ]]; then
  echo "Missing $TARGETS_FILE." >&2
  exit 1
fi

indices=()
unavailable=()
idx=0
while IFS= read -r target; do
  idx=$((idx + 1))
  [[ -z "$target" ]] && continue
  if [[ ! -d "$PDB_BASE_DIR/$target" ]]; then
    unavailable+=("$target (PDB directory missing)")
    continue
  fi
  if [[ ! -f "$GRAPH_DATA_DIR/${target}.hdf5" && ! -f "$GRAPH_DATA_DIR/${target}.h5" ]]; then
    unavailable+=("$target (graph HDF5 missing)")
    continue
  fi
  if [[ ! -f "$COMMIT_MARKER_DIR/${target}.done" ]]; then
    indices+=("$idx")
  fi
done < "$TARGETS_FILE"

if [[ ${#unavailable[@]} -gt 0 ]]; then
  echo "Skipping ${#unavailable[@]} unavailable target(s):"
  printf '  %s\n' "${unavailable[@]}"
fi

if [[ ${#indices[@]} -eq 0 ]]; then
  echo "All available targets complete."
  exit 0
fi

array_spec=$(IFS=,; echo "${indices[*]}")
echo "Resubmitting ${#indices[@]} incomplete target(s): $array_spec"
sbatch --array="${array_spec}%64" "$PROJECT_DIR/cluster/dispatch_voronoi_jobs.slurm"
