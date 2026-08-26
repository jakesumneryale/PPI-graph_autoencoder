#!/usr/bin/env bash
# Resubmit only the targets without a completion marker -- covers targets that
# never ran, timed out mid-run, or hit per-model errors. Finished models are
# skipped automatically by the worker, so this never redoes completed work.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder}"
PDB_BASE_DIR="${PDB_BASE_DIR:-/nfs/roberts/pi/pi_co54/jas485/uniformly_sampled_target_data}"
RUN_DIR="$PROJECT_DIR/apbs_run"
TARGETS_FILE="$PROJECT_DIR/apbs_analysis/cluster/targets.txt"
DONE_MARKER_DIR="$RUN_DIR/completed"
THROTTLE="${THROTTLE:-24}"

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
  if [[ ! -d "$PDB_BASE_DIR/sampled_$target" ]]; then
    unavailable+=("$target (sampled PDB directory missing)")
    continue
  fi
  if [[ ! -f "$DONE_MARKER_DIR/${target}.done" ]]; then
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
sbatch --array="${array_spec}%${THROTTLE}" "$PROJECT_DIR/apbs_analysis/cluster/dispatch_apbs_jobs.slurm"
