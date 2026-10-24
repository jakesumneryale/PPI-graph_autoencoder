#!/usr/bin/env bash
# Dataset-size comparison: the rank model (configs/gate_rank.toml) x seeds 7/17/27 on
#   expanded       train = original + batch-1 targets of stratified fold 1 (254 targets)
#   original_only  train = only the original-146 targets of that train split (102 targets)
# with identical val (35) and test (61) targets. 6 GPU runs. See DATASET_SIZE_COMPARISON.md.
#
# Run from the repository on a Bouchet login node, after the batch-1 graphs (with v_es) are in NEW_GRAPH_DIR:
#   bash cluster/submit_dataset_size_comparison.sh audit     # 1-CPU array: model lists for the new graphs
#   DEPEND=<copy job id> bash cluster/submit_dataset_size_comparison.sh audit
#       # audit starts only if cluster/copy_expansion_graphs.slurm succeeds (afterok); the graphs are then checked
#       # in the copy job's source (COPY_SRC) instead of NEW_GRAPH_DIR, which is still being filled
#   bash cluster/submit_dataset_size_comparison.sh train     # check + stage data dirs, submit 2 x 3 GPU runs
#   DRY_RUN=1 bash cluster/submit_dataset_size_comparison.sh train
set -euo pipefail
STAGE=${1:?usage: submit_dataset_size_comparison.sh audit|train}
# The repository this script is in (exported, so the Slurm jobs use it too); the original graphs sit next to it.
export PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}"
OLD_GRAPH_DIR="${OLD_GRAPH_DIR:-$(dirname "$PROJECT_DIR")/ppi_processed_graphs}"
NEW_GRAPH_DIR="${NEW_GRAPH_DIR:-/gpfs/boyd/pi/pi_co54/jas485/ppi_expansion_graphs}"
OLD_MODEL_LISTS="${OLD_MODEL_LISTS:-$PROJECT_DIR/gate_run/full_pooling_repaired/model_lists}"
EXP="$PROJECT_DIR/gate_run/dataset_size_comparison"
CONFIG=configs/gate_rank.toml
ARMS=(expanded original_only)
cd "$PROJECT_DIR"
for f in "${ARMS[@]/#/$EXP/target_splits_}"; do [[ -f ${f}.json ]] || { echo "Missing ${f}.json; sync gate_run/dataset_size_comparison." >&2; exit 2; }; done
ALL_TARGETS=$(sort -u "$EXP"/targets_*.txt)
is_original() { [[ -f $OLD_MODEL_LISTS/$1.txt ]]; }

case "$STAGE" in
audit)
  mkdir -p "$EXP/logs" "$EXP/tasks"
  tasks="$EXP/tasks/audit_$(date +%Y%m%d_%H%M%S).txt"
  check_dir=$NEW_GRAPH_DIR
  [[ -n ${DEPEND:-} ]] && check_dir="${COPY_SRC:-/home/jas485/project_pi_co54/jas485/ppi_expansion_graphs}"
  missing=()
  for t in $ALL_TARGETS; do
    is_original "$t" && continue
    if [[ -s $check_dir/$t.hdf5 ]]; then echo "$NEW_GRAPH_DIR/$t.hdf5" >> "$tasks"; else missing+=("$t"); fi
  done
  (( ${#missing[@]} == 0 )) || { echo "Missing batch-1 graph files in $check_dir: ${missing[*]}" >&2; exit 2; }
  n=$(grep -c . "$tasks")
  args=(--parsable --export=ALL --array="1-$n" --output="$EXP/logs/audit_%A_%a.out" --error="$EXP/logs/audit_%A_%a.err")
  [[ -n ${DEPEND:-} ]] && args+=(--dependency="afterok:$DEPEND" --kill-on-invalid-dep=yes)
  if [[ -n ${DRY_RUN:-} ]]; then echo "sbatch ${args[*]} cluster/audit_expansion_graphs.slurm $tasks $EXP"; exit 0; fi
  echo "Submitted audit of $n batch-1 graphs: $(sbatch "${args[@]}" cluster/audit_expansion_graphs.slurm "$tasks" "$EXP")"
  echo "Model lists: $EXP/model_lists; reports: $EXP/audit"
  ;;
train)
  # One model-list directory: the existing lists for original targets, the audit's lists for batch-1 targets.
  # Every target needs a non-empty list; any missing graph, list or v_es-less batch-1 audit stops here.
  problems=()
  for t in $ALL_TARGETS; do
    if is_original "$t"; then
      [[ -s $OLD_GRAPH_DIR/$t.hdf5 ]] || problems+=("$t: no graph in $OLD_GRAPH_DIR")
      [[ -e $EXP/model_lists/$t.txt ]] || { mkdir -p "$EXP/model_lists"; ln -s "$OLD_MODEL_LISTS/$t.txt" "$EXP/model_lists/$t.txt"; }
    else
      [[ -s $NEW_GRAPH_DIR/$t.hdf5 ]] || problems+=("$t: no graph in $NEW_GRAPH_DIR")
      [[ -f $EXP/audit/$t.json ]] || problems+=("$t: not audited (run the audit stage)")
    fi
    [[ -s $EXP/model_lists/$t.txt ]] || problems+=("$t: empty or missing model list")
  done
  (( ${#problems[@]} == 0 )) || { printf '%s\n' "${problems[@]}" >&2; echo "${#problems[@]} problems; nothing submitted." >&2; exit 2; }

  module load miniconda
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate py311_env
  source "$PROJECT_DIR/venv/bin/activate"
  python train_gate.py --config "$CONFIG" --print-config > /dev/null
  # The manifests were written with an assumed cluster path; point them at the data directories made below.
  for arm in "${ARMS[@]}"; do
    python - "$EXP/target_splits_$arm.json" "$EXP/data_$arm" <<'PY'
import json, sys
path, data = sys.argv[1], sys.argv[2]
m = json.load(open(path))
m['data_path'] = data
for split in m['splits'].values():
    split['paths'] = [f'{data}/{t}.hdf5' for t in split['targets']]
open(path, 'w').write(json.dumps(m, indent=2) + '\n')
PY
  done

  # ~1,300 models per original target and ~2,000 per batch-1 target, so the expanded arm has ~3x the training
  # graphs per epoch. Validating every 2,000 steps would then validate the 35-target set ~14x per epoch; scaling
  # the interval keeps both arms at ~4 validations per epoch (as in the rank follow-up) and inside the walltime.
  declare -A VAL_EVERY=([expanded]=6000 [original_only]=2000)
  for arm in "${ARMS[@]}"; do
    data="$EXP/data_$arm"; out="$EXP/$arm"
    [[ -e $out ]] && { echo "Refusing to reuse $out; move it away first." >&2; exit 2; }
    mkdir -p "$data"
    while read -r t; do
      if is_original "$t"; then src="$OLD_GRAPH_DIR/$t.hdf5"; else src="$NEW_GRAPH_DIR/$t.hdf5"; fi
      ln -sfn "$src" "$data/$t.hdf5"
    done < "$EXP/targets_$arm.txt"
    extra=$(comm -13 <(sort "$EXP/targets_$arm.txt") <(ls "$data" | sed 's/\.hdf5$//' | sort))
    [[ -z $extra ]] || { echo "$data has files outside the manifest: $extra" >&2; exit 2; }
    args=(--parsable --export=ALL --array=1-3 --time=6-23:00:00 --job-name="gate_size_$arm"
          --output="$out/logs/train_%A_%a.out" --error="$out/logs/train_%A_%a.err")
    targs=("$CONFIG" "$out" --data "$data" --model-list-dir "$EXP/model_lists"
           --split-manifest "$EXP/target_splits_$arm.json" --val-every-steps "${VAL_EVERY[$arm]}")
    if [[ -n ${DRY_RUN:-} ]]; then echo "sbatch ${args[*]} cluster/train_gate_config.slurm ${targs[*]}"; continue; fi
    mkdir -p "$out/logs"
    echo "$arm: $(sbatch "${args[@]}" cluster/train_gate_config.slurm "${targs[@]}") -> $out/gate_rank_seed{7,17,27}"
  done
  ;;
*) echo "Unknown stage: $STAGE (audit|train)" >&2; exit 2 ;;
esac
