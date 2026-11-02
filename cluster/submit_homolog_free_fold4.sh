#!/usr/bin/env bash
# Homolog-free fold 4, four arms (3 seeds, 10 epochs, the rank 0.3 + 8-decoys recipe). See HOMOLOG_FREE_FOLD4.md.
#   arm_a_no_esm        the recipe as is
#   arm_b_esm16         the recipe + ESM2 layer-33 embeddings, projection 16
#   arm_c_no_aa         the recipe without aa_type (input and reconstruction target)
#   arm_d_esm16_no_aa   ESM2 as in arm B, without aa_type
# Run from the repository on a Bouchet login node, in this order:
#   bash cluster/submit_homolog_free_fold4.sh setup    # once: fair-esm into the venv, check the ESM2 weights file
#   bash cluster/submit_homolog_free_fold4.sh esm      # GPU embedding job for the 204 new targets, then 1-CPU sidecar array
#                                                      # for all 350 (the old sidecars predate the cohort repair: 21 lack models)
#   bash cluster/submit_homolog_free_fold4.sh train    # staged data + manifest, submit all arms (DRY_RUN=1 to preview)
# Arms A and C do not need ESM. If sidecars are still building: ESM_DEPEND=<sidecar job> ... train   (arms B and D wait with afterok)
# RUNS="arm_a_no_esm" submits a subset; an existing run directory is refused.
set -euo pipefail
STAGE=${1:?usage: submit_homolog_free_fold4.sh setup|esm|train}
export PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}"
export TORCH_HOME="${TORCH_HOME:-$(dirname "$PROJECT_DIR")/esm_models}"
SRC="$PROJECT_DIR/gate_run/dataset_size_comparison"
OLD_MODEL_LISTS="${OLD_MODEL_LISTS:-$PROJECT_DIR/gate_run/full_pooling_repaired/model_lists}"
OLD_GRAPH_DIR="${OLD_GRAPH_DIR:-$(dirname "$PROJECT_DIR")/ppi_processed_graphs}"
OLD_EMBEDS="${OLD_EMBEDS:-/gpfs/boyd/pi/pi_co54/nb685/scratch_backup/SS_embeds}"
NEW_GRAPH_DIR="${NEW_GRAPH_DIR:-/gpfs/boyd/pi/pi_co54/jas485/ppi_expansion_graphs}"
EXP="$PROJECT_DIR/gate_run/homolog_free_fold4"
DATA="$SRC/data_expanded"; LISTS="$SRC/model_lists"; SIDECARS="$EXP/esm_sidecars"
CONFIG=configs/gate_rank.toml
cd "$PROJECT_DIR"
[[ -f $EXP/target_splits.json ]] || { echo "Missing $EXP/target_splits.json; sync the repository." >&2; exit 2; }

activate() {
  module load miniconda
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate py311_env
  source "$PROJECT_DIR/venv/bin/activate"
}
ALL=$(python3 -c "import json,sys; s=json.load(open(sys.argv[1]))['splits']; print(*sorted(t for v in s.values() for t in v['targets']))" "$EXP/target_splits.json")
is_original() { [[ -f $OLD_MODEL_LISTS/$1.txt ]]; }

case "$STAGE" in
setup)
  activate
  export OMP_NUM_THREADS=1
  python -c "import esm" 2>/dev/null || pip install --no-deps fair-esm==2.0.0
  # The weights URL is blocked from Bouchet (HTTP 403), so they are copied here from a machine that can reach it. Loading the
  # model is killed on a login node, so it is first loaded in the GPU job, which must reproduce stored embeddings before it starts.
  w=$TORCH_HOME/hub/checkpoints/esm2_t33_650M_UR50D.pt
  [[ $(stat -c %s "$w" 2>/dev/null) == 2604537549 && -s ${w%.pt}-contact-regression.pt ]] || { echo "ESM2 weights missing or incomplete in $TORCH_HOME/hub/checkpoints." >&2; exit 2; }
  python -c "import esm; print('fair-esm', esm.__version__)"
  echo "Weights OK in $TORCH_HOME. Next: bash cluster/submit_homolog_free_fold4.sh esm"
  ;;
esm)
  [[ -s $TORCH_HOME/hub/checkpoints/esm2_t33_650M_UR50D.pt ]] || { echo "No weights in $TORCH_HOME; see the setup stage." >&2; exit 2; }
  mkdir -p "$EXP/logs" "$EXP/tasks" "$SIDECARS" "$EXP/esm_embeds"
  stamp=$(date +%Y%m%d_%H%M%S); targets="$EXP/tasks/esm_targets_$stamp.txt"; tasks="$EXP/tasks/esm_sidecars_$stamp.txt"
  problems=()
  for t in $ALL; do
    [[ -s $LISTS/$t.txt ]] || problems+=("$t: no model list in $LISTS")
    if is_original "$t"; then
      [[ -s $OLD_GRAPH_DIR/$t.hdf5 ]] || problems+=("$t: no graph in $OLD_GRAPH_DIR")
      [[ -s $OLD_EMBEDS/${t}_all.fasta ]] || problems+=("$t: no embeddings in $OLD_EMBEDS")
      echo "$OLD_GRAPH_DIR/$t.hdf5 $OLD_EMBEDS" >> "$tasks"
    else
      [[ -s $NEW_GRAPH_DIR/$t.hdf5 ]] || problems+=("$t: no graph in $NEW_GRAPH_DIR")
      echo "$t" >> "$targets"; echo "$NEW_GRAPH_DIR/$t.hdf5 $EXP/esm_embeds" >> "$tasks"
    fi
  done
  (( ${#problems[@]} == 0 )) || { printf '%s\n' "${problems[@]}" >&2; echo "${#problems[@]} problems; nothing submitted." >&2; exit 2; }
  n=$(grep -c . "$tasks")
  a1=(--parsable --export=ALL --output="$EXP/logs/esm_embed_%j.out" --error="$EXP/logs/esm_embed_%j.err")
  a2=(--parsable --export=ALL --array="1-$n" --output="$EXP/logs/esm_sidecar_%A_%a.out" --error="$EXP/logs/esm_sidecar_%A_%a.err")
  if [[ -n ${DRY_RUN:-} ]]; then
    echo "sbatch ${a1[*]} cluster/compute_esm_embeddings.slurm $targets $NEW_GRAPH_DIR $LISTS $EXP/esm_embeds $OLD_EMBEDS"
    echo "sbatch --dependency=afterok:<embed job> ${a2[*]} cluster/build_esm_sidecars.slurm $tasks $LISTS $SIDECARS"
    exit 0
  fi
  j1=$(sbatch "${a1[@]}" cluster/compute_esm_embeddings.slurm "$targets" "$NEW_GRAPH_DIR" "$LISTS" "$EXP/esm_embeds" "$OLD_EMBEDS")
  j2=$(sbatch --dependency="afterok:$j1" --kill-on-invalid-dep=yes "${a2[@]}" cluster/build_esm_sidecars.slurm "$tasks" "$LISTS" "$SIDECARS")
  echo "Embeddings: job $j1 ($(grep -c . "$targets") targets). Sidecars: job $j2 ($n targets; starts after embeddings succeed)."
  echo "Next: ESM_DEPEND=$j2 bash cluster/submit_homolog_free_fold4.sh train"
  ;;
train)
  n_data=$(find "$DATA" -maxdepth 1 -name '*.hdf5' | wc -l); broken=$(find "$DATA" -maxdepth 1 -name '*.hdf5' -xtype l | wc -l)
  (( n_data == 350 && broken == 0 )) || { echo "$DATA: $n_data graph links ($broken broken); expected 350." >&2; exit 2; }
  n_lists=$(find "$LISTS" -maxdepth 1 -name '*.txt' | wc -l)
  (( n_lists == 350 )) || { echo "$LISTS: $n_lists model lists; expected 350." >&2; exit 2; }
  activate
  # The committed manifest has targets only; staged/ gets the paths for this checkout (git-ignored).
  mkdir -p "$EXP/staged"; MANIFEST="$EXP/staged/target_splits.json"
  python - "$EXP/target_splits.json" "$DATA" "$MANIFEST" <<'PY'
import json, sys
src, data, path = sys.argv[1:4]
m = json.load(open(src)); m['data_path'] = data
for split in m['splits'].values():
    split['paths'] = [f'{data}/{t}.hdf5' for t in split['targets']]
open(path, 'w').write(json.dumps(m, indent=2) + '\n')
PY
  # Same recipe as rank0.3_8decoys in the ranking sweep; the 3-day walltime and validation interval also match it.
  COMMON=(--data "$DATA" --model-list-dir "$LISTS" --split-manifest "$MANIFEST" --epochs 10 --val-every-steps 6000
          --rank-loss-weight 0.3 --targets-per-batch 2)
  ESM="--use-esm --esm-sidecar-dir $SIDECARS --esm-projection-dim 16"
  # node_mse averages over columns, so dropping 20 of 24 would weight the other 4 six times more; 4/24 keeps their weight unchanged.
  NO_AA="--node-features chain,interface_nodes,interface_node_degree,rsasa_i_node --node-weight 0.166667"
  declare -A RUN_ARGS=([arm_a_no_esm]="" [arm_b_esm16]="$ESM" [arm_c_no_aa]="$NO_AA" [arm_d_esm16_no_aa]="$ESM $NO_AA")
  for run in ${RUNS:-arm_a_no_esm arm_b_esm16 arm_c_no_aa arm_d_esm16_no_aa}; do
    [[ -v "RUN_ARGS[$run]" ]] || { echo "Unknown run: $run" >&2; exit 2; }
    out="$EXP/$run"; dep=()
    [[ -e $out ]] && { echo "Refusing to reuse $out; move it away first." >&2; exit 2; }
    if [[ $run == *esm16* ]]; then
      if [[ -n ${ESM_DEPEND:-} ]]; then
        dep=(--dependency="afterok:$ESM_DEPEND" --kill-on-invalid-dep=yes)
      else
        n_side=$( { find -L "$SIDECARS" -maxdepth 1 -name '*.hdf5' 2>/dev/null || true; } | wc -l)
        (( n_side == 350 )) || { echo "$SIDECARS: $n_side sidecars, expected 350. Run the esm stage, or pass ESM_DEPEND=<sidecar job>." >&2; exit 2; }
      fi
    fi
    # shellcheck disable=SC2206
    targs=("$CONFIG" "$out" "${COMMON[@]}" ${RUN_ARGS[$run]})
    python train_gate.py --config "$CONFIG" "${targs[@]:2}" --output-dir /tmp/unused --print-config > /dev/null
    args=(--parsable --export=ALL --array=1-3 --time=3-00:00:00 --job-name="gate_hf4_$run" "${dep[@]}"
          --output="$out/logs/train_%A_%a.out" --error="$out/logs/train_%A_%a.err")
    if [[ -n ${DRY_RUN:-} ]]; then echo "sbatch ${args[*]} cluster/train_gate_config.slurm ${targs[*]}"; continue; fi
    mkdir -p "$out/logs"
    echo "$run: $(sbatch "${args[@]}" cluster/train_gate_config.slurm "${targs[@]}") -> $out/gate_rank_seed{7,17,27}"
  done
  ;;
*) echo "Unknown stage: $STAGE (setup|esm|train)" >&2; exit 2 ;;
esac
