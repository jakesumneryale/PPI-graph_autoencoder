#!/usr/bin/env bash
#SBATCH --job-name=repair_cohort
#SBATCH --partition=pi_co54
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
set -euo pipefail
cd "$PROJECT_DIR"
module load miniconda
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate py311_env
source "$PROJECT_DIR/venv/bin/activate"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
[[ "${SLURM_CPUS_PER_TASK:-1}" == 1 ]] || { echo 'Serial stage requires one CPU.' >&2; exit 2; }
echo "Cohort repair $STAGE: 1 process, 1 numerical-library thread"
case "$STAGE" in
 inventory)
  python full_pooling_validation.py inventory --data "$DATA_DIR" --output "$EXPERIMENT_DIR/paths.json" ;;
 repair)
  # The only stage that writes to the source graphs, and it writes exactly one dataset.
  python repair_interface_node_degree.py run --paths "$EXPERIMENT_DIR/paths.json" \
   --task "$SLURM_ARRAY_TASK_ID" --output "$REPAIR_REPORTS" ${REPAIR_DRY_RUN:+--dry-run} ;;
 repair_summary)
  python repair_interface_node_degree.py summarize --reports "$REPAIR_REPORTS" \
   --expected-targets "$TARGET_COUNT" ;;
 audit)
  python full_pooling_validation.py audit --paths "$EXPERIMENT_DIR/paths.json" \
   --task "$SLURM_ARRAY_TASK_ID" --rsasa "$RSASA_DIR" --output "$AUDIT_DIR" ;;
 build)
  # Rebuilds the cohort, the split manifest and the model lists from the fresh audit.
  # Target assignment comes from PRIOR_SPLIT; sample counts are recomputed, so a manifest
  # whose counts predate the repair is corrected rather than trusted.
  python full_pooling_validation.py build --paths "$EXPERIMENT_DIR/paths.json" \
   --audit-dir "$AUDIT_DIR" --prior-split "$PRIOR_SPLIT" --source-matrix "$SOURCE_MATRIX" \
   --output "$EXPERIMENT_DIR" --rsasa "$RSASA_DIR" --minimum "$MIN_ELIGIBLE_MODELS" ;;
 *) echo "Unknown STAGE: $STAGE" >&2; exit 2 ;;
esac
