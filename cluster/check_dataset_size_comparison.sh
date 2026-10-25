#!/usr/bin/env bash
# Read-only pre-training check for the dataset-size comparison (nothing is written or submitted):
#   1. the batch-1 graphs are all in NEW_GRAPH_DIR (/gpfs/boyd) and match the Globus landing copy in size
#   2. the copy job's log ends in its checksum OK line
#   3. every batch-1 target has an audit report and a non-empty model list; rejection summary
#   4. every original target has its graph and existing model list
#   5. your queued/running jobs
# Run from the repository on a Bouchet login node:
#   bash cluster/check_dataset_size_comparison.sh
set -uo pipefail
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}"
OLD_GRAPH_DIR="${OLD_GRAPH_DIR:-$(dirname "$PROJECT_DIR")/ppi_processed_graphs}"
NEW_GRAPH_DIR="${NEW_GRAPH_DIR:-/gpfs/boyd/pi/pi_co54/jas485/ppi_expansion_graphs}"
COPY_SRC="${COPY_SRC:-/home/jas485/project_pi_co54/jas485/ppi_expansion_graphs}"
OLD_MODEL_LISTS="${OLD_MODEL_LISTS:-$PROJECT_DIR/gate_run/full_pooling_repaired/model_lists}"
EXP="$PROJECT_DIR/gate_run/dataset_size_comparison"
fail=0
bad() { echo "  PROBLEM: $*"; fail=1; }

ALL_TARGETS=$(sort -u "$EXP"/targets_*.txt)
NEW=(); OLD=()
for t in $ALL_TARGETS; do if [[ -f $OLD_MODEL_LISTS/$t.txt ]]; then OLD+=("$t"); else NEW+=("$t"); fi; done
echo "Targets in the manifests: ${#OLD[@]} original, ${#NEW[@]} batch-1 (expected 146 + 204)"
(( ${#OLD[@]} == 146 && ${#NEW[@]} == 204 )) || bad "unexpected target counts"

echo "1. Batch-1 graphs in $NEW_GRAPH_DIR"
n=0; mism=0
for t in "${NEW[@]}"; do
  f=$NEW_GRAPH_DIR/$t.hdf5
  if [[ ! -s $f ]]; then bad "missing $f"; continue; fi
  n=$((n + 1))
  [[ -f $COPY_SRC/$t.hdf5 && $(stat -c %s "$f") != $(stat -c %s "$COPY_SRC/$t.hdf5") ]] && { bad "$t size differs from $COPY_SRC"; mism=$((mism + 1)); }
done
echo "  $n/${#NEW[@]} present, $(du -sh "$NEW_GRAPH_DIR" 2>/dev/null | cut -f1) total, $mism size mismatches vs landing copy"

echo "2. Copy job log"
log=$(ls -t "$PROJECT_DIR"/copy_expansion_graphs_*.out 2>/dev/null | head -1)
if [[ -z $log ]]; then bad "no copy_expansion_graphs_*.out in $PROJECT_DIR"
else echo "  $(basename "$log"): $(tail -1 "$log")"; grep -q ' OK: ' "$log" || bad "copy log has no OK line"; fi

echo "3. Audit of the batch-1 graphs ($EXP/audit, model_lists)"
python3 - "$EXP" "${NEW[@]}" <<'PY' || fail=1
import json, sys
from collections import Counter
from pathlib import Path
exp, targets = Path(sys.argv[1]), sys.argv[2:]
raw = acc = 0; reasons = Counter(); low = []; missing = []; empty = []
for t in targets:
    r = exp / 'audit' / f'{t}.json'
    if not r.is_file(): missing.append(t); continue
    d = json.loads(r.read_text())
    raw += d['raw_models']; acc += len(d['accepted']); reasons.update(d['rejection_counts'])
    if d['raw_models'] and len(d['accepted']) < 0.9 * d['raw_models']: low.append(f"{t} {len(d['accepted'])}/{d['raw_models']}")
    ml = exp / 'model_lists' / f'{t}.txt'
    if not ml.is_file() or not ml.read_text().strip(): empty.append(t)
print(f'  {len(targets) - len(missing)}/{len(targets)} audited: {acc:,} of {raw:,} models accepted ({acc / max(raw, 1):.2%})')
for why, k in reasons.most_common(5): print(f'    rejected {k:,}: {why[:100]}')
for label, items in (('not audited', missing), ('empty/missing model list', empty), ('<90% accepted', low)):
    if items: print(f'  PROBLEM: {label}: {" ".join(items[:20])}{" ..." if len(items) > 20 else ""}')
sys.exit(1 if missing or empty else 0)
PY
errs=$(grep -l -E 'Traceback|Error|Killed|CANCELLED' "$EXP"/logs/audit_*.err 2>/dev/null | wc -l)
(( errs == 0 )) && echo "  audit logs: no errors" || bad "$errs audit .err logs with errors (grep -l Traceback $EXP/logs/audit_*.err)"

echo "4. Original targets"
miss_g=0; miss_l=0
for t in "${OLD[@]}"; do
  [[ -s $OLD_GRAPH_DIR/$t.hdf5 ]] || miss_g=$((miss_g + 1))
  [[ -s $OLD_MODEL_LISTS/$t.txt ]] || miss_l=$((miss_l + 1))
done
echo "  graphs missing in $OLD_GRAPH_DIR: $miss_g; empty model lists: $miss_l"
(( miss_g == 0 && miss_l == 0 )) || bad "original-target inputs incomplete"

echo "5. Your jobs"
squeue --me --format='  %.10i %.9P %.24j %.2t %.10M %R' 2>/dev/null || echo "  (squeue unavailable)"

echo
(( fail == 0 )) && echo "ALL CHECKS PASSED. Next: DRY_RUN=1 bash cluster/submit_dataset_size_comparison.sh train" \
                || echo "SOME CHECKS FAILED (see PROBLEM lines above)."
exit $fail
