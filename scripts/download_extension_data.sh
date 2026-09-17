#!/usr/bin/env bash
# Run on the workstation. Globus CLI and an authenticated account are required.
# Set SOURCE_COLLECTION and SOURCE_BUNDLE (path as exposed by that collection).
# Destination defaults to the locally configured GCP endpoint and /scratch.
# No deletion, no CLI installation, and no endpoint permission changes occur.
set -euo pipefail
SOURCE_COLLECTION="${SOURCE_COLLECTION:-a2bf0df9-5633-4565-b083-b8907423bb77}"
SOURCE_BUNDLE="${SOURCE_BUNDLE:-/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder/local_transfer_full}"
GLOBUS_BIN="${GLOBUS_BIN:-globus}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LOCAL_ROOT="${LOCAL_ROOT:-/scratch/ppi_extension_data/full}"
DESTINATION_PATH="${DESTINATION_PATH:-$LOCAL_ROOT}"
DESTINATION_COLLECTION="${DESTINATION_COLLECTION:-60ab259b-1d82-11f1-bd71-0e34a6ec9899}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
command -v "$GLOBUS_BIN" >/dev/null || { echo 'Globus CLI not found. Set GLOBUS_BIN to its executable or install globus-cli in your chosen environment.' >&2; exit 2; }
case "${1:-}" in
  --dry-run)
    exec "$GLOBUS_BIN" transfer "$SOURCE_COLLECTION:${SOURCE_BUNDLE%/}/" \
      "$DESTINATION_COLLECTION:${DESTINATION_PATH%/}/" --recursive \
      --sync-level checksum --verify-checksum --notify off --dry-run ;;
  '') ;;
  *) echo 'Usage: bash scripts/download_extension_data.sh [--dry-run]' >&2; exit 2 ;;
esac
mkdir -p "$LOCAL_ROOT"
# Download the manifest first to inspect size before transferring large data.
submit() {
  "$GLOBUS_BIN" transfer "$1" "$2" --sync-level checksum --verify-checksum \
    --notify off --format json "${@:3}" |
    "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["task_id"])'
}
wait_task() {
  local task_id="$1" result
  while true; do
    if "$GLOBUS_BIN" task wait "$task_id" --timeout 60 --polling-interval 10 --timeout-exit-code 50; then
      return
    else
      result=$?
      if [[ "$result" != 50 ]]; then
        echo "Transfer failed or needs attention: $task_id. Inspect with: $GLOBUS_BIN task show $task_id" >&2
        return "$result"
      fi
      echo "Transfer still running: $task_id"
    fi
  done
}
MANIFEST_TASK=$(submit "$SOURCE_COLLECTION:${SOURCE_BUNDLE%/}/bundle_manifest.json" \
  "$DESTINATION_COLLECTION:${DESTINATION_PATH%/}/bundle_manifest.json")
echo "Manifest task: $MANIFEST_TASK"
wait_task "$MANIFEST_TASK"
"$PYTHON_BIN" "$SCRIPT_DIR/../plan_extension_download.py" --local-root "$LOCAL_ROOT" \
  --source-bundle "$SOURCE_BUNDLE" --destination-path "$DESTINATION_PATH" \
  --reuse-pdb "${REUSE_PDB_ROOT:-/scratch/uniformly_sampled_ppi_data}" \
  --reuse-rsasa "${REUSE_RSASA_DIR:-/scratch/ppi_autoencoder_code/rsasa_i_data}" \
  --reuse-esm "${REUSE_ESM_ROOT:-$SCRIPT_DIR/../ESM2_embedding_ex}"
if [[ -s "$LOCAL_ROOT/.globus-transfer.txt" ]]; then
  DATA_TASK=$(submit "$SOURCE_COLLECTION" "$DESTINATION_COLLECTION" --batch "$LOCAL_ROOT/.globus-transfer.txt")
  printf '%s\n' "$DATA_TASK" > "$LOCAL_ROOT/.globus_transfer_task"
  echo "Data task: $DATA_TASK"
  wait_task "$DATA_TASK"
fi
"$PYTHON_BIN" - "$LOCAL_ROOT" <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
m = json.loads((root / 'bundle_manifest.json').read_text())
for name, size in m['files'].items():
    p = root / name
    if not p.is_file() or p.stat().st_size != size:
        raise SystemExit(f'Incomplete download: {p}')
    digest = hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != m['file_sha256'][name]:
        raise SystemExit(f'Checksum mismatch: {p}')
print(f'Complete: {root}. Globus verified transfer checksums.')
PY
