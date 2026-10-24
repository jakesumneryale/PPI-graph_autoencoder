#!/usr/bin/env bash
#SBATCH --job-name=copy_expansion_graphs
#SBATCH --partition=pi_co54
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --output=copy_expansion_graphs_%j.out

# Second hop of the batch-1 graph transfer. Globus cannot write to /gpfs/boyd on Bouchet yet, so
# transfer_graphs_to_bouchet.sh (target expansion, local PC) puts the graphs in ~/project_pi_co54/jas485/ppi_expansion_graphs;
# this copies them to the migrated location used by cluster/submit_dataset_size_comparison.sh (NEW_GRAPH_DIR).
#
#   sbatch cluster/copy_expansion_graphs.slurm
#
# One serial rsync (1 CPU), then a full checksum comparison of the two copies; the source is left in place
# (delete it yourself once the check has passed). Rerunning only copies what is missing or different.
set -euo pipefail
SRC="${SRC:-/home/jas485/project_pi_co54/jas485/ppi_expansion_graphs}"
DEST="${DEST:-/gpfs/boyd/pi/pi_co54/jas485/ppi_expansion_graphs}"
EXPECTED="${EXPECTED:-204}"

n_src=$(find "$SRC" -maxdepth 1 -name '*.hdf5' | wc -l)
(( n_src == EXPECTED )) || { echo "Expected $EXPECTED graph files in $SRC, found $n_src; is the Globus transfer finished?" >&2; exit 2; }
mkdir -p "$DEST"
echo "$(date) copying $n_src graphs ($(du -sh "$SRC" | cut -f1)) $SRC -> $DEST"
rsync -a "$SRC"/ "$DEST"/
echo "$(date) copied; comparing checksums"
diffs=$(rsync -rc --dry-run --out-format='%n' "$SRC"/ "$DEST"/)
[[ -z $diffs ]] || { echo "Files differ after copy:" >&2; echo "$diffs" >&2; exit 1; }
echo "$(date) OK: $(find "$DEST" -maxdepth 1 -name '*.hdf5' | wc -l) graph files in $DEST identical to $SRC"
