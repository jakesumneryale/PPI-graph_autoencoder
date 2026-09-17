# Local data transfer and prototype workflow

The transfer uses Bouchet collection `a2bf0df9-5633-4565-b083-b8907423bb77` and workstation collection `60ab259b-1d82-11f1-bd71-0e34a6ec9899`. Defaults download the complete existing 10% subset into `/scratch/ppi_extension_data/full`. Source/destination paths are **paths exposed by those collections**; override `SOURCE_BUNDLE` or `DESTINATION_PATH` if Globus exposes a different root than the filesystem paths below.

## Existing data inspected locally

- `/scratch/ppi_autoencoder_code/processed_graph_data`: 146 HDF5 target files. Sampled files have older feature coverage: 1acb has Voronoi area but lacks its missing mask; 1avx/1ay7 examples lack the area too. These are not assumed interchangeable with current cluster inputs.
- `/scratch/ppi_autoencoder_code/rsasa_i_data`: 146 rSASA CSVs.
- `/scratch/uniformly_sampled_ppi_data`: sampled PDB directories.
- Repository `ESM2_embedding_ex`: 1acb chain embeddings and FASTA.
- No APBS surface HDF5 stores were found in the bounded local directory scan (excluding trash and environments).

The downloader compares each reusable file's SHA-256 against the cluster manifest. Identical PDBs, rSASA CSVs, and ESM examples are symlinked into the local bundle directory without downloading or duplicating them. Different/missing files are transferred. Existing source files are never overwritten. Downloaded graph files contain only the selected subset's models, with current features copied from the full cluster graph files after verifying node references, contacts, and DockQ. This avoids downloading a second full graph dataset.

## 1. Stage a bundle on Bouchet

Sync the new scripts to the cluster repository, then run there:

```bash
cd /nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder
sbatch cluster/bundle_local_extension_data.slurm
```

This requests **one CPU, no GPU**, and performs serial copying/hashing. It creates `local_transfer_full/`, containing every model in each nonempty target of the existing 10% subset. The original selected graph names are preserved; this does not resample the full dataset. The job writes its log to Slurm's default `slurm-JOBID.out`; `bundle_manifest.json` appears only when all copies complete successfully. Do not download an unfinished bundle.

Before copying, the job now checks APBS record coverage across every selected
target and writes `local_transfer_full.preflight.json` alongside the output
directory. Both the plain graph name and its `_corrected` alias are checked.
By default, any missing records stop the job before large files are copied.
The report lists every missing graph, source store, attempted key, PDB path,
and whether it is a sampled model or a random negative.

To proceed with the common cohort that has APBS records, explicitly run:

```bash
sbatch cluster/bundle_local_extension_data.slurm --missing-apbs exclude
```

This excludes missing graphs from the bundled graph dataset itself, so the
baseline and all extensions use the same retained cohort. Targets with no
matches are excluded too. Nothing is filled with zero or resampled. The final
manifest and `apbs_coverage.json` retain the exclusions and counts. Check the
attrition before interpreting results, particularly if missing records are
concentrated among random negatives. Compare all models using this same new
cohort; earlier runs may have different test graphs. To preserve the entire
original subset instead, regenerate the missing APBS records and rerun the
default strict job.

For an audit alone, with no data copying:

```bash
sbatch cluster/bundle_local_extension_data.slurm --preflight-only --missing-apbs exclude
```

An unreadable/corrupt APBS store remains an error even with exclusion enabled.
This preflight checks record existence, not scientific validity; local
preparation still validates APBS content and alignment. The older bundler
removed its temporary staging directory on failure, so completed target copies
from that failed attempt cannot be resumed. The new upfront check prevents
that repeated copying for missing-APBS failures; it is not a general resume
mechanism for other copy failures.

For an optional tiny software smoke test (three models each from 1acb, 1avx, and 1ay7), use:

```bash
sbatch --export=ALL,MODE=smoke cluster/bundle_local_extension_data.slurm
```

Smoke mode creates `local_transfer_smoke/`; these nine models are not suitable for accuracy comparisons. Empty targets are recorded and excluded; missing required source files cause an actionable failure. No scientific validation is bypassed, and APBS weighting metadata is preserved as-is. No APBS, ESM, or Voronoi recalculation occurs. Existing completed bundles are immutable: reuse them, or set `BUNDLE_DIR` to a new path when source data changes. Failed staging is cleaned up; successful source datasets remain untouched.

Python options can be forwarded after the Slurm script, e.g. `--targets 1acb 1avx 1ay7` or `--optional-dir PATH`. The standard A/B FASTA and embedding naming scheme in the supplied listing is used. An optional source audit directory is copied for provenance; all selected names are in the manifest. A source missing one of those required files must be repaired before a complete bundle can be published.

## 2. Enable the workstation destination

Globus Connect Personal (the endpoint service) and `globus` (the Python CLI) are separate programs. The service was found at `/home/jake-sumner/globusconnectpersonal-3.2.8/globusconnectpersonal`; `globus` was not on the agent shell's PATH. Activate the environment where your CLI is installed, or set `GLOBUS_BIN=/absolute/path/to/globus`. If needed, install `globus-cli` into a dedicated environment and run `globus login` yourself.

The current personal endpoint config at `~/.globusonline/lta/config-paths` lists only `~/,0,1`. **Add `/scratch/ppi_extension_data` as a writable accessible directory** in the Globus Connect Personal preferences (or add `/scratch/ppi_extension_data,0,1` to that file and restart the service when no other transfers need it). The scripts do not change endpoint permissions or restart services. See the [official Linux setup instructions](https://docs.globus.org/globus-connect-personal/install/linux/).

## 3. Download on the workstation

From the local repository, after the bundle job succeeds:

```bash
# Preview the endpoint/path selection without submitting a transfer:
bash scripts/download_extension_data.sh --dry-run

# Download the 10% bundle, automatically reusing verified local files:
bash scripts/download_extension_data.sh
```

The script first transfers the manifest, finds reusable local files, checks disk headroom, and then transfers only remaining files as a Globus batch. It records the data task ID in `.globus_transfer_task`, waits for success, and checks every file's SHA-256. Re-running skips matching files. It never requests destination deletion and disables transfer email notifications. If interrupted while a task is active, check that task before rerunning so you do not submit duplicate active transfers:

```bash
globus task show TASK_ID
globus task wait TASK_ID --timeout 60
```

To download the optional smoke bundle, use a separate destination:

```bash
SOURCE_BUNDLE=/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder/local_transfer_smoke \
LOCAL_ROOT=/scratch/ppi_extension_data/smoke \
bash scripts/download_extension_data.sh
```

`SOURCE_COLLECTION`, `DESTINATION_COLLECTION`, `SOURCE_BUNDLE`, `LOCAL_ROOT`, and `DESTINATION_PATH` are configurable. Destination endpoint paths must refer to the same local directory as `LOCAL_ROOT`. Reuse paths can be overridden with `REUSE_PDB_ROOT`, `REUSE_RSASA_DIR`, and `REUSE_ESM_ROOT`. The initial dry-run previews the directory transfer; the real transfer uses the checksum-filtered file batch after obtaining the manifest. CLI behavior follows the [official transfer reference](https://docs.globus.org/cli/reference/transfer/).

## 4. Prepare and test locally

Activate the local scientific Python environment, then:

```bash
source scripts/local_extension_env.sh
# To use the smoke bundle instead:
# export LOCAL_ROOT=/scratch/ppi_extension_data/smoke
# source scripts/local_extension_env.sh

python preflight_model_extensions.py --data "$SUBSET_DIR" --output "$LOCAL_ROOT/preflight"
while IFS= read -r target; do
  python prepare_model_extensions.py --target "$target" \
    --data "$SUBSET_DIR" --output "$EXTENSION_DATA" \
    --pdb-root "$PDB_ROOT" --apbs-dir "$APBS_DIR" --esm-root "$ESM_ROOT" || break
done < "$LOCAL_ROOT/preflight/targets.txt"
```

Current full graph features are already in the bundled subset files; no second `--feature-data` dataset is necessary. Unmarked APBS stores still require verification before using `--assume-area-weighted`. This local workflow deliberately exposes the same alignment/provenance checks that run on the cluster.

Once every target prepares successfully, create a one-epoch, one-seed local matrix:

```bash
python model_extension_experiments.py prepare --data "$EXTENSION_DATA" \
  --output "$EXPERIMENT_DIR" --targets-file "$LOCAL_ROOT/preflight/targets.txt" \
  --optional-node-features-dir "$OPTIONAL_NODE_FEATURES_DIR" \
  --seeds 7 --epochs 1 --batch-size 2 --num-workers 0 --device cuda

# Run one model; matrix indices are one-based. Run additional models sequentially
# on the single workstation GPU, not 19 simultaneous GPU jobs.
python model_extension_experiments.py run --matrix "$EXPERIMENT_DIR/matrix.json" --task 1
```

Use a fresh local experiment directory for each new matrix. Batch size 2 is an initial memory check, not a performance recommendation. The coding session currently cannot communicate with the NVIDIA driver (`nvidia-smi` failed), so successful CUDA execution must be verified before claiming a 4090 test. CPU tests can use `--device cpu` meanwhile. GPU model code is the same on the workstation and cluster; cluster runs retain their L40S allocation and selected training settings. Do not directly compare accuracy from different batches/epochs or the tiny smoke bundle.

The local preparation and example training above use one CPU process. Enable loader workers explicitly only when allocating additional CPUs. Only the cluster launchers request Slurm resources.

Update from downloaded-bundle testing: the host driver and RTX 4090 work. The
restricted coding-session sandbox hides `/dev/nvidia*`; approved host execution
successfully ran PyTorch 2.10.0+cu130 CUDA forward/backward. Use host GPU access
for coding-agent CUDA jobs. A sandbox-only failure is not evidence that the
driver needs reinstalling.

For a full read-only PDB/graph/ESM/APBS/rSASA alignment audit:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python validate_extension_bundle.py \
  --bundle /scratch/ppi_extension_data/full \
  --output local_data_audit/downloaded_bundle/alignment
```

For sequential two-epoch execution checks of all 19 configurations on nine
real graphs, including checkpoint reload and comparison export:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python smoke_test_model_extensions.py \
  --bundle /scratch/ppi_extension_data/full \
  --output local_data_audit/downloaded_bundle/smoke19_cuda --device cuda
```

Use a fresh output directory if that smoke test already exists. This test
explicitly uses stored APBS means only for numerical software checks; it does
not attest to unmarked area weighting. Its outputs are marked SOFTWARE TEST
ONLY and its metrics must not be interpreted as accuracy comparisons. Normal
preparation still enforces APBS provenance.


## Audit all existing local graph files separately

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python audit_local_graphs.py
```

This reads every graph from the 146-file local dataset in one process and writes
`local_data_audit/graphs.csv` and `local_data_audit/summary.json`. It checks required
feature shapes/finiteness, contact indices, node references, DockQ, interface
flags, and Voronoi area/mask coverage. It accepts scalar columns stored either
as `[N]` or `[N,1]`, as the training loader does. It does not modify files or
establish that the underlying physical calculations are correct. Missing
interface-node degree is reported separately; preparation can derive it.
Training remains restricted to the transferred 10% subset, whose newer features
must not be inferred from the older local full-dataset audit.

The completed local audit checked 189,710 graphs across all 146 files:
188,467 passed the core graph checks, and 1,243 were missing required node/edge
datasets. Only 1,389 had valid Voronoi area; none had the Voronoi missing mask
or stored interface-node degree. These results describe the existing local
files, not the newer cluster dataset. Detailed results are in
`local_data_audit/summary.json` and `local_data_audit/graphs.csv`.

Local validation also matched the real 1acb E/I chains to the standardized
ESM A/B files and completed finite CPU forward/backward passes for GAT and
EGNN with 304 nodes and 1,280-dimensional ESM embeddings. The four transfer
workflow tests passed using a mocked Globus CLI; no actual transfer or GPU
training has been performed by this workflow yet.
