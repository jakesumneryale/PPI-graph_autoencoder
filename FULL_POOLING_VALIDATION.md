# Full-dataset pooling validation

After syncing the code, run from the cluster repository:

```bash
bash cluster/submit_full_pooling_validation.sh
```

Default graph directory:
`/nfs/roberts/project/pi_co54/jas485/ppi_processed_graphs`.
Default rSASA directory:
`/nfs/roberts/project/pi_co54/jas485/rsasa_i_graph_data`.
Override with `DATA_DIR` and `RSASA_DIR` if needed. This launcher deliberately
does not use `extension_data_10pct_*`, a subset HDF5 directory, old subset model
lists, or the old launcher's "full" mode (which meant more seeds on 10% data).

## Cohort checks before GPU work

- A serial inventory rejects HDF5 files marked as 10% subsets.
- One CPU array task per target checks every graph in the original files,
  including rSASA availability, finite/dimensionally consistent features,
  interface masks, valid contacts, label range, area missing masks, and
  interface degrees recomputed from contacts. Source graphs are read-only.
- Rejections and exact model names are recorded under `full_pooling_audit/`.
  Audits can be reused when graph/CSV path, size, modification time and audit
  code signature agree. Do not mutate graph files while this experiment runs.
- A serial build step writes `gate_run/full_pooling_validation/coverage.json`
  with raw/eligible/rejected counts and reasons. It requires at least **180,000
  eligible graphs**, not merely 180,000 raw groups, before releasing GPU jobs.
- Target assignments are preserved from the completed 10% experiment. The
  manifest paths/counts are rebuilt for full data. A previously included target
  becoming empty or a new eligible target outside the split blocks training;
  no target is silently reassigned or dropped. Already excluded targets with
  no usable graphs remain reported exclusions.

**Existing missing features may prevent the 180,000 threshold being met.**
The earlier 10% audit had 21 empty targets, including missing Voronoi areas.
This setup does not invent missing features or claim those graphs are usable.
Inspect the coverage report before deciding whether to repair the source
features or deliberately use a smaller common cohort. An explicit
`MIN_ELIGIBLE_MODELS` override exists for that later decision; it is not lowered
automatically. If a dependency fails, queued GPU tasks do not execute; cancel
those blocked tasks before submitting a fresh experiment directory.

No ESM/APBS/PDB preparation or source mutation is required. Eligibility is
feature-based, never selected using model performance. Label reads check finite
values and [0,1] validity; test predictions/metrics are not computed.

## Twelve from-scratch runs

| Factor | Settings |
|---|---|
| Pooling | Whole graph; combined whole graph + interface |
| Training DockQ weighting exponent | 0.5; 0.75 |
| Seed | 7; 17; 27 |

This 2×2×3 design separates the pooling and weighting effects. All runs train
on the full eligible **training targets**, with all eligible validation/test
models remaining held out by the frozen split. "Full dataset" does not mean
training on held-out targets. The builder prints counts for each split.

Shared settings: four-layer residual GAT, hidden width 64, latent width 32,
four attention heads, dropout 0.3, reconstruction lambda 1, batch size 16,
50 epochs, learning rate 0.0003, cosine schedule and 1,000-step warmup.
Full epochs have more optimizer updates than the earlier 10% epochs; all new
arms have the same data and update budget. Full runs may take substantially
longer. Existing training Slurm limits are retained (one GPU, 80 GB, 6d23h).

Combined pooling concatenates whole-graph mean/max and interface-node mean/max
embeddings before the graph projector. This adds graph-projector parameters;
the comparison is not parameter-count matched.

## Exact input and reconstruction features

Node inputs:
- `aa_type`: stored amino-acid encoding.
- `chain`: stored chain encoding.
- `interface_nodes`: interface membership flag.
- `rsasa_i`: `avg_rsasa_i` from the target CSV, one per-model scalar repeated on
  each node by the current loader (not a residue-specific rSASA measurement).
- `interface_node_degree`: unique neighbors that are interface nodes, treating
  contacts as undirected and ignoring duplicate contacts/self-edges.

Edge inputs:
- `interface_edges`: interface edge feature.
- `ca_dist`: Cα distance.
- `voronoi_contact_area`: log1p transformed and standardized using training-only
  statistics, with the statistics sampling seed fixed at 7.
- `voronoi_contact_missing`: Voronoi-area missingness mask.

Node reconstruction uses the node input features. Edge-attribute reconstruction
uses **only `interface_edges,ca_dist`**; Voronoi area and its mask are inputs only.
Edge-presence reconstruction is also retained. Objective:
`weighted DockQ MSE + node MSE + edge-attribute MSE + 0.1 * edge-presence BCE`.
Range weights use training labels only and mean-one normalization, with a
maximum weight ratio of three. APBS, ESM2 and coordinate-based EGNN inputs are
not used.

## Outputs and resources

Results: `gate_run/full_pooling_validation/`.
Logs: `logs/audit_JOB_TASK.out`, `logs/build_JOB.out`, `logs/train_JOB_TASK.out`.
Each new experiment directory must be fresh; cached audits are kept separately.
After completing/downloading all results:

```bash
python summarize_full_pooling_validation.py gate_run/full_pooling_validation
```

The summary checks complete histories, identical validation cohorts and
prediction/checkpoint consistency, then exports per-seed, per-target and
per-range metrics. Validation MSE remains ordinary/unweighted. This stage
keeps test evaluation sealed; increasing dataset size alone does not create a
new independent validation split after the previous rounds of tuning.

Audit/build: one CPU each, one process, single-threaded numerical libraries.
Training: eight CPUs, one main process plus seven DataLoader workers;
nonpersistent pools and one numerical-library/PyTorch thread per process.
No array concurrency caps are added. Large full-training prediction exports
are disabled; validation predictions and per-epoch range diagnostics remain.

## Local verification

Six tests passed for the full-cohort threshold, subset rejection, unchanged
source files, audit-cache invalidation, frozen target membership, matrix
construction, and mocked Slurm dependency submission. Two real-data CUDA smoke
runs (whole-graph and combined pooling) passed through the new audit/model-list/
split/training/checkpoint/summary pipeline on a copied nine-graph software
fixture. The low minimum used in that fixture is not the production default.
No full cluster cohort count or full-scale runtime has been verified locally.
