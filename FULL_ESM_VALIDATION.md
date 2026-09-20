# Six full-dataset ESM2 additions

These are paired additions to `full_pooling_validation`, not replacements or
resubmissions of its 12 control jobs:

| Pooling | ESM2 | Seeds | DockQ weighting exponent |
|---|---|---|---|
| Whole graph mean/max | Layer 33 | 7, 17, 27 | 0.75 |
| Whole graph + interface mean/max | Layer 33 | 7, 17, 27 | 0.75 |

All six retain the full parent cohort, frozen target split, original structural
features (including current average rSASA_i), reconstruction lambda 1, and other
training settings. APBS is excluded. ESM2 adds 1280 residue features, projected to
64 dimensions by the existing trainable LayerNorm/linear/SiLU encoder input.
ESM vectors are not extra reconstruction targets. Compare each run against its
matching pooling/seed/weighting control; keep the test split untouched.

After pulling the code on Bouchet, wait for the original full-pooling **matrix
build** to succeed (the 12 GPU runs need not finish), then run:

```bash
bash cluster/submit_full_esm_validation.sh
```

If the parent experiment has a custom location, set `FULL_PARENT_DIR`.
`EXPERIMENT_DIR`, `ESM_ROOT`, and `ESM_SIDECARS` may also be overridden. The default
embedding root is `/nfs/roberts/pi/pi_co54/nb685/scratch_backup/SS_embeds`.

The launcher submits one serial preparation task per target, then a serial
cohort validation job, then six GPU tasks. Dependencies block training if any
preparation fails. Preparation requests one CPU; GPU training requests eight
CPUs and uses one main process plus seven DataLoader workers, each with one
numerical-library thread. There is no array concurrency cap.

Sidecars align graph node_reference sequences to FASTA sequences and embedding
labels, rather than assuming graph chain IDs equal A/B. The existing alignment
rules reject ambiguous mappings unless the candidate embeddings are identical.
Identical node layouts share one HDF5 embedding dataset via hard links. Raw
source graphs are unchanged. Every selected graph must have a valid embedding;
failures are reported in per-target audit JSON without silently shrinking the
cohort. The inherited full-dataset coverage gate still applies: this does not
manufacture missing structural data to reach 180,000 graphs.

Results and combined stdout/stderr logs:
`gate_run/full_pooling_esm_validation/` and its `logs/` directory.
Preparation audits: `full_pooling_esm_sidecars/*.audit.json`.
Use a fresh experiment directory for a new submission.

When finished:

```bash
python summarize_full_esm_validation.py \
  --parent gate_run/full_pooling_validation \
  --esm gate_run/full_pooling_esm_validation
```

The report verifies paired validation identities and labels, reports pooled and
per-target MSE and Spearman correlation, and writes paired deltas plus DockQ-bin
metrics under each experiment's analysis directory. Negative MSE deltas favor
ESM. Three seeds measure training variability; targets are the relevant units
for assessing generalization. Do not claim a benefit from training loss alone.

## Missing parent matrix

The launcher requires the full-pooling inventory (`paths.json`) and its cohort
matrix (`matrix.json`). A missing matrix does not prove the build is running:
check `gate_run/full_pooling_validation/logs/build_*.out` for a failed build,
or point `FULL_PARENT_DIR` to your actual parent directory.
If the parent has not been submitted, launch `cluster/submit_full_pooling_validation.sh`
first. Do not repeat that submission if it already exists.

With an existing inventory and a pending parent build, you can queue ESM jobs
immediately using the numeric **Coverage/matrix** job ID printed by that launcher:

```bash
FULL_BUILD_JOB_ID=12345678 bash cluster/submit_full_esm_validation.sh
```

Replace the example ID with the real parent build ID. Preparation receives an
`afterok` dependency, so a failed parent build does not release ESM training.
