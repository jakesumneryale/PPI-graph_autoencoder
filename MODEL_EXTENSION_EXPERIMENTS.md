# APBS, interface pooling, ESM2, and EGNN experiments

The control is the **current full-feature residual four-layer GAT**: amino-acid identity, chain, interface flag, rSASA_i, interface-node degree, CA distance, Voronoi contact area, and its missing mask. Voronoi and the new APBS quantities are encoder inputs only; edge reconstruction still targets interface_edges and ca_dist. This matches the `full` arm of the current seed-replicate scripts. Those scripts and their running jobs are unchanged.

## 1. Prepare the same existing 10% subset

Run preprocessing on the cluster, where the APBS stores, original decoy PDBs, full feature graphs, and embeddings live. There is no need to recompute Voronoi tessellations or change contact topology for the initial EGNN. The preparer adds CA coordinates from each original decoy PDB, outside the scalar feature matrix.

`prepare_model_extensions.py` copies only graph names already in the subset. `--feature-data` optionally supplies newer committed features from the full graph files, after verifying identical node references, contacts, and DockQ. It never writes to either source. Existing residue indices must be zero-based and agree with PDB sequence and chain order. Interface-node degree uses the existing repository definition: number of unique neighboring nodes labeled as interface, including neighbors on the same chain.

Outputs are `<target>.hdf5` plus `<target>.audit.json`. Each target file is replaced atomically after processing. Every rejected graph and reason is recorded. Graphs with absent APBS models, ambiguous ESM alignment, missing CA atoms, or empty interfaces are rejected. A target with no surviving graphs fails the preparation job. All variants subsequently use this **same intersection**, including the baseline. Review attrition before interpreting the experiment: results apply to this common cohort, which can be smaller than the original 10% subset.

### APBS semantics and provenance

The onboard surface pipeline already samples the DX map on solvent-accessible surface points and computes the residue mean

`phi_i = sum(point_area * point_potential) / sum(point_area)`.

These are whole-complex surface potentials, in kT/e. For each existing undirected contact `(i,j)`, the new features are:

| Feature | Definition |
| --- | --- |
| apbs_pair_mean | `(phi_i + phi_j) / 2` |
| apbs_pair_absdiff | `abs(phi_i - phi_j)` |
| apbs_pair_product | `phi_i * phi_j` |
| apbs_pair_area_product | `Voronoi_area_ij * phi_i * phi_j` |
| apbs_pair_missing | either residue has no valid surface potential |

The product retains sign information about the two surface means. The area-product is an **additional contact-area descriptor**, with units Å²(kT/e)². Neither is a physical pair interaction energy or an average over the exact contact patch. The area-weighting of a residue's solvent-accessible surface and the Voronoi contact area are different geometric quantities. Computing true patch-specific partner potentials would need retained surface geometry/maps, suitable sampling at the patch, and potentially separate partner electrostatic solves; residue means alone cannot recover that quantity.

Buried residues, residues absent from PQR, and truncated residues are marked missing. Their continuous edge features become zero **after** normalization. Normalization uses only valid observations sampled from training targets, with a fixed statistics seed across runs. The area-product also excludes missing Voronoi areas. A feature with no valid sampled training observations causes training to fail. APBS inputs are standardized, without applying log1p to signed potentials/products. The checkpoints retain the training statistics for evaluation.

Older APBS stores may lack provenance for area-weighted means. New model groups now record `residue_statistics_area_weighted=True`. The preparer also recognizes the file-level marker written by the existing migration utility. For unmarked stores:

- Inspect their generation version; do not assume an old unweighted mean is an area-weighted mean.
- If surface points were saved, the existing `python -m apbs_analysis.migrate_area_weighting STORE --dry-run` can assess migration. Without suitable saved points, migration may require regeneration.
- If you have verified that an unmarked store used the current area-weighted implementation, explicitly set `ASSUME_AREA_WEIGHTED=1` for the cluster wrapper, or pass `--assume-area-weighted` to the preparer. This assertion is recorded in the audit.

APBS background: [APBS software/methods paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC5734301/). The exact area-weighted sampling implementation is in `apbs_analysis/electrostatics.py` and described in `apbs_analysis/METHODS.md`.

### ESM alignment

The supplied examples contain `representations[0]`, `[32]`, and `[33]`. Layer 33 is the default: **241 × 1280** for 1acb.A and **63 × 1280** for 1acb.B. The preparer uses per-residue representations, never mean_representations. It verifies the embedding label against FASTA, sequence lengths (no BOS/EOS rows), chain sequences, and finite values. Only a full exact sequence or a unique contiguous fragment is accepted; internal gaps/mutations require embeddings regenerated for the corresponding PDB sequence. No positional guessing through ambiguous alignments occurs.

Default templates reproduce the examples:

```text
{target}_all.fasta
{target}.{chain}.pt
```

For graph-specific directories, set `FASTA_TEMPLATE` and `EMBEDDING_TEMPLATE`, for example:

```bash
export FASTA_TEMPLATE='{target}/{graph}/{target}_all.fasta'
export EMBEDDING_TEMPLATE='{target}/{graph}/{target}.{chain}.pt'
```

Here `{chain}` refers to the embedding label suffix in the FASTA (usually A/B), not the original PDB chain ID. Source chains such as E/I are assigned one-to-one to FASTA records by exact sequence matching, regardless of chain order, and the mapping is recorded in provenance. Multiple assignments are accepted only if they yield identical embeddings for every node; otherwise preparation rejects the ambiguous mapping. Use the actual SS_embeds layout. Target-level embeddings are appropriate only when decoy chain sequences match those target sequences. Loaded chains are cached during preparation, avoiding repeated reads of the same large tensor files. Source file hashes, selected layer, and matched offsets are recorded. ESM features are layer-normalized and projected through a trainable 64-dimensional projection before entering the encoder. They are not reconstruction targets.

## 2. Freeze the target split and training protocol

The matrix preparer scans every surviving graph, checks all feature widths/finite values and interface masks, and verifies rSASA CSV coverage matches the prepared cohort exactly. Missing optional CSV entries cannot silently shrink one variant's dataset.

Use `PRIOR_SPLIT` to reuse a previous target split: paths are remapped by target name to the new prepared files, and coverage/disjointness are validated. Otherwise a target-disjoint split is generated once with seed 7 and validation/test fractions 0.15. As in the existing trainer, validation fraction applies to the targets remaining after test selection. Every seed/configuration loads this same manifest.

All runs use 50 epochs, batch size 16, LR 3e-4, cosine decay, 1,000 warmup steps, and seeds 7/17/27. These defaults match the current seed study. Checkpoints are selected using validation DockQ MSE; test evaluation occurs after checkpoint selection. All autoencoder runs now use the original fixed loss coefficients node/edge/link/target = 1/1/0.1/1 (`--loss-weight-mode fixed`). There is no adaptive loss reweighting. These coefficients are not percentage contributions; raw term scales still affect the total. Supervised controls use 0/0/0/1, so reconstruction heads receive no training signal. Reconstruction scores from these supervised controls are not meaningful.

## 3. Run the 57-job study or 19-job first pass

| Family | Configurations | Seeds | Jobs |
| --- | ---: | ---: | ---: |
| GAT factorial: APBS none/plain/plain+area × pooling all/interface × ESM off/on | 12 | 3 | 36 |
| EGNN with reconstruction: baseline inputs / combined plain APBS+interface+ESM | 2 | 3 | 6 |
| EGNN supervised: baseline / combined | 2 | 3 | 6 |
| GAT supervised controls: baseline / combined | 2 | 3 | 6 |
| Original seed-replicate base: no Voronoi or interface-node degree | 1 | 3 | 3 |
| Total | 19 | 3 | 57 |

Interface pooling applies both mean and max to interface nodes only, with batch boundaries preserved. Message passing and node/edge reconstruction still operate on the full graph. Thus this tests readout focus while holding topology constant.

The EGNN follows scalar-message and equivariant-coordinate-update equations from [Satorras et al., ICML 2021](https://proceedings.mlr.press/v139/satorras21a.html), with bounded normalized coordinate updates. It uses the same contact edges and graph-level heads. Raw coordinates never enter scalar features or reconstruction targets. The supervised arm is the initial feed-forward EGNN experiment; the reconstruction arm adds the existing node/edge/link objectives. Width defaults are held fixed, not parameter counts; this is an initial architecture comparison, not a claim of equal compute.

On the cluster, from the repository:

```bash
# Override defaults here if your stores are elsewhere.
export APBS_DIR=/nfs/roberts/pi/pi_co54/jas485/ppi_gnn_data_store/apbs_model_data
export ESM_ROOT=/nfs/roberts/pi/pi_co54/nb685/scratch_backup/SS_embeds
# Optional: preserve an existing target split.
export PRIOR_SPLIT="$PWD/gate_run/voronoi_10pct_50epochs/target_splits.json"

# Set only after verifying an unmarked APBS store's generation method:
# export ASSUME_AREA_WEIGHTED=1

bash cluster/submit_model_extensions.sh full
```

For a first pass with **all 19 configurations and seed 7 only**, run:

```bash
bash cluster/submit_model_extensions.sh quick
```

Quick mode keeps the same 50 epochs by default, changing only seed count. To also shorten training, use `EPOCHS=10 bash cluster/submit_model_extensions.sh quick`. Full and quick modes use separate default output and prepared-data directories ending in `_full` and `_quick`. The original base is named `seed_replicate_base`; the full Voronoi control remains `baseline`. Both are retrained with fixed weights on the same 10% common cohort, so they are feature/architecture controls rather than reproductions of the older adaptive-loss runs.

Once quick preparation has finished, reuse its data and target split for the full study:

```bash
SKIP_PREP=1 \
EXTENSION_DATA="$PROJECT_DIR/extension_data_10pct_quick" \
PRIOR_SPLIT="$PROJECT_DIR/gate_run/model_extensions_10pct_quick/target_splits.json" \
bash cluster/submit_model_extensions.sh full
```

Set `PROJECT_DIR` in your shell to the checkout used for these experiments; adjust these paths if you overrode quick-mode defaults. `SKIP_PREP=1` reads the existing prepared data without rewriting it. The full study still trains all three seeds, including seed 7, in its own run directories. With one seed, seed standard deviation is reported as NaN (unavailable), not zero; target bootstrap intervals do not measure seed variability.

The submission chain is a one-CPU subset preflight → one nonempty target preparation → remaining per-target CPU preparation → cohort/matrix validation → 57-task GPU array (19 in quick mode) → comparison. Arrays have **no `%N` concurrency throttle**. Slurm still applies cluster resource availability and account limits. Each GPU task requests one GPU, 8 CPUs, and 80 GB RAM. CPU preparation, matrix validation, and comparison each request 1 CPU and 32 GB because their per-job Python loops are serial. Preparation runs concurrently across target-array tasks. Training requests 8 CPUs and defaults to 7 DataLoader worker processes plus the main process; its launcher caps workers at the actual Slurm allocation minus one. OMP, MKL, OpenBLAS, NumExpr, and Accelerate thread counts are set to one per process to avoid nested thread pools. The CPU/worker allocation is printed in each training log. This enables CPU parallelism but does not guarantee every CPU stays busy: HDF5 reads, filesystem throughput, and GPU compute can limit utilization. Existing environment activation follows the current training scripts (`py311_env` plus repository `venv`). Dependencies include PyTorch, PyG, NumPy, h5py, SciPy, and Biopython.

Paths also configurable through `PROJECT_DIR`, `SUBSET_DIR`, `FEATURE_DATA`, `PDB_ROOT`, `EXTENSION_DATA`, `EXPERIMENT_DIR`, and `OPTIONAL_NODE_FEATURES_DIR`. Use a dedicated prepared-data directory and a new experiment directory. Do not regenerate its HDF5 inputs while training jobs are running.

To inspect one target before dispatching the whole experiment:

```bash
python prepare_model_extensions.py --target 1acb \
  --data "$SUBSET_DIR" --feature-data "$FEATURE_DATA" --output "$EXTENSION_DATA" \
  --pdb-root "$PDB_ROOT" --apbs-dir "$APBS_DIR" --esm-root "$ESM_ROOT"
```

The preflight writes `EXPERIMENT_DIR/targets.txt` for nonempty subset files and records all empty files in `excluded_targets.json`. These empty files supply no graphs to any variant. Unreadable files fail preflight; nonempty targets with no successfully prepared graphs still fail preparation. Prior target splits retain surviving target assignments and can omit only targets explicitly recorded as empty by preflight. The first nonempty target must prepare successfully before the rest of the CPU array is released. This dependency is a validation gate, not an array concurrency cap. Once preprocessing succeeds, `matrix.json` records every command, graph key, seed, configuration, and cohort count. For a failed GPU task, rerun just its one-based matrix index:

```bash
sbatch --array=TASK_ID \
  --output="$EXPERIMENT_DIR/logs/retry_%A_%a.out" cluster/train_model_extensions.slurm
```

This restarts that training run; optimizer-state resume is not implemented. After any retries, run the comparison manually if its original afterok dependency did not complete:

```bash
python compare_model_extensions.py --matrix "$EXPERIMENT_DIR/matrix.json" \
  --output "$EXPERIMENT_DIR/comparison"
```

## 4. Compare matched targets, then confirm on the full dataset

`comparison/comparison.csv` reports target-macro DockQ MSE (primary), seed spread, per-target Spearman correlation, and top-1 DockQ regret. Undefined correlations are counted and excluded from the correlation mean. Top-1 ties use stable graph-name order. `paired_targets.csv` gives each target's improvement over baseline. The cluster comparison stage also writes `comparison_vs_seed_base/`, using the original base as the reference. For manual comparisons, pass `--reference seed_replicate_base` and a separate output directory.

The comparator requires all requested runs and exactly matching prediction keys/labels. It averages the matched seed differences within each target, then bootstraps targets to produce confidence intervals. Decoys and seed replicates are not treated as independent biological samples. Positive `mse_improvement` means lower error than baseline. Both ordinary 95% and Bonferroni-adjusted familywise bootstrap intervals are reported for the planned baseline comparisons. These intervals measure target sampling uncertainty conditional on the chosen seeds/split, not every source of training uncertainty.

Use the factorial cells to inspect whether gains persist with/without ESM, interface pooling, and the extra area descriptor. Compare EGNN reconstruction against GAT reconstruction, and EGNN supervised against its GAT supervised control; this separates an architecture change from removing auxiliary losses. Do not choose a winner from reconstruction loss alone.

Treat this 10% experiment as screening. Select a small number of candidates using validation behavior and effect consistency, then lock their settings and rerun against the full-feature baseline on the full dataset, ideally with at least five seeds and an untouched confirmation holdout. A target-disjoint split does not guarantee sequence-family separation: use a homology-grouped prior manifest if generalization across families is the intended claim. If the existing held-out targets have already informed model choices, report them as exploratory rather than a fresh final test.

## Local verification

`test_model_extensions.py` checks ESM example alignment, source-preserving preparation, APBS feature values and missing-data normalization, interface-only pooling, ESM gradient flow, rigid-motion equivariance/permutation behavior, and miniature CPU runs through training, checkpoint reloading, prediction export, and paired comparison. Run alongside the existing normalization/reconstruction tests:

```bash
python -m pytest -q test_model_extensions.py test_edge_feature_normalization.py \
  test_edge_recon_subset.py test_interface_node_degree.py
```

The cluster datasets and Slurm are not mounted/available in the local coding workspace. Local fixture tests verify execution and scientific invariants; they do not establish an improvement in model accuracy on the biological dataset.
