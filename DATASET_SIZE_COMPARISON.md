# Dataset-size comparison: original 146 vs expanded 350 targets

Does training on the expanded dataset (146 original + 204 target-expansion batch-1 targets) improve the rank model?
Six GPU runs: the rank model (`configs/gate_rank.toml`, unchanged) × seeds 7/17/27 × two arms that share the same
validation and test targets and differ only in training targets.

| arm | train | val | test |
|---|---|---|---|
| `expanded` | 254 (102 original + 152 batch 1, incl. 45 training-only) | 35 (20 original) | 61 (24 original) |
| `original_only` | 102 (the original targets of the same train split) | same 35 | same 61 |

The split is `gate_run/expanded_target_splits/stratified_fold1.json` (difficulty-stratified, homology-grouped).
Fold 1 rather than fold 0 because the three graphs that last failed on memory (3viq, 8g0p, 2b5u) are all in train
there; fold 0 has 3viq and 8g0p in test. Results are not comparable with the earlier runs on the 29-target test set.

## Steps

1. **Grace: v_es for the batch-1 graphs** (from the `cluster_pipeline` directory of the target expansion):
   `bash setup_graph_env.sh` (adds `pdb2pqr` to the existing env), then `bash submit_ves.sh`.
2. **Transfer** the batch-1 graphs from Grace to `/gpfs/boyd/pi/pi_co54/jas485/ppi_expansion_graphs/` (Bouchet), in
   two hops because the Bouchet Globus collection does not allow `/gpfs/boyd` yet:
   - `sbatch --partition=day --dependency=afterany:<v_es job> verify_ves.slurm` on Grace writes `VES_COMPLETE`;
     `transfer_graphs_to_bouchet.sh` (target expansion, local PC) waits for it and copies the graphs with Globus to
     `~/project_pi_co54/jas485/ppi_expansion_graphs/`.
   - `sbatch cluster/copy_expansion_graphs.slurm` on Bouchet copies them to `/gpfs/boyd` and checksum-compares.
   The original graphs stay where they are; nothing is copied into `ppi_processed_graphs`.
3. **Bouchet: sync** `train_gate.py` and friends, `audit_expansion_graphs.py`, `make_dataset_size_manifests.py`,
   `cluster/audit_expansion_graphs.slurm`, `cluster/submit_dataset_size_comparison.sh` and
   `gate_run/dataset_size_comparison/` (the two manifests and target lists).
4. `bash cluster/submit_dataset_size_comparison.sh audit`: one 1-CPU task per batch-1 graph, writing
   `gate_run/dataset_size_comparison/model_lists/<pdb>.txt` (models passing the full-pooling audit checks plus finite
   `rsasa_i_node` and `v_es`) and `audit/<pdb>.json` (rejections).
5. `bash cluster/submit_dataset_size_comparison.sh train` (try `DRY_RUN=1` first): checks every graph, audit and model
   list, links the original targets' existing model lists, builds one symlink data directory per arm (train_gate.py
   requires the directory to hold exactly the manifest's targets) and submits two 3-task arrays of
   `cluster/train_gate_config.slurm`. Results: `gate_run/dataset_size_comparison/<arm>/gate_rank_seed<seed>/`.

## Differences from the rank follow-up recipe

- **Validation interval.** Batch-1 targets have ~2,000 models each against ~1,300 for the originals, so the
  expanded arm has ~3× the training graphs per epoch. `--val-every-steps` is 6,000 there and 2,000 for
  `original_only`, so both validate the (larger, 35-target) validation set ~4× per epoch as the follow-up did.
- **Walltime** 6 days 23 h (GPU, 8 CPUs, 80 GB as before). Estimated ~4–5 h/epoch for `expanded` from the
  75 min/epoch measured on 136k training graphs; `original_only` should take about as long as the follow-up.
- Same 20 epochs in both arms, so `expanded` takes ~3× more optimizer steps. That is part of what "more data" means
  here; a steps-matched arm would be a separate question.

## Reading the result

Compare test DockQ MSE and mean within-target Spearman over the same 61 targets, and also separately on the 24
original and 37 batch-1 test targets: the batch-1 models were prepared with the new pipeline (GROMACS minimization),
so a gain only on batch-1 targets would mean the model learned the new pipeline, not more about interfaces.

## Local verification

- `v_es` port reproduces the original 1acb values exactly (15 decoys; end-to-end through `ves_features.py` on 20).
- Audit rejects a model without `v_es` and lists the rest.
- Submit script run against a mock tree with stub `sbatch`: refuses before the audit, stages 350 / 198 links, and
  refuses to reuse a finished run directory.
- `train_gate.py --prepare-splits-only` loads a manifest through a symlinked data directory and model lists
  (3 local graphs; CPU, no training).
