# Dataset-size comparison: original 146 vs expanded 350 targets

Does training on the expanded dataset (146 original + 204 target-expansion batch-1 targets) improve the rank model?
Nine GPU runs: the rank model (`configs/gate_rank.toml`) × seeds 7/17/27 × three arms that share the same
validation and test targets.

| arm | train | val | test |
|---|---|---|---|
| `expanded` | 254 (102 original + 152 batch 1, incl. 45 training-only) | 35 (20 original) | 61 (24 original) |
| `original_only` | 102 (the original targets of the same train split) | same 35 | same 61 |
| `expanded_large` | same 254 as `expanded` | same 35 | same 61 |

`expanded_large` is the rank model widened to `hidden_dim` 128 / `latent_dim` 64 (0.97 M parameters vs 0.26 M);
everything else is identical, so `expanded` vs `expanded_large` isolates model width on the larger dataset. Width
costs little time: on an RTX 4090, forward + backward for a 16-graph batch of real-sized graphs took 8.3 ms (64/32)
vs 15.5 ms (128/64), against ~560 ms per step measured on the cluster, where HDF5 data loading dominates.

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
5. `bash cluster/check_dataset_size_comparison.sh` (read-only: graphs, copy log, audit, original inputs, queue),
   then `bash cluster/submit_dataset_size_comparison.sh train` (try `DRY_RUN=1` first; `RUNS="expanded_large"`
   submits a subset): checks every graph, audit and model
   list, links the original targets' existing model lists, builds one symlink data directory per arm (train_gate.py
   requires the directory to hold exactly the manifest's targets) and submits three 3-task arrays of
   `cluster/train_gate_config.slurm`. Results: `gate_run/dataset_size_comparison/<arm>/gate_rank_seed<seed>/`.

## Results at epoch 10

Every run is submitted with `--snapshot-epochs 10`: after epoch 10, the checkpoint with the lowest validation DockQ
MSE so far is evaluated exactly as at the end (validation and test predictions, `prediction_metrics.json` with the
selected checkpoint) into `<run>/epoch10/`, together with a copy of that checkpoint and `snapshot.json`. It is
evaluated on copies of the model, and the random-number state is restored afterwards, so training continues to
epoch 20 exactly as it would have without the snapshot (tested). The reconstruction evaluation is final-only. The
expanded arms should reach it after ~2 days instead of ~4. Looking at test results at epoch 10 does not influence
training or checkpoint selection, but report the epoch-10 and epoch-20 numbers together.

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
