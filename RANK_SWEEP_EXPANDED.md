# Ranking-loss sweep on the expanded dataset

Does a stronger within-target ranking signal improve the rank model on the expanded dataset? Fifteen GPU runs:
five arms × seeds 7/17/27, 10 epochs each, all on exactly the data, split (stratified fold 1) and model lists of
the dataset-size comparison's `expanded` arm.

| arm | ranking-loss weight | batch composition (16 graphs) |
|---|---|---|
| `control` | 0.1 | 4 targets × 4 decoys (current recipe) |
| `rank0.3` | 0.3 | 4 × 4 |
| `rank1.0` | 1.0 | 4 × 4 |
| `rank0.3_8decoys` | 0.3 | 2 targets × 8 decoys |
| `interface_pool` | 0.1 | 4 × 4, and `--pooling interface` (added later; compare with `control`) |

Everything else is `configs/gate_rank.toml`, plus `--val-every-steps 6000` as in the expanded arm.

## Why this sweep

From the expanded arm's final results (`gate_run/analyze_dataset_size_comparison.ipynb`) and earlier sweeps:

- 81% of the test DockQ MSE is within-target error (ranking and spread), 19% per-target offset.
- The model is already calibrated: a linear recalibration fitted on validation has slope ~1 and does not lower test
  MSE, so the compressed predictions are shrinkage under uncertainty. Output tweaks cannot fix it; better
  within-target discrimination can.
- Training DockQ MSE keeps falling after epoch ~5 while validation is flat; the wider model overfits more.
- Already tested without benefit: weight decay 3e-2, dropout 0.5, reconstruction λ = 3 (and λ < 1 earlier), hinge
  loss, CAPRI head, learning rate 1e-4, target adversary, augmentation, EMA, detached head, width 128/64.
- The ranking loss was the one recipe change that improved ⟨ρ⟩ (+0.027 in the long screen), but only weight 0.1
  was ever run. It is ~1.8% of the training loss (DockQ MSE 3.8%, reconstruction ~94%), and with 4 decoys per
  target a batch has at most 24 within-target pairs (~15 usable per step). 2 × 8 raises that to at most 56 and
  includes more close-quality pairs, the comparisons that decide top-1 selection.

## The interface-pooling arm

Added after the first four arms were running. On the expanded model's hardest test targets (1kac, 4hwi, 1he8, 9wlp)
predictions collapse to a flat, low value: whole targets are under-predicted by 0.13--0.21 DockQ although 36--54\% of
their decoys are acceptable or better. Across all 61 test targets that target-level under-prediction tracks the
interface's share of the complex (Spearman $r = +0.53$, $p = 10^{-5}$) and not the targets' actual quality
($r = -0.10$). One explanation: combined pooling includes a whole-graph mean, and the mean of `interface_nodes` over
all residues is the interface share, so a large complex with a small true binding patch looks "incorrect" in every
pose, and the near-identical whole-graph summary flattens differences between its decoys. `interface_pool` removes the
whole-graph half (the DockQ path pools interface residues only); everything else equals `control`. Check the whole-target
offset against interface share, and those four targets, besides the overall metrics. Submit it alone:

```bash
RUNS="interface_pool" bash cluster/submit_rank_sweep_expanded.sh
```

## 10 epochs

The cosine schedule runs over the 10 epochs. In the 20-epoch runs every best validation checkpoint came by epoch 12
and the second ten epochs changed no test metric materially, so this halves the cost (~33 h of training per run)
without dropping the useful part of training. The control is rerun at 10 epochs so every arm shares the schedule.

## Selecting the winner

Choose on **validation** (validation DockQ MSE and ⟨ρ⟩, top-1) only: this fold's test targets have already been
looked at. The final test evaluation still runs and is written as usual, for comparability, but should not be used
to pick the arm. Confirm the chosen arm once on another fold (e.g. `expanded_target_splits/stratified_fold2.json`),
whose test set is untouched.

## Running

On Bouchet, from the repository (after syncing `cluster/submit_rank_sweep_expanded.sh` and this file):

```bash
DRY_RUN=1 bash cluster/submit_rank_sweep_expanded.sh
bash cluster/submit_rank_sweep_expanded.sh
```

`RUNS="rank1.0"` submits a subset; an existing run directory is refused. Results:
`gate_run/rank_sweep_expanded/<arm>/gate_rank_seed<seed>/`. Resources as before: one GPU, 8 CPUs (main process +
7 loader workers), 80 GB, 3-day walltime.

## Local verification

- Every arm's options resolve against `gate_rank.toml` (10 epochs, cosine over 10, rank weight, targets per batch,
  pooling).
- CPU training run with `--pooling interface` finishes and evaluates on test; its checkpoint's graph projector takes
  2 × latent inputs (interface mean + max), not 4 × latent as with combined pooling.
- CPU training runs with 4 × 2 and 2 × 4 grouped batches finish, train with the ranking loss and evaluate on test.
- The grouped sampler uses every training decoy once per epoch in chunks of distinct decoys from one target, so on
  real data (1,300–2,000 decoys per target) both batch compositions give ~25,700 steps per epoch.
- Submit script against a mock tree with a stub `sbatch`: four arrays with the intended options; refuses an
  existing run directory and a broken data link. On Bouchet: 350 intact data links, the staged manifest and 350
  model lists are present.
