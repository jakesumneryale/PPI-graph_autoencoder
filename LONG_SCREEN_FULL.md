# 40-epoch screen: long-run stability and high-DockQ accuracy (full dataset)

After syncing the code (`train_gate.py`, `dockq_objectives.py`, `anti_memorization.py`,
`test_anti_memorization.py`, `cluster/train_gate_long_screen_full.slurm`,
`cluster/submit_long_screen_full.sh`), run on Bouchet from the repository:

```bash
bash cluster/submit_long_screen_full.sh
```

21 GPU array tasks (7 configurations × seeds 7/17/27), no concurrency cap.
Results: `gate_run/long_screen_full/<config>_seed<seed>/`; logs in
`gate_run/long_screen_full/logs/`. Existing directories and runs are never overwritten.

## Questions

1. **Can validation DockQ MSE keep falling over 40 epochs?** The cause ablation
   identified fixed loss weights, dropout 0.3 and range weighting as what keeps
   validation from rising, all by limiting how tightly the training targets are fit.
   The control (A) still reaches its best near epoch 9 and is then roughly flat as
   its 20-epoch learning rate anneals. The first three levers below limit training
   fit further.
2. **Can high-DockQ decoys be predicted accurately?** At A's best checkpoint, truly
   high decoys (DockQ ≥ 0.80) are predicted 0.54 on average and only ~5% of them
   ≥ 0.80. The model is calibrated (true DockQ averages ≈ prediction in every
   prediction bin) but cannot yet tell high from medium decoys, so the last three
   levers target that distinction.

## Configurations

All are A (fixed weights, reconstruction λ 1, dropout 0.3, capped inverse-sqrt
range weighting, combined pooling, batch 16, LR 3e-4, 1,000 warmup steps) trained
for **40 epochs with the cosine schedule spread over all 40**, changing one thing:

| config | change | targets |
|---|---|---|
| control | none | reference at 40 epochs |
| lambda3 | reconstruction λ = 3 | less DockQ pull on the encoder late in training |
| wd3e-2 | AdamW weight decay 3e-2 (was 1e-5) | limits training fit |
| dropout05 | dropout 0.5 | limits training fit (dropout 0.3 was a main factor) |
| hinge_max_b5 | DockQ weight 1 below 0.23, rising linearly to β = 5 at DockQ 1, evaluated at max(true, predicted), normalized to mean 1 over training labels; **replaces** range weighting | high-DockQ accuracy without rewarding inflated predictions |
| capri_head | + 4-class CAPRI head (incorrect / acceptable / medium / high, edges 0.23 / 0.49 / 0.80), softmax cross-entropy, weight 0.2; DockQ MSE kept | richer high-DockQ signal; gives P(high) per decoy |
| rank | + within-target pairwise ranking loss, weight 0.1, temperature 0.1, pairs where at least one decoy ≥ 0.23, weighted by DockQ difference; batches of 4 targets × 4 decoys | encoder learns to separate good from very good decoys |

The `rank` arm necessarily changes batch composition (grouped batches, still every
decoy once per epoch); its effect is that of the loss plus grouped batching. The
γ = 0.2 and δ = 0.1 weights are first estimates chosen so each extra term's
gradient is comparable to the DockQ MSE's; they have not been tuned.

## Measurement and outputs

As before: validation every 2,000 steps and at epoch ends, exact graph-average
validation DockQ MSE for checkpointing, no test evaluation. New outputs:

- `capri_head`: `validation_predictions.csv` gains `p_incorrect, p_acceptable,
  p_medium, p_high`; `loss_history.csv` gains `train_/val_capri_ce` and
  `train_/val_capri_acc`; the checkpoint stores `capri_head_state_dict` (the GATE
  model state itself is unchanged).
- `rank`: `loss_history.csv` gains `train_rank_loss` and `train_rank_pairs` (mean
  pairs per batch).
- `hinge_max_b5`: `dockq_training_distribution.json` gains a `hinge` block with β,
  start and the normalizer.

## How to read it

- **Long-run stability:** validation slope over epochs 20–40 (≤ 0 is the goal),
  epoch of the best checkpoint (later is better), best and final validation MSE.
- **High DockQ** at the best checkpoint: per-CAPRI-class MSE and bias; share of
  true high decoys predicted ≥ 0.80; share of decoys predicted ≥ 0.80 that are
  truly high (false highs); CAPRI class accuracy; top-1 and within-target ρ; for
  `capri_head`, how reliable P(high) is.
- Selection stays on plain validation MSE so arms are comparable; high-DockQ
  accuracy is reported as a co-primary outcome.

Three seeds and 12 validation targets; treat small differences as ties. The test
split is untouched.

## Resources

One GPU, 8 CPUs (main process + 7 non-persistent spawn loader workers, one
thread each), 80 GB, 3.5-day walltime. 20-epoch runs of this recipe logged about
74 min per epoch including step validations, so 40 epochs is about 50 hours.
No checkpoint/resume is needed at this length.

## Local verification

- `test_anti_memorization.py` now has 37 tests, including: hinge shape and
  mean-one normalization; hinge-max penalizing confident false highs with the
  weight detached from the gradient; CAPRI class thresholds; ranking loss using
  only same-target pairs above 0.23 and preferring the correct order; grouped
  batches covering every decoy once with at most 4 targets per batch; end-to-end
  runs of each new arm (predictions match the best validation, CAPRI probabilities
  sum to 1, history columns present); argument guards. 128 tests pass across the
  training suites.
- All 21 Slurm tasks dry-run with stub commands; each argument list parses in
  `train_gate.py`.
- Real-data CUDA smoke runs of `hinge_max_b5`, `capri_head` and `rank` on the
  local 10% subset (3,000 graphs, 2 epochs): `local_data_audit/long_screen_smoke/`.
  Software check only.
