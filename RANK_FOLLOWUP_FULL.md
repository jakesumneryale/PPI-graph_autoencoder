# Ranking-loss follow-up with test-set evaluation (full dataset)

After syncing the code (`train_gate.py`, `dockq_objectives.py`, `anti_memorization.py`,
`test_anti_memorization.py`, `cluster/train_gate_rank_followup_full.slurm`,
`cluster/submit_rank_followup_full.sh`), run on Bouchet from the repository:

```bash
bash cluster/submit_rank_followup_full.sh
```

12 GPU array tasks (4 configurations × seeds 7/17/27), 20 epochs each, no
concurrency cap. Results: `gate_run/rank_followup_full/<config>_seed<seed>/`;
logs in `gate_run/rank_followup_full/logs/`.

## Configurations

The within-target ranking loss from the 40-epoch screen (weight 0.1, temperature
0.1, pairs with a decoy ≥ 0.23, grouped batches of 4 targets × 4 decoys) is now
standard. Every configuration is the control recipe (fixed weights, dropout 0.3,
capped inverse-sqrt range weighting, combined pooling, cosine over 20 epochs) plus
the ranking loss, with one further change:

| config | change |
|---|---|
| rank | none: the new reference |
| rank_capri | + CAPRI-class head, cross-entropy weight 0.2 (same weight as the screen) |
| rank_lambda3 | reconstruction weight λ = 3 |
| rank_lr1e-4 | peak learning rate 1e-4 instead of 3e-4 |

Why the learning-rate arm: in the screen, validation MSE moved by 0.002–0.004 between
consecutive validations (2,000 steps apart), about as much as the differences between
configurations, and it settled as the learning rate decayed (0.0021 → 0.0012 in the
20-epoch control). Best checkpoints arrived while the learning rate was still near its
peak (2–2.7e-4). A lower peak should give smoother curves and a later, more reliable best
checkpoint; whether it also improves the best value is what the arm tests.

## Test-set evaluation

The test split (29 targets, 34,704 decoys) is sealed during training
(`--validation-only-during-training`). After training, the checkpoint with the lowest
validation DockQ MSE (from validations every 2,000 steps and at epoch ends) is reloaded
and evaluated once on the test targets:

- `test_predictions.csv` (with `p_incorrect … p_high` for `rank_capri`);
- `prediction_metrics.json` gains `test` (pooled, target-macro and per-DockQ-bin MSE and
  bias) and `selected_checkpoint` (epoch, step and validation MSE of the evaluated model);
- `reconstruction_summary.csv` / `reconstruction_by_target.csv` on the test targets.

`validation_predictions.csv` from the same checkpoint is exported as before. The
selection rule is fixed in advance and every run is evaluated, so the test numbers for
all configurations should be reported together, not only for the one that does best on
test.

## Code change

`train_gate.py` now exports validation predictions and metrics from the selected
checkpoint in every mode, and when test evaluation is enabled also writes the `test`
and `selected_checkpoint` blocks to `prediction_metrics.json` and logs the test DockQ
MSE. Previously, runs with test evaluation enabled wrote test predictions and the
reconstruction summary but no validation predictions or test DockQ metrics.

## Resources

One GPU, 8 CPUs (main process + 7 non-persistent spawn loader workers, one thread
each), 80 GB, 2-day walltime. Measured ~75 min/epoch on this recipe, so ~25 h of
training plus test prediction and reconstruction evaluation.

## Local verification

- New test: a run with the test split sealed, ranking loss and CAPRI head writes NaN
  test columns during training, exports validation predictions matching the best
  validation, evaluates the selected checkpoint on held-out test targets only, and
  writes consistent test metrics and the reconstruction summary. 129 tests pass across
  the training suites.
- All 12 Slurm tasks dry-run with stub commands; each argument list parses in
  `train_gate.py`.
- Real-data CUDA smoke run (local 10% subset, 3,000 graphs, 2 epochs, ranking loss +
  CAPRI head + test evaluation): `local_data_audit/rank_followup_smoke/`. Software check
  only.
