# Diagnose unseen-target performance

The quick run shows early improvement followed by a widening training/validation
gap, not a general inability to reduce the training objective. For the full GAT,
validation DockQ MSE goes from 0.10484 to 0.07661 (epoch 5), then 0.11181 at
epoch 50; training DockQ MSE falls from 0.10399 to 0.02484. Interface pooling
reaches its validation minimum at epoch 16. ESM and combined models often select
epoch 1. Overfitting is a leading explanation; these curves alone do not identify
the causal feature or exclude subtler preprocessing issues.

Use raw validation DockQ MSE to assess the prediction objective. Total loss also
includes reconstruction and cannot be compared directly across AE/supervised
objectives. Existing exports already use validation-selected checkpoints, so
selecting epoch 5 again will not improve the saved baseline predictions.

## Local preliminary checks

Train and validation label distributions in the available local bundle are close:
mean 0.33389 / 0.33090 and SD 0.30873 / 0.30788. This does not rule out feature,
protein-family, or geometry distribution shift.

A training-mean constant gives validation MSE 0.09471, an rSASA-only ridge model
0.08709, and a seven-descriptor ridge model 0.07739. Descriptors are node/edge
counts, interface node/edge fractions, CA-distance mean/SD, and rSASA. Scaling is
fit on training data only, ridge alpha is fixed at 10, and the test set is unused.
These are preliminary results: the local bundle has 10,825/10,910 training and
1,110/1,132 validation graphs, because it previously excluded missing APBS
records. They are not an exact paired comparison with the full cluster cohort.
See `local_data_audit/generalization/simple_baselines.json` and `descriptors.csv`.

## Four controlled experiments

Reuse the completed full-feature GAT seed-7 baseline as the reference. Preserve
the original data, target split, objective, scheduler, and epoch budget.

| Diagnostic | Only factor changed |
| --- | --- |
| lower_lr | Learning rate 3e-4 → 3e-5 |
| smaller_encoder | Hidden width 64 → 32, latent width 32 → 16, heads 4 → 2 |
| more_dropout | Dropout 0.1 → 0.3 |
| more_weight_decay | AdamW weight decay 1e-5 → 1e-3 |

These are hypotheses, not guaranteed improvements. Evaluate best validation
MSE and its persistence through training; a slower learning curve alone is not
a success. If regularization/capacity helps, confirm across training seeds and
additional target-grouped validation splits before reopening test evaluation.
If it does not, prioritize feature/representation and objective diagnostics
rather than expanding the architecture search.

After syncing the changes, run on Bouchet:

```bash
bash cluster/submit_generalization_diagnostics.sh
```

This writes `gate_run/generalization_diagnostics_seed7/matrix.json`, reuses
existing prepared data, and submits four GPU jobs with seven loader workers
plus the main process (eight allocated CPUs; one numerical thread per process).
No array cap is applied. No old runs or inputs are modified. These jobs were
not submitted by the coding session.

`--no-test-evaluation` in `train_gate.py` disables both per-epoch and final test
evaluation and exports `validation_predictions.csv` from the best checkpoint.
Use those predictions for target-level validation diagnosis. Do not feed this
diagnostic matrix into the test-comparison script, which expects test exports.
Logs are in `gate_run/generalization_diagnostics_seed7/logs/`.
