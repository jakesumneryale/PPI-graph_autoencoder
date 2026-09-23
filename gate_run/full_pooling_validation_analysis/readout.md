All 18 runs completed 50 epochs on the same 11,325 validation graphs across 10 targets, with identical labels and no test evaluation.

Best control configuration: combined (whole graph + interface) pooling at alpha=0.5, validation DockQ MSE 0.07065 +/- 0.00194 (seed SD).

The spread across all four configurations (0.00468) is comparable to the within-configuration seed SD (0.00148-0.00379). Combined pooling wins 5/6 matched pairs and is not parameter-matched (+4,096 parameters); the weighting exponent wins 2/6. Treat both as weak preferences.

Checkpoints are selected extremely early: median epoch 2, with 9/12 runs selecting epoch 5 or earlier. Validation MSE ends 16% above its minimum while training MSE falls to 0.0219.

Training past the checkpoint keeps improving the 0--0.2 range and steadily degrades everything above 0.4; the high-DockQ bias grows more negative throughout. Early stopping masks this trade rather than removing it.

ESM2 raises validation DockQ MSE in 6/6 matched pairs (+16.2% relative), while raising mean within-target Spearman correlation by +0.0235 in 4/6 pairs and cutting final training MSE by 3.1x. Better ranking with worse calibration.

ESM2 improves the two lowest DockQ ranges and worsens the three highest; the net pooled change (+0.01190) comes from the upper ranges, where the underprediction deepens from -0.411 to -0.529.

No test predictions exist for any of the 18 runs, so every rho above is validation-only on 10 targets. On the same frozen 25-target test split, the earlier full-data feature-ablation models reach macro rho 0.557-0.648 (median per-target rho 0.768), with the mean pulled down by 4 weak targets. Those are different models and do not license a test claim for the pooling or ESM2 arms.

On this evidence, do not adopt ESM2 for this objective as configured. The defensible next step is regularization and schedule, not more input features: these runs overfit within a handful of epochs on 109,147 training graphs.

All numbers are validation-only and come from the checkpoint-selection metric itself, so they are optimistic. Three seeds and ten validation targets do not establish significance, and the test split remains sealed.
