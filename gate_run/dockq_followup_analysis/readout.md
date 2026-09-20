All 36 jobs passed completion/cohort checks; three reused controls reproduced their validation predictions.

Best pooling: combined, alpha=0.75. Validation MSE 0.06967 +/- 0.00167 (seed SD), versus 0.07339 +/- 0.00229: 5.06% lower.

Improved in 3/3 paired seeds and 8/10 validation targets.

Macro within-target Spearman: 0.428 -> 0.487; macro-target MSE: 0.07159 -> 0.06762.

Previous best, high-DockQ bin: training observed 0.864, predicted 0.567; validation observed 0.864, predicted 0.441.

Selected pooling, high-DockQ validation bin: observed 0.864, predicted 0.461; MSE 0.1878 versus 0.2143.

Fine-tuning retained epoch zero in 9/9 runs. These runs completed 20 epochs; the fallback prevents reporting a worse checkpoint as an improvement.

Combined pooling is worth confirming, but its larger graph-projector input adds parameters, so this is not a pure parameter-matched pooling ablation.

No test predictions were read. Confirm on additional target-level validation splits before more extensive tuning of this same ten-target cohort.
