All 36 runs completed 50 epochs; 1132 validation graphs across 10 targets.

Best mean validation MSE: lambda=1, weighting=inverse-sqrt: 0.07339 +/- 0.00229 (seed SD).

Matched lambda=1 ordinary-MSE control: 0.07579 +/- 0.00262; relative improvement 3.17%.

Selected configuration improves MSE in 3/3 paired seeds and 8/10 targets (target MSE averaged across seeds).

Range weighting lowers mean validation MSE at 6/6 lambda values.

Selected mean final-epoch MSE is 0.08059; mean selected epoch is 19.7.

For true DockQ 0-0.2: mean observed 0.053, predicted 0.230. For 0.8-1.0: observed 0.864, predicted 0.441.

The score-range compression is not eliminated; inspect the bin errors and biases before claiming full-range accuracy.

This sweep does not support reducing reconstruction weight as a general fix. Preserve the lambda=1 range-weighted configuration as the leading candidate and confirm on additional target splits before adding complexity.

The 0–0.2 bin accounts for 92.7% of the net pooled-MSE improvement. The gain is therefore largely at low DockQ; the high-score underprediction remains essentially unresolved.
