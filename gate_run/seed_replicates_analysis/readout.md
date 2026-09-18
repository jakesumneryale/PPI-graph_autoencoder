**Full versus base, averaged over three seeds:**

- Target-macro MSE: **0.0795 → 0.0806** (-1.4% reduction).
- Paired MSE improvement: **-0.0011**, target-bootstrap 95% interval **[-0.0128, 0.0118]**.
- Within-target Spearman: **0.681 → 0.655**.
- Top-1 true DockQ: **0.377 → 0.439**; random single-selection reference **0.331**.
- Selected checkpoints: base seed 7: 1, base seed 17: 1, base seed 27: 1, full seed 7: 6, full seed 17: 1, full seed 27: 6.

The raw training losses and validation curves are more informative than the adaptive weighted total alone. Use the validation-selected predictions as saved; do not pick a later epoch or favorable seed based on this test analysis.
