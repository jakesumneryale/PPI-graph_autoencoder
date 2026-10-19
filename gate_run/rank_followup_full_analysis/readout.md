| config | val MSE | test MSE | val ⟨ρ⟩ | test ⟨ρ⟩ | test top-1 | test high recall / precision | test high bias |
|---|---|---|---|---|---|---|---|
| rank | 0.0470 ± 0.0006 | 0.0586 ± 0.0024 | 0.728 ± 0.012 | 0.686 ± 0.014 | 0.447 ± 0.020 | 9.2% / 49% | -0.298 ± 0.001 |
| rank_capri | 0.0455 ± 0.0033 | 0.0599 ± 0.0010 | 0.736 ± 0.013 | 0.676 ± 0.005 | 0.469 ± 0.027 | 7.0% / 45% | -0.307 ± 0.026 |
| rank_lambda3 | 0.0494 ± 0.0019 | 0.0606 ± 0.0018 | 0.710 ± 0.021 | 0.659 ± 0.019 | 0.407 ± 0.016 | 4.6% / 45% | -0.314 ± 0.040 |
| rank_lr1e-4 | 0.0460 ± 0.0010 | 0.0604 ± 0.0013 | 0.724 ± 0.005 | 0.661 ± 0.008 | 0.435 ± 0.006 | 12.7% / 49% | -0.263 ± 0.015 |

Test references: constant-predictor MSE 0.0967; random / oracle top-1 0.334 / 0.905.

Same test targets, earlier recipe (seed mean):
- Ladder, core: MSE 0.1070, ⟨ρ⟩ 0.553, top-1 0.319, high bias -0.538
- Ladder, all features: MSE 0.0808, ⟨ρ⟩ 0.644, top-1 0.454, high bias -0.401
- Rank (this experiment): MSE 0.0586, ⟨ρ⟩ 0.686, top-1 0.447, high bias -0.298

Paired vs rank on the 29 test targets (95% bootstrap CI; positive favours the configuration):
- rank_capri: −ΔMSE -0.0013 [-0.0050, +0.0022]; Δρ -0.010 [-0.031, +0.009]
- rank_lambda3: −ΔMSE -0.0017 [-0.0047, +0.0011]; Δρ -0.026 [-0.042, -0.011]
- rank_lr1e-4: −ΔMSE -0.0022 [-0.0053, +0.0009]; Δρ -0.025 [-0.043, -0.008]