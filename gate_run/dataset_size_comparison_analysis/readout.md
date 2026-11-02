Stage: **final**. Seed mean ± SD.

| arm | val MSE | test MSE | test ⟨ρ⟩ | ⟨ρ⟩ original, homolog | ⟨ρ⟩ original, no homolog | ⟨ρ⟩ new | test top-1 | selected epoch |
|---|---|---|---|---|---|---|---|---|
| Original only | 0.0494 ± 0.0005 | 0.0473 ± 0.0012 | 0.707 ± 0.003 | 0.639 ± 0.025 | 0.727 ± 0.005 | 0.712 ± 0.007 | 0.538 ± 0.010 | 11.0 ± 4.6 |
| Expanded | 0.0434 ± 0.0010 | 0.0397 ± 0.0011 | 0.760 ± 0.007 | 0.844 ± 0.020 | 0.750 ± 0.013 | 0.746 ± 0.002 | 0.599 ± 0.002 | 7.7 ± 4.0 |
| Expanded, wide | 0.0485 ± 0.0012 | 0.0426 ± 0.0037 | 0.735 ± 0.003 | 0.844 ± 0.019 | 0.748 ± 0.009 | 0.707 ± 0.012 | 0.553 ± 0.050 | 2.7 ± 2.1 |

References (all test targets): constant-predictor MSE 0.0908; random / oracle top-1 0.278 / 0.941.

Paired per-target effects (95% bootstrap CI over targets; positive favours the first arm):
- Expanded vs Original only, All 61 (n=61): Δρ +0.053 [+0.025, +0.086], −ΔMSE +0.0089 [+0.0042, +0.0141], Δtop-1 +0.061
- Expanded vs Original only, Original, homolog in train (n=8): Δρ +0.205 [+0.095, +0.359], −ΔMSE +0.0331 [+0.0132, +0.0551], Δtop-1 +0.164
- Expanded vs Original only, Original, no homolog (n=16): Δρ +0.023 [-0.030, +0.078], −ΔMSE +0.0028 [-0.0045, +0.0113], Δtop-1 +0.061
- Expanded vs Original only, New (batch 1) (n=37): Δρ +0.033 [+0.013, +0.055], −ΔMSE +0.0063 [+0.0023, +0.0105], Δtop-1 +0.039
- Expanded, wide vs Original only, All 61 (n=61): Δρ +0.029 [+0.000, +0.063], −ΔMSE +0.0061 [+0.0011, +0.0114], Δtop-1 +0.015
- Expanded, wide vs Original only, Original, homolog in train (n=8): Δρ +0.205 [+0.086, +0.366], −ΔMSE +0.0303 [+0.0103, +0.0534], Δtop-1 +0.213
- Expanded, wide vs Original only, Original, no homolog (n=16): Δρ +0.021 [-0.010, +0.054], −ΔMSE +0.0010 [-0.0084, +0.0109], Δtop-1 +0.011
- Expanded, wide vs Original only, New (batch 1) (n=37): Δρ -0.006 [-0.033, +0.020], −ΔMSE +0.0031 [-0.0012, +0.0076], Δtop-1 -0.026
- Expanded, wide vs Expanded, All 61 (n=61): Δρ -0.024 [-0.043, -0.006], −ΔMSE -0.0027 [-0.0048, -0.0009], Δtop-1 -0.046
- Expanded, wide vs Expanded, Original, homolog in train (n=8): Δρ +0.000 [-0.028, +0.021], −ΔMSE -0.0028 [-0.0050, -0.0008], Δtop-1 +0.049
- Expanded, wide vs Expanded, Original, no homolog (n=16): Δρ -0.002 [-0.036, +0.043], −ΔMSE -0.0018 [-0.0076, +0.0035], Δtop-1 -0.050
- Expanded, wide vs Expanded, New (batch 1) (n=37): Δρ -0.039 [-0.063, -0.019], −ΔMSE -0.0031 [-0.0052, -0.0013], Δtop-1 -0.065

After epoch 10 (validation DockQ MSE improved on the epoch-10 best):
- Original only: epochs done 20–20; improved in 2/3 seeds; best epochs 15, 12, 6
- Expanded: epochs done 20–20; improved in 1/3 seeds; best epochs 12, 7, 4
- Expanded, wide: epochs done 20–20; improved in 0/3 seeds; best epochs 1, 2, 5