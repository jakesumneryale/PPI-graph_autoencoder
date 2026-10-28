Stage: **epoch10**. Seed mean ± SD.

| arm | val MSE | test MSE | test ⟨ρ⟩ | ⟨ρ⟩ original, homolog | ⟨ρ⟩ original, no homolog | ⟨ρ⟩ new | test top-1 | selected epoch |
|---|---|---|---|---|---|---|---|---|
| Original only | 0.0505 ± 0.0010 | 0.0459 ± 0.0008 | 0.706 ± 0.007 | 0.663 ± 0.029 | 0.718 ± 0.005 | 0.710 ± 0.007 | 0.535 ± 0.003 | 5.7 ± 0.6 |
| Expanded | 0.0434 ± 0.0011 | 0.0397 ± 0.0010 | 0.760 ± 0.007 | 0.845 ± 0.021 | 0.752 ± 0.016 | 0.745 ± 0.003 | 0.584 ± 0.025 | 7.0 ± 3.0 |
| Expanded, wide | 0.0485 ± 0.0012 | 0.0426 ± 0.0037 | 0.735 ± 0.003 | 0.844 ± 0.019 | 0.748 ± 0.009 | 0.707 ± 0.012 | 0.553 ± 0.050 | 2.7 ± 2.1 |

References (all test targets): constant-predictor MSE 0.0908; random / oracle top-1 0.278 / 0.941.

Paired per-target effects (95% bootstrap CI over targets; positive favours the first arm):
- Expanded vs Original only, All 61 (n=61): Δρ +0.054 [+0.024, +0.088], −ΔMSE +0.0073 [+0.0026, +0.0126], Δtop-1 +0.049
- Expanded vs Original only, Original, homolog in train (n=8): Δρ +0.182 [+0.080, +0.333], −ΔMSE +0.0320 [+0.0105, +0.0568], Δtop-1 +0.127
- Expanded vs Original only, Original, no homolog (n=16): Δρ +0.033 [-0.034, +0.104], −ΔMSE +0.0000 [-0.0065, +0.0067], Δtop-1 +0.089
- Expanded vs Original only, New (batch 1) (n=37): Δρ +0.035 [+0.012, +0.059], −ΔMSE +0.0051 [+0.0010, +0.0090], Δtop-1 +0.015
- Expanded, wide vs Original only, All 61 (n=61): Δρ +0.029 [-0.002, +0.064], −ΔMSE +0.0045 [-0.0003, +0.0096], Δtop-1 +0.018
- Expanded, wide vs Original only, Original, homolog in train (n=8): Δρ +0.181 [+0.071, +0.334], −ΔMSE +0.0285 [+0.0082, +0.0533], Δtop-1 +0.174
- Expanded, wide vs Original only, Original, no homolog (n=16): Δρ +0.029 [-0.015, +0.077], −ΔMSE -0.0014 [-0.0098, +0.0048], Δtop-1 +0.033
- Expanded, wide vs Original only, New (batch 1) (n=37): Δρ -0.003 [-0.036, +0.026], −ΔMSE +0.0019 [-0.0023, +0.0060], Δtop-1 -0.023
- Expanded, wide vs Expanded, All 61 (n=61): Δρ -0.024 [-0.043, -0.005], −ΔMSE -0.0028 [-0.0049, -0.0009], Δtop-1 -0.031
- Expanded, wide vs Expanded, Original, homolog in train (n=8): Δρ -0.001 [-0.027, +0.019], −ΔMSE -0.0035 [-0.0053, -0.0014], Δtop-1 +0.047
- Expanded, wide vs Expanded, Original, no homolog (n=16): Δρ -0.004 [-0.041, +0.045], −ΔMSE -0.0015 [-0.0077, +0.0042], Δtop-1 -0.055
- Expanded, wide vs Expanded, New (batch 1) (n=37): Δρ -0.038 [-0.062, -0.018], −ΔMSE -0.0032 [-0.0052, -0.0013], Δtop-1 -0.038

After epoch 10 (validation DockQ MSE improved on the epoch-10 best):
- Original only: epochs done 20–20; improved in 2/3 seeds; best epochs 15, 12, 6
- Expanded: epochs done 20–20; improved in 1/3 seeds; best epochs 12, 7, 4
- Expanded, wide: epochs done 16–19; improved in 0/3 seeds; best epochs 1, 2, 5