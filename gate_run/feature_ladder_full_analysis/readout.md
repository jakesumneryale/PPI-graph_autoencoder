| rung | test MSE | $\langle\rho\rangle_t$ | top-1 DockQ | AA acc. (%) | edge $F_1$ (%) |
|---|---|---|---|---|---|
| core | 0.1070 ± 0.0054 | 0.553 ± 0.010 | 0.319 | 100.0 | 74.2 |
| degree | 0.0901 ± 0.0144 | 0.599 ± 0.009 | 0.346 | 97.7 | 80.5 |
| rsasa | 0.0784 ± 0.0020 | 0.621 ± 0.004 | 0.428 | 87.2 | 79.4 |
| voronoi | 0.0752 ± 0.0081 | 0.633 ± 0.008 | 0.424 | 84.2 | 75.7 |
| electrostatics | 0.0808 ± 0.0097 | 0.644 ± 0.029 | 0.454 | 86.9 | 78.5 |

- Constant-predictor MSE is **0.0967**; random/oracle top-1 DockQ **0.334 / 0.905**.
- Best mean MSE: **voronoi**; best mean $\langle\rho\rangle_t$: **electrostatics**.
- Paired $\Delta\langle\rho\rangle_t$ per step:
  - core → degree: +0.047 [-0.009, +0.105], 18/29 targets
  - degree → rSASA$_i$: +0.021 [-0.042, +0.069], 18/29 targets
  - rSASA$_i$ → Voronoi: +0.012 [-0.006, +0.034], 17/29 targets
  - Voronoi → $V_{es}$: +0.011 [-0.017, +0.037], 20/29 targets
  - core → full: +0.091 [+0.004, +0.175], 24/29 targets
- Selected epochs: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2] -- every checkpoint is from epoch 1 or 2.
