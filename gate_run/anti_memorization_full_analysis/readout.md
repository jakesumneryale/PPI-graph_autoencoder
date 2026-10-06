| arm | best val MSE | final val MSE | best epoch | ⟨ρ⟩ₜ | top-1 |
|---|---|---|---|---|---|
| A control | 0.0485 ± 0.0015 | 0.0537 ± 0.0044 | 9.3 ± 2.1 | 0.712 ± 0.013 | 0.524 ± 0.035 |
| B detached | 0.0570 ± 0.0032 | 0.0607 ± 0.0045 | 10.0 ± 7.0 | 0.670 ± 0.034 | 0.412 ± 0.052 |
| C adversary | 0.0516 ± 0.0028 | 0.0987 ± 0.0004 | 3.0 ± 0.0 | 0.687 ± 0.013 | 0.438 ± 0.021 |
| D augment | 0.0504 ± 0.0029 | 0.0611 ± 0.0049 | 5.0 ± 1.7 | 0.705 ± 0.014 | 0.478 ± 0.122 |
| E balanced+EMA | 0.0524 ± 0.0003 | 0.0539 ± 0.0020 | 12.7 ± 8.7 | 0.685 ± 0.018 | 0.452 ± 0.060 |
| F combined | 0.0530 ± 0.0014 | 0.0652 ± 0.0206 | 13.3 ± 5.9 | 0.686 ± 0.003 | 0.519 ± 0.066 |

- Constant predictor: **0.0903**. Feature ladder (same features/split, old recipe): best
  **0.0716**, final **0.1305**.
- Lowest mean best validation MSE: **A control**. Paired by target, no arm beats the control on MSE:
  - B detached: Δ = -0.0061 [-0.0151, +0.0041], better on 3/12 targets
  - C adversary: Δ = -0.0017 [-0.0098, +0.0069], better on 4/12 targets
  - D augment: Δ = -0.0013 [-0.0061, +0.0031], better on 4/12 targets
  - E balanced+EMA: Δ = -0.0022 [-0.0088, +0.0059], better on 3/12 targets
  - F combined: Δ = -0.0028 [-0.0123, +0.0066], better on 6/12 targets