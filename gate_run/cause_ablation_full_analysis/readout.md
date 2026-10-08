Status: 26/27 ablation runs complete; common horizon 18 epochs.

| setting | Δ late MSE, added to L | Δ late MSE, kept in A |
|---|---|---|
| fixed loss weights | -0.0217 ± 0.0056 (n=3) | -0.0341 ± 0.0062 (n=3) |
| dropout 0.3 | -0.0185 ± 0.0058 (n=3) | -0.0201 ± 0.0084 (n=3) |
| 20-epoch schedule | +0.0025 ± 0.0043 (n=3) | -0.0032 ± 0.0051 (n=3) |
| range weighting | not tested | -0.0091 ± 0.0058 (n=3) |
| combined pooling | not tested | -0.0002 ± 0.0053 (n=3) |

| config | best | late | rise | best epoch |
|---|---|---|---|---|
| L_ladder | 0.0552 | 0.1144 | +0.0592 | 1.3 |
| L_plus_fixed | 0.0524 | 0.0927 | +0.0403 | 2.3 |
| L_plus_dropout | 0.0641 | 0.0958 | +0.0317 | 3.7 |
| L_plus_sched20 | 0.0582 | 0.1169 | +0.0587 | 1.0 |
| A_minus_fixed | 0.0512 | 0.0879 | +0.0367 | 1.0 |
| A_minus_dropout | 0.0515 | 0.0740 | +0.0225 | 2.0 |
| A_minus_sched | 0.0475 | 0.0570 | +0.0095 | 8.7 |
| A_minus_range | 0.0482 | 0.0630 | +0.0148 | 5.0 |
| A_minus_combined | 0.0498 | 0.0541 | +0.0042 | 7.3 |
| A_control | 0.0485 | 0.0539 | +0.0054 | 9.3 |