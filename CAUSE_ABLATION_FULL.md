# Cause ablation: why did validation DockQ MSE stop rising? (full dataset)

After syncing the code (`train_gate.py`, `test_anti_memorization.py`,
`cluster/train_gate_cause_ablation_full.slurm`, `cluster/submit_cause_ablation_full.sh`),
run on Bouchet from the repository:

```bash
bash cluster/submit_cause_ablation_full.sh
```

This submits 27 GPU array tasks (9 configurations × seeds 7/17/27), no
concurrency cap. Results go to `gate_run/cause_ablation_full/<config>_seed<seed>/`,
logs to `gate_run/cause_ablation_full/logs/`. Existing directories and
completed runs are never overwritten. The launcher requires the three
`gate_run/anti_memorization_full/A_control_seed*` runs, which serve as the
control reference and are not retrained.

## Question

With the same features, cohort and split, the feature ladder's validation DockQ
MSE was best at epoch 1–2 (0.072) and rose to 0.13, whereas the
anti-memorization control (A) was best at epochs 7–11 (0.0485) and ended at
0.054. A changed five training settings at once (plus measurement details), so
the cause is unidentified. Leading suspect: the adaptive loss shares, whose
DockQ weight climbed from ~14 to ~260 as the training DockQ loss shrank.

| setting | ladder (L) | control (A) |
|---|---|---|
| loss weighting | adaptive shares .15/.15/.10/.60 | fixed, reconstruction λ = 1 |
| DockQ range weighting | none | capped inverse-sqrt, α = 0.5 |
| pooling | whole graph | whole graph + interface |
| dropout | 0.1 | 0.3 |
| cosine schedule | 50 epochs | 20 epochs |

## Configurations

Each `L_plus_*` adds one control setting to the ladder recipe; each `A_minus_*`
removes one from the control. Agreement between the two directions is the
strongest evidence; disagreement indicates an interaction.

| config | weighting | range | pooling | dropout | schedule |
|---|---|---|---|---|---|
| L_ladder | shares | none | whole | 0.1 | 50 |
| L_plus_fixed | **fixed** | none | whole | 0.1 | 50 |
| L_plus_dropout | shares | none | whole | **0.3** | 50 |
| L_plus_sched20 | shares | none | whole | 0.1 | **20** |
| A_minus_fixed | **shares** | **none** | combined | 0.3 | 20 |
| A_minus_dropout | fixed | inv-sqrt | combined | **0.1** | 20 |
| A_minus_sched | fixed | inv-sqrt | combined | 0.3 | **50** |
| A_minus_range | fixed | **none** | combined | 0.3 | 20 |
| A_minus_combined | fixed | inv-sqrt | **whole** | 0.3 | 20 |
| A (reused) | fixed | inv-sqrt | combined | 0.3 | 20 |

Range weighting requires fixed weights in `train_gate.py`, so `A_minus_fixed`
also drops range weighting; `A_minus_range` isolates range weighting alone.
Pooling is tested from the control side only: whole vs combined pooling was
already a weak effect in the earlier full pooling validation.

**Shared:** 20 training epochs; LR 3e-4 with 1,000 warmup steps then cosine;
batch 16; four-layer residual GAT (hidden 64, latent 32, 4 heads); the
electrostatics-rung features, `full_pooling_repaired` model lists and frozen
split. A "50-epoch schedule" run trains the first 20 epochs of a 50-epoch cosine
(new `--lr-schedule-epochs`), which is exactly the learning rate the ladder had
over those epochs. The ladder's rise was clear by epoch 4–5, so 20 epochs
captures it at well under half the GPU cost.

**Identical measurement for every run,** including the ladder recipe: validation
every 2,000 steps and at each epoch end, exact graph-average validation DockQ MSE
(`--dockq-range-diagnostics`), best-step checkpointing, no test evaluation. The
original ladder measured once per epoch with batch-averaged MSE; `L_ladder`
reproduces its recipe under the new measurement, so it is the proper
comparison baseline, not the old ladder numbers.

## Reading the result

For each configuration: best validation MSE, end-of-training MSE, the
final/best ratio and the slope after the best step (does it rise?), plus the
selected epoch. A setting is implicated if adding it to L removes the rise
*and* removing it from A brings the rise back. Compare paired by seed and by
validation target. Three seeds and 12 validation targets; treat small
differences as ties.

## Code changes

- `--lr-schedule-epochs N`: cosine schedule length in epochs, independent of
  `--epochs` (default: equal). Requires `--lr-schedule cosine`.
- `--val-every-steps` now also works with `--loss-weight-mode shares` (it is
  measurement only); the training interventions still require fixed weights.
- Epoch log lines report wall-clock minutes per epoch.

## Resources

One GPU, 8 CPUs (main process + 7 non-persistent spawn loader workers, one
thread each), 80 GB, 3-day walltime per task. The same 20-epoch recipe finished
within a day in `gate_run/anti_memorization_full`.

## Local verification

- `test_anti_memorization.py` (23 tests) adds checks that the schedule length is
  decoupled from training length, that step validation works with adaptive
  shares (and that the balancer still adapts), and that the guards reject invalid
  combinations at argument parsing. 114 tests pass across the training suites.
- The Slurm script was dry-run for all 27 tasks: configuration/seed mapping and
  each argument list parse in `train_gate.py`.
- A real-data CUDA smoke run of the `L_ladder` recipe (shares + step validation +
  decoupled schedule; local 10% subset without `rsasa_i_node`/`v_es`, 3,000
  graphs, 2 epochs) completed; exported predictions match the best step
  validation, and the adaptive DockQ weight rose across epochs as in the ladder:
  `local_data_audit/cause_ablation_smoke/`. Software check only.
