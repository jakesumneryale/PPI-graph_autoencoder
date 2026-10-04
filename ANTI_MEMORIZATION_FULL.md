# Anti-memorization experiment (full dataset)

After syncing the code (`train_gate.py`, `GATE_model.py`, `anti_memorization.py`,
`cluster/train_gate_anti_memorization_full.slurm`,
`cluster/submit_anti_memorization_full.sh`), run on Bouchet from the repository:

```bash
bash cluster/submit_anti_memorization_full.sh
```

This submits 18 GPU array tasks (6 arms × seeds 7/17/27) without a concurrency
cap. Results go to `gate_run/anti_memorization_full/<arm>_seed<seed>/`, logs to
`gate_run/anti_memorization_full/logs/`. Set `OUTPUT_ROOT` for a fresh
directory; existing directories and completed runs are never overwritten.

## Why

All 15 full-dataset feature-ladder runs selected epoch 1 or 2. Validation DockQ
MSE then rose from ~0.08 to ~0.14 (above the 0.097 constant-predictor MSE) while
training DockQ MSE fell to ~0.002. With ~150k decoys but only 105 training
targets, the leading hypothesis is per-target memorization: the network can
learn each training target's native contacts and score decoys by overlap with
them, which transfers to no unseen target. Supporting evidence: the selected
epoch shrinks as decoys per target per epoch grow (10% subset: epochs 5–20;
full: 1–2), and ESM2, which makes residues easier to identify, cut training MSE
3.1× while raising validation MSE in 6/6 matched pairs. Earlier attempts
(learning rate, capacity, dropout, weight decay, reconstruction weight, range
weighting, pooling, fine-tuning) used general regularization; none targeted
this shortcut.

## Arms

| Arm | Options | Mechanism |
|---|---|---|
| A_control | — | Shared objective below; the reference |
| B_detached | `--dockq-head-mode detached` | DockQ head (graph projector + quality head) trains on detached node embeddings; only reconstruction gradients reach the encoder, so it cannot fit DockQ labels |
| C_adversary | `--target-adversary-weight 0.1` | A 105-way training-target classifier on the graph embedding behind gradient reversal (DANN ramp 0→1 over training) pushes that embedding to be target-invariant |
| D_augment | `--edge-dropout 0.1 --node-feature-mask 0.1 --ca-dist-noise 0.25` | Denoising input corruption for the encoder only; reconstruction and DockQ targets are the clean graph. Contacts are dropped as undirected pairs; C-alpha noise is in Å, clamped at 0 |
| E_balanced_ema | `--decoys-per-target 256 --ema-decay 0.999` | 256 decoys per training target per epoch, redrawn each epoch; an exponential moving average of the weights is what gets validated and checkpointed |
| F_combined | C + D + E | Joint-training regularizers together. B is a different training paradigm and is not stacked here; combine it with the winners in a follow-up if it helps |

**Shared settings:** full electrostatics-rung features (node:
`aa_type, chain, interface_nodes, interface_node_degree, rsasa_i_node`; edge:
`interface_edges, ca_dist, voronoi_contact_area (log1p, z), voronoi_contact_missing, v_es (z)`;
reconstruction of `interface_edges, ca_dist`), the `full_pooling_repaired` model
lists and frozen split, four-layer residual GAT (hidden 64, latent 32, 4 heads),
combined whole-graph + interface pooling, dropout 0.3, batch 16, AdamW LR 3e-4
with 1,000 warmup steps then cosine decay, **20 epochs**. Objective:
`DockQ MSE + 1 × (node MSE + edge MSE + 0.1 × edge BCE)` with fixed weights and
capped inverse-sqrt DockQ range weighting (α 0.5). This replaces the ladder's
adaptive loss shares, whose DockQ weight climbed from ~14 to ~260 during training.

**Budgets differ by design.** E and F see 256 × 105 ≈ 26.9k graphs per epoch
(≈ 4.5 full-data passes over 20 epochs) against ≈ 120k per epoch for A–D. Since
every ladder run peaked within two full passes, the shorter budget still covers
that regime at finer resolution. Compare arms by best validation MSE and its
persistence, not by epoch number.

## Checkpointing and outputs

Every arm validates every 2,000 optimizer steps (`--val-every-steps`) and at
each epoch end, and keeps the best checkpoint by exact graph-average
validation DockQ MSE (unweighted, as in the DockQ sweeps). With EMA on, the
validated and saved weights are the EMA weights. No test predictions are made.

New per-run outputs, in addition to the usual `loss_history.csv`,
`dockq_epoch_metrics.jsonl`, `dockq_training_distribution.json`,
`validation_predictions.csv`, `prediction_metrics.json` and `gate_model.pt`:

- `validation_steps.csv`: every validation (step and epoch end) with pooled,
  target-macro and bin-macro validation MSE and whether it was saved. This is the
  curve to judge persistence with.
- `loss_history.csv` gains `train_target_adversary_{ce,acc,coefficient}` for
  adversary arms (chance accuracy 1/105 ≈ 0.0095). Its `val_*` columns are
  end-of-epoch values, from EMA weights in E/F.
- `gate_model.pt` records `best_step`, `ema_weights` and, for adversary arms, the
  adversary state and its target order.

## How to read the result

Primary: best validation DockQ MSE (mean ± seed SD) versus A_control, paired by
seed and by validation target. Secondary: final/best MSE ratio and the slope of
`validation_steps.csv` after the best step (does the curve still rise?),
high-DockQ bias, and within-target Spearman. For C, check the adversary accuracy:
near chance means the embedding became target-invariant; high accuracy means the
reversal was too weak. Then run the chosen protocol once on the sealed test split.

The 10-target validation set has informed earlier choices, and three seeds are
a small basis for inference; treat small differences as ties.

## Resources

One GPU, 8 CPUs, 80 GB, 4d12h walltime per task. One main process (one
PyTorch/BLAS thread) plus 7 spawn DataLoader workers; workers are not
persistent. During a mid-epoch validation the training pool is paused and idle
while the 7-worker validation pool runs, so at most 8 processes are busy. The
launcher's login-node checks only confirm that the synced code has the new
options; they read no data.

## Local verification

- `test_anti_memorization.py`: 18 tests covering gradient reversal, the DANN
  ramp, augmentation (input untouched, symmetric contact dropout, noise on
  `ca_dist` only), balanced sampling, EMA, zero encoder gradient from the DockQ
  loss in detached mode, and end-to-end `train_gate.py` runs of every arm
  checking that the exported validation predictions match the best step
  validation and `best_step`. The existing 91 training regression tests pass
  unchanged.
- Real-data CUDA smoke runs of all six arms on the local 10% subset (3,000 graphs,
  2 epochs, RTX 4090, 7 spawn workers):
  `local_data_audit/anti_memorization_smoke/`. Local graphs lack
  `rsasa_i_node`/`v_es`, so the smoke runs use the degree + Voronoi features;
  feature loading is unchanged code. These are software checks, not evidence of
  accuracy.
- The Slurm script was dry-run with stub commands for all 18 tasks' arm/seed
  mapping, and each arm's exact argument list parses in `train_gate.py`.
