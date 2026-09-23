# Full-dataset ESM2 tuning

Every arm uses ESM2 layer 33. The sweep targets one question: **make validation DockQ
error fall with training instead of bottoming out in the first few epochs.**

```bash
STAGE=screen bash cluster/submit_full_esm_tuning.sh
```

Reuses the completed `full_pooling_validation` cohort, its model lists and the existing
`full_pooling_esm_sidecars`. Nothing is re-audited, no source graph is touched, and the
sidecars are not rebuilt. Set `SPLIT_MANIFEST` to use a different target split (see below),
`FULL_PARENT_DIR` for a parent elsewhere, and `EXPERIMENT_DIR` for the output, which must
not already exist.

## What the completed runs actually showed

Three measurements from `analyze_full_pooling_validation.ipynb` set the agenda. They are
descriptive findings from twelve control and six ESM2 runs, not established mechanisms.

**The DockQ term is a rounding error in the loss.** With `reconstruction_lambda=1`, the
training loss decomposes as edge-attribute reconstruction 88–91%, node reconstruction
4–7%, edge-presence BCE ~1%, and **DockQ MSE 3.1–4.9%**. The encoder is optimized almost
entirely to reconstruct `ca_dist` and `interface_edges`. Whatever else is wrong, the
objective currently asks mostly for something other than DockQ prediction.

**The models are close to constant predictors on validation.** Validation label variance
is 0.0944. The best control reaches 0.0727 and the ESM2 arm 0.0898 — about 23% and 5% of
variance explained. An oracle that predicted each target's own mean would score 0.0929, so
the useful signal is within-target, and there is very little of it.

**ESM2 was worse on pooled MSE but not uniformly.** It raised validation MSE in 6/6
matched pairs (+16.2%) while improving within-target Spearman in 38/60 paired target
observations. Per target the picture splits: on 1ugq, ESM2 gives MSE 0.0064 against the
control's 0.0634 with bias −0.008 against +0.212 — a genuinely better, calibrated fit.
On 2grn it collapses to a near-constant band and Spearman flips from +0.463 to −0.466.
The aggregate hid both.

At the baseline, projected ESM occupies **64 of the 88 encoder input columns**, so
structural features are outnumbered roughly three to one, and the ESM2 runs reach final
training MSE about three times lower than the controls. That is the motivation for the
sequence-conditioning arms rather than a diagnosis.

## Arms

Thirteen arms, each exactly one declared change from the ESM2 configuration that ran in
`full_pooling_esm_validation` (combined pooling, α=0.75, λ=1, dropout 0.3, LR 3e-4,
projection 64). `full_esm_tuning.py` asserts the one-change property when it builds the
matrix, so an arm cannot silently drift.

| Group | Arm | Change | Why |
|---|---|---|---|
| Reference | `baseline` | — | The run every delta is measured against |
| | `alpha0.5` | exponent 0.5 | α=0.5 won the control comparison; untested with ESM2 |
| Sequence conditioning | `esm_dim32`, `esm_dim16` | projection 32, 16 | Cut the sequence share of the encoder input |
| | `esm_scale0.25` | scale 0.25 | Same width, smaller magnitude |
| | `esm_gate` | learnable scale from 0.5 | Let the model choose its own reliance on sequence |
| | `esm_dropout0.3` | dropout on the ESM block | Regularize sequence only, leave structure intact |
| Objective balance | `lambda0.3`, `lambda0.1`, `lambda0.03` | reconstruction λ | Raise DockQ from ~3% of the loss to roughly 10%, 25%, 50% |
| Optimization | `lr1e-4` | LR 1e-4 | Baseline minima arrive before warmup ends |
| | `dropout0.5` | encoder dropout 0.5 | Stronger regularization |
| | `weight_decay1e-3` | weight decay 1e-3 | Hundredfold increase |

`reconstruction_lambda` scales all three reconstruction terms and leaves `target_weight`
at 1, so it is the direct lever on the 3% figure. The λ arms are the ones with a mechanism
behind them; the rest are plausible but speculative.

Run `STAGE=screen` first: thirteen arms at seed 7 only. One seed cannot separate a real
effect from seed noise — the twelve control runs had a seed SD of 0.0015–0.0038 against a
0.0047 spread across all four configurations — so treat screening as a filter, not a
result. Then confirm the survivors across three seeds:

```bash
STAGE=confirm ARMS="baseline lambda0.1 esm_dim16" bash cluster/submit_full_esm_tuning.sh
```

`STAGE=confirm` refuses to run without an explicit `ARMS` list.

## Validation and test DockQ RMSE

`train_gate.py` now records `train_target_rmse`, `val_target_rmse` and `test_target_rmse`
in `loss_history.csv` alongside the MSE columns, so DockQ error is readable in label units
without transforming the history. For reference, a constant predictor scores RMSE 0.307 on
the current validation split and the best completed run scores 0.270.

These runs omit `--no-test-evaluation` and `--validation-only-during-training`, so test
metrics are computed every epoch and `test_predictions.csv` is written at the end from the
selected checkpoint. **Checkpoint selection remains `val_target_mse`**; `train_gate.py`
only ever compares `val_<checkpoint-metric>`, and the matrix records
`selection_metric: val_target_mse`.

**This spends the test split.** Watching test RMSE across 13 or more runs and then picking
a configuration makes the test set a second validation set, whatever the code selects on —
the selection happens in the person reading the curves. After this sweep, the test numbers
are no longer an unbiased estimate of generalization, and a later paper-grade claim needs a
split that has not been looked at. Two ways to keep that option open, in order of
preference:

1. Carve a **development split** from the training targets and monitor that instead. It
   gives the same "is it going down?" signal at the cost of a few training targets, and
   leaves the test set sealed. This is the recommended path and needs only a new manifest.
2. Keep test monitoring, and treat every test number from this sweep as diagnostic. Declare
   a single final configuration, then evaluate it once on a split reserved in advance.

Per-epoch test evaluation also costs real time: 30,466 test graphs against 11,325
validation graphs, so expect the epoch time to roughly triple.

## Repairing the cohort first

Twenty-one targets contribute no models because their graphs lack the derived
`interface_node_degree` array, which is cheap to recompute from data already in the files.
The pipeline below repairs them, re-audits, rebuilds the cohort on a chosen split, rebuilds
the ESM2 sidecars and then launches this sweep, with `afterok` dependencies so a failure
stops everything before any GPU time is spent:

```bash
CONFIRM_SOURCE_MUTATION=yes bash cluster/submit_repair_and_retune.sh
```

| Stage | Array | What it does |
|---|---|---|
| `inventory` | serial | Writes `paths.json` |
| `repair` | 1–146 | Writes `interface_node_degree` and nothing else; verifies each write by recomputation |
| `repair_summary` | serial | Aggregates and fails the pipeline if any target failed |
| `audit` | 1–146 | Full re-audit; independently recomputes the degree and rejects disagreement |
| `build` | serial | Rebuilds cohort, split manifest and model lists from the fresh audit |
| `prepare` | 1–146 | Rebuilds ESM2 sidecars for the expanded cohort |
| `tune` | serial | Submits this sweep against the repaired parent |

Expected outcome: **188,466 of 189,710 raw models eligible**, up from 150,938. The 1,243
models missing their `node_features` group entirely and the one missing rSASA are not
repairable this way and stay out of the cohort. `MIN_ELIGIBLE_MODELS` defaults to 185,000 —
low enough to tolerate a small explained shortfall, high enough that a silently failed
repair blocks the sweep.

Two cautions. First, the repair **writes into the source graphs**, so every cached audit
keyed to the old file signatures goes stale and the completed pooling and ESM2 experiments
can no longer be rebuilt from unaltered sources. The launcher refuses to run until
`CONFIRM_SOURCE_MUTATION=yes` is set; snapshot `DATA_DIR` first, and use `REPAIR_DRY_RUN=1`
to see what would change without writing. Second, the repair is idempotent and treats a
present-but-stale degree array the same as a missing one, comparing values rather than
shape, because the audit rejects any array that disagrees with its own contacts.

Sidecars must be rebuilt, not reused: the existing ones were prepared against the
pre-repair cohort and are empty for exactly these 21 targets. `full_esm_tuning.py build`
checks each split target's sidecar audit and refuses a sweep whose sidecars do not cover
the cohort, so this ordering is enforced rather than merely documented.

## Changing the target split

`SPLIT_MANIFEST` accepts any manifest in the `target_splits.json` format. The builder
checks that the three splits are non-empty and disjoint and that **every** target in the
manifest has an ESM sidecar, refusing otherwise rather than shrinking the cohort.

The manifest at `target_splits.json` (seed 5; 105 train / 12 validation / 29 test targets,
covering all 146) is the intended split for these runs. Its recorded sample counts —
136,684 / 15,155 / 36,628, totalling 188,467 — presuppose the repair above: that total is
exactly the raw pool minus the 1,243 models with no `node_features` group. It is one model
above what the repair can actually deliver, because a single model in 3u82 also lacks
rSASA. The `build` stage recomputes every count from the fresh audit and writes a corrected
manifest, keeping the target assignment and fixing the bookkeeping, so use the manifest it
emits rather than the hand-written one.

A larger, difficulty-stratified validation set is a real methodological improvement: ten
validation targets is too few, and several of them are pathological — on the current split
the model produces a near-constant band for 1us7 and 1zhi, where the fitted
predicted-on-observed slope is about zero and Spearman is negative for both arms. Across
the 25 test targets that slope tracks Spearman at r = 0.92, so "hard" here means the model
collapses to a per-target constant, not that the labels are degenerate: every test target
spans at least 0.83 of the DockQ range.

One caution. Reshuffling until validation looks better is selection on the split and
inflates every number that follows. A stratified split is defensible if it is **declared
once, in advance, on a stated difficulty criterion, and then left alone**. Changing the
split also breaks comparability with the twelve control and six ESM2 runs, which used the
frozen split; anything compared across the two is not a matched comparison.

## Beyond this sweep

If none of the thirteen arms produces a falling validation curve, the remaining
explanations are structural rather than hyperparameter, and each needs its own experiment:

- **Per-target calibration.** The failure mode is a constant prediction per target, so a
  ranking or pairwise loss within a target would optimize what the panels show the model
  can sometimes do, even where absolute DockQ is hopeless.
- **Decoupled objectives.** Pretrain the autoencoder, freeze or strongly down-weight
  reconstruction, then fit the DockQ head. The λ arms are a cheap approximation of this.
- **Target-level generalization.** 90 training targets may simply be too few for the model
  to generalize to unseen complexes, which no amount of per-graph data fixes. A
  leave-targets-out curve — accuracy against number of training targets — would answer
  whether more targets or a better model is the binding constraint.

## Outputs

`gate_run/full_esm_tuning/` with `matrix.json`, one directory per run, and `logs/`. Each
run directory holds `loss_history.csv` (now with RMSE columns), `validation_predictions.csv`,
`test_predictions.csv`, `prediction_metrics.json`, `dockq_epoch_metrics.jsonl` and
`gate_model.pt`. Resources match the existing training jobs: one GPU, eight CPUs, 80 GB,
6d23h, one main process plus seven DataLoader workers.

```bash
python full_esm_tuning.py verify --output gate_run/full_esm_tuning
```

`verify` reports completion per arm and reads no test metric, so a partial sweep can be
checked without looking at test results.
