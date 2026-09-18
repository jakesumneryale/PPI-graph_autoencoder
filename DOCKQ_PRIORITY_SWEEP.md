# DockQ-priority weekend sweep

Run on the cluster after syncing the updated code:

```bash
bash cluster/submit_dockq_priority_sweep.sh
```

This creates `gate_run/dockq_priority_sweep/matrix.json` and submits 36 GPU array
jobs without a concurrency cap. A serial preflight checks existing split files,
cohort counts, and training-label coverage before any GPU jobs are submitted. Existing directories are never overwritten.
For a new experiment directory, set `EXPERIMENT_DIR` before the command.

## Controlled experiment

- Reconstruction lambda: 0, 0.01, 0.03, 0.1, 0.3, 1.
- DockQ weighting: ordinary MSE or capped inverse-square-root training-bin frequency.
- Seeds: 7, 17, 27. The target split and edge-statistics seed remain fixed.
- Dropout 0.3, 50 epochs, LR 0.0003, existing cosine schedule and warmup.
- Existing APBS-free 10% dataset and structural inputs; no preprocessing rerun.
- Objective: `DockQ MSE + lambda * (node MSE + edge MSE + 0.1 * edge BCE)`.
  DockQ weighting changes only the training DockQ term. Lambda zero retains
  the graph inputs and message passing but removes reconstruction gradients.
- No adaptive loss-share balancing. All coefficients remain fixed.

Bins are [0,.2), [.2,.4), [.4,.6), [.6,.8), [.8,1]. Only training labels determine
weights. Raw weights are `min(sqrt(max_bin_count / bin_count), 3)`, then
normalized to average one over training examples. Empty bins use count one
for a finite stored weight but contribute no training examples. The cap bounds
the ratio between weights, not their final absolute maximum. No batch-local
renormalization is used. No resampling, sigmoid, prediction clipping, or forced
output-range expansion is applied.

## Validation and output

All checkpoints use exact **unweighted graph-average validation DockQ MSE**.
This corrects the older history's equal weighting of batches, including its
short last batch. All 36 jobs use the same definition; the old dropout-0.3
seed-7 job is therefore not silently substituted for the new control.
No test evaluation is performed during or after the sweep.

Each run writes:

- `loss_history.csv`: optimization losses and coefficients. With range weighting,
  `train_target_mse` is the weighted training objective; `val_target_mse` is
  always ordinary graph-average MSE. Other reconstruction histories retain
  the trainer's existing batch averaging.
- `dockq_training_distribution.json`: training counts and candidate bin weights.
  The `none` arm records the distribution but does not apply those weights.
- `dockq_epoch_metrics.jsonl`: validation graph-average, target-average and
  bin-average MSE, per-bin counts/MSE/bias, and first-training-batch gradient
  norms for weighted DockQ and weighted reconstruction at shared node embeddings.
  These are sampled latent-gradient diagnostics, not full-epoch encoder-parameter
  gradient norms. Empty validation bins are null, excluded from bin averages,
  and counted explicitly through `occupied_bins`.
- `gate_model.pt` and selected-checkpoint `validation_predictions.csv`.

Combined stdout/stderr logs: `gate_run/dockq_priority_sweep/logs/train_JOB_TASK.out`.
Summarize after completion:

```bash
python summarize_dockq_priority_sweep.py gate_run/dockq_priority_sweep
```

The report verifies identical validation cohorts and matching checkpoint MSE,
reports missing jobs, and exports per-run, per-bin, and across-seed CSVs under
`analysis/`. Select using ordinary validation MSE, inspect high-DockQ bias and
macro-bin/target MSE, and compare matching seed sets. Validation-based selection
across many configurations can be optimistic; preserve the test set for the
final chosen protocol.

## Resources

Each job uses one GPU and eight CPUs: one main process plus seven DataLoader
workers for the active loader. Persistent worker pools are disabled so training
and validation pools do not accumulate. Numerical-library and PyTorch intra-op
threads are one, with one inter-op thread. The trainer and runner log the actual
configuration. Preparation and summary are serial. Existing training Slurm
memory/time limits are retained. No jobs are submitted from the workstation.

## Local verification

Twenty targeted regression tests passed across the objective, matrix, preflight,
validation-only execution, existing loss balancing, and submission checks. Six
real-data two-epoch CUDA smoke runs passed on the RTX 4090: lambda 0, 0.1 and 1,
each with both weighting modes. These used two spawned loader workers plus one
main process, single-threaded numerical libraries, and verified checkpoint
reload, validation exports, and the comparison report. Their tiny balanced
training fixture is a software check, not evidence of accuracy; nonuniform
weight values and gradients are covered by the numerical regression test.
Smoke artifacts are under `local_data_audit/dockq_priority_smoke/`.
