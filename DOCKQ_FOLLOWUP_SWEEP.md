# DockQ follow-up experiments

After syncing the code to the cluster, from the repository:

```bash
bash cluster/submit_dockq_followup_sweep.sh
```

Default output: `gate_run/dockq_followup_sweep/`. Set `EXPERIMENT_DIR` for a
fresh location. The builder refuses to overwrite existing experiment directories.
The completed `gate_run/dockq_priority_sweep/` must remain available.

## Jobs and comparisons

36 jobs are submitted, with no array concurrency caps:

1. Tasks 1–3 evaluate the three existing lambda=1 range-weighted checkpoints
   on training and validation graphs, with dropout disabled. No optimization,
   retraining, or test evaluation occurs. These jobs establish whether high
   DockQ compression is already present on training targets.
2. Tasks 4–27 run 24 new pooling/weighting combinations. Together with three
   completed controls reused from the prior sweep, these cover 3 pooling modes
   × 3 weighting exponents × 3 seeds. Pooling modes are whole graph, interface
   only, and concatenated whole-graph/interface mean and max pools. Exponents
   are 0.5, 0.75, and 1.0. Lambda=1, dropout=0.3, LR=0.0003, 50 epochs and the
   existing cosine schedule remain fixed. Combined pooling doubles the input
   width to the graph projector and thus adds parameters; it is not a strictly
   parameter-count-matched comparison.
3. Tasks 28–36 fine-tune each existing selected checkpoint for 20 additional
   epochs at LR=0.00003, cosine decay, zero warmup, and lambda=0, 0.1 or 1.
   The optimizer is reset identically for all arms. Lambda=1 is the matched
   continued-training control. This is supervised structural pretraining from
   the existing multitask checkpoint, not a new unsupervised pretraining stage.
   The starting checkpoint is evaluated and retained as epoch zero unless
   ordinary validation DockQ MSE improves. Range weighting remains exponent
   0.5; pooling remains whole graph. Do not compare different training budgets
   as though lambda alone explains their difference.

The 33 training jobs depend on successful completion of all three diagnostics.
This is an execution check, not an automated scientific decision based on their
results. If diagnostics fail, inspect their logs before resubmitting; training
jobs will not start under the failed dependency.

## Objective and evaluation

Weights use training-bin counts only: `min((max_count/count)**alpha, 3)`, normalized
to mean weight one over training examples. Empty bins have finite placeholders
but no examples. Five fixed bins span [0,1]. The cap is a weight ratio before
normalization; alpha=0.5 reproduces the preceding inverse-square-root method.
The existing option is still called `--dockq-range-weighting inverse-sqrt` for
backward compatibility; `--dockq-weight-exponent` specifies the actual power.

All jobs use the original APBS-free 10% data and frozen target split. No test
predictions are generated. Selection uses unweighted graph-average validation
MSE. All new trained models also export training predictions from the selected
checkpoint in evaluation mode, so their training/validation comparisons have
matching dropout behavior. Per-range MSE, bias, target-average MSE, histories
and sampled latent-gradient diagnostics remain available.

Outputs include `training_predictions.csv`, `training_target_coverage.csv`
(counts by target and DockQ bin), `validation_predictions.csv`, and
`prediction_metrics.json`; fine-tunes also write `initial_validation.json`.
Logs: `gate_run/dockq_followup_sweep/logs/train_JOB_TASK.out`.

After downloading or completing results:

```bash
python summarize_dockq_followup.py gate_run/dockq_followup_sweep
```

The summary validates cohorts/checkpoint errors, includes reused controls,
reports missing runs, and writes per-run/per-range/across-seed tables in
`analysis/`. Compare fine-tuning arms against their starting checkpoints and
against lambda=1 continued training, not just against a from-scratch model.
Only three seeds and one small validation split are involved. Keep any final
claims of generalization for a fixed protocol evaluated on unseen targets.

## Resource configuration

One GPU and eight CPUs per job: one main process and seven active DataLoader
workers, each with single-threaded numerical libraries. Persistent worker pools
are disabled. PyTorch main intra-op and inter-op threads are one. Preparation
and reporting are serial. Existing preprocessing and completed control runs
are reused. The launcher runs on the login node and submits GPU arrays; it
does not allocate a GPU for the builder or summary.

## Local verification

The 32-test model/training regression suite passed; three targeted checks also
passed after adding training-target coverage and interface preflight validation.
Seven real-data RTX 4090 smoke jobs passed: evaluation-only reload, all three
pooling modes, and fine-tuning at lambda 0, 0.1 and 1. Checkpoint fallback,
training/validation exports and the summary command were exercised. These tiny
software fixtures do not measure scientific improvement. Evidence is under
`local_data_audit/dockq_followup_smoke/`.
