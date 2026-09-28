# Ch. 4 vs. Ch. 3 head-to-head comparison (TODO at dissertation.tex:1569)

Builds the comparison figure/table for GATE against the eight baselines
(ZRank2, ITScorePP, Rosetta, PyDock, VoroMQA, Deeprank-GNN-ESM, GNN-DOVE,
two-feature SVR), restricted to the **29-target test set** of the current
GATE split (`target_splits.json`, seed 5, 146-target cohort -- the split the
`gat4_residual_full` seed-replicate cluster job is training on right now).

```
cd gate_run/ch3_comparison
../../.venv/bin/python build_comparison_figure.py
```

Writes `chapter_4_figs/Fig4_comparison_test_set_PRELIMINARY.{pdf,png}` and
`output/*_per_target_summary.csv` (one row per target x method: rho, AUC@0.23,
n_decoys, n_scored). Nothing in `dissertation.tex` is touched.

## Data provenance

**16 of 29 test targets are in the main 84-PDB dataset.** Loaded from
`all_supersampled_balanced_datasets/<pdb>_supersampled_balanced_scores.csv`
-- the same final, class-rebalanced (~51% DockQ >= 0.23) per-decoy tables
Chapter 3's Figs. 3.4/3.7 are built from. SVR scores are joined from
`svr_2_score_results_no_overlap` (the leave-one-out, no-data-leakage run;
`svr_2_metric_enum.py` confirms `svr_cols = ["Intertwined", "Contact_45"]`,
i.e. exactly S and N_c). One target (7JXV) has no SVR file in that directory
and is reported as missing rather than silently dropped.

**13 of 29 are ZDOCK Benchmark-5.5 targets**, with no equivalent balanced
CSV on disk anywhere (checked). Parsed directly from the raw
`bm_55_scoring/scores_sampled_<target>/*` files with `raw_score_parsers.py`.
**GNN-DOVE was never scored on BM5.5** (no `dove_*` file exists in any of
the 13 directories) -- its aggregate is over 16/29 targets, not 29, and this
is stated on the figure itself.

## Validated, not assumed

Every raw-file parser was checked against 1ACB, an 84-set target with an
independently-computable reference: parsed scores reproduce
`all_supersampled_balanced_datasets/1acb_...csv` to **zero** absolute
difference over all 961 shared decoys, and Spearman rho against DockQ
(computed on the full 1394-decoy balanced set) matches
`supersampled_correlation_dataset_final_manuscript.csv` to **4 decimal
places** for all 7 baseline methods (see `parse_raw_scores` validation
run in the session transcript). AUC values are computed fresh here rather
than read from that CSV: the published `dove_5_auc` column has an
unflipped-sign bug (verified against 1ACB: published 0.273 vs. the
semantically-correct 0.729 my code and manual recomputation both give) --
`HIGHER_IS_BETTER` in `raw_score_parsers.py` fixes this throughout.

The two-feature SVR reproduction is close but not exact: recomputing over
all 84 targets from these directories gives mean |rho| = 0.693-0.680
(no_overlap / with_overlap respectively) against the dissertation's
headline 0.750. Same features, same leave-one-out RBF procedure per
`svr_2_metric_enum.py` -- the residual gap is unexplained (possibly a later
re-run under a different name not found on disk) and is worth flagging to
Jake rather than silently absorbing.

## Known asymmetry between the two halves of the test set

The BM5.5 `scores_sampled_*` sets are uniform-in-DockQ-bin (matching the
dissertation's stated sampling procedure) but **not** separately rebalanced
by class the way the final 84-set tables are: 60-75% DockQ >= 0.23 in the
13 BM5.5 test targets vs. ~51% in the 16 84-set test targets. Both are far
better than raw docking output (~5% positive), but they are not the same
procedure. This is stated in the figure footer and should be called out
in the caption/text if this figure goes into the dissertation as-is.

## Plugging in GATE

`gate_loader.py` reads a `test_predictions.csv` in GATE's own output format
and reshapes it to match every baseline table. Point
`GATE_PREDICTIONS_PATH` at the finished run (once rsynced locally) and
rerun `build_comparison_figure.py` -- everything else regenerates
automatically, no other code changes needed. The loader hard-fails if the
run's target list doesn't exactly match these 29 targets, so it cannot
silently plot a mismatched split.
