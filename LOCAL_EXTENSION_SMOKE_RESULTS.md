# Local extension validation — September 17, 2026

Subsequent decision: APBS is deferred for the next cluster experiment. The
APBS-free production preparer accepted the nine real smoke graphs without
an APBS path, and all 11 distinct configurations passed two-epoch CUDA runs,
checkpoint reload, finite prediction/loss checks, and comparison export.
The test suite passed 24 regression tests plus two mocked submission tests.
See `MODEL_EXTENSION_EXPERIMENTS.md` for the new launcher and
`local_data_audit/no_apbs_smoke/results.json` for execution results. The older
APBS-on checks below remain historical evidence, not a launch requirement.

The downloaded bundle is intact and numerically compatible. All 19 experiment
configurations completed real-data CPU and RTX 4090 software smoke runs.
APBS area-weighting provenance remains unresolved; this is not approval to
label the stored means as verified area-weighted surface potentials.

## Download and alignment

- Bundle: `/scratch/ppi_extension_data/full` (19.34 GiB).
- All 15,660 files matched their manifest sizes and SHA-256 checksums.
- All 14,906 graphs across 125 targets passed graph feature, PDB residue/chain,
  coordinate, ESM sequence/shape, APBS residue identity/units, and rSASA checks.
- Original selection: 15,086 graphs; APBS availability excluded 180 (157 random
  negatives and 23 sampled models). No additional targets were lost to APBS.
  The earlier 21 empty Voronoi targets remain excluded.
- ESM layer 33 has 1,280 features per residue for every graph.
- 9,591,795 of 30,241,331 stored edges have masked APBS pair features (31.7%).
  Buried residues or other invalid surface samples can cause this mask; it does
  not mean these graph records are missing. Missing values stay neutral after
  feature scaling.
- All APBS area-weighting markers are absent. Alignment and finite values do
  not establish how residue surface means were weighted. No marker was added,
  no original data was changed, and normal preparation still enforces provenance.

Evidence: `local_data_audit/downloaded_bundle/integrity.json`,
`alignment/summary.json`, and the bundle's `apbs_coverage.json`.

## Numerical software tests

The nine-graph fixture uses low, middle, and high DockQ examples from each of
1acb, 1avx, and 1ay7. It is explicitly marked SOFTWARE TEST ONLY. Raw stored
APBS means exercise the numerical code without attesting to their weighting.

- All 19 configurations ran two epochs, reloaded their selected checkpoints,
  and exported finite predictions on both CPU and CUDA.
- All runs use the same target split; both reference comparison exports passed.
- 23 regression tests passed, including ESM alignment, missing-value scaling,
  interface pooling, EGNN symmetry, and allocation limits.
- Directed CUDA checks on two real graphs (608 nodes, 6,856 directed edges)
  confirmed finite gradients and nonzero gradients to ESM projection weights.
- Interface pooling matched explicit mean/max calculations and was unchanged
  when only non-interface embeddings were perturbed.
- EGNN rigid-motion prediction discrepancy was approximately 1.5e-8.
- With dropout disabled, a 60-step same-batch fitting check reduced target MSE
  from 0.2721 to 0.0000475 for GAT and from 0.2870 to 0.0000224 for EGNN.

These are execution and optimization checks, not generalization estimates.
Do not use the smoke comparison metrics to select the best scientific model.

The capacity test repeated the largest graph (915 nodes, 5,473 stored edges)
16 times: 14,640 nodes and 175,136 directed edges in a batch. Forward/backward
and optimizer steps passed with the full ESM/APBS/area GAT configuration
(6.85 GiB peak allocated GPU memory) and combined EGNN (6.51 GiB).
Seven batches were loaded with seven spawn workers; worker IDs 0 through 6
were all observed processing data. The main process used one PyTorch thread,
and numerical library pools were restricted to one thread per process.
This checks the intended eight-CPU configuration and batch-size-16 memory
capacity on the 4090; it does not measure L40S training throughput.

Evidence: `local_data_audit/downloaded_bundle/smoke19/results.json`,
`smoke19_cuda/results.json`, `smoke19_cuda/directed_checks.json`,
and `regression_tests.log`.
Capacity details and its reproducible script are in `capacity_checks.json`
and `capacity_check.py` in the same audit directory.

## GPU access and next experiment

The host RTX 4090 works with NVIDIA driver 590.48.01 and PyTorch 2.10.0+cu130.
The restricted session hides NVIDIA device files; approved host execution
provides CUDA access. No driver reinstall or reboot was needed.

Before preparing the real 19-run cohort, resolve the APBS interpretation:
either explicitly use stored surface means for an exploratory experiment
without claiming verified area weighting, or regenerate the missing provenance
with the area-weighted implementation. The current production preparer still
requires recorded weighting or an explicit attestation; this audit supplies
neither. Do not use `--assume-area-weighted` based on these smoke results.

Reproduce the audit and smoke tests with the commands in
`LOCAL_DATA_WORKFLOW.md`. Original downloaded inputs remain unchanged.
