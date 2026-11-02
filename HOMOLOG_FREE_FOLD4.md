# Homolog-free fold 4, with and without ESM2

Three questions on a split whose test set has not been read: how good is the rank model on targets with no homolog in training, and do
ESM2 embeddings or the `aa_type` one-hot help on those targets? Twelve GPU runs: four arms × seeds 7/17/27, 10 epochs each.

| arm | recipe |
|---|---|
| `arm_a_no_esm` | rank-loss weight 0.3, 2 targets × 8 decoys per batch (the arm chosen on validation in `RANK_SWEEP_EXPANDED.md`) |
| `arm_b_esm16` | the same + ESM2 layer-33 embeddings (`--use-esm`, `--esm-projection-dim 16`) |
| `arm_c_no_aa` | arm A without `aa_type` |
| `arm_d_esm16_no_aa` | arm B without `aa_type` |

Everything else is `configs/gate_rank.toml`, plus `--val-every-steps 6000`.

## The split

`make_homolog_free_split.py --fold 4` starts from `expanded_target_splits/stratified_fold4.json` and moves any validation or test target that
shares an RCSB 30% chain cluster with a training target into training, repeatedly, until none has a homolog there. Nothing is dropped.

| | train | val | test |
|---|---|---|---|
| fold 4 before | 253 | 35 | 62 |
| homolog-free fold 4 | 263 | 35 | 52 (28 new, 24 original) |

Ten test targets moved (`gate_run/homolog_free_fold4/moved_to_train.csv`). Difficulty bins stay balanced (10/12/11/9/10 on test).

- **Why not the `homolog_leakage_splits` folds?** They *remove* homologs from training and leave 5 validation targets with homologs; they answer how
  much leakage is worth, not how the model does when homolog families are only in training.
- **Why fold 4?** It keeps all 35 validation targets and its test set was never read: none of its targets were in fold 1's test, and only 4 were in fold
  1's validation. Homolog-free fold 1 would reuse a test set read several times, and its 52 targets have no homolog in training either way, so it would
  mostly repeat existing runs.
- **Known hard targets move.** 1KAC is in validation and 4HWI, 1HE8 and 9WLP in training, so this test set cannot re-test them; it will have its own worst
  targets. 1KAC in validation can still pull on checkpoint selection, as any hard target can.

## ESM2

- **Embeddings.** `compute_esm_embeddings.py` runs `esm2_t33_650M_UR50D` once per chain, on the sequence in the graph's `node_reference`, in the layout of the existing
  `SS_embeds`. Only the 204 new targets need it; the 146 originals keep their stored embeddings. 5GZT's 1,134-residue chain exceeds ESM2's limit of 1,022: it uses two overlapping
  windows, averaged where they overlap. 5GZT is in the test set.
- **Sidecars.** `esm_sidecars.py --graph` stores one embedding per residue layout, so once per target (the 146 originals took 230 MB). All 350 are rebuilt: 21 of the old
  `full_pooling_esm_sidecars` lack models added by the cohort repair (26,026 of 176,964), which would crash training.
- **Projection 16, not 64.** In `full_pooling_esm_validation`, ESM2 had 64 of 88 encoder input columns, raised validation MSE by 16% and reached training MSE about 3× lower than
  controls, though within-target Spearman improved on 38 of 60 target pairs. At 16 it has 16 of 40 columns. The sweep meant to test this (`FULL_ESM_TUNING.md`) was never run, so 16 is a judgment call.

## Without `aa_type`

- **What changes.** Arms C and D use `--node-features chain,interface_nodes,interface_node_degree,rsasa_i_node`. The 20 one-hot columns leave both the encoder input and the node reconstruction target. ESM2 is an input
  only, so arm D reconstructs the same 4 columns as arm C.
- **Node loss weight.** `node_mse` averages over columns, so with 4 columns instead of 24 each one would count six times more against DockQ and the edge terms. `--node-weight 0.166667` (4/24) keeps their weight as in arms A and B,
  so the only change is the removal. Comparing reconstruction metrics across arms needs the same care: `node_feature_mse` averages over different columns.
- **Not sequence-blind.** `v_es` still depends on residue charges, and the contact and burial features depend on residue type indirectly. These arms measure the explicit one-hot, not all residue-identity information.

Read as a 2×2: A vs C and B vs D give the effect of `aa_type` without and with ESM2; A vs B and C vs D give the effect of ESM2 with and without `aa_type`.

## Decision rule (fixed in advance)

Compare the arms on **validation** (DockQ MSE, ⟨ρ⟩, top-1); read the test set once, after that comparison. Report on test: all four arms, the worst-target scorecard, GATE vs ZRank2, and
the 3-seed ensembles. Targets scored on relaxed decoys that are not GATE's (`2oul`, `2wmp`, `3qc8`, `3k9p`) or lacking classical scores are left out of classical comparisons.

## Running

From the repository on a Bouchet login node, after pulling:

```bash
bash cluster/submit_homolog_free_fold4.sh setup
bash cluster/submit_homolog_free_fold4.sh esm
ESM_DEPEND=<sidecar job> bash cluster/submit_homolog_free_fold4.sh train
```

`setup` installs `fair-esm` into the venv and checks the weights file. The weights URL is blocked from Bouchet (HTTP 403), so they were copied to
`jas485/esm_models/hub/checkpoints` (checksum verified). `esm` submits the GPU embedding job (one CPU; it first re-embeds two stored chains and stops if they differ) and then the 350-task 1-CPU sidecar array. `train` writes the staged manifest and submits all four arms; arms A and C
start at once and arms B and D wait for the sidecars. `DRY_RUN=1` previews each stage; `RUNS="arm_a_no_esm"` submits one arm. Results: `gate_run/homolog_free_fold4/<arm>/gate_rank_seed<seed>/`.
Resources per run: one GPU, 8 CPUs (main process + 7 loader workers), 80 GB, 3-day walltime.

## Verification

- **Split:** asserted in the script: all 350 targets kept, no overlap, no validation or test target sharing a 30% chain cluster with training.
- **Embeddings:** the script reproduces the stored embeddings of 1ACB, 1KAC, 1HE8, 1J2J and 3WHQ from their graphs (both chains, max difference 2×10⁻⁵, correlation 1.000; chains up to 749 residues).
- **On Bouchet:** the weights checksum matches, and all 146 stored embedding sets, 350 data links and 350 model lists are present. PyTorch 2.13 there could not be tested on the model itself, because a
  login node kills the model load; the GPU job's reproduce-or-stop check covers this.
- **Windowing:** 5GZT's real chain runs. On a 749-residue chain with windows of the same relative size, windowed vs whole gives correlation 0.998, median per-residue cosine 1.000, minimum 0.93.
- **Sidecars:** the new path gives one shared layout per target, and values match the stored embeddings in node order.
- **Training:** on local graphs, short GPU runs of both arms finish through test and reconstruction evaluation. The ESM run exposed a bug, now fixed: `evaluate_gate_reconstruction.py` did not pass
  `esm_sidecar_dir`, so an ESM run with test evaluation would have crashed at the very end.
- **Submit script:** against a mock tree with stub `sbatch`: 204 embedding targets, 350 sidecars, arm B waits on the sidecar job and otherwise refuses without 350 sidecars, reruns refused.
