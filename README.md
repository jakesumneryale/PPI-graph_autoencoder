# GATE: Graph Attention autoEncoder for protein–protein docking models

GATE scores protein–protein docking models. Each model of a two-chain complex becomes a residue graph (nodes =
residues, edges = residue contacts), and a graph attention network learns two things from it at once:

1. **to predict the model's DockQ** (agreement with the native complex, 0–1), which is the quality score used for
   ranking; and
2. **to reconstruct the interface graph** (residue identities, contacts, Cα distances): an autoencoder objective
   that shapes the embedding.

The current default is the **rank model**, configured by [`configs/gate_rank.toml`](configs/gate_rank.toml).

---

## Contents

- [Results of the default (rank) model](#results-of-the-default-rank-model)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Configuration files](#configuration-files)
- [Default model vs built-in defaults (what is set differently)](#default-model-vs-built-in-defaults-what-is-set-differently)
- [How the rank model works](#how-the-rank-model-works)
- [Data](#data)
- [Outputs of a run](#outputs-of-a-run)
- [Other configurations that were tried](#other-configurations-that-were-tried)
- [Repository map](#repository-map)
- [Preparing graph data (Voronoi contact areas, 10% subset)](#preparing-graph-data-voronoi-contact-areas-10-subset)

---

## Results of the default (rank) model

29 held-out test targets, 3 seeds (7, 17, 27), each evaluated once at its best-validation checkpoint
(details: [`RANK_FOLLOWUP_FULL.md`](RANK_FOLLOWUP_FULL.md),
[`gate_run/analyze_rank_followup_full.ipynb`](gate_run/analyze_rank_followup_full.ipynb)).

| model | test DockQ MSE ↓ | mean within-target Spearman ρ ↑ | top-1 DockQ ↑ |
|---|---|---|---|
| constant predictor (training mean) | 0.0967 | – | 0.334 (random pick) |
| feature ladder, core features (old recipe) | 0.1070 | 0.553 | 0.319 |
| feature ladder, all features (old recipe) | 0.0808 | 0.644 | 0.454 |
| **rank (default)** | **0.0586 ± 0.0024** | **0.686 ± 0.014** | 0.447 ± 0.020 |

Top-1 DockQ is the true DockQ of the model each method ranks first for a target (oracle 0.905).

**Against established scoring functions** on the 18 targets from the main dataset with scores for every method
(validation and test pooled; Spearman ρ with DockQ, sign oriented so that more negative is better):

| ZRank2 | PyDock | ITScorePP | **GATE (rank)** | VoroIF-GNN | Deeprank-GNN-ESM | Rosetta | VoroMQA |
|---|---|---|---|---|---|---|---|
| −0.745 | −0.718 | −0.687 | **−0.673** | −0.652 | −0.634 | −0.633 | −0.492 |

**Known limitation.** With fixed loss weights the Cα-distance term (Å²) dominates reconstruction, and the rank
model recovers only ~36% of amino-acid identities (the old ladder recipe: 87%). DockQ prediction is unaffected,
but the autoencoder half is weaker; rebalancing the reconstruction terms is an open follow-up.

---

## Installation

Python ≥ 3.11 (configs are read with the standard-library `tomllib`). Install the CUDA build of PyTorch for your
machine from [pytorch.org](https://pytorch.org) first, then:

```bash
pip install -r requirements.txt
```

On the Yale cluster the runs use `module load miniconda; conda activate py311_env` plus the project `venv`
(see [`cluster/train_gate_config.slurm`](cluster/train_gate_config.slurm)).

Checks:

```bash
python protein_hdf5_dataset.py --require-target     # inspect the HDF5 graph data
python smoke_test_gate.py --device auto             # one forward/backward pass
python -m pytest -q                                 # unit tests
```

---

## Quick start

Train the default model:

```bash
python train_gate.py --config configs/gate_rank.toml --seed 7 --output-dir runs/rank_seed7
```

See exactly what a run will use, without training:

```bash
python train_gate.py --config configs/gate_rank.toml --print-config
```

Change one setting without editing the file (command-line flags override the config):

```bash
python train_gate.py --config configs/gate_rank.toml --capri-head-weight 0.2 --output-dir runs/rank_capri
```

For a local run, copy the config and change the site-specific part (`cluster = false`, `data`, `model_list_dir`,
`split_manifest`, `device`, `num_workers`). On/off options such as `cluster` can only be switched *on* from the
command line, so switching them off is done in the file.

On the cluster, three seeds as a Slurm array (1 GPU, 8 CPUs = main process + 7 loader workers per seed):

```bash
sbatch cluster/train_gate_config.slurm configs/gate_rank.toml gate_run/my_experiment
sbatch cluster/train_gate_config.slurm configs/gate_rank.toml gate_run/rank_capri --capri-head-weight 0.2
```

Repeat any finished run exactly:

```bash
python train_gate.py --config runs/rank_seed7/run_config.toml
```

---

## Configuration files

A config is a [TOML](https://toml.io) file that sets `train_gate.py` options.

```toml
[model]                       # sections are only for readability; any option can go in any section
dropout = 0.3
residual_connections = true   # on/off options take true / false
pooling = "combined"          # strings in quotes; choices are checked

[features]
node_features = ["aa_type", "chain", "interface_nodes"]   # lists or "a,b,c" strings both work

[optimization]
lr = 3e-4
```

- **Keys** are the option names without the leading `--`; dashes or underscores both work
  (`batch-size` = `batch_size`).
- **Precedence:** command-line flags > config file > built-in defaults.
- **Checked on load:** a misspelled option, a value outside its allowed choices, or a wrong type stops the run with
  a clear message (`unknown option "dropuot" (did you mean dropout?)`).
- **Every run writes `run_config.toml`** to its output directory: every option with its resolved value, the git
  commit (flagged if there were uncommitted changes) and the exact command line. Pass it back with `--config` to
  repeat the run.
- Implementation: [`gate_config.py`](gate_config.py); tests: [`test_gate_config.py`](test_gate_config.py).

Shipped configs:

| file | what it is |
|---|---|
| [`configs/gate_rank.toml`](configs/gate_rank.toml) | the default rank model, with the cluster data paths it was trained on |

---

## Default model vs built-in defaults (what is set differently)

`train_gate.py`'s built-in defaults are the **original** settings and are deliberately left unchanged so the older
experiment scripts in `cluster/` (feature ladder, anti-memorization, cause ablation, …) still reproduce exactly.
The rank model therefore lives in the config file. The table lists **every option where the rank model differs
from the built-in default** (generated from `--print-config` with and without the config).

### Changed for the rank model

| option | built-in default | **rank model (what you use)** | why |
|---|---|---|---|
| **Model** | | | |
| `dropout` | 0.1 | **0.3** | with 0.1 the validation DockQ loss rose during training (cause ablation) |
| `residual_connections` | false (2 GAT layers) | **true** (4 residual GAT layers) | deeper encoder; residuals keep it trainable |
| `pooling` | `all` | **`combined`** | whole-graph *and* interface mean/max pooling feed the DockQ head |
| **Features** | | | |
| `node_features` | `aa_type, chain, interface_nodes` | **`aa_type, chain, interface_nodes, interface_node_degree, rsasa_i_node`** | the feature ladder's full feature set |
| `edge_features` | `interface_edges, ca_dist` | **`interface_edges, ca_dist, voronoi_contact_area, voronoi_contact_missing, v_es`** | Voronoi contact area and electrostatics as edge inputs |
| `edge_recon_features` | (all edge features) | **`interface_edges, ca_dist`** | Voronoi area / v_es are inputs only; reconstructing them swamped the loss |
| `edge_feature_transforms` | (none) | **`voronoi_contact_area=log1p`** | heavy-tailed areas |
| `standardize_edge_features` | (none) | **`voronoi_contact_area, v_es`** | z-scored with training-split statistics |
| `edge_stats_seed` | (= `seed`) | **7** | identical input normalization for every seed |
| **Optimization** | | | |
| `epochs` | 100 | **20** | the best validation checkpoints fell at epochs 3–12 |
| `batch_size` | 4 | **16** | |
| `lr` | 1e-3 | **3e-4** | |
| `lr_schedule` | `constant` | **`cosine`** | 1000-step linear warmup, then cosine decay to 3% of `lr` |
| `targets_per_batch` | 0 (plain shuffle) | **4** | each batch = 4 targets × 4 models, so the ranking loss has within-target pairs |
| **Loss** | | | |
| `dockq_range_weighting` | `none` | **`inverse-sqrt`** | up-weights rare high-DockQ models (bin weight (n_max/n_bin)^0.5, capped at 3×) |
| `rank_loss_weight` | 0.0 | **0.1** | within-target pairwise ranking loss (see below) |
| **Evaluation** | | | |
| `checkpoint_metric` | `loss` (total loss) | **`target_mse`** | keep the checkpoint that predicts DockQ best, not the one that reconstructs best |
| `val_every_steps` | 0 (epoch ends only) | **2000** | finer checkpoint selection |
| `validation_only_during_training` | false | **true** | test set sealed; scored once at the selected checkpoint |
| `dockq_range_diagnostics` | false | **true** | per-DockQ-range metrics; required by the range weighting |
| **Data and runtime** (site-specific, change freely) | | | |
| `cluster` | false | **true** | cluster default directories |
| `data` | (local default) | **`/nfs/roberts/project/pi_co54/jas485/ppi_processed_graphs`** | |
| `model_list_dir` | (all models) | **`gate_run/full_pooling_repaired/model_lists`** | the usable models of each target |
| `split_manifest` | (new random split) | **`gate_run/full_pooling_repaired/target_splits.json`** | the fixed train/val/test target split |
| `device` | `auto` | **`cuda`** | |
| `num_workers` | 0 | **7** | 8-CPU allocation = main process + 7 loader workers |
| `worker_start_method` | `auto` | **`spawn`** | avoids HDF5/fork hangs |
| `cpu_threads` | (PyTorch default) | **1** | one BLAS/PyTorch thread in the main process |
| `output_dir` | `checkpoints` | **`runs/gate_rank`** | set per run with `--output-dir` |

### Same as the built-in default (set explicitly or left alone)

| group | options and values |
|---|---|
| model | `architecture = gat`, `hidden_dim = 64`, `latent_dim = 32`, `gat_heads = 4`, `dockq_head_mode = joint` |
| optimization | `weight_decay = 1e-5`, `warmup_steps = 1000`, `lr_final_fraction = 0.03`, `grad_clip = 1.0`, `seed = 7` (runs use 7, 17, 27), `decoys_per_target = 0`, `ema_decay = 0` |
| loss | `loss_weight_mode = fixed`, `target_weight = 1.0`, `node_weight = 1.0`, `edge_attr_weight = 1.0`, `edge_presence_weight = 0.1`, `reconstruction_lambda = 1.0`, `dockq_weight_exponent = 0.5`, `dockq_weight_cap = 3.0`, `rank_temperature = 0.1`, `dockq_hinge_start = 0.23` (also the ranking-loss pair threshold), `capri_head_weight = 0`, `target_adversary_weight = 0` |
| augmentation | `edge_dropout = 0`, `node_feature_mask = 0`, `ca_dist_noise = 0` |
| data | `target_name = DockQ`, `test_fraction = 0.2`, `val_fraction = 0.1` (both ignored when `split_manifest` is given), `exclude_random_negatives = false` (true drops the random-negative models) |
| features | `use_rsasa_i = false`, `use_drsasa = false`, `use_esm = false`, `edge_stats_max_graphs = 200` |

The shares options (`target_share`, `node_share`, …) only matter with `loss_weight_mode = shares`, which the old
feature-ladder runs used (DockQ 0.60, node 0.15, edge attributes 0.15, edge presence 0.10).

---

## How the rank model works

**Encoder.** Node features (plus optional ESM-2 embeddings) pass through 4 `GATConv` layers (64 channels × 4
heads, edge features as attention inputs, dropout 0.3, a residual connection from the input) and are projected to a
32-dimensional embedding per residue.

**Decoders (reconstruction).** From the residue embeddings: node features (MSE), the reconstructed edge
features `interface_edges` and `ca_dist` (MSE), and edge presence (binary cross-entropy against sampled
non-edges).

**DockQ head.** Mean and max pooling over all residues and over interface residues (`pooling = combined`) →
MLP → one DockQ prediction per model.

**Training objective**

$$\mathcal{L}=\underbrace{\tfrac{1}{N}\sum_i w_{b(y_i)}(\hat y_i-y_i)^2}_{\text{range-weighted DockQ MSE}}
+\lambda\big(\mathcal{L}_\mathrm{node}+\mathcal{L}_\mathrm{edge\,attr}+0.1\,\mathcal{L}_\mathrm{edge\,presence}\big)
+0.1\,\mathcal{L}_\mathrm{rank}$$

- $w_{b(y)}$: weight of the DockQ bin (0–0.2, …, 0.8–1.0) of the true DockQ, $(n_\max/n_b)^{0.5}$ capped at 3×
  and normalized to mean 1 over the training labels.
- $\lambda = 1$ (`reconstruction_lambda`).
- $\mathcal{L}_\mathrm{rank}$, the within-target pairwise logistic ranking loss over pairs $P$ of models from the
  same target with different DockQ where at least one has DockQ ≥ 0.23, weighted by their true DockQ gap
  (temperature $\tau = 0.1$):

$$\mathcal{L}_\mathrm{rank}=\frac{\sum_{(i,j)\in P}|y_i-y_j|\,\log\!\left(1+\exp\!\left[-\frac{\mathrm{sgn}(y_i-y_j)(\hat y_i-\hat y_j)}{\tau}\right]\right)}{\sum_{(i,j)\in P}|y_i-y_j|}$$

  It only rewards ordering models correctly within a target; the MSE term keeps the predictions calibrated.
  Slide-ready renderings of these equations: [`gate_run/rank_loss_equations/`](gate_run/rank_loss_equations/).

**Checkpoint selection.** Validation runs every 2000 steps and at each epoch end; the checkpoint with the lowest
*plain* (unweighted) validation DockQ MSE is kept. Only that checkpoint is evaluated on the test targets, once.

Code: loss terms in [`train_gate.py`](train_gate.py) (`compute_gate_loss`, `run_epoch`), range weighting and
ranking loss in [`dockq_objectives.py`](dockq_objectives.py), grouped batches in
[`anti_memorization.py`](anti_memorization.py) (`GroupedTargetBatchSampler`), model in
[`GATE_model.py`](GATE_model.py).

---

## Data

One HDF5 file per target (`<target>.hdf5`); one group per docking model, with

| group | contents |
|---|---|
| `node_features/` | `aa_type` (one-hot, 20), `chain`, `interface_nodes`, `interface_node_degree`, `rsasa_i_node`, … |
| `edge_features/` | `contacts` (edge list), `interface_edges`, `ca_dist`, `voronoi_contact_area`, `voronoi_contact_missing`, `v_es` |
| `target_scores/` | `DockQ` (and other labels) |

Models come from ZDOCK, sampled uniformly over DockQ (20 bins) plus random negatives, ~2000 per target.

- **Splits are by target**, never by model, so no target appears in two splits. The default run uses the fixed
  split in `split_manifest` (29 test targets). Without a manifest, 20% of targets go to test and 10% of the rest to
  validation, and the split is written to `<output_dir>/target_splits.json`; pass that file as `split_manifest` to
  reuse it.
- **`model_list_dir`** holds `<target>.txt` files listing which models of each target to use (e.g. only models
  whose Voronoi features passed the audit).
- Data location: `--data` / `data`, else the `PPI_HDF5_DATA` environment variable, else
  `/scratch/ppi_autoencoder_code/processed_graph_data`; `cluster = true` switches to the cluster directories.
  Unreadable HDF5 files are skipped with a warning unless `strict_hdf5 = true`.

---

## Outputs of a run

In `output_dir`:

| file | contents |
|---|---|
| `run_config.toml` | every resolved option, git commit, command line (rerun with `--config`) |
| `gate_model.pt` | the selected checkpoint |
| `loss_history.csv`, `validation_steps.csv` | per-epoch and per-2000-step losses |
| `target_splits.json` | the train/val/test targets |
| `validation_predictions.csv`, `test_predictions.csv` | predicted and true DockQ per model |
| `prediction_metrics.json` | MSE, per-range metrics; `test` and `selected_checkpoint` blocks |
| `dockq_epoch_metrics.jsonl`, `dockq_training_distribution.json` | per-range diagnostics and the range weights |
| `reconstruction_summary.csv`, `reconstruction_by_target.csv` | reconstruction quality on test (AA identity, edge F1, Cα distance error in Å) |

`python evaluate_gate_reconstruction.py` re-evaluates reconstruction for any saved checkpoint.

---

## Other configurations that were tried

Each is the rank config plus one change (3 seeds, same 29 test targets):

| configuration | change | test MSE | test ρ |
|---|---|---|---|
| **rank** | — | **0.0586** | **0.686** |
| rank + CAPRI head | `--capri-head-weight 0.2` (auxiliary CAPRI-class cross-entropy) | 0.0599 | 0.676 |
| rank, λ = 3 | `--reconstruction-lambda 3` | 0.0606 | 0.659 |
| rank, lr 1e-4 | `--lr 1e-4` | 0.0604 | 0.661 |

None beat the plain rank configuration. Earlier experiments that led to it (each with its own write-up and
notebook): feature ladder, anti-memorization, cause ablation, long (40-epoch) screen. See the `*_FULL.md` files and
`gate_run/analyze_*.ipynb`.

---

## Repository map

| path | what |
|---|---|
| `train_gate.py` | training / evaluation entry point |
| `gate_config.py`, `configs/` | TOML run configurations |
| `GATE_model.py` | the GAT autoencoder |
| `dockq_objectives.py` | DockQ range weighting, ranking loss, CAPRI head, per-range metrics |
| `anti_memorization.py` | grouped and target-balanced samplers, augmentation, target adversary, EMA |
| `protein_hdf5_dataset.py` | HDF5 graph dataset and data-path defaults |
| `evaluate_gate_reconstruction.py` | reconstruction metrics for a checkpoint |
| `cluster/` | Slurm scripts (`train_gate_config.slurm` runs any config; the others reproduce past experiments) |
| `gate_run/` | experiment outputs, `build_*_notebook.py` → `analyze_*.ipynb` analyses and `*_analysis/` figures |
| `test_*.py` | unit tests (`python -m pytest -q`) |
| `nb_scripts/` | graph construction from PDB files |

---

## Preparing graph data (Voronoi contact areas, 10% subset)

Residue–residue Voronoi contact areas are computed from the model PDB files and aligned to the graph's contact
list:

```bash
python precompute_voronoi_model_references.py --cluster          # per-target model reference lists
python generate_voronoi_contact_area_data.py --cluster /path/to/<target>
python generate_voronoi_contact_area_data.py --cluster --write-graph-feature \
    --feature-name voronoi_contact_area /path/to/<target>        # write it into the graph HDF5
```

Audit the Voronoi features and build a balanced 10% subset (stratified by DockQ bin, random negatives separate):

```bash
sbatch cluster/audit_voronoi_dataset.slurm
python audit_voronoi_dataset.py --audit-only                     # audit and success lists only
```

Outputs go to `voronoi_dataset_audit/`: `successful_models/<target>.txt` (usable as `model_list_dir`),
`model_audit.csv`, `target_attrition.csv`, `subset_manifest.csv` and `subset_hdf5/<target>.hdf5`.
