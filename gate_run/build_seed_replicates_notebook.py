"""Build the executed-result analysis notebook; inputs are never modified."""
from pathlib import Path
import nbformat as nbf

cells = []
def md(text): cells.append(nbf.v4.new_markdown_cell(text.strip()))
def code(text): cells.append(nbf.v4.new_code_cell(text.strip()))

md('''# GAT4 seed replicates: base versus full features

Six completed runs: **base/full × seeds 7, 17, 27**, evaluated on a shared target split. This notebook reads the downloaded prediction/history CSVs and a checkpoint-metadata snapshot; it never changes run artifacts or loads model weights.

The full configuration adds **interface-node degree, log-transformed Voronoi contact area, and its missing mask**. This comparison does not isolate the effect of area alone. These runs used **adaptive loss-share weighting**, not the fixed-weight objective planned for the extension experiments.

We evaluate score accuracy, within-target ranking, top-structure selection, calibration, seed variation, and training behavior. Test metrics describe the checkpoints selected by validation; no test-based checkpoint or seed selection is performed.''')
code('''from pathlib import Path
import hashlib
import json
import re
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
get_ipython().run_line_magic('matplotlib', 'inline')

# Works when launched from the repository, gate_run, or seed_replicates.
candidates = [Path.cwd(), *Path.cwd().parents]
GATE_DIR = next((p for parent in candidates for p in (parent, parent / 'gate_run')
                 if (p / 'seed_replicates/shared_target_splits.json').is_file()), None)
if GATE_DIR is None:
    raise FileNotFoundError('Set GATE_DIR to the repository gate_run directory.')
RUN_ROOT = GATE_DIR / 'seed_replicates'
EXPORT = GATE_DIR / 'seed_replicates_analysis'
EXPORT.mkdir(exist_ok=True)
metadata = json.loads((GATE_DIR / 'seed_replicates_metadata.json').read_text())
SEEDS = [7, 17, 27]
CONFIGS = ['base', 'full']
COLORS = {'base': '#585858', 'full': '#247ba0'}
LABELS = {'base': 'Base', 'full': 'Full features'}
mpl.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
                     'mathtext.fontset': 'cm', 'text.usetex': False,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'legend.frameon': False, 'figure.dpi': 110, 'savefig.dpi': 180})
pd.set_option('display.max_columns', 25)

def savefig(fig, name):
    fig.savefig(EXPORT / f'{name}.png', bbox_inches='tight')
    fig.savefig(EXPORT / f'{name}.pdf', bbox_inches='tight')
    plt.show()

def export(frame, name):
    frame.to_csv(EXPORT / f'{name}.csv', index=False)

print('Input:', RUN_ROOT)
print('Figures and tables:', EXPORT)''')
md('''## Validate the experiment and its shared cohort

All runs must have identical `(target_id, graph_name)` keys and labels. Missing runs, duplicate keys, nonfinite predictions, metadata hash mismatches, and split overlap stop the analysis rather than silently changing the cohort. The checkpoint snapshot includes hashes of the original CSVs and checkpoint files.''')
code('''predictions, histories, inventory = {}, {}, []
reference = None
for config in CONFIGS:
    for seed in SEEDS:
        run = f'gat4_residual_{config}_seed{seed}'
        directory = RUN_ROOT / run
        meta = metadata[run]
        for filename, expected_hash in meta['sha256'].items():
            if hashlib.sha256((directory / filename).read_bytes()).hexdigest() != expected_hash:
                raise ValueError(f'{run}/{filename}: changed since checkpoint metadata was extracted')
        frame = pd.read_csv(directory / 'test_predictions.csv')
        history = pd.read_csv(directory / 'loss_history.csv')
        if frame.duplicated(['target_id', 'graph_name']).any():
            raise ValueError(f'{run}: duplicate predictions')
        if not np.isfinite(frame[['true_target', 'predicted_target']].to_numpy()).all():
            raise ValueError(f'{run}: nonfinite predictions')
        if not frame.true_target.between(0, 1).all():
            raise ValueError(f'{run}: invalid DockQ labels')
        frame = frame.sort_values(['target_id', 'graph_name']).reset_index(drop=True)
        identity = frame[['target_id', 'graph_name', 'true_target']]
        if reference is None:
            reference = identity
        elif not reference.equals(identity):
            raise ValueError(f'{run}: cohort or labels differ; a paired analysis is not valid')
        if history.epoch.duplicated().any() or not history.epoch.is_monotonic_increasing:
            raise ValueError(f'{run}: invalid epoch history')
        if len(history) != meta['args']['epochs']:
            raise ValueError(f'{run}: incomplete epoch history')
        best_row = history.loc[history.epoch.eq(meta['best_epoch'])].iloc[0]
        if not np.isclose(best_row.val_target_mse, meta['best_metric_value']):
            raise ValueError(f'{run}: checkpoint/history validation metric mismatch')
        predictions[config, seed] = frame
        histories[config, seed] = history
        inventory.append(dict(config=config, seed=seed, graphs=len(frame), targets=frame.target_id.nunique(),
            epochs=len(history), checkpoint_epoch=meta['best_epoch'], checkpoint_metric=meta['best_metric'],
            checkpoint_val_mse=meta['best_metric_value'], loss_mode=meta['args']['loss_weight_mode'],
            learning_rate=meta['args']['lr'], batch_size=meta['args']['batch_size']))
inventory = pd.DataFrame(inventory)
splits = json.loads((RUN_ROOT / 'shared_target_splits.json').read_text())['splits']
split_sets = {name: set(info.get('targets', [Path(p).stem for p in info['paths']])) for name, info in splits.items()}
assert not any(split_sets[a] & split_sets[b] for a, b in [('train','val'), ('train','test'), ('val','test')])
assert set(reference.target_id) == split_sets['test'], 'Test targets differ from shared split'
display(inventory)
export(inventory, 'run_inventory')
print(f'Exact paired cohort: {len(reference):,} graphs / {reference.target_id.nunique()} test targets.')
feature_table = pd.DataFrame([dict(config=c,
    node_features=', '.join(metadata[f'gat4_residual_{c}_seed7']['node_features']),
    edge_features=', '.join(metadata[f'gat4_residual_{c}_seed7']['edge_features']),
    transforms=str(metadata[f'gat4_residual_{c}_seed7']['edge_feature_transforms'])) for c in CONFIGS])
display(feature_table)
export(feature_table, 'feature_definitions')''')
md('''## Metrics: score accuracy and within-target selection

Pooled metrics weight every graph equally; **macro metrics weight each target equally**. Spearman is calculated separately within each target before averaging. Top-1 is the actual DockQ of the highest predicted structure; top-5/top-10 are the best actual DockQ among the top predicted candidates (assuming a subsequent way to identify the best). Regret is `best available DockQ − top-1 DockQ`; lower is better. Prediction ties use graph-name ordering.

The random-selection reference is the mean true DockQ within each target. The constant-mean MSE below uses the **test mean only as a descriptive reference**, not a deployable baseline fitted on test labels.''')
code('''def correlation(y, p, rank=False):
    if len(y) < 2 or np.std(y) == 0 or np.std(p) == 0:
        return np.nan
    return float((spearmanr if rank else pearsonr)(y, p).statistic)

per_target_rows, run_rows = [], []
for (config, seed), frame in predictions.items():
    start = len(per_target_rows)
    for target, group in frame.groupby('target_id', sort=True):
        y, p = group.true_target.to_numpy(), group.predicted_target.to_numpy()
        order = np.argsort(-p, kind='stable')
        per_target_rows.append(dict(config=config, seed=seed, target=target, n=len(group),
            mse=np.mean((p-y)**2), mae=np.mean(abs(p-y)), bias=np.mean(p-y),
            spearman=correlation(y,p,True), pearson=correlation(y,p),
            top1=y[order[0]], top5=y[order[:5]].max(), top10=y[order[:10]].max(),
            oracle=y.max(), random=y.mean(), regret=y.max()-y[order[0]]))
    macro = pd.DataFrame(per_target_rows[start:])
    y, p = frame.true_target.to_numpy(), frame.predicted_target.to_numpy()
    mse = np.mean((p-y)**2)
    run_rows.append(dict(config=config, seed=seed, mse=mse, rmse=np.sqrt(mse),
        mae=np.mean(abs(p-y)), r2=1-mse/np.var(y), pearson=correlation(y,p),
        spearman=correlation(y,p,True), bias=np.mean(p-y), macro_mse=macro.mse.mean(),
        macro_spearman=macro.spearman.mean(), defined_target_correlations=macro.spearman.notna().sum(),
        top1=macro.top1.mean(), top5=macro.top5.mean(), top10=macro.top10.mean(),
        regret=macro.regret.mean(), oracle=macro.oracle.mean(), random=macro.random.mean(),
        outside_0_1=int(((p<0)|(p>1)).sum())))
per_target = pd.DataFrame(per_target_rows)
metrics = pd.DataFrame(run_rows)
display(metrics.round(4))
export(metrics, 'metrics_by_run')
export(per_target, 'metrics_by_target_and_seed')
summary_columns = ['rmse','mae','r2','macro_mse','macro_spearman','top1','top5','regret','bias']
seed_summary = metrics.groupby('config')[summary_columns].agg(['mean','std'])
display(seed_summary.round(4))
seed_summary.to_csv(EXPORT / 'seed_mean_and_sd.csv')
print('SD is variation across three training seeds on the same split, not three independent test cohorts.')
print('Descriptive test-mean constant MSE:', np.var(reference.true_target))''')
code('''fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
for ax, metric, title in zip(axes, ['macro_mse','macro_spearman','top1'],
                           ['Target-macro MSE ↓','Within-target Spearman ↑','Top-1 true DockQ ↑']):
    for seed in SEEDS:
        values = [metrics.loc[(metrics.config==c)&(metrics.seed==seed),metric].iloc[0] for c in CONFIGS]
        ax.plot([0,1], values, color='0.75', lw=1)
        for x,c,value in zip([0,1], CONFIGS, values):
            ax.scatter(x,value,color=COLORS[c],s=45,zorder=3)
        ax.annotate(f'seed {seed}', (1,values[1]), xytext=(6,0), textcoords='offset points',fontsize=8)
    ax.set(xticks=[0,1], xticklabels=['Base','Full'], title=title, xlim=(-.25,1.65))
savefig(fig, 'paired_seed_metrics')''')
md('## Seed-by-seed bar comparison\n\nBoth groups use the four-layer residual GAT. Black bars show the original base features; gray bars add interface-node degree and Voronoi area/mask. Each pair is one training seed on the identical test cohort.\n\nBoth panels give each target equal weight: the left is the mean within-target Spearman correlation (higher is better), and the right is mean target MSE (lower is better). These are not pooled correlations across unrelated targets. Values are individual runs, so no seed error bars are shown.')
code('# Match the grouped black/gray comparison style, with slide-readable text.\n# These are target-macro metrics; the notebook also reports pooled metrics above.\nimport shutil\nif shutil.which("latex") is None:\n    raise RuntimeError("LaTeX is required to render this figure in Computer Modern.")\n\nwith mpl.rc_context({\n    "text.usetex": True,\n    "font.family": "serif",\n    "font.serif": ["Computer Modern Roman"],\n    "mathtext.fontset": "cm",\n    "axes.spines.top": True,\n    "axes.spines.right": True,\n    "axes.labelsize": 22,\n    "xtick.labelsize": 18,\n    "ytick.labelsize": 18,\n    "legend.fontsize": 18,\n}):\n    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), constrained_layout=True)\n    positions = np.arange(len(SEEDS))\n    bar_width = 0.36\n    bar_groups = [\n        ("base", "Base features", "black"),\n        ("full", "+ Interface degree + Voronoi area", "lightgray"),\n    ]\n    panels = [\n        ("macro_spearman", r"Mean within-target Spearman $\\rho$", "{:.3f}"),\n        ("macro_mse", "Target-average DockQ MSE", "{:.4f}"),\n    ]\n    for ax, (metric, ylabel, value_format) in zip(axes, panels):\n        for group_index, (config, legend_label, color) in enumerate(bar_groups):\n            subset = metrics.loc[metrics.config.eq(config)].set_index("seed").loc[SEEDS]\n            values = subset[metric].to_numpy(float)\n            bars = ax.bar(positions + (group_index - 0.5) * bar_width, values,\n                          bar_width, color=color, edgecolor="black", linewidth=1.2,\n                          label=legend_label)\n            ax.bar_label(bars, labels=[value_format.format(v) for v in values],\n                         padding=5, fontsize=16)\n        ymax = metrics[metric].max()\n        ymin = min(0.0, metrics[metric].min() * 1.15)\n        ax.set_ylim(ymin, ymax * 1.22)\n        ax.set_xticks(positions, [f"Seed {seed}" for seed in SEEDS])\n        ax.set_xlabel("Training seed", labelpad=10)\n        ax.set_ylabel(ylabel, labelpad=10)\n        ax.tick_params(axis="both", width=1.1, length=5)\n        ax.yaxis.get_offset_text().set_fontsize(18)\n        for spine in ax.spines.values():\n            spine.set_visible(True)\n            spine.set_linewidth(1.1)\n    handles, labels = axes[0].get_legend_handles_labels()\n    fig.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False)\n    fig.savefig(EXPORT / "seed_spearman_mse_bars.pdf", bbox_inches="tight")\n    fig.savefig(EXPORT / "seed_spearman_mse_bars.png", dpi=200, bbox_inches="tight")\n    plt.show()')
md('## Best-case per-target selection\n\nCompare base seed 17 with the best Spearman correlation available for each target across all six runs (base/full × seeds 7, 17, 27). Both bars average targets equally. The oracle selects runs using test labels: it is a retrospective best-case benchmark, not an unbiased estimate for a deployable model.\n\nZRank2 is shown at the user-provided reference value of 0.775; it is not recomputed from the seed replicate predictions.\n')
code('# Compare the best single run with a retrospective per-target oracle.\nordered = per_target.sort_values(["target", "config", "seed"]).copy()\nassert not ordered.duplicated(["target", "config", "seed"]).any()\nassert ordered.groupby("target").size().eq(len(CONFIGS) * len(SEEDS)).all()\nassert np.isfinite(ordered["spearman"]).all(), "Undefined correlations require explicit cohort handling."\nwinners = ordered.loc[ordered.groupby("target")["spearman"].idxmax()].copy()\nbase17 = ordered.loc[ordered.config.eq("base") & ordered.seed.eq(17)]\nassert set(base17.target) == set(winners.target)\nbase_mean = base17.spearman.mean()\noracle_mean = winners.spearman.mean()\ncomparison = pd.DataFrame({\n    "selection": ["Base seed 17", "Per-target oracle (test-label selection)", "ZRank2"],\n    "mean_target_spearman": [base_mean, oracle_mean, 0.775],\n    "targets": [len(base17), len(winners), None],\n    "source": ["Seed replicate predictions", "Seed replicate predictions", "User-provided reference"],\n})\ncomparison.to_csv(EXPORT / "oracle_vs_base_seed17_spearman.csv", index=False)\n\nwith mpl.rc_context({\n    "text.usetex": True,\n    "font.family": "serif",\n    "font.serif": ["Computer Modern Roman"],\n    "mathtext.fontset": "cm",\n    "axes.labelsize": 24,\n    "xtick.labelsize": 21,\n    "ytick.labelsize": 20,\n}):\n    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)\n    bars = ax.bar([0, 1, 2], [base_mean, oracle_mean, 0.775], width=0.62,\n                  color=["black", "lightgray", "gray"], edgecolor="black", linewidth=1.2)\n    ax.bar_label(bars, fmt="%.3f", padding=8, fontsize=23)\n    ax.set_xticks([0, 1, 2], ["Base seed 17", r"Hand-picked best $\\rho$", "ZRank2"])\n    ax.set_ylabel(r"$\\rho$", labelpad=12)\n    ax.set_ylim(0, 1)\n    ax.set_xlim(-0.6, 2.6)\n    ax.set_yticks(np.arange(0, 1.01, 0.2))\n    ax.tick_params(axis="both", width=1.1, length=5, pad=8)\n    for spine in ax.spines.values():\n        spine.set_visible(True)\n        spine.set_linewidth(1.1)\n    fig.savefig(EXPORT / "oracle_vs_base_seed17_spearman.pdf", bbox_inches="tight")\n    fig.savefig(EXPORT / "oracle_vs_base_seed17_spearman.png", dpi=200, bbox_inches="tight")\n    plt.show()\nprint(f"{len(winners)} targets; oracle improvement: {oracle_mean - base_mean:+.3f}")\n')
md('''## Paired uncertainty across targets

For each target, average each metric over the three seeds **before** computing the base/full difference. Bootstrap targets together across configurations (10,000 draws). Positive values always mean improvement with full features. These intervals describe target sampling uncertainty conditional on this split and these three seeds; they do not treat decoys or seeds as independent test samples. Intervals are pointwise, with no multiple-comparison correction. Target-macro MSE is the main score-error comparison; other metrics are diagnostic.''')
code('''target_mean = per_target.groupby(['config','target']).mean(numeric_only=True)
base = target_mean.loc['base'].sort_index()
full = target_mean.loc['full'].sort_index()
assert base.index.equals(full.index)
deltas = pd.DataFrame(index=base.index)
for metric in ['mse','mae','spearman','top1','top5','regret']:
    deltas[metric] = full[metric]-base[metric] if metric in ['spearman','top1','top5'] else base[metric]-full[metric]
rng = np.random.default_rng(123)
indices = rng.integers(0, len(deltas), size=(10000,len(deltas)))
paired_rows = []
for metric in deltas:
    values = deltas[metric].to_numpy()
    if not np.isfinite(values).all():
        raise ValueError(f'Undefined {metric}; explicitly define the paired target cohort before bootstrapping')
    boot = values[indices].mean(axis=1)
    paired_rows.append(dict(metric=metric, mean_improvement=values.mean(),
        ci95_low=np.quantile(boot,.025), ci95_high=np.quantile(boot,.975),
        improved_targets=int((values>0).sum()), tied_targets=int((values==0).sum()), targets=len(values)))
paired = pd.DataFrame(paired_rows)
display(paired.round(5))
export(paired, 'paired_target_bootstrap')
export(deltas.reset_index(), 'paired_target_improvements')
fig, ax = plt.subplots(figsize=(10,4), constrained_layout=True)
ordered = deltas.mse.sort_values()
ax.bar(ordered.index, ordered, color=np.where(ordered>0,COLORS['full'],'#c65a46'))
ax.axhline(0,color='black',lw=.8)
ax.set(ylabel='Base MSE − full MSE',title='Per-target change, averaged over three seeds')
ax.tick_params(axis='x',rotation=65)
savefig(fig,'target_mse_improvements')''')
md('''## Training and checkpoint selection

The exported predictions come from the **best validation checkpoint**, not necessarily epoch 50. Vertical lines mark the checkpoint epoch from saved metadata. Training improvement combined with worsening validation performance is consistent with overfitting; these histories alone cannot establish whether adaptive weighting caused it. The old runs recorded test losses during training; we do not use those curves for model selection here.''')
code('''fig, axes = plt.subplots(2,3,figsize=(13,7),sharex=True,sharey=True,constrained_layout=True)
for row,config in enumerate(CONFIGS):
    for col,seed in enumerate(SEEDS):
        ax=axes[row,col]; h=histories[config,seed]
        best=metadata[f'gat4_residual_{config}_seed{seed}']['best_epoch']
        ax.plot(h.epoch,h.train_target_mse,label='Train',color='#444444')
        ax.plot(h.epoch,h.val_target_mse,label='Validation',color=COLORS['full'])
        ax.axvline(best,color='#c65a46',ls='--',lw=1,label=f'Checkpoint: {best}')
        ax.set(title=f'{LABELS[config]} · seed {seed}',xlabel='Epoch',ylabel='DockQ MSE')
        ax.legend(fontsize=8)
savefig(fig,'train_validation_dockq')
training_rows=[]
for (config,seed),h in histories.items():
    training_rows.append(dict(config=config,seed=seed,train_mse_first=h.train_target_mse.iloc[0],
        train_mse_last=h.train_target_mse.iloc[-1],val_mse_first=h.val_target_mse.iloc[0],
        val_mse_last=h.val_target_mse.iloc[-1],train_mse_reduction_percent=100*(1-h.train_target_mse.iloc[-1]/h.train_target_mse.iloc[0])))
training_summary=pd.DataFrame(training_rows)
display(training_summary.round(5)); export(training_summary,'training_change')''')
md('''### Why can the total training loss look flat?

With adaptive loss shares, weights change during optimization. A nearly constant weighted sum does **not** imply constant raw reconstruction/regression errors. This diagnostic uses the logged raw losses, weights, and weighted contributions; it does not reconstruct epoch-average contributions by multiplying two epoch averages.

Change `DIAGNOSTIC_CONFIG` and `DIAGNOSTIC_SEED` to inspect another run.''')
code('''DIAGNOSTIC_CONFIG, DIAGNOSTIC_SEED = 'full', 7
h=histories[DIAGNOSTIC_CONFIG,DIAGNOSTIC_SEED]
terms=['node_mse','edge_attr_mse','edge_presence_bce','target_mse']
fig,axes=plt.subplots(1,3,figsize=(13,3.7),constrained_layout=True)
for term in terms:
    axes[0].plot(h.epoch,h['train_'+term],label=term)
    axes[1].plot(h.epoch,h['weight_'+term],label=term)
    axes[2].plot(h.epoch,h['train_weighted_'+term],label=term)
axes[2].plot(h.epoch,h.train_loss,color='black',lw=2,label='Total')
for ax,title in zip(axes,['Raw training losses','Adaptive weights','Weighted contributions']):
    ax.set(xlabel='Epoch',title=title); ax.legend(fontsize=7)
axes[0].set_yscale('log'); axes[1].set_yscale('log')
fig.suptitle(f'{LABELS[DIAGNOSTIC_CONFIG]} · seed {DIAGNOSTIC_SEED}')
savefig(fig,f'loss_diagnostics_{DIAGNOSTIC_CONFIG}_seed{DIAGNOSTIC_SEED}')''')
md('''## Prediction spread and calibration

All six panels share axes. The diagonal is perfect agreement. The binned plot groups by **true DockQ**, exposing underprediction of high-quality structures; it is a residual diagnostic, not a reliability plot binned by predicted score. Predictions are never clipped for metrics.''')
code('''low=min(0.,min(f.predicted_target.min() for f in predictions.values()))-.02
high=max(1.,max(f.predicted_target.max() for f in predictions.values()))+.02
fig,axes=plt.subplots(2,3,figsize=(12,7),sharex=True,sharey=True,constrained_layout=True)
for row,c in enumerate(CONFIGS):
    for col,s in enumerate(SEEDS):
        ax=axes[row,col];f=predictions[c,s]
        ax.hexbin(f.true_target,f.predicted_target,gridsize=55,bins='log',mincnt=1,cmap='viridis',rasterized=True)
        ax.plot([0,1],[0,1],'k--',lw=.8)
        ax.set(xlim=(low,high),ylim=(low,high),xlabel='True DockQ',ylabel='Predicted DockQ',title=f'{LABELS[c]} · seed {s}')
savefig(fig,'observed_predicted_all_runs')
calibration=[]
for (c,s),f in predictions.items():
    for interval,g in f.groupby(pd.cut(f.true_target,np.linspace(0,1,11),include_lowest=True),observed=True):
        calibration.append(dict(config=c,seed=s,bin=str(interval),n=len(g),true_mean=g.true_target.mean(),
            predicted_mean=g.predicted_target.mean(),bias=(g.predicted_target-g.true_target).mean()))
calibration=pd.DataFrame(calibration);export(calibration,'true_dockq_bins')
fig,ax=plt.subplots(figsize=(6,4.5),constrained_layout=True)
for c in CONFIGS:
    for i,s in enumerate(SEEDS):
        g=calibration[(calibration.config==c)&(calibration.seed==s)]
        ax.plot(g.true_mean,g.predicted_mean,color=COLORS[c],marker='o',alpha=.65,ls=['-','--',':'][i],label=f'{LABELS[c]} seed {s}')
ax.plot([0,1],[0,1],'k--',lw=.8)
ax.set(xlabel='Mean true DockQ in bin',ylabel='Mean predicted DockQ',xlim=(0,1),ylim=(low,high))
ax.legend(fontsize=8);savefig(fig,'binned_prediction_bias')''')
md('''## Top-ranked candidate selection

Good correlation across all decoys can coexist with poor top-1 selection. These curves summarize the best true DockQ among the top *k* predictions for each target, averaged equally across targets. The random reference applies only to selecting **one** structure; the oracle shows the best available structure per target.''')
code('''fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
ks=[1,5,10]
for c in CONFIGS:
    values=np.array([[metrics.loc[(metrics.config==c)&(metrics.seed==s),f'top{k}'].iloc[0] for k in ks] for s in SEEDS])
    ax.errorbar(ks,values.mean(0),yerr=values.std(0,ddof=1),color=COLORS[c],marker='o',capsize=4,label=LABELS[c]+' (mean ± seed SD)')
ax.axhline(metrics.random.iloc[0],ls=':',color='0.5',label='Random single selection')
ax.axhline(metrics.oracle.iloc[0],ls='--',color='0.3',label='Best available')
ax.set(xticks=ks,xlabel='Number of top-ranked candidates inspected',ylabel='Target-macro best true DockQ',ylim=(0,1))
ax.legend(fontsize=8);savefig(fig,'top_k_selection')
selection_table=per_target.groupby(['config','target'])[['spearman','top1','top5','oracle','regret']].mean().unstack('config')
display(selection_table.round(3))
selection_table.to_csv(EXPORT/'selection_by_target.csv')''')
md('''## Target-by-target prediction panels

Choose a seed to compare base and full predictions for all held-out targets. In-panel annotations report both within-target Spearman correlations. Common axes expose target-specific calibration failures. The plots use all available predictions, with rasterized points in PDF exports.''')
code('PANEL_SEED = 17\ncomparison_configs = ("base", "full")\ncomparison_colors = {"base": "black", "full": "#247ba0"}\ncomparison_labels = {"base": "Base features", "full": "Full features (interface degree + Voronoi)"}\ntargets = sorted(reference.target_id.unique())\nncols = 5\nnrows = int(np.ceil(len(targets) / ncols))\naxis_low = min(0.0, *(predictions[c, PANEL_SEED].predicted_target.min() for c in comparison_configs)) - 0.02\naxis_high = max(1.0, *(predictions[c, PANEL_SEED].predicted_target.max() for c in comparison_configs)) + 0.02\n\nwith mpl.rc_context({\n    "text.usetex": True,\n    "font.family": "serif",\n    "font.serif": ["Computer Modern Roman"],\n    "mathtext.fontset": "cm",\n    "axes.spines.top": True,\n    "axes.spines.right": True,\n    "axes.labelsize": 24,\n    "xtick.labelsize": 20,\n    "ytick.labelsize": 20,\n}):\n    fig, axes = plt.subplots(nrows, ncols, figsize=(27, 5.2 * nrows),\n                             sharex=True, sharey=True, constrained_layout=True)\n    for ax, target in zip(np.atleast_1d(axes).flat, targets):\n        rhos = {}\n        for config in comparison_configs:\n            frame = predictions[config, PANEL_SEED]\n            group = frame.loc[frame.target_id.eq(target)]\n            rhos[config] = correlation(group.true_target.to_numpy(), group.predicted_target.to_numpy(), rank=True)\n            ax.scatter(group.true_target, group.predicted_target,\n                       color=comparison_colors[config], marker="o", alpha=0.30, s=22,\n                       edgecolors="none", rasterized=True, label=comparison_labels[config])\n        ax.plot([axis_low, axis_high], [axis_low, axis_high],\n                color="0.45", linestyle="--", linewidth=1.3)\n        annotation = (target + "\\n" + rf"$\\rho_{{\\mathrm{{base}}}} = {rhos[\'base\']:.3f}$"\n                      + "\\n" + rf"$\\rho_{{\\mathrm{{full}}}} = {rhos[\'full\']:.3f}$")\n        ax.text(0.04, 0.96, annotation, transform=ax.transAxes,\n                ha="left", va="top", fontsize=23,\n                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=2))\n        ax.set(xlim=(axis_low, axis_high), ylim=(axis_low, axis_high),\n               xlabel="Observed DockQ", ylabel="Predicted DockQ")\n        ax.set_xticks([0, 0.5, 1.0])\n        ax.set_yticks([0, 0.5, 1.0])\n        ax.set_aspect("equal", adjustable="box")\n        ax.tick_params(labelbottom=True, labelleft=True, length=6, width=1.3, pad=6)\n        ax.xaxis.labelpad = 9\n        ax.yaxis.labelpad = 9\n        for spine in ax.spines.values():\n            spine.set_visible(True)\n            spine.set_linewidth(1.3)\n    for ax in list(np.atleast_1d(axes).flat)[len(targets):]:\n        ax.set_visible(False)\n    handles, labels = axes.flat[0].get_legend_handles_labels()\n    legend = fig.legend(handles, labels, loc="outside upper center", ncol=2,\n                        fontsize=24, frameon=False, markerscale=2)\n    for handle in legend.legend_handles:\n        handle.set_alpha(1.0)\n    fig.savefig(EXPORT / f"prediction_panels_seed{PANEL_SEED}.pdf", bbox_inches="tight")\n    fig.savefig(EXPORT / f"prediction_panels_seed{PANEL_SEED}.png", dpi=180, bbox_inches="tight")\n    plt.show()')
md('## Base seed 17: observed versus predicted DockQ by target\n\nThe requested base seed-17 model is shown for every held-out target. Each translucent black circle is one structure. All panels share axis limits, the dashed line indicates perfect agreement, and the in-panel annotation identifies the target and its Spearman correlation. This is the selected model for visualization; selection here does not establish that it is best on every metric.')
code('SELECTED_CONFIG, SELECTED_SEED = "base", 17\nselected_predictions = predictions[SELECTED_CONFIG, SELECTED_SEED]\nselected_targets = sorted(selected_predictions.target_id.unique())\nncols = 5\nnrows = int(np.ceil(len(selected_targets) / ncols))\naxis_low = min(0.0, selected_predictions.true_target.min(), selected_predictions.predicted_target.min()) - 0.02\naxis_high = max(1.0, selected_predictions.true_target.max(), selected_predictions.predicted_target.max()) + 0.02\n\nwith mpl.rc_context({\n    "text.usetex": True,\n    "font.family": "serif",\n    "font.serif": ["Computer Modern Roman"],\n    "mathtext.fontset": "cm",\n    "axes.spines.top": True,\n    "axes.spines.right": True,\n    "axes.labelsize": 24,\n    "xtick.labelsize": 20,\n    "ytick.labelsize": 20,\n}):\n    fig, axes = plt.subplots(nrows, ncols, figsize=(27, 5.2 * nrows),\n                             sharex=True, sharey=True, constrained_layout=True)\n    for ax, target in zip(np.atleast_1d(axes).flat, selected_targets):\n        group = selected_predictions.loc[selected_predictions.target_id.eq(target)]\n        rho = correlation(group.true_target.to_numpy(), group.predicted_target.to_numpy(), rank=True)\n        ax.scatter(group.true_target, group.predicted_target,\n                   color="black", marker="o", alpha=0.30, s=22,\n                   edgecolors="none", rasterized=True)\n        ax.plot([axis_low, axis_high], [axis_low, axis_high],\n                color="0.45", linestyle="--", linewidth=1.3)\n        ax.text(0.04, 0.96, target + "\\n" + rf"$\\rho = {rho:.3f}$",\n                transform=ax.transAxes, ha="left", va="top", fontsize=23,\n                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=2))\n        ax.set(xlim=(axis_low, axis_high), ylim=(axis_low, axis_high),\n               xlabel="Observed DockQ", ylabel="Predicted DockQ")\n        ax.set_xticks([0, 0.5, 1.0])\n        ax.set_yticks([0, 0.5, 1.0])\n        ax.set_aspect("equal", adjustable="box")\n        ax.tick_params(labelbottom=True, labelleft=True, length=6, width=1.3, pad=6)\n        ax.xaxis.labelpad = 9\n        ax.yaxis.labelpad = 9\n        for spine in ax.spines.values():\n            spine.set_visible(True)\n            spine.set_linewidth(1.3)\n    for ax in list(np.atleast_1d(axes).flat)[len(selected_targets):]:\n        ax.set_visible(False)\n    fig.savefig(EXPORT / "base_seed17_dockq_by_target.pdf", bbox_inches="tight")\n    fig.savefig(EXPORT / "base_seed17_dockq_by_target.png", dpi=180, bbox_inches="tight")\n    plt.show()')
md('''## Optional three-seed ensemble

Averaging predictions is a **different predictor** from averaging metrics across training runs. This exploratory calculation averages all three seeds equally, without choosing a seed on test performance. It should not replace the individual-run comparison above.''')
code('''ensemble_rows=[]
for c in CONFIGS:
    f=predictions[c,SEEDS[0]].copy()
    f['predicted_target']=np.mean([predictions[c,s].predicted_target.to_numpy() for s in SEEDS],axis=0)
    target_rows=[]
    for t,g in f.groupby('target_id'):
        y,p=g.true_target.to_numpy(),g.predicted_target.to_numpy()
        order=np.argsort(-p,kind='stable')
        target_rows.append(dict(mse=np.mean((p-y)**2),rho=correlation(y,p,True),top1=y[order[0]],top5=y[order[:5]].max()))
    t=pd.DataFrame(target_rows)
    ensemble_rows.append(dict(config=c,rmse=np.sqrt(np.mean((f.predicted_target-f.true_target)**2)),
        macro_mse=t.mse.mean(),macro_spearman=t.rho.mean(),top1=t.top1.mean(),top5=t.top5.mean()))
ensemble=pd.DataFrame(ensemble_rows)
display(ensemble.round(4));export(ensemble,'three_seed_ensemble')''')
md('''## Readout and interpretation limits

The summary below is generated from the current inputs. Seed SD and target-bootstrap uncertainty answer different questions. With only three seeds and one fixed target split, neither establishes robustness to a different train/test split. The full-feature comparison changes multiple inputs together, and adaptive weighting is shared by both groups: neither the benefit of Voronoi area alone nor the causal effect of loss weighting is isolated.''')
code('''means=metrics.groupby('config').mean(numeric_only=True)
mse_row=paired.set_index('metric').loc['mse']
relative=100*(means.loc['base','macro_mse']-means.loc['full','macro_mse'])/means.loc['base','macro_mse']
selected_epochs=', '.join(f"{r.config} seed {r.seed}: {r.checkpoint_epoch}" for r in inventory.itertuples())
readout=f"""**Full versus base, averaged over three seeds:**

- Target-macro MSE: **{means.loc['base','macro_mse']:.4f} → {means.loc['full','macro_mse']:.4f}** ({relative:+.1f}% reduction).
- Paired MSE improvement: **{mse_row.mean_improvement:.4f}**, target-bootstrap 95% interval **[{mse_row.ci95_low:.4f}, {mse_row.ci95_high:.4f}]**.
- Within-target Spearman: **{means.loc['base','macro_spearman']:.3f} → {means.loc['full','macro_spearman']:.3f}**.
- Top-1 true DockQ: **{means.loc['base','top1']:.3f} → {means.loc['full','top1']:.3f}**; random single-selection reference **{means.loc['base','random']:.3f}**.
- Selected checkpoints: {selected_epochs}.

The raw training losses and validation curves are more informative than the adaptive weighted total alone. Use the validation-selected predictions as saved; do not pick a later epoch or favorable seed based on this test analysis.
"""
display(Markdown(readout))
(EXPORT/'readout.md').write_text(readout)
print('Saved CSV tables and PNG/PDF figures to',EXPORT)''')

notebook = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec': {'display_name':'Python 3','language':'python','name':'python3'}, 'language_info': {'name':'python'}})
path = Path(__file__).with_name('analyze_seed_replicates.ipynb')
nbf.write(notebook, path)
print(path)

if __name__ == '__main__':
    pass
