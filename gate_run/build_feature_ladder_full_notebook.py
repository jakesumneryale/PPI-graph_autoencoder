"""Build the analysis of the corrected 5-rung full-dataset feature ladder:
core -> +degree -> +rsasa_i_node -> +voronoi -> +v_es electrostatics."""
from pathlib import Path
import nbformat as nbf

cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip()))

md(r'''
# Feature ladder (full dataset): core $\to$ +degree $\to$ +rSASA$_i$ $\to$ +Voronoi $\to$ +electrostatics

Fifteen runs: a cumulative five-rung feature ladder $\times$ three seeds (7, 17, 27) on the **full dataset**, all
four-layer residual GAT (`gat4_residual`). This is the corrected ladder that the 10% baseline notebook
(`analyze_feature_ladder_10pct.ipynb`) was waiting on:

| rung | node features | edge features |
|---|---|---|
| `core` | `aa_type, chain, interface_nodes` | `interface_edges, ca_dist` |
| `degree` | + `interface_node_degree` | |
| `rsasa` | + `rsasa_i_node` (genuine per-residue) | |
| `voronoi` | | + `voronoi_contact_area, voronoi_contact_missing` |
| `electrostatics` | | + `v_es` |

The edge-reconstruction target is `interface_edges, ca_dist` in every rung, so reconstruction metrics are
comparable across the ladder. The feature lists above are read back from each run's own
`reconstruction_summary.csv` and checked below rather than assumed.

Figures use LaTeX Computer Modern, large labels, full borders and no titles. Error bars and bands are **sample
standard deviation across three seeds**, not confidence intervals. Where a paired uncertainty is shown it is a
**bootstrap over the 29 test targets**, which is the honest unit of replication here (decoys within a target
are not independent).
''')

code(r'''
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

GATE = Path.cwd() if (Path.cwd() / 'feature_ladder_full').is_dir() else Path.cwd() / 'gate_run'
ROOT = GATE / 'feature_ladder_full'
ROOT_10PCT = GATE / 'feature_ladder_10pct'
EXPORT = GATE / 'feature_ladder_full_analysis'; EXPORT.mkdir(exist_ok=True)

RUNGS = ['core', 'degree', 'rsasa', 'voronoi', 'electrostatics']
SEEDS = [7, 17, 27]
RUNG_LABEL = {'core': 'core', 'degree': '+degree', 'rsasa': '+rSASA$_i$', 'voronoi': '+Voronoi',
              'electrostatics': '+$V_{\\mathrm{es}}$'}
RUNG_COLOR = {'core': '#0072B2', 'degree': '#56B4E9', 'rsasa': '#009E73', 'voronoi': '#D55E00',
              'electrostatics': '#CC79A7'}
EXPECTED = {
    'core': ('aa_type,chain,interface_nodes', 'interface_edges,ca_dist'),
    'degree': ('aa_type,chain,interface_nodes,interface_node_degree', 'interface_edges,ca_dist'),
    'rsasa': ('aa_type,chain,interface_nodes,interface_node_degree,rsasa_i_node', 'interface_edges,ca_dist'),
    'voronoi': ('aa_type,chain,interface_nodes,interface_node_degree,rsasa_i_node',
                'interface_edges,ca_dist,voronoi_contact_area,voronoi_contact_missing'),
    'electrostatics': ('aa_type,chain,interface_nodes,interface_node_degree,rsasa_i_node',
                       'interface_edges,ca_dist,voronoi_contact_area,voronoi_contact_missing,v_es'),
}

mpl.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.serif': ['Computer Modern Roman'],
    'mathtext.fontset': 'cm', 'axes.labelsize': 23, 'xtick.labelsize': 17, 'ytick.labelsize': 17,
    'legend.fontsize': 17, 'axes.spines.top': True, 'axes.spines.right': True, 'axes.linewidth': 1.1})

def save(fig, name):
    fig.savefig(EXPORT / (name + '.pdf'), bbox_inches='tight')
    fig.savefig(EXPORT / (name + '.png'), dpi=180, bbox_inches='tight')
    plt.show()

def run_dir(rung, seed, root=ROOT):
    return root / f'gat4_residual_{rung}_seed{seed}'

history, summaries, predictions, recon_targets = [], [], {}, []
for rung in RUNGS:
    for seed in SEEDS:
        d = run_dir(rung, seed)
        h = pd.read_csv(d / 'loss_history.csv'); h['rung'] = rung; h['seed'] = seed; history.append(h)
        s = pd.read_csv(d / 'reconstruction_summary.csv').iloc[0].to_dict(); s['rung'] = rung; s['seed'] = seed
        if (s['node_features'], s['edge_features']) != EXPECTED[rung]:
            raise ValueError(f'{d.name}: features {s["node_features"]} | {s["edge_features"]} do not match rung')
        if s['edge_recon_features'] != 'interface_edges,ca_dist':
            raise ValueError(f'{d.name}: unexpected edge reconstruction target {s["edge_recon_features"]}')
        summaries.append(s)
        rt = pd.read_csv(d / 'reconstruction_by_target.csv'); rt['rung'] = rung; rt['seed'] = seed
        recon_targets.append(rt)
        predictions[rung, seed] = pd.read_csv(d / 'test_predictions.csv')
history = pd.concat(history, ignore_index=True)
summary = pd.DataFrame(summaries)
recon_targets = pd.concat(recon_targets, ignore_index=True)

# Every run must score the identical test cohort with identical labels, or paired comparisons are invalid.
reference = predictions['core', 7][['target_id', 'graph_name', 'true_target']]
for key, p in predictions.items():
    if not (p.target_id.equals(reference.target_id) and p.graph_name.equals(reference.graph_name)
            and np.allclose(p.true_target, reference.true_target)):
        raise ValueError(f'{key}: test cohort or labels differ from core seed 7')
    if not np.isfinite(p.predicted_target).all():
        raise ValueError(f'{key}: nonfinite predictions')
if reference.duplicated(['target_id', 'graph_name']).any():
    raise ValueError('duplicate (target_id, graph_name) keys in test cohort')

print(f'{len(summary)} runs, {history.epoch.max()} epochs each, {len(history)} loss-history rows.')
print(f'Exact paired test cohort: {len(reference):,} decoys / {reference.target_id.nunique()} targets, '
      f'identical across all runs.')
''')

md(r'''## Reference baselines

Before comparing rungs it helps to know what "no skill" looks like on this test cohort. A constant predictor at
the test-set mean DockQ has MSE equal to the test-set variance; a per-target constant does essentially no
better, because target-mean DockQ barely varies across these 29 targets. All of the useful signal is therefore
**within-target ranking**, which is why macro Spearman $\langle\rho\rangle_t$ is the headline metric alongside MSE.''')

code(r'''
y_all = reference.true_target.to_numpy()
MEAN_MSE = np.var(y_all)
TARGET_MEAN_MSE = ((reference.true_target - reference.groupby('target_id').true_target.transform('mean')) ** 2).mean()
target_stats = reference.groupby('target_id').true_target.agg(n='size', mean='mean', sd='std', best='max')
RANDOM_TOP1 = target_stats['mean'].mean(); ORACLE_TOP1 = target_stats['best'].mean()
print(f'Constant (global-mean) predictor MSE: {MEAN_MSE:.4f}')
print(f'Constant per-target-mean predictor MSE: {TARGET_MEAN_MSE:.4f}')
print(f'Top-1 DockQ: random pick {RANDOM_TOP1:.3f}, oracle {ORACLE_TOP1:.3f} (target-macro).')
display(target_stats.describe().round(3))
''')

md(r'''## Model selection

Checkpoints are selected on minimum validation `target_mse`; the test-split `target_mse` is read off at that
*same* epoch (test is logged every epoch for monitoring only).''')

code(r'''
best_rows = []
for (rung, seed), g in history.groupby(['rung', 'seed']):
    b = g.loc[g.val_target_mse.idxmin()]
    best_rows.append(dict(rung=rung, seed=seed, best_epoch=int(b.epoch), val_target_mse=b.val_target_mse,
                          test_target_mse_logged=b.test_target_mse,
                          val_target_mse_final=g.iloc[-1].val_target_mse,
                          train_target_mse_final=g.iloc[-1].train_target_mse))
best = pd.DataFrame(best_rows)
best['rung'] = pd.Categorical(best.rung, RUNGS, ordered=True)
best = best.sort_values(['rung', 'seed']).reset_index(drop=True)
display(best.round(4))
print(f'Selected epochs across the fifteen runs: {sorted(best.best_epoch)} (of {history.epoch.max()}).')
''')

md(r'''**Every run selects epoch 1 or 2.** The DockQ head is at its best almost immediately and gets worse for
the remaining ~48 epochs (final-epoch validation MSE is roughly 1.5$\times$ the selected value, and above the
constant-predictor baseline). So everything that follows compares *very early* checkpoints, whose
reconstruction quality is also far from converged. The training-dynamics section below shows why; it is the
single most important caveat on this run.''')

md(r'''## Per-run test metrics

Computed from `test_predictions.csv` at the selected checkpoint. Pooled metrics weight every decoy equally;
**macro metrics weight each target equally** (computed within target, then averaged). Top-1 is the true DockQ
of the decoy the model ranks first; top-5 is the best true DockQ among the top five; regret is best-available
minus top-1 (lower is better).''')

code(r'''
def target_metrics(g):
    y = g.true_target.to_numpy(); p = g.predicted_target.to_numpy()
    order = np.lexsort((g.graph_name.to_numpy(), -p))
    return pd.Series(dict(mse=np.mean((p - y) ** 2), rho=spearmanr(y, p).statistic,
                          top1=y[order[0]], top5=y[order[:5]].max(), top10=y[order[:10]].max(),
                          best=y.max(), regret=y.max() - y[order[0]]))

per_target, metric_rows = [], []
for (rung, seed), p in predictions.items():
    t = p.groupby('target_id').apply(target_metrics, include_groups=False)
    t['rung'] = rung; t['seed'] = seed; per_target.append(t.reset_index())
    y = p.true_target.to_numpy(); yh = p.predicted_target.to_numpy(); mse = np.mean((yh - y) ** 2)
    metric_rows.append(dict(rung=rung, seed=seed, mse=mse, rmse=np.sqrt(mse), mae=np.mean(abs(yh - y)),
        r2=1 - mse / np.var(y), pearson=pearsonr(y, yh).statistic, spearman=spearmanr(y, yh).statistic,
        bias=np.mean(yh - y), pred_sd=yh.std(), macro_mse=t.mse.mean(), macro_rho=t.rho.mean(),
        top1=t.top1.mean(), top5=t.top5.mean(), top10=t.top10.mean(), regret=t.regret.mean()))
per_target = pd.concat(per_target, ignore_index=True)
metrics = pd.DataFrame(metric_rows)
metrics = metrics.merge(best[['rung', 'seed', 'best_epoch']].astype({'rung': str}), on=['rung', 'seed'])
metrics['rung'] = pd.Categorical(metrics.rung, RUNGS, ordered=True)
metrics = metrics.sort_values(['rung', 'seed']).reset_index(drop=True)

check = metrics.merge(best.astype({'rung': str})[['rung', 'seed', 'test_target_mse_logged']].assign(
    rung=lambda d: pd.Categorical(d.rung, RUNGS, ordered=True)), on=['rung', 'seed'])
print('Max |MSE from predictions - MSE logged at selected epoch|:',
      f'{(check.mse - check.test_target_mse_logged).abs().max():.2e}')

metrics.to_csv(EXPORT / 'metrics_per_run.csv', index=False)
per_target.to_csv(EXPORT / 'metrics_per_target.csv', index=False)
display(metrics.round(4))

agg_cols = ['mse', 'r2', 'macro_rho', 'spearman', 'pearson', 'top1', 'top5', 'regret', 'bias', 'pred_sd']
agg = metrics.groupby('rung', observed=True)[agg_cols].agg(['mean', 'std'])
agg.to_csv(EXPORT / 'metrics_by_rung.csv')
display(agg.round(4))
''')

md(r'''## The ladder: does each added feature help?

Bars are the three-seed mean $\pm$ seed sd; dots are individual seeds. The dashed line on the MSE panel is the
constant-predictor baseline (test-set variance) -- bars above it are worse than predicting the mean for every
decoy. Right panel is macro within-target Spearman $\rho$ (higher is better).''')

code(r'''
x = np.arange(len(RUNGS)); labels = [RUNG_LABEL[r] for r in RUNGS]

def ladder_bars(ax, col, ylabel, fmt, colors, dot_style, label_fs=17, baseline=None, ylim_pad=1.35):
    values = agg[(col, 'mean')].to_numpy(); errors = agg[(col, 'std')].to_numpy()
    top = max((values + errors).max(), baseline or 0)
    ax.bar(x, values, color=[colors[r] for r in RUNGS], edgecolor='black', yerr=errors, capsize=6,
           error_kw=dict(lw=1.6))
    for i, r in enumerate(RUNGS):
        sv = metrics.loc[metrics.rung == r, col].to_numpy()
        ax.scatter(i + np.linspace(-.13, .13, len(sv)), sv, zorder=6, **dot_style)
        ax.text(i, max(values[i] + errors[i], sv.max()) + .045 * top, fmt % values[i], ha='center',
                fontsize=label_fs, bbox=dict(facecolor='white', edgecolor='none', pad=1.5), zorder=7)
    if baseline is not None:
        ax.axhline(baseline, color='0.3', ls='--', lw=1.6, zorder=0, label='constant predictor')
        ax.legend(frameon=False, loc='upper right', fontsize=16)
    ax.set_xticks(x, labels, fontsize=16); ax.set_ylabel(ylabel)
    ax.set_ylim(0, top * ylim_pad)

fig, axes = plt.subplots(1, 2, figsize=(18, 6.8), constrained_layout=True)
dots = dict(color='black', s=26)
ladder_bars(axes[0], 'mse', 'Test DockQ MSE', '%.4f', RUNG_COLOR, dots, baseline=MEAN_MSE)
ladder_bars(axes[1], 'macro_rho', r'$\langle\rho\rangle_t$', '%.3f', RUNG_COLOR, dots)
save(fig, 'ladder_summary')
''')

md(r'''## Step-wise effect of each feature, paired by target

The bar chart mixes seed noise with target heterogeneity. Here each rung is compared with the rung immediately
before it **on the same targets**: for every target, average the metric over the three seeds, take the
difference (new rung minus previous), then bootstrap the 29 per-target differences (10,000 resamples) for a 95%
interval. Positive is an improvement in both panels (MSE differences are sign-flipped). The right-hand numbers
are the fraction of targets that improved.''')

code(r'''
rng = np.random.default_rng(0)
seed_mean = per_target.groupby(['rung', 'target_id'])[['mse', 'rho', 'top1']].mean()
steps = list(zip(RUNGS[:-1], RUNGS[1:])) + [('core', 'electrostatics')]
step_rows = []
for prev, new in steps:
    for metric, sign in [('rho', 1), ('mse', -1), ('top1', 1)]:
        d = sign * (seed_mean.loc[new, metric] - seed_mean.loc[prev, metric]).to_numpy()
        boots = rng.choice(d, size=(10_000, len(d)), replace=True).mean(1)
        step_rows.append(dict(step=f'{prev}->{new}', prev=prev, new=new, metric=metric, mean=d.mean(),
                              lo=np.percentile(boots, 2.5), hi=np.percentile(boots, 97.5),
                              frac_improved=(d > 0).mean()))
step_effects = pd.DataFrame(step_rows)
step_effects.to_csv(EXPORT / 'stepwise_paired_effects.csv', index=False)
display(step_effects.round(4))

STEP_LABEL = {f'{a}->{b}': (f'{RUNG_LABEL[b]} vs {RUNG_LABEL[a]}' if a != 'core' or b == 'degree'
              else f'{RUNG_LABEL[b]} vs core') for a, b in steps}
STEP_LABEL['core->degree'] = '+degree vs core'
fig, axes = plt.subplots(1, 2, figsize=(18, 6.2), constrained_layout=True, sharey=True)
ypos = np.arange(len(steps))[::-1]
for ax, metric, xlabel in [(axes[0], 'rho', r'$\Delta\langle\rho\rangle_t$ (new $-$ previous)'),
                           (axes[1], 'mse', r'$-\Delta$ MSE (previous $-$ new)')]:
    s = step_effects[step_effects.metric == metric]
    for yv, (_, row) in zip(ypos, s.iterrows()):
        col = RUNG_COLOR[row.new] if row.prev != 'core' or row.new == 'degree' else '0.25'
        ax.errorbar(row['mean'], yv, xerr=[[row['mean'] - row.lo], [row.hi - row['mean']]], fmt='o',
                    color=col, ms=11, capsize=6, lw=2.2, mec='black')
        ax.text(1.02, yv, f'{row.frac_improved * 29:.0f}/29', transform=ax.get_yaxis_transform(),
                va='center', fontsize=16)
    ax.axvline(0, color='black', lw=1.2)
    ax.axhline(.5, color='0.6', lw=1, ls=':')
    ax.set_xlabel(xlabel)
axes[0].set_yticks(ypos, [STEP_LABEL[f'{a}->{b}'] for a, b in steps], fontsize=17)
save(fig, 'stepwise_paired_effects')
''')

md(r'''## Training dynamics: why does every run stop at epoch 1--2?

Left: validation DockQ MSE by epoch (seed mean $\pm$ sd band per rung) against the constant-predictor baseline.
Middle: the matching **training** DockQ MSE on a log scale. Right: the DockQ term's weight in the loss as set by
the adaptive loss-share balancer. The pattern is classic early overfitting of the regression head: training
DockQ MSE keeps falling while validation MSE rises after epoch 1--2, and the balancer keeps *raising* the
DockQ weight as the training DockQ loss shrinks, which pushes the shared encoder further toward fitting the
training targets.''')

code(r'''
fig, axes = plt.subplots(1, 3, figsize=(24, 6.6), constrained_layout=True)
for rung in RUNGS:
    g = history[history.rung == rung].groupby('epoch')
    for ax, col in zip(axes, ['val_target_mse', 'train_target_mse', 'weight_target_mse']):
        s = g[col].agg(['mean', 'std'])
        ax.plot(s.index, s['mean'], color=RUNG_COLOR[rung], lw=2.2, label=RUNG_LABEL[rung])
        ax.fill_between(s.index, s['mean'] - s['std'], s['mean'] + s['std'], color=RUNG_COLOR[rung], alpha=.15)
axes[0].axhline(MEAN_MSE, color='0.3', ls='--', lw=1.6)
axes[0].set_ylabel('Validation DockQ MSE'); axes[1].set_ylabel('Training DockQ MSE')
axes[1].set_yscale('log'); axes[1].set_yticks([.003, .01, .03], ['0.003', '0.01', '0.03'])
axes[2].set_ylabel('DockQ loss weight')
for ax in axes: ax.set_xlabel('Epoch')
axes[0].legend(frameon=False, loc='lower right', ncol=2, fontsize=15)
save(fig, 'training_dynamics')

ep = history.groupby('epoch')[['train_target_mse', 'val_target_mse', 'weight_target_mse']].mean()
print('All-run mean at epochs 1, 2, 10, 50:'); display(ep.loc[[1, 2, 10, 50]].round(4))
''')

md(r'''Reconstruction losses on the validation split, by contrast, keep improving well past epoch 2 -- so the
checkpoint chosen for DockQ freezes the autoencoder long before its reconstruction has converged. (Node
MSE is averaged over each rung's own node-feature vector, so its *level* is not comparable across rungs -- core
reconstructs only one-hot identity/chain/interface flags -- but the downward trend within each rung is.)''')

code(r'''
recon_loss_cols = [('val_node_mse', 'Val node MSE'), ('val_edge_attr_mse', 'Val edge-attribute MSE'),
                   ('val_edge_presence_bce', 'Val edge-presence BCE')]
fig, axes = plt.subplots(1, 3, figsize=(24, 6.2), constrained_layout=True)
for ax, (col, ylabel) in zip(axes, recon_loss_cols):
    for rung in RUNGS:
        s = history[history.rung == rung].groupby('epoch')[col].agg(['mean', 'std'])
        ax.plot(s.index, s['mean'], color=RUNG_COLOR[rung], lw=2.2, label=RUNG_LABEL[rung])
        ax.fill_between(s.index, s['mean'] - s['std'], s['mean'] + s['std'], color=RUNG_COLOR[rung], alpha=.15)
    ax.axvspan(.5, 2.5, color='0.85', zorder=0)
    ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel); ax.set_yscale('log')
axes[0].legend(frameon=False, loc='upper right', ncol=2, fontsize=15)
save(fig, 'reconstruction_loss_curves')
''')

md(r'''## Reconstruction quality at the selected checkpoint

Seed mean $\pm$ sd. `rsasa_i_mae` is empty for every run (rSASA$_i$ is an input here, not a reconstruction
target) and is omitted. Remember these are epoch 1--2 checkpoints.

Note the trade-off: amino-acid identity accuracy is essentially perfect at `core` and *falls* as continuous
node features are added (to ~84--87% from `rsasa` onward, with a large seed spread), whereas edge $F_1$ and
C$\alpha$-distance error improve from `core` to `degree` and then stay roughly flat. With a fixed-width
latent, extra input channels compete with identity for capacity at this early stage of training.''')

code(r'''
recon_cols = ['aa_identity_accuracy_pct', 'interface_node_accuracy_pct', 'edge_f1_pct', 'ca_dist_mae_angstrom']
RECON_LABEL = {'aa_identity_accuracy_pct': 'AA identity accuracy (\\%)',
               'interface_node_accuracy_pct': 'Interface-node accuracy (\\%)',
               'edge_f1_pct': 'Edge $F_1$ (\\%)', 'ca_dist_mae_angstrom': r'C$\alpha$ distance MAE (\AA)'}
assert summary.rsasa_i_mae.isna().all()
summary['rung'] = pd.Categorical(summary.rung, RUNGS, ordered=True)
recon = summary.groupby('rung', observed=True)[recon_cols].agg(['mean', 'std'])
display(recon.round(3)); recon.to_csv(EXPORT / 'reconstruction_by_rung.csv')

fig, axes = plt.subplots(1, len(recon_cols), figsize=(6 * len(recon_cols), 6), constrained_layout=True)
for ax, col in zip(axes, recon_cols):
    v = recon[(col, 'mean')].to_numpy(); e = recon[(col, 'std')].to_numpy()
    ax.bar(x, v, color=[RUNG_COLOR[r] for r in RUNGS], edgecolor='black', yerr=e, capsize=5)
    for i, r in enumerate(RUNGS):
        sv = summary.loc[summary.rung == r, col].to_numpy()
        ax.scatter(i + np.linspace(-.13, .13, 3), sv, color='black', s=18, zorder=6)
    if 'pct' in col:
        lo = min(v - e); ax.set_ylim(lo - .3 * (100 - lo), 100 + .08 * (100 - lo))
    else:
        ax.set_ylim(0, max(v + e) * 1.15)
    ax.set_xticks(x, labels, fontsize=14, rotation=25); ax.set_ylabel(RECON_LABEL[col])
save(fig, 'reconstruction_by_rung')
''')

md(r'''## Per-target view: which targets drive the ladder?

Seed-averaged within-target Spearman $\rho$ for every test target (rows sorted by the core-rung value). The right
block is the change from core to the full (`electrostatics`) model. This shows whether rung-level gains are broad
or concentrated in a handful of targets.''')

code(r'''
heat = seed_mean['rho'].unstack('rung')[RUNGS]
heat = heat.sort_values('core')
delta = (heat['electrostatics'] - heat['core']).to_frame('$\\Delta$')
heat.to_csv(EXPORT / 'per_target_rho_by_rung.csv')

fig, (ax, axd) = plt.subplots(1, 2, figsize=(12, 16), gridspec_kw=dict(width_ratios=[5, 1.3]),
                              constrained_layout=True)
lim = np.nanmax(np.abs(heat.to_numpy()))
im = ax.imshow(heat.to_numpy(), cmap='RdBu_r', vmin=-lim, vmax=lim, aspect='auto')
ax.set_xticks(range(len(RUNGS)), labels, fontsize=16, rotation=25)
ax.set_yticks(range(len(heat)), heat.index, fontsize=13)
ax.set_ylabel('Test target')
cb = fig.colorbar(im, ax=ax, location='bottom', pad=0.02, shrink=.8, aspect=30); cb.set_label(r'Within-target $\rho$', fontsize=20)
cb.ax.tick_params(labelsize=15)
dl = np.nanmax(np.abs(delta.to_numpy()))
imd = axd.imshow(delta.to_numpy(), cmap='PiYG', vmin=-dl, vmax=dl, aspect='auto')
axd.set_xticks([0], [r'$+V_{\mathrm{es}}$ $-$ core'], fontsize=15); axd.set_yticks([])
for i, v in enumerate(delta.iloc[:, 0]):
    axd.text(0, i, f'{v:+.2f}', ha='center', va='center', fontsize=12)
save(fig, 'per_target_rho_heatmap')
''')

md(r'''## Top-ranked candidate selection

A scorer is ultimately used to pick decoys. Curves give the target-macro best true DockQ among the top-$k$
ranked decoys (seed mean $\pm$ sd), against the random single-pick expectation and the per-target oracle.''')

code(r'''
ks = [1, 5, 10]
fig, ax = plt.subplots(figsize=(10, 6.8), constrained_layout=True)
for j, rung in enumerate(RUNGS):
    vals = np.array([[metrics.loc[(metrics.rung == rung) & (metrics.seed == s), f'top{k}'].iloc[0] for k in ks]
                     for s in SEEDS])
    xs = np.array(ks) + (j - 2) * .12
    ax.errorbar(xs, vals.mean(0), yerr=vals.std(0, ddof=1), color=RUNG_COLOR[rung], marker='o', ms=9, lw=2.2,
                capsize=5, label=RUNG_LABEL[rung], mec='black')
ax.axhline(ORACLE_TOP1, color='black', ls='-', lw=1.4, label='oracle')
ax.axhline(RANDOM_TOP1, color='0.4', ls='--', lw=1.4, label='random (top-1)')
ax.set_xticks(ks); ax.set_xlim(.4, 10.8); ax.set_ylim(.25, 1.0)
ax.set_xlabel('Top-$k$ decoys inspected'); ax.set_ylabel('Best true DockQ (target mean)')
ax.legend(frameon=False, loc='lower right', ncol=2, fontsize=15)
save(fig, 'top_k_selection')
''')

md(r'''## Prediction spread and calibration

Mean predicted DockQ within bins of true DockQ (pooled over seeds; bars are seed sd of the bin mean). A
well-calibrated scorer follows the diagonal; a collapsed one is flat. The right panel shows the distribution of
predictions per rung against the true-label distribution (grey).''')

code(r'''
bins = np.linspace(0, 1, 11); mids = (bins[:-1] + bins[1:]) / 2
fig, (ax, axh) = plt.subplots(1, 2, figsize=(19, 7), constrained_layout=True)
calib_rows = []
for rung in RUNGS:
    per_seed = []
    for s in SEEDS:
        p = predictions[rung, s]
        b = pd.cut(p.true_target, bins, include_lowest=True)
        per_seed.append(p.groupby(b, observed=False).predicted_target.mean().to_numpy())
    per_seed = np.array(per_seed)
    calib_rows += [dict(rung=rung, bin_mid=m, mean_pred=v, sd=e) for m, v, e in
                   zip(mids, per_seed.mean(0), per_seed.std(0, ddof=1))]
    ax.errorbar(mids, per_seed.mean(0), yerr=per_seed.std(0, ddof=1), color=RUNG_COLOR[rung], marker='o',
                ms=7, lw=2, capsize=4, label=RUNG_LABEL[rung])
    pooled = np.concatenate([predictions[rung, s].predicted_target for s in SEEDS])
    axh.hist(pooled, bins=np.linspace(-.1, 1.2, 66), histtype='step', lw=2.2, density=True, color=RUNG_COLOR[rung])
pd.DataFrame(calib_rows).to_csv(EXPORT / 'calibration_by_rung.csv', index=False)
ax.plot([0, 1], [0, 1], color='0.4', ls='--', lw=1.4)
ax.set(xlim=(0, 1), ylim=(0, 1)); ax.set_xlabel('True DockQ (bin centre)'); ax.set_ylabel('Mean predicted DockQ')
ax.legend(frameon=False, loc='upper left', fontsize=15)
axh.hist(y_all, bins=np.linspace(-.1, 1.2, 66), density=True, color='0.8', label='true')
axh.set_xlabel('DockQ'); axh.set_ylabel('Density')
save(fig, 'calibration_and_spread')
''')

md(r'''## Target-by-target predictions: core vs full model

All 29 test targets for one seed, core (black) against the full `electrostatics` model (pink). Shared axes;
dashed line is perfect agreement. Seed 17 is chosen for display only; it is not selected on test performance.''')

code(r'''
PANEL_SEED = 17
pair = ('core', 'electrostatics')
targets = sorted(reference.target_id.unique())
ncols = 6; nrows = int(np.ceil(len(targets) / ncols))
lo_ax = min(0.0, *(predictions[r, PANEL_SEED].predicted_target.min() for r in pair)) - .02
hi_ax = max(1.0, *(predictions[r, PANEL_SEED].predicted_target.max() for r in pair)) + .02
with mpl.rc_context({'axes.labelsize': 22, 'xtick.labelsize': 17, 'ytick.labelsize': 17}):
    fig, axes = plt.subplots(nrows, ncols, figsize=(30, 5.1 * nrows), sharex=True, sharey=True,
                             constrained_layout=True)
    for ax, t in zip(axes.flat, targets):
        rhos = {}
        for rung, color in zip(pair, ['black', RUNG_COLOR['electrostatics']]):
            g = predictions[rung, PANEL_SEED]; g = g[g.target_id == t]
            rhos[rung] = spearmanr(g.true_target, g.predicted_target).statistic
            ax.scatter(g.true_target, g.predicted_target, color=color, alpha=.28, s=16, edgecolors='none',
                       rasterized=True, label=RUNG_LABEL[rung] if rung == 'core' else r'full ($+V_{\mathrm{es}}$)')
        ax.plot([lo_ax, hi_ax], [lo_ax, hi_ax], color='0.45', ls='--', lw=1.2)
        ax.text(.04, .96, t + '\n' + rf'$\rho_{{\mathrm{{core}}}}={rhos["core"]:.2f}$' + '\n'
                + rf'$\rho_{{\mathrm{{full}}}}={rhos["electrostatics"]:.2f}$', transform=ax.transAxes,
                ha='left', va='top', fontsize=18, bbox=dict(facecolor='white', edgecolor='none', alpha=.8, pad=2))
        ax.set(xlim=(lo_ax, hi_ax), ylim=(lo_ax, hi_ax)); ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([0, .5, 1]); ax.set_yticks([0, .5, 1])
        ax.tick_params(labelbottom=True, labelleft=True)
    for ax in axes.flat[len(targets):]: ax.set_visible(False)
    for ax in axes[:, 0]: ax.set_ylabel('Predicted DockQ')
    for ax in axes[-1, :]: ax.set_xlabel('Observed DockQ')
    for ax in axes.flat[len(targets) - ncols:len(targets)]: ax.set_xlabel('Observed DockQ')
    leg = fig.legend(*axes.flat[0].get_legend_handles_labels(), loc='outside upper center', ncol=2,
                     fontsize=24, frameon=False, markerscale=3)
    for h in leg.legend_handles: h.set_alpha(1)
    save(fig, f'prediction_panels_seed{PANEL_SEED}')
''')

md(r'''## Three-seed ensemble

Averaging the three seeds' predictions per decoy is a cheap way to see how much of the rung-to-rung differences
are seed noise. Shown as a reference, not as the reported model.''')

code(r'''
ens_rows = []
for rung in RUNGS:
    p = reference.copy()
    p['predicted_target'] = np.mean([predictions[rung, s].predicted_target.to_numpy() for s in SEEDS], axis=0)
    t = p.groupby('target_id').apply(target_metrics, include_groups=False)
    ens_rows.append(dict(rung=rung, mse=np.mean((p.predicted_target - p.true_target) ** 2),
                         macro_rho=t.rho.mean(), top1=t.top1.mean(), top5=t.top5.mean(),
                         seed_mean_mse=agg.loc[rung, ('mse', 'mean')],
                         seed_mean_macro_rho=agg.loc[rung, ('macro_rho', 'mean')]))
ensemble = pd.DataFrame(ens_rows)
ensemble.to_csv(EXPORT / 'ensemble_by_rung.csv', index=False)
display(ensemble.round(4))
''')

md(r'''## Full dataset vs the 10% baseline ladder

Rung means for the four rungs the two ladders share. **Not a paired comparison:** the 10% run used a different
split (only 6 of 29 test targets overlap) and its `rsasa`/`voronoi` rungs used the graph-averaged `rsasa_i`
rather than `rsasa_i_node`. Read it only as "did the ladder's shape survive the move to the full data".''')

code(r'''
SHARED = ['core', 'degree', 'rsasa', 'voronoi']
rows = []
for rung in SHARED:
    for seed in SEEDS:
        p = pd.read_csv(run_dir(rung, seed, ROOT_10PCT) / 'test_predictions.csv')
        t = p.groupby('target_id').apply(target_metrics, include_groups=False)
        rows.append(dict(rung=rung, seed=seed, mse=np.mean((p.predicted_target - p.true_target) ** 2),
                         macro_rho=t.rho.mean(), dataset='10%'))
comp = pd.concat([pd.DataFrame(rows),
                  metrics[metrics.rung.isin(SHARED)].astype({'rung': str})[['rung', 'seed', 'mse', 'macro_rho']]
                  .assign(dataset='full')], ignore_index=True)
comp_agg = comp.groupby(['dataset', 'rung'])[['mse', 'macro_rho']].agg(['mean', 'std'])
display(comp_agg.round(4)); comp_agg.to_csv(EXPORT / 'full_vs_10pct.csv')

fig, axes = plt.subplots(1, 2, figsize=(17, 6.4), constrained_layout=True)
xs = np.arange(len(SHARED)); w = .38
for ax, col, ylabel in [(axes[0], 'mse', 'Test DockQ MSE'), (axes[1], 'macro_rho', r'$\langle\rho\rangle_t$')]:
    for k, (ds, face) in enumerate([('10%', 'white'), ('full', '0.35')]):
        v = comp_agg.loc[ds].loc[SHARED, (col, 'mean')]; e = comp_agg.loc[ds].loc[SHARED, (col, 'std')]
        ax.bar(xs + (k - .5) * w, v, w, yerr=e, color=face, edgecolor='black', capsize=5, lw=1.2,
               label={'10%': r'10\% subset', 'full': 'full dataset'}[ds], hatch='//' if ds == '10%' else None)
    ax.set_xticks(xs, [RUNG_LABEL[r] for r in SHARED], fontsize=16); ax.set_ylabel(ylabel)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
axes[0].legend(frameon=False, loc='upper right')
save(fig, 'full_vs_10pct')
''')

md(r'''## Dissertation figure

DockQ MSE and $\langle\rho\rangle_t$ (top), with the paired step-wise $\Delta\langle\rho\rangle_t$ and the
validation-curve evidence for the early stopping point (bottom). Grayscale, darkest at the full model, so it
prints; seed dots are white-on-black to stay visible on dark bars.''')

code(r'''
GRAY = {'core': '#e0e0e0', 'degree': '#bdbdbd', 'rsasa': '#8c8c8c', 'voronoi': '#525252', 'electrostatics': '#111111'}
fig = plt.figure(figsize=(20, 13), constrained_layout=True)
gs = fig.add_gridspec(2, 2)
axa, axb, axc, axd = (fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2))
wdots = dict(facecolors='white', edgecolors='black', linewidths=1.1, s=34)
ladder_bars(axa, 'mse', 'Test DockQ MSE', '%.4f', GRAY, wdots, label_fs=15, baseline=MEAN_MSE)
ladder_bars(axb, 'macro_rho', r'$\langle\rho\rangle_t$', '%.3f', GRAY, wdots, label_fs=15)

s = step_effects[step_effects.metric == 'rho']
for yv, (_, row) in zip(ypos, s.iterrows()):
    axc.errorbar(row['mean'], yv, xerr=[[row['mean'] - row.lo], [row.hi - row['mean']]], fmt='o',
                 mfc=GRAY[row.new], mec='black', ecolor='black', ms=12, capsize=6, lw=2)
axc.axvline(0, color='black', lw=1.2); axc.axhline(.5, color='0.6', lw=1, ls=':')
axc.set_yticks(ypos, [STEP_LABEL[f'{a}->{b}'] for a, b in steps], fontsize=16)
axc.set_xlabel(r'Paired $\Delta\langle\rho\rangle_t$ (95\% bootstrap CI)')

for rung in RUNGS:
    g = history[history.rung == rung].groupby('epoch').val_target_mse.agg(['mean', 'std'])
    axd.plot(g.index, g['mean'], color=GRAY[rung] if rung != 'core' else '#9e9e9e', lw=2.4, label=RUNG_LABEL[rung])
axd.axhline(MEAN_MSE, color='0.3', ls='--', lw=1.4)
axd.set_xlabel('Epoch'); axd.set_ylabel('Validation DockQ MSE')
axd.legend(frameon=False, loc='lower right', ncol=2, fontsize=15)

for ax, lab in zip([axa, axb, axc, axd], ['(a)', '(b)', '(c)', '(d)']):
    ax.text(-0.1, 1.03, lab, transform=ax.transAxes, fontsize=28, fontweight='bold', va='bottom', ha='right')
save(fig, 'dissertation_ladder_figure')
''')

md(r'''## Readout''')

code(r'''
m = agg
def mr(r, c): return m.loc[r, (c, 'mean')], m.loc[r, (c, 'std')]
best_rho_rung = m[('macro_rho', 'mean')].idxmax(); best_mse_rung = m[('mse', 'mean')].idxmin()
lines = ['| rung | test MSE | $\\langle\\rho\\rangle_t$ | top-1 DockQ | AA acc. (%) | edge $F_1$ (%) |', '|---|---|---|---|---|---|']
for r in RUNGS:
    lines.append(f'| {r} | {mr(r,"mse")[0]:.4f} ± {mr(r,"mse")[1]:.4f} | {mr(r,"macro_rho")[0]:.3f} ± {mr(r,"macro_rho")[1]:.3f} '
                 f'| {mr(r,"top1")[0]:.3f} | {recon.loc[r, ("aa_identity_accuracy_pct","mean")]:.1f} '
                 f'| {recon.loc[r, ("edge_f1_pct","mean")]:.1f} |')
se = step_effects.set_index(['step', 'metric'])
def step_txt(st):
    r = se.loc[(st, 'rho')]
    return f'{r["mean"]:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}], {r.frac_improved*29:.0f}/29 targets'
text = '\n'.join(lines) + f"""

- Constant-predictor MSE is **{MEAN_MSE:.4f}**; random/oracle top-1 DockQ **{RANDOM_TOP1:.3f} / {ORACLE_TOP1:.3f}**.
- Best mean MSE: **{best_mse_rung}**; best mean $\\langle\\rho\\rangle_t$: **{best_rho_rung}**.
- Paired $\\Delta\\langle\\rho\\rangle_t$ per step:
  - core → degree: {step_txt('core->degree')}
  - degree → rSASA$_i$: {step_txt('degree->rsasa')}
  - rSASA$_i$ → Voronoi: {step_txt('rsasa->voronoi')}
  - Voronoi → $V_{{es}}$: {step_txt('voronoi->electrostatics')}
  - core → full: {step_txt('core->electrostatics')}
- Selected epochs: {sorted(best.best_epoch)} -- every checkpoint is from epoch 1 or 2.
"""
display(Markdown(text))
(EXPORT / 'readout.md').write_text(text)
''')

md(r'''## Limitations

- **Early-epoch checkpoints.** All fifteen runs select epoch 1 or 2; validation DockQ MSE then rises for the
  rest of training while training DockQ MSE falls and the loss-share balancer escalates the DockQ weight. The
  ladder therefore compares what each feature set gives you after one or two passes, not converged models.
  Reconstruction metrics are similarly early. A rerun with fixed loss weights, a lower learning rate or a
  capped DockQ share (or simply evaluating a later reconstruction-selected checkpoint) is needed before treating
  the rung ordering as final.
- Test numbers are read at a validation-selected epoch on one split per run; three seeds give a crude variance
  estimate, and seed-to-seed spread is comparable to several of the rung-to-rung gaps.
- The paired bootstrap treats the 29 test targets as the unit of replication; it does not account for seed
  variance beyond averaging over the three seeds, and 29 targets is a small sample.
- No multiple-comparison correction is applied across the five step-wise comparisons.
- The 10%-vs-full comparison is unpaired (different splits, different rSASA$_i$ feature) and only descriptive.
''')

notebook = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}})
path = Path(__file__).with_name('analyze_feature_ladder_full.ipynb')
nbf.write(notebook, path)
print(path)
