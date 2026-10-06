"""Build the analysis of the full-dataset anti-memorization experiment (6 arms x 3 seeds)."""
from pathlib import Path
import nbformat as nbf

cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip()))

md(r'''
# Anti-memorization experiment (full dataset)

Eighteen runs: six arms $\times$ three seeds (7, 17, 27), 20 epochs each on the full repaired cohort, using
the electrostatics-rung features of the feature ladder and the same frozen target split. Validation is on
12 held-out targets (14,155 decoys); **the test split was not evaluated**. Design and motivation are in
`ANTI_MEMORIZATION_FULL.md`.

| arm | change relative to A |
|---|---|
| A control | -- (fixed-weight objective, range weighting, combined pooling, dropout 0.3) |
| B detached | DockQ head on detached embeddings; only reconstruction trains the encoder |
| C adversary | training-target classifier behind gradient reversal (weight 0.1, DANN ramp) |
| D augment | edge dropout 0.1, node-feature masking 0.1, C$\alpha$ jitter 0.25 \AA{} on encoder inputs |
| E balanced + EMA | 256 decoys/target/epoch; EMA (0.999) weights validated and checkpointed |
| F combined | C + D + E |

The motivating problem: every feature-ladder run picked epoch 1--2 and validation DockQ MSE then rose to
$\sim$0.13. The question is which arm lowers the best validation MSE and keeps it from rising.

Every run validated every 2,000 optimizer steps and at each epoch end; the checkpoint is the best of those.
E and F use 26,880 graphs per epoch (vs. 127,807), so curves are plotted against **training graphs seen**,
not epochs. Figures use LaTeX Computer Modern, large labels, full borders and no titles. Spreads are sample
SD over three seeds; paired intervals are bootstraps over the 12 validation targets.
''')

code(r'''
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

GATE = Path.cwd() if (Path.cwd() / 'anti_memorization_full').is_dir() else Path.cwd() / 'gate_run'
ROOT = GATE / 'anti_memorization_full'
LADDER = GATE / 'feature_ladder_full'
EXPORT = GATE / 'anti_memorization_full_analysis'; EXPORT.mkdir(exist_ok=True)

ARMS = ['A_control', 'B_detached', 'C_adversary', 'D_augment', 'E_balanced_ema', 'F_combined']
SEEDS = [7, 17, 27]
ARM_LABEL = {'A_control': 'A control', 'B_detached': 'B detached', 'C_adversary': 'C adversary',
             'D_augment': 'D augment', 'E_balanced_ema': 'E balanced+EMA', 'F_combined': 'F combined'}
SHORT = {a: a[0] for a in ARMS}
ARM_COLOR = {'A_control': '#000000', 'B_detached': '#0072B2', 'C_adversary': '#D55E00',
             'D_augment': '#009E73', 'E_balanced_ema': '#CC79A7', 'F_combined': '#E69F00'}
BATCH = 16

mpl.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.serif': ['Computer Modern Roman'],
    'mathtext.fontset': 'cm', 'axes.labelsize': 23, 'xtick.labelsize': 17, 'ytick.labelsize': 17,
    'legend.fontsize': 17, 'axes.spines.top': True, 'axes.spines.right': True, 'axes.linewidth': 1.1})

def save(fig, name):
    fig.savefig(EXPORT / (name + '.pdf'), bbox_inches='tight')
    fig.savefig(EXPORT / (name + '.png'), dpi=180, bbox_inches='tight')
    plt.show()

def run_dir(arm, seed):
    return ROOT / f'{arm}_seed{seed}'

steps, history, predictions, reports, train_dist = [], [], {}, [], {}
for arm in ARMS:
    for seed in SEEDS:
        d = run_dir(arm, seed)
        s = pd.read_csv(d / 'validation_steps.csv').assign(arm=arm, seed=seed); steps.append(s)
        h = pd.read_csv(d / 'loss_history.csv').assign(arm=arm, seed=seed); history.append(h)
        predictions[arm, seed] = pd.read_csv(d / 'validation_predictions.csv')
        reports.append(dict(arm=arm, seed=seed, **json.loads((d / 'prediction_metrics.json').read_text())['validation']))
        train_dist[arm, seed] = json.loads((d / 'dockq_training_distribution.json').read_text())
steps = pd.concat(steps, ignore_index=True)
history = pd.concat(history, ignore_index=True)
steps['graphs_seen_M'] = steps.global_step * BATCH / 1e6

# Checks: identical validation cohort/labels, identical training label profile, finished runs,
# and exported predictions equal to the best validation in validation_steps.csv.
reference = predictions['A_control', 7][['target_id', 'graph_name', 'true_target']]
for (arm, seed), p in predictions.items():
    if not (p.target_id.equals(reference.target_id) and p.graph_name.equals(reference.graph_name)
            and np.allclose(p.true_target, reference.true_target)):
        raise ValueError(f'{arm} seed {seed}: validation cohort or labels differ')
    if train_dist[arm, seed]['counts'] != train_dist['A_control', 7]['counts']:
        raise ValueError(f'{arm} seed {seed}: training label distribution differs')
    s = steps[(steps.arm == arm) & (steps.seed == seed)]
    if history[(history.arm == arm) & (history.seed == seed)].epoch.max() != 20:
        raise ValueError(f'{arm} seed {seed}: did not finish 20 epochs')
    mse = np.mean((p.predicted_target - p.true_target) ** 2)
    if not np.isclose(mse, s.val_target_mse.min(), rtol=1e-5, atol=1e-7):
        raise ValueError(f'{arm} seed {seed}: exported predictions are not the best validation')
n_train = sum(train_dist['A_control', 7]['counts'])
print(f'{len(predictions)} runs; identical validation cohort: {len(reference):,} decoys / '
      f'{reference.target_id.nunique()} targets; {n_train:,} training graphs.')
print('Exported predictions match the best step validation in every run.')
''')

md(r'''## Reference points

A constant predictor at the validation mean has MSE equal to the validation variance (an optimistic
no-skill baseline: it uses validation labels). The feature-ladder electrostatics rung used the same
features, cohort and split but a different recipe (below); its validation history is logged once per
epoch.''')

code(r'''
CONST_MSE = reference.true_target.var(ddof=0)
lad = pd.concat([pd.read_csv(LADDER / f'gat4_residual_electrostatics_seed{s}' / 'loss_history.csv').assign(seed=s)
                 for s in SEEDS], ignore_index=True)
lad['graphs_seen_M'] = lad.epoch * n_train / 1e6
lad_best = lad.groupby('seed').val_target_mse.min()
lad_final = lad[lad.epoch == lad.epoch.max()].set_index('seed').val_target_mse
print(f'Constant-predictor validation MSE: {CONST_MSE:.4f}')
print(f'Feature ladder (electrostatics): best {lad_best.mean():.4f} +/- {lad_best.std():.4f}, '
      f'final (epoch 50) {lad_final.mean():.4f} +/- {lad_final.std():.4f}')
''')

md(r'''## Per-run metrics

From `validation_steps.csv` (every validation) and the checkpoint's `validation_predictions.csv`.
**best** is the selected validation MSE; **final** is the last validation of training; **after best** is the
median of all validations from the best step onward (how well the gain persists). Macro metrics weight the
12 targets equally; $\rho$ is within-target Spearman; top-1 is the true DockQ of each target's top-ranked
decoy.''')

code(r'''
def target_metrics(g):
    y = g.true_target.to_numpy(); p = g.predicted_target.to_numpy()
    order = np.lexsort((g.graph_name.to_numpy(), -p))
    return pd.Series(dict(mse=np.mean((p - y) ** 2), rho=spearmanr(y, p).statistic, top1=y[order[0]],
                          top5=y[order[:5]].max(), best=y.max()))

per_target, rows = [], []
for (arm, seed), p in predictions.items():
    t = p.groupby('target_id')[['graph_name', 'true_target', 'predicted_target']].apply(target_metrics)
    per_target.append(t.reset_index().assign(arm=arm, seed=seed))
    s = steps[(steps.arm == arm) & (steps.seed == seed)].sort_values('global_step')
    b = s.loc[s.val_target_mse.idxmin()]
    rep = next(r for r in reports if r['arm'] == arm and r['seed'] == seed)
    rows.append(dict(arm=arm, seed=seed, best=b.val_target_mse, best_epoch=int(b.epoch),
                     best_graphs_M=b.graphs_seen_M, final=s.val_target_mse.iloc[-1],
                     after_best=s[s.global_step >= b.global_step].val_target_mse.median(),
                     macro_mse=t.mse.mean(), macro_rho=t.rho.mean(), top1=t.top1.mean(), top5=t.top5.mean(),
                     high_bias=rep['bins'][4]['bias'], low_bias=rep['bins'][0]['bias'],
                     macro_bin_mse=rep['macro_bin_mse']))
per_target = pd.concat(per_target, ignore_index=True)
metrics = pd.DataFrame(rows)
metrics['final_over_best'] = metrics.final / metrics.best
metrics.to_csv(EXPORT / 'metrics_per_run.csv', index=False)
per_target.to_csv(EXPORT / 'metrics_per_target.csv', index=False)
display(metrics.round(4))

cols = ['best', 'final', 'after_best', 'final_over_best', 'best_epoch', 'macro_mse', 'macro_rho', 'top1',
        'high_bias', 'low_bias']
agg = metrics.groupby('arm')[cols].agg(['mean', 'std']).loc[ARMS]
agg.to_csv(EXPORT / 'metrics_by_arm.csv')
display(agg.round(4))
''')

md(r'''## Headline: best and end-of-training validation MSE

Solid bars: best (selected) validation MSE; hatched bars: final validation of training. Dots are seeds; error
bars are seed SD. Dashed line: constant predictor. Dotted line: feature-ladder electrostatics best (mean of 3
seeds). F seed 7 and all three C runs diverged late in training (next sections); their final values sit at
the constant predictor.''')

code(r'''
fig, ax = plt.subplots(figsize=(17, 7.2), constrained_layout=True)
x = np.arange(len(ARMS)); w = .38
for k, (col, hatch, alpha) in enumerate([('best', None, 1.0), ('final', '//', .45)]):
    v = agg[(col, 'mean')].to_numpy(); e = agg[(col, 'std')].to_numpy()
    ax.bar(x + (k - .5) * w, v, w, yerr=e, capsize=5, color=[ARM_COLOR[a] for a in ARMS], alpha=alpha,
           hatch=hatch, edgecolor='black', lw=1.1, error_kw=dict(lw=1.4),
           label='best (selected)' if col == 'best' else 'end of training')
    for i, a in enumerate(ARMS):
        sv = metrics.loc[metrics.arm == a, col].to_numpy()
        ax.scatter(x[i] + (k - .5) * w + np.linspace(-.08, .08, 3), sv, s=26, zorder=6,
                   facecolors='white', edgecolors='black', linewidths=1)
        if col == 'best':
            ax.text(x[i] + (k - .5) * w, max(v[i] + e[i], sv.max()) + .003, f'{v[i]:.4f}', ha='center',
                    fontsize=14, bbox=dict(facecolor='white', edgecolor='none', pad=1), zorder=7)
ax.axhline(CONST_MSE, color='0.3', ls='--', lw=1.6, label='constant predictor')
ax.axhline(lad_best.mean(), color='0.3', ls=':', lw=2, label='feature ladder best')
ax.set_xticks(x, [ARM_LABEL[a] for a in ARMS], fontsize=18)
ax.set_ylabel('Validation DockQ MSE'); ax.set_ylim(0, .125)
ax.legend(frameon=False, loc='upper left', ncol=4, fontsize=16)
save(fig, 'best_and_final_mse')
''')

md(r'''## Validation curves against training graphs seen

Each panel: one arm's three seeds (thin) and their mean (thick); the control's mean is repeated in grey on
every panel. Panel A also shows the feature ladder (electrostatics rung, one point per epoch, first 2.6M
graphs). The y-axis is capped at 0.11; values above it (C and F seed 7 divergence) are clipped at the top
edge.''')

code(r'''
YMAX = .11
ctrl = steps[steps.arm == 'A_control'].groupby('global_step').agg(x=('graphs_seen_M', 'first'),
                                                                   y=('val_target_mse', 'mean'))
fig, axes = plt.subplots(2, 3, figsize=(24, 12), sharey=True, constrained_layout=True)
for ax, arm in zip(axes.flat, ARMS):
    for seed in SEEDS:
        s = steps[(steps.arm == arm) & (steps.seed == seed)].sort_values('global_step')
        ax.plot(s.graphs_seen_M, s.val_target_mse.clip(upper=YMAX), color=ARM_COLOR[arm], lw=1, alpha=.45)
    m = steps[steps.arm == arm].groupby('global_step').agg(x=('graphs_seen_M', 'first'), y=('val_target_mse', 'mean'))
    if arm != 'A_control':
        ax.plot(ctrl.x, ctrl.y, color='0.6', lw=2.2, label='A control (mean)')
    ax.plot(m.x, m.y.clip(upper=YMAX), color=ARM_COLOR[arm], lw=2.8, label=ARM_LABEL[arm] + ' (mean)')
    if arm == 'A_control':
        lg = lad.groupby('epoch').agg(x=('graphs_seen_M', 'first'), y=('val_target_mse', 'mean'))
        lg = lg[lg.x <= ctrl.x.max()]
        ax.plot(lg.x, lg.y, color='0.35', ls=':', marker='s', ms=6, lw=2, label='feature ladder (mean)')
    ax.axhline(CONST_MSE, color='0.3', ls='--', lw=1.3)
    ax.set_ylim(.04, YMAX); ax.set_xlim(0, ctrl.x.max() * 1.02)
    ax.legend(frameon=False, loc='center right', fontsize=15)
for ax in axes[-1]: ax.set_xlabel('Training graphs seen (millions)')
for ax in axes[:, 0]: ax.set_ylabel('Validation DockQ MSE')
save(fig, 'validation_curves_by_arm')
''')

md(r'''## The control vs the feature ladder

Same features, cohort and split. Left: validation DockQ MSE (seed mean $\pm$ SD) against graphs seen.
Right: training DockQ MSE on a log scale. **Not a controlled comparison.** The control changed several things
at once relative to the ladder, so the improvement cannot be attributed to any single one:

| setting | feature ladder | A control |
|---|---|---|
| loss weighting | adaptive shares (DockQ weight 14 $\to$ 260) | fixed: DockQ $+ 1\times$ reconstruction |
| DockQ range weighting | none | capped inverse-sqrt ($\alpha$ 0.5) |
| pooling | whole graph | whole graph + interface |
| dropout | 0.1 | 0.3 |
| schedule | 50-epoch cosine | 20-epoch cosine (LR decays 2.5$\times$ faster per epoch) |
| checkpoint resolution | once per epoch | every 2,000 steps |
| validation MSE | batch-averaged | exact graph-averaged |

Training DockQ MSE for the control is range-weighted (mean weight 1) while the ladder's is unweighted; the
difference is small next to the order-of-magnitude gap.''')

code(r'''
fig, axes = plt.subplots(1, 2, figsize=(20, 7), constrained_layout=True)
xmax = 20 * n_train / 1e6
lg = lad.groupby('epoch').agg(x=('graphs_seen_M', 'first'), m=('val_target_mse', 'mean'), s=('val_target_mse', 'std'),
                              tm=('train_target_mse', 'mean'))
a = steps[steps.arm == 'A_control'].groupby('global_step').agg(x=('graphs_seen_M', 'first'),
        m=('val_target_mse', 'mean'), s=('val_target_mse', 'std'))
ax = axes[0]
ax.plot(lg.x, lg.m, color='0.45', marker='s', ms=6, lw=2.2, label='feature ladder (electrostatics)')
ax.fill_between(lg.x, lg.m - lg.s, lg.m + lg.s, color='0.45', alpha=.15)
ax.plot(a.x, a.m, color='black', lw=2.6, label='A control')
ax.fill_between(a.x, a.m - a.s, a.m + a.s, color='black', alpha=.15)
ax.axhline(CONST_MSE, color='0.3', ls='--', lw=1.4, label='constant predictor')
ax.set_xlim(0, lad.graphs_seen_M.max()); ax.set_xlabel('Training graphs seen (millions)')
ax.set_ylabel('Validation DockQ MSE'); ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(.5, -.16), ncol=3, fontsize=16)
ax.axvline(xmax, color='0.6', lw=1, ls=':')
ax.text(xmax, .045, ' control ends', fontsize=15, color='0.4')

ax = axes[1]
ha = history[history.arm == 'A_control'].groupby('epoch').train_target_mse.mean()
ax.plot(lg.x, lg.tm, color='0.45', marker='s', ms=6, lw=2.2, label='feature ladder')
ax.plot(ha.index * n_train / 1e6, ha.values, color='black', marker='o', ms=6, lw=2.6, label='A control')
ax.set_yscale('log'); ax.set_xlim(0, lad.graphs_seen_M.max())
ax.set_xlabel('Training graphs seen (millions)'); ax.set_ylabel('Training DockQ MSE')
ax.legend(frameon=False, loc='upper right', fontsize=16)
save(fig, 'control_vs_feature_ladder')
print(f"Training DockQ MSE at the last epoch: control {ha.iloc[-1]:.4f}, ladder at epoch 20 "
      f"{lad[lad.epoch == 20].train_target_mse.mean():.4f}, at epoch 50 {lad[lad.epoch == 50].train_target_mse.mean():.4f}")
''')

md(r'''## Each arm against the control, paired by target

For every validation target, the arm's MSE (or within-target $\rho$) is averaged over seeds and differenced
against the control's; the 12 differences are bootstrapped (10,000 resamples) for a 95% interval. Positive
values favour the arm in both panels. Numbers on the right count the targets where the arm beat the control.''')

code(r'''
rng = np.random.default_rng(0)
seed_mean = per_target.groupby(['arm', 'target_id'])[['mse', 'rho', 'top1']].mean()
eff = []
for arm in ARMS[1:]:
    for metric, sign in [('mse', -1), ('rho', 1), ('top1', 1)]:
        d = sign * (seed_mean.loc[arm, metric] - seed_mean.loc['A_control', metric]).to_numpy()
        boots = rng.choice(d, size=(10_000, len(d))).mean(1)
        eff.append(dict(arm=arm, metric=metric, mean=d.mean(), lo=np.percentile(boots, 2.5),
                        hi=np.percentile(boots, 97.5), wins=int((d > 0).sum()), n=len(d)))
eff = pd.DataFrame(eff); eff.to_csv(EXPORT / 'paired_vs_control.csv', index=False)
display(eff.round(4))

fig, axes = plt.subplots(1, 2, figsize=(19, 6.4), sharey=True, constrained_layout=True)
yp = np.arange(len(ARMS) - 1)[::-1]
for ax, metric, xlabel in [(axes[0], 'mse', r'$-\Delta$ MSE (control $-$ arm)'),
                           (axes[1], 'rho', r'$\Delta\langle\rho\rangle_t$ (arm $-$ control)')]:
    e = eff[eff.metric == metric].set_index('arm').loc[ARMS[1:]]
    for yv, (arm, r) in zip(yp, e.iterrows()):
        ax.errorbar(r['mean'], yv, xerr=[[r['mean'] - r.lo], [r.hi - r['mean']]], fmt='o', ms=12, capsize=6,
                    lw=2.2, color=ARM_COLOR[arm], mec='black')
        ax.text(1.02, yv, f'{r.wins}/{r.n}', transform=ax.get_yaxis_transform(), va='center', fontsize=16)
    ax.axvline(0, color='black', lw=1.2); ax.set_xlabel(xlabel)
axes[0].set_yticks(yp, [ARM_LABEL[a] for a in ARMS[1:]], fontsize=18)
save(fig, 'paired_vs_control')
''')

md(r'''## What happened to the adversary (C, and F)

Training loss terms by epoch for the adversary runs. The reversed gradient asks the encoder to *maximize*
the adversary's cross-entropy, which is unbounded. Once the DANN coefficient passed $\sim$0.5 (epoch 3), the
encoder in all three C runs found it could inflate the cross-entropy by blowing up its activations: CE rose
to $10^{4}$--$10^{6}$, training DockQ MSE exploded, and validation MSE settled at the constant-predictor level.
F (with augmentation and EMA) held out until epoch 12 for seed 7 and not at all for seeds 17 and 27. The
early validation numbers for C are real (they are from before the collapse), but the method as configured is
unstable.''')

code(r'''
adv = history[history.arm.isin(['C_adversary', 'F_combined'])]
fig, axes = plt.subplots(1, 3, figsize=(24, 6.4), constrained_layout=True)
for (arm, seed), h in adv.groupby(['arm', 'seed']):
    ls = '-' if arm == 'C_adversary' else '--'
    lab = f"{SHORT[arm]} seed {seed}"
    axes[0].plot(h.epoch, h.train_target_adversary_ce, ls, color=ARM_COLOR[arm], lw=2, alpha=.5 + .25 * (seed == 7), label=lab)
    axes[1].plot(h.epoch, h.train_target_mse, ls, color=ARM_COLOR[arm], lw=2, alpha=.5 + .25 * (seed == 7))
    axes[2].plot(h.epoch, h.val_target_mse, ls, color=ARM_COLOR[arm], lw=2, alpha=.5 + .25 * (seed == 7))
coef = adv.groupby('epoch').train_target_adversary_coefficient.mean()
axes[0].axhline(np.log(105), color='0.4', ls=':', lw=1.4)
axes[0].text(20, np.log(105) * 1.4, r'$\ln 105$ (chance)', ha='right', fontsize=15, color='0.3')
for ax, lab in zip(axes, ['Adversary cross-entropy', 'Training DockQ MSE', 'Validation DockQ MSE (epoch end)']):
    ax.set_yscale('log'); ax.set_xlabel('Epoch'); ax.set_ylabel(lab)
    ax.axvline(coef[coef >= .5].index.min(), color='0.6', ls=':', lw=1.2)
axes[2].axhline(CONST_MSE, color='0.3', ls='--', lw=1.4)
axes[0].legend(frameon=False, fontsize=13, ncol=2, loc='upper left')
save(fig, 'adversary_instability')
print('DANN coefficient by epoch:', coef.round(3).to_dict())
''')

md(r'''## Training vs validation: is memorization reduced?

End-of-training training DockQ MSE (range-weighted, comparable across arms because every arm uses the same
weights) against the best validation MSE. Arms further right fit the training targets less tightly. Runs
that diverged (all three C runs and F seed 7, whose end-of-training validation MSE is above 0.08) are omitted.

The relationship runs the opposite way to the memorization hypothesis: the arms that fit training DockQ
*less* tightly (B, F, E) also have *higher* validation MSE. Restraining the fit mostly cost accuracy rather
than buying generalization. Under this recipe, remaining error looks more like underfitting/limited
signal than like memorization.''')

code(r'''
end = history[history.epoch == 20][['arm', 'seed', 'train_target_mse']].merge(metrics[['arm', 'seed', 'best']])
end['diverged'] = end.merge(metrics[['arm', 'seed', 'final']]).final.to_numpy() > .08
fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
for arm in ARMS:
    e = end[(end.arm == arm) & ~end.diverged]
    if e.empty:
        continue
    ax.scatter(e.train_target_mse, e.best, s=110, color=ARM_COLOR[arm], edgecolors='black', label=ARM_LABEL[arm],
               zorder=5)
ax.axhline(CONST_MSE, color='0.3', ls='--', lw=1.4)
ax.set_xlabel('Training DockQ MSE at epoch 20'); ax.set_ylabel('Best validation DockQ MSE')
ax.set_ylim(.045, .064)
ax.legend(frameon=False, fontsize=15, loc='upper left', ncol=2)
save(fig, 'train_vs_validation')
display(end.round(4))
''')

md(r'''## Score range: where does each arm err?

Mean prediction error (bias) and MSE by true-DockQ bin at the selected checkpoint, averaged over seeds.
Every arm still over-predicts poor decoys and under-predicts high-quality ones (score compression). The
control has the smallest under-prediction in the 0.8--1 bin, and the lowest MSE there.''')

code(r'''
bins = pd.DataFrame([dict(arm=r['arm'], seed=r['seed'], **b) for r in reports for b in r['bins']])
bins['mid'] = (bins.low + bins.high) / 2
bm = bins.groupby(['arm', 'mid'])[['bias', 'mse']].mean().reset_index()
bm.to_csv(EXPORT / 'bins_by_arm.csv', index=False)
fig, axes = plt.subplots(1, 2, figsize=(20, 6.6), constrained_layout=True)
for arm in ARMS:
    b = bm[bm.arm == arm]
    axes[0].plot(b.mid, b.bias, marker='o', ms=8, lw=2.2, color=ARM_COLOR[arm], label=ARM_LABEL[arm])
    axes[1].plot(b.mid, b.mse, marker='o', ms=8, lw=2.2, color=ARM_COLOR[arm])
axes[0].axhline(0, color='0.3', lw=1.2)
axes[0].set_ylabel('Mean error (pred $-$ true)'); axes[1].set_ylabel('Validation MSE in bin')
for ax in axes: ax.set_xlabel('True DockQ (bin centre)'); ax.set_xticks([.1, .3, .5, .7, .9])
axes[0].legend(frameon=False, fontsize=15, loc='lower left', ncol=2)
save(fig, 'score_range_by_arm')
''')

md(r'''## Within-target ranking and decoy selection

Seed mean $\pm$ SD of macro within-target Spearman $\rho$ and mean top-1 true DockQ, against the random
single-pick expectation and the per-target oracle.''')

code(r'''
tstats = reference.groupby('target_id').true_target.agg(['mean', 'max'])
fig, axes = plt.subplots(1, 2, figsize=(19, 6.6), constrained_layout=True)
for ax, col, ylabel in [(axes[0], 'macro_rho', r'$\langle\rho\rangle_t$'), (axes[1], 'top1', 'Top-1 true DockQ')]:
    v = agg[(col, 'mean')].to_numpy(); e = agg[(col, 'std')].to_numpy()
    ax.bar(x, v, yerr=e, capsize=5, color=[ARM_COLOR[a] for a in ARMS], edgecolor='black', alpha=.85)
    for i, a in enumerate(ARMS):
        sv = metrics.loc[metrics.arm == a, col].to_numpy()
        ax.scatter(x[i] + np.linspace(-.12, .12, 3), sv, s=26, zorder=6, facecolors='white', edgecolors='black')
    ax.set_xticks(x, [SHORT[a] for a in ARMS], fontsize=20); ax.set_ylabel(ylabel); ax.set_xlabel('Arm')
axes[1].axhline(tstats['mean'].mean(), color='0.3', ls='--', lw=1.4, label='random pick')
axes[1].axhline(tstats['max'].mean(), color='black', lw=1.4, label='oracle')
axes[1].set_ylim(0, 1.05); axes[0].set_ylim(0, .85)
axes[1].legend(frameon=False, loc='upper right', ncol=2, fontsize=15)
save(fig, 'ranking_and_selection')
''')

md(r'''## Readout''')

code(r'''
def ms(arm, col, f='.4f'):
    return f"{agg.loc[arm, (col, 'mean')]:{f}} ± {agg.loc[arm, (col, 'std')]:{f}}"
lines = ['| arm | best val MSE | final val MSE | best epoch | ⟨ρ⟩ₜ | top-1 |', '|---|---|---|---|---|---|']
for a in ARMS:
    lines.append(f"| {ARM_LABEL[a]} | {ms(a, 'best')} | {ms(a, 'final')} | {ms(a, 'best_epoch', '.1f')} "
                 f"| {ms(a, 'macro_rho', '.3f')} | {ms(a, 'top1', '.3f')} |")
e = eff.set_index(['arm', 'metric'])
win = metrics.groupby('arm').best.mean().idxmin()
text = '\n'.join(lines) + f"""

- Constant predictor: **{CONST_MSE:.4f}**. Feature ladder (same features/split, old recipe): best
  **{lad_best.mean():.4f}**, final **{lad_final.mean():.4f}**.
- Lowest mean best validation MSE: **{ARM_LABEL[win]}**. Paired by target, no arm beats the control on MSE:
""" + '\n'.join(f"  - {ARM_LABEL[a]}: Δ = {e.loc[(a, 'mse'), 'mean']:+.4f} "
                f"[{e.loc[(a, 'mse'), 'lo']:+.4f}, {e.loc[(a, 'mse'), 'hi']:+.4f}], better on "
                f"{e.loc[(a, 'mse'), 'wins']}/12 targets" for a in ARMS[1:])
display(Markdown(text)); (EXPORT / 'readout.md').write_text(text)
''')

md(r'''## Interpretation and limitations

- **The rise is mostly gone in the control itself.** With fixed loss weights, range weighting, combined
  pooling, dropout 0.3 and a 20-epoch schedule, the best checkpoint moves from epoch 1--2 to epochs 7--11, best
  validation MSE falls from $\sim$0.072 to $\sim$0.049, and the end-of-training value ($\sim$0.054) stays far
  below the ladder's $\sim$0.13. Training DockQ MSE ends an order of magnitude higher than in the ladder, i.e.
  the control memorizes much less. Which of the changed settings is responsible is **not identified**: the
  adaptive loss shares (whose DockQ weight climbed to $\sim$260 as training DockQ MSE fell) are the leading
  suspect, but dropout, the shorter cosine schedule and the other changes moved at the same time.
- **None of the targeted interventions improved on that control.** D (augmentation) is closest on best MSE but
  rises more afterwards; E (balanced + EMA) gives the flattest, most seed-consistent curve at a slightly worse
  best value; B (detached) is worse than the control in all three seeds (0.057 vs 0.049; target-level
  interval still spans zero), so the encoder appears to need DockQ gradients; C diverged. Every paired
  interval against the control includes zero: these arms are at best ties, not improvements.
- **Less training fit did not mean better validation.** Across the non-diverged runs, the control fits the
  training targets most tightly *and* validates best; the memorization-targeted arms traded training fit for
  no validation gain.
- **C's failure is a method problem, not evidence against the idea.** Gradient reversal on an unbounded
  cross-entropy let the encoder win by inflating activations. A bounded objective (e.g. pushing the adversary's
  prediction toward uniform) would be needed to test target-invariance fairly.
- Twelve validation targets and three seeds; the paired intervals are wide, and this validation set has now
  informed several rounds of choices. The test split has not been touched.
- Graphs-seen axes treat every step as 16 graphs (each epoch's last batch is slightly smaller).
''')

notebook = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}})
path = Path(__file__).with_name('analyze_anti_memorization_full.ipynb')
nbf.write(notebook, path)
print(path)
