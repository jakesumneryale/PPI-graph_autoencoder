"""ESM2 tuning matrix on the full cohort: sequence conditioning, objective balance and regularization.

Every arm uses ESM2. Checkpoints are always selected on validation DockQ MSE; the test
split is evaluated per epoch for monitoring only and is never a selection criterion.
Test predictions are written once at the end from the selected checkpoint by train_gate.py.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile

BASELINE = dict(pooling='combined', exponent=.75, reconstruction_lambda=1., dropout=.3, lr=3e-4,
                weight_decay=1e-5, esm_projection_dim=64, esm_scale=1., esm_dropout=0., esm_gate=False,
                epochs=50, warmup_steps=1000)

# One declared change per screening arm, against the ESM2 configuration that ran in
# full_pooling_esm_validation. Rationale is recorded so the matrix is self-documenting.
ARMS = [
    ('baseline', {}, 'Matches full_pooling_esm_validation combined/alpha0.75; the reference for every delta.'),
    ('alpha0.5', dict(exponent=.5), 'Lower DockQ weighting exponent; alpha=0.5 won the control comparison.'),
    # Sequence conditioning. Projected ESM is 64 of the 88 encoder input columns at the
    # baseline, so structural features are outnumbered roughly three to one.
    ('esm_dim32', dict(esm_projection_dim=32), 'Halve the sequence share of the encoder input.'),
    ('esm_dim16', dict(esm_projection_dim=16), 'Quarter the sequence share of the encoder input.'),
    ('esm_scale0.25', dict(esm_scale=.25), 'Keep the width, shrink the ESM block magnitude.'),
    ('esm_gate', dict(esm_gate=True, esm_scale=.5), 'Learnable scalar ESM scale initialised at 0.5.'),
    ('esm_dropout0.3', dict(esm_dropout=.3), 'Dropout on the ESM block only; structural features untouched.'),
    # Objective balance. At the baseline, edge-attribute reconstruction is ~88-91% of the
    # training loss and DockQ MSE is ~3-5%, so the encoder is optimized mainly to reconstruct.
    ('lambda0.3', dict(reconstruction_lambda=.3), 'Raise the DockQ share of the loss to roughly 10%.'),
    ('lambda0.1', dict(reconstruction_lambda=.1), 'Raise the DockQ share to roughly 25%.'),
    ('lambda0.03', dict(reconstruction_lambda=.03), 'Raise the DockQ share to roughly 50%.'),
    # Optimization. Baseline ESM runs reach their best validation epoch at 1-11 of 50 and
    # then rise, with final training MSE about three times lower than the controls.
    ('lr1e-4', dict(lr=1e-4), 'Slower schedule; the baseline minimum arrives before the warmup ends.'),
    ('dropout0.5', dict(dropout=.5), 'Stronger encoder regularization.'),
    ('weight_decay1e-3', dict(weight_decay=1e-3), 'Hundredfold weight decay.'),
]
SCREEN_SEED = 7
CONFIRM_SEEDS = (7, 17, 27)
NODES = 'aa_type,chain,interface_nodes,rsasa_i,interface_node_degree'
EDGES = 'interface_edges,ca_dist,voronoi_contact_area,voronoi_contact_missing'


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name)
    try:
        with os.fdopen(fd, 'w') as f: json.dump(value, f, indent=2, allow_nan=False); f.write('\n')
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def code_signature(*names):
    root = Path(__file__).parent
    return {n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in names}


def arm_argv(name, changes, seed, output, split_manifest, data, rsasa, model_lists, sidecars):
    """Build one run's argv. Only the declared changes may differ between arms."""
    settings = {**BASELINE, **changes}
    argv = [
        '--data', str(data),
        '--optional-node-features-dir', str(rsasa),
        '--split-manifest', str(split_manifest),
        '--model-list-dir', str(model_lists),
        '--strict-hdf5', '--device', 'cuda',
        '--epochs', str(settings['epochs']),
        '--batch-size', '16', '--num-workers', '7', '--worker-start-method', 'spawn',
        '--lr', f'{settings["lr"]:g}', '--lr-schedule', 'cosine',
        '--warmup-steps', str(settings['warmup_steps']),
        '--weight-decay', f'{settings["weight_decay"]:g}',
        '--edge-recon-features', 'interface_edges,ca_dist', '--edge-stats-seed', '7',
        '--loss-weight-mode', 'fixed',
        # Selection is always validation DockQ MSE. Test metrics are logged per epoch but
        # never consulted here; train_gate.py only ever compares val_<checkpoint-metric>.
        '--checkpoint-metric', 'target_mse',
        '--residual_connections', '--architecture', 'gat',
        '--pooling', settings['pooling'],
        '--node-features', NODES, '--edge-features', EDGES,
        '--seed', str(seed), '--output-dir', str(output),
        '--standardize-edge-features', 'voronoi_contact_area',
        '--edge-feature-transforms', 'voronoi_contact_area=log1p',
        '--node-weight', '1', '--edge-attr-weight', '1', '--edge-presence-weight', '0.1',
        '--target-weight', '1',
        '--dropout', f'{settings["dropout"]:g}',
        '--reconstruction-lambda', f'{settings["reconstruction_lambda"]:g}',
        '--dockq-range-weighting', 'inverse-sqrt', '--dockq-weight-cap', '3',
        '--dockq-weight-exponent', f'{settings["exponent"]:g}',
        '--dockq-range-diagnostics', '--cpu-threads', '1',
        '--use-esm', '--esm-sidecar-dir', str(sidecars),
        '--esm-projection-dim', str(settings['esm_projection_dim']),
        '--esm-scale', f'{settings["esm_scale"]:g}',
        '--esm-dropout', f'{settings["esm_dropout"]:g}',
    ]
    if settings['esm_gate']: argv.append('--esm-gate')
    # Deliberately absent: --no-test-evaluation and --validation-only-during-training.
    # Their absence is what produces per-epoch test metrics and the end-of-run test
    # predictions from the selected checkpoint.
    return argv, settings


def build(parent, output, sidecars, split_manifest, data, rsasa, stage, arms, model_lists_override=None):
    if (output / 'matrix.json').exists():
        raise FileExistsError('Experiment matrix already exists; choose a fresh experiment directory')
    parent_matrix = json.loads((parent / 'matrix.json').read_text())
    if not parent_matrix.get('full_dataset'):
        raise ValueError('Parent is not a full-dataset experiment')
    reference_argv = parent_matrix['runs'][0]['argv']
    recorded = Path(reference_argv[reference_argv.index('--model-list-dir') + 1])
    # The parent matrix records cluster-absolute paths. Prefer an explicit override, then
    # the recorded location, then the copy beside the parent matrix, so the same builder
    # runs on the cluster and against a downloaded experiment directory.
    candidates = [c for c in (model_lists_override, recorded, parent / 'model_lists') if c is not None]
    model_lists = next((c for c in candidates if c.is_dir()), None)
    if model_lists is None:
        raise ValueError('Parent model lists not found in: ' + ', '.join(str(c) for c in candidates))
    if not sidecars.is_dir():
        raise ValueError(f'ESM sidecar directory missing: {sidecars}')
    split = json.loads(split_manifest.read_text())['splits']
    counts = {s: split[s]['sample_count'] for s in ('train', 'val', 'test')}
    targets = {s: len(split[s]['targets']) for s in ('train', 'val', 'test')}
    seen = set()
    for s in ('train', 'val', 'test'):
        overlap = seen & set(split[s]['targets'])
        if overlap: raise ValueError(f'Overlapping split membership: {sorted(overlap)}')
        seen |= set(split[s]['targets'])
        if not split[s]['targets']: raise ValueError(f'Empty {s} split')
    # Every selected target must have an ESM sidecar that actually covers its cohort. A
    # sidecar file exists for empty targets too, so presence alone is not enough: check the
    # preparation audit reports a non-empty, fully accepted cohort.
    missing = [t for t in sorted(seen) if not (sidecars / f'{t}.hdf5').exists()]
    if missing: raise ValueError(f'{len(missing)} targets without ESM sidecars, e.g. {missing[:5]}')
    empty, short = [], []
    for t in sorted(seen):
        audit_path = sidecars / f'{t}.audit.json'
        if not audit_path.exists(): raise ValueError(f'Missing sidecar audit for {t}')
        a = json.loads(audit_path.read_text())
        if a['expected'] == 0: empty.append(t)
        elif a['accepted'] != a['expected'] or a['failures']: short.append(t)
    if empty:
        raise ValueError(f'{len(empty)} split targets have empty ESM sidecars, e.g. {empty[:5]}. '
                         'Rebuild the sidecars against the current cohort before tuning.')
    if short:
        raise ValueError(f'{len(short)} targets have incomplete ESM sidecars, e.g. {short[:5]}')
    selected = [a for a in ARMS if not arms or a[0] in arms]
    if arms and len(selected) != len(arms):
        raise ValueError(f'Unknown arm(s): {sorted(set(arms) - {a[0] for a in selected})}')
    seeds = (SCREEN_SEED,) if stage == 'screen' else CONFIRM_SEEDS
    runs = []
    for name, changes, rationale in selected:
        for seed in seeds:
            run_name = f'{name}_seed{seed}'
            argv, settings = arm_argv(name, changes, seed, output / run_name, split_manifest,
                                      data, rsasa, model_lists, sidecars)
            runs.append(dict(config=dict(name=run_name, arm=name, architecture='gat', apbs='none',
                                         pooling=settings['pooling'], esm=True, objective='ae'),
                             seed=seed, arm=name, changes=changes, rationale=rationale,
                             settings={k: v for k, v in settings.items()},
                             output=str((output / run_name).resolve()), argv=argv))
    if len({r['config']['name'] for r in runs}) != len(runs):
        raise ValueError('Duplicate run names')
    reference = next(r for r in runs if r['arm'] == 'baseline')
    for r in runs:
        declared = set(r['changes']) | {'output_dir', 'seed'}
        differing = {k for k in set(reference['settings']) | set(r['settings'])
                     if reference['settings'].get(k) != r['settings'].get(k)}
        if not differing <= declared:
            raise ValueError(f'{r["config"]["name"]} differs beyond its declared changes: {differing - declared}')
    matrix = dict(runs=runs, stage=stage, seeds=list(seeds), arms=[a[0] for a in selected],
                  esm_enabled=True, apbs_enabled=False, full_dataset=True,
                  validation_only=False, test_monitoring=True,
                  selection_metric='val_target_mse',
                  test_use='monitoring and one final prediction export from the selected checkpoint',
                  baseline=BASELINE, parent_experiment=str(parent.resolve()),
                  split=str(split_manifest.resolve()), model_list_dir=str(model_lists.resolve()),
                  esm_sidecars=str(sidecars.resolve()), esm_layer=33, esm_width=1280,
                  split_target_counts=targets, split_sample_counts=counts,
                  input_features=dict(nodes=NODES.split(','), edges=EDGES.split(',')),
                  code_sha256=code_signature('full_esm_tuning.py', 'train_gate.py', 'GATE_model.py',
                                             'esm_sidecars.py', 'protein_hdf5_dataset.py'))
    write_json(output / 'matrix.json', matrix)
    print(f'{len(runs)} runs ({len(selected)} arms x {len(seeds)} seed(s)) -> {output / "matrix.json"}')
    print(f'Split: {targets["train"]}/{targets["val"]}/{targets["test"]} targets, '
          f'{counts["train"]:,}/{counts["val"]:,}/{counts["test"]:,} graphs.')
    print('Checkpoint metric: val_target_mse. Test metrics are logged per epoch for monitoring only.')
    return matrix


def verify(output):
    """Report completion without reading any test metric, so a partial sweep can be inspected safely."""
    matrix = json.loads((output / 'matrix.json').read_text())
    done, missing, incomplete = [], [], []
    for r in matrix['runs']:
        path = output / Path(r['output']).name
        if not (path / 'loss_history.csv').exists(): missing.append(path.name); continue
        lines = (path / 'loss_history.csv').read_text().splitlines()
        epochs = len(lines) - 1
        expected = int(r['argv'][r['argv'].index('--epochs') + 1])
        if epochs != expected: incomplete.append(f'{path.name} ({epochs}/{expected} epochs)'); continue
        if not (path / 'test_predictions.csv').exists(): incomplete.append(f'{path.name} (no test predictions)'); continue
        done.append(path.name)
    print(f'{len(done)}/{len(matrix["runs"])} runs complete with test predictions.')
    for label, group in [('missing', missing), ('incomplete', incomplete)]:
        if group: print(f'{len(group)} {label}: ' + ', '.join(sorted(group)))
    counts = Counter(r['arm'] for r in matrix['runs'] if (output / Path(r['output']).name).is_dir())
    print('Runs present per arm: ' + ', '.join(f'{a}={counts.get(a, 0)}' for a in matrix['arms']))
    return done


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build', help='Write the tuning matrix')
    b.add_argument('--parent', type=Path, required=True, help='Completed full_pooling_validation directory')
    b.add_argument('--output', type=Path, required=True, help='Fresh experiment directory')
    b.add_argument('--esm-sidecars', type=Path, required=True)
    b.add_argument('--split-manifest', type=Path, default=None,
                   help='Defaults to the parent split. Point this at the stratified-difficulty manifest when it exists.')
    b.add_argument('--data', type=Path, required=True)
    b.add_argument('--rsasa', type=Path, required=True)
    b.add_argument('--model-lists', type=Path, default=None,
                   help='Override the parent model-list directory recorded in its matrix.')
    b.add_argument('--stage', choices=('screen', 'confirm'), default='screen')
    b.add_argument('--arms', nargs='*', default=None, help='Subset of arm names; defaults to all')
    v = sub.add_parser('verify', help='Report run completion')
    v.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.command == 'build':
        build(a.parent, a.output, a.esm_sidecars, a.split_manifest or (a.parent / 'target_splits.json'),
              a.data, a.rsasa, a.stage, a.arms, a.model_lists)
    else:
        verify(a.output)
