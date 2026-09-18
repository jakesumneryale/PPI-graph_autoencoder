"""Create four controlled validation-only experiments from the completed baseline.

Reuse the existing baseline as the control. No preprocessing or test evaluation.
These runs test hypotheses; none is presumed to fix generalization.
"""
import argparse
import copy
import json
from pathlib import Path


def build(source, output):
    matrix = json.loads(source.read_text())
    if matrix.get('apbs_enabled') is not False:
        raise ValueError('Use the completed APBS-free experiment matrix')
    baseline = [r for r in matrix['runs'] if r['config']['name'] == 'baseline' and r['seed'] == 7]
    if len(baseline) != 1:
        raise ValueError('Expected exactly one baseline seed 7')
    if output.exists():
        raise FileExistsError('Choose a fresh diagnostic output directory')
    output.mkdir(parents=True)
    changes = {
        'lower_lr': {'--lr': '3e-5'},
        'smaller_encoder': {'--hidden-dim': '32', '--latent-dim': '16', '--gat-heads': '2'},
        'more_dropout': {'--dropout': '0.3'},
        'more_weight_decay': {'--weight-decay': '0.001'},
    }
    runs = []
    for name, overrides in changes.items():
        run = copy.deepcopy(baseline[0])
        run['config']['name'] = name
        run['output'] = str((output / name).resolve())
        argv = run['argv']
        for flag, value in {**overrides, '--output-dir': run['output']}.items():
            if flag in argv:
                argv[argv.index(flag)+1] = value
            else:
                argv.extend([flag, value])
        if '--no-test-evaluation' not in argv:
            argv.append('--no-test-evaluation')
        run['diagnostic_changes'] = overrides
        runs.append(run)
    matrix.update(runs=runs, diagnostic_reference=baseline[0],
        diagnostic_source=str(source.resolve()), validation_only=True,
        primary_metric='Validation DockQ MSE; use existing baseline history as control')
    path = output / 'matrix.json'
    path.write_text(json.dumps(matrix, indent=2)+'\n')
    print(f'Four validation-only runs written to {path}. Existing baseline is reused, not retrained.')
    return matrix


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.source, args.output)
