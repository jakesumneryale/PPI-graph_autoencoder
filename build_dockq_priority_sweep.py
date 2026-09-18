"""Prepare a 36-run, validation-only DockQ/structure factorial sweep."""
import argparse
import copy
import json
import math
import hashlib
from pathlib import Path


def set_option(argv, flag, value):
    if flag in argv:
        argv[argv.index(flag)+1] = str(value)
    else:
        argv.extend([flag, str(value)])


def build(source, output, seeds=(7,17,27), lambdas=(0.,.01,.03,.1,.3,1.)):
    matrix=json.loads(source.read_text())
    if matrix.get('apbs_enabled') is not False:
        raise ValueError('Use the existing APBS-free matrix')
    reference=[r for r in matrix['runs'] if r['config']['name']=='baseline' and r['seed']==7]
    if len(reference)!=1:
        raise ValueError('Expected exactly one baseline seed 7')
    if output.exists():
        raise FileExistsError(f'{output} already exists; choose a fresh directory to protect results')
    runs=[]
    for seed in seeds:
        for lam in lambdas:
            if lam < 0 or not math.isfinite(lam):
                raise ValueError('Lambda must be finite and nonnegative')
            for weighting in ('none','inverse-sqrt'):
                tag=f'lambda{lam:g}_{weighting}_seed{seed}'
                run=copy.deepcopy(reference[0]); run['seed']=seed
                run['config']['name']=tag
                run['output']=str((output/tag).resolve())
                argv=run['argv']
                # Disable persistent pools: only the active loader has seven workers.
                if '--persistent-workers' in argv: argv.remove('--persistent-workers')
                values={'--seed':seed, '--output-dir':run['output'], '--dropout':.3,
                    '--reconstruction-lambda':lam, '--dockq-range-weighting':weighting,
                    '--dockq-weight-cap':3, '--node-weight':1, '--edge-attr-weight':1,
                    '--edge-presence-weight':.1, '--target-weight':1, '--loss-weight-mode':'fixed',
                    '--checkpoint-metric':'target_mse', '--num-workers':7, '--cpu-threads':1,
                    '--worker-start-method':'spawn', '--epochs':50}
                for k,v in values.items(): set_option(argv,k,v)
                for flag in ('--no-test-evaluation','--validation-only-during-training','--dockq-range-diagnostics'):
                    if flag not in argv: argv.append(flag)
                run['dockq_sweep']={'reconstruction_lambda':lam,'range_weighting':weighting,'dropout':.3}
                runs.append(run)
    matrix.update(runs=runs, diagnostic_reference=reference[0], validation_only=True,
        primary_metric='Unweighted graph-average validation DockQ MSE',
        sweep_description='6 reconstruction weights x 2 range weightings x 3 seeds; fixed target split and edge-stats seed')
    code_root=Path(__file__).parent
    matrix['code_sha256']={name:hashlib.sha256((code_root/name).read_bytes()).hexdigest()
        for name in ('train_gate.py','GATE_model.py','dockq_objectives.py','protein_hdf5_dataset.py','build_dockq_priority_sweep.py')}
    output.mkdir(parents=True)
    (output/'logs').mkdir()
    (output/'matrix.json').write_text(json.dumps(matrix,indent=2)+'\n')
    print(f'{len(runs)} runs; frozen split: {matrix.get("split")}; matrix: {output / "matrix.json"}')
    return matrix


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); build(args.source,args.output)
