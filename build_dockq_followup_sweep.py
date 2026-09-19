"""Prepare diagnostics, pooling/range-weight sweeps and checkpoint fine-tuning."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from build_dockq_priority_sweep import set_option


def build(source, output, verify_inputs=True):
    prior=json.loads(source.read_text())
    if not prior.get('validation_only') or prior.get('apbs_enabled') is not False:
        raise ValueError('Expected the completed APBS-free DockQ priority sweep')
    if output.exists(): raise FileExistsError(f'Choose a fresh output directory: {output}')
    controls={}
    for seed in (7,17,27):
        candidates=[r for r in prior['runs'] if r['seed']==seed and r['dockq_sweep']['reconstruction_lambda']==1 and r['dockq_sweep']['range_weighting']=='inverse-sqrt']
        if len(candidates)!=1: raise ValueError(f'Missing or ambiguous control seed {seed}')
        r=copy.deepcopy(candidates[0]); path=source.parent/Path(r['output']).name
        r['output']=str(path.resolve())
        if verify_inputs:
            for name in ('gate_model.pt','validation_predictions.csv','loss_history.csv'):
                if not (path/name).is_file(): raise FileNotFoundError(path/name)
        controls[seed]=r
    runs=[]; reused=[]
    def add(seed, name, stage, pooling='all', exponent=.5, lam=1., initialize=False):
        run=copy.deepcopy(controls[seed]); run['output']=str((output/name).resolve())
        run['config'].update(name=name,pooling=pooling)
        argv=run['argv']
        for flag in ('--persistent-workers','--evaluation-only','--export-training-predictions'):
            if flag in argv: argv.remove(flag)
        opts={'--output-dir':run['output'],'--pooling':pooling,'--dockq-weight-exponent':exponent,
            '--reconstruction-lambda':lam,'--num-workers':7,'--cpu-threads':1,'--worker-start-method':'spawn',
            '--epochs':20 if stage=='finetune' else 50,'--lr':'3e-5' if stage=='finetune' else '3e-4',
            '--warmup-steps':0 if stage=='finetune' else 1000,'--dockq-weight-cap':3,
            '--dropout':.3,'--loss-weight-mode':'fixed','--checkpoint-metric':'target_mse'}
        if initialize: opts['--initialize-checkpoint']=str(Path(controls[seed]['output'])/'gate_model.pt')
        for k,v in opts.items(): set_option(argv,k,v)
        for flag in ('--no-test-evaluation','--validation-only-during-training','--dockq-range-diagnostics','--export-training-predictions'):
            if flag not in argv: argv.append(flag)
        if stage=='diagnostic': argv.append('--evaluation-only')
        run['followup']=dict(stage=stage,pooling=pooling,exponent=exponent,reconstruction_lambda=lam,
                             parent_checkpoint=opts.get('--initialize-checkpoint'))
        run.pop('dockq_sweep',None)
        runs.append(run)
    for seed in (7,17,27): add(seed,f'diagnostic_seed{seed}','diagnostic',initialize=True)
    for seed in (7,17,27):
        for pooling in ('all','interface','combined'):
            for exponent in (.5,.75,1.):
                if pooling=='all' and exponent==.5:
                    reused.append(dict(seed=seed,output=controls[seed]['output'],pooling=pooling,exponent=exponent))
                else: add(seed,f'pool_{pooling}_alpha{exponent:g}_seed{seed}','pooling',pooling,exponent)
    for seed in (7,17,27):
        for lam in (0.,.1,1.): add(seed,f'finetune_lambda{lam:g}_seed{seed}','finetune',lam=lam,initialize=True)
    root=Path(__file__).parent
    m=dict(runs=runs,reused_controls=reused,apbs_enabled=False,validation_only=True,
        source_matrix=str(source.resolve()),split=prior['split'],cohort_keys=prior.get('cohort_keys'),
        primary_metric='Unweighted graph-average validation DockQ MSE',
        code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                     [root/'train_gate.py',root/'GATE_model.py',root/'dockq_objectives.py',Path(__file__)]})
    output.mkdir(parents=True); (output/'logs').mkdir()
    (output/'matrix.json').write_text(json.dumps(m,indent=2)+'\n')
    print(f'{len(runs)} jobs: 3 evaluation diagnostics, 24 new pooling runs, 9 fine-tunes; 3 completed controls reused.')
    return m

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); build(args.source,args.output)
