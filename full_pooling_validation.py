"""Read-only full-graph audit and fixed-split 12-run pooling validation matrix."""
import argparse
from collections import Counter
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
import h5py
import numpy as np
from add_interface_node_degree import calculate_interface_node_degree, normalize_contacts
from build_dockq_priority_sweep import set_option

NODES=('aa_type','chain','interface_nodes','rsasa_i','interface_node_degree')
EDGES=('interface_edges','ca_dist','voronoi_contact_area','voronoi_contact_missing')


def write_json(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=path.parent,prefix='.'+path.name)
    try:
        with os.fdopen(fd,'w') as f: json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def audit_code_signature():
    root=Path(__file__).parent
    return hashlib.sha256(Path(__file__).read_bytes()+(root/'add_interface_node_degree.py').read_bytes()).hexdigest()


def signature(path):
    stat=path.stat()
    return dict(path=str(path.resolve()),size=stat.st_size,mtime_ns=stat.st_mtime_ns)


def inventory(data, output):
    paths=sorted([*data.glob('*.hdf5'),*data.glob('*.h5')])
    if not paths:raise ValueError(f'No graph files: {data}')
    if len({p.stem for p in paths})!=len(paths):raise ValueError('Duplicate target filenames')
    for p in paths:
        with h5py.File(p,'r') as f:
            if 'voronoi_subset_model_count' in f.attrs:raise ValueError(f'Subset input rejected: {p}')
    write_json(output,[str(p.resolve()) for p in paths])
    print(len(paths))


def read_rsasa(path):
    values={}
    with path.open() as f:
        reader=csv.DictReader(f)
        if not {'Decoy','avg_rsasa_i'}<=set(reader.fieldnames or []):raise ValueError(f'Invalid rSASA columns: {path}')
        for row in reader:
            name=row['Decoy'].strip()
            if not name:continue
            if name in values:raise ValueError(f'Duplicate rSASA graph: {name}')
            if row['avg_rsasa_i'].strip():values[name]=float(row['avg_rsasa_i'])
    return values


def validate_graph(g, rsasa):
    nf=g['node_features'];ef=g['edge_features']
    node_values={name:np.asarray(nf[name][()]) for name in NODES if name!='rsasa_i'}
    n=len(node_values['interface_nodes'])
    if n==0:raise ValueError('No nodes')
    for name,v in node_values.items():
        if v.ndim not in (1,2) or len(v)!=n or not np.isfinite(v).all():raise ValueError(f'Invalid node feature {name}')
    mask=node_values['interface_nodes'].reshape(-1)
    if len(mask)!=n or not np.isin(mask,[0,1]).all() or not mask.any():raise ValueError('Invalid or empty interface mask')
    if not np.isfinite(rsasa):raise ValueError('Missing/nonfinite rSASA')
    # Labels are validated for completeness/range only, never used to select performance.
    y=float(g['target_scores/DockQ'][()])
    if not np.isfinite(y) or not 0<=y<=1:raise ValueError('Invalid DockQ label')
    contacts=np.asarray(ef['contacts'][()]); normalized=normalize_contacts(contacts,n)
    if not len(normalized):raise ValueError('No contacts')
    for name in EDGES:
        v=np.asarray(ef[name][()])
        if v.ndim not in (1,2) or len(v)!=len(contacts) or not np.isfinite(v).all():raise ValueError(f'Invalid edge feature {name}')
        if name in ('ca_dist','voronoi_contact_area') and (v<0).any():raise ValueError(f'Negative {name}')
        if name=='voronoi_contact_missing' and not np.isin(v,[0,1]).all():raise ValueError('Invalid area missing mask')
    degree=calculate_interface_node_degree(node_values['interface_nodes'],contacts)
    if not np.array_equal(degree.reshape(-1),node_values['interface_node_degree'].reshape(-1)):
        raise ValueError('Interface-node degree inconsistent with contacts')
    return dict(node_width=sum(1 if v.ndim==1 else v.shape[1] for v in node_values.values())+1,
                edge_width=sum(1 if ef[k].ndim==1 else ef[k].shape[1] for k in EDGES))


def audit(path, rsasa_dir, output):
    csv_path=rsasa_dir/f'{path.stem}_avg_rSASA_i.csv'
    stamp=dict(graph=signature(path),rsasa=signature(csv_path) if csv_path.exists() else None,
               audit_code=audit_code_signature())
    report_path=output/f'{path.stem}.json'
    if report_path.exists():
        old=json.loads(report_path.read_text())
        if old.get('signature')==stamp:
            print(f'{path.stem}: reused audit; {len(old["accepted"])} eligible graphs',flush=True);return old
    rsasa=read_rsasa(csv_path) if csv_path.exists() else {}
    accepted=[];rejected={};widths=set()
    with h5py.File(path,'r') as f:
        if 'voronoi_subset_model_count' in f.attrs:raise ValueError(f'Subset input rejected: {path}')
        raw=len(f)
        for name in sorted(f):
            try:
                dimensions=validate_graph(f[name],rsasa.get(name,float('nan')))
                widths.add((dimensions['node_width'],dimensions['edge_width']));accepted.append(name)
            except (KeyError,ValueError,TypeError,IndexError) as exc:rejected[name]=str(exc)
    if stamp['graph']!=signature(path) or stamp['rsasa']!=(signature(csv_path) if csv_path.exists() else None):
        raise RuntimeError('Inputs changed during audit')
    result=dict(target=path.stem,signature=stamp,raw_models=raw,accepted=accepted,rejected=rejected,
                rejection_counts=dict(Counter(rejected.values())),feature_widths=sorted(widths))
    write_json(report_path,result)
    print(f'{path.stem}: {len(accepted)}/{raw} eligible; {len(rejected)} rejected',flush=True)
    return result


def build(paths_file, audit_dir, prior_split, source_matrix, output, rsasa_dir, minimum=180000):
    if minimum < 1:raise ValueError('Minimum cohort size must be positive')
    if (output/'matrix.json').exists():raise FileExistsError('Experiment matrix already exists; choose a fresh experiment directory')
    paths=[Path(p) for p in json.loads(paths_file.read_text())]
    original_split=json.loads(prior_split.read_text())
    target_assignment={}
    for split in ('train','val','test'):
        for target in original_split['splits'][split]['targets']:
            if target in target_assignment:raise ValueError('Overlapping prior target split')
            target_assignment[target]=split
    reports=[]
    for path in paths:
        r=json.loads((audit_dir/f'{path.stem}.json').read_text())
        csv_path=rsasa_dir/f'{path.stem}_avg_rSASA_i.csv'
        expected=dict(graph=signature(path),rsasa=signature(csv_path) if csv_path.exists() else None,
                      audit_code=audit_code_signature())
        if r['signature']!=expected:raise ValueError(f'Stale audit: {path}')
        reports.append(r)
    raw=sum(r['raw_models'] for r in reports);eligible=sum(len(r['accepted']) for r in reports)
    outside=[r['target'] for r in reports if r['accepted'] and r['target'] not in target_assignment]
    absent=sorted(set(target_assignment)-{r['target'] for r in reports if r['accepted']})
    coverage=dict(raw_models=raw,eligible_models=eligible,rejected_models=raw-eligible,minimum_required=minimum,
                  eligible_targets_outside_frozen_split=outside,frozen_targets_without_eligible_models=absent,
                  targets=[dict(target=r['target'],raw=r['raw_models'],eligible=len(r['accepted']),rejection_counts=r['rejection_counts']) for r in reports])
    write_json(output/'coverage.json',coverage)
    if outside or absent:raise ValueError('Full dataset differs in eligible target membership from frozen split; see coverage.json. No automatic reassignment.')
    if eligible<minimum:raise ValueError(f'Only {eligible:,} eligible of {raw:,} raw models; minimum {minimum:,}. See coverage.json. GPU training blocked.')
    widths={tuple(v) for r in reports for v in r['feature_widths']}
    if len(widths)!=1:raise ValueError(f'Inconsistent input widths: {widths}')
    lists=output/'model_lists';lists.mkdir(parents=True,exist_ok=True)
    split_data={s:dict(targets=[],paths=[],sample_count=0,target_count=0) for s in ('train','val','test')}
    for path,r in zip(paths,reports):
        (lists/f'{path.stem}.txt').write_text(''.join(n+'\n' for n in r['accepted']))
        if not r['accepted']:continue
        s=split_data[target_assignment[path.stem]];s['targets'].append(path.stem);s['paths'].append(str(path.resolve()));s['sample_count']+=len(r['accepted']);s['target_count']+=1
    manifest=output/'target_splits.json'
    write_json(manifest,dict(data_path=str(paths[0].parent.resolve()),seed=7,splits=split_data,
        source_target_split=str(prior_split.resolve()),model_list_dir=str(lists.resolve())))
    source=json.loads(source_matrix.read_text())
    refs=[r for r in source['runs'] if r.get('followup',{}).get('stage')=='pooling' and r['followup']['pooling']=='combined' and r['followup']['exponent']==.75 and r['seed']==7]
    if len(refs)!=1:raise ValueError('Expected unique combined alpha=.75 seed7 reference')
    runs=[]
    for seed in (7,17,27):
        for pool in ('all','combined'):
            for alpha in (.5,.75):
                r=copy.deepcopy(refs[0]);name=f'full_{pool}_alpha{alpha:g}_seed{seed}'
                r.update(seed=seed,output=str((output/name).resolve()))
                r['config'].update(name=name,pooling=pool)
                r.pop('followup',None);a=r['argv']
                for flag in ('--export-training-predictions','--persistent-workers'):
                    if flag in a:a.remove(flag)
                if any(flag in a for flag in ('--initialize-checkpoint','--evaluation-only','--use-esm')):raise ValueError('Expected from-scratch reference')
                if '--no-test-evaluation' not in a or '--dockq-range-diagnostics' not in a:
                    raise ValueError('Reference must be validation-only with DockQ range diagnostics')
                opts={'--data':paths[0].parent.resolve(),'--model-list-dir':lists.resolve(),'--split-manifest':manifest.resolve(),
                      '--optional-node-features-dir':rsasa_dir.resolve(),'--seed':seed,'--output-dir':r['output'],
                      '--pooling':pool,'--dockq-weight-exponent':alpha,'--epochs':50,'--batch-size':16,'--num-workers':7,
                      '--cpu-threads':1,'--worker-start-method':'spawn','--dropout':.3,'--lr':'3e-4',
                      '--warmup-steps':1000,'--reconstruction-lambda':1,'--dockq-range-weighting':'inverse-sqrt',
                      '--dockq-weight-cap':3,'--checkpoint-metric':'target_mse','--node-features':','.join(NODES),'--edge-features':','.join(EDGES)}
                for flag,value in opts.items():set_option(a,flag,value)
                r['full_validation']=dict(pooling=pool,exponent=alpha,reconstruction_lambda=1,eligible_models=eligible)
                runs.append(r)
    write_json(output/'matrix.json',dict(runs=runs,apbs_enabled=False,validation_only=True,
        full_dataset=True,eligible_models=eligible,raw_models=raw,split=str(manifest.resolve()),
        source_matrix=str(source_matrix.resolve()),
        code_sha256={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
            for name in ('train_gate.py','GATE_model.py','dockq_objectives.py','protein_hdf5_dataset.py')},primary_metric='Unweighted validation DockQ MSE',
        input_features=dict(nodes=NODES,edges=EDGES),feature_widths=sorted(widths)))
    print(f'{eligible:,} eligible full-dataset graphs; {len(runs)} GPU jobs ready. Split counts: '+str({k:v['sample_count'] for k,v in split_data.items()}),flush=True)
    return runs


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    a=sub.add_parser('inventory');a.add_argument('--data',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a=sub.add_parser('audit');a.add_argument('--paths',type=Path,required=True);a.add_argument('--task',type=int,required=True);a.add_argument('--rsasa',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a=sub.add_parser('build')
    for name in ('paths','audit-dir','prior-split','source-matrix','output','rsasa'):a.add_argument('--'+name,type=Path,required=True)
    a.add_argument('--minimum',type=int,default=180000)
    args=p.parse_args()
    if args.action=='inventory':inventory(args.data,args.output)
    elif args.action=='audit':
        paths=json.loads(args.paths.read_text());audit(Path(paths[args.task-1]),args.rsasa,args.output)
    else:build(args.paths,args.audit_dir,args.prior_split,args.source_matrix,args.output,args.rsasa,args.minimum)
if __name__=='__main__':main()
