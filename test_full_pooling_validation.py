import json
from pathlib import Path
import h5py
import numpy as np
import pytest
from full_pooling_validation import inventory,audit,build


def fixture(tmp_path):
    data=tmp_path/'data';data.mkdir();rs=tmp_path/'rsasa';rs.mkdir();cache=tmp_path/'audit';cache.mkdir()
    for target in ('1abc','2abc','3abc'):
        with h5py.File(data/f'{target}.hdf5','w') as f:
            g=f.create_group('g');n=g.create_group('node_features');e=g.create_group('edge_features')
            for key,v in dict(aa_type=np.eye(3),chain=[[0],[1],[1]],interface_nodes=[[1],[1],[0]],interface_node_degree=[[1],[1],[1]]).items():n[key]=v
            for key,v in dict(contacts=[[0,1],[1,2]],interface_edges=[[1],[0]],ca_dist=[[3.],[4.]],voronoi_contact_area=[[2.],[0.]],voronoi_contact_missing=[[0],[1]]).items():e[key]=v
            g['target_scores/DockQ']=.7
        (rs/f'{target}_avg_rSASA_i.csv').write_text('Decoy,avg_rsasa_i\ng,0.3\n')
    paths=tmp_path/'paths.json';inventory(data,paths)
    for path in data.glob('*.hdf5'):audit(path,rs,cache)
    split=tmp_path/'split.json';split.write_text(json.dumps({'splits':{s:{'targets':[t]} for s,t in zip(('train','val','test'),('1abc','2abc','3abc'))}}))
    source=tmp_path/'source.json'
    source.write_text(json.dumps({'runs':[{'seed':7,'config':{'name':'pool_combined_alpha0.75_seed7'},'followup':{'stage':'pooling','pooling':'combined','exponent':.75},
        'argv':['--residual_connections','--architecture','gat','--loss-weight-mode','fixed','--node-weight','1','--edge-attr-weight','1','--edge-presence-weight','.1','--target-weight','1',
        '--checkpoint-metric','target_mse','--no-test-evaluation','--dockq-range-diagnostics','--dockq-range-weighting','inverse-sqrt','--edge-recon-features','interface_edges,ca_dist',
        '--standardize-edge-features','voronoi_contact_area','--edge-feature-transforms','voronoi_contact_area=log1p','--edge-stats-seed','7','--dropout','.3','--lr','.0003','--reconstruction-lambda','1']}]}))
    return data,rs,cache,paths,split,source


def test_full_selection_counts_and_flags(tmp_path):
    data,rs,cache,paths,split,source=fixture(tmp_path)
    out=tmp_path/'out';runs=build(paths,cache,split,source,out,rs,minimum=3)
    assert len(runs)==12 and len({r['output'] for r in runs})==12
    m=json.loads((out/'target_splits.json').read_text())
    assert all(s['sample_count']==1 for s in m['splits'].values())
    for r in runs:
        a=r['argv'];assert a[a.index('--data')+1]==str(data.resolve())
        assert a[a.index('--model-list-dir')+1]==str((out/'model_lists').resolve())
        assert '--no-test-evaluation' not in a and '--validation-only-during-training' not in a and '--max-samples' not in a
        assert a[a.index('--seed')+1] in ('7','17','27')
    with pytest.raises(FileExistsError):build(paths,cache,split,source,out,rs,minimum=3)


def test_coverage_gate_never_silently_trains_subset(tmp_path):
    _,rs,cache,paths,split,source=fixture(tmp_path);out=tmp_path/'out'
    with pytest.raises(ValueError,match='GPU training blocked'):build(paths,cache,split,source,out,rs)
    assert json.loads((out/'coverage.json').read_text())['eligible_models']==3
    assert not (out/'matrix.json').exists()


def test_audit_missing_feature_cache_invalidation_and_source_preservation(tmp_path):
    data,rs,cache,*_=fixture(tmp_path);p=data/'1abc.hdf5'
    before=p.read_bytes();r=audit(p,rs,cache);assert r['accepted']==['g'] and p.read_bytes()==before
    with h5py.File(p,'r+') as f:del f['g/edge_features/voronoi_contact_area']
    r=audit(p,rs,cache);assert not r['accepted'] and 'g' in r['rejected']


def test_subset_marker_rejected(tmp_path):
    data,rs,cache,*_=fixture(tmp_path);p=data/'1abc.hdf5'
    with h5py.File(p,'r+') as f:f.attrs['voronoi_subset_model_count']=1
    with pytest.raises(ValueError,match='Subset'):inventory(data,tmp_path/'new.json')


def test_target_split_not_silently_changed(tmp_path):
    data,rs,cache,paths,split,source=fixture(tmp_path)
    s=json.loads(split.read_text());s['splits']['train']['targets']=['other'];split.write_text(json.dumps(s))
    with pytest.raises(ValueError,match='target membership'):build(paths,cache,split,source,tmp_path/'out',rs,minimum=3)


def test_launcher_dependencies_and_resource_requests(tmp_path):
    import os
    import subprocess
    repo=Path(__file__).resolve().parent;project=tmp_path/'project';project.mkdir()
    (project/'venv/bin').mkdir(parents=True);(project/'venv/bin/activate').write_text('')
    conda=tmp_path/'conda';(conda/'etc/profile.d').mkdir(parents=True);(conda/'etc/profile.d/conda.sh').write_text('')
    bootstrap=tmp_path/'bootstrap.sh'
    bootstrap.write_text('module() { :; }\nconda() { if [[ "$1" == info ]]; then echo "$MOCK_CONDA"; fi; }\n')
    binary=tmp_path/'bin';binary.mkdir()
    fakepython=binary/'python';fakepython.write_text('#!/bin/bash\nif [[ "$*" == *" inventory "* ]]; then echo 146; fi\n');fakepython.chmod(0o755)
    fake=binary/'sbatch';fake.write_text('''#!/usr/bin/env python3
import json,os,sys
from pathlib import Path
p=Path(os.environ['CAPTURE'])
rows=p.read_text().splitlines() if p.exists() else []
with p.open('a') as f:f.write(json.dumps(sys.argv[1:])+'\\n')
print(100+len(rows))
''');fake.chmod(0o755)
    env=dict(os.environ,PROJECT_DIR=str(project),EXPERIMENT_DIR=str(project/'run'),AUDIT_DIR=str(project/'audit'),
             BASH_ENV=str(bootstrap),MOCK_CONDA=str(conda),CAPTURE=str(tmp_path/'capture'),PATH=str(binary)+os.pathsep+os.environ['PATH'])
    r=subprocess.run(['bash',str(repo/'cluster/submit_full_pooling_validation.sh')],env=env,capture_output=True,text=True)
    assert r.returncode==0,r.stdout+r.stderr
    calls=[json.loads(line) for line in (tmp_path/'capture').read_text().splitlines()]
    assert len(calls)==3
    assert '--array=1-146' in calls[0]
    assert '--dependency=afterok:100' in calls[1]
    assert '--dependency=afterok:101' in calls[2] and '--array=1-12' in calls[2]
    assert not any('%' in arg for c in calls for arg in c if arg.startswith('--array'))
    assert '#SBATCH --cpus-per-task=1' in (repo/'cluster/full_pooling_preflight.slurm').read_text()
