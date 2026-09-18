import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from dockq_objectives import bin_indices, fit_range_weights, weighted_dockq_mse, prediction_report
from build_dockq_priority_sweep import build
from train_gate import resolve_loss_weights
from test_model_extensions import write_fixture


def test_weight_normalization_boundaries_and_cap():
    labels=[0.]*100+[.2,.4,.6,.8,1.]
    profile=fit_range_weights(labels)
    bins=bin_indices(labels); w=np.array(profile['weights'])
    assert bin_indices([0,.2,.4,.6,.8,1]).tolist()==[0,1,2,3,4,4]
    assert np.isclose(w[bins].mean(),1)
    assert w.max()/w.min()<=3+1e-9
    pred=torch.zeros(6,requires_grad=True); y=torch.tensor([0,.2,.4,.6,.8,1.])
    loss=weighted_dockq_mse(pred,y,w.tolist()); loss.backward()
    assert np.allclose(pred.grad.numpy(), -2*y.numpy()*w[[0,1,2,3,4,4]]/6)
    with pytest.raises(ValueError): fit_range_weights([1.1])


def test_report_weights_graphs_targets_and_bins_separately():
    r=prediction_report([0,0,1],[1,1,1],['a','a','b'])
    assert r['pooled_mse']==pytest.approx(2/3)
    assert r['macro_target_mse']==.5 and r['macro_bin_mse']==.5
    assert r['bins'][1]['mse'] is None


def test_lambda_preserves_target_weight():
    args=SimpleNamespace(node_weight=1,edge_attr_weight=1,edge_presence_weight=.1,target_weight=1,reconstruction_lambda=.03)
    assert resolve_loss_weights(args)==dict(node_mse=.03,edge_attr_mse=.03,edge_presence_bce=.003,target_mse=1)


def test_matrix_factorial_and_frozen_split(tmp_path):
    source=tmp_path/'source.json'
    original=dict(apbs_enabled=False,runs=[dict(config={'name':'baseline'},seed=7,output='old',
        argv=['--output-dir','old','--split-manifest','frozen.json','--edge-stats-seed','7','--num-workers','7'])])
    source.write_text(json.dumps(original)); m=build(source,tmp_path/'out')
    assert len(m['runs'])==36
    assert len({r['output'] for r in m['runs']})==36
    for r in m['runs']:
        a=r['argv']; assert a[a.index('--split-manifest')+1]=='frozen.json'
        assert a[a.index('--edge-stats-seed')+1]=='7'
        assert '--no-test-evaluation' in a and '--persistent-workers' not in a
    assert json.loads(source.read_text())==original


@pytest.mark.parametrize('lam,weighting',[(0,'inverse-sqrt'),(.1,'none'),(.03,'inverse-sqrt')])
def test_training_weighting_checkpoints_and_validation(tmp_path,lam,weighting):
    data=tmp_path/'data'; data.mkdir()
    for target in ('1abc','2abc','3abc'): write_fixture(data/f'{target}.hdf5')
    out=tmp_path/'run'
    cmd=[sys.executable,'train_gate.py','--data',str(data),'--output-dir',str(out),
        '--epochs','2','--num-workers','0','--cpu-threads','1','--device','cpu',
        '--no-test-evaluation','--checkpoint-metric','target_mse','--hidden-dim','8',
        '--latent-dim','4','--loss-weight-mode','fixed','--reconstruction-lambda',str(lam),
        '--dockq-range-weighting',weighting,'--dockq-range-diagnostics']
    result=subprocess.run(cmd,text=True,capture_output=True)
    assert result.returncode==0,result.stdout+result.stderr
    h=pd.read_csv(out/'loss_history.csv'); f=pd.read_csv(out/'validation_predictions.csv')
    mse=np.mean((f.true_target-f.predicted_target)**2)
    assert mse==pytest.approx(h.val_target_mse.min(),rel=1e-5,abs=1e-7)
    records=[json.loads(line) for line in (out/'dockq_epoch_metrics.jsonl').read_text().splitlines()]
    assert len(records)==2
    for r in records:
        assert np.isfinite(list(r['first_train_batch_gradients'].values())).all()
        if lam==0: assert r['first_train_batch_gradients']['gradient_structure']==0
    assert not (out/'test_predictions.csv').exists()
    assert h.filter(regex='^test_').isna().all().all()


def test_preflight_reads_only_training_labels_and_rejects_overlap(tmp_path):
    import h5py
    from preflight_dockq_priority_sweep import preflight
    data=tmp_path/'data'; data.mkdir()
    splits={}
    for split,name in [('train','1abc'),('val','2abc'),('test','3abc')]:
        path=data/(name+'.hdf5')
        with h5py.File(path,'w') as f:
            g=f.create_group('graph')
            # Invalid holdout labels prove they are not used to fit weights.
            g.create_dataset('target_scores/DockQ',data=.8 if split=='train' else 99.)
        splits[split]={'paths':[str(path)],'sample_count':1}
    manifest=tmp_path/'split.json'; manifest.write_text(json.dumps({'splits':splits}))
    source=tmp_path/'matrix.json'
    source.write_text(json.dumps({'runs':[{'config':{'name':'baseline'},'seed':7,
        'argv':['--data',str(data),'--optional-node-features-dir',str(data),'--split-manifest',str(manifest)]}]}))
    preflight(source)
    splits['val']['paths']=splits['train']['paths']
    manifest.write_text(json.dumps({'splits':splits}))
    with pytest.raises(ValueError,match='overlapping'): preflight(source)
