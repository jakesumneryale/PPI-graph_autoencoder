import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Batch
from test_model_extensions import graph, write_fixture
from EGNN_model import build_graph_model
from dockq_objectives import fit_range_weights
from build_dockq_followup_sweep import build


@pytest.mark.parametrize('architecture',['gat','egnn'])
def test_combined_pooling_and_gradients(architecture):
    model=build_graph_model(architecture,in_node_feats=3,in_edge_feats=2,hidden_dim=8,latent_dim=4,gat_heads=2,pooling='combined',dropout=0)
    data=Batch.from_data_list([graph(),graph()]); z=torch.arange(32).reshape(8,4).float()
    pooled=model.pool_nodes(z,data.batch,data)
    torch.testing.assert_close(pooled[0,:4],z[:4].mean(0))
    torch.testing.assert_close(pooled[0,4:8],z[:4].max(0).values)
    torch.testing.assert_close(pooled[0,8:12],z[1:3].mean(0))
    torch.testing.assert_close(pooled[0,12:],z[1:3].max(0).values)
    result=model(data);result['quality_pred'].sum().backward()
    assert model.node_projector.weight.grad is not None
    data.interface_mask[:4]=False
    with pytest.raises(ValueError,match='no interface'):model(data)


def test_exponents():
    labels=[0.]*100+[.9]*10
    w=[fit_range_weights(labels,cap=10,exponent=e)['weights'] for e in (.5,.75,1.)]
    assert w[0][4]<w[1][4]<w[2][4]
    for weights in w: assert np.isclose((100*weights[0]+10*weights[4])/110,1)
    with pytest.raises(ValueError): fit_range_weights(labels,exponent=-1)


def test_followup_matrix(tmp_path):
    source=Path('gate_run/dockq_priority_sweep/matrix.json')
    matrix=build(source,tmp_path/'sweep',verify_inputs=False)
    assert len(matrix['runs'])==36 and len(matrix['reused_controls'])==3
    assert len({r['output'] for r in matrix['runs']})==36
    for r in matrix['runs']:
        a=r['argv']; f=r['followup']
        assert '--no-test-evaluation' in a and '--persistent-workers' not in a
        assert a[a.index('--cpu-threads')+1]=='1'
        if f['stage']=='pooling': assert '--initialize-checkpoint' not in a
        else: assert '--initialize-checkpoint' in a
        if f['stage']=='finetune': assert a[a.index('--epochs')+1]=='20'


def test_reload_diagnostics_and_finetune(tmp_path):
    data=tmp_path/'data';data.mkdir()
    for t in ('1abc','2abc','3abc'): write_fixture(data/(t+'.hdf5'))
    common=['--data',str(data),'--epochs','1','--num-workers','0','--cpu-threads','1','--device','cpu',
        '--hidden-dim','8','--latent-dim','4','--no-test-evaluation','--checkpoint-metric','target_mse',
        '--loss-weight-mode','fixed','--dockq-range-diagnostics','--dockq-range-weighting','inverse-sqrt']
    def run(name,extra):
        out=tmp_path/name
        result=subprocess.run([sys.executable,'train_gate.py',*common,'--output-dir',str(out),*extra],capture_output=True,text=True)
        assert result.returncode==0,result.stdout+result.stderr
        return out
    source=run('base',[])
    diag=run('diag',['--initialize-checkpoint',str(source/'gate_model.pt'),'--evaluation-only'])
    a=pd.read_csv(source/'validation_predictions.csv');b=pd.read_csv(diag/'validation_predictions.csv')
    np.testing.assert_allclose(a.predicted_target,b.predicted_target)
    assert (diag/'training_predictions.csv').is_file()
    coverage=pd.read_csv(diag/'training_target_coverage.csv')
    assert coverage.n.sum()==len(pd.read_csv(diag/'training_predictions.csv'))
    assert not (diag/'loss_history.csv').exists()
    fine=run('fine',['--initialize-checkpoint',str(source/'gate_model.pt'),'--reconstruction-lambda','0','--lr','1e-30','--export-training-predictions'])
    ckpt=torch.load(fine/'gate_model.pt',map_location='cpu')
    initial=json.loads((fine/'initial_validation.json').read_text())
    f=pd.read_csv(fine/'validation_predictions.csv'); mse=np.mean((f.true_target-f.predicted_target)**2)
    assert mse<=initial['pooled_mse']+1e-7
    assert ckpt['best_epoch']==0
    assert not (fine/'test_predictions.csv').exists()


def test_interface_preflight_rejects_empty_interfaces(tmp_path):
    import h5py
    from preflight_dockq_priority_sweep import preflight
    data=tmp_path/'data';data.mkdir();splits={}
    for split,target in [('train','1abc'),('val','2abc'),('test','3abc')]:
        path=data/f'{target}.hdf5'
        with h5py.File(path,'w') as f:
            f.create_dataset('g/target_scores/DockQ',data=.5)
            f.create_dataset('g/node_features/interface_nodes',data=[0,0])
        splits[split]=dict(paths=[str(path)],sample_count=1)
    manifest=tmp_path/'split.json';manifest.write_text(json.dumps(dict(splits=splits)))
    source=tmp_path/'matrix.json';source.write_text(json.dumps(dict(runs=[dict(config={'name':'baseline'},seed=7,
        argv=['--data',str(data),'--optional-node-features-dir',str(data),'--split-manifest',str(manifest)])])))
    with pytest.raises(ValueError,match='interface nodes'):preflight(source,require_interface=True)
