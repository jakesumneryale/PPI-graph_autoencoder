import copy
import json
import subprocess
import sys

import pandas as pd

from build_generalization_diagnostics import build
from test_model_extensions import write_fixture


def test_diagnostic_matrix_changes_one_factor_and_preserves_source(tmp_path):
    source = tmp_path/'source.json'
    original = {'apbs_enabled': False, 'runs': [dict(config={'name':'baseline'}, seed=7,
        output='old', argv=['--data','unchanged','--split-manifest','frozen.json',
        '--lr','3e-4','--output-dir','old','--num-workers','7'])]}
    source.write_text(json.dumps(original))
    result = build(source,tmp_path/'diagnostics')
    assert json.loads(source.read_text()) == original
    assert len(result['runs']) == 4 and result['diagnostic_reference'] == original['runs'][0]
    for run in result['runs']:
        argv = run['argv']
        assert '--no-test-evaluation' in argv
        assert argv[argv.index('--data')+1] == 'unchanged'
        assert argv[argv.index('--split-manifest')+1] == 'frozen.json'


def test_validation_only_never_exports_test_predictions(tmp_path):
    data=tmp_path/'data';data.mkdir()
    for target in ('1abc','2abc','3abc'):
        write_fixture(data/f'{target}.hdf5')
    output=tmp_path/'run'
    result=subprocess.run([sys.executable,'train_gate.py','--data',str(data),'--output-dir',str(output),
        '--epochs','1','--num-workers','0','--device','cpu','--no-test-evaluation',
        '--checkpoint-metric','target_mse','--hidden-dim','8','--latent-dim','4'],
        text=True,capture_output=True)
    assert result.returncode == 0,result.stdout+result.stderr
    assert (output/'gate_model.pt').is_file()
    assert (output/'validation_predictions.csv').is_file()
    assert not (output/'test_predictions.csv').exists()
    assert not (output/'reconstruction_summary.csv').exists()
    history=pd.read_csv(output/'loss_history.csv')
    assert history.test_target_mse.isna().all()
