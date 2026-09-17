"""Check launcher configuration without submitting any cluster jobs."""
import json
import os
from pathlib import Path
import subprocess

import pytest


@pytest.mark.parametrize('mode', ['quick', 'full'])
def test_no_apbs_launcher_uses_original_subset_and_separate_outputs(tmp_path, mode):
    repo = Path(__file__).resolve().parent
    project = tmp_path / 'project'
    (project / 'gate_run/seed_replicates').mkdir(parents=True)
    (project / 'gate_run/seed_replicates/shared_target_splits.json').write_text('{}')
    (project / 'cluster').symlink_to(repo / 'cluster', target_is_directory=True)
    binary = tmp_path / 'bin'; binary.mkdir()
    fake = binary / 'sbatch'
    fake.write_text('''#!/usr/bin/env python3
import json,os,sys
from pathlib import Path
Path(os.environ['CAPTURE']).write_text(json.dumps({'argv':sys.argv[1:],
 'environment':{k:os.environ[k] for k in ['APBS_ENABLED','RUN_MODE','SUBSET_DIR','EXTENSION_DATA','EXPERIMENT_DIR','PRIOR_SPLIT']}}))
print('12345')
''')
    fake.chmod(0o755)
    env = {k:v for k,v in os.environ.items() if k not in
           ('SUBSET_DIR','EXTENSION_DATA','EXPERIMENT_DIR','PRIOR_SPLIT')}
    env.update(PROJECT_DIR=str(project), PATH=str(binary)+os.pathsep+env['PATH'],
               CAPTURE=str(tmp_path/'capture.json'), APBS_ENABLED='1')
    completed = subprocess.run(['bash',str(repo/'cluster/submit_model_extensions_no_apbs.sh'),mode],
                               env=env,text=True,capture_output=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    captured = json.loads((tmp_path/'capture.json').read_text())
    settings = captured['environment']
    assert settings['APBS_ENABLED'] == '0'
    assert settings['RUN_MODE'] == mode
    assert settings['SUBSET_DIR'] == str(project/'voronoi_dataset_audit/subset_hdf5')
    assert settings['EXTENSION_DATA'].endswith(f'extension_data_10pct_{mode}_no_apbs')
    assert settings['EXPERIMENT_DIR'].endswith(f'model_extensions_10pct_{mode}_no_apbs')
    assert settings['PRIOR_SPLIT'].endswith('seed_replicates/shared_target_splits.json')
    assert '--export=ALL,STAGE=preflight' in captured['argv']
