import copy
import json
import hashlib
import h5py
import numpy as np
import pytest
from test_full_pooling_validation import fixture
from full_pooling_validation import build as build_parent
from build_full_esm_validation import build
from esm_sidecars import reference_rows,reference_hash,read_sidecar


def setup(tmp_path):
    data,rs,cache,paths,split,source=fixture(tmp_path)
    parent=tmp_path/'parent';build_parent(paths,cache,split,source,parent,rs,minimum=3)
    (parent/'paths.json').write_text(paths.read_text())
    sidecars=tmp_path/'esm';sidecars.mkdir()
    for p in data.glob('*.hdf5'):
        with h5py.File(p,'r+') as f:
            f['g/node_reference']=np.array([['0','1','A'],['1','2','C'],['2','2','D']],dtype=h5py.string_dtype())
            digest=reference_hash(reference_rows(f['g']))
        with h5py.File(sidecars/p.name,'w') as f:
            ds=f.create_dataset('models/g',data=np.ones((3,1280),dtype=np.float32));ds.attrs['reference_sha256']=digest
        (sidecars/f'{p.stem}.audit.json').write_text(json.dumps({'accepted':1,'failures':{}}))
    return parent,sidecars,data


def test_six_matched_runs_preserve_controls_and_sources(tmp_path):
    parent,sidecars,data=setup(tmp_path)
    before=(parent/'matrix.json').read_bytes();raw={p:p.read_bytes() for p in data.glob('*')}
    result=build(parent,sidecars,tmp_path/'out')
    assert len(result['runs'])==6
    assert {(r['seed'],r['full_validation']['pooling']) for r in result['runs']}=={(s,p) for s in (7,17,27) for p in ('all','combined')}
    controls={r['output']:r for r in json.loads(before)['runs']}
    for r in result['runs']:
        old=controls[r['matched_control']]['argv'];new=r['argv']
        assert '--use-esm' in new
        for flag in ('--data','--model-list-dir','--split-manifest','--seed','--reconstruction-lambda'):
            assert new[new.index(flag)+1]==old[old.index(flag)+1]
    assert before==(parent/'matrix.json').read_bytes()
    assert all(p.read_bytes()==b for p,b in raw.items())


def test_missing_graph_blocks_matrix(tmp_path):
    parent,sidecars,_=setup(tmp_path)
    with h5py.File(sidecars/'1abc.hdf5','r+') as f:del f['models/g']
    with pytest.raises(ValueError,match='cohort mismatch'):build(parent,sidecars,tmp_path/'out')


def test_changed_node_order_rejected(tmp_path):
    parent,sidecars,data=setup(tmp_path)
    with h5py.File(data/'1abc.hdf5','r+') as f:
        assert read_sidecar(sidecars,'1abc','g',f['g']).shape==(3,1280)
        f['g/node_reference'][1,2]='E'
        with pytest.raises(ValueError,match='order mismatch'):read_sidecar(sidecars,'1abc','g',f['g'])
    with pytest.raises(ValueError,match='alignment mismatch'):build(parent,sidecars,tmp_path/'out')


def test_launcher_dependencies_and_serial_preparation():
    from pathlib import Path
    launcher=Path('cluster/submit_full_esm_validation.sh').read_text()
    assert '--array=1-6' in launcher
    assert '--dependency="afterok:$PREP_ID"' in launcher
    assert '--dependency="afterok:$BUILD_ID"' in launcher
    assert 'nb685/scratch_backup' in launcher
    assert '#SBATCH --cpus-per-task=1' in Path('cluster/full_esm_preflight.slurm').read_text()
    import re
    assert not any('%' in arg for arg in re.findall(r'--array=\S+', launcher))


def test_pending_parent_build_queues_with_dependency(tmp_path):
    import os
    import subprocess
    from pathlib import Path
    parent=tmp_path/'parent';parent.mkdir();(parent/'paths.json').write_text('[]')
    project=tmp_path/'project';(project/'venv/bin').mkdir(parents=True)
    (project/'venv/bin/activate').write_text('')
    conda=tmp_path/'conda';(conda/'etc/profile.d').mkdir(parents=True)
    (conda/'etc/profile.d/conda.sh').write_text('')
    bootstrap=tmp_path/'bootstrap'
    bootstrap.write_text('module() { :; }\nconda() { [[ "$1" != info ]] || echo "$MOCK_CONDA"; }\npython() { echo 3; }\nsbatch() { echo "$*" >> "$CAPTURE"; echo 123; }\n')
    env=dict(os.environ,PROJECT_DIR=str(project),FULL_PARENT_DIR=str(parent),EXPERIMENT_DIR=str(tmp_path/'out'),ESM_SIDECARS=str(tmp_path/'sidecars'),BASH_ENV=str(bootstrap),MOCK_CONDA=str(conda),CAPTURE=str(tmp_path/'calls'),FULL_BUILD_JOB_ID='456')
    script=str(Path('cluster/submit_full_esm_validation.sh').resolve())
    result=subprocess.run(['bash',script],env=env,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    calls=(tmp_path/'calls').read_text().splitlines()
    assert len(calls)==3
    assert '--dependency=afterok:456' in calls[0]
    assert '--array=1-6' in calls[2]


def test_missing_parent_reports_inventory_without_submission(tmp_path):
    import os
    import subprocess
    from pathlib import Path
    env=dict(os.environ,PROJECT_DIR=str(tmp_path),FULL_PARENT_DIR=str(tmp_path/'missing'))
    result=subprocess.run(['bash',str(Path('cluster/submit_full_esm_validation.sh').resolve())],env=env,text=True,capture_output=True)
    assert result.returncode==2
    assert 'Missing parent inventory' in result.stderr
    assert 'Do not resubmit existing runs' in result.stderr
