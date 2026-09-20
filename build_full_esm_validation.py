"""Append six matched ESM runs in a separate experiment; never alter controls."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import h5py
from build_dockq_priority_sweep import set_option
from esm_sidecars import reference_rows,reference_hash


def build(parent, sidecars, output):
    prior=json.loads((parent/'matrix.json').read_text())
    if not prior.get('full_dataset') or not prior.get('validation_only'):raise ValueError('Expected full-dataset validation parent')
    if (output/'matrix.json').exists():raise FileExistsError('ESM matrix already exists')
    refs=[r for r in prior['runs'] if r['full_validation']['exponent']==.75]
    if {(r['full_validation']['pooling'],r['seed']) for r in refs}!={(p,s) for p in ('all','combined') for s in (7,17,27)} or len(refs)!=6:
        raise ValueError('Expected six matched alpha=.75 controls')
    a=refs[0]['argv'];lists=Path(a[a.index('--model-list-dir')+1]);total=0
    for path in map(Path,json.loads((parent/'paths.json').read_text())):
        names={n for n in (lists/f'{path.stem}.txt').read_text().splitlines() if n}
        audit=json.loads((sidecars/f'{path.stem}.audit.json').read_text())
        if audit['failures'] or audit['accepted']!=len(names):raise ValueError(f'Incomplete ESM audit: {path.stem}')
        with h5py.File(sidecars/f'{path.stem}.hdf5','r') as f,h5py.File(path,'r') as source:
            if set(f['models'])!=names:raise ValueError('ESM cohort mismatch')
            for name in names:
                ds=f['models'][name]
                if ds.shape!=(len(reference_rows(source[name])),1280) or ds.attrs['reference_sha256']!=reference_hash(reference_rows(source[name])):
                    raise ValueError(f'ESM alignment mismatch {path.stem}/{name}')
        total+=len(names)
    if total!=prior['eligible_models']:raise ValueError('Full ESM cohort size differs from controls')
    runs=[]
    for ref in refs:
        r=copy.deepcopy(ref);name=Path(ref['output']).name+'_esm2';r['output']=str((output/name).resolve())
        r['config'].update(name=name,esm=True);r['full_validation']['esm']=True
        r['matched_control']=ref['output']
        if '--use-esm' not in r['argv']:r['argv'].append('--use-esm')
        for flag,value in {'--output-dir':r['output'],'--esm-sidecar-dir':sidecars.resolve(),'--esm-projection-dim':64}.items():set_option(r['argv'],flag,value)
        runs.append(r)
    output.mkdir(parents=True,exist_ok=True)
    # Preserve portable local analysis paths without altering parent artifacts.
    (output/'target_splits.json').write_text((parent/'target_splits.json').read_text())
    m={**prior,'runs':runs,'esm_enabled':True,'parent_experiment':str(parent.resolve()),'esm_layer':33,'esm_width':1280,'esm_projection_width':64}
    m['parent_code_sha256']=prior.get('code_sha256',{})
    m['code_sha256']={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in ('train_gate.py','GATE_model.py','protein_hdf5_dataset.py','esm_sidecars.py','build_full_esm_validation.py')}
    (output/'matrix.json').write_text(json.dumps(m,indent=2)+'\n')
    print(f'Six ESM2 runs ready on the same {total:,} full-dataset graphs; no control runs resubmitted.')
    return m
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','sidecars','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();build(a.parent,a.sidecars,a.output)
