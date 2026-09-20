"""Sequence-aligned ESM sidecars; source graphs and cohort remain unchanged."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import h5py
import numpy as np


def reference_rows(graph):
    ref=graph['node_reference'].asstr()[()]
    if ref.ndim!=2 or ref.shape[1]!=3 or not np.array_equal(ref[:,0].astype(int),np.arange(len(ref))):
        raise ValueError('Expected sequential [node ID, chain ID, amino acid] node_reference')
    rows=[(str(c),str(aa)) for _,c,aa in ref]
    if any(len(aa)!=1 for _,aa in rows):raise ValueError('Invalid residue code')
    return rows


def reference_hash(rows):
    return hashlib.sha256(json.dumps(rows,separators=(',',':')).encode()).hexdigest()


def read_sidecar(directory,target,name,graph):
    with h5py.File(Path(directory)/f'{target}.hdf5','r') as f:
        ds=f['models'][name]
        if ds.attrs['reference_sha256']!=reference_hash(reference_rows(graph)):
            raise ValueError(f'ESM node order mismatch: {target}/{name}')
        return ds[()]


def prepare(parent, target, esm_root, output):
    import torch
    from prepare_model_extensions import load_embeddings
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    matrix=json.loads((parent/'matrix.json').read_text());argv=matrix['runs'][0]['argv']
    def value(flag):return argv[argv.index(flag)+1]
    paths=[Path(p) for p in json.loads((parent/'paths.json').read_text())]
    path=next(p for p in paths if p.stem==target)
    names=sorted(n for n in (Path(value('--model-list-dir'))/f'{target}.txt').read_text().splitlines() if n)
    output.mkdir(parents=True,exist_ok=True)
    final=output/f'{target}.hdf5'
    # Outputs are rebuilt atomically; existing graph data is never modified.
    fd,tmp=tempfile.mkstemp(prefix=f'.{target}.',suffix='.hdf5',dir=output);os.close(fd)
    failures={};layouts={};widths=set()
    try:
        with h5py.File(path,'r') as source,h5py.File(tmp,'w') as out:
            models=out.create_group('models');unique=out.create_group('embeddings')
            for name in names:
                try:
                    rows=reference_rows(source[name]);digest=reference_hash(rows)
                    if digest not in unique:
                        values,provenance=load_embeddings(esm_root,target,name,rows,'{target}_all.fasta','{target}.{chain}.pt',33)
                        if not np.isfinite(values).all():raise ValueError('Nonfinite ESM embedding')
                        if values.shape!=(len(rows),1280):raise ValueError(f'Expected layer-33 ESM2 width 1280, found {values.shape}')
                        ds=unique.create_dataset(digest,data=values,compression='gzip')
                        ds.attrs['reference_sha256']=digest;ds.attrs['provenance']=json.dumps(provenance)
                    models[name]=unique[digest];widths.add(unique[digest].shape[1]);layouts[digest]=True
                except (KeyError,ValueError,FileNotFoundError) as exc:failures[name]=str(exc)
            out.attrs['source_graph']=str(path);out.attrs['layer']=33
        audit=dict(target=target,expected=len(names),accepted=len(names)-len(failures),failures=failures,
                   unique_layouts=len(layouts),widths=sorted(widths))
        (output/f'{target}.audit.json').write_text(json.dumps(audit,indent=2)+'\n')
        if failures:raise ValueError(f'{target}: {len(failures)} ESM alignment failures; cohort not reduced; see audit')
        os.replace(tmp,final)
        print(f'{target}: {len(names)} aligned graphs, {len(layouts)} shared embedding layouts; 1 process/1 thread',flush=True)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--task',type=int,required=True)
    p.add_argument('--esm-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();paths=json.loads((a.parent/'paths.json').read_text())
    prepare(a.parent,Path(paths[a.task-1]).stem,a.esm_root,a.output)
