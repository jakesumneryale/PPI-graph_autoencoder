"""Serial input preflight before allocating GPUs; never reads test labels."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
from dockq_objectives import fit_range_weights


def preflight(source, require_interface=False):
    matrix=json.loads(source.read_text())
    run=next(r for r in matrix['runs'] if r['config']['name']=='baseline' and r['seed']==7)
    argv=run['argv']
    def value(flag): return argv[argv.index(flag)+1]
    data=Path(value('--data')).resolve()
    optional=Path(value('--optional-node-features-dir'))
    if not data.is_dir() or not optional.is_dir():
        raise FileNotFoundError(f'Missing data or optional features: {data}, {optional}')
    manifest=json.loads(Path(value('--split-manifest')).read_text())
    seen=set(); train_labels=[]
    for split in ('train','val','test'):
        paths=[Path(p).resolve() for p in manifest['splits'][split]['paths']]
        if not paths or seen.intersection(paths): raise ValueError('Empty or overlapping target split')
        seen.update(paths)
        count=0
        for p in paths:
            if p.parent!=data: raise ValueError(f'Split points outside data directory: {p}')
            with h5py.File(p,'r') as f:
                count+=len(f)
                if not len(f): raise ValueError(f'Empty target file: {p}')
                if require_interface and split in ('train','val'):
                    for name in f:
                        mask=f[name]['node_features']['interface_nodes'][()]
                        if not np.isfinite(mask).all() or not np.any(mask):
                            raise ValueError(f'Missing/invalid interface nodes: {p}:{name}')
                if split=='train':
                    train_labels.extend(float(f[k]['target_scores']['DockQ'][()]) for k in f)
        expected=manifest['splits'][split]['sample_count']
        if count!=expected: raise ValueError(f'{split}: {count} graphs versus expected {expected}')
        print(f'{split}: {len(paths)} targets, {count} graphs',flush=True)
    print('Training-only DockQ distribution:',json.dumps(fit_range_weights(train_labels)),flush=True)
    print('Input preflight passed: 1 process, 1 numerical-library thread.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--source',type=Path,required=True)
    p.add_argument("--require-interface", action="store_true")
    args=p.parse_args(); preflight(args.source, args.require_interface)
