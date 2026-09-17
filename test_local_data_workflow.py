import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from bundle_local_extension_data import build_bundle, file_sha256
from plan_extension_download import plan
from audit_local_graphs import inspect_graph


def inputs(tmp_path):
    dirs = {k: tmp_path / k for k in ('subset', 'feature_data', 'apbs_dir', 'esm_root', 'pdb_root', 'optional_dir')}
    for d in dirs.values(): d.mkdir()
    name = 'complex.0_0_0'
    for key in ('subset', 'feature_data'):
        with h5py.File(dirs[key] / '1abc.hdf5', 'w') as f:
            g = f.create_group(name)
            g['node_reference'] = np.array([[0, 1, 'A'], [1, 2, 'G']], dtype='S')
            g['node_features/aa_type'] = np.eye(20)[:2]
            g['node_features/chain'] = [[0], [1]]
            g['node_features/interface_nodes'] = [[1], [1]]
            g['edge_features/contacts'] = [[0, 1]]
            g['edge_features/ca_dist'] = [3.]
            g['edge_features/interface_edges'] = [[1]]
            g['target_scores/DockQ'] = 0.5
            if key == 'feature_data':
                g['edge_features/voronoi_contact_area'] = [[2.]]
                g['edge_features/voronoi_contact_missing'] = [[0]]
        with h5py.File(dirs[key] / 'empty.hdf5', 'w'): pass
    with h5py.File(dirs['apbs_dir'] / '1abc_apbs_surface.hdf5', 'w') as f:
        f.attrs['potential_units'] = 'kT/e'
        f.create_group(name)['residue_potential_mean'] = [1., 2.]
        f.create_group('unselected')
    (dirs['esm_root'] / '1abc_all.fasta').write_text('>1abc.A\nA\n>1abc.B\nG\n')
    for chain in 'AB': (dirs['esm_root'] / f'1abc.{chain}.pt').write_bytes(b'tensor placeholder')
    target = dirs['pdb_root'] / 'sampled_1abc'; target.mkdir()
    (target / f'{name}_corrected_H_0001.pdb').write_text('PDB placeholder\n')
    (dirs['optional_dir'] / '1abc_avg_rSASA_i.csv').write_text('Decoy,avg_rsasa_i\ncomplex.0_0_0,0.3\n')
    return SimpleNamespace(**dirs, mode='smoke', targets=['1abc', 'empty'], output=tmp_path / 'bundle')


def test_bundle_preserves_identity_features_and_provenance(tmp_path):
    args = inputs(tmp_path)
    before = file_sha256(args.subset / '1abc.hdf5')
    manifest = build_bundle(args)
    assert manifest['excluded_empty_targets'] == ['empty']
    assert file_sha256(args.subset / '1abc.hdf5') == before
    with h5py.File(args.output / 'subset_hdf5/1abc.hdf5') as f:
        result = inspect_graph(f['complex.0_0_0'])
        assert result['base_graph_ok'] and result['voronoi_ready'] and result['mask_ready']
    with h5py.File(args.output / 'apbs_model_data/1abc_apbs_surface.hdf5') as f:
        assert list(f) == ['complex.0_0_0']
        assert 'residue_statistics_area_weighted' not in f.attrs
    assert manifest['total_bytes'] == sum(manifest['files'].values())
    with pytest.raises(FileExistsError): build_bundle(args)


def test_bundle_missing_input_never_publishes_partial(tmp_path):
    args = inputs(tmp_path)
    (args.esm_root / '1abc.B.pt').unlink()
    with pytest.raises(FileNotFoundError): build_bundle(args)
    assert not args.output.exists()
    assert not list(tmp_path.glob('.bundle.*'))


def test_downloader_reuses_local_files_and_runs_batch(tmp_path):
    args = inputs(tmp_path); build_bundle(args)
    destination = tmp_path / 'download'
    fake = tmp_path / 'globus'
    fake.write_text('''#!/usr/bin/env python3
import json, pathlib, shlex, shutil, sys
args = sys.argv[1:]
if args[:2] == ['task', 'wait']:
    sys.exit(0)
assert args[0] == 'transfer'
if '--dry-run' in args:
    print('dry-run'); sys.exit(0)
if '--batch' in args:
    pairs = [shlex.split(line) for line in pathlib.Path(args[args.index('--batch')+1]).read_text().splitlines()]
else:
    pairs = [[args[1].split(':',1)[1], args[2].split(':',1)[1]]]
for src, dst in pairs:
    pathlib.Path(dst).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
print(json.dumps({'task_id': 'fake-task'}))
''')
    fake.chmod(0o755)
    env = {**os.environ, 'GLOBUS_BIN': str(fake), 'PYTHON_BIN': sys.executable,
           'SOURCE_BUNDLE': str(args.output), 'LOCAL_ROOT': str(destination),
           'REUSE_PDB_ROOT': str(args.pdb_root), 'REUSE_RSASA_DIR': str(args.optional_dir),
           'REUSE_ESM_ROOT': str(args.esm_root)}
    script = 'scripts/download_extension_data.sh'
    for _ in range(2):  # Re-running must not redownload already matched files.
        result = subprocess.run(['bash', script], env=env, text=True, capture_output=True)
        assert result.returncode == 0, result.stdout + result.stderr
    assert (destination / 'pdb/sampled_1abc/complex.0_0_0_corrected_H_0001.pdb').is_symlink()
    assert (destination / '.globus-transfer.txt').read_text() == ''
    manifest = json.loads((destination / 'bundle_manifest.json').read_text())
    for name, digest in manifest['file_sha256'].items():
        assert file_sha256(destination / name) == digest


def test_local_reuse_requires_content_match(tmp_path):
    args = inputs(tmp_path); build_bundle(args)
    local = tmp_path / 'local'; local.mkdir()
    (local / 'bundle_manifest.json').write_text((args.output / 'bundle_manifest.json').read_text())
    source = args.pdb_root / 'sampled_1abc/complex.0_0_0_corrected_H_0001.pdb'
    source.write_text('X' * source.stat().st_size)  # Same size is insufficient.
    config = SimpleNamespace(local_root=local, source_bundle=str(args.output), destination_path=str(local),
        reuse_pdb=args.pdb_root, reuse_rsasa=args.optional_dir, reuse_esm=args.esm_root)
    plan(config)
    assert 'complex.0_0_0_corrected_H_0001.pdb' in (local / '.globus-transfer.txt').read_text()
    assert not (local / 'pdb/sampled_1abc/complex.0_0_0_corrected_H_0001.pdb').exists()
