"""Scientific invariants and end-to-end extension integration checks."""
import json
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pytest
import torch
from torch_geometric.data import Data, Batch
from EGNN_model import build_graph_model
from prepare_model_extensions import apbs_features, load_embeddings
from protein_hdf5_dataset import ProteinGraphHDF5Dataset, compute_edge_feature_stats


def graph():
    return Data(x=torch.randn(4, 3), pos=torch.randn(4, 3),
        edge_index=torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]),
        edge_attr=torch.randn(6, 2), interface_mask=torch.tensor([False, True, True, False]),
        esm=torch.randn(4, 8))


@pytest.mark.parametrize('architecture', ['gat', 'egnn'])
def test_pooling_and_esm(architecture):
    torch.manual_seed(3)
    model = build_graph_model(architecture, in_node_feats=3, in_edge_feats=2,
        hidden_dim=8, latent_dim=4, gat_heads=2, esm_dim=8, esm_projection_dim=4,
        pooling='interface', dropout=0).eval()
    data = Batch.from_data_list([graph(), graph()])
    z = torch.arange(32).reshape(8, 4).float()
    pooled = model.pool_nodes(z, data.batch, data)
    torch.testing.assert_close(pooled[0, :4], z[1:3].mean(0))
    torch.testing.assert_close(pooled[0, 4:], z[1:3].max(0).values)
    result = model(data)
    assert result['node_recon'].shape == (8, 3)  # ESM is input-only
    result['quality_pred'].sum().backward()
    assert model.esm_projector[1].weight.grad.abs().sum() > 0
    data.interface_mask[:4] = False
    with pytest.raises(ValueError, match='no interface'):
        model(data)


def test_egnn_equivariance_and_permutation():
    torch.manual_seed(4)
    data = graph()
    model = build_graph_model('egnn', in_node_feats=3, in_edge_feats=2,
        hidden_dim=8, latent_dim=4, dropout=0).eval()
    z, pos = model.encode_geometry(data)
    rotation, _ = torch.linalg.qr(torch.randn(3, 3))
    rotation[:, 0] *= -1  # Reflections are covered too.
    translation = torch.tensor([3., -8., 4.])
    moved = data.clone(); moved.pos = data.pos @ rotation + translation
    moved_z, moved_pos = model.encode_geometry(moved)
    torch.testing.assert_close(z, moved_z, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(pos @ rotation + translation, moved_pos, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(model(data)['quality_pred'], model(moved)['quality_pred'])
    permutation = torch.tensor([2, 0, 3, 1]); inverse = torch.argsort(permutation)
    permuted = data.clone()
    for name in ('x', 'pos', 'interface_mask', 'esm'):
        setattr(permuted, name, getattr(data, name)[permutation])
    permuted.edge_index = inverse[data.edge_index]
    perm_z, perm_pos = model.encode_geometry(permuted)
    torch.testing.assert_close(z[permutation], perm_z)
    torch.testing.assert_close(pos[permutation], perm_pos)


def test_real_esm_examples():
    from Bio import SeqIO
    root = Path(__file__).parent / 'ESM2_embedding_ex'
    records = list(SeqIO.parse(root / '1acb_all.fasta', 'fasta'))
    residues = [(record.id.split('.')[-1], aa) for record in records for aa in str(record.seq)]
    embedding, provenance = load_embeddings(root, '1acb', 'unused', residues,
        '{target}_all.fasta', '{target}.{chain}.pt', 33)
    assert embedding.shape == (304, 1280)
    expected = torch.load(root / '1acb.B.pt', weights_only=True)['representations'][33]
    np.testing.assert_array_equal(embedding[241:], expected.numpy())
    with pytest.raises(ValueError, match='sequence alignment'):
        load_embeddings(root, '1acb', 'unused', [('A', 'X')] + residues[1:],
            '{target}_all.fasta', '{target}.{chain}.pt', 33)


def write_fixture(path):
    with h5py.File(path, 'w') as f:
        for i in range(2):
            g = f.create_group(f'complex.{i}_0_0')
            n = g.create_group('node_features')
            n['aa_type'] = np.eye(20, dtype=np.float32)[:4]
            n['chain'] = [[0], [0], [1], [1]]
            n['interface_nodes'] = [[0], [1], [1], [0]]
            n['interface_node_degree'] = [[1], [1], [1], [1]]
            e = g.create_group('edge_features')
            e['contacts'] = [[0, 1], [1, 2], [2, 3]]
            e['interface_edges'] = [[0], [1], [0]]
            e['ca_dist'] = [[3], [4], [5]]
            e['voronoi_contact_area'] = [[2], [3], [4]]
            e['voronoi_contact_missing'] = np.zeros((3, 1))
            for name in ('mean', 'absdiff', 'product', 'area_product'):
                e[f'apbs_pair_{name}'] = [[0], [2], [3]]
            e['apbs_pair_missing'] = [[1], [0], [0]]
            g['esm2'] = np.ones((4, 8), dtype=np.float32)
            g['pos'] = np.arange(12, dtype=np.float32).reshape(4, 3)
            g.create_group('target_scores')['DockQ'] = i * 0.5


def test_apbs_pair_and_missing_scaling(tmp_path):
    path = tmp_path / 'demo.hdf5'; write_fixture(path)
    with h5py.File(path, 'a') as f:
        g = f['complex.0_0_0']
        a = f.create_group('apbs')
        a['residue_aa_id'] = np.arange(4)
        a['residue_chain'] = np.array(['A', 'A', 'B', 'B'], dtype='S')
        a['residue_name'] = np.array(['ALA'] * 4, dtype='S')
        a['residue_potential_mean'] = [1., -2., 3., np.nan]
        a['residue_surface_point_count'] = [1, 2, 3, 0]
        a['residue_in_pqr'] = [1, 1, 1, 1]
        features = apbs_features(a, g, [('A', 'A'), ('A', 'A'), ('B', 'A'), ('B', 'A')])
        np.testing.assert_array_equal(features['apbs_pair_product'].ravel(), [-2, -6, 0])
        np.testing.assert_array_equal(features['apbs_pair_area_product'].ravel(), [-4, -18, 0])
        np.testing.assert_array_equal(features['apbs_pair_missing'].ravel(), [0, 0, 1])
        del f['apbs']
    stats = compute_edge_feature_stats(path, ['apbs_pair_product'], {}, max_graphs=2)
    assert stats['apbs_pair_product'] == (2.5, 0.5)
    dataset = ProteinGraphHDF5Dataset(path, edge_features=['apbs_pair_product', 'apbs_pair_missing'],
        edge_feature_stats=stats)
    assert dataset[0].edge_attr[0, 0] == 0  # Missing stays neutral after scaling


def test_matrix_training_reload_and_comparison(tmp_path):
    data = tmp_path / 'data'; data.mkdir()
    optional = tmp_path / 'optional'; optional.mkdir()
    for target in ('1abc', '2abc', '3abc'):
        write_fixture(data / f'{target}.hdf5')
        names = ['complex.0_0_0', 'complex.1_0_0']
        (data / f'{target}.audit.json').write_text(json.dumps({'accepted': names, 'rejected': {}}))
        (optional / f'{target}_avg_rSASA_i.csv').write_text('Decoy,avg_rsasa_i\n' + '\n'.join(f'{n},0.3' for n in names))
    output = tmp_path / 'runs'
    def run(*argv):
        completed = subprocess.run([sys.executable, *map(str, argv)], capture_output=True, text=True)
        assert completed.returncode == 0, completed.stdout + completed.stderr
    run('model_extension_experiments.py', 'prepare', '--data', data, '--output', output,
        '--optional-node-features-dir', optional, '--epochs', '1', '--num-workers', '0',
        '--device', 'cpu', '--seeds', '7')
    matrix_path = output / 'matrix.json'
    matrix = json.loads(matrix_path.read_text())
    assert len(matrix['runs']) == 19
    for item in matrix['runs']:
        command = item['argv']
        assert command[command.index('--loss-weight-mode') + 1] == 'fixed'
        assert not any(flag.endswith('-share') for flag in command)
        expected = ['1', '1', '0.1', '1'] if item['config']['objective'] == 'ae' else ['0', '0', '0', '1']
        assert [command[command.index('--' + term + '-weight') + 1] for term in ('node', 'edge-attr', 'edge-presence', 'target')] == expected
        if item['config']['name'] == 'seed_replicate_base':
            assert command[command.index('--node-features') + 1] == 'aa_type,chain,interface_nodes,rsasa_i'
            assert command[command.index('--edge-features') + 1] == 'interface_edges,ca_dist'
            assert '--edge-feature-transforms' not in command
    # Exercise both encoders with ESM + APBS + interface pooling, and FF objective.
    wanted = {'seed_replicate_base', 'baseline', 'gat_plain_interface_esm1', 'egnn_ae_combined', 'egnn_supervised_combined'}
    for i, item in enumerate(matrix['runs']):
        if item['config']['name'] in wanted:
            run('model_extension_experiments.py', 'run', '--matrix', matrix_path, '--task', i + 1)
            assert (Path(item['output']) / 'test_predictions.csv').is_file()
    matrix['runs'] = [item for item in matrix['runs'] if item['config']['name'] in wanted]
    matrix_path.write_text(json.dumps(matrix))
    run('compare_model_extensions.py', '--matrix', matrix_path, '--output', output / 'comparison', '--bootstrap', '100')
    assert (output / 'comparison/comparison.csv').is_file()
    run('compare_model_extensions.py', '--matrix', matrix_path, '--reference', 'seed_replicate_base', '--output', output / 'base_comparison', '--bootstrap', '100')
    full = tmp_path / 'full_runs'
    # Prior manifests may include known empty targets; preserve every surviving
    # target's assignment rather than reshuffling after excluding those targets.
    prior = json.loads((output / 'target_splits.json').read_text())
    prior['splits']['train']['paths'].append(str(data / 'empty.hdf5'))
    prior_path = tmp_path / 'prior_with_empty.json'
    prior_path.write_text(json.dumps(prior))
    (tmp_path / 'targets.txt').write_text('1abc\n2abc\n3abc\n')
    (tmp_path / 'excluded_targets.json').write_text(json.dumps({
        'empty': {'reason': 'empty_source_subset', 'source_entry_count': 0}}))
    run('model_extension_experiments.py', 'prepare', '--data', data, '--output', full,
        '--optional-node-features-dir', optional, '--prior-split', prior_path,
        '--targets-file', tmp_path / 'targets.txt')
    full_matrix = json.loads((full / 'matrix.json').read_text())
    assert len(full_matrix['runs']) == 57
    assert {item['seed'] for item in full_matrix['runs']} == {7, 17, 27}
    assert json.loads((full / 'target_splits.json').read_text())['splits'] == json.loads((output / 'target_splits.json').read_text())['splits']


def test_preparer_preserves_sources_and_aligns_rows(tmp_path):
    from types import SimpleNamespace
    from prepare_model_extensions import prepare_target, sha256
    source = tmp_path / 'source'; source.mkdir()
    feature_data = tmp_path / 'latest'; feature_data.mkdir()
    pdb_root = tmp_path / 'pdb'; (pdb_root / 'sampled_1abc').mkdir(parents=True)
    esm_root = tmp_path / 'esm'; esm_root.mkdir()
    apbs_dir = tmp_path / 'apbs'; apbs_dir.mkdir()
    target = '1abc'; name = 'complex.0_0_0'
    source_path = source / f'{target}.hdf5'; write_fixture(source_path)
    with h5py.File(source_path, 'a') as f:
        del f['complex.1_0_0']
        f[name]['node_reference'] = np.array([[0, 1, 'A'], [1, 1, 'A'], [2, 2, 'A'], [3, 2, 'A']], dtype='S')
    pdb = pdb_root / 'sampled_1abc' / f'{name}_corrected_H_0001.pdb'
    lines = []
    for i, chain in enumerate(['A', 'A', 'B', 'B']):
        lines.append(f'ATOM  {i+1:5d}  CA  ALA {chain}{i+1:4d}    {float(i):8.3f}{0.:8.3f}{0.:8.3f}  1.00 20.00           C  \n')
    pdb.write_text(''.join(lines) + 'END\n')
    (esm_root / '1abc_all.fasta').write_text('>1abc.A\nAA\n>1abc.B\nAA\n')
    for chain in ('A', 'B'):
        torch.save({'label': f'1abc.{chain}', 'representations': {33: torch.arange(16).reshape(2, 8).float()}}, esm_root / f'1abc.{chain}.pt')
    with h5py.File(apbs_dir / '1abc_apbs_surface.hdf5', 'w') as f:
        a = f.create_group(name)
        a.attrs['residue_statistics_area_weighted'] = True
        a.attrs['potential_units'] = 'kT/e'
        a['residue_aa_id'] = np.arange(4)
        a['residue_chain'] = np.array(['A', 'A', 'B', 'B'], dtype='S')
        a['residue_name'] = np.array(['ALA'] * 4, dtype='S')
        a['residue_potential_mean'] = [1., -2., 3., np.nan]
        a['residue_surface_point_count'] = [1, 2, 3, 0]
        a['residue_in_pqr'] = [1, 1, 1, 1]
    import shutil
    shutil.copyfile(source_path, feature_data / '1abc.hdf5')
    # Existing subset lacks the newer mask; hydrate from the full feature graphs.
    with h5py.File(source_path, 'a') as f:
        del f[name]['edge_features/voronoi_contact_missing']
    before = sha256(source_path)
    args = SimpleNamespace(data=source, feature_data=feature_data, output=tmp_path / 'prepared',
        apbs_dir=apbs_dir, pdb_root=pdb_root, esm_root=esm_root, esm_layer=33,
        fasta_template='{target}_all.fasta', embedding_template='{target}.{chain}.pt', assume_area_weighted=False)
    prepare_target(args, target)
    assert sha256(source_path) == before
    with h5py.File(args.output / '1abc.hdf5') as f:
        g = f[name]
        np.testing.assert_array_equal(g['pos'][:, 0], [0, 1, 2, 3])
        np.testing.assert_array_equal(g['node_features/interface_node_degree'][:].ravel(), [1, 1, 1, 1])
        assert g['esm2'].shape == (4, 8)
    # Bad ordering produces an audit and never replaces the last complete output.
    prepared_digest = sha256(args.output / '1abc.hdf5')
    with h5py.File(apbs_dir / '1abc_apbs_surface.hdf5', 'a') as f:
        f[name]['residue_chain'][0] = b'B'
    with pytest.raises(ValueError, match='no eligible'):
        prepare_target(args, target)
    assert sha256(args.output / '1abc.hdf5') == prepared_digest


def test_egnn_coincident_coordinates_have_finite_gradients():
    data = graph(); data.pos.zero_()
    model = build_graph_model('egnn', in_node_feats=3, in_edge_feats=2,
        hidden_dim=8, latent_dim=4, dropout=0, residual_connections=True)
    model(data)['quality_pred'].sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@pytest.mark.parametrize('cpus,requested,expected', [(8, 7, 7), (4, 7, 3), (1, 7, 0), (8, 0, 0)])
def test_training_workers_fit_cpu_allocation(cpus, requested, expected):
    from model_extension_experiments import training_command
    original = {'argv': ['--num-workers', str(requested)]}
    command = training_command(original, cpus)
    assert int(command[command.index('--num-workers') + 1]) == expected
    assert original['argv'][1] == str(requested)  # Frozen matrix is unchanged.


def test_training_cpu_allocation_must_be_positive():
    from model_extension_experiments import training_command
    with pytest.raises(ValueError, match='positive'):
        training_command({'argv': ['--num-workers', '7']}, 0)


def test_empty_subset_fails_before_opening_external_inputs(tmp_path):
    from types import SimpleNamespace
    from prepare_model_extensions import prepare_target
    source = tmp_path / 'source'; source.mkdir()
    with h5py.File(source / '3gfk.hdf5', 'w') as f:
        f.attrs['voronoi_subset_model_count'] = 0
    output = tmp_path / 'prepared'; output.mkdir()
    existing = output / '3gfk.hdf5'
    existing.write_bytes(b'previous output must be preserved')
    args = SimpleNamespace(data=source, output=output, apbs_dir=tmp_path / 'missing_apbs')
    with pytest.raises(ValueError, match='zero graph entries'):
        prepare_target(args, '3gfk')
    audit = json.loads((output / '3gfk.audit.json').read_text())
    assert audit['input_error'] == 'empty_source_subset'
    assert audit['source_entry_count'] == audit['source_subset_model_count'] == 0
    assert existing.read_bytes() == b'previous output must be preserved'
    assert not list(output.glob('.3gfk.*'))



def test_preflight_records_only_empty_target_exclusions(tmp_path):
    from preflight_model_extensions import prepare_target_list
    data = tmp_path / 'data'; data.mkdir()
    for target in ('a', 'b', 'c', 'empty'):
        with h5py.File(data / f'{target}.hdf5', 'w') as f:
            if target != 'empty':
                f.create_group('model')
    included, excluded = prepare_target_list(data, tmp_path / 'output')
    assert included == ['a', 'b', 'c']
    assert list(excluded) == ['empty']
    assert (tmp_path / 'output/targets.txt').read_text() == 'a\nb\nc\n'
    with h5py.File(data / 'empty.hdf5') as f:
        assert len(f) == 0
    # A corrupt file is an error, not another permissible target exclusion.
    (data / 'broken.hdf5').write_text('not hdf5')
    with pytest.raises(OSError):
        prepare_target_list(data, tmp_path / 'bad_output')
