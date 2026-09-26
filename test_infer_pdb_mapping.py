import h5py
import numpy as np
import pytest
from infer_pdb_mapping import build_map, write_map, match_group, load_candidates
from voronoi_edge_features.model_reference import build_target_model_references


def write_pdb(path, xyz):
    lines = [f"ATOM  {i:5d}  CA  ALA A{i:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C" for i, (x, y, z) in enumerate(xyz, 1)]
    path.write_text("\n".join(lines) + "\nEND\n")


def make_target(tmp_path, families=('sampled', 'random'), per=3, n=6):
    rng = np.random.default_rng(0)
    sampled = tmp_path / 'sampled_1abc'
    (sampled / 'random_negatives').mkdir(parents=True)
    contacts = np.array([[0, 3], [1, 4], [2, 5], [0, 5]])
    truth = {}
    with h5py.File(tmp_path / '1abc.hdf5', 'w') as f:
        for fam in families:
            for i in range(1, per + 1):
                xyz = rng.normal(scale=5, size=(n, 3))
                name = f'{fam}_1abc_model_{i}'
                g = f.create_group(name)
                g['node_features/interface_nodes'] = np.ones((n, 1))
                g['edge_features/contacts'] = contacts
                g['edge_features/ca_dist'] = np.linalg.norm(xyz[contacts[:, 0]] - xyz[contacts[:, 1]], axis=1)[:, None]
                fname = f'complex.{9 - i}_{i * 7}_corrected_H_0001.pdb'
                folder = sampled / 'random_negatives' if fam == 'random' else sampled
                write_pdb(folder / fname, xyz)
                truth[name] = f'sampled_1abc/{"random_negatives/" if fam == "random" else ""}{fname}'
    return tmp_path / '1abc.hdf5', sampled, truth


def test_geometry_recovers_renamed_files(tmp_path):
    graph, sampled, truth = make_target(tmp_path)
    rows, problems = build_map(graph, sampled)
    assert {r[0]: r[1] for r in rows} == truth and not problems['unmatched'] and not problems['ambiguous']
    assert {r[0]: r[2] for r in rows}['random_1abc_model_1'] == 'random_negative'


def test_reference_builder_uses_the_map(tmp_path):
    graph, sampled, truth = make_target(tmp_path)
    rows, _ = build_map(graph, sampled)
    write_map(rows, tmp_path / 'maps' / '1abc.csv')
    refs = build_target_model_references(graph, '1abc', name_map_dir=tmp_path / 'maps')
    assert {r.graph_group_name: r.relative_pdb_path for r in refs} == truth


def test_without_a_map_unknown_names_still_raise(tmp_path):
    graph, _, _ = make_target(tmp_path, families=('other',))
    with pytest.raises(ValueError, match='Unsupported'):
        build_target_model_references(graph, '1abc', name_map_dir=tmp_path / 'none')


def test_wrong_structure_is_not_matched(tmp_path):
    graph, sampled, _ = make_target(tmp_path, families=('sampled',))
    for f in sampled.glob('*.pdb'):
        write_pdb(f, np.random.default_rng(9).normal(scale=5, size=(6, 3)))
    rows, problems = build_map(graph, sampled)
    assert not rows and len(problems['unmatched']) + len(problems['ambiguous']) == 3
