import pytest
from voronoi_edge_features.common import infer_relative_pdb_path


def test_positive_complex_pattern_unchanged():
    path, location = infer_relative_pdb_path('1gla', 'complex.0_0_15')
    assert path == 'sampled_1gla/complex.0_0_15_corrected_H_0001.pdb'
    assert location == 'sampled'


def test_negative_complex_pattern_unchanged():
    path, location = infer_relative_pdb_path('1gla', 'complex.12345_6')
    assert path == 'sampled_1gla/random_negatives/complex.12345_6_corrected_H_0001.pdb'
    assert location == 'random_negative'


def test_uniformly_sampled_batch_resolves_to_relaxed_pdb():
    path, location = infer_relative_pdb_path('1c3a', 'random_1c3a_model_1')
    assert path == 'sampled_1c3a/relaxed_1c3a_model_1.pdb'
    assert location == 'uniformly_sampled'


def test_uniformly_sampled_batch_target_mismatch_rejected():
    with pytest.raises(ValueError, match='Unsupported graph/model naming scheme'):
        infer_relative_pdb_path('1c3a', 'random_1kfu_model_1')


def test_unknown_naming_scheme_still_raises():
    with pytest.raises(ValueError, match='Unsupported graph/model naming scheme'):
        infer_relative_pdb_path('1gla', 'totally_unrecognized_name')
