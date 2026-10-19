"""Tests for gate_config (TOML run configurations for train_gate.py)."""
import argparse
import tomllib

import pytest

from gate_config import load_config, parse_args_with_config, render_config


def make_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--pooling', choices=('all', 'interface', 'combined'), default='all')
    p.add_argument('--residual-connections', '--residual_connections', action='store_true')
    p.add_argument('--node-features', default=None)
    p.add_argument('--output-dir', default='checkpoints')
    p.add_argument('--max-samples', type=int, default=None)
    return p


def write(tmp_path, text):
    path = tmp_path / 'cfg.toml'
    path.write_text(text)
    return str(path)


def test_config_sets_values_and_sections_are_flattened(tmp_path):
    cfg = write(tmp_path, '[model]\ndropout = 0.3\npooling = "combined"\nresidual_connections = true\n'
                          '[optimization]\nepochs = 20\nlr = 3e-4\n')
    args = parse_args_with_config(make_parser(), ['--config', cfg])
    assert (args.dropout, args.pooling, args.residual_connections, args.epochs, args.lr) == (0.3, 'combined', True, 20, 3e-4)


def test_cli_overrides_config(tmp_path):
    cfg = write(tmp_path, 'epochs = 20\ndropout = 0.3\n')
    args = parse_args_with_config(make_parser(), ['--config', cfg, '--epochs', '5'])
    assert args.epochs == 5 and args.dropout == 0.3


def test_dashes_lists_and_int_to_float(tmp_path):
    cfg = write(tmp_path, 'node-features = ["aa_type", "chain"]\nlr = 1\n')
    args = parse_args_with_config(make_parser(), ['--config', cfg])
    assert args.node_features == 'aa_type,chain' and args.lr == 1.0 and isinstance(args.lr, float)


@pytest.mark.parametrize('text, message', [
    ('dropuot = 0.3\n', 'did you mean dropout'),
    ('pooling = "mean"\n', 'must be one of'),
    ('residual_connections = 1\n', 'true or false'),
    ('epochs = "many"\n', 'not a valid int'),
    ('epochs = true\n', 'not true/false'),
    ('[a]\nepochs = 1\n[b]\nepochs = 2\n', 'set twice'),
])
def test_bad_configs_fail_clearly(tmp_path, text, message):
    with pytest.raises(SystemExit) as err:
        load_config(write(tmp_path, text), make_parser())
    assert message in str(err.value)


def test_rendered_config_round_trips(tmp_path):
    parser = make_parser()
    args = parse_args_with_config(parser, ['--epochs', '7', '--pooling', 'combined', '--residual-connections'])
    text = render_config(args, parser)
    tomllib.loads(text)                                   # valid TOML
    assert '# max_samples = (unset)' in text and 'config' not in tomllib.loads(text).get('other', {})
    path = write(tmp_path, text)
    again = parse_args_with_config(make_parser(), ['--config', path])
    assert {k: v for k, v in vars(again).items() if k not in ('config', 'print_config')} == \
           {k: v for k, v in vars(args).items() if k not in ('config', 'print_config')}
