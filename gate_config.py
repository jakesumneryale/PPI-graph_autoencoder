"""TOML run configurations for train_gate.py.

A config file sets any train_gate.py option. Keys are the option names without the leading dashes (dashes or
underscores both work) and may be grouped in any [section] for readability; sections are ignored when loading.
Options that take a comma-separated list (node_features, edge_features, ...) also accept a TOML list.

Precedence: command-line flags > config file > train_gate.py built-in defaults.

    python train_gate.py --config configs/gate_rank.toml --seed 17 --output-dir runs/rank_seed17
    python train_gate.py --config configs/gate_rank.toml --print-config      # show the resolved settings and exit

Every run writes <output-dir>/run_config.toml with every resolved option (plus the git commit and command line),
so `python train_gate.py --config <output-dir>/run_config.toml` repeats it.
"""
from __future__ import annotations

import argparse
import datetime
import difflib
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path

# Display order and grouping of options in written configs. Options not listed go under [other].
SECTIONS: dict[str, tuple[str, ...]] = {
    'data': ('cluster', 'data', 'model_list_dir', 'split_manifest', 'target_name', 'test_fraction', 'val_fraction',
             'max_samples', 'strict_hdf5', 'optional_node_features_dir'),
    'features': ('node_feature_set', 'node_features', 'use_rsasa_i', 'use_drsasa', 'edge_features',
                 'edge_recon_features', 'edge_feature_transforms', 'standardize_edge_features', 'edge_stats_seed',
                 'edge_stats_max_graphs', 'use_esm', 'esm_sidecar_dir', 'esm_projection_dim', 'esm_scale',
                 'esm_dropout', 'esm_gate'),
    'model': ('architecture', 'hidden_dim', 'latent_dim', 'gat_heads', 'dropout', 'residual_connections', 'pooling',
              'dockq_head_mode'),
    'optimization': ('epochs', 'batch_size', 'lr', 'weight_decay', 'lr_schedule', 'warmup_steps', 'lr_final_fraction',
                     'lr_schedule_epochs', 'grad_clip', 'seed', 'targets_per_batch', 'decoys_per_target', 'ema_decay'),
    'loss': ('loss_weight_mode', 'target_weight', 'node_weight', 'edge_attr_weight', 'edge_presence_weight',
             'reconstruction_lambda', 'target_share', 'node_share', 'edge_attr_share', 'edge_presence_share',
             'loss_share_momentum', 'dockq_range_weighting', 'dockq_weight_exponent', 'dockq_weight_cap',
             'dockq_hinge_beta', 'dockq_hinge_start', 'rank_loss_weight', 'rank_temperature', 'capri_head_weight',
             'target_adversary_weight'),
    'augmentation': ('edge_dropout', 'node_feature_mask', 'ca_dist_noise'),
    'evaluation': ('checkpoint_metric', 'val_every_steps', 'validation_only_during_training', 'no_test_evaluation',
                   'dockq_range_diagnostics', 'eval_edge_threshold', 'eval_negative_ratio', 'eval_log_every',
                   'export_training_predictions', 'initialize_checkpoint', 'evaluation_only', 'prepare_splits_only'),
    'runtime': ('device', 'output_dir', 'num_workers', 'worker_start_method', 'persistent_workers', 'prefetch_factor',
                'cpu_threads'),
}
NOT_WRITTEN = {'help', 'config', 'print_config'}


def _actions(parser: argparse.ArgumentParser) -> dict[str, argparse.Action]:
    return {a.dest: a for a in parser._actions if a.dest not in NOT_WRITTEN and a.option_strings}


def load_config(path: str | Path, parser: argparse.ArgumentParser) -> dict[str, object]:
    """Read a TOML config and return {dest: value}, validated against the parser's options."""
    with open(path, 'rb') as fh:
        raw = tomllib.load(fh)
    flat: dict[str, object] = {}
    for key, value in raw.items():
        items = value.items() if isinstance(value, dict) else [(key, value)]   # [section] tables are flattened
        for k, v in items:
            dest = k.replace('-', '_')
            if dest in flat:
                raise SystemExit(f'{path}: option "{k}" is set twice')
            flat[dest] = v
    actions = _actions(parser)
    values: dict[str, object] = {}
    for dest, v in flat.items():
        action = actions.get(dest)
        if action is None:
            near = difflib.get_close_matches(dest, actions, n=3)
            raise SystemExit(f'{path}: unknown option "{dest}"' + (f' (did you mean {", ".join(near)}?)' if near else ''))
        if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
            if not isinstance(v, bool):
                raise SystemExit(f'{path}: "{dest}" must be true or false')
        elif isinstance(v, list):                                   # list -> the comma-separated form
            v = ','.join(str(x) for x in v)
        elif isinstance(v, bool):
            raise SystemExit(f'{path}: "{dest}" takes a value, not true/false')
        elif action.type is not None:
            try:
                v = action.type(v)
            except (TypeError, ValueError):
                raise SystemExit(f'{path}: "{dest}" = {v!r} is not a valid {getattr(action.type, "__name__", "value")}')
        if action.choices is not None and v not in action.choices:
            raise SystemExit(f'{path}: "{dest}" = {v!r} must be one of {", ".join(map(str, action.choices))}')
        values[dest] = v
    return values


def parse_args_with_config(parser: argparse.ArgumentParser, argv: list[str] | None = None) -> argparse.Namespace:
    """parse_args with --config support: config values become the parser defaults, so CLI flags still win."""
    parser.add_argument('--config', default=None, help='TOML file of option values (see gate_config.py / README).')
    parser.add_argument('--print-config', action='store_true',
                        help='Print the fully resolved configuration as TOML and exit.')
    pre, _ = parser.parse_known_args(argv)
    if pre.config:
        parser.set_defaults(**load_config(pre.config, parser))
    args = parser.parse_args(argv)
    if args.print_config:
        sys.stdout.write(render_config(args, parser))
        raise SystemExit(0)
    return args


def _toml_value(v: object) -> str:
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, (int, float)):
        return repr(v)
    s = str(v).replace('\\', '\\\\').replace('"', '\\"')
    return f'"{s}"'


def _git_commit() -> str:
    try:
        here = Path(__file__).resolve().parent
        commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=here, capture_output=True, text=True,
                                timeout=10).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=here,
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return (commit + (' (uncommitted changes)' if dirty else '')) if commit else 'unknown'
    except (OSError, subprocess.SubprocessError):
        return 'unknown'


def render_config(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str:
    """Every option of the run as a TOML config. Unset options (None) are written as comments."""
    actions = _actions(parser)
    placed = [d for keys in SECTIONS.values() for d in keys]
    sections = dict(SECTIONS)
    others = tuple(d for d in actions if d not in placed)
    if others:
        sections['other'] = others
    lines = ['# Resolved train_gate.py configuration. Re-run with: python train_gate.py --config <this file>',
             f'# written {datetime.datetime.now().isoformat(timespec="seconds")}, git commit {_git_commit()}',
             f'# command: {shlex.join([Path(sys.argv[0]).name] + sys.argv[1:])}']
    for section, keys in sections.items():
        keys = [k for k in keys if k in actions and hasattr(args, k)]
        if not keys:
            continue
        lines.append(f'\n[{section}]')
        for k in keys:
            v = getattr(args, k)
            lines.append(f'# {k} = (unset)' if v is None else f'{k} = {_toml_value(v)}')
    return '\n'.join(lines) + '\n'


def write_run_config(args: argparse.Namespace, parser: argparse.ArgumentParser, output_dir: str | Path) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'run_config.toml'
    path.write_text(render_config(args, parser))
    return path
