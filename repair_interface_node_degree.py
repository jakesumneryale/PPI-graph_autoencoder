"""Repair the missing interface_node_degree feature, one target per array task.

This is the only step in the pipeline that writes to the source graphs. It adds or
replaces exactly one node dataset, interface_node_degree, recomputed from the stored
contacts and interface mask by the same function the audit uses to validate it. No other
dataset, attribute or group is created, deleted or modified.

Models whose node_features group is absent entirely cannot be repaired here and are
recorded as unrepairable rather than treated as failures; they stay out of the cohort.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import tempfile
import h5py
import numpy as np
from add_interface_node_degree import (FEATURE_NAME, TEMPORARY_NAME, calculate_interface_node_degree,
                                       dataset_is_valid, normalize_contacts)

REQUIRED = ('node_features', 'edge_features')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name)
    try:
        with os.fdopen(fd, 'w') as f: json.dump(value, f, indent=2, allow_nan=False); f.write('\n')
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def signature(path):
    stat = path.stat()
    return dict(path=str(path.resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def inspect(model):
    """Return (num_nodes, contacts) or raise, without writing anything."""
    for name in REQUIRED:
        if name not in model: raise KeyError(f"missing {name}")
    node_features = model['node_features']; edge_features = model['edge_features']
    if 'interface_nodes' not in node_features: raise KeyError('missing interface_nodes')
    if 'contacts' not in edge_features: raise KeyError('missing contacts')
    interface = np.asarray(node_features['interface_nodes'][()])
    contacts = np.asarray(edge_features['contacts'][()])
    n = len(interface.reshape(len(interface), -1))
    normalize_contacts(contacts, n)
    return interface, contacts, n


def repair(path, output, overwrite=False, dry_run=False):
    before = signature(path)
    repaired = []; already_valid = 0; unrepairable = {}; failed = {}
    inventory = {}
    with h5py.File(path, 'r' if dry_run else 'r+') as handle:
        names = sorted(handle.keys())
        for name in names:
            try:
                model = handle[name]
                interface, contacts, n = inspect(model)
                node_features = model['node_features']
                # Record every other node dataset so the write can be shown to be scoped.
                inventory[name] = sorted(k for k in node_features.keys() if k != FEATURE_NAME)
                if TEMPORARY_NAME in node_features and not dry_run:
                    del node_features[TEMPORARY_NAME]
                values = calculate_interface_node_degree(interface, contacts)
                # "Valid" must mean agreeing with the contacts, not merely well shaped: the
                # audit rejects a degree array that disagrees with its own graph, so a stale
                # but correctly shaped dataset has to be rewritten like a missing one.
                if FEATURE_NAME in node_features and not overwrite:
                    existing = node_features[FEATURE_NAME]
                    if dataset_is_valid(existing, n) and np.array_equal(
                            np.asarray(existing[()]).reshape(-1), values.reshape(-1)):
                        already_valid += 1
                        continue
                if dry_run:
                    repaired.append(name); continue
                if FEATURE_NAME in node_features: del node_features[FEATURE_NAME]
                temporary = node_features.create_dataset(TEMPORARY_NAME, data=values, dtype=np.int32)
                temporary.attrs['definition'] = 'number of unique adjacent nodes whose interface_nodes value is nonzero'
                temporary.attrs['contacts_treated_as_undirected'] = True
                handle.flush()
                if not dataset_is_valid(temporary, n): raise RuntimeError('new dataset failed validation')
                node_features.move(TEMPORARY_NAME, FEATURE_NAME)
                handle.flush()
                repaired.append(name)
            except KeyError as exc:
                unrepairable[name] = str(exc)
            except Exception as exc:  # noqa: BLE001 - one malformed graph must not abort the target
                failed[name] = str(exc)
        # Verification pass: recompute independently and confirm nothing else moved.
        mismatched = []; disturbed = []
        for name in repaired if not dry_run else []:
            model = handle[name]
            interface, contacts, n = inspect(model)
            expected = calculate_interface_node_degree(interface, contacts).reshape(-1)
            stored = np.asarray(model['node_features'][FEATURE_NAME][()]).reshape(-1)
            if not np.array_equal(expected, stored): mismatched.append(name)
            current = sorted(k for k in model['node_features'].keys() if k != FEATURE_NAME)
            if current != inventory[name]: disturbed.append(name)
        raw = len(names)
    report = dict(target=path.stem, raw_models=raw, repaired=len(repaired), already_valid=already_valid,
                  unrepairable=len(unrepairable), failed=len(failed),
                  unrepairable_reasons=dict(Counter(unrepairable.values())),
                  failures={k: v for k, v in list(failed.items())[:20]},
                  verified_mismatch=len(mismatched), other_node_datasets_changed=len(disturbed),
                  dry_run=dry_run, signature_before=before,
                  signature_after=None if dry_run else signature(path))
    write_json(output / f'{path.stem}.json', report)
    print(f'{path.stem}: {len(repaired)} repaired, {already_valid} already valid, '
          f'{len(unrepairable)} unrepairable, {len(failed)} failed', flush=True)
    if mismatched: raise SystemExit(f'{path.stem}: {len(mismatched)} models disagree with recomputation')
    if disturbed: raise SystemExit(f'{path.stem}: {len(disturbed)} models had other node datasets change')
    if failed: raise SystemExit(f'{path.stem}: {len(failed)} models failed to repair; see {output}/{path.stem}.json')
    return report


def summarize(reports_dir, expected_targets=None):
    reports = [json.loads(p.read_text()) for p in sorted(Path(reports_dir).glob('*.json'))]
    if not reports: raise SystemExit(f'No repair reports in {reports_dir}')
    raw = sum(r['raw_models'] for r in reports)
    repaired = sum(r['repaired'] for r in reports)
    valid = sum(r['already_valid'] for r in reports)
    unrepairable = sum(r['unrepairable'] for r in reports)
    reasons = Counter()
    for r in reports: reasons.update(r['unrepairable_reasons'])
    print(f'{len(reports)} targets; {raw:,} models: {repaired:,} repaired, {valid:,} already valid, '
          f'{unrepairable:,} unrepairable.')
    for reason, n in reasons.most_common(): print(f'  unrepairable: {n:,} x {reason}')
    print(f'Expected eligible after re-audit: {repaired + valid:,} (subject to the other audit checks).')
    if expected_targets is not None and len(reports) != expected_targets:
        raise SystemExit(f'Expected {expected_targets} targets, found {len(reports)}')
    if any(r['verified_mismatch'] or r['other_node_datasets_changed'] for r in reports):
        raise SystemExit('Verification failed in at least one target')
    return reports


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run', help='Repair one target')
    r.add_argument('--paths', type=Path, required=True, help='paths.json inventory')
    r.add_argument('--task', type=int, required=True, help='One-based array index')
    r.add_argument('--output', type=Path, required=True, help='Directory for per-target reports')
    r.add_argument('--overwrite', action='store_true', help='Recompute even valid datasets')
    r.add_argument('--dry-run', action='store_true', help='Open read-only and report what would change')
    s = sub.add_parser('summarize', help='Aggregate the per-target reports')
    s.add_argument('--reports', type=Path, required=True)
    s.add_argument('--expected-targets', type=int, default=None)
    a = p.parse_args()
    if a.command == 'run':
        paths = [Path(x) for x in json.loads(a.paths.read_text())]
        if not 1 <= a.task <= len(paths): raise SystemExit('Task index outside inventory')
        repair(paths[a.task - 1], a.output, overwrite=a.overwrite, dry_run=a.dry_run)
    else:
        summarize(a.reports, a.expected_targets)
