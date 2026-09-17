"""Stage a portable smoke-test or complete 10% input bundle on the cluster.

Serial I/O: request one CPU. No APBS/Voronoi/ESM computation is performed.
Source stores remain read-only; a completed bundle is published atomically.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import h5py
import numpy as np
from voronoi_edge_features.common import infer_relative_pdb_path, resolve_target_graph_hdf5


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def copy_file(source, destination):
    if not source.is_file():
        raise FileNotFoundError(f'Required bundle input is missing: {source}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def preflight_apbs(args, targets, audit_path):
    """Check every selected APBS key before copying any large input files."""
    report = {'missing_apbs_policy': getattr(args, 'missing_apbs', 'error'),
              'targets': {}, 'excluded_empty_targets': [], 'excluded_apbs_targets': [],
              'missing_graphs': [], 'errors': []}
    selected = {}
    for target in targets:
        source = resolve_target_graph_hdf5(args.subset, target)
        with h5py.File(source, 'r') as subset:
            names = sorted(subset)
        if args.mode == 'smoke':
            names = names[:3]
        if not names:
            report['excluded_empty_targets'].append(target)
            continue
        path = args.apbs_dir / f'{target}_apbs_surface.hdf5'
        matches = {}
        missing = []
        # A missing store is recorded explicitly; a corrupt/unreadable store
        # is a hard error, not grounds for quietly excluding its graphs.
        try:
            if path.exists():
                with h5py.File(path, 'r') as apbs:
                    for name in names:
                        base = name.removesuffix('_corrected')
                        candidates = [base, base + '_corrected']
                        key = next((k for k in candidates if isinstance(apbs.get(k), h5py.Group)), None)
                        if key is None:
                            missing.append(name)
                        else:
                            matches[name] = key
            else:
                missing = names
        except (OSError, RuntimeError) as exc:
            report['errors'].append({'target': target, 'path': str(path), 'error': str(exc)})
        for name in missing:
            relative, kind = infer_relative_pdb_path(target, name)
            base = name.removesuffix('_corrected')
            report['missing_graphs'].append({'target': target, 'graph': name,
                'apbs_path': str(path), 'tried_keys': [base, base + '_corrected'],
                'pdb_path': str(args.pdb_root / relative), 'model_kind': kind,
                'reason': 'missing_record' if path.exists() else 'missing_store'})
        report['targets'][target] = {'selected': len(names), 'matched': len(matches), 'missing': len(missing)}
        if matches:
            selected[target] = matches
        elif missing:
            report['excluded_apbs_targets'].append(target)
    report['selected_graphs'] = sum(v['selected'] for v in report['targets'].values())
    report['matched_graphs'] = sum(len(v) for v in selected.values())
    audit_path.write_text(json.dumps(report, indent=2) + '\n')
    print(f'APBS preflight: {report["matched_graphs"]}/{report["selected_graphs"]} graphs matched; '
          f'{len(report["missing_graphs"])} missing. Audit: {audit_path}', flush=True)
    if report['errors']:
        raise ValueError(f'Unreadable APBS stores; see {audit_path}')
    if report['missing_graphs'] and report['missing_apbs_policy'] == 'error':
        raise ValueError(f'Missing APBS inputs; see {audit_path}. Repair these records, or explicitly '
                         'use --missing-apbs exclude to omit these graphs from every model configuration.')
    if not selected:
        raise ValueError(f'No graphs with matching APBS records; see {audit_path}')
    return selected, report


def build_bundle(args):
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f'{output} exists. Reuse it or choose a new bundle name.')
    output.parent.mkdir(parents=True, exist_ok=True)
    targets = args.targets or (['1acb', '1avx', '1ay7'] if args.mode == 'smoke' else
        sorted({p.stem for pattern in ('*.h5', '*.hdf5') for p in args.subset.glob(pattern)}))
    if not targets or len(set(targets)) != len(targets):
        raise ValueError('Targets must be nonempty and unique')
    selected, coverage = preflight_apbs(args, targets, output.with_name(output.name + '.preflight.json'))
    if getattr(args, 'preflight_only', False):
        return coverage
    stage = Path(tempfile.mkdtemp(prefix=f'.{output.name}.', dir=output.parent))
    manifest = {'schema': 1, 'mode': args.mode, 'targets': {},
                'excluded_empty_targets': coverage['excluded_empty_targets'], 'apbs_coverage': coverage,
                'source_paths': {k: str(getattr(args, k)) for k in
                                 ('subset', 'feature_data', 'apbs_dir', 'esm_root', 'pdb_root', 'optional_dir')}}
    try:
        for target, apbs_matches in selected.items():
            source = resolve_target_graph_hdf5(args.subset, target)
            with h5py.File(source, 'r') as subset:
                names = list(apbs_matches)
                full_path = resolve_target_graph_hdf5(args.feature_data, target)
                apbs_path = args.apbs_dir / f'{target}_apbs_surface.hdf5'
                for directory in ('subset_hdf5', 'apbs_model_data'):
                    (stage / directory).mkdir(exist_ok=True)
                with h5py.File(full_path, 'r') as full, h5py.File(apbs_path, 'r') as apbs, \
                     h5py.File(stage / 'subset_hdf5' / f'{target}.hdf5', 'w') as graphs_out, \
                     h5py.File(stage / 'apbs_model_data' / apbs_path.name, 'w') as apbs_out:
                    for key, value in subset.attrs.items():
                        graphs_out.attrs[key] = value
                    graphs_out.attrs['voronoi_subset_model_count'] = len(names)
                    # Preserve provenance, including absence of area-weighting
                    # markers. Packaging must not certify an unverified store.
                    for key, value in apbs.attrs.items():
                        apbs_out.attrs[key] = value
                    for name in names:
                        for key in ('node_reference', 'edge_features/contacts', 'target_scores/DockQ'):
                            if not np.array_equal(subset[name][key][()], full[name][key][()]):
                                raise ValueError(f'{target}/{name}: subset and full graph differ at {key}')
                        full.copy(name, graphs_out, name=name)
                        apbs_name = apbs_matches[name]
                        if apbs_name not in apbs_out:
                            apbs.copy(apbs_name, apbs_out, name=apbs_name)
                        relative, _ = infer_relative_pdb_path(target, name)
                        copy_file(args.pdb_root / relative, stage / 'pdb' / relative)
            fasta = args.esm_root / f'{target}_all.fasta'
            copy_file(fasta, stage / 'SS_embeds' / fasta.name)
            labels = [line[1:].split()[0] for line in fasta.read_text().splitlines() if line.startswith('>')]
            if not labels:
                raise ValueError(f'No FASTA records in {fasta}')
            for label in labels:
                chain = label.rsplit('.', 1)[-1]
                if not chain.isalnum():
                    raise ValueError(f'Unexpected FASTA chain label: {label}')
                filename = f'{target}.{chain}.pt'
                copy_file(args.esm_root / filename, stage / 'SS_embeds' / filename)
            filename = f'{target}_avg_rSASA_i.csv'
            copy_file(args.optional_dir / filename, stage / 'rsasa_i' / filename)
            manifest['targets'][target] = names
            print(f'{target}: bundled {len(names)} graphs and matching inputs', flush=True)
        if not manifest['targets']:
            raise ValueError('No nonempty targets available')
        (stage / 'apbs_coverage.json').write_text(json.dumps(coverage, indent=2) + '\n')
        metadata = args.subset.parent
        for filename in ('subset_manifest.csv', 'target_attrition.csv', 'summary.json'):
            if (metadata / filename).is_file():
                copy_file(metadata / filename, stage / 'source_audit' / filename)
        manifest['files'] = {str(p.relative_to(stage)): p.stat().st_size
                             for p in sorted(stage.rglob('*')) if p.is_file()}
        manifest['file_sha256'] = {name: file_sha256(stage / name) for name in manifest['files']}
        manifest['total_bytes'] = sum(manifest['files'].values())
        (stage / 'bundle_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        os.replace(stage, output)
        print(f'Bundle ready: {output}; {manifest["total_bytes"] / 2**30:.2f} GiB')
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    project = Path('/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder')
    parser.add_argument('--mode', choices=('smoke', 'full'), default='full')
    parser.add_argument('--targets', nargs='+')
    parser.add_argument('--missing-apbs', choices=('error', 'exclude'), default='error',
                        help='Fail before copying (default), or exclude missing APBS graphs from the shared cohort')
    parser.add_argument('--preflight-only', action='store_true', help='Write APBS coverage audit without copying data')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--subset', type=Path, default=project / 'voronoi_dataset_audit/subset_hdf5')
    parser.add_argument('--feature-data', type=Path, default=Path('/nfs/roberts/project/pi_co54/jas485/ppi_processed_graphs'))
    parser.add_argument('--apbs-dir', type=Path, default=Path('/nfs/roberts/pi/pi_co54/jas485/ppi_gnn_data_store/apbs_model_data'))
    parser.add_argument('--esm-root', type=Path, default=Path('/nfs/roberts/pi/pi_co54/nb685/scratch_backup/SS_embeds'))
    parser.add_argument('--pdb-root', type=Path, default=Path('/nfs/roberts/pi/pi_co54/jas485/uniformly_sampled_target_data'))
    parser.add_argument('--optional-dir', type=Path, default=Path('/home/jas485/project_pi_co54/jas485/rsasa_i_graph_data'))
    build_bundle(parser.parse_args())


if __name__ == '__main__':
    main()
