"""Prepare isolated, audited graph copies for the extension experiment.

Input must be the existing 10% subset. No sampling or source mutation occurs.
One invocation per target can run concurrently; each output is atomically replaced.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from contextlib import ExitStack
from functools import lru_cache
from collections import Counter
from add_interface_node_degree import calculate_interface_node_degree

import h5py
import numpy as np
import torch
from Bio.Data.PDBData import protein_letters_3to1
from Bio.PDB import PDBParser
from Bio import SeqIO
from voronoi_edge_features.common import infer_relative_pdb_path, resolve_target_graph_hdf5


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_residues(path):
    model = next(PDBParser(QUIET=True).get_structure('decoy', path).get_models())
    rows, positions = [], []
    for chain in model:
        for residue in chain:
            if residue.id[0] != ' ':
                continue
            name = protein_letters_3to1.get(residue.resname)
            if name is None or 'CA' not in residue:
                raise ValueError(f'Nonstandard residue or missing CA: {chain.id}:{residue.id}')
            rows.append((chain.id, name))
            positions.append(residue['CA'].coord)
    return rows, np.asarray(positions, dtype=np.float32)


def validate_nodes(graph, residues):
    reference = graph['node_reference'].asstr()[()]
    n = len(residues)
    if reference.shape != (n, 3):
        raise ValueError('PDB / node_reference size mismatch')
    if not np.array_equal(reference[:, 0].astype(int), np.arange(n)):
        raise ValueError('Expected zero-based sequential node_reference IDs')
    if list(reference[:, 2]) != [r[1] for r in residues]:
        raise ValueError('PDB / graph residue sequence mismatch')
    chain_order = list(dict.fromkeys(r[0] for r in residues))
    expected = np.array([chain_order.index(r[0]) + 1 for r in residues])
    if not np.array_equal(reference[:, 1].astype(int), expected):
        raise ValueError('PDB / graph chain order mismatch')
    return chain_order


@lru_cache(maxsize=32)
def read_embedding(path, layer):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or 'representations' not in payload:
        raise ValueError(f'{path}: expected ESM representations dictionary')
    tensor = payload['representations'][layer].detach().cpu().float()
    return payload.get('label'), tensor.numpy().copy(), sha256(path)


@lru_cache(maxsize=32)
def read_fasta(path):
    records = list(SeqIO.parse(path, 'fasta'))
    sequences = {record.id: str(record.seq) for record in records}
    if len(sequences) != len(records):
        raise ValueError(f'{path}: duplicate FASTA labels')
    return sequences, sha256(path)


def load_embeddings(root, target, graph_name, residues, fasta_template, embedding_template, layer):
    context = dict(target=target, graph=graph_name)
    fasta = root / fasta_template.format(**context)
    records, fasta_hash = read_fasta(fasta)
    chains = list(dict.fromkeys(r[0] for r in residues))
    candidates = {}
    for chain in chains:
        graph_sequence = ''.join(aa for c, aa in residues if c == chain)
        candidates[chain] = []
        for label, sequence in records.items():
            # FASTA labels define embedding filenames; PDB labels do not.
            # Example: source E/I can match standardized 1acb.A/1acb.B.
            embedding_chain = label.rsplit('.', 1)[-1]
            starts = [i for i in range(len(sequence) - len(graph_sequence) + 1)
                      if sequence[i:i + len(graph_sequence)] == graph_sequence]
            if len(starts) != 1:
                continue
            path = root / embedding_template.format(**context, chain=embedding_chain)
            stored_label, tensor, embedding_hash = read_embedding(path, layer)
            if stored_label != label:
                raise ValueError(f'{path}: embedding label {stored_label!r} does not match FASTA {label!r}')
            if tensor.ndim != 2 or tensor.shape[0] != len(sequence):
                raise ValueError(f'{path}: embedding length must equal FASTA length (no BOS/EOS rows)')
            start = starts[0]
            block = tensor[start:start + len(graph_sequence)]
            if not np.isfinite(block).all():
                raise ValueError(f'{path}: nonfinite embedding')
            candidates[chain].append((label, block, {
                'path': str(path), 'sha256': embedding_hash, 'label': label,
                'embedding_chain': embedding_chain, 'start': start, 'length': len(graph_sequence)}))
        if not candidates[chain]:
            raise ValueError(f'{fasta}: PDB chain {chain} has no unique exact sequence alignment; '
                             'regenerate embeddings for its PDB sequence')

    # Require a one-to-one chain assignment. Homomer ambiguity is harmless only
    # when every possible assignment gives exactly the same per-node features.
    first = None
    equivalent = 0
    def assign(index, used, mapping):
        nonlocal first, equivalent
        if index == len(chains):
            if first is None:
                first = dict(mapping)
            elif any(not np.array_equal(first[c][1], mapping[c][1]) for c in chains):
                raise ValueError(f'{fasta}: ambiguous chain sequence alignment produces different embeddings')
            equivalent += 1
            return
        chain = chains[index]
        for candidate in candidates[chain]:
            if candidate[0] not in used:
                mapping[chain] = candidate
                assign(index + 1, used | {candidate[0]}, mapping)
    assign(0, set(), {})
    if first is None:
        raise ValueError(f'{fasta}: no one-to-one chain sequence alignment')
    widths = {first[c][1].shape[1] for c in chains}
    if len(widths) != 1:
        raise ValueError(f'{fasta}: inconsistent embedding widths')
    embedding = np.empty((len(residues), widths.pop()), dtype=np.float32)
    provenance = {'fasta': str(fasta), 'fasta_sha256': fasta_hash, 'layer': layer,
                  'mapping_method': 'sequence_one_to_one', 'equivalent_assignments': equivalent,
                  'chains': {}}
    for chain in chains:
        indices = [i for i, (c, _) in enumerate(residues) if c == chain]
        embedding[indices] = first[chain][1]
        provenance['chains'][chain] = first[chain][2]
    return embedding, provenance


def contact_indices(graph, n):
    contacts = np.asarray(graph['edge_features/contacts'][()], dtype=np.int64)
    if contacts.ndim != 2 or contacts.shape[1] != 2:
        raise ValueError('contacts must be [E, 2]')
    # IDs are explicitly validated against node_reference; no indexing heuristic.
    if contacts.size and (contacts.min() < 0 or contacts.max() >= n):
        raise ValueError('contacts are not zero-based node IDs')
    return contacts


def apbs_features(group, graph, residues):
    n = len(residues)
    if not np.array_equal(group['residue_aa_id'][()], np.arange(n)):
        raise ValueError('APBS / graph residue IDs differ')
    names = [protein_letters_3to1.get(s, '?') for s in group['residue_name'].asstr()[()]]
    chains = list(group['residue_chain'].asstr()[()])
    if list(zip(chains, names)) != residues:
        raise ValueError('APBS / PDB residue identity or chain mismatch')
    phi = np.asarray(group['residue_potential_mean'][()], dtype=float)
    valid = ((group['residue_surface_point_count'][()] > 0) &
             (group['residue_in_pqr'][()] > 0) & np.isfinite(phi))
    if 'residue_truncated' in group:
        valid &= group['residue_truncated'][()] == 0
    src, dst = contact_indices(graph, n).T
    missing = ~(valid[src] & valid[dst])
    a, b = np.where(valid, phi, 0)[src], np.where(valid, phi, 0)[dst]
    area = np.asarray(graph['edge_features/voronoi_contact_area'][()]).reshape(-1)
    area_missing = np.asarray(graph['edge_features/voronoi_contact_missing'][()]).reshape(-1) > 0
    result = {'apbs_pair_mean': (a + b) / 2, 'apbs_pair_absdiff': np.abs(a - b),
              'apbs_pair_product': a * b, 'apbs_pair_area_product': a * b * area}
    for name, values in result.items():
        values[missing] = 0
        if name == 'apbs_pair_area_product':
            values[area_missing] = 0
        if not np.isfinite(values).all():
            raise ValueError(f'Nonfinite {name}')
        result[name] = values.astype(np.float32)[:, None]
    result['apbs_pair_missing'] = missing.astype(np.float32)[:, None]
    return result


def prepare_target(args, target):
    source = resolve_target_graph_hdf5(args.data, target)
    destination = args.output / f'{target}.hdf5'
    if source.resolve() == destination.resolve():
        raise ValueError('Output must be separate from the source subset')
    args.output.mkdir(parents=True, exist_ok=True)
    apbs_path = args.apbs_dir / f'{target}_apbs_surface.hdf5'
    audit = {'target': target, 'source': str(source), 'source_sha256': sha256(source),
             'apbs_path': str(apbs_path), 'accepted': [], 'rejected': {}, 'models': {}}
    fd, temporary = tempfile.mkstemp(prefix=f'.{target}.', suffix='.hdf5', dir=args.output)
    os.close(fd)
    try:
        with ExitStack() as stack:
            original = stack.enter_context(h5py.File(source))
            audit['source_entry_count'] = len(original)
            audit['source_subset_model_count'] = (
                int(original.attrs['voronoi_subset_model_count'])
                if 'voronoi_subset_model_count' in original.attrs else None)
            if not len(original):
                audit['input_error'] = 'empty_source_subset'
                audit_path = args.output / f'{target}.audit.json'
                audit_path.write_text(json.dumps(audit, indent=2) + '\n')
                raise ValueError(
                    f'{target}: source subset {source} has zero graph entries. '
                    'No APBS or ESM checks ran. Check subset_manifest.csv and '
                    f'the upstream Voronoi audit before rebuilding the subset; see {audit_path}')
            apbs = stack.enter_context(h5py.File(apbs_path))
            output = stack.enter_context(h5py.File(temporary, 'w'))
            latest = (stack.enter_context(h5py.File(resolve_target_graph_hdf5(args.feature_data, target)))
                      if args.feature_data else original)
            weighted = bool(apbs.attrs.get('residue_statistics_area_weighted', False))
            audit['area_weighting'] = ('recorded' if weighted else
                'user_asserted' if args.assume_area_weighted else 'unverified_at_file_level')
            audit['apbs_settings'] = {k: str(v) for k, v in apbs.attrs.items()}
            for name in sorted(original):
                graph = original[name]
                try:
                    if latest is not original:
                        updated = latest[name]
                        for key in ('node_reference', 'edge_features/contacts', 'target_scores/DockQ'):
                            if not np.array_equal(graph[key][()], updated[key][()]):
                                raise ValueError(f'Updated feature graph differs from subset: {key}')
                        graph = updated
                    relative, _ = infer_relative_pdb_path(target, name)
                    pdb = args.pdb_root / relative
                    residues, pos = read_residues(pdb)
                    validate_nodes(graph, residues)
                    if not np.isfinite(pos).all():
                        raise ValueError('Nonfinite coordinates')
                    if not np.any(graph['node_features/interface_nodes'][()]):
                        raise ValueError('No interface nodes')
                    for key in ('interface_edges', 'ca_dist', 'voronoi_contact_area', 'voronoi_contact_missing'):
                        if not np.isfinite(graph[f'edge_features/{key}'][()]).all():
                            raise ValueError(f'Nonfinite {key}')
                    if not np.isfinite(float(graph['target_scores/DockQ'][()])):
                        raise ValueError('Nonfinite DockQ')
                    embeddings, provenance = load_embeddings(args.esm_root, target, name, residues,
                        args.fasta_template, args.embedding_template, args.esm_layer)
                    apbs_name = name.removesuffix('_corrected')
                    if apbs_name not in apbs:
                        apbs_name += '_corrected'
                    apbs_group = apbs[apbs_name]
                    if not (weighted or apbs_group.attrs.get('residue_statistics_area_weighted', False) or args.assume_area_weighted):
                        raise ValueError('APBS area-weighting provenance absent: validate/migrate the store, or explicitly use --assume-area-weighted')
                    units = apbs_group.attrs.get('potential_units', apbs.attrs.get('potential_units'))
                    if units != 'kT/e':
                        raise ValueError(f'Expected APBS potential units kT/e, got {units!r}')
                    features = apbs_features(apbs_group, graph, residues)
                    graph.file.copy(graph, output, name=name)
                    copied = output[name]
                    for key, values in (('pos', pos), ('esm2', embeddings)):
                        if key in copied:
                            del copied[key]
                        copied.create_dataset(key, data=values, compression='gzip')
                    copied.attrs['esm_provenance'] = json.dumps(provenance)
                    copied.attrs['source_pdb'] = str(pdb)
                    for key, values in features.items():
                        if key in copied['edge_features']:
                            del copied['edge_features'][key]
                        copied['edge_features'].create_dataset(key, data=values, compression='gzip')
                    degree = calculate_interface_node_degree(
                        graph['node_features/interface_nodes'][()], contact_indices(graph, len(residues)))
                    nodes = copied['node_features']
                    if 'interface_node_degree' in nodes:
                        del nodes['interface_node_degree']
                    nodes.create_dataset('interface_node_degree', data=degree)
                    audit['accepted'].append(name)
                    audit['models'][name] = {'pdb_sha256': sha256(pdb), 'esm': provenance,
                        'apbs_missing_edges': int(features['apbs_pair_missing'].sum()),
                        'apbs_area_weighting': 'recorded' if weighted or apbs_group.attrs.get('residue_statistics_area_weighted', False) else 'user_asserted'}
                except (KeyError, ValueError, FileNotFoundError, OSError) as exc:
                    if name in output:
                        del output[name]
                    audit['rejected'][name] = str(exc)
            output.attrs['extension_schema'] = 1
        audit_path = args.output / f'{target}.audit.json'
        audit_path.write_text(json.dumps(audit, indent=2) + '\n')
        if not audit['accepted']:
            reasons = Counter(audit['rejected'].values())
            summary = '\n'.join(f'  {count} graph(s): {reason}'
                                for reason, count in reasons.most_common(5))
            raise ValueError(
                f'{target}: no eligible graphs; {len(audit["rejected"])} rejected. '
                f'Most common reasons:\n{summary}\nFull audit: {audit_path}')
        os.replace(temporary, destination)
        print(f"{target}: {len(audit['accepted'])} accepted, {len(audit['rejected'])} rejected")
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True, help='Existing 10% subset HDF5 directory')
    p.add_argument('--feature-data', type=Path, help='Optional full graph directory with recently committed Voronoi/mask features')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--target', required=True)
    p.add_argument('--pdb-root', type=Path, required=True)
    p.add_argument('--apbs-dir', type=Path, required=True)
    p.add_argument('--esm-root', type=Path, default=Path('/nfs/roberts/pi/pi_co54/nb685/scratch_backup/SS_embeds'))
    p.add_argument('--fasta-template', default='{target}_all.fasta')
    p.add_argument('--embedding-template', default='{target}.{chain}.pt')
    p.add_argument('--esm-layer', type=int, default=33)
    p.add_argument('--assume-area-weighted', action='store_true', help='Explicitly attest that an unmarked store used the area-weighted implementation')
    args = p.parse_args()
    prepare_target(args, args.target)


if __name__ == '__main__':
    main()
