"""ESM2 embeddings (esm2_t33_650M_UR50D, layer 33) for each chain of a target, in the layout esm_sidecars.py reads.

Sequences come from the graph itself (node_reference of the target's first listed model), which is what the existing
SS_embeds FASTA files contain. One forward pass per chain: writes <output>/<pdb>_all.fasta and <pdb>.A.pt / <pdb>.B.pt with
{'label', 'representations': {33: tensor(L, 1280)}, 'mean_representations': {33: tensor(1280)}}. Finished targets are skipped.

ESM2 accepts at most 1022 residues. Longer chains (5GZT) are embedded in overlapping windows (WINDOW long, starting every
STRIDE residues) and overlapping positions are averaged, rather than truncated.

With --check-dir (an existing SS_embeds directory) the job first re-embeds two stored chains and stops unless they match, so a
different PyTorch or fair-esm cannot silently change the embeddings relative to the stored ones.

    python compute_esm_embeddings.py --targets targets.txt --graph-dir graphs/ --model-list-dir model_lists/ --output esm_embeds/
"""
import argparse
import os
import time
from pathlib import Path

import h5py
import numpy as np
import torch

LAYER = 33
WINDOW = 1022
STRIDE = 511


def chain_sequences(graph_path: Path, model: str) -> list[str]:
    with h5py.File(graph_path, 'r') as f:
        ref = f[model]['node_reference'].asstr()[()]
    chains: dict[str, list[str]] = {}
    for _, chain, aa in ref:
        chains.setdefault(str(chain), []).append(str(aa))
    return [''.join(v) for v in chains.values()]


def windows(length: int, window: int = WINDOW, stride: int = STRIDE) -> list[tuple[int, int]]:
    if length <= window:
        return [(0, length)]
    starts = list(range(0, length - window, stride)) + [length - window]
    return [(s, s + window) for s in starts]


CHECK_CHAINS = (('1j2j', 'B'), ('1acb', 'B'))
CHECK_TOLERANCE = 1e-3


def check_reproduces(root: Path, model, converter, device) -> None:
    for target, chain in CHECK_CHAINS:
        lines = (root / f'{target}_all.fasta').read_text().split()
        sequence = dict(zip((l[1:] for l in lines[::2]), lines[1::2]))[f'{target}.{chain}']
        stored = torch.load(root / f'{target}.{chain}.pt', map_location='cpu', weights_only=True)['representations'][LAYER].float()
        diff = (embed(sequence, model, converter, device) - stored).abs().max().item()
        print(f'check {target}.{chain}: max difference from stored embedding {diff:.1e}', flush=True)
        if diff > CHECK_TOLERANCE:
            raise SystemExit(f'{target}.{chain} differs from {root} by {diff:.1e} (limit {CHECK_TOLERANCE:g}); stopping')


@torch.no_grad()
def embed(sequence: str, model, converter, device) -> torch.Tensor:
    """Per-residue layer-33 embedding, shape (L, 1280), without the BOS/EOS rows."""
    total = torch.zeros(len(sequence), 1280); count = torch.zeros(len(sequence), 1)
    for start, end in windows(len(sequence)):
        _, _, tokens = converter([('chain', sequence[start:end])])
        rep = model(tokens.to(device), repr_layers=[LAYER], return_contacts=False)['representations'][LAYER]
        total[start:end] += rep[0, 1:end - start + 1].float().cpu(); count[start:end] += 1
    return total / count


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--targets', type=Path, required=True, help='file with one target per line')
    p.add_argument('--graph-dir', type=Path, required=True)
    p.add_argument('--model-list-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--check-dir', type=Path, help='Existing embedding directory to reproduce before starting')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    import esm
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.eval().to(a.device); converter = alphabet.get_batch_converter()
    if a.check_dir:
        check_reproduces(a.check_dir, model, converter, a.device)
    a.output.mkdir(parents=True, exist_ok=True)
    targets = [t for t in a.targets.read_text().split() if t]
    print(f'{len(targets)} targets on {a.device}', flush=True)
    for i, t in enumerate(targets, 1):
        fasta = a.output / f'{t}_all.fasta'
        if fasta.exists():
            continue
        start = time.monotonic()
        names = [n for n in (a.model_list_dir / f'{t}.txt').read_text().splitlines() if n]
        sequences = chain_sequences(a.graph_dir / f'{t}.hdf5', names[0])
        if len(sequences) != 2:
            raise SystemExit(f'{t}: expected 2 chains, found {len(sequences)}')
        for label, seq in zip('AB', sequences):
            rep = embed(seq, model, converter, a.device)
            if not torch.isfinite(rep).all():
                raise SystemExit(f'{t}.{label}: nonfinite embedding')
            torch.save({'label': f'{t}.{label}', 'representations': {LAYER: rep.clone()},
                        'mean_representations': {LAYER: rep.mean(0).clone()}}, a.output / f'{t}.{label}.pt')
        # written last: its presence marks the target as finished
        tmp = fasta.with_suffix('.tmp')
        tmp.write_text(''.join(f'>{t}.{label}\n{seq}\n' for label, seq in zip('AB', sequences)))
        os.replace(tmp, fasta)
        print(f'[{i}/{len(targets)}] {t}: chains {[len(s) for s in sequences]}, {time.monotonic() - start:.1f} s', flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    main()
