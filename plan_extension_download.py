"""Reuse checksum-identical local inputs and write a missing-file Globus batch.

Uses only the Python standard library. Existing source files stay untouched;
verified reusable files are exposed through symlinks in the new bundle directory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil


def checksum(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def plan(args):
    root = args.local_root.resolve()
    manifest = json.loads((root / 'bundle_manifest.json').read_text())
    reusable = {'pdb': args.reuse_pdb, 'rsasa_i': args.reuse_rsasa, 'SS_embeds': args.reuse_esm}
    batch, remaining, reused, present = [], 0, 0, 0
    for name, size in manifest['files'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError(f'Unsafe manifest path: {name}')
        destination = root / relative
        if not destination.parent.resolve().is_relative_to(root):
            raise ValueError(f'Destination parent escapes bundle directory: {destination}')
        expected = manifest['file_sha256'][name]
        def matches(path):
            return path.is_file() and path.stat().st_size == size and checksum(path) == expected
        if matches(destination):
            present += 1
            continue
        if destination.is_symlink():
            raise ValueError(f'Existing symlink differs from bundle; choose a fresh LOCAL_ROOT: {destination}')
        candidates_root = reusable.get(relative.parts[0])
        candidate = candidates_root / Path(*relative.parts[1:]) if candidates_root else None
        if not destination.exists() and candidate is not None and matches(candidate):
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.symlink_to(candidate.resolve())
            reused += 1
            continue
        source = args.source_bundle.rstrip('/') + '/' + name
        target = args.destination_path.rstrip('/') + '/' + name
        batch.append(shlex.quote(source) + ' ' + shlex.quote(target))
        remaining += size
    free = shutil.disk_usage(root).free
    if free < remaining + 2 * 2**30:
        raise ValueError(f'Insufficient disk space: need {remaining / 2**30:.2f} GiB plus 2 GiB reserve; free {free / 2**30:.2f} GiB')
    (root / '.globus-transfer.txt').write_text(''.join(line + '\n' for line in batch))
    print(f'{reused} existing local files reused; {present} bundle files already match; '
          f'{len(batch)} files to transfer ({remaining / 2**30:.2f} GiB).', flush=True)
    return len(batch)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--local-root', type=Path, required=True)
    p.add_argument('--source-bundle', required=True)
    p.add_argument('--destination-path', required=True)
    p.add_argument('--reuse-pdb', type=Path, default=Path('/scratch/uniformly_sampled_ppi_data'))
    p.add_argument('--reuse-rsasa', type=Path, default=Path('/scratch/ppi_autoencoder_code/rsasa_i_data'))
    p.add_argument('--reuse-esm', type=Path, default=Path(__file__).parent / 'ESM2_embedding_ex')
    plan(p.parse_args())


if __name__ == '__main__':
    main()
