"""Read-only subset scan before dispatching preparation jobs (one CPU).

Empty subset files contribute no graphs to any model. Record them explicitly
and schedule only the nonempty targets. Corrupt files remain fatal errors.
"""
import argparse
import json
from pathlib import Path
import h5py


def prepare_target_list(data, output):
    paths = sorted(set(data.glob('*.h5')) | set(data.glob('*.hdf5')))
    if not paths:
        raise ValueError(f'No subset HDF5 files in {data}')
    included, excluded, seen = [], {}, set()
    for path in paths:
        if path.stem in seen:
            raise ValueError(f'Duplicate target files for {path.stem}')
        seen.add(path.stem)
        with h5py.File(path, 'r') as handle:
            count = len(handle)
        if count:
            included.append(path.stem)
        else:
            excluded[path.stem] = {'reason': 'empty_source_subset', 'source': str(path.resolve()),
                                   'source_entry_count': 0}
    output.mkdir(parents=True, exist_ok=True)
    (output / 'excluded_targets.json').write_text(json.dumps(excluded, indent=2) + '\n')
    if len(included) < 3:
        raise ValueError('Need at least three nonempty targets for train/validation/test')
    (output / 'targets.txt').write_text(''.join(name + '\n' for name in included))
    print(f'{len(included)} nonempty targets; {len(excluded)} empty targets excluded. '
          f'Details: {output / "excluded_targets.json"}', flush=True)
    return included, excluded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare_target_list(args.data, args.output)


if __name__ == '__main__':
    main()
