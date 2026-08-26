"""Rewrite an existing store with the current on-disk layout, without recomputing.

Only needed for stores written before string columns moved from HDF5's
variable-length strings to fixed-width bytes. The values are copied verbatim,
so this is a format migration, not a recalculation.

    python -m apbs_analysis.repack_store <store.hdf5>
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys

import h5py
import numpy as np

from apbs_analysis.storage import COMPRESSION, _write_strings


def repack(source_path: Path, destination_path: Path) -> None:
    with h5py.File(source_path, "r") as source, h5py.File(destination_path, "w") as destination:
        for key, value in source.attrs.items():
            destination.attrs[key] = value
        for index, name in enumerate(source, start=1):
            group = source[name]
            new_group = destination.create_group(name)
            for key, value in group.attrs.items():
                new_group.attrs[key] = value
            for dataset_name in group:
                dataset = group[dataset_name]
                if h5py.check_string_dtype(dataset.dtype):
                    _write_strings(new_group, dataset_name, dataset.asstr()[:])
                else:
                    new_group.create_dataset(
                        dataset_name, data=dataset[()], dtype=dataset.dtype, **COMPRESSION
                    )
            if index % 10 == 0:
                print(f"  repacked {index} groups", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("store_path")
    parser.add_argument("--output", help="Write here instead of replacing the input in place")
    args = parser.parse_args()

    source_path = Path(args.store_path).expanduser()
    if not source_path.is_file():
        sys.exit(f"No store at {source_path}")

    temporary_path = source_path.with_suffix(".repack.tmp")
    repack(source_path, temporary_path)

    if args.output:
        shutil.move(temporary_path, Path(args.output).expanduser())
        final = Path(args.output).expanduser()
    else:
        os.replace(temporary_path, source_path)  # atomic: never a half-written store
        final = source_path
    print(f"Wrote {final} ({final.stat().st_size / 1024**3:.2f} GiB)")


if __name__ == "__main__":
    main()
