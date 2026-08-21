#!/usr/bin/env python3
"""Fail-fast dependency and backend check for a Voronoi array worker."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


REQUIRED_IMPORTS = (
    "numpy",
    "pandas",
    "h5py",
    "scipy",
    "Bio",
    "freesasa",
    "sklearn",
    "matplotlib",
    "pyvista",
    "vtk",
    "pyvoro",
    "trimesh",
    "manifold3d",
)


def main() -> None:
    failures = []
    for name in REQUIRED_IMPORTS:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"import {name}: {exc}")

    if not failures:
        try:
            import trimesh

            boxes = [
                trimesh.creation.box(extents=(2, 2, 2)),
                trimesh.creation.box(
                    extents=(2, 2, 2),
                    transform=trimesh.transformations.translation_matrix((1, 0, 0)),
                ),
            ]
            result = trimesh.boolean.union(boxes, engine="manifold")
            if result is None or not result.is_volume:
                failures.append("trimesh manifold boolean returned no valid volume")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"trimesh manifold backend: {exc}")

    if not failures:
        try:
            from voronoi_edge_features.contact_area import load_voronoi_dependencies

            load_voronoi_dependencies()
        except Exception as exc:  # noqa: BLE001
            failures.append(f"project Voronoi dependency load: {exc}")

    if failures:
        print("Voronoi environment preflight FAILED:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        raise SystemExit(1)

    print(f"Voronoi environment preflight passed with {sys.executable}")


if __name__ == "__main__":
    main()
