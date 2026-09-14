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

    # Exercise the exact pyvista -> trimesh handoff create_surface() performs.
    # pyvista >= 0.44 removed PolyData.n_faces, which broke create_surface
    # while every import above still succeeded; a preflight that only imports
    # would have passed and the array tasks would have died on the first model.
    if not failures:
        try:
            import pyvista as pv
            import trimesh

            sphere = pv.Icosphere(radius=2.5, center=(0.0, 0.0, 0.0), nsub=2)
            faces = sphere.faces.reshape((-1, 4))[:, 1:]
            mesh = trimesh.Trimesh(sphere.points, faces)
            if not mesh.is_watertight or mesh.faces.shape[0] != faces.shape[0]:
                failures.append("pyvista icosphere -> trimesh conversion produced a bad mesh")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"pyvista icosphere -> trimesh: {exc}")

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
