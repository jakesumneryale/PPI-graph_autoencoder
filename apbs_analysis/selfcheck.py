"""Fast correctness checks for the pure-Python pieces (no APBS required).

Covers the parts where a silent error would corrupt every downstream number:
grid dimensioning, the OpenDX round trip and its axis order, surface-area
sampling, and the residue renumbering that the graph join depends on.

    python -m apbs_analysis.selfcheck
"""

from __future__ import annotations

from pathlib import Path
import re
import sys
import tempfile

import numpy as np

from apbs_analysis.common import model_id_from_pdb_name, graph_group_candidates
from apbs_analysis.dx_grid import DxGrid, read_dx
from apbs_analysis.export_dx import write_dx
from apbs_analysis.grid_sizing import DIME_STEP, compute_grid_parameters
from apbs_analysis.electrostatics import (
    fibonacci_sphere,
    parse_pqr,
    solvent_accessible_points,
    write_pqr,
)
from apbs_analysis.structure_prep import prepare_structure


def check_grid_sizing() -> None:
    coords = np.array([[0.0, 0.0, 0.0], [30.0, 10.0, 5.0]])
    radii = np.array([2.0, 2.0])
    grid = compute_grid_parameters(coords, radii)
    assert all((value - 1) % DIME_STEP == 0 for value in grid.dime), grid.dime
    assert all(f <= c + 1e-9 for f, c in zip(grid.fglen, grid.cglen)), "fine box exceeds coarse box"
    assert np.allclose(grid.center, [15.0, 5.0, 2.5]), grid.center
    assert all(spacing <= 0.5 + 1e-9 for spacing in grid.fine_spacing), grid.fine_spacing

    # A tight memory ceiling must coarsen the grid rather than be ignored.
    tight = compute_grid_parameters(coords, radii, memory_ceiling_mb=2.0)
    assert tight.memory_mb <= grid.memory_mb, (tight.memory_mb, grid.memory_mb)
    print("  grid sizing: dime valid, boxes nested, memory ceiling respected")


def check_dx_roundtrip() -> None:
    # Deliberately non-cubic with distinct spacings: a transposed axis order or
    # a swapped delta would survive a symmetric test but not this one.
    values = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    grid = DxGrid(
        origin=np.array([-1.0, 2.0, 0.5]),
        spacing=np.array([0.25, 0.5, 1.0]),
        values=values,
    )
    with tempfile.TemporaryDirectory() as temporary:
        path = write_dx(grid, Path(temporary) / "test.dx")
        restored = read_dx(path)
        assert restored.shape == grid.shape, (restored.shape, grid.shape)
        assert np.allclose(restored.origin, grid.origin)
        assert np.allclose(restored.spacing, grid.spacing)
        assert np.allclose(restored.values, grid.values), "values or axis order changed"

        gzipped = write_dx(grid, Path(temporary) / "test.dx.gz")
        assert np.allclose(read_dx(gzipped).values, grid.values), "gzip path differs"

    # Sampling exactly on grid nodes must return those nodes' values.
    nodes = np.array([[-1.0, 2.0, 0.5], [-1.0 + 0.25 * 3, 2.0 + 0.5 * 2, 0.5 + 1.0 * 4]])
    sampled = grid.sample(nodes)
    assert np.allclose(sampled, [values[0, 0, 0], values[3, 2, 4]]), sampled
    print("  dx round trip: shape, origin, spacing, axis order, gzip, node sampling")


def check_surface_sampling() -> None:
    # One isolated atom: its accessible area must be the analytic sphere area.
    radius = 2.0
    probe = 1.4
    points, owner, sasa = solvent_accessible_points(
        np.zeros((1, 3)), np.array([radius]), probe_radius=probe, sphere_points=500
    )
    expected = 4.0 * np.pi * (radius + probe) ** 2
    assert abs(sasa[0] - expected) < 1e-6, (sasa[0], expected)
    assert len(points) == 500 and set(owner.tolist()) == {0}
    assert np.allclose(np.linalg.norm(points, axis=1), radius + probe)

    # A small atom engulfed by a large one is fully buried; the large one is not.
    _, _, engulfed = solvent_accessible_points(
        np.zeros((2, 3)), np.array([1.0, 5.0]), probe_radius=probe, sphere_points=200
    )
    assert np.isclose(engulfed[0], 0.0), engulfed
    assert engulfed[1] > 0.0, engulfed

    # Two nearly coincident atoms lose roughly half their area to each other.
    _, _, touching = solvent_accessible_points(
        np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]]),
        np.array([radius, radius]),
        probe_radius=probe,
        sphere_points=400,
    )
    assert all(0.3 * expected < value < 0.7 * expected for value in touching), touching

    directions = fibonacci_sphere(1000)
    assert np.allclose(np.linalg.norm(directions, axis=1), 1.0)
    assert np.linalg.norm(directions.mean(axis=0)) < 0.01, "sphere sampling is not centred"
    print("  surface sampling: analytic area, full occlusion, uniform directions")


DUPLICATE_NUMBERING_PDB = """\
ATOM      1  N   SER A  20      0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  SER A  20      1.500   0.000   0.000  1.00  0.00           C
ATOM      3  C   SER A  20      2.000   1.400   0.000  1.00  0.00           C
ATOM      4  O   SER A  20      1.300   2.400   0.000  1.00  0.00           O
ATOM      5  CB  SER A  20      2.000  -0.800   1.200  1.00  0.00           C
ATOM      6  OG  SER A  20      3.400  -0.900   1.200  1.00  0.00           O
ATOM      0  HG  SER A  20      3.700  -1.100   2.100  1.00  0.00           H
ATOM      7  N   SER A  20     10.000   0.000   0.000  1.00  0.00           N
ATOM      8  CA  SER A  20     11.500   0.000   0.000  1.00  0.00           C
ATOM      9  C   SER A  20     12.000   1.400   0.000  1.00  0.00           C
ATOM     10  O   SER A  20     11.300   2.400   0.000  1.00  0.00           O
ATOM     11  CB  SER A  20     12.000  -0.800   1.200  1.00  0.00           C
ATOM     12  OG  SER A  20     13.400  -0.900   1.200  1.00  0.00           O
ATOM     13  N   LYS B  31     20.000   0.000   0.000  1.00  0.00           N
ATOM     14  CA  LYS B  31     21.500   0.000   0.000  1.00  0.00           C
ATOM     15  C   LYS B  31     22.000   1.400   0.000  1.00  0.00           C
ATOM     16  O   LYS B  31     21.300   2.400   0.000  1.00  0.00           O
ATOM     17  CB  LYS B  31     22.000  -0.800   1.200  1.00  0.00           C
"""


def check_structure_prep() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        work_dir = Path(temporary)
        source = work_dir / "duplicate.pdb"
        source.write_text(DUPLICATE_NUMBERING_PDB)

        prepared = prepare_structure(source, work_dir / "prepared.pdb")
        # The two same-numbered SER residues must stay distinct.
        assert len(prepared) == 3, len(prepared)
        assert list(prepared.number) == [20, 20, 31], prepared.number
        assert list(prepared.chain) == ["A", "A", "B"], prepared.chain

        written = [
            line for line in prepared.pdb_path.read_text().splitlines() if line.startswith("ATOM")
        ]
        # Renumbering must make the residue number equal aa_id + 1 ...
        numbers = sorted({int(line[22:26]) for line in written})
        assert numbers == [1, 2, 3], numbers
        # ... and hydrogens must be gone.
        assert not any(line[76:78].strip() == "H" for line in written), "hydrogen survived"
        assert len(written) == 6 + 6 + 5, len(written)  # two full SER, one CB-truncated LYS

        # LYS is missing CG/CD/CE/NZ: flagged always, truncated only on request.
        assert list(prepared.incomplete) == [False, False, True], prepared.incomplete
        assert not prepared.truncated.any(), "truncation must not happen by default"
        assert prepared.has_incomplete_residues

        truncated = prepare_structure(source, work_dir / "trunc.pdb", truncate_incomplete=True)
        assert list(truncated.truncated) == [False, False, True], truncated.truncated
        assert truncated.modeled_name[2] == "ALA", truncated.modeled_name
        # Residue blocks and aa_ids must be identical across both preparations.
        assert list(truncated.number) == list(prepared.number)
        assert list(truncated.chain) == list(prepared.chain)
    print("  structure prep: duplicates split, aa_id renumbering, H stripped, truncation fallback")


# Line 2 is the case that breaks a whitespace split: pdb2pqr's fixed-width
# columns run 60.423 straight into -177.683.
SAMPLE_PQR = """\
ATOM      1  N   CYS E   1       2.323 -16.405  18.812 -0.3200 2.0000
ATOM      2  N   GLN A   1     138.038  60.423-177.683 -0.3200 2.0000
ATOM      3  CG2 VAL B 445      34.563-100.766 -44.817  0.0000 2.0000
"""


def check_pqr_parsing() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "sample.pqr"
        path.write_text(SAMPLE_PQR)
        structure = parse_pqr(path)

    assert len(structure) == 3, len(structure)
    assert list(structure.chain) == ["E", "A", "B"], structure.chain
    assert list(structure.resnum) == [1, 1, 445], structure.resnum
    assert list(structure.resname) == ["CYS", "GLN", "VAL"], structure.resname
    assert list(structure.atom_name) == ["N", "N", "CG2"], structure.atom_name
    assert np.allclose(structure.xyz[0], [2.323, -16.405, 18.812]), structure.xyz[0]
    assert np.allclose(structure.xyz[1], [138.038, 60.423, -177.683]), structure.xyz[1]
    assert np.allclose(structure.xyz[2], [34.563, -100.766, -44.817]), structure.xyz[2]
    assert np.allclose(structure.charge, [-0.32, -0.32, 0.0]), structure.charge
    assert np.allclose(structure.radius, [2.0, 2.0, 2.0]), structure.radius
    # write_pqr must round-trip those same far-from-origin coordinates, since
    # APBS reads that file rather than pdb2pqr's fixed-width one.
    with tempfile.TemporaryDirectory() as temporary:
        rewritten = write_pqr(structure, Path(temporary) / "apbs_input.pqr")
        text = rewritten.read_text()
        assert not re.search(r"\d-\d", text), "rewritten PQR still has concatenated fields"
        again = parse_pqr(rewritten)

    assert np.allclose(again.xyz, structure.xyz), (again.xyz, structure.xyz)
    assert np.allclose(again.charge, structure.charge)
    assert np.allclose(again.radius, structure.radius)
    assert list(again.chain) == list(structure.chain)
    assert list(again.resnum) == list(structure.resnum)
    assert list(again.atom_name) == list(structure.atom_name)
    print("  pqr parsing: collided columns, and a separation-safe APBS round trip")


def check_string_storage() -> None:
    """Fixed-width string columns must round-trip, including empty values."""
    import h5py

    from apbs_analysis.storage import _write_strings

    values = np.array(["A", "", "GLY", "HIS", "CG2"], dtype="<U8")
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "strings.hdf5"
        with h5py.File(path, "w") as handle:
            _write_strings(handle, "column", values)
        with h5py.File(path, "r") as handle:
            dataset = handle["column"]
            assert dataset.dtype.kind == "S", f"expected fixed-width bytes, got {dataset.dtype}"
            assert h5py.check_string_dtype(dataset.dtype), "asstr() path would break"
            restored = dataset.asstr()[:]
    assert list(restored) == list(values), (restored, values)
    print("  string storage: fixed-width columns round-trip via asstr()")


def check_model_naming() -> None:
    assert model_id_from_pdb_name("complex.0_0_11_corrected_H_0001.pdb") == "complex.0_0_11"
    assert model_id_from_pdb_name("complex.1234_5_corrected_H_0001.pdb") == "complex.1234_5"
    assert model_id_from_pdb_name("1acb_complex_H.pdb") == "1acb"
    assert graph_group_candidates("complex.0_0_11") == ("complex.0_0_11", "complex.0_0_11_corrected")
    assert graph_group_candidates("complex.0_0_11_corrected") == (
        "complex.0_0_11",
        "complex.0_0_11_corrected",
    )
    print("  model naming: PDB filename <-> graph group mapping")


def main() -> None:
    checks = (
        check_grid_sizing,
        check_dx_roundtrip,
        check_surface_sampling,
        check_structure_prep,
        check_pqr_parsing,
        check_string_storage,
        check_model_naming,
    )
    failures = 0
    for check in checks:
        try:
            check()
        except AssertionError as exc:
            print(f"  FAIL {check.__name__}: {exc}", file=sys.stderr)
            failures += 1
    if failures:
        raise SystemExit(f"{failures}/{len(checks)} self-checks failed")
    print(f"All {len(checks)} self-checks passed.")


if __name__ == "__main__":
    main()
