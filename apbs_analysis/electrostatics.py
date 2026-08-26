"""Run pdb2pqr + APBS on one PDB and project the potential onto its surface.

Pipeline per structure:
  1. pdb2pqr assigns a forcefield charge and radius to every atom (PQR).
  2. APBS solves the Poisson-Boltzmann equation on a focused multigrid and
     writes the potential as an OpenDX volume.
  3. A Shrake-Rupley style solvent-accessible point cloud is generated from the
     PQR radii, each point tagged with its parent atom/residue.
  4. The volumetric potential is trilinearly sampled at those surface points
     (and at atom centres), then aggregated per residue.

Step 4 is what makes this a *surface* charge map: the raw DX volume is mostly
solvent and protein interior, whereas the per-residue surface statistics are
the compact, structure-aligned quantity worth carrying to thousands of models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import gzip
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import numpy as np

from apbs_analysis.dx_grid import DxGrid, read_dx
from apbs_analysis.grid_sizing import GridParameters, compute_grid_parameters
from apbs_analysis.structure_prep import PreparedStructure, prepare_structure


DEFAULT_PROBE_RADIUS = 1.4      # Angstrom, water probe for the accessible surface
DEFAULT_SPHERE_POINTS = 100     # Shrake-Rupley samples per atom
POTENTIAL_UNITS = "kT/e"        # APBS default output units


@dataclass
class ApbsSettings:
    """Everything that changes the numbers, kept in one place so it can be
    recorded verbatim as HDF5 attributes alongside the results."""

    forcefield: str = "PARSE"          # PARSE is the standard choice for continuum electrostatics
    ph: float | None = 7.0             # None -> skip titration-state assignment
    titration_method: str = "propka"   # "propka" or "none"
    protein_dielectric: float = 2.0
    solvent_dielectric: float = 78.54
    ionic_strength: float = 0.150      # mol/L of a symmetric 1:1 salt
    ion_radius: float = 2.0
    temperature: float = 298.15
    solute_radius: float = 1.4         # APBS srad
    surface_sdens: float = 10.0
    surface_window: float = 0.3        # APBS swin
    pbe_solver: str = "lpbe"           # "lpbe" or "npbe"
    coarse_factor: float = 1.7
    fine_padding: float = 20.0
    target_spacing: float = 0.5
    memory_ceiling_mb: float = 4000.0
    probe_radius: float = DEFAULT_PROBE_RADIUS
    sphere_points: int = DEFAULT_SPHERE_POINTS
    pdb2pqr_executable: str = "pdb2pqr"
    apbs_executable: str = "apbs"
    apbs_threads: int = 1

    def as_attributes(self) -> dict[str, object]:
        return {
            "forcefield": self.forcefield,
            "ph": -1.0 if self.ph is None else float(self.ph),
            "titration_method": self.titration_method,
            "protein_dielectric": float(self.protein_dielectric),
            "solvent_dielectric": float(self.solvent_dielectric),
            "ionic_strength_molar": float(self.ionic_strength),
            "ion_radius": float(self.ion_radius),
            "temperature_kelvin": float(self.temperature),
            "pbe_solver": self.pbe_solver,
            "probe_radius": float(self.probe_radius),
            "sphere_points": int(self.sphere_points),
            "potential_units": POTENTIAL_UNITS,
        }


@dataclass
class PqrStructure:
    """Atom table parsed from a pdb2pqr PQR file."""

    chain: np.ndarray       # (N,) <U4
    resnum: np.ndarray      # (N,) int32
    resname: np.ndarray     # (N,) <U8
    atom_name: np.ndarray   # (N,) <U8
    xyz: np.ndarray         # (N, 3) float64
    charge: np.ndarray      # (N,) float64, elementary charges
    radius: np.ndarray      # (N,) float64, Angstrom

    def __len__(self) -> int:
        return len(self.xyz)

    def atom_aa_id(self, residue_count: int) -> np.ndarray:
        """Each atom's aa_id, recovered from the prepared structure's renumbering.

        prepare_structure numbered residue i as i + 1, so this is a subtraction
        rather than a (chain, number) match -- which is the point, since that
        key is ambiguous in files whose insertion codes were dropped.
        """
        aa_id = self.resnum.astype(np.int32) - 1
        if aa_id.min() < 0 or aa_id.max() >= residue_count:
            raise ValueError(
                f"PQR residue numbers {aa_id.min() + 1}-{aa_id.max() + 1} fall outside "
                f"the {residue_count} residues of the prepared structure"
            )
        return aa_id


@dataclass
class SurfaceElectrostatics:
    """Per-model result: atom-level, residue-level, and (optionally) point-level."""

    model_id: str
    pdb_path: Path
    structure: PqrStructure
    prepared: PreparedStructure
    atom_aa_id: np.ndarray                # (N,) graph node id for each PQR atom
    atom_potential: np.ndarray            # (N,)
    # Residue arrays are indexed by aa_id and cover every residue of the
    # original PDB, so they align 1:1 with the graph nodes even when pdb2pqr
    # dropped a residue (those entries are NaN and residue_in_pqr is False).
    residue_chain: np.ndarray             # (R,)
    residue_number: np.ndarray            # (R,)
    residue_name: np.ndarray              # (R,)
    residue_in_pqr: np.ndarray            # (R,) bool
    residue_charge: np.ndarray            # (R,)
    residue_sasa: np.ndarray              # (R,) Angstrom^2
    residue_surface_point_count: np.ndarray   # (R,)
    residue_potential_mean: np.ndarray    # (R,)
    residue_potential_min: np.ndarray     # (R,)
    residue_potential_max: np.ndarray     # (R,)
    residue_potential_std: np.ndarray     # (R,)
    grid_origin: np.ndarray               # (3,)
    grid_spacing: np.ndarray              # (3,)
    grid_shape: tuple[int, int, int]
    grid_parameters: GridParameters
    surface_xyz: np.ndarray | None = None
    surface_potential: np.ndarray | None = None
    surface_atom_index: np.ndarray | None = None
    surface_residue_index: np.ndarray | None = None
    potential_grid: np.ndarray | None = None
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# pdb2pqr / APBS invocation
# --------------------------------------------------------------------------


def _run(command: list[str], cwd: Path, log_path: Path, timeout: float | None) -> None:
    """Run an external tool, tee-ing combined output to a log for diagnosis."""
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write(" ".join(command) + "\n\n")
        log_handle.flush()
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    if completed.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]
        raise RuntimeError(
            f"{Path(command[0]).name} failed (exit {completed.returncode}):\n  "
            + "\n  ".join(tail)
        )


def run_pdb2pqr(
    pdb_path: Path,
    pqr_path: Path,
    settings: ApbsSettings,
    work_dir: Path,
    timeout: float | None = None,
) -> None:
    """Assign forcefield charges/radii. --keep-chain is essential: without it
    pdb2pqr strips chain IDs, and chain is how results map back to graph nodes."""
    command = [
        settings.pdb2pqr_executable,
        f"--ff={settings.forcefield}",
        "--keep-chain",
        "--drop-water",
        str(pdb_path),
        str(pqr_path),
    ]
    if settings.ph is not None and settings.titration_method.lower() != "none":
        command[3:3] = [
            f"--with-ph={settings.ph}",
            f"--titration-state-method={settings.titration_method}",
        ]
    _run(command, work_dir, work_dir / "pdb2pqr.log", timeout)
    if not pqr_path.is_file() or pqr_path.stat().st_size == 0:
        raise RuntimeError(f"pdb2pqr produced no PQR at {pqr_path}")


# x, y, z, charge, radius -- matched individually so a value that runs into its
# neighbour still splits correctly (see parse_pqr).
_PQR_NUMBER_PATTERN = re.compile(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?")


def parse_pqr(pqr_path: str | Path) -> PqrStructure:
    """Read a PQR atom table.

    PQR is nominally whitespace-delimited, but pdb2pqr writes fixed-width
    columns, so a coordinate wide enough to fill its field runs straight into
    the next one: `60.423-177.683` is two values, not one. Splitting on
    whitespace silently mis-parses those lines -- and docked poses translated
    far from the origin hit this routinely. The numeric tail is therefore
    matched by pattern, which separates `60.423` and `-177.683` correctly, with
    a whitespace split kept as a fallback for PQRs from other tools.
    """
    chains: list[str] = []
    resnums: list[int] = []
    resnames: list[str] = []
    atom_names: list[str] = []
    coordinates: list[tuple[float, float, float]] = []
    charges: list[float] = []
    radii: list[float] = []

    with Path(pqr_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue

            # Identity fields sit in fixed PDB columns and never collide;
            # everything from the residue number onward is matched by pattern.
            numbers = _PQR_NUMBER_PATTERN.findall(line[22:])
            if len(numbers) == 6:
                residue_number = int(float(numbers[0]))
                x, y, z, charge, radius = (float(value) for value in numbers[1:6])
                chain = line[21].strip()
                resname = line[17:21].strip()
                atom_name = line[12:16].strip()
            else:
                # Fallback for a genuinely whitespace-delimited PQR.
                fields = line.split()
                if len(fields) < 9:
                    raise ValueError(f"Unparseable PQR line in {pqr_path}: {line!r}")
                radius = float(fields[-1])
                charge = float(fields[-2])
                x, y, z = (float(value) for value in fields[-5:-2])
                residue_number = int(fields[-6])
                chain = fields[4] if len(fields) - 6 >= 5 else ""
                resname = fields[3]
                atom_name = fields[2]

            chains.append(chain)
            resnums.append(residue_number)
            resnames.append(resname)
            atom_names.append(atom_name)
            coordinates.append((x, y, z))
            charges.append(charge)
            radii.append(radius)

    if not coordinates:
        raise ValueError(f"No ATOM records parsed from {pqr_path}")

    return PqrStructure(
        chain=np.array(chains, dtype="<U4"),
        resnum=np.array(resnums, dtype=np.int32),
        resname=np.array(resnames, dtype="<U8"),
        atom_name=np.array(atom_names, dtype="<U8"),
        xyz=np.array(coordinates, dtype=np.float64),
        charge=np.array(charges, dtype=np.float64),
        radius=np.array(radii, dtype=np.float64),
    )


def write_pqr(structure: PqrStructure, pqr_path: str | Path) -> Path:
    """Re-emit a PQR with fields that cannot run together.

    APBS's own reader is whitespace-delimited and rejects the file outright
    ("make sure there are no concatenated fields") when pdb2pqr's fixed-width
    columns collide -- which happens for any structure sitting far from the
    origin, so docked poses hit it routinely. Writing wide, explicitly padded
    numeric fields removes the failure mode entirely.
    """
    pqr_path = Path(pqr_path)
    lines = []
    for index in range(len(structure)):
        chain = structure.chain[index] or " "
        lines.append(
            f"ATOM  {index + 1:5d} {structure.atom_name[index]:<4s} "
            f"{structure.resname[index]:<3s} {chain:1s} {structure.resnum[index]:5d} "
            f"{structure.xyz[index, 0]:12.4f} {structure.xyz[index, 1]:12.4f} "
            f"{structure.xyz[index, 2]:12.4f} {structure.charge[index]:9.4f} "
            f"{structure.radius[index]:8.4f}"
        )
    pqr_path.write_text("\n".join(lines) + "\nTER\nEND\n", encoding="utf-8")
    return pqr_path


def write_apbs_input(
    input_path: Path,
    pqr_name: str,
    grid: GridParameters,
    settings: ApbsSettings,
    potential_stem: str = "potential",
) -> None:
    """Emit an APBS mg-auto (focusing) input deck for a single molecule."""
    ion_lines = ""
    if settings.ionic_strength > 0:
        ion_lines = (
            f"    ion charge  1 conc {settings.ionic_strength:.4f} radius {settings.ion_radius:.2f}\n"
            f"    ion charge -1 conc {settings.ionic_strength:.4f} radius {settings.ion_radius:.2f}\n"
        )

    input_path.write_text(
        f"""read
    mol pqr {pqr_name}
end
elec name solvated
    mg-auto
    dime   {grid.dime[0]} {grid.dime[1]} {grid.dime[2]}
    cglen  {grid.cglen[0]:.4f} {grid.cglen[1]:.4f} {grid.cglen[2]:.4f}
    fglen  {grid.fglen[0]:.4f} {grid.fglen[1]:.4f} {grid.fglen[2]:.4f}
    cgcent mol 1
    fgcent mol 1
    mol 1
    {settings.pbe_solver}
    bcfl sdh
    pdie {settings.protein_dielectric}
    sdie {settings.solvent_dielectric}
    srfm smol
    chgm spl2
    sdens {settings.surface_sdens}
    srad {settings.solute_radius}
    swin {settings.surface_window}
    temp {settings.temperature}
{ion_lines}    calcenergy no
    calcforce no
    write pot dx {potential_stem}
end
quit
""",
        encoding="utf-8",
    )


def run_apbs(
    input_path: Path,
    work_dir: Path,
    settings: ApbsSettings,
    potential_stem: str = "potential",
    timeout: float | None = None,
) -> Path:
    _run([settings.apbs_executable, input_path.name], work_dir, work_dir / "apbs.log", timeout)
    dx_path = work_dir / f"{potential_stem}.dx"
    if not dx_path.is_file():
        raise RuntimeError(f"APBS finished but wrote no potential map at {dx_path}")
    return dx_path


# --------------------------------------------------------------------------
# Solvent-accessible surface sampling
# --------------------------------------------------------------------------


def fibonacci_sphere(count: int) -> np.ndarray:
    """`count` near-uniformly spaced unit vectors (golden-spiral construction)."""
    indices = np.arange(count, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * indices / count
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    azimuth = np.pi * (1.0 + 5.0**0.5) * indices
    return np.stack([radius * np.cos(azimuth), radius * np.sin(azimuth), z], axis=1)


def solvent_accessible_points(
    coords: np.ndarray,
    radii: np.ndarray,
    probe_radius: float = DEFAULT_PROBE_RADIUS,
    sphere_points: int = DEFAULT_SPHERE_POINTS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shrake-Rupley surface sampling.

    Returns (points (P, 3), owning atom index (P,), per-atom SASA (N,)).
    A test point on atom i survives if it lies outside every other atom's
    probe-expanded sphere; the surviving fraction times 4*pi*R_i^2 is atom i's
    solvent-accessible area.
    """
    from scipy.spatial import cKDTree

    coords = np.asarray(coords, dtype=np.float64)
    expanded = np.asarray(radii, dtype=np.float64) + probe_radius
    unit_sphere = fibonacci_sphere(sphere_points)
    tree = cKDTree(coords)
    search_radius = float(expanded.max()) + float(expanded.max())

    kept_points: list[np.ndarray] = []
    kept_owner: list[np.ndarray] = []
    atom_sasa = np.zeros(len(coords), dtype=np.float64)

    neighbour_lists = tree.query_ball_point(coords, r=search_radius)
    for atom_index, neighbours in enumerate(neighbour_lists):
        candidates = coords[atom_index] + expanded[atom_index] * unit_sphere
        others = np.asarray([n for n in neighbours if n != atom_index], dtype=int)
        if others.size:
            # Buried where the point falls inside any neighbour's expanded sphere.
            squared_distance = ((candidates[:, None, :] - coords[others][None, :, :]) ** 2).sum(axis=2)
            exposed = np.all(squared_distance >= expanded[others][None, :] ** 2, axis=1)
        else:
            exposed = np.ones(len(candidates), dtype=bool)

        exposed_count = int(exposed.sum())
        if exposed_count:
            kept_points.append(candidates[exposed])
            kept_owner.append(np.full(exposed_count, atom_index, dtype=np.int32))
        atom_sasa[atom_index] = (
            4.0 * np.pi * expanded[atom_index] ** 2 * exposed_count / sphere_points
        )

    if not kept_points:
        return np.empty((0, 3)), np.empty(0, dtype=np.int32), atom_sasa
    return np.concatenate(kept_points), np.concatenate(kept_owner), atom_sasa


# --------------------------------------------------------------------------
# Per-residue aggregation
# --------------------------------------------------------------------------


def _grouped_statistics(
    values: np.ndarray, group_index: np.ndarray, group_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """mean/min/max/std/count per group; empty groups get NaN statistics."""
    counts = np.bincount(group_index, minlength=group_count).astype(np.int64)
    totals = np.bincount(group_index, weights=values, minlength=group_count)
    squares = np.bincount(group_index, weights=values**2, minlength=group_count)

    populated = counts > 0
    means = np.full(group_count, np.nan)
    stds = np.full(group_count, np.nan)
    means[populated] = totals[populated] / counts[populated]
    variance = np.maximum(0.0, squares[populated] / counts[populated] - means[populated] ** 2)
    stds[populated] = np.sqrt(variance)

    # Seed with +/-inf (not NaN) because np.minimum/np.maximum propagate NaN,
    # which would swallow every real value.
    minima = np.full(group_count, np.inf)
    maxima = np.full(group_count, -np.inf)
    np.minimum.at(minima, group_index, values)
    np.maximum.at(maxima, group_index, values)
    minima = np.where(populated, minima, np.nan)
    maxima = np.where(populated, maxima, np.nan)

    return means, minima, maxima, stds, counts


def aggregate_by_residue(
    structure: PqrStructure,
    atom_aa_id: np.ndarray,
    residue_count: int,
    surface_atom_index: np.ndarray,
    surface_potential: np.ndarray,
    atom_sasa: np.ndarray,
) -> dict[str, np.ndarray]:
    """Roll atom/point quantities up to residues.

    Surface points are near-equal-area samples, so the plain mean over a
    residue's points is already an area-weighted mean surface potential.
    """
    surface_residue_index = (
        atom_aa_id[surface_atom_index]
        if surface_atom_index.size
        else np.empty(0, dtype=np.int32)
    )
    means, minima, maxima, stds, counts = _grouped_statistics(
        surface_potential.astype(np.float64), surface_residue_index, residue_count
    )
    return {
        "surface_residue_index": surface_residue_index.astype(np.int32),
        "residue_charge": np.bincount(
            atom_aa_id, weights=structure.charge, minlength=residue_count
        ),
        "residue_sasa": np.bincount(
            atom_aa_id, weights=atom_sasa, minlength=residue_count
        ),
        "residue_in_pqr": np.bincount(atom_aa_id, minlength=residue_count) > 0,
        "residue_surface_point_count": counts.astype(np.int32),
        "residue_potential_mean": means,
        "residue_potential_min": minima,
        "residue_potential_max": maxima,
        "residue_potential_std": stds,
    }


def compute_surface_electrostatics(
    pdb_path: str | Path,
    model_id: str,
    settings: ApbsSettings | None = None,
    scratch_dir: str | Path | None = None,
    keep_intermediates_dir: str | Path | None = None,
    keep_grid: bool = False,
    keep_surface_points: bool = False,
    timeout: float | None = None,
) -> SurfaceElectrostatics:
    """Full pdb2pqr -> APBS -> surface-projection pipeline for one structure.

    All intermediates live in a scratch directory that is deleted on exit
    unless keep_intermediates_dir is given (which copies the PQR, APBS input,
    logs, and DX map there for inspection).
    """
    settings = settings or ApbsSettings()
    pdb_path = Path(pdb_path).resolve()
    if not pdb_path.is_file():
        raise FileNotFoundError(f"No PDB at {pdb_path}")

    warnings: list[str] = []
    scratch_parent = Path(scratch_dir) if scratch_dir else None
    if scratch_parent is not None:
        scratch_parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"apbs_{model_id}_", dir=scratch_parent) as temporary:
        work_dir = Path(temporary)
        prepared_path = work_dir / "prepared.pdb"
        prepared = prepare_structure(pdb_path, prepared_path)
        pqr_path = work_dir / "structure.pqr"
        try:
            run_pdb2pqr(prepared.pdb_path, pqr_path, settings, work_dir, timeout=timeout)
        except RuntimeError as exc:
            # pdb2pqr normally rebuilds unresolved side chains correctly, so
            # the full structure is always tried first; on the structures where
            # that rebuild crashes, retry with those side chains cut back.
            if not prepared.has_incomplete_residues:
                raise
            detail = [line.strip() for line in str(exc).splitlines() if line.strip()][-1]
            warnings.append(
                f"pdb2pqr failed on the complete structure ({detail[:160]}); "
                "retried with incomplete side chains truncated"
            )
            prepared = prepare_structure(pdb_path, prepared_path, truncate_incomplete=True)
            run_pdb2pqr(prepared.pdb_path, pqr_path, settings, work_dir, timeout=timeout)
        warnings.extend(prepared.warnings)
        structure = parse_pqr(pqr_path)

        # APBS reads this normalised copy, never pdb2pqr's own fixed-width file.
        apbs_pqr_path = write_pqr(structure, work_dir / "apbs_input.pqr")

        grid = compute_grid_parameters(
            structure.xyz,
            structure.radius,
            coarse_factor=settings.coarse_factor,
            fine_padding=settings.fine_padding,
            target_spacing=settings.target_spacing,
            memory_ceiling_mb=settings.memory_ceiling_mb,
        )
        input_path = work_dir / "apbs.in"
        write_apbs_input(input_path, apbs_pqr_path.name, grid, settings)
        dx_path = run_apbs(input_path, work_dir, settings, timeout=timeout)
        potential: DxGrid = read_dx(dx_path)

        if keep_intermediates_dir is not None:
            destination = Path(keep_intermediates_dir)
            destination.mkdir(parents=True, exist_ok=True)
            for name, target_name in (
                ("structure.pqr", f"{model_id}.pqr"),
                ("apbs_input.pqr", f"{model_id}_apbs_input.pqr"),
                ("prepared.pdb", f"{model_id}_prepared.pdb"),
                ("apbs.in", f"{model_id}.in"),
                ("pdb2pqr.log", f"{model_id}_pdb2pqr.log"),
                ("apbs.log", f"{model_id}_apbs.log"),
            ):
                source = work_dir / name
                if source.is_file():
                    shutil.copy2(source, destination / target_name)
            # The OpenDX text map is ~40x the float32 array it encodes, so it
            # is always gzipped on the way out (read_dx handles .gz directly).
            with dx_path.open("rb") as source, gzip.open(
                destination / f"{model_id}_potential.dx.gz", "wb", compresslevel=5
            ) as archive:
                shutil.copyfileobj(source, archive)

    surface_xyz, surface_atom_index, atom_sasa = solvent_accessible_points(
        structure.xyz,
        structure.radius,
        probe_radius=settings.probe_radius,
        sphere_points=settings.sphere_points,
    )
    surface_potential = (
        potential.sample(surface_xyz) if surface_xyz.size else np.empty(0, dtype=np.float32)
    )
    atom_potential = potential.sample(structure.xyz)

    residue_count = len(prepared)
    atom_aa_id = structure.atom_aa_id(residue_count)
    aggregates = aggregate_by_residue(
        structure,
        atom_aa_id,
        residue_count,
        surface_atom_index,
        surface_potential,
        atom_sasa,
    )

    # Residues pdb2pqr dropped entirely carry no charge or potential; make that
    # explicit rather than letting a zero read as a real, neutral result.
    absent = ~aggregates["residue_in_pqr"]
    if absent.any():
        warnings.append(f"{int(absent.sum())}/{residue_count} residues absent from the PQR")
        for key in ("residue_charge", "residue_sasa"):
            aggregates[key] = np.where(absent, np.nan, aggregates[key])
    buried = int(np.sum((aggregates["residue_surface_point_count"] == 0) & ~absent))
    if buried:
        warnings.append(f"{buried}/{residue_count} residues are fully buried (no surface points)")

    return SurfaceElectrostatics(
        model_id=model_id,
        pdb_path=pdb_path,
        structure=structure,
        prepared=prepared,
        atom_aa_id=atom_aa_id,
        atom_potential=atom_potential,
        residue_chain=prepared.chain,
        residue_number=prepared.number,
        residue_name=prepared.name,
        residue_in_pqr=aggregates["residue_in_pqr"],
        residue_charge=aggregates["residue_charge"],
        residue_sasa=aggregates["residue_sasa"],
        residue_surface_point_count=aggregates["residue_surface_point_count"],
        residue_potential_mean=aggregates["residue_potential_mean"],
        residue_potential_min=aggregates["residue_potential_min"],
        residue_potential_max=aggregates["residue_potential_max"],
        residue_potential_std=aggregates["residue_potential_std"],
        grid_origin=potential.origin,
        grid_spacing=potential.spacing,
        grid_shape=potential.shape,
        grid_parameters=grid,
        surface_xyz=surface_xyz.astype(np.float32) if keep_surface_points else None,
        surface_potential=surface_potential if keep_surface_points else None,
        surface_atom_index=surface_atom_index if keep_surface_points else None,
        surface_residue_index=aggregates["surface_residue_index"] if keep_surface_points else None,
        potential_grid=potential.values if keep_grid else None,
        warnings=warnings,
    )
