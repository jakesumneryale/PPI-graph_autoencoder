"""APBS multigrid dimensioning, following the standard psize.py conventions.

APBS `mg-auto` needs three things per axis: the coarse box length (a padded
box carrying the Debye-Huckel boundary condition), the fine box length (the
region actually resolved), and `dime`, the number of grid points. `dime` is
constrained to c * 2^(nlev+1) + 1, i.e. 32c + 1 for the default nlev = 4.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# APBS allocates roughly 200 bytes per fine-grid point for a sequential run.
BYTES_PER_GRID_POINT = 200.0
NLEV = 4
DIME_STEP = 2 ** (NLEV + 1)  # 32
MIN_DIME = DIME_STEP + 1     # 33


@dataclass(frozen=True)
class GridParameters:
    dime: tuple[int, int, int]
    cglen: tuple[float, float, float]
    fglen: tuple[float, float, float]
    center: tuple[float, float, float]

    @property
    def fine_spacing(self) -> tuple[float, float, float]:
        return tuple(
            length / (points - 1) for length, points in zip(self.fglen, self.dime)
        )

    @property
    def memory_mb(self) -> float:
        return float(np.prod(self.dime)) * BYTES_PER_GRID_POINT / 1024**2


def _round_up_to_dime(points: np.ndarray) -> np.ndarray:
    """Round each axis up to the next valid c * 32 + 1 value."""
    coefficients = np.maximum(np.ceil((points - 1) / DIME_STEP), 1.0)
    return (DIME_STEP * coefficients + 1).astype(int)


def compute_grid_parameters(
    coords: np.ndarray,
    radii: np.ndarray,
    coarse_factor: float = 1.7,
    fine_padding: float = 20.0,
    target_spacing: float = 0.5,
    memory_ceiling_mb: float = 4000.0,
) -> GridParameters:
    """Size the APBS focusing grids for one molecule.

    coarse_factor/fine_padding/target_spacing default to the psize values that
    pdb2pqr uses. If the resulting grid would exceed memory_ceiling_mb, dime is
    stepped down uniformly (coarsening the fine mesh) until it fits, which is
    what keeps very large complexes from blowing up a job's memory request.
    """
    coords = np.asarray(coords, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64).reshape(-1, 1)
    if coords.size == 0:
        raise ValueError("No atoms to size a grid around")

    lower = (coords - radii).min(axis=0)
    upper = (coords + radii).max(axis=0)
    molecule_length = upper - lower
    center = (upper + lower) / 2.0

    cglen = coarse_factor * molecule_length
    # The fine box only needs the molecule plus a solvent shell, but it can
    # never usefully exceed the coarse box.
    fglen = np.minimum(molecule_length + fine_padding, cglen)

    dime = _round_up_to_dime(np.ceil(fglen / target_spacing) + 1)
    dime = np.maximum(dime, MIN_DIME)

    while float(np.prod(dime)) * BYTES_PER_GRID_POINT / 1024**2 > memory_ceiling_mb:
        if np.all(dime <= MIN_DIME):
            break
        # Coarsen the longest axis first so the mesh stays roughly isotropic.
        axis = int(np.argmax(np.where(dime > MIN_DIME, dime, -1)))
        dime[axis] = max(MIN_DIME, dime[axis] - DIME_STEP)

    return GridParameters(
        dime=tuple(int(value) for value in dime),
        cglen=tuple(float(value) for value in cglen),
        fglen=tuple(float(value) for value in fglen),
        center=tuple(float(value) for value in center),
    )
