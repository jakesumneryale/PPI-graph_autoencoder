"""Screened electrostatic interaction as two monomers approach their bound pose.

One chain is held fixed (B) and the other (A) is walked in from a long-range
starting separation to the crystallographic bound state, with the screened
Coulomb interaction evaluated at every step.

Model
-----
Each atom carries the PARSE partial charge APBS was given. The potential that
the stationary chain B creates at a point r is the Debye-Huckel (screened
Coulomb) form

    phi_B(r) = (1 / (4 pi eps0 eps_r)) * sum_j  q_j * exp(-|r - r_j| / lambda_D) / |r - r_j|

and the interaction energy is that potential contracted with A's charges

    U(d) = sum_{i in A} q_i * phi_B(r_i(d))

This is symmetric -- contracting phi_A with B's charges gives the same number --
so "of B on A" fixes only which chain the per-residue decomposition is
attributed to, not the total. Both decompositions are written out.

Why this model rather than a Poisson-Boltzmann solve at every step: PB would
need ~20 s per step per structure (roughly 500 CPU-hours over 84 targets x 101
steps), and the quantity asked for -- a screened interaction with a biological
Debye length -- is exactly what Debye-Huckel provides in closed form. Its
limitations are real and stated in METHODS.md: uniform solvent dielectric with
no low-dielectric protein interior, no salt exclusion from the protein volume,
and no desolvation. It is at its best where most of the trajectory lives (well
separated chains) and degrades at contact, which is where the APBS maps in the
rest of this package are the better tool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist


# --- physical constants (SI, then the practical combination) -----------------
ELEMENTARY_CHARGE = 1.602176634e-19      # C
VACUUM_PERMITTIVITY = 8.8541878128e-12   # F/m
BOLTZMANN = 1.380649e-23                 # J/K
AVOGADRO = 6.02214076e23                 # /mol
JOULES_PER_KCAL = 4184.0

# e^2 / (4 pi eps0), expressed so that U[kcal/mol] = COULOMB_CONSTANT * q_i q_j / (eps_r * r[A])
COULOMB_CONSTANT = 332.0637              # kcal.A/(mol.e^2)

WATER_DIELECTRIC = 78.54                 # matches the APBS runs (sdie)
CYTOSOLIC_IONIC_STRENGTH = 0.150         # mol/L; see docstring of debye_length


def kt_in_kcal_per_mol(temperature: float = 298.15) -> float:
    return BOLTZMANN * temperature * AVOGADRO / JOULES_PER_KCAL


def debye_length(
    ionic_strength: float = CYTOSOLIC_IONIC_STRENGTH,
    temperature: float = 298.15,
    dielectric: float = WATER_DIELECTRIC,
) -> float:
    """Debye screening length in Angstrom.

        lambda_D = sqrt(eps0 * eps_r * kB * T / (2 * I * e^2))

    The default 0.150 M is the standard figure for cytosolic ionic strength
    (K+ ~140 mM with Na+, Cl-, Mg2+, phosphates and charged metabolites making
    up the rest) and is the same value the APBS runs used, so the two parts of
    this package screen identically. It gives lambda_D = 7.86 A at 298.15 K.

    The result is insensitive to the temperature choice -- 310 K with water's
    dielectric at 37 C gives 7.78 A -- so 25 C is kept for consistency with the
    APBS work rather than switched to body temperature for its own sake.
    """
    ions_per_m3 = ionic_strength * 1000.0 * AVOGADRO
    length_m = np.sqrt(
        VACUUM_PERMITTIVITY * dielectric * BOLTZMANN * temperature
        / (2.0 * ions_per_m3 * ELEMENTARY_CHARGE**2)
    )
    return float(length_m * 1e10)


@dataclass
class ScreeningModel:
    ionic_strength: float = CYTOSOLIC_IONIC_STRENGTH
    temperature: float = 298.15
    dielectric: float = WATER_DIELECTRIC

    @property
    def debye_length(self) -> float:
        return debye_length(self.ionic_strength, self.temperature, self.dielectric)

    @property
    def kt_kcal(self) -> float:
        return kt_in_kcal_per_mol(self.temperature)

    def as_attributes(self) -> dict[str, float]:
        return {
            "ionic_strength_molar": self.ionic_strength,
            "temperature_kelvin": self.temperature,
            "dielectric": self.dielectric,
            "debye_length_angstrom": self.debye_length,
            "kt_kcal_per_mol": self.kt_kcal,
        }


def screened_potential(
    points: np.ndarray,
    source_xyz: np.ndarray,
    source_charge: np.ndarray,
    model: ScreeningModel,
    screened: bool = True,
    chunk: int = 1024,
) -> np.ndarray:
    """Potential at `points` from the charges at `source_xyz`, in kcal/(mol.e).

    Evaluated exactly over all pairs rather than with a distance cutoff: cdist
    makes the full sum as fast as a 60 A cutoff, and a cutoff biases the total
    systematically (~0.2-0.3% low at 40 A) because every truncated term has the
    same sign as its charge product.
    """
    points = np.asarray(points, dtype=np.float64)
    source_xyz = np.asarray(source_xyz, dtype=np.float64)
    source_charge = np.asarray(source_charge, dtype=np.float64)
    lam = model.debye_length

    potential = np.empty(len(points), dtype=np.float64)
    for start in range(0, len(points), chunk):
        distance = cdist(points[start : start + chunk], source_xyz)
        np.maximum(distance, 1e-6, out=distance)  # guard coincident atoms
        weight = source_charge / distance
        if screened:
            weight *= np.exp(-distance / lam)
        potential[start : start + chunk] = weight.sum(axis=1)
    return potential * (COULOMB_CONSTANT / model.dielectric)


def pair_potentials(
    moving_xyz: np.ndarray,
    moving_charge: np.ndarray,
    fixed_xyz: np.ndarray,
    fixed_charge: np.ndarray,
    model: ScreeningModel,
    chunk: int = 1024,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Everything the trajectory needs from one pass over the atom pairs.

    Returns, all in kcal/(mol.e):
      phi_on_moving   -- screened potential from the fixed chain at each moving atom
      phi_on_fixed    -- screened potential from the moving chain at each fixed atom
      phi_on_moving_unscreened -- the same as the first, with the Debye factor dropped

    Computing them together matters: the distance matrix is the expensive part,
    and evaluating the three separately triples the cost of the whole sweep.
    The two screened potentials contract to the *same* total energy; they are
    two decompositions of it, one per chain.
    """
    moving_xyz = np.asarray(moving_xyz, dtype=np.float64)
    fixed_xyz = np.asarray(fixed_xyz, dtype=np.float64)
    lam = model.debye_length
    scale = COULOMB_CONSTANT / model.dielectric

    phi_moving = np.empty(len(moving_xyz))
    phi_moving_bare = np.empty(len(moving_xyz))
    phi_fixed = np.zeros(len(fixed_xyz))

    for start in range(0, len(moving_xyz), chunk):
        stop = min(start + chunk, len(moving_xyz))
        distance = cdist(moving_xyz[start:stop], fixed_xyz)
        np.maximum(distance, 1e-6, out=distance)
        inverse = 1.0 / distance
        damped = np.exp(-distance / lam) * inverse

        phi_moving[start:stop] = damped @ fixed_charge
        phi_moving_bare[start:stop] = inverse @ fixed_charge
        phi_fixed += moving_charge[start:stop] @ damped

    return phi_moving * scale, phi_fixed * scale, phi_moving_bare * scale


def heavy_atom_mask(atom_names: np.ndarray) -> np.ndarray:
    """True for non-hydrogen atoms, by PDB atom-name convention.

    Geometry -- clash detection and path finding -- uses heavy atoms only.
    Hydrogen positions are model-built (reduce, then rebuilt by pdb2pqr) and
    routinely sit 1.2-1.4 A from a partner atom, which is a normal hydrogen
    bond, not a clash. Judging paths on all-atom distances therefore flags
    every native interface as clashing and makes the criterion useless.
    Electrostatics keeps every atom, because every atom carries charge.
    """
    return np.array([
        not (name[:1] == "H" or (len(name) > 1 and name[0].isdigit() and name[1] == "H"))
        for name in atom_names
    ], dtype=bool)


# --- path construction -------------------------------------------------------


@dataclass
class ApproachPath:
    """The rigid-body trajectory of the moving chain, bound pose last.

    `offset[k]` is the displacement of the moving chain from its bound
    position at frame k, as a full 3-vector -- the path is not required to be
    straight. `offset[-1]` is exactly zero, so the last frame is the
    crystallographic complex by construction.
    """

    offset: np.ndarray                # (S, 3) displacement of the centroid from the bound pose
    rotation: np.ndarray              # (S, 3, 3) rotation about the bound centroid; identity at the end
    centroid: np.ndarray              # (3,) bound-pose centroid the rotation is applied about
    com_displacement: np.ndarray      # (S,) |offset|
    path_length: np.ndarray           # (S,) arc length travelled from the bound pose
    min_gap: np.ndarray               # (S,) closest atom-atom approach at each frame
    axis: np.ndarray                  # (3,) net escape direction
    mode: str                         # "linear" or "steered"
    axis_angle_from_com: float
    native_gap: float
    worst_gap: float                  # smallest clearance anywhere on the path
    clash_free: bool
    notes: list[str] = field(default_factory=list)

    def position(self, index: int, bound_xyz: np.ndarray) -> np.ndarray:
        """Place `bound_xyz` at frame `index`: rotate about the bound centroid, then translate."""
        return (bound_xyz - self.centroid) @ self.rotation[index].T + self.centroid + self.offset[index]

    @property
    def max_rotation_degrees(self) -> float:
        traces = np.clip((np.trace(self.rotation, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
        return float(np.degrees(np.arccos(traces)).max())


def _fibonacci_directions(count: int) -> np.ndarray:
    indices = np.arange(count, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * indices / count
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    azimuth = np.pi * (1.0 + 5.0**0.5) * indices
    return np.stack([radius * np.cos(azimuth), radius * np.sin(azimuth), z], axis=1)


def _min_separation(tree_fixed: cKDTree, moving_xyz: np.ndarray) -> float:
    distance, _ = tree_fixed.query(moving_xyz, k=1)
    return float(distance.min())


def _rotation_about(axis: np.ndarray, degrees: float) -> np.ndarray:
    """Rodrigues rotation matrix."""
    axis = axis / np.linalg.norm(axis)
    theta = np.radians(degrees)
    cross = np.array([[0.0, -axis[2], axis[1]],
                      [axis[2], 0.0, -axis[0]],
                      [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(theta) * cross + (1.0 - np.cos(theta)) * (cross @ cross)


def _rotation_candidates(degrees: float) -> list[np.ndarray]:
    """Identity plus small rotations about each Cartesian axis, both senses."""
    candidates = [np.eye(3)]
    for axis in np.eye(3):
        candidates.append(_rotation_about(axis, degrees))
        candidates.append(_rotation_about(axis, -degrees))
    return candidates


def choose_separation_axis(
    fixed_xyz: np.ndarray,
    moving_xyz: np.ndarray,
    max_displacement: float,
    candidate_count: int = 256,
    probe_step: float = 2.0,
    subsample: int = 4,
    clash_floor: float = 2.6,
) -> tuple[np.ndarray, float, float, str]:
    """Pick the straight-line pull-out direction, scored on clashes then reach.

    Directions are ranked lexicographically: first by how much they clash
    (total depth below the bound state's own closest contact, so the native
    interface is not itself counted as a clash), then by the separation
    actually reached. A direction that scrapes is always worse than one that
    does not, however far it eventually gets.

    Returns (axis, clash_penalty, final_gap, description).
    """
    fixed_probe = fixed_xyz[::subsample]
    moving_probe = moving_xyz[::subsample]
    tree = cKDTree(fixed_probe)

    com_axis = moving_xyz.mean(axis=0) - fixed_xyz.mean(axis=0)
    com_axis /= np.linalg.norm(com_axis)
    native_gap = _min_separation(tree, moving_probe)
    floor = min(native_gap, clash_floor) - 0.05

    # Directions pointing back into the partner are excluded; everything else
    # is fair game, because the escape route out of a groove can be nearly
    # perpendicular to the line joining the centroids.
    candidates = [com_axis] + [
        d for d in _fibonacci_directions(candidate_count) if float(d @ com_axis) > -0.3
    ]
    steps = np.arange(probe_step, max_displacement + probe_step, probe_step)

    def score(direction, probe_tree, probe_points, probe_steps):
        gaps = np.array([_min_separation(probe_tree, probe_points + d * direction) for d in probe_steps])
        return float(np.maximum(0.0, floor - gaps).sum()), float(gaps[-1])

    # Coarse pass over every candidate, then rescore a shortlist against the
    # full atom set. Subsampling for the coarse pass is what makes 256
    # directions affordable, but it skips atoms and so under-reports contacts;
    # deciding on the subsampled score alone picks axes that clash in reality.
    coarse = [(*score(d, tree, moving_probe, steps), d) for d in candidates]
    shortlist = [row[2] for row in sorted(coarse, key=lambda r: (r[0], -r[1]))[:12]]
    if not any(np.allclose(d, com_axis) for d in shortlist):
        shortlist.insert(0, com_axis)

    fine_tree = cKDTree(fixed_xyz)
    fine_steps = np.arange(1.0, max_displacement + 1.0, 1.0)
    fine = [(*score(d, fine_tree, moving_xyz, fine_steps), d) for d in shortlist]
    com_penalty, com_reach = score(com_axis, fine_tree, moving_xyz, fine_steps)
    best_penalty, best_reach, best_axis = min(fine, key=lambda row: (row[0], -row[1]))

    # Keep the COM axis when it is also clash-free and nearly as far-reaching;
    # it is the more interpretable choice and the scoring difference is noise.
    if com_penalty <= 1e-9 and com_reach >= best_reach - 3.0:
        return com_axis, com_penalty, com_reach, "centre-of-mass vector"

    angle = float(np.degrees(np.arccos(np.clip(float(best_axis @ com_axis), -1.0, 1.0))))
    return (
        best_axis, best_penalty, best_reach,
        f"clearance-optimised, {angle:.0f} deg off the COM axis",
    )


def build_path(
    fixed_xyz: np.ndarray,
    moving_xyz: np.ndarray,
    max_displacement: float = 50.0,
    step: float = 0.5,
    clash_tolerance: float = 0.05,
    clash_floor: float = 2.6,
    linear_tolerance: float = 0.5,
    rotation_step_degrees: float = 4.0,
    clearance_target: float = 12.0,
    cone_degrees: float = 85.0,
    cone_directions: int = 128,
    subsample: int = 2,
) -> ApproachPath:
    """Build the unbinding path, then reverse it into an approach.

    A straight pull-out along the best axis is tried first and kept whenever it
    is clash-free -- that is the simplest, most interpretable trajectory and it
    works for most dimers.

    When no straight line works (for some complexes *no* rigid translation
    separates the chains -- 1kfu has zero clash-free directions out of 167
    tested, because the small subunit is nestled against an extended partner),
    the path is steered instead, in two phases:

      escape  -- while the chains are still close, step in whichever direction
                 within a cone about the escape axis opens the largest gap.
                 This lets the chain slide out along a groove rather than
                 straight through its wall.
      recede  -- once clear of the partner, run straight along the escape axis
                 until the required displacement is reached.

    Generating outward from the bound pose and reversing guarantees the final
    frame is the crystal structure exactly, not approximately.
    """
    tree_probe = cKDTree(fixed_xyz[::subsample])
    tree_full = cKDTree(fixed_xyz)
    moving_probe = moving_xyz[::subsample]

    com_axis = moving_xyz.mean(axis=0) - fixed_xyz.mean(axis=0)
    com_axis /= np.linalg.norm(com_axis)
    axis, penalty, _reach, axis_source = choose_separation_axis(
        fixed_xyz, moving_xyz, max_displacement, subsample=subsample
    )
    axis_angle = float(np.degrees(np.arccos(np.clip(float(axis @ com_axis), -1.0, 1.0))))
    native_gap = _min_separation(tree_full, moving_xyz)
    # An absolute heavy-atom floor, not a purely relative one: a native contact
    # that is already tight should not licence an equally tight contact
    # somewhere it does not belong.
    floor = min(native_gap, clash_floor) - clash_tolerance

    notes: list[str] = []
    step_count = int(round(max_displacement / step))
    centroid = moving_xyz.mean(axis=0)

    def place(rotation, translation):
        return (moving_xyz - centroid) @ rotation.T + centroid + translation

    def place_probe(rotation, translation):
        return (moving_probe - centroid) @ rotation.T + centroid + translation

    if penalty <= linear_tolerance:
        offsets = np.array([k * step * axis for k in range(step_count + 1)])
        rotations = np.repeat(np.eye(3)[None], len(offsets), axis=0)
        mode = "linear"
    else:
        notes.append(
            f"no clash-free straight pull-out (best straight axis dips {penalty:.1f} A-steps "
            "below the clash floor); path steered with rotation"
        )
        cone = [axis] + [
            d for d in _fibonacci_directions(cone_directions)
            if float(d @ axis) > np.cos(np.radians(cone_degrees))
        ]
        turns = _rotation_candidates(rotation_step_degrees)

        offsets = [np.zeros(3)]
        rotations = [np.eye(3)]
        translation = np.zeros(3)
        rotation = np.eye(3)
        for _ in range(step_count):
            gap_now = _min_separation(tree_probe, place_probe(rotation, translation))
            if gap_now >= clearance_target:
                trial = translation + step * axis
                if _min_separation(tree_probe, place_probe(rotation, trial)) >= floor:
                    translation = trial
                    offsets.append(translation.copy()); rotations.append(rotation.copy())
                    continue

            # Best translation first, then a small rotation if still tight.
            # Rotation is what makes interdigitated pairs separable at all --
            # for some complexes no pure translation works, because the chains
            # have to unwind, not just slide.
            best_dir, best_gap = axis, -np.inf
            for direction in cone:
                gap = _min_separation(tree_probe, place_probe(rotation, translation + step * direction))
                if gap > best_gap:
                    best_gap, best_dir = gap, direction
            translation = translation + step * best_dir

            if best_gap < clearance_target:
                best_turn, turn_gap = np.eye(3), best_gap
                for turn in turns:
                    candidate = turn @ rotation
                    gap = _min_separation(tree_probe, place_probe(candidate, translation))
                    if gap > turn_gap:
                        turn_gap, best_turn = gap, turn
                rotation = best_turn @ rotation

            offsets.append(translation.copy()); rotations.append(rotation.copy())
        offsets = np.array(offsets); rotations = np.array(rotations)
        mode = "steered"

    # Extend straight out until the net centroid displacement is exactly what
    # was asked for; a steered path spends some of its length turning.
    final_displacement = float(np.linalg.norm(offsets[-1]))
    if mode == "steered" and final_displacement > 1e-6:
        extra = max_displacement - final_displacement
        if extra > step / 2:
            direction = offsets[-1] / final_displacement
            tail = offsets[-1] + np.outer(
                np.arange(1, int(round(extra / step)) + 1) * step, direction)
            offsets = np.vstack([offsets, tail])
            rotations = np.concatenate([rotations, np.repeat(rotations[-1][None], len(tail), axis=0)])

    gaps = np.array([_min_separation(tree_full, place(r, o)) for r, o in zip(rotations, offsets)])
    lengths = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(offsets, axis=0), axis=1))])
    worst = float(gaps[1:].min()) if len(gaps) > 1 else native_gap
    clash_free = bool(worst >= floor)
    if not clash_free:
        notes.append(
            f"closest approach en route is {worst:.2f} A against a native contact of "
            f"{native_gap:.2f} A -- the chains still graze on the way out"
        )

    order = slice(None, None, -1)   # reverse: far away first, bound pose last
    return ApproachPath(
        offset=offsets[order].copy(),
        rotation=rotations[order].copy(),
        centroid=centroid,
        com_displacement=np.linalg.norm(offsets, axis=1)[order].copy(),
        path_length=lengths[order].copy(),
        min_gap=gaps[order].copy(),
        axis=axis,
        mode=mode,
        axis_angle_from_com=axis_angle if axis_source != "centre-of-mass vector" else 0.0,
        native_gap=native_gap,
        worst_gap=worst,
        clash_free=clash_free,
        notes=notes,
    )


# --- per-pair analysis -------------------------------------------------------


@dataclass
class ChainData:
    """One chain's atoms and its residue bookkeeping."""

    label: str
    xyz: np.ndarray
    charge: np.ndarray
    atom_aa_id: np.ndarray        # index into this chain's own residue list
    residue_number: np.ndarray
    residue_name: np.ndarray
    residue_charge: np.ndarray
    parent_aa_id: np.ndarray      # aa_id in the whole complex, for joining back
    heavy: np.ndarray = None      # bool mask; geometry uses these, electrostatics uses all

    @property
    def heavy_xyz(self) -> np.ndarray:
        return self.xyz if self.heavy is None else self.xyz[self.heavy]

    @property
    def net_charge(self) -> float:
        return float(self.charge.sum())

    @property
    def centre(self) -> np.ndarray:
        return self.xyz.mean(axis=0)


def load_chain_pair(store_path: str | Path, complex_id: str) -> tuple[ChainData, ChainData]:
    """Split a solved complex into its two chains.

    Uses the complex's own PQR atoms, so the charges are exactly those APBS
    was given and are identical to the monomer stores'.
    """
    import h5py

    with h5py.File(Path(store_path).expanduser(), "r") as handle:
        group = handle[complex_id]
        atom_chain = group["atom_chain"].asstr()[:]
        atom_name = group["atom_name"].asstr()[:]
        atom_aa_id = group["atom_aa_id"][:]
        xyz = group["atom_xyz"][:].astype(np.float64)
        charge = group["atom_charge"][:].astype(np.float64)
        residue_chain = group["residue_chain"].asstr()[:]
        residue_number = group["residue_number"][:]
        residue_name = group["residue_name"].asstr()[:]
        residue_charge = group["residue_charge"][:]

    labels = sorted(set(atom_chain.tolist()))
    if len(labels) != 2:
        raise ValueError(f"{complex_id} has {len(labels)} chains ({labels}); expected a heterodimer")

    chains = []
    for label in labels:
        atom_mask = atom_chain == label
        residue_mask = residue_chain == label
        parent = np.flatnonzero(residue_mask).astype(np.int32)
        remap = np.full(len(residue_chain), -1, dtype=np.int32)
        remap[parent] = np.arange(len(parent), dtype=np.int32)
        chains.append(
            ChainData(
                label=label,
                xyz=xyz[atom_mask],
                charge=charge[atom_mask],
                atom_aa_id=remap[atom_aa_id[atom_mask]],
                residue_number=residue_number[residue_mask],
                residue_name=residue_name[residue_mask],
                residue_charge=residue_charge[residue_mask],
                parent_aa_id=parent,
                heavy=heavy_atom_mask(atom_name[atom_mask]),
            )
        )
    return chains[0], chains[1]


def analyse_pair(
    moving: ChainData,
    fixed: ChainData,
    model: ScreeningModel,
    max_displacement: float = 50.0,
    step: float = 0.5,
    residue_stride: int = 1,
):
    """Walk `moving` in to its bound pose against `fixed`, scoring every step.

    Returns (path, trajectory rows, residue rows).
    """
    import pandas as pd

    path = build_path(
        fixed.heavy_xyz, moving.heavy_xyz, max_displacement=max_displacement, step=step
    )
    kt = model.kt_kcal
    lam = model.debye_length

    fixed_centre = fixed.centre
    net_product = moving.net_charge * fixed.net_charge

    trajectory: list[dict] = []
    residue_rows: list[pd.DataFrame] = []

    for index in range(len(path.com_displacement)):
        moved = path.position(index, moving.xyz)

        potential, potential_on_fixed, unscreened = pair_potentials(
            moved, moving.charge, fixed.xyz, fixed.charge, model
        )
        atom_energy = moving.charge * potential
        fixed_atom_energy = fixed.charge * potential_on_fixed
        energy = float(atom_energy.sum())
        energy_unscreened = float(moving.charge @ unscreened)

        centre_distance = float(np.linalg.norm(moved.mean(axis=0) - fixed_centre))
        monopole = (
            COULOMB_CONSTANT * net_product * np.exp(-centre_distance / lam)
            / (model.dielectric * centre_distance)
        )
        trajectory.append({
            "step": index,
            "displacement_angstrom": float(path.com_displacement[index]),
            "path_length_angstrom": float(path.path_length[index]),
            "centre_distance_angstrom": centre_distance,
            "min_heavy_atom_gap_angstrom": float(path.min_gap[index]),
            # Frames where the chains overlap sterically. The Debye-Huckel sum
            # still returns a number there, but interpenetrating charge clouds
            # make it unphysical -- drop these rows before fitting anything.
            "steric_overlap": bool(path.min_gap[index] < 2.0),
            "rotation_degrees": float(np.degrees(np.arccos(np.clip(
                (np.trace(path.rotation[index]) - 1.0) / 2.0, -1.0, 1.0)))),
            "interaction_energy_kcal_per_mol": energy,
            "interaction_energy_kT": energy / kt,
            "interaction_energy_unscreened_kcal_per_mol": energy_unscreened,
            "monopole_energy_kcal_per_mol": float(monopole),
            "potential_at_moving_mean_kT_per_e": float(potential.mean() / kt),
            "potential_at_moving_min_kT_per_e": float(potential.min() / kt),
            "potential_at_moving_max_kT_per_e": float(potential.max() / kt),
        })

        if residue_stride and index % residue_stride == 0:
            for chain, other, tag in ((moving, fixed, "moving"), (fixed, moving, "fixed")):
                # Two decompositions of the same total, already in hand.
                per_atom = atom_energy if tag == "moving" else fixed_atom_energy
                contribution = np.bincount(
                    chain.atom_aa_id, weights=per_atom, minlength=len(chain.residue_number)
                )
                residue_rows.append(pd.DataFrame({
                    "step": index,
                    "displacement_angstrom": float(path.com_displacement[index]),
                    "chain": chain.label,
                    "chain_role": tag,
                    "residue_aa_id": chain.parent_aa_id,
                    "residue_number": chain.residue_number,
                    "residue_name": chain.residue_name,
                    "residue_charge": chain.residue_charge,
                    "contribution_kcal_per_mol": contribution,
                    "contribution_kT": contribution / kt,
                }))

    return path, pd.DataFrame(trajectory), (
        pd.concat(residue_rows, ignore_index=True) if residue_rows else None
    )
