"""Animate a water-sized probe rolling over one amino acid, for PyMOL.

The solvent-accessible surface *is* the locus of the probe centre when the
probe is in contact with the van der Waals surface, so the animation is exact
rather than illustrative: at every frame the probe touches the residue and
never overlaps it.

For each frame the probe centre p satisfies

    |p - a_i| >= r_i + 1.4   for every atom i,   with equality for the atom it
                                                 is currently resting on

Radii are the Bondi values PyMOL itself uses to draw spheres (H 1.20, C 1.70,
N 1.55, O 1.52), so the probe visually kisses the spheres on screen instead of
floating above them or sinking in.

The roll is a single sweep about one axis, taken as the residue's shortest
principal axis so the probe travels around the widest profile.

Writes into the output directory:
    asparagine.pdb          the residue, static
    probe_path.pdb          multi-state, one MODEL per frame, probe centre only
    probe_with_marker.pdb   same, plus a dot on the probe that rotates with
                            true rolling-without-slipping kinematics
    sas_trace.pdb           the whole path at once, as a closed curve
    contact_points.pdb      where the probe touches, on the vdW surface
    roll_animation.pml      loads and sets all of it up
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


PROBE_RADIUS = 1.4
# Bondi radii, matching PyMOL's own table so the drawn spheres and the computed
# surface agree.
VDW = {"H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80}


def read_atoms(pdb_path: Path):
    names, elements, coords = [], [], []
    for line in pdb_path.read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        names.append(line[12:16].strip())
        element = line[76:78].strip() or line[12:16].strip()[:1]
        elements.append(element)
        coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    radii = np.array([VDW.get(e, 1.70) for e in elements])
    return names, np.array(elements), np.array(coords), radii


def surface_point(origin, direction, coords, expanded, hi=30.0, tol=1e-9):
    """Where a ray from `origin` leaves the solvent-accessible surface.

    f(R) = max_i (expanded_i - |p(R) - a_i|) is positive inside the surface and
    negative outside, so the crossing is found by bisection. Returns the point
    and the index of the atom the probe is resting on there.
    """
    def gap(distance):
        point = origin + distance * direction
        return (expanded - np.linalg.norm(coords - point, axis=1))

    low, high = 0.0, hi
    if gap(high).max() > 0:
        raise ValueError("ray does not leave the surface within the search radius")
    while high - low > tol:
        middle = 0.5 * (low + high)
        if gap(middle).max() > 0:
            low = middle
        else:
            high = middle
    point = origin + high * direction
    return point, int(np.argmax(gap(high)))


def rodrigues(axis, angle):
    norm = np.linalg.norm(axis)
    if norm < 1e-12 or abs(angle) < 1e-12:
        return np.eye(3)
    axis = axis / norm
    cross = np.array([[0.0, -axis[2], axis[1]],
                      [axis[2], 0.0, -axis[0]],
                      [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def build_path(coords, radii, frames=180, probe=PROBE_RADIUS):
    """Sweep the probe once around the residue's widest profile."""
    expanded = radii + probe
    centroid = coords.mean(axis=0)

    # Principal axes of the atom cloud. The shortest one is the rotation axis,
    # so the sweep happens in the plane of the two long axes.
    centred = coords - centroid
    _u, _s, vectors = np.linalg.svd(centred, full_matrices=False)
    first, second, axis = vectors[0], vectors[1], vectors[2]

    angles = np.linspace(0.0, 2.0 * np.pi, frames, endpoint=False)
    centres, contacts, resting = [], [], []
    for angle in angles:
        direction = np.cos(angle) * first + np.sin(angle) * second
        point, atom = surface_point(centroid, direction, coords, expanded)
        centres.append(point)
        resting.append(atom)
        # The contact sits on the line from the atom centre to the probe centre,
        # at that atom's own vdW radius.
        offset = point - coords[atom]
        contacts.append(coords[atom] + offset / np.linalg.norm(offset) * radii[atom])
    return np.array(centres), np.array(contacts), np.array(resting), axis, centroid


def rolling_marker(centres, contacts, probe=PROBE_RADIUS):
    """A point on the probe surface, rotated by rolling without slipping.

    At each step the contact point is instantaneously stationary, so the probe
    turns about (n x dp) by |dp| / r, where n is the outward contact normal.
    """
    normals = centres - contacts
    normals /= np.linalg.norm(normals, axis=1)[:, None]

    rotation = np.eye(3)
    start = normals[0] * probe            # marker starts at the contact point
    markers = [centres[0] + start]
    for index in range(1, len(centres)):
        step = centres[index] - centres[index - 1]
        angle = np.linalg.norm(step) / probe
        rotation = rodrigues(np.cross(normals[index - 1], step), angle) @ rotation
        markers.append(centres[index] + rotation @ start)
    return np.array(markers)


def _atom_line(serial, name, resname, chain, resnum, xyz, element, bfactor=0.0):
    return (f"HETATM{serial:5d} {name:<4s}{'':1s}{resname:>3s} {chain:1s}{resnum:4d}    "
            f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00{bfactor:6.2f}"
            f"          {element:>2s}")


def write_multistate(path, frames_xyz, names, elements, resname, header):
    lines = [f"REMARK   1 {header}"]
    for model, frame in enumerate(frames_xyz, start=1):
        lines.append(f"MODEL     {model:4d}")
        for serial, (name, element, xyz) in enumerate(zip(names, elements, frame), start=1):
            lines.append(_atom_line(serial, name, resname, "P", serial, xyz, element))
        lines.append("ENDMDL")
    lines.append("END")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_curve(path, points, resname, element, header, close_loop=True):
    lines = [f"REMARK   1 {header}"]
    for serial, xyz in enumerate(points, start=1):
        lines.append(_atom_line(serial, "X", resname, "T", serial, xyz, element))
    lines.append("TER")
    count = len(points)
    for index in range(count):
        partners = []
        if index > 0:
            partners.append(index)
        if index < count - 1:
            partners.append(index + 2)
        if close_loop and index == 0:
            partners.append(count)
        if close_loop and index == count - 1:
            partners.append(1)
        if partners:
            lines.append("CONECT" + f"{index + 1:5d}" + "".join(f"{p:5d}" for p in sorted(set(partners))))
    lines.append("END")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


PML = '''# Water-sized probe rolling over asparagine
# The probe centre traces the solvent-accessible surface, so it touches the
# van der Waals spheres at every frame and never overlaps them.
# Radii here are PyMOL's own (H 1.20, C 1.70, N 1.55, O 1.52); the probe is {probe}.

load asparagine.pdb,        asn
load probe_path.pdb,        probe
load sas_trace.pdb,         trace
load contact_points.pdb,    contacts
# Optional: the probe with a dot showing it genuinely rolling rather than sliding
# load probe_with_marker.pdb, probe_rolling

bg_color white
hide everything

# --- the residue, as spheres with hydrogens ---
show spheres, asn
color grey70, asn and elem C
color skyblue, asn and elem N
color salmon, asn and elem O
color white, asn and elem H
set sphere_transparency, 0.0, asn

# --- the probe, exactly 1.4 A ---
show spheres, probe
alter probe, vdw={probe}
rebuild
color red, probe

# --- the path it traces (this is the SAS) and where it touches ---
show sticks, trace
set stick_radius, 0.06, trace
color yellow, trace
show spheres, contacts
set sphere_scale, 0.12, contacts
color orange, contacts
disable contacts

set ray_opaque_background, 1
set orthoscopic, 1
orient asn
zoom asn or trace, 2

mset 1 -{frames}
set movie_fps, 30

# Play it, or scrub with the frame slider. To render frames:
#   set ray_trace_frames, 1
#   mpng frame_
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--residue-pdb",
                        default=str(Path.home() / "Documents" / "sasa_probe_animation"
                                    / "asparagine.pdb"))
    parser.add_argument("--output-dir",
                        default=str(Path.home() / "Documents" / "sasa_probe_animation"))
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--probe", type=float, default=PROBE_RADIUS)
    args = parser.parse_args()

    residue_pdb = Path(args.residue_pdb).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    names, elements, coords, radii = read_atoms(residue_pdb)
    print(f"residue: {len(coords)} atoms from {residue_pdb.name}")

    centres, contacts, resting, axis, centroid = build_path(
        coords, radii, frames=args.frames, probe=args.probe)
    markers = rolling_marker(centres, contacts, probe=args.probe)

    # Check the probe really is in contact and never overlapping.
    distances = np.linalg.norm(centres[:, None, :] - coords[None, :, :], axis=2)
    clearance = distances - (radii + args.probe)[None, :]
    print(f"worst overlap across all frames : {clearance.min():+.2e} A  (0 = touching)")
    print(f"contact gap per frame           : max {np.abs(clearance.min(axis=1)).max():.2e} A")
    print(f"atoms the probe rests on        : "
          f"{sorted({names[i] for i in resting})}")

    write_multistate(output_dir / "probe_path.pdb", centres[:, None, :], ["O"], ["O"],
                     "HOH", f"probe centre, r = {args.probe} A, {args.frames} frames")
    both = np.stack([centres, markers], axis=1)
    write_multistate(output_dir / "probe_with_marker.pdb", both, ["O", "X"], ["O", "N"],
                     "HOH", "probe centre plus a surface dot rolling without slipping")
    write_curve(output_dir / "sas_trace.pdb", centres, "SAS", "C",
                "solvent-accessible surface traced by the probe centre")
    write_curve(output_dir / "contact_points.pdb", contacts, "CNT", "C",
                "points where the probe touches the van der Waals surface",
                close_loop=False)
    (output_dir / "roll_animation.pml").write_text(
        PML.format(probe=args.probe, frames=args.frames), encoding="utf-8")

    print(f"\nwrote {args.frames} frames to {output_dir}")


if __name__ == "__main__":
    main()
