"""PyMOL renderer for dissertation Fig. 4.3 (Poisson-Boltzmann surface electrostatics).

Runs inside PyMOL's own interpreter, driven by build_fig4_3.py:

    pymol -cq fig4_3_render.py -- config.json

Nothing is moved. The open-book panels are made by moving the *camera*, never the
molecules: rotating coordinates would leave the volumetric potential maps behind in
the original frame and every surface would be coloured from the wrong voxels. Each
monomer is instead rendered alone, viewed from its partner's side along the
interface normal with a shared up axis. Contacting points then appear at mirror
positions across the gap between the two images, which is exactly the open-book
correspondence.

Every image is rendered orthoscopically at the same scale (config "px_per_angstrom"),
so the composer can place them in Angstrom coordinates and add a scale bar.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from pymol import cmd

FOV_DEGREES = 20.0


# --------------------------------------------------------------------- geometry
def interface_basis(complex_obj: str, chain_a: str, chain_b: str, cutoff: float):
    """Right-handed frame with ex pointing from chain B to chain A.

    ey is the long axis of the interface footprint (principal component of the
    interface atoms projected onto the interface plane), so the hinge of the open
    book runs along the footprint and the two faces come out tall rather than wide.
    """
    sel_a = f"{complex_obj} and chain {chain_a}"
    sel_b = f"{complex_obj} and chain {chain_b}"
    iface = f"({sel_a} within {cutoff} of {sel_b}) or ({sel_b} within {cutoff} of {sel_a})"
    xyz_iface = cmd.get_coords(iface)
    if xyz_iface is None or len(xyz_iface) < 10:
        raise RuntimeError(f"Too few interface atoms within {cutoff} A to define a frame.")
    center = xyz_iface.mean(axis=0)
    ex = cmd.get_coords(sel_a).mean(axis=0) - cmd.get_coords(sel_b).mean(axis=0)
    ex /= np.linalg.norm(ex)
    projected = (xyz_iface - center) - np.outer((xyz_iface - center) @ ex, ex)
    _, _, vt = np.linalg.svd(projected, full_matrices=False)
    ey = vt[0] - (vt[0] @ ex) * ex
    ey /= np.linalg.norm(ey)
    ez = np.cross(ex, ey)
    return center, ex, ey, ez


def camera(right: np.ndarray, up: np.ndarray):
    right = right / np.linalg.norm(right)
    up = up / np.linalg.norm(up)
    look = np.cross(up, right)          # r = d x u  <=>  d = u x r
    return right, up, look


def projected_extent(selection: str, center, right, up, pad: float):
    xyz = cmd.get_coords(selection) - center
    sx, sy = xyz @ right, xyz @ up
    return (float(sx.min() - pad), float(sx.max() + pad), float(sy.min() - pad), float(sy.max() + pad))


def set_camera(center, right, up, look, frame, px_per_a):
    """Aim an orthoscopic camera so `frame` (Angstrom, in screen axes) fills the image."""
    x0, x1, y0, y1 = frame
    width_a, height_a = x1 - x0, y1 - y0
    # Recentre on the frame so asymmetric extents are not clipped.
    origin = center + right * (x0 + x1) / 2 + up * (y0 + y1) / 2
    dist = height_a / (2.0 * math.tan(math.radians(FOV_DEGREES) / 2.0))
    depth = 400.0
    rotation = np.vstack([right, up, -look])
    view = list(rotation.T.flatten()) + [0.0, 0.0, -dist] + list(origin) + [dist - depth, dist + depth, 1.0]
    cmd.set("field_of_view", FOV_DEGREES)
    cmd.set_view(view)
    return int(round(width_a * px_per_a)), int(round(height_a * px_per_a))


# ---------------------------------------------------------------------- styling
def ramp_colors(stops):
    names = []
    for index, hex_color in enumerate(stops):
        rgb = [int(hex_color[i:i + 2], 16) / 255.0 for i in (1, 3, 5)]
        name = f"fig43_stop_{index}"
        cmd.set_color(name, rgb)
        names.append(name)
    return names


def interpolate(stops, value, vmax):
    """Linear interpolation over evenly spaced stops on [-vmax, vmax].

    Identical to matplotlib's LinearSegmentedColormap.from_list on the same stops,
    so the PyMOL surface and the matplotlib colourbar agree exactly.
    """
    rgb = np.array([[int(h[i:i + 2], 16) / 255.0 for i in (1, 3, 5)] for h in stops])
    t = (np.clip(value, -vmax, vmax) + vmax) / (2 * vmax) * (len(stops) - 1)
    lo = int(min(math.floor(t), len(stops) - 2))
    frac = t - lo
    return list((1 - frac) * rgb[lo] + frac * rgb[lo + 1])


def publication_look():
    cmd.bg_color("white")
    cmd.set("ray_opaque_background", 0)
    cmd.set("orthoscopic", 1)
    cmd.set("depth_cue", 0)
    cmd.set("ray_trace_fog", 0)
    cmd.set("ray_shadows", 0)
    cmd.set("antialias", 2)
    cmd.set("surface_quality", 1)
    cmd.set("two_sided_lighting", 1)
    cmd.set("ambient", 0.45)
    cmd.set("direct", 0.55)
    cmd.set("reflect", 0.25)
    cmd.set("specular", 0.15)
    cmd.set("shininess", 30)
    cmd.set("light_count", 2)
    # Colour each vertex by the map one probe radius out, i.e. on the solvent-accessible
    # surface -- the same place the per-residue potentials were sampled.
    cmd.set("surface_ramp_above_mode", 1)


def flat_look():
    """Unshaded rendering for the interface-footprint masks."""
    cmd.set("ambient", 1.0)
    cmd.set("direct", 0.0)
    cmd.set("reflect", 0.0)
    cmd.set("specular", 0.0)
    cmd.set("light_count", 1)
    cmd.set("antialias", 0)


def show_surface(obj: str):
    cmd.hide("everything", obj)
    cmd.show("surface", obj)


def render(path: Path, size):
    cmd.png(str(path), width=size[0], height=size[1], dpi=300, ray=1)


# ------------------------------------------------------------------------- main
def main(config_path: str) -> None:
    cfg = json.loads(Path(config_path).read_text())
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    px = cfg["px_per_angstrom"]
    pad = cfg.get("pad_angstrom", 3.0)
    chain_a, chain_b = cfg["chain_a"], cfg["chain_b"]

    cmd.reinitialize()
    publication_look()
    cmd.load(cfg["complex_pqr"], "cx")
    cmd.load(cfg["complex_map"], "cx_map")
    cmd.load(cfg["monomer_pqr"][chain_a], "mA")
    cmd.load(cfg["monomer_pqr"][chain_b], "mB")
    cmd.load(cfg["monomer_map"][chain_a], "mA_map")
    cmd.load(cfg["monomer_map"][chain_b], "mB_map")
    cmd.remove("solvent")

    center, ex, ey, ez = interface_basis("cx", chain_a, chain_b, cfg.get("interface_cutoff", 5.0))

    # Views. (a) looks along the interface plane with A on the right; the open-book
    # cameras share that up axis and face each monomer from its partner's side.
    view_a = camera(right=ex, up=ey)
    view_open = {chain_a: camera(right=ez, up=ey), chain_b: camera(right=-ez, up=ey)}
    obj = {chain_a: "mA", chain_b: "mB"}
    obj_map = {chain_a: "mA_map", chain_b: "mB_map"}

    frame_a = projected_extent("cx", center, view_a[0], view_a[1], pad)
    extents = [projected_extent(obj[ch], center, *view_open[ch][:2], pad) for ch in (chain_a, chain_b)]
    half_w = max(max(abs(e[0]), abs(e[1])) for e in extents)
    frame_open = (-half_w, half_w, min(e[2] for e in extents), max(e[3] for e in extents))

    phi_stops = ramp_colors(cfg["phi_stops"])
    phi_max = cfg["phi_max"]
    for name in ("cx", "mA", "mB"):
        cmd.hide("everything", name)

    # (a) complex coloured by the complex map.
    cmd.ramp_new("ramp_cx", "cx_map", [-phi_max + i * 2 * phi_max / (len(phi_stops) - 1) for i in range(len(phi_stops))], phi_stops)
    cmd.disable("ramp_cx")
    show_surface("cx")
    cmd.set("surface_color", "ramp_cx", "cx")
    size = set_camera(center, *view_a, frame_a, px)
    render(out / "a_complex_phi.png", size)
    cmd.hide("everything", "cx")

    # (b) each monomer coloured by its own map.
    for ch in (chain_a, chain_b):
        ramp = f"ramp_{obj[ch]}"
        cmd.ramp_new(ramp, obj_map[ch], [-phi_max + i * 2 * phi_max / (len(phi_stops) - 1) for i in range(len(phi_stops))], phi_stops)
        cmd.disable(ramp)
        show_surface(obj[ch])
        cmd.set("surface_color", ramp, obj[ch])
        size_open = set_camera(center, *view_open[ch], frame_open, px)
        render(out / f"b_{ch}_phi.png", size_open)
        cmd.hide("everything", obj[ch])

    # (c) each monomer coloured per residue by the interaction potential.
    dphi_stops, dphi_max = cfg["dphi_stops"], cfg["dphi_max"]
    for ch in (chain_a, chain_b):
        values = {str(k): v for k, v in cfg["dphi_residue"][ch].items()}
        cmd.set("surface_color", -1, obj[ch])            # back to atom colours
        cmd.color(cfg.get("missing_color", "grey70"), obj[ch])
        painted = 0
        for resi, value in values.items():
            if value is None or not np.isfinite(value):
                continue
            color_name = f"c_{obj[ch]}_{resi}".replace("-", "m")
            cmd.set_color(color_name, interpolate(dphi_stops, value, dphi_max))
            painted += cmd.color(color_name, f"{obj[ch]} and resi \\{resi}") or 1
        show_surface(obj[ch])
        size_open = set_camera(center, *view_open[ch], frame_open, px)
        render(out / f"c_{ch}_dphi.png", size_open)

        # Interface-footprint mask: residues that lose SASA on binding, drawn flat
        # black so the composer can trace the outline over panels (b) and (c).
        cmd.color("white", obj[ch])
        for resi in cfg["interface_residues"][ch]:
            cmd.color("black", f"{obj[ch]} and resi \\{resi}")
        cmd.set("antialias", 0)
        saved = {k: cmd.get(k) for k in ("ambient", "direct", "reflect", "specular", "light_count")}
        flat_look()
        render(out / f"mask_{ch}.png", size_open)
        for k, v in saved.items():
            cmd.set(k, v)
        cmd.set("antialias", 2)
        cmd.hide("everything", obj[ch])

    geometry = {
        "center": center.tolist(), "ex": ex.tolist(), "ey": ey.tolist(), "ez": ez.tolist(),
        "px_per_angstrom": px,
        "frame_a": frame_a, "frame_open": list(frame_open),
        "images": {
            "a": "a_complex_phi.png",
            "b": {ch: f"b_{ch}_phi.png" for ch in (chain_a, chain_b)},
            "c": {ch: f"c_{ch}_dphi.png" for ch in (chain_a, chain_b)},
            "mask": {ch: f"mask_{ch}.png" for ch in (chain_a, chain_b)},
        },
    }
    (out / "geometry.json").write_text(json.dumps(geometry, indent=2))
    print(f"FIG43_RENDER_OK {out}")


# PyMOL executes command-line scripts with __name__ == "pymol", not "__main__".
if __name__ in ("__main__", "pymol"):
    main(sys.argv[-1])
