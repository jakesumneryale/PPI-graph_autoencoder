"""The radical-Voronoi surface two residues share, as an interactive 3D figure.

Same visual language as examples/voronoi_1acb_demo (Plotly, translucent atom-
level face polygons, Å² labels) but reduced to a single residue pair on a white
background, for slides.

The tessellation is still computed over the *whole* protein. That is not
optional: a residue's Voronoi cell is bounded by all of its neighbours, so
tessellating two residues in isolation would give faces bounded only by the
box and the shared area would be wrong. The picture is simplified, the
calculation is not -- the area printed on the figure is exactly the
`voronoi_contact_area` edge feature the pipeline produces.

    python -m apbs_analysis.voronoi_pair --cache ~/Documents/voronoi_pair_demo/cache_3dgp.pkl
    python -m apbs_analysis.voronoi_pair --pdb targets_84_complex_only/3dgp_complex_H.pdb \\
        --pair 12 140 -o ~/Documents/voronoi_pair_demo
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import plotly.graph_objects as go  # noqa: E402
from plotly.colors import sample_colorscale  # noqa: E402

from voronoi_edge_features.contact_area import (  # noqa: E402
    compute_bounded_voronoi_tessellation,
    compute_face_area,
    compute_residue_contact_area_table,
    load_protein_dataframe,
)

# Chain colours as in the demo; residue A is the first chain, B the second.
CHAIN_COLOURS = {0: "#2E86DE", 1: "#FF8C42"}
FACE_COLOURSCALE = "Viridis"
BOND_CUTOFF = 1.9          # A, heavy-heavy; H-heavy handled separately
BOND_CUTOFF_H = 1.3


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------


def compute_or_load(pdb_path: Path | None, cache: Path | None, probe_size: float = 1.4) -> dict:
    """Tessellate the full protein, or reuse a cached tessellation."""
    if cache is not None and cache.is_file():
        data = pickle.load(open(cache, "rb"))
        return data
    if pdb_path is None:
        raise SystemExit("give --pdb, or --cache pointing at an existing cache")
    pdb_path = pdb_path.resolve()
    protein_df = load_protein_dataframe(pdb_path.name, pdb_path.parent)
    shift = protein_df[["x_coord", "y_coord", "z_coord"]].to_numpy(float).mean(axis=0)
    tessellation = compute_bounded_voronoi_tessellation(protein_df, probe_size)
    contacts = compute_residue_contact_area_table(protein_df, tessellation)
    data = {
        "pdb": str(pdb_path), "protein_df": protein_df, "tessellation": tessellation,
        "contacts": contacts, "coordinate_shift": shift, "probe_size": probe_size,
    }
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(data, open(cache, "wb"))
    return data


def residue_table(protein_df: pd.DataFrame) -> pd.DataFrame:
    table = (protein_df[["aa_id", "aa_ind", "aa_name", "chain_id", "chain_name"]]
             .drop_duplicates("aa_id").set_index("aa_id").sort_index())
    return table


def pick_default_pair(contacts: pd.DataFrame, protein_df: pd.DataFrame) -> tuple[int, int]:
    """Largest inter-chain contact: the most demonstrative single face."""
    residues = residue_table(protein_df)
    chain1 = contacts["aa_id1"].map(residues["chain_id"])
    chain2 = contacts["aa_id2"].map(residues["chain_id"])
    inter = contacts[chain1 != chain2]
    if inter.empty:
        inter = contacts
    best = inter.sort_values("voronoi_contact_area", ascending=False).iloc[0]
    return int(best.aa_id1), int(best.aa_id2)


def shared_faces(protein_df: pd.DataFrame, tessellation: list[dict], aa1: int, aa2: int):
    """Every atom-level Voronoi face between residues aa1 and aa2.

    Returns (triangles (T,3,3), face_area per triangle (T,), atom pair per
    triangle (T,2), polygon list) in the tessellation's shifted frame.
    """
    atom_aa = protein_df["aa_id"].astype(int).to_numpy()
    want = {aa1, aa2}
    triangles, tri_area, tri_atoms, polygons = [], [], [], []
    for cell in tessellation:
        cid = int(cell.get("cell_id", -1))
        if cid < 0 or int(atom_aa[cid]) not in want:
            continue
        for face in cell.get("faces", []):
            adj = int(face.get("adjacent_cell", -1))
            if adj < 0 or cid >= adj:
                continue
            if {int(atom_aa[cid]), int(atom_aa[adj])} != want:
                continue
            ids = face.get("vertices", [])
            if len(ids) < 3:
                continue
            verts = np.asarray([cell["vertices"][i] for i in ids], dtype=float)
            area = compute_face_area(cell, face)
            polygons.append((verts, area, (cid, adj)))
            for k in range(1, len(verts) - 1):
                triangles.append(np.stack([verts[0], verts[k], verts[k + 1]]))
                tri_area.append(area)
                tri_atoms.append((cid, adj))
    if not triangles:
        raise SystemExit(f"residues {aa1} and {aa2} share no Voronoi face")
    return (np.stack(triangles), np.asarray(tri_area), np.asarray(tri_atoms), polygons)


# --------------------------------------------------------------------------
# geometry helpers for drawing
# --------------------------------------------------------------------------


def _icosphere(radius: float, centre: np.ndarray, nsub: int = 2):
    import pyvista as pv

    sphere = pv.Icosphere(radius=radius, center=centre, nsub=nsub)
    faces = sphere.faces.reshape(-1, 4)[:, 1:]
    return np.asarray(sphere.points), faces


def residue_atoms(protein_df: pd.DataFrame, aa_id: int, shift: np.ndarray) -> pd.DataFrame:
    atoms = protein_df[protein_df["aa_id"] == aa_id].copy()
    xyz = atoms[["x_coord", "y_coord", "z_coord"]].to_numpy(float) - shift
    atoms["x"], atoms["y"], atoms["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    return atoms


def bonds(atoms: pd.DataFrame) -> list[tuple[int, int]]:
    """Distance-based bonds within one residue, just for drawing sticks."""
    xyz = atoms[["x", "y", "z"]].to_numpy(float)
    is_h = atoms["hyd_bool"].to_numpy(int) == 1
    pairs = []
    for i in range(len(xyz)):
        for j in range(i + 1, len(xyz)):
            d = np.linalg.norm(xyz[i] - xyz[j])
            cutoff = BOND_CUTOFF_H if (is_h[i] or is_h[j]) else BOND_CUTOFF
            if d < cutoff and not (is_h[i] and is_h[j]):
                pairs.append((i, j))
    return pairs


# --------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------


def _label(res: pd.Series) -> str:
    return f"{res.chain_name}:{res.aa_name}{int(res.aa_ind)}"


def build_figure(data: dict, aa1: int, aa2: int, show_hydrogens: bool = True,
                 face_opacity: float = 0.85, sphere_opacity: float = 0.30) -> go.Figure:
    protein_df, tess, shift = data["protein_df"], data["tessellation"], data["coordinate_shift"]
    residues = residue_table(protein_df)
    tri, tri_area, tri_atoms, polygons = shared_faces(protein_df, tess, aa1, aa2)
    total_area = sum(a for _v, a, _p in polygons)

    fig = go.Figure()

    # --- the two residues: spheres at the tessellation's own radii, plus sticks
    for slot, aa in enumerate((aa1, aa2)):
        atoms = residue_atoms(protein_df, aa, shift)
        if not show_hydrogens:
            atoms = atoms[atoms["hyd_bool"] == 0]
        colour = CHAIN_COLOURS[slot]
        name = _label(residues.loc[aa])
        xs, ys, zs, ii, jj, kk = [], [], [], [], [], []
        offset = 0
        for row in atoms.itertuples(index=False):
            pts, faces = _icosphere(float(row.atom_radius), np.array([row.x, row.y, row.z]))
            xs.extend(pts[:, 0]); ys.extend(pts[:, 1]); zs.extend(pts[:, 2])
            ii.extend(faces[:, 0] + offset); jj.extend(faces[:, 1] + offset); kk.extend(faces[:, 2] + offset)
            offset += len(pts)
        fig.add_trace(go.Mesh3d(
            x=xs, y=ys, z=zs, i=ii, j=jj, k=kk, color=colour, opacity=sphere_opacity,
            name=f"{name}  (atoms at tessellation radii)", showlegend=True, hoverinfo="skip",
            lighting={"ambient": 0.6, "diffuse": 0.8, "specular": 0.2, "roughness": 0.6},
            flatshading=False,
        ))
        # sticks
        xyz = atoms[["x", "y", "z"]].to_numpy(float)
        bx, by, bz = [], [], []
        for i, j in bonds(atoms):
            bx += [xyz[i, 0], xyz[j, 0], None]; by += [xyz[i, 1], xyz[j, 1], None]; bz += [xyz[i, 2], xyz[j, 2], None]
        fig.add_trace(go.Scatter3d(
            x=bx, y=by, z=bz, mode="lines", line={"color": colour, "width": 6},
            name=f"{name} bonds", showlegend=False, hoverinfo="skip",
        ))
        # atom centres, hoverable
        hover = [f"{name} · {n}  r = {r:.2f} Å" for n, r in zip(atoms["atom_name"], atoms["atom_radius"])]
        fig.add_trace(go.Scatter3d(
            x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers",
            marker={"size": 3, "color": colour}, text=hover,
            hovertemplate="%{text}<extra></extra>", showlegend=False,
        ))

    # --- the shared Voronoi surface, one colour per atom-level face by its area
    verts = tri.reshape(-1, 3)
    starts = np.arange(0, len(verts), 3)
    lo, hi = float(tri_area.min()), float(tri_area.max())
    norm = (tri_area - lo) / ((hi - lo) or 1.0)
    face_colours = sample_colorscale(FACE_COLOURSCALE, norm.tolist())
    atom_names = protein_df["atom_name"].to_numpy()
    hover_faces = [f"face {atom_names[a]}–{atom_names[b]}: {area:.2f} Å²"
                   for (a, b), area in zip(tri_atoms, tri_area)]
    fig.add_trace(go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2], i=starts, j=starts + 1, k=starts + 2,
        facecolor=face_colours, opacity=face_opacity, flatshading=True,
        name=f"Shared Voronoi surface · {len(polygons)} atom faces · {total_area:.1f} Å²",
        showlegend=True, text=hover_faces, hovertemplate="%{text}<extra></extra>",
        lighting={"ambient": 0.7, "diffuse": 0.7, "specular": 0.1},
    ))
    # polygon outlines so individual atom faces read clearly
    ox, oy, oz = [], [], []
    for pv_, _a, _p in polygons:
        loop = np.vstack([pv_, pv_[:1]])
        ox += loop[:, 0].tolist() + [None]; oy += loop[:, 1].tolist() + [None]; oz += loop[:, 2].tolist() + [None]
    fig.add_trace(go.Scatter3d(
        x=ox, y=oy, z=oz, mode="lines", line={"color": "rgba(30,30,30,0.55)", "width": 2},
        name="face edges", showlegend=False, hoverinfo="skip",
    ))

    # Look at the shared surface face-on. The radical-Voronoi face sits between
    # the two residues' spheres, so the default oblique view shows it edge-on and
    # half hidden; the mean triangle normal points straight at it.
    centroid = verts.mean(axis=0)
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    weighted = (normals.T * tri_area).T
    normal = weighted.sum(axis=0)
    normal /= np.linalg.norm(normal) or 1.0
    # Orient the normal so residue 1 is on the near side; the sign is arbitrary
    # otherwise and either face-on view is fine.
    centre1 = residue_atoms(protein_df, aa1, shift)[["x", "y", "z"]].to_numpy(float).mean(axis=0)
    if np.dot(centre1 - centroid, normal) < 0:
        normal = -normal

    # --- area label, pushed toward the camera so the mesh does not hide it
    label_pos = centroid + 2.0 * normal
    fig.add_trace(go.Scatter3d(
        x=[label_pos[0]], y=[label_pos[1]], z=[label_pos[2]], mode="text",
        text=[f"<b>{total_area:.1f} Å²</b>"], textfont={"size": 20, "color": "black"},
        textposition="middle center", showlegend=False, hoverinfo="skip",
    ))
    # Tilt a little off the exact normal so the face has visible depth.
    up_hint = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(up_hint, normal)) > 0.9:
        up_hint = np.array([0.0, 1.0, 0.0])
    side = np.cross(normal, up_hint); side /= np.linalg.norm(side)
    eye = 1.9 * (0.88 * normal + 0.30 * side + 0.35 * np.cross(side, normal))

    r1, r2 = residues.loc[aa1], residues.loc[aa2]
    fig.update_layout(
        template="plotly_white",
        title={"text": f"{Path(data['pdb']).stem.replace('_complex_H', '')} · {_label(r1)} ↔ {_label(r2)}"
                       f"<br><sup>Bounded radical Voronoi; shared face area {total_area:.2f} Å² "
                       f"over {len(polygons)} atom–atom faces (probe {data['probe_size']} Å)</sup>"},
        height=820, margin={"l": 0, "r": 0, "t": 90, "b": 0},
        legend={"x": 0.01, "y": 0.99, "bgcolor": "rgba(255,255,255,0.7)"},
        paper_bgcolor="white",
        scene={
            "aspectmode": "data", "bgcolor": "white",
            "xaxis": {"title": "x (Å)", "showbackground": False, "gridcolor": "#e5e5e5"},
            "yaxis": {"title": "y (Å)", "showbackground": False, "gridcolor": "#e5e5e5"},
            "zaxis": {"title": "z (Å)", "showbackground": False, "gridcolor": "#e5e5e5"},
            "camera": {"eye": {"x": float(eye[0]), "y": float(eye[1]), "z": float(eye[2])}},
        },
    )
    return fig


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdb", type=Path, help="Complex PDB to tessellate")
    parser.add_argument("--cache", type=Path, help="Pickle to read/write the full tessellation")
    parser.add_argument("--pair", type=int, nargs=2, metavar=("AA_ID1", "AA_ID2"),
                        help="aa_id of the two residues; default = largest inter-chain contact")
    parser.add_argument("-o", "--output-dir", type=Path,
                        default=Path("~/Documents/voronoi_pair_demo").expanduser())
    parser.add_argument("--no-hydrogens", action="store_true")
    parser.add_argument("--png", action="store_true", help="Also write a static PNG (needs kaleido)")
    args = parser.parse_args()

    data = compute_or_load(args.pdb, args.cache)
    protein_df, contacts = data["protein_df"], data["contacts"]
    aa1, aa2 = args.pair if args.pair else pick_default_pair(contacts, protein_df)
    aa1, aa2 = min(aa1, aa2), max(aa1, aa2)

    residues = residue_table(protein_df)
    row = contacts[(contacts.aa_id1 == aa1) & (contacts.aa_id2 == aa2)]
    stem = Path(data["pdb"]).stem.replace("_complex_H", "")
    tag = f"{stem}_{_label(residues.loc[aa1])}_{_label(residues.loc[aa2])}".replace(":", "")
    print(f"{stem}: {_label(residues.loc[aa1])} ↔ {_label(residues.loc[aa2])}  "
          f"(aa_id {aa1}, {aa2})")
    if len(row):
        print(f"  contact area from the feature table: {float(row.voronoi_contact_area.iloc[0]):.3f} Å² "
              f"over {int(row.atom_face_count.iloc[0])} atom faces")

    fig = build_figure(data, aa1, aa2, show_hydrogens=not args.no_hydrogens)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    html = args.output_dir / f"{tag}.html"
    fig.write_html(html, include_plotlyjs="cdn")
    print(f"  wrote {html}")
    if args.png:
        png = args.output_dir / f"{tag}.png"
        fig.write_image(png, width=1400, height=900, scale=2)
        print(f"  wrote {png}")


if __name__ == "__main__":
    main()
