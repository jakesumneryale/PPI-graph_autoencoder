"""Plotly helpers for the interactive 1acb Voronoi proof-of-concept notebook."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale


CHAIN_COLORS = ("#2E86DE", "#FF8C42", "#2ECC71", "#9B59B6")


def _decode(values: np.ndarray) -> np.ndarray:
    return np.asarray([value.decode() if isinstance(value, bytes) else str(value) for value in values])


def load_demo(path: str | Path) -> dict[str, object]:
    path = Path(path)
    with h5py.File(path) as handle:
        residues = pd.DataFrame({name: handle["residues"][name][()] for name in handle["residues"]})
        for column in ("aa_name", "chain_name"):
            residues[column] = _decode(residues[column].to_numpy())
        contacts = pd.DataFrame({name: handle["contacts"][name][()] for name in handle["contacts"]})
        faces = {name: handle["faces"][name][()] for name in handle["faces"]}
        for column in ("chain1", "chain2"):
            faces[column] = _decode(faces[column])
        attrs = {name: value for name, value in handle.attrs.items()}
    return {"path": path, "attrs": attrs, "residues": residues, "contacts": contacts, "faces": faces}


def interface_contact_table(data: dict[str, object]) -> pd.DataFrame:
    residues = data["residues"].set_index("aa_id")
    contacts = data["contacts"].copy()
    for side in (1, 2):
        ids = contacts[f"aa_id{side}"].astype(int)
        contacts[f"chain{side}"] = ids.map(residues["chain_name"])
        contacts[f"resnum{side}"] = ids.map(residues["aa_ind"])
        contacts[f"resname{side}"] = ids.map(residues["aa_name"])
    return contacts[contacts["chain1"] != contacts["chain2"]].sort_values(
        "voronoi_contact_area", ascending=False
    ).reset_index(drop=True)


def _residue_label(row: pd.Series, side: int) -> str:
    return f"{row[f'chain{side}']}:{row[f'resname{side}']}{int(row[f'resnum{side}'])}"


def build_figure(data: dict[str, object], top_n: int = 12, show_labels: bool = True) -> go.Figure:
    residues = data["residues"].copy()
    faces = data["faces"]
    top = interface_contact_table(data).head(top_n).copy()
    selected_pairs = {
        (int(row.aa_id1), int(row.aa_id2)): rank
        for rank, row in enumerate(top.itertuples(index=False))
    }
    triangle_pairs = list(zip(faces["aa_id1"].astype(int), faces["aa_id2"].astype(int)))
    mask = np.asarray([pair in selected_pairs for pair in triangle_pairs], dtype=bool)
    triangles = faces["triangles"][mask]
    triangle_areas = faces["contact_area"][mask].astype(float)

    figure = go.Figure()
    chains = sorted(residues["chain_name"].unique())
    for index, chain in enumerate(chains):
        chain_rows = residues[residues["chain_name"] == chain].sort_values("aa_ind")
        hover = [
            f"Chain {chain} · {row.aa_name}{int(row.aa_ind)} · aa_id {int(row.aa_id)}"
            for row in chain_rows.itertuples(index=False)
        ]
        figure.add_trace(
            go.Scatter3d(
                x=chain_rows["x"], y=chain_rows["y"], z=chain_rows["z"],
                mode="lines+markers", name=f"Protein chain {chain}",
                line={"color": CHAIN_COLORS[index % len(CHAIN_COLORS)], "width": 7},
                marker={"color": CHAIN_COLORS[index % len(CHAIN_COLORS)], "size": 3},
                text=hover, hovertemplate="%{text}<extra></extra>",
            )
        )

    if len(triangles):
        vertices = triangles.reshape(-1, 3)
        starts = np.arange(0, len(vertices), 3)
        minimum, maximum = float(triangle_areas.min()), float(triangle_areas.max())
        scale = maximum - minimum or 1.0
        normalized = (triangle_areas - minimum) / scale
        face_colors = sample_colorscale("Turbo", normalized.tolist())
        figure.add_trace(
            go.Mesh3d(
                x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                i=starts, j=starts + 1, k=starts + 2,
                facecolor=face_colors, opacity=0.58, flatshading=True,
                name=f"Voronoi faces (top {len(top)})", showlegend=True,
                hoverinfo="skip", lighting={"ambient": 0.65, "diffuse": 0.75, "specular": 0.15},
            )
        )

    residue_lookup = residues.set_index("aa_id")
    label_x, label_y, label_z, label_text = [], [], [], []
    max_area = max(float(top["voronoi_contact_area"].max()), 1.0) if len(top) else 1.0
    for row in top.itertuples(index=False):
        first = residue_lookup.loc[int(row.aa_id1)]
        second = residue_lookup.loc[int(row.aa_id2)]
        p1 = first[["x", "y", "z"]].to_numpy(dtype=float)
        p2 = second[["x", "y", "z"]].to_numpy(dtype=float)
        area = float(row.voronoi_contact_area)
        hover = (
            f"{first.chain_name}:{first.aa_name}{int(first.aa_ind)} ↔ "
            f"{second.chain_name}:{second.aa_name}{int(second.aa_ind)}<br>"
            f"Voronoi contact area: {area:.2f} Å²<br>Atom faces: {int(row.atom_face_count)}"
        )
        figure.add_trace(
            go.Scatter3d(
                x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
                mode="lines", line={"color": "rgba(250,250,250,0.82)", "width": 2 + 7 * area / max_area},
                text=[hover, hover], hovertemplate="%{text}<extra></extra>",
                showlegend=False,
            )
        )
        midpoint = (p1 + p2) / 2
        label_x.append(midpoint[0]); label_y.append(midpoint[1]); label_z.append(midpoint[2])
        label_text.append(f"{area:.1f} Å²")

    if show_labels and label_text:
        figure.add_trace(
            go.Scatter3d(
                x=label_x, y=label_y, z=label_z, mode="markers+text",
                marker={"size": 3, "color": "white"}, text=label_text,
                textfont={"color": "white", "size": 11}, textposition="top center",
                name="Contact area labels", hoverinfo="skip",
            )
        )

    model_name = str(data["attrs"].get("model_name", data["path"].stem))
    figure.update_layout(
        title={"text": f"1acb · {model_name}<br><sup>Top {len(top)} inter-chain Voronoi contacts; area in Å²</sup>"},
        template="plotly_dark", height=800, margin={"l": 0, "r": 0, "t": 80, "b": 0},
        legend={"x": 0.01, "y": 0.99},
        scene={
            "aspectmode": "data",
            "xaxis": {"title": "x (Å)", "showbackground": False},
            "yaxis": {"title": "y (Å)", "showbackground": False},
            "zaxis": {"title": "z (Å)", "showbackground": False},
            "camera": {"eye": {"x": 1.55, "y": 1.55, "z": 1.15}},
        },
    )
    return figure

