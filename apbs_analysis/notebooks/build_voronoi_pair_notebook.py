"""Generate the two-residue Voronoi contact notebook (white background).

Same structure as examples/voronoi_1acb_demo/build_notebook.py, reduced to a
single residue pair. Run once; the notebook is the deliverable.

    python apbs_analysis/notebooks/build_voronoi_pair_notebook.py
"""

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "voronoi_pair_figure.ipynb"

nb = nbf.v4.new_notebook()
nb["metadata"]["kernelspec"] = {"display_name": "Python (our_env)", "language": "python", "name": "python3"}
cells = []
md = lambda t: cells.append(nbf.v4.new_markdown_cell(t))  # noqa: E731
code = lambda t: cells.append(nbf.v4.new_code_cell(t))   # noqa: E731

md("""# Shared radical-Voronoi surface between two residues

The face two residues share in the **bounded radical Voronoi tessellation**, drawn on its own.
Each translucent polygon is one atom–atom Voronoi face; the number on the figure is their summed
area, which is exactly the `voronoi_contact_area` edge feature the pipeline stores.

The tessellation is computed over the *whole* protein — a residue's cell is bounded by all of its
neighbours, so two residues tessellated in isolation would give the wrong area. Only the drawing is
reduced to two residues.

Drag to rotate, scroll to zoom, hover a polygon for the atom pair and its area.""")

code('''import sys
from pathlib import Path

REPO_ROOT = Path.cwd()
while REPO_ROOT.name != "PPI-graph_autoencoder" and REPO_ROOT != REPO_ROOT.parent:
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from apbs_analysis.voronoi_pair import (
    build_figure, compute_or_load, pick_default_pair, residue_table,
)

CACHE = Path("~/Documents/voronoi_pair_demo/cache_3dgp.pkl").expanduser()
PDB = REPO_ROOT / "targets_84_complex_only" / "3dgp_complex_H.pdb"

# Reuses the cached full-protein tessellation if present (~20 s to rebuild otherwise).
data = compute_or_load(PDB, CACHE)
protein_df, contacts = data["protein_df"], data["contacts"]
residues = residue_table(protein_df)
print(f"{Path(data['pdb']).name}: {len(residues)} residues, {len(contacts)} residue contacts, "
      f"{len(data['tessellation'])} Voronoi cells, probe {data['probe_size']} Å")''')

md("""## Pick the pair

Default is the largest inter-chain contact. Change `aa1, aa2` to any two residue `aa_id`s that
share a face — the table below lists the strongest inter-chain ones.""")

code('''chain1 = contacts["aa_id1"].map(residues["chain_id"])
chain2 = contacts["aa_id2"].map(residues["chain_id"])
inter = contacts[chain1 != chain2].sort_values("voronoi_contact_area", ascending=False).head(15).copy()
lab = lambda a: f"{residues.loc[a].chain_name}:{residues.loc[a].aa_name}{int(residues.loc[a].aa_ind)}"
inter["contact"] = [f"{lab(a)} ↔ {lab(b)}" for a, b in zip(inter.aa_id1, inter.aa_id2)]
inter[["aa_id1", "aa_id2", "contact", "voronoi_contact_area", "atom_face_count"]].rename(
    columns={"voronoi_contact_area": "area_Å²"}).style.format({"area_Å²": "{:.2f}"}).hide(axis="index")''')

code('''aa1, aa2 = pick_default_pair(contacts, protein_df)   # or e.g. aa1, aa2 = 7, 69
fig = build_figure(data, aa1, aa2, show_hydrogens=True)
fig.show()''')

md("""## Numbers behind the picture

Per atom–atom face, the polygon area (the colour scale on the figure) and the residue total.""")

code('''from apbs_analysis.voronoi_pair import shared_faces
import pandas as pd

_, _, _, polygons = shared_faces(protein_df, data["tessellation"], aa1, aa2)
names = protein_df["atom_name"].to_numpy()
faces = pd.DataFrame({
    "atom_1": [names[a] for _v, _area, (a, b) in polygons],
    "atom_2": [names[b] for _v, _area, (a, b) in polygons],
    "vertices": [len(v) for v, _area, _p in polygons],
    "area_Å²": [area for _v, area, _p in polygons],
}).sort_values("area_Å²", ascending=False)
print(f"{lab(aa1)} ↔ {lab(aa2)}: {len(faces)} faces, total {faces['area_Å²'].sum():.3f} Å²")
row = contacts[(contacts.aa_id1 == min(aa1, aa2)) & (contacts.aa_id2 == max(aa1, aa2))].iloc[0]
print(f"feature table:            {row.voronoi_contact_area:.3f} Å² over {int(row.atom_face_count)} faces")
faces.style.format({"area_Å²": "{:.3f}"}).hide(axis="index")''')

md("""## Export

Static PNG for slides (needs `kaleido`), and a standalone HTML that keeps the interactivity.""")

code('''out = Path("~/Documents/voronoi_pair_demo").expanduser(); out.mkdir(exist_ok=True)
tag = f"{Path(data['pdb']).stem.replace('_complex_H','')}_{lab(aa1)}_{lab(aa2)}".replace(":", "")
fig.write_html(out / f"{tag}.html", include_plotlyjs="cdn")
fig.write_image(out / f"{tag}.png", width=1400, height=900, scale=2)
print("wrote", out / f"{tag}.html", "and", out / f"{tag}.png")''')

nb["cells"] = cells
nbf.write(nb, OUTPUT)
print(OUTPUT)
