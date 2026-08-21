"""Generate the interactive 1acb demonstration notebook."""

from pathlib import Path

import nbformat as nbf


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "interactive_1acb_voronoi_demo.ipynb"

notebook = nbf.v4.new_notebook()
notebook["metadata"]["kernelspec"] = {
    "display_name": "Python (ppi-voronoi)",
    "language": "python",
    "name": "ppi-voronoi",
}
notebook["metadata"]["language_info"] = {"name": "python", "version": "3.10"}
notebook["cells"] = [
    nbf.v4.new_markdown_cell(
        """# Interactive 1acb Voronoi contact-area proof

This notebook displays **actual bounded radical-Voronoi face polygons** for the strongest inter-chain residue contacts. Drag to rotate, scroll to zoom, and hover over the white contact segments for residue identities and contact areas in **Å²**.

- Chain **E** is blue and chain **I** is orange.
- Translucent colored polygons are the atom-level Voronoi faces contributing to the selected residue contacts.
- Each white segment joins representative Cα positions; its label is the summed shared-face area for that residue pair.
"""
    ),
    nbf.v4.new_code_cell(
        """from pathlib import Path
import json
import sys

import ipywidgets as widgets
from IPython.display import display

REPO_ROOT = Path.cwd()
while REPO_ROOT.name != "PPI-graph_autoencoder" and REPO_ROOT != REPO_ROOT.parent:
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT / "examples" / "voronoi_1acb_demo"))

from demo_visualization import build_figure, interface_contact_table, load_demo

DATA_DIR = REPO_ROOT / "voronoi_demo_outputs" / "1acb_three_models"
MODEL_FILES = sorted(DATA_DIR.glob("complex.*.h5"))
assert MODEL_FILES, f"No demo data found in {DATA_DIR}; run prepare_demo.py first"
[(path.stem, round(path.stat().st_size / 1024**2, 1)) for path in MODEL_FILES]"""
    ),
    nbf.v4.new_markdown_cell(
        """## Interactive viewer

Choose any of the three successful models and how many of its strongest inter-chain contacts to expose. Keeping the subset small makes the geometry legible while still showing the true tessellation boundaries."""
    ),
    nbf.v4.new_code_cell(
        """model_picker = widgets.Dropdown(
    options=[(path.stem, str(path)) for path in MODEL_FILES],
    description="Model:", layout=widgets.Layout(width="520px")
)
contact_count = widgets.IntSlider(value=12, min=3, max=30, step=1, description="Top contacts:", continuous_update=False)
labels = widgets.Checkbox(value=True, description="Show Å² labels")
output = widgets.Output()

def redraw(*_):
    with output:
        output.clear_output(wait=True)
        data = load_demo(model_picker.value)
        figure = build_figure(data, top_n=contact_count.value, show_labels=labels.value)
        figure.show()

for control in (model_picker, contact_count, labels):
    control.observe(redraw, names="value")
display(widgets.HBox([model_picker, contact_count, labels]), output)
redraw()"""
    ),
    nbf.v4.new_markdown_cell("## Numerical evidence behind the picture"),
    nbf.v4.new_code_cell(
        """data = load_demo(model_picker.value)
table = interface_contact_table(data).head(20).copy()
table["contact"] = (
    table["chain1"] + ":" + table["resname1"] + table["resnum1"].astype(int).astype(str)
    + " ↔ " + table["chain2"] + ":" + table["resname2"] + table["resnum2"].astype(int).astype(str)
)
table[["contact", "voronoi_contact_area", "atom_face_count"]].rename(
    columns={"voronoi_contact_area": "contact_area_Å²"}
).style.format({"contact_area_Å²": "{:.2f}"}).background_gradient(subset=["contact_area_Å²"], cmap="YlOrRd")"""
    ),
    nbf.v4.new_markdown_cell(
        """### What this proves

The polygons are reconstructed from the vertices returned by the bounded radical Voronoi calculation. For each residue pair, the reported contact area is the sum of its shared atom-level polygon areas. The table and labels therefore come from the same faces shown in 3D—not from distance-based proxy edges."""
    ),
]

nbf.write(notebook, OUTPUT)
print(OUTPUT)
