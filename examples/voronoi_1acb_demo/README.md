# Interactive 1acb Voronoi demo

This example computes three successful `1acb` models into an isolated output
directory and displays actual bounded radical-Voronoi face polygons in an
interactive Plotly/Jupyter view. Chain E is blue, chain I is orange, and the
selected inter-chain contact areas are labeled in Å².

## Recompute the demo data

From the repository root, using the `ppi-voronoi` Conda environment:

```bash
conda activate ppi-voronoi
python -m examples.voronoi_1acb_demo.prepare_demo --count 3
```

Outputs are written to `voronoi_demo_outputs/1acb_three_models/`, separate
from production feature output. Existing successful model files are reused;
pass `--overwrite` to recompute them.

The script records a numerical failure and tries another model until it has
three successes. This matters because Voro++ can occasionally reject a point
configuration for numerical reasons.

## Open the interactive notebook

```bash
conda activate ppi-voronoi
jupyter lab examples/voronoi_1acb_demo/interactive_1acb_voronoi_demo.ipynb
```

Select the `Python (ppi-voronoi)` kernel if Jupyter does not select it
automatically. Use the dropdown to switch models, the slider to change the
number of displayed contacts, drag to rotate, scroll to zoom, and hover over
white contact segments for residue identities and exact areas.

## Rebuild the notebook source

The checked-in notebook is generated from `build_notebook.py`:

```bash
python examples/voronoi_1acb_demo/build_notebook.py
```
