# apbs_analysis — surface charge potential maps from PDB models

Generates Poisson–Boltzmann surface electrostatic potential maps for PPI
heterodimer models using **pdb2pqr** (forcefield charges and radii) and
**APBS** (the PB solve), then projects the potential onto each structure's
solvent-accessible surface and aggregates it per residue.

**Nothing here writes into the training graph HDF5 files.** Results live in
their own per-target store, so electrostatics can be regenerated, re-
parameterised, or thrown away without touching the graphs. A join key back to
graph nodes is stored (`residue_aa_id`) for whenever that's wanted.

---

## What gets computed

Per structure:

1. `structure_prep.prepare_structure` normalises the PDB (see *Input handling*).
2. `pdb2pqr` assigns a PARSE charge and radius to every atom → PQR.
3. `apbs` solves the linearised PB equation on a focused multigrid (`mg-auto`)
   → volumetric potential in kT/e.
4. A Shrake–Rupley solvent-accessible point cloud is built from the PQR radii,
   each point tagged with its parent atom and residue.
5. The volume is trilinearly sampled at those surface points and at atom
   centres, then rolled up per residue.

Step 5 is what makes this a *surface* map rather than a volume: the raw grid is
mostly bulk solvent and protein interior, whereas the per-residue surface
statistics are the compact, structure-aligned quantity that scales to many
thousands of models.

Because the surface points are near-equal-area samples, a residue's plain mean
over its points is already an area-weighted mean surface potential.

## Layout

| File | Role |
| --- | --- |
| `structure_prep.py` | Normalise a PDB into something pdb2pqr accepts; build the aa_id map |
| `electrostatics.py` | pdb2pqr/APBS invocation, PQR parsing, SAS sampling, residue aggregation |
| `grid_sizing.py` | APBS `mg-auto` grid dimensioning (psize conventions) |
| `dx_grid.py` | OpenDX reader + trilinear sampling |
| `storage.py` | HDF5 layout and atomic per-model commits |
| `pipeline.py` | Resumable batch runner (sequential or multi-process) |
| `cli_options.py` | Shared argparse wiring, so both entry points expose identical physics |
| `generate_apbs_surface_data.py` | **Cluster entry point** — one target per invocation |
| `run_local_84_targets.py` | **Local entry point** — the 84 bound complexes |
| `inspect_apbs_output.py` | Read-only QC on an output HDF5 |
| `analysis.py` | pandas loaders for the store |
| `repack_store.py` | Migrate a pre-existing store to the current on-disk layout |
| `selfcheck.py` | Fast correctness checks for the pure-Python pieces (no APBS needed) |
| `export_dx.py` | Regenerate an OpenDX map from stored HDF5 for PyMOL/ChimeraX |
| `cluster/` | SLURM array job, worker, env setup + activation, preflight, resubmit, progress check |

---

## Local run (the 84 bound complexes)

One-time environment (APBS has no PyPI wheel, so it must come from conda-forge):

```bash
conda create -y -n apbs_env -c conda-forge python=3.11 apbs pdb2pqr numpy scipy h5py pandas
```

Then, from the repository root:

```bash
conda activate apbs_env && python -m apbs_analysis.run_local_84_targets --workers 3
```

Writes to `~/Documents/apbs_electrostatics_84_targets/`:

- `targets_84_apbs_surface.hdf5` — one group per PDB id
- `targets_84_apbs_summary.csv` — one row per structure (status, timing, totals)

The local run stores **everything**: per-residue statistics, the full surface
point cloud, and the raw potential volume. Each structure is one bound
heterodimer, so that is affordable here in a way it is not on the cluster.

Useful flags: `--targets 1acb 2grn` (subset), `--no-store-grid` (skip volumes),
`--sphere-points 250` (finer surface sampling), `--keep-intermediates` (also
write each PQR, APBS input, log, and a gzipped DX map), `--overwrite`.

Reruns are resumable — completed structures are skipped, so an interrupted run
just needs the same command again.

## Cluster run

```bash
bash apbs_analysis/cluster/setup_env.sh          # once, see "Naming the environment" below
python apbs_analysis/cluster/list_targets.py \
    /nfs/roberts/pi/pi_co54/jas485/uniformly_sampled_target_data \
    apbs_analysis/cluster/targets.txt            # only if the target set changed
sbatch --array=1 apbs_analysis/cluster/dispatch_apbs_jobs.slurm   # validate on one target
sbatch apbs_analysis/cluster/dispatch_apbs_jobs.slurm             # full array
```

Results land in
`/nfs/roberts/pi/pi_co54/jas485/ppi_gnn_data_store/apbs_model_data/` as
`<target>_apbs_surface.hdf5` + `<target>_apbs_summary.csv`.

The worker mirrors the Voronoi array's operational contract — one target per
array task, resumable, SIGTERM-safe, requeueable — with two differences:

- **No commit phase.** The Voronoi worker has one because it writes features
  back into the graph HDF5. This one has nothing to commit; each model group is
  renamed into place only once fully written, which is all the atomicity needed.
- **Much smaller memory request** (16 GiB vs 64 GiB). APBS memory is set by the
  grid, which `--memory-ceiling-mb` caps directly, and nothing here builds a mesh.

### Naming the environment

`setup_env.sh` builds the environment with `conda create --prefix`, which
produces a **nameless** env: it must live on the shared filesystem the compute
nodes read, not in `~/.conda/envs`, and a prefix env has no name to activate.
There is no file inside the env that names it — `conda activate <name>` only
resolves names against the directories listed in `envs_dirs`.

To make `conda activate apbs_env` work, add its parent directory to
`~/.condarc`. **Order matters**: the first entry is where `conda create -n`
writes new environments, so keep the default first.

```yaml
envs_dirs:
  - /home/jas485/.conda/envs
  - /nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder
```

`setup_env.sh` detects whether the name already resolves and prints this block
with the right paths filled in if it does not.

The cluster scripts source `cluster/activate_env.sh`, which tries the short
name first and falls back to the full prefix. That fallback is deliberate:
`~/.condarc` lives outside this repo, so an array task should not fail because
a personal config file was reset. Override the name with `APBS_ENV_NAME` or the
location with `APBS_ENV_PREFIX`.

Progress/QC for one target:

```bash
bash apbs_analysis/cluster/check_apbs_output.sh 1acb
```

Resubmit only incomplete targets:

```bash
bash apbs_analysis/cluster/resubmit_failed.sh
```

### Why the cluster run stores less

Per model, a surface point cloud is roughly 7 MB and a raw potential volume 15–70 MB.
At thousands of models per target that is tens of terabytes, so the cluster
worker stores only the atom- and residue-level results — **~147 KiB per model**
for a size-representative sample, i.e. ~200 MiB per 1,400-model target and
~29 GiB across all 146. Pass `--store-surface-points` / `--store-grid` for a
deliberately chosen subset if the fine detail is needed.

The atom-level table is ~90% of that; residue-level alone is ~31 B/residue,
so dropping atoms would shrink the store roughly tenfold if it turns out not to
be needed.

## Runtime and sizing

Measured on Apple M-series in the cluster storage configuration, PROPKA on,
one model at a time:

| Structure | Heavy atoms | Seconds |
| --- | ---: | ---: |
| 3dgp | 1,017 | 16.2 |
| 5dmb | 1,694 | 17.3 |
| 2dvw | 2,327 | 18.8 |
| 7joe | 2,947 | 26.5 |
| 6kp3 | 3,762 | 33.9 |
| 5mu7 | 4,169 | 42.5 |
| 1kfu | 7,119 | 55.9 |

Runtime is close to linear in size: **≈ 5.7 s + 7.3 s per 1,000 heavy atoms**
(R² = 0.95). The 84 targets have a median of 2,316 heavy atoms, so **~23 s per
model** is the number to plan with; peak RSS tops out at 3.45 GiB on the
largest, since `--memory-ceiling-mb` caps the grid regardless of molecule size.

**Parallelism is per-model, not per-solve.** APBS's `mg-auto` solver is
single-threaded (measured at 91% of one core), so `--workers N` runs N models
in separate processes while the parent does all HDF5 writes. Measured scaling
on the seven-structure sample: 170 s at 1 worker, 102 s at 2 (1.7×), 64 s at 4
(2.7×). The worker script sets `--workers` from `SLURM_CPUS_PER_TASK`, so the
shipped `--cpus-per-task=2` means two models at a time per target.

For a 1,400-model target at ~23 s/model that is roughly **9 h single-threaded,
5.5 h at 2 workers** — before derating for slower cluster cores, so plan on
8–11 h against the 24 h wall limit. Going past 2 workers needs more memory than
the shipped 16 GiB (worst case is ~4.5 GiB per concurrent solve); raising the
array throttle is the cheaper way to buy throughput.

### Skipping PROPKA

`--titration-method none` is **not** a meaningful speedup: measured at 3.8%
(169 s → 163 s over the same seven structures), because runtime is dominated by
the APBS solve, not by pKa prediction. It does change the answer — net charge
differed on 3 of 7 structures, since PROPKA is what decides HIS protonation and
any shifted ASP/GLU/LYS pKa at pH 7. Not worth the trade.

---

## Checking the code itself

```bash
python -m apbs_analysis.selfcheck
```

Covers grid dimensioning, the OpenDX round trip and its axis order, surface-area
sampling against analytic sphere areas, and the residue renumbering the graph
join depends on. Needs no APBS install, runs in seconds, and is also run by the
cluster preflight before every array task.

## Input handling

Three things in real model PDBs break or silently corrupt pdb2pqr, and
`structure_prep.py` handles each explicitly:

- **Duplicate residue numbering.** Some files were written with insertion codes
  dropped, so one chain carries two distinct residues both numbered e.g. 20
  (`3sgb` and `7joe` among the 84). pdb2pqr merges them and then fails with a
  structural "gap" — or, worse, silently produces wrong chemistry. Residues are
  therefore renumbered sequentially before pdb2pqr sees them.
- **Incomplete side chains.** pdb2pqr normally rebuilds unresolved side chains
  correctly, so the complete structure is always attempted first. On the
  structures where that rebuild crashes (`1euv` among the 84), the run retries
  with those side chains cut back to ALA/GLY. That discards their side-chain
  charge, so it is recorded per residue (`residue_truncated`) rather than
  applied silently. Residues merely missing atoms are flagged
  `residue_incomplete` whether or not they were truncated.
- **Pre-existing hydrogens.** pdb2pqr re-protonates regardless, and the
  reduce-added hydrogens in these files carry serial number 0, so they are
  dropped up front.

A fourth problem appears on the way *out* of pdb2pqr: it writes PQR in
fixed-width columns, so a coordinate wide enough to fill its field runs into the
next one (`60.423-177.683` is two values, not one). APBS's own reader rejects
such a file outright, and 3 of the 84 hit it. Both `parse_pqr` and the
normalised PQR that APBS is actually given handle this — relevant well beyond
these 84, since any docked pose translated far from the origin triggers it.

### Measured on the 84 bound complexes

All 84 complete. Across 22,689 surface-exposed residues, mean surface potential
comes out at **+1.03 kT/e for ARG/LYS**, **−1.76 for ASP/GLU**, −0.29 for HIS,
and ≈0 for polar and apolar residues; the correlation between residue formal
charge and mean surface potential is positive in every one of the 84 models
(min +0.11, median +0.29). That sign structure is the end-to-end check that the
charges, the solve, and the grid-to-surface sampling all line up — a transposed
grid axis or a broken residue mapping would flatten it.

Of 26,533 residues, 3,844 are fully buried (no surface points, NaN statistics),
192 are flagged `residue_incomplete`, 6 needed truncation (all in `1euv`), and
none were dropped by pdb2pqr. Total runtime was about 25 minutes at
`--workers 3`; the store is 1.7 GiB with volumes and point clouds included.

## Joining back to the graphs

`residue_aa_id` is the join key. The renumbering is chosen so that residue *i*
in the store is the *(i+1)*-th residue in the source PDB's file order — exactly
the sequential counter `create_protein_graph_structure.py` assigns as `aa_id`.
Residue arrays span **every** residue of the source PDB, so they align 1:1 with
graph nodes even where pdb2pqr dropped a residue (those entries are NaN, with
`residue_in_pqr` false).

Matching on `(chain, residue_number)` would be ambiguous on exactly the files
whose insertion codes were dropped, which is why the index is used instead.
`common.graph_group_candidates(model_id)` gives the graph group names a model id
may correspond to (the PDB filename is identical for `complex.X_Y_Z` and its
`_corrected` variant, so that inverse mapping is genuinely ambiguous).

## Reading the output

```python
from apbs_analysis.analysis import load_residue_table, interface_residues, load_surface_points

path = "~/Documents/apbs_electrostatics_84_targets/targets_84_apbs_surface.hdf5"
residues = load_residue_table(path, exposed_only=True)   # one row per residue per model
per_chain = interface_residues(residues)                 # SASA-weighted per-chain summary
cloud = load_surface_points(path, "1acb")                # per-point map for one model
```

Buried residues have no surface points, so their potential statistics are NaN —
`exposed_only=True` drops them rather than letting NaNs propagate.

For visualisation, regenerate an OpenDX map on demand instead of storing one:

```bash
python -m apbs_analysis.export_dx <store.hdf5> 1acb -o 1acb_potential.dx
```

## Output HDF5 layout

Root attributes record the full parameter set (forcefield, pH, dielectrics,
ionic strength, solver, probe radius, sphere points, units) so any file is
self-describing. Per model group:

| Group | Datasets |
| --- | --- |
| Atoms (N) | `atom_aa_id`, `atom_chain`, `atom_resnum`, `atom_resname`, `atom_name`, `atom_pqr_resname`, `atom_xyz`, `atom_charge`, `atom_radius`, `atom_potential` |
| Residues (R) | `residue_aa_id`, `residue_chain`, `residue_number`, `residue_insertion_code`, `residue_name`, `residue_modeled_name`, `residue_incomplete`, `residue_truncated`, `residue_in_pqr`, `residue_charge`, `residue_sasa`, `residue_surface_point_count`, `residue_potential_{mean,min,max,std}` |
| Surface points (P, optional) | `surface_xyz`, `surface_potential`, `surface_atom_index`, `surface_residue_index` |
| Volume (optional) | `potential_grid` + `grid_origin` / `grid_spacing` / `grid_shape` attributes |

`atom_pqr_resname` preserves the protonation state pdb2pqr chose (HIS → HID/HIE/HIP etc.);
`atom_resname` is the original name from the source PDB.

Potentials are in **kT/e** (APBS's native output; multiply by 25.7 for mV at 298 K).

## Defaults

PARSE forcefield, pH 7.0 with PROPKA titration states, ε_protein 2.0,
ε_solvent 78.54, 0.150 M 1:1 salt, 298.15 K, linearised PB, ~0.5 Å target fine-grid
spacing (coarsened if the grid would exceed `--memory-ceiling-mb`), 1.4 Å probe,
100 Shrake–Rupley points per atom. Every one is a flag on both entry points, and
every one is recorded in the output file.

`--titration-method none` skips PROPKA and uses standard protonation states,
but see *Skipping PROPKA* above — it saves under 4% and changes the chemistry,
so it is not a worthwhile trade.
