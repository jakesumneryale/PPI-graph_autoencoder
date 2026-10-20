# README

## Overview

These scripts compute relative solvent‑accessible surface area (rSASA) features, partial charges, and electrostatic interaction energies on graph edges for use as node/edge features in downstream GNN models (e.g., for DockQ‑based decoy scoring).

---

## 1. `calc_rsasa_features.py`

**Purpose:**
Computes relative solvent-accessible surface area (rSASA) features for decoy protein–protein complexes, compares bound (complex) vs. unbound (monomer) rSASA values, normalizes them against reference averages/standard deviations, and aggregates per-target summary statistics (e.g., average ΔrSASA and rSASA_i vs. DockQ score). Also includes a helper to annotate graph edge features with Cα–Cα distances.

**Command-line arguments:**
- `-p`, `--pdb`: PDB/target ID to process.

**Key imports/dependencies:**
`create_protein_graph_structure` (aliased `jk`), `h5py`, `pandas`, `numpy`

**Functions:**

- **`add_ca_contact_dist(pdb)`**
  Opens the target's HDF5 graph file and, for each decoy graph, computes the Cα–Cα Euclidean distance for every contact/edge (using atomic coordinates from `jk.get_protein_information`) and stores it as a new `ca_dist` dataset under `edge_features` in the HDF5 file.

- **`calc_rsasa(pdbid)`**
  For every decoy PDB in a target's decoy directory:
  - Splits the complex into receptor/ligand chains (`pdb_splitchain.py`).
  - Computes rSASA for the complex and for each isolated chain (monomer) via `jk.get_protein_information` and `jk.calculate_rsasa_for_protein`.
  - Merges complex vs. monomer rSASA per residue, computes `delta_rSASA` (complex − monomer).
  - Saves a per-decoy CSV of these features and cleans up temporary files.

- **`calc_rsasa_i(pdbid)`**
  Loads previously computed per-decoy rSASA CSVs and computes node-level rSASA features `rSASA_i_complex` and `rSASA_i_monomer` using per-residue average and standard deviation from `norm_rSASA.csv`. Overwrites the CSVs with these added columns.

- **`calc_avg_rsasa_i(pdb)`**
  For each decoy in the target's supersampled score table, loads its rSASA CSV, computes/saves the mean standardized `rSASA_i_complex` per decoy alongside DockQ scores, saving to `{pdb}_avg_rSASA_i.csv`.

- **`main()`**
  Runs `calc_rsasa_i(args.pdb)` followed by `calc_avg_rsasa_i(args.pdb)` for the target passed via `-p/--pdb`.

**Note:** File paths are hard-coded to a specific HPC cluster directory structure (`/gpfs/gibbs/pi/ohern/...`), so this script is tailored to a particular lab's compute environment and dataset layout.

---

## 2. `calc_partial_charge.py`

**Purpose:**
Assigns partial atomic charges to all PDB structures in a given directory using `pdb2pqr` with the PARSE force field, producing `.pqr` (charge/radius) and corresponding `.pdb` output files. These charge files are later consumed by `create_graph_edge_ves.py` to compute electrostatic potentials.

**Command-line arguments:**
- `-p`, `--pdbid`: PDB ID used to name the output subdirectory.
- `-d`, `--dir`: Directory containing the complex `.pdb` files to process.

**Functions:**
- `main()`
  Parses command-line arguments, creates a `{pdbid}_partial_charges` output subdirectory inside the given complex directory, finds all `.pdb` files in that directory, and for each one runs:
  ```
  pdb2pqr <input.pdb> <output.pqr> --ff PARSE --pdb-output <output.pdb> --whitespace
  ```
  producing a `_Q.pqr` and `_Q.pdb` file per input structure.

**Dependencies:** `pdb2pqr` (external CLI tool), `pathlib`, `os`, `argparse`, 

---



## 3. `create_graph_edge_ves.py`

**Purpose:**
Computes screened electrostatic interaction energies between interface residue pairs in protein–protein complex decoys, using Debye–Hückel-like screened Coulomb potentials derived from partial charges (from `.pqr` files produced by `calc_partial_charge.py`) and structural graph edge/interface annotations stored in HDF5 files.

**Physical constants defined:**
- `q_e`: elementary charge
- `e_0`: vacuum permittivity
- `D_perm`: permittivity constant (80, water)
- `sig`: a length scale (Å → m, 4.8 Å, average distances between neighboring amino acids) 
- `ep_es`: electrostatic energy scale factor
- `lam`: Debye screening length (9 Å)


**Command-line arguments:**
- `-p`, `--pdbid`: Target PDB ID to process.

**Functions:**

- **`main()`**
  1. Loads interface edge indices, edge-to-residue-contact references, and interface node flags from the target's HDF5 graph file (one entry per decoy/key).
  2. For each decoy:
     - Loads the corresponding `.pqr` file (atomic charges & coordinates) based on whether it's the "complex" structure or a monomer.
     - Filters atoms down to those flagged as interface residues.
     - Computes a pairwise atomic distance matrix (converted to meters) and a pairwise charge-product matrix.
     - Computes a screened Coulomb potential matrix: `charge_product * sig / distance * exp(-distance/lambda)`.
     - Sums atom-pair potentials into residue-pair interaction energies (`aa_interact` matrix) for all interface residue pairs.
     - Maps residue-pair interaction energies back onto the original graph edge list (`interface_edge_ref`), producing a per-decoy list of edge electrostatic potentials (`Edge_Ves`), with non-interface edges set to 0.
  3. Collects all decoys' edge potential lists into a DataFrame (`all_ves_df`) indexed by decoy key.
  4. Saves results as both `{pdbid}_edge_ves.csv` and `{pdbid}_edge_ves.pkl`.


---

