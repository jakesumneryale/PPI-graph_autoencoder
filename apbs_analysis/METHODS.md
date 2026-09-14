# Methods: what each quantity is, and how it was computed

Everything below refers to the datasets in `*_apbs_surface.hdf5`. Every
parameter is also recorded as an attribute on the file itself, so any store is
self-describing — `python -m apbs_analysis.inspect_apbs_output <store>` prints them.

---

## 0. Nomenclature

Every symbol used anywhere in this document, grouped by where it appears.

### Poisson–Boltzmann (§2)

| Symbol | Meaning | Units |
| --- | --- | --- |
| `φ(r)` | electrostatic potential at position **r** | kT/e (APBS output) |
| `r` | a position in space (bold in the equations) | Å |
| `∇` | gradient; `∇·` divergence | Å⁻¹ |
| `ε(r)` | dielectric permittivity, position-dependent | dimensionless (relative) |
| `ε_p`, `ε_s` | solute and solvent dielectric — APBS `pdie` 2.0, `sdie` 78.54 | dimensionless |
| `ρ_f(r)` | fixed charge density: the solute's own PARSE partial charges | e·Å⁻³ |
| `κ̄²(r)` | screening (modified Debye–Hückel) parameter, `= ε(r)κ²` | Å⁻² |

### Screening (§2, §9)

| Symbol | Meaning | Units |
| --- | --- | --- |
| `ρ_ion(r)` | mobile-ion charge density — the screening charge | e·Å⁻³ |
| `κ` | inverse Debye length | m⁻¹ |
| `λ_D` | Debye screening length, `= 1/κ` = **7.86 Å** here | Å |
| `I` | ionic strength, `= ½Σᵢcᵢzᵢ²` = **0.150 M** here | mol·L⁻¹ (or mol·m⁻³ in SI form) |
| `cᵢ` | concentration of ion species *i* (0.150 M each) | mol·m⁻³ |
| `zᵢ` | valence of ion species *i* (+1 and −1) | dimensionless |
| `e` | elementary charge, 1.602177×10⁻¹⁹ | C |
| `ε₀` | vacuum permittivity, 8.854188×10⁻¹² | F·m⁻¹ |
| `ε_w` | water dielectric, 78.54 (same value as `sdie`) | dimensionless |
| `k_B` | Boltzmann constant, 1.380649×10⁻²³ | J·K⁻¹ |
| `T` | temperature, 298.15 | K |
| `N_A` | Avogadro constant, 6.022141×10²³ | mol⁻¹ |

### Surface and per-residue averaging (§3, §5)

| Symbol | Meaning | Units |
| --- | --- | --- |
| `φ̄_r` | **area-weighted mean surface potential of residue r** (this is the "phi-bar" symbol) | kT/e |
| `φ_k` | potential sampled at surface point *k* | kT/e |
| `a_k` | area that surface point *k* represents | Å² |
| `k` | index over surface points | — |
| `i` | index over atoms | — |
| `r` (subscript) | index over residues — **not** a radius in `φ̄_r` | — |
| `R_i`, `r_i` | van der Waals radius of atom *i* (same quantity, both spellings appear) | Å |
| `p` | solvent probe radius, 1.4 | Å |
| `M` | Shrake–Rupley sample points per atom, 100 | — |
| `SASA_i` | solvent-accessible surface area of atom *i* | Å² |
| `dA` | surface area element in the continuous form | Å² |

### Interaction potential (§8)

| Symbol | Meaning | Units |
| --- | --- | --- |
| `Δφ(r)` | complex potential minus the isolated chains', voxel-wise | kT/e |
| `Δφ_r` | the same difference averaged over residue *r*'s surface | kT/e |
| `x_k` | a surface point at which both maps are sampled | Å |

### Screened approach curves (§9)

| Symbol | Meaning | Units |
| --- | --- | --- |
| `U(d)` | interaction energy at separation *d* | kcal·mol⁻¹ (or kT) |
| `d` | displacement of the moving chain from its bound pose | Å |
| `A`, `B` | the moving and stationary chains | — |
| `q_i`, `q_j` | PARSE partial charge on atom *i* (in A) or *j* (in B) | e |
| `φ_B(r_i)` | potential from chain B evaluated at atom *i* of chain A | kcal·mol⁻¹·e⁻¹ |
| `r_ij` | distance between atoms *i* and *j* — **a distance, not a radius** | Å |
| `q_A`, `q_B` | net charge of each chain (monopole term) | e |
| `R` | centre-to-centre distance between chains (monopole term) | Å |
| `ε_r` | relative dielectric, 78.54 — same as `ε_w` | dimensionless |

### Grid sizing (§2)

| Symbol | Meaning | Units |
| --- | --- | --- |
| `dime` | grid points per axis, constrained to `c·2^(nlev+1)+1` | — |
| `nlev` | multigrid levels, 4 | — |
| `cglen`, `fglen` | coarse and fine box lengths | Å |
| `cfac` | coarse-box padding factor, 1.7 | — |
| `fadd` | fine-box padding added to the molecule, 20 | Å |

### Constants and conversions

| Quantity | Value |
| --- | --- |
| `k_BT` at 298.15 K | 0.5925 kcal·mol⁻¹ |
| `1 kT/e` | 25.693 mV |
| Coulomb constant `e²N_A/4πε₀` | 332.0637 kcal·Å·mol⁻¹·e⁻² |

### Notation collisions to watch

Three symbols are reused with different meanings, which is worth stating
explicitly rather than leaving to context:

- **`r`** is a position vector in the PB equations (`φ(r)`, `ε(r)`), a residue
  index in `φ̄_r`, and an atomic radius in `r_i` and `a_k`.
- **`ε`** is the dielectric throughout, but `ε₀` is vacuum permittivity with
  units, while `ε_p`, `ε_s`, `ε_w`, `ε_r` are all *relative* and dimensionless.
  `ε_s`, `ε_w` and `ε_r` are the same number, 78.54.
- **`λ_D`** and a bare `λ` are the same Debye length; `λ` appears unsubscripted
  in the practical-units form of `U`.

---

## 1. Charges and radii — pdb2pqr

Each structure is normalised (see *Input handling* in the README) and passed to
**pdb2pqr 3.6.1** with the **PARSE** forcefield at **pH 7.0**, with titration
states from **PROPKA**. pdb2pqr adds hydrogens, picks a protonation state for
every ionisable group, and assigns each atom a partial charge \(q_i\) (units of
elementary charge, *e*) and a radius \(R_i\) (Å).

PARSE is the standard choice for continuum electrostatics: its radii and charges
were fit to reproduce experimental small-molecule hydration free energies using
the PB equation itself, rather than for molecular-mechanics dynamics
(Sitkoff, Sharp & Honig, *J. Phys. Chem.* **98**, 1978, 1994).

It is **united-atom for aliphatic carbons**, which is the consequence worth
knowing here. An aliphatic carbon is inflated to 2.0 Å to absorb the hydrogens
bonded to it, and those hydrogens are then given **radius 0.0 and charge 0.0**.
Polar hydrogens — those on N or O — keep radius 1.0 and carry the charge.
The full radius set in this data is:

| Atom | Radius (Å) |
| --- | ---: |
| aliphatic H (on C) | 0.0 |
| polar H (on N, O) | 1.0 |
| O | 1.4 |
| N | 1.5 |
| carbonyl / aromatic C | 1.7 |
| S | 1.85 |
| aliphatic C (united with its H) | 2.0 |

So a "hydrogen" in a PARSE PQR is either invisible to the solver (0.0) or a
charged point with a small radius (1.0), depending on what it is bonded to.

- `atom_charge` = \(q_i\), in *e*.
- `atom_radius` = \(R_i\), in Å.
- `atom_pqr_resname` records the protonation state pdb2pqr chose (HIS → HID/HIE/HIP,
  etc.); `atom_resname` is the original name from the source PDB.

### `total_charge`

```
total_charge = Σ_i q_i        over every atom in the PQR
```

A sum over **atoms**, not over amino acids — but since every atom belongs to
exactly one residue, it equals `Σ residue_charge` too. It is the net charge of
**everything solved together**: for a complex store that is both chains; for a
monomer store it is that one chain. Units: *e*.

It is not necessarily an integer — it is a sum of forcefield partial charges,
and non-integer values reflect PROPKA fractional-state choices and truncated
residues. In practice it comes out within ~1e-12 of an integer for these files.

### `residue_charge`

```
residue_charge[r] = Σ_{i ∈ residue r} q_i
```

For standard residues at their assigned protonation state this reproduces the
formal charge (ARG/LYS +1, ASP/GLU −1, neutral otherwise). Units: *e*. `NaN`
for any residue pdb2pqr dropped (`residue_in_pqr` false).

---

## 2. The potential — APBS

**APBS 3.4.1** solves the **linearised Poisson–Boltzmann equation**

$$\nabla\cdot\left[\varepsilon(\mathbf r)\nabla\phi(\mathbf r)\right]
 - \bar\kappa^2(\mathbf r)\,\phi(\mathbf r) = -4\pi\rho_{\text{solute}}(\mathbf r)$$

on a focused multigrid (`mg-auto`), with:

| Parameter | Value | APBS keyword |
| --- | --- | --- |
| Solute dielectric | 2.0 | `pdie` |
| Solvent dielectric | 78.54 | `sdie` |
| Ionic strength | 0.150 M 1:1 salt, ion radius 2.0 Å | `ion` |
| Temperature | 298.15 K | `temp` |
| Solvent probe | 1.4 Å | `srad` |
| Dielectric surface | smoothed molecular (`smol`), window 0.3 Å | `srfm`, `swin` |
| Charge discretisation | cubic B-spline (`spl2`) | `chgm` |
| Boundary condition | single Debye–Hückel (`sdh`) | `bcfl` |
| Grid | ~0.5 Å target fine spacing | `dime`, `fglen`, `cglen` |

**Units.** APBS writes the potential in **kT/e** — i.e. \(\phi\) already divided
by \(k_BT/e\), so the stored numbers are dimensionless. At 298.15 K:

```
1 kT/e = 25.693 mV = 0.5925 kcal·mol⁻¹·e⁻¹
```

Multiply by 25.693 for millivolts. This is the same convention APBS/PyMOL use
by default, so a ±5 ramp in PyMOL is ±5 kT/e ≈ ±129 mV.

`potential_grid` (when stored) is the raw volume, with `grid_origin`,
`grid_spacing` and `grid_shape` giving the lattice: value `[i,j,k]` sits at
`grid_origin + (i,j,k) * grid_spacing`.

---

## 3. The surface — Shrake–Rupley

The **solvent-accessible surface (SAS)** is sampled directly, rather than
meshed. For each atom *i*, `sphere_points` = 100 near-uniform directions
(golden-spiral construction) are placed on a sphere of radius \(R_i + p\) with
probe \(p\) = 1.4 Å. A point survives if it lies outside every *other* atom's
probe-expanded sphere:

```
point k on atom i is exposed  ⟺  |x_k − x_j| ≥ R_j + p   for all j ≠ i
```

Each surviving point carries the area it represents:

```
surface_point_area[k] = 4π (R_i + p)² / M        M = sphere_points = 100
```

and per-atom SASA is the surviving fraction of the full sphere:

```
SASA_i = 4π (R_i + p)² · (exposed points on i) / M
```

- `residue_sasa[r] = Σ_{i ∈ r} SASA_i`, in Å². By construction this exactly
  equals the sum of `surface_point_area` over that residue's points.
- `surface_xyz`, `surface_potential`, `surface_atom_index`,
  `surface_residue_index`, `surface_point_area` describe the point cloud.

> **This is the SAS, not the SES.** Points sit on the probe-*centre* surface
> (\(R_i + 1.4\)), not the probe-contact (Connolly/molecular) surface that
> PyMOL draws with `show surface`. The potential is therefore sampled ~1.4 Å
> out from the van der Waals surface. That is the usual convention for
> residue-level electrostatic descriptors — it is where a water molecule's
> centre actually sits — but it means these numbers are systematically closer
> to zero than potentials read off the molecular surface, and it is why the
> PyMOL picture and the per-residue numbers are not expected to agree
> point-for-point.

---

## 4. Sampling the potential at the surface

The volumetric solution is **trilinearly interpolated** at each surface point
and at each atom centre (`scipy.ndimage.map_coordinates`, `order=1`, edge
clamping):

```
surface_potential[k] = φ(x_k)     interpolated from potential_grid
atom_potential[i]    = φ(x_i)     interpolated at the atom centre
```

`atom_potential` is the potential *inside* the low-dielectric solute, which is
dominated by that atom's own charge. It is useful for per-atom bookkeeping but
is **not** a surface quantity — use the residue statistics for that.

---

## 5. Per-residue statistics — the area-weighted surface average

`residue_potential_mean` is the discretised surface integral over that
residue's solvent-accessible patch:

$$\bar\phi_r=\frac{\int_{\mathrm{SAS}_r}\phi\,dA}{\int_{\mathrm{SAS}_r}dA}
\;\longrightarrow\;
\frac{\sum_{k\in r}\phi_k\,a_k}{\sum_{k\in r}a_k},
\qquad a_k=\frac{4\pi (R_{i(k)}+p)^2}{M}$$

**The weighting is not cosmetic.** Because PARSE radii span 0.0–2.0 Å, the area
represented by a point differs by up to **5.9×** between an aliphatic hydrogen
(radius 0.0) and a united aliphatic carbon (radius 2.0). An unweighted mean over points silently over-counts small-radius atoms;
against the correctly weighted value it differs by a median of 0.03 kT/e but by
**up to 4.2 kT/e** on individual residues. Stores written before this was fixed
can be corrected in place with `python -m apbs_analysis.migrate_area_weighting`
(it recomputes from the stored point cloud — no new APBS run), and carry
`residue_statistics_area_weighted = True` once they have been.

The other per-residue fields:

| Field | Definition |
| --- | --- |
| `residue_potential_std` | area-weighted standard deviation, \(\sqrt{\langle\phi^2\rangle_a-\bar\phi^2}\) |
| `residue_potential_min` / `_max` | plain min/max over the residue's points (unweighted — weighting is meaningless for an extremum) |
| `residue_surface_point_count` | number of surviving SAS points |
| `residue_sasa` | Å², = Σ a_k |

A residue with `residue_surface_point_count == 0` is fully buried: it has no
solvent-accessible surface, so its mean/min/max/std are **NaN**, not zero.
`load_residue_table(..., exposed_only=True)` drops those rows.

### `mean_surface_potential` in the summary CSV

The **SASA-weighted average over the whole molecule**:

```
mean_surface_potential = Σ_r (φ̄_r · SASA_r) / Σ_r SASA_r      over exposed residues
```

which is algebraically the same as averaging φ over the entire SAS. It is *not*
an unweighted mean of per-residue means — that would count a barely-exposed
residue the same as a fully exposed one.

For the same quantity per chain, use `analysis.interface_residues()`.

---

## 6. Joining to other data

`residue_aa_id` is the join key: entry *i* is the *(i+1)*-th residue in the
source PDB's file order, matching the sequential counter
`create_protein_graph_structure.py` assigns as `aa_id`. Residue arrays span
every residue of the source PDB, so they align 1:1 with graph nodes.

Do **not** join on `(chain, residue_number)` — two of the 84 files
(`3sgb`, `7joe`) had insertion codes stripped and reuse a residue number within
a chain, so that key is not unique. `residue_insertion_code` is retained but is
empty for these files.

**`residue_number` is not the deposited PDB numbering.** 80 of the 84 input
files were renumbered sequentially 1..N across the whole complex before this
pipeline saw them (`1acb` chain E is 1–241, chain I 242–304). The remaining
four — `2bkr`, `2wmp`, `3sgb`, `7joe` — are nearly sequential with gaps or
duplicates, which is where the duplicate-numbering problem comes from. So these
numbers cannot be cross-referenced to literature or UniProt positions without
realigning to the deposited entry first.

For monomer stores, `residue_parent_aa_id` (a group attribute) maps each
monomer residue onto its `aa_id` in the parent complex.

---

## 7. Monomers

Monomer maps are computed by removing one chain from the **already-solved
complex**, keeping:

- **the same coordinates, charges and radii** — taken from the complex's PQR
  rather than a fresh pdb2pqr run, so a residue carries an identical charge in
  both. Re-running PROPKA on an isolated chain can change titration states
  (a buried salt bridge becomes solvent-exposed), which would put part of any
  complex-minus-monomers difference into the protonation rather than the
  interaction.
- **the same grid** — the complex's exact box, with the centre pinned
  explicitly. APBS's default `cgcent mol 1` centres on the molecule, which for
  a monomer sits up to ~0.1 Å from the complex's centre; that is enough to
  misalign voxels and make a difference map meaningless.

This is the rigid-body, fixed-charge decomposition. With it, the maps are on a
bit-identical lattice and

```
Δφ(r) = φ_complex(r) − φ_chainA(r) − φ_chainB(r)
```

is a valid voxel-wise difference. Verified: origin, spacing and shape match
exactly between a complex and both its monomers.

What this decomposition does **not** include: no conformational relaxation of
the unbound monomers (they keep their bound geometry), and no protonation
change on binding. Both are deliberate — they are what make the difference
interpretable as an electrostatic interaction rather than a mixture of effects.
If the *relaxed* monomer is the target instead, split chains to PDB with
`structure_prep.split_chains` and run them through the normal PDB entry point,
which re-runs pdb2pqr and gives each its own box.

---

## 8. Interaction potential (complex − monomers)

### The volumetric map

```
Δφ(r) = φ_complex(r) − Σ_chains φ_chain(r)
```

computed voxel by voxel on the shared lattice (`interaction_map.py`). It
refuses to run if the grids do not match exactly.

**Why this is not identically zero — the part worth understanding.** The
linearised PB equation is *linear in the charge density* for a fixed dielectric
and ion-exclusion map. The monomers inherit the complex's charges verbatim, so

```
ρ_complex(r) = ρ_A(r) + ρ_B(r)      exactly (verified: max |Δq| = 0)
```

If ε(r) and κ̄(r) were also the same in all three calculations, superposition
would give φ_complex = φ_A + φ_B and Δφ ≡ 0. They are *not* the same: in the
complex, the interface region is low-dielectric protein interior with ions
excluded; in each isolated monomer, that same region is high-dielectric solvent
with ions present.

So **Δφ isolates the effect of the dielectric and ionic boundary changing on
binding** — desolvation of the interface and the altered screening — with the
Coulombic part of the two charge sets exactly cancelled out. That is precisely
the non-trivial part of binding electrostatics; the additive part carries no
information.

This is verifiable in the data. |Δφ| for `1acb`, by distance from the contact zone:

| Distance from contact | median &#124;Δφ&#124; (kT/e) |
| --- | ---: |
| 0–4 Å | 3.33 |
| 4–6 Å | 0.67 |
| 6–8 Å | 0.33 |
| 8–12 Å | 0.14 |
| 12–20 Å | 0.033 |
| 20–40 Å | 0.0065 |

A ~500-fold decay, localised at the interface, exactly as a boundary-change
effect must be. If the grids were misaligned the signal would instead be
scattered over the whole surface — which is why the map doubles as a check
that the alignment held.

### Per-residue

Two columns are written, because there are two defensible definitions:

**`interaction_potential`** (use this one) — both maps sampled at the **same**
points, namely the monomer's SAS points for that residue:

```
Δφ_r = Σ_k [φ_complex(x_k) − φ_monomer(x_k)] a_k / Σ_k a_k
```

a pure field change at fixed geometry.

**`interaction_potential_own_surface`** — each state averaged over its own
surface, `φ̄_r(complex) − φ̄_r(monomer)`. This additionally folds in the surface
the residue *loses* on binding, since an interface residue's accessible patch
shrinks and the two averages are then over different areas.

Away from the interface the two agree to ~1e-3 kT/e. **On interface residues
they differ by a mean of ~4 kT/e and by up to 32 kT/e**, and correlate only
0.51–0.84 — so the choice matters exactly where the analysis is interesting.

**`buried_sasa`** = `sasa_monomer − sasa_complex`, the SASA a residue loses on
binding (Å²). Purely geometric; the usual interface definition. The default
cutoff calling a residue "interface" is > 1 Å².

For `1acb` this gives a 45-residue interface (27 in chain E, 18 in chain I)
burying 1,608 Å² total — a normal size for a protease–inhibitor complex — with
the largest Δφ on a contiguous surface patch of chain E plus the inhibitor
residue burying the most area (180 Å²).

### What it is not

Δφ is a **potential**, in kT/e, not an energy. It is not ΔG_binding and not a
ΔΔG. Turning it into an energy requires integrating against the charge
distribution (and, for a free energy, the solvation terms APBS reports with
`calcenergy`), which this pipeline does not currently do.

---

## 9. Screened approach curves (monomer brought in from long range)

`approach.py` walks one chain from a long-range separation to its bound pose and
evaluates the screened interaction at every step.

### Screening length

The Debye length is

$$\lambda_D=\sqrt{\frac{\varepsilon_0\varepsilon_r k_BT}{2Ie^2}}$$

At **I = 0.150 M** (the standard figure for cytosolic ionic strength: K⁺ ≈ 140 mM
with Na⁺, Cl⁻, Mg²⁺, phosphates and charged metabolites making up the rest),
298.15 K and ε = 78.54, this gives **λ_D = 7.86 Å**. That is the same ionic
strength the APBS runs used, so both halves of the package screen identically.

The choice is robust: 310 K with water's dielectric at 37 °C gives 7.78 Å, a
1% change. 25 °C is kept for consistency with the APBS work.

### Interaction model

Every atom carries its PARSE partial charge. The stationary chain B creates a
Debye–Hückel potential

$$\phi_B(\mathbf r)=\frac{1}{4\pi\varepsilon_0\varepsilon_r}\sum_{j\in B}q_j\frac{e^{-|\mathbf r-\mathbf r_j|/\lambda_D}}{|\mathbf r-\mathbf r_j|}$$

and the interaction energy is that potential contracted with A's charges:

$$U(d)=\sum_{i\in A}q_i\,\phi_B(\mathbf r_i(d))$$

In practical units, `U[kcal/mol] = 332.0637 · Σ qᵢqⱼ e^(−r/λ) / (ε·r)` with
charges in *e* and r in Å. Divide by kT = 0.5925 kcal/mol for kT.

**U is symmetric** — contracting φ_A with B's charges gives the same number — so
"of B on A" fixes only which chain the per-residue decomposition is attributed
to, not the total. Both decompositions are written, and each sums to U (not 2U).

Every atom pair is summed exactly, with no distance cutoff: `cdist` makes the
full sum as fast as a 60 Å cutoff, and truncation biases the total
systematically low (~0.2–0.3% at 40 Å) because every dropped term shares the
sign of its charge product.

### Why Debye–Hückel and not PB at every step

A PB solve per step is ~20 s per structure, i.e. roughly 500 CPU-hours over 84
dimers × 101 steps. More importantly, a screened interaction with a biological
Debye length is exactly what Debye–Hückel gives in closed form.

Its limitations are real and matter at contact: a **uniform solvent dielectric
with no low-dielectric protein interior**, no salt exclusion from the protein
volume, and no desolvation. Charges buried at an interface are in reality
screened far less than ε = 78.54 implies, so contact energies here are
**underestimates**. The model is at its best over most of the trajectory (well
separated chains) and weakest exactly where the APBS maps in the rest of this
package are the better tool.

### The path

Generated *outward* from the bound pose and then reversed, so the final frame is
the crystal structure exactly rather than by convergence.

A straight pull-out is tried first, along the direction chosen by ranking
candidates lexicographically — clashes first, then reach. **60 of 84 dimers
separate cleanly this way.** Geometry uses **heavy atoms only**: hydrogens are
model-built and routinely sit 1.2–1.4 Å from a partner atom, which is a normal
hydrogen bond, and judging paths on all-atom distances flags every native
interface as clashing. Electrostatics still uses every atom.

Where no straight line works, the path is **steered** in two phases: while the
chains are close, step in whichever direction opens the largest gap (letting a
chain slide along a groove), plus small rotations when translation alone is not
enough; once clear, run straight out to the required separation.

**Some pairs cannot be separated by rigid-body motion at all.** For 1kfu, zero
of 167 candidate directions give a clash-free translation — 45% of rays cast
from the small subunit's centroid hit its partner, and the best direction still
squeezes to 0.7 Å for a 15 Å stretch. The subunits interdigitate. 34 of 84
dimers have some steric overlap on the path; those frames are flagged
`steric_overlap` in the trajectory CSV and their energies are **not physical**
— drop them before fitting anything.

Two coordinates are reported. `path_length_angstrom` is arc length and is
monotone by construction. `displacement_angstrom` is the straight-line
separation from the bound pose; it is monotone on linear paths, but a steered
path arcs and its centroid can track back — measured at most 0.35 Å, on 14 of
84 dimers. Use path length as the progress coordinate when that matters.

### What is reported

Per step: both energies (screened and unscreened, so the screening factor is
visible), the monopole–monopole estimate `332·q_A·q_B·e^(−R/λ)/(εR)` (comparing
it to the full sum shows when higher multipoles matter), the potential
statistics at the moving chain, geometry, and the overlap flag. Per residue,
per step: that residue's share of U, for both chains.

### Results over the 84 dimers

| | |
| --- | --- |
| Attractive at the bound pose | **84/84** |
| Interaction energy at contact | median −3.62, range −15.41 to −0.00 kcal/mol |
| \|U\| > 1 kT at contact | 82/84 |
| Screening factor at contact (screened/unscreened) | median 0.749 |
| Onset of \|U\| > 0.5 kT | median 11.0 Å of separation (≈1.4 λ_D) |
| Sign flips between long range and contact | **16/84** |

Every interface is electrostatically complementary at contact. The 16 sign
flips are the interesting cases: net-charge repulsion at long range that
becomes attraction on close approach, i.e. steering that reverses as the
complementary patches come into register.

---

## 10. Known limitations

- **Linearised PB.** `lpbe` is accurate for these systems at 0.15 M but
  underestimates potentials in regions of very high charge density. Switch with
  `--pbe-solver npbe` if that matters.
- **Grid resolution.** Fine spacing is ~0.43–0.50 Å, coarsened for very large
  complexes if the grid would exceed `--memory-ceiling-mb`. The stored
  `grid_spacing` is the truth for each model.
- **Single conformation.** One structure, no ensemble or thermal averaging.
- **Truncated side chains.** 6 residues across the 84 (all in `1euv`) had
  incomplete density and were cut to ALA/GLY, losing their side-chain charge;
  `residue_truncated` flags them. 192 residues are flagged `residue_incomplete`
  (missing atoms in the input) but were successfully rebuilt by pdb2pqr.
- **SAS not SES**, as described in §3.
