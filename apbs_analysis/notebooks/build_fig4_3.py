"""Build dissertation Fig. 4.3: Poisson-Boltzmann surface electrostatics.

    (a) molecular surface of one heterodimer coloured by phi (kT/e)
    (b) the two monomers opened out, each coloured by its own phi
    (c) per-residue interaction potential dphi_r on the same open-book faces
    (d) dphi_r versus buried SASA per residue, over all 84 targets

    python -m apbs_analysis.notebooks.build_fig4_3 --target 1ay7
    python -m apbs_analysis.notebooks.build_fig4_3 --preview 1ay7 4hwi 2grn 2fhz

Everything expensive is cached next to the APBS stores, so re-running only redoes
the composition. --refresh-data / --refresh-renders force the earlier stages.

The interaction potential follows the dissertation's definition,
    dphi = phi_complex - sum_chains phi_chain,
sampled per residue on the monomer's own surface points and area-weighted. Note
that apbs_analysis.interaction_map.interaction_residues() subtracts only the
residue's *own* chain, which leaves the partner's Coulomb field in; its values are
kept as column `interaction_potential` and selectable here with --dphi own.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from apbs_analysis.common import LOCAL_OUTPUT_DIR
from apbs_analysis.dx_grid import DxGrid
from apbs_analysis.interaction_map import interaction_residues

HERE = Path(__file__).resolve().parent
RENDERER = HERE / "fig4_3_render.py"
THESIS_FIG_DIR = Path.home() / "Desktop" / "phd_thesis" / "chapter_4_figs"
PYMOL = shutil.which("pymol") or "/opt/homebrew/bin/pymol"

INTERFACE_CUTOFF_A2 = 1.0          # buried SASA above which a residue is "interface"
PHI_MAX = 5.0                      # kT/e, the conventional APBS ramp
DPHI_MAX = 10.0                    # kT/e, ~90th percentile of |dphi_r| on interface residues
# ColorBrewer RdBu and PuOr, both colour-blind safe. Negative = red / orange.
PHI_STOPS = ["#b2182b", "#ef8a62", "#f7f7f7", "#67a9cf", "#2166ac"]
DPHI_STOPS = ["#b35806", "#f1a340", "#f7f7f7", "#998ec3", "#542788"]
TEXT_WIDTH_IN = 469.75502 / 72.27  # \textwidth of dissertation.tex


# ------------------------------------------------------------------------ data
def stores(output_dir: Path) -> tuple[Path, Path]:
    return output_dir / "targets_84_apbs_surface.hdf5", output_dir / "targets_84_monomers_apbs_surface.hdf5"


def _grid(group: h5py.Group, values: np.ndarray) -> DxGrid:
    return DxGrid(
        origin=np.asarray(group.attrs["grid_origin"], dtype=float),
        spacing=np.asarray(group.attrs["grid_spacing"], dtype=float),
        values=values,
    )


def residue_table(output_dir: Path, refresh: bool = False) -> pd.DataFrame:
    """Per-residue dphi and buried SASA for every residue of every complex (cached)."""
    path = output_dir / "targets_84_interaction_residues.csv"
    if path.is_file() and not refresh:
        table = pd.read_csv(path)
        if "interaction_potential_both" in table:
            return table

    complex_store, monomer_store = stores(output_dir)
    with h5py.File(complex_store, "r") as handle:
        ids = sorted(name for name in handle if not name.startswith("__"))

    frames, both_parts = [], []
    with h5py.File(complex_store, "r") as complexes, h5py.File(monomer_store, "r") as monomers:
        for index, complex_id in enumerate(ids, start=1):
            frames.append(interaction_residues(complex_store, monomer_store, complex_id))
            complex_group = complexes[complex_id]
            chain_names = sorted(
                name for name in monomers if monomers[name].attrs.get("parent_complex_id") == complex_id
            )
            delta = complex_group["potential_grid"][:].astype(np.float64)
            for name in chain_names:
                delta -= monomers[name]["potential_grid"][:].astype(np.float64)
            delta_grid = _grid(complex_group, delta.astype(np.float32))
            for name in chain_names:
                group = monomers[name]
                parent = group.attrs["residue_parent_aa_id"]
                residue_index = group["surface_residue_index"][:]
                area = group["surface_point_area"][:].astype(np.float64)
                sampled = delta_grid.sample(group["surface_xyz"][:]).astype(np.float64)
                total = np.bincount(residue_index, area, len(parent))
                both_parts.append(pd.DataFrame({
                    "complex_id": complex_id,
                    "chain": str(group.attrs["chain"]),
                    "residue_aa_id": parent,
                    "interaction_potential_both": np.where(
                        total > 0,
                        np.bincount(residue_index, sampled * area, len(parent)) / np.maximum(total, 1e-12),
                        np.nan,
                    ),
                }))
            if index % 21 == 0:
                print(f"  residue table {index}/{len(ids)}", flush=True)

    table = pd.concat(frames, ignore_index=True).merge(
        pd.concat(both_parts, ignore_index=True),
        on=["complex_id", "chain", "residue_aa_id"],
        how="left",
        validate="one_to_one",
    )
    table.to_csv(path, index=False)
    return table


def complementarity_table(output_dir: Path, residues: pd.DataFrame, refresh: bool = False) -> pd.DataFrame:
    """Electrostatic complementarity per complex (cached).

    EC = -mean over both chains of corr(phi_self, phi_partner), each monomer's own map
    and its partner's map sampled at the same interface surface points. Positive EC
    means one side's positive patches face the other side's negative ones.
    """
    path = output_dir / "targets_84_electrostatic_complementarity.csv"
    if path.is_file() and not refresh:
        cached = pd.read_csv(path)
        if {"complex_id", "EC"} <= set(cached.columns):
            return cached

    complex_store, monomer_store = stores(output_dir)
    rows = []
    with h5py.File(monomer_store, "r") as monomers:
        for complex_id in sorted(residues.complex_id.unique()):
            chain_names = sorted(
                name for name in monomers if monomers[name].attrs.get("parent_complex_id") == complex_id
            )
            groups = [monomers[name] for name in chain_names]
            grids = [_grid(g, g["potential_grid"][:]) for g in groups]
            correlations, points = [], 0
            for self_index in (0, 1):
                group = groups[self_index]
                chain_rows = residues[(residues.complex_id == complex_id) & (residues.chain == str(group.attrs["chain"]))]
                interface_index = np.flatnonzero(chain_rows.buried_sasa.to_numpy() > INTERFACE_CUTOFF_A2)
                mask = np.isin(group["surface_residue_index"][:], interface_index)
                if mask.sum() < 30:
                    continue
                xyz = group["surface_xyz"][:][mask]
                points += int(mask.sum())
                correlations.append(
                    np.corrcoef(grids[self_index].sample(xyz), grids[1 - self_index].sample(xyz))[0, 1]
                )
            complex_rows = residues[residues.complex_id == complex_id]
            rows.append({
                "complex_id": complex_id,
                "chains": "/".join(str(g.attrs["chain"]) for g in groups),
                "EC": -float(np.mean(correlations)) if correlations else np.nan,
                "interface_points": points,
                "interface_residues": int((complex_rows.buried_sasa > INTERFACE_CUTOFF_A2).sum()),
                "buried_sasa_total": float(complex_rows.buried_sasa.sum()),
            })
    table = pd.DataFrame(rows).sort_values("EC", ascending=False).reset_index(drop=True)
    table.to_csv(path, index=False)
    return table


# --------------------------------------------------------------------- renders
def render_target(output_dir: Path, residues: pd.DataFrame, complex_id: str, render_dir: Path,
                  px_per_angstrom: float, dphi_column: str, refresh: bool = False) -> dict:
    geometry_path = render_dir / "geometry.json"
    if geometry_path.is_file() and not refresh:
        return json.loads(geometry_path.read_text())

    rows = residues[residues.complex_id == complex_id]
    chains = sorted(rows.chain.unique())
    if len(chains) != 2:
        raise ValueError(f"{complex_id} has {len(chains)} chains; the open book needs exactly two.")
    for chain in chains:
        numbers = rows[rows.chain == chain].residue_number
        if numbers.duplicated().any():
            raise ValueError(
                f"{complex_id}:{chain} has duplicate residue numbers (dropped insertion codes), "
                "so per-residue colours cannot be mapped onto the PQR unambiguously."
            )
    complexes_dir, monomers_dir = output_dir / "pymol_complexes", output_dir / "pymol_monomers"
    config = {
        "output_dir": str(render_dir),
        "px_per_angstrom": px_per_angstrom,
        "chain_a": chains[0],
        "chain_b": chains[1],
        "complex_pqr": str(complexes_dir / f"{complex_id}.pqr"),
        "complex_map": str(complexes_dir / f"{complex_id}_potential.dx.gz"),
        "monomer_pqr": {c: str(monomers_dir / f"{complex_id}_{c}.pqr") for c in chains},
        "monomer_map": {c: str(monomers_dir / f"{complex_id}_{c}_potential.dx.gz") for c in chains},
        "phi_stops": PHI_STOPS,
        "phi_max": PHI_MAX,
        "dphi_stops": DPHI_STOPS,
        "dphi_max": DPHI_MAX,
        "dphi_residue": {
            c: {
                str(int(n)): (None if not np.isfinite(v) else float(v))
                for n, v in zip(rows[rows.chain == c].residue_number, rows[rows.chain == c][dphi_column])
            }
            for c in chains
        },
        "interface_residues": {
            c: [str(int(n)) for n in rows[(rows.chain == c) & (rows.buried_sasa > INTERFACE_CUTOFF_A2)].residue_number]
            for c in chains
        },
    }
    for key in ("complex_pqr", "complex_map"):
        if not Path(config[key]).is_file():
            raise FileNotFoundError(config[key])
    render_dir.mkdir(parents=True, exist_ok=True)
    config_path = render_dir / "config.json"
    config_path.write_text(json.dumps(config))
    result = subprocess.run([PYMOL, "-cq", str(RENDERER), "--", str(config_path)],
                            capture_output=True, text=True)
    if "FIG43_RENDER_OK" not in result.stdout:
        sys.stderr.write(result.stdout[-4000:] + result.stderr[-4000:])
        raise RuntimeError(f"PyMOL render failed for {complex_id}")
    return json.loads(geometry_path.read_text())


# ---------------------------------------------------------------------- styling
def latex_available() -> bool:
    """MacTeX installs to /Library/TeX/texbin, which is often missing from a venv's PATH."""
    if shutil.which("latex"):
        return True
    texbin = Path("/Library/TeX/texbin")
    if (texbin / "latex").exists():
        os.environ["PATH"] = f"{texbin}{os.pathsep}{os.environ.get('PATH', '')}"
        return True
    return False


def thesis_style() -> None:
    mpl.rcParams.update({
        "text.usetex": latex_available(),
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        # Matches the other Chapter 4 figures: boxed axes, ~10 pt labels at \textwidth.
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "savefig.dpi": 600,
    })


def colormap(stops, name):
    return LinearSegmentedColormap.from_list(name, stops)


def angstrom() -> str:
    return r"\AA" if mpl.rcParams["text.usetex"] else r"$\mathrm{\AA}$"


def kt_e() -> str:
    return r"$k_BT/e$"


def panel_label(ax, letter, x=-0.02, y=1.0):
    ax.text(x, y, rf"\textbf{{({letter})}}" if mpl.rcParams["text.usetex"] else f"({letter})",
            transform=ax.transAxes, ha="right", va="top", fontsize=12,
            fontweight="normal" if mpl.rcParams["text.usetex"] else "bold")


def load_rgba(path: Path) -> np.ndarray:
    return plt.imread(str(path))


def scale_bar(ax, x, y, length, label):
    ax.plot([x, x + length], [y, y], color="black", lw=1.4, solid_capstyle="butt", clip_on=False)
    ax.text(x + length / 2, y - 1.5, label, ha="center", va="top", fontsize=7.5)


def footprint_outline(ax, mask_path: Path, extent):
    """Trace the interface footprint from the flat black/white mask render."""
    rgba = load_rgba(mask_path)
    dark = (rgba[..., :3].mean(axis=-1) < 0.5) & (rgba[..., 3] > 0.5)
    if not dark.any():
        return
    ax.contour(dark.astype(float), levels=[0.5], extent=extent, origin="upper",
               colors="black", linewidths=0.7, linestyles=[(0, (2.5, 1.5))])


def open_book(ax, geometry, render_dir: Path, key: str, chain_a: str, chain_b: str, gap: float,
              labels: dict[str, str] | None, outline: bool):
    x0, x1, y0, y1 = geometry["frame_open"]
    width = x1 - x0
    extents = {
        chain_b: (-gap / 2 - width, -gap / 2, y0, y1),
        chain_a: (gap / 2, gap / 2 + width, y0, y1),
    }
    for chain in (chain_a, chain_b):
        ax.imshow(load_rgba(render_dir / geometry["images"][key][chain]),
                  extent=extents[chain], interpolation="lanczos")
        if outline:
            footprint_outline(ax, render_dir / geometry["images"]["mask"][chain], extents[chain])
        if labels:
            left, right, bottom, _ = extents[chain]
            ax.text((left + right) / 2, bottom - 1.0, labels[chain], ha="center", va="top", fontsize=8)
    ax.set_xlim(-gap / 2 - width, gap / 2 + width)
    ax.set_ylim(y0 - (7 if labels else 1), y1)
    ax.set_aspect("equal")
    ax.axis("off")
    return extents


# --------------------------------------------------------------------- compose
def compose(residues: pd.DataFrame, geometry: dict, render_dir: Path, complex_id: str,
            chain_labels: dict[str, str], dphi_column: str, output_path: Path) -> None:
    thesis_style()
    chain_a, chain_b = sorted(residues[residues.complex_id == complex_id].chain.unique())
    phi_cmap, dphi_cmap = colormap(PHI_STOPS, "phi"), colormap(DPHI_STOPS, "dphi")
    gap = 6.0

    frame_open = geometry["frame_open"]
    book_w = 2 * (frame_open[1] - frame_open[0]) + gap
    book_h = frame_open[3] - frame_open[2] + 7
    fa = geometry["frame_a"]
    a_w, a_h = fa[1] - fa[0], fa[3] - fa[2]

    fig = plt.figure(figsize=(TEXT_WIDTH_IN, TEXT_WIDTH_IN * 0.66))
    outer = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.72, 0.055], height_ratios=[1, 1],
                             left=0.035, right=0.925, top=0.975, bottom=0.075, wspace=0.08, hspace=0.16)

    # (a) complex, same px/A as the open book so the three renders share one scale.
    ax_a = fig.add_subplot(outer[0, 0])
    ax_a.imshow(load_rgba(render_dir / geometry["images"]["a"]), extent=(fa[0], fa[1], fa[2], fa[3]),
                interpolation="lanczos")
    ax_a.set_xlim(fa[0], fa[1])
    ax_a.set_ylim(fa[2] - 11, fa[3])
    ax_a.set_aspect("equal")
    ax_a.axis("off")
    # The frame is centred on the interface, so chain B occupies x < 0 and chain A x > 0.
    ax_a.text(fa[0] / 2, fa[2] - 1.0, chain_labels[chain_b], ha="center", va="top", fontsize=8)
    ax_a.text(fa[1] / 2, fa[2] - 1.0, chain_labels[chain_a], ha="center", va="top", fontsize=8)
    scale_bar(ax_a, fa[0] + 2, fa[2] - 7.0, 10, f"10 {angstrom()}")
    panel_label(ax_a, "a", x=0.0)

    # (b) monomers, own phi.
    ax_b = fig.add_subplot(outer[0, 1])
    open_book(ax_b, geometry, render_dir, "b", chain_a, chain_b, gap, chain_labels, outline=True)
    panel_label(ax_b, "b", x=0.0)
    cax_phi = fig.add_subplot(outer[0, 2])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(-PHI_MAX, PHI_MAX), cmap=phi_cmap),
                      cax=cax_phi, extend="both")
    cb.set_label(rf"$\phi$ ({kt_e()})", labelpad=2)
    cb.set_ticks([-PHI_MAX, -PHI_MAX / 2, 0, PHI_MAX / 2, PHI_MAX])
    cb.outline.set_linewidth(0.5)

    # (c) monomers, per-residue dphi_r.
    ax_c = fig.add_subplot(outer[1, 1])
    open_book(ax_c, geometry, render_dir, "c", chain_a, chain_b, gap, chain_labels, outline=True)
    panel_label(ax_c, "c", x=0.0)
    cax_dphi = fig.add_subplot(outer[1, 2])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(-DPHI_MAX, DPHI_MAX), cmap=dphi_cmap),
                      cax=cax_dphi, extend="both")
    cb.set_label(rf"$\Delta\phi_r$ ({kt_e()})", labelpad=2)
    cb.set_ticks([-DPHI_MAX, -DPHI_MAX / 2, 0, DPHI_MAX / 2, DPHI_MAX])
    cb.outline.set_linewidth(0.5)

    # (d) dphi_r vs buried SASA, all targets.
    ax_d = fig.add_subplot(outer[1, 0])
    panel_d(ax_d, residues, dphi_column, complex_id)
    panel_label(ax_d, "d", x=-0.30)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def panel_d(ax, residues: pd.DataFrame, dphi_column: str, highlight: str | None) -> None:
    """dphi_r against buried SASA for every interface residue of every target.

    The signed median stays near zero at all burial levels; what grows is the
    magnitude, so the band is the interquartile range, which fans out with burial.
    Residues that lose no surface on binding are summarised in text rather than
    plotted, because they would form a single column at x = 0.
    """
    data = residues.dropna(subset=[dphi_column])
    interface = data[data.buried_sasa > INTERFACE_CUTOFF_A2]
    exposed = data[data.buried_sasa <= INTERFACE_CUTOFF_A2]
    x, y = interface.buried_sasa.to_numpy(), interface[dphi_column].to_numpy()

    ax.set_yscale("symlog", linthresh=1.0, linscale=0.7)
    ax.axhline(0, color="0.55", lw=0.5, zorder=0)
    ax.scatter(x, y, s=2.0, c="0.45", alpha=0.25, linewidths=0, rasterized=True, zorder=1)

    edges = np.array([1, 10, 20, 35, 50, 70, 95, 130, 230])
    centers, median, q1, q3 = [], [], [], []
    for left, right in zip(edges[:-1], edges[1:]):
        in_bin = (x > left) & (x <= right)
        if in_bin.sum() < 25:
            continue
        centers.append(np.median(x[in_bin]))
        median.append(np.median(y[in_bin]))
        q1.append(np.percentile(y[in_bin], 25))
        q3.append(np.percentile(y[in_bin], 75))
    ax.fill_between(centers, q1, q3, color="0.15", alpha=0.18, lw=0, zorder=2, label="interquartile range")
    ax.plot(centers, median, color="black", lw=1.2, zorder=4, label="median")

    if highlight is not None:
        own = interface[interface.complex_id == highlight]
        ax.scatter(own.buried_sasa, own[dphi_column], s=11, facecolor="none", edgecolor="black",
                   linewidths=0.6, zorder=5, label=f"{highlight}, panels (a)--(c)")

    ax.set_xlim(0, np.percentile(x, 99.8))
    lim = np.abs(y).max() * 1.25
    ax.set_ylim(-lim, lim)
    ticks = [t for t in (-30, -10, -3, -1, 0, 1, 3, 10, 30) if abs(t) < lim]
    ax.set_yticks(ticks)
    ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: rf"${v:g}$"))
    ax.set_xlabel(rf"buried SASA ({angstrom()}$^2$)", labelpad=2)
    ax.set_ylabel(rf"$\Delta\phi_r$ ({kt_e()})", labelpad=1)
    legend = ax.legend(loc="upper right", fontsize=7.5, frameon=True, handlelength=1.4,
                       borderaxespad=0.3, borderpad=0.3, labelspacing=0.25)
    legend.get_frame().set(facecolor="white", edgecolor="none", alpha=0.85)
    n_targets = data.complex_id.nunique()
    ax.set_title(f"{len(interface):,} interface residues, {n_targets} targets", fontsize=8, pad=3)
    print(f"  panel (d): non-interface residues (n={len(exposed):,}) "
          f"median |dphi_r| = {exposed[dphi_column].abs().median():.3f} kT/e")


# ----------------------------------------------------------------------- preview
def preview_sheet(render_root: Path, geometries: dict[str, dict], path: Path) -> None:
    thesis_style()
    mpl.rcParams["text.usetex"] = False
    fig, axes = plt.subplots(len(geometries), 3, figsize=(10, 2.6 * len(geometries)))
    axes = np.atleast_2d(axes)
    for row, (complex_id, geometry) in enumerate(geometries.items()):
        directory = render_root / complex_id
        chain_a, chain_b = sorted(geometry["images"]["b"])
        fa = geometry["frame_a"]
        axes[row, 0].imshow(load_rgba(directory / geometry["images"]["a"]), extent=fa)
        axes[row, 0].set_title(f"{complex_id} (a)")
        for column, key in ((1, "b"), (2, "c")):
            open_book(axes[row, column], geometry, directory, key, chain_a, chain_b, 6.0, None, outline=True)
            axes[row, column].set_title(f"({key})")
        for ax in axes[row]:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default=str(LOCAL_OUTPUT_DIR), help="APBS output directory")
    parser.add_argument("--target", default="1ay7", help="Heterodimer shown in panels (a)-(c)")
    parser.add_argument("--chain-labels", default=None,
                        help="JSON {chain: label}, e.g. '{\"A\": \"RNase Sa\", \"B\": \"barstar\"}'")
    parser.add_argument("--preview", nargs="+", default=None, help="Render candidates to a contact sheet")
    parser.add_argument("--dphi", choices=("both", "own"), default="both",
                        help="'both': thesis Eq. (dphi = phi_complex - sum phi_chain). 'own': own chain only.")
    parser.add_argument("--px-per-angstrom", type=float, default=16.0)
    parser.add_argument("--output", default=str(THESIS_FIG_DIR / "Fig4_3_pb_electrostatics.pdf"))
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--refresh-renders", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser()
    dphi_column = "interaction_potential_both" if args.dphi == "both" else "interaction_potential"
    residues = residue_table(output_dir, refresh=args.refresh_data)
    ec = complementarity_table(output_dir, residues, refresh=args.refresh_data)
    render_root = output_dir / "fig4_3_renders" / args.dphi

    if args.preview:
        geometries = {}
        for complex_id in args.preview:
            rank = int(ec.index[ec.complex_id == complex_id][0]) + 1
            print(f"rendering {complex_id} (EC rank {rank}/{len(ec)})", flush=True)
            geometries[complex_id] = render_target(
                output_dir, residues, complex_id, render_root / f"preview_{complex_id}",
                px_per_angstrom=6.0, dphi_column=dphi_column, refresh=args.refresh_renders,
            )
        sheet = output_dir / "figures" / "fig4_3_candidates.png"
        preview_sheet(render_root, {f"preview_{k}": v for k, v in geometries.items()}, sheet)
        print(f"contact sheet: {sheet}")
        return

    rows = residues[residues.complex_id == args.target]
    chains = sorted(rows.chain.unique())
    labels = json.loads(args.chain_labels) if args.chain_labels else {c: f"chain {c}" for c in chains}
    geometry = render_target(output_dir, residues, args.target, render_root / args.target,
                             px_per_angstrom=args.px_per_angstrom, dphi_column=dphi_column,
                             refresh=args.refresh_renders)
    compose(residues, geometry, render_root / args.target, args.target, labels, dphi_column, Path(args.output))
    rank = int(ec.index[ec.complex_id == args.target][0]) + 1
    print(f"Fig. 4.3 written to {args.output}")
    print(f"  showcase {args.target}: EC {ec.EC.iloc[rank - 1]:+.2f}, rank {rank}/{len(ec)} "
          f"(median over the 84: {ec.EC.median():+.2f})")


if __name__ == "__main__":
    main()
