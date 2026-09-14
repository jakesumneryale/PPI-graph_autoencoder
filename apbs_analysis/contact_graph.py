"""Write residue contact graphs as PDB files, for viewing in PyMOL.

Three graphs per complex, in separate files so they can be loaded together and
coloured independently. All coordinates are the original ones, so the files
overlay the source structure with no fitting:

    <id>_chain_<X>_graph.pdb    one chain: its residues and their contacts
    <id>_chain_<Y>_graph.pdb    the other chain
    <id>_interface_graph.pdb    only the contacts that cross between chains

Nodes sit on the C-alpha and are written as HETATM carbons, one residue each,
keeping the original chain ID and residue number so `resi 45` still selects
what it should. Edges are CONECT records.

PyMOL will not invent bonds here: nodes are ~3.8 A apart at minimum, far beyond
any covalent distance, so distance-based bonding finds nothing and the CONECT
records are the only connectivity. HETATM is deliberate -- PyMOL's default
connect_mode reads CONECT for HETATM records but applies its own residue
templates to ATOM records.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


DEFAULT_CUTOFF = 4.5      # A, heavy-atom to heavy-atom between residues
NODE_RESNAME = "NOD"


def _is_hydrogen(line: str) -> bool:
    element = line[76:78].strip()
    if element:
        return element == "H"
    name = line[12:16]
    return name.strip()[:1] == "H" or name[1:2] == "H"


@dataclass
class ResidueGraph:
    """Residues of one structure plus the contacts between them."""

    chain: np.ndarray          # (R,) chain ID
    number: np.ndarray         # (R,) original residue number
    name: np.ndarray           # (R,) residue name
    node_xyz: np.ndarray       # (R, 3) C-alpha position
    edges: np.ndarray          # (E, 2) residue indices, i < j
    atom_xyz: np.ndarray       # heavy atoms, for rebuilding subsets
    atom_residue: np.ndarray   # residue index per heavy atom

    def __len__(self) -> int:
        return len(self.number)


def build_contact_graph(pdb_path: str | Path, cutoff: float = DEFAULT_CUTOFF,
                        keep_altloc: str = "A") -> ResidueGraph:
    """Parse a PDB and connect residues whose heavy atoms come within `cutoff`.

    Sequential neighbours are kept: their backbone atoms are always inside the
    cutoff, and the resulting edges are what make the graph trace the chain,
    which is what you want to look at.
    """
    pdb_path = Path(pdb_path)

    blocks: list[tuple[tuple[str, str, str], list[str]]] = []
    previous_key = None
    seen_names: set[str] = set()
    for line in pdb_path.read_text(errors="replace").splitlines():
        if not line.startswith("ATOM"):
            continue
        if line[16] not in (" ", keep_altloc):
            continue
        if _is_hydrogen(line):
            continue
        key = (line[21], line[22:27], line[17:20].strip())
        atom_name = line[12:16].strip()
        # Same split rule the rest of the package uses: a repeated atom name
        # starts a new residue, so two adjacent residues sharing a number (the
        # files whose insertion codes were stripped) do not get merged.
        if key != previous_key or atom_name in seen_names:
            blocks.append((key, []))
            previous_key = key
            seen_names = set()
        blocks[-1][1].append(line)
        seen_names.add(atom_name)

    if not blocks:
        raise ValueError(f"No usable ATOM records in {pdb_path}")

    chains, numbers, names, node_xyz = [], [], [], []
    atom_xyz, atom_residue = [], []
    for index, ((chain, number, resname), lines) in enumerate(blocks):
        coords = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in lines])
        alpha = [l for l in lines if l[12:16].strip() == "CA"]
        if alpha:
            position = np.array([float(alpha[0][30:38]), float(alpha[0][38:46]),
                                 float(alpha[0][46:54])])
        else:
            # No C-alpha (rare, truncated residues): fall back to the centroid so
            # the node still lands inside the residue it represents.
            position = coords.mean(axis=0)
        chains.append(chain)
        numbers.append(int(number[:4]))
        names.append(resname)
        node_xyz.append(position)
        atom_xyz.append(coords)
        atom_residue.append(np.full(len(coords), index))

    atom_xyz = np.vstack(atom_xyz)
    atom_residue = np.concatenate(atom_residue)

    tree = cKDTree(atom_xyz)
    pairs = tree.query_pairs(cutoff, output_type="ndarray")
    residue_pairs = atom_residue[pairs]
    residue_pairs = residue_pairs[residue_pairs[:, 0] != residue_pairs[:, 1]]
    residue_pairs.sort(axis=1)
    edges = np.unique(residue_pairs, axis=0) if len(residue_pairs) else np.empty((0, 2), int)

    return ResidueGraph(
        chain=np.array(chains, dtype="<U2"),
        number=np.array(numbers, dtype=int),
        name=np.array(names, dtype="<U4"),
        node_xyz=np.array(node_xyz),
        edges=edges,
        atom_xyz=atom_xyz,
        atom_residue=atom_residue,
    )


def _conect_lines(serial: int, partners: list[int]) -> list[str]:
    """CONECT holds at most four partners; longer lists continue on new lines."""
    lines = []
    for start in range(0, len(partners), 4):
        chunk = partners[start : start + 4]
        lines.append("CONECT" + f"{serial:5d}" + "".join(f"{p:5d}" for p in chunk))
    return lines


def write_graph_pdb(
    path: str | Path,
    graph: ResidueGraph,
    node_indices: np.ndarray,
    edges: np.ndarray,
    title: str,
) -> tuple[int, int]:
    """Write a subset of nodes and edges as HETATM + CONECT."""
    path = Path(path)
    node_indices = np.asarray(node_indices)
    serial_of = {int(node): position + 1 for position, node in enumerate(node_indices)}

    lines = [f"REMARK   1 {title}",
             f"REMARK   1 nodes on C-alpha; edges = residue pairs within "
             f"{DEFAULT_CUTOFF} A heavy atom to heavy atom",
             "REMARK   1 B-factor is the node degree within this graph"]

    degree = np.zeros(len(node_indices), dtype=int)
    neighbours: dict[int, list[int]] = {int(n): [] for n in node_indices}
    for left, right in edges:
        neighbours[int(left)].append(serial_of[int(right)])
        neighbours[int(right)].append(serial_of[int(left)])
    for position, node in enumerate(node_indices):
        degree[position] = len(neighbours[int(node)])

    for position, node in enumerate(node_indices):
        node = int(node)
        x, y, z = graph.node_xyz[node]
        lines.append(
            f"HETATM{position + 1:5d}  CA  {NODE_RESNAME} {graph.chain[node]:1s}"
            f"{graph.number[node] % 10000:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{min(degree[position], 999):6.2f}"
            f"          C"
        )
    lines.append("TER")
    for node in node_indices:
        partners = sorted(neighbours[int(node)])
        if partners:
            lines.extend(_conect_lines(serial_of[int(node)], partners))
    lines.append("END")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(node_indices), len(edges)


def write_complex_graphs(
    pdb_path: str | Path, output_dir: str | Path, cutoff: float = DEFAULT_CUTOFF
) -> dict:
    """Write the two per-chain graphs and the interface graph for one complex."""
    pdb_path = Path(pdb_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = pdb_path.stem.replace("_complex_H", "")

    graph = build_contact_graph(pdb_path, cutoff=cutoff)
    labels = sorted(set(graph.chain.tolist()))
    if len(labels) != 2:
        raise ValueError(f"{model}: expected 2 chains, found {labels}")

    same_chain = graph.chain[graph.edges[:, 0]] == graph.chain[graph.edges[:, 1]]
    record = {"complex_id": model, "cutoff_angstrom": cutoff}

    # Fixed column names, with the chain letter as a value. Naming the columns
    # after the letters themselves gives a sparse, NaN-filled table once the
    # complexes use different chain IDs.
    for position, label in enumerate(labels, start=1):
        nodes = np.flatnonzero(graph.chain == label)
        keep = same_chain & np.isin(graph.edges[:, 0], nodes)
        node_count, edge_count = write_graph_pdb(
            output_dir / f"{model}_chain_{label}_graph.pdb", graph, nodes,
            graph.edges[keep], f"{model} chain {label} residue contact graph")
        record[f"chain_{position}"] = label
        record[f"chain_{position}_nodes"] = node_count
        record[f"chain_{position}_edges"] = edge_count

    # Interface graph: only the edges that cross between the chains, and only
    # the residues those edges touch.
    crossing = graph.edges[~same_chain]
    interface_nodes = np.unique(crossing) if len(crossing) else np.empty(0, int)
    node_count, edge_count = write_graph_pdb(
        output_dir / f"{model}_interface_graph.pdb", graph, interface_nodes, crossing,
        f"{model} interface contact graph ({labels[0]} to {labels[1]})")
    record["interface_nodes"] = node_count
    record["interface_edges"] = edge_count
    record["chains"] = "".join(labels)
    return record


SESSION_PML = '''# {model} residue contact graphs
# Nodes on C-alpha, edges within {cutoff} A (heavy atom to heavy atom).
# B-factor on each node is its degree within that graph.

load {model}_chain_{a}_graph.pdb, {model}_chain_{a}
load {model}_chain_{b}_graph.pdb, {model}_chain_{b}
load {model}_interface_graph.pdb, {model}_interface

bg_color white
hide everything

show spheres, {model}_chain_{a} or {model}_chain_{b} or {model}_interface
show sticks,  {model}_chain_{a} or {model}_chain_{b} or {model}_interface

color skyblue, {model}_chain_{a}
color salmon,  {model}_chain_{b}
color yellow,  {model}_interface

set sphere_scale, 0.55, {model}_chain_{a}
set sphere_scale, 0.55, {model}_chain_{b}
set sphere_scale, 0.75, {model}_interface
set stick_radius, 0.14, {model}_chain_{a}
set stick_radius, 0.14, {model}_chain_{b}
set stick_radius, 0.25, {model}_interface

set ray_opaque_background, 1
orient
zoom all, 3

# Colour a chain graph by node degree instead:
#   spectrum b, blue_white_red, {model}_chain_{a}
'''


def write_session_pml(output_dir: str | Path, model: str, labels: list[str],
                      cutoff: float = DEFAULT_CUTOFF) -> Path:
    path = Path(output_dir) / f"{model}_graphs.pml"
    path.write_text(
        SESSION_PML.format(model=model, a=labels[0], b=labels[1], cutoff=cutoff),
        encoding="utf-8",
    )
    return path


def main() -> None:
    import argparse
    import sys

    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdb-dir", default=str(Path(__file__).resolve().parent.parent
                                                 / "targets_84_complex_only"))
    parser.add_argument("--output-dir",
                        default=str(Path.home() / "Documents" / "ppi_contact_graphs"))
    parser.add_argument("--cutoff", type=float, default=DEFAULT_CUTOFF)
    parser.add_argument("--targets", nargs="*")
    parser.add_argument("--no-pml", action="store_true", help="Skip the per-target PyMOL script")
    args = parser.parse_args()

    pdb_dir = Path(args.pdb_dir).expanduser()
    if not pdb_dir.is_dir():
        sys.exit(f"No PDB directory at {pdb_dir}")
    output_dir = Path(args.output_dir).expanduser()

    paths = sorted(pdb_dir.glob("*.pdb"))
    if args.targets:
        wanted = set(args.targets)
        paths = [p for p in paths if p.stem.replace("_complex_H", "") in wanted]

    print(f"PDB source: {pdb_dir}")
    print(f"Output:     {output_dir}")
    print(f"Cutoff:     {args.cutoff} A heavy atom to heavy atom")
    print(f"Complexes:  {len(paths)}", flush=True)

    rows, failures = [], []
    for index, path in enumerate(paths, start=1):
        try:
            record = write_complex_graphs(path, output_dir, cutoff=args.cutoff)
            if not args.no_pml:
                write_session_pml(output_dir, record["complex_id"],
                                  list(record["chains"]), args.cutoff)
            rows.append(record)
        except Exception as exc:  # noqa: BLE001
            failures.append((path.stem, str(exc)[:200]))
        if index % 20 == 0:
            print(f"  {index}/{len(paths)}", flush=True)

    summary = pd.DataFrame(rows)
    summary_path = output_dir / "contact_graph_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\n{len(rows)} complexes written, {len(failures)} failed -> {summary_path}")
    for name, message in failures:
        print(f"  {name}: {message}")


if __name__ == "__main__":
    main()
