"""Normalise a model PDB into something pdb2pqr will reliably accept.

Three problems show up in real inputs and each silently corrupts or crashes the
downstream calculation, so all three are handled here rather than hoped away:

  * Duplicated residue numbering. Several of these files were written with the
    insertion code dropped, so one chain can carry two distinct residues both
    numbered e.g. 20. pdb2pqr merges them and then reports a structural "gap".
    Residues are therefore renumbered sequentially before it ever sees them.
  * Incomplete side chains. Crystal structures with unresolved side-chain
    density usually rebuild fine, but on some residues pdb2pqr's hydrogen
    rebuild crashes outright. truncate_incomplete cuts those residues back to
    ALA (backbone + CB) or GLY (backbone only); it is a *fallback*, used only
    after a full-structure attempt has failed, because truncating discards
    that side chain's charge. Either way the state is recorded per residue, so
    the loss is visible rather than silent.
  * Pre-existing hydrogens. pdb2pqr rebuilds and re-protonates regardless, and
    the reduce-added hydrogens in these files carry serial number 0, so they
    are dropped up front.

The renumbering is chosen so the new residue number is aa_id + 1, where aa_id
is the sequential residue counter (in file order, across chains) that
create_protein_graph_structure.py uses as its node id. That makes the mapping
from PQR back to graph nodes a subtraction rather than a name match -- which
matters precisely for the duplicate-numbering files, where matching on
(chain, residue number) is ambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


BACKBONE_ATOMS = ("N", "CA", "C", "O")

# Heavy side-chain atoms expected for each standard residue.
SIDECHAIN_ATOMS: dict[str, tuple[str, ...]] = {
    "ALA": ("CB",),
    "ARG": ("CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"),
    "ASN": ("CB", "CG", "OD1", "ND2"),
    "ASP": ("CB", "CG", "OD1", "OD2"),
    "CYS": ("CB", "SG"),
    "GLN": ("CB", "CG", "CD", "OE1", "NE2"),
    "GLU": ("CB", "CG", "CD", "OE1", "OE2"),
    "GLY": (),
    "HIS": ("CB", "CG", "ND1", "CD2", "CE1", "NE2"),
    "ILE": ("CB", "CG1", "CG2", "CD1"),
    "LEU": ("CB", "CG", "CD1", "CD2"),
    "LYS": ("CB", "CG", "CD", "CE", "NZ"),
    "MET": ("CB", "CG", "SD", "CE"),
    "PHE": ("CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "PRO": ("CB", "CG", "CD"),
    "SER": ("CB", "OG"),
    "THR": ("CB", "OG1", "CG2"),
    "TRP": ("CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
    "TYR": ("CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"),
    "VAL": ("CB", "CG1", "CG2"),
}

MAX_PDB_RESIDUE_NUMBER = 9999


@dataclass
class PreparedStructure:
    """The cleaned PDB written for pdb2pqr, plus the map back to the original.

    Every array is indexed by aa_id, i.e. entry i describes the (i+1)-th
    residue in the original file order, and the prepared PDB numbers that same
    residue i+1.
    """

    pdb_path: Path
    chain: np.ndarray            # (R,) original chain ID
    number: np.ndarray           # (R,) original PDB residue number
    insertion_code: np.ndarray   # (R,) original insertion code ("" if none)
    name: np.ndarray             # (R,) original residue name
    modeled_name: np.ndarray     # (R,) what pdb2pqr was actually given
    incomplete: np.ndarray       # (R,) bool: side chain missing atoms in the input
    truncated: np.ndarray        # (R,) bool: that side chain was actually removed
    atom_count: np.ndarray       # (R,) heavy atoms written
    warnings: list[str]

    def __len__(self) -> int:
        return len(self.chain)

    @property
    def has_incomplete_residues(self) -> bool:
        return bool(self.incomplete.any())


def _atom_name(line: str) -> str:
    return line[12:16].strip()


def _is_hydrogen(line: str) -> bool:
    element = line[76:78].strip()
    if element:
        return element == "H"
    # Fall back to the atom-name convention when the element column is blank.
    name = line[12:16]
    return name.strip()[:1] == "H" or name[1:2] == "H"


def prepare_structure(
    pdb_path: str | Path,
    output_path: str | Path,
    truncate_incomplete: bool = False,
    keep_altloc: str = "A",
) -> PreparedStructure:
    """Write a normalised heavy-atom PDB and return the residue map.

    Residue blocks, their order, and their aa_ids are identical whether or not
    truncate_incomplete is set -- only the atoms written change -- so a
    truncating retry stays aligned with the first attempt.
    """
    pdb_path = Path(pdb_path)
    output_path = Path(output_path)

    # Group atoms into residue blocks in file order. A new block starts whenever
    # the (chain, number, insertion code, name) key changes, so two same-numbered
    # residues stay distinct instead of being merged.
    blocks: list[tuple[tuple[str, str, str, str], list[str]]] = []
    previous_key = None
    block_atom_names: set[str] = set()
    for line in pdb_path.read_text(errors="replace").splitlines():
        if not line.startswith("ATOM"):
            continue
        altloc = line[16]
        if altloc not in (" ", keep_altloc):
            continue
        if _is_hydrogen(line):
            continue
        key = (line[21], line[22:26].strip(), line[26].strip(), line[17:20].strip())
        atom_name = _atom_name(line)
        # A new residue starts when the identity key changes, or when an atom
        # name repeats: two *adjacent* residues sharing chain, number, and name
        # are indistinguishable by key alone, and merging them is exactly the
        # corruption this module exists to prevent.
        if key != previous_key or atom_name in block_atom_names:
            blocks.append((key, []))
            previous_key = key
            block_atom_names = set()
        blocks[-1][1].append(line)
        block_atom_names.add(atom_name)

    if not blocks:
        raise ValueError(f"No usable ATOM records in {pdb_path}")
    if len(blocks) > MAX_PDB_RESIDUE_NUMBER:
        raise ValueError(
            f"{pdb_path.name} has {len(blocks)} residues, which overflows the "
            "4-character PDB residue-number field used for the aa_id mapping"
        )

    chains: list[str] = []
    numbers: list[int] = []
    insertion_codes: list[str] = []
    names: list[str] = []
    modeled_names: list[str] = []
    incomplete_flags: list[bool] = []
    truncated_flags: list[bool] = []
    atom_counts: list[int] = []
    warnings: list[str] = []
    output_lines: list[str] = []
    serial = 0

    for aa_id, ((chain, number, insertion_code, resname), lines) in enumerate(blocks):
        present = {_atom_name(line) for line in lines}
        missing_backbone = [name for name in BACKBONE_ATOMS if name not in present]
        expected_sidechain = SIDECHAIN_ATOMS.get(resname)
        if expected_sidechain is None:
            warnings.append(f"{chain}{number} {resname}: non-standard residue kept as-is")
            missing_sidechain: list[str] = []
        else:
            missing_sidechain = [name for name in expected_sidechain if name not in present]

        kept_lines = lines
        modeled_name = resname
        if missing_sidechain and truncate_incomplete:
            # Degrade to the largest residue the present atoms can support.
            modeled_name = "ALA" if "CB" in present else "GLY"
            allowed = set(BACKBONE_ATOMS) | ({"CB"} if modeled_name == "ALA" else set())
            kept_lines = [line for line in lines if _atom_name(line) in allowed]
        if missing_backbone:
            warnings.append(
                f"{chain}{number} {resname}: missing backbone {','.join(missing_backbone)}"
            )

        for line in kept_lines:
            serial += 1
            # Rewrite serial, residue name, residue number; blank the insertion
            # code, occupancy and B-factor. Coordinates are copied verbatim.
            output_lines.append(
                f"ATOM  {serial:5d} {line[12:16]}{' '}{modeled_name:>3s} {chain}"
                f"{aa_id + 1:4d}    {line[30:54]}  1.00  0.00          {line[76:78]}"
            )

        chains.append(chain)
        numbers.append(int(number))
        insertion_codes.append(insertion_code)
        names.append(resname)
        modeled_names.append(modeled_name)
        incomplete_flags.append(bool(missing_sidechain))
        truncated_flags.append(bool(missing_sidechain) and truncate_incomplete)
        atom_counts.append(len(kept_lines))

    truncated_count = sum(truncated_flags)
    if truncated_count:
        warnings.insert(
            0,
            f"{truncated_count}/{len(blocks)} residues truncated to ALA/GLY "
            "(incomplete side chain in the input structure); their side-chain "
            "charge is not represented",
        )

    output_path.write_text("\n".join(output_lines) + "\nTER\nEND\n", encoding="utf-8")

    return PreparedStructure(
        pdb_path=output_path,
        chain=np.array(chains, dtype="<U4"),
        number=np.array(numbers, dtype=np.int32),
        insertion_code=np.array(insertion_codes, dtype="<U2"),
        name=np.array(names, dtype="<U8"),
        modeled_name=np.array(modeled_names, dtype="<U8"),
        incomplete=np.array(incomplete_flags, dtype=bool),
        truncated=np.array(truncated_flags, dtype=bool),
        atom_count=np.array(atom_counts, dtype=np.int32),
        warnings=warnings,
    )
