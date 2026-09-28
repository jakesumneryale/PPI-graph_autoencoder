"""Match graph model groups to PDB files by geometry (CA distances at stored contacts).

The graph stores ca_dist for each contact; a PDB whose CA coordinates reproduce it exactly is
the structure the model was built from, whatever the file is called. Never writes to the graph.

  show      print the matches for the first few groups of one family
  write-map write graph_group_name -> relative PDB path for every sampled_/random_ group
"""
import argparse
import csv
import re
import sys
from pathlib import Path
import h5py
import numpy as np

TOLERANCE = 0.05
FAMILY_PATTERN = re.compile(r"^(?P<family>sampled|random)_(?P<target>.+)_model_(?P<index>\d+)$")


def ca_coords(path):
    seen, coords = set(), []
    for line in open(path):
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            key = (line[21], line[22:27])
            if key not in seen:
                seen.add(key)
                coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return np.asarray(coords, dtype=np.float64).reshape(-1, 3)


def load_candidates(pdb_dir):
    """Return {residue_count: (file_names, coordinate stack)} for every PDB directly in pdb_dir."""
    grouped = {}
    for path in sorted(Path(pdb_dir).glob("*.pdb")):
        xyz = ca_coords(path)
        if len(xyz):
            grouped.setdefault(len(xyz), []).append((path.name, xyz))
    return {n: ([name for name, _ in items], np.stack([xyz for _, xyz in items])) for n, items in grouped.items()}


def match_group(group, candidates, tolerance=TOLERANCE):
    """Return (file_name, max_abs_diff, unique) for one graph group, or None if nothing fits."""
    if "node_features" not in group or "edge_features" not in group:
        return None
    n = len(group["node_features"]["interface_nodes"])
    if n not in candidates:
        return None
    contacts = np.asarray(group["edge_features"]["contacts"][()]).astype(int)
    expected = np.asarray(group["edge_features"]["ca_dist"][()], dtype=np.float64).reshape(-1)
    if len(contacts) != len(expected) or contacts.max() >= n:
        return None
    names, stack = candidates[n]
    dist = np.linalg.norm(stack[:, contacts[:, 0]] - stack[:, contacts[:, 1]], axis=2)
    diff = np.abs(dist - expected).max(axis=1)
    order = np.argsort(diff)
    best = float(diff[order[0]])
    unique = best <= tolerance and (len(order) < 2 or float(diff[order[1]]) > tolerance)
    return names[order[0]], best, unique


def family_dirs(sampled_dir):
    sampled_dir = Path(sampled_dir)
    return {"sampled": sampled_dir, "random": sampled_dir / "random_negatives"}


def build_map(graph_path, sampled_dir, tolerance=TOLERANCE, limit=None):
    rows, problems = [], {"unmatched": [], "ambiguous": []}
    target_dir = Path(sampled_dir).name
    with h5py.File(graph_path, "r") as handle:
        by_family = {"sampled": [], "random": []}
        for name in sorted(handle):
            m = FAMILY_PATTERN.fullmatch(name)
            if m:
                by_family[m.group("family")].append(name)
        for family, directory in family_dirs(sampled_dir).items():
            names = by_family[family][:limit]
            if not names:
                continue
            candidates = load_candidates(directory)
            for name in names:
                result = match_group(handle[name], candidates, tolerance)
                if result is None or result[1] > tolerance:
                    problems["unmatched"].append(name)
                elif not result[2]:
                    problems["ambiguous"].append(name)
                else:
                    sub = "random_negatives/" if family == "random" else ""
                    rows.append((name, f"{target_dir}/{sub}{result[0]}", "random_negative" if family == "random" else "sampled"))
    return rows, problems


def write_map(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("graph_group_name", "relative_pdb_path", "location_type"))
        writer.writerows(rows)
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    s = sub.add_parser("show")
    s.add_argument("--graph", type=Path, required=True)
    s.add_argument("--pdb-dir", type=Path, required=True)
    s.add_argument("--prefix", required=True)
    s.add_argument("--count", type=int, default=20)
    w = sub.add_parser("write-map")
    w.add_argument("--graph", type=Path, required=True)
    w.add_argument("--sampled-dir", type=Path, required=True, help="sampled_<target> directory")
    w.add_argument("--output", type=Path, required=True)
    w.add_argument("--min-matched-fraction", type=float, default=0.95,
                   help="Fail (writing nothing) if fewer of the sampled_/random_ groups match uniquely")
    a = p.parse_args()

    if a.action == "show":
        candidates = load_candidates(a.pdb_dir)
        with h5py.File(a.graph, "r") as handle:
            names = sorted((n for n in handle if n.startswith(a.prefix)), key=lambda n: int(n.rsplit("_", 1)[1]))[: a.count]
            for name in names:
                r = match_group(handle[name], candidates)
                print(f"{name:28s} -> {r[0] if r else 'NONE':45s} max|diff|={r[1] if r else float('nan'):.4f} "
                      f"{'UNIQUE' if r and r[2] else 'CHECK'}")
        return

    with h5py.File(a.graph, "r") as handle:
        total = sum(1 for n in handle if FAMILY_PATTERN.fullmatch(n))
    if total == 0:
        print(f"{a.graph.stem}: no sampled_/random_ groups; no mapping needed.")
        return
    rows, problems = build_map(a.graph, a.sampled_dir)
    print(f"{a.graph.stem}: matched {len(rows)}/{total} groups by geometry; "
          f"{len(problems['unmatched'])} unmatched, {len(problems['ambiguous'])} ambiguous")
    if len(rows) < a.min_matched_fraction * total:
        print("Too few unique matches; refusing to write a mapping.", file=sys.stderr)
        sys.exit(2)
    write_map(rows, a.output)
    print(f"Wrote {a.output}")


if __name__ == "__main__":
    main()
