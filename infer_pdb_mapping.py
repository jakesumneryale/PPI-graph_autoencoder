"""Match graph model groups to PDB files by geometry (CA distances at stored contacts).

Read-only. For each requested graph group it finds the PDB in a directory whose CA-CA
distances at the group's contacts reproduce the stored ca_dist, then prints the pairs so
a renaming rule can be read off and verified.
"""
import argparse
from pathlib import Path
import h5py
import numpy as np


def ca_coords(path):
    seen, coords = set(), []
    for line in open(path):
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            key = (line[21], line[22:27])
            if key not in seen:
                seen.add(key)
                coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return np.asarray(coords)


def fingerprint(coords, contacts):
    if len(coords) == 0 or contacts.max() >= len(coords):
        return None
    return np.linalg.norm(coords[contacts[:, 0]] - coords[contacts[:, 1]], axis=1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph", type=Path, required=True)
    p.add_argument("--pdb-dir", type=Path, required=True, help="Directory of candidate PDB files")
    p.add_argument("--prefix", required=True, help="Graph group prefix, e.g. sampled_1ggp_model_")
    p.add_argument("--count", type=int, default=20, help="How many graph groups to match")
    p.add_argument("--tolerance", type=float, default=0.05)
    a = p.parse_args()

    pdbs = sorted(a.pdb_dir.glob("*.pdb"))
    coords = {f.name: ca_coords(f) for f in pdbs}
    print(f"{len(pdbs)} candidate PDB files in {a.pdb_dir}")
    with h5py.File(a.graph, "r") as h:
        names = sorted((n for n in h if n.startswith(a.prefix)), key=lambda n: int(n.rsplit("_", 1)[1]))[: a.count]
        for name in names:
            ef = h[name]["edge_features"]
            contacts = np.asarray(ef["contacts"][()]).astype(int)
            expected = np.asarray(ef["ca_dist"][()]).reshape(-1)
            best = []
            for fname, xyz in coords.items():
                fp = fingerprint(xyz, contacts)
                if fp is not None and len(fp) == len(expected):
                    best.append((float(np.abs(fp - expected).max()), fname))
            best.sort()
            top = best[0] if best else (float("nan"), "NONE")
            ok = top[0] <= a.tolerance and (len(best) < 2 or best[1][0] > a.tolerance)
            print(f"{name:28s} -> {top[1]:45s} max|diff|={top[0]:.4f} {'UNIQUE' if ok else 'CHECK'}")


if __name__ == "__main__":
    main()
