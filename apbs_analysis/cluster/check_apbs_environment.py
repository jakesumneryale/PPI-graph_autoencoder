#!/usr/bin/env python3
"""Fail-fast preflight for an APBS array worker.

Checks the Python imports, that both external binaries run, and -- most
usefully -- that a two-atom toy system actually solves end to end. A missing
forcefield data file or a broken APBS build only shows up on a real solve, and
finding that here costs two seconds instead of a whole array task.
"""

from __future__ import annotations

import importlib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

REQUIRED_IMPORTS = ("numpy", "scipy", "h5py", "pandas")

TOY_PQR = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  -0.4000 1.8240
ATOM      2  CA  ALA A   1       1.500   0.000   0.000   0.4000 1.9080
"""


def main() -> None:
    failures: list[str] = []

    for name in REQUIRED_IMPORTS:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"import {name}: {exc}")

    for executable in ("apbs", "pdb2pqr"):
        if shutil.which(executable) is None:
            failures.append(f"{executable} not found on PATH")

    if not failures:
        try:
            from apbs_analysis.electrostatics import ApbsSettings, parse_pqr, run_apbs, write_apbs_input
            from apbs_analysis.dx_grid import read_dx
            from apbs_analysis.grid_sizing import compute_grid_parameters
        except Exception as exc:  # noqa: BLE001
            failures.append(f"apbs_analysis import: {exc}")

    # End-to-end toy solve: exercises the APBS binary, the input writer, and
    # the DX reader together.
    if not failures:
        try:
            with tempfile.TemporaryDirectory() as temporary:
                work_dir = Path(temporary)
                pqr_path = work_dir / "toy.pqr"
                pqr_path.write_text(TOY_PQR)
                structure = parse_pqr(pqr_path)
                settings = ApbsSettings()
                grid = compute_grid_parameters(
                    structure.xyz, structure.radius, memory_ceiling_mb=64.0
                )
                input_path = work_dir / "toy.in"
                write_apbs_input(input_path, pqr_path.name, grid, settings)
                potential = read_dx(run_apbs(input_path, work_dir, settings))
                if potential.values.size == 0:
                    failures.append("toy APBS solve produced an empty potential grid")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"toy APBS solve: {exc}")

    if not failures:
        try:
            subprocess.run(["pdb2pqr", "--version"], capture_output=True, check=True, timeout=120)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"pdb2pqr --version: {exc}")

    # The pure-Python correctness checks are seconds of work and catch a bad
    # numpy/scipy pairing that would otherwise corrupt every number quietly.
    if not failures:
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "apbs_analysis.selfcheck"],
                cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=300,
            )
            if completed.returncode != 0:
                failures.append(f"apbs_analysis.selfcheck: {completed.stdout.strip()} {completed.stderr.strip()}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"apbs_analysis.selfcheck: {exc}")

    if failures:
        print("APBS environment preflight FAILED:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        raise SystemExit(1)

    print(f"APBS environment preflight passed with {sys.executable}")


if __name__ == "__main__":
    main()
