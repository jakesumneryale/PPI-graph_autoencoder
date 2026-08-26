"""Argparse wiring shared by both entry points, so the cluster and local runs
are guaranteed to expose (and record) exactly the same physical settings."""

from __future__ import annotations

import argparse

from apbs_analysis.electrostatics import ApbsSettings


def add_apbs_arguments(parser: argparse.ArgumentParser) -> None:
    physics = parser.add_argument_group("electrostatics")
    physics.add_argument("--forcefield", default="PARSE", help="pdb2pqr forcefield (PARSE, AMBER, CHARMM, ...)")
    physics.add_argument("--ph", type=float, default=7.0)
    physics.add_argument(
        "--titration-method",
        default="propka",
        choices=("propka", "none"),
        help="'none' skips pKa prediction and uses standard states (much faster at scale)",
    )
    physics.add_argument("--protein-dielectric", type=float, default=2.0)
    physics.add_argument("--solvent-dielectric", type=float, default=78.54)
    physics.add_argument("--ionic-strength", type=float, default=0.150, help="mol/L of 1:1 salt; 0 disables ions")
    physics.add_argument("--temperature", type=float, default=298.15)
    physics.add_argument("--pbe-solver", default="lpbe", choices=("lpbe", "npbe"))

    grid = parser.add_argument_group("grid")
    grid.add_argument("--target-spacing", type=float, default=0.5, help="Requested fine-grid spacing (Angstrom)")
    grid.add_argument("--coarse-factor", type=float, default=1.7)
    grid.add_argument("--fine-padding", type=float, default=20.0)
    grid.add_argument(
        "--memory-ceiling-mb",
        type=float,
        default=4000.0,
        help="Coarsen the grid rather than exceed this APBS memory estimate",
    )

    surface = parser.add_argument_group("surface")
    surface.add_argument("--probe-radius", type=float, default=1.4)
    surface.add_argument("--sphere-points", type=int, default=100, help="Shrake-Rupley samples per atom")

    tools = parser.add_argument_group("executables")
    tools.add_argument("--pdb2pqr", dest="pdb2pqr_executable", default="pdb2pqr")
    tools.add_argument("--apbs", dest="apbs_executable", default="apbs")


def settings_from_args(args: argparse.Namespace) -> ApbsSettings:
    return ApbsSettings(
        forcefield=args.forcefield,
        ph=args.ph,
        titration_method=args.titration_method,
        protein_dielectric=args.protein_dielectric,
        solvent_dielectric=args.solvent_dielectric,
        ionic_strength=args.ionic_strength,
        temperature=args.temperature,
        pbe_solver=args.pbe_solver,
        coarse_factor=args.coarse_factor,
        fine_padding=args.fine_padding,
        target_spacing=args.target_spacing,
        memory_ceiling_mb=args.memory_ceiling_mb,
        probe_radius=args.probe_radius,
        sphere_points=args.sphere_points,
        pdb2pqr_executable=args.pdb2pqr_executable,
        apbs_executable=args.apbs_executable,
    )
