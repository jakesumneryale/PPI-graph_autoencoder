"""Render an approach movie to PNG frames and an mp4, with a label band on top.

The readout is composited onto the frames afterwards rather than drawn as a
PyMOL label. Placing a label in the scene means placing it in camera space,
where perspective, the render aspect ratio and PyMOL's own label handling all
fight you; compositing puts the text exactly where it is asked to go, keeps it
crisp, and leaves the structure untouched underneath.

PyMOL renders the structures on a plain canvas, then a white band is added
above and the text drawn into it, so nothing ever overlaps the molecules.

Uses whichever `pymol` is on PATH (the open-source build, so no watermark).

    python -m apbs_analysis.approach_render 1ugh --turn 180 --suffix viewA
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from PIL import Image, ImageDraw, ImageFont


PYMOL_SCRIPT = '''
from pymol import cmd

cmd.load(r"{pdb}", "ap")
cmd.hide("everything")
cmd.show("surface", "ap")
cmd.bg_color("white")
cmd.set("ray_opaque_background", 1)
cmd.set("orthoscopic", 1)          # no perspective distortion of the separation
cmd.set("surface_quality", {quality})
cmd.set("two_sided_lighting", "on")
cmd.set("antialias", 2)

cmd.color("grey60", "chain {fixed}")

# Put the approach axis across the screen. Orienting on the structure instead
# leaves that axis pointing mostly into the screen, so the travel is invisible
# and the depth extent inflates the bounding sphere, shrinking everything.
# Two markers at the start and end of the moving chain's path, oriented on,
# gets this right using PyMOL's own convention rather than a hand-built matrix.
import numpy as _np

def _com(selection, state):
    model = cmd.get_model(selection, state=state)
    return _np.array([atom.coord for atom in model.atom]).mean(axis=0)

_last = cmd.count_states("ap")
cmd.pseudoatom("axis_helper", pos=list(_com("chain {moving}", 1)))
cmd.pseudoatom("axis_helper", pos=list(_com("chain {moving}", _last)))
cmd.orient("axis_helper")
cmd.delete("axis_helper")

# Roll about that axis to show a different face of the interface; the approach
# stays horizontal because the roll is about the horizontal axis.
cmd.turn("x", {roll})

cmd.set("all_states", 1)
cmd.zoom("ap", {buffer})
cmd.set("all_states", 0)

import csv as _csv
RES = {{}}
with open(r"{residues}") as handle:
    for row in _csv.DictReader(handle):
        RES.setdefault(int(row["frame"]), []).append(row)

SPAN = {span}
for frame in range(1, {frames} + 1):
    cmd.frame(frame)
    cmd.alter("chain {moving}", "b=0.0")
    for entry in RES.get(frame, []):
        cmd.alter("chain {moving} and resi %s" % entry["residue_number"],
                  "b=%s" % entry["contribution_kcal_per_mol"])
    cmd.spectrum("b", "red_white_blue", "chain {moving}", -SPAN, SPAN)
    cmd.png(r"{outdir}/raw_%04d.png" % frame, width={width}, height={height}, ray=1)
print("rendered {frames} frames")
'''


def load_font(size: int):
    """A clean sans with the glyphs we need; matplotlib always ships DejaVu."""
    import matplotlib

    candidates = [
        Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf" / "DejaVuSans.ttf",
        Path("/System/Library/Fonts/Helvetica.ttc"),
    ]
    for path in candidates:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size)
            except OSError:
                continue
    return ImageFont.load_default()


def compose(raw_path: Path, out_path: Path, band: int, lines: list[tuple[str, int]],
            colour=(26, 26, 26)):
    """Paste the rendered frame below a white band and draw the text into it."""
    scene = Image.open(raw_path).convert("RGB")
    width, height = scene.size
    canvas = Image.new("RGB", (width, height + band), "white")
    canvas.paste(scene, (0, band))

    draw = ImageDraw.Draw(canvas)
    total = sum(size for _text, size in lines) + 8 * (len(lines) - 1)
    y = max(6, (band - total) // 2)
    for text, size in lines:
        font = load_font(size)
        box = draw.textbbox((0, 0), text, font=font)
        draw.text(((width - (box[2] - box[0])) / 2, y), text, font=font, fill=colour)
        y += size + 8
    canvas.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("complex_id")
    parser.add_argument("--movie-dir",
                        default=str(Path.home() / "Documents"
                                    / "apbs_electrostatics_84_targets" / "approach_movies"))
    parser.add_argument("--roll", type=float, default=0.0,
                        help="Roll about the approach axis, in degrees, to show another face")
    parser.add_argument("--suffix", default="", help="Tag appended to the output names")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=660)
    parser.add_argument("--band", type=int, default=130, help="Label band height in pixels")
    parser.add_argument("--buffer", type=float, default=2.0,
                        help="Zoom buffer; smaller is a closer view")
    parser.add_argument("--quality", type=int, default=1)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--keep-frames", action="store_true")
    args = parser.parse_args()

    movie_dir = Path(args.movie_dir).expanduser()
    pdb = movie_dir / f"{args.complex_id}_approach.pdb"
    frames_csv = movie_dir / f"{args.complex_id}_frames.csv"
    residues_csv = movie_dir / f"{args.complex_id}_residues.csv"
    for path in (pdb, frames_csv, residues_csv):
        if not path.is_file():
            sys.exit(f"missing {path}; run apbs_analysis.approach_movie first")
    if shutil.which("pymol") is None:
        sys.exit("no `pymol` on PATH")

    with frames_csv.open() as handle:
        frames = list(csv.DictReader(handle))
    moving = fixed = None
    with residues_csv.open() as handle:
        chains = {row.get("chain") for row in csv.DictReader(handle)} - {None}
    # The chain identities are on the source PDB; read them off the model.
    labels = sorted({line[21] for line in pdb.read_text().splitlines()
                     if line.startswith("ATOM")})
    counts = {label: sum(1 for line in pdb.read_text().splitlines()
                         if line.startswith("ATOM") and line[21] == label and "MODEL" not in line)
              for label in labels}
    moving = min(counts, key=counts.get)
    fixed = max(counts, key=counts.get)

    contributions = []
    with residues_csv.open() as handle:
        for row in csv.DictReader(handle):
            contributions.append(abs(float(row["contribution_kcal_per_mol"])))
    contributions.sort()
    span = max(contributions[int(0.98 * len(contributions))], 1e-3)

    suffix = f"_{args.suffix}" if args.suffix else ""
    frame_dir = movie_dir / f"{args.complex_id}{suffix}_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    print(f"{args.complex_id}: moving chain {moving}, fixed chain {fixed}")
    print(f"approach axis across screen, rolled {args.roll} deg, zoom buffer {args.buffer}")
    print(f"{len(frames)} frames at {args.width}x{args.height} + {args.band} px label band",
          flush=True)

    with tempfile.TemporaryDirectory() as temporary:
        raw_dir = Path(temporary)
        script = raw_dir / "render.py"
        script.write_text(PYMOL_SCRIPT.format(
            pdb=pdb, residues=residues_csv, outdir=raw_dir, frames=len(frames),
            moving=moving, fixed=fixed, span=span, roll=args.roll,
            buffer=args.buffer, quality=args.quality,
            width=args.width, height=args.height))
        result = subprocess.run(["pymol", "-cq", str(script)],
                                capture_output=True, text=True)
        if result.returncode != 0:
            sys.exit(f"pymol failed:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}")
        print(result.stdout.strip().splitlines()[-1] if result.stdout.strip() else "rendered")

        for index, row in enumerate(frames, start=1):
            raw = raw_dir / f"raw_{index:04d}.png"
            if not raw.exists():
                sys.exit(f"pymol did not write {raw.name}")
            overlap = row["steric_overlap"].strip().lower() in ("true", "1")
            top = (f"{float(row['displacement_angstrom']):.1f} Å apart"
                   f"     closest contact {float(row['min_heavy_atom_gap_angstrom']):.1f} Å"
                   + ("     [steric overlap]" if overlap else ""))
            bottom = (f"U = {float(row['interaction_energy_kcal_per_mol']):+.3f} kcal/mol"
                      f"   =   {float(row['interaction_energy_kT']):+.2f} kT")
            compose(raw, frame_dir / f"frame_{index:04d}.png", args.band,
                    [(top, 34), (bottom, 30)])
        print(f"composited {len(frames)} frames -> {frame_dir}")

    movie = movie_dir / f"{args.complex_id}{suffix}_approach.mp4"
    encode = subprocess.run(
        ["ffmpeg", "-y", "-framerate", str(args.fps),
         "-i", str(frame_dir / "frame_%04d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
         "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(movie)],
        capture_output=True, text=True)
    if encode.returncode != 0:
        sys.exit(f"ffmpeg failed:\n{encode.stderr[-1500:]}")
    print(f"wrote {movie} ({movie.stat().st_size / 1024**2:.1f} MB)")

    if not args.keep_frames:
        shutil.rmtree(frame_dir, ignore_errors=True)
        print("(frames removed; pass --keep-frames to keep them)")


if __name__ == "__main__":
    main()
