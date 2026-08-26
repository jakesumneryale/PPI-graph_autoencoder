"""Minimal OpenDX reader for APBS potential maps, plus trilinear sampling.

APBS writes uniform "gridpositions/gridconnections" OpenDX files. Parsing the
handful of header lines directly avoids a dependency on gridDataFormats and
keeps the cluster environment small.
"""

from __future__ import annotations

from dataclasses import dataclass
import gzip
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DxGrid:
    """A uniform scalar grid: values[i, j, k] sits at origin + (i, j, k) * spacing."""

    origin: np.ndarray   # (3,) float64, Angstrom
    spacing: np.ndarray  # (3,) float64, Angstrom
    values: np.ndarray   # (nx, ny, nz) float32

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(n) for n in self.values.shape)

    def sample(self, points: np.ndarray) -> np.ndarray:
        """Trilinearly interpolate at arbitrary Cartesian points (P, 3).

        Points outside the box clamp to the nearest edge value. APBS boxes are
        padded well beyond the molecule, so out-of-box sampling should not
        happen for surface points; clamping just keeps it non-fatal.
        """
        from scipy.ndimage import map_coordinates

        points = np.asarray(points, dtype=np.float64)
        fractional = (points - self.origin) / self.spacing
        sampled = map_coordinates(
            self.values.astype(np.float64, copy=False),
            fractional.T,
            order=1,
            mode="nearest",
        )
        return sampled.astype(np.float32)


def _open_maybe_gzip(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return path.open("r")


def read_dx(path: str | Path) -> DxGrid:
    """Parse an OpenDX file (optionally .gz) written by APBS."""
    path = Path(path)
    counts: tuple[int, int, int] | None = None
    origin: np.ndarray | None = None
    deltas: list[np.ndarray] = []
    item_count: int | None = None
    data_chunks: list[np.ndarray] = []

    with _open_maybe_gzip(path) as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            if item_count is None:
                # Still in the header.
                if stripped.startswith("object") and "gridpositions" in stripped:
                    fields = stripped.split()
                    counts = (int(fields[-3]), int(fields[-2]), int(fields[-1]))
                elif stripped.startswith("origin"):
                    origin = np.array([float(value) for value in stripped.split()[1:4]])
                elif stripped.startswith("delta"):
                    deltas.append(np.array([float(value) for value in stripped.split()[1:4]]))
                elif stripped.startswith("object") and "data follows" in stripped:
                    item_count = int(stripped.split("items")[1].split()[0])
                continue

            # Data body: read it in one chunk rather than line by line -- a
            # 129^3 fine grid is ~2M numbers and per-line parsing dominates.
            body = stripped + "\n" + handle.read()
            numeric_lines = []
            for body_line in body.splitlines():
                body_line = body_line.strip()
                if not body_line:
                    continue
                if body_line[0].isalpha():
                    break  # trailing attribute/component footer
                numeric_lines.append(body_line)
            data_chunks.append(
                np.array(" ".join(numeric_lines).split(), dtype=np.float64)
            )
            break

    if counts is None or origin is None or item_count is None or len(deltas) < 3:
        raise ValueError(f"{path} is not a recognisable uniform OpenDX grid")

    values = np.concatenate(data_chunks) if data_chunks else np.empty(0)
    if values.size != item_count:
        raise ValueError(f"{path}: expected {item_count} values, parsed {values.size}")

    # APBS always writes axis-aligned deltas, one per row.
    spacing = np.array([deltas[0][0], deltas[1][1], deltas[2][2]])
    if not np.all(spacing > 0):
        raise ValueError(f"{path}: non-axis-aligned or degenerate grid deltas {deltas}")

    return DxGrid(
        origin=origin,
        spacing=spacing,
        values=values.reshape(counts).astype(np.float32),  # OpenDX is z-fastest row-major
    )
