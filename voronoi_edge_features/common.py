"""Shared helpers for standalone Voronoi edge-feature utilities."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "voronoi_edge_features_data"
DEFAULT_REFERENCE_DIR = DEFAULT_DATA_DIR / "model_references"
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_DIR / "contact_area_hdf5"
DEFAULT_GRAPH_DATA_DIR = Path("/scratch/ppi_autoencoder_code/processed_graph_data")
CLUSTER_GRAPH_DATA_DIR = Path("/home/jas485/project_pi_co54/jas485/ppi_processed_graphs")
GRAPH_DATA_ENV_VAR = "PPI_HDF5_DATA"

POSITIVE_MODEL_PATTERN = re.compile(r"^complex\.\d{1,2}_\d{1,2}_\d{1,2}_corrected$")
NEGATIVE_MODEL_PATTERN = re.compile(r"^complex\.\d{1,5}_\d_corrected$")


@dataclass(frozen=True)
class ModelReference:
    target_name: str
    graph_group_name: str
    relative_pdb_path: str
    location_type: str

    @property
    def pdb_filename(self) -> str:
        return Path(self.relative_pdb_path).name


def target_name_from_dir(target_dir: str | Path) -> str:
    return Path(target_dir).resolve().name


def infer_relative_pdb_path(target_name: str, graph_group_name: str) -> tuple[str, str]:
    sampled_dir = Path(f"sampled_{target_name}")
    pdb_filename = f"{graph_group_name}_H_0001.pdb"

    if POSITIVE_MODEL_PATTERN.fullmatch(graph_group_name):
        return str(sampled_dir / pdb_filename), "sampled"

    if NEGATIVE_MODEL_PATTERN.fullmatch(graph_group_name):
        return str(sampled_dir / "random_negatives" / pdb_filename), "random_negative"

    raise ValueError(
        "Unsupported graph/model naming scheme for "
        f"{graph_group_name!r}. Expected complex.X_X_X_corrected or complex.X_X_corrected."
    )


def reference_csv_path(reference_dir: str | Path, target_name: str) -> Path:
    return Path(reference_dir) / f"{target_name}.csv"


def reference_txt_path(reference_dir: str | Path, target_name: str) -> Path:
    return Path(reference_dir) / f"{target_name}.txt"


def target_output_hdf5_path(output_dir: str | Path, target_name: str) -> Path:
    return Path(output_dir) / f"{target_name}_voronoi_contact_areas.hdf5"


def target_summary_csv_path(output_dir: str | Path, target_name: str) -> Path:
    return Path(output_dir) / f"{target_name}_voronoi_contact_areas_summary.csv"


def default_graph_data_path(cluster: bool = False) -> str:
    if cluster:
        return str(CLUSTER_GRAPH_DATA_DIR)

    env_path = os.environ.get(GRAPH_DATA_ENV_VAR)
    if env_path:
        return env_path
    return str(DEFAULT_GRAPH_DATA_DIR)
