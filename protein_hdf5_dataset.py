"""PyTorch/PyG dataset utilities for protein graph HDF5 files."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable, Sequence
import warnings

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


DEFAULT_NODE_FEATURES = ("aa_type", "chain", "interface_nodes")
DEFAULT_EDGE_FEATURES = ("interface_edges", "ca_dist")
DEFAULT_PROCESSED_GRAPH_DATA_DIR = Path("/scratch/ppi_autoencoder_code/processed_graph_data")
DEFAULT_OPTIONAL_NODE_FEATURES_DIR = Path("/scratch/ppi_autoencoder_code/rsasa_i_data")
CLUSTER_PROJECT_ROOT = Path("/home/jas485/project_pi_co54/jas485/ppi_autoencoder_project")
CLUSTER_PROCESSED_GRAPH_DATA_DIR = Path("/home/jas485/project_pi_co54/jas485/ppi_processed_graphs")
CLUSTER_OPTIONAL_NODE_FEATURES_DIR = Path("/home/jas485/project_pi_co54/jas485/rsasa_i_graph_data")
DATA_PATH_ENV_VAR = "PPI_HDF5_DATA"
OPTIONAL_NODE_FEATURE_SPECS = {
    "rsasa_i": {
        "file_suffix": "_avg_rSASA_i.csv",
        "value_column": "avg_rsasa_i",
    },
    "drsasa": {
        "file_suffix": "_avg_dRSASA.csv",
        "value_column": "avg_d_rsasa",
    },
}


def default_hdf5_data_path(cluster: bool = False) -> str:
    if cluster:
        return str(CLUSTER_PROCESSED_GRAPH_DATA_DIR)

    env_path = os.environ.get(DATA_PATH_ENV_VAR)
    if env_path:
        return env_path

    for candidate in (
        DEFAULT_PROCESSED_GRAPH_DATA_DIR,
        Path("data_for_testing"),
        Path("data_for_training"),
    ):
        if candidate.exists():
            return str(candidate)
    return "data_for_testing"


def default_optional_node_features_dir(cluster: bool = False) -> str:
    if cluster:
        return str(CLUSTER_OPTIONAL_NODE_FEATURES_DIR)
    return str(DEFAULT_OPTIONAL_NODE_FEATURES_DIR)


def apply_cluster_path_defaults(args) -> None:
    cluster_enabled = bool(getattr(args, "cluster", False))
    if getattr(args, "data", None) is None:
        args.data = default_hdf5_data_path(cluster=cluster_enabled)
    if hasattr(args, "optional_node_features_dir") and getattr(args, "optional_node_features_dir", None) is None:
        args.optional_node_features_dir = default_optional_node_features_dir(cluster=cluster_enabled)


@dataclass(frozen=True)
class HDF5GraphKey:
    path: Path
    group_name: str
    format: str


def _resolve_hdf5_paths(paths: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(paths, (str, Path)):
        paths = [paths]

    files: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            files.extend(sorted(path.glob("*.h5")))
            files.extend(sorted(path.glob("*.hdf5")))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"HDF5 path does not exist: {path}")

    unique_files = sorted({file.resolve() for file in files})
    if not unique_files:
        raise FileNotFoundError(f"No .h5 or .hdf5 files found in: {paths}")
    return unique_files


def _as_float_matrix(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"Expected a 1D or 2D feature array, got shape {array.shape}.")
    return array.astype(np.float32, copy=False)


def _stack_feature_group(group: h5py.Group, feature_names: Sequence[str]) -> np.ndarray:
    arrays = []
    missing = []
    for name in feature_names:
        if name not in group:
            missing.append(name)
            continue
        arrays.append(_as_float_matrix(group[name][()]))

    if missing:
        available = ", ".join(sorted(group.keys()))
        raise KeyError(f"Missing feature(s) {missing}; available features: {available}")
    if not arrays:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate(arrays, axis=1)


def _concatenate_feature_matrices(matrices: Sequence[np.ndarray]) -> np.ndarray:
    non_empty = [matrix for matrix in matrices if matrix.size > 0]
    if not non_empty:
        return np.empty((0, 0), dtype=np.float32)
    row_counts = {matrix.shape[0] for matrix in non_empty}
    if len(row_counts) != 1:
        raise ValueError(f"Cannot concatenate feature matrices with mismatched row counts: {sorted(row_counts)}")
    return np.concatenate(non_empty, axis=1)


class ProteinGraphHDF5Dataset(Dataset):
    """Lazily read PPI graph samples from one or more HDF5 files.

    Rich graph files created by ``create_graph_format.py`` are exposed as PyG
    ``Data`` objects with:

    - ``x``: residue node features built from ``node_features``.
    - ``edge_index``: contact pairs from ``edge_features/contacts``.
    - ``edge_attr``: edge/contact features built from ``edge_features``.
    - ``y``: scalar target, usually ``target_scores/DockQ`` when present.

    The older ``test_targets/test-data.h5`` format with ``X`` node features and
    dense ``A`` adjacency matrices is also supported for smoke testing.
    """

    def __init__(
        self,
        paths: str | Path | Sequence[str | Path],
        node_features: Sequence[str] = DEFAULT_NODE_FEATURES,
        edge_features: Sequence[str] = DEFAULT_EDGE_FEATURES,
        target_name: str = "DockQ",
        require_target: bool = False,
        make_undirected: bool = True,
        include_self_edges: bool = False,
        max_samples: int | None = None,
        skip_invalid_files: bool = True,
        optional_node_features_dir: str | Path = DEFAULT_OPTIONAL_NODE_FEATURES_DIR,
    ) -> None:
        self.paths = _resolve_hdf5_paths(paths)
        self.node_features = tuple(node_features)
        self.edge_features = tuple(edge_features)
        self.target_name = target_name
        self.require_target = require_target
        self.make_undirected = make_undirected
        self.include_self_edges = include_self_edges
        self.skip_invalid_files = skip_invalid_files
        self.optional_node_features_dir = Path(optional_node_features_dir)
        self.hdf5_node_features = tuple(
            feature_name
            for feature_name in self.node_features
            if feature_name not in OPTIONAL_NODE_FEATURE_SPECS
        )
        self.optional_node_features = tuple(
            feature_name
            for feature_name in self.node_features
            if feature_name in OPTIONAL_NODE_FEATURE_SPECS
        )
        unknown_node_features = sorted(
            set(self.node_features) - set(self.hdf5_node_features) - set(self.optional_node_features)
        )
        if unknown_node_features:
            raise KeyError(f"Unsupported node feature(s): {unknown_node_features}")
        self.skipped_files: list[tuple[Path, str]] = []
        self.optional_node_feature_values = self._load_optional_node_feature_values()
        self.samples = self._build_index(max_samples=max_samples)
        if not self.samples:
            raise ValueError("No usable graph samples were found in the provided HDF5 files.")

    def _build_index(self, max_samples: int | None) -> list[HDF5GraphKey]:
        samples: list[HDF5GraphKey] = []
        for path in self.paths:
            try:
                h5 = h5py.File(path, "r")
            except OSError as exc:
                if not self.skip_invalid_files:
                    raise
                message = f"Skipping unreadable HDF5 file {path}: {exc}"
                warnings.warn(message, RuntimeWarning, stacklevel=2)
                self.skipped_files.append((path, str(exc)))
                continue

            with h5:
                for group_name in sorted(h5.keys()):
                    group = h5[group_name]
                    if self._is_rich_graph_group(group):
                        has_target = self._has_target(group)
                        sample = HDF5GraphKey(path, group_name, "rich")
                        if self.require_target and not has_target:
                            continue
                        if not self._sample_has_optional_features(sample):
                            continue
                        samples.append(sample)
                    elif self._is_legacy_graph_group(group) and not self.require_target:
                        samples.append(HDF5GraphKey(path, group_name, "legacy_ax"))

                    if max_samples is not None and len(samples) >= max_samples:
                        return samples
        return samples

    def _load_optional_node_feature_values(self) -> dict[str, dict[str, dict[str, float]]]:
        feature_values: dict[str, dict[str, dict[str, float]]] = {}
        if not self.optional_node_features:
            return feature_values
        if not self.optional_node_features_dir.exists():
            raise FileNotFoundError(
                f"Optional node feature directory does not exist: {self.optional_node_features_dir}"
            )

        for feature_name in self.optional_node_features:
            spec = OPTIONAL_NODE_FEATURE_SPECS[feature_name]
            target_values: dict[str, dict[str, float]] = {}
            matching_files = sorted(self.optional_node_features_dir.glob(f"*{spec['file_suffix']}"))
            if not matching_files:
                raise FileNotFoundError(
                    f"No files matching *{spec['file_suffix']} were found in {self.optional_node_features_dir}"
                )
            for csv_path in matching_files:
                target_id = csv_path.name[:4]
                decoy_values: dict[str, float] = {}
                with csv_path.open(newline="", encoding="ascii") as csv_file:
                    reader = csv.DictReader(csv_file)
                    if reader.fieldnames is None or "Decoy" not in reader.fieldnames or spec["value_column"] not in reader.fieldnames:
                        raise ValueError(
                            f"{csv_path} must contain Decoy and {spec['value_column']} columns; "
                            f"found {reader.fieldnames}"
                        )
                    for row in reader:
                        decoy_name = row["Decoy"].strip()
                        if not decoy_name:
                            continue
                        raw_value = row[spec["value_column"]].strip()
                        if not raw_value:
                            continue
                        if decoy_name in decoy_values:
                            raise ValueError(f"Duplicate decoy entry {decoy_name!r} in {csv_path}")
                        decoy_values[decoy_name] = float(raw_value)
                target_values[target_id] = decoy_values
            feature_values[feature_name] = target_values
        return feature_values

    def _sample_has_optional_features(self, sample: HDF5GraphKey) -> bool:
        if not self.optional_node_features:
            return True
        target_id = sample.path.stem[:4]
        for feature_name in self.optional_node_features:
            target_values = self.optional_node_feature_values[feature_name].get(target_id)
            if target_values is None or sample.group_name not in target_values:
                return False
        return True

    def _is_rich_graph_group(self, group: h5py.Group) -> bool:
        if not all(name in group for name in ("node_features", "edge_features")):
            return False
        node_group = group["node_features"]
        edge_group = group["edge_features"]
        required_node_features = all(name in node_group for name in self.hdf5_node_features)
        required_edge_features = all(name in edge_group for name in self.edge_features)
        return required_node_features and required_edge_features and "contacts" in edge_group

    def _is_legacy_graph_group(self, group: h5py.Group) -> bool:
        return "X" in group and "A" in group

    def _has_target(self, group: h5py.Group) -> bool:
        return "target_scores" in group and self.target_name in group["target_scores"]

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def num_node_features(self) -> int:
        return int(self[0].x.size(1))

    @property
    def num_edge_features(self) -> int:
        edge_attr = getattr(self[0], "edge_attr", None)
        return 0 if edge_attr is None else int(edge_attr.size(1))

    def __getitem__(self, index: int) -> Data:
        sample = self.samples[index]
        with h5py.File(sample.path, "r") as h5:
            group = h5[sample.group_name]
            if sample.format == "rich":
                data = self._read_rich_graph(sample, group)
            elif sample.format == "legacy_ax":
                data = self._read_legacy_graph(group)
            else:
                raise ValueError(f"Unsupported graph format: {sample.format}")

        data.sample_index = torch.tensor([index], dtype=torch.long)
        data.graph_name = sample.group_name
        data.hdf5_path = str(sample.path)
        data.graph_format = sample.format
        return data

    def _build_optional_node_feature_matrix(self, sample: HDF5GraphKey, num_nodes: int) -> np.ndarray:
        if not self.optional_node_features:
            return np.empty((num_nodes, 0), dtype=np.float32)
        target_id = sample.path.stem[:4]
        feature_columns = []
        for feature_name in self.optional_node_features:
            scalar_value = self.optional_node_feature_values[feature_name][target_id][sample.group_name]
            feature_columns.append(np.full((num_nodes, 1), scalar_value, dtype=np.float32))
        return np.concatenate(feature_columns, axis=1)

    def _read_rich_graph(self, sample: HDF5GraphKey, group: h5py.Group) -> Data:
        x_hdf5 = _stack_feature_group(group["node_features"], self.hdf5_node_features)
        if x_hdf5.shape[0] > 0:
            num_nodes = x_hdf5.shape[0]
        else:
            first_node_dataset = next(iter(group["node_features"].values()))
            num_nodes = int(first_node_dataset.shape[0])
        x_optional = self._build_optional_node_feature_matrix(sample, num_nodes)
        x = _concatenate_feature_matrices((x_hdf5, x_optional))
        contacts = np.asarray(group["edge_features"]["contacts"][()], dtype=np.int64)
        edge_attr = _stack_feature_group(group["edge_features"], self.edge_features)

        if contacts.ndim != 2 or contacts.shape[1] != 2:
            raise ValueError(f"contacts must have shape [num_edges, 2], got {contacts.shape}.")
        if contacts.size and contacts.min() >= 1 and contacts.max() == x.shape[0]:
            contacts = contacts - 1

        valid_edges = (
            (contacts[:, 0] >= 0)
            & (contacts[:, 0] < x.shape[0])
            & (contacts[:, 1] >= 0)
            & (contacts[:, 1] < x.shape[0])
        )
        if not self.include_self_edges:
            valid_edges &= contacts[:, 0] != contacts[:, 1]
        contacts = contacts[valid_edges]
        edge_attr = edge_attr[valid_edges]

        edge_index = torch.from_numpy(contacts.T.copy()).long()
        edge_attr_tensor = torch.from_numpy(edge_attr).float()
        if self.make_undirected and edge_index.numel() > 0:
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            edge_attr_tensor = torch.cat([edge_attr_tensor, edge_attr_tensor], dim=0)

        data = Data(
            x=torch.from_numpy(x).float(),
            edge_index=edge_index,
            edge_attr=edge_attr_tensor,
        )
        if self._has_target(group):
            target = float(group["target_scores"][self.target_name][()])
            data.y = torch.tensor([target], dtype=torch.float32)
        return data

    def _read_legacy_graph(self, group: h5py.Group) -> Data:
        x = _as_float_matrix(group["X"][()])
        adjacency = np.asarray(group["A"][()], dtype=np.float32)
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError(f"A must be a square adjacency matrix, got {adjacency.shape}.")

        src, dst = np.nonzero(adjacency)
        if not self.include_self_edges:
            mask = src != dst
            src = src[mask]
            dst = dst[mask]

        edge_index = torch.from_numpy(np.vstack([src, dst]).astype(np.int64))
        edge_values = adjacency[src, dst].astype(np.float32)[:, None]
        return Data(
            x=torch.from_numpy(x).float(),
            edge_index=edge_index,
            edge_attr=torch.from_numpy(edge_values).float(),
        )


def build_dataloader(
    paths: str | Path | Sequence[str | Path],
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    **dataset_kwargs,
):
    from torch_geometric.loader import DataLoader

    dataset = ProteinGraphHDF5Dataset(paths, **dataset_kwargs)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return dataset, loader


def _parse_feature_list(raw: str | None, defaults: Iterable[str]) -> tuple[str, ...]:
    if raw is None:
        return tuple(defaults)
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect PPI graph HDF5 files.")
    parser.add_argument("--cluster", action="store_true", help="Use the cluster default data directories.")
    parser.add_argument("--data", default=None, help="HDF5 file or directory.")
    parser.add_argument("--target-name", default="DockQ")
    parser.add_argument("--node-features", default=None, help="Comma-separated node feature names.")
    parser.add_argument("--edge-features", default=None, help="Comma-separated edge feature names.")
    parser.add_argument("--use-rsasa-i", action="store_true", help="Append the per-decoy avg rSASA_i scalar to each node.")
    parser.add_argument("--use-drsasa", action="store_true", help="Append the per-decoy avg dRSASA scalar to each node.")
    parser.add_argument(
        "--optional-node-features-dir",
        default=None,
        help="Directory containing optional rsasa/drsasa CSV files.",
    )
    parser.add_argument("--max-samples", type=int, default=3)
    parser.add_argument("--require-target", action="store_true")
    parser.add_argument("--strict-hdf5", action="store_true", help="Fail instead of skipping unreadable HDF5 files.")
    args = parser.parse_args()
    apply_cluster_path_defaults(args)

    node_features = list(_parse_feature_list(args.node_features, DEFAULT_NODE_FEATURES))
    if args.use_rsasa_i and "rsasa_i" not in node_features:
        node_features.append("rsasa_i")
    if args.use_drsasa and "drsasa" not in node_features:
        node_features.append("drsasa")

    dataset = ProteinGraphHDF5Dataset(
        args.data,
        node_features=tuple(node_features),
        edge_features=_parse_feature_list(args.edge_features, DEFAULT_EDGE_FEATURES),
        target_name=args.target_name,
        require_target=args.require_target,
        max_samples=args.max_samples,
        skip_invalid_files=not args.strict_hdf5,
        optional_node_features_dir=args.optional_node_features_dir,
    )

    print(f"Loaded {len(dataset)} sample(s) from {len(dataset.paths)} HDF5 file(s).")
    if dataset.skipped_files:
        print(f"Skipped unreadable HDF5 file(s): {len(dataset.skipped_files)}")
    print(f"Node features: {dataset.node_features} -> {dataset.num_node_features} columns")
    print(f"Edge features: {dataset.edge_features} -> {dataset.num_edge_features} columns")
    for idx in range(min(len(dataset), args.max_samples)):
        data = dataset[idx]
        target = getattr(data, "y", None)
        target_text = "none" if target is None else f"{float(target.item()):.4f}"
        print(
            f"[{idx}] {data.graph_name}: nodes={data.num_nodes}, "
            f"edges={data.num_edges}, x={tuple(data.x.shape)}, "
            f"edge_attr={tuple(data.edge_attr.shape)}, target={target_text}"
        )


if __name__ == "__main__":
    main()
