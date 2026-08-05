"""Framework-agnostic per-case graph dataset, plus a thin PyTorch Geometric wrapper."""
import numpy as np
import torch
from torch_geometric.data import Data, Dataset

from src.data import load_case
from src.graph import build_graph, build_subsampled_graph


class AirfRANSGraphDataset:
    def __init__(self, dataset_root, names, stats=None, max_nodes=None, k=6):
        """max_nodes: if set, use build_subsampled_graph (local-only cheap sanity
        check) instead of the real full-mesh graph -- see src/graph.py for why."""
        self.dataset_root = dataset_root
        self.names = names
        self.stats = stats
        self.max_nodes = max_nodes
        self.k = k

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        simulation, name = load_case(self.dataset_root, self.names[idx])
        if self.max_nodes is not None:
            node_features, edge_index, edge_attr, targets, sub_idx = build_subsampled_graph(
                simulation, n_nodes=self.max_nodes, k=self.k
            )
            surface = simulation.surface[sub_idx]
        else:
            node_features, edge_index, edge_attr = build_graph(simulation)
            targets = np.concatenate(
                [simulation.velocity, simulation.pressure, simulation.nu_t], axis=1
            )
            surface = simulation.surface

        if self.stats is not None:
            node_features = (node_features - self.stats["node_mean"]) / self.stats["node_std"]
            targets = (targets - self.stats["target_mean"]) / self.stats["target_std"]

        return {
            "name": name,
            "node_features": node_features,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "targets": targets,
            "surface": surface,
        }


def to_pyg_data(item):
    return Data(
        x=torch.tensor(item["node_features"], dtype=torch.float32),
        edge_index=torch.tensor(item["edge_index"], dtype=torch.long),
        edge_attr=torch.tensor(item["edge_attr"], dtype=torch.float32),
        y=torch.tensor(item["targets"], dtype=torch.float32),
        name=item["name"],
    )


class PyGAirfRANSDataset(Dataset):
    """Wraps AirfRANSGraphDataset, converting each item to a torch_geometric Data object."""

    def __init__(self, dataset_root, names, stats=None, max_nodes=None, k=6):
        super().__init__()
        self.inner = AirfRANSGraphDataset(
            dataset_root, names, stats=stats, max_nodes=max_nodes, k=k
        )

    def len(self):
        return len(self.inner)

    def get(self, idx):
        return to_pyg_data(self.inner[idx])
