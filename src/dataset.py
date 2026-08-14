"""Framework-agnostic per-case graph dataset, plus a thin PyTorch Geometric wrapper."""
import numpy as np
import torch
from torch_geometric.data import Data, Dataset

from src.data import load_case
from src.graph import build_graph, build_subsampled_graph
from src.preprocess import cache_path


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

        # Raw (pre-normalization) wall distance -- column 2 of node_features,
        # see src/graph.py. Needed in physical units for the distance-weighted
        # loss (src/train.py), so this is captured before normalizing below.
        wall_distance = node_features[:, 2:3].copy()

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
            "wall_distance": wall_distance,
        }


def to_pyg_data(item):
    return Data(
        x=torch.tensor(item["node_features"], dtype=torch.float32),
        edge_index=torch.tensor(item["edge_index"], dtype=torch.long),
        edge_attr=torch.tensor(item["edge_attr"], dtype=torch.float32),
        y=torch.tensor(item["targets"], dtype=torch.float32),
        name=item["name"],
        surface=torch.tensor(item["surface"], dtype=torch.bool),
        wall_distance=torch.tensor(item["wall_distance"], dtype=torch.float32),
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


class CachedPyGAirfRANSDataset(Dataset):
    """Reads pre-built graph tensors from src.preprocess's cache -- no VTU parsing
    at train time. Run src.preprocess.preprocess_split over `names` first."""

    def __init__(self, cache_dir, names, stats=None):
        super().__init__()
        self.cache_dir = cache_dir
        self.names = names
        self.stats = stats

    def len(self):
        return len(self.names)

    def get(self, idx):
        name = self.names[idx]
        item = torch.load(cache_path(self.cache_dir, name), weights_only=True)
        x, targets = item["node_features"], item["targets"]
        edge_index = item["edge_index"].long()  # PyG requires int64 edge_index

        # edge_attr isn't cached (see src/preprocess.py) -- recompute from RAW
        # (pre-normalization) position, matching what build_graph() would return.
        position = x[:, :2]
        edge_attr = position[edge_index[1]] - position[edge_index[0]]
        # Raw wall distance (column 2), captured before x gets normalized below --
        # needed in physical units for the distance-weighted loss (src/train.py).
        wall_distance = x[:, 2:3].clone()

        if self.stats is not None:
            node_mean = torch.as_tensor(self.stats["node_mean"], dtype=torch.float32)
            node_std = torch.as_tensor(self.stats["node_std"], dtype=torch.float32)
            target_mean = torch.as_tensor(self.stats["target_mean"], dtype=torch.float32)
            target_std = torch.as_tensor(self.stats["target_std"], dtype=torch.float32)
            x = (x - node_mean) / node_std
            targets = (targets - target_mean) / target_std

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=targets,
            name=item["name"],
            surface=item["surface"].bool(),
            wall_distance=wall_distance,
        )
