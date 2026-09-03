"""Framework-agnostic per-case graph dataset, plus a thin PyTorch Geometric wrapper."""
import numpy as np
import torch
from torch_geometric.data import Data, Dataset

from src.data import load_case
from src.graph import airfoil_polyline, build_graph, build_subsampled_graph, radius_graph_edges
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
        # Raw (pre-normalization) surface normal -- columns 5:7. Needed as an
        # actual direction vector for the wall-shear-gradient proxy loss
        # (src/train.py) -- normalizing a unit vector by subtracting a mean
        # and dividing by a std would distort it away from being a direction.
        normal = node_features[:, 5:7].copy()

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
            "normal": normal,
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
        normal=torch.tensor(item["normal"], dtype=torch.float32),
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
        # Cached as float16 on disk (src/preprocess.py) -- cast up to float32
        # immediately so every downstream computation (normalization, model
        # forward/backward, the distance-weighted and wss-proxy losses) runs
        # at the same precision it always has; only the on-disk footprint
        # changed, not the training math.
        x5 = item["node_features"].float()
        targets = item["targets"].float()
        # Some cases cached before src/preprocess.py clamped raw targets can
        # still have literal inf baked in (float16 overflow at cache time --
        # see preprocess_case) -- nan_to_num here fixes those already-built
        # caches without needing a full re-preprocess.
        targets = torch.nan_to_num(targets, nan=0.0, posinf=6.0e4, neginf=-6.0e4)
        edge_index = item["edge_index"].long()  # PyG requires int64 edge_index
        surface = item["surface"].bool()

        # edge_attr isn't cached (see src/preprocess.py) -- recompute from RAW
        # (pre-normalization) position, matching what build_graph() would return.
        position = x5[:, :2]
        edge_attr = position[edge_index[1]] - position[edge_index[0]]
        # Raw wall distance (column 2) -- needed in physical units for the
        # distance-weighted loss (src/train.py).
        wall_distance = x5[:, 2:3].clone()
        # normal isn't cached densely (see src/preprocess.py -- it's ~0.5%
        # nonzero) -- scatter the cached surface-only rows back into a full
        # (N, 2) array using the surface mask. A real direction vector, so it
        # must stay unnormalized (see AirfRANSGraphDataset.__getitem__ above).
        normal = torch.zeros(x5.size(0), 2, dtype=x5.dtype)
        normal[surface] = item["normal_at_surface"].float()
        x = torch.cat([x5, normal], dim=1)

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
            surface=surface,
            wall_distance=wall_distance,
            normal=normal,
        )


class CachedRadiusSubsampledDataset(Dataset):
    """Same cache as CachedPyGAirfRANSDataset, but instead of the real mesh
    connectivity, resamples n_nodes and rebuilds a radius-graph fresh on
    every get() call -- the actual training regime behind the AirfRANS
    paper's own best baseline (GraphSAGE, Cd relative error ~4%), not our
    from-scratch attempt to fix Cd via loss-shape tuning. See
    build_radius_subsampled_graph / radius_graph_edges (src/graph.py) for
    why this differs from the mesh's own edges and why cKDTree instead of
    torch_geometric.nn.radius_graph.

    Called once per epoch per case by the DataLoader (PyG re-fetches via
    get() every epoch), so the subsample and its graph are genuinely
    different each epoch -- matching the paper's per-epoch resampling,
    and providing implicit data augmentation the fixed-mesh dataset never
    had.
    """

    def __init__(self, cache_dir, names, stats=None, n_nodes=32000, r=0.05, max_neighbors=64):
        super().__init__()
        self.cache_dir = cache_dir
        self.names = names
        self.stats = stats
        self.n_nodes = n_nodes
        self.r = r
        self.max_neighbors = max_neighbors

    def len(self):
        return len(self.names)

    def get(self, idx):
        name = self.names[idx]
        item = torch.load(cache_path(self.cache_dir, name), weights_only=True)
        x5_full = item["node_features"].float()
        targets_full = item["targets"].float()
        # See CachedPyGAirfRANSDataset.get() -- fixes caches built before
        # preprocess_case clamped raw targets, where float16 overflow left
        # literal inf baked in for a couple of cases (stagnation-point
        # pressure spikes ~80k-97k, past float16's ~65504 max).
        targets_full = torch.nan_to_num(targets_full, nan=0.0, posinf=6.0e4, neginf=-6.0e4)
        surface_full = item["surface"].bool()
        normal_full = torch.zeros(x5_full.size(0), 2, dtype=x5_full.dtype)
        normal_full[surface_full] = item["normal_at_surface"].float()

        n_total = x5_full.size(0)
        sub_idx = torch.from_numpy(
            np.random.default_rng().choice(n_total, size=min(self.n_nodes, n_total), replace=False)
        )
        x5 = x5_full[sub_idx]
        targets = targets_full[sub_idx]
        surface = surface_full[sub_idx]
        normal = normal_full[sub_idx]
        wall_distance = x5[:, 2:3].clone()

        position = x5[:, :2].numpy()
        # Full-resolution surface points (not the subsample) -- see
        # airfoil_polyline's docstring for why the trailing edge needs full
        # resolution to represent correctly.
        poly = airfoil_polyline(x5_full[surface_full, :2].numpy())
        edge_index_np, edge_attr_np = radius_graph_edges(
            position, self.r, self.max_neighbors,
            wall_distance=wall_distance[:, 0].numpy(), surface_polyline=poly,
        )
        edge_index = torch.tensor(edge_index_np, dtype=torch.long)
        # Real mesh edges (build_graph) are chord-fraction-scale (~1e-4 to
        # 1e-2), the edge encoder's weight init implicitly assumes inputs in
        # that range. A radius-graph edge can be as long as `r` itself --
        # up to ~500x larger -- so leaving edge_attr in raw units feeds the
        # encoder values far outside what its init was ever calibrated for.
        # Dividing by r bounds every component to roughly [-1, 1], the same
        # scale the model has always seen. Confirmed via detect_anomaly=True
        # that raw edge_attr produced a NaN in the decoder's Addmm backward
        # on the very first training step (fresh weights, before any update
        # -- ruling out a training-dynamics divergence and pointing at input
        # scale instead).
        edge_attr = torch.tensor(edge_attr_np, dtype=torch.float32) / self.r

        x = torch.cat([x5, normal], dim=1)

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
            surface=surface,
            wall_distance=wall_distance,
            normal=normal,
        )
