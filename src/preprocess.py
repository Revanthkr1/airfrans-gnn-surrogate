"""One-time conversion: VTU meshes -> cached graph tensors (real mesh connectivity).

Training re-reads the same cases every epoch; without caching, that means
re-parsing VTU via PyVista every single epoch, which dominated runtime even
locally (~1-11s/case just to load). This is a Colab-scale job for a full split
(hundreds of cases), but preprocessing a case at a time is cheap and safe to
run locally too -- it was *training* on many full-res graphs at once that hung
the local machine (see src/graph.py), not loading them one at a time.
"""
import os

import numpy as np
import torch

from src.data import load_case
from src.graph import build_graph


def cache_path(cache_dir, name):
    return os.path.join(cache_dir, f"{name}.pt")


def preprocess_case(dataset_root, name, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    out_path = cache_path(cache_dir, name)
    if os.path.exists(out_path):
        return out_path

    simulation, _ = load_case(dataset_root, name)
    # edge_attr is dropped here -- it's fully determined by position + edge_index
    # (dst - src), so CachedPyGAirfRANSDataset recomputes it instead of storing it.
    # edge_index as int32 (not PyG's usual int64): node counts are ~180k, well
    # under int32 range, and it's the single biggest piece of the cache -- these
    # two changes take a case from ~24MB to ~12MB, which mattered once a Colab
    # disk filled up caching the full 800-case split.
    node_features, edge_index, _ = build_graph(simulation)
    targets = np.concatenate(
        [simulation.velocity, simulation.pressure, simulation.nu_t], axis=1
    )
    torch.save(
        {
            "node_features": torch.tensor(node_features, dtype=torch.float32),
            "edge_index": torch.tensor(edge_index, dtype=torch.int32),
            "targets": torch.tensor(targets, dtype=torch.float32),
            "surface": torch.tensor(simulation.surface),
            "name": name,
        },
        out_path,
    )
    return out_path


def preprocess_split(dataset_root, names, cache_dir, log_every=50):
    for i, name in enumerate(names):
        preprocess_case(dataset_root, name, cache_dir)
        if (i + 1) % log_every == 0:
            print(f"cached {i + 1}/{len(names)}", flush=True)
