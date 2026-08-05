"""Per-feature mean/std for node inputs and targets, over a (sub)sample of cases.

Vectorized sum / sum-of-squares accumulation -- no need for a numerically fancier
streaming algorithm here, these are well-scaled physical quantities, not extreme values.
"""
import numpy as np

from src.data import load_case
from src.graph import build_graph


def compute_stats(dataset_root, names):
    node_sum = node_sumsq = target_sum = target_sumsq = None
    count = 0

    for name in names:
        simulation, _ = load_case(dataset_root, name)
        node_features, _, _ = build_graph(simulation)
        targets = np.concatenate(
            [simulation.velocity, simulation.pressure, simulation.nu_t], axis=1
        )

        if node_sum is None:
            node_sum = np.zeros(node_features.shape[1])
            node_sumsq = np.zeros(node_features.shape[1])
            target_sum = np.zeros(targets.shape[1])
            target_sumsq = np.zeros(targets.shape[1])

        node_sum += node_features.sum(axis=0)
        node_sumsq += (node_features**2).sum(axis=0)
        target_sum += targets.sum(axis=0)
        target_sumsq += (targets**2).sum(axis=0)
        count += node_features.shape[0]

    node_mean = node_sum / count
    node_var = node_sumsq / count - node_mean**2
    target_mean = target_sum / count
    target_var = target_sumsq / count - target_mean**2

    return {
        "node_mean": node_mean,
        "node_std": np.sqrt(np.maximum(node_var, 1e-12)),
        "target_mean": target_mean,
        "target_std": np.sqrt(np.maximum(target_var, 1e-12)),
        "n_simulations": len(names),
        "n_nodes": count,
    }
