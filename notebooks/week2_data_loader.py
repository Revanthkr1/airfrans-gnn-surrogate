"""Week 2: official splits + normalization stats, sanity-check the full data loader."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data import split_names
from src.dataset import AirfRANSGraphDataset
from src.normalization import compute_stats

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
STATS_PATH = os.path.join(DATA_ROOT, "norm_stats.npz")
# ~11s/case to load locally -> all 800 training cases would be ~2.5hrs. A 100-case
# random subsample gives a stable mean/std estimate in ~18min instead. Full training
# runs (and could recompute stats over everything) happen on Colab in week 4.
N_STATS_SAMPLE = 100


def main():
    dataset_root = os.path.join(DATA_ROOT, "Dataset")
    train_names = split_names(dataset_root, task="full", train=True)
    test_names = split_names(dataset_root, task="full", train=False)
    print(f"full task: {len(train_names)} train, {len(test_names)} test")

    rng = np.random.default_rng(0)
    sample_names = list(rng.choice(train_names, size=N_STATS_SAMPLE, replace=False))
    print(f"Computing normalization stats over a {N_STATS_SAMPLE}-case subsample of training data...")
    stats = compute_stats(dataset_root, sample_names)
    np.savez(STATS_PATH, **stats)
    print(f"Saved stats to {STATS_PATH}")
    print("node_mean [x, y, sdf, inlet_vx, inlet_vy]:", stats["node_mean"])
    print("node_std :", stats["node_std"])
    print("target_mean [vx, vy, pressure, nu_t]:", stats["target_mean"])
    print("target_std :", stats["target_std"])

    train_ds = AirfRANSGraphDataset(dataset_root, train_names, stats=stats)
    test_ds = AirfRANSGraphDataset(dataset_root, test_names, stats=stats)
    print(f"train dataset len: {len(train_ds)}, test dataset len: {len(test_ds)}")

    item = train_ds[0]
    print("sample item:", item["name"])
    print("node_features shape:", item["node_features"].shape)
    print("node_features mean (should be ~0):", item["node_features"].mean(axis=0))
    print("edge_index shape:", item["edge_index"].shape)
    print("targets shape:", item["targets"].shape)
    print("targets mean (should be ~0):", item["targets"].mean(axis=0))


if __name__ == "__main__":
    main()
