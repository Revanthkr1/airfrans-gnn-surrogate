"""Week 3: does the model + training loop even work? Overfit on 5 cases. Nothing else."""
import os
import sys

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data import split_names
from src.dataset import PyGAirfRANSDataset
from src.model import MeshGraphNet

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
N_CASES = 5
MAX_EPOCHS = 300
# Full-res cases (~180k nodes each) hung the machine even batched just 5 at a time.
# Subsample nodes + k-NN edges locally; full-mesh training is a Colab-time job (week 4+).
MAX_NODES_PER_CASE = 2000
K_NEIGHBORS = 6


class _CachedDataset(torch.utils.data.Dataset):
    """Holds pre-loaded Data objects in memory so the DataLoader doesn't re-read
    from disk every epoch -- irrelevant for a real training set, but here we
    deliberately reuse the same 5 cases hundreds of times."""

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class OverfitModule(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = MeshGraphNet()
        self.losses = []

    def training_step(self, batch, batch_idx):
        pred = self.model(batch.x, batch.edge_index, batch.edge_attr)
        loss = torch.nn.functional.mse_loss(pred, batch.y)
        self.losses.append(loss.item())
        print(f"epoch {self.current_epoch}: loss={loss.item():.5f}", flush=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-3)


def main():
    dataset_root = os.path.join(DATA_ROOT, "Dataset")
    stats = dict(np.load(os.path.join(DATA_ROOT, "norm_stats.npz")))
    names = split_names(dataset_root, task="full", train=True)[:N_CASES]

    ds = PyGAirfRANSDataset(
        dataset_root, names, stats=stats, max_nodes=MAX_NODES_PER_CASE, k=K_NEIGHBORS
    )
    cached = _CachedDataset([ds[i] for i in range(len(ds))])
    loader = DataLoader(cached, batch_size=N_CASES, shuffle=False)

    module = OverfitModule()
    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(module, loader)

    print(f"First loss: {module.losses[0]:.5f}, last loss: {module.losses[-1]:.5f}")

    fig, ax = plt.subplots()
    ax.plot(module.losses, marker="o")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss (normalized targets)")
    ax.set_yscale("log")
    ax.set_title(f"Overfit sanity check: {N_CASES} cases")
    out_path = os.path.join(os.path.dirname(__file__), "week3_overfit_loss.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
