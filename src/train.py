"""Week 4: full-dataset training on the official 'full' split.

Colab-GPU job -- full-resolution graphs (~180k nodes / ~720k edges per case)
hung the local dev machine even 5 at a time (see src/graph.py). Run
src/preprocess.py over the split first so this doesn't re-parse VTU every epoch.

The official 200-case test split is deliberately NOT touched here -- it's held
out for week 5's final evaluation (lift/drag surface integration). This module
carves its own validation set out of the 800 training cases instead.
"""
import os

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch_geometric.loader import DataLoader

from src.data import split_names
from src.dataset import CachedPyGAirfRANSDataset
from src.metrics import relative_l2_per_field
from src.model import MeshGraphNet


class TrainModule(L.LightningModule):
    def __init__(self, target_mean, target_std, lr=1e-3, **model_kwargs):
        super().__init__()
        self.model = MeshGraphNet(**model_kwargs)
        self.register_buffer("target_mean", torch.as_tensor(target_mean, dtype=torch.float32))
        self.register_buffer("target_std", torch.as_tensor(target_std, dtype=torch.float32))
        self.lr = lr

    def forward(self, batch):
        return self.model(batch.x, batch.edge_index, batch.edge_attr)

    def training_step(self, batch, batch_idx):
        pred = self(batch)
        loss = torch.nn.functional.mse_loss(pred, batch.y)
        self.log("train_loss", loss, batch_size=batch.num_graphs)
        return loss

    def validation_step(self, batch, batch_idx):
        pred = self(batch)
        loss = torch.nn.functional.mse_loss(pred, batch.y)
        self.log("val_loss", loss, batch_size=batch.num_graphs, prog_bar=True)

        pred_phys = pred * self.target_std + self.target_mean
        target_phys = batch.y * self.target_std + self.target_mean
        for field, err in relative_l2_per_field(pred_phys, target_phys).items():
            self.log(f"val_rel_l2_{field}", err, batch_size=batch.num_graphs)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def main(
    dataset_root,
    cache_dir,
    stats_path,
    checkpoint_path,
    max_epochs=100,
    batch_size=1,
    accumulate_grad_batches=4,
    n_val=80,
    lr=1e-3,
    model_kwargs=None,
    checkpoint_every_n_epochs=5,
    num_workers=2,
    precision="32-true",
):
    """batch_size=1 by default: batching multiple full-resolution graphs (each
    ~180k nodes/~720k edges) into one forward pass blew past a 16GB T4's memory
    with batch_size=4 (~2.88M edges x 4 message-passing rounds retaining
    activations for backprop). accumulate_grad_batches recovers a larger
    effective batch size (gradient-averaged over that many steps) without
    holding more than one graph in memory at a time.

    checkpoint_every_n_epochs: this Colab session has already dropped its GPU
    and hit OOM once each -- a single end-of-training checkpoint would lose
    everything to a disconnect. Periodic checkpoints land next to
    checkpoint_path so progress survives a crash.
    """
    stats = dict(np.load(stats_path))
    all_train_names = split_names(dataset_root, task="full", train=True)
    # Official test split (200 cases) is NOT used here -- reserved for week 5.
    train_names = all_train_names[:-n_val]
    val_names = all_train_names[-n_val:]

    train_ds = CachedPyGAirfRANSDataset(cache_dir, train_names, stats=stats)
    val_ds = CachedPyGAirfRANSDataset(cache_dir, val_names, stats=stats)

    loader_kwargs = dict(
        num_workers=num_workers, persistent_workers=num_workers > 0, pin_memory=True
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)

    module = TrainModule(
        stats["target_mean"], stats["target_std"], lr=lr, **(model_kwargs or {})
    )
    checkpoint_dir = os.path.dirname(checkpoint_path)
    periodic_ckpt = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="mgn-{epoch:03d}",
        every_n_epochs=checkpoint_every_n_epochs,
        save_top_k=-1,  # keep all of them -- checkpoints are a few MB, not worth pruning
    )
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        precision=precision,
        log_every_n_steps=10,
        logger=False,
        accumulate_grad_batches=accumulate_grad_batches,
        callbacks=[periodic_ckpt],
    )
    trainer.fit(module, train_loader, val_loader)

    trainer.save_checkpoint(checkpoint_path)
    print(f"Saved final checkpoint to {checkpoint_path}")
    print(f"Periodic checkpoints (every {checkpoint_every_n_epochs} epochs) in {checkpoint_dir}")


if __name__ == "__main__":
    DATA_ROOT = "data"
    main(
        dataset_root=os.path.join(DATA_ROOT, "Dataset"),
        cache_dir=os.path.join(DATA_ROOT, "cache", "full"),
        stats_path=os.path.join(DATA_ROOT, "norm_stats.npz"),
        checkpoint_path=os.path.join(DATA_ROOT, "meshgraphnet.ckpt"),
        batch_size=1,
        accumulate_grad_batches=4,
        num_workers=2,
    )
