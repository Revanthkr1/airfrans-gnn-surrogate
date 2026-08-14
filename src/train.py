"""Week 4: full-dataset training on the official 'full' split.

Colab-GPU job -- full-resolution graphs (~180k nodes / ~720k edges per case)
hung the local dev machine even 5 at a time (see src/graph.py). Run
src/preprocess.py over the split first so this doesn't re-parse VTU every epoch.

The official 200-case test split is deliberately NOT touched here -- it's held
out for week 5's final evaluation (lift/drag surface integration). This module
carves its own validation set out of the 800 training cases instead.
"""
import glob
import os
import shutil

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch_geometric.loader import DataLoader

from src.data import split_names
from src.dataset import CachedPyGAirfRANSDataset
from src.metrics import mean_abs_error_per_field, relative_l2_per_field
from src.model import MeshGraphNet

# Sized for real GPU training, not the tiny Week-3 local CPU overfit sanity check
# (latent_dim=32, hidden_dim=64, n_message_passing=4, still MeshGraphNet's own
# defaults -- used directly by notebooks/week3_overfit.py). First real training
# run on this bigger size got 0.24-0.57 relative L2 per field and a Cd relative
# L2 of 3.03 (Cl was a much more usable 0.29) -- see src/evaluate.py. Roughly
# 4x the activation memory of the previous size; estimated to still fit a 16GB
# T4 with precision="16-mixed" and batch_size=1, but this is deliberately NOT
# the largest size discussed (128/128/10 was estimated closer to ~7.5x memory,
# risking OOM again) -- a smaller, safer step up first.
DEFAULT_MODEL_KWARGS = {"latent_dim": 64, "hidden_dim": 128, "n_message_passing": 8}

# Surface nodes are only ~0.56% of a typical mesh (1025 of 181794, measured
# directly) -- a binary "surface-only" weight would also be physically too
# narrow anyway: wall shear stress (what drag actually depends on) comes from
# a velocity *gradient* computed over a neighborhood of near-wall points, not
# just the literal zero-distance surface line. So this weights every node by
# closeness to the wall (using the already-available wall_distance/sdf
# feature) instead of a hard surface/not-surface split. LENGTH_SCALE is a
# heuristic (~5% of the ~1-unit chord), not a precisely derived boundary-layer
# thickness -- tunable if evaluation shows it's off.
WALL_WEIGHT_PEAK = 20.0
WALL_WEIGHT_LENGTH_SCALE = 0.05


def distance_weighted_mse(pred, target, wall_distance, peak_weight, length_scale):
    weight = 1.0 + (peak_weight - 1.0) * torch.exp(-wall_distance / length_scale)
    return (weight * (pred - target) ** 2).mean()


class TrainModule(L.LightningModule):
    def __init__(
        self,
        target_mean,
        target_std,
        lr=1e-3,
        max_epochs=100,
        wall_weight_peak=WALL_WEIGHT_PEAK,
        wall_weight_length_scale=WALL_WEIGHT_LENGTH_SCALE,
        **model_kwargs,
    ):
        super().__init__()
        self.model = MeshGraphNet(**model_kwargs)
        self.register_buffer("target_mean", torch.as_tensor(target_mean, dtype=torch.float32))
        self.register_buffer("target_std", torch.as_tensor(target_std, dtype=torch.float32))
        self.lr = lr
        self.max_epochs = max_epochs
        self.wall_weight_peak = wall_weight_peak
        self.wall_weight_length_scale = wall_weight_length_scale

    def forward(self, batch):
        return self.model(batch.x, batch.edge_index, batch.edge_attr)

    def training_step(self, batch, batch_idx):
        pred = self(batch)
        loss = distance_weighted_mse(
            pred, batch.y, batch.wall_distance, self.wall_weight_peak, self.wall_weight_length_scale
        )
        self.log("train_loss", loss, batch_size=batch.num_graphs)
        return loss

    def validation_step(self, batch, batch_idx):
        pred = self(batch)
        # Plain (unweighted) loss stays the headline val_loss for comparability
        # across runs -- the weighting only changes what's optimized, not how
        # overall convergence is judged.
        loss = torch.nn.functional.mse_loss(pred, batch.y)
        self.log("val_loss", loss, batch_size=batch.num_graphs, prog_bar=True)

        pred_phys = pred * self.target_std + self.target_mean
        target_phys = batch.y * self.target_std + self.target_mean
        for field, err in relative_l2_per_field(pred_phys, target_phys).items():
            self.log(f"val_rel_l2_{field}", err, batch_size=batch.num_graphs)

        # Cheap proxy for drag-relevant accuracy, logged every epoch -- unlike
        # true Cd/Cl (src/evaluate.py), this needs no raw Simulation/mesh object,
        # just the already-cached surface mask, so it can run inline during
        # training instead of only as a manual post-hoc checkpoint comparison
        # (which is what caught the epoch-54-vs-99 regression in the first place).
        # MAE, not relative L2: velocity (and nu_t) are ~0 at the wall by the
        # no-slip condition, so relative error's denominator blows up there.
        surface = batch.surface
        if surface.any():
            surf_errors = mean_abs_error_per_field(pred_phys[surface], target_phys[surface])
            for field, err in surf_errors.items():
                self.log(f"val_surface_mae_{field}", err, batch_size=batch.num_graphs)

    def configure_optimizers(self):
        # Fixed lr=1e-3 for all 100 epochs of the first run likely limited fine
        # convergence in the later epochs -- cosine decay to ~0 over max_epochs.
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


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
    resume_from_checkpoint=None,
    wall_weight_peak=WALL_WEIGHT_PEAK,
    wall_weight_length_scale=WALL_WEIGHT_LENGTH_SCALE,
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

    resume_from_checkpoint: path to resume from. If None, auto-detects the
    most recent periodic checkpoint in checkpoint_dir (by epoch number) and
    resumes from that -- so just re-running after a disconnect picks up where
    training left off instead of restarting at epoch 0. Pass an explicit path
    to override, or a nonexistent dir if you genuinely want to start fresh.
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
        stats["target_mean"],
        stats["target_std"],
        lr=lr,
        max_epochs=max_epochs,
        wall_weight_peak=wall_weight_peak,
        wall_weight_length_scale=wall_weight_length_scale,
        **(model_kwargs or DEFAULT_MODEL_KWARGS),
    )
    checkpoint_dir = os.path.dirname(checkpoint_path)
    os.makedirs(checkpoint_dir, exist_ok=True)
    periodic_ckpt = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="mgn-{epoch:03d}",
        every_n_epochs=checkpoint_every_n_epochs,
        save_top_k=-1,  # keep all of them -- checkpoints are a few MB, not worth pruning
    )

    if resume_from_checkpoint is None:
        # filenames zero-pad the epoch number (mgn-epoch=003.ckpt), so lexicographic
        # sort order matches numeric order -- last one is the most recent.
        existing = sorted(glob.glob(os.path.join(checkpoint_dir, "mgn-epoch=*.ckpt")))
        resume_from_checkpoint = existing[-1] if existing else None

    if resume_from_checkpoint and os.path.dirname(
        os.path.abspath(resume_from_checkpoint)
    ) != os.path.abspath(checkpoint_dir):
        # Lightning's ModelCheckpoint can end up inferring its save directory from
        # wherever the *resumed* checkpoint lives, ignoring the dirpath passed above
        # -- if that's a read-only mount (e.g. a Kaggle input Dataset), every
        # subsequent periodic save crashes with "Read-only file system" (hit this
        # for real: resumed from a Kaggle Dataset, epoch 29's save tried writing
        # back into that same read-only input path). Copying it into checkpoint_dir
        # first means there's no mismatched directory left to infer wrong.
        local_resume_path = os.path.join(
            checkpoint_dir, os.path.basename(resume_from_checkpoint)
        )
        shutil.copy2(resume_from_checkpoint, local_resume_path)
        resume_from_checkpoint = local_resume_path

    if resume_from_checkpoint:
        print(f"Resuming from {resume_from_checkpoint}")

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        precision=precision,
        log_every_n_steps=10,
        logger=False,
        accumulate_grad_batches=accumulate_grad_batches,
        callbacks=[periodic_ckpt],
    )
    trainer.fit(module, train_loader, val_loader, ckpt_path=resume_from_checkpoint)

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
