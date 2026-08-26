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
import re
import shutil

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter

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

# Minimum direction cosine between a candidate edge and the local outward
# normal for wall_shear_gradient_proxy to use it (see that function's
# docstring). Deliberately a *cosine* (scale-invariant), not a raw distance --
# an absolute-length threshold either zeroes out every real edge (too big,
# confirmed locally: 1e-3 was larger than every real near-wall edge's raw
# projected length in a test case, ~1e-4 to 7e-4, and filtered out 100% of
# them) or lets through near-tangent edges whose alignment is only positive
# due to floating-point noise, dividing by which blows the estimate up by
# orders of magnitude (confirmed locally: produced a "proxy MSE" of ~1e9).
# 0.3 requires the edge to point at least somewhat toward straight-out, not
# just technically away from tangent.
WSS_PROXY_MIN_COS = 0.3
# Last-resort floor on the finite-difference denominator itself, purely to
# avoid a literal division by an exactly-zero-length edge -- not expected to
# ever actually bind once WSS_PROXY_MIN_COS is filtering by direction.
WSS_PROXY_MIN_ALIGNMENT = 1e-8
# Hard clamp on the proxy's per-component output. This is what should have
# been here from the start: a real Kaggle run silently froze at ~2700
# optimizer steps because this proxy is computed under precision="16-mixed"
# (fp16 max ~65504), and a division by a near-zero denominator on a real
# mesh edge produced a value in the hundreds of millions -- overflowing to
# inf and poisoning the AMP gradient scaler into skipping every subsequent
# step, even though the term was never added to the loss (weight was 0).
# 50 keeps (pred - true)^2 comfortably under fp16's range with real margin;
# this proxy is explicitly not trusted for precision anyway (see the
# WSS_PROXY_WEIGHT docstring), so clamping this aggressively costs nothing
# real.
WSS_PROXY_MAX_ABS = 50.0
# Off by default (0.0) and should STAY off -- see ARCHITECTURE.md section 11.
# A local check of this proxy's edge selection on a real training case found
# it's not reliable on this dataset's actual mesh topology: for a surface
# node's neighbors with a positive (outward) cosine to the local normal, the
# MEDIAN cosine was 0.0002 (i.e. barely distinguishable from tangent) and only
# 2 edges in the entire ~720k-edge mesh exceeded cos>0.1 -- a genuinely
# wall-normal-aligned mesh edge into the interior is rare, not the common
# case this proxy assumed. Before this is trustworthy even as a *logged*
# diagnostic, it needs validating against AirfRANS's own wall shear stress
# (Simulation.wallshearstress()) on a held-out case to see if it correlates
# at all, and likely needs a proper multi-neighbor least-squares gradient
# (or a KDTree-based nearest-off-wall-point lookup) instead of a single
# nearest-mesh-edge finite difference. Left in place, inert, for that future
# work rather than removed outright.
WSS_PROXY_WEIGHT = 0.0


def distance_weighted_mse(pred, target, wall_distance, peak_weight, length_scale):
    weight = 1.0 + (peak_weight - 1.0) * torch.exp(-wall_distance / length_scale)
    return (weight * (pred - target) ** 2).mean()


def wall_shear_gradient_proxy(
    velocity, edge_index, edge_attr, normal, surface, min_cos=WSS_PROXY_MIN_COS, min_alignment=WSS_PROXY_MIN_ALIGNMENT
):
    """Cheap, self-consistent proxy for the wall-normal velocity gradient that
    friction drag (cdv, see ARCHITECTURE.md section 11) actually depends on --
    NOT AirfRANS's own wall shear stress (that needs a full VTK least-squares
    neighborhood derivative, impractical to differentiate through every
    training step). For each mesh edge leaving a surface node whose direction
    is reasonably aligned with that node's outward normal (cosine > min_cos),
    project the edge onto the normal and take a one-sided finite difference of
    velocity along it; average over every such edge per surface node. Only
    ever compared against this SAME proxy computed from the true velocity
    field (never against AirfRANS's real wall shear stress value), so the
    loss is well-posed regardless of how physically accurate the proxy is in
    absolute terms -- it just has to move the same way true and predicted
    velocity do.

    Filtering by direction *cosine* rather than raw projected length matters:
    an absolute-length cutoff either zeroes out every real edge (too big
    relative to this dataset's near-wall mesh scale) or lets through
    near-tangent edges whose alignment is positive only from floating-point
    noise, exploding the estimate when divided by it (both failure modes
    confirmed in a local smoke test -- see WSS_PROXY_MIN_COS's comment).

    velocity: (N, 2). edge_index: (2, E). edge_attr: (E, 2) = position[dst] -
    position[src], raw units. normal: (N, 2), raw unit vectors (zero off the
    surface). surface: (N,) bool.
    """
    src, dst = edge_index
    edge_length = edge_attr.norm(dim=-1)
    alignment = (edge_attr * normal[src]).sum(dim=-1)  # (E,) projected length
    cos_theta = alignment / edge_length.clamp(min=min_alignment)
    valid = surface[src] & (cos_theta > min_cos)
    if not valid.any():
        return torch.zeros_like(velocity)

    directional_deriv = (velocity[dst] - velocity[src]) / alignment.clamp(min=min_alignment).unsqueeze(-1)
    directional_deriv = directional_deriv.clamp(min=-WSS_PROXY_MAX_ABS, max=WSS_PROXY_MAX_ABS)

    idx = src[valid]
    contrib = directional_deriv[valid]
    summed = scatter(contrib, idx, dim=0, dim_size=velocity.size(0), reduce="sum")
    count = scatter(
        torch.ones(idx.size(0), device=velocity.device), idx, dim=0, dim_size=velocity.size(0), reduce="sum"
    )
    return summed / count.clamp(min=1).unsqueeze(-1)


class TrainModule(L.LightningModule):
    def __init__(
        self,
        target_mean,
        target_std,
        lr=1e-3,
        max_epochs=100,
        wall_weight_peak=WALL_WEIGHT_PEAK,
        wall_weight_length_scale=WALL_WEIGHT_LENGTH_SCALE,
        wss_proxy_weight=WSS_PROXY_WEIGHT,
        wss_proxy_min_cos=WSS_PROXY_MIN_COS,
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
        self.wss_proxy_weight = wss_proxy_weight
        self.wss_proxy_min_cos = wss_proxy_min_cos
        self._last_param_fingerprint = None
        self._frozen_epochs = 0

    def on_validation_epoch_end(self):
        # Frozen-weights canary. Hit this exact failure mode twice already
        # (fp16 overflow poisoning the AMP grad scaler into silently
        # skipping every optimizer step -- ARCHITECTURE.md section 11) --
        # both times it went undetected for dozens of epochs / hours of
        # real GPU time because nothing was actually watching for it, only
        # caught after the fact by manually diffing two checkpoints'
        # state_dicts. This does that check automatically, every epoch,
        # cheaply (one sum over all params). Two consecutive identical
        # fingerprints is decisive -- a real training step essentially
        # never leaves every single parameter bit-for-bit unchanged.
        with torch.no_grad():
            fingerprint = sum(p.sum().item() for p in self.parameters())
        if self._last_param_fingerprint is not None and fingerprint == self._last_param_fingerprint:
            self._frozen_epochs += 1
        else:
            self._frozen_epochs = 0
        self._last_param_fingerprint = fingerprint
        self.log("frozen_epochs", float(self._frozen_epochs))
        if self._frozen_epochs >= 2:
            print(
                f"\n{'!' * 70}\nWEIGHTS HAVE NOT CHANGED FOR {self._frozen_epochs + 1} EPOCHS "
                f"(param fingerprint identical). Stopping training now instead of "
                f"wasting further GPU time -- this exact signature has meant a "
                f"poisoned AMP grad scaler both previous times (see ARCHITECTURE.md "
                f"section 11). The last checkpoint before this run's weights froze "
                f"is still the one to resume from, not the most recent one.\n{'!' * 70}\n",
                flush=True,
            )
            self.trainer.should_stop = True

    def forward(self, batch):
        return self.model(batch.x, batch.edge_index, batch.edge_attr)

    def _wss_proxy_mse(self, pred_phys, target_phys, batch):
        # Physical-unit velocity in, physical-unit velocity out -- edge_attr
        # and normal are already raw (see src/dataset.py), and the gradient
        # estimate itself is only meaningful in physical units (chord-scale
        # distances, not per-feature-normalized ones).
        wss_pred = wall_shear_gradient_proxy(
            pred_phys[:, :2], batch.edge_index, batch.edge_attr, batch.normal, batch.surface,
            min_cos=self.wss_proxy_min_cos,
        )
        wss_true = wall_shear_gradient_proxy(
            target_phys[:, :2], batch.edge_index, batch.edge_attr, batch.normal, batch.surface,
            min_cos=self.wss_proxy_min_cos,
        )
        surface = batch.surface
        if not surface.any():
            return torch.zeros((), device=pred_phys.device)
        return torch.nn.functional.mse_loss(wss_pred[surface], wss_true[surface])

    def training_step(self, batch, batch_idx):
        pred = self(batch)
        loss = distance_weighted_mse(
            pred, batch.y, batch.wall_distance, self.wall_weight_peak, self.wall_weight_length_scale
        )
        self.log("train_loss", loss, batch_size=batch.num_graphs)

        # Gated on weight > 0, NOT "always computed for visibility" (that was
        # the actual bug, see below) -- only touch pred_phys/target_phys and
        # run the proxy at all when it's going to be used.
        if self.wss_proxy_weight > 0:
            pred_phys = pred * self.target_std + self.target_mean
            target_phys = batch.y * self.target_std + self.target_mean
            wss_mse = self._wss_proxy_mse(pred_phys, target_phys, batch)
            self.log("train_wss_proxy_mse", wss_mse, batch_size=batch.num_graphs)
            loss = loss + self.wss_proxy_weight * wss_mse

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
            # Single scalar for ModelCheckpoint's `monitor` (see best_surface_ckpt,
            # main() below) -- val_loss/val_rel_l2_* improved every epoch through
            # epoch 91 while true Cd relative L2 (src/evaluate.py) got worse after
            # epoch 47 (see project history), so neither is a safe stand-in for
            # "pick the checkpoint to actually ship." This surface-MAE average is
            # the cheapest available proxy that's shown to track the same
            # regression without needing the full Simulation/force_coefficient()
            # pass per case.
            surf_mae_mean = sum(surf_errors.values()) / len(surf_errors)
            self.log("val_surface_mae_mean", surf_mae_mean, batch_size=batch.num_graphs)

        # Gated on weight > 0, same as training_step -- see that comment for
        # why "always compute, just for visibility" was the actual bug that
        # froze a real Kaggle run's weights for 2700+ steps (ARCHITECTURE.md
        # section 11 update). Not folded into val_surface_mae_mean /
        # best_surface_ckpt's monitor even when enabled -- this proxy is
        # still unvalidated, logged separately so it can be inspected
        # without changing the already-working checkpoint-selection criterion.
        if self.wss_proxy_weight > 0:
            wss_mse = self._wss_proxy_mse(pred_phys, target_phys, batch)
            self.log("val_wss_proxy_mse", wss_mse, batch_size=batch.num_graphs)

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
    wss_proxy_weight=WSS_PROXY_WEIGHT,
    wss_proxy_min_cos=WSS_PROXY_MIN_COS,
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
        wss_proxy_weight=wss_proxy_weight,
        wss_proxy_min_cos=wss_proxy_min_cos,
        **(model_kwargs or DEFAULT_MODEL_KWARGS),
    )
    checkpoint_dir = os.path.dirname(checkpoint_path)
    os.makedirs(checkpoint_dir, exist_ok=True)
    periodic_ckpt = ModelCheckpoint(
        dirpath=checkpoint_dir,
        # Explicit "epoch=" in the template + auto_insert_metric_name=False,
        # not relying on Lightning's own auto-insertion of "name=" before a
        # placeholder -- that behavior isn't pinned across Lightning versions
        # (Kaggle's `pip install lightning` has no version pin), and a run
        # whose installed version doesn't auto-insert "=" silently produces
        # "mgn-epoch009.ckpt" instead of "mgn-epoch=009.ckpt". The resume glob
        # below then finds nothing and every restart silently begins at epoch
        # 0 instead of resuming -- confirmed hitting this for real.
        filename="mgn-epoch={epoch:03d}",
        auto_insert_metric_name=False,
        every_n_epochs=checkpoint_every_n_epochs,
        save_top_k=-1,  # keep all of them -- checkpoints are a few MB, not worth pruning
    )
    # Task-metric selection, not "last epoch wins": every field-level metric and
    # val_loss improved monotonically through epoch 91 of the weighted-loss run
    # while true Cd relative L2 bottomed out at epoch 47 and got worse after --
    # picking the final checkpoint would ship a worse-drag model than one from
    # partway through training. val_surface_mae_mean is a cheap per-epoch proxy
    # for that downstream metric; this callback tracks the single best epoch by
    # it automatically instead of relying on a manual post-hoc sweep every run.
    best_surface_ckpt = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="mgn-best-surface-mae",
        monitor="val_surface_mae_mean",
        mode="min",
        save_top_k=1,
    )

    if resume_from_checkpoint is None:
        # Matches both "mgn-epoch=003.ckpt" (this version's explicit template
        # above) and "mgn-epoch003.ckpt" (older runs / environments where
        # Lightning's auto_insert_metric_name didn't add the "=") -- a run
        # resuming a checkpoint directory from before this fix must still
        # find it, not silently restart at epoch 0. Sorted by the extracted
        # epoch NUMBER, not lexicographically -- "=" (0x3D) sorts after every
        # digit character, so a plain string sort would put "mgn-epoch009"
        # ahead of "mgn-epoch=015" and pick the wrong "latest" whenever both
        # naming styles coexist in the same directory.
        existing = glob.glob(os.path.join(checkpoint_dir, "mgn-epoch*.ckpt"))
        with_epoch = []
        for path in existing:
            match = re.search(r"epoch=?(\d+)\.ckpt$", os.path.basename(path))
            if match:
                with_epoch.append((int(match.group(1)), path))
        with_epoch.sort(key=lambda item: item[0])
        resume_from_checkpoint = with_epoch[-1][1] if with_epoch else None

    if resume_from_checkpoint:
        # Two independent reasons a resumed checkpoint's ORIGINAL location
        # can't be trusted as writable, regardless of what directory-equality
        # checks say (a prior version of this code tried to detect the
        # mismatch instead of just always avoiding it, and that detection
        # silently failed to trigger on a real Kaggle run for reasons that
        # weren't worth chasing further):
        #   1. It may not even be in checkpoint_dir (e.g. a Kaggle input
        #      Dataset, read-only).
        #   2. Lightning's ModelCheckpoint saves each callback's OWN internal
        #      state (best_model_path/dirpath/etc.) INSIDE the checkpoint
        #      file -- confirmed directly by inspecting one. If any past run
        #      ever pointed best_surface_ckpt's dirpath somewhere that's
        #      since unwritable, that stale path is embedded in the file
        #      itself and gets restored into the live callback on resume,
        #      independent of where the file currently lives.
        # So: unconditionally copy (if needed) into checkpoint_dir AND strip
        # embedded callback state, every time, rather than trying to detect
        # which of these needs doing. Costs only each ModelCheckpoint's
        # "best score so far" bookkeeping, never the model/optimizer/epoch
        # state actually needed to resume correctly.
        local_resume_path = os.path.join(
            checkpoint_dir, os.path.basename(resume_from_checkpoint)
        )
        if os.path.abspath(resume_from_checkpoint) != os.path.abspath(local_resume_path):
            shutil.copy2(resume_from_checkpoint, local_resume_path)
        resume_from_checkpoint = local_resume_path

        ckpt = torch.load(resume_from_checkpoint, map_location="cpu", weights_only=False)
        if ckpt.get("callbacks"):
            ckpt["callbacks"] = {}
        torch.save(ckpt, resume_from_checkpoint)
        print(f"Resuming from {resume_from_checkpoint}")

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        precision=precision,
        log_every_n_steps=10,
        logger=False,
        accumulate_grad_batches=accumulate_grad_batches,
        callbacks=[periodic_ckpt, best_surface_ckpt],
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
