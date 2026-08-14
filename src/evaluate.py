"""Week 5: evaluate a trained checkpoint on the official held-out test split.

Lift/drag reuses AirfRANS's own Simulation.force_coefficient() rather than
reimplementing wall-shear-stress + surface integration by hand -- self.velocity
and self.pressure are plain overwritable numpy attributes, so assigning our
model's predictions into them lets force_coefficient() compute lift/drag from
our fields using the library's own validated code. reference=True/False on the
same call reads either the original VTU fields or whatever we just assigned,
so both predicted and reference coefficients come off the same Simulation
object without needing to reload or reset() anything.
"""
import numpy as np
import torch

from src.data import load_case, split_names
from src.graph import build_graph
from src.metrics import relative_l2_per_field
from src.train import DEFAULT_MODEL_KWARGS, TrainModule


def load_trained_model(checkpoint_path, **model_kwargs):
    # target_mean/target_std are only placeholders here (need the right shape,
    # (4,)) -- load_state_dict overwrites these registered buffers with the
    # real values saved in the checkpoint right after construction.
    #
    # model_kwargs defaults to DEFAULT_MODEL_KWARGS (src/train.py) -- without
    # this, TrainModule falls back to MeshGraphNet's own tiny class defaults
    # (32/64/4) regardless of what the checkpoint was actually trained with,
    # and load_state_dict fails on a shape mismatch. If you trained with
    # explicitly different model_kwargs, pass the same ones here.
    module = TrainModule.load_from_checkpoint(
        checkpoint_path,
        target_mean=np.zeros(4),
        target_std=np.ones(4),
        map_location="cpu",
        **(model_kwargs or DEFAULT_MODEL_KWARGS),
    )
    module.eval()
    return module.model


@torch.no_grad()
def evaluate_case(model, dataset_root, name, stats):
    simulation, _ = load_case(dataset_root, name)
    node_features, edge_index, edge_attr = build_graph(simulation)

    x = torch.tensor(
        (node_features - stats["node_mean"]) / stats["node_std"], dtype=torch.float32
    )
    edge_index_t = torch.tensor(edge_index, dtype=torch.long)
    edge_attr_t = torch.tensor(edge_attr, dtype=torch.float32)

    pred_norm = model(x, edge_index_t, edge_attr_t).numpy()
    pred = pred_norm * stats["target_std"] + stats["target_mean"]

    target = np.concatenate(
        [simulation.velocity, simulation.pressure, simulation.nu_t], axis=1
    )
    field_errors = relative_l2_per_field(torch.tensor(pred), torch.tensor(target))

    # Overwriting these doesn't touch the underlying VTU-loaded mesh data --
    # force_coefficient(reference=True) reads that directly, unaffected.
    simulation.velocity = pred[:, :2]
    simulation.pressure = pred[:, 2:3]
    (cd_pred, _, _), (cl_pred, _, _) = simulation.force_coefficient(reference=False)
    (cd_ref, _, _), (cl_ref, _, _) = simulation.force_coefficient(reference=True)

    return {
        "name": name,
        "field_errors": field_errors,
        "cd_pred": float(cd_pred),
        "cd_ref": float(cd_ref),
        "cl_pred": float(cl_pred),
        "cl_ref": float(cl_ref),
    }


def evaluate_split(
    checkpoint_path,
    dataset_root,
    stats_path,
    task="full",
    log_every=20,
    names=None,
    **model_kwargs,
):
    """names: evaluate this explicit list of cases instead of the full official
    test split -- e.g. a small fixed subset for comparing several checkpoints
    quickly (see notebooks/compare_checkpoints.py), where running the full
    200-case split per checkpoint would take ~20-25min each."""
    stats = dict(np.load(stats_path))
    model = load_trained_model(checkpoint_path, **model_kwargs)
    if names is None:
        names = split_names(dataset_root, task=task, train=False)

    results = []
    for i, name in enumerate(names):
        results.append(evaluate_case(model, dataset_root, name, stats))
        if (i + 1) % log_every == 0:
            print(f"evaluated {i + 1}/{len(names)}", flush=True)
    return results


def summarize(results):
    """Aggregate a list of evaluate_case() results into headline numbers --
    mean relative L2 per field, and Cd/Cl relative L2 across the whole set."""
    summary = {}
    for field in FIELD_NAMES:
        errors = [r["field_errors"][field] for r in results]
        summary[f"{field}_mean"] = float(np.mean(errors))

    cd_pred = np.array([r["cd_pred"] for r in results])
    cd_ref = np.array([r["cd_ref"] for r in results])
    cl_pred = np.array([r["cl_pred"] for r in results])
    cl_ref = np.array([r["cl_ref"] for r in results])
    summary["cd_rel_l2"] = float(np.linalg.norm(cd_pred - cd_ref) / np.linalg.norm(cd_ref))
    summary["cl_rel_l2"] = float(np.linalg.norm(cl_pred - cl_ref) / np.linalg.norm(cl_ref))
    return summary
