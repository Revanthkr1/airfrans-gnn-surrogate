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
from src.train import TrainModule


def load_trained_model(checkpoint_path, **model_kwargs):
    # target_mean/target_std are only placeholders here (need the right shape,
    # (4,)) -- load_state_dict overwrites these registered buffers with the
    # real values saved in the checkpoint right after construction.
    module = TrainModule.load_from_checkpoint(
        checkpoint_path,
        target_mean=np.zeros(4),
        target_std=np.ones(4),
        map_location="cpu",
        **model_kwargs,
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
    checkpoint_path, dataset_root, stats_path, task="full", log_every=20, **model_kwargs
):
    stats = dict(np.load(stats_path))
    model = load_trained_model(checkpoint_path, **model_kwargs)
    test_names = split_names(dataset_root, task=task, train=False)

    results = []
    for i, name in enumerate(test_names):
        results.append(evaluate_case(model, dataset_root, name, stats))
        if (i + 1) % log_every == 0:
            print(f"evaluated {i + 1}/{len(test_names)}", flush=True)
    return results
