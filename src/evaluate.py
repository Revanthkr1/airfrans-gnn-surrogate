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
from src.metrics import FIELD_NAMES, mean_abs_error_per_field, relative_l2_per_field
from src.train import DEFAULT_MODEL_KWARGS, TrainModule

# Near-wall band, in chord units -- distinct from the literal `surface` mask
# (sdf==0 exactly, ~0.56% of nodes). Not the same length scale as training's
# WALL_WEIGHT_LENGTH_SCALE=0.05 (src/train.py): that's a smooth decay for
# loss weighting, and 61% of a typical mesh already falls under sdf<0.05 (RANS
# meshes cluster nodes tightly near the wall for boundary-layer resolution) --
# too broad to call "near-wall" as a region label. 0.01 is still large enough
# to have real coverage (~47% of nodes) while narrower than the 0.05 decay.
NEAR_WALL_THRESHOLD = 0.01
# Wake band: downstream of the trailing edge (chord runs x in [0, 1]), within
# a fixed y half-width of the centerline. This is a simplification -- it
# doesn't account for the wake deflecting at nonzero angle of attack, so at
# high AoA some true wake gets misclassified as far-field -- good enough for
# a diagnostic breakdown, not a precise wake extraction.
WAKE_X_THRESHOLD = 1.0
WAKE_Y_HALFWIDTH = 0.15


def regional_errors(
    pred,
    target,
    wall_distance,
    position,
    surface,
    near_wall_threshold=NEAR_WALL_THRESHOLD,
    wake_x_threshold=WAKE_X_THRESHOLD,
    wake_y_halfwidth=WAKE_Y_HALFWIDTH,
):
    """pred, target: (N, 4) numpy arrays in physical units. Buckets nodes into
    surface / near_wall / wake / far_field and reports the error appropriate
    to each -- MAE at/near the wall (velocity, and nu_t, are ~0 there by the
    no-slip condition, so relative L2's denominator blows up -- same reasoning
    as mean_abs_error_per_field's docstring), relative L2 elsewhere.
    """
    wall_distance = np.asarray(wall_distance).reshape(-1)
    surface = np.asarray(surface).reshape(-1).astype(bool)
    x, y = position[:, 0], position[:, 1]

    near_wall = (~surface) & (wall_distance < near_wall_threshold)
    wake = (
        ~surface
        & ~near_wall
        & (x > wake_x_threshold)
        & (np.abs(y) < wake_y_halfwidth)
    )
    far_field = ~surface & ~near_wall & ~wake

    out = {}
    for region_name, mask, use_mae in [
        ("surface", surface, True),
        ("near_wall", near_wall, True),
        ("wake", wake, False),
        ("far_field", far_field, False),
    ]:
        if not mask.any():
            continue
        p, t = torch.tensor(pred[mask]), torch.tensor(target[mask])
        out[region_name] = {
            "n": int(mask.sum()),
            "errors": mean_abs_error_per_field(p, t) if use_mae else relative_l2_per_field(p, t),
            "metric": "mae" if use_mae else "rel_l2",
        }
    return out


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
    region_errors = regional_errors(
        pred, target, node_features[:, 2:3], simulation.position, simulation.surface
    )

    # Overwriting these doesn't touch the underlying VTU-loaded mesh data --
    # force_coefficient(reference=True) reads that directly, unaffected.
    simulation.velocity = pred[:, :2]
    simulation.pressure = pred[:, 2:3]
    # cdp/cdv split pressure drag (needs only the predicted pressure field)
    # from viscous/friction drag (needs the wall-normal *gradient* of the
    # predicted velocity field, a noisier quantity than the field itself) --
    # keeping both instead of just the summed cd/cl lets us tell which half
    # is actually driving a Cd regression instead of guessing.
    (cd_pred, cdp_pred, cdv_pred), (cl_pred, clp_pred, clv_pred) = simulation.force_coefficient(
        reference=False
    )
    (cd_ref, cdp_ref, cdv_ref), (cl_ref, clp_ref, clv_ref) = simulation.force_coefficient(
        reference=True
    )

    return {
        "name": name,
        "field_errors": field_errors,
        "region_errors": region_errors,
        "cd_pred": float(cd_pred),
        "cd_ref": float(cd_ref),
        "cl_pred": float(cl_pred),
        "cl_ref": float(cl_ref),
        "cdp_pred": float(cdp_pred),
        "cdp_ref": float(cdp_ref),
        "cdv_pred": float(cdv_pred),
        "cdv_ref": float(cdv_ref),
        "clp_pred": float(clp_pred),
        "clp_ref": float(clp_ref),
        "clv_pred": float(clv_pred),
        "clv_ref": float(clv_ref),
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

    # Split by contribution -- pressure drag (cdp, needs only the predicted
    # pressure field) vs. friction drag (cdv, needs the wall-normal *gradient*
    # of the predicted velocity field). Reported separately, never blended
    # into the summed cd/cl above, since they can regress independently (see
    # ARCHITECTURE.md section 10) and a fixed cd_rel_l2 number can't show that.
    for prefix in ("cdp", "cdv", "clp", "clv"):
        pred = np.array([r[f"{prefix}_pred"] for r in results])
        ref = np.array([r[f"{prefix}_ref"] for r in results])
        summary[f"{prefix}_rel_l2"] = float(np.linalg.norm(pred - ref) / np.linalg.norm(ref))

    # Sanity check: negative predicted Cd is physically impossible (see
    # project history's cancellation-failure discussion) -- if this is ever
    # true, the model has stopped predicting drag and started predicting noise.
    summary["cd_pred_min"] = float(cd_pred.min())
    summary["cd_pred_max"] = float(cd_pred.max())
    summary["any_negative_cd_pred"] = bool((cd_pred < 0).any())
    return summary


def summarize_regions(results):
    """Mean, across cases, of each region's per-field error (see
    regional_errors) -- same simple-average-across-cases convention as
    summarize(), just broken out by surface/near_wall/wake/far_field instead
    of pooling every node together."""
    region_names = results[0]["region_errors"].keys()
    summary = {}
    for region_name in region_names:
        metric = results[0]["region_errors"][region_name]["metric"]
        summary[region_name] = {"metric": metric, "fields": {}}
        for field in FIELD_NAMES:
            errors = [
                r["region_errors"][region_name]["errors"][field]
                for r in results
                if region_name in r["region_errors"]
            ]
            summary[region_name]["fields"][field] = float(np.mean(errors))
    return summary
