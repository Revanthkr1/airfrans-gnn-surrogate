"""Week 5: evaluate a trained checkpoint on the official 200-case test split.

Reports relative L2 per field (mean across cases -- never blended across
fields, per CLAUDE.md) and lift/drag coefficient accuracy against AirfRANS's
own ground-truth force_coefficient(). Inference-only, one case at a time --
safe to run locally (no backward pass, no batching multiple full-res graphs
together, which is what actually hung this machine during training).
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.evaluate import evaluate_split
from src.metrics import FIELD_NAMES

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
CHECKPOINT_PATH = os.path.join(DATA_ROOT, "meshgraphnet.ckpt")  # replace with your actual checkpoint


def main():
    dataset_root = os.path.join(DATA_ROOT, "Dataset")
    results = evaluate_split(
        checkpoint_path=CHECKPOINT_PATH,
        dataset_root=dataset_root,
        stats_path=os.path.join(DATA_ROOT, "norm_stats.npz"),
    )

    print(f"\n=== Relative L2 per field, mean over {len(results)} test cases ===")
    for field in FIELD_NAMES:
        errors = [r["field_errors"][field] for r in results]
        print(f"{field}: mean={np.mean(errors):.4f}  median={np.median(errors):.4f}  worst={np.max(errors):.4f}")

    cd_pred = np.array([r["cd_pred"] for r in results])
    cd_ref = np.array([r["cd_ref"] for r in results])
    cl_pred = np.array([r["cl_pred"] for r in results])
    cl_ref = np.array([r["cl_ref"] for r in results])

    print("\n=== Lift/drag coefficient accuracy ===")
    print(f"Cd relative L2: {np.linalg.norm(cd_pred - cd_ref) / np.linalg.norm(cd_ref):.4f}")
    print(f"Cl relative L2: {np.linalg.norm(cl_pred - cl_ref) / np.linalg.norm(cl_ref):.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, pred, ref, label in [(axes[0], cd_pred, cd_ref, "Cd"), (axes[1], cl_pred, cl_ref, "Cl")]:
        ax.scatter(ref, pred, s=10, alpha=0.6)
        lims = [min(ref.min(), pred.min()), max(ref.max(), pred.max())]
        ax.plot(lims, lims, "k--", linewidth=1, label="perfect")
        ax.set_xlabel(f"{label} (ground truth)")
        ax.set_ylabel(f"{label} (predicted)")
        ax.set_title(label)
        ax.legend()
    fig.tight_layout()
    ckpt_tag = os.path.splitext(os.path.basename(CHECKPOINT_PATH))[0]
    out_path = os.path.join(os.path.dirname(__file__), f"week5_lift_drag_{ckpt_tag}.png")
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
