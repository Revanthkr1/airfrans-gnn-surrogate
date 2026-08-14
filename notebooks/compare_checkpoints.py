"""Formalizes the manual epoch-54-vs-99 comparison into a reusable script.

Training-time field-level val_loss doesn't reveal a Cd/Cl regression (that's
exactly what happened between epoch 54 and 99 -- fields kept improving while
drag got dramatically worse). Until training itself monitors something
drag-aware, the reliable way to pick a checkpoint is: evaluate several of them
against real Cd/Cl on a shared subset and compare directly, rather than
trusting the final epoch by default.

Uses a small fixed subset of the test split (not all 200) so comparing many
checkpoints stays fast -- full-split evaluation (src/evaluate.evaluate_split
with names=None) is still the right call for the one checkpoint you settle on.
"""
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data import split_names
from src.evaluate import evaluate_split, summarize

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
N_COMPARE_CASES = 30  # subset size -- keeps each checkpoint's eval to a few minutes

# Fill in the checkpoint files to compare -- e.g. a glob over everything you
# downloaded: CHECKPOINT_PATHS = sorted(glob.glob("/path/to/mgn-epoch=*.ckpt"))
CHECKPOINT_PATHS = [
    "/path/to/mgn-epoch=A.ckpt",
    "/path/to/mgn-epoch=B.ckpt",
]


def main():
    dataset_root = os.path.join(DATA_ROOT, "Dataset")
    stats_path = os.path.join(DATA_ROOT, "norm_stats.npz")
    test_names = split_names(dataset_root, task="full", train=False)[:N_COMPARE_CASES]

    rows = []
    for ckpt_path in CHECKPOINT_PATHS:
        print(f"\n=== {os.path.basename(ckpt_path)} ===", flush=True)
        results = evaluate_split(
            checkpoint_path=ckpt_path,
            dataset_root=dataset_root,
            stats_path=stats_path,
            names=test_names,
        )
        summary = summarize(results)
        summary["checkpoint"] = os.path.basename(ckpt_path)
        rows.append(summary)
        print(
            f"  fields(mean): vx={summary['vx_mean']:.3f} vy={summary['vy_mean']:.3f} "
            f"pressure={summary['pressure_mean']:.3f} nu_t={summary['nu_t_mean']:.3f}"
        )
        print(f"  Cd rel L2: {summary['cd_rel_l2']:.3f}   Cl rel L2: {summary['cl_rel_l2']:.3f}")

    print(f"\n=== Summary over {N_COMPARE_CASES} test cases ===")
    header = f"{'checkpoint':<24} {'vx':>7} {'vy':>7} {'press':>7} {'nu_t':>7} {'Cd':>8} {'Cl':>7}"
    print(header)
    for r in rows:
        print(
            f"{r['checkpoint']:<24} {r['vx_mean']:>7.3f} {r['vy_mean']:>7.3f} "
            f"{r['pressure_mean']:>7.3f} {r['nu_t_mean']:>7.3f} {r['cd_rel_l2']:>8.3f} {r['cl_rel_l2']:>7.3f}"
        )

    best = min(rows, key=lambda r: r["cd_rel_l2"])
    print(f"\nBest by Cd: {best['checkpoint']} (Cd rel L2 = {best['cd_rel_l2']:.3f})")


if __name__ == "__main__":
    main()
