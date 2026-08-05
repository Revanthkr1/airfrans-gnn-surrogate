"""Day 1: does the AirfRANS pipeline even work? Load one case, plot pressure. Nothing else."""
import os
import sys

import airfrans as af
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data import load_case

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")


def main():
    dataset_root = os.path.join(DATA_ROOT, "Dataset")
    if not os.path.isdir(dataset_root) or not os.listdir(dataset_root):
        af.dataset.download(root=DATA_ROOT, file_name="Dataset", unzip=True, OpenFOAM=False)

    simulation, name = load_case(dataset_root)
    print(f"Loading simulation: {name}")

    print("nodes:", simulation.position.shape[0])
    print("inlet velocity:", simulation.inlet_velocity)
    print("angle of attack (rad):", simulation.angle_of_attack)

    x, y = simulation.position[:, 0], simulation.position[:, 1]
    fig, ax = plt.subplots(figsize=(10, 4))
    sc = ax.scatter(x, y, c=simulation.pressure[:, 0], s=2, cmap="RdBu_r")
    ax.set_aspect("equal")
    ax.set_title(f"Pressure field: {name}")
    fig.colorbar(sc, ax=ax, label="pressure (/ rho)")
    fig.savefig(os.path.join(os.path.dirname(__file__), "day1_pressure_field.png"), dpi=150)
    print("Saved notebooks/day1_pressure_field.png")


if __name__ == "__main__":
    main()
