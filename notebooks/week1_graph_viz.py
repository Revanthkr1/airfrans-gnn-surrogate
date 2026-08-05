"""Week 1: turn one case into a graph, print counts, visualize edges over the airfoil."""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data import load_case
from src.graph import build_graph

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
# Zoom window around the airfoil -- plotting all ~360k mesh edges over the
# full domain is unreadable and slow; this is the region that actually
# confirms the connectivity looks right.
ZOOM_X = (-0.5, 1.5)
ZOOM_Y = (-0.6, 0.6)


def main():
    dataset_root = os.path.join(DATA_ROOT, "Dataset")
    simulation, name = load_case(dataset_root)

    node_features, edge_index, edge_attr = build_graph(simulation)
    n_nodes = node_features.shape[0]
    n_directed_edges = edge_index.shape[1]
    print(f"Simulation: {name}")
    print(f"Nodes: {n_nodes}")
    print(f"Edges: {n_directed_edges} directed ({n_directed_edges // 2} undirected)")
    print(f"Node feature shape: {node_features.shape} [x, y, wall_dist, inlet_vx, inlet_vy]")
    print(f"Edge feature shape: {edge_attr.shape} [dx, dy]")

    pos = node_features[:, :2]
    src, dst = edge_index[0], edge_index[1]
    in_zoom = (
        (pos[src, 0] > ZOOM_X[0]) & (pos[src, 0] < ZOOM_X[1])
        & (pos[src, 1] > ZOOM_Y[0]) & (pos[src, 1] < ZOOM_Y[1])
        & (pos[dst, 0] > ZOOM_X[0]) & (pos[dst, 0] < ZOOM_X[1])
        & (pos[dst, 1] > ZOOM_Y[0]) & (pos[dst, 1] < ZOOM_Y[1])
    )
    segments = np.stack([pos[src[in_zoom]], pos[dst[in_zoom]]], axis=1)
    print(f"Edges drawn in zoom window: {segments.shape[0]}")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.add_collection(LineCollection(segments, linewidths=0.3, colors="steelblue", alpha=0.6))
    surface = simulation.surface
    ax.scatter(pos[surface, 0], pos[surface, 1], s=3, c="black", zorder=3, label="airfoil surface")
    ax.set_xlim(*ZOOM_X)
    ax.set_ylim(*ZOOM_Y)
    ax.set_aspect("equal")
    ax.set_title(f"Mesh graph near airfoil: {name}")
    ax.legend(loc="upper right")
    fig.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "week1_graph.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
