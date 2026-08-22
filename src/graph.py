"""Turn one AirfRANS Simulation into a graph: node features + edge_index + edge features."""
import numpy as np
from scipy.spatial import cKDTree


def build_graph(simulation):
    """
    Node features: [x, y, wall_distance, inlet_vx, inlet_vy, normal_x, normal_y]  (N, 7)
    Edges: mesh connectivity from the internal mesh, made bidirectional.
    Edge features: relative position (dst - src)  (2*E, 2)

    simulation.normals is a unit vector at surface nodes, [0, 0] everywhere
    else -- wall_distance alone tells the model *how far* from the wall a
    node is, but not *which direction* into it, so it can't resolve the
    boundary layer's anisotropy (very different physics along the wall vs.
    across it) from that scalar alone.

    Returns:
        node_features: (N, 7) float array
        edge_index: (2, 2*E) int array, row 0 = source, row 1 = destination
        edge_attr: (2*E, 2) float array
    """
    node_features = np.concatenate(
        [simulation.position, simulation.sdf, simulation.input_velocity, simulation.normals],
        axis=1,
    )

    edges_poly = simulation.internal.extract_all_edges()
    pairs = edges_poly.lines.reshape(-1, 3)[:, 1:]  # (E, 2), format [n_pts, i, j] per cell

    src = np.concatenate([pairs[:, 0], pairs[:, 1]])
    dst = np.concatenate([pairs[:, 1], pairs[:, 0]])
    edge_index = np.stack([src, dst], axis=0)

    edge_attr = simulation.position[dst] - simulation.position[src]

    return node_features, edge_index, edge_attr


def build_subsampled_graph(simulation, n_nodes=2000, k=6, seed=0):
    """Cheap local-only sanity-check graph: random node subsample + k-NN edges.

    NOT the real mesh connectivity -- that's build_graph(), meant for full-resolution
    runs on Colab. A full case is ~180k nodes / ~720k edges, and even 5 of those
    batched together is enough to hang a 4GB-GPU laptop. This trades mesh-accurate
    connectivity for a graph small enough to run a training step locally without
    that happening.

    Returns:
        node_features: (n_nodes, 7), edge_index: (2, n_nodes*k), edge_attr: (n_nodes*k, 2),
        targets: (n_nodes, 4) [vx, vy, pressure, nu_t]
    """
    rng = np.random.default_rng(seed)
    n_total = simulation.position.shape[0]
    idx = rng.choice(n_total, size=min(n_nodes, n_total), replace=False)

    position = simulation.position[idx]
    node_features = np.concatenate(
        [position, simulation.sdf[idx], simulation.input_velocity[idx], simulation.normals[idx]],
        axis=1,
    )

    tree = cKDTree(position)
    _, neighbors = tree.query(position, k=k + 1)  # column 0 is each point itself
    src = np.repeat(np.arange(position.shape[0]), k)
    dst = neighbors[:, 1:].reshape(-1)
    edge_index = np.stack([src, dst], axis=0)
    edge_attr = position[dst] - position[src]

    targets = np.concatenate(
        [simulation.velocity[idx], simulation.pressure[idx], simulation.nu_t[idx]], axis=1
    )

    return node_features, edge_index, edge_attr, targets, idx
