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


def radius_graph_edges(position, r, max_neighbors):
    """Shared by build_radius_subsampled_graph (below, VTU-based) and
    CachedRadiusSubsampledDataset (src/dataset.py, cache-based) -- both need
    the identical subsample-then-reconnect step, just starting from
    different sources for `position`. See build_radius_subsampled_graph's
    docstring for why cKDTree instead of torch_geometric.nn.radius_graph,
    and for what this approximates vs. replicates exactly.

    position: (N, 2) already-subsampled node positions.

    Returns:
        edge_index: (2, E) int array, edge_attr: (E, 2) float array
    """
    tree = cKDTree(position)
    dist, neighbors = tree.query(position, k=max_neighbors + 1, distance_upper_bound=r)
    # Column 0 is each point itself (distance 0) -- excluded here and
    # re-added below as an explicit self-loop, rather than counted against
    # max_neighbors.
    valid = np.isfinite(dist[:, 1:])
    src = np.repeat(np.arange(position.shape[0]), max_neighbors)[valid.reshape(-1)]
    dst = neighbors[:, 1:].reshape(-1)[valid.reshape(-1)]

    self_loops = np.arange(position.shape[0])
    src = np.concatenate([src, self_loops])
    dst = np.concatenate([dst, self_loops])

    edge_index = np.stack([src, dst], axis=0)
    edge_attr = position[dst] - position[src]
    return edge_index, edge_attr


def build_radius_subsampled_graph(simulation, n_nodes=32000, r=0.05, max_neighbors=64, seed=None):
    """Random node subsample + radius-graph edges -- NOT a cheap dev-only
    approximation like build_subsampled_graph below; this is the actual
    training regime the AirfRANS paper's own best-performing baseline
    (GraphSAGE, Cd relative error ~4%, vs. our full-fixed-mesh training)
    uses: resample `n_nodes` of the mesh and rebuild neighbor edges via a
    real spatial radius search, freshly every call (so every epoch, if
    called from a Dataset's get()) -- not the mesh's own triangulation
    edges. Matters specifically because the real mesh's near-wall edges are
    mostly tangential to the surface (median cosine to the normal ~0.0002,
    measured directly -- ARCHITECTURE.md section 11), so a spatial-radius
    reconstruction gives a materially different, more isotropic near-wall
    neighborhood than build_graph()'s real connectivity does.

    Uses scipy's cKDTree, not torch_geometric.nn.radius_graph -- the latter
    needs torch_cluster (a compiled extension), which both isn't installed
    here and hung rather than failing on import when tested locally. This
    project already avoids torch-scatter/torch-sparse for the same
    "avoid compiled-extension install friction" reason (src/model.py uses
    torch_geometric.utils.scatter instead).

    Approximates, doesn't bit-exactly replicate, radius_graph(r, loop=True,
    max_num_neighbors): finds up to `max_neighbors` nearest OTHER points
    within radius `r` per node (not counting the self-loop against that
    cap), then adds one explicit self-loop per node separately.

    seed: None (default) draws a fresh random subsample every call, which
    is the point when called from inside a Dataset's get() (resampled every
    epoch, matching the paper's approach) -- pass a fixed int only for a
    reproducible one-off check.

    Returns:
        node_features: (n_nodes, 7), edge_index: (2, E) int array,
        edge_attr: (E, 2), targets: (n_nodes, 4) [vx, vy, pressure, nu_t],
        idx: (n_nodes,) indices into the original full mesh
    """
    rng = np.random.default_rng(seed)
    n_total = simulation.position.shape[0]
    idx = rng.choice(n_total, size=min(n_nodes, n_total), replace=False)

    position = simulation.position[idx]
    node_features = np.concatenate(
        [position, simulation.sdf[idx], simulation.input_velocity[idx], simulation.normals[idx]],
        axis=1,
    )

    edge_index, edge_attr = radius_graph_edges(position, r, max_neighbors)

    targets = np.concatenate(
        [simulation.velocity[idx], simulation.pressure[idx], simulation.nu_t[idx]], axis=1
    )

    return node_features, edge_index, edge_attr, targets, idx


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
