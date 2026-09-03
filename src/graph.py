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


def airfoil_polyline(surface_position):
    """Order surface points into a closed polyline by angle around their
    centroid. Not a general planar-curve sorter -- relies on the airfoil
    boundary being star-shaped from its own centroid, true for a single
    non-self-intersecting airfoil profile. Used to test whether a
    radius-graph edge cuts through the solid body (see radius_graph_edges).
    Pass the FULL-resolution surface points, not a subsample -- the sharp
    trailing-edge cusp needs fine resolution to represent correctly."""
    centroid = surface_position.mean(axis=0)
    angle = np.arctan2(
        surface_position[:, 1] - centroid[1], surface_position[:, 0] - centroid[0]
    )
    poly = surface_position[np.argsort(angle)]
    return np.vstack([poly, poly[:1]])  # close it


def _segments_cross_polyline(p, q, poly, k=16):
    """Vectorized segment-vs-polyline intersection. p, q: (E, 2) segment
    endpoints. poly: (M+1, 2) closed polyline. Returns (E,) bool -- does
    segment p[i]->q[i] cross any polyline edge?

    Only tests each query segment against its k nearest polyline segments
    (by midpoint distance), not all M -- a real surface polyline has
    hundreds to low thousands of segments, and testing every candidate
    edge against every one of them is O(E*M): at real scale (E up to
    ~2e5 near-wall candidates, M ~1000) that's ~1.3GB+ per intermediate
    array, and far too slow to run every epoch. k=16 was checked against
    k=64 on real cached cases (same edges flagged, ~5x faster) -- plenty
    of slack over how many polyline segments can plausibly fall within a
    window of a few edge-lengths (r) along the boundary.
    """
    a, b = poly[:-1], poly[1:]  # (M, 2) each
    mid = (a + b) / 2
    tree = cKDTree(mid)
    k_eff = min(k, mid.shape[0])
    _, seg_idx = tree.query((p + q) / 2, k=k_eff)  # (E, k_eff)
    if k_eff == 1:
        seg_idx = seg_idx[:, None]

    a_k, b_k = a[seg_idx], b[seg_idx]  # (E, k_eff, 2)
    d1 = b_k - a_k
    d2 = (q - p)[:, None, :]  # broadcasts to (E, k_eff, 2)
    denom = d1[..., 0] * d2[..., 1] - d1[..., 1] * d2[..., 0]  # (E, k_eff)
    ok = np.abs(denom) > 1e-12
    denom_safe = np.where(ok, denom, 1.0)
    dp = p[:, None, :] - a_k
    t = (dp[..., 0] * d2[..., 1] - dp[..., 1] * d2[..., 0]) / denom_safe
    u = (dp[..., 0] * d1[..., 1] - dp[..., 1] * d1[..., 0]) / denom_safe
    hit = ok & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    return hit.any(axis=1)


def radius_graph_edges(position, r, max_neighbors, wall_distance=None, surface_polyline=None):
    """Shared by build_radius_subsampled_graph (below, VTU-based) and
    CachedRadiusSubsampledDataset (src/dataset.py, cache-based) -- both need
    the identical subsample-then-reconnect step, just starting from
    different sources for `position`. See build_radius_subsampled_graph's
    docstring for why cKDTree instead of torch_geometric.nn.radius_graph,
    and for what this approximates vs. replicates exactly.

    position: (N, 2) already-subsampled node positions.

    wall_distance, surface_polyline: if both given, drops candidate edges
    whose straight-line segment crosses the airfoil surface -- a pure
    Euclidean radius search has no notion of the airfoil as a solid
    obstacle, so near thin geometry (the trailing edge especially) it can
    connect two points on opposite sides of the body that are close in
    (x,y) but physically unconnected without going around it (confirmed on
    real cached cases: edges as short as ~1e-6 in position with a
    non-negligible target gap, ARCHITECTURE.md section 11). The crossing
    test is restricted to candidates with either endpoint within r of the
    wall (checked against a looser 2r on real cases -- identical edges
    flagged, so r alone already catches every true crossing: no edge is
    longer than r, so a crossing edge's endpoints can't be farther than r
    from the body either).

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

    if wall_distance is not None and surface_polyline is not None and src.size:
        near_wall = (wall_distance[src] < r) | (wall_distance[dst] < r)
        candidates = np.where(near_wall)[0]
        if candidates.size:
            crosses = _segments_cross_polyline(
                position[src[candidates]], position[dst[candidates]], surface_polyline
            )
            drop = np.zeros(src.shape[0], dtype=bool)
            drop[candidates[crosses]] = True
            src, dst = src[~drop], dst[~drop]

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

    # Full-resolution surface points (not the subsample) -- see
    # airfoil_polyline's docstring for why the trailing edge needs full
    # resolution to represent correctly.
    poly = airfoil_polyline(simulation.position[simulation.surface])
    edge_index, edge_attr = radius_graph_edges(
        position, r, max_neighbors, wall_distance=simulation.sdf[idx, 0], surface_polyline=poly
    )
    # See CachedRadiusSubsampledDataset.get() (src/dataset.py) -- raw edge_attr
    # here can be up to ~500x larger than a real mesh edge, which produced a
    # NaN in the decoder's backward pass on the very first training step.
    # Dividing by r keeps this function's output consistent with what the
    # cache-based path actually feeds the model.
    edge_attr = edge_attr / r

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
