"""One-time conversion: VTU meshes -> cached graph tensors (real mesh connectivity).

Training re-reads the same cases every epoch; without caching, that means
re-parsing VTU via PyVista every single epoch, which dominated runtime even
locally (~1-11s/case just to load). This is a Colab-scale job for a full split
(hundreds of cases), but preprocessing a case at a time is cheap and safe to
run locally too -- it was *training* on many full-res graphs at once that hung
the local machine (see src/graph.py), not loading them one at a time.
"""
import os
import shutil
import zipfile

import airfrans as af
import numpy as np
import torch

from src.data import load_case
from src.graph import build_graph


def cache_path(cache_dir, name):
    return os.path.join(cache_dir, f"{name}.pt")


def preprocess_case(dataset_root, name, cache_dir, delete_raw_after=False):
    os.makedirs(cache_dir, exist_ok=True)
    out_path = cache_path(cache_dir, name)
    if not os.path.exists(out_path):
        simulation, _ = load_case(dataset_root, name)
        # edge_attr is dropped here -- it's fully determined by position + edge_index
        # (dst - src), so CachedPyGAirfRANSDataset recomputes it instead of storing it.
        # edge_index as int32 (not PyG's usual int64): node counts are ~180k, well
        # under int32 range, and it's the single biggest piece of the cache -- these
        # two changes take a case from ~24MB to ~12MB, which mattered once a Colab
        # disk filled up caching the full 800-case split. Sparse normal storage and
        # float16 (below) take it down further, to ~9MB -- see their own comments.
        node_features, edge_index, _ = build_graph(simulation)
        surface = simulation.surface
        # normal (columns 5:7 of node_features, see src/graph.py) is exactly
        # zero everywhere except the ~0.5% of nodes on the surface -- caching
        # it densely would nearly double node_features' on-disk size for
        # almost all zeros. Store only the surface rows; CachedPyGAirfRANSDataset
        # reconstructs the full (N, 2) array by scattering back using the
        # already-cached `surface` mask.
        normal_at_surface = node_features[surface, 5:7]
        node_features = node_features[:, :5]
        targets = np.concatenate(
            [simulation.velocity, simulation.pressure, simulation.nu_t], axis=1
        )
        # Two cases in the full-800 split have a raw pressure magnitude (~81k,
        # ~97k, at what's almost certainly the stagnation point) that overflows
        # float16's max (~65504) -- silently becoming `inf` at the cast below,
        # baked into the cache forever, and eventually producing a genuine NaN
        # loss/gradient whenever training happens to draw that case (confirmed
        # via torch.autograd.set_detect_anomaly -- see ARCHITECTURE.md section
        # 11). Clamped with margin so no future preprocessing run reintroduces
        # this; CachedPyGAirfRANSDataset/CachedRadiusSubsampledDataset.get()
        # also nan_to_num on load to fix caches already built before this fix.
        targets = np.clip(targets, -6.0e4, 6.0e4)
        # float16 on disk, not float32: halves node_features/targets/normal's
        # footprint. Cast back to float32 immediately on load
        # (CachedPyGAirfRANSDataset.get()) so training math is unaffected --
        # this only shrinks the on-disk representation, and the project
        # already trains under precision="16-mixed" on Kaggle/Colab anyway,
        # so it isn't introducing a new precision regime. Needed for real
        # headroom against the raw zip (9.34GB, undeletable until every case
        # has been extracted from it -- see stream_preprocess_from_zip)
        # sitting on disk *simultaneously* with the growing cache for the
        # entire preprocessing run: this combination hit Kaggle's disk quota
        # at 750/800 cases even after normals were already stored sparsely.
        payload = {
            "node_features": torch.tensor(node_features, dtype=torch.float16),
            "normal_at_surface": torch.tensor(normal_at_surface, dtype=torch.float16),
            "edge_index": torch.tensor(edge_index, dtype=torch.int32),
            "targets": torch.tensor(targets, dtype=torch.float16),
            "surface": torch.tensor(surface),
            "name": name,
        }
        # Write to a temp path then rename, not directly to out_path: an
        # interrupted torch.save (disk full, session timeout/kill) would
        # otherwise leave a truncated .pt file that still satisfies
        # os.path.exists() above, silently skipping this case forever on
        # every future call instead of ever re-caching it. rename() is
        # atomic on the same filesystem, so out_path only ever exists once
        # the write has actually finished.
        tmp_path = out_path + ".tmp"
        torch.save(payload, tmp_path)
        os.replace(tmp_path, out_path)

    if delete_raw_after:
        shutil.rmtree(os.path.join(dataset_root, name), ignore_errors=True)

    return out_path


def preprocess_split(dataset_root, names, cache_dir, log_every=50, delete_raw_after=False):
    for i, name in enumerate(names):
        preprocess_case(dataset_root, name, cache_dir, delete_raw_after=delete_raw_after)
        if (i + 1) % log_every == 0:
            print(f"cached {i + 1}/{len(names)}", flush=True)


def download_zip_only(root, file_name="Dataset", OpenFOAM=False):
    """Just the zip, no extraction. af.dataset.download's own extractall() needs
    the zip (~9.34GB) AND the full extracted dataset (~15GB) on disk at once --
    that alone exceeded a Kaggle session's disk quota. Extracting/caching one
    case at a time (see stream_preprocess_from_zip) avoids ever needing both."""
    os.makedirs(root, exist_ok=True)
    zip_path = os.path.join(root, f"{file_name}.zip")
    if not os.path.exists(zip_path):
        af.dataset.download(root=root, file_name=file_name, unzip=False, OpenFOAM=OpenFOAM)
    return zip_path


def stream_preprocess_from_zip(zip_path, names, cache_dir, work_dir, log_every=50):
    """Extract + cache one case at a time straight from the zip, deleting each
    case's raw files immediately after caching. Peak disk usage stays at
    ~(zip + one case + growing cache) instead of (zip + the full raw dataset +
    growing cache) all at once.

    Doesn't need manifest.json out of the zip -- `names` comes from
    split_names() reading the git-tracked data/manifest.json instead.
    """
    with zipfile.ZipFile(zip_path) as zf:
        all_members = zf.namelist()
        for i, name in enumerate(names):
            prefix = f"Dataset/{name}/"
            members = [m for m in all_members if m.startswith(prefix)]
            zf.extractall(work_dir, members=members)
            preprocess_case(
                os.path.join(work_dir, "Dataset"), name, cache_dir, delete_raw_after=True
            )
            if (i + 1) % log_every == 0:
                print(f"cached {i + 1}/{len(names)}", flush=True)
