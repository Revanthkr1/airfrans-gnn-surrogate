"""Loading AirfRANS simulations off disk, including the official manifest.json splits."""
import json
import os

import airfrans as af


def list_simulation_names(dataset_root):
    return sorted(
        d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))
    )


def load_case(dataset_root, name=None):
    if name is None:
        name = list_simulation_names(dataset_root)[0]
    return af.Simulation(root=dataset_root, name=name), name


def load_manifest(dataset_root):
    with open(os.path.join(dataset_root, "manifest.json")) as f:
        return json.load(f)


def split_names(dataset_root, task="full", train=True):
    """Official simulation names for a task/split, straight from manifest.json.

    Tasks: 'full' (800/200), 'scarce' (200/200, shares full's test set),
    'reynolds' (504/496), 'aoa' (804/196).
    """
    manifest = load_manifest(dataset_root)
    key = f"{task}_{'train' if train else 'test'}"
    return manifest[key]
