"""Read the frozen Pushing/Inserting evaluation cases from the OOD benchmark."""
import json
import pickle

import numpy as np

from configs.config import REPO


def load_condition(task, name):
    directory = REPO / 'fm_baseline/ood_benchmark' / task
    manifest = json.loads((directory / 'manifest.json').read_text())
    if name not in manifest['conditions']:
        raise ValueError(f'Unknown {task} condition {name}. Choose from {list(manifest["conditions"])}')
    path = directory / manifest['conditions'][name]['file']
    with path.open('rb') as handle:
        dataset = pickle.load(handle)
    if (dataset['cond'] != name or dataset['task'] != task or
            len(dataset['episodes']) != manifest['n_episodes'] or
            dataset['factors'] != manifest['conditions'][name]['factors']):
        raise ValueError(f'Condition file disagrees with manifest: {path}')
    return path, dataset


def observe_boxes(boxes, factor, rng):
    if not factor:
        return boxes
    noisy = []
    for pos, quat, yaw in boxes:
        p = np.array(pos, dtype=float, copy=True)
        p[:2] += rng.normal(0., factor['pos_m'], 2)
        noisy.append((p, quat, yaw + np.deg2rad(rng.normal(0., factor['yaw_deg']))))
    return noisy
