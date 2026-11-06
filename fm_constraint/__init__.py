"""Configuration and execution filters for the unchanged FM baseline."""

import importlib
from contextlib import nullcontext
from pathlib import Path
import sys


def import_baseline(name):
    """Reject legacy absolute-import collisions instead of replacing modules."""
    baseline = Path(__file__).resolve().parents[1] / 'fm_baseline'
    for legacy in ('train', 'eval_pushing'):
        module = sys.modules.get(legacy)
        if module is not None and Path(getattr(module, '__file__', '')).resolve() != baseline / f'{legacy}.py':
            raise ImportError(f'Legacy import {legacy!r} is occupied; use a fresh python -m process')
    return importlib.import_module(f'fm_baseline.{name}')


def checkpoint_reads():
    """Allow the baseline's version metadata with modern weights-only loading.

    PyTorch 2.3 wrote TorchVersion into config metadata. Do not disable
    weights-only loading or change the teammate's torch.load calls globally.
    """
    import torch
    if hasattr(torch.serialization, 'safe_globals'):
        return torch.serialization.safe_globals([torch.torch_version.TorchVersion])
    return nullcontext()
