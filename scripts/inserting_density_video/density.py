"""Training-set log KDE used for the A_10 presentation's floor and counters."""

from hashlib import sha256

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import logsumexp

# Empirical log-density quantiles over all 711,908 desired training XY positions.
CUTOFFS = [0.0263594751088443, 2.058791232195018, 2.482032065730343]
SUPPORT_SHA = "6dbbfcac70f86aa5c75a776c478be5d58e3a6bcd6b121a25008a13e76bef6f50"
KERNEL = "7b9ba10cb5404191d206047d278f9e7edb368f50ed4523d89ba897caf04ab0b8.npz"
PALETTE = [(231, 234, 231), (182, 215, 209), (116, 182, 171), (49, 138, 128)]
GOALS = np.array([[0.3575, 0.276], [0.525, 0.4535], [0.6925, 0.276]])
CONTEXT = "env_050_05.npz"


def load_support(source):
    with np.load(source / "data/inserting_sincos_v1/pf2mp_cache" / KERNEL) as cache:
        support = cache["support"]
    if sha256(support.tobytes()).hexdigest() != SUPPORT_SHA:
        raise ValueError("KDE support differs from the training calibration")
    return support


def log_density(points, support):
    variance = 0.025**2
    normalizer = np.log(len(support) * 2 * np.pi * variance)
    chunks = []
    for start in range(0, len(points), 160):
        squared = cdist(points[start : start + 160], support, "sqeuclidean")
        chunks.append(logsumexp(-squared / (2 * variance), axis=1) - normalizer)
    return np.concatenate(chunks)
