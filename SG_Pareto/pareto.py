from __future__ import annotations

import numpy as np


def pareto_mask_minimize(points: np.ndarray) -> np.ndarray:
    """Return boolean mask selecting Pareto-optimal points for minimization."""
    P = np.asarray(points, dtype=float)
    n = P.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        dominates = np.all(P <= P[i], axis=1) & np.any(P < P[i], axis=1)
        if np.any(dominates):
            keep[i] = False
            continue
        dominated_by_i = np.all(P[i] <= P, axis=1) & np.any(P[i] < P, axis=1)
        keep[dominated_by_i] = False
        keep[i] = True
    return keep
