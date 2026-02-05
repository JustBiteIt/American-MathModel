
# HVAC_Pareto/pareto.py
from __future__ import annotations
from typing import List, Tuple, Dict, Any
import numpy as np

def pareto_mask_minimize(points: np.ndarray) -> np.ndarray:
    """
    Return boolean mask selecting Pareto-optimal points for minimization.
    points: (n, m) with all objectives to MINIMIZE
    """
    P = np.asarray(points, dtype=float)
    n = P.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        # any point that dominates i will set keep[i]=False
        # j dominates i if all <= and at least one <
        dominates = np.all(P <= P[i], axis=1) & np.any(P < P[i], axis=1)
        if np.any(dominates):
            keep[i] = False
            continue
        # i dominates others -> remove them
        dominated_by_i = np.all(P[i] <= P, axis=1) & np.any(P[i] < P, axis=1)
        keep[dominated_by_i] = False
        keep[i] = True
    return keep

def select_representative_by_weights(
    E_cool: np.ndarray,
    E_heat: np.ndarray | None = None,
    mask_pareto: np.ndarray | None = None,
    *,
    w_cool: float,
    w_heat: float,
) -> int:
    """
    Pick argmin of weighted sum within Pareto set.

    Supports two call styles:
      1) points array (n,2) for a Pareto-only set:
         select_representative_by_weights(points, w_cool=..., w_heat=...)
         -> returns index within that array.
      2) full arrays + mask:
         select_representative_by_weights(E_cool, E_heat, mask_pareto, w_cool=..., w_heat=...)
         -> returns index within the full array.
    """
    w_cool = float(w_cool)
    w_heat = float(w_heat)

    if E_heat is None and mask_pareto is None:
        pts = np.asarray(E_cool, dtype=float)
        if pts.ndim != 2 or pts.shape[1] < 2:
            raise ValueError("When E_heat/mask_pareto not provided, E_cool must be (n,2) array")
        J = w_cool * pts[:, 0] + w_heat * pts[:, 1]
        return int(np.argmin(J))

    if E_heat is None or mask_pareto is None:
        raise ValueError("E_heat and mask_pareto must be provided together")

    idx = np.where(np.asarray(mask_pareto, dtype=bool))[0]
    J = w_cool * np.asarray(E_cool)[idx] + w_heat * np.asarray(E_heat)[idx]
    return int(idx[int(np.argmin(J))])
