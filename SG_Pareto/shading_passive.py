from __future__ import annotations

from typing import Dict

import numpy as np


def compute_u_facades(
    alpha_rad: np.ndarray,
    psi_rad: np.ndarray,
    dN_m: float,
    dS_m: float,
    etaE_rad: float,
    etaW_rad: float,
    gamma_dict_rad: Dict[str, float],
    window_height_m: float,
    eps: float = 1e-6,
) -> Dict[str, np.ndarray]:
    """
    Passive shading transmissivity per facade.

    North/South: overhang
      tan(phi) = tan(alpha) / (|cos(psi-gamma)| + eps)
      h = d * tan(phi)
      S = clip(h/H_win, 0..1)
      u = 1 - S

    East/West: louvers
      tan(eta) = sin(psi-gamma) / (tan(alpha) + eps)
      u = max(0, 1 - |eta| / H_f)
    """
    alpha = np.asarray(alpha_rad, dtype=float)
    psi = np.asarray(psi_rad, dtype=float)
    H_win = max(float(window_height_m), eps)

    # North/South overhang
    def _u_overhang(d_m: float, gamma_rad: float) -> np.ndarray:
        denom = np.abs(np.cos(psi - gamma_rad)) + eps
        tan_phi = np.tan(alpha) / denom
        h = float(d_m) * tan_phi
        S = np.clip(h / H_win, 0.0, 1.0)
        return 1.0 - S

    # East/West louvers
    def _u_louver(H_f: float, gamma_rad: float) -> np.ndarray:
        H = max(float(H_f), eps)
        tan_eta = np.sin(psi - gamma_rad) / (np.tan(alpha) + eps)
        eta = np.arctan(tan_eta)
        return np.maximum(0.0, 1.0 - np.abs(eta) / H)

    uN = _u_overhang(dN_m, gamma_dict_rad["N"])
    uS = _u_overhang(dS_m, gamma_dict_rad["S"])
    uE = _u_louver(etaE_rad, gamma_dict_rad["E"])
    uW = _u_louver(etaW_rad, gamma_dict_rad["W"])

    return {"N": uN, "E": uE, "S": uS, "W": uW}
