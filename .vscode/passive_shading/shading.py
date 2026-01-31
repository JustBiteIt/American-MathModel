# shading.py
import numpy as np

def u_overhang(alpha: np.ndarray, psi: np.ndarray, gamma_deg: float,
               d_m: float, window_height_m: float, eps: float = 1e-6) -> np.ndarray:
    """
    South/North overhang transmittance u(t) based on vertical profile angle.
      tan(phi) = tan(alpha) / (|cos(psi-gamma)| + eps)
      shading_ratio = clip(d * tan(phi) / H, 0, 1)
      u = 1 - shading_ratio
    """
    gamma = np.deg2rad(gamma_deg)
    denom = np.abs(np.cos(psi - gamma)) + eps
    tan_phi = np.tan(alpha) / denom
    h_shade = d_m * tan_phi
    S = np.clip(h_shade / max(window_height_m, eps), 0.0, 1.0)
    return 1.0 - S

def u_louver(alpha: np.ndarray, psi: np.ndarray, gamma_deg: float,
             eta_cutoff_rad: float, eps: float = 1e-6) -> np.ndarray:
    """
    East/West louver transmittance u(t) via horizontal profile angle.
      tan(eta) = sin(psi-gamma) / (tan(alpha) + eps)
      u = max(0, 1 - |eta| / Eta)
    """
    gamma = np.deg2rad(gamma_deg)
    tan_eta = np.sin(psi - gamma) / (np.tan(alpha) + eps)
    eta = np.arctan(tan_eta)
    return np.maximum(0.0, 1.0 - np.abs(eta) / max(eta_cutoff_rad, eps))
