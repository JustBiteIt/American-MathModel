# solar.py
import numpy as np
import pandas as pd

def solar_position(times: pd.DatetimeIndex, latitude: float, longitude: float):
    """
    Uses pvlib if available; otherwise raises ImportError with clear message.
    Returns alpha (elevation, rad) and psi (azimuth, rad; from North, East-positive).
    """
    try:
        import pvlib
    except ImportError as e:
        raise ImportError("pvlib is required for solar position. Install via pip install pvlib") from e

    sp = pvlib.solarposition.get_solarposition(times, latitude, longitude)
    # pvlib azimuth: degrees clockwise from North
    elev_deg = sp["apparent_elevation"].to_numpy()
    az_deg = sp["azimuth"].to_numpy()

    alpha = np.deg2rad(elev_deg)
    psi = np.deg2rad(az_deg)
    return alpha, psi

def facade_incidence_cos(alpha: np.ndarray, psi: np.ndarray, gamma_deg: float) -> np.ndarray:
    """
    Vertical facade incidence cosine under the convention:
      cos(theta) = max(0, cos(alpha) * cos(psi - gamma))
    where alpha is solar elevation (rad), psi is solar azimuth (rad),
    gamma is facade outward normal azimuth (deg).
    """
    gamma = np.deg2rad(gamma_deg)
    cos_theta = np.cos(alpha) * np.cos(psi - gamma)
    return np.maximum(0.0, cos_theta)

def facade_direct_irradiance(DNI: np.ndarray, cos_theta: np.ndarray) -> np.ndarray:
    """Direct irradiance on facade plane: I_f = DNI * cos(theta)."""
    return DNI * cos_theta

def facade_diffuse_irradiance_isotropic(DHI: np.ndarray, beta_deg: float = 90.0) -> np.ndarray:
    """
    各向同性天空散射到倾角为 beta 的平面：
      I_d = DHI * (1 + cos(beta)) / 2
    垂直立面 beta=90° => 0.5 * DHI
    """
    beta = np.deg2rad(beta_deg)
    return DHI * (1.0 + np.cos(beta)) / 2.0

def facade_ground_reflected_irradiance(GHI: np.ndarray, rho_g: float = 0.2, beta_deg: float = 90.0) -> np.ndarray:
    """
    地面反射到倾角为 beta 的平面：
      I_r = rho_g * GHI * (1 - cos(beta)) / 2
    垂直立面 beta=90° => 0.5 * rho_g * GHI
    """
    beta = np.deg2rad(beta_deg)
    return rho_g * GHI * (1.0 - np.cos(beta)) / 2.0
