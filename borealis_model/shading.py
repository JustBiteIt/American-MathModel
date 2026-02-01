# borealis_model/shading.py
from __future__ import annotations
import numpy as np

def _safe_div(a, b, eps=1e-12):
    return a / np.where(np.abs(b) < eps, eps, b)

def u_overhang(alpha, psi, facade_azimuth_deg, d_m, Hwin_m):
    """
    简化挑檐遮阳（只作用于直射 beam）：
      - 计算“立面法向剖面内”的等效太阳高度角 beta_p
      - 阴影长度 L = d * tan(beta_p)
      - 透过率 u = max(0, 1 - L/Hwin)
    """
    gamma = np.deg2rad(facade_azimuth_deg)
    az_diff = psi - gamma

    # sun in front: cos(az_diff) > 0
    front = np.cos(az_diff) > 1e-6

    # beta_p = arctan( tan(alpha) / cos(az_diff) )
    beta_p = np.arctan(_safe_div(np.tan(alpha), np.cos(az_diff)))

    L = d_m * np.tan(np.clip(beta_p, 0.0, np.pi/2 - 1e-6))
    u = 1.0 - _safe_div(L, max(Hwin_m, 1e-6))
    u = np.clip(u, 0.0, 1.0)

    # 不在正面时：beam 本来就≈0，这里 u 设 1 不影响
    u = np.where(front, u, 1.0)
    return u

def u_louver(alpha, psi, facade_azimuth_deg, eta_rad):
    """
    简化百叶（垂直百叶 + 水平剖面角 HPA）：
      HPA = arctan( |sin(az_diff)| / tan(alpha) )
      u = max(0, 1 - HPA/eta)
    """
    gamma = np.deg2rad(facade_azimuth_deg)
    az_diff = psi - gamma

    # alpha 太小会导致 tan(alpha) ~ 0 -> HPA 很大 -> u -> 0（符合低太阳角更需遮挡）
    HPA = np.arctan(_safe_div(np.abs(np.sin(az_diff)), np.tan(np.maximum(alpha, 1e-6))))

    eta = np.maximum(float(eta_rad), 1e-6)
    u = 1.0 - HPA / eta
    u = np.clip(u, 0.0, 1.0)

    # 不在正面时：beam 本来就≈0，这里 u 设 1 不影响
    front = np.cos(az_diff) > 1e-6
    u = np.where(front, u, 1.0)
    return u
