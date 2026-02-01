# borealis_model/albedo.py
from __future__ import annotations
import numpy as np

def rho_ground_timeseries(
    T_out_C: np.ndarray,
    rho_bare: float = 0.20,
    rho_snow: float = 0.75,
    Ta_thr_C: float = 0.0,
    transition_C: float = 1.5,
) -> np.ndarray:
    """
    用室外温度生成雪地反照率时间序列 rho_g(t)，并做平滑过渡：
      snow_frac = sigmoid( (Ta_thr - T_out)/transition )
      rho_g = rho_bare*(1-snow_frac) + rho_snow*snow_frac
    """
    T = np.asarray(T_out_C, dtype=float)
    w = max(float(transition_C), 1e-6)
    x = (float(Ta_thr_C) - T) / w
    snow_frac = 1.0 / (1.0 + np.exp(-x))
    return rho_bare * (1.0 - snow_frac) + rho_snow * snow_frac
