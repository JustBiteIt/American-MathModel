# gains.py
import numpy as np
from typing import Dict

def solar_gains_by_facade(I_facade: Dict[str, np.ndarray],
                          A_win: Dict[str, float],
                          u_facade: Dict[str, np.ndarray],
                          tau_heat: float) -> Dict[str, np.ndarray]:
    """Q_f(t) = A_win,f * tau_heat * u_f(t) * I_f(t)."""
    Q = {}
    for f in ["N", "E", "S", "W"]:
        Q[f] = A_win[f] * tau_heat * u_facade[f] * I_facade[f]
    return Q

def visible_gains_total(I_facade: Dict[str, np.ndarray],
                        A_win: Dict[str, float],
                        u_facade: Dict[str, np.ndarray],
                        tau_vis: float) -> np.ndarray:
    """Q_vis(t) = sum_f A_win,f * tau_vis * u_f(t) * I_f(t)."""
    out = 0.0
    for f in ["N", "E", "S", "W"]:
        out = out + A_win[f] * tau_vis * u_facade[f] * I_facade[f]
    return out

def solar_gains_by_facade_split(
    I_beam_facade: Dict[str, np.ndarray],
    I_other_W_per_m2: np.ndarray,  # I_other = I_diffuse + I_reflected (对四个立面这里相同)
    A_win: Dict[str, float],
    u_facade: Dict[str, np.ndarray],
    tau_heat: float
) -> Dict[str, np.ndarray]:
    """
    热得热（拆分版）：
      Q_f = A * tau_heat * ( u * I_beam + I_other )
    解释：
      - u(t) 主要作用于直射（几何遮挡）
      - I_other（天空散射+地面反射）先不加几何遮挡（简化）
    """
    Q = {}
    for f in ["N", "E", "S", "W"]:
        Q[f] = A_win[f] * tau_heat * (u_facade[f] * I_beam_facade[f] + I_other_W_per_m2)
    return Q

def visible_gains_total_split(
    I_beam_facade: Dict[str, np.ndarray],
    I_other_W_per_m2: np.ndarray,
    A_win: Dict[str, float],
    u_facade: Dict[str, np.ndarray],
    tau_vis: float
) -> np.ndarray:
    """
    可见光（拆分版）：
      Q_vis = Σ A * tau_vis * ( u * I_beam + I_other )
    """
    out = 0.0
    for f in ["N", "E", "S", "W"]:
        out = out + A_win[f] * tau_vis * (u_facade[f] * I_beam_facade[f] + I_other_W_per_m2)
    return out
