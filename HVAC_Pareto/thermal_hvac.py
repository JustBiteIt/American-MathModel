
# HVAC_Pareto/thermal_hvac.py
from __future__ import annotations
import numpy as np
from typing import Tuple

def simulate_2r2c_with_deadband_hvac(
    Phi_s_W: np.ndarray,
    T_out_C: np.ndarray,
    *,
    Ci: float,
    Cm: float,
    Ria: float,
    Rim: float,
    eta: float,
    T_heat_C: float,
    T_cool_C: float,
    dt_hours: float = 1.0,
    T_init_C: float = 21.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    2R2C (Ti air + Tm mass) with ideal deadband HVAC:
      - if Ti_free < T_heat -> add heating to clamp Ti_next = T_heat
      - if Ti_free > T_cool -> add cooling (negative) to clamp Ti_next = T_cool
      - else HVAC=0

    Returns
      Ti: (n+1,)
      Tm: (n+1,)
      Phi_h: (n,)  HVAC power W (positive=heating, negative=cooling)
    """
    Phi_s = np.asarray(Phi_s_W, dtype=float)
    Ta = np.asarray(T_out_C, dtype=float)
    n = len(Ta)
    if len(Phi_s) != n:
        raise ValueError("Phi_s_W and T_out_C must have the same length")

    dt = float(dt_hours) * 3600.0

    Ti = np.zeros(n + 1, dtype=float)
    Tm = np.zeros(n + 1, dtype=float)
    Phi_h = np.zeros(n, dtype=float)

    Ti[0] = float(T_init_C)
    Tm[0] = float(T_init_C)

    Ci = float(Ci); Cm = float(Cm)
    Ria = float(Ria); Rim = float(Rim)
    eta = float(eta)

    Theat = float(T_heat_C)
    Tcool = float(T_cool_C)
    if not (Theat < Tcool):
        raise ValueError("Require T_heat_C < T_cool_C for deadband")

    for k in range(n):
        term_ia = (Ta[k] - Ti[k]) / Ria
        term_im = (Tm[k] - Ti[k]) / Rim
        base = term_ia + term_im + eta * Phi_s[k]

        Ti_free = Ti[k] + (dt / Ci) * base

        if Ti_free < Theat:
            Tset = Theat
            Phi_req = (Ci / dt) * (Tset - Ti[k]) - base
            Phi_h[k] = max(0.0, Phi_req)  # heating
        elif Ti_free > Tcool:
            Tset = Tcool
            Phi_req = (Ci / dt) * (Tset - Ti[k]) - base
            Phi_h[k] = min(0.0, Phi_req)  # cooling (negative)
        else:
            Phi_h[k] = 0.0

        Ti[k + 1] = Ti[k] + (dt / Ci) * (base + Phi_h[k])

        term_m = (Ti[k] - Tm[k]) / Rim + (1.0 - eta) * Phi_s[k]
        Tm[k + 1] = Tm[k] + (dt / Cm) * term_m

    return Ti, Tm, Phi_h


def hvac_energy_kwh(Phi_h_W: np.ndarray, dt_hours: float = 1.0) -> Tuple[float, float]:
    """Return (E_heat_kWh, E_cool_kWh) from HVAC power time series."""
    Phi = np.asarray(Phi_h_W, dtype=float)
    dt = float(dt_hours)
    E_heat = float(np.sum(np.maximum(Phi, 0.0)) * dt / 1000.0)
    E_cool = float(np.sum(np.maximum(-Phi, 0.0)) * dt / 1000.0)
    return E_heat, E_cool
