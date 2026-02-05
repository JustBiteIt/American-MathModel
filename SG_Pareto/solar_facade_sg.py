from __future__ import annotations

from typing import Dict, Any

import numpy as np
import pandas as pd

from borealis_model.albedo import rho_ground_timeseries
from borealis_model.solar import solar_position
from SG_Pareto.shading_passive import compute_u_facades


def compute_facade_irradiance(
    df: pd.DataFrame,
    *,
    location,
    facade_azimuths_deg: Dict[str, float],
    dN: float,
    dS: float,
    etaE: float,
    etaW: float,
    window_height_m: float,
    optics_borealis,
    eps: float = 1e-6,
    enable_passive_shading: bool = True,
    return_debug: bool = False,
) -> Dict[str, Any]:
    """
    Compute facade shortwave irradiance per document:
      I_f = u_f * DNI * mu_f + 0.5*V_sky*DHI + 0.5*V_gr*rho*GHI
    """
    times = df.index
    DNI = df["DNI"].to_numpy(float)
    DHI = df["DHI"].to_numpy(float)
    GHI = df["GHI"].to_numpy(float)
    T_out = df["T_out"].to_numpy(float)

    alpha, psi = solar_position(times, location.latitude, location.longitude)

    gamma_rad = {f: np.deg2rad(float(facade_azimuths_deg[f])) for f in ("N", "E", "S", "W")}
    mu = {f: np.maximum(np.cos(alpha) * np.cos(psi - gamma_rad[f]), 0.0) for f in ("N", "E", "S", "W")}

    if enable_passive_shading:
        u_facade = compute_u_facades(
            alpha_rad=alpha,
            psi_rad=psi,
            dN_m=dN,
            dS_m=dS,
            etaE_rad=etaE,
            etaW_rad=etaW,
            gamma_dict_rad=gamma_rad,
            window_height_m=window_height_m,
            eps=eps,
        )
    else:
        u_facade = {f: np.ones_like(alpha) for f in ("N", "E", "S", "W")}

    # ground albedo
    rho_g = rho_ground_timeseries(
        T_out_C=T_out,
        rho_bare=optics_borealis.rho_bare,
        rho_snow=optics_borealis.rho_snow,
        Ta_thr_C=optics_borealis.snow_Ta_threshold_C,
    )

    I_sky = 0.5 * DHI
    I_ground = 0.5 * rho_g * GHI
    I_other = optics_borealis.V_sky * I_sky + optics_borealis.V_gr * I_ground

    I_f = {f: (u_facade[f] * (DNI * mu[f]) + I_other) for f in ("N", "E", "S", "W")}

    out: Dict[str, Any] = {
        "I_f": I_f,
        "rho_g": rho_g,
        "I_other": I_other,
    }
    if return_debug:
        out.update(
            {
                "alpha_rad": alpha,
                "psi_rad": psi,
                "mu": mu,
                "u": u_facade,
                "gamma_rad": gamma_rad,
            }
        )
    return out
