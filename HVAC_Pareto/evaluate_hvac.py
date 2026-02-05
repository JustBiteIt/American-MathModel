
# HVAC_Pareto/evaluate_hvac.py
from __future__ import annotations

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

# Reuse Borealis physics blocks
from borealis_model.albedo import rho_ground_timeseries
from borealis_model.solar import solar_position, facade_incidence_cos, facade_direct_irradiance
from borealis_model.shading import u_overhang, u_louver

from .thermal_hvac import simulate_2r2c_with_deadband_hvac, hvac_energy_kwh


def evaluate_design(
    df: pd.DataFrame,
    *,
    location,
    building,
    optical_base,
    optics_borealis,
    hvac_2r2c,
    x: Tuple[float, float, float, float],
    return_series: bool = False,
) -> Dict[str, Any]:
    """
    Pipeline:
      weather -> facade irradiance -> indoor solar gain -> 2R2C -> HVAC load

    Decision vector x = (dN, dS, etaE, etaW)
      dN, dS: overhang depth (m)
      etaE, etaW: louver angle parameter (rad) used by borealis_model.shading.u_louver

    Returns a dict with:
      - E_heat_kWh, E_cool_kWh
      - optional series (Ti, Tm, Phi_h, Phi_s)
    """
    dN, dS, etaE, etaW = x
    times = df.index

    DNI = df["DNI"].to_numpy(float)
    DHI = df["DHI"].to_numpy(float)
    GHI = df["GHI"].to_numpy(float)
    T_out = df["T_out"].to_numpy(float)

    # 1) solar geometry
    alpha, psi = solar_position(times, location.latitude, location.longitude)

    # 2) direct irradiance on each facade
    I_beam = {}
    for f in ["N", "E", "S", "W"]:
        cos_t = facade_incidence_cos(alpha, psi, building.facade_azimuth_deg[f])
        I_beam[f] = facade_direct_irradiance(DNI, cos_t)

    # 3) beam transmissivity from shading (reuse Borealis functions)
    Hwin = building.window_height_m
    u_facade = {
        "N": u_overhang(alpha, psi, building.facade_azimuth_deg["N"], dN, Hwin),
        "S": u_overhang(alpha, psi, building.facade_azimuth_deg["S"], dS, Hwin),
        "E": u_louver(alpha, psi, building.facade_azimuth_deg["E"], etaE),
        "W": u_louver(alpha, psi, building.facade_azimuth_deg["W"], etaW),
    }

    # 4) ground albedo timeseries (snow effect)
    rho_g = rho_ground_timeseries(
        T_out_C=T_out,
        rho_bare=optics_borealis.rho_bare,
        rho_snow=optics_borealis.rho_snow,
        Ta_thr_C=optics_borealis.snow_Ta_threshold_C,
    )

    # 5) isotropic sky & ground terms
    I_sky = 0.5 * DHI
    I_ground = 0.5 * rho_g * GHI

    # BorealisOptical: k_other*(V_sky*I_sky + V_gr*I_ground)
    I_other_eff = optics_borealis.k_other * (optics_borealis.V_sky * I_sky + optics_borealis.V_gr * I_ground)

    # 6) total facade irradiance
    I_f = {f: (u_facade[f] * I_beam[f] + I_other_eff) for f in ["N", "E", "S", "W"]}

    # 7) indoor solar gain
    A_win = building.window_areas()  # uses legacy wwr_south/wwr_other if wwr_by_facade is None
    g_eq = float(getattr(optical_base, "tau_heat", 0.6))

    Phi_s = (
        A_win["N"] * g_eq * I_f["N"]
        + A_win["E"] * g_eq * I_f["E"]
        + A_win["S"] * g_eq * I_f["S"]
        + A_win["W"] * g_eq * I_f["W"]
    )

    # 8) 2R2C + deadband HVAC
    Ti, Tm, Phi_h = simulate_2r2c_with_deadband_hvac(
        Phi_s_W=Phi_s,
        T_out_C=T_out,
        Ci=hvac_2r2c.Ci_J_per_K,
        Cm=hvac_2r2c.Cm_J_per_K,
        Ria=hvac_2r2c.Ria_K_per_W,
        Rim=hvac_2r2c.Rim_K_per_W,
        eta=hvac_2r2c.eta_air,
        T_heat_C=hvac_2r2c.T_heat_C,
        T_cool_C=hvac_2r2c.T_cool_C,
        dt_hours=hvac_2r2c.dt_hours,
        T_init_C=hvac_2r2c.T_init_C,
    )

    E_heat_kWh, E_cool_kWh = hvac_energy_kwh(Phi_h, dt_hours=hvac_2r2c.dt_hours)

    out = {
        "x": tuple(float(v) for v in x),
        "E_heat_kWh": float(E_heat_kWh),
        "E_cool_kWh": float(E_cool_kWh),
    }
    if return_series:
        out.update({
            "Phi_s_W": Phi_s,
            "Ti_C": Ti,
            "Tm_C": Tm,
            "Phi_h_W": Phi_h,
        })
    return out
