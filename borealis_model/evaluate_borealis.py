# borealis_model/evaluate_borealis.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from albedo import rho_ground_timeseries
from thermal_2r2c import (
    simulate_2r2c_with_heating,
    simulate_2r2c_with_heating_metrics,
)
from metrics import month_mask, heating_energy_kwh, overheat_metrics

from solar import solar_position, facade_incidence_cos, facade_direct_irradiance
from shading import u_overhang, u_louver


def evaluate_borealis(
    df: pd.DataFrame,
    location,
    building,
    optical_base,
    schedule,
    bo_optical,
    bo_therm,
    bo_seasons,
    x: Tuple[float, float, float, float],
    *,
    return_series: bool = True,
) -> Dict[str, Any]:

    dN, dS, etaE, etaW = x
    times = df.index

    DNI = df["DNI"].to_numpy(float)
    DHI = df["DHI"].to_numpy(float)
    GHI = df["GHI"].to_numpy(float)
    T_out = df["T_out"].to_numpy(float)

    # solar
    alpha, psi = solar_position(times, location.latitude, location.longitude)

    # direct to facade
    I_beam: Dict[str, np.ndarray] = {}
    for f in ["N", "E", "S", "W"]:
        cos_t = facade_incidence_cos(alpha, psi, building.facade_azimuth_deg[f])
        I_beam[f] = facade_direct_irradiance(DNI, cos_t)

    # shading transmissivity for beam
    Hwin = building.window_height_m
    u_facade = {
        "N": u_overhang(alpha, psi, building.facade_azimuth_deg["N"], dN, Hwin),
        "S": u_overhang(alpha, psi, building.facade_azimuth_deg["S"], dS, Hwin),
        "E": u_louver(alpha, psi, building.facade_azimuth_deg["E"], etaE),
        "W": u_louver(alpha, psi, building.facade_azimuth_deg["W"], etaW),
    }

    # rho_g(t)
    rho_g = rho_ground_timeseries(
        T_out_C=T_out,
        rho_bare=bo_optical.rho_bare,
        rho_snow=bo_optical.rho_snow,
        Ta_thr_C=bo_optical.snow_Ta_threshold_C
    )

    # I_sky / I_ground
    I_sky = 0.5 * DHI
    I_ground = 0.5 * rho_g * GHI

    # k_other (keeps rho_g inside I_ground)
    I_other_eff = bo_optical.k_other * (bo_optical.V_sky * I_sky + bo_optical.V_gr * I_ground)

    # If(t)
    I_f = {f: (u_facade[f] * I_beam[f] + I_other_eff) for f in ["N", "E", "S", "W"]}

    # solar gain
    A_win = building.window_areas()
    g_eq = float(getattr(optical_base, "tau_heat", 0.6))

    Phi_s = (
        A_win["N"] * g_eq * I_f["N"]
        + A_win["E"] * g_eq * I_f["E"]
        + A_win["S"] * g_eq * I_f["S"]
        + A_win["W"] * g_eq * I_f["W"]
    )

    # masks
    heat_mask = month_mask(times, bo_seasons.heating_months)
    sum_mask = month_mask(times, bo_seasons.summer_months)

    if not return_series:
        Eh_kWh, OH, Hoh = simulate_2r2c_with_heating_metrics(
            Phi_s_W=Phi_s,
            T_out_C=T_out,
            heat_mask=heat_mask,
            summer_mask=sum_mask,
            Ci=bo_therm.Ci_J_per_K,
            Cm=bo_therm.Cm_J_per_K,
            Ria=bo_therm.Ria_K_per_W,
            Rim=bo_therm.Rim_K_per_W,
            eta=bo_therm.eta_air,
            T_min_C=bo_therm.T_min_C,
            T_max_C=bo_therm.T_max_C,
            dt_hours=bo_therm.dt_hours,
            T_init_C=bo_therm.T_init_C,
        )
        return {
            "x": x,
            "Eh_kWh": Eh_kWh,
            "OH_degC_h": OH,
            "Hoh_h": Hoh,
        }

    # full series (for plotting / debug)
    Ti, Tm, Phi_h = simulate_2r2c_with_heating(
        Phi_s_W=Phi_s,
        T_out_C=T_out,
        heat_mask=heat_mask,
        Ci=bo_therm.Ci_J_per_K,
        Cm=bo_therm.Cm_J_per_K,
        Ria=bo_therm.Ria_K_per_W,
        Rim=bo_therm.Rim_K_per_W,
        eta=bo_therm.eta_air,
        T_min_C=bo_therm.T_min_C,
        dt_hours=bo_therm.dt_hours,
        T_init_C=bo_therm.T_init_C,
    )

    Eh_kWh = heating_energy_kwh(Phi_h, heat_mask, bo_therm.dt_hours)
    OH, Hoh = overheat_metrics(Ti, sum_mask, bo_therm.T_max_C, bo_therm.dt_hours)

    return {
        "x": x,
        "Eh_kWh": Eh_kWh,
        "OH_degC_h": OH,
        "Hoh_h": Hoh,
        "series": {
            "Ti_C": Ti,
            "Tm_C": Tm,
            "Phi_s_W": Phi_s,
            "Phi_h_W": Phi_h,
            "I_other_eff_Wm2": I_other_eff,
            "rho_g": rho_g,
        }
    }
