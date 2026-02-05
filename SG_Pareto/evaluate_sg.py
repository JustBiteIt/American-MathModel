from __future__ import annotations

from typing import Dict, Any
import numpy as np
import pandas as pd

from SG_Pareto.thermal_sg import simulate_2r2c_deadband, hvac_energy_kwh
from SG_Pareto.geometry import GeometryDecision
from SG_Pareto.solar_facade_sg import compute_facade_irradiance


def compute_solar_gains(
    df: pd.DataFrame,
    *,
    location,
    optical_base,
    optics_borealis,
    geometry: GeometryDecision,
    dN: float,
    dS: float,
    etaE: float,
    etaW: float,
    window_height_m: float,
    enable_passive_shading: bool = True,
) -> Dict[str, Any]:
    """Compute solar gains with geometry + facade WWR + orientation theta."""
    T_out = df["T_out"].to_numpy(float)
    GHI = df["GHI"].to_numpy(float)

    facade_az = geometry.facade_azimuths_deg()
    irr = compute_facade_irradiance(
        df,
        location=location,
        facade_azimuths_deg=facade_az,
        dN=dN,
        dS=dS,
        etaE=etaE,
        etaW=etaW,
        window_height_m=window_height_m,
        optics_borealis=optics_borealis,
        enable_passive_shading=enable_passive_shading,
    )
    I_f = irr["I_f"]

    A_win = geometry.window_areas_m2()
    g_eq = float(getattr(optical_base, "tau_heat", 0.55))

    Phi_s = (
        A_win["N"] * g_eq * I_f["N"]
        + A_win["E"] * g_eq * I_f["E"]
        + A_win["S"] * g_eq * I_f["S"]
        + A_win["W"] * g_eq * I_f["W"]
    )

    return {
        "Phi_s_W": Phi_s,
        "GHI": GHI,
        "T_out_C": T_out,
        "A_win": A_win,
    }


def evaluate_design(
    df_base: pd.DataFrame,
    *,
    location,
    optical_base,
    optics_borealis,
    hvac_cfg,
    material,
    pv_cfg,
    dN: float,
    dS: float,
    etaE: float,
    etaW: float,
    geometry: GeometryDecision,
    window_height_m: float,
    delta_trend_C: float,
    enable_passive_shading: bool = True,
    return_series: bool = False,
) -> Dict[str, Any]:
    solar = compute_solar_gains(
        df_base,
        location=location,
        optical_base=optical_base,
        optics_borealis=optics_borealis,
        geometry=geometry,
        dN=dN,
        dS=dS,
        etaE=etaE,
        etaW=etaW,
        window_height_m=window_height_m,
        enable_passive_shading=enable_passive_shading,
    )

    Phi_s = solar["Phi_s_W"]
    T_out_base = solar["T_out_C"]
    T_out_2040 = T_out_base + float(delta_trend_C)

    Ti, Tm, Phi_h = simulate_2r2c_deadband(
        Phi_s_W=Phi_s,
        T_out_C=T_out_base,
        Ci=material.Ci_J_per_K,
        Cm=material.Cm_J_per_K,
        Ria=material.Ria_K_per_W,
        Rim=material.Rim_K_per_W,
        eta=material.eta_air,
        T_heat_C=hvac_cfg.T_heat_C,
        T_cool_C=hvac_cfg.T_cool_C,
        dt_hours=hvac_cfg.dt_hours,
        T_init_C=hvac_cfg.T_init_C,
    )
    _, E_cool_th = hvac_energy_kwh(Phi_h, dt_hours=hvac_cfg.dt_hours)
    E_cool_el = E_cool_th / float(hvac_cfg.COP_cool)

    Ti2, Tm2, Phi_h2 = simulate_2r2c_deadband(
        Phi_s_W=Phi_s,
        T_out_C=T_out_2040,
        Ci=material.Ci_J_per_K,
        Cm=material.Cm_J_per_K,
        Ria=material.Ria_K_per_W,
        Rim=material.Rim_K_per_W,
        eta=material.eta_air,
        T_heat_C=hvac_cfg.T_heat_C,
        T_cool_C=hvac_cfg.T_cool_C,
        dt_hours=hvac_cfg.dt_hours,
        T_init_C=hvac_cfg.T_init_C,
    )
    _, E_cool_th_2040 = hvac_energy_kwh(Phi_h2, dt_hours=hvac_cfg.dt_hours)
    COP = float(hvac_cfg.COP_cool)
    E_cool_el_2040 = E_cool_th_2040 / COP
    cool_th_2040_W = np.maximum(-np.asarray(Phi_h2, dtype=float), 0.0)
    P_cool_el_2040_kW = (cool_th_2040_W / 1000.0) / COP
    if P_cool_el_2040_kW.size:
        peak_idx = int(np.argmax(P_cool_el_2040_kW))
        P_cool_el_peak_2040_kW = float(P_cool_el_2040_kW[peak_idx])
    else:
        peak_idx = 0
        P_cool_el_peak_2040_kW = 0.0

    A_rf = geometry.roof_area_m2()
    eta_pv = float(pv_cfg.eta_pv)
    alpha_to_cool = float(pv_cfg.alpha_to_cool)
    GHI = solar["GHI"]
    E_pv_el = float(np.sum(eta_pv * A_rf * GHI) * hvac_cfg.dt_hours / 1000.0)
    E_pv_to_cool = alpha_to_cool * E_pv_el

    MNZ = E_pv_to_cool - E_cool_el_2040

    peak_hour_local = int(df_base.index[peak_idx].hour) if len(df_base.index) else 0
    peak_datetime_local = str(df_base.index[peak_idx]) if len(df_base.index) else ""
    T_out_2040_at_peak_C = float(T_out_2040[peak_idx]) if len(T_out_2040) else float("nan")
    solar_gain_at_peak_W = float(Phi_s[peak_idx]) if len(Phi_s) else float("nan")

    out = {
        "E_cool_th_kWh": float(E_cool_th),
        "E_cool_el_kWh": float(E_cool_el),
        "E_cool_th_2040_kWh": float(E_cool_th_2040),
        "E_cool_el_2040_kWh": float(E_cool_el_2040),
        "P_cool_el_peak_2040_kW": float(P_cool_el_peak_2040_kW),
        "E_pv_el_kWh": float(E_pv_el),
        "MNZ_kWh_el": float(MNZ),
        "A_roof_m2": float(A_rf),
        "A_win_total_m2": float(geometry.window_area_total_m2()),
        "A_win_N_m2": float(solar["A_win"]["N"]),
        "A_win_E_m2": float(solar["A_win"]["E"]),
        "A_win_S_m2": float(solar["A_win"]["S"]),
        "A_win_W_m2": float(solar["A_win"]["W"]),
        "peak_hour_local": peak_hour_local,
        "peak_datetime_local": peak_datetime_local,
        "T_out_2040_at_peak_C": T_out_2040_at_peak_C,
        "solar_gain_at_peak_W": solar_gain_at_peak_W,
    }

    if return_series:
        out.update(
            {
                "Ti_C": Ti2,
                "Tm_C": Tm2,
                "Phi_h_W": Phi_h2,
                "Phi_s_W": Phi_s,
                "P_cool_el_kW_2040": P_cool_el_2040_kW,
                "T_out_2040_C": T_out_2040,
            }
        )

    return out
