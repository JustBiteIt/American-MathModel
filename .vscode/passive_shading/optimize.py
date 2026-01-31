# optimize.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional

from solar import (
    solar_position,
    facade_incidence_cos,
    facade_direct_irradiance,
    facade_diffuse_irradiance_isotropic,
    facade_ground_reflected_irradiance,
)
from shading import u_overhang, u_louver
from gains import solar_gains_by_facade_split, visible_gains_total_split
from thermal import simulate_1r1c_hourly
from constraints import occupancy_mask, sunlit_mask, evaluate_visual_constraints
from objective import annual_cooling_load_kwh


def _get_ground_albedo(optical, default: float = 0.20) -> float:
    """
    地表反照率 rho_g 的获取策略：
    - 若 optical 里有 rho_g 字段，则使用它；
    - 否则取默认 0.20（草地/一般地面常用粗略值）。
    """
    return float(getattr(optical, "rho_g", default))


def _get_k_other(optical, default: float = 1.0) -> float:
    """
    I_other = (I_sky + I_ground) 的折减系数 k_other
    - 若 optical 里有 k_other 字段，则使用它（建议 0~1）
    - 否则默认 1.0（不改变当前结果）
    """
    try:
        k = float(getattr(optical, "k_other", default))
    except Exception:
        k = default
    # 防御：限制范围，避免误填导致数值爆炸
    return float(np.clip(k, 0.0, 1.0))


def evaluate_design(
    df: pd.DataFrame,
    location,
    building,
    optical,
    thermal,
    schedule,
    visual,
    x: Tuple[float, float, float, float],  # (dN, dS, etaE, etaW)
) -> Dict[str, Any]:
    """
    Full closed-loop evaluation (3-part irradiance, beta=90°):
      weather + solar position
        -> beam on each facade: I_b,f = DNI * cos(theta_f)
        -> diffuse on vertical: I_d = DHI*(1+cosβ)/2 = 0.5*DHI
        -> ground-reflected on vertical: I_r = rho_g*GHI*(1-cosβ)/2 = 0.5*rho_g*GHI
        -> I_other = k_other*(I_d + I_r)   (isotropic simplification; same for all vertical facades)

      shading transmittance u_f(t) applies primarily to beam component:
        -> gains: Q_f = A_win,f * tau * ( u_f*I_b,f + I_other )

      -> 1R1C + setpoint clamp -> annual cooling load
      -> visual constraints feasibility
    """
    dN, dS, etaE, etaW = x

    # ---------- Inputs ----------
    times = df.index

    # 必需列：DNI, DHI, GHI, T_out
    DNI = df["DNI"].to_numpy(dtype=float)
    DHI = df["DHI"].to_numpy(dtype=float)
    GHI = df["GHI"].to_numpy(dtype=float)
    T_out = df["T_out"].to_numpy(dtype=float)

    # ---------- Solar position ----------
    alpha, psi = solar_position(times, location.latitude, location.longitude)

    # ---------- 1) Beam irradiance on each facade ----------
    I_beam_facade: Dict[str, np.ndarray] = {}
    for f in ["N", "E", "S", "W"]:
        cos_t = facade_incidence_cos(alpha, psi, building.facade_azimuth_deg[f])
        I_beam_facade[f] = facade_direct_irradiance(DNI, cos_t)

    # ---------- 2) Diffuse + ground-reflected on vertical (beta=90°) ----------
    rho_g = _get_ground_albedo(optical, default=0.20)
    I_sky = facade_diffuse_irradiance_isotropic(DHI, beta_deg=90.0)                 # 0.5*DHI
    I_ground = facade_ground_reflected_irradiance(GHI, rho_g=rho_g, beta_deg=90.0)  # 0.5*rho_g*GHI

    # ✅ 关键改动：给 I_other 加折减开关（默认=1.0 不破坏现有）
    k_other = _get_k_other(optical, default=1.0)
    I_other = k_other * (I_sky + I_ground)

    # ---------- Shading transmittance u_f(t) (beam only) ----------
    Hwin = building.window_height_m
    u_facade = {
        "N": u_overhang(alpha, psi, building.facade_azimuth_deg["N"], dN, Hwin),
        "S": u_overhang(alpha, psi, building.facade_azimuth_deg["S"], dS, Hwin),
        "E": u_louver(alpha, psi, building.facade_azimuth_deg["E"], etaE),
        "W": u_louver(alpha, psi, building.facade_azimuth_deg["W"], etaW),
    }

    A_win = building.window_areas()

    # ---------- Heat gains (split: beam shaded, diffuse+reflected unshaded) ----------
    Q_by_f = solar_gains_by_facade_split(
        I_beam_facade=I_beam_facade,
        I_other_W_per_m2=I_other,
        A_win=A_win,
        u_facade=u_facade,
        tau_heat=optical.tau_heat
    )
    Q_sol = Q_by_f["N"] + Q_by_f["E"] + Q_by_f["S"] + Q_by_f["W"]

    # ---------- Schedule masks ----------
    occ = occupancy_mask(times, schedule.occ_start_hour, schedule.occ_end_hour)

    # 白天判定：优先用扩展版（日照=alpha>0 且 DNI/DHI/GHI 任意有能量）；若你的 constraints.py 还是旧版，会自动降级
    try:
        day = sunlit_mask(DNI, alpha, DHI=DHI, GHI=GHI)
    except TypeError:
        day = sunlit_mask(DNI, alpha)

    day_occ = occ & day

    # ---------- 1R1C simulation with setpoint clamp ----------
    T, Q_cool, L_kWh = simulate_1r1c_hourly(
        Q_sol_W=Q_sol,
        T_out_C=T_out,
        occ_mask=occ,
        C_J_per_K=thermal.C_J_per_K,
        H_W_per_K=thermal.H_W_per_K,
        T_init_C=thermal.T_init_C,
        T_set_C=thermal.T_set_C,
        dt_hours=thermal.dt_hours,
        Q_int_W=None
    )
    L_year = annual_cooling_load_kwh(L_kWh)

    # ---------- Visual proxy (split logic consistent with heat gains) ----------
    Q_vis = visible_gains_total_split(
        I_beam_facade=I_beam_facade,
        I_other_W_per_m2=I_other,
        A_win=A_win,
        u_facade=u_facade,
        tau_vis=optical.tau_vis
    )
    A_total = sum(A_win.values())
    E_vis = Q_vis / max(A_total, 1e-12)

    feasible, vis_stats = evaluate_visual_constraints(
        E_vis_W_per_m2=E_vis,
        mask_day_occ=day_occ,
        E_min=visual.E_min,
        p_min=visual.p_min,
        E_glare=visual.E_glare,
        H_glare_max=visual.H_glare_max
    )

    return {
        "x": x,
        "feasible": feasible,
        "L_year_kWh": L_year,
        "vis": vis_stats,
        "series": {
            "T_C": T,
            "Q_cool_W": Q_cool,
            "L_kWh": L_kWh,
            "E_vis_Wm2": E_vis,
            # 额外输出：便于你画图/写论文
            "I_sky_Wm2": I_sky,
            "I_ground_Wm2": I_ground,
            "I_other_Wm2": I_other,
            "I_beam_N_Wm2": I_beam_facade["N"],
            "I_beam_E_Wm2": I_beam_facade["E"],
            "I_beam_S_Wm2": I_beam_facade["S"],
            "I_beam_W_Wm2": I_beam_facade["W"],
            "u_N": u_facade["N"],
            "u_E": u_facade["E"],
            "u_S": u_facade["S"],
            "u_W": u_facade["W"],
        }
    }


def optimize_with_scipy_de(
    df, location, building, optical, thermal, schedule, visual, bounds
):
    """
    Differential Evolution with feasibility-first penalty.
    改进点：
    - baseline_x 用 bounds 上界（避免你改 bounds 后口径不一致）
    - 记录“搜索过程中遇到的最优可行解”，最终输出 best 保证可行（只要搜索中出现过可行点）
    """
    try:
        from scipy.optimize import differential_evolution
    except ImportError as e:
        raise ImportError("scipy is required for optimization. Install via pip install scipy") from e

    # ✅ Baseline：无挑檐 + 百叶尽量“开”（取当前 bounds 上界）
    baseline_x = (0.0, 0.0, bounds.etaE_bounds_rad[1], bounds.etaW_bounds_rad[1])
    baseline = evaluate_design(df, location, building, optical, thermal, schedule, visual, baseline_x)

    best_feasible_x: Optional[Tuple[float, float, float, float]] = None
    best_feasible_L: float = float("inf")

    def penalized_obj(v):
        nonlocal best_feasible_x, best_feasible_L

        x = (float(v[0]), float(v[1]), float(v[2]), float(v[3]))
        res = evaluate_design(df, location, building, optical, thermal, schedule, visual, x)

        if res["feasible"]:
            if res["L_year_kWh"] < best_feasible_L:
                best_feasible_L = float(res["L_year_kWh"])
                best_feasible_x = x
            return float(res["L_year_kWh"])

        # 不可行：罚项（采光违约权重大一点，避免算法直接把 u 压成 0 导致“黑屋子但不眩光”）
        pen = (
            1e6
            + 1e5 * max(0.0, visual.p_min - float(res["vis"]["p_daylight"]))
            + 1e3 * max(0.0, float(res["vis"]["H_glare"]) - float(visual.H_glare_max))
        )
        return float(res["L_year_kWh"]) + float(pen)

    de_bounds = [
        bounds.dN_bounds_m,
        bounds.dS_bounds_m,
        bounds.etaE_bounds_rad,
        bounds.etaW_bounds_rad,
    ]

    result = differential_evolution(
        penalized_obj,
        bounds=de_bounds,
        maxiter=40,
        popsize=12,
        polish=True,
        seed=1
    )

    # SciPy 的最终点（可能不可行）
    best_x_raw = (float(result.x[0]), float(result.x[1]), float(result.x[2]), float(result.x[3]))
    best_raw = evaluate_design(df, location, building, optical, thermal, schedule, visual, best_x_raw)

    # ✅ 最终输出：优先给“搜索过程中最优可行解”，否则兜底 baseline
    if best_feasible_x is not None:
        best_out = evaluate_design(df, location, building, optical, thermal, schedule, visual, best_feasible_x)
    else:
        best_out = baseline

    return {"baseline": baseline, "best": best_out, "best_raw": best_raw, "scipy_result": result}
