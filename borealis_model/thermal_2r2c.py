# borealis_model/thermal_2r2c.py
from __future__ import annotations
import numpy as np

def simulate_2r2c_with_heating(
    Phi_s_W: np.ndarray,
    T_out_C: np.ndarray,
    heat_mask: np.ndarray,
    Ci: float,
    Cm: float,
    Ria: float,
    Rim: float,
    eta: float,
    T_min_C: float,
    dt_hours: float = 1.0,
    T_init_C: float = 22.0,
):
    """
    2R2C（空气 Ti + 热质量 Tm）+ 冬季供暖钳位（只在 heat_mask=True 的时段启用）。
    输出：
      Ti: (n+1,)
      Tm: (n+1,)
      Phi_h: (n,) 供暖功率 W
    """
    Phi_s = np.asarray(Phi_s_W, dtype=float)
    Ta = np.asarray(T_out_C, dtype=float)
    hm = np.asarray(heat_mask, dtype=bool)

    n = len(Ta)
    if len(Phi_s) != n:
        raise ValueError("Phi_s_W 与 T_out_C 长度必须一致")

    dt = float(dt_hours) * 3600.0

    Ti = np.zeros(n + 1, dtype=float)
    Tm = np.zeros(n + 1, dtype=float)
    Phi_h = np.zeros(n, dtype=float)

    Ti[0] = float(T_init_C)
    Tm[0] = float(T_init_C)

    Ci = float(Ci); Cm = float(Cm)
    Ria = float(Ria); Rim = float(Rim)
    eta = float(eta)

    for k in range(n):
        term_ia = (Ta[k] - Ti[k]) / Ria
        term_im = (Tm[k] - Ti[k]) / Rim
        base = term_ia + term_im + eta * Phi_s[k]

        Ti_free = Ti[k] + (dt / Ci) * base

        if hm[k] and (Ti_free < T_min_C):
            Phi_h_req = (Ci / dt) * (T_min_C - Ti[k]) - base
            Phi_h[k] = max(0.0, Phi_h_req)
        else:
            Phi_h[k] = 0.0

        Ti[k + 1] = Ti[k] + (dt / Ci) * (base + Phi_h[k])

        term_m = (Ti[k] - Tm[k]) / Rim + (1.0 - eta) * Phi_s[k]
        Tm[k + 1] = Tm[k] + (dt / Cm) * term_m

    return Ti, Tm, Phi_h


def simulate_2r2c_with_heating_metrics(
    Phi_s_W: np.ndarray,
    T_out_C: np.ndarray,
    heat_mask: np.ndarray,
    summer_mask: np.ndarray,
    Ci: float,
    Cm: float,
    Ria: float,
    Rim: float,
    eta: float,
    T_min_C: float,
    T_max_C: float,
    dt_hours: float = 1.0,
    T_init_C: float = 22.0,
):
    """
    ✅ 只计算 Eh / OH / Hoh，不返回整条时间序列（优化阶段用，速度快很多）
    - Eh_kWh = sum(Phi_h * dt)/1000 over heating months
    - OH = sum(max(Ti - Tmax, 0)*dt) over summer months
    - Hoh = sum(1(Ti > Tmax)*dt) over summer months
    """
    Phi_s = np.asarray(Phi_s_W, dtype=float)
    Ta = np.asarray(T_out_C, dtype=float)
    hm = np.asarray(heat_mask, dtype=bool)
    sm = np.asarray(summer_mask, dtype=bool)

    n = len(Ta)
    if len(Phi_s) != n:
        raise ValueError("Phi_s_W 与 T_out_C 长度必须一致")

    dt_h = float(dt_hours)
    dt = dt_h * 3600.0

    Ci = float(Ci); Cm = float(Cm)
    Ria = float(Ria); Rim = float(Rim)
    eta = float(eta)
    T_min_C = float(T_min_C)
    T_max_C = float(T_max_C)

    Ti = float(T_init_C)
    Tm = float(T_init_C)

    Eh_kWh = 0.0
    OH = 0.0
    Hoh = 0.0

    for k in range(n):
        # 过热指标用 Ti[k]（与 times[k] 对齐）
        if sm[k]:
            ex = Ti - T_max_C
            if ex > 0:
                OH += ex * dt_h
                Hoh += 1.0 * dt_h

        # thermal update with heating clamp
        term_ia = (Ta[k] - Ti) / Ria
        term_im = (Tm - Ti) / Rim
        base = term_ia + term_im + eta * Phi_s[k]

        Ti_free = Ti + (dt / Ci) * base

        Phi_h = 0.0
        if hm[k] and (Ti_free < T_min_C):
            Phi_h_req = (Ci / dt) * (T_min_C - Ti) - base
            Phi_h = max(0.0, Phi_h_req)

        # Eh accumulate (only when heating season)
        if hm[k]:
            Eh_kWh += (Phi_h * dt_h) / 1000.0

        # update states
        Ti_next = Ti + (dt / Ci) * (base + Phi_h)
        term_m = (Ti - Tm) / Rim + (1.0 - eta) * Phi_s[k]
        Tm_next = Tm + (dt / Cm) * term_m

        Ti, Tm = Ti_next, Tm_next

    return float(Eh_kWh), float(OH), float(Hoh)
