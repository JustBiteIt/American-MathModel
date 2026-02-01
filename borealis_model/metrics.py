# borealis_model/metrics.py
from __future__ import annotations
import numpy as np
import pandas as pd


def month_mask(times: pd.DatetimeIndex, months) -> np.ndarray:
    """
    返回 bool mask，长度 = len(times)
    months: iterable of int (1..12)
    """
    mset = set(int(x) for x in months)
    return np.array([(int(mm) in mset) for mm in times.month], dtype=bool)


def heating_energy_kwh(Phi_h_W: np.ndarray, heat_mask: np.ndarray, dt_hours: float) -> float:
    Phi = np.asarray(Phi_h_W, dtype=float)
    hm = np.asarray(heat_mask, dtype=bool)
    return float(np.sum(Phi[hm]) * float(dt_hours) / 1000.0)


def overheat_metrics(Ti_C: np.ndarray, summer_mask: np.ndarray, T_max_C: float, dt_hours: float):
    """
    OH = sum max(Ti - Tmax, 0) * dt
    Hoh = sum 1(Ti > Tmax) * dt
    注意：Ti 是 (n+1,)，与 times (n,) 对齐用 Ti[:-1]
    """
    T = np.asarray(Ti_C, dtype=float)[:-1]
    sm = np.asarray(summer_mask, dtype=bool)

    exceed = np.maximum(T - float(T_max_C), 0.0)
    OH = float(np.sum(exceed[sm]) * float(dt_hours))
    Hoh = float(np.sum((T[sm] > float(T_max_C)).astype(float)) * float(dt_hours))
    return OH, Hoh
