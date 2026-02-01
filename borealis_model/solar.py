# borealis_model/solar.py
from __future__ import annotations
import numpy as np
import pandas as pd

def _to_numpy_times_utc(times: pd.DatetimeIndex) -> np.ndarray:
    if times.tz is None:
        raise ValueError("times 必须是 tz-aware 的 DatetimeIndex")
    t_utc = times.tz_convert("UTC")
    return t_utc.to_numpy(dtype="datetime64[s]")

def solar_position(times: pd.DatetimeIndex, latitude_deg: float, longitude_deg: float):
    """
    返回：
      alpha: solar elevation (rad)
      psi:   solar azimuth (rad), 0=N, +pi/2=E
    """
    # 转 UTC 秒
    t = _to_numpy_times_utc(times)
    # 转为 pandas 时间做字段提取
    t_pd = pd.DatetimeIndex(t).tz_localize("UTC")

    lat = np.deg2rad(latitude_deg)
    lon = longitude_deg

    # 日序与小时
    doy = t_pd.dayofyear.to_numpy(dtype=float)
    hour = (t_pd.hour + t_pd.minute / 60.0 + t_pd.second / 3600.0).to_numpy(dtype=float)

    # 近似太阳赤纬（Cooper）
    gamma = 2.0 * np.pi * (doy - 1.0) / 365.0
    decl = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.00148  * np.sin(3 * gamma)
    )

    # equation of time (minutes)
    eot = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma)
        - 0.040849 * np.sin(2 * gamma)
    )

    # 本地太阳时：UTC -> minutes -> 加经度修正
    # 这里用简化：LST_minutes = hour*60 + eot + 4*lon
    # （lon 为东经正；新加坡是正）
    lst_min = hour * 60.0 + eot + 4.0 * lon
    ha = np.deg2rad((lst_min / 4.0) - 180.0)  # hour angle, rad

    # elevation
    sin_alpha = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(ha)
    sin_alpha = np.clip(sin_alpha, -1.0, 1.0)
    alpha = np.arcsin(sin_alpha)

    # azimuth (0=N, +E)
    # 公式：tan(psi) = sin(ha) / (cos(ha)*sin(lat) - tan(decl)*cos(lat))
    denom = np.cos(ha) * np.sin(lat) - np.tan(decl) * np.cos(lat)
    psi = np.arctan2(np.sin(ha), denom)  # [-pi,pi]
    # 转成 0..2pi
    psi = (psi + 2.0 * np.pi) % (2.0 * np.pi)

    return alpha, psi

def facade_incidence_cos(alpha: np.ndarray, psi: np.ndarray, facade_azimuth_deg: float) -> np.ndarray:
    """
    竖直立面：cos(theta_f) = cos(alpha) * cos(psi - gamma_f)
    gamma_f: 立面外法线方位角（deg, N=0 E=90）
    """
    gamma = np.deg2rad(facade_azimuth_deg)
    cos_t = np.cos(alpha) * np.cos(psi - gamma)
    return np.clip(cos_t, 0.0, 1.0)

def facade_direct_irradiance(DNI: np.ndarray, cos_theta: np.ndarray) -> np.ndarray:
    return DNI * np.clip(cos_theta, 0.0, None)

def facade_diffuse_irradiance_isotropic(DHI: np.ndarray, beta_deg: float = 90.0) -> np.ndarray:
    beta = np.deg2rad(beta_deg)
    return DHI * (1.0 + np.cos(beta)) / 2.0

def facade_ground_reflected_irradiance(GHI: np.ndarray, rho_g: np.ndarray, beta_deg: float = 90.0) -> np.ndarray:
    beta = np.deg2rad(beta_deg)
    return rho_g * GHI * (1.0 - np.cos(beta)) / 2.0
