# pro2_generate_timeseries.py
from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd

from config import (
    LocationConfig,
    BuildingConfig,
    OpticalConfig,
    ScheduleConfig,
    BorealisOptical,
    Borealis2R2C,
    BorealisSeasons,
)
from data_io import load_weather_csv
from evaluate_borealis import evaluate_borealis


# -----------------------------
# helpers
# -----------------------------
def _slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _parse_x(x_str: str) -> Optional[Tuple[float, float, float, float]]:
    """
    parse "(0.1, 0.2, 1.3, 1.5)" or "(0, 0, 1.57, 1.57)" safely
    """
    if not isinstance(x_str, str) or not x_str.strip():
        return None
    try:
        x = ast.literal_eval(x_str)
        if isinstance(x, (list, tuple)) and len(x) == 4:
            return (float(x[0]), float(x[1]), float(x[2]), float(x[3]))
    except Exception:
        return None
    return None


def _make_building_fi() -> BuildingConfig:
    """
    Finland WWR strategy:
      WWR_S=0.40, WWR_W=0.45, WWR_E=0.55, WWR_N=0.25
    兼容两种 BuildingConfig：
    - 只有 wwr_south/wwr_other：退化近似（N 用 0.25，其他用 0.45/0.55 无法区分）
    - 若你已经扩展成 wwr_north/wwr_east/wwr_west/wwr_south：则完整使用
    """
    fields = getattr(BuildingConfig, "__dataclass_fields__", {})
    if all(k in fields for k in ["wwr_north", "wwr_east", "wwr_west", "wwr_south"]):
        return BuildingConfig(
            wwr_south=0.40,
            wwr_west=0.45,
            wwr_east=0.55,
            wwr_north=0.25,
        )
    else:
        # 退化版本：至少把 south=0.40, north=0.25 体现出来
        # 由于结构只有 wwr_other，E/W 无法分别 0.45/0.55，这里用折中 0.50
        return BuildingConfig(
            wwr_south=0.40,
            wwr_other=0.50,  # E/W/N(退化) 统一
        )


def _make_building_passive() -> BuildingConfig:
    """
    Passive_shading 的对照 WWR（你现在 Borealis 里用的）：
      south=0.45, other=0.30
    """
    fields = getattr(BuildingConfig, "__dataclass_fields__", {})
    if all(k in fields for k in ["wwr_north", "wwr_east", "wwr_west", "wwr_south"]):
        return BuildingConfig(
            wwr_south=0.45,
            wwr_west=0.30,
            wwr_east=0.30,
            wwr_north=0.30,
        )
    else:
        return BuildingConfig(
            wwr_south=0.45,
            wwr_other=0.30,
        )


def _get_total_window_area(building: BuildingConfig) -> float:
    A_win = building.window_areas()
    return float(sum(A_win.values()))


def _export_timeseries_csv(
    out_csv: Path,
    df_weather: pd.DataFrame,
    res: Dict[str, Any],
    *,
    case_name: str,
    variant: str,
    material: str,
    wwr_scheme: str,
    x: Tuple[float, float, float, float],
    g_eq: float,
    A_win_total: float,
    schedule: ScheduleConfig,
) -> None:
    times = df_weather.index
    n = len(times)

    series = res.get("series", {})

    Ti = np.asarray(series.get("Ti_C"), dtype=float)  # (n+1,)
    Tm = np.asarray(series.get("Tm_C"), dtype=float)  # (n+1,)
    Phi_s = np.asarray(series.get("Phi_s_W"), dtype=float)  # (n,)
    Phi_h = np.asarray(series.get("Phi_h_W"), dtype=float)  # (n,)
    rho_g = np.asarray(series.get("rho_g"), dtype=float)  # (n,)
    I_other = np.asarray(series.get("I_other_eff_Wm2"), dtype=float)  # (n,)

    if len(Phi_s) != n:
        raise ValueError("Phi_s_W length mismatch with weather")
    if len(Phi_h) != n:
        raise ValueError("Phi_h_W length mismatch with weather")
    if len(rho_g) != n:
        raise ValueError("rho_g length mismatch with weather")
    if len(I_other) != n:
        raise ValueError("I_other_eff length mismatch with weather")
    if len(Ti) != n + 1:
        raise ValueError("Ti_C length mismatch with weather")
    if len(Tm) != n + 1:
        raise ValueError("Tm_C length mismatch with weather")

    # 对齐到每个小时的时刻：用 Ti[:-1], Tm[:-1]
    Ti_h = Ti[:-1]
    Tm_h = Tm[:-1]

    # Occupied mask
    hours = times.hour.to_numpy()
    occ = (hours >= schedule.occ_start_hour) & (hours < schedule.occ_end_hour)

    # 一个给“E_sol 分布图/KDE”用的 proxy（W/m^2）
    # - Esol_win = Phi_s / A_win_total  (窗单位面积太阳得热功率)
    # - Einc_avg = Phi_s / (g_eq*A_win_total)  (反推平均立面入射 If 的加权平均)
    eps = 1e-9
    Esol_win = Phi_s / max(A_win_total, eps)
    Einc_avg = Phi_s / max(g_eq * A_win_total, eps)

    # ✅ 修复 tz-aware 报错：先去掉时区，再转字符串
    time_col = times.tz_localize(None).astype(str)

    out = pd.DataFrame(
        {
            "time": time_col,  # ← 唯一关键修复点：保留原列名 "time"
            "month": times.month.to_numpy(),
            "hour": hours,
            "is_occupied": occ.astype(int),

            # weather
            "T_out_C": df_weather["T_out"].to_numpy(float),
            "DNI": df_weather["DNI"].to_numpy(float),
            "DHI": df_weather["DHI"].to_numpy(float),
            "GHI": df_weather["GHI"].to_numpy(float),

            # states / fluxes
            "Ti_C": Ti_h,
            "Tm_C": Tm_h,
            "Phi_s_W": Phi_s,
            "Phi_h_W": Phi_h,

            # optics
            "rho_g": rho_g,
            "I_other_eff_Wm2": I_other,
            "Esol_win_Wm2": Esol_win,
            "Einc_avg_Wm2": Einc_avg,

            # metadata (重复列，方便后续直接 groupby)
            "case": case_name,
            "variant": variant,      # baseline / best / ctrl
            "material": material,
            "wwr_scheme": wwr_scheme,
            "dN_m": x[0],
            "dS_m": x[1],
            "etaE_rad": x[2],
            "etaW_rad": x[3],
            "g_eq": g_eq,
            "A_win_total_m2": A_win_total,
        }
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"✅ wrote {out_csv} ({len(out)} rows)")


# -----------------------------
# main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weather_csv", type=str, default="")
    parser.add_argument("--summary_csv", type=str, default="")
    parser.add_argument("--out_dir", type=str, default=r"D:\ICM_RESULT\pro2")
    args = parser.parse_args()

    ROOT = Path(__file__).resolve().parents[1]  # -> D:\ICM_CODE

    weather_csv = Path(args.weather_csv) if args.weather_csv else (ROOT / "weather_helsinki_tmy_hourly.csv")
    if not weather_csv.exists():
        raise FileNotFoundError(f"weather_csv not found: {weather_csv}")

    summary_csv = Path(args.summary_csv) if args.summary_csv else (ROOT / "borealis_results_summary.csv")
    if not summary_csv.exists():
        print(f"⚠️ summary_csv not found: {summary_csv} (best_x 将退化为 baseline)")
        summary_df = None
    else:
        summary_df = pd.read_csv(summary_csv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- configs ----
    location = LocationConfig()
    df = load_weather_csv(str(weather_csv), tz=location.tz)

    optical_base = OpticalConfig()
    schedule = ScheduleConfig()
    seasons = BorealisSeasons()

    # 你现在 Borealis 光学（按你要求：bare=0.2 snow=0.7）
    bo_opt = BorealisOptical(
        k_other=0.35,
        V_sky=1.0,
        V_gr=1.0,
        rho_bare=0.20,
        rho_snow=0.70,
        snow_Ta_threshold_C=0.0,
    )

    # 你目前 main 里用过的两套热参数（如果你 main 改了，就把这里同步改一下）
    therm_brick = Borealis2R2C(
        Cm_J_per_K=6.0e8,
        Rim_K_per_W=1.6e-4,
    )
    therm_concrete = Borealis2R2C(
        Cm_J_per_K=1.2e9,
        Rim_K_per_W=9.0e-5,
    )

    building_fi = _make_building_fi()
    building_passive = _make_building_passive()

    # baseline（你定义的“无任何遮阳措施/百叶全开”）
    X_BASELINE = (0.0, 0.0, 1.57, 1.57)

    # g_eq：evaluate_borealis 里就是用 optical_base.tau_heat
    g_eq = float(getattr(optical_base, "tau_heat", 0.55))

    def find_best_x(case_name: str) -> Tuple[float, float, float, float]:
        if summary_df is None:
            return X_BASELINE
        sub = summary_df[summary_df["case"].astype(str) == case_name]
        if len(sub) == 0:
            return X_BASELINE
        # 优先用 x_best
        x = _parse_x(str(sub.iloc[0].get("x_best", "")))
        return x if x is not None else X_BASELINE

    # ---- runs to export ----
    runs: List[Dict[str, Any]] = [
        {
            "case": "FI_WWR + BRICK",
            "building": building_fi,
            "therm": therm_brick,
            "material": "brick_wall_hydronic",
            "wwr_scheme": "FINLAND",
            "export_baseline": True,
            "export_best": True,
        },
        {
            "case": "FI_WWR + CONCRETE",
            "building": building_fi,
            "therm": therm_concrete,
            "material": "concrete_wall_hydronic",
            "wwr_scheme": "FINLAND",
            "export_baseline": True,
            "export_best": True,
        },
        {
            "case": "CTRL: PASSIVE_WWR + NO_SHADING + BRICK_THERM",
            "building": building_passive,
            "therm": therm_brick,
            "material": "brick_wall_hydronic",
            "wwr_scheme": "PASSIVE",
            "export_baseline": True,
            "export_best": False,
        },
        {
            "case": "CTRL: PASSIVE_WWR + NO_SHADING + CONCRETE_THERM",
            "building": building_passive,
            "therm": therm_concrete,
            "material": "concrete_wall_hydronic",
            "wwr_scheme": "PASSIVE",
            "export_baseline": True,
            "export_best": False,
        },
    ]

    print("\n===== Generating timeseries CSVs =====\n")
    print(f"weather: {weather_csv}")
    print(f"out_dir: {out_dir}")
    if summary_df is not None:
        print(f"summary: {summary_csv}\n")

    for r in runs:
        case = r["case"]
        building = r["building"]
        therm = r["therm"]
        material = r["material"]
        wwr_scheme = r["wwr_scheme"]
        A_win_total = _get_total_window_area(building)

        # baseline
        if r["export_baseline"]:
            res_base = evaluate_borealis(
                df=df,
                location=location,
                building=building,
                optical_base=optical_base,
                schedule=schedule,
                bo_optical=bo_opt,
                bo_therm=therm,
                bo_seasons=seasons,
                x=X_BASELINE,
            )
            out_csv = out_dir / f"timeseries_{_slug(case)}_baseline.csv"
            _export_timeseries_csv(
                out_csv,
                df,
                res_base,
                case_name=case,
                variant="baseline",
                material=material,
                wwr_scheme=wwr_scheme,
                x=X_BASELINE,
                g_eq=g_eq,
                A_win_total=A_win_total,
                schedule=schedule,
            )

        # best (from summary)
        if r["export_best"]:
            best_x = find_best_x(case)
            res_best = evaluate_borealis(
                df=df,
                location=location,
                building=building,
                optical_base=optical_base,
                schedule=schedule,
                bo_optical=bo_opt,
                bo_therm=therm,
                bo_seasons=seasons,
                x=best_x,
            )
            out_csv = out_dir / f"timeseries_{_slug(case)}_best.csv"
            _export_timeseries_csv(
                out_csv,
                df,
                res_best,
                case_name=case,
                variant="best",
                material=material,
                wwr_scheme=wwr_scheme,
                x=best_x,
                g_eq=g_eq,
                A_win_total=A_win_total,
                schedule=schedule,
            )

    print("\n✅ Done. Now you should see timeseries_*.csv in your out_dir.\n")


if __name__ == "__main__":
    main()