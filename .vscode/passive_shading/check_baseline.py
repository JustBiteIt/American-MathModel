# check_baseline.py
# -*- coding: utf-8 -*-
"""
一键“统一 baseline 并验证”
-------------------------------------------------
运行后会：
1) 读取 weather_singapore_hourly.csv
2) 重建 TMY 虚拟年（连续 8760 小时）
3) 计算 baseline（统一定义）
4) 打印 df 指纹 + baseline 结果
5) 输出 baseline 的时序 CSV + baseline_report.txt 到 D:\ICM_RESULT\pro1passive_shading
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from config import (
    LocationConfig, BuildingConfig, OpticalConfig, ThermalConfig,
    ScheduleConfig, VisualConstraintConfig, DecisionBounds
)
from data_io import load_weather_csv
from optimize import evaluate_design

from baseline_utils import (
    find_weather_csv,
    build_tmy_virtual_year,
    baseline_x_from_bounds,
    df_fingerprint,
)


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_baseline_outputs(result_dir: Path, df_tmy: pd.DataFrame, baseline_res: dict) -> None:
    """
    输出 baseline 的关键时序：
      - baseline_series_temperature.csv
      - baseline_series_cooling_visual.csv
      - baseline_report.txt
    """
    series = baseline_res["series"]

    # 温度：T_C 长度 n+1，所以给它配一个 n+1 的时间轴
    t_hour = df_tmy.index
    t_temp = t_hour.append(pd.DatetimeIndex([t_hour[-1] + pd.Timedelta(hours=1)]))

    temp_df = pd.DataFrame({
        "datetime": t_temp.astype(str),
        "T_C": series["T_C"],
    })
    temp_df.to_csv(result_dir / "baseline_series_temperature.csv", index=False, encoding="utf-8-sig")

    # 其它：长度 n
    other_df = pd.DataFrame({
        "datetime": t_hour.astype(str),
        "Q_cool_W": series["Q_cool_W"],
        "L_kWh": series["L_kWh"],
        "E_vis_Wm2": series["E_vis_Wm2"],
    })
    # 如果你在 evaluate_design 里输出了更多列（I_sky/I_ground/...），这里也一起带上
    for k in ["I_sky_Wm2", "I_ground_Wm2", "I_other_Wm2", "u_N", "u_E", "u_S", "u_W",
              "I_beam_N_Wm2", "I_beam_E_Wm2", "I_beam_S_Wm2", "I_beam_W_Wm2"]:
        if k in series:
            other_df[k] = series[k]

    other_df.to_csv(result_dir / "baseline_series_cooling_visual.csv", index=False, encoding="utf-8-sig")

    # 报告
    lines = []
    lines.append("=== BASELINE (Unified) ===")
    lines.append(f"x = {baseline_res['x']}")
    lines.append(f"L_year_kWh = {baseline_res['L_year_kWh']:.6f}")
    lines.append(f"feasible = {baseline_res['feasible']}")
    lines.append(f"vis = {baseline_res['vis']}")
    (result_dir / "baseline_report.txt").write_text("\n".join(lines), encoding="utf-8")


def main():
    script_dir = Path(__file__).resolve().parent

    # configs
    loc = LocationConfig()
    bld = BuildingConfig()
    opt = OpticalConfig()
    th = ThermalConfig()
    sch = ScheduleConfig()
    vis = VisualConstraintConfig()
    bounds = DecisionBounds()

    # data
    weather_path = find_weather_csv(script_dir)
    print("[INFO] weather_path =", weather_path)

    df_raw = load_weather_csv(str(weather_path), tz=loc.tz)

    # 统一口径：TMY 虚拟年（连续 8760 小时）
    df_tmy = build_tmy_virtual_year(df_raw, tz=loc.tz, target_year=2023, freq="1h", strict=True)

    # 指纹（非常重要：以后所有脚本都要打印它，确保同一份 df_tmy）
    fp = df_fingerprint(df_tmy)

    print("=== 数据检查（Unified TMY Virtual Year）===")
    print("时区:", loc.tz)
    print("时间范围:", df_tmy.index.min(), "->", df_tmy.index.max())
    print("总小时数:", len(df_tmy), "(期望 8760)")
    print("fingerprint:", fp)
    print("DNI(min/max):", float(df_tmy["DNI"].min()), "/", float(df_tmy["DNI"].max()))
    print("T_out(min/max):", float(df_tmy["T_out"].min()), "/", float(df_tmy["T_out"].max()))
    if "DHI" in df_tmy.columns:
        print("DHI(min/max):", float(df_tmy["DHI"].min()), "/", float(df_tmy["DHI"].max()))
    if "GHI" in df_tmy.columns:
        print("GHI(min/max):", float(df_tmy["GHI"].min()), "/", float(df_tmy["GHI"].max()))

    # baseline（统一定义）
    baseline_x = baseline_x_from_bounds(bounds)
    baseline_res = evaluate_design(df_tmy, loc, bld, opt, th, sch, vis, baseline_x)

    print("\n=== BASELINE（统一口径）===")
    print("baseline_x =", baseline_x)
    print("L_year_kWh =", baseline_res["L_year_kWh"])
    print("vis =", baseline_res["vis"])
    print("feasible =", baseline_res["feasible"])

    # 输出
    result_dir = ensure_dir(Path(r"D:\ICM_RESULT") / "pro1passive_shading")
    save_baseline_outputs(result_dir, df_tmy, baseline_res)
    print("\n[OK] baseline 输出完成：", result_dir)
    print("  - baseline_report.txt")
    print("  - baseline_series_temperature.csv")
    print("  - baseline_series_cooling_visual.csv")


if __name__ == "__main__":
    main()
