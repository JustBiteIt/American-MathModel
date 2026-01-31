# main.py
# -*- coding: utf-8 -*-
"""
主程序（TMY 虚拟完整年：8760 小时连续索引）
-------------------------------------------------
你当前的 weather_singapore_hourly.csv 来自 PVGIS TMY：
- 行数为 8760（非闰年）
- month/day/hour 连续，但 year 是“拼月来源年”（2006/2013/2023...）
因此：
✅ 应把它当作“虚拟年”使用：重建连续时间索引后再仿真/优化
❌ 不要按真实年份切片（会只剩几个月）
❌ 不要对原始多年份索引 resample（会膨胀到十几万行）

功能：
1) 读取 weather_singapore_hourly.csv（含 DNI/DHI/GHI/T_out）
2) 强制取前 8760 行作为 TMY 全年
3) 重建连续索引：2023-01-01 起每小时一步（Asia/Singapore）
4) 调用优化模块求解 (dN, dS, etaE, etaW)
5) 输出结果到 D:\ICM_RESULT\pro1passive_shading
6) 输出图到 D:\ICM_PLOT\pro1passive_shading
7) 保存 baseline 与 best 两套时序 CSV（含 datetime，便于复现）

依赖：
- numpy, pandas, matplotlib
- 项目模块：config/data_io/optimize/objective 等
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    LocationConfig, BuildingConfig, OpticalConfig, ThermalConfig,
    ScheduleConfig, VisualConstraintConfig, DecisionBounds
)
from data_io import load_weather_csv
from optimize import optimize_with_scipy_de
from objective import baseline_savings


# =========================
# 0. 输出目录（固定到 D 盘）
# =========================
PLOT_ROOT = Path(r"D:\ICM_PLOT") / "pro1passive_shading"
RESULT_ROOT = Path(r"D:\ICM_RESULT") / "pro1passive_shading"

# =========================
# 1. 虚拟年设置
# =========================
VIRTUAL_YEAR = 2023          # 只是“标签”，用于生成连续索引
TMY_HOURS = 8760             # PVGIS TMY 默认非闰年


# -------------------------
# 工具函数：目录
# -------------------------
def _ensure_dir(p: Path) -> Path:
    """确保目录存在。"""
    p.mkdir(parents=True, exist_ok=True)
    return p


# -------------------------
# 工具函数：数据检查
# -------------------------
def _quick_stats(df: pd.DataFrame, tz: str) -> None:
    """打印数据概况（用于确认是否为连续 8760 小时虚拟年）。"""
    print("=== 数据检查（TMY 虚拟年）===")
    print(f"时区: {tz}")
    print(f"时间范围: {df.index.min()} -> {df.index.max()}")
    print(f"总小时数: {len(df)} (期望 {TMY_HOURS})")

    for col in ["DNI", "T_out"]:
        if col in df.columns:
            print(f"{col}(min/max): {df[col].min():.2f} / {df[col].max():.2f}")

    # 可选列
    for col in ["DHI", "GHI"]:
        if col in df.columns:
            print(f"{col}(min/max): {df[col].min():.2f} / {df[col].max():.2f}")

    # 连续性检查
    dt = df.index.to_series().diff().dropna()
    bad = int((dt != pd.Timedelta(hours=1)).sum())
    print(f"相邻时间差 != 1h 次数: {bad}")
    if bad > 0:
        idx_bad = dt[dt != pd.Timedelta(hours=1)].index[:8]
        print("异常示例(最多8个):", list(idx_bad))


# -------------------------
# 工具函数：将 TMY 数据重建为连续虚拟年
# -------------------------
def _prepare_tmy_virtual_year(df_raw: pd.DataFrame, tz: str, base_year: int = VIRTUAL_YEAR) -> pd.DataFrame:
    """
    输入 df_raw：load_weather_csv() 读入后的 DataFrame（DatetimeIndex 可能跨多年份）
    输出 df：长度 8760，索引为 base_year 的连续整点小时序列
    """
    if len(df_raw) < TMY_HOURS:
        raise ValueError(f"原始数据行数不足 {TMY_HOURS}：{len(df_raw)}。请确认你的 CSV 是全年 TMY。")

    # 1) 只取前 8760 行（你已确认你的 CSV 正好 8760 行；这里是防御性写法）
    df = df_raw.iloc[:TMY_HOURS].copy()

    # 2) 重建连续虚拟年索引（避免拼月年份造成切片/重采样错误）
    df = df.reset_index(drop=True)
    df.index = pd.date_range(
        f"{base_year}-01-01 00:00:00",
        periods=len(df),
        freq="1h",
        tz=tz
    )

    # 3) 清洗：辐照度不允许为负
    for col in ["DNI", "DHI", "GHI"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).clip(lower=0.0)

    # 4) 温度允许缺失但不建议；这里严格要求无 NaN
    df["T_out"] = pd.to_numeric(df["T_out"], errors="coerce")
    if df["T_out"].isna().any():
        n_bad = int(df["T_out"].isna().sum())
        raise ValueError(f"T_out 存在缺失值 {n_bad} 条，请检查输入 CSV。")

    return df


# -------------------------
# 工具函数：保存时序 CSV（含 datetime）
# -------------------------
def _save_series_csv(result_dir: Path, times: pd.DatetimeIndex, series: dict, prefix: str) -> None:
    """
    保存关键时序到 CSV（包含 datetime，便于复现与论文作图）。
    - times: 长度 n
    - T_C: 长度 n+1（末端补 1 小时）
    """
    n = len(times)
    times_plus1 = times.append(pd.DatetimeIndex([times[-1] + pd.Timedelta(hours=1)]))

    out_T = pd.DataFrame({
        "datetime": times_plus1.astype(str),
        "T_C": series["T_C"],
    })
    out_T.to_csv(result_dir / f"{prefix}_series_temperature.csv", index=False, encoding="utf-8-sig")

    out = pd.DataFrame({
        "datetime": times.astype(str),
        "Q_cool_W": series["Q_cool_W"],
        "L_kWh": series["L_kWh"],
        "E_vis_Wm2": series["E_vis_Wm2"],
    })
    out.to_csv(result_dir / f"{prefix}_series_cooling_visual.csv", index=False, encoding="utf-8-sig")


# -------------------------
# 工具函数：画图（用真实时间轴）
# -------------------------
def _plot_key_figures(plot_dir: Path, df: pd.DataFrame, base: dict, best: dict, T_set: float = 26.0) -> None:
    """
    生成基础可视化：
    1) DNI 与室外温度（前 14 天）
    2) baseline vs best 室内温度对比（前 14 天）
    3) baseline vs best 制冷功率对比（前 14 天）
    4) 全年：每日制冷量（baseline vs best）对比
    """
    # -------- 前 14 天示例 --------
    n_show = min(len(df), 24 * 14)
    t = df.index[:n_show]

    # 1) 天气驱动
    fig = plt.figure()
    plt.plot(t, df["DNI"].to_numpy()[:n_show], label="DNI (W/m^2)")
    plt.plot(t, df["T_out"].to_numpy()[:n_show], label="T_out (°C)")
    plt.xlabel("Local time (Asia/Singapore)")
    plt.legend()
    plt.title("Weather drivers (first 14 days, TMY virtual year)")
    plt.gcf().autofmt_xdate()
    fig.savefig(plot_dir / "weather_drivers_first14days.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 2) 室内温度对比（T_C 长度 n+1）
    T_base = base["series"]["T_C"]
    T_best = best["series"]["T_C"]
    t_T = df.index[:min(len(df), n_show)].append(pd.DatetimeIndex([df.index[min(len(df), n_show)-1] + pd.Timedelta(hours=1)]))
    # 保险：取同样长度
    m = min(len(t_T), len(T_base), len(T_best))
    t_T = t_T[:m]
    T_base = T_base[:m]
    T_best = T_best[:m]

    fig = plt.figure()
    plt.plot(t_T, T_base, label="Baseline indoor T (°C)")
    plt.plot(t_T, T_best, label="Best indoor T (°C)")
    plt.axhline(T_set, linestyle="--", label=f"{T_set}°C cap")
    plt.xlabel("Local time (Asia/Singapore)")
    plt.legend()
    plt.title("Indoor temperature: baseline vs best (first 14 days)")
    plt.gcf().autofmt_xdate()
    fig.savefig(plot_dir / "indoor_temperature_baseline_vs_best_first14days.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 3) 制冷功率对比（长度 n）
    Qb = base["series"]["Q_cool_W"][:n_show]
    Qk = best["series"]["Q_cool_W"][:n_show]
    fig = plt.figure()
    plt.plot(t, Qb, label="Baseline Q_cool (W)")
    plt.plot(t, Qk, label="Best Q_cool (W)")
    plt.xlabel("Local time (Asia/Singapore)")
    plt.legend()
    plt.title("Cooling power: baseline vs best (first 14 days)")
    plt.gcf().autofmt_xdate()
    fig.savefig(plot_dir / "cooling_power_baseline_vs_best_first14days.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # -------- 全年：每日制冷量 --------
    # 用 best/base 的 L_kWh 聚合到日
    L_base = pd.Series(base["series"]["L_kWh"], index=df.index)
    L_best = pd.Series(best["series"]["L_kWh"], index=df.index)

    daily_base = L_base.resample("1D").sum()
    daily_best = L_best.resample("1D").sum()

    fig = plt.figure()
    plt.plot(daily_base.index, daily_base.to_numpy(), label="Baseline daily cooling (kWh/day)")
    plt.plot(daily_best.index, daily_best.to_numpy(), label="Best daily cooling (kWh/day)")
    plt.xlabel("Local date")
    plt.legend()
    plt.title("Daily cooling energy: baseline vs best (TMY virtual year)")
    plt.gcf().autofmt_xdate()
    fig.savefig(plot_dir / "daily_cooling_energy_baseline_vs_best.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# -------------------------
# 工具函数：写文本报告
# -------------------------
def _write_report_txt(result_dir: Path, base: dict, best: dict, report: dict) -> None:
    """写一个简洁的文本报告，便于你论文引用与复核。"""
    lines = []
    lines.append("=== Passive Shading Optimization Report (TMY virtual year) ===")
    lines.append("")

    lines.append("[Baseline]")
    lines.append(f"  x = {base['x']}  # (dN, dS, etaE, etaW)")
    lines.append(f"  feasible = {base['feasible']}")
    lines.append(f"  L_year_kWh = {base['L_year_kWh']:.6f}")
    lines.append(f"  vis = {base['vis']}")
    lines.append("")

    lines.append("[Best]")
    lines.append(f"  x = {best['x']}  # (dN, dS, etaE, etaW)")
    lines.append(f"  feasible = {best['feasible']}")
    lines.append(f"  L_year_kWh = {best['L_year_kWh']:.6f}")
    lines.append(f"  vis = {best['vis']}")
    lines.append("")

    lines.append("[Savings vs Baseline]")
    for k, v in report.items():
        if isinstance(v, float):
            lines.append(f"  {k}: {v:.6f}")
        else:
            lines.append(f"  {k}: {v}")

    (result_dir / "report.txt").write_text("\n".join(lines), encoding="utf-8")


# -------------------------
# 主程序
# -------------------------
def main():
    # ---------- 1) 配置 ----------
    location = LocationConfig()
    building = BuildingConfig()
    optical = OpticalConfig()
    thermal = ThermalConfig()
    schedule = ScheduleConfig()
    visual = VisualConstraintConfig()
    bounds = DecisionBounds()

    # ---------- 2) 输入数据路径（优先 D:\ICM_CODE，其次脚本同目录） ----------
    base_dir = Path(__file__).resolve().parent
    candidates = [
        Path(r"D:\ICM_CODE\weather_singapore_hourly.csv"),
        base_dir / "weather_singapore_hourly.csv",
    ]
    weather_path = None
    for p in candidates:
        if p.exists():
            weather_path = p
            break

    if weather_path is None:
        raise FileNotFoundError(
            "找不到天气文件 weather_singapore_hourly.csv。\n"
            "请确认以下任一路径存在：\n" + "\n".join([f"  - {c}" for c in candidates])
        )

    # ---------- 3) 读取并构造 TMY 虚拟年 ----------
    df_raw = load_weather_csv(str(weather_path), tz=location.tz)
    df = _prepare_tmy_virtual_year(df_raw, tz=location.tz, base_year=VIRTUAL_YEAR)

    _quick_stats(df, tz=location.tz)

    # ---------- 4) 输出目录 ----------
    plot_dir = _ensure_dir(PLOT_ROOT)
    result_dir = _ensure_dir(RESULT_ROOT)

    print(f"输出目录(Plot): {plot_dir}")
    print(f"输出目录(Result): {result_dir}")

    # ---------- 5) 优化 ----------
    print("=== 开始优化（Differential Evolution）===")
    out = optimize_with_scipy_de(df, location, building, optical, thermal, schedule, visual, bounds)
    print("=== 优化结束 ===")

    baseline = out["baseline"]
    best = out["best"]

    # ---------- 6) 结果汇总 ----------
    base_L = baseline["L_year_kWh"]
    best_L = best["L_year_kWh"]
    report = baseline_savings(base_L, best_L)

    print("\n=== Baseline vs Best (kWh) ===")
    for k, v in report.items():
        print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")

    print("\n=== Best x = (dN, dS, etaE, etaW) ===")
    print(best["x"])
    print("\n=== Visual constraints stats ===")
    print(best["vis"])
    print("\n=== Feasible ===")
    print(best["feasible"])

    # ---------- 7) 写文件（CSV + 报告 + 图） ----------
    # 保存 baseline & best 两套时序（带 datetime）
    _save_series_csv(result_dir, df.index, baseline["series"], prefix="baseline")
    _save_series_csv(result_dir, df.index, best["series"], prefix="best")

    # 报告
    _write_report_txt(result_dir, baseline, best, report)

    # 图
    _plot_key_figures(plot_dir, df, baseline, best, T_set=thermal.T_set_C)

    print("\n=== 已完成输出 ===")
    print("图像目录:", plot_dir)
    print("结果目录:", result_dir)
    print("报告文件:", result_dir / "report.txt")


if __name__ == "__main__":
    main()
