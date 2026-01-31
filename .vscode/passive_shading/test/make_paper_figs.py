# test/plot_paper_figs.py
# -*- coding: utf-8 -*-
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# =========================
# 你需要根据自己情况改的几个参数
# =========================
RESULT_ROOT = Path(r"D:\ICM_RESULT\pro1passive_shading")
PLOT_ROOT   = Path(r"D:\ICM_PLOT\pro1passive_shading_paper")
WEATHER_CSV = Path(r"D:\ICM_CODE\weather_singapore_hourly.csv")

TZ = "Asia/Singapore"
VIRTUAL_YEAR = 2023
TMY_HOURS = 8760

# 论文里建议用“占用+白天”作为视觉统计口径（你代码里的 day_occ）
OCC_START_HOUR = 8
OCC_END_HOUR   = 18

# 这两个阈值请与你 config.py 的 VisualConstraintConfig 一致
# （你之前图里大概是这些数，你自己确认一下）
E_MIN   = 7.8922
E_GLARE = 108.5986

# 图的输出 dpi（论文建议 300+）
DPI = 350


# =========================
# 工具函数：读你保存的 series CSV
# =========================
def read_series(prefix: str) -> pd.DataFrame:
    p = RESULT_ROOT / f"{prefix}_series_cooling_visual.csv"
    if not p.exists():
        raise FileNotFoundError(f"找不到：{p}")

    df = pd.read_csv(p)
    df["datetime"] = pd.to_datetime(df["datetime"])

    # 处理时区
    if df["datetime"].dt.tz is None:
        df["datetime"] = df["datetime"].dt.tz_localize(TZ)
    else:
        df["datetime"] = df["datetime"].dt.tz_convert(TZ)

    df = df.set_index("datetime").sort_index()
    return df


# =========================
# 工具函数：读天气（并“虚拟年”对齐到 2023 连续 8760h）
# =========================
def read_weather_tmy(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到天气文件：{path}")

    w = pd.read_csv(path)
    if "datetime" not in w.columns:
        raise ValueError("天气 CSV 缺少 datetime 列")
    w["datetime"] = pd.to_datetime(w["datetime"])

    if w["datetime"].dt.tz is None:
        w["datetime"] = w["datetime"].dt.tz_localize(TZ)
    else:
        w["datetime"] = w["datetime"].dt.tz_convert(TZ)

    w = w.set_index("datetime").sort_index()

    # 取前 8760 行，并重建连续虚拟年索引（跟你 main.py 一致的口径）
    if len(w) < TMY_HOURS:
        raise ValueError(f"天气行数不足 {TMY_HOURS}：{len(w)}")

    w = w.iloc[:TMY_HOURS].copy()
    w = w.reset_index(drop=True)
    w.index = pd.date_range(
        f"{VIRTUAL_YEAR}-01-01 00:00:00",
        periods=len(w),
        freq="1h",
        tz=TZ
    )

    # 确保辐照度列存在
    for c in ["DNI", "DHI", "GHI"]:
        if c not in w.columns:
            raise ValueError(f"天气 CSV 缺少列：{c}")
        w[c] = pd.to_numeric(w[c], errors="coerce").fillna(0.0).clip(lower=0.0)

    return w


def make_day_occ_mask(times: pd.DatetimeIndex, weather: pd.DataFrame) -> pd.Series:
    """不依赖 solar alpha，直接用 (DNI/DHI/GHI 有能量) 来判定白天，和你现在 sunlit_mask 的新口径一致。"""
    # 白天：任一辐照度 > 0
    day = (weather["DNI"] > 0) | (weather["DHI"] > 0) | (weather["GHI"] > 0)

    # 占用：小时在 [start, end)
    h = times.hour
    occ = (h >= OCC_START_HOUR) & (h < OCC_END_HOUR)

    return (day.reindex(times).fillna(False) & occ)


# =========================
# ECharts 导出：时间序列转成 [ms, value] 格式
# =========================
def to_echarts_series(df: pd.DataFrame, col: str) -> list:
    # ECharts 常用毫秒时间戳
    ms = (df.index.tz_convert("UTC").view("int64") // 10**6).astype(np.int64)
    vals = df[col].to_numpy()
    return [[int(t), float(v)] for t, v in zip(ms, vals)]


def save_echarts_payload(out_dir: Path, payload: dict, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{name}.json"
    p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print("Saved ECharts JSON:", p)


# =========================
# 画图（论文友好版）
# =========================
def set_pub_style():
    plt.rcParams.update({
        "figure.figsize": (9, 4.8),
        "figure.dpi": 120,
        "savefig.dpi": DPI,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def fig1_daily_cooling(plot_dir: Path, no_shading: pd.DataFrame, optimized: pd.DataFrame):
    # 日总制冷量
    d0 = no_shading["L_kWh"].resample("1D").sum()
    d1 = optimized["L_kWh"].resample("1D").sum()

    fig = plt.figure()
    plt.plot(d0.index, d0.values, label="No shading (daily cooling)")
    plt.plot(d1.index, d1.values, label="Optimized shading (daily cooling)")
    plt.ylabel("Cooling energy (kWh/day)")
    plt.xlabel("Date")
    plt.title("Daily cooling energy (TMY virtual year)")
    plt.legend(frameon=True)

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    fig.tight_layout()
    fig.savefig(plot_dir / "fig1_daily_cooling.png")
    plt.close(fig)


def fig2_evis_vs_dni_14days(plot_dir: Path, no_shading: pd.DataFrame, optimized: pd.DataFrame, weather: pd.DataFrame):
    n = 24 * 14
    t = no_shading.index[:n]

    e0 = no_shading.loc[t, "E_vis_Wm2"]
    e1 = optimized.loc[t, "E_vis_Wm2"]
    dni = weather.loc[t, "DNI"]

    fig = plt.figure()
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    ax1.plot(t, e0.values, label="No shading: $E_{vis}$")
    ax1.plot(t, e1.values, label="Optimized shading: $E_{vis}$")
    ax2.plot(t, dni.values, label="DNI", linestyle="--")

    ax1.set_ylabel(r"$E_{vis}$ (W/m$^2$)")
    ax2.set_ylabel(r"DNI (W/m$^2$)")
    ax1.set_xlabel("Local time")

    ax1.set_title(r"$E_{vis}$ vs DNI (first 14 days)")
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b-%d"))

    # 合并双轴图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=True, loc="upper right")

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(plot_dir / "fig2_evis_vs_dni_first14days.png")
    plt.close(fig)


def fig3_evis_hist_dayocc(plot_dir: Path, no_shading: pd.DataFrame, optimized: pd.DataFrame, day_occ: pd.Series):
    # 关键：只统计 day_occ，去掉夜间大量 0 值，左边怪柱子就消失/显著变小
    e0 = no_shading.loc[day_occ, "E_vis_Wm2"].astype(float)
    e1 = optimized.loc[day_occ, "E_vis_Wm2"].astype(float)

    # 再做一个很小的阈值过滤，防止“接近0”的残留占太多视觉面积（可选）
    eps = 1e-6
    e0 = e0[e0 > eps]
    e1 = e1[e1 > eps]

    bins = np.linspace(0, max(e0.max(), e1.max(), E_GLARE) * 1.05, 45)

    fig = plt.figure()
    plt.hist(e0.values, bins=bins, alpha=0.6, density=True, label="No shading")
    plt.hist(e1.values, bins=bins, alpha=0.6, density=True, label="Optimized shading")

    plt.axvline(E_MIN, linestyle="--", label=f"$E_{{min}}$ = {E_MIN:.2f}")
    plt.axvline(E_GLARE, linestyle="--", label=f"$E_{{glare}}$ = {E_GLARE:.2f}")

    plt.xlabel(r"$E_{vis}$ (W/m$^2$)")
    plt.ylabel("Probability density")
    plt.title(r"Distribution of $E_{vis}$ during occupied daytime")
    plt.legend(frameon=True)

    fig.tight_layout()
    fig.savefig(plot_dir / "fig3_evis_hist_dayocc.png")
    plt.close(fig)


def fig4_monthly_cooling_bar(plot_dir: Path, no_shading: pd.DataFrame, optimized: pd.DataFrame):
    # 每月总制冷量（论文正文更好放）
    m0 = no_shading["L_kWh"].resample("MS").sum()
    m1 = optimized["L_kWh"].resample("MS").sum()

    fig = plt.figure(figsize=(9, 4.2))
    x = np.arange(len(m0))
    width = 0.42

    plt.bar(x - width/2, m0.values, width=width, label="No shading")
    plt.bar(x + width/2, m1.values, width=width, label="Optimized shading")

    plt.xticks(x, [d.strftime("%b") for d in m0.index])
    plt.ylabel("Cooling energy (kWh/month)")
    plt.xlabel("Month")
    plt.title("Monthly cooling energy (TMY virtual year)")
    plt.legend(frameon=True)

    fig.tight_layout()
    fig.savefig(plot_dir / "fig4_monthly_cooling_bar.png")
    plt.close(fig)


def main():
    PLOT_ROOT.mkdir(parents=True, exist_ok=True)
    set_pub_style()

    # 1) 读结果
    no_shading = read_series("baseline")  # 你这份 baseline 本质就是“无遮阳”
    optimized = read_series("best")

    # 2) 读天气（用于 DNI + day_occ）
    weather = read_weather_tmy(WEATHER_CSV)

    # 对齐检查
    if len(weather) != len(no_shading):
        print("WARNING: weather and series length differ:", len(weather), len(no_shading))

    # 3) day_occ mask（修直方图最关键）
    day_occ = make_day_occ_mask(no_shading.index, weather)

    # 4) 画 4 张图
    fig1_daily_cooling(PLOT_ROOT, no_shading, optimized)
    fig2_evis_vs_dni_14days(PLOT_ROOT, no_shading, optimized, weather)
    fig3_evis_hist_dayocc(PLOT_ROOT, no_shading, optimized, day_occ)
    fig4_monthly_cooling_bar(PLOT_ROOT, no_shading, optimized)

    print("✅ Saved figures to:", PLOT_ROOT)

    # 5) 导出给 ECharts 用的数据（JSON）
    echarts_dir = RESULT_ROOT / "echarts_payload"
    payload = {
        "tz": TZ,
        "series": {
            "cooling_hourly_no_shading_kWh": to_echarts_series(no_shading, "L_kWh"),
            "cooling_hourly_optimized_kWh": to_echarts_series(optimized, "L_kWh"),
            "evis_no_shading": to_echarts_series(no_shading, "E_vis_Wm2"),
            "evis_optimized": to_echarts_series(optimized, "E_vis_Wm2"),
            "dni": to_echarts_series(weather.reindex(no_shading.index), "DNI"),
        },
        "thresholds": {"E_min": E_MIN, "E_glare": E_GLARE},
        "meta": {"occ_start": OCC_START_HOUR, "occ_end": OCC_END_HOUR, "virtual_year": VIRTUAL_YEAR},
    }
    save_echarts_payload(echarts_dir, payload, "paper_fig_data")

    print("✅ Saved ECharts payload to:", echarts_dir)


if __name__ == "__main__":
    main()
