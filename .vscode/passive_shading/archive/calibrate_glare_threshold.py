# calibrate_glare_threshold.py
# -*- coding: utf-8 -*-
"""
用途：标定 VisualConstraintConfig 里的 E_min / E_glare（基于 baseline 分布）
-----------------------------------------------------------------------
你现在的数据是 PVGIS TMY：8760 行，但年份是“拼凑的”（2006/2007/2013/...）
所以不能用真实年份去切片；必须构造一个“虚拟连续年时间轴”做统计。

本脚本做的事：
1) 读取 weather_singapore_hourly.csv（需要列：datetime, DNI, DHI, GHI, T_out）
2) 构造 target_year 的连续 8760 小时索引（默认 2023）
3) 运行 baseline 设计（无遮阳）得到 E_vis
4) 在 day_occ（白天&占用）样本上：
   - 标定 E_min：让 p_daylight ≈ p_min
   - 标定 E_glare：让 H_glare ≈ H_glare_max（尽量不超过）

运行方式（建议在 passive_shading 目录下）：
  python calibrate_glare_threshold.py

可选参数：
  python calibrate_glare_threshold.py --H_glare_max 183
  python calibrate_glare_threshold.py --p_min 0.80
  python calibrate_glare_threshold.py --target_year 2023
  python calibrate_glare_threshold.py --weather_path D:\\ICM_CODE\\weather_singapore_hourly.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from config import (
    LocationConfig,
    BuildingConfig,
    OpticalConfig,
    ThermalConfig,
    ScheduleConfig,
    VisualConstraintConfig,
    DecisionBounds,
)
from data_io import load_weather_csv
from optimize import evaluate_design
from solar import solar_position


# ---------------------------
# 路径工具：自动定位项目根
# ---------------------------
def locate_project_root(script_dir: Path) -> Path:
    """
    从脚本目录向上找 ICM_CODE 作为根目录；找不到就退回到父目录。
    """
    for p in script_dir.parents:
        if p.name.upper() == "ICM_CODE":
            return p
    return script_dir.parent


def find_weather_csv(script_dir: Path, project_root: Path, user_path: str | None) -> Path:
    """
    输入文件查找策略：
    1) 如果用户指定 --weather_path，优先用它
    2) 否则依次尝试：
       - 脚本同目录
       - 项目根目录
       - D:\ICM_CODE\
    """
    if user_path:
        p = Path(user_path)
        if not p.exists():
            raise FileNotFoundError(f"你指定的 weather_path 不存在：{p}")
        return p

    candidates = [
        script_dir / "weather_singapore_hourly.csv",
        project_root / "weather_singapore_hourly.csv",
        Path(r"D:\ICM_CODE\weather_singapore_hourly.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        "找不到 weather_singapore_hourly.csv。\n"
        "请把文件放到以下任一位置，或用 --weather_path 指定绝对路径：\n"
        + "\n".join([f"  - {c}" for c in candidates])
    )


# ---------------------------
# 核心：构造 TMY 虚拟年
# ---------------------------
def build_tmy_virtual_year(df_raw: pd.DataFrame, tz: str, target_year: int = 2023) -> pd.DataFrame:
    """
    把“拼年份的 TMY”映射为 target_year 的连续 8760 小时。
    关键点：按行顺序映射，不按真实年份切片。
    """
    if not isinstance(df_raw.index, pd.DatetimeIndex):
        raise TypeError("df_raw.index 不是 DatetimeIndex（请用 load_weather_csv 读取）")

    # 统一到本地时区
    if df_raw.index.tz is None:
        df_raw = df_raw.copy()
        df_raw.index = df_raw.index.tz_localize(tz)
    else:
        df_raw = df_raw.tz_convert(tz)

    # 按时间排序（通常月序会正确），但更重要的是：总行数应为 8760
    df = df_raw.sort_index().copy()

    # 截断/提醒
    if len(df) > 8760:
        df = df.iloc[:8760].copy()
    if len(df) < 8760:
        print(f"[WARN] 行数 {len(df)} < 8760：这不是完整全年，标定结果需谨慎。")

    # 构造 target_year 的连续小时索引
    idx = pd.date_range(
        start=pd.Timestamp(f"{target_year}-01-01 00:00:00", tz=tz),
        periods=len(df),
        freq="1h",
    )
    df.index = idx
    return df


# ---------------------------
# 统计 day_occ 样本（白天&占用）
# ---------------------------
def make_day_occ_mask(df: pd.DataFrame, loc: LocationConfig, sched: ScheduleConfig) -> np.ndarray:
    """
    白天判定：太阳高度角 alpha>0 且 (DNI/DHI/GHI 至少一个>0)
    占用判定：小时在 [occ_start, occ_end)
    """
    times = df.index
    alpha, _psi = solar_position(times, loc.latitude, loc.longitude)

    DNI = df["DNI"].to_numpy(dtype=float)
    DHI = df["DHI"].to_numpy(dtype=float) if "DHI" in df.columns else np.zeros_like(DNI)
    GHI = df["GHI"].to_numpy(dtype=float) if "GHI" in df.columns else np.zeros_like(DNI)

    day = (alpha > 0.0) & ((DNI > 0.0) | (DHI > 0.0) | (GHI > 0.0))
    hours = np.array([t.hour for t in times])
    occ = (hours >= sched.occ_start_hour) & (hours < sched.occ_end_hour)

    return day & occ


# ---------------------------
# 标定：E_min（采光下限）
# ---------------------------
def calibrate_E_min(E: np.ndarray, p_min: float) -> tuple[float, float]:
    """
    令 p_daylight = mean(E >= E_min) ≈ p_min
    取 E_min 为 (1 - p_min) 分位数
    返回：E_min, 实际 p_daylight
    """
    E = np.asarray(E, dtype=float)
    E = E[np.isfinite(E)]
    if len(E) == 0:
        return float("nan"), 0.0

    q = max(0.0, min(1.0, 1.0 - float(p_min)))
    E_min = float(np.quantile(E, q))
    p = float(np.mean(E >= E_min))
    return E_min, p


# ---------------------------
# 标定：E_glare（眩光阈值）
# ---------------------------
def calibrate_E_glare(E: np.ndarray, H_glare_max: int) -> tuple[float, int]:
    """
    令 H_glare = count(E >= E_glare) 不超过 H_glare_max，且尽量接近它。
    返回：E_glare, 实际 H_glare
    """
    E = np.asarray(E, dtype=float)
    E = E[np.isfinite(E)]
    if len(E) == 0:
        return float("inf"), 0

    H_target = int(H_glare_max)
    if H_target <= 0:
        return float(np.nextafter(np.max(E), np.inf)), 0

    # 候选阈值取唯一值（升序）
    candidates = np.unique(np.sort(E))

    best_thr = None
    best_cnt = -1

    # 遍历候选阈值，找“cnt <= H_target 且 cnt 最大”的那个（最接近上限但不超）
    for thr in candidates:
        cnt = int(np.sum(E >= thr))
        if cnt <= H_target and cnt > best_cnt:
            best_cnt = cnt
            best_thr = float(thr)

    # 如果连最大值都出现次数 > H_target，那么只能把阈值设到 >max，得到 0 次眩光
    if best_thr is None:
        best_thr = float(np.nextafter(np.max(E), np.inf))
        best_cnt = 0

    return best_thr, best_cnt


def print_quantiles(E: np.ndarray) -> None:
    E = np.asarray(E, dtype=float)
    E = E[np.isfinite(E)]
    if len(E) == 0:
        print("[WARN] E_vis 全是 NaN，无法给分位数。")
        return

    qs = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99]
    print("\n--- E_vis 在 day_occ 下的分位数（W/m^2）---")
    for q in qs:
        print(f"q={q:>4}: {float(np.quantile(E, q)):.6f}")
    print(f"max: {float(np.max(E)):.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weather_path", type=str, default=None, help="weather_singapore_hourly.csv 的路径（可选）")
    parser.add_argument("--target_year", type=int, default=2023, help="虚拟年年份（默认 2023）")
    parser.add_argument("--H_glare_max", type=int, default=None, help="覆盖 config.H_glare_max（可选）")
    parser.add_argument("--p_min", type=float, default=None, help="覆盖 config.p_min（可选）")
    args = parser.parse_args()

    # 配置
    loc = LocationConfig()
    bld = BuildingConfig()
    opt = OpticalConfig()
    th = ThermalConfig()
    sc = ScheduleConfig()
    vc = VisualConstraintConfig()
    bounds = DecisionBounds()

    if args.H_glare_max is not None:
        H_glare_max = int(args.H_glare_max)
    else:
        H_glare_max = int(vc.H_glare_max)

    if args.p_min is not None:
        p_min = float(args.p_min)
    else:
        p_min = float(vc.p_min)

    # 路径
    script_dir = Path(__file__).resolve().parent
    project_root = locate_project_root(script_dir)
    weather_path = find_weather_csv(script_dir, project_root, args.weather_path)

    print("[INFO] script_dir   =", script_dir)
    print("[INFO] project_root =", project_root)
    print("[INFO] weather_path =", weather_path)

    # 读数据（注意：不要 resample 到真实时间轴，否则会跨多年出现巨大空洞）
    df_raw = load_weather_csv(str(weather_path), tz=loc.tz)

    # 检查列
    need_cols = ["DNI", "T_out", "DHI", "GHI"]
    missing = [c for c in need_cols if c not in df_raw.columns]
    if missing:
        raise ValueError(
            f"weather 文件缺列：{missing}\n"
            f"当前列名：{list(df_raw.columns)}\n"
            "请确认你的 weather_singapore_hourly.csv 包含 DNI/DHI/GHI/T_out。"
        )

    # 构造虚拟年
    df = build_tmy_virtual_year(df_raw, tz=loc.tz, target_year=args.target_year)

    # baseline（“最接近无遮阳”的参数取当前 bounds 上界）
    baseline_x = (0.0, 0.0, bounds.etaE_bounds_rad[1], bounds.etaW_bounds_rad[1])
    res = evaluate_design(df, loc, bld, opt, th, sc, vc, baseline_x)

    E_vis = res["series"]["E_vis_Wm2"]

    # day_occ 样本
    day_occ = make_day_occ_mask(df, loc, sc)
    E = np.asarray(E_vis, dtype=float)[day_occ]

    print("\n=== 标定数据概况（TMY 虚拟年）===")
    print(f"虚拟年时间范围: {df.index.min()} -> {df.index.max()}")
    print(f"总小时数: {len(df)}")
    print(f"day_occ 样本数 = {int(np.sum(day_occ))}（白天&占用小时）")

    print_quantiles(E)

    # 1) 标定 E_min
    E_min_suggest, p_actual = calibrate_E_min(E, p_min=p_min)

    # 2) 标定 E_glare
    E_glare_suggest, H_actual = calibrate_E_glare(E, H_glare_max=H_glare_max)

    print("\n=== 推荐阈值（用于修改 config.py）===")
    print(f"当前 p_min = {p_min:.3f}")
    print(f"建议 E_min = {E_min_suggest:.6f}  （采用后 baseline 下 p_daylight ≈ {p_actual:.6f}）")
    print("")
    print(f"当前 H_glare_max = {H_glare_max:d}")
    print(f"建议 E_glare = {E_glare_suggest:.6f}（采用后 baseline 下 H_glare = {H_actual:d}，应 <= {H_glare_max:d}）")

    print("\n下一步：打开 config.py，把 VisualConstraintConfig 里的 E_min / E_glare 改为上面建议值。")


if __name__ == "__main__":
    main()
