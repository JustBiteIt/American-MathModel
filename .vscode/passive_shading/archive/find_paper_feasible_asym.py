# find_paper_feasible_asym.py
# -*- coding: utf-8 -*-
"""
方案二：非对称（dN,dS,etaE,etaW 独立）快速找“论文可用的可行解（待选解）”。

输出：
1) D:\ICM_RESULT\pro1passive_shading\paper_feasible_candidates_asym.csv
   - 包含抽样到的所有可行解与指标（年冷负荷、采光比例、眩光小时数、工程评分等）
2) 控制台打印：
   - baseline（无遮阳近似）
   - best_energy（最省电可行解）
   - paper_pick（更工程合理的可行解，适合写论文）

依赖：
- 你的项目代码：config.py, optimize.py, data_io.py 等
- pvlib（太阳位置）
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


# =========================
# 你可以在这里改参数
# =========================
WEATHER_PATH = Path(r"D:\ICM_CODE\weather_singapore_hourly.csv")
OUT_DIR = Path(r"D:\ICM_RESULT\pro1passive_shading")
OUT_CSV = OUT_DIR / "paper_feasible_candidates_asym.csv"

TARGET_YEAR = 2023          # 构造“TMY虚拟年”的年份（只是时间轴标签，不代表真实年份）
N_TRIALS = 2500             # 抽样次数：越大越容易挑到更好的方案
SEED = 7

# 工程偏好：避免贴边界（你可以调）
EDGE_FRAC_GUARD = 0.08      # 距离上下界小于 8% 区间算“贴边”
ASYM_TOL_D = 0.12           # 强制“不要完全对称”：|dN-dS| < 0.12m 则跳过
ASYM_TOL_ETA = 0.05         # 强制“不要完全对称”：|etaE-etaW| < 0.05rad 则跳过


# =========================
# 工具函数
# =========================
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def build_tmy_virtual_year(df_raw: pd.DataFrame, tz: str, target_year: int = 2023) -> pd.DataFrame:
    """
    将“拼接年（TMY）”重建为一个连续的虚拟年时间轴：
    - 不按真实年份切片（因为每个月可能来自不同年份）
    - 直接按“行顺序”赋予一个连续的 hourly index：target_year-01-01 00:00 -> 12-31 23:00

    要求：
    - df_raw 至少包含 8760 行（非闰年）
    - df_raw index 已经是 tz-aware DatetimeIndex（load_weather_csv 会处理）
    """
    if not isinstance(df_raw.index, pd.DatetimeIndex):
        raise TypeError("df_raw.index 必须是 DatetimeIndex")

    # 统一时区
    if df_raw.index.tz is None:
        df_raw = df_raw.copy()
        df_raw.index = df_raw.index.tz_localize(tz)
    else:
        df_raw = df_raw.tz_convert(tz)

    # 只取前 8760 行（PVGIS TMY 标准是 8760）
    if len(df_raw) < 8760:
        raise ValueError(f"数据行数不足 8760：{len(df_raw)}，无法构造完整虚拟年。")
    df = df_raw.iloc[:8760].copy()

    # 构造连续虚拟年时间轴
    idx = pd.date_range(
        start=pd.Timestamp(f"{target_year}-01-01 00:00:00", tz=tz),
        periods=8760,
        freq="1h"
    )
    df.index = idx

    # 保险：去掉可能的 -0.00
    for c in ["DNI", "DHI", "GHI"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
            df[c] = df[c].clip(lower=0.0)

    return df


def engineering_score(x: tuple[float, float, float, float], bounds: DecisionBounds) -> float:
    """
    工程合理性评分（越小越好）：
    - 不希望贴边界
    - 不希望极端值
    - 不希望南北/东西差异特别离谱（差太大施工不经济）
    """
    dN, dS, etaE, etaW = x

    def norm01(v, lo, hi):
        return (v - lo) / (hi - lo + 1e-12)

    # 归一化到 [0,1]
    dNn = norm01(dN, *bounds.dN_bounds_m)
    dSn = norm01(dS, *bounds.dS_bounds_m)
    eEn = norm01(etaE, *bounds.etaE_bounds_rad)
    eWn = norm01(etaW, *bounds.etaW_bounds_rad)

    # 1) 贴边惩罚：靠近 0 或 1 都罚
    def edge_pen(u):
        return max(0.0, EDGE_FRAC_GUARD - min(u, 1.0 - u)) / EDGE_FRAC_GUARD

    pen_edge = edge_pen(dNn) + edge_pen(dSn) + edge_pen(eEn) + edge_pen(eWn)

    # 2) 中心偏离惩罚：离 0.5 越远越罚（鼓励中庸工程值）
    pen_center = (abs(dNn - 0.5) + abs(dSn - 0.5) + abs(eEn - 0.5) + abs(eWn - 0.5))

    # 3) 差异惩罚：南北/东西差太大也不工程
    pen_asym = 0.6 * abs(dN - dS) + 0.4 * abs(etaE - etaW)

    return 2.0 * pen_edge + 1.0 * pen_center + pen_asym


def sample_x_asymmetric(rng: np.random.Generator, bounds: DecisionBounds) -> tuple[float, float, float, float]:
    """
    独立抽样四个变量，并避免完全对称（按容差过滤）。
    为了更工程：使用“截断区间”避免总抽到边界附近。
    """
    d_lo, d_hi = bounds.dN_bounds_m
    s_lo, s_hi = bounds.dS_bounds_m
    e_lo, e_hi = bounds.etaE_bounds_rad
    w_lo, w_hi = bounds.etaW_bounds_rad

    # 工程上通常不会做 0 或 3 的极端，先内缩一点（你可调）
    def shrink(lo, hi, frac=0.06):
        span = hi - lo
        return lo + frac * span, hi - frac * span

    d_lo2, d_hi2 = shrink(d_lo, d_hi)
    s_lo2, s_hi2 = shrink(s_lo, s_hi)
    e_lo2, e_hi2 = shrink(e_lo, e_hi)
    w_lo2, w_hi2 = shrink(w_lo, w_hi)

    for _ in range(200):  # 最多尝试 200 次，确保抽到不完全对称的
        dN = float(rng.uniform(d_lo2, d_hi2))
        dS = float(rng.uniform(s_lo2, s_hi2))
        etaE = float(rng.uniform(e_lo2, e_hi2))
        etaW = float(rng.uniform(w_lo2, w_hi2))

        # 强制“不完全对称”（避免你又看到相等然后疑惑）
        if abs(dN - dS) < ASYM_TOL_D and abs(etaE - etaW) < ASYM_TOL_ETA:
            continue

        return (dN, dS, etaE, etaW)

    # 实在抽不到就放宽（理论上不会走到这里）
    return (dN, dS, etaE, etaW)


def find_paper_feasible_asymmetric() -> None:
    """
    ✅ 新名字：find_paper_feasible_asymmetric
    用来区别你之前的“对称版 / 原版”。
    """
    ensure_dir(OUT_DIR)

    # ---------- 1) 读配置 ----------
    loc = LocationConfig()
    bld = BuildingConfig()
    opt = OpticalConfig()
    th = ThermalConfig()
    sch = ScheduleConfig()
    vis = VisualConstraintConfig()
    bd = DecisionBounds()

    # ---------- 2) 读天气 + 构造虚拟年 ----------
    if not WEATHER_PATH.exists():
        raise FileNotFoundError(f"找不到天气文件：{WEATHER_PATH}")

    df_raw = load_weather_csv(str(WEATHER_PATH), tz=loc.tz)

    # 你现在的 CSV 有 DNI/DHI/GHI/T_out，确保都在
    for c in ["DNI", "DHI", "GHI", "T_out"]:
        if c not in df_raw.columns:
            if c in ["DHI", "GHI"]:
                df_raw[c] = 0.0
            else:
                raise ValueError(f"天气数据缺少列：{c}，现有列：{list(df_raw.columns)}")

    df = build_tmy_virtual_year(df_raw, tz=loc.tz, target_year=TARGET_YEAR)

    print("[INFO] weather_path =", WEATHER_PATH)
    print("=== 数据检查（TMY 虚拟年）===")
    print("时区:", loc.tz)
    print("时间范围:", df.index.min(), "->", df.index.max())
    print("总小时数:", len(df), "(期望 8760)")
    print(f"DNI(min/max): {df['DNI'].min():.2f} / {df['DNI'].max():.2f}")
    print(f"T_out(min/max): {df['T_out'].min():.2f} / {df['T_out'].max():.2f}")
    print(f"DHI(min/max): {df['DHI'].min():.2f} / {df['DHI'].max():.2f}")
    print(f"GHI(min/max): {df['GHI'].min():.2f} / {df['GHI'].max():.2f}")
    print()

    # ---------- 3) baseline（与当前 bounds 一致的“近似无遮阳”） ----------
    baseline_x = (0.0, 0.0, bd.etaE_bounds_rad[1], bd.etaW_bounds_rad[1])
    base = evaluate_design(df, loc, bld, opt, th, sch, vis, baseline_x)
    print("[BASELINE] x =", baseline_x,
          "L_year_kWh =", base["L_year_kWh"],
          "vis =", base["vis"],
          "feasible =", base["feasible"])
    print()

    # ---------- 4) 随机找可行解池 ----------
    rng = np.random.default_rng(SEED)

    rows = []
    n_feas = 0

    best_energy = None
    best_energy_L = float("inf")

    for i in range(N_TRIALS):
        x = sample_x_asymmetric(rng, bd)
        res = evaluate_design(df, loc, bld, opt, th, sch, vis, x)
        if not res["feasible"]:
            continue

        n_feas += 1
        L = float(res["L_year_kWh"])
        p = float(res["vis"]["p_daylight"])
        hg = int(res["vis"]["H_glare"])
        score = float(engineering_score(x, bd))

        rows.append({
            "dN": x[0], "dS": x[1], "etaE": x[2], "etaW": x[3],
            "L_year_kWh": L,
            "p_daylight": p,
            "H_glare": hg,
            "eng_score": score,
        })

        if L < best_energy_L:
            best_energy_L = L
            best_energy = (x, res, score)

    if n_feas == 0:
        print("[WARN] 没找到任何可行解。说明你当前 config 约束太严或 bounds 太窄。")
        print("建议：先降低 p_min 或放宽 H_glare_max 或扩大 bounds 再跑。")
        return

    cand = pd.DataFrame(rows).sort_values(["L_year_kWh", "eng_score"]).reset_index(drop=True)
    cand.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print("[OK] 候选已保存：", OUT_CSV)
    print(f"[INFO] 抽样次数={N_TRIALS}，可行解数量={n_feas}")
    print()

    # ---------- 5) 输出 best_energy ----------
    x_be, res_be, score_be = best_energy
    print("=== 可行解中最省电 best_energy ===")
    print("x =", x_be)
    print("L_year_kWh =", float(res_be["L_year_kWh"]))
    print("vis =", res_be["vis"])
    print("eng_score =", score_be)
    print()

    # ---------- 6) 论文推荐 paper_pick（更工程，不贴边界，不极端） ----------
    # 思路：在可行池里，挑“eng_score”较小且 L 也不错的折中点
    # 这里用：minimize( L_norm + 0.18 * score_norm )
    Lmin, Lmax = cand["L_year_kWh"].min(), cand["L_year_kWh"].max()
    Smin, Smax = cand["eng_score"].min(), cand["eng_score"].max()

    def norm(v, vmin, vmax):
        if vmax <= vmin + 1e-12:
            return 0.0
        return (v - vmin) / (vmax - vmin)

    cand2 = cand.copy()
    cand2["L_norm"] = cand2["L_year_kWh"].apply(lambda v: norm(v, Lmin, Lmax))
    cand2["S_norm"] = cand2["eng_score"].apply(lambda v: norm(v, Smin, Smax))
    cand2["paper_obj"] = cand2["L_norm"] + 0.18 * cand2["S_norm"]

    pick = cand2.sort_values("paper_obj").iloc[0]
    x_pp = (float(pick["dN"]), float(pick["dS"]), float(pick["etaE"]), float(pick["etaW"]))
    res_pp = evaluate_design(df, loc, bld, opt, th, sch, vis, x_pp)

    print("=== 论文推荐 paper_pick（更工程、不贴边界、不极端）===")
    print("x =", x_pp)
    print("L_year_kWh =", float(res_pp["L_year_kWh"]))
    print("vis =", res_pp["vis"])
    print("eng_score =", float(pick["eng_score"]))
    print()

    # 论文一句话（你可直接复制）
    save_ratio = (float(base["L_year_kWh"]) - float(res_pp["L_year_kWh"])) / max(float(base["L_year_kWh"]), 1e-12)
    print("=== 论文一句话可用 ===")
    print(
        f"在工程可实现的遮阳参数范围内随机筛选得到可行方案 "
        f"x=({x_pp[0]:.3f},{x_pp[1]:.3f},{x_pp[2]:.3f},{x_pp[3]:.3f})，"
        f"满足采光与眩光约束（p_daylight={res_pp['vis']['p_daylight']:.3f}, H_glare={res_pp['vis']['H_glare']}），"
        f"年制冷负荷由 {float(base['L_year_kWh']):.1f} kWh 降至 {float(res_pp['L_year_kWh']):.1f} kWh，"
        f"节能约 {save_ratio*100:.2f}% 。"
    )


if __name__ == "__main__":
    find_paper_feasible_asymmetric()
