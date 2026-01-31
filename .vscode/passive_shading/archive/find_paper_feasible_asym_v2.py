# find_paper_feasible_asym_v2.py
# -*- coding: utf-8 -*-
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


def build_tmy_virtual_year(df_raw: pd.DataFrame, tz: str, target_year: int = 2023) -> pd.DataFrame:
    """
    将 PVGIS TMY（跨多年拼接、时间不连续）重排成一个“虚拟连续年”：
      - 先把 index 转到本地 tz
      - 若分钟恒为 30（PVGIS offset=0.5 常见），整体减 30min 让其落到整点
      - 按 (month, day, hour) 排序
      - 重新赋予 target_year 的连续 8760 小时索引
    """
    df = df_raw.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize(tz)
    else:
        df = df.tz_convert(tz)

    # 常见：PVGIS 转本地后全是 :30（offset=0.5h），这里自动对齐到整点
    mins = np.unique(df.index.minute)
    if len(mins) == 1 and int(mins[0]) == 30:
        df.index = df.index - pd.Timedelta(minutes=30)

    # 用月/日/小时排序，构造“典型年序列”
    key = pd.DataFrame({
        "month": df.index.month,
        "day": df.index.day,
        "hour": df.index.hour,
    }, index=df.index)

    df2 = df.copy()
    df2["_month"] = key["month"].to_numpy()
    df2["_day"] = key["day"].to_numpy()
    df2["_hour"] = key["hour"].to_numpy()

    df2 = df2.sort_values(["_month", "_day", "_hour"]).drop(columns=["_month", "_day", "_hour"])

    # 重新赋予连续时间轴（8760小时）
    n = len(df2)
    idx_new = pd.date_range(
        start=pd.Timestamp(f"{target_year}-01-01 00:00:00", tz=tz),
        periods=n,
        freq="1h"
    )
    df2.index = idx_new

    # 清洗一下
    if "DNI" in df2.columns:
        df2["DNI"] = pd.to_numeric(df2["DNI"], errors="coerce").fillna(0.0).clip(lower=0.0)

    return df2


def quick_stats(df: pd.DataFrame, tz: str) -> None:
    print("=== 数据检查（TMY 虚拟年）===")
    print("时区:", tz)
    print("时间范围:", df.index.min(), "->", df.index.max())
    print(f"总小时数: {len(df)} (期望 8760)")
    for c in ["DNI", "DHI", "GHI", "T_out"]:
        if c in df.columns:
            print(f"{c}(min/max): {df[c].min():.2f} / {df[c].max():.2f}")


def engineering_score(x, bounds: DecisionBounds) -> float:
    """
    越小越“工程”：
    - 惩罚贴边界（太极端）
    - 惩罚过强不对称（便于论文解释与施工）
    """
    dN, dS, etaE, etaW = x

    b = [
        ("dN", dN, bounds.dN_bounds_m),
        ("dS", dS, bounds.dS_bounds_m),
        ("etaE", etaE, bounds.etaE_bounds_rad),
        ("etaW", etaW, bounds.etaW_bounds_rad),
    ]

    score = 0.0

    # 1) 距离区间中心（越靠中心越好）
    for _, v, (lo, hi) in b:
        rng = max(hi - lo, 1e-12)
        mid = 0.5 * (lo + hi)
        score += ((v - mid) / (0.5 * rng)) ** 2  # 0~1 量级

    # 2) 贴边界额外惩罚：落在两端 10% 区域就加分（不希望论文推荐解贴边）
    for _, v, (lo, hi) in b:
        rng = max(hi - lo, 1e-12)
        if v < lo + 0.10 * rng:
            score += 2.0
        if v > hi - 0.10 * rng:
            score += 2.0

    # 3) 不对称惩罚（不是不允许不对称，只是别太夸张）
    score += 1.0 * ((dN - dS) / max(bounds.dN_bounds_m[1] - bounds.dN_bounds_m[0], 1e-12)) ** 2
    score += 1.0 * ((etaE - etaW) / max(bounds.etaE_bounds_rad[1] - bounds.etaE_bounds_rad[0], 1e-12)) ** 2

    return float(score)


def main():
    # ---------- 配置 ----------
    loc = LocationConfig()
    bld = BuildingConfig()
    opt = OpticalConfig()
    th = ThermalConfig()
    sc = ScheduleConfig()
    vc = VisualConstraintConfig()
    bd = DecisionBounds()

    # ---------- 输入路径 ----------
    # 优先用 D:\ICM_CODE\weather_singapore_hourly.csv
    candidates = [
        Path(r"D:\ICM_CODE\weather_singapore_hourly.csv"),
        Path(__file__).resolve().parent / "weather_singapore_hourly.csv",
    ]
    weather_path = None
    for p in candidates:
        if p.exists():
            weather_path = p
            break
    if weather_path is None:
        raise FileNotFoundError("找不到 weather_singapore_hourly.csv，请检查路径。")

    print("[INFO] weather_path =", weather_path)

    df_raw = load_weather_csv(str(weather_path), tz=loc.tz)  # 允许包含 DHI/GHI 额外列
    df = build_tmy_virtual_year(df_raw, tz=loc.tz, target_year=2023)

    # 只保留需要的列
    need_cols = ["DNI", "DHI", "GHI", "T_out"]
    miss = [c for c in need_cols if c not in df.columns]
    if miss:
        raise ValueError(f"缺少列 {miss}。你现在的 CSV 表头应包含 {need_cols}。")

    quick_stats(df, tz=loc.tz)

    # ---------- baseline（无挑檐 + 百叶“开” = eta 取上界） ----------
    baseline_x = (0.0, 0.0, bd.etaE_bounds_rad[1], bd.etaW_bounds_rad[1])
    baseline = evaluate_design(df, loc, bld, opt, th, sc, vc, baseline_x)

    print("\n[BASELINE] x =", baseline_x,
          "L_year_kWh =", baseline["L_year_kWh"],
          "vis =", baseline["vis"],
          "feasible =", baseline["feasible"])

    # ---------- 随机抽样找“论文可用工程解” ----------
    rng = np.random.default_rng(1)
    n_samples = 2500

    def sample_one():
        dN = float(rng.uniform(*bd.dN_bounds_m))
        dS = float(rng.uniform(*bd.dS_bounds_m))
        etaE = float(rng.uniform(*bd.etaE_bounds_rad))
        etaW = float(rng.uniform(*bd.etaW_bounds_rad))
        return (dN, dS, etaE, etaW)

    rows = []
    feasible_count = 0

    for _ in range(n_samples):
        x = sample_one()
        res = evaluate_design(df, loc, bld, opt, th, sc, vc, x)
        if res["feasible"]:
            feasible_count += 1

        eng = engineering_score(x, bd)

        rows.append({
            "dN": x[0], "dS": x[1], "etaE": x[2], "etaW": x[3],
            "L_year_kWh": float(res["L_year_kWh"]),
            "p_daylight": float(res["vis"]["p_daylight"]),
            "H_glare": int(res["vis"]["H_glare"]),
            "feasible": bool(res["feasible"]),
            "eng_score": float(eng),
        })

    out_df = pd.DataFrame(rows)

    # 输出候选
    out_dir = Path(r"D:\ICM_RESULT\pro1passive_shading")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "paper_feasible_candidates_asym_v2.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("\n[OK] 候选已保存：", out_path)
    print(f"[INFO] 抽样次数={n_samples}，可行解数量={feasible_count}")

    feas = out_df[out_df["feasible"]].copy()
    if feas.empty:
        print("\n[WARN] 没有找到可行解。建议：放宽 bounds 或放宽 config 里的 visual 约束。")
        return

    # 1) 可行解里最省电
    best_energy = feas.sort_values("L_year_kWh", ascending=True).iloc[0]

    # 2) “论文推荐”：不贴边界、不极端 => eng_score 小 + 能耗也不错
    #    这里做一个折中：min( L + w*eng_score )
    w = 1500.0  # 你可以调整：越大越偏“工程可信”，越小越偏“省电”
    feas["paper_cost"] = feas["L_year_kWh"] + w * feas["eng_score"]
    paper_pick = feas.sort_values("paper_cost", ascending=True).iloc[0]

    print("\n=== 可行解中最省电 best_energy ===")
    print("x = (%.3f, %.3f, %.3f, %.3f)" % (best_energy["dN"], best_energy["dS"], best_energy["etaE"], best_energy["etaW"]))
    print("L_year_kWh =", float(best_energy["L_year_kWh"]))
    print("vis =", {"p_daylight": float(best_energy["p_daylight"]), "H_glare": int(best_energy["H_glare"])})
    print("eng_score =", float(best_energy["eng_score"]))

    print("\n=== 论文推荐 paper_pick（更工程、不贴边界、不极端）===")
    print("x = (%.3f, %.3f, %.3f, %.3f)" % (paper_pick["dN"], paper_pick["dS"], paper_pick["etaE"], paper_pick["etaW"]))
    print("L_year_kWh =", float(paper_pick["L_year_kWh"]))
    print("vis =", {"p_daylight": float(paper_pick["p_daylight"]), "H_glare": int(paper_pick["H_glare"])})
    print("eng_score =", float(paper_pick["eng_score"]))

    # 一句话（相对 baseline）
    base_L = float(baseline["L_year_kWh"])
    pick_L = float(paper_pick["L_year_kWh"])
    save_ratio = (base_L - pick_L) / base_L if base_L > 0 else 0.0

    print("\n=== 论文一句话可用 ===")
    print(
        "在工程可实现的遮阳参数范围内随机筛选得到可行方案 "
        f"x=({paper_pick['dN']:.3f},{paper_pick['dS']:.3f},{paper_pick['etaE']:.3f},{paper_pick['etaW']:.3f})，"
        f"满足采光与眩光约束（p_daylight={paper_pick['p_daylight']:.3f}, H_glare={int(paper_pick['H_glare'])}），"
        f"年制冷负荷由 {base_L:.1f} kWh 降至 {pick_L:.1f} kWh，节能约 {100*save_ratio:.2f}% 。"
    )


if __name__ == "__main__":
    main()
