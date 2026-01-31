# find_paper_feasible.py
# -*- coding: utf-8 -*-
"""
目的：快速找一个“论文里好看、工程上合理”的可行解（不用追求全局最优）

思路：
1) 读 weather_singapore_hourly.csv（包含 DNI/DHI/GHI/T_out）
2) 重采样为整点小时，并构建 2023 虚拟年（8760h 连续）
3) 在工程合理范围内抽样 (dN, dS, etaE, etaW)
4) evaluate_design 判断可行性并计算年制冷负荷
5) 输出：
   - best_energy：可行里最省电的
   - paper_pick ：可行里“不贴边界、更工程合理”的（推荐论文用）
并保存候选到 D:\ICM_RESULT\pro1passive_shading\paper_feasible_candidates.csv
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


# ============ 你可以改的“工程范围” ============
# 悬挑深度（m）：一般 0.3~2.0 比较像真能做出来
D_MIN, D_MAX = 0.3, 2.0

# 百叶 cutoff（rad）：别用 0.1 和 10 这种极端；0.3~1.2 比较正常
ETA_MIN, ETA_MAX = 0.3, 1.2

# 抽样次数：越大越容易找到更好看的解，但会更慢（先用 1500~3000 就够）
N_SAMPLES = 2500

# 是否强制“对称方案”（论文更好解释）：dN=dS 且 etaE=etaW
FORCE_SYMMETRY = True

# 随机种子：保证每次跑出来一样（便于论文复现）
SEED = 1


def locate_weather_csv(script_dir: Path) -> Path:
    """
    尝试在常见位置找 weather_singapore_hourly.csv：
    1) 脚本同目录
    2) 项目根目录 D:\ICM_CODE
    """
    candidates = [
        script_dir / "weather_singapore_hourly.csv",
        script_dir.parents[1] / "weather_singapore_hourly.csv",  # ...\ICM_CODE\weather...
        script_dir.parents[2] / "weather_singapore_hourly.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "找不到 weather_singapore_hourly.csv。\n"
        "请把它放在：\n"
        f"  1) {candidates[0]}\n"
        f"  2) {candidates[1]}\n"
        f"  3) {candidates[2]}\n"
    )


def build_tmy_virtual_year(df_raw: pd.DataFrame, tz: str, target_year: int = 2023) -> pd.DataFrame:
    """
    把 PVGIS TMY（年份可能跳来跳去）重建成“连续虚拟年”：
    - 先重采样到整点小时（1h）
    - 然后忽略原始时间戳顺序，直接按行顺序映射到 target_year 的 8760 小时
    """
    df = df_raw.copy()

    # 确保时区
    if df.index.tz is None:
        df.index = df.index.tz_localize(tz)
    else:
        df = df.tz_convert(tz)

    # 整点小时
    df = df.resample("1h").mean()

    # 截取前 8760 行（TMY 非闰年）
    if len(df) < 8760:
        raise ValueError(f"重采样后小时数不足 8760：{len(df)}，请检查原始数据。")
    df = df.iloc[:8760].copy()

    # 构造连续虚拟年时间轴
    idx = pd.date_range(f"{target_year}-01-01 00:00:00", periods=8760, freq="1h", tz=tz)
    df.index = idx

    return df


def is_not_extreme(x: tuple[float, float, float, float]) -> bool:
    """
    定义“工程不极端”：不贴边界、数值在合理区间内
    你可以按需要调整这条规则。
    """
    dN, dS, eE, eW = x
    if not (D_MIN <= dN <= D_MAX and D_MIN <= dS <= D_MAX):
        return False
    if not (ETA_MIN <= eE <= ETA_MAX and ETA_MIN <= eW <= ETA_MAX):
        return False

    # 进一步：别太贴边界（留 10% 缓冲）
    def away_from_bounds(v, lo, hi, frac=0.10):
        margin = (hi - lo) * frac
        return (v >= lo + margin) and (v <= hi - margin)

    return (
        away_from_bounds(dN, D_MIN, D_MAX) and
        away_from_bounds(dS, D_MIN, D_MAX) and
        away_from_bounds(eE, ETA_MIN, ETA_MAX) and
        away_from_bounds(eW, ETA_MIN, ETA_MAX)
    )


def main():
    rng = np.random.default_rng(SEED)

    # ---- 配置对象 ----
    loc = LocationConfig()
    bld = BuildingConfig()
    opt = OpticalConfig()
    th = ThermalConfig()
    sc = ScheduleConfig()
    vc = VisualConstraintConfig()
    bounds = DecisionBounds()

    # ---- 读天气 ----
    script_dir = Path(__file__).resolve().parent
    weather_path = locate_weather_csv(script_dir)
    print("[INFO] weather_path =", weather_path)

    df_raw = load_weather_csv(str(weather_path), tz=loc.tz)

    # 检查列（你现在的 optimize 用到了 DHI/GHI）
    need_cols = ["DNI", "DHI", "GHI", "T_out"]
    miss = [c for c in need_cols if c not in df_raw.columns]
    if miss:
        raise ValueError(f"天气数据缺少列：{miss}，当前列：{list(df_raw.columns)}")

    # ---- 构建虚拟年 ----
    df = build_tmy_virtual_year(df_raw, tz=loc.tz, target_year=2023)
    print("=== 数据检查（TMY 虚拟年）===")
    print("时区:", loc.tz)
    print("时间范围:", df.index.min(), "->", df.index.max())
    print("总小时数:", len(df), "(期望 8760)")
    print("DNI(min/max):", float(df["DNI"].min()), "/", float(df["DNI"].max()))
    print("T_out(min/max):", float(df["T_out"].min()), "/", float(df["T_out"].max()))
    print("DHI(min/max):", float(df["DHI"].min()), "/", float(df["DHI"].max()))
    print("GHI(min/max):", float(df["GHI"].min()), "/", float(df["GHI"].max()))
    print()

    # ---- baseline（用于对比写论文）----
    baseline_x = (0.0, 0.0, bounds.etaE_bounds_rad[1], bounds.etaW_bounds_rad[1])
    base = evaluate_design(df, loc, bld, opt, th, sc, vc, baseline_x)
    print("[BASELINE] x =", base["x"], "L_year_kWh =", base["L_year_kWh"], "vis =", base["vis"], "feasible =", base["feasible"])
    print()

    records = []
    best_energy = None
    best_energy_L = float("inf")

    paper_pick = None
    paper_pick_L = float("inf")

    # ---- 抽样搜索 ----
    for i in range(N_SAMPLES):
        if FORCE_SYMMETRY:
            d = float(rng.uniform(D_MIN, D_MAX))
            eta = float(rng.uniform(ETA_MIN, ETA_MAX))
            x = (d, d, eta, eta)
        else:
            x = (
                float(rng.uniform(D_MIN, D_MAX)),
                float(rng.uniform(D_MIN, D_MAX)),
                float(rng.uniform(ETA_MIN, ETA_MAX)),
                float(rng.uniform(ETA_MIN, ETA_MAX)),
            )

        res = evaluate_design(df, loc, bld, opt, th, sc, vc, x)

        rec = {
            "dN": x[0], "dS": x[1], "etaE": x[2], "etaW": x[3],
            "feasible": bool(res["feasible"]),
            "L_year_kWh": float(res["L_year_kWh"]),
            "p_daylight": float(res["vis"]["p_daylight"]),
            "H_glare": int(res["vis"]["H_glare"]),
        }
        records.append(rec)

        if res["feasible"]:
            L = float(res["L_year_kWh"])

            # 1) 可行里最省电
            if L < best_energy_L:
                best_energy_L = L
                best_energy = res

            # 2) 论文推荐：可行且不极端
            if is_not_extreme(x) and (L < paper_pick_L):
                paper_pick_L = L
                paper_pick = res

    out_df = pd.DataFrame(records).sort_values(["feasible", "L_year_kWh"], ascending=[False, True])

    # ---- 输出 CSV ----
    result_dir = Path(r"D:\ICM_RESULT") / "pro1passive_shading"
    result_dir.mkdir(parents=True, exist_ok=True)
    out_csv = result_dir / "paper_feasible_candidates.csv"
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("[OK] 候选已保存：", out_csv)
    print()

    # ---- 打印结果 ----
    feasible_count = int(out_df["feasible"].sum())
    print(f"[INFO] 抽样次数={N_SAMPLES}，可行解数量={feasible_count}")
    if feasible_count == 0:
        print("❌ 没找到可行解：建议先放宽约束（比如调 p_min / E_min / H_glare_max / E_glare），或者扩大工程范围。")
        return

    print("\n=== 可行解中最省电 best_energy ===")
    print("x =", best_energy["x"])
    print("L_year_kWh =", float(best_energy["L_year_kWh"]))
    print("vis =", best_energy["vis"])

    print("\n=== 论文推荐 paper_pick（更工程、不贴边界）===")
    if paper_pick is None:
        print("⚠️ 可行解里没有满足“不极端规则”的，退而求其次建议用 best_energy。")
        paper_pick = best_energy
    print("x =", paper_pick["x"])
    print("L_year_kWh =", float(paper_pick["L_year_kWh"]))
    print("vis =", paper_pick["vis"])

    # 给你一个可直接复制进论文/报告的对比句子
    save_ratio = (base["L_year_kWh"] - paper_pick["L_year_kWh"]) / max(base["L_year_kWh"], 1e-12)
    print("\n=== 论文一句话可用 ===")
    print(f"在工程可实现的遮阳参数范围内筛选得到可行方案 x={paper_pick['x']}，"
          f"满足采光与眩光约束（p_daylight={paper_pick['vis']['p_daylight']:.3f}, H_glare={paper_pick['vis']['H_glare']}），"
          f"年制冷负荷由 {base['L_year_kWh']:.1f} kWh 降至 {paper_pick['L_year_kWh']:.1f} kWh，"
          f"节能约 {save_ratio*100:.2f}%。")


if __name__ == "__main__":
    main()
