# test/smoke_test_eval.py
# -*- coding: utf-8 -*-
from pathlib import Path
import sys

# 把 passive_shading 根目录加入 sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from config import (
    LocationConfig, BuildingConfig, OpticalConfig, ThermalConfig,
    ScheduleConfig, VisualConstraintConfig, DecisionBounds
)
from data_io import load_weather_csv, ensure_hourly
from optimize import evaluate_design


def find_weather_csv() -> str:
    """
    优先用你明确的绝对路径，其次找项目根目录同名文件。
    """
    candidates = [
        Path(r"D:\ICM_CODE\weather_singapore_hourly.csv"),
        Path(__file__).resolve().parents[1] / "weather_singapore_hourly.csv",  # 项目根
        Path.cwd() / "weather_singapore_hourly.csv",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    raise FileNotFoundError("找不到 weather_singapore_hourly.csv，请检查路径。")


def load_df_hourly(path: str, tz: str) -> pd.DataFrame:
    df = load_weather_csv(path, tz=tz)

    # 如果你的数据已经是 8760 整点小时，这一步不会改变；
    # 如果是半点/非整点，这一步会对齐到整点小时。
    df = ensure_hourly(df)

    # 基本断言（和你 check_weather 的一致）
    print(f"使用气象文件： {path}")
    print("len:", len(df))
    print("start:", df.index[0])
    print("end:", df.index[-1])
    print("tz:", df.index.tz)

    assert 8000 <= len(df) <= 9000, "小时数不在合理范围（8000-9000），时间序列可能不完整或被重复拼接。"
    assert pd.Series(df.index).diff().dropna().dt.total_seconds().median() == 3600, "时间步长不是 1h。"

    return df


def summarize(res: dict, name: str) -> None:
    vis = res["vis"]
    print(f"\n===== {name} =====")
    print("x =", res["x"])
    print("feasible =", res["feasible"])
    print("L_year_kWh =", res["L_year_kWh"])
    print("p_daylight =", vis.get("p_daylight"))
    print("H_glare =", vis.get("H_glare"))

    # 额外：把“样本集合大小”打印出来（这对你判断口径变化极重要）
    # day_occ = occ & day，所以样本量就是 mask_day_occ 为 True 的个数
    # 但 evaluate_design 没返回 mask，这里用一个近似办法：从 E_vis 的长度无法得出样本数
    # ——所以建议你在 optimize.py 里把 N_day_occ 加进 vis_stats（我下面会告诉你怎么改）


def main():
    # configs
    location = LocationConfig()
    building = BuildingConfig()
    optical = OpticalConfig()
    thermal = ThermalConfig()
    schedule = ScheduleConfig()
    visual = VisualConstraintConfig()
    bounds = DecisionBounds()

    # data
    weather_path = find_weather_csv()
    df = load_df_hourly(weather_path, tz=location.tz)

    # 三个点：baseline / 中等遮阳 / 强遮阳
    # ✅ 注意：eta 是弧度，范围你说的对：0~1.5 左右
    eta_open = bounds.etaE_bounds_rad[1]   # ≈ 1.5
    eta_mid  = 0.8
    eta_tight = bounds.etaE_bounds_rad[0]  # ≈ 0.1

    cases = [
        ("Baseline (no overhang + louvers open)", (0.0, 0.0, eta_open, eta_open)),
        ("Medium shading", (1.0, 1.0, eta_mid, eta_mid)),
        ("Strong shading", (2.5, 2.5, eta_tight, eta_tight)),
    ]

    for name, x in cases:
        res = evaluate_design(df, location, building, optical, thermal, schedule, visual, x)
        summarize(res, name)

    print("\n✅ smoke test 完成：现在你能看到 3 个方案的可行性/采光/眩光指标和年负荷。")


if __name__ == "__main__":
    main()
