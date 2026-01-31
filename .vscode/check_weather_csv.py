# check_weather_csv.py
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

CSV_PATH = r"D:\ICM_CODE\weather_singapore_hourly.csv"
TZ = "Asia/Singapore"

def main():
    df = pd.read_csv(CSV_PATH)
    if "datetime" not in df.columns:
        raise ValueError("CSV缺少 datetime 列")
    if "DNI" not in df.columns or "T_out" not in df.columns:
        raise ValueError("CSV至少需要 DNI 与 T_out 列")

    t = pd.to_datetime(df["datetime"])
    # 若已带时区则转换；若不带时区则按新加坡解释
    if t.dt.tz is None:
        t = t.dt.tz_localize(TZ)
    else:
        t = t.dt.tz_convert(TZ)

    df = df.copy()
    df.index = t
    df = df.sort_index()

    print("=== 基础信息 ===")
    print("行数:", len(df))
    print("时间范围:", df.index.min(), "->", df.index.max())
    print("唯一年份:", sorted(df.index.year.unique().tolist()))
    print("月份覆盖:", sorted(df.index.month.unique().tolist()))

    # 连续性检查（以相邻时间差为准）
    dt = df.index.to_series().diff().dropna()
    dt_hours = dt.dt.total_seconds() / 3600.0

    n_not_1h = int(np.sum(np.abs(dt_hours - 1.0) > 1e-9))
    print("\n=== 连续性检查 ===")
    print("相邻时间差 != 1h 的次数:", n_not_1h)
    if n_not_1h > 0:
        bad = df.index.to_series().diff()
        bad_idx = bad[(bad.dt.total_seconds() != 3600)].index[:10]
        print("示例异常位置(最多10个):", list(bad_idx))

    # 重复时间戳检查
    dup = df.index.duplicated().sum()
    print("\n=== 重复检查 ===")
    print("重复时间戳数量:", int(dup))

    # 缺失值检查
    print("\n=== 缺失值检查 ===")
    print("DNI NaN:", int(df["DNI"].isna().sum()))
    print("T_out NaN:", int(df["T_out"].isna().sum()))

    # 是否像一个“完整年”（仅作为提示）
    print("\n=== 判定建议 ===")
    if len(df) == 8760 and n_not_1h == 0 and dup == 0 and set(df.index.month.unique()) == set(range(1,13)):
        print("✅ 看起来是完整的 8760 小时连续年数据，可直接用于仿真/优化。")
    else:
        print("⚠️ 数据未满足“连续完整年”特征：建议不要按真实年份切片；可考虑重建一个连续的虚拟年时间轴。")

if __name__ == "__main__":
    main()
