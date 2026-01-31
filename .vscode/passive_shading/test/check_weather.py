# test/check_weather.py
# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))   # .../passive_shading/test
PROJECT_DIR = os.path.dirname(THIS_DIR)                 # .../passive_shading
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from config import LocationConfig
from data_io import load_weather_csv, ensure_hourly


def main():
    # 向上找 CSV（因为你的真实文件在 D:\ICM_CODE\ 下）
    ROOT = PROJECT_DIR
    path = None
    for _ in range(8):  # 多给几层，稳一点
        candidate = os.path.join(ROOT, "weather_singapore_hourly.csv")
        if os.path.exists(candidate):
            path = candidate
            break
        ROOT = os.path.dirname(ROOT)

    if path is None:
        raise FileNotFoundError("找不到 weather_singapore_hourly.csv（已向上搜索 8 层）")

    print("使用气象文件：", path)

    tz = LocationConfig().tz
    df = load_weather_csv(path, tz=tz)
    df = ensure_hourly(df)

    print("len:", len(df))
    print("start:", df.index[0])
    print("end:", df.index[-1])
    print("tz:", df.index.tz)

    assert 8000 <= len(df) <= 9000, f"行数异常：{len(df)}（期望接近 8760）"

    step_sec = pd.Series(df.index).diff().dropna().dt.total_seconds()
    med = step_sec.median()
    assert med == 3600, f"时间步中位数不是 1h：median={med} sec"

    print("✅ 数据检查通过")


if __name__ == "__main__":
    main()
