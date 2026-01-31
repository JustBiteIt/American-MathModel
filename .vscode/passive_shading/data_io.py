# data_io.py
# -*- coding: utf-8 -*-
import pandas as pd

# 现在我们需要三分量：DNI/DHI/GHI + T_out
REQUIRED_COLS = ["datetime", "DNI", "DHI", "GHI", "T_out"]

def _normalize_typical_year_index(idx: pd.DatetimeIndex, year: int = 2023) -> pd.DatetimeIndex:
    """
    将“拼好月/典型年”这种跨年份拼接的时间索引，统一映射到同一个虚拟年份。
    保留 月-日-时-分-秒 不变，只改 year，避免后续任何 resample 因跨度过大膨胀数据。
    默认 year=2023（非闰年）。若数据里有 2/29 会报错，但你的 PVGIS 典型年通常不会有 2/29。
    """
    tz = idx.tz
    # 用 list comprehension 保留时区
    new = [
        pd.Timestamp(year=year, month=t.month, day=t.day, hour=t.hour, minute=t.minute, second=t.second, tz=tz)
        for t in idx
    ]
    return pd.DatetimeIndex(new)

def load_weather_csv(path: str, tz: str) -> pd.DataFrame:
    """
    读取本项目的气象输入 CSV，并返回以 datetime 为索引的 DataFrame（tz-aware）。

    输入 CSV 必须包含：
      - datetime: 时间戳（建议 tz-aware，例如 2006-01-01 08:30:00+08:00）
      - DNI, DHI, GHI (W/m^2)
      - T_out (°C)
    """
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"缺少列：{missing}。\n"
            f"当前列：{list(df.columns)}\n"
            "请确认 CSV 列名与本项目一致。"
        )

    # 解析时间
    t = pd.to_datetime(df["datetime"], errors="coerce")
    if t.isna().any():
        bad = df.loc[t.isna(), "datetime"].head(5).tolist()
        raise ValueError(f"datetime 解析失败，示例：{bad}")

    if t.dt.tz is None:
        # 不带时区：按 tz 本地化（不推荐；但兼容）
        t = t.dt.tz_localize(tz)
    else:
        # 带时区：转换到目标时区
        t = t.dt.tz_convert(tz)

    out = df.copy()
    out.index = pd.DatetimeIndex(t)
    out = out.drop(columns=["datetime"]).sort_index()

    # 基本清洗：辐照度非负 + 数值化
    for col in ["DNI", "DHI", "GHI"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0).clip(lower=0.0)

    out["T_out"] = pd.to_numeric(out["T_out"], errors="coerce")
    out = out.dropna(subset=["T_out"])  # 温度用于 1R1C

    # ---- 关键：自动识别“典型年拼接”（跨年份但行数接近 8760）并归一化到同一年 ----
    years = out.index.year
    if years.nunique() > 1 and 8000 <= len(out) <= 9000:
        out = out.copy()
        out.index = _normalize_typical_year_index(out.index, year=2023)
        out = out.sort_index()

    return out


def ensure_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    对齐为逐小时序列（安全版）。

    你的 PVGIS 典型年常见时间戳为 :30（08:30, 09:30...），
    直接 resample("1h") 在“跨年份拼接”的数据上可能会把 18 年跨度补齐，生成十几万小时的空数据 → 灾难。

    这里采用：
      1) 若分钟全是 30：整体 shift(-30min)，再 floor('h')
      2) 用 groupby(index).mean() 合并重复（不会生成缺失小时）
    """
    out = df.copy()
    idx = out.index

    # 如果已经看起来是逐小时（相邻差值的中位数≈1小时），就只做必要的 :30 -> 整点对齐
    minutes = pd.Index(idx.minute)

    if minutes.nunique() == 1 and minutes[0] == 30:
        # 把 08:30 当作 08:00 这一小时的代表（等价于“区间中心”转“区间起点”）
        new_idx = (idx - pd.Timedelta(minutes=30)).floor("h")
        out.index = new_idx
        out = out.groupby(out.index).mean()
        # 保留时区（groupby 后 index 仍是 tz-aware）
        return out.sort_index()

    # 如果不是 :30，但可能有一些非整点噪声，也可以做一个“就近整点”归并
    # 不做 resample，避免跨年跨度膨胀
    if not ((minutes == 0).all()):
        new_idx = idx.round("h")
        out.index = new_idx
        out = out.groupby(out.index).mean()
        return out.sort_index()

    # 已经是整点：直接返回（不 resample）
    return out.sort_index()
