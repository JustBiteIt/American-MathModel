# baseline_utils.py
# -*- coding: utf-8 -*-
"""
统一 Baseline 口径（秦始皇统一度量衡版）
-------------------------------------------------
1) 统一读入 weather_singapore_hourly.csv
2) 将“跨年拼接的 TMY 月份”重建为一个连续的“虚拟年”(target_year)，保证 8760 小时连续
3) 统一 Baseline 的定义：
   - 北/南挑檐深度 dN=dS=0（无挑檐）
   - 东/西百叶 cutoff 取当前 bounds 的上界（尽量“不开遮挡”）
4) 提供 df 指纹 fingerprint，确保所有脚本用的是同一套 df_tmy
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import numpy as np
import pandas as pd


def find_weather_csv(script_dir: Path) -> Path:
    """
    查找 weather_singapore_hourly.csv 的路径：
    优先：项目根目录 D:\ICM_CODE\weather_singapore_hourly.csv
    其次：脚本目录同级 / 上级
    """
    candidates = [
        script_dir.parent.parent / "weather_singapore_hourly.csv",  # ...\ICM_CODE\weather_singapore_hourly.csv
        script_dir / "weather_singapore_hourly.csv",
        script_dir.parent / "weather_singapore_hourly.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "找不到 weather_singapore_hourly.csv。请确认文件存在于以下任一位置：\n"
        + "\n".join([f"  - {c}" for c in candidates])
    )


def _ensure_datetime_index(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    """确保 df.index 是 tz-aware DatetimeIndex，并转换到 tz。"""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df.index 必须是 DatetimeIndex。请先用 load_weather_csv 读入。")

    if df.index.tz is None:
        df = df.copy()
        df.index = df.index.tz_localize(tz)
    else:
        df = df.tz_convert(tz)

    return df


def _align_to_hour_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    """
    处理 PVGIS 常见的“半小时偏移”（例如全是 :30）。
    目标：把时间对齐到整点小时，避免后续 reindex(整点) 对不上。
    """
    idx = df.index
    mins = np.unique(idx.minute)

    # 情况1：所有分钟都一样且不是 0 -> 直接整体平移到整点
    if len(mins) == 1 and int(mins[0]) != 0:
        shift_min = int(mins[0])
        df = df.copy()
        df.index = df.index - pd.Timedelta(minutes=shift_min)

    # 情况2：分钟不一致 -> 用 floor('h') 对齐（可能会产生重复）
    elif len(mins) > 1:
        df = df.copy()
        df.index = df.index.floor("h")

    # 去重（对齐后可能出现同一小时重复）
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


def build_tmy_virtual_year(
    df_raw: pd.DataFrame,
    tz: str,
    target_year: int = 2023,
    freq: str = "1h",
    strict: bool = True,
) -> pd.DataFrame:
    """
    把“跨年份拼起来的 TMY 月份”重建成一个连续的虚拟年：
    - 先把时间对齐到整点（处理 :30 偏移）
    - 保留每条记录的 月-日-时(分)
    - 年份统一改成 target_year
    - 最终强制 reindex 成 [target_year-01-01 00:00, target_year+1-01-01 00:00) 的 8760 小时连续序列

    strict=True：
      - 如果重建后不是完整 8760 小时，会直接报错（强制你修数据/流程）
    strict=False：
      - 缺的小时用时间插值补齐（调试用，不推荐论文最终版）
    """
    df = _ensure_datetime_index(df_raw, tz).copy()

    # 只保留我们关心的列
    keep_cols = [c for c in ["DNI", "DHI", "GHI", "T_out"] if c in df.columns]
    if "DNI" not in keep_cols or "T_out" not in keep_cols:
        raise ValueError(f"df_raw 至少要包含 DNI 和 T_out。当前列：{list(df.columns)}")
    df = df[keep_cols].copy()

    # 清理
    for c in ["DNI", "DHI", "GHI"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).clip(lower=0.0)
    df["T_out"] = pd.to_numeric(df["T_out"], errors="coerce")
    df = df.dropna(subset=["T_out"])

    # 关键：先把时间对齐到整点（否则后面整点 reindex 会全 NaN）
    df = _align_to_hour_if_needed(df)

    # 用“月-日-时-分”重建虚拟年时间轴
    idx = df.index
    naive = pd.to_datetime(
        {
            "year": np.full(len(idx), target_year, dtype=int),
            "month": idx.month.to_numpy(),
            "day": idx.day.to_numpy(),
            "hour": idx.hour.to_numpy(),
            "minute": idx.minute.to_numpy(),
        },
        errors="coerce",
    )

    # 这里 naive 可能是 Series（不同 pandas 版本行为不同），统一转 DatetimeIndex 再 tz_localize
    new_idx = pd.DatetimeIndex(naive)
    if new_idx.isna().any():
        bad_n = int(new_idx.isna().sum())
        raise ValueError(f"虚拟年时间构造出现无效日期/时间 {bad_n} 条（可能包含 2/29 等）。")

    new_idx = new_idx.tz_localize(tz)

    df.index = new_idx
    df = df.sort_index()

    # 去重（稳健）
    df = df[~df.index.duplicated(keep="first")]

    # 目标：连续整点 8760 小时
    start = pd.Timestamp(f"{target_year}-01-01 00:00:00", tz=tz)
    end = pd.Timestamp(f"{target_year+1}-01-01 00:00:00", tz=tz)
    full_idx = pd.date_range(start, end, freq=freq, inclusive="left")

    df = df.reindex(full_idx)

    if strict:
        if df.isna().any().any():
            na_counts = df.isna().sum()
            bad = na_counts[na_counts > 0]
            raise ValueError(
                "TMY 虚拟年重建失败：出现缺失小时/缺失值。\n"
                f"缺失统计：\n{bad}\n"
                "这通常表示：你的原始时间不是整点，或原始文件本身缺小时。\n"
                "你可以先 strict=False 调试，但论文最终建议 strict=True。"
            )
    else:
        df = df.interpolate(method="time").ffill().bfill()

    return df


def baseline_x_from_bounds(bounds) -> tuple[float, float, float, float]:
    """
    统一 Baseline 定义：
      - dN=0, dS=0：无挑檐
      - etaE/etaW 取 bounds 上界：百叶尽量“放开”，最接近无遮阳
    """
    return (0.0, 0.0, float(bounds.etaE_bounds_rad[1]), float(bounds.etaW_bounds_rad[1]))


def df_fingerprint(df: pd.DataFrame) -> str:
    """
    给 df 生成一个“指纹”：df 内容变了，指纹就变。
    """
    h = hashlib.md5()
    idx_ns = df.index.view("int64")
    h.update(idx_ns.tobytes())

    cols = [c for c in ["DNI", "DHI", "GHI", "T_out"] if c in df.columns]
    for c in cols:
        arr = np.asarray(df[c].to_numpy(dtype=float))
        arr = np.nan_to_num(arr, nan=-9999.0, posinf=1e9, neginf=-1e9)
        h.update(arr.tobytes())

    return h.hexdigest()[:12]
