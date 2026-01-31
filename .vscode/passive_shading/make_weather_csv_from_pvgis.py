# make_weather_csv_from_pvgis.py
# -*- coding: utf-8 -*-
"""
将 PVGIS TMY CSV 转换为本项目所需输入文件：
  weather_singapore_hourly.csv

输出列：
  datetime : Asia/Singapore 时区（UTC+8）的时间字符串（tz-aware）
  DNI      : 直射法向辐照度（W/m^2），来自 PVGIS 的 Gb(n)
  DHI      : 水平面散射辐照度（W/m^2），来自 PVGIS 的 Gd(h)
  GHI      : 水平面总辐照度（W/m^2），来自 PVGIS 的 G(h)
  T_out    : 室外温度（°C），来自 PVGIS 的 T2m

说明：
- PVGIS CSV 的时间列为 time(UTC)，格式 YYYYMMDD:HHMM
- 文件头可能包含 Irradiance Time Offset (h): 0.5
- 文件尾部可能附加变量说明行（非时间戳），脚本会自动过滤
"""

from __future__ import annotations

import re
from pathlib import Path
import pandas as pd


TZ_LOCAL = "Asia/Singapore"


def find_header_row(lines: list[str]) -> int:
    """定位数据表头行（以 time(UTC), 开头），返回其 0-based 行号。"""
    for i, line in enumerate(lines):
        s = line.strip().replace(" ", "")
        if s.lower().startswith("time(utc),"):
            return i
    raise ValueError("未找到数据表头行（应以 'time(UTC),' 开头）。")


def read_irradiance_time_offset_hours(lines: list[str]) -> float:
    """从文件头读取 Irradiance Time Offset (h): x；找不到则返回 0。"""
    pat = re.compile(r"Irradiance Time Offset \(h\):\s*([-\d.]+)")
    for line in lines[:120]:
        m = pat.search(line)
        if m:
            return float(m.group(1))
    return 0.0


def pick_time_col(df: pd.DataFrame) -> str:
    """选择时间列名：优先 time(UTC)，否则模糊匹配包含 time & utc 的列。"""
    if "time(UTC)" in df.columns:
        return "time(UTC)"
    for c in df.columns:
        cc = c.lower().replace(" ", "")
        if ("time" in cc) and ("utc" in cc):
            return c
    raise ValueError(f"未找到时间列。现有列名：{list(df.columns)}")


def parse_pvgis_time_utc(series: pd.Series) -> pd.DatetimeIndex:
    """
    PVGIS 时间严格格式：YYYYMMDD:HHMM
    用 errors='coerce' 将异常值置为 NaT，方便后续过滤/报错。
    """
    s = series.astype(str).str.strip()
    t = pd.to_datetime(s, format="%Y%m%d:%H%M", utc=True, errors="coerce")
    return pd.DatetimeIndex(t)


def locate_project_root(script_dir: Path) -> Path:
    """
    从脚本目录向上寻找名为 ICM_CODE 的目录作为项目根。
    找不到则退回到 parents[2]（适配你的当前目录结构）。
    """
    for p in script_dir.parents:
        if p.name.upper() == "ICM_CODE":
            return p
    return script_dir.parents[2]


def _require_cols(df: pd.DataFrame, cols: list[str]) -> None:
    """必需列检查（集中写，报错更清晰）。"""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"缺少列：{missing}。现有列名：{list(df.columns)}")


def main():
    # ---------- 路径策略 ----------
    script_dir = Path(__file__).resolve().parent
    project_root = locate_project_root(script_dir)

    print("[INFO] script_dir    =", script_dir)
    print("[INFO] project_root  =", project_root)

    candidates = [
        project_root / "pvgis_tmy.csv",
        script_dir / "pvgis_tmy.csv",
    ]

    in_path = None
    for p in candidates:
        if p.exists():
            in_path = p
            break

    if in_path is None:
        raise FileNotFoundError(
            "找不到输入文件 pvgis_tmy.csv。\n"
            "请将你下载的 PVGIS CSV 重命名为 pvgis_tmy.csv，并放在以下任一位置：\n"
            f"  1) {candidates[0]}\n"
            f"  2) {candidates[1]}\n"
            "特别注意不要出现 pvgis_tmy.csv.csv 这种双后缀。\n"
        )

    out_path = project_root / "weather_singapore_hourly.csv"

    # ---------- 读取原始文件文本（用于找表头/offset） ----------
    raw_text = in_path.read_text(encoding="utf-8", errors="ignore")
    lines = raw_text.splitlines()

    header_row = find_header_row(lines)
    offset_h = read_irradiance_time_offset_hours(lines)

    print(f"[INFO] 输入文件: {in_path}")
    print(f"[INFO] 数据表头行号(0-based): {header_row}")
    print(f"[INFO] Irradiance Time Offset (h): {offset_h}")

    # ---------- 读取数据表 ----------
    df = pd.read_csv(in_path, skiprows=header_row)
    df.columns = [c.strip() for c in df.columns]

    time_col = pick_time_col(df)

    # 必需列检查：新增 Gd(h), G(h)
    _require_cols(df, ["Gb(n)", "Gd(h)", "G(h)", "T2m"])

    # ---------- 过滤尾注/说明行：只保留形如 YYYYMMDD:HHMM 的时间戳行 ----------
    time_raw = df[time_col].astype(str).str.strip()
    mask_time = time_raw.str.match(r"^\d{8}:\d{4}$", na=False)

    if not mask_time.all():
        bad = time_raw[~mask_time].head(8).tolist()
        print(f"[WARN] 检测到非时间戳行 {(~mask_time).sum()} 条，将自动过滤。示例：{bad}")
        df = df.loc[mask_time].copy()

    # PVGIS TMY 应为 8760 行；若多则截断
    if len(df) > 8760:
        print(f"[WARN] 数据行数 {len(df)} > 8760，已截断为前 8760 行")
        df = df.iloc[:8760].copy()
    if len(df) < 8760:
        print(f"[WARN] 数据行数 {len(df)} < 8760（可能缺测或文件非全年）。仍会继续输出，但年统计需谨慎。")

    # ---------- 时间解析：UTC -> (UTC + offset) -> Asia/Singapore ----------
    t_utc = parse_pvgis_time_utc(df[time_col])
    if t_utc.isna().any():
        n_bad = int(t_utc.isna().sum())
        bad_vals = df.loc[t_utc.isna(), time_col].astype(str).head(5).tolist()
        raise ValueError(f"时间解析失败 {n_bad} 条，示例：{bad_vals}")

    if abs(offset_h) > 1e-12:
        t_utc = t_utc + pd.to_timedelta(offset_h, unit="h")

    t_local = t_utc.tz_convert(TZ_LOCAL)

    # ---------- 字段映射与清洗 ----------
    # DNI: Gb(n) 直射法向
    DNI = pd.to_numeric(df["Gb(n)"], errors="coerce").fillna(0.0).clip(lower=0.0)

    # DHI: Gd(h) 水平散射
    DHI = pd.to_numeric(df["Gd(h)"], errors="coerce").fillna(0.0).clip(lower=0.0)

    # GHI: G(h) 水平总辐照
    GHI = pd.to_numeric(df["G(h)"], errors="coerce").fillna(0.0).clip(lower=0.0)

    # 温度
    T_out = pd.to_numeric(df["T2m"], errors="coerce")

    out = pd.DataFrame({
        # 写 tz-aware 字符串，避免后续被误当“无时区本地时间”
        "datetime": t_local.astype(str),
        "DNI": DNI.astype(float),
        "DHI": DHI.astype(float),
        "GHI": GHI.astype(float),
        "T_out": T_out.astype(float),
    }).dropna(subset=["T_out"])

    # 排序/去重（稳健性）
    out = out.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)

    # ---------- 输出 ----------
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"[OK] 已生成: {out_path}")
    print(f"[OK] 行数: {len(out)}")
    print("[OK] 预览前 5 行：")
    print(out.head(5))
    print("[OK] 列名：", list(out.columns))


if __name__ == "__main__":
    main()
