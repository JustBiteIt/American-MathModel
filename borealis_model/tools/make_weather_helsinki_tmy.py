# borealis_model/tools/make_weather_helsinki_tmy.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests


PVGIS_TMY_API = "https://re.jrc.ec.europa.eu/api/tmy"  # PVGIS TMY endpoint  :contentReference[oaicite:2]{index=2}


def _parse_pvgis_tmy_csv(text: str) -> pd.DataFrame:
    """
    PVGIS TMY CSV 格式前面有若干 meta 行 + 12 行 months_selected，
    真正数据部分从以 'time(UTC)' 开头的 header 行开始，后面固定 8760 行。
    CSV header 典型为：time(UTC),T2m,RH,G(h),Gb(n),Gd(h),...  :contentReference[oaicite:3]{index=3}
    """
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("time(UTC)"):
            start = i
            break
    if start is None:
        raise ValueError("PVGIS CSV 中找不到 'time(UTC)' 表头行，无法解析。")

    # header + 8760 行数据
    csv_block = "\n".join(lines[start : start + 1 + 8760])
    df = pd.read_csv(io.StringIO(csv_block))
    return df


def download_pvgis_tmy(lat: float, lon: float, outputformat: str = "csv", timeout: int = 60) -> pd.DataFrame:
    params = {"lat": lat, "lon": lon, "outputformat": outputformat}
    r = requests.get(PVGIS_TMY_API, params=params, timeout=timeout)
    r.raise_for_status()
    raw = _parse_pvgis_tmy_csv(r.text)

    # 时间列：time(UTC) 形如 19900101:0000（PVGIS/解析器用这个格式） :contentReference[oaicite:4]{index=4}
    t_utc = pd.to_datetime(raw["time(UTC)"], format="%Y%m%d:%H%M", utc=True)

    out = pd.DataFrame(
        {
            "datetime": t_utc.astype(str),     # 先保留 UTC +00:00（最稳，不引入 DST 麻烦）
            "DNI": raw["Gb(n)"],               # DNI  :contentReference[oaicite:5]{index=5}
            "DHI": raw["Gd(h)"],               # DHI  :contentReference[oaicite:6]{index=6}
            "GHI": raw["G(h)"],                # GHI  :contentReference[oaicite:7]{index=7}
            "T_out": raw["T2m"],               # Temperature  :contentReference[oaicite:8]{index=8}
        }
    )

    # 清洗：非负
    for c in ["DNI", "DHI", "GHI"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).clip(lower=0.0)
    out["T_out"] = pd.to_numeric(out["T_out"], errors="coerce")

    return out


def main():
    # Helsinki（你也可以按需要改坐标）
    lat, lon = 60.1699, 24.9384
    out_path = Path(r"D:\ICM_CODE\weather_helsinki_tmy_hourly.csv")

    df = download_pvgis_tmy(lat, lon)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("✅ Saved:", out_path)
    print("rows:", len(df), "start:", df["datetime"].iloc[0], "end:", df["datetime"].iloc[-1])


if __name__ == "__main__":
    main()
