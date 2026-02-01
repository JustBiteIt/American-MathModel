# borealis_model/data_io.py
# -*- coding: utf-8 -*-
import pandas as pd

REQUIRED_COLS = ["datetime", "DNI", "DHI", "GHI", "T_out"]

def load_weather_csv(path: str, tz: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"缺少列：{missing}。\n"
            f"当前列：{list(df.columns)}\n"
            "需要：datetime,DNI,DHI,GHI,T_out"
        )

    t = pd.to_datetime(df["datetime"])
    if t.dt.tz is None:
        t = t.dt.tz_localize(tz)
    else:
        t = t.dt.tz_convert(tz)

    out = df.copy()
    out.index = t
    out = out.drop(columns=["datetime"]).sort_index()

    for col in ["DNI", "DHI", "GHI"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0).clip(lower=0.0)

    out["T_out"] = pd.to_numeric(out["T_out"], errors="coerce")
    out = out.dropna(subset=["T_out"])

    return out

def ensure_hourly(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample("1h").mean()
