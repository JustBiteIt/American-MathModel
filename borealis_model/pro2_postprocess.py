# -*- coding: utf-8 -*-
"""
Pro2 postprocess:
- Read hourly CSVs (timeseries_*.csv)
- Aggregate monthly E_h (kWh/month)
- Build E_sol KDE (occupied daytime)
- Extract hottest 14-day Ti segment (for T_max line plot)
- Export pro2_charts.json for ECharts dashboard

Run:
  python D:\ICM_CODE\pro2_postprocess.py ^
    --in_dir D:\ICM_RESULT\pro2 ^
    --out_json D:\ICM_RESULT\pro2\pro2_charts.json ^
    --tmax_plot 26 ^
    --dt_hours 1
"""

from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

def to_ms(dt: pd.DatetimeIndex) -> np.ndarray:
    return (dt.view("int64") // 1_000_000).astype(np.int64)

def monthly_sum(times: pd.DatetimeIndex, values: np.ndarray) -> List[float]:
    out = []
    for m in range(1, 13):
        out.append(float(values[times.month == m].sum()))
    return out

def rolling_hottest_window_idx(T: np.ndarray, window: int) -> int:
    if len(T) <= window:
        return 0
    s = pd.Series(T)
    idx = int(s.rolling(window=window, min_periods=window).mean().idxmax())
    return max(0, idx - window + 1)

def silverman_bw(samples: np.ndarray) -> float:
    n = len(samples)
    if n < 2:
        return 1.0
    std = float(np.std(samples, ddof=1))
    if std <= 1e-9:
        return 1.0
    return 1.06 * std * (n ** (-1/5))

def kde_gaussian(x_grid: np.ndarray, samples: np.ndarray, max_n: int = 20000) -> np.ndarray:
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    if len(samples) == 0:
        return np.zeros_like(x_grid, dtype=float)

    if len(samples) > max_n:
        idx = np.linspace(0, len(samples)-1, max_n).astype(int)
        samples = samples[idx]

    bw = max(silverman_bw(samples), 1e-6)
    diff = (x_grid[:, None] - samples[None, :]) / bw
    pdf = np.exp(-0.5 * diff**2).mean(axis=1) / (bw * np.sqrt(2*np.pi))
    return pdf

def parse_case_tag(filename: str) -> Tuple[str, str]:
    """
    Expect: timeseries_<caseId>_<tag>.csv
    """
    m = re.match(r"^timeseries_(.+)_(.+)\.csv$", filename)
    if not m:
        raise ValueError(f"Bad filename: {filename}")
    return m.group(1), m.group(2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, required=True)
    ap.add_argument("--out_json", type=str, required=True)
    ap.add_argument("--tmax_plot", type=float, default=26.0)
    ap.add_argument("--dt_hours", type=float, default=None, help="If not set, infer from time column.")
    ap.add_argument("--E_min", type=float, default=7.89)
    ap.add_argument("--E_glare", type=float, default=87.24)
    ap.add_argument("--occ_start", type=int, default=8)
    ap.add_argument("--occ_end", type=int, default=18)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("timeseries_*.csv"))
    if not files:
        raise FileNotFoundError(f"No timeseries_*.csv in {in_dir}")

    charts: Dict[str, Any] = {
        "meta": {
            "tmax_plot_C": float(args.tmax_plot),
            "E_min": float(args.E_min),
            "E_glare": float(args.E_glare),
            "months": MONTHS,
        },
        "cases": {}
    }

    for fp in files:
        case_id, tag = parse_case_tag(fp.name)

        df = pd.read_csv(fp)
        if "time" in df.columns:
            t = pd.to_datetime(df["time"], utc=False)
        elif "time_ms" in df.columns:
            t = pd.to_datetime(df["time_ms"].astype("int64"), unit="ms", utc=True)
        else:
            raise ValueError(f"{fp.name} must have 'time' or 'time_ms' column.")

        # ensure tz-aware ok; ECharts only needs ms
        times = pd.DatetimeIndex(t)

        # dt_hours
        if args.dt_hours is not None:
            dt_h = float(args.dt_hours)
        else:
            dt_sec = np.median(np.diff(times.view("int64"))) / 1e9
            dt_h = float(dt_sec / 3600.0)

        # required
        if "Ti_C" not in df.columns or "Phi_h_W" not in df.columns:
            raise ValueError(f"{fp.name} must include Ti_C and Phi_h_W")

        Ti = df["Ti_C"].to_numpy(dtype=float)
        Phi_h = df["Phi_h_W"].to_numpy(dtype=float)

        Eh_hour_kWh = Phi_h * dt_h / 1000.0
        monthly_Eh = monthly_sum(times, Eh_hour_kWh)

        # OH/Hoh (use tmax_plot for paper)
        tmax = float(args.tmax_plot)
        exceed = np.maximum(Ti - tmax, 0.0)
        OH = float(exceed.sum() * dt_h)
        Hoh = float((Ti > tmax).sum() * dt_h)

        monthly_OH = monthly_sum(times, exceed * dt_h)
        monthly_Hoh = monthly_sum(times, (Ti > tmax).astype(float) * dt_h)

        # hottest 14d
        window = 14 * 24
        sidx = rolling_hottest_window_idx(Ti, window)
        eidx = min(sidx + window, len(Ti))
        hot_t = times[sidx:eidx]
        hot_pairs = [[int(ms), float(v)] for ms, v in zip(to_ms(hot_t), Ti[sidx:eidx])]

        # E_sol distribution (optional but recommended for your 2nd figure)
        if "E_sol_Wm2" in df.columns:
            Esol = df["E_sol_Wm2"].to_numpy(dtype=float)
            if "occupied_daytime" in df.columns:
                occ = df["occupied_daytime"].to_numpy(dtype=int).astype(bool)
            else:
                occ = (times.hour >= args.occ_start) & (times.hour < args.occ_end)

            x_grid = np.linspace(0.0, 180.0, 400)
            pdf = kde_gaussian(x_grid, Esol[occ])
            esol_density = {
                "x": x_grid.tolist(),
                "pdf": pdf.tolist(),
                "E_min": float(args.E_min),
                "E_glare": float(args.E_glare),
            }
        else:
            esol_density = None

        charts["cases"].setdefault(case_id, {"scenarios": {}})
        charts["cases"][case_id]["scenarios"][tag] = {
            "monthly": {
                "months": MONTHS,
                "Eh_kWh": monthly_Eh,
                "OH_degC_h": monthly_OH,
                "Hoh_h": monthly_Hoh,
            },
            "metrics": {
                "Eh_kWh_total": float(np.sum(Eh_hour_kWh)),
                "OH_degC_h": OH,
                "Hoh_h": Hoh,
                "Ti_max_C": float(np.max(Ti)) if len(Ti) else float("nan"),
            },
            "Ti_hot_14d": hot_pairs,
            "Esol_density": esol_density,
        }

    out_json.write_text(json.dumps(charts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] wrote {out_json}")

if __name__ == "__main__":
    main()
