from __future__ import annotations

from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_root = None
for parent in Path(__file__).resolve().parents:
    if (parent / "borealis_model").exists():
        _root = parent
        break
if _root is not None and str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from borealis_model.data_io import load_weather_csv
from SG_Pareto.config import ANALYSIS_ROOT, SINGAPORE, DELTA_TREND_C, USE_AR1_NOISE, AR1_SEED


def _find_latest_weather_csv() -> Path | None:
    if not ANALYSIS_ROOT.exists():
        return None
    candidates = sorted(ANALYSIS_ROOT.glob("sg_pareto_*/weather/weather_singapore_tmy_hourly.csv"))
    return candidates[-1] if candidates else None


def _ar1_noise(n: int, *, phi: float = 0.8, sigma: float = 0.35, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, sigma, size=n)
    x = np.zeros(n, dtype=float)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + eps[i]
    return x


def _month_ticks_for_year(year: int) -> tuple[list[int], list[str]]:
    dates = pd.date_range(f"{year}-01-01", f"{year}-12-01", freq="MS")
    ticks = [int(d.dayofyear) for d in dates]
    labels = [d.strftime("%b") for d in dates]
    return ticks, labels


def _write_chart_json(out_dir: Path, payload: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "charts_outdoor_temp.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_chart_html(out_dir: Path, payload: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2)
    html = """<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>Outdoor Temperature: Full Year</title>
  <style>
    :root {
      --bg: #ffffff;
      --grid: #e6e6e6;
      --axis: #333333;
      --text: #222222;
      --blue: #4c78a8;
      --orange: #f28e2b;
    }
    body {
      margin: 24px;
      font-family: "Segoe UI", Arial, sans-serif;
      color: var(--text);
      background: var(--bg);
    }
    .header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 12px;
    }
    h2 {
      font-size: 20px;
      font-weight: 600;
      margin: 0;
    }
    .btn {
      border: 1px solid #cfcfcf;
      background: #fafafa;
      padding: 6px 12px;
      cursor: pointer;
      border-radius: 4px;
      font-size: 12px;
    }
    canvas {
      width: 100%;
      max-width: 1100px;
      height: 520px;
      border: 1px solid #f0f0f0;
      background: #fff;
    }
    .note {
      font-size: 12px;
      color: #666;
      margin-top: 6px;
    }
  </style>
</head>
<body>
  <div class=\"header\">
    <h2>Outdoor Temperature: Full Year</h2>
    <button class=\"btn\" id=\"btn-download\">Download PNG</button>
  </div>
  <canvas id=\"chart\" width=1100 height=520></canvas>
  <div class=\"note\">Baseline vs 2040 scenario (daily mean). Month axis shared.</div>

  <script>
    const DATA = __PAYLOAD__;
    const canvas = document.getElementById("chart");
    const ctx = canvas.getContext("2d");

    const W = canvas.width;
    const H = canvas.height;
    const pad = { left: 70, right: 30, top: 50, bottom: 60 };
    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top - pad.bottom;

    const baseline = DATA.series.baseline.filter(d => isFinite(d.y));
    const scenario = DATA.series.scenario.filter(d => isFinite(d.y));
    const ticks = DATA.month_ticks;

    const allY = baseline.map(d => d.y).concat(scenario.map(d => d.y));
    const allX = baseline.map(d => d.x).concat(scenario.map(d => d.x));
    let yMin = Math.min(...allY);
    let yMax = Math.max(...allY);
    if (!isFinite(yMin) || !isFinite(yMax) || Math.abs(yMax - yMin) < 1e-6) {
      yMin = 0; yMax = 1;
    }
    let xMin = Math.min(...allX);
    let xMax = Math.max(...allX);
    if (!isFinite(xMin) || !isFinite(xMax) || Math.abs(xMax - xMin) < 1e-6) {
      xMin = 1; xMax = 365;
    }

    const xScale = x => pad.left + (x - xMin) / (xMax - xMin) * plotW;
    const yScale = y => pad.top + plotH - (y - yMin) / (yMax - yMin) * plotH;

    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, W, H);

    // grid + axes
    ctx.strokeStyle = "#e6e6e6";
    ctx.lineWidth = 1;
    const yTicks = 5;
    for (let i = 0; i <= yTicks; i++) {
      const y = pad.top + plotH * i / yTicks;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(pad.left + plotW, y);
      ctx.stroke();
      const val = yMax - (yMax - yMin) * i / yTicks;
      ctx.fillStyle = "#555";
      ctx.font = "12px Arial";
      ctx.fillText(val.toFixed(1), 10, y + 4);
    }

    ctx.strokeStyle = "#333";
    ctx.beginPath();
    ctx.moveTo(pad.left, pad.top);
    ctx.lineTo(pad.left, pad.top + plotH);
    ctx.lineTo(pad.left + plotW, pad.top + plotH);
    ctx.stroke();

    // x ticks (months)
    ctx.fillStyle = "#555";
    ctx.font = "12px Arial";
    ticks.forEach(t => {
      const x = xScale(t.x);
      ctx.beginPath();
      ctx.moveTo(x, pad.top + plotH);
      ctx.lineTo(x, pad.top + plotH + 4);
      ctx.strokeStyle = "#333";
      ctx.stroke();
      ctx.fillText(t.label, x - 10, pad.top + plotH + 22);
    });

    function drawLine(points, color) {
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      points.forEach((p, i) => {
        const x = xScale(p.x);
        const y = yScale(p.y);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }

    drawLine(baseline, "#4c78a8");
    drawLine(scenario, "#f28e2b");

    // legend
    ctx.fillStyle = "#222";
    ctx.font = "12px Arial";
    const lx = pad.left + plotW - 210;
    const ly = pad.top + 10;
    ctx.fillRect(lx, ly, 12, 3);
    ctx.fillText("Baseline (daily mean)", lx + 18, ly + 6);
    ctx.fillStyle = "#f28e2b";
    ctx.fillRect(lx, ly + 16, 12, 3);
    ctx.fillStyle = "#222";
    ctx.fillText("2040 scenario (daily mean)", lx + 18, ly + 22);

    // labels
    ctx.fillStyle = "#222";
    ctx.font = "13px Arial";
    ctx.fillText("Month", pad.left + plotW / 2 - 20, H - 15);
    ctx.save();
    ctx.translate(16, pad.top + plotH / 2 + 40);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Outdoor Air Temperature (degC)", 0, 0);
    ctx.restore();

    document.getElementById("btn-download").addEventListener("click", () => {
      const link = document.createElement("a");
      link.download = "outdoor_temperature_full_year.png";
      link.href = canvas.toDataURL("image/png");
      link.click();
    });
  </script>
</body>
</html>
"""
    html_path = out_dir / "index_outdoor_temp.html"
    html_path.write_text(html.replace("__PAYLOAD__", data), encoding="utf-8")
    return html_path


def build_plot(
    df_weather: pd.DataFrame,
    *,
    out_dir: Path,
    delta_trend_C: float,
    use_ar1_noise: bool,
    ar1_seed: int,
) -> dict[str, Path]:
    plots_dir = out_dir / "plots"
    data_dir = out_dir / "data"
    echarts_dir = out_dir / "echarts"
    plots_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    echarts_dir.mkdir(parents=True, exist_ok=True)

    times = df_weather.index
    if getattr(times, "tz", None) is not None:
        times_local = times.tz_localize(None)
    else:
        times_local = times

    T_base = df_weather["T_out"].to_numpy(float)
    noise = _ar1_noise(len(T_base), seed=ar1_seed) if use_ar1_noise else np.zeros(len(T_base), dtype=float)
    T_2040 = T_base + float(delta_trend_C) + noise

    doy_all = times_local.dayofyear.to_numpy(int)
    df_daily = pd.DataFrame(
        {
            "doy": doy_all,
            "T_out_baseline_C": T_base,
            "T_out_2040_C": T_2040,
        }
    )
    # drop leap day to keep 365-day axis
    df_daily = df_daily[df_daily["doy"] != 366]
    daily = df_daily.groupby("doy")["T_out_baseline_C"].mean()
    daily_2040 = df_daily.groupby("doy")["T_out_2040_C"].mean()

    daily = daily.reindex(range(1, 366)).astype(float)
    daily_2040 = daily_2040.reindex(range(1, 366)).astype(float)

    dummy_dates = pd.date_range("2001-01-01", periods=365, freq="D")
    daily_df = pd.DataFrame(
        {
            "date_local": dummy_dates.date.astype(str),
            "T_out_baseline_C": daily.values,
            "T_out_2040_C": daily_2040.values,
        }
    )
    daily_csv = data_dir / "outdoor_temperature_daily.csv"
    daily_df.to_csv(daily_csv, index=False)

    doy = daily.index.to_numpy(int)
    doy_2040 = daily_2040.index.to_numpy(int)

    # consistency check
    delta = daily_2040.values - daily.values
    delta = delta[np.isfinite(delta)]
    if delta.size == 0:
        raise ValueError("日均温度序列为空或全为 NaN，无法进行一致性校验。")
    mean_delta = float(np.mean(delta))
    max_abs_delta = float(np.max(np.abs(delta)))
    print(f"一致性校验：mean_delta={mean_delta:.4f}, max_abs_delta={max_abs_delta:.4f}")
    if not use_ar1_noise:
        tol = 1e-4
        if abs(mean_delta - float(delta_trend_C)) > tol or abs(max_abs_delta - float(delta_trend_C)) > tol:
            raise ValueError("2040 与 baseline 差异不符合 DELTA_TREND_C，可能列引用或聚合步骤有误。")

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#333333",
            "xtick.color": "#666666",
            "ytick.color": "#666666",
            "grid.color": "#e6e6e6",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.8,
            "font.size": 11,
        }
    )

    fig, ax = plt.subplots(figsize=(10, 4.6))
    baseline_vals = daily_df["T_out_baseline_C"].to_numpy(float)
    scenario_vals = daily_df["T_out_2040_C"].to_numpy(float)
    ax.plot(doy, baseline_vals, color="#4c78a8", label="Baseline (daily mean)", linewidth=1.4)
    ax.plot(doy_2040, scenario_vals, color="#f28e2b", label="2040 scenario (daily mean)", linewidth=1.4)
    ax.set_xlabel("Month")
    ax.set_ylabel("Outdoor Air Temperature (degC)")
    ax.set_title("Outdoor Temperature: Full Year")
    ax.grid(True, axis="y")
    ax.legend(frameon=False, loc="upper right")

    ticks, labels = _month_ticks_for_year(2001)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlim(1, 365)

    png_path = plots_dir / "outdoor_temperature_full_year_with_zoom.png"
    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    plt.close(fig)

    month_ticks = [{"x": int(t), "label": lbl} for t, lbl in zip(ticks, labels)]
    series_baseline = [
        {"x": int(x), "y": float(y)}
        for x, y in zip(doy, daily.values)
        if np.isfinite(y)
    ]
    series_scenario = [
        {"x": int(x), "y": float(y)}
        for x, y in zip(doy_2040, daily_2040.values)
        if np.isfinite(y)
    ]
    payload = {
        "meta": {
            "title": "Outdoor Temperature: Full Year",
            "x_min": int(min(doy.min(), doy_2040.min())),
            "x_max": int(max(doy.max(), doy_2040.max())),
        },
        "month_ticks": month_ticks,
        "series": {
            "baseline": series_baseline,
            "scenario": series_scenario,
        },
    }
    json_path = _write_chart_json(echarts_dir, payload)
    html_path = _write_chart_html(echarts_dir, payload)

    return {
        "png": png_path,
        "daily_csv": daily_csv,
        "json": json_path,
        "html": html_path,
    }


def main() -> None:
    weather_csv = _find_latest_weather_csv()
    if weather_csv is None or not weather_csv.exists():
        raise SystemExit("???????????? main_sg_pareto.py ?? weather ???")

    out_dir = ANALYSIS_ROOT / "outdoor_temperature_latest"
    df = load_weather_csv(str(weather_csv), tz=SINGAPORE.tz)

    result = build_plot(
        df,
        out_dir=out_dir,
        delta_trend_C=float(DELTA_TREND_C),
        use_ar1_noise=bool(USE_AR1_NOISE),
        ar1_seed=int(AR1_SEED),
    )

    baseline_mean = float(np.mean(df["T_out"].to_numpy(float)))
    baseline_max = float(np.max(df["T_out"].to_numpy(float)))
    scenario = df["T_out"].to_numpy(float) + float(DELTA_TREND_C)
    if USE_AR1_NOISE:
        scenario = scenario + _ar1_noise(len(scenario), seed=int(AR1_SEED))
    scenario_mean = float(np.mean(scenario))
    scenario_max = float(np.max(scenario))

    print("Outdoor temperature plot generated:")
    print(f"PNG: {result['png']}")
    print(f"daily.csv: {result['daily_csv']}")
    print(f"JSON: {result['json']}")
    print(f"HTML: {result['html']}")
    print(f"baseline mean/max: {baseline_mean:.2f} degC / {baseline_max:.2f} degC")
    print(f"2040 mean/max: {scenario_mean:.2f} degC / {scenario_max:.2f} degC")


if __name__ == "__main__":
    main()
