from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class OutdoorTempOutputs:
    png_path: Path
    daily_csv: Path
    zoom_csv: Path
    echarts_json: Path | None
    echarts_html: Path | None
    zoom_start: str
    zoom_end: str
    baseline_mean_C: float
    baseline_max_C: float
    scenario_mean_C: float
    scenario_max_C: float


def _ar1_noise(n: int, *, phi: float = 0.8, sigma: float = 0.35, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, sigma, size=n)
    x = np.zeros(n, dtype=float)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + eps[i]
    return x


def _write_echarts_min_js(out_dir: Path) -> None:
    js = r"""(function(){
  function clamp(v, a, b){ return Math.max(a, Math.min(b, v)); }
  function init(dom){
    const canvas = document.createElement("canvas");
    const rect = dom.getBoundingClientRect();
    const w = Math.max(320, rect.width || dom.clientWidth || 900);
    const h = Math.max(240, rect.height || dom.clientHeight || 480);
    canvas.width = w;
    canvas.height = h;
    dom.innerHTML = "";
    dom.appendChild(canvas);
    const ctx = canvas.getContext("2d");
    let option = null;
    function setOption(opt){ option = opt || {}; render(); }
    function render(){
      if(!option) return;
      const grid = option.grid || {};
      const left = grid.left || 70;
      const right = grid.right || 70;
      const top = grid.top || 60;
      const bottom = grid.bottom || 70;
      const plotW = w - left - right;
      const plotH = h - top - bottom;
      ctx.clearRect(0,0,w,h);
      ctx.fillStyle = "#fff";
      ctx.fillRect(0,0,w,h);

      const series = option.series || [];
      let xs = [], ys = [];
      series.forEach(s => {
        (s.data || []).forEach(p => {
          if(p && p.length >= 2 && isFinite(p[0]) && isFinite(p[1])){
            xs.push(p[0]); ys.push(p[1]);
          }
        });
      });
      let xmin = Math.min.apply(null, xs);
      let xmax = Math.max.apply(null, xs);
      let ymin = Math.min.apply(null, ys);
      let ymax = Math.max.apply(null, ys);
      if(!isFinite(xmin) || !isFinite(xmax)){ xmin = 0; xmax = 1; }
      if(!isFinite(ymin) || !isFinite(ymax)){ ymin = 0; ymax = 1; }
      if(xmax - xmin < 1e-9){ xmax = xmin + 1; }
      if(ymax - ymin < 1e-9){ ymax = ymin + 1; }

      function xScale(x){ return left + (x - xmin) / (xmax - xmin) * plotW; }
      function yScale(y){ return top + plotH - (y - ymin) / (ymax - ymin) * plotH; }

      ctx.strokeStyle = "#222";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(left, top);
      ctx.lineTo(left, top + plotH);
      ctx.lineTo(left + plotW, top + plotH);
      ctx.stroke();

      ctx.fillStyle = "#222";
      ctx.font = "12px Arial, sans-serif";
      const nt = 5;
      for(let i=0;i<=nt;i++){
        const tx = left + plotW * i / nt;
        const vx = xmin + (xmax - xmin) * i / nt;
        ctx.strokeStyle = "#ddd";
        ctx.beginPath();
        ctx.moveTo(tx, top);
        ctx.lineTo(tx, top + plotH);
        ctx.stroke();
        ctx.fillStyle = "#222";
        ctx.fillText(vx.toFixed(0), tx - 10, top + plotH + 18);
      }
      for(let i=0;i<=nt;i++){
        const ty = top + plotH - plotH * i / nt;
        const vy = ymin + (ymax - ymin) * i / nt;
        ctx.strokeStyle = "#ddd";
        ctx.beginPath();
        ctx.moveTo(left, ty);
        ctx.lineTo(left + plotW, ty);
        ctx.stroke();
        ctx.fillStyle = "#222";
        ctx.fillText(vy.toFixed(2), left - 50, ty + 4);
      }

      if(option.xAxis && option.xAxis.name){
        ctx.fillText(option.xAxis.name, left + plotW/2 - 120, h - 10);
      }
      if(option.yAxis && option.yAxis.name){
        ctx.save();
        ctx.translate(16, top + plotH/2 + 60);
        ctx.rotate(-Math.PI/2);
        ctx.fillText(option.yAxis.name, 0, 0);
        ctx.restore();
      }

      const colors = ["#3b82f6", "#ef4444", "#10b981", "#f59e0b"];
      series.forEach((s, idx) => {
        const col = colors[idx % colors.length];
        const size = s.symbolSize || 2;
        const data = s.data || [];
        if(s.type === "line"){
          ctx.strokeStyle = col;
          ctx.lineWidth = (s.lineStyle && s.lineStyle.width) ? s.lineStyle.width : 2;
          ctx.beginPath();
          let started = false;
          data.forEach(p => {
            if(!p || p.length < 2) return;
            const x = xScale(p[0]);
            const y = yScale(p[1]);
            if(!started){ ctx.moveTo(x, y); started = true; }
            else { ctx.lineTo(x, y); }
          });
          ctx.stroke();
        }
        ctx.fillStyle = col;
        data.forEach(p => {
          if(!p || p.length < 2) return;
          const x = xScale(p[0]);
          const y = yScale(p[1]);
          ctx.beginPath();
          ctx.arc(x, y, size, 0, Math.PI * 2);
          ctx.fill();
        });
      });
    }
    return { setOption };
  }
  window.echarts = { init: init };
})();"""
    (out_dir / "echarts.min.js").write_text(js, encoding="utf-8")


def _write_echarts_html(html_path: Path, data: Dict[str, Any]) -> None:
    payload = json.dumps(data, indent=2)
    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Outdoor Temperature</title>
  <script src="./echarts.min.js"></script>
  <style>
    body { font-family: Arial, sans-serif; margin: 20px; }
    .chart { width: 100%; height: 480px; margin-bottom: 28px; }
  </style>
</head>
<body>
  <h2>Daily Outdoor Temperature: Baseline vs 2040</h2>
  <div id="chart-daily" class="chart"></div>
  <h2>Zoom Window (Hourly)</h2>
  <div id="chart-zoom" class="chart"></div>

  <script>
    const DATA = __PAYLOAD__;
    const daily = DATA.daily;
    const zoom = DATA.zoom;

    const dailyBase = daily.map(r => [r.day_index, r.T_out_baseline_C]);
    const daily2040 = daily.map(r => [r.day_index, r.T_out_2040_C]);
    const zoomBase = zoom.map(r => [r.hour_index, r.T_out_baseline_C]);
    const zoom2040 = zoom.map(r => [r.hour_index, r.T_out_2040_C]);

    const c1 = echarts.init(document.getElementById("chart-daily"));
    c1.setOption({
      grid: { left: 70, right: 70, top: 50, bottom: 70, containLabel: true },
      xAxis: { name: "Day index (local)", nameGap: 30 },
      yAxis: { name: "Outdoor Air Temperature (degC)", nameGap: 35 },
      series: [
        { name: "Baseline", type: "line", data: dailyBase, symbolSize: 2 },
        { name: "2040", type: "line", data: daily2040, symbolSize: 2 }
      ]
    });

    const c2 = echarts.init(document.getElementById("chart-zoom"));
    c2.setOption({
      grid: { left: 70, right: 70, top: 50, bottom: 70, containLabel: true },
      xAxis: { name: "Hour index (zoom)", nameGap: 30 },
      yAxis: { name: "Outdoor Air Temperature (degC)", nameGap: 35 },
      series: [
        { name: "Baseline", type: "line", data: zoomBase, symbolSize: 2 },
        { name: "2040", type: "line", data: zoom2040, symbolSize: 2 }
      ]
    });
  </script>
</body>
</html>
"""
    html_path.write_text(html.replace("__PAYLOAD__", payload), encoding="utf-8")


def build_outdoor_temperature_with_zoom(
    df_weather: pd.DataFrame,
    *,
    delta_trend_C: float,
    use_ar1_noise: bool,
    ar1_seed: int,
    out_dir: Path,
    enable_echarts: bool = True,
) -> OutdoorTempOutputs:
    data_dir = out_dir / "data"
    plots_dir = out_dir / "plots"
    echarts_dir = out_dir / "echarts"
    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    if enable_echarts:
        echarts_dir.mkdir(parents=True, exist_ok=True)

    times = df_weather.index
    if getattr(times, "tz", None) is not None:
        times_local = times.tz_localize(None)
    else:
        times_local = times
    T_base = df_weather["T_out"].to_numpy(float)
    if use_ar1_noise:
        noise = _ar1_noise(len(T_base), seed=int(ar1_seed))
    else:
        noise = np.zeros(len(T_base), dtype=float)
    T_2040 = T_base + float(delta_trend_C) + noise

    daily = pd.Series(T_base, index=times_local).resample("D").mean()
    daily_2040 = pd.Series(T_2040, index=times_local).resample("D").mean()
    daily_df = pd.DataFrame(
        {
            "date_local": daily.index.date.astype(str),
            "T_out_baseline_C": daily.values,
            "T_out_2040_C": daily_2040.values,
        }
    )
    daily_csv = data_dir / "outdoor_temperature_daily.csv"
    daily_df.to_csv(daily_csv, index=False)

    # pick hottest 7-day window by 2040 daily mean
    roll = daily_2040.rolling(7).mean()
    if roll.isna().all() or daily_2040.empty:
        start_day = daily_2040.index[0] if len(daily_2040) else times_local[0]
    else:
        start_day = roll.idxmax() - pd.Timedelta(days=6)
    end_day = start_day + pd.Timedelta(days=7)

    zoom_mask = (times_local >= start_day) & (times_local < end_day)
    zoom_times = times_local[zoom_mask]
    zoom_base = T_base[zoom_mask]
    zoom_2040 = T_2040[zoom_mask]
    zoom_df = pd.DataFrame(
        {
            "datetime_local": zoom_times.astype(str),
            "T_out_baseline_C": zoom_base,
            "T_out_2040_C": zoom_2040,
        }
    )
    zoom_csv = data_dir / "outdoor_temperature_zoom_hourly.csv"
    zoom_df.to_csv(zoom_csv, index=False)

    # Matplotlib main plot (monthly x-axis)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(daily.index, daily.values, label="Baseline (daily mean)", linewidth=1.2)
    ax.plot(daily_2040.index, daily_2040.values, label="2040 scenario (daily mean)", linewidth=1.2)
    ax.set_xlabel("Month")
    ax.set_ylabel("Outdoor Air Temperature (degC)")
    ax.set_title("Outdoor Temperature: Full Year")
    ax.legend()

    month_starts = pd.date_range(daily.index.min().floor("D"), daily.index.max().ceil("D"), freq="MS")
    ax.set_xticks(month_starts)
    ax.set_xticklabels([d.strftime("%b") for d in month_starts])

    png_path = plots_dir / "outdoor_temperature_full_year_with_zoom.png"
    fig.set_layout_engine("none")
    fig.savefig(png_path, dpi=200)
    plt.close(fig)

    echarts_json = None
    echarts_html = None
    if enable_echarts:
        _write_echarts_min_js(echarts_dir)
        daily_payload = daily_df.copy()
        daily_payload.insert(0, "day_index", np.arange(len(daily_payload)))
        zoom_payload = zoom_df.copy()
        zoom_payload.insert(0, "hour_index", np.arange(len(zoom_payload)))
        charts = {
            "daily": daily_payload.to_dict("records"),
            "zoom": zoom_payload.to_dict("records"),
        }
        echarts_json = echarts_dir / "charts_outdoor_temp.json"
        echarts_json.write_text(json.dumps(charts, indent=2), encoding="utf-8")
        echarts_html = echarts_dir / "index_outdoor_temp.html"
        _write_echarts_html(echarts_html, charts)

    return OutdoorTempOutputs(
        png_path=png_path,
        daily_csv=daily_csv,
        zoom_csv=zoom_csv,
        echarts_json=echarts_json,
        echarts_html=echarts_html,
        zoom_start=str(start_day.date()),
        zoom_end=str((end_day - pd.Timedelta(days=1)).date()),
        baseline_mean_C=float(np.mean(T_base)),
        baseline_max_C=float(np.max(T_base)),
        scenario_mean_C=float(np.mean(T_2040)),
        scenario_max_C=float(np.max(T_2040)),
    )
