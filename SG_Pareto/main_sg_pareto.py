from __future__ import annotations

# Allow running by clicking VSCode "Run Python File" on this file
import sys
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from borealis_model.data_io import load_weather_csv

from SG_Pareto.config import (
    ANALYSIS_ROOT,
    SINGAPORE,
    DEFAULT_BUILDING,
    DEFAULT_OPTICAL,
    DEFAULT_BOREALIS_OPTICAL,
    DEFAULT_BOUNDS,
    DEFAULT_HVAC,
    DEFAULT_PV,
    MATERIALS,
    DELTA_TREND_C,
    USE_AR1_NOISE,
    AR1_SEED,
    N_SAMPLES,
    RNG_SEED,
    A0_M2,
    L0_M,
    W0_M,
    H0_M,
    EPS_A,
    R_MIN,
    R_MAX,
    EPS_H,
    THETA_MAX_DEG,
    WWR_MIN,
    WWR_MAX,
    WINDOW_HEIGHT_M,
    PEAK_COOLING_LIMIT_KW,
    PEAK_LIMIT_LIST_KW,
    BASELINE_DN_M,
    BASELINE_DS_M,
    BASELINE_ETA_E_RAD,
    BASELINE_ETA_W_RAD,
    BASELINE_THETA_DEG,
    COMFORT_LOW_C,
    COMFORT_HIGH_C,
)
from SG_Pareto.pvgis_fetch import fetch_pvgis_tmy_to_csv
from SG_Pareto.evaluate_sg import evaluate_design
from SG_Pareto.geometry import GeometryDecision
from SG_Pareto.solar_facade_sg import compute_facade_irradiance
from SG_Pareto.paper_plots import generate_paper_outputs


def _ensure_dirs(root: Path) -> dict[str, Path]:
    weather_dir = root / "weather"
    results_dir = root / "results"
    plots_dir = root / "plots"
    echarts_dir = root / "echarts"
    data_dir = root / "data"
    for d in [weather_dir, results_dir, plots_dir, echarts_dir, data_dir]:
        d.mkdir(parents=True, exist_ok=True)
    return {
        "weather": weather_dir,
        "results": results_dir,
        "plots": plots_dir,
        "echarts": echarts_dir,
        "data": data_dir,
    }


def _sample_designs(bounds, n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    dN = rng.uniform(bounds.dN_bounds_m[0], bounds.dN_bounds_m[1], size=n)
    dS = rng.uniform(bounds.dS_bounds_m[0], bounds.dS_bounds_m[1], size=n)
    eE = rng.uniform(bounds.etaE_bounds_rad[0], bounds.etaE_bounds_rad[1], size=n)
    eW = rng.uniform(bounds.etaW_bounds_rad[0], bounds.etaW_bounds_rad[1], size=n)
    m = rng.integers(1, 3, size=n)

    A_min = (1.0 - float(EPS_A)) * float(A0_M2)
    A_max = (1.0 + float(EPS_A)) * float(A0_M2)
    A = rng.uniform(A_min, A_max, size=n)
    r = rng.uniform(float(R_MIN), float(R_MAX), size=n)
    L = np.sqrt(A * r)
    W = np.sqrt(A / r)

    H_min = (1.0 - float(EPS_H)) * float(H0_M)
    H_max = (1.0 + float(EPS_H)) * float(H0_M)
    H = rng.uniform(H_min, H_max, size=n)

    theta = rng.uniform(-float(THETA_MAX_DEG), float(THETA_MAX_DEG), size=n)

    wwr_N = rng.uniform(float(WWR_MIN), float(WWR_MAX), size=n)
    wwr_E = rng.uniform(float(WWR_MIN), float(WWR_MAX), size=n)
    wwr_S = rng.uniform(float(WWR_MIN), float(WWR_MAX), size=n)
    wwr_W = rng.uniform(float(WWR_MIN), float(WWR_MAX), size=n)

    return {
        "dN_m": dN,
        "dS_m": dS,
        "etaE_rad": eE,
        "etaW_rad": eW,
        "material_m": m,
        "L_m": L,
        "W_m": W,
        "H_m": H,
        "theta_deg": theta,
        "wwr_N": wwr_N,
        "wwr_E": wwr_E,
        "wwr_S": wwr_S,
        "wwr_W": wwr_W,
    }


def _write_local_echarts_js(out_dir: Path) -> Path:
    """Write a minimal offline echarts-compatible stub for scatter/line plots."""
    js = r"""(function(){
  function clamp(v, a, b){ return Math.max(a, Math.min(b, v)); }
  function init(dom){
    const canvas = document.createElement("canvas");
    const rect = dom.getBoundingClientRect();
    const w = Math.max(320, rect.width || dom.clientWidth || 900);
    const h = Math.max(240, rect.height || dom.clientHeight || 620);
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

      // axes
      ctx.strokeStyle = "#222";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(left, top);
      ctx.lineTo(left, top + plotH);
      ctx.lineTo(left + plotW, top + plotH);
      ctx.stroke();

      // ticks
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

      // axis labels
      if(option.xAxis && option.xAxis.name){
        ctx.fillText(option.xAxis.name, left + plotW/2 - 110, h - 10);
      }
      if(option.yAxis && option.yAxis.name){
        ctx.save();
        ctx.translate(16, top + plotH/2 + 60);
        ctx.rotate(-Math.PI/2);
        ctx.fillText(option.yAxis.name, 0, 0);
        ctx.restore();
      }

      // series points / lines
      const colors = ["#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#111827"];
      series.forEach((s, idx) => {
        const col = colors[idx % colors.length];
        const size = s.symbolSize || 3;
        const opacity = (s.itemStyle && s.itemStyle.opacity != null) ? s.itemStyle.opacity : 0.7;
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
        ctx.globalAlpha = clamp(opacity, 0.05, 1.0);
        data.forEach(p => {
          if(!p || p.length < 2) return;
          const x = xScale(p[0]);
          const y = yScale(p[1]);
          ctx.beginPath();
          ctx.arc(x, y, size, 0, Math.PI * 2);
          ctx.fill();
        });
        ctx.globalAlpha = 1.0;
      });
    }
    function getDataURL(){ return canvas.toDataURL("image/png"); }
    function resize(){
      const r = dom.getBoundingClientRect();
      const nw = Math.max(320, r.width || dom.clientWidth || 900);
      const nh = Math.max(240, r.height || dom.clientHeight || 620);
      canvas.width = nw; canvas.height = nh;
      render();
    }
    return { setOption, getDataURL, resize };
  }
  window.echarts = { init: init };
})();"""
    path = out_dir / "echarts.min.js"
    path.write_text(js, encoding="utf-8")
    return path


def _write_echarts_html(charts_path: Path, data: dict) -> None:
    payload = json.dumps(data, indent=2)
    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Feasible Cooling Solutions</title>
  <script src="./echarts.min.js"></script>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; }}
    .header {{ display: flex; align-items: center; justify-content: space-between; gap: 12px; }}
    #chart {{ width: 100%; height: 620px; }}
    .note {{ color: #666; font-size: 12px; }}
    .btn {{ padding: 6px 12px; border: 1px solid #bbb; background: #f7f7f7; cursor: pointer; }}
  </style>
</head>
<body>
  <div class="header">
    <h2>Feasible Cooling Solutions (2040)</h2>
    <button id="btn-download" class="btn">Download PNG</button>
  </div>
  <div id="chart"></div>
  <p class="note">Offline-ready: open this file directly to view the chart.</p>
  <script>
    const DATA = __PAYLOAD__;
    const feas = DATA.points.feasible.map(p => [p.E_cool_el_2040_kWh, p.P_cool_el_peak_2040_kW]);
    const infeas = DATA.points.infeasible.map(p => [p.E_cool_el_2040_kWh, p.P_cool_el_peak_2040_kW]);
    const best = DATA.best ? [[DATA.best.E_cool_el_2040_kWh, DATA.best.P_cool_el_peak_2040_kW]] : [];
    const limitLine = [[DATA.meta.E_cool_min_kWh, DATA.meta.P_max_kW], [DATA.meta.E_cool_max_kWh, DATA.meta.P_max_kW]];

    const chart = echarts.init(document.getElementById("chart"));
    chart.setOption({
      grid: { left: 70, right: 70, top: 60, bottom: 70, containLabel: true },
      tooltip: { trigger: "item" },
      legend: { data: ["Feasible", "Infeasible", "Best Feasible", "P_MAX"], top: 10 },
      xAxis: { name: "Annual Cooling Electricity 2040 (kWh_el)", nameGap: 30 },
      yAxis: { name: "Peak Cooling Electric Power 2040 (kW_el)", nameGap: 35 },
      series: [
        { name: "Feasible", type: "scatter", data: feas, symbolSize: 4, itemStyle: { opacity: 0.35 } },
        { name: "Infeasible", type: "scatter", data: infeas, symbolSize: 4, itemStyle: { opacity: 0.35 } },
        { name: "Best Feasible", type: "scatter", data: best, symbolSize: 10 },
        { name: "P_MAX", type: "line", data: limitLine, symbolSize: 0, lineStyle: { width: 2 } }
      ]
    });

    document.getElementById("btn-download").addEventListener("click", () => {
      const url = chart.getDataURL({ type: "png", pixelRatio: 2, backgroundColor: "#fff" });
      const a = document.createElement("a");
      a.href = url;
      a.download = "feasible_scatter.png";
      a.click();
    });
  </script>
</body>
</html>
"""
    charts_path.write_text(html.replace("__PAYLOAD__", payload), encoding="utf-8")


def _write_shading_echarts_html(html_path: Path, data: dict) -> None:
    payload = json.dumps(data, indent=2)
    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Passive Shading Transmissivity</title>
  <script src="./echarts.min.js"></script>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; }}
    .header {{ display: flex; align-items: center; justify-content: space-between; gap: 12px; }}
    #chart {{ width: 100%; height: 520px; }}
    .note {{ color: #666; font-size: 12px; }}
    .btn {{ padding: 6px 12px; border: 1px solid #bbb; background: #f7f7f7; cursor: pointer; }}
  </style>
</head>
<body>
  <div class="header">
    <h2>Passive Shading Transmissivity (u_f)</h2>
    <button id="btn-download" class="btn">Download PNG</button>
  </div>
  <div id="chart"></div>
  <p class="note">Offline-ready: open this file directly to view the chart.</p>
  <script>
    const DATA = __PAYLOAD__;
    const x = DATA.series.hour_index;
    function pairs(arr) {{ return x.map((v, i) => [v, arr[i]]); }}

    const chart = echarts.init(document.getElementById("chart"));
    chart.setOption({
      grid: { left: 70, right: 40, top: 60, bottom: 70, containLabel: true },
      tooltip: { trigger: "item" },
      legend: { data: ["uN", "uE", "uS", "uW"], top: 10 },
      xAxis: { name: "Hour index (sampled)", nameGap: 30 },
      yAxis: { name: "u (0-1)", nameGap: 35 },
      series: [
        { name: "uN", type: "line", data: pairs(DATA.series.uN), symbolSize: 2, lineStyle: { width: 2 } },
        { name: "uE", type: "line", data: pairs(DATA.series.uE), symbolSize: 2, lineStyle: { width: 2 } },
        { name: "uS", type: "line", data: pairs(DATA.series.uS), symbolSize: 2, lineStyle: { width: 2 } },
        { name: "uW", type: "line", data: pairs(DATA.series.uW), symbolSize: 2, lineStyle: { width: 2 } }
      ]
    });

    document.getElementById("btn-download").addEventListener("click", () => {
      const url = chart.getDataURL({ type: "png", pixelRatio: 2, backgroundColor: "#fff" });
      const a = document.createElement("a");
      a.href = url;
      a.download = "shading_u_facades.png";
      a.click();
    });
  </script>
</body>
</html>
"""
    html_path.write_text(html.replace("__PAYLOAD__", payload), encoding="utf-8")


def main(enable_passive_shading: bool = True) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = ANALYSIS_ROOT / f"sg_pareto_{ts}"
    dirs = _ensure_dirs(run_root)

    shading_root = PROJECT_ROOT / "data_analysis" / "sg_passive_shading" / ts
    shading_echarts_dir = shading_root / "echarts"
    shading_echarts_dir.mkdir(parents=True, exist_ok=True)

    print("========== SG_Pareto：开始 ==========")
    print(f"运行目录：{run_root}")
    print("步骤1：准备 PVGIS 天气数据（Singapore）...")

    weather_csv = dirs["weather"] / "weather_singapore_tmy_hourly.csv"
    if weather_csv.exists():
        row_count = max(0, sum(1 for _ in weather_csv.open("r", encoding="utf-8")) - 1)
        print("PVGIS：命中缓存，直接使用已有天气文件。")
        print(f"天气文件：{weather_csv}（{row_count} 行）")
    else:
        info = fetch_pvgis_tmy_to_csv(
            lat=SINGAPORE.latitude,
            lon=SINGAPORE.longitude,
            out_csv=weather_csv,
            overwrite=False,
        )
        print(f"PVGIS：下载完成，已生成天气文件：{info['path']}（{info['rows']} 行）")

    print("步骤2：生成 2040 温度情景（DeltaT=+1.5C）...")
    df_raw = pd.read_csv(weather_csv)
    df_trend = df_raw.copy()
    df_trend["T_out"] = pd.to_numeric(df_trend["T_out"], errors="coerce") + float(DELTA_TREND_C)
    weather_trend_csv = dirs["weather"] / "weather_singapore_tmy_hourly_trend_1p5C.csv"
    df_trend.to_csv(weather_trend_csv, index=False)
    print(f"已生成：{weather_trend_csv}")

    print("步骤3：读取天气并运行模型...")
    df_base = load_weather_csv(str(weather_csv), tz=SINGAPORE.tz)
    print(f"天气行数：{len(df_base)}  时区：{SINGAPORE.tz}")
    print(f"被动遮阳：{'开启' if enable_passive_shading else '关闭'}")

    rng = np.random.default_rng(RNG_SEED)
    X = _sample_designs(DEFAULT_BOUNDS, N_SAMPLES, rng)
    print(f"采样数：{N_SAMPLES}  随机种子：{RNG_SEED}")

    rows = []
    for i in range(N_SAMPLES):
        dN = float(X["dN_m"][i])
        dS = float(X["dS_m"][i])
        etaE = float(X["etaE_rad"][i])
        etaW = float(X["etaW_rad"][i])
        m = int(X["material_m"][i])

        geom = GeometryDecision(
            L_m=float(X["L_m"][i]),
            W_m=float(X["W_m"][i]),
            H_m=float(X["H_m"][i]),
            theta_deg=float(X["theta_deg"][i]),
            wwr_N=float(X["wwr_N"][i]),
            wwr_E=float(X["wwr_E"][i]),
            wwr_S=float(X["wwr_S"][i]),
            wwr_W=float(X["wwr_W"][i]),
        )
        material = MATERIALS[int(m)]

        res = evaluate_design(
            df_base,
            location=SINGAPORE,
            optical_base=DEFAULT_OPTICAL,
            optics_borealis=DEFAULT_BOREALIS_OPTICAL,
            hvac_cfg=DEFAULT_HVAC,
            material=material,
            pv_cfg=DEFAULT_PV,
            dN=dN,
            dS=dS,
            etaE=etaE,
            etaW=etaW,
            geometry=geom,
            window_height_m=float(WINDOW_HEIGHT_M),
            delta_trend_C=float(DELTA_TREND_C),
            enable_passive_shading=enable_passive_shading,
        )

        rows.append(
            {
                "dN_m": dN,
                "dS_m": dS,
                "etaE_rad": etaE,
                "etaW_rad": etaW,
                "material_m": int(m),
                "material_id": int(m),
                "L_m": float(geom.L_m),
                "W_m": float(geom.W_m),
                "H_m": float(geom.H_m),
                "theta_deg": float(geom.theta_deg),
                "wwr_N": float(geom.wwr_N),
                "wwr_E": float(geom.wwr_E),
                "wwr_S": float(geom.wwr_S),
                "wwr_W": float(geom.wwr_W),
                "A_roof_m2": res["A_roof_m2"],
                "A_win_total_m2": res["A_win_total_m2"],
                "A_win_N_m2": res["A_win_N_m2"],
                "A_win_E_m2": res["A_win_E_m2"],
                "A_win_S_m2": res["A_win_S_m2"],
                "A_win_W_m2": res["A_win_W_m2"],
                "E_cool_th_kWh": res["E_cool_th_kWh"],
                "E_cool_el_kWh_current": res["E_cool_el_kWh"],
                "E_cool_th_2040_kWh": res["E_cool_th_2040_kWh"],
                "E_cool_el_2040_kWh": res["E_cool_el_2040_kWh"],
                "P_cool_el_peak_2040_kW": res["P_cool_el_peak_2040_kW"],
                "E_pv_el_kWh": res["E_pv_el_kWh"],
                "MNZ_kWh_el": res["MNZ_kWh_el"],
                "peak_hour_local": res["peak_hour_local"],
                "peak_datetime_local": res["peak_datetime_local"],
                "T_out_2040_at_peak_C": res["T_out_2040_at_peak_C"],
                "solar_gain_at_peak_W": res["solar_gain_at_peak_W"],
            }
        )

    df = pd.DataFrame(rows)
    df["feasible"] = df["P_cool_el_peak_2040_kW"] <= float(PEAK_COOLING_LIMIT_KW)

    samples_csv = dirs["results"] / "samples.csv"
    df.to_csv(samples_csv, index=False)

    # Baseline run (single design)
    geom_base = GeometryDecision(
        L_m=float(L0_M),
        W_m=float(W0_M),
        H_m=float(H0_M),
        theta_deg=float(BASELINE_THETA_DEG),
        wwr_N=float(DEFAULT_BUILDING.wwr_other),
        wwr_E=float(DEFAULT_BUILDING.wwr_other),
        wwr_S=float(DEFAULT_BUILDING.wwr_south),
        wwr_W=float(DEFAULT_BUILDING.wwr_other),
    )
    base_material = MATERIALS[1]
    base_res = evaluate_design(
        df_base,
        location=SINGAPORE,
        optical_base=DEFAULT_OPTICAL,
        optics_borealis=DEFAULT_BOREALIS_OPTICAL,
        hvac_cfg=DEFAULT_HVAC,
        material=base_material,
        pv_cfg=DEFAULT_PV,
        dN=float(BASELINE_DN_M),
        dS=float(BASELINE_DS_M),
        etaE=float(BASELINE_ETA_E_RAD),
        etaW=float(BASELINE_ETA_W_RAD),
        geometry=geom_base,
        window_height_m=float(WINDOW_HEIGHT_M),
        delta_trend_C=float(DELTA_TREND_C),
        enable_passive_shading=enable_passive_shading,
    )
    baseline_row = {
        "dN_m": float(BASELINE_DN_M),
        "dS_m": float(BASELINE_DS_M),
        "etaE_rad": float(BASELINE_ETA_E_RAD),
        "etaW_rad": float(BASELINE_ETA_W_RAD),
        "material_m": 1,
        "material_id": 1,
        "L_m": float(geom_base.L_m),
        "W_m": float(geom_base.W_m),
        "H_m": float(geom_base.H_m),
        "theta_deg": float(geom_base.theta_deg),
        "wwr_N": float(geom_base.wwr_N),
        "wwr_E": float(geom_base.wwr_E),
        "wwr_S": float(geom_base.wwr_S),
        "wwr_W": float(geom_base.wwr_W),
        "A_roof_m2": base_res["A_roof_m2"],
        "A_win_total_m2": base_res["A_win_total_m2"],
        "A_win_N_m2": base_res["A_win_N_m2"],
        "A_win_E_m2": base_res["A_win_E_m2"],
        "A_win_S_m2": base_res["A_win_S_m2"],
        "A_win_W_m2": base_res["A_win_W_m2"],
        "E_cool_th_kWh": base_res["E_cool_th_kWh"],
        "E_cool_el_kWh_current": base_res["E_cool_el_kWh"],
        "E_cool_th_2040_kWh": base_res["E_cool_th_2040_kWh"],
        "E_cool_el_2040_kWh": base_res["E_cool_el_2040_kWh"],
        "P_cool_el_peak_2040_kW": base_res["P_cool_el_peak_2040_kW"],
        "E_pv_el_kWh": base_res["E_pv_el_kWh"],
        "MNZ_kWh_el": base_res["MNZ_kWh_el"],
        "peak_hour_local": base_res["peak_hour_local"],
        "peak_datetime_local": base_res["peak_datetime_local"],
        "T_out_2040_at_peak_C": base_res["T_out_2040_at_peak_C"],
        "solar_gain_at_peak_W": base_res["solar_gain_at_peak_W"],
    }
    baseline_csv = dirs["results"] / "baseline.csv"
    pd.DataFrame([baseline_row]).to_csv(baseline_csv, index=False)

    feasible_df = df[df["feasible"]].copy()
    feasible_csv = dirs["results"] / "feasible.csv"
    feasible_df.to_csv(feasible_csv, index=False)

    best_path = dirs["results"] / "best_feasible.csv"
    infeasible_report = dirs["results"] / "infeasible_report.json"

    best_row = None
    if feasible_df.empty:
        infeasible_report.write_text(
            json.dumps(
                {
                    "message": "no feasible solution",
                    "P_max_kW": float(PEAK_COOLING_LIMIT_KW),
                    "n_samples": int(len(df)),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    else:
        best_idx = feasible_df["E_cool_el_2040_kWh"].idxmin()
        best_row = feasible_df.loc[best_idx].to_dict()
        pd.DataFrame([best_row]).to_csv(best_path, index=False)

    # Plot feasible scatter
    plt.figure()
    feas = df[df["feasible"]]
    infeas = df[~df["feasible"]]
    plt.scatter(feas["E_cool_el_2040_kWh"], feas["P_cool_el_peak_2040_kW"], s=10, alpha=0.35, label="Feasible")
    plt.scatter(infeas["E_cool_el_2040_kWh"], infeas["P_cool_el_peak_2040_kW"], s=10, alpha=0.35, label="Infeasible")
    plt.axhline(float(PEAK_COOLING_LIMIT_KW), color="black", linewidth=1.5, label="P_MAX")
    if best_row is not None:
        plt.scatter([best_row["E_cool_el_2040_kWh"]], [best_row["P_cool_el_peak_2040_kW"]], s=120, marker="*", label="Best Feasible")
    plt.xlabel("Annual Cooling Electricity 2040 (kWh_el)")
    plt.ylabel("Peak Cooling Electric Power 2040 (kW_el)")
    plt.title("Feasible Cooling Solutions (2040)")
    plt.legend()
    plt.tight_layout()
    feasible_png = dirs["plots"] / "feasible_scatter.png"
    plt.savefig(feasible_png, dpi=200)
    plt.close()

    # ECharts payload (offline)
    meta = {
        "site": {
            "name": SINGAPORE.name,
            "latitude": SINGAPORE.latitude,
            "longitude": SINGAPORE.longitude,
            "tz": SINGAPORE.tz,
        },
        "delta_trend_C": DELTA_TREND_C,
        "peak_cooling_limit_kw": float(PEAK_COOLING_LIMIT_KW),
        "samples": int(len(df)),
    }

    E_min = float(df["E_cool_el_2040_kWh"].min()) if len(df) else 0.0
    E_max = float(df["E_cool_el_2040_kWh"].max()) if len(df) else 1.0

    charts = {
        "meta": {
            "P_max_kW": float(PEAK_COOLING_LIMIT_KW),
            "E_cool_min_kWh": E_min,
            "E_cool_max_kWh": E_max,
        },
        "points": {
            "feasible": feas[["E_cool_el_2040_kWh", "P_cool_el_peak_2040_kW"]].to_dict("records"),
            "infeasible": infeas[["E_cool_el_2040_kWh", "P_cool_el_peak_2040_kW"]].to_dict("records"),
        },
        "best": None if best_row is None else {
            "E_cool_el_2040_kWh": float(best_row["E_cool_el_2040_kWh"]),
            "P_cool_el_peak_2040_kW": float(best_row["P_cool_el_peak_2040_kW"]),
        },
    }

    charts_json = dirs["echarts"] / "charts.json"
    charts_json.write_text(json.dumps(charts, indent=2), encoding="utf-8")
    charts_html = dirs["echarts"] / "index.html"
    _write_local_echarts_js(dirs["echarts"])
    _write_echarts_html(charts_html, charts)

    # Passive shading debug timeseries (use first sample)
    geom_dbg = GeometryDecision(
        L_m=float(X["L_m"][0]),
        W_m=float(X["W_m"][0]),
        H_m=float(X["H_m"][0]),
        theta_deg=float(X["theta_deg"][0]),
        wwr_N=float(X["wwr_N"][0]),
        wwr_E=float(X["wwr_E"][0]),
        wwr_S=float(X["wwr_S"][0]),
        wwr_W=float(X["wwr_W"][0]),
    )
    irr_dbg = compute_facade_irradiance(
        df_base,
        location=SINGAPORE,
        facade_azimuths_deg=geom_dbg.facade_azimuths_deg(),
        dN=float(X["dN_m"][0]),
        dS=float(X["dS_m"][0]),
        etaE=float(X["etaE_rad"][0]),
        etaW=float(X["etaW_rad"][0]),
        window_height_m=float(WINDOW_HEIGHT_M),
        optics_borealis=DEFAULT_BOREALIS_OPTICAL,
        enable_passive_shading=enable_passive_shading,
        return_debug=True,
    )

    times = df_base.index
    time_col = times.tz_localize(None).astype(str)
    mu = irr_dbg["mu"]
    u = irr_dbg["u"]
    I_f = irr_dbg["I_f"]

    dbg_df = pd.DataFrame(
        {
            "datetime": time_col,
            "alpha_rad": irr_dbg["alpha_rad"],
            "psi_rad": irr_dbg["psi_rad"],
            "muN": mu["N"],
            "muE": mu["E"],
            "muS": mu["S"],
            "muW": mu["W"],
            "uN": u["N"],
            "uE": u["E"],
            "uS": u["S"],
            "uW": u["W"],
            "I_N": I_f["N"],
            "I_E": I_f["E"],
            "I_S": I_f["S"],
            "I_W": I_f["W"],
        }
    )
    shading_csv = shading_root / "timeseries_shading_debug.csv"
    shading_root.mkdir(parents=True, exist_ok=True)
    dbg_df.to_csv(shading_csv, index=False)

    # shading echarts (downsample every 24 hours)
    step = 24
    idx = np.arange(0, len(times), step)
    shading_payload = {
        "meta": {
            "sample_step_hours": step,
            "total_hours": int(len(times)),
        },
        "series": {
            "hour_index": idx.tolist(),
            "uN": u["N"][idx].tolist(),
            "uE": u["E"][idx].tolist(),
            "uS": u["S"][idx].tolist(),
            "uW": u["W"][idx].tolist(),
        },
    }
    charts_shading_json = shading_echarts_dir / "charts_shading.json"
    charts_shading_json.write_text(json.dumps(shading_payload, indent=2), encoding="utf-8")
    _write_local_echarts_js(shading_echarts_dir)
    _write_shading_echarts_html(shading_echarts_dir / "index_shading.html", shading_payload)

    paper_info = generate_paper_outputs(
        df_base=df_base,
        df_samples=df,
        feasible_df=feasible_df,
        best_row=best_row,
        baseline_row=baseline_row,
        out_dirs=dirs,
        location=SINGAPORE,
        optical_base=DEFAULT_OPTICAL,
        optics_borealis=DEFAULT_BOREALIS_OPTICAL,
        hvac_cfg=DEFAULT_HVAC,
        materials=MATERIALS,
        delta_trend_C=float(DELTA_TREND_C),
        peak_limit_kw=float(PEAK_COOLING_LIMIT_KW),
        peak_limit_list=PEAK_LIMIT_LIST_KW,
        comfort_low_C=float(COMFORT_LOW_C),
        comfort_high_C=float(COMFORT_HIGH_C),
        window_height_m=float(WINDOW_HEIGHT_M),
        enable_passive_shading=enable_passive_shading,
    )

    print("步骤4：写出结果与统计摘要...")
    print(f"samples.csv：{samples_csv}")
    print(f"feasible.csv：{feasible_csv}")
    print(f"best_feasible.csv：{best_path}")
    if feasible_df.empty:
        print("无可行解，建议提高 P_MAX 或扩大采样范围")
        print(f"infeasible_report.json：{infeasible_report}")
    print(f"feasible_scatter.png：{feasible_png}")
    print(f"charts.json：{charts_json}")
    print(f"index.html：{charts_html}")

    def _stat(arr: np.ndarray) -> str:
        return f"min={float(np.min(arr)):.3f}, max={float(np.max(arr)):.3f}, mean={float(np.mean(arr)):.3f}"

    print("被动遮阳 u_f 统计：")
    print(f"uN: {_stat(u['N'])}")
    print(f"uE: {_stat(u['E'])}")
    print(f"uS: {_stat(u['S'])}")
    print(f"uW: {_stat(u['W'])}")
    print(f"遮阳调试 CSV：{shading_csv}")
    print(f"遮阳图表：{shading_echarts_dir / 'index_shading.html'}")
    print("论文图表输出：")
    print(f"data：{paper_info['data_dir']}")
    print(f"plots：{paper_info['plots_dir']}")
    print(f"echarts：{paper_info['echarts_dir']}")
    print(f"已选择最热周：{paper_info['heatwave_start']} 至 {paper_info['heatwave_end']}")
    print(f"无制冷最高室温：{paper_info['max_temp_no_cooling_C']:.2f}°C；有制冷最高室温：{paper_info['max_temp_with_cooling_C']:.2f}°C")
    print(f"过热小时数（>26°C）：无制冷 {paper_info['overheat_hours_no']} h，有制冷 {paper_info['overheat_hours_with']} h")

    print("========== SG_Pareto：完成 ==========")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--disable-passive-shading", action="store_true", help="Disable passive shading (direct beam).")
    args = parser.parse_args()
    main(enable_passive_shading=not args.disable_passive_shading)
