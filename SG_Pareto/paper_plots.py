
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Iterable

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from SG_Pareto.geometry import GeometryDecision
from SG_Pareto.solar_facade_sg import compute_facade_irradiance
from SG_Pareto.thermal_sg import simulate_2r2c_deadband, simulate_2r2c_no_hvac


@dataclass(frozen=True)
class DesignRow:
    dN_m: float
    dS_m: float
    etaE_rad: float
    etaW_rad: float
    material_id: int
    L_m: float
    W_m: float
    H_m: float
    theta_deg: float
    wwr_N: float
    wwr_E: float
    wwr_S: float
    wwr_W: float


def _row_to_design(row: Dict[str, Any]) -> DesignRow:
    return DesignRow(
        dN_m=float(row["dN_m"]),
        dS_m=float(row["dS_m"]),
        etaE_rad=float(row["etaE_rad"]),
        etaW_rad=float(row["etaW_rad"]),
        material_id=int(row.get("material_id", row.get("material_m", 1))),
        L_m=float(row["L_m"]),
        W_m=float(row["W_m"]),
        H_m=float(row["H_m"]),
        theta_deg=float(row["theta_deg"]),
        wwr_N=float(row["wwr_N"]),
        wwr_E=float(row["wwr_E"]),
        wwr_S=float(row["wwr_S"]),
        wwr_W=float(row["wwr_W"]),
    )


def _compute_design_series(
    df_base: pd.DataFrame,
    *,
    location,
    optical_base,
    optics_borealis,
    hvac_cfg,
    material,
    design: DesignRow,
    window_height_m: float,
    delta_trend_C: float,
    enable_passive_shading: bool,
) -> Dict[str, Any]:
    geom = GeometryDecision(
        L_m=design.L_m,
        W_m=design.W_m,
        H_m=design.H_m,
        theta_deg=design.theta_deg,
        wwr_N=design.wwr_N,
        wwr_E=design.wwr_E,
        wwr_S=design.wwr_S,
        wwr_W=design.wwr_W,
    )

    irr = compute_facade_irradiance(
        df_base,
        location=location,
        facade_azimuths_deg=geom.facade_azimuths_deg(),
        dN=design.dN_m,
        dS=design.dS_m,
        etaE=design.etaE_rad,
        etaW=design.etaW_rad,
        window_height_m=window_height_m,
        optics_borealis=optics_borealis,
        enable_passive_shading=enable_passive_shading,
        return_debug=True,
    )

    A_win = geom.window_areas_m2()
    g_eq = float(getattr(optical_base, "tau_heat", 0.55))
    I_f = irr["I_f"]
    Phi_s = (
        A_win["N"] * g_eq * I_f["N"]
        + A_win["E"] * g_eq * I_f["E"]
        + A_win["S"] * g_eq * I_f["S"]
        + A_win["W"] * g_eq * I_f["W"]
    )

    T_out_2040 = df_base["T_out"].to_numpy(float) + float(delta_trend_C)

    Ti_with, Tm_with, Phi_h = simulate_2r2c_deadband(
        Phi_s_W=Phi_s,
        T_out_C=T_out_2040,
        Ci=material.Ci_J_per_K,
        Cm=material.Cm_J_per_K,
        Ria=material.Ria_K_per_W,
        Rim=material.Rim_K_per_W,
        eta=material.eta_air,
        T_heat_C=hvac_cfg.T_heat_C,
        T_cool_C=hvac_cfg.T_cool_C,
        dt_hours=hvac_cfg.dt_hours,
        T_init_C=hvac_cfg.T_init_C,
    )

    Ti_no, Tm_no = simulate_2r2c_no_hvac(
        Phi_s_W=Phi_s,
        T_out_C=T_out_2040,
        Ci=material.Ci_J_per_K,
        Cm=material.Cm_J_per_K,
        Ria=material.Ria_K_per_W,
        Rim=material.Rim_K_per_W,
        eta=material.eta_air,
        dt_hours=hvac_cfg.dt_hours,
        T_init_C=hvac_cfg.T_init_C,
    )

    COP = float(hvac_cfg.COP_cool)
    P_cool_el_kW = np.maximum(-Phi_h, 0.0) / 1000.0 / COP

    return {
        "geom": geom,
        "irr": irr,
        "Phi_s_W": Phi_s,
        "T_out_2040_C": T_out_2040,
        "Ti_with_C": Ti_with,
        "Tm_with_C": Tm_with,
        "Ti_no_C": Ti_no,
        "Tm_no_C": Tm_no,
        "Phi_h_W": Phi_h,
        "P_cool_el_kW": P_cool_el_kW,
    }


def _daily_mean(times: pd.DatetimeIndex, values: np.ndarray) -> pd.Series:
    arr = np.asarray(values, dtype=float)
    n = min(len(times), len(arr))
    s = pd.Series(arr[:n], index=times[:n])
    return s.resample("D").mean()

def _index_to_str(idx: pd.DatetimeIndex) -> np.ndarray:
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    return idx.astype(str).to_numpy()


def _monthly_sum_kwh(times: pd.DatetimeIndex, power_kW: np.ndarray, dt_hours: float = 1.0) -> pd.Series:
    arr = np.asarray(power_kW, dtype=float)
    n = min(len(times), len(arr))
    s = pd.Series(arr[:n] * float(dt_hours), index=times[:n])
    return s.resample("MS").sum()


def _hottest_week_start(times: pd.DatetimeIndex, T_out_2040: np.ndarray) -> int:
    s = pd.Series(np.asarray(T_out_2040, dtype=float), index=times)
    win = 24 * 7
    if len(s) <= win:
        return 0
    r = s.rolling(win).mean()
    end_idx = int(r.idxmax().value)
    end_ts = pd.Timestamp(end_idx, tz=times.tz)
    start_ts = end_ts - pd.Timedelta(hours=win - 1)
    return int(np.where(times == start_ts)[0][0]) if start_ts in times else 0


def _hottest_day_start(times: pd.DatetimeIndex, T_out_2040: np.ndarray) -> int:
    s = pd.Series(np.asarray(T_out_2040, dtype=float), index=times)
    day = s.resample("D").mean()
    if day.empty:
        return 0
    hottest_day = day.idxmax()
    start_ts = hottest_day.floor("D")
    return int(np.where(times == start_ts)[0][0]) if start_ts in times else 0


def _write_echarts_min_js(out_dir: Path) -> None:
    js = r"""(function(){
  function clamp(v, a, b){ return Math.max(a, Math.min(b, v)); }
  function init(dom){
    const canvas = document.createElement("canvas");
    const rect = dom.getBoundingClientRect();
    const w = Math.max(320, rect.width || dom.clientWidth || 900);
    const h = Math.max(240, rect.height || dom.clientHeight || 520);
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
      const nh = Math.max(240, r.height || dom.clientHeight || 520);
      canvas.width = nw; canvas.height = nh;
      render();
    }
    return { setOption, getDataURL, resize };
  }
  window.echarts = { init: init };
})();"""
    (out_dir / "echarts.min.js").write_text(js, encoding="utf-8")


def _build_echarts_payload(
    *,
    df_samples: pd.DataFrame,
    feasible_df: pd.DataFrame,
    best_row: Dict[str, Any] | None,
    peak_limit_kw: float,
    sens_df: pd.DataFrame,
    temp_daily_df: pd.DataFrame,
    indoor_daily_df: pd.DataFrame,
) -> Dict[str, Any]:
    infeasible_df = df_samples[~df_samples["feasible"]]
    temp_daily = temp_daily_df.copy()
    temp_daily.insert(0, "day_index", np.arange(len(temp_daily)))
    indoor_daily = indoor_daily_df.copy()
    indoor_daily.insert(0, "day_index", np.arange(len(indoor_daily)))
    payload = {
        "meta": {
            "P_max_kW": float(peak_limit_kw),
        },
        "points": {
            "feasible": feasible_df[["E_cool_el_2040_kWh", "P_cool_el_peak_2040_kW"]].to_dict("records"),
            "infeasible": infeasible_df[["E_cool_el_2040_kWh", "P_cool_el_peak_2040_kW"]].to_dict("records"),
        },
        "best": None,
        "sensitivity": sens_df.to_dict("records"),
        "temperature_daily": temp_daily.to_dict("records"),
        "indoor_temp_daily": indoor_daily.to_dict("records"),
    }
    if best_row is not None:
        payload["best"] = {
            "E_cool_el_2040_kWh": float(best_row["E_cool_el_2040_kWh"]),
            "P_cool_el_peak_2040_kW": float(best_row["P_cool_el_peak_2040_kW"]),
        }
    return payload


def _write_paper_echarts_html(html_path: Path, data: Dict[str, Any]) -> None:
    payload = json.dumps(data, indent=2)
    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>SG Pareto Paper Charts</title>
  <script src="./echarts.min.js"></script>
  <style>
    body { font-family: Arial, sans-serif; margin: 20px; }
    .chart { width: 100%; height: 520px; margin-bottom: 28px; }
    h2 { margin-top: 24px; }
    .note { color: #666; font-size: 12px; }
  </style>
</head>
<body>
  <h2>Feasible Region Scatter (2040)</h2>
  <div id="chart-feasible" class="chart"></div>

  <h2>Peak Limit Sensitivity</h2>
  <div id="chart-sensitivity" class="chart"></div>

  <h2>Daily Outdoor Temperature: Baseline vs 2040</h2>
  <div id="chart-temperature" class="chart"></div>

  <h2>2040 Indoor Temperature: With vs Without Cooling</h2>
  <div id="chart-indoor" class="chart"></div>

  <p class="note">Offline-ready: open this file directly to view the charts.</p>

  <script>
    const DATA = __PAYLOAD__;

    function pairsFromRecords(records, xKey, yKey){
      return records.map(r => [r[xKey], r[yKey]]);
    }

    const feas = DATA.points.feasible.map(p => [p.E_cool_el_2040_kWh, p.P_cool_el_peak_2040_kW]);
    const infeas = DATA.points.infeasible.map(p => [p.E_cool_el_2040_kWh, p.P_cool_el_peak_2040_kW]);
    const best = DATA.best ? [[DATA.best.E_cool_el_2040_kWh, DATA.best.P_cool_el_peak_2040_kW]] : [];
    const allE = feas.map(p => p[0]).concat(infeas.map(p => p[0]));
    const eMin = allE.length ? Math.min(...allE) : 0;
    const eMax = allE.length ? Math.max(...allE) : 1;
    const limitLine = [[eMin, DATA.meta.P_max_kW], [eMax, DATA.meta.P_max_kW]];

    const chart1 = echarts.init(document.getElementById("chart-feasible"));
    chart1.setOption({
      grid: { left: 70, right: 70, top: 50, bottom: 70, containLabel: true },
      xAxis: { name: "Annual Cooling Electricity 2040 (kWh_el)", nameGap: 30 },
      yAxis: { name: "Peak Cooling Electric Power 2040 (kW_el)", nameGap: 35 },
      series: [
        { name: "Feasible", type: "scatter", data: feas, symbolSize: 4, itemStyle: { opacity: 0.35 } },
        { name: "Infeasible", type: "scatter", data: infeas, symbolSize: 4, itemStyle: { opacity: 0.35 } },
        { name: "Best", type: "scatter", data: best, symbolSize: 10 },
        { name: "P_max", type: "line", data: limitLine, symbolSize: 0, lineStyle: { width: 2 } }
      ]
    });

    const sens = DATA.sensitivity;
    const sensBest = sens.map(r => [r.P_max_kW, r.best_feasible_E_cool_el_2040_kWh]);
    const sensFrac = sens.map(r => [r.P_max_kW, r.feasible_fraction]);
    const chart2 = echarts.init(document.getElementById("chart-sensitivity"));
    chart2.setOption({
      grid: { left: 70, right: 70, top: 50, bottom: 70, containLabel: true },
      xAxis: { name: "P_max (kW_el)", nameGap: 30 },
      yAxis: { name: "Value", nameGap: 35 },
      series: [
        { name: "Best feasible energy", type: "line", data: sensBest, symbolSize: 3 },
        { name: "Feasible fraction", type: "line", data: sensFrac, symbolSize: 3 }
      ]
    });

    const temp = DATA.temperature_daily;
    const tempBase = temp.map(r => [r.day_index, r.T_out_baseline_C]);
    const temp2040 = temp.map(r => [r.day_index, r.T_out_2040_C]);
    const chart3 = echarts.init(document.getElementById("chart-temperature"));
    chart3.setOption({
      grid: { left: 70, right: 70, top: 50, bottom: 70, containLabel: true },
      xAxis: { name: "Day index (local)", nameGap: 30 },
      yAxis: { name: "Outdoor Air Temperature (°C)", nameGap: 35 },
      series: [
        { name: "Baseline", type: "line", data: tempBase, symbolSize: 2, lineStyle: { width: 2 } },
        { name: "2040", type: "line", data: temp2040, symbolSize: 2, lineStyle: { width: 2 } }
      ]
    });

    const indoor = DATA.indoor_temp_daily;
    const indWith = indoor.map(r => [r.day_index, r.T_air_2040_with_cooling_C]);
    const indNo = indoor.map(r => [r.day_index, r.T_air_2040_no_cooling_C]);
    const chart4 = echarts.init(document.getElementById("chart-indoor"));
    chart4.setOption({
      grid: { left: 70, right: 70, top: 50, bottom: 70, containLabel: true },
      xAxis: { name: "Day index (local)", nameGap: 30 },
      yAxis: { name: "Indoor Air Temperature (°C)", nameGap: 35 },
      series: [
        { name: "With cooling", type: "line", data: indWith, symbolSize: 2, lineStyle: { width: 2 } },
        { name: "No cooling", type: "line", data: indNo, symbolSize: 2, lineStyle: { width: 2 } }
      ]
    });
  </script>
</body>
</html>
"""
    html_path.write_text(html.replace("__PAYLOAD__", payload), encoding="utf-8")

def generate_paper_outputs(
    *,
    df_base: pd.DataFrame,
    df_samples: pd.DataFrame,
    feasible_df: pd.DataFrame,
    best_row: Dict[str, Any] | None,
    baseline_row: Dict[str, Any],
    out_dirs: Dict[str, Path],
    location,
    optical_base,
    optics_borealis,
    hvac_cfg,
    materials: Dict[int, Any],
    delta_trend_C: float,
    peak_limit_kw: float,
    peak_limit_list: Iterable[float],
    comfort_low_C: float,
    comfort_high_C: float,
    window_height_m: float,
    enable_passive_shading: bool,
) -> Dict[str, Any]:
    data_dir = out_dirs["data"]
    plots_dir = out_dirs["plots"]
    echarts_dir = out_dirs["echarts"]

    if best_row is None:
        design_row = baseline_row
    else:
        design_row = best_row

    base_design = _row_to_design(baseline_row)
    best_design = _row_to_design(design_row)

    base_series = _compute_design_series(
        df_base,
        location=location,
        optical_base=optical_base,
        optics_borealis=optics_borealis,
        hvac_cfg=hvac_cfg,
        material=materials[int(base_design.material_id)],
        design=base_design,
        window_height_m=window_height_m,
        delta_trend_C=delta_trend_C,
        enable_passive_shading=enable_passive_shading,
    )
    best_series = _compute_design_series(
        df_base,
        location=location,
        optical_base=optical_base,
        optics_borealis=optics_borealis,
        hvac_cfg=hvac_cfg,
        material=materials[int(best_design.material_id)],
        design=best_design,
        window_height_m=window_height_m,
        delta_trend_C=delta_trend_C,
        enable_passive_shading=enable_passive_shading,
    )

    times = df_base.index
    feas = df_samples[df_samples["feasible"]]
    infeas = df_samples[~df_samples["feasible"]]

    # 1) feasible_region_scatter
    plt.figure()
    plt.scatter(feas["E_cool_el_2040_kWh"], feas["P_cool_el_peak_2040_kW"], s=10, alpha=0.35, label="Feasible")
    plt.scatter(infeas["E_cool_el_2040_kWh"], infeas["P_cool_el_peak_2040_kW"], s=10, alpha=0.35, label="Infeasible")
    plt.axhline(float(peak_limit_kw), color="black", linewidth=1.5, label="P_max")
    if best_row is not None:
        plt.scatter([best_row["E_cool_el_2040_kWh"]], [best_row["P_cool_el_peak_2040_kW"]], s=120, marker="*", label="Best feasible")
    plt.xlabel("Annual Cooling Electricity 2040 (kWh_el)")
    plt.ylabel("Peak Cooling Electric Power 2040 (kW_el)")
    plt.title("Feasible Region: Cooling Energy vs Peak Power")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "feasible_region_scatter.png", dpi=200)
    plt.close()

    # 2) peak_limit_sensitivity
    rows = []
    for pmax in peak_limit_list:
        feas_sub = df_samples[df_samples["P_cool_el_peak_2040_kW"] <= float(pmax)]
        frac = float(len(feas_sub)) / float(len(df_samples)) if len(df_samples) else 0.0
        best_val = float(feas_sub["E_cool_el_2040_kWh"].min()) if len(feas_sub) else float("nan")
        rows.append({"P_max_kW": float(pmax), "best_feasible_E_cool_el_2040_kWh": best_val, "feasible_fraction": frac})
    sens_df = pd.DataFrame(rows)
    sens_df.to_csv(data_dir / "sensitivity_peak_limit.csv", index=False)

    fig, ax1 = plt.subplots()
    ax1.plot(sens_df["P_max_kW"], sens_df["best_feasible_E_cool_el_2040_kWh"], marker="o", label="Best feasible energy")
    ax1.set_xlabel("P_max (kW_el)")
    ax1.set_ylabel("Best feasible E_cool_el_2040 (kWh_el)")
    ax2 = ax1.twinx()
    ax2.plot(sens_df["P_max_kW"], sens_df["feasible_fraction"], marker="s", color="orange", label="Feasible fraction")
    ax2.set_ylabel("Feasible fraction")
    plt.title("Peak Limit Sensitivity")
    fig.tight_layout()
    fig.savefig(plots_dir / "peak_limit_sensitivity.png", dpi=200)
    plt.close(fig)

    # 3) baseline_vs_best_bar
    best_for_bar = baseline_row if best_row is None else best_row
    labels = ["Baseline", "Best feasible"]
    vals_energy = [baseline_row["E_cool_el_2040_kWh"], best_for_bar["E_cool_el_2040_kWh"]]
    vals_peak = [baseline_row["P_cool_el_peak_2040_kW"], best_for_bar["P_cool_el_peak_2040_kW"]]
    vals_mnz = [baseline_row.get("MNZ_kWh_el", np.nan), best_for_bar.get("MNZ_kWh_el", np.nan)]
    x = np.arange(len(labels))
    w = 0.25
    plt.figure()
    plt.bar(x - w, vals_energy, width=w, label="Annual Cooling Electricity 2040")
    plt.bar(x, vals_peak, width=w, label="Peak Cooling Power 2040")
    plt.bar(x + w, vals_mnz, width=w, label="MNZ")
    plt.xticks(x, labels)
    plt.ylabel("Value")
    plt.title("Baseline vs Best Feasible")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "baseline_vs_best_bar.png", dpi=200)
    plt.close()

    # 4) peak_hour_hist
    plt.figure()
    plt.hist(feas["peak_hour_local"], bins=24, alpha=0.6, label="Feasible")
    plt.hist(infeas["peak_hour_local"], bins=24, alpha=0.6, label="Infeasible")
    plt.xlabel("Peak hour (local)")
    plt.ylabel("Count")
    plt.title("Peak Hour Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "peak_hour_hist.png", dpi=200)
    plt.close()

    # 5) peak_attribution_scatter
    plt.figure()
    sc = plt.scatter(
        df_samples["T_out_2040_at_peak_C"],
        df_samples["P_cool_el_peak_2040_kW"],
        c=df_samples["solar_gain_at_peak_W"],
        cmap="viridis",
        s=10,
        alpha=0.7,
    )
    plt.xlabel("T_out_2040_at_peak (°C)")
    plt.ylabel("Peak Cooling Electric Power 2040 (kW_el)")
    plt.title("Peak Attribution by Solar Gain")
    plt.colorbar(sc, label="Solar gain at peak (W)")
    plt.tight_layout()
    plt.savefig(plots_dir / "peak_attribution_scatter.png", dpi=200)
    plt.close()

    # 6) shading_u_timeseries_example
    def _plot_u_examples(series: Dict[str, Any]) -> None:
        u = series["irr"]["u"]
        T_out = series["T_out_2040_C"]
        hottest_day = int(np.argmax(pd.Series(T_out, index=times).resample("D").mean()))
        day_list = [80, 200, hottest_day]
        fig, axs = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
        for ax, day in zip(axs, day_list):
            if day >= len(times) // 24:
                day = len(times) // 24 - 1
            start = day * 24
            end = start + 24
            hrs = np.arange(24)
            ax.plot(hrs, u["N"][start:end], label="uN")
            ax.plot(hrs, u["E"][start:end], label="uE")
            ax.plot(hrs, u["S"][start:end], label="uS")
            ax.plot(hrs, u["W"][start:end], label="uW")
            ax.set_ylabel("u")
            ax.set_title(f"Day {day}")
        axs[-1].set_xlabel("Hour")
        axs[0].legend()
        fig.suptitle("Shading u_f Timeseries Example")
        fig.tight_layout()
        fig.savefig(plots_dir / "shading_u_timeseries_example.png", dpi=200)
        plt.close(fig)

    _plot_u_examples(best_series)

    # 7) beam_with_vs_without_shading (South facade, hottest day)
    irr = best_series["irr"]
    mu = irr["mu"]["S"]
    uS = irr["u"]["S"]
    DNI = df_base["DNI"].to_numpy(float)
    T_out = best_series["T_out_2040_C"]
    hottest_day = int(np.argmax(pd.Series(T_out, index=times).resample("D").mean())) if len(T_out) else 0
    start = max(0, hottest_day * 24)
    end = min(start + 24, len(DNI))
    if end > start:
        hrs = np.arange(end - start)
        beam = DNI[start:end] * mu[start:end]
        beam_shaded = uS[start:end] * DNI[start:end] * mu[start:end]
        plt.figure()
        plt.plot(hrs, beam, label="DNI*mu")
        plt.plot(hrs, beam_shaded, label="u*DNI*mu")
        plt.xlabel("Hour")
        plt.ylabel("Beam irradiance (W/m2)")
        plt.title("Beam: With vs Without Shading (South, hottest day)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "beam_with_vs_without_shading.png", dpi=200)
        plt.close()
    else:
        plt.figure()
        plt.text(0.5, 0.5, "No valid beam data", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(plots_dir / "beam_with_vs_without_shading.png", dpi=200)
        plt.close()

    # 8) solar_gains_facade_breakdown
    def _annual_facade_gains(series: Dict[str, Any], geom: GeometryDecision) -> Dict[str, float]:
        g_eq = float(getattr(optical_base, "tau_heat", 0.55))
        A_win = geom.window_areas_m2()
        I_f = series["irr"]["I_f"]
        dt_h = float(hvac_cfg.dt_hours)
        out = {}
        for f in ["N", "E", "S", "W"]:
            Phi = A_win[f] * g_eq * I_f[f]
            out[f] = float(np.sum(Phi) * dt_h / 1000.0)
        return out

    base_g = _annual_facade_gains(base_series, base_series["geom"])
    best_g = _annual_facade_gains(best_series, best_series["geom"])
    facades = ["N", "E", "S", "W"]
    base_vals = [base_g[f] for f in facades]
    best_vals = [best_g[f] for f in facades]
    fig, ax = plt.subplots()
    ax.bar([0, 1], [base_vals[0], best_vals[0]], label="N")
    bottom = [base_vals[0], best_vals[0]]
    for i, f in enumerate(facades[1:], start=1):
        ax.bar([0, 1], [base_vals[i], best_vals[i]], bottom=bottom, label=f)
        bottom = [bottom[0] + base_vals[i], bottom[1] + best_vals[i]]
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Baseline", "Best"])
    ax.set_ylabel("Annual solar gains (kWh)")
    ax.set_title("Annual Solar Gains by Facade")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "solar_gains_facade_breakdown.png", dpi=200)
    plt.close(fig)

    # 9) cooling_duration_curve
    plt.figure()
    p_base = np.sort(base_series["P_cool_el_kW"])[::-1]
    p_best = np.sort(best_series["P_cool_el_kW"])[::-1]
    plt.plot(p_base, label="Baseline")
    plt.plot(p_best, label="Best")
    plt.xlabel("Hours (sorted)")
    plt.ylabel("Cooling power (kW_el)")
    plt.title("Cooling Power Duration Curve (2040)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "cooling_duration_curve.png", dpi=200)
    plt.close()

    # 10) monthly_cooling_electricity
    m_base = _monthly_sum_kwh(times, base_series["P_cool_el_kW"], dt_hours=hvac_cfg.dt_hours)
    m_best = _monthly_sum_kwh(times, best_series["P_cool_el_kW"], dt_hours=hvac_cfg.dt_hours)
    months = np.arange(1, len(m_base) + 1)
    w = 0.35
    plt.figure()
    plt.bar(months - w/2, m_base.values, width=w, label="Baseline")
    plt.bar(months + w/2, m_best.values, width=w, label="Best")
    plt.xlabel("Month")
    plt.ylabel("Monthly cooling electricity (kWh_el)")
    plt.title("Monthly Cooling Electricity (2040)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "monthly_cooling_electricity.png", dpi=200)
    plt.close()

    # 11) material_boxplot_energy
    plt.figure()
    data1 = df_samples[df_samples["material_id"] == 1]["E_cool_el_2040_kWh"].to_numpy()
    data2 = df_samples[df_samples["material_id"] == 2]["E_cool_el_2040_kWh"].to_numpy()
    plt.boxplot([data1, data2], labels=["Material 1", "Material 2"])
    plt.ylabel("Annual Cooling Electricity 2040 (kWh_el)")
    plt.title("Energy by Material")
    plt.tight_layout()
    plt.savefig(plots_dir / "material_boxplot_energy.png", dpi=200)
    plt.close()

    # 12) material_boxplot_peak
    plt.figure()
    data1 = df_samples[df_samples["material_id"] == 1]["P_cool_el_peak_2040_kW"].to_numpy()
    data2 = df_samples[df_samples["material_id"] == 2]["P_cool_el_peak_2040_kW"].to_numpy()
    plt.boxplot([data1, data2], labels=["Material 1", "Material 2"])
    plt.ylabel("Peak Cooling Electric Power 2040 (kW_el)")
    plt.title("Peak Power by Material")
    plt.tight_layout()
    plt.savefig(plots_dir / "material_boxplot_peak.png", dpi=200)
    plt.close()

    # 13) state_trajectories_week
    start = _hottest_week_start(times, best_series["T_out_2040_C"])
    end = start + 24 * 7
    t_slice = times[start:end]
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axs[0].plot(t_slice, base_series["Ti_with_C"][start:end], label="T_air")
    axs[0].plot(t_slice, base_series["Tm_with_C"][start:end], label="T_mass")
    axs[0].plot(t_slice, base_series["T_out_2040_C"][start:end], label="T_out_2040")
    axs[0].set_title("Baseline: Hottest Week")
    axs[0].legend()
    axs[1].plot(t_slice, best_series["Ti_with_C"][start:end], label="T_air")
    axs[1].plot(t_slice, best_series["Tm_with_C"][start:end], label="T_mass")
    axs[1].plot(t_slice, best_series["T_out_2040_C"][start:end], label="T_out_2040")
    axs[1].set_title("Best Feasible: Hottest Week")
    axs[1].legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "state_trajectories_week.png", dpi=200)
    plt.close(fig)

    # 14) violation_hist
    violation = infeas["P_cool_el_peak_2040_kW"] - float(peak_limit_kw)
    plt.figure()
    plt.hist(violation, bins=30)
    plt.xlabel("Violation (kW_el)")
    plt.ylabel("Count")
    plt.title("Peak Limit Violation Histogram")
    plt.tight_layout()
    plt.savefig(plots_dir / "violation_hist.png", dpi=200)
    plt.close()

    # 15) correlation_heatmap
    cols = [
        "wwr_N", "wwr_E", "wwr_S", "wwr_W",
        "dN_m", "dS_m", "etaE_rad", "etaW_rad",
        "L_m", "W_m", "H_m", "theta_deg", "material_id",
        "E_cool_el_2040_kWh", "P_cool_el_peak_2040_kW", "MNZ_kWh_el",
        "solar_gain_at_peak_W",
    ]
    corr_df = df_samples[cols].corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    plt.imshow(corr_df.values, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(cols)), cols, rotation=90)
    plt.yticks(range(len(cols)), cols)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(plots_dir / "correlation_heatmap.png", dpi=200)
    plt.close()

    # 16) temperature_baseline_vs_2040
    T_base = df_base["T_out"].to_numpy(float)
    T_2040 = T_base + float(delta_trend_C)
    daily_base = _daily_mean(times, T_base)
    daily_2040 = _daily_mean(times, T_2040)
    temp_daily_df = pd.DataFrame(
        {
            "date_local": daily_base.index.date.astype(str),
            "T_out_baseline_C": daily_base.values,
            "T_out_2040_C": daily_2040.values,
        }
    )
    temp_daily_df.to_csv(data_dir / "temperature_daily.csv", index=False)

    plt.figure()
    plt.plot(daily_base.index, daily_base.values, marker="o", markersize=2, linewidth=1, label="Baseline T_out")
    plt.plot(daily_2040.index, daily_2040.values, marker="o", markersize=2, linewidth=1, label="2040 T_out")
    plt.xlabel("Date")
    plt.ylabel("Outdoor Air Temperature (°C)")
    plt.title("Daily Outdoor Temperature: Baseline vs 2040")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "temperature_baseline_vs_2040.png", dpi=200)
    plt.close()

    # 17) indoor_temperature_2040_with_vs_without_cooling (daily)
    daily_with = _daily_mean(times, best_series["Ti_with_C"])
    daily_no = _daily_mean(times, best_series["Ti_no_C"])
    indoor_daily_df = pd.DataFrame(
        {
            "date_local": daily_with.index.date.astype(str),
            "T_air_2040_with_cooling_C": daily_with.values,
            "T_air_2040_no_cooling_C": daily_no.values,
        }
    )
    indoor_daily_df.to_csv(
        data_dir / "indoor_temperature_2040_with_vs_without_cooling_daily.csv",
        index=False,
    )

    plt.figure()
    plt.plot(daily_with.index, daily_with.values, marker="o", markersize=2, linewidth=1, label="With cooling")
    plt.plot(daily_no.index, daily_no.values, marker="o", markersize=2, linewidth=1, label="No cooling")
    plt.xlabel("Date")
    plt.ylabel("Indoor Air Temperature (°C)")
    plt.title("2040 Indoor Air Temperature: With vs Without Cooling")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "indoor_temperature_2040_with_vs_without_cooling.png", dpi=200)
    plt.close()

    # 18A) heatwave week and hottest day hourly outputs
    heat_start = _hottest_week_start(times, best_series["T_out_2040_C"])
    heat_end = heat_start + 24 * 7
    heat_slice = times[heat_start:heat_end]
    heat_df = pd.DataFrame(
        {
            "datetime_local": _index_to_str(heat_slice),
            "T_out_2040_C": best_series["T_out_2040_C"][heat_start:heat_end],
            "T_air_with_cooling_C": best_series["Ti_with_C"][heat_start:heat_end],
            "T_air_no_cooling_C": best_series["Ti_no_C"][heat_start:heat_end],
            "P_cool_el_kW": best_series["P_cool_el_kW"][heat_start:heat_end],
        }
    )
    heat_df.to_csv(data_dir / "indoor_temp_2040_heatwave_week_hourly.csv", index=False)

    plt.figure(figsize=(10, 4))
    plt.plot(heat_slice, heat_df["T_air_with_cooling_C"], label="Indoor with cooling")
    plt.plot(heat_slice, heat_df["T_air_no_cooling_C"], label="Indoor no cooling")
    plt.plot(heat_slice, heat_df["T_out_2040_C"], label="Outdoor 2040", color="gray", linewidth=1)
    plt.axhline(float(hvac_cfg.T_cool_C), color="black", linestyle="--", linewidth=1, label="Cooling setpoint")
    plt.xlabel("Datetime (local)")
    plt.ylabel("Indoor Air Temperature (°C)")
    plt.title("Heatwave Week: Indoor Temperature (2040)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "indoor_temp_2040_heatwave_week_with_vs_without_cooling_hourly.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(heat_slice, heat_df["T_air_with_cooling_C"], label="Indoor with cooling")
    plt.plot(heat_slice, heat_df["T_air_no_cooling_C"], label="Indoor no cooling")
    plt.plot(heat_slice, heat_df["T_out_2040_C"], label="Outdoor 2040", color="gray", linewidth=1)
    plt.axhline(float(hvac_cfg.T_cool_C), color="black", linestyle="--", linewidth=1, label="Cooling setpoint")
    ax2 = plt.gca().twinx()
    ax2.plot(heat_slice, heat_df["P_cool_el_kW"], color="tab:red", alpha=0.6, label="Cooling power (kW)")
    ax2.set_ylabel("Cooling power (kW_el)")
    plt.xlabel("Datetime (local)")
    plt.ylabel("Indoor Air Temperature (°C)")
    plt.title("Heatwave Week: Temperature & Cooling Power")
    plt.tight_layout()
    plt.savefig(plots_dir / "heatwave_week_temp_and_cooling_power.png", dpi=200)
    plt.close()

    hottest_start = _hottest_day_start(times, best_series["T_out_2040_C"])
    hottest_end = hottest_start + 24
    hottest_slice = times[hottest_start:hottest_end]
    hottest_df = pd.DataFrame(
        {
            "datetime_local": _index_to_str(hottest_slice),
            "T_out_2040_C": best_series["T_out_2040_C"][hottest_start:hottest_end],
            "T_air_with_cooling_C": best_series["Ti_with_C"][hottest_start:hottest_end],
            "T_air_no_cooling_C": best_series["Ti_no_C"][hottest_start:hottest_end],
            "P_cool_el_kW": best_series["P_cool_el_kW"][hottest_start:hottest_end],
        }
    )
    hottest_df.to_csv(data_dir / "indoor_temp_2040_hottest_day_hourly.csv", index=False)

    plt.figure(figsize=(10, 4))
    plt.plot(hottest_slice, hottest_df["T_air_with_cooling_C"], label="Indoor with cooling")
    plt.plot(hottest_slice, hottest_df["T_air_no_cooling_C"], label="Indoor no cooling")
    plt.plot(hottest_slice, hottest_df["T_out_2040_C"], label="Outdoor 2040", color="gray", linewidth=1)
    plt.axhline(float(hvac_cfg.T_cool_C), color="black", linestyle="--", linewidth=1, label="Cooling setpoint")
    plt.xlabel("Datetime (local)")
    plt.ylabel("Indoor Air Temperature (°C)")
    plt.title("Hottest Day: Indoor Temperature (2040)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "indoor_temp_2040_hottest_day_with_vs_without_cooling_hourly.png", dpi=200)
    plt.close()

    # 18B) overheat metrics
    overheat_thr = 26.0
    overheat_hours_no = int(np.sum(best_series["Ti_no_C"] > overheat_thr))
    overheat_hours_with = int(np.sum(best_series["Ti_with_C"] > overheat_thr))
    overheat_deg_no = float(np.sum(np.maximum(best_series["Ti_no_C"] - overheat_thr, 0.0)))
    overheat_deg_with = float(np.sum(np.maximum(best_series["Ti_with_C"] - overheat_thr, 0.0)))
    overheat_df = pd.DataFrame(
        [
            {
                "design_id": "best" if best_row is not None else "baseline",
                "scenario": "2040",
                "overheat_hours_no_cooling": overheat_hours_no,
                "overheat_hours_with_cooling": overheat_hours_with,
                "overheat_degree_hours_no_cooling": overheat_deg_no,
                "overheat_degree_hours_with_cooling": overheat_deg_with,
            }
        ]
    )
    overheat_df.to_csv(data_dir / "overheat_metrics_2040.csv", index=False)

    plt.figure()
    plt.bar(["No cooling", "With cooling"], [overheat_hours_no, overheat_hours_with], label="Overheat hours")
    plt.ylabel("Hours")
    plt.title("2040 Overheating Risk: With vs Without Cooling")
    plt.tight_layout()
    plt.savefig(plots_dir / "overheat_metrics_2040_bar.png", dpi=200)
    plt.close()

    # 18C) indoor temperature distribution
    plt.figure()
    plt.hist(best_series["Ti_with_C"], bins=40, alpha=0.6, label="With cooling")
    plt.hist(best_series["Ti_no_C"], bins=40, alpha=0.6, label="No cooling")
    plt.xlabel("Indoor Air Temperature (°C)")
    plt.ylabel("Frequency")
    plt.title("Indoor Temperature Distribution (2040)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "indoor_temp_distribution_2040_with_vs_without_cooling.png", dpi=200)
    plt.close()

    # 18D) comfort fraction
    c_low = float(comfort_low_C)
    c_high = float(comfort_high_C)
    comfort_with = float(np.mean((best_series["Ti_with_C"] >= c_low) & (best_series["Ti_with_C"] <= c_high)))
    comfort_no = float(np.mean((best_series["Ti_no_C"] >= c_low) & (best_series["Ti_no_C"] <= c_high)))
    comfort_df = pd.DataFrame(
        [
            {
                "comfort_low_C": c_low,
                "comfort_high_C": c_high,
                "comfort_fraction_with_cooling": comfort_with,
                "comfort_fraction_no_cooling": comfort_no,
            }
        ]
    )
    comfort_df.to_csv(data_dir / "comfort_fraction_2040.csv", index=False)

    plt.figure()
    plt.bar(["With cooling", "No cooling"], [comfort_with, comfort_no])
    plt.ylabel("Comfort fraction")
    plt.title("Comfort Fraction (2040)")
    plt.tight_layout()
    plt.savefig(plots_dir / "comfort_fraction_2040_bar.png", dpi=200)
    plt.close()

    # ECharts output (feasible scatter + sensitivity + temperature + indoor temperature)
    _write_echarts_min_js(echarts_dir)
    charts_payload = _build_echarts_payload(
        df_samples=df_samples,
        feasible_df=feasible_df,
        best_row=best_row,
        peak_limit_kw=peak_limit_kw,
        sens_df=sens_df,
        temp_daily_df=temp_daily_df,
        indoor_daily_df=indoor_daily_df,
    )
    (echarts_dir / "charts.json").write_text(json.dumps(charts_payload, indent=2), encoding="utf-8")
    _write_paper_echarts_html(echarts_dir / "index.html", charts_payload)

    return {
        "plots_dir": plots_dir,
        "data_dir": data_dir,
        "echarts_dir": echarts_dir,
        "heatwave_start": str(times[heat_start].date()) if len(times) else "",
        "heatwave_end": str((times[min(heat_end - 1, len(times) - 1)]).date()) if len(times) else "",
        "max_temp_no_cooling_C": float(np.max(best_series["Ti_no_C"])),
        "max_temp_with_cooling_C": float(np.max(best_series["Ti_with_C"])),
        "overheat_hours_no": overheat_hours_no,
        "overheat_hours_with": overheat_hours_with,
    }
