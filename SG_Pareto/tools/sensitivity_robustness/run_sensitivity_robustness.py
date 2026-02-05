from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from datetime import datetime
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

_root = None
for parent in Path(__file__).resolve().parents:
    if (parent / "borealis_model").exists():
        _root = parent
        break
if _root is not None and str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from borealis_model.data_io import load_weather_csv

from SG_Pareto.config import (
    ANALYSIS_ROOT,
    SINGAPORE,
    DEFAULT_OPTICAL,
    DEFAULT_BOREALIS_OPTICAL,
    DEFAULT_HVAC,
    DEFAULT_PV,
    MATERIALS,
    WINDOW_HEIGHT_M,
    USE_AR1_NOISE,
    AR1_SEED,
    DELTA_TREND_C,
)
from SG_Pareto.geometry import GeometryDecision
from SG_Pareto.solar_facade_sg import compute_facade_irradiance
from SG_Pareto.thermal_sg import simulate_2r2c_deadband


@dataclass(frozen=True)
class FixedDesign:
    geometry: GeometryDecision
    dN_m: float
    dS_m: float
    etaE_rad: float
    etaW_rad: float
    material_id: int


def _find_latest_run_dir() -> Path | None:
    runs = sorted(ANALYSIS_ROOT.glob("sg_pareto_*"))
    return runs[-1] if runs else None


def _load_fixed_design(run_dir: Path) -> tuple[FixedDesign, str]:
    results_dir = run_dir / "results"
    best_path = results_dir / "best_feasible.csv"
    baseline_path = results_dir / "baseline.csv"
    source = "baseline"
    if best_path.exists():
        df_best = pd.read_csv(best_path)
        if not df_best.empty:
            row = df_best.iloc[0].to_dict()
            source = "best_feasible"
        else:
            row = pd.read_csv(baseline_path).iloc[0].to_dict()
    else:
        row = pd.read_csv(baseline_path).iloc[0].to_dict()

    geom = GeometryDecision(
        L_m=float(row["L_m"]),
        W_m=float(row["W_m"]),
        H_m=float(row["H_m"]),
        theta_deg=float(row["theta_deg"]),
        wwr_N=float(row["wwr_N"]),
        wwr_E=float(row["wwr_E"]),
        wwr_S=float(row["wwr_S"]),
        wwr_W=float(row["wwr_W"]),
    )
    design = FixedDesign(
        geometry=geom,
        dN_m=float(row.get("dN_m", 0.0)),
        dS_m=float(row.get("dS_m", 0.0)),
        etaE_rad=float(row.get("etaE_rad", 0.0)),
        etaW_rad=float(row.get("etaW_rad", 0.0)),
        material_id=int(row.get("material_id", row.get("material_m", 2))),
    )
    return design, source


def _ar1_noise(n: int, *, phi: float = 0.8, sigma: float = 0.35, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, sigma, size=n)
    x = np.zeros(n, dtype=float)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + eps[i]
    return x


def _prepare_typical_year(
    df_base: pd.DataFrame,
    Phi_s_W: np.ndarray,
) -> dict[str, np.ndarray]:
    times = df_base.index
    if getattr(times, "tz", None) is not None:
        times_local = times.tz_localize(None)
    else:
        times_local = times

    doy = times_local.dayofyear.to_numpy(int)
    hour = times_local.hour.to_numpy(int)
    df = pd.DataFrame(
        {
            "doy": doy,
            "hour": hour,
            "T_out": df_base["T_out"].to_numpy(float),
            "GHI": df_base["GHI"].to_numpy(float),
            "Phi_s": Phi_s_W,
        }
    )
    df = df[df["doy"] != 366]
    index = pd.MultiIndex.from_product([range(1, 366), range(24)], names=["doy", "hour"])
    grouped = df.groupby(["doy", "hour"]).mean().reindex(index)
    grouped = grouped.interpolate().ffill().bfill()

    return {
        "doy": grouped.index.get_level_values("doy").to_numpy(int),
        "hour": grouped.index.get_level_values("hour").to_numpy(int),
        "T_out": grouped["T_out"].to_numpy(float),
        "GHI": grouped["GHI"].to_numpy(float),
        "Phi_s": grouped["Phi_s"].to_numpy(float),
    }


def _compute_margin_daily(
    *,
    T_out_base: np.ndarray,
    GHI: np.ndarray,
    Phi_s: np.ndarray,
    design: FixedDesign,
    delta_T: float,
    COP: float,
    ar1_noise: np.ndarray,
    dt_hours: float,
) -> dict[str, np.ndarray]:
    T_out_2040 = T_out_base + float(delta_T) + ar1_noise
    material = MATERIALS[int(design.material_id)]

    Ti, Tm, Phi_h = simulate_2r2c_deadband(
        Phi_s_W=Phi_s,
        T_out_C=T_out_2040,
        Ci=material.Ci_J_per_K,
        Cm=material.Cm_J_per_K,
        Ria=material.Ria_K_per_W,
        Rim=material.Rim_K_per_W,
        eta=material.eta_air,
        T_heat_C=DEFAULT_HVAC.T_heat_C,
        T_cool_C=DEFAULT_HVAC.T_cool_C,
        dt_hours=dt_hours,
        T_init_C=DEFAULT_HVAC.T_init_C,
    )
    P_el_kW = np.maximum(-Phi_h, 0.0) / 1000.0 / float(COP)

    A_rf = design.geometry.roof_area_m2()
    P_pv_kW = (float(DEFAULT_PV.eta_pv) * A_rf * GHI) / 1000.0
    P_pv_kW = P_pv_kW * float(DEFAULT_PV.alpha_to_cool)

    net_kWh = (P_pv_kW - P_el_kW) * float(dt_hours)
    n_days = len(net_kWh) // 24
    net_kWh = net_kWh[: n_days * 24]
    margin_kWh = np.cumsum(net_kWh)
    margin_daily = margin_kWh.reshape(n_days, 24)[:, -1]
    net_daily = net_kWh.reshape(n_days, 24).sum(axis=1)

    return {
        "margin_daily": margin_daily,
        "net_daily": net_daily,
    }


def _select_vulnerable_window(net_daily: np.ndarray, window_days: int = 14) -> tuple[int, int]:
    if len(net_daily) < window_days:
        return 1, len(net_daily)
    rolling = np.convolve(net_daily, np.ones(window_days), mode="valid")
    start = int(np.argmin(rolling)) + 1
    end = start + window_days - 1
    return start, end


def _plot_sensitivity(
    out_dir: Path,
    timeseries: list[dict],
    window: tuple[int, int],
    y_limits: tuple[float, float],
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6), sharey=True)
    (w_start, w_end) = window

    for ax, subplot in zip(axes, ["a", "b"]):
        curves = [t for t in timeseries if t["subplot"] == subplot]
        for t in curves:
            ax.plot(t["day"], t["margin"], label=t["label"], linewidth=1.5)
        ax.axhline(0.0, color="#333333", linestyle="--", linewidth=1.0, label="Net-zero threshold")
        ax.set_xlim(1, 365)
        ax.set_ylim(y_limits[0], y_limits[1])
        ax.set_xlabel("Time (day)")
        ax.set_ylabel("Cumulative Net-zero Margin (kWh)")
        if subplot == "a":
            ax.set_title("(a) Adjustment of ΔT_trend")
        else:
            ax.set_title("(b) Adjustment of COP")
        ax.legend(frameon=False, fontsize=8, loc="upper left")

        inset = inset_axes(ax, width="45%", height="45%", loc="upper right", borderpad=1.2)
        window_vals = []
        for t in curves:
            window_vals.extend(t["margin"][w_start - 1 : w_end])
            inset.plot(
                t["day"][w_start - 1 : w_end],
                t["margin"][w_start - 1 : w_end],
                linewidth=1.0,
            )
        if window_vals:
            vmin = float(np.min(window_vals))
            vmax = float(np.max(window_vals))
            pad = 0.05 * (vmax - vmin + 1e-6)
            inset.set_ylim(vmin - pad, vmax + pad)
        inset.set_xlim(w_start, w_end)
        if inset.get_ylim()[0] <= 0.0 <= inset.get_ylim()[1]:
            inset.axhline(0.0, color="#333333", linestyle="--", linewidth=0.8)
        inset.set_title("Zoom: most vulnerable 2-week window", fontsize=8)
        inset.tick_params(axis="both", labelsize=7)
        mark_inset(ax, inset, loc1=2, loc2=4, fc="none", ec="black", lw=0.7)

    fig.set_layout_engine("none")
    png_path = out_dir / "plots" / "figure1_sensitivity_margin.png"
    fig.savefig(png_path, dpi=220)
    plt.close(fig)
    return png_path


def _write_echarts_figure1(
    out_dir: Path,
    df_ts: pd.DataFrame,
    inset_window: tuple[int, int],
) -> tuple[Path, Path]:
    echarts_dir = out_dir / "echarts"
    echarts_dir.mkdir(parents=True, exist_ok=True)

    def _build_subplot(subplot: str, label_fmt) -> list[dict]:
        sub = df_ts[df_ts["subplot"] == subplot].copy()
        groups = sorted(sub.groupby("param_value"), key=lambda kv: kv[0])
        series = []
        for val, g in groups:
            g = g.sort_values("day")
            data = g[["day", "margin_kwh"]].values.tolist()
            series.append({"name": label_fmt(float(val)), "data": data})
        return series

    subplot_a = _build_subplot("a", lambda v: f"ΔT={v:.1f}°C")
    subplot_b = _build_subplot("b", lambda v: f"COP={v:.2f}")

    all_vals = []
    for s in subplot_a + subplot_b:
        all_vals.extend([pt[1] for pt in s["data"]])
    if not all_vals:
        all_vals = [0.0, 1.0]
    y_min = float(np.min(all_vals))
    y_max = float(np.max(all_vals))
    pad = 0.03 * (y_max - y_min + 1e-6)
    y_main = {"min": y_min - pad, "max": y_max + pad}

    def _inset_range(series_list: list[dict], start: int, end: int) -> dict:
        vals = []
        for s in series_list:
            for day, val in s["data"]:
                if start <= day <= end:
                    vals.append(val)
        if not vals:
            vals = all_vals
        vmin = float(np.min(vals))
        vmax = float(np.max(vals))
        pad_local = 0.05 * (vmax - vmin + 1e-6)
        return {"min": vmin - pad_local, "max": vmax + pad_local}

    inset_start, inset_end = inset_window
    inset_a = _inset_range(subplot_a, inset_start, inset_end)
    inset_b = _inset_range(subplot_b, inset_start, inset_end)

    data = {
        "meta": {
            "x_min": 1,
            "x_max": 365,
            "y_label": "Cumulative Net-zero Margin (kWh)",
            "y_min": y_main["min"],
            "y_max": y_main["max"],
        },
        "subplot_a": subplot_a,
        "subplot_b": subplot_b,
        "inset": {
            "start": int(inset_start),
            "end": int(inset_end),
            "y_inset_a": inset_a,
            "y_inset_b": inset_b,
        },
    }

    json_path = echarts_dir / "figure1_sensitivity_margin.json"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    template = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Sensitivity Analysis Figure</title>
  <script src="./echarts.min.js"></script>
  <style>
    body { margin: 16px; font-family: "Segoe UI", Arial, sans-serif; background: #fff; }
    #chart { width: 1600px; max-width: 100%; height: 700px; margin: 0 auto; background: #fff; }
  </style>
</head>
<body>
  <div id="chart"></div>
  <script>
    const DATA = __DATA__;
    const chart = echarts.init(document.getElementById('chart'));

    const palette = ['#1f77b4', '#ff7f0e', '#2ca02c'];

    function fmtTick(val) {
      const rounded = Math.round(val);
      if (Math.abs(val - rounded) < 1e-6) return rounded.toLocaleString('en-US');
      return val.toFixed(1);
    }

    const zoom = DATA.inset;
    const yMainMin = DATA.meta.y_min;
    const yMainMax = DATA.meta.y_max;

    const series = [];
    const legendA = [];
    const legendB = [];

    DATA.subplot_a.forEach((s, idx) => {
      legendA.push(s.name);
      const color = palette[idx % palette.length];
      series.push({
        name: s.name,
        type: 'line',
        data: s.data,
        xAxisIndex: 0,
        yAxisIndex: 0,
        showSymbol: false,
        lineStyle: { width: 2, color: color }
      });
      series.push({
        name: s.name + '_inset',
        type: 'line',
        data: s.data.filter(pt => pt[0] >= zoom.start && pt[0] <= zoom.end),
        xAxisIndex: 2,
        yAxisIndex: 2,
        showSymbol: false,
        lineStyle: { width: 1.5, color: color },
        silent: true,
        tooltip: { show: false }
      });
    });

    DATA.subplot_b.forEach((s, idx) => {
      legendB.push(s.name);
      const color = palette[idx % palette.length];
      series.push({
        name: s.name,
        type: 'line',
        data: s.data,
        xAxisIndex: 1,
        yAxisIndex: 1,
        showSymbol: false,
        lineStyle: { width: 2, color: color }
      });
      series.push({
        name: s.name + '_inset',
        type: 'line',
        data: s.data.filter(pt => pt[0] >= zoom.start && pt[0] <= zoom.end),
        xAxisIndex: 3,
        yAxisIndex: 3,
        showSymbol: false,
        lineStyle: { width: 1.5, color: color },
        silent: true,
        tooltip: { show: false }
      });
    });

    legendA.push('Net-zero threshold (a)');
    legendB.push('Net-zero threshold (b)');

    series.push({
      name: 'Net-zero threshold (a)',
      type: 'line',
      data: [[1, 0], [365, 0]],
      xAxisIndex: 0,
      yAxisIndex: 0,
      showSymbol: false,
      lineStyle: { width: 1, type: 'dashed', color: '#222' },
      silent: true
    });
    series.push({
      name: 'Net-zero threshold (b)',
      type: 'line',
      data: [[1, 0], [365, 0]],
      xAxisIndex: 1,
      yAxisIndex: 1,
      showSymbol: false,
      lineStyle: { width: 1, type: 'dashed', color: '#222' },
      silent: true
    });

    if (zoom.y_inset_a.min <= 0 && zoom.y_inset_a.max >= 0) {
      series.push({
        name: '__zero_inset_a',
        type: 'line',
        data: [[zoom.start, 0], [zoom.end, 0]],
        xAxisIndex: 2,
        yAxisIndex: 2,
        showSymbol: false,
        lineStyle: { width: 1, type: 'dashed', color: '#777' },
        silent: true,
        tooltip: { show: false }
      });
    }
    if (zoom.y_inset_b.min <= 0 && zoom.y_inset_b.max >= 0) {
      series.push({
        name: '__zero_inset_b',
        type: 'line',
        data: [[zoom.start, 0], [zoom.end, 0]],
        xAxisIndex: 3,
        yAxisIndex: 3,
        showSymbol: false,
        lineStyle: { width: 1, type: 'dashed', color: '#777' },
        silent: true,
        tooltip: { show: false }
      });
    }

    const option = {
      backgroundColor: '#fff',
      animation: false,
      toolbox: { right: 10, top: 6, feature: { saveAsImage: { pixelRatio: 2, title: 'Download' } } },
      title: [
        { text: '(a) Adjustment of ΔT_trend', left: '7%', top: 6, textStyle: { fontSize: 13, fontWeight: 'normal' } },
        { text: '(b) Adjustment of COP', left: '53%', top: 6, textStyle: { fontSize: 13, fontWeight: 'normal' } },
        { text: 'Zoom: most vulnerable 2-week window', left: '28%', top: 100, textStyle: { fontSize: 10, color: '#444' } },
        { text: 'Zoom: most vulnerable 2-week window', left: '74%', top: 100, textStyle: { fontSize: 10, color: '#444' } }
      ],
      legend: [
        { data: legendA, left: '7%', top: 28, orient: 'vertical', itemWidth: 12, itemHeight: 2, itemGap: 4,
          formatter: name => name.replace(' (a)', ''), textStyle: { fontSize: 11 } },
        { data: legendB, left: '53%', top: 28, orient: 'vertical', itemWidth: 12, itemHeight: 2, itemGap: 4,
          formatter: name => name.replace(' (b)', ''), textStyle: { fontSize: 11 } }
      ],
      grid: [
        { left: '7%', right: '53%', top: 80, bottom: 70 },
        { left: '53%', right: '7%', top: 80, bottom: 70 },
        { left: '28%', top: 120, width: '18%', height: '32%' },
        { left: '74%', top: 120, width: '18%', height: '32%' }
      ],
      xAxis: [
        { type: 'value', min: 1, max: 365, name: 'Time (day)', nameLocation: 'middle', nameGap: 36, gridIndex: 0,
          axisLabel: { fontSize: 11 }, splitLine: { lineStyle: { color: '#e6e9ef' } } },
        { type: 'value', min: 1, max: 365, name: 'Time (day)', nameLocation: 'middle', nameGap: 36, gridIndex: 1,
          axisLabel: { fontSize: 11 }, splitLine: { lineStyle: { color: '#e6e9ef' } } },
        { type: 'value', min: zoom.start, max: zoom.end, splitNumber: 4, gridIndex: 2,
          axisLabel: { fontSize: 9, formatter: fmtTick }, splitLine: { lineStyle: { color: '#eef1f6' } } },
        { type: 'value', min: zoom.start, max: zoom.end, splitNumber: 4, gridIndex: 3,
          axisLabel: { fontSize: 9, formatter: fmtTick }, splitLine: { lineStyle: { color: '#eef1f6' } } }
      ],
      yAxis: [
        { type: 'value', name: DATA.meta.y_label, nameLocation: 'middle', nameGap: 55, min: yMainMin, max: yMainMax, gridIndex: 0,
          axisLabel: { fontSize: 11, formatter: v => v.toLocaleString('en-US') }, splitLine: { lineStyle: { color: '#e6e9ef' } } },
        { type: 'value', name: '', min: yMainMin, max: yMainMax, gridIndex: 1,
          axisLabel: { fontSize: 11, formatter: v => v.toLocaleString('en-US') }, splitLine: { lineStyle: { color: '#e6e9ef' } } },
        { type: 'value', min: zoom.y_inset_a.min, max: zoom.y_inset_a.max, gridIndex: 2,
          axisLabel: { fontSize: 9, formatter: v => v.toLocaleString('en-US') }, splitLine: { lineStyle: { color: '#eef1f6' } } },
        { type: 'value', min: zoom.y_inset_b.min, max: zoom.y_inset_b.max, gridIndex: 3,
          axisLabel: { fontSize: 9, formatter: v => v.toLocaleString('en-US') }, splitLine: { lineStyle: { color: '#eef1f6' } } }
      ],
      tooltip: {
        trigger: 'axis',
        formatter: params => {
          if (!Array.isArray(params)) params = [params];
          const day = params[0]?.axisValue ?? '';
          const lines = [`Day ${day}`];
          params.forEach(p => {
            if (!p.seriesName || p.seriesName.includes('_inset')) return;
            if (p.seriesName.includes('threshold')) return;
            const v = Array.isArray(p.data) ? p.data[1] : p.value;
            const val = (v ?? 0).toLocaleString('en-US', { maximumFractionDigits: 0 });
            lines.push(`${p.marker} ${p.seriesName}: ${val} kWh`);
          });
          return lines.join('<br/>');
        }
      },
      series: series
    };

    chart.setOption(option);

    function updateConnectors() {
      const mainTL = chart.convertToPixel({ xAxisIndex: 0, yAxisIndex: 0 }, [zoom.start, zoom.y_inset_a.max]);
      const mainBR = chart.convertToPixel({ xAxisIndex: 0, yAxisIndex: 0 }, [zoom.end, zoom.y_inset_a.min]);
      const insetTL = chart.convertToPixel({ xAxisIndex: 2, yAxisIndex: 2 }, [zoom.start, zoom.y_inset_a.max]);
      const insetBR = chart.convertToPixel({ xAxisIndex: 2, yAxisIndex: 2 }, [zoom.end, zoom.y_inset_a.min]);

      const mainTLB = chart.convertToPixel({ xAxisIndex: 1, yAxisIndex: 1 }, [zoom.start, zoom.y_inset_b.max]);
      const mainBRB = chart.convertToPixel({ xAxisIndex: 1, yAxisIndex: 1 }, [zoom.end, zoom.y_inset_b.min]);
      const insetTLB = chart.convertToPixel({ xAxisIndex: 3, yAxisIndex: 3 }, [zoom.start, zoom.y_inset_b.max]);
      const insetBRB = chart.convertToPixel({ xAxisIndex: 3, yAxisIndex: 3 }, [zoom.end, zoom.y_inset_b.min]);

      if (!mainTL || !mainBR || !insetTL || !insetBR || !mainTLB || !mainBRB || !insetTLB || !insetBRB) return;

      function rectFrom(p1, p2) {
        const x = Math.min(p1[0], p2[0]);
        const y = Math.min(p1[1], p2[1]);
        const w = Math.abs(p2[0] - p1[0]);
        const h = Math.abs(p2[1] - p1[1]);
        return { x, y, width: w, height: h };
      }

      const rectA = rectFrom(mainTL, mainBR);
      const rectB = rectFrom(mainTLB, mainBRB);
      const rectInsetA = rectFrom(insetTL, insetBR);
      const rectInsetB = rectFrom(insetTLB, insetBRB);

      const graphics = [
        { type: 'rect', shape: rectA, style: { fill: 'rgba(0,0,0,0)', stroke: '#333', lineWidth: 1 } },
        { type: 'rect', shape: rectB, style: { fill: 'rgba(0,0,0,0)', stroke: '#333', lineWidth: 1 } },
        { type: 'rect', shape: rectInsetA, style: { fill: 'rgba(0,0,0,0)', stroke: '#333', lineWidth: 1 } },
        { type: 'rect', shape: rectInsetB, style: { fill: 'rgba(0,0,0,0)', stroke: '#333', lineWidth: 1 } },
        { type: 'line', shape: { x1: rectA.x + rectA.width, y1: rectA.y, x2: rectInsetA.x, y2: rectInsetA.y }, style: { stroke: '#333', lineWidth: 1 } },
        { type: 'line', shape: { x1: rectA.x + rectA.width, y1: rectA.y + rectA.height, x2: rectInsetA.x, y2: rectInsetA.y + rectInsetA.height }, style: { stroke: '#333', lineWidth: 1 } },
        { type: 'line', shape: { x1: rectB.x + rectB.width, y1: rectB.y, x2: rectInsetB.x, y2: rectInsetB.y }, style: { stroke: '#333', lineWidth: 1 } },
        { type: 'line', shape: { x1: rectB.x + rectB.width, y1: rectB.y + rectB.height, x2: rectInsetB.x, y2: rectInsetB.y + rectInsetB.height }, style: { stroke: '#333', lineWidth: 1 } }
      ];

      chart.setOption({ graphic: graphics }, { replaceMerge: ['graphic'] });
    }

    chart.on('finished', updateConnectors);
    window.addEventListener('resize', () => {
      chart.resize();
      updateConnectors();
    });
  </script>
</body>
</html>
"""
    html = template.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    html_path = echarts_dir / "figure1_sensitivity_margin.html"
    html_path.write_text(html, encoding="utf-8")

    bundled = Path(__file__).resolve().parent / "echarts.min.js"
    if bundled.exists():
        target = echarts_dir / "echarts.min.js"
        if not target.exists():
            target.write_bytes(bundled.read_bytes())

    return html_path, json_path


def _plot_robustness(
    out_dir: Path,
    samples_df: pd.DataFrame,
    pass_rate: float,
    q05: float,
) -> Path:
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    data = samples_df["margin_year_kwh"].to_numpy()
    bp = ax.boxplot(data, vert=True, widths=0.5, patch_artist=True)
    for box in bp["boxes"]:
        box.set(facecolor="#cfe3f7", edgecolor="#4c78a8", linewidth=1.2)
    for median in bp["medians"]:
        median.set(color="#333333", linewidth=1.2)

    ax.set_title("Robustness Stress Test under Joint Uncertainty")
    ax.set_ylabel("Annual Net-zero Margin (kWh)")
    ax.set_xticks([1])
    ax.set_xticklabels(["Margin"])

    q1 = float(np.percentile(data, 25))
    q3 = float(np.percentile(data, 75))
    iqr = max(q3 - q1, 1e-6)
    y_min = q1 - 1.5 * iqr
    y_max = q3 + 1.5 * iqr
    ax.set_ylim(y_min, y_max)

    if y_min <= 0.0 <= y_max:
        ax.axhline(0.0, color="#333333", linestyle="--", linewidth=1.0, label="Net-zero threshold")
        ax.legend(frameon=False, fontsize=8, loc="lower left")
    else:
        ax.annotate(
            "Net-zero threshold (0 kWh) is far below the displayed range",
            xy=(0.5, 0.0),
            xycoords="axes fraction",
            xytext=(0.08, -0.08),
            textcoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color="#333333"),
            fontsize=8,
            ha="left",
            va="top",
        )

    ax.text(
        0.60,
        0.95,
        f"pass_rate={pass_rate:.2f}, Q0.05={q05:.0f} kWh",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )

    fig.set_layout_engine("none")
    png_path = out_dir / "plots" / "figure2_robustness_boxplot.png"
    fig.savefig(png_path, dpi=220)
    plt.close(fig)
    return png_path


def main() -> None:
    run_dir = _find_latest_run_dir()
    if run_dir is None:
        raise SystemExit("未找到 sg_pareto 运行目录，请先运行主流程。")

    design, source = _load_fixed_design(run_dir)
    weather_csv = run_dir / "weather" / "weather_singapore_tmy_hourly.csv"
    df_base = load_weather_csv(str(weather_csv), tz=SINGAPORE.tz)

    geom = design.geometry
    irr = compute_facade_irradiance(
        df_base,
        location=SINGAPORE,
        facade_azimuths_deg=geom.facade_azimuths_deg(),
        dN=design.dN_m,
        dS=design.dS_m,
        etaE=design.etaE_rad,
        etaW=design.etaW_rad,
        window_height_m=float(WINDOW_HEIGHT_M),
        optics_borealis=DEFAULT_BOREALIS_OPTICAL,
        enable_passive_shading=True,
        return_debug=False,
    )
    A_win = geom.window_areas_m2()
    g_eq = float(getattr(DEFAULT_OPTICAL, "tau_heat", 0.55))
    I_f = irr["I_f"]
    Phi_s = (
        A_win["N"] * g_eq * I_f["N"]
        + A_win["E"] * g_eq * I_f["E"]
        + A_win["S"] * g_eq * I_f["S"]
        + A_win["W"] * g_eq * I_f["W"]
    )

    typ = _prepare_typical_year(df_base, Phi_s_W=Phi_s)
    ar1_path = _ar1_noise(len(typ["T_out"]), seed=int(AR1_SEED)) if USE_AR1_NOISE else np.zeros(len(typ["T_out"]))
    dt_hours = float(DEFAULT_HVAC.dt_hours)

    # Sensitivity (a): delta T
    delta_levels = [1.5, 1.6, 1.7]
    cop0 = float(DEFAULT_HVAC.COP_cool)
    sensitivity_curves = []
    rows = []

    for dT in delta_levels:
        res = _compute_margin_daily(
            T_out_base=typ["T_out"],
            GHI=typ["GHI"],
            Phi_s=typ["Phi_s"],
            design=design,
            delta_T=dT,
            COP=cop0,
            ar1_noise=ar1_path,
            dt_hours=dt_hours,
        )
        day = np.arange(1, len(res["margin_daily"]) + 1)
        label = f"ΔT={dT:.1f}°C"
        sensitivity_curves.append({"subplot": "a", "day": day, "margin": res["margin_daily"], "label": label})
        for i, m in enumerate(res["margin_daily"], start=1):
            rows.append(
                {
                    "case_id": f"deltaT_{dT:.1f}",
                    "subplot": "a",
                    "param_name": "delta_T_trend_C",
                    "param_value": float(dT),
                    "day": i,
                    "margin_kwh": float(m),
                }
            )

    # Sensitivity (b): COP
    cop_levels = [0.9 * cop0, 1.0 * cop0, 1.1 * cop0]
    base_res = _compute_margin_daily(
        T_out_base=typ["T_out"],
        GHI=typ["GHI"],
        Phi_s=typ["Phi_s"],
        design=design,
        delta_T=float(DELTA_TREND_C),
        COP=cop0,
        ar1_noise=ar1_path,
        dt_hours=dt_hours,
    )
    for cop in cop_levels:
        res = _compute_margin_daily(
            T_out_base=typ["T_out"],
            GHI=typ["GHI"],
            Phi_s=typ["Phi_s"],
            design=design,
            delta_T=float(DELTA_TREND_C),
            COP=cop,
            ar1_noise=ar1_path,
            dt_hours=dt_hours,
        )
        day = np.arange(1, len(res["margin_daily"]) + 1)
        label = f"COP={cop:.2f}"
        sensitivity_curves.append({"subplot": "b", "day": day, "margin": res["margin_daily"], "label": label})
        for i, m in enumerate(res["margin_daily"], start=1):
            rows.append(
                {
                    "case_id": f"cop_{cop:.2f}",
                    "subplot": "b",
                    "param_name": "COP",
                    "param_value": float(cop),
                    "day": i,
                    "margin_kwh": float(m),
                }
            )

    inset_start, inset_end = _select_vulnerable_window(base_res["net_daily"])

    # plot figure1
    all_margins = np.concatenate([c["margin"] for c in sensitivity_curves])
    y_min, y_max = float(np.min(all_margins)), float(np.max(all_margins))
    pad = 0.05 * (y_max - y_min + 1e-6)
    y_limits = (y_min - pad, y_max + pad)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"sensitivity_robustness_{ts}"
    out_root = ANALYSIS_ROOT / run_id / "sensitivity_robustness"
    (out_root / "plots").mkdir(parents=True, exist_ok=True)
    (out_root / "data").mkdir(parents=True, exist_ok=True)

    fig1_path = _plot_sensitivity(out_root, sensitivity_curves, (inset_start, inset_end), y_limits)

    df_ts = pd.DataFrame(rows)
    ts_csv = out_root / "data" / "figure1_margin_timeseries.csv"
    df_ts.to_csv(ts_csv, index=False)

    _write_echarts_figure1(out_root, df_ts, (inset_start, inset_end))

    # Robustness stress test
    N = 400
    rng = np.random.default_rng(42)
    delta_samples = rng.uniform(1.5, 1.7, size=N)
    cop_samples = rng.uniform(0.9 * cop0, 1.1 * cop0, size=N)

    records = []
    for i in range(N):
        res = _compute_margin_daily(
            T_out_base=typ["T_out"],
            GHI=typ["GHI"],
            Phi_s=typ["Phi_s"],
            design=design,
            delta_T=float(delta_samples[i]),
            COP=float(cop_samples[i]),
            ar1_noise=ar1_path,
            dt_hours=dt_hours,
        )
        margin_year = float(res["margin_daily"][-1])
        pass_flag = 1 if margin_year >= 0 else 0
        records.append(
            {
                "sample_id": i + 1,
                "delta_T_trend_C": float(delta_samples[i]),
                "COP_cool": float(cop_samples[i]),
                "margin_year_kwh": margin_year,
                "pass_flag": pass_flag,
            }
        )

    df_samples = pd.DataFrame(records)
    samples_csv = out_root / "data" / "figure2_robustness_samples.csv"
    df_samples.to_csv(samples_csv, index=False)

    pass_rate = float(df_samples["pass_flag"].mean())
    q05 = float(np.quantile(df_samples["margin_year_kwh"], 0.05))
    if pass_rate >= 0.90 and q05 >= 0:
        decision = "PASS"
    elif pass_rate < 0.70:
        decision = "FAIL"
    else:
        decision = "PARTIAL"

    summary = {
        "N": N,
        "pass_rate": pass_rate,
        "q05_kwh": q05,
        "decision": decision,
        "COP0": cop0,
        "deltaT_range": [1.5, 1.7],
        "COP_range": [0.9 * cop0, 1.1 * cop0],
    }
    summary_json = out_root / "data" / "figure2_robustness_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig2_path = _plot_robustness(out_root, df_samples, pass_rate, q05)

    # Chinese logs
    print("已生成敏感性与稳健性图表。")
    print(f"固定方案来源：{source}")
    print(f"Inset 窗口：day {inset_start} - {inset_end}")
    for c in sensitivity_curves:
        if c["subplot"] == "a":
            year_margin = float(c["margin"][-1])
            print(f"ΔT 情景 {c['label']} 年末裕度: {year_margin:.2f} kWh")
    for c in sensitivity_curves:
        if c["subplot"] == "b":
            year_margin = float(c["margin"][-1])
            print(f"COP 情景 {c['label']} 年末裕度: {year_margin:.2f} kWh")
    print(f"图1输出：{fig1_path}")
    print(f"图1数据：{ts_csv}")
    print(f"稳健性 N={N}, ΔT∈[1.5,1.7], COP∈[{0.9*cop0:.2f},{1.1*cop0:.2f}]")
    print(f"pass_rate={pass_rate:.2f}, Q0.05={q05:.2f}, 判定={decision}")
    print(f"图2输出：{fig2_path}")
    print(f"图2数据：{samples_csv}")
    print(f"汇总：{summary_json}")


if __name__ == "__main__":
    main()
