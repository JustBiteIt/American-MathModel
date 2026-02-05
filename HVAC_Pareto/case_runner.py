from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from borealis_model.data_io import load_weather_csv

from HVAC_Pareto.evaluate_hvac import evaluate_design
from HVAC_Pareto.pareto import pareto_mask_minimize, select_representative_by_weights

Logger = Callable[[str], None]


def compute_hdd_cdd_from_hourly(df: pd.DataFrame, T_base_C: float = 18.3) -> Tuple[float, float]:
    """HDD/CDD from hourly data using day-mean method."""
    T = df["T_out"].astype(float)
    day_mean = T.resample("D").mean()
    HDD = float(np.sum(np.maximum(T_base_C - day_mean, 0.0)))
    CDD = float(np.sum(np.maximum(day_mean - T_base_C, 0.0)))
    return HDD, CDD


def climate_xi(HDD: float, CDD: float) -> float:
    denom = float(HDD + CDD)
    if denom <= 0:
        return 0.0
    return float((CDD - HDD) / denom)


def sample_designs(bounds, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample x=(dN,dS,etaE,etaW) uniformly within bounds."""
    dN = rng.uniform(bounds.dN_bounds_m[0], bounds.dN_bounds_m[1], size=n)
    dS = rng.uniform(bounds.dS_bounds_m[0], bounds.dS_bounds_m[1], size=n)
    eE = rng.uniform(bounds.etaE_bounds_rad[0], bounds.etaE_bounds_rad[1], size=n)
    eW = rng.uniform(bounds.etaW_bounds_rad[0], bounds.etaW_bounds_rad[1], size=n)
    return np.stack([dN, dS, eE, eW], axis=1)


def _write_echarts_case_json(
    out_dir: Path,
    *,
    case_info: Dict[str, Any],
    climate: Dict[str, Any],
    representative: Dict[str, Any],
    points_all: List[Dict[str, Any]],
    points_pareto: List[Dict[str, Any]],
) -> Path:
    payload = {
        "case": case_info,
        "climate": climate,
        "representative": representative,
        "pointsAll": points_all,
        "pointsPareto": points_pareto,
    }
    path = out_dir / "echarts_case.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_echarts_case_html(out_dir: Path) -> Path:
    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>ECharts Case</title>
  <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
  <style>
    body { font-family: Arial, sans-serif; margin: 20px; }
    .header { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
    #chart { width: 100%; height: 600px; }
    .note { color: #666; font-size: 12px; }
    .btn { padding: 6px 12px; border: 1px solid #bbb; background: #f7f7f7; cursor: pointer; }
  </style>
</head>
<body>
  <div class="header">
    <h2>Pareto Front (Heating vs Cooling Load)</h2>
    <button id="btn-download" class="btn">Download PNG</button>
  </div>
  <div id="chart"></div>
  <p class="note">If the chart doesn't load, open this file via a local web server (e.g., VSCode Live Server).</p>
  <script>
    fetch("./echarts_case.json")
      .then(r => r.json())
      .then(data => {
        const all = data.pointsAll.map(p => [p.E_cool_kWh, p.E_heat_kWh]);
        const pareto = data.pointsPareto.map(p => [p.E_cool_kWh, p.E_heat_kWh]);
        const rep = [data.representative.E_cool_kWh, data.representative.E_heat_kWh];

        const chart = echarts.init(document.getElementById("chart"));
        chart.setOption({
          grid: { left: 70, right: 70, top: 60, bottom: 70, containLabel: true },
          tooltip: {
            trigger: "item"
          },
          legend: { data: ["All", "Pareto", "Representative"], top: 10 },
          xAxis: { name: "Cooling load (kWh_th)", nameGap: 30 },
          yAxis: { name: "Heating load (kWh_th)", nameGap: 35 },
          series: [
            { name: "All", type: "scatter", data: all, symbolSize: 4, itemStyle: { opacity: 0.4 } },
            { name: "Pareto", type: "scatter", data: pareto, symbolSize: 6 },
            { name: "Representative", type: "scatter", data: [rep], symbolSize: 14, symbol: "diamond" }
          ]
        });

        document.getElementById("btn-download").addEventListener("click", () => {
          const url = chart.getDataURL({ type: "png", pixelRatio: 2, backgroundColor: "#fff" });
          const a = document.createElement("a");
          a.href = url;
          a.download = "pareto_case.png";
          a.click();
        });
      });
  </script>
</body>
</html>
"""
    path = out_dir / "echarts_case.html"
    path.write_text(html, encoding="utf-8")
    return path


def run_case_from_weather_csv(
    *,
    weather_csv: str,
    location,
    config,
    n_samples: int,
    seed: int,
    out_dir: str | Path,
    case_name: str,
    logger: Logger | None = None,
) -> Dict[str, Any]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if logger:
        logger(f"开始处理：{case_name}")
        logger(f"天气文件：{weather_csv}")

    df = load_weather_csv(weather_csv, tz=location.tz)
    if logger:
        logger(f"天气行数：{len(df)}  时区：{location.tz}")

    HDD, CDD = compute_hdd_cdd_from_hourly(df, T_base_C=18.3)
    xi = climate_xi(HDD, CDD)
    w_cool = (1.0 + xi) / 2.0
    w_heat = (1.0 - xi) / 2.0
    if logger:
        logger(f"HDD={HDD:.0f}  CDD={CDD:.0f}  xi={xi:.3f}")

    rng = np.random.default_rng(seed)
    X = sample_designs(config.bounds, n_samples, rng)
    if logger:
        logger(f"采样数：{n_samples}  随机种子：{seed}")

    rows: List[Dict[str, Any]] = []
    for x in X:
        r = evaluate_design(
            df,
            location=location,
            building=config.building,
            optical_base=config.optical,
            optics_borealis=config.optics_borealis,
            hvac_2r2c=config.hvac_2r2c,
            x=(float(x[0]), float(x[1]), float(x[2]), float(x[3])),
            return_series=False,
        )
        r.update({
            "dN_m": float(x[0]),
            "dS_m": float(x[1]),
            "etaE_rad": float(x[2]),
            "etaW_rad": float(x[3]),
        })
        rows.append(r)

    res = pd.DataFrame(rows)
    res_path = out_path / "results.csv"
    res.to_csv(res_path, index=False)

    pts = res[["E_cool_kWh", "E_heat_kWh"]].to_numpy(float)
    mask = pareto_mask_minimize(pts)
    pareto_df = res.loc[mask].copy()
    pareto_path = out_path / "pareto.csv"
    pareto_df.to_csv(pareto_path, index=False)

    rep_idx_full = select_representative_by_weights(
        res["E_cool_kWh"].to_numpy(float),
        res["E_heat_kWh"].to_numpy(float),
        mask,
        w_cool=w_cool,
        w_heat=w_heat,
    )
    pareto_indices = np.where(mask)[0]
    rep_idx_pareto = int(np.where(pareto_indices == int(rep_idx_full))[0][0])
    rep_row = res.iloc[int(rep_idx_full)]

    # Plot
    fig = plt.figure()
    plt.scatter(res["E_cool_kWh"], res["E_heat_kWh"], alpha=0.25, s=12, label="Samples")
    plt.scatter(pareto_df["E_cool_kWh"], pareto_df["E_heat_kWh"], s=18, label="Pareto")
    plt.scatter([rep_row["E_cool_kWh"]], [rep_row["E_heat_kWh"]], marker="*", s=160, label="Representative")
    plt.xlabel("Cooling load (kWh_th)")
    plt.ylabel("Heating load (kWh_th)")
    plt.title(f"Pareto Front - {case_name}")
    plt.legend()
    plt.tight_layout()
    fig_path = out_path / "pareto.png"
    plt.savefig(fig_path, dpi=200)
    plt.close(fig)

    meta = {
        "case": {
            "name": case_name,
            "latitude": float(location.latitude),
            "longitude": float(location.longitude),
            "tz": str(location.tz),
        },
        "weather_csv": str(weather_csv),
        "hvac": asdict(config.hvac_2r2c),
        "HDD": HDD,
        "CDD": CDD,
        "xi": xi,
        "w_cool": w_cool,
        "w_heat": w_heat,
        "pareto_size": int(len(pareto_df)),
        "representative": {
            "row_index_in_results": int(rep_idx_full),
            "row_index_in_pareto": int(rep_idx_pareto),
            "E_cool_kWh": float(rep_row["E_cool_kWh"]),
            "E_heat_kWh": float(rep_row["E_heat_kWh"]),
            "dN_m": float(rep_row["dN_m"]),
            "dS_m": float(rep_row["dS_m"]),
            "etaE_rad": float(rep_row["etaE_rad"]),
            "etaW_rad": float(rep_row["etaW_rad"]),
        },
    }
    meta_path = out_path / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # ECharts outputs
    points_all = res[["E_cool_kWh", "E_heat_kWh", "dN_m", "dS_m", "etaE_rad", "etaW_rad"]].to_dict("records")
    points_pareto = pareto_df[["E_cool_kWh", "E_heat_kWh", "dN_m", "dS_m", "etaE_rad", "etaW_rad"]].to_dict("records")
    case_info = {
        "name": case_name,
        "latitude": float(location.latitude),
        "longitude": float(location.longitude),
        "tz": str(location.tz),
        "weatherCsv": str(weather_csv),
    }
    climate = {
        "HDD": HDD,
        "CDD": CDD,
        "xi": xi,
        "wHeat": w_heat,
        "wCool": w_cool,
    }
    representative = {
        "dN_m": float(rep_row["dN_m"]),
        "dS_m": float(rep_row["dS_m"]),
        "etaE_rad": float(rep_row["etaE_rad"]),
        "etaW_rad": float(rep_row["etaW_rad"]),
        "E_heat_kWh": float(rep_row["E_heat_kWh"]),
        "E_cool_kWh": float(rep_row["E_cool_kWh"]),
        "paretoIndex": int(rep_idx_pareto),
        "resultsIndex": int(rep_idx_full),
    }
    echarts_json = _write_echarts_case_json(
        out_path,
        case_info=case_info,
        climate=climate,
        representative=representative,
        points_all=points_all,
        points_pareto=points_pareto,
    )
    echarts_html = _write_echarts_case_html(out_path)

    if logger:
        logger(f"Pareto 点数：{len(pareto_df)}")
        logger(
            "代表方案："
            f"dN={rep_row['dN_m']:.3f}, dS={rep_row['dS_m']:.3f}, "
            f"etaE={rep_row['etaE_rad']:.3f}, etaW={rep_row['etaW_rad']:.3f}"
        )
        logger(
            f"代表负荷：E_heat={rep_row['E_heat_kWh']:.2f} kWh_th, "
            f"E_cool={rep_row['E_cool_kWh']:.2f} kWh_th"
        )
        logger(f"输出文件：results.csv, pareto.csv, pareto.png, meta.json")
        logger(f"ECharts：{echarts_json.name}, {echarts_html.name}")

    return {
        "out_dir": str(out_path),
        "results_csv": str(res_path),
        "pareto_csv": str(pareto_path),
        "pareto_png": str(fig_path),
        "meta_json": str(meta_path),
        "echarts_json": str(echarts_json),
        "echarts_html": str(echarts_html),
        "HDD": HDD,
        "CDD": CDD,
        "xi": xi,
        "w_cool": w_cool,
        "w_heat": w_heat,
        "pareto_size": int(len(pareto_df)),
        "representative": meta["representative"],
    }
