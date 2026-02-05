from __future__ import annotations

# Allow running by clicking VSCode ▶ on this file:
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import matplotlib.pyplot as plt

from HVAC_Pareto.config import ModelConfig, LocationOverrides, make_location
from HVAC_Pareto.case_runner import run_case_from_weather_csv
from HVAC_Pareto.pvgis_fetch import fetch_pvgis_tmy_to_csv


# Fixed longitude for scan (can be edited)
FIXED_LON = -0.1278  # London longitude by default
LAT_STEP = 6


def _lat_list() -> list[int]:
    return list(range(90, -1, -LAT_STEP))


def _write_echarts_lat_json(out_dir: Path, payload: dict) -> Path:
    path = out_dir / "echarts_latitude_scan.json"
    path.write_text(__import__("json").dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_echarts_lat_html(out_dir: Path) -> Path:
    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Latitude Scan</title>
  <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
  <style>
    body { font-family: Arial, sans-serif; margin: 20px; }
    .header { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
    #chart { width: 100%; height: 620px; }
    .note { color: #666; font-size: 12px; }
    .btn { padding: 6px 12px; border: 1px solid #bbb; background: #f7f7f7; cursor: pointer; }
  </style>
</head>
<body>
  <div class="header">
    <h2>Latitude Scan (Representative Variables)</h2>
    <button id="btn-download" class="btn">Download PNG</button>
  </div>
  <div id="chart"></div>
  <p class="note">If the chart doesn't load, open this file via a local web server (e.g., VSCode Live Server).</p>
  <script>
    fetch("./echarts_latitude_scan.json")
      .then(r => r.json())
      .then(data => {
        const lat = data.series.latitudes;
        const chart = echarts.init(document.getElementById("chart"));
        chart.setOption({
          grid: { left: 70, right: 120, top: 60, bottom: 70, containLabel: true },
          tooltip: { trigger: "axis" },
          legend: { data: ["dN_m", "dS_m", "etaE_rad", "etaW_rad"], top: 10 },
          xAxis: { type: "category", data: lat, name: "Latitude (deg)", nameGap: 30 },
          yAxis: [
            { type: "value", name: "Overhang depth (m)", nameGap: 35 },
            { type: "value", name: "Louver parameter (rad)", nameGap: 45 }
          ],
          series: [
            { name: "dN_m", type: "line", yAxisIndex: 0, data: data.series.dN_m, symbol: "circle", symbolSize: 8, lineStyle: { width: 3 } },
            { name: "dS_m", type: "line", yAxisIndex: 0, data: data.series.dS_m, symbol: "circle", symbolSize: 8, lineStyle: { width: 3 } },
            { name: "etaE_rad", type: "line", yAxisIndex: 1, data: data.series.etaE_rad, symbol: "triangle", symbolSize: 9, lineStyle: { width: 3 } },
            { name: "etaW_rad", type: "line", yAxisIndex: 1, data: data.series.etaW_rad, symbol: "triangle", symbolSize: 9, lineStyle: { width: 3 } }
          ]
        });

        document.getElementById("btn-download").addEventListener("click", () => {
          const url = chart.getDataURL({ type: "png", pixelRatio: 2, backgroundColor: "#fff" });
          const a = document.createElement("a");
          a.href = url;
          a.download = "latitude_scan.png";
          a.click();
        });
      });
  </script>
</body>
</html>
"""
    path = out_dir / "echarts_latitude_scan.html"
    path.write_text(html, encoding="utf-8")
    return path


def main() -> None:
    print("========== 纬度扫描实验：开始 ==========")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_root = PROJECT_ROOT / "analysis_runs" / f"latitude_scan_{ts}"
    weather_dir = exp_root / "weather"
    results_dir = exp_root / "results"
    summary_dir = exp_root / "summary"
    for d in [weather_dir, results_dir, summary_dir]:
        d.mkdir(parents=True, exist_ok=True)

    cfg = ModelConfig()
    summaries = []
    failures = []

    latitudes = _lat_list()
    total = len(latitudes)
    for idx, lat in enumerate(latitudes, start=1):
        print(f"---------- [{idx}/{total}] lat={lat} ----------")
        print(f"地点信息：纬度={lat:.1f}，经度={FIXED_LON:.4f}")
        fname = f"lon_{FIXED_LON:.2f}_lat_{lat:.0f}_tmy_hourly.csv".replace("-", "m").replace(".", "p")
        weather_path = weather_dir / fname

        try:
            if weather_path.exists():
                print("PVGIS：命中缓存，直接使用已有天气文件。")
                row_count = max(0, sum(1 for _ in weather_path.open("r", encoding="utf-8")) - 1)
                print(f"天气文件：{weather_path}（{row_count} 行）")
            else:
                print("PVGIS：未发现缓存，开始下载典型年数据...")
                info = fetch_pvgis_tmy_to_csv(lat=lat, lon=FIXED_LON, out_csv=weather_path, overwrite=False)
                print(f"PVGIS：下载完成，已生成天气文件：{info['path']}（{info['rows']} 行）")

            location = make_location(cfg.location, LocationOverrides(latitude=lat, longitude=FIXED_LON, tz="UTC"))
            case_out = results_dir / f"lat_{lat:02d}"
            def log_case(msg: str) -> None:
                print(f"  {msg}")

            result = run_case_from_weather_csv(
                weather_csv=str(weather_path),
                location=location,
                config=cfg,
                n_samples=2000,
                seed=42,
                out_dir=case_out,
                case_name=f"lat_{lat:02d}",
                logger=log_case,
            )

            summaries.append({
                "latitude": lat,
                "longitude": FIXED_LON,
                "HDD": result["HDD"],
                "CDD": result["CDD"],
                "xi": result["xi"],
                "dN_m": result["representative"]["dN_m"],
                "dS_m": result["representative"]["dS_m"],
                "etaE_rad": result["representative"]["etaE_rad"],
                "etaW_rad": result["representative"]["etaW_rad"],
                "E_heat_kWh": result["representative"]["E_heat_kWh"],
                "E_cool_kWh": result["representative"]["E_cool_kWh"],
                "pareto_size": result["pareto_size"],
                "weather_csv": str(weather_path),
                "results_dir": str(case_out),
            })

        except Exception as exc:
            failures.append({"latitude": lat, "error": str(exc)})
            print(f"PVGIS/模型运行失败：{exc}")
            print("该纬度点已跳过，继续后续纬度。")
            continue

    summary_csv = summary_dir / "latitude_scan.csv"
    df_sum = pd.DataFrame(summaries)
    df_sum.to_csv(summary_csv, index=False)

    # Latitude scan plot (double y-axis)
    plot_path = summary_dir / "latitude_scan.png"
    if not df_sum.empty:
        lat_vals = df_sum["latitude"].to_numpy()
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        ax1.plot(lat_vals, df_sum["dN_m"], "o-", label="dN_m")
        ax1.plot(lat_vals, df_sum["dS_m"], "o-", label="dS_m")
        ax2.plot(lat_vals, df_sum["etaE_rad"], "s--", label="etaE_rad")
        ax2.plot(lat_vals, df_sum["etaW_rad"], "s--", label="etaW_rad")

        ax1.set_xlabel("Latitude (deg)")
        ax1.set_ylabel("Overhang depth (m)")
        ax2.set_ylabel("Louver parameter (rad)")
        plt.title(f"Latitude Scan (lon={FIXED_LON:.2f})")
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="best")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close(fig)

    # ECharts outputs
    series = {
        "latitudes": df_sum["latitude"].tolist(),
        "dN_m": df_sum["dN_m"].tolist(),
        "dS_m": df_sum["dS_m"].tolist(),
        "etaE_rad": df_sum["etaE_rad"].tolist(),
        "etaW_rad": df_sum["etaW_rad"].tolist(),
    }
    payload = {
        "experiment": {
            "type": "latitude_scan",
            "fixedLongitude": FIXED_LON,
            "latStepDeg": LAT_STEP,
            "createdAt": ts,
            "rootDir": str(exp_root),
        },
        "scan": [
            {
                "latitude": r["latitude"],
                "longitude": r["longitude"],
                "weatherCsv": r["weather_csv"],
                "climate": {
                    "HDD": r["HDD"],
                    "CDD": r["CDD"],
                    "xi": r["xi"],
                },
                "representative": {
                    "dN_m": r["dN_m"],
                    "dS_m": r["dS_m"],
                    "etaE_rad": r["etaE_rad"],
                    "etaW_rad": r["etaW_rad"],
                    "E_heat_kWh": r["E_heat_kWh"],
                    "E_cool_kWh": r["E_cool_kWh"],
                },
                "paretoSize": r["pareto_size"],
            }
            for r in summaries
        ],
        "series": series,
    }
    echarts_json = _write_echarts_lat_json(summary_dir, payload)
    echarts_html = _write_echarts_lat_html(summary_dir)

    print("========== 纬度扫描实验：结束 ==========")
    print(f"实验目录：{exp_root}")
    print(f"成功纬度点数：{len(summaries)}  失败点数：{len(failures)}")
    print(f"汇总文件：{summary_csv}")
    if plot_path.exists():
        print(f"扫描图：{plot_path}")
    print(f"ECharts JSON：{echarts_json}")
    print(f"ECharts HTML：{echarts_html}")
    print("若 HTML 无法加载数据，请用本地静态服务器打开（如 VSCode Live Server）。")

    if failures:
        for f in failures:
            print(f"失败纬度：{f['latitude']}  错误：{f['error']}")


if __name__ == "__main__":
    main()
