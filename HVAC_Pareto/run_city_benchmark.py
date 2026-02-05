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


CITY_LIST = [
    ("London", 51.5074, -0.1278, "Europe/London"),
    ("Calgary", 51.0447, -114.0719, "America/Edmonton"),
    ("Rome", 41.9028, 12.4964, "Europe/Rome"),
    ("Beijing", 39.9042, 116.4074, "Asia/Shanghai"),
    ("San Francisco", 37.7749, -122.4194, "America/Los_Angeles"),
    ("Seville", 37.3891, -5.9845, "Europe/Madrid"),
]


def _slug(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def _write_echarts_city_json(out_dir: Path, payload: dict) -> Path:
    path = out_dir / "echarts_city_benchmark.json"
    path.write_text(__import__("json").dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_echarts_city_html(out_dir: Path) -> Path:
    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>City Benchmark</title>
  <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
  <style>
    body { font-family: Arial, sans-serif; margin: 20px; }
    .header { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-top: 6px; }
    #pareto { width: 100%; height: 520px; }
    #climate { width: 100%; height: 380px; margin-top: 16px; }
    .note { color: #666; font-size: 12px; }
    .btn { padding: 6px 12px; border: 1px solid #bbb; background: #f7f7f7; cursor: pointer; }
  </style>
</head>
<body>
  <div class="header">
    <h2>City Benchmark: Pareto Fronts</h2>
    <button id="btn-pareto" class="btn">Download Pareto PNG</button>
  </div>
  <div id="pareto"></div>
  <div class="header">
    <h2>Climate Index (xi, dimensionless)</h2>
    <button id="btn-climate" class="btn">Download Climate PNG</button>
  </div>
  <div id="climate"></div>
  <p class="note">If the chart doesn't load, open this file via a local web server (e.g., VSCode Live Server).</p>
  <script>
    fetch("./echarts_city_benchmark.json")
      .then(r => r.json())
      .then(data => {
        const cities = data.cities.map(c => c.name);
        const paretoSeries = cities.map(name => ({
          name,
          type: "scatter",
          data: (data.paretoFronts[name] || []).map(p => [p.E_cool_kWh, p.E_heat_kWh]),
          symbolSize: 6
        }));

        const repSeries = {
          name: "Representative",
          type: "scatter",
          data: data.cities.map(c => [c.representative.E_cool_kWh, c.representative.E_heat_kWh]),
          symbol: "diamond",
          symbolSize: 12
        };

        const paretoChart = echarts.init(document.getElementById("pareto"));
        paretoChart.setOption({
          grid: { left: 70, right: 70, top: 60, bottom: 70, containLabel: true },
          legend: { data: [...cities, "Representative"], top: 10 },
          tooltip: { trigger: "item" },
          xAxis: { name: "Cooling load (kWh_th)", nameGap: 30 },
          yAxis: { name: "Heating load (kWh_th)", nameGap: 35 },
          series: [...paretoSeries, repSeries]
        });

        const xiVals = data.cities.map(c => c.climate.xi);
        const climateChart = echarts.init(document.getElementById("climate"));
        climateChart.setOption({
          grid: { left: 70, right: 40, top: 50, bottom: 60, containLabel: true },
          tooltip: { trigger: "axis" },
          xAxis: { type: "category", data: cities, name: "City", nameGap: 30 },
          yAxis: { type: "value", name: "xi (dimensionless)", nameGap: 35 },
          series: [{ type: "bar", data: xiVals, name: "xi" }]
        });

        document.getElementById("btn-pareto").addEventListener("click", () => {
          const url = paretoChart.getDataURL({ type: "png", pixelRatio: 2, backgroundColor: "#fff" });
          const a = document.createElement("a");
          a.href = url;
          a.download = "city_pareto.png";
          a.click();
        });
        document.getElementById("btn-climate").addEventListener("click", () => {
          const url = climateChart.getDataURL({ type: "png", pixelRatio: 2, backgroundColor: "#fff" });
          const a = document.createElement("a");
          a.href = url;
          a.download = "city_climate.png";
          a.click();
        });
      });
  </script>
</body>
</html>
"""
    path = out_dir / "echarts_city_benchmark.html"
    path.write_text(html, encoding="utf-8")
    return path


def main() -> None:
    print("========== 城市对比实验：开始 ==========")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_root = PROJECT_ROOT / "analysis_runs" / f"city_benchmark_{ts}"
    weather_dir = exp_root / "weather"
    results_dir = exp_root / "results"
    summary_dir = exp_root / "summary"
    for d in [weather_dir, results_dir, summary_dir]:
        d.mkdir(parents=True, exist_ok=True)

    cfg = ModelConfig()
    summaries = []
    pareto_fronts = {}
    failures = []

    total = len(CITY_LIST)
    for idx, (name, lat, lon, tz) in enumerate(CITY_LIST, start=1):
        print(f"---------- [{idx}/{total}] {name} ----------")
        print(f"地点信息：纬度={lat:.4f}，经度={lon:.4f}，时区={tz}")
        case_slug = _slug(name)
        weather_path = weather_dir / f"{case_slug}_tmy_hourly.csv"

        try:
            if weather_path.exists():
                print("PVGIS：命中缓存，直接使用已有天气文件。")
                row_count = max(0, sum(1 for _ in weather_path.open("r", encoding="utf-8")) - 1)
                print(f"天气文件：{weather_path}（{row_count} 行）")
            else:
                print("PVGIS：未发现缓存，开始下载典型年数据...")
                info = fetch_pvgis_tmy_to_csv(lat=lat, lon=lon, out_csv=weather_path, overwrite=False)
                print(f"PVGIS：下载完成，已生成天气文件：{info['path']}（{info['rows']} 行）")

            location = make_location(cfg.location, LocationOverrides(latitude=lat, longitude=lon, tz=tz))
            case_out = results_dir / case_slug
            def log_case(msg: str) -> None:
                print(f"  {msg}")

            result = run_case_from_weather_csv(
                weather_csv=str(weather_path),
                location=location,
                config=cfg,
                n_samples=2000,
                seed=42,
                out_dir=case_out,
                case_name=name,
                logger=log_case,
            )

            summaries.append({
                "city": name,
                "latitude": lat,
                "longitude": lon,
                "tz": tz,
                "HDD": result["HDD"],
                "CDD": result["CDD"],
                "xi": result["xi"],
                "w_heat": result["w_heat"],
                "w_cool": result["w_cool"],
                "E_heat_kWh": result["representative"]["E_heat_kWh"],
                "E_cool_kWh": result["representative"]["E_cool_kWh"],
                "dN_m": result["representative"]["dN_m"],
                "dS_m": result["representative"]["dS_m"],
                "etaE_rad": result["representative"]["etaE_rad"],
                "etaW_rad": result["representative"]["etaW_rad"],
                "pareto_size": result["pareto_size"],
                "weather_csv": str(weather_path),
                "results_dir": str(case_out),
            })

            pareto_df = pd.read_csv(Path(result["pareto_csv"]))
            pareto_fronts[name] = pareto_df[["E_cool_kWh", "E_heat_kWh", "dN_m", "dS_m", "etaE_rad", "etaW_rad"]].to_dict("records")

        except Exception as exc:
            failures.append({"city": name, "error": str(exc)})
            print(f"PVGIS/模型运行失败：{exc}")
            print("该城市已跳过，继续后续城市。")
            continue

    # summary csv
    summary_csv = summary_dir / "city_benchmark_summary.csv"
    pd.DataFrame(summaries).to_csv(summary_csv, index=False)

    # overlay plot
    overlay_png = summary_dir / "city_pareto_overlay.png"
    if pareto_fronts:
        plt.figure()
        for name, pts in pareto_fronts.items():
            x = [p["E_cool_kWh"] for p in pts]
            y = [p["E_heat_kWh"] for p in pts]
            plt.scatter(x, y, s=10, label=name)
        plt.xlabel("Cooling load (kWh_th)")
        plt.ylabel("Heating load (kWh_th)")
        plt.title("Pareto Fronts (City Benchmark)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(overlay_png, dpi=200)
        plt.close()

    # ECharts outputs
    city_payload = {
        "experiment": {
            "type": "city_benchmark",
            "createdAt": ts,
            "rootDir": str(exp_root),
        },
        "cities": [
            {
                "name": s["city"],
                "latitude": s["latitude"],
                "longitude": s["longitude"],
                "tz": s["tz"],
                "weatherCsv": s["weather_csv"],
                "climate": {
                    "HDD": s["HDD"],
                    "CDD": s["CDD"],
                    "xi": s["xi"],
                    "wHeat": s["w_heat"],
                    "wCool": s["w_cool"],
                },
                "representative": {
                    "dN_m": s["dN_m"],
                    "dS_m": s["dS_m"],
                    "etaE_rad": s["etaE_rad"],
                    "etaW_rad": s["etaW_rad"],
                    "E_heat_kWh": s["E_heat_kWh"],
                    "E_cool_kWh": s["E_cool_kWh"],
                },
                "paretoSize": s["pareto_size"],
            }
            for s in summaries
        ],
        "paretoFronts": pareto_fronts,
    }
    echarts_json = _write_echarts_city_json(summary_dir, city_payload)
    echarts_html = _write_echarts_city_html(summary_dir)

    print("========== 城市对比实验：结束 ==========")
    print(f"实验目录：{exp_root}")
    print(f"成功城市数：{len(summaries)}  失败城市数：{len(failures)}")
    print(f"汇总文件：{summary_csv}")
    if overlay_png.exists():
        print(f"叠加图：{overlay_png}")
    print(f"ECharts JSON：{echarts_json}")
    print(f"ECharts HTML：{echarts_html}")
    print("若 HTML 无法加载数据，请用本地静态服务器打开（如 VSCode Live Server）。")

    if failures:
        for f in failures:
            print(f"失败城市：{f['city']}  错误：{f['error']}")


if __name__ == "__main__":
    main()
