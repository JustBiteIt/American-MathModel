# analysis_runs

## 模块定位
`analysis_runs` 是 **实验输出目录**，保存由 `HVAC_Pareto` 批量脚本生成的结果（不是源代码）。  
该目录会随着实验运行自动增长，可视为“运行快照”。

主要由以下脚本生成：
- `HVAC_Pareto/run_city_benchmark.py`
- `HVAC_Pareto/run_latitude_scan.py`

---

## 目录命名与结构
每次运行都会新建一个以时间戳命名的子目录：
- `city_benchmark_YYYYMMDD_HHMMSS/`
- `latitude_scan_YYYYMMDD_HHMMSS/`

典型结构：
```
analysis_runs/
  city_benchmark_YYYYMMDD_HHMMSS/
    weather/       # PVGIS 下载的 TMY
    results/       # 每个城市/站点的 Pareto 输出
    summary/       # 汇总 CSV + 图表 + ECharts

  latitude_scan_YYYYMMDD_HHMMSS/
    weather/       # 每个纬度点的 TMY
    results/       # 每个纬度点的 Pareto 输出
    summary/       # 汇总 CSV + 扫描曲线 + ECharts
```

---

## 结果文件说明
### results/（单站点/单纬度）
常见输出：
- `results.csv`：所有采样点  
- `pareto.csv`：Pareto 前沿点  
- `pareto.png`：散点图  
- `meta.json`：代表点、气候指标  
- `echarts_case.json/html`：交互图

### summary/（汇总层）
常见输出：
- `city_benchmark_summary.csv`  
- `latitude_scan.csv`  
- `city_pareto_overlay.png`  
- `latitude_scan.png`  
- `echarts_*.json/html`

---

## 使用建议
- 如果只关心最新结果，可只保留最新时间戳目录。  
- 目录内容可安全删除，脚本可重复生成。  
- 若需要可视化，请使用本地静态服务器打开 ECharts HTML（如 VSCode Live Server）。
