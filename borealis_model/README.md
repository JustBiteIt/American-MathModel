# borealis_model

## 模块定位
`borealis_model` 是整个项目的物理核心：用 **TMY 天气数据 + 太阳几何 + 简化遮阳模型 + 2R2C 热模型** 来评估遮阳设计的能耗与过热风险。它主要服务于 “Borealis” 场景（寒冷气候），核心输出包括：

- 供暖能耗 `Eh_kWh`
- 过热强度 `OH_degC_h`
- 过热时长 `Hoh_h`

同时，它提供了一套可重复的 **随机搜索/方案输出** 流程，用于生成论文/报告需要的汇总表、时序 CSV、ECharts 图表数据等。

---

## 计算流程（从天气到指标）
1. **读天气数据**  
   `data_io.load_weather_csv()` 读取 `datetime, DNI, DHI, GHI, T_out` 五列，转为带时区的 `DatetimeIndex`。

2. **太阳几何与立面入射**  
   `solar.solar_position()` 计算太阳高度角/方位角；  
   `solar.facade_incidence_cos()` 与 `facade_direct_irradiance()` 得到各立面直射辐照度。

3. **遮阳透射（设计变量 x）**  
   `shading.u_overhang()`：北/南向挑檐透射系数  
   `shading.u_louver()`：东/西向百叶透射系数  
   设计变量 `x = (dN, dS, etaE, etaW)`。

4. **地面反照率（雪/地表）**  
   `albedo.rho_ground_timeseries()` 根据室外温度生成 `rho_g(t)`，并平滑过渡。

5. **立面总辐照度**  
   组合直射 + 天空散射 + 地面反射（BorealisOptical 的 `k_other, V_sky, V_gr`）。

6. **室内太阳得热**  
   `BuildingConfig.window_areas()` × `tau_heat` × 立面辐照度 → `Phi_s(t)`。

7. **2R2C 动态热模型**  
   `thermal_2r2c.simulate_2r2c_with_heating()`  
   仅冬季开启供暖，夏季不主动制冷。

8. **指标计算**  
   `metrics.heating_energy_kwh()`  
   `metrics.overheat_metrics()`  
   输出 `Eh_kWh`, `OH_degC_h`, `Hoh_h`。

---

## 核心参数与数据结构
- **LocationConfig**：纬度、经度、时区  
- **BuildingConfig**：几何尺寸、层数、窗墙比（WWR）、立面方位  
  - 支持 `wwr_south/wwr_other` 旧式模式  
  - 也支持 `wwr_by_facade` 精细立面 WWR
- **OpticalConfig**：`tau_heat` (等效透射)、`tau_vis` 等  
- **BorealisOptical**：雪地反照率、`k_other`、`V_sky/V_gr`  
- **Borealis2R2C**：热容/热阻 + 供暖上下限  
- **DecisionBounds**：设计变量的采样范围  

设计变量 `x = (dN, dS, etaE, etaW)`：
- `dN, dS`：北/南向挑檐深度（m）
- `etaE, etaW`：东/西向百叶参数（rad）

---

## 主要脚本与用途
### 1) 主计算
`main_borealis.py`
- 读取天气 → 运行基准方案 → 随机搜索可行遮阳方案  
- 约束：`OH/Hoh` 不超过基准的一定比例  
- 输出：`D:\ICM_CODE\borealis_results_summary.csv`

### 2) 核心评价函数
`evaluate_borealis.py`
- 物理主流程封装为 `evaluate_borealis()`  
- `return_series=True` 时输出完整时序（Ti/Tm/Phi_s/Phi_h 等）

### 3) PRO2 产出链
- `pro2_make_outputs.py`  
  生成 ECharts payload、序列 CSV、PNG 预览图、平铺 summary CSV。
- `pro2_generate_timeseries.py`  
  根据 summary 生成用于可视化的时序 CSV。
- `pro2_postprocess.py`  
  后处理时序 CSV，生成图表用 JSON（KDE、月度能耗、最热 14 天等）。
- `pro2_run_6cases.py`  
  固定 6 个 case 的快速对照输出（`results.json`）。

### 4) 工具脚本
`tools/make_weather_helsinki_tmy.py`  
从 PVGIS 获取赫尔辛基 TMY，标准化为 `datetime,DNI,DHI,GHI,T_out`。

---

## 输入与输出
### 输入
默认读取项目根目录：
- `weather_helsinki_tmy_hourly.csv`（包含 `datetime,DNI,DHI,GHI,T_out`）

### 输出
常见输出包括：
- `borealis_results_summary.csv`（主汇总表）
- `D:\ICM_RESULT\pro2\` 下的：
  - `series_csv/`：完整时序
  - `echarts_payload/`：ECharts JSON
  - `fig_png/`：快速 PNG
  - `pro2_summary_flat.csv`：平铺汇总

---

## 常用运行方式
```powershell
# 基础评估 + 随机搜索
python borealis_model/main_borealis.py

# PRO2 论文级输出
python borealis_model/pro2_make_outputs.py

# 生成时序 CSV（可指定 out_dir）
python borealis_model/pro2_generate_timeseries.py --out_dir D:\ICM_RESULT\pro2

# 后处理时序为 ECharts JSON
python borealis_model/pro2_postprocess.py --in_dir D:\ICM_RESULT\pro2 --out_json D:\ICM_RESULT\pro2\pro2_charts.json

# 6-case 快速输出
python borealis_model/pro2_run_6cases.py
```

---

## 典型扩展点
- **换城市/气候**：改 `LocationConfig` + 替换天气 CSV  
- **换 WWR**：改 `BuildingConfig` 的 WWR 或 `wwr_by_facade`  
- **换热参数**：`Borealis2R2C` 与 `MaterialConfig`  
- **优化策略**：替换 `main_borealis.py` 中的随机搜索  

---

## 注意事项
- 所有时间索引必须是 **tz-aware**（否则 `solar_position` 会报错）。  
- `return_series=False` 可显著加速优化循环。  
- 过热指标只在 `summer_months` 上统计。  
- 模型不包含显式制冷，仅供暖 + 过热评价。  
