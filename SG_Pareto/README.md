# SG_Pareto

## 模块定位
`SG_Pareto` 是针对 **新加坡气候** 的多目标设计评估模块，核心关注：

- **制冷电耗**（kWh_el）
- **净零裕度 MNZ**（PV 供能 - 2040 年制冷电耗）

模型将 **遮阳设计 + 材料热参数 + 未来升温情景** 统一到 Pareto 框架内，用于评估不同方案在低能耗与“净零潜力”之间的权衡。

---

## 核心流程
1. **PVGIS 获取 TMY 天气**（新加坡）  
2. **生成 2040 情景**：在 `T_out` 上叠加 `+1.5°C`（`DELTA_TREND_C`）  
3. **遮阳计算**：复用 `borealis_model` 的太阳/遮阳/反照率  
4. **2R2C + deadband HVAC**：仅关注制冷负荷  
5. **电耗换算**：`E_cool_el = E_cool_th / COP_cool`  
6. **PV 供电**：`E_pv_el = eta_pv * A_roof * GHI`  
7. **净零裕度**：`MNZ = alpha_to_cool * E_pv_el - E_cool_el_2040`  
8. **Pareto 筛选**：最小化 `E_cool_el` 与 `-MNZ`

---

## 关键文件与职责
- `config.py`  
  - `SINGAPORE`：站点与时区  
  - `MATERIALS`：材料热参数  
  - `DEFAULT_HVAC`、`DEFAULT_PV`、`DEFAULT_BOUNDS`
  - `N_SAMPLES`、`RNG_SEED`

- `evaluate_sg.py`  
  计算太阳得热、两套温度情景下的制冷能耗、PV 产能、MNZ。

- `thermal_sg.py`  
  2R2C + deadband HVAC（与 HVAC_Pareto 类似，但默认初温不同）。

- `pareto.py`  
  Pareto 过滤函数。

- `pvgis_fetch.py`  
  PVGIS TMY 获取并标准化为 `datetime,DNI,DHI,GHI,T_out`。

- `main_sg_pareto.py`  
  主入口：完整流程 + 输出 + ECharts 可视化。

---

## 设计变量
在采样中，每个方案包含 5 个变量：
`(dN, dS, etaE, etaW, material_m)`
- `dN, dS`：北/南向挑檐深度（m）  
- `etaE, etaW`：东/西向百叶参数（rad）  
- `material_m`：材料编号（见 `config.MATERIALS`）

---

## 输入与输出
### 输入
PVGIS 生成的标准天气 CSV（由脚本自动下载并缓存）：
- `datetime, DNI, DHI, GHI, T_out`

### 输出（按运行批次）
位于 `SG_Pareto/analysis_runs/sg_pareto_YYYYMMDD_HHMMSS/`：
- `weather/`：原始与趋势天气  
- `results/`：
  - `samples.csv`：全部样本  
  - `pareto.csv`：Pareto 前沿  
  - `meta.json`：参数与输出摘要  
- `plots/pareto.png`：散点图  
- `echarts/charts.json` + `echarts/index.html`

---

## 常用运行方式
```powershell
python SG_Pareto/main_sg_pareto.py
```
运行后会自动：
- 获取/复用 PVGIS 天气
- 生成 2040 情景天气
- 输出 Pareto 与图表文件

---

## 注意事项
- 目标函数是 **最小化** `(E_cool_el, -MNZ)`；MNZ 越大越好。  
- `DELTA_TREND_C` 默认为 +1.5°C，可在 `config.py` 调整。  
- `COP_cool` 影响电耗换算；PV 参数影响 MNZ。  
