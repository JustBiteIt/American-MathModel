# HVAC_Pareto

## 模块定位
`HVAC_Pareto` 是面向 **供暖/制冷负荷双目标优化** 的实验模块。它复用 `borealis_model` 的太阳与遮阳计算，但热模型改为 **2R2C + 理想死区 HVAC**，并通过随机采样生成设计空间，在 **(E_cool, E_heat)** 平面上提取 Pareto 前沿与代表解。

适用场景：
- 评估遮阳参数对冷/热负荷的权衡
- 多城市对比（city benchmark）
- 纬度扫描（latitude scan）

---

## 核心流程（单 case）
1. 读取天气 CSV（标准列：`datetime,DNI,DHI,GHI,T_out`）  
2. 太阳几何 + 立面直射  
3. 遮阳透射（x = dN,dS,etaE,etaW）  
4. 室内太阳得热 `Phi_s(t)`  
5. 2R2C + deadband HVAC → `Phi_h(t)`  
6. 统计年冷负荷 `E_cool_kWh` 和年热负荷 `E_heat_kWh`  
7. 采样多设计点 → Pareto 前沿 → 代表点（基于气候指数权重）

---

## 关键文件与职责
### 配置与入口
- `config.py`  
  - `SITES`：城市/站点字典  
  - `DEFAULT_SITE`：默认站点  
  - `ModelConfig`：统一模型参数  
  - `resolve_weather_csv()`：天气文件解析  
  - `resolve_output_dir()`：输出目录约束

- `main_hvac_pareto.py`  
  CLI 入口：执行单站点 Pareto 运行。

### 物理与指标
- `evaluate_hvac.py`  
  核心评估：天气 → 辐照度 → 太阳得热 → HVAC 负荷。
- `thermal_hvac.py`  
  2R2C + deadband HVAC (加热/制冷)。
- `pareto.py`  
  Pareto 过滤与代表点选择逻辑。

### 实验工具
- `case_runner.py`  
  批量采样、生成 Pareto、输出 PNG + JSON + CSV。
- `pvgis_fetch.py`  
  从 PVGIS 下载 TMY 天气并格式化为标准 CSV。
- `run_city_benchmark.py`  
  多城市比较，生成汇总与叠加 Pareto 图。
- `run_latitude_scan.py`  
  固定经度，扫描纬度，输出遮阳代表参数随纬度变化曲线。

---

## 设计变量
`x = (dN, dS, etaE, etaW)`
- `dN, dS`：北/南向挑檐深度（m）  
- `etaE, etaW`：东/西向百叶参数（rad）

采样范围由 `DecisionBounds` 控制。

---

## 输入与输出
### 输入
标准天气 CSV（可本地/远程获取）：
- `datetime`（带时区或可转时区）
- `DNI`, `DHI`, `GHI`（W/m2）
- `T_out`（摄氏度）

### 输出（单 case）
默认输出在 `HVAC_Pareto/hvac_outputs/`：
- `results.csv`：全部随机样本  
- `pareto.csv`：Pareto 前沿  
- `pareto.png`：散点图  
- `meta.json`：代表点/气候指标等  
- `echarts_case.json/html`：网页交互图

### 输出（批量实验）
多城市与纬度扫描会写入项目根目录：
`D:\ICM_CODE\analysis_runs\...`

---

## 常用运行方式
```powershell
# 单站点 Pareto
python HVAC_Pareto/main_hvac_pareto.py

# 指定天气与采样数
python HVAC_Pareto/main_hvac_pareto.py --weather D:\ICM_CODE\weather_helsinki_tmy_hourly.csv --n 3000 --seed 123

# 多城市 benchmark（自动 PVGIS 下载）
python HVAC_Pareto/run_city_benchmark.py

# 纬度扫描（自动 PVGIS 下载）
python HVAC_Pareto/run_latitude_scan.py
```

---

## 代表点选取逻辑
`case_runner.compute_hdd_cdd_from_hourly()`  
→ 计算 HDD/CDD  
→ 得到气候指数 `xi = (CDD - HDD)/(CDD + HDD)`  
→ 权重 `w_cool`、`w_heat`  
→ 在 Pareto 集内最小化 `w_cool*E_cool + w_heat*E_heat`

---

## 注意事项
- 模块依赖 `borealis_model` 的太阳/遮阳/反照率实现。  
- 输出目录被限制在 `HVAC_Pareto/` 内（防止误写）。  
- 运行 `run_city_benchmark.py` / `run_latitude_scan.py` 会自动在 `analysis_runs` 生成新实验目录。  
