import sys
from pathlib import Path
from dataclasses import replace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataclasses import replace
import numpy as np

from config import LocationConfig, BuildingConfig, OpticalConfig, ThermalConfig, ScheduleConfig, VisualConstraintConfig, DecisionBounds
from data_io import load_weather_csv
from optimize import evaluate_design

def main():
    location = LocationConfig()
    building = BuildingConfig()
    optical0 = OpticalConfig()
    thermal = ThermalConfig()
    schedule = ScheduleConfig()
    visual = VisualConstraintConfig()      # 这里你已把 H_glare_max 改成 183
    bounds = DecisionBounds()

    df = load_weather_csv(r"D:\ICM_CODE\weather_singapore_hourly.csv", tz=location.tz)

    # baseline：无挑檐 + 百叶尽量开
    x_base = (0.0, 0.0, bounds.etaE_bounds_rad[1], bounds.etaW_bounds_rad[1])

    ks = np.arange(0.34, 0.401, 0.01)  # 精扫
    for k in ks:
        optical = replace(optical0, k_other=float(k))  # 关键：Frozen dataclass 用 replace
        r = evaluate_design(df, location, building, optical, thermal, schedule, visual, x_base)
        H = r["vis"]["H_glare"]
        p = r["vis"]["p_daylight"]
        print(f"k_other={k:.2f}  feasible={r['feasible']}  H={H}  p={p:.3f}  L={r['L_year_kWh']:.1f}")

if __name__ == "__main__":
    main()
