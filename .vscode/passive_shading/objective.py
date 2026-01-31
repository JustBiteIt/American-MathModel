# objective.py
import numpy as np
from typing import Dict, Any

def annual_cooling_load_kwh(L_kWh: np.ndarray) -> float:
    return float(np.sum(L_kWh))

def baseline_savings(baseline: float, design: float) -> Dict[str, float]:
    delta = baseline - design
    ratio = delta / baseline if baseline > 0 else 0.0
    return {"L_base_kWh": baseline, "L_design_kWh": design, "delta_kWh": delta, "save_ratio": ratio}
