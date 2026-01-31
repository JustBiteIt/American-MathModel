# constraints.py
import numpy as np

def occupancy_mask(times, occ_start_hour: int, occ_end_hour: int) -> np.ndarray:
    """Occupied if local hour in [start, end)."""
    hours = np.array([t.hour for t in times])
    return (hours >= occ_start_hour) & (hours < occ_end_hour)

def sunlit_mask(DNI: np.ndarray, alpha_rad: np.ndarray, DHI: np.ndarray | None = None, GHI: np.ndarray | None = None) -> np.ndarray:
    """
    白天判定：太阳在地平线上方(alpha>0) 即认为是白天。
    为了兼容旧逻辑，如果传入了DHI/GHI，也允许在阴天(DNI=0)时仍然计入白天。
    """
    base = (alpha_rad > 0.0)

    if (DHI is None) and (GHI is None):
        # 兼容旧逻辑
        return base & (DNI > 0.0)

    d = 0.0 if DHI is None else DHI
    g = 0.0 if GHI is None else GHI
    return base & ((DNI > 0.0) | (d > 0.0) | (g > 0.0))

def evaluate_visual_constraints(
    E_vis_W_per_m2: np.ndarray,
    mask_day_occ: np.ndarray,
    E_min: float,
    p_min: float,
    E_glare: float,
    H_glare_max: int
):
    """
    Daylight adequacy: fraction of (day & occupied) hours with E_vis >= E_min >= p_min
    Glare: count of (day & occupied) hours with E_vis >= E_glare <= H_glare_max
    """
    idx = np.where(mask_day_occ)[0]
    if len(idx) == 0:
        # no occupied sunlit hours in data window -> treat as infeasible unless user changes definition
        return False, {"p_daylight": 0.0, "H_glare": int(1e9)}

    E = E_vis_W_per_m2[idx]
    p_daylight = float(np.mean(E >= E_min))
    H_glare = int(np.sum(E >= E_glare))

    feasible = (p_daylight >= p_min) and (H_glare <= H_glare_max)
    return feasible, {"p_daylight": p_daylight, "H_glare": H_glare}
