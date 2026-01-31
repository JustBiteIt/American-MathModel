# config.py
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class LocationConfig:
    latitude: float = 1.3521     # Singapore                            
    longitude: float = 103.8198
    tz: str = "Asia/Singapore"


@dataclass(frozen=True)
class BuildingConfig:
    length_m: float = 60.0
    width_m: float = 24.0
    floors: int = 2

    # Window-to-wall ratios
    wwr_south: float = 0.45
    wwr_other: float = 0.30

    story_height_m: float = 3.6
    window_height_m: float = 2.0

    # Facade azimuths (deg): N=0, E=90, S=180, W=270
    facade_azimuth_deg: Dict[str, float] = None

    def __post_init__(self):
        object.__setattr__(self, "facade_azimuth_deg",
                           {"N": 0.0, "E": 90.0, "S": 180.0, "W": 270.0})

    def facade_areas(self) -> Dict[str, float]:
        """Gross facade areas for the whole building (all floors)."""
        height = self.floors * self.story_height_m
        return {
            "N": self.length_m * height,
            "S": self.length_m * height,
            "E": self.width_m * height,
            "W": self.width_m * height,
        }

    def window_areas(self) -> Dict[str, float]:
        """Window areas by facade = WWR * gross facade area."""
        A_fac = self.facade_areas()
        return {
            "S": self.wwr_south * A_fac["S"],
            "N": self.wwr_other * A_fac["N"],
            "E": self.wwr_other * A_fac["E"],
            "W": self.wwr_other * A_fac["W"],
        }


@dataclass(frozen=True)
class OpticalConfig:
    # tau_heat：太阳短波得热通道透过（用于制冷负荷）
    tau_heat: float = 0.55
    # tau_vis：可见光通道透过（用于采光/炫光代理）
    tau_vis: float = 0.60
    # 地表反照率（如果你已经有了就别重复）
    rho_g: float = 0.20

    # ✅ 新增：散射+地反的“透过比例”开关（0~1）
    k_other: float = 0.45

@dataclass(frozen=True)
class ThermalConfig:
    # 1R1C lumped parameters
    C_J_per_K: float = 2.0e8
    H_W_per_K: float = 4.0e4

    T_init_C: float = 26.0
    T_set_C: float = 26.0
    dt_hours: float = 1.0


@dataclass(frozen=True)
class ScheduleConfig:
    occ_start_hour: int = 8
    occ_end_hour: int = 18


@dataclass(frozen=True)
class VisualConstraintConfig:
    # E_vis = Q_vis / A_win_total (W/m^2)
    E_min: float = 7.8922
    p_min: float = 0.70
    E_glare: float = 108.5986
    H_glare_max: int = 183


@dataclass(frozen=True)
class DecisionBounds:
    dN_bounds_m: Tuple[float, float] = (0.0, 3.0)
    dS_bounds_m: Tuple[float, float] = (0.0, 3.0)
    etaE_bounds_rad: Tuple[float, float] = (0.10, 1.57)
    etaW_bounds_rad: Tuple[float, float] = (0.10, 1.57)