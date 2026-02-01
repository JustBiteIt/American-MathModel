# borealis_model/config.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional


# =========================
# Base configs (location / geometry / optics / schedule)
# =========================

@dataclass(frozen=True)
class LocationConfig:
    # Helsinki (Borealis) default
    latitude: float = 60.1699
    longitude: float = 24.9384
    tz: str = "Europe/Helsinki"


@dataclass(frozen=True)
class BuildingConfig:
    length_m: float = 60.0
    width_m: float = 24.0
    floors: int = 2

    # legacy: passive_shading 风格（南向单独，其它一致）
    wwr_south: float = 0.45
    wwr_other: float = 0.30

    # ✅ 新增：按立面分别给 WWR（优先级高于 wwr_south/wwr_other）
    # keys: "N","E","S","W"
    wwr_by_facade: Optional[Dict[str, float]] = None

    story_height_m: float = 3.6
    window_height_m: float = 2.0

    # Facade azimuths (deg): N=0, E=90, S=180, W=270
    facade_azimuth_deg: Dict[str, float] = None

    def __post_init__(self):
        if self.facade_azimuth_deg is None:
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

        if self.wwr_by_facade is not None:
            w = self.wwr_by_facade
            return {
                "N": float(w["N"]) * A_fac["N"],
                "E": float(w["E"]) * A_fac["E"],
                "S": float(w["S"]) * A_fac["S"],
                "W": float(w["W"]) * A_fac["W"],
            }

        # fallback: legacy mode
        return {
            "S": self.wwr_south * A_fac["S"],
            "N": self.wwr_other * A_fac["N"],
            "E": self.wwr_other * A_fac["E"],
            "W": self.wwr_other * A_fac["W"],
        }


@dataclass(frozen=True)
class OpticalConfig:
    # 用于“热得热”的等效透过（在 evaluate_borealis 里当 g_eq 用）
    tau_heat: float = 0.55
    # 可见光通道（未来做采光评价可用）
    tau_vis: float = 0.60
    # 兼容字段（Borealis 实际用 bo_optical 的 rho_bare/rho_snow + rho_ground_timeseries）
    rho_g: float = 0.20


@dataclass(frozen=True)
class ScheduleConfig:
    occ_start_hour: int = 8
    occ_end_hour: int = 18


@dataclass(frozen=True)
class DecisionBounds:
    # (dN, dS, etaE, etaW)
    dN_bounds_m: Tuple[float, float] = (0.0, 3.0)
    dS_bounds_m: Tuple[float, float] = (0.0, 3.0)
    etaE_bounds_rad: Tuple[float, float] = (0.10, 1.57)
    etaW_bounds_rad: Tuple[float, float] = (0.10, 1.57)


# =========================
# Borealis-specific configs (snow albedo + 2R2C + seasons + k_other)
# =========================

@dataclass(frozen=True)
class BorealisOptical:
    """
    If(t) = u_f(t)*I_beam,f(t) + k_other*(V_sky*I_sky + V_gr*I_ground)
    where I_sky = 0.5*DHI, I_ground = 0.5*rho_g(t)*GHI
    """
    k_other: float = 0.35
    V_sky: float = 1.0
    V_gr: float = 1.0

    # ✅ 按你要求：bare=0.2, snow=0.7
    rho_bare: float = 0.20
    rho_snow: float = 0.70
    snow_Ta_threshold_C: float = 0.0


@dataclass(frozen=True)
class Borealis2R2C:
    Ci_J_per_K: float = 2.0e8
    Cm_J_per_K: float = 8.0e8
    Ria_K_per_W: float = 2.5e-4
    Rim_K_per_W: float = 1.0e-4
    eta_air: float = 0.35

    T_min_C: float = 18.0
    T_max_C: float = 26.0

    T_init_C: float = 21.0
    dt_hours: float = 1.0


@dataclass(frozen=True)
class BorealisSeasons:
    heating_months: Tuple[int, ...] = (10, 11, 12, 1, 2, 3, 4)  # Oct-Apr
    summer_months: Tuple[int, ...] = (5, 6, 7, 8, 9)            # May-Sep


# =========================
# ✅ Material presets for your comparison study
# =========================

@dataclass(frozen=True)
class MaterialConfig:
    name: str
    Cm_J_per_K: float
    Rim_K_per_W: float


# 你要的两种材料（推荐初值，用于对比）
MATERIAL_BRICK_HYDRONIC = MaterialConfig(
    name="brick_wall_hydronic",
    Cm_J_per_K=6.0e8,
    Rim_K_per_W=1.3e-4,
)

MATERIAL_CONCRETE_HYDRONIC = MaterialConfig(
    name="concrete_wall_hydronic",
    Cm_J_per_K=9.0e8,
    Rim_K_per_W=0.9e-4,
)

# 对照基线材料：用你 Borealis2R2C 默认（最不容易“扯淡”）
MATERIAL_BASELINE = MaterialConfig(
    name="baseline_default",
    Cm_J_per_K=8.0e8,
    Rim_K_per_W=1.0e-4,
)


# =========================
# ✅ WWR schemes
# =========================

WWR_SCHEME_FINLAND = {"S": 0.40, "W": 0.45, "E": 0.55, "N": 0.25}
# passive_shading 风格（默认 South=0.45, other=0.30）不用 dict 也行，但这里给一个显式版本方便对照
WWR_SCHEME_PASSIVE = {"S": 0.45, "W": 0.30, "E": 0.30, "N": 0.30}
