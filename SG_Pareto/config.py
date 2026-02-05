from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

from borealis_model.config import (
    LocationConfig,
    BuildingConfig,
    OpticalConfig,
    BorealisOptical,
    DecisionBounds,
)


PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
PACKAGE_ROOT: Path = Path(__file__).resolve().parent
ANALYSIS_ROOT: Path = PACKAGE_ROOT / "analysis_runs"


@dataclass(frozen=True)
class SiteConfig:
    name: str
    latitude: float
    longitude: float
    tz: str


SINGAPORE = SiteConfig(
    name="Singapore",
    latitude=1.3521,
    longitude=103.8198,
    tz="Asia/Singapore",
)


DELTA_TREND_C: float = 1.5
USE_AR1_NOISE: bool = False
AR1_SEED: int = 42


@dataclass(frozen=True)
class HVACConfig:
    T_heat_C: float = 18.0
    T_cool_C: float = 26.0
    T_init_C: float = 24.0
    dt_hours: float = 1.0
    COP_cool: float = 3.0


@dataclass(frozen=True)
class PVConfig:
    eta_pv: float = 0.18
    alpha_to_cool: float = 1.0


@dataclass(frozen=True)
class MaterialRC:
    Ci_J_per_K: float
    Cm_J_per_K: float
    Ria_K_per_W: float
    Rim_K_per_W: float
    eta_air: float


MATERIALS: Dict[int, MaterialRC] = {
    1: MaterialRC(
        Ci_J_per_K=2.0e8,
        Cm_J_per_K=8.0e8,
        Ria_K_per_W=2.5e-4,
        Rim_K_per_W=1.0e-4,
        eta_air=0.35,
    ),
    2: MaterialRC(
        Ci_J_per_K=2.0e8,
        Cm_J_per_K=1.2e9,
        Ria_K_per_W=4.0e-4,
        Rim_K_per_W=1.2e-4,
        eta_air=0.25,
    ),
}


DEFAULT_BUILDING: BuildingConfig = BuildingConfig(
    wwr_south=0.45,
    wwr_other=0.30,
    wwr_by_facade=None,
)

DEFAULT_OPTICAL: OpticalConfig = OpticalConfig()
DEFAULT_BOREALIS_OPTICAL: BorealisOptical = BorealisOptical()
DEFAULT_BOUNDS: DecisionBounds = DecisionBounds()

DEFAULT_HVAC: HVACConfig = HVACConfig()
DEFAULT_PV: PVConfig = PVConfig()

N_SAMPLES: int = 2000
RNG_SEED: int = 42

# -------------------------
# Geometry & facade WWR constraints (per document)
# -------------------------
L0_M: float = 60.0
W0_M: float = 24.0
H0_M: float = 7.2
A0_M2: float = L0_M * W0_M

EPS_A: float = 0.05
R_MIN: float = 2.0
R_MAX: float = 3.0
EPS_H: float = 0.10
THETA_MAX_DEG: float = 30.0

WWR_MIN: float = 0.20
WWR_MAX: float = 0.60

# Passive shading window height (H_win, meters)
WINDOW_HEIGHT_M: float = 2.0

# Hard constraint for peak cooling electric power (kW_el)
PEAK_COOLING_LIMIT_KW: float = 350.0

# Sensitivity scan list for peak limit (kW_el)
PEAK_LIMIT_LIST_KW = [200, 250, 300, 350, 400, 450, 500]

# Baseline decision variables (no shading)
BASELINE_DN_M: float = 0.0
BASELINE_DS_M: float = 0.0
BASELINE_ETA_E_RAD: float = 1.57
BASELINE_ETA_W_RAD: float = 1.57
BASELINE_THETA_DEG: float = 0.0

# Comfort band for indoor temperature (°C)
COMFORT_LOW_C: float = 24.0
COMFORT_HIGH_C: float = 26.0
