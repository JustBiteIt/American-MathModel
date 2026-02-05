from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


FACADES = ("N", "E", "S", "W")


@dataclass(frozen=True)
class GeometryDecision:
    L_m: float
    W_m: float
    H_m: float
    theta_deg: float
    wwr_N: float
    wwr_E: float
    wwr_S: float
    wwr_W: float

    def roof_area_m2(self) -> float:
        return float(self.L_m * self.W_m)

    def facade_areas_m2(self) -> Dict[str, float]:
        # 文档约定：N/S = W*H, E/W = L*H
        return {
            "N": float(self.W_m * self.H_m),
            "S": float(self.W_m * self.H_m),
            "E": float(self.L_m * self.H_m),
            "W": float(self.L_m * self.H_m),
        }

    def window_areas_m2(self) -> Dict[str, float]:
        A_fac = self.facade_areas_m2()
        return {
            "N": float(self.wwr_N) * A_fac["N"],
            "E": float(self.wwr_E) * A_fac["E"],
            "S": float(self.wwr_S) * A_fac["S"],
            "W": float(self.wwr_W) * A_fac["W"],
        }

    def window_area_total_m2(self) -> float:
        A_win = self.window_areas_m2()
        return float(sum(A_win.values()))

    def facade_azimuths_deg(self) -> Dict[str, float]:
        # 基准方位角：N=0, E=90, S=180, W=270
        theta = float(self.theta_deg)
        return {
            "N": (0.0 + theta) % 360.0,
            "E": (90.0 + theta) % 360.0,
            "S": (180.0 + theta) % 360.0,
            "W": (270.0 + theta) % 360.0,
        }


def sample_geometry_and_wwr(
    rng: np.random.Generator,
    *,
    A0_m2: float,
    eps_A: float,
    r_min: float,
    r_max: float,
    H0_m: float,
    eps_H: float,
    theta_max_deg: float,
    wwr_min: float,
    wwr_max: float,
) -> GeometryDecision:
    """Sample geometry + facade WWR with document constraints."""
    A_min = (1.0 - float(eps_A)) * float(A0_m2)
    A_max = (1.0 + float(eps_A)) * float(A0_m2)
    A = rng.uniform(A_min, A_max)

    r = rng.uniform(float(r_min), float(r_max))
    L = float(np.sqrt(A * r))
    W = float(np.sqrt(A / r))

    H_min = (1.0 - float(eps_H)) * float(H0_m)
    H_max = (1.0 + float(eps_H)) * float(H0_m)
    H = float(rng.uniform(H_min, H_max))

    theta = float(rng.uniform(-float(theta_max_deg), float(theta_max_deg)))

    wwr_N = float(rng.uniform(float(wwr_min), float(wwr_max)))
    wwr_E = float(rng.uniform(float(wwr_min), float(wwr_max)))
    wwr_S = float(rng.uniform(float(wwr_min), float(wwr_max)))
    wwr_W = float(rng.uniform(float(wwr_min), float(wwr_max)))

    return GeometryDecision(
        L_m=L,
        W_m=W,
        H_m=H,
        theta_deg=theta,
        wwr_N=wwr_N,
        wwr_E=wwr_E,
        wwr_S=wwr_S,
        wwr_W=wwr_W,
    )

