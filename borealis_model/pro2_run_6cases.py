# pro2_run_6cases.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd

from config import (
    LocationConfig,
    BuildingConfig,
    OpticalConfig,
    ScheduleConfig,
    BorealisOptical,
    Borealis2R2C,
    BorealisSeasons,
    DecisionBounds,
)
from data_io import load_weather_csv
from evaluate_borealis import evaluate_borealis


# -----------------------------
# user settings
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]  # -> D:\ICM_CODE (按你 main_borealis 的逻辑)
WEATHER = ROOT / "weather_helsinki_tmy_hourly.csv"

# Scheme A settings (跟你 main_borealis 一致)
N_SAMPLES = 2000
SEED = 20260131
PROGRESS_EVERY = 200
RATIO = 0.6
OH_FLOOR = 5.0
HOH_FLOOR = 5.0

# "no shading" baseline x (跟你代码一致)
X_BASELINE: Tuple[float, float, float, float] = (0.0, 0.0, 1.57, 1.57)

OUT_JSON = Path("results.json")


# -----------------------------
# optimizer (copy from your main_borealis with minimal changes)
# -----------------------------
def _optimize_scheme_A_feasible_min_Eh(
    df: pd.DataFrame,
    location: LocationConfig,
    building: BuildingConfig,
    optical_base: OpticalConfig,
    schedule: ScheduleConfig,
    bo_opt: BorealisOptical,
    bo_therm: Borealis2R2C,
    bo_seasons: BorealisSeasons,
    bounds: DecisionBounds,
    *,
    baseline_x: Tuple[float, float, float, float],
    baseline_res: Dict[str, Any],
    n_samples: int = N_SAMPLES,
    seed: int = SEED,
    progress_every: int = PROGRESS_EVERY,
    oh_floor: float = OH_FLOOR,
    hoh_floor: float = HOH_FLOOR,
    ratio: float = RATIO,
) -> Tuple[Tuple[float, float, float, float], Dict[str, Any], Dict[str, Any]]:
    rng = np.random.default_rng(seed)

    base_OH = float(baseline_res["OH_degC_h"])
    base_Hoh = float(baseline_res["Hoh_h"])

    OH_max = max(float(oh_floor), float(ratio) * base_OH)
    Hoh_max = max(float(hoh_floor), float(ratio) * base_Hoh)

    baseline_feasible = (base_OH <= OH_max) and (base_Hoh <= Hoh_max)

    best_x: Optional[Tuple[float, float, float, float]] = baseline_x if baseline_feasible else None
    best_res: Optional[Dict[str, Any]] = baseline_res if baseline_feasible else None
    best_Eh: float = float(baseline_res["Eh_kWh"]) if baseline_feasible else float("inf")

    feasible_count = 1 if baseline_feasible else 0
    t0 = pd.Timestamp.now()

    for i in range(1, n_samples + 1):
        dN = rng.uniform(*bounds.dN_bounds_m)
        dS = rng.uniform(*bounds.dS_bounds_m)
        etaE = rng.uniform(*bounds.etaE_bounds_rad)
        etaW = rng.uniform(*bounds.etaW_bounds_rad)
        x = (float(dN), float(dS), float(etaE), float(etaW))

        res = evaluate_borealis(
            df=df,
            location=location,
            building=building,
            optical_base=optical_base,
            schedule=schedule,
            bo_optical=bo_opt,
            bo_therm=bo_therm,
            bo_seasons=bo_seasons,
            x=x,
            return_series=False,  # speed
        )

        OH = float(res["OH_degC_h"])
        Hoh = float(res["Hoh_h"])
        if (OH <= OH_max) and (Hoh <= Hoh_max):
            feasible_count += 1
            Eh = float(res["Eh_kWh"])
            if Eh < best_Eh:
                best_Eh = Eh
                best_x = x
                best_res = res

        if progress_every > 0 and (i % progress_every == 0):
            elapsed = (pd.Timestamp.now() - t0).total_seconds()
            print(
                f"  ... {i}/{n_samples} done | feasible={feasible_count} | "
                f"best_Eh={best_Eh:.2f} | {elapsed:.1f}s"
            )

    if best_x is None or best_res is None:
        best_x = baseline_x
        best_res = baseline_res
        status = "no_feasible_found"
    else:
        status = "feasible_found"

    info = {
        "status": status,
        "baseline_feasible": bool(baseline_feasible),
        "OH_max": float(OH_max),
        "Hoh_max": float(Hoh_max),
        "feasible_count": int(feasible_count),
        "n_samples": int(n_samples),
        "seed": int(seed),
        "ratio": float(ratio),
    }
    return best_x, best_res, info


def _assert_metrics(res: Dict[str, Any], where: str) -> None:
    need = ["Eh_kWh", "OH_degC_h", "Hoh_h"]
    miss = [k for k in need if k not in res]
    if miss:
        raise KeyError(f"{where} missing keys: {miss}. Got keys={list(res.keys())}")


# -----------------------------
# main
# -----------------------------
def main() -> None:
    if not WEATHER.exists():
        raise FileNotFoundError(f"weather csv not found: {WEATHER}")

    location = LocationConfig()
    df = load_weather_csv(str(WEATHER), tz=location.tz)

    optical_base = OpticalConfig()
    schedule = ScheduleConfig()
    seasons = BorealisSeasons()
    bounds = DecisionBounds()

    bo_opt = BorealisOptical(
        k_other=0.35,
        V_sky=1.0,
        V_gr=1.0,
        rho_bare=0.20,
        rho_snow=0.70,
        snow_Ta_threshold_C=0.0,
    )

    therm_brick = Borealis2R2C(Cm_J_per_K=6.0e8, Rim_K_per_W=1.6e-4)
    therm_concrete = Borealis2R2C(Cm_J_per_K=1.2e9, Rim_K_per_W=9.0e-5)

    # 你当前口径：
    # old WWR = passive; new WWR = fi
    building_old = BuildingConfig(wwr_south=0.45, wwr_other=0.30)  # old/passive
    building_new = BuildingConfig(wwr_south=0.40, wwr_other=0.25)  # new/fi

    # 6 cases mapping (与你手绘①~⑥一致)
    case_specs = [
        ("case1", "① Brick + old WWR + no shading", building_old, therm_brick, False),
        ("case2", "② Brick + new WWR + no shading", building_new, therm_brick, False),
        ("case3", "③ Brick + new WWR + shading",    building_new, therm_brick, True),
        ("case4", "④ Concrete + old WWR + no shading", building_old, therm_concrete, False),
        ("case5", "⑤ Concrete + new WWR + no shading", building_new, therm_concrete, False),
        ("case6", "⑥ Concrete + new WWR + shading",    building_new, therm_concrete, True),
    ]

    out: Dict[str, Any] = {
        "meta": {
            "weather_csv": str(WEATHER),
            "X_BASELINE": list(X_BASELINE),
            "schemeA": {
                "n_samples": N_SAMPLES,
                "seed": SEED,
                "ratio": RATIO,
                "oh_floor": OH_FLOOR,
                "hoh_floor": HOH_FLOOR,
            }
        },
        "cases": {}
    }

    print("\n===== Running 6 cases -> results.json =====\n")

    for case_id, label, building, therm, is_shading in case_specs:
        print(f"[{case_id}] {label}")

        # baseline (no shading x)
        base = evaluate_borealis(
            df=df,
            location=location,
            building=building,
            optical_base=optical_base,
            schedule=schedule,
            bo_optical=bo_opt,
            bo_therm=therm,
            bo_seasons=seasons,
            x=X_BASELINE,
            return_series=False,
        )
        _assert_metrics(base, f"{case_id}.baseline")

        if is_shading:
            best_x, best_res, info = _optimize_scheme_A_feasible_min_Eh(
                df=df,
                location=location,
                building=building,
                optical_base=optical_base,
                schedule=schedule,
                bo_opt=bo_opt,
                bo_therm=therm,
                bo_seasons=seasons,
                bounds=bounds,
                baseline_x=X_BASELINE,
                baseline_res=base,
            )
            _assert_metrics(best_res, f"{case_id}.best")
            use = best_res
            x_used = best_x
        else:
            info = {"status": "no_shading"}
            use = base
            x_used = X_BASELINE

        print(f"  x_used = {tuple(x_used)}")
        print(f"  Eh={use['Eh_kWh']:.2f} kWh, OH={use['OH_degC_h']:.2f}, Hoh={use['Hoh_h']:.1f}\n")

        out["cases"][case_id] = {
            "label": label,
            "has_shading": bool(is_shading),
            "building": {
                "wwr_south": float(building.wwr_south),
                "wwr_other": float(building.wwr_other),
            },
            "thermal": {
                "Cm_J_per_K": float(therm.Cm_J_per_K),
                "Rim_K_per_W": float(therm.Rim_K_per_W),
                "T_max_C": float(therm.T_max_C),
            },
            "scenarios": {
                "baseline": {
                    "x": list(x_used),
                    "metrics": {
                        # dashboard 侧用 Eh_kWh_total
                        "Eh_kWh_total": float(use["Eh_kWh"]),
                        "OH_degC_h": float(use["OH_degC_h"]),
                        "Hoh_h": float(use["Hoh_h"]),
                    },
                    "schemeA_info": info,
                }
            }
        }

    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ Wrote: {OUT_JSON.resolve()}")
    print("\nNext: put results.json next to dashboard.html and open via http.server.\n")


if __name__ == "__main__":
    main()
