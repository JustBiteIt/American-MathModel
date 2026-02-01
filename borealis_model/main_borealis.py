# borealis_model/main_borealis.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import time

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


def _fmt_x(x: Tuple[float, float, float, float]) -> str:
    return f"({x[0]:.6g}, {x[1]:.6g}, {x[2]:.6g}, {x[3]:.6g})"


def _feasible_thresholds_from_baseline(
    OH0: float,
    Hoh0: float,
    *,
    OH_ratio: float = 0.60,
    Hoh_ratio: float = 0.60,
    OH_abs_floor: float = 5.0,
    Hoh_abs_floor: float = 5.0,
) -> Tuple[float, float]:
    OH_max = max(float(OH_abs_floor), float(OH_ratio) * float(OH0))
    Hoh_max = max(float(Hoh_abs_floor), float(Hoh_ratio) * float(Hoh0))
    return OH_max, Hoh_max


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
    OH_max: float,
    Hoh_max: float,
    n_samples: int = 2000,
    seed: int = 20260201,
    progress_every: int = 200,
) -> Tuple[Tuple[float, float, float, float], Dict[str, Any], Dict[str, Any]]:

    rng = np.random.default_rng(seed)
    t0 = time.time()

    best_feas_x: Optional[Tuple[float, float, float, float]] = None
    best_feas_res: Optional[Dict[str, Any]] = None
    best_feas_Eh = float("inf")
    feas_count = 0

    best_fallback_x = None
    best_fallback_res = None
    best_fallback_key = (float("inf"), float("inf"))  # (violation, Eh)

    def violation_score(res: Dict[str, Any]) -> float:
        OH = float(res["OH_degC_h"])
        Hoh = float(res["Hoh_h"])
        v1 = max(0.0, OH - float(OH_max)) / max(float(OH_max), 1e-9)
        v2 = max(0.0, Hoh - float(Hoh_max)) / max(float(Hoh_max), 1e-9)
        return v1 + v2

    for i in range(int(n_samples)):
        dN = rng.uniform(*bounds.dN_bounds_m)
        dS = rng.uniform(*bounds.dS_bounds_m)
        etaE = rng.uniform(*bounds.etaE_bounds_rad)
        etaW = rng.uniform(*bounds.etaW_bounds_rad)
        x = (float(dN), float(dS), float(etaE), float(etaW))

        # ✅ optimization stage: metrics-only
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
            return_series=False,
        )

        Eh = float(res["Eh_kWh"])
        OH = float(res["OH_degC_h"])
        Hoh = float(res["Hoh_h"])

        feas = (OH <= float(OH_max)) and (Hoh <= float(Hoh_max))
        if feas:
            feas_count += 1
            if Eh < best_feas_Eh:
                best_feas_Eh = Eh
                best_feas_x = x
                best_feas_res = res
        else:
            v = violation_score(res)
            key = (v, Eh)
            if key < best_fallback_key:
                best_fallback_key = key
                best_fallback_x = x
                best_fallback_res = res

        if (i + 1) % int(progress_every) == 0:
            dt = time.time() - t0
            msg = f"  ... {i+1}/{n_samples} done | feasible={feas_count} | best_Eh={best_feas_Eh if best_feas_x else None} | {dt:.1f}s"
            print(msg)

    if best_feas_x is not None and best_feas_res is not None:
        info = {"status": "feasible_found", "OH_max": OH_max, "Hoh_max": Hoh_max, "n_samples": n_samples, "seed": seed, "feasible_count": feas_count}
        return best_feas_x, best_feas_res, info

    if best_fallback_x is None or best_fallback_res is None:
        raise RuntimeError("Optimization failed: no candidates evaluated.")

    info = {"status": "no_feasible_solution", "OH_max": OH_max, "Hoh_max": Hoh_max, "n_samples": n_samples, "seed": seed,
            "feasible_count": feas_count, "fallback_violation_score": best_fallback_key[0]}
    return best_fallback_x, best_fallback_res, info


def main() -> None:
    ROOT = Path(__file__).resolve().parents[1]
    weather_csv = ROOT / "weather_helsinki_tmy_hourly.csv"
    if not weather_csv.exists():
        alt = Path(__file__).resolve().parent / "weather_helsinki_tmy_hourly.csv"
        if alt.exists():
            weather_csv = alt
        else:
            raise FileNotFoundError(f"找不到: {weather_csv}")

    print(f"✅ Using weather file: {weather_csv}")

    location = LocationConfig()
    df = load_weather_csv(str(weather_csv), tz=location.tz)

    optical_base = OpticalConfig()
    schedule = ScheduleConfig()
    bo_seasons = BorealisSeasons()
    bounds = DecisionBounds()

    bo_opt = BorealisOptical(
        k_other=0.35,
        V_sky=1.0,
        V_gr=1.0,
        rho_bare=0.20,
        rho_snow=0.70,
        snow_Ta_threshold_C=0.0,
    )

    X_BASELINE = (0.0, 0.0, 1.57, 1.57)

    # ✅ 你可以先用小样本确认跑通，再加大
    N_SAMPLES = 2000
    SEED = 20260201
    PROGRESS_EVERY = 200

    therm_brick = Borealis2R2C(Cm_J_per_K=6.0e8, Rim_K_per_W=1.6e-4)
    therm_concrete = Borealis2R2C(Cm_J_per_K=1.2e9, Rim_K_per_W=9.0e-5)

    building_fi = BuildingConfig(wwr_south=0.40, wwr_other=0.25)
    building_passive = BuildingConfig(wwr_south=0.45, wwr_other=0.30)

    print("\n===== Borealis runs (Scheme A: feasible min Eh) =====\n")

    rows: List[Dict[str, Any]] = []

    def run_case(name: str, building: BuildingConfig, therm: Borealis2R2C):
        # baseline metrics-only
        res0 = evaluate_borealis(df, location, building, optical_base, schedule, bo_opt, therm, bo_seasons, X_BASELINE, return_series=False)
        OH0, Hoh0, Eh0 = float(res0["OH_degC_h"]), float(res0["Hoh_h"]), float(res0["Eh_kWh"])

        OH_max, Hoh_max = _feasible_thresholds_from_baseline(OH0, Hoh0, OH_ratio=0.60, Hoh_ratio=0.60, OH_abs_floor=5.0, Hoh_abs_floor=5.0)

        best_x, best_res_metrics, info = _optimize_scheme_A_feasible_min_Eh(
            df=df, location=location, building=building, optical_base=optical_base, schedule=schedule,
            bo_opt=bo_opt, bo_therm=therm, bo_seasons=bo_seasons, bounds=bounds,
            OH_max=OH_max, Hoh_max=Hoh_max, n_samples=N_SAMPLES, seed=SEED, progress_every=PROGRESS_EVERY
        )

        # best full series (for later plotting/debug)
        best_res_full = evaluate_borealis(df, location, building, optical_base, schedule, bo_opt, therm, bo_seasons, best_x, return_series=True)

        print(f"[{name}] schemeA_feasible")
        print(f"  baseline x={_fmt_x(X_BASELINE)} Eh={Eh0:.2f} kWh, OH={OH0:.2f}, Hoh={Hoh0:.1f}")
        print(f"  constr   OH_max={OH_max:.2f}, Hoh_max={Hoh_max:.1f} ({info['status']}, feasible={info['feasible_count']})")
        print(f"  best     x={_fmt_x(best_x)} Eh={float(best_res_full['Eh_kWh']):.2f} kWh, OH={float(best_res_full['OH_degC_h']):.2f}, Hoh={float(best_res_full['Hoh_h']):.1f}\n")

        rows.append({
            "case": name,
            "mode": "schemeA_feasible",
            "OH_max": OH_max,
            "Hoh_max": Hoh_max,
            "opt_status": info["status"],
            "feasible_count": info["feasible_count"],
            "n_samples": N_SAMPLES,
            "seed": SEED,
            "x_baseline": str(X_BASELINE),
            "Eh_kWh_baseline": Eh0,
            "OH_degC_h_baseline": OH0,
            "Hoh_h_baseline": Hoh0,
            "x_best": str(best_x),
            "Eh_kWh_best": float(best_res_full["Eh_kWh"]),
            "OH_degC_h_best": float(best_res_full["OH_degC_h"]),
            "Hoh_h_best": float(best_res_full["Hoh_h"]),
        })

    run_case("FI_WWR + BRICK", building_fi, therm_brick)
    run_case("FI_WWR + CONCRETE", building_fi, therm_concrete)

    # ctrl no-shading (no optimization)
    for tag, therm in [("BRICK_THERM", therm_brick), ("CONCRETE_THERM", therm_concrete)]:
        name = f"CTRL: PASSIVE_WWR + NO_SHADING + {tag}"
        res = evaluate_borealis(df, location, building_passive, optical_base, schedule, bo_opt, therm, bo_seasons, X_BASELINE, return_series=False)
        print(f"[{name}] NO_SHADING")
        print(f"  x={_fmt_x(X_BASELINE)} Eh={float(res['Eh_kWh']):.2f} kWh, OH={float(res['OH_degC_h']):.2f}, Hoh={float(res['Hoh_h']):.1f}\n")

        rows.append({
            "case": name,
            "mode": "no_shading_ctrl",
            "OH_max": np.nan,
            "Hoh_max": np.nan,
            "opt_status": "not_optimized",
            "feasible_count": 0,
            "n_samples": 0,
            "seed": np.nan,
            "x_baseline": str(X_BASELINE),
            "Eh_kWh_baseline": float(res["Eh_kWh"]),
            "OH_degC_h_baseline": float(res["OH_degC_h"]),
            "Hoh_h_baseline": float(res["Hoh_h"]),
            "x_best": str(X_BASELINE),
            "Eh_kWh_best": float(res["Eh_kWh"]),
            "OH_degC_h_best": float(res["OH_degC_h"]),
            "Hoh_h_best": float(res["Hoh_h"]),
        })

    out_csv = ROOT / "borealis_results_summary.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"✅ Saved summary: {out_csv}")


if __name__ == "__main__":
    main()
