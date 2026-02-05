# borealis_model/main_borealis.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

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
# Local helper (avoid import error)
# -----------------------------
def month_mask_bool(times: pd.DatetimeIndex, months) -> np.ndarray:
    """Return boolean mask (n,) for given months."""
    mset = set(int(x) for x in months)
    return np.array([(int(mm) in mset) for mm in times.month], dtype=bool)


# -----------------------------
# Pretty print
# -----------------------------
def _fmt_x(x: Tuple[float, float, float, float]) -> str:
    return f"({x[0]:.6g}, {x[1]:.6g}, {x[2]:.6g}, {x[3]:.6g})"


def _summer_stats(
    df: pd.DataFrame,
    Ti_C: np.ndarray,  # (n+1,)
    seasons: BorealisSeasons,
    T_max_C: float,
) -> Dict[str, float]:
    sm = month_mask_bool(df.index, seasons.summer_months)  # (n,)
    T = np.asarray(Ti_C, dtype=float)[:-1]                # 对齐 df.index (n,)
    Ts = T[sm]
    if Ts.size == 0:
        return {"Ti_summer_max": float("nan"), "Ti_summer_min": float("nan"), "exceed_hours": 0.0}
    return {
        "Ti_summer_max": float(np.max(Ts)),
        "Ti_summer_min": float(np.min(Ts)),
        "exceed_hours": float(np.sum(Ts > float(T_max_C))),
    }


# -----------------------------
# Scheme A optimizer
#   feasible: OH<=OH_max and Hoh<=Hoh_max
#   objective: minimize Eh_kWh
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
    n_samples: int = 2000,
    seed: int = 20260131,
    progress_every: int = 200,
    oh_floor: float = 5.0,
    hoh_floor: float = 5.0,
    ratio: float = 0.6,
) -> Tuple[Tuple[float, float, float, float], Dict[str, Any], Dict[str, Any]]:
    """
    约束阈值来自 baseline：
      OH_max  = max(oh_floor,  ratio * OH_baseline)
      Hoh_max = max(hoh_floor, ratio * Hoh_baseline)

    在可行域里找 Eh 最小的 x。
    """
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
                f"best_Eh={best_Eh} | {elapsed:.1f}s"
            )

    if best_x is None or best_res is None:
        best_x = baseline_x
        best_res = baseline_res
        status = "no_feasible_found"
    else:
        status = "feasible_found"

    info = {
        "status": status,
        "baseline_feasible": baseline_feasible,
        "OH_max": OH_max,
        "Hoh_max": Hoh_max,
        "feasible_count": feasible_count,
        "n_samples": n_samples,
        "seed": seed,
    }
    return best_x, best_res, info


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ROOT = Path(__file__).resolve().parents[1]  # -> D:\ICM_CODE

    weather_csv = ROOT / "weather_helsinki_tmy_hourly.csv"
    if not weather_csv.exists():
        alt = Path(__file__).resolve().parent / "weather_helsinki_tmy_hourly.csv"
        if alt.exists():
            weather_csv = alt
        else:
            raise FileNotFoundError(f"找不到: {weather_csv}")

    print(f"✅ Using weather file: {weather_csv}")

    # ---- Load weather ----
    location = LocationConfig()
    df = load_weather_csv(str(weather_csv), tz=location.tz)

    # ---- Shared configs ----
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

    therm_brick = Borealis2R2C(
        Cm_J_per_K=6.0e8,
        Rim_K_per_W=1.6e-4,
    )
    therm_concrete = Borealis2R2C(
        Cm_J_per_K=1.2e9,
        Rim_K_per_W=9.0e-5,
    )

    building_fi = BuildingConfig(
        wwr_south=0.40,
        wwr_other=0.25,
    )
    building_passive = BuildingConfig(
        wwr_south=0.45,
        wwr_other=0.30,
    )

    X_BASELINE = (0.0, 0.0, 1.57, 1.57)

    N_SAMPLES = 2000
    SEED = 20260131
    PROGRESS_EVERY = 200
    DEBUG_SUMMER = True

    print("\n===== Borealis runs (Scheme A: feasible min Eh) =====\n")

    rows: List[Dict[str, Any]] = []

    def run_case(case_name: str, building: BuildingConfig, therm: Borealis2R2C, *, mode: str) -> None:
        nonlocal rows

        base = evaluate_borealis(
            df=df,
            location=location,
            building=building,
            optical_base=optical_base,
            schedule=schedule,
            bo_optical=bo_opt,
            bo_therm=therm,
            bo_seasons=bo_seasons,
            x=X_BASELINE,
        )

        best_x, best_res, info = _optimize_scheme_A_feasible_min_Eh(
            df=df,
            location=location,
            building=building,
            optical_base=optical_base,
            schedule=schedule,
            bo_opt=bo_opt,
            bo_therm=therm,
            bo_seasons=bo_seasons,
            bounds=bounds,
            baseline_x=X_BASELINE,
            baseline_res=base,
            n_samples=N_SAMPLES,
            seed=SEED,
            progress_every=PROGRESS_EVERY,
            oh_floor=5.0,
            hoh_floor=5.0,
            ratio=0.6,
        )

        print(f"[{case_name}] {mode}")
        print(
            f"  baseline x={_fmt_x(X_BASELINE)} "
            f"Eh={base['Eh_kWh']:.2f} kWh, OH={base['OH_degC_h']:.2f}, Hoh={base['Hoh_h']:.1f}"
        )
        print(
            f"  constr   OH_max={info['OH_max']:.2f}, Hoh_max={info['Hoh_max']:.1f} "
            f"({info['status']}, baseline_feasible={info['baseline_feasible']}, feasible={info['feasible_count']})"
        )
        print(
            f"  best     x={_fmt_x(best_x)} "
            f"Eh={best_res['Eh_kWh']:.2f} kWh, OH={best_res['OH_degC_h']:.2f}, Hoh={best_res['Hoh_h']:.1f}\n"
        )

        if DEBUG_SUMMER:
            s0 = _summer_stats(df, base["series"]["Ti_C"], bo_seasons, therm.T_max_C)
            s1 = _summer_stats(df, best_res["series"]["Ti_C"], bo_seasons, therm.T_max_C)
            print(
                f"  [debug] baseline summer: Ti_max={s0['Ti_summer_max']:.2f}C, exceed_hours={s0['exceed_hours']:.0f}"
            )
            print(
                f"  [debug] best     summer: Ti_max={s1['Ti_summer_max']:.2f}C, exceed_hours={s1['exceed_hours']:.0f}\n"
            )

        rows.append(
            {
                "case": case_name,
                "mode": mode,
                "baseline_x": str(X_BASELINE),
                "best_x": str(best_x),
                "Eh_kWh_baseline": float(base["Eh_kWh"]),
                "OH_degC_h_baseline": float(base["OH_degC_h"]),
                "Hoh_h_baseline": float(base["Hoh_h"]),
                "Eh_kWh_best": float(best_res["Eh_kWh"]),
                "OH_degC_h_best": float(best_res["OH_degC_h"]),
                "Hoh_h_best": float(best_res["Hoh_h"]),
                "OH_max": float(info["OH_max"]),
                "Hoh_max": float(info["Hoh_max"]),
                "baseline_feasible": bool(info["baseline_feasible"]),
                "feasible_count": int(info["feasible_count"]),
                "n_samples": int(info["n_samples"]),
                "seed": int(info["seed"]),
                "wwr_south": float(getattr(building, "wwr_south", np.nan)),
                "wwr_other": float(getattr(building, "wwr_other", np.nan)),
                "Cm_J_per_K": float(therm.Cm_J_per_K),
                "Rim_K_per_W": float(therm.Rim_K_per_W),
                "T_max_C": float(therm.T_max_C),
                "T_min_C": float(therm.T_min_C),
                "k_other": float(bo_opt.k_other),
                "rho_bare": float(bo_opt.rho_bare),
                "rho_snow": float(bo_opt.rho_snow),
            }
        )

    run_case("FI_WWR + BRICK", building_fi, therm_brick, mode="schemeA_feasible")
    run_case("FI_WWR + CONCRETE", building_fi, therm_concrete, mode="schemeA_feasible")

    def run_ctrl(case_name: str, building: BuildingConfig, therm: Borealis2R2C) -> None:
        res = evaluate_borealis(
            df=df,
            location=location,
            building=building,
            optical_base=optical_base,
            schedule=schedule,
            bo_optical=bo_opt,
            bo_therm=therm,
            bo_seasons=bo_seasons,
            x=X_BASELINE,
        )
        print(f"[{case_name}] NO_SHADING")
        print(
            f"  x={_fmt_x(X_BASELINE)} "
            f"Eh={res['Eh_kWh']:.2f} kWh, OH={res['OH_degC_h']:.2f}, Hoh={res['Hoh_h']:.1f}\n"
        )

        if DEBUG_SUMMER:
            s = _summer_stats(df, res["series"]["Ti_C"], bo_seasons, therm.T_max_C)
            print(f"  [debug] ctrl summer: Ti_max={s['Ti_summer_max']:.2f}C, exceed_hours={s['exceed_hours']:.0f}\n")

        rows.append(
            {
                "case": case_name,
                "mode": "no_shading",
                "baseline_x": str(X_BASELINE),
                "best_x": str(X_BASELINE),
                "Eh_kWh_baseline": float(res["Eh_kWh"]),
                "OH_degC_h_baseline": float(res["OH_degC_h"]),
                "Hoh_h_baseline": float(res["Hoh_h"]),
                "Eh_kWh_best": float(res["Eh_kWh"]),
                "OH_degC_h_best": float(res["OH_degC_h"]),
                "Hoh_h_best": float(res["Hoh_h"]),
                "OH_max": np.nan,
                "Hoh_max": np.nan,
                "baseline_feasible": np.nan,
                "feasible_count": np.nan,
                "n_samples": 0,
                "seed": np.nan,
                "wwr_south": float(getattr(building, "wwr_south", np.nan)),
                "wwr_other": float(getattr(building, "wwr_other", np.nan)),
                "Cm_J_per_K": float(therm.Cm_J_per_K),
                "Rim_K_per_W": float(therm.Rim_K_per_W),
                "T_max_C": float(therm.T_max_C),
                "T_min_C": float(therm.T_min_C),
                "k_other": float(bo_opt.k_other),
                "rho_bare": float(bo_opt.rho_bare),
                "rho_snow": float(bo_opt.rho_snow),
            }
        )

    run_ctrl("CTRL: PASSIVE_WWR + NO_SHADING + BRICK_THERM", building_passive, therm_brick)
    run_ctrl("CTRL: PASSIVE_WWR + NO_SHADING + CONCRETE_THERM", building_passive, therm_concrete)

    out_csv = ROOT / "borealis_results_summary.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"✅ Saved summary: {out_csv}")


if __name__ == "__main__":
    main()
