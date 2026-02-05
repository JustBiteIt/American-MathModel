# borealis_model/pro2_make_outputs.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


# =========================
# User knobs (你只需要改这里)
# =========================
ROOT = Path(__file__).resolve().parents[1]  # D:\ICM_CODE
OUT_DIR = Path(r"D:\ICM_RESULT") / "pro2"

WEATHER_CANDIDATES = [
    ROOT / "weather_helsinki_tmy_hourly.csv",
    Path(__file__).resolve().parent / "weather_helsinki_tmy_hourly.csv",
]

# Scheme A: 可行域约束强度（相对 baseline 的比例）
GAMMA = 0.60        # OH_max = max(OH_MIN, GAMMA * OH_baseline)
OH_MIN = 5.0
HOH_MIN = 5.0

# Scheme A: 随机搜索采样数（你电脑跑得动就加；论文更稳）
N_SAMPLES = 2000
SEED = 20260201

# 输出 PNG（给你先验算趋势；论文最终你用 ECharts）
MAKE_PNG = True
PNG_DPI = 220


# =========================
# Helpers
# =========================
def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _fmt_x(x: Tuple[float, float, float, float]) -> str:
    return f"({x[0]:.6g}, {x[1]:.6g}, {x[2]:.6g}, {x[3]:.6g})"

def _slug(s: str) -> str:
    return (
        s.replace(" ", "_")
         .replace(":", "")
         .replace("+", "plus")
         .replace("/", "_")
         .replace("__", "_")
    )

def _pairs_from_series(idx: pd.DatetimeIndex, values: np.ndarray) -> List[List[Any]]:
    # ECharts 用：[[ts_ms, val], ...]
    ts = (idx.view("int64") // 1_000_000).tolist()  # ns -> ms
    vals = np.asarray(values, dtype=float).tolist()
    return [[ts[i], vals[i]] for i in range(min(len(ts), len(vals)))]

def _monthly_sum_kwh_from_power(times: pd.DatetimeIndex, power_W: np.ndarray, dt_hours: float = 1.0) -> List[float]:
    s = pd.Series(np.asarray(power_W, dtype=float) * dt_hours / 1000.0, index=times)  # kWh per step
    m = s.resample("MS").sum()
    # 返回 12个月（TMY通常全年）
    # 若不是12个月也照样返回
    return m.to_numpy(dtype=float).tolist()

def _pick_hot_window(times: pd.DatetimeIndex, Ti: np.ndarray, days: int = 14) -> Tuple[pd.DatetimeIndex, np.ndarray]:
    """
    选一个“最热窗口”用于画 Ti 线图：
    用 rolling mean(Ti) 找出平均温度最高的连续 days 天窗口。
    """
    Ti_step = np.asarray(Ti, dtype=float)[:-1]  # (n,)
    s = pd.Series(Ti_step, index=times)
    win = days * 24
    if len(s) <= win + 1:
        return times, Ti_step
    r = s.rolling(win).mean()
    end = int(r.idxmax().value)  # ns
    end_ts = pd.Timestamp(end, tz=times.tz)
    start_ts = end_ts - pd.Timedelta(hours=win - 1)
    s2 = s.loc[start_ts:end_ts]
    return s2.index, s2.to_numpy(dtype=float)

def _save_series_csv(case_key: str, times: pd.DatetimeIndex, res: Dict[str, Any], out_dir: Path) -> Dict[str, str]:
    """
    输出两份 CSV：
    - state: Ti/Tm (n+1)
    - flux : Phi_s/Phi_h/I_other_eff/rho_g (n)
    """
    series = res["series"]
    Ti = np.asarray(series["Ti_C"], dtype=float)
    Tm = np.asarray(series["Tm_C"], dtype=float)
    Phi_s = np.asarray(series["Phi_s_W"], dtype=float)
    Phi_h = np.asarray(series["Phi_h_W"], dtype=float)

    I_other = np.asarray(series.get("I_other_eff_Wm2", np.full_like(Phi_h, np.nan)), dtype=float)
    rho_g = np.asarray(series.get("rho_g", np.full_like(Phi_h, np.nan)), dtype=float)

    n = len(times)
    times_plus1 = times.append(pd.DatetimeIndex([times[-1] + pd.Timedelta(hours=1)]))

    # state (n+1)
    df_state = pd.DataFrame({
        "datetime": times_plus1.astype(str),
        "Ti_C": Ti[: len(times_plus1)],
        "Tm_C": Tm[: len(times_plus1)],
    })
    p_state = out_dir / f"{case_key}__series_state.csv"
    df_state.to_csv(p_state, index=False, encoding="utf-8-sig")

    # flux (n)
    df_flux = pd.DataFrame({
        "datetime": times.astype(str),
        "Phi_s_W": Phi_s[:n],
        "Phi_h_W": Phi_h[:n],
        "I_other_eff_Wm2": I_other[:n],
        "rho_g": rho_g[:n],
    })
    p_flux = out_dir / f"{case_key}__series_flux.csv"
    df_flux.to_csv(p_flux, index=False, encoding="utf-8-sig")

    return {"state_csv": str(p_state), "flux_csv": str(p_flux)}


# =========================
# Scheme A optimizer:
#   min Eh  subject to OH<=OH_max and Hoh<=Hoh_max
# =========================
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
    baseline_res: Dict[str, Any],
    n_samples: int = N_SAMPLES,
    seed: int = SEED,
    gamma: float = GAMMA,
    oh_min: float = OH_MIN,
    hoh_min: float = HOH_MIN,
    log_every: int = 200,
) -> Tuple[Tuple[float, float, float, float], Dict[str, Any], Dict[str, Any]]:
    rng = np.random.default_rng(seed)

    OH0 = float(baseline_res["OH_degC_h"])
    Hoh0 = float(baseline_res["Hoh_h"])
    OH_max = max(float(oh_min), float(gamma) * OH0)
    Hoh_max = max(float(hoh_min), float(gamma) * Hoh0)

    best_x: Optional[Tuple[float, float, float, float]] = None
    best_res: Optional[Dict[str, Any]] = None
    best_Eh = float("inf")
    feasible_count = 0

    t0 = time.time()
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

        if (i % log_every) == 0:
            dt = time.time() - t0
            print(f"  ... {i}/{n_samples} done | feasible={feasible_count} | best_Eh={best_Eh} | {dt:.1f}s")

    info = {
        "OH0": OH0,
        "Hoh0": Hoh0,
        "OH_max": OH_max,
        "Hoh_max": Hoh_max,
        "feasible_count": feasible_count,
        "n_samples": n_samples,
        "seed": seed,
        "gamma": gamma,
    }

    # 若 baseline 本身可行，并且你希望允许 best=baseline（常见）
    base_feasible = (OH0 <= OH_max) and (Hoh0 <= Hoh_max)
    info["baseline_feasible"] = bool(base_feasible)

    if best_x is None or best_res is None:
        # 没找到可行点：兜底返回 baseline
        info["feasible_found"] = False
        return baseline_res["x"], baseline_res, info

    info["feasible_found"] = True
    return best_x, best_res, info


# =========================
# Main export
# =========================
def main() -> None:
    out_dir = _ensure_dir(OUT_DIR)
    fig_dir = _ensure_dir(out_dir / "fig_png")
    payload_dir = _ensure_dir(out_dir / "echarts_payload")
    series_dir = _ensure_dir(out_dir / "series_csv")

    # ---- weather ----
    weather_csv = None
    for p in WEATHER_CANDIDATES:
        if p.exists():
            weather_csv = p
            break
    if weather_csv is None:
        raise FileNotFoundError("找不到 weather_helsinki_tmy_hourly.csv（请放在 D:\\ICM_CODE\\ 下或 borealis_model 目录下）")

    location = LocationConfig()
    print(f"✅ Using weather file: {weather_csv}")
    df = load_weather_csv(str(weather_csv), tz=location.tz)

    # ---- shared configs ----
    optical_base = OpticalConfig()
    schedule = ScheduleConfig()
    bo_seasons = BorealisSeasons()
    bounds = DecisionBounds()

    # 你要求的地表反照率：bare=0.2, snow=0.7
    bo_opt = BorealisOptical(
        k_other=0.35,
        V_sky=1.0,
        V_gr=1.0,
        rho_bare=0.20,
        rho_snow=0.70,
        snow_Ta_threshold_C=0.0,
    )

    # ---- thermal presets（按你目前 main 的口径；想改就改这里）----
    therm_brick = Borealis2R2C(
        Cm_J_per_K=6.0e8,
        Rim_K_per_W=1.6e-4,
    )
    therm_concrete = Borealis2R2C(
        Cm_J_per_K=1.2e9,
        Rim_K_per_W=9.0e-5,
    )

    # ---- building presets（按你目前口径；想改 WWR 就改这里）----
    building_fi = BuildingConfig(
        wwr_south=0.40,
        wwr_other=0.25,
    )
    building_passive = BuildingConfig(
        wwr_south=0.45,
        wwr_other=0.30,
    )

    # ---- x baseline（无任何遮阳等效）----
    X_BASELINE = (0.0, 0.0, 1.57, 1.57)

    print("\n===== Borealis runs (Scheme A + exports) =====\n")

    # 要跑的 cases：两套“优化组” + 两套“对照无遮阳组（砖/混凝土）”
    case_specs = [
        ("FI_WWR + BRICK", building_fi, therm_brick, "schemeA_feasible"),
        ("FI_WWR + CONCRETE", building_fi, therm_concrete, "schemeA_feasible"),
        ("CTRL: PASSIVE_WWR + NO_SHADING + BRICK_THERM", building_passive, therm_brick, "no_shading"),
        ("CTRL: PASSIVE_WWR + NO_SHADING + CONCRETE_THERM", building_passive, therm_concrete, "no_shading"),
    ]

    summary_rows: List[Dict[str, Any]] = []
    all_payload: Dict[str, Any] = {
        "tz": location.tz,
        "meta": {
            "weather_csv": str(weather_csv),
            "GAMMA": GAMMA,
            "OH_MIN": OH_MIN,
            "HOH_MIN": HOH_MIN,
            "N_SAMPLES": N_SAMPLES,
            "SEED": SEED,
        },
        "cases": {}
    }

    # 运行并导出
    for case_name, building, therm, mode in case_specs:
        case_key_base = _slug(case_name)

        # baseline
        base_res = evaluate_borealis(
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

        # best
        if mode == "schemeA_feasible":
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
                baseline_res=base_res,
                n_samples=N_SAMPLES,
                seed=SEED,
                gamma=GAMMA,
                oh_min=OH_MIN,
                hoh_min=HOH_MIN,
            )
        else:
            best_x, best_res, info = (X_BASELINE, base_res, {"mode": "no_shading"})

        # 打印
        print(f"[{case_name}] mode={mode}")
        print(f"  baseline x={_fmt_x(X_BASELINE)} Eh={base_res['Eh_kWh']:.2f} kWh, OH={base_res['OH_degC_h']:.2f}, Hoh={base_res['Hoh_h']:.1f}")
        if mode == "schemeA_feasible":
            print(f"  constr   OH_max={info['OH_max']:.2f}, Hoh_max={info['Hoh_max']:.1f} (feasible={info['feasible_count']}, baseline_feasible={info['baseline_feasible']})")
        print(f"  best     x={_fmt_x(best_x)} Eh={best_res['Eh_kWh']:.2f} kWh, OH={best_res['OH_degC_h']:.2f}, Hoh={best_res['Hoh_h']:.1f}\n")

        # ---- save series csv (baseline/best) ----
        base_key = f"{case_key_base}__baseline"
        best_key = f"{case_key_base}__best" if mode == "schemeA_feasible" else f"{case_key_base}__no_shading"

        base_paths = _save_series_csv(base_key, df.index, base_res, series_dir)
        best_paths = _save_series_csv(best_key, df.index, best_res, series_dir)

        # ---- summary row ----
        row = {
            "case": case_name,
            "mode": mode,
            "building": {
                "wwr_south": float(building.wwr_south),
                "wwr_other": float(building.wwr_other),
            },
            "thermal": {
                "Ci_J_per_K": float(therm.Ci_J_per_K),
                "Cm_J_per_K": float(therm.Cm_J_per_K),
                "Ria_K_per_W": float(therm.Ria_K_per_W),
                "Rim_K_per_W": float(therm.Rim_K_per_W),
                "eta_air": float(therm.eta_air),
                "T_min_C": float(therm.T_min_C),
                "T_max_C": float(therm.T_max_C),
            },
            "optical_borealis": {
                "k_other": float(bo_opt.k_other),
                "rho_bare": float(bo_opt.rho_bare),
                "rho_snow": float(bo_opt.rho_snow),
                "snow_Ta_threshold_C": float(bo_opt.snow_Ta_threshold_C),
                "V_sky": float(bo_opt.V_sky),
                "V_gr": float(bo_opt.V_gr),
            },
            "baseline": {
                "x": tuple(base_res["x"]),
                "Eh_kWh": float(base_res["Eh_kWh"]),
                "OH_degC_h": float(base_res["OH_degC_h"]),
                "Hoh_h": float(base_res["Hoh_h"]),
                "series_state_csv": base_paths["state_csv"],
                "series_flux_csv": base_paths["flux_csv"],
            },
            "best": {
                "x": tuple(best_res["x"]),
                "Eh_kWh": float(best_res["Eh_kWh"]),
                "OH_degC_h": float(best_res["OH_degC_h"]),
                "Hoh_h": float(best_res["Hoh_h"]),
                "series_state_csv": best_paths["state_csv"],
                "series_flux_csv": best_paths["flux_csv"],
            },
            "constraints": {
                "OH_max": float(info.get("OH_max", np.nan)),
                "Hoh_max": float(info.get("Hoh_max", np.nan)),
                "feasible_count": int(info.get("feasible_count", 0)),
                "baseline_feasible": bool(info.get("baseline_feasible", False)) if "baseline_feasible" in info else None,
            }
        }
        summary_rows.append(row)

        # ---- payload for echarts ----
        # 只放“作图必要数据”：summary点 + （热周 Ti）+ 月度采暖
        hot_t_base, hot_Ti_base = _pick_hot_window(df.index, base_res["series"]["Ti_C"], days=14)
        hot_t_best, hot_Ti_best = _pick_hot_window(df.index, best_res["series"]["Ti_C"], days=14)

        monthly_Eh_base = _monthly_sum_kwh_from_power(df.index, base_res["series"]["Phi_h_W"], dt_hours=float(therm.dt_hours))
        monthly_Eh_best = _monthly_sum_kwh_from_power(df.index, best_res["series"]["Phi_h_W"], dt_hours=float(therm.dt_hours))

        all_payload["cases"][case_key_base] = {
            "name": case_name,
            "mode": mode,
            "baseline": {
                "x": list(base_res["x"]),
                "Eh_kWh": float(base_res["Eh_kWh"]),
                "OH_degC_h": float(base_res["OH_degC_h"]),
                "Hoh_h": float(base_res["Hoh_h"]),
                "Ti_hot_14d": _pairs_from_series(hot_t_base, hot_Ti_base),
                "monthly_Eh_kWh": monthly_Eh_base,
            },
            "best": {
                "x": list(best_res["x"]),
                "Eh_kWh": float(best_res["Eh_kWh"]),
                "OH_degC_h": float(best_res["OH_degC_h"]),
                "Hoh_h": float(best_res["Hoh_h"]),
                "Ti_hot_14d": _pairs_from_series(hot_t_best, hot_Ti_best),
                "monthly_Eh_kWh": monthly_Eh_best,
            },
            "constraints": {
                "OH_max": float(info.get("OH_max", np.nan)),
                "Hoh_max": float(info.get("Hoh_max", np.nan)),
            },
            "building": {
                "wwr_south": float(building.wwr_south),
                "wwr_other": float(building.wwr_other),
            },
            "thermal": {
                "Cm_J_per_K": float(therm.Cm_J_per_K),
                "Rim_K_per_W": float(therm.Rim_K_per_W),
                "T_max_C": float(therm.T_max_C),
                "T_min_C": float(therm.T_min_C),
            }
        }

    # ---- write summary json + flattened summary csv ----
    p_json = payload_dir / "pro2_fig_payload.json"
    p_json.write_text(json.dumps(all_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # 扁平化 summary（方便你做论文表格）
    flat_rows = []
    for r in summary_rows:
        flat_rows.append({
            "case": r["case"],
            "mode": r["mode"],
            "wwr_south": r["building"]["wwr_south"],
            "wwr_other": r["building"]["wwr_other"],
            "Cm_J_per_K": r["thermal"]["Cm_J_per_K"],
            "Rim_K_per_W": r["thermal"]["Rim_K_per_W"],
            "T_min_C": r["thermal"]["T_min_C"],
            "T_max_C": r["thermal"]["T_max_C"],
            "k_other": r["optical_borealis"]["k_other"],
            "rho_bare": r["optical_borealis"]["rho_bare"],
            "rho_snow": r["optical_borealis"]["rho_snow"],

            "baseline_x": str(r["baseline"]["x"]),
            "baseline_Eh_kWh": r["baseline"]["Eh_kWh"],
            "baseline_OH_degC_h": r["baseline"]["OH_degC_h"],
            "baseline_Hoh_h": r["baseline"]["Hoh_h"],

            "best_x": str(r["best"]["x"]),
            "best_Eh_kWh": r["best"]["Eh_kWh"],
            "best_OH_degC_h": r["best"]["OH_degC_h"],
            "best_Hoh_h": r["best"]["Hoh_h"],

            "OH_max": r["constraints"]["OH_max"],
            "Hoh_max": r["constraints"]["Hoh_max"],
            "feasible_count": r["constraints"]["feasible_count"],
            "baseline_feasible": r["constraints"]["baseline_feasible"],
        })

    p_csv = out_dir / "pro2_summary_flat.csv"
    pd.DataFrame(flat_rows).to_csv(p_csv, index=False, encoding="utf-8-sig")

    print(f"✅ Saved series CSV to: {series_dir}")
    print(f"✅ Saved flat summary CSV: {p_csv}")
    print(f"✅ Saved ECharts payload JSON: {p_json}")

    # ---- quick png figures (optional) ----
    if MAKE_PNG:
        _make_png_figures(all_payload, fig_dir)
        print(f"✅ Saved PNG figures to: {fig_dir}")

def _make_png_figures(payload: Dict[str, Any], fig_dir: Path) -> None:
    cases = payload["cases"]

    # 1) Scatter: Eh vs Hoh
    pts = []
    for key, c in cases.items():
        name = c["name"]
        for kind in ["baseline", "best"]:
            pts.append({
                "label": f"{name}__{kind}",
                "Eh": c[kind]["Eh_kWh"],
                "Hoh": c[kind]["Hoh_h"],
                "OH": c[kind]["OH_degC_h"],
            })

    fig = plt.figure()
    for p in pts:
        plt.scatter(p["Eh"], p["Hoh"])
        plt.text(p["Eh"], p["Hoh"], p["label"], fontsize=7)
    plt.xlabel("Heating energy Eh (kWh)")
    plt.ylabel("Hours overheat Hoh (h)")
    plt.title("Trade-off: Eh vs Hoh (baseline vs optimized)")
    fig.savefig(fig_dir / "fig1_scatter_Eh_vs_Hoh.png", dpi=PNG_DPI, bbox_inches="tight")
    plt.close(fig)

    # 2) Bar: Hoh baseline vs best（按 case）
    labels = list(cases.keys())
    xlab = [cases[k]["name"] for k in labels]
    H0 = [cases[k]["baseline"]["Hoh_h"] for k in labels]
    Hb = [cases[k]["best"]["Hoh_h"] for k in labels]

    fig = plt.figure(figsize=(10, 4))
    x = np.arange(len(labels))
    w = 0.38
    plt.bar(x - w/2, H0, width=w, label="baseline/no shading")
    plt.bar(x + w/2, Hb, width=w, label="optimized/best")
    plt.xticks(x, xlab, rotation=20, ha="right")
    plt.ylabel("Hoh (hours)")
    plt.title("Overheat hours Hoh: baseline vs best")
    plt.legend()
    fig.savefig(fig_dir / "fig2_bar_Hoh.png", dpi=PNG_DPI, bbox_inches="tight")
    plt.close(fig)

    # 3) Bar: OH baseline vs best
    O0 = [cases[k]["baseline"]["OH_degC_h"] for k in labels]
    Ob = [cases[k]["best"]["OH_degC_h"] for k in labels]

    fig = plt.figure(figsize=(10, 4))
    plt.bar(x - w/2, O0, width=w, label="baseline/no shading")
    plt.bar(x + w/2, Ob, width=w, label="optimized/best")
    plt.xticks(x, xlab, rotation=20, ha="right")
    plt.ylabel("OH (°C·h)")
    plt.title("Overheat severity OH: baseline vs best")
    plt.legend()
    fig.savefig(fig_dir / "fig3_bar_OH.png", dpi=PNG_DPI, bbox_inches="tight")
    plt.close(fig)

    # 4) Ti hot 14d line（每个 case 单独一张：baseline vs best）
    for key, c in cases.items():
        b = c["baseline"]["Ti_hot_14d"]
        k = c["best"]["Ti_hot_14d"]
        tb = [p[0] for p in b]
        vb = [p[1] for p in b]
        tk = [p[0] for p in k]
        vk = [p[1] for p in k]

        fig = plt.figure(figsize=(10, 4))
        plt.plot(tb, vb, label="baseline/no shading")
        plt.plot(tk, vk, label="optimized/best")
        plt.xlabel("time (ms since epoch)")
        plt.ylabel("Ti (°C)")
        plt.title(f"Indoor temperature Ti (hottest 14 days): {c['name']}")
        plt.legend()
        fig.savefig(fig_dir / f"fig4_Ti_hot14d__{_slug(c['name'])}.png", dpi=PNG_DPI, bbox_inches="tight")
        plt.close(fig)

    # 5) Monthly Eh (baseline vs best)（每个 case 单独一张）
    for key, c in cases.items():
        mb = c["baseline"]["monthly_Eh_kWh"]
        mk = c["best"]["monthly_Eh_kWh"]
        m = np.arange(1, len(mb) + 1)

        fig = plt.figure(figsize=(10, 4))
        plt.plot(m, mb, marker="o", label="baseline/no shading")
        plt.plot(m, mk, marker="o", label="optimized/best")
        plt.xticks(m, [str(i) for i in m])
        plt.xlabel("Month index")
        plt.ylabel("Monthly heating energy (kWh)")
        plt.title(f"Monthly Eh: {c['name']}")
        plt.legend()
        fig.savefig(fig_dir / f"fig5_monthly_Eh__{_slug(c['name'])}.png", dpi=PNG_DPI, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
