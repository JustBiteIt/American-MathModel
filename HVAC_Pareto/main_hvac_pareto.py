# HVAC_Pareto/main_hvac_pareto.py
from __future__ import annotations

# Allow running by clicking VSCode ▶ on this file:
# ensure project root (D:\ICM_CODE) is on sys.path so `borealis_model` is importable.
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, Any

from HVAC_Pareto.config import (
    ModelConfig,
    LocationOverrides,
    make_location,
    get_active_site,
    resolve_weather_csv,
    resolve_output_dir,
)
from HVAC_Pareto.case_runner import run_case_from_weather_csv


def run_pareto(
    weather_csv: str | None = None,
    *,
    config: ModelConfig = ModelConfig(),
    location_overrides: LocationOverrides = LocationOverrides(),
    n_samples: int = 2000,
    seed: int = 42,
    out_dir: str | None = None,
) -> Dict[str, Any]:
    """
    Full pipeline:
      weather -> irradiance -> solar gain -> 2R2C -> HVAC heat/cool -> Pareto front

    If weather_csv is None, it is taken from HVAC_Pareto/config.py (DEFAULT_SITE).
    """
    out_path = resolve_output_dir(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Site defaults (lazy mode: edit config.py only)
    site = get_active_site()
    if weather_csv is None:
        weather_csv = resolve_weather_csv(site)

    # location (configurable)
    base_loc = make_location(config.location, location_overrides)

    # If user didn't override lat/lon/tz, use site values
    loc_over = LocationOverrides(
        latitude=location_overrides.latitude if location_overrides.latitude is not None else site.latitude,
        longitude=location_overrides.longitude if location_overrides.longitude is not None else site.longitude,
        tz=location_overrides.tz if location_overrides.tz is not None else site.tz,
    )
    location = make_location(base_loc, loc_over)

    def log_cn(msg: str) -> None:
        print(f"[HVAC_Pareto] {msg}")

    log_cn(f"输出目录：{out_path}")
    log_cn(f"站点：{site.name}  weather_csv：{weather_csv}")

    result = run_case_from_weather_csv(
        weather_csv=weather_csv,
        location=location,
        config=config,
        n_samples=n_samples,
        seed=seed,
        out_dir=out_path,
        case_name=site.name,
        logger=log_cn,
    )

    log_cn(f"完成：{site.name}")
    return result


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    # weather is optional now (lazy config mode)
    p.add_argument("--weather", type=str, default=None, help="Path to weather csv (optional; defaults from config.py)")
    p.add_argument("--lat", type=float, default=None)
    p.add_argument("--lon", type=float, default=None)
    p.add_argument("--tz", type=str, default=None)
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default=None, help="Output subfolder under HVAC_Pareto (optional)")
    args = p.parse_args()

    cfg = ModelConfig()
    loc_over = LocationOverrides(latitude=args.lat, longitude=args.lon, tz=args.tz)
    run_pareto(args.weather, config=cfg, location_overrides=loc_over, n_samples=args.n, seed=args.seed, out_dir=args.out)
