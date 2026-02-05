# HVAC_Pareto/config.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict

# Reuse Borealis base configs as much as possible
from borealis_model.config import (
    LocationConfig,
    BuildingConfig,
    OpticalConfig,
    ScheduleConfig,
    BorealisOptical,
    DecisionBounds,
)

# -------------------------
# Project / data locations
# -------------------------
# D:\ICM_CODE\HVAC_Pareto\config.py -> project root is D:\ICM_CODE
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

# ✅ You store CSVs in PROJECT_ROOT (D:\ICM_CODE). Keep this as default.
# Still supports a future subfolder convention via resolve_weather_csv() fallback.
WEATHER_DIR: Path = PROJECT_ROOT

# All HVAC_Pareto outputs must live under this folder.
OUTPUTS_DIR: Path = Path(__file__).resolve().parent / "hvac_outputs"


# -------------------------
# Site configuration
# -------------------------
@dataclass(frozen=True)
class SiteConfig:
    """All site-dependent knobs in one place.
    Edit DEFAULT_SITE or entries in SITES to switch locations lazily.
    """
    name: str
    latitude: float
    longitude: float
    tz: str
    # Can be: absolute path, relative path, or just a filename.
    weather_csv: str


# Pick the site name you want to run by default (when you just click ▶ in VSCode)
DEFAULT_SITE: str = "borealis"

# Add as many as you want. Since your CSVs are in PROJECT_ROOT,
# it's recommended to set weather_csv as a filename only.
SITES: Dict[str, SiteConfig] = {
    "borealis": SiteConfig(
        name="borealis",
        latitude=60.1699,
        longitude=24.9384,
        tz="Europe/Helsinki",
        weather_csv="weather_helsinki_tmy_hourly.csv",
    ),
    "singapore": SiteConfig(
        name="singapore",
        latitude=1.3521,
        longitude=103.8198,
        tz="Asia/Singapore",
        weather_csv="weather_singapore_hourly.csv",
    ),
    # If you don't have phoenix csv yet, leave it commented or point to an existing file.
    # "sungrove": SiteConfig(
    #     name="sungrove",
    #     latitude=33.4484,
    #     longitude=-112.0740,
    #     tz="America/Phoenix",
    #     weather_csv="weather_phoenix_tmy_hourly.csv",
    # ),
}


def get_active_site() -> SiteConfig:
    """Returns the default site. Change DEFAULT_SITE to switch."""
    if DEFAULT_SITE not in SITES:
        raise KeyError(f"DEFAULT_SITE={DEFAULT_SITE!r} not found in SITES keys={list(SITES.keys())}")
    return SITES[DEFAULT_SITE]


def resolve_weather_csv(site: SiteConfig) -> str:
    """Resolve weather path robustly for your 'CSV in project root' layout.

    Accepted site.weather_csv forms:
      - absolute path: used directly
      - relative path: resolved against PROJECT_ROOT
      - filename only: searched in PROJECT_ROOT first, then PROJECT_ROOT/weather
    """
    raw = Path(site.weather_csv)

    # 1) Absolute path
    if raw.is_absolute():
        if raw.exists():
            return str(raw)
        raise FileNotFoundError(f"[HVAC_Pareto] weather_csv absolute path not found: {raw}")

    candidates = []

    # 2) If it's a relative path like "data/xxx.csv"
    candidates.append((PROJECT_ROOT / raw).resolve())

    # 3) Filename-only common case: look in PROJECT_ROOT
    candidates.append((PROJECT_ROOT / raw.name).resolve())

    # 4) Backward-compatible: look in PROJECT_ROOT/weather
    candidates.append((PROJECT_ROOT / "weather" / raw.name).resolve())

    for p in candidates:
        if p.exists():
            return str(p)

    msg = (
        f"[HVAC_Pareto] Weather CSV not found for site={site.name!r}.\n"
        f"  site.weather_csv = {site.weather_csv!r}\n"
        "  Tried:\n"
        + "".join([f"    - {c}\n" for c in candidates])
        + "  Fix: put the CSV in D:\\ICM_CODE (project root), or update SITES[...].weather_csv to the correct path.\n"
    )
    raise FileNotFoundError(msg)


def resolve_output_dir(out_dir: str | None) -> Path:
    """Resolve output directory under HVAC_Pareto only."""
    base = Path(__file__).resolve().parent
    if out_dir is None or str(out_dir).strip() == "":
        return OUTPUTS_DIR

    p = Path(out_dir)
    if p.is_absolute():
        try:
            p.relative_to(base)
        except ValueError as exc:
            raise ValueError(
                f"[HVAC_Pareto] out_dir must be inside {base}. Got: {p}"
            ) from exc
        return p

    # Force relative paths to live under HVAC_Pareto
    return (base / p).resolve()


# -------------------------
# Optional CLI overrides
# -------------------------
@dataclass(frozen=True)
class LocationOverrides:
    """Optional overrides to switch site without touching code."""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    tz: Optional[str] = None


def make_location(base: LocationConfig = LocationConfig(), overrides: LocationOverrides = LocationOverrides()) -> LocationConfig:
    """Create a LocationConfig with optional overrides (lat/lon/tz)."""
    return LocationConfig(
        latitude=float(overrides.latitude) if overrides.latitude is not None else base.latitude,
        longitude=float(overrides.longitude) if overrides.longitude is not None else base.longitude,
        tz=str(overrides.tz) if overrides.tz is not None else base.tz,
    )


# -------------------------
# HVAC / thermal model config
# -------------------------
@dataclass(frozen=True)
class HVAC2R2C:
    """2R2C + ideal deadband HVAC (heat/cool) config."""
    Ci_J_per_K: float = 2.0e8
    Cm_J_per_K: float = 8.0e8
    Ria_K_per_W: float = 2.5e-4
    Rim_K_per_W: float = 1.0e-4
    eta_air: float = 0.35

    # Deadband setpoints (docx: 18-26 °C)
    T_heat_C: float = 18.0
    T_cool_C: float = 26.0

    T_init_C: float = 21.0
    dt_hours: float = 1.0


# -------------------------
# Model config (single knob set)
# -------------------------
@dataclass(frozen=True)
class ModelConfig:
    """All knobs in one place."""
    location: LocationConfig = LocationConfig()

    # ✅ legacy passive_shading WWR style (South separate, others same)
    building: BuildingConfig = BuildingConfig(
        wwr_south=0.45,
        wwr_other=0.30,
        wwr_by_facade=None,  # ensure we do NOT use dict WWR schemes
    )

    optical: OpticalConfig = OpticalConfig()
    schedule: ScheduleConfig = ScheduleConfig()
    optics_borealis: BorealisOptical = BorealisOptical()
    bounds: DecisionBounds = DecisionBounds()

    hvac_2r2c: HVAC2R2C = HVAC2R2C()
