from __future__ import annotations

import json
import ssl
import time as _time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen, Request

import pandas as pd

PVGIS_TMY_URL = "https://re.jrc.ec.europa.eu/api/tmy"


@dataclass(frozen=True)
class PVGISTmyRequest:
    lat: float
    lon: float
    usehorizon: int = 1
    outputformat: str = "json"
    startyear: Optional[int] = None
    endyear: Optional[int] = None
    raddatabase: Optional[str] = None

    def to_params(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "lat": self.lat,
            "lon": self.lon,
            "outputformat": self.outputformat,
            "usehorizon": self.usehorizon,
        }
        if self.startyear is not None:
            params["startyear"] = int(self.startyear)
        if self.endyear is not None:
            params["endyear"] = int(self.endyear)
        if self.raddatabase:
            params["raddatabase"] = str(self.raddatabase)
        return params


def _parse_pvgis_time(t: str) -> str:
    s = str(t).strip()
    s = s.replace("T", " ").replace("UTC", "").replace("Z", "").strip()
    for fmt in ("%Y%m%d:%H%M", "%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
            return dt.isoformat().replace("T", " ")
        except ValueError:
            continue
    try:
        dt = pd.to_datetime(s, utc=True)
        if hasattr(dt, "to_pydatetime"):
            dt = dt.to_pydatetime()
        return dt.isoformat().replace("T", " ")
    except Exception as exc:
        raise ValueError(f"Unrecognized PVGIS time format: {t!r}") from exc


def _extract_series(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        raise ValueError("PVGIS response contains empty tmy_hourly data")
    if not isinstance(rows[0], dict):
        raise ValueError("PVGIS tmy_hourly data is not a list of records (dict).")

    keys = list(rows[0].keys())
    time_key = None
    for k in keys:
        if "time" in k.lower():
            time_key = k
            break
    if time_key is None:
        raise ValueError(f"PVGIS record missing time field. Keys={keys}")

    out_rows = []
    for r in rows:
        time_str = r.get(time_key)
        if not time_str:
            raise ValueError(f"PVGIS record missing time field for key={time_key!r}")
        out_rows.append(
            {
                "datetime": _parse_pvgis_time(time_str),
                "DNI": r.get("Gb(n)"),
                "DHI": r.get("Gd(h)"),
                "GHI": r.get("G(h)"),
                "T_out": r.get("T2m"),
            }
        )

    df = pd.DataFrame(out_rows)
    for c in ["DNI", "DHI", "GHI", "T_out"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["DNI", "DHI", "GHI"]:
        df[c] = df[c].fillna(0.0).clip(lower=0.0)
    df = df.dropna(subset=["T_out"])
    return df


def fetch_pvgis_tmy_to_csv(
    *,
    lat: float,
    lon: float,
    out_csv: Path,
    overwrite: bool = False,
    timeout_s: int = 60,
    req: Optional[PVGISTmyRequest] = None,
) -> Dict[str, Any]:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if out_csv.exists() and not overwrite:
        row_count = max(0, sum(1 for _ in out_csv.open("r", encoding="utf-8")) - 1)
        return {"path": str(out_csv), "cached": True, "rows": row_count, "url": None}

    req = req or PVGISTmyRequest(lat=lat, lon=lon)
    params = req.to_params()
    params["lat"] = float(lat)
    params["lon"] = float(lon)
    url = f"{PVGIS_TMY_URL}?{urlencode(params)}"

    request = Request(url, headers={"User-Agent": "SG_Pareto/1.0"})
    last_err = None
    payload = None
    for attempt in range(3):
        try:
            with urlopen(request, timeout=timeout_s) as resp:
                payload = resp.read().decode("utf-8")
            last_err = None
            break
        except (URLError, ssl.SSLError) as exc:
            last_err = exc
            if attempt < 2:
                _time.sleep(1.5 * (attempt + 1))
                continue
            raise ValueError(f"PVGIS 请求失败：{exc}") from exc
    if last_err is not None or payload is None:
        raise ValueError(f"PVGIS 请求失败：{last_err}")

    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError("PVGIS 响应不是有效 JSON，无法解析。") from exc

    try:
        rows = data["outputs"]["tmy_hourly"]
    except KeyError as exc:
        raise ValueError("PVGIS 响应缺少 outputs.tmy_hourly 字段。") from exc

    df = _extract_series(rows)
    if len(df) != 8760:
        raise ValueError(f"PVGIS 返回行数异常：{len(df)} 行（期望 8760 行）")

    df.to_csv(out_csv, index=False)
    return {"path": str(out_csv), "cached": False, "rows": int(len(df)), "url": url}
