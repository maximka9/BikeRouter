"""Погодный контекст: Open-Meteo, кэш, множители для весов рёбер."""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import requests

from .policy_data import load_weather_policy

logger = logging.getLogger(__name__)

_OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
_OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"

_CACHE_LOCK = threading.Lock()
_CACHE: Dict[str, Tuple[float, "WeatherSnapshot"]] = {}
_CACHE_TTL_SEC = 900.0


@dataclass
class WeatherSnapshot:
    """Нормализованный снимок погоды для маршрутизации."""

    temperature_c: float = 20.0
    apparent_temperature_c: Optional[float] = None
    precipitation_mm: float = 0.0
    precipitation_probability: Optional[float] = None
    wind_speed_ms: float = 3.0
    wind_gusts_ms: Optional[float] = None
    cloud_cover_pct: float = 50.0
    humidity_pct: float = 60.0
    shortwave_radiation_wm2: Optional[float] = None


@dataclass
class WeatherMultipliers:
    """Множители к компонентам стоимости ребра."""

    physical: float = 1.0
    heat: float = 1.0
    green: float = 1.0
    stress: float = 1.0
    surface: float = 1.0


@dataclass
class WeatherWeightParams:
    """Параметры применения погоды к весам (передаётся в RouteService)."""

    mults: WeatherMultipliers = field(default_factory=WeatherMultipliers)
    green_coupling: float = 0.55
    enabled: bool = False


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def snapshot_from_manual(
    *,
    temperature_c: Optional[float] = None,
    precipitation_mm: Optional[float] = None,
    wind_speed_ms: Optional[float] = None,
    cloud_cover_pct: Optional[float] = None,
    humidity_pct: Optional[float] = None,
    apparent_temperature_c: Optional[float] = None,
) -> WeatherSnapshot:
    return WeatherSnapshot(
        temperature_c=float(temperature_c if temperature_c is not None else 20.0),
        apparent_temperature_c=apparent_temperature_c,
        precipitation_mm=float(precipitation_mm if precipitation_mm is not None else 0.0),
        wind_speed_ms=float(wind_speed_ms if wind_speed_ms is not None else 3.0),
        cloud_cover_pct=float(cloud_cover_pct if cloud_cover_pct is not None else 50.0),
        humidity_pct=float(humidity_pct if humidity_pct is not None else 60.0),
    )


def build_weather_weight_params(
    snap: WeatherSnapshot,
    *,
    enabled: bool,
    policy: Optional[Dict[str, Any]] = None,
) -> WeatherWeightParams:
    """Собрать параметры для RouteService из снимка погоды."""
    if not enabled:
        return WeatherWeightParams(enabled=False)
    pol = policy if policy is not None else load_weather_policy()
    gcc = float((pol or {}).get("green_edge_coupling", 0.55))
    mults = compute_weather_multipliers(snap, policy=pol)
    return WeatherWeightParams(
        enabled=True, mults=mults, green_coupling=gcc
    )


def compute_weather_multipliers(
    snap: WeatherSnapshot,
    *,
    policy: Optional[Dict[str, Any]] = None,
) -> WeatherMultipliers:
    """Вычислить множители по политике и снимку погоды."""
    pol = policy if policy is not None else load_weather_policy()
    if not pol:
        return WeatherMultipliers()

    m = pol.get("multipliers") or {}
    tc = pol.get("temperature_c") or {}
    pr = pol.get("precipitation_mm_h") or {}
    wn = pol.get("wind_ms") or {}
    cc = pol.get("cloud_cover_pct") or {}
    hu = pol.get("humidity_pct") or {}

    T = snap.temperature_c
    rain = snap.precipitation_mm
    wind = snap.wind_speed_ms
    clouds = snap.cloud_cover_pct
    hum = snap.humidity_pct

    ref_heat = 22.0
    mp = m.get("physical") or {}
    mh = m.get("heat") or {}
    mg = m.get("green") or {}
    ms = m.get("stress") or {}
    msv = m.get("surface") or {}

    # physical
    phys = float(mp.get("base", 1.0))
    cold_b = float(tc.get("cold_below", 5))
    hot_a = float(tc.get("hot_above", 30))
    if T < cold_b:
        phys += float(mp.get("per_temp_cold", 0.008)) * (cold_b - T)
    if T > hot_a:
        phys += float(mp.get("per_temp_hot", 0.012)) * (T - hot_a)
    dry_max = float(pr.get("dry_max", 0.1))
    light_max = float(pr.get("light_max", 1.0))
    heavy = float(pr.get("heavy_above", 2.0))
    if rain > dry_max:
        phys *= float(mp.get("rain_light" if rain < light_max else "rain_heavy", 1.06))
    calm = float(wn.get("calm_max", 4))
    strong = float(wn.get("strong_above", 12))
    if wind > strong:
        phys *= 1.0 + float(mp.get("wind_headwind_proxy", 0.04)) * min(2.0, (wind - strong) / strong)
    if hum > float(hu.get("humid_above", 80)):
        phys *= float(mp.get("humid_surface", 1.05))

    # heat (thermal discomfort routing component)
    heat = float(mh.get("base", 1.0))
    if T > ref_heat:
        heat += float(mh.get("per_temp_above_22", 0.022)) * (T - ref_heat)
    heat *= max(0.65, 1.0 - float(mh.get("cloud_reduction_per_pct", 0.0022)) * clouds)
    if rain > dry_max:
        heat *= float(
            mh.get(
                "rain_reduction_heavy" if rain >= heavy else "rain_reduction_light",
                0.9,
            )
        )

    # green comfort multiplier (applied with trees_pct on edge)
    green = float(mg.get("base", 1.0))
    warm_min = float(tc.get("warm_min", 25))
    clear_max = float(cc.get("clear_max", 30))
    overcast = float(cc.get("overcast_min", 70))
    if T >= warm_min and clouds <= clear_max and rain <= dry_max:
        green *= float(mg.get("hot_sunny_bonus", 1.18))
    elif T < float(tc.get("cool_max", 15)) or clouds >= overcast:
        green *= float(mg.get("cool_cloud_penalty", 0.92))
    if rain > light_max:
        green *= float(mg.get("rain_neutralize", 0.72))

    # stress
    stress = float(ms.get("base", 1.0))
    if rain > dry_max:
        stress *= float(ms.get("rain", 1.12))
    if wind > strong:
        stress *= float(ms.get("wind_strong", 1.1))
    if clouds > float(cc.get("overcast_min", 70)) and rain > dry_max * 2:
        stress *= float(ms.get("low_visibility_cloud", 1.05))

    # surface (merged with physical in routing; separate knob)
    surface = float(msv.get("base", 1.0))
    if rain > dry_max:
        surface *= float(msv.get("rain", 1.08))
    if hum > float(hu.get("humid_above", 80)):
        surface *= float(msv.get("humid", 1.04))

    # wet + high LTS roads — approximate: bump stress slightly if rain and wind
    if rain > light_max and wind > float(wn.get("breeze_max", 8)):
        stress *= float(ms.get("wet_and_fast_road", 1.08))

    return WeatherMultipliers(
        physical=_clamp(phys, 0.75, 1.45),
        heat=_clamp(heat, 0.55, 1.75),
        green=_clamp(green, 0.6, 1.35),
        stress=_clamp(stress, 0.85, 1.45),
        surface=_clamp(surface, 0.9, 1.25),
    )


def _cache_key(lat: float, lon: float, iso_hour: str, historical: bool) -> str:
    return f"{round(lat, 3)}:{round(lon, 3)}:{iso_hour[:13]}:{'a' if historical else 'f'}"


def _weather_target_in_past(when_iso: str) -> bool:
    """True, если момент *when_iso* уже наступил — для выбора Archive API вместо Forecast."""
    from datetime import datetime

    raw = str(when_iso).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return False
    if dt.tzinfo is not None:
        return dt < datetime.now(dt.tzinfo)
    return dt < datetime.now()


def fetch_open_meteo_hourly(
    lat: float,
    lon: float,
    when_iso: str,
    *,
    historical: bool = False,
) -> WeatherSnapshot:
    """Почасовая погода Open-Meteo. *when_iso* — ISO (локальное или с offset)."""
    from datetime import datetime, timedelta, timezone

    def _as_utc(dt: datetime) -> datetime:
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    raw = str(when_iso).strip().replace("Z", "+00:00")
    try:
        dt = _as_utc(datetime.fromisoformat(raw))
    except ValueError:
        dt = datetime.now(timezone.utc)
    dstr = dt.date().isoformat()

    url = _OPEN_METEO_ARCHIVE if historical else _OPEN_METEO_FORECAST
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(
            [
                "temperature_2m",
                "apparent_temperature",
                "precipitation",
                "precipitation_probability",
                "cloud_cover",
                "relative_humidity_2m",
                "wind_speed_10m",
                "wind_gusts_10m",
                "shortwave_radiation",
            ]
        ),
        "start_date": dstr,
        "end_date": dstr,
        "timezone": "auto",
        # Явно м/с: иначе по умолчанию км/ч (документация Forecast/Archive API).
        "wind_speed_unit": "ms",
    }
    try:
        r = requests.get(url, params=params, timeout=25)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        logger.warning("Open-Meteo запрос не удался: %s", exc)
        return WeatherSnapshot()

    hourly = data.get("hourly") or {}
    times = hourly.get("time") or []
    if not times:
        return WeatherSnapshot()

    # Open-Meteo отдаёт time без tz — в локальном поясe точки; utc_offset_seconds в корне ответа.
    offset_sec = int(data.get("utc_offset_seconds") or 0)
    local_tz = timezone(timedelta(seconds=offset_sec))

    # ближайший почасовой слот: сравниваем всё в UTC ( aware vs naive из API).
    target_h = dt.replace(minute=0, second=0, microsecond=0)
    best_i = 0
    best_d = 1e9
    for i, ts in enumerate(times):
        try:
            tsi = datetime.fromisoformat(str(ts))
        except ValueError:
            continue
        if tsi.tzinfo is not None:
            tsi_utc = tsi.astimezone(timezone.utc)
        else:
            tsi_utc = tsi.replace(tzinfo=local_tz).astimezone(timezone.utc)
        d = abs((tsi_utc - target_h).total_seconds())
        if d < best_d:
            best_d = d
            best_i = i

    def _arr(name: str, default: float = 0.0) -> float:
        arr = hourly.get(name)
        if not arr or best_i >= len(arr):
            return default
        v = arr[best_i]
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return default
        return float(v)

    w_raw = _arr("wind_speed_10m", 10.0)
    # Параметр wind_speed_unit=ms — значения в м/с (по умолчанию у API — км/ч).
    wind_ms = max(0.0, w_raw)

    app_t = None
    apt_arr = hourly.get("apparent_temperature")
    if apt_arr and best_i < len(apt_arr) and apt_arr[best_i] is not None:
        app_t = float(apt_arr[best_i])

    pr_arr = hourly.get("precipitation_probability")
    pr_prob = None
    if pr_arr and best_i < len(pr_arr) and pr_arr[best_i] is not None:
        pr_prob = float(pr_arr[best_i])

    gust_arr = hourly.get("wind_gusts_10m")
    gust = None
    if gust_arr and best_i < len(gust_arr) and gust_arr[best_i] is not None:
        gust = max(0.0, float(gust_arr[best_i]))

    sw_arr = hourly.get("shortwave_radiation")
    sw = None
    if sw_arr and best_i < len(sw_arr) and sw_arr[best_i] is not None:
        sw = float(sw_arr[best_i])

    return WeatherSnapshot(
        temperature_c=_arr("temperature_2m", 20.0),
        apparent_temperature_c=app_t,
        precipitation_mm=max(0.0, _arr("precipitation", 0.0)),
        precipitation_probability=pr_prob,
        wind_speed_ms=wind_ms,
        wind_gusts_ms=gust,
        cloud_cover_pct=_clamp(_arr("cloud_cover", 50.0), 0.0, 100.0),
        humidity_pct=_clamp(_arr("relative_humidity_2m", 60.0), 0.0, 100.0),
        shortwave_radiation_wm2=sw,
    )


def resolve_weather_for_route(
    *,
    lat: float,
    lon: float,
    weather_mode: str,
    use_live_weather: bool,
    weather_time_iso: Optional[str],
    departure_time: Optional[str],
    manual: Optional[WeatherSnapshot] = None,
) -> Tuple[WeatherSnapshot, str, WeatherWeightParams]:
    """Вернуть снимок погоды, строку источника и параметры весов."""
    mode = (weather_mode or "none").strip().lower()
    if mode in ("none", "off", ""):
        snap = WeatherSnapshot()
        return snap, "none", WeatherWeightParams(enabled=False)
    if mode in ("fixed-snapshot", "fixed_snapshot") and manual is None:
        snap = WeatherSnapshot()
        return snap, "fixed_snapshot_unset", WeatherWeightParams(enabled=False)

    when = weather_time_iso or departure_time
    if not when:
        from datetime import datetime

        when = datetime.now().isoformat(timespec="seconds")

    if mode in ("manual", "fixed-snapshot", "fixed_snapshot") and manual is not None:
        snap = manual
        src = "fixed_snapshot" if "fixed" in mode else "manual"
    elif mode == "auto" or use_live_weather or mode == "live":
        use_archive = _weather_target_in_past(when)
        key = _cache_key(lat, lon, when, use_archive)
        now = time.time()
        with _CACHE_LOCK:
            ent = _CACHE.get(key)
            if ent and now - ent[0] < _CACHE_TTL_SEC:
                snap = ent[1]
                src = "open_meteo_cache"
            else:
                snap = fetch_open_meteo_hourly(
                    lat, lon, when, historical=use_archive
                )
                _CACHE[key] = (now, snap)
                src = "open_meteo_archive" if use_archive else "open_meteo"
    else:
        snap = manual if manual is not None else WeatherSnapshot()
        src = "manual" if manual is not None else "default"

    enabled = mode not in ("none", "off", "")
    return snap, src, build_weather_weight_params(snap, enabled=enabled)
