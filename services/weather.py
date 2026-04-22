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
    # Устар.: только логи/аналитика (classify_weather_regime), не ветвление стоимости ребра.
    regime: str = "neutral"
    hot_tree_bonus_scale: float = 1.0
    hot_open_penalty_scale: float = 1.0
    cold_canyon_bonus_scale: float = 1.0
    cold_tree_damping: float = 0.5
    weather_response_scale: float = 1.0
    # Непрерывная тепло-микроклиматическая модель (см. routing.effective_edge_components).
    heat_continuous: bool = False
    tree_shade_bonus: float = 1.0
    open_sky_penalty: float = 1.0
    building_shade_bonus: float = 1.0
    covered_bonus: float = 1.0
    wind_open_penalty: float = 1.0
    wet_surface_penalty: float = 1.0
    normalized_signals: Dict[str, float] = field(default_factory=dict)
    # Пороги и k для ребра (копия из Settings на момент запроса).
    heat_edge_k_open: float = 0.54
    heat_edge_k_tree: float = 0.44
    heat_edge_k_building: float = 0.34
    heat_edge_k_covered: float = 0.16
    heat_edge_k_wet: float = 0.28
    heat_edge_k_wind: float = 0.30
    heat_wind_exp_w1: float = 0.45
    heat_wind_exp_w2: float = 0.35
    heat_wind_exp_w3: float = 0.35
    heat_wet_surface_edge_bad_max: float = 0.85
    heat_open_wet_synergy: float = 0.14
    heat_edge_factor_min: float = 0.65
    heat_edge_factor_max: float = 1.75
    heat_edge_rain_open_mult: float = 0.34
    heat_edge_rain_building_mult: float = 0.42
    heat_edge_rain_wind_exp_mult: float = 0.16
    # Нормированные сигналы и edge-stress (заполняются при enabled + settings).
    weather_stress_global_blend: float = 0.38
    stress_edge_rain_slip: float = 0.22
    stress_edge_wind_open: float = 0.20
    stress_edge_lts_fast: float = 0.17
    stress_edge_building_shelter: float = 0.11
    stress_edge_factor_min: float = 0.82
    stress_edge_factor_max: float = 1.48
    phys_wet_tier0_cap: float = 0.016
    phys_wet_tier1_coef: float = 0.048
    phys_wet_tier2_coef: float = 0.12


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def snapshot_from_manual(
    *,
    temperature_c: Optional[float] = None,
    precipitation_mm: Optional[float] = None,
    wind_speed_ms: Optional[float] = None,
    wind_gusts_ms: Optional[float] = None,
    cloud_cover_pct: Optional[float] = None,
    humidity_pct: Optional[float] = None,
    apparent_temperature_c: Optional[float] = None,
    shortwave_radiation_wm2: Optional[float] = None,
) -> WeatherSnapshot:
    return WeatherSnapshot(
        temperature_c=float(temperature_c if temperature_c is not None else 20.0),
        apparent_temperature_c=apparent_temperature_c,
        precipitation_mm=float(precipitation_mm if precipitation_mm is not None else 0.0),
        wind_speed_ms=float(wind_speed_ms if wind_speed_ms is not None else 3.0),
        wind_gusts_ms=wind_gusts_ms,
        cloud_cover_pct=float(cloud_cover_pct if cloud_cover_pct is not None else 50.0),
        humidity_pct=float(humidity_pct if humidity_pct is not None else 60.0),
        shortwave_radiation_wm2=shortwave_radiation_wm2,
    )


def _normalized_weather_signals(snap: WeatherSnapshot, s: Any) -> Dict[str, float]:
    """Нормированные сигналы [0..1] и cold_like [0..1] для непрерывной тепловой модели."""
    T = float(snap.temperature_c)
    rain = float(snap.precipitation_mm)
    wind = float(snap.wind_speed_ms)
    gust_o = snap.wind_gusts_ms
    gust = float(gust_o) if gust_o is not None else wind
    clouds = float(snap.cloud_cover_pct)
    hum = float(snap.humidity_pct)

    tmax = max(float(getattr(s, "heat_temp_ref_max", 30.0)), 1e-6)
    rmax = max(float(getattr(s, "heat_rain_ref_max", 3.0)), 1e-6)
    wmax = max(float(getattr(s, "heat_wind_ref_max", 12.0)), 1e-6)
    gdmax = max(float(getattr(s, "heat_gust_delta_ref_max", 10.0)), 1e-6)
    tcref = float(getattr(s, "heat_temp_cool_ref", 10.0))
    tc_rng = max(float(getattr(s, "heat_temp_cool_range", 15.0)), 1e-6)

    temp_norm = _clamp(T / tmax, 0.0, 1.0)
    rain_norm = _clamp(rain / rmax, 0.0, 1.0)
    wind_norm = _clamp(wind / wmax, 0.0, 1.0)
    gust_delta = max(0.0, gust - wind)
    gust_norm = _clamp(gust_delta / gdmax, 0.0, 1.0)
    cloud_norm = _clamp(clouds / 100.0, 0.0, 1.0)
    humidity_norm = _clamp(hum / 100.0, 0.0, 1.0)
    cold_like_norm = _clamp((tcref - T) / tc_rng, 0.0, 1.0)
    return {
        "temp_norm": temp_norm,
        "rain_norm": rain_norm,
        "wind_norm": wind_norm,
        "gust_norm": gust_norm,
        "cloud_norm": cloud_norm,
        "humidity_norm": humidity_norm,
        "cold_like_norm": cold_like_norm,
    }


def _continuous_six_coefficients(
    sig: Dict[str, float], s: Any
) -> Tuple[float, float, float, float, float, float]:
    """Шесть непрерывных множителей для поправки тепловой компоненты ребра."""
    tn = sig["temp_norm"]
    rn = sig["rain_norm"]
    wn = sig["wind_norm"]
    gn = sig["gust_norm"]
    cn = sig["cloud_norm"]
    hn = sig["humidity_norm"]
    cl = sig["cold_like_norm"]

    a1 = float(getattr(s, "heat_tree_shade_temp_gain", 0.42))
    a2 = float(getattr(s, "heat_tree_shade_rain_damp", 0.35))
    a3 = float(getattr(s, "heat_tree_shade_cold_damp", 0.38))
    tree_shade_bonus = 1.0 + a1 * tn - a2 * rn - a3 * cl

    b1 = float(getattr(s, "heat_open_sky_temp_gain", 0.52))
    b2 = float(getattr(s, "heat_open_sky_wind_gain", 0.28))
    b3 = float(getattr(s, "heat_open_sky_gust_gain", 0.22))
    b4 = float(getattr(s, "heat_open_sky_humid_gain", 0.18))
    open_sky_penalty = (
        1.0
        + b1 * tn * (1.0 - cn)
        + b2 * wn
        + b3 * gn
        + b4 * hn * tn
    )

    c1 = float(getattr(s, "heat_building_shade_wind_gain", 0.32))
    c2 = float(getattr(s, "heat_building_shade_gust_gain", 0.22))
    c3 = float(getattr(s, "heat_building_shade_rain_gain", 0.26))
    c4 = float(getattr(s, "heat_building_shade_humid_gain", 0.14))
    building_shade_bonus = (
        1.0 + c1 * wn + c2 * gn + c3 * rn + c4 * hn * (1.0 - cn)
    )

    d1 = float(getattr(s, "heat_covered_rain_gain", 0.45))
    d2 = float(getattr(s, "heat_covered_wind_gain", 0.22))
    d3 = float(getattr(s, "heat_covered_gust_gain", 0.18))
    covered_bonus = 1.0 + d1 * rn + d2 * wn + d3 * gn

    wg = float(getattr(s, "heat_wind_open_penalty_gain", 0.34))
    gg = float(getattr(s, "heat_wind_open_gust_gain", 0.24))
    wind_open_penalty = 1.0 + wg * wn + gg * gn

    e1 = float(getattr(s, "heat_wet_surface_rain_gain", 0.38))
    e2 = float(getattr(s, "heat_wet_surface_humid_gain", 0.22))
    wet_surface_penalty = 1.0 + e1 * rn + e2 * hn

    lo = float(getattr(s, "heat_coeff_clamp_lo", 0.72))
    hi = float(getattr(s, "heat_coeff_clamp_hi", 1.42))
    return (
        _clamp(tree_shade_bonus, lo, hi),
        _clamp(open_sky_penalty, lo, hi),
        _clamp(building_shade_bonus, lo, hi),
        _clamp(covered_bonus, lo, hi),
        _clamp(wind_open_penalty, lo, hi),
        _clamp(wet_surface_penalty, lo, hi),
    )


def _edge_params_from_settings(s: Any) -> Dict[str, float]:
    return {
        "heat_edge_k_open": float(getattr(s, "heat_edge_k_open", 0.54)),
        "heat_edge_k_tree": float(getattr(s, "heat_edge_k_tree", 0.44)),
        "heat_edge_k_building": float(getattr(s, "heat_edge_k_building", 0.34)),
        "heat_edge_k_covered": float(getattr(s, "heat_edge_k_covered", 0.16)),
        "heat_edge_k_wet": float(getattr(s, "heat_edge_k_wet", 0.28)),
        "heat_edge_k_wind": float(getattr(s, "heat_edge_k_wind", 0.30)),
        "heat_wind_exp_w1": float(getattr(s, "heat_wind_exp_w1", 0.45)),
        "heat_wind_exp_w2": float(getattr(s, "heat_wind_exp_w2", 0.35)),
        "heat_wind_exp_w3": float(getattr(s, "heat_wind_exp_w3", 0.35)),
        "heat_wet_surface_edge_bad_max": float(
            getattr(s, "heat_wet_surface_edge_bad_max", 0.85)
        ),
        "heat_open_wet_synergy": float(getattr(s, "heat_open_wet_synergy", 0.14)),
        "heat_edge_factor_min": float(getattr(s, "heat_edge_factor_min", 0.65)),
        "heat_edge_factor_max": float(getattr(s, "heat_edge_factor_max", 1.75)),
        "heat_edge_rain_open_mult": float(
            getattr(s, "heat_edge_rain_open_mult", 0.34)
        ),
        "heat_edge_rain_building_mult": float(
            getattr(s, "heat_edge_rain_building_mult", 0.42)
        ),
        "heat_edge_rain_wind_exp_mult": float(
            getattr(s, "heat_edge_rain_wind_exp_mult", 0.16)
        ),
    }


def _stress_and_phys_wet_params_from_settings(s: Any) -> Dict[str, float]:
    return {
        "weather_stress_global_blend": float(
            getattr(s, "weather_stress_global_blend", 0.38)
        ),
        "stress_edge_rain_slip": float(getattr(s, "weather_stress_edge_rain_slip", 0.22)),
        "stress_edge_wind_open": float(
            getattr(s, "weather_stress_edge_wind_open", 0.20)
        ),
        "stress_edge_lts_fast": float(getattr(s, "weather_stress_edge_lts_fast", 0.17)),
        "stress_edge_building_shelter": float(
            getattr(s, "weather_stress_edge_building_shelter", 0.11)
        ),
        "stress_edge_factor_min": float(
            getattr(s, "weather_stress_edge_factor_min", 0.82)
        ),
        "stress_edge_factor_max": float(
            getattr(s, "weather_stress_edge_factor_max", 1.48)
        ),
        "phys_wet_tier0_cap": float(getattr(s, "weather_phys_wet_penalty_tier0_cap", 0.016)),
        "phys_wet_tier1_coef": float(
            getattr(s, "weather_phys_wet_penalty_tier1_coef", 0.048)
        ),
        "phys_wet_tier2_coef": float(
            getattr(s, "weather_phys_wet_penalty_tier2_coef", 0.12)
        ),
    }


def build_weather_weight_params(
    snap: WeatherSnapshot,
    *,
    enabled: bool,
    policy: Optional[Dict[str, Any]] = None,
    thermal_scales: Optional[Dict[str, float]] = None,
    settings: Optional[Any] = None,
) -> WeatherWeightParams:
    """Собрать параметры для RouteService из снимка погоды."""
    if not enabled:
        return WeatherWeightParams(enabled=False)
    pol = policy if policy is not None else load_weather_policy()
    gcc = float((pol or {}).get("green_edge_coupling", 0.55))
    scales = thermal_scales or {}
    rs = float(scales.get("response", 1.0))
    mults = compute_weather_multipliers(snap, policy=pol, response_scale=rs)
    regime = classify_weather_regime(snap)

    sig: Dict[str, float] = {}
    spw: Dict[str, float] = {}
    if settings is not None:
        sig = _normalized_weather_signals(snap, settings)
        spw = _stress_and_phys_wet_params_from_settings(settings)

    base_kw: Dict[str, Any] = dict(
        enabled=True,
        mults=mults,
        green_coupling=gcc,
        regime=regime,
        hot_tree_bonus_scale=float(scales.get("hot_tree", 1.0)),
        hot_open_penalty_scale=float(scales.get("hot_open", 1.0)),
        cold_canyon_bonus_scale=float(scales.get("cold_canyon", 1.0)),
        cold_tree_damping=float(scales.get("cold_tree_damp", 0.65)),
        weather_response_scale=rs,
        heat_continuous=False,
        normalized_signals=sig,
        **spw,
    )

    if settings is not None and bool(
        getattr(settings, "heat_continuous_enable", True)
    ):
        tsb, osp, bsb, cbn, wop, wsp = _continuous_six_coefficients(sig, settings)
        ep = _edge_params_from_settings(settings)
        base_kw.update(
            heat_continuous=True,
            tree_shade_bonus=tsb,
            open_sky_penalty=osp,
            building_shade_bonus=bsb,
            covered_bonus=cbn,
            wind_open_penalty=wop,
            wet_surface_penalty=wsp,
            **ep,
        )
    return WeatherWeightParams(**base_kw)


def classify_weather_regime(snap: WeatherSnapshot) -> str:
    """Грубая классификация: жара/инсоляция vs холод/ветер/влага для микроклимата на рёбрах."""
    T = float(snap.temperature_c)
    Ta = float(snap.apparent_temperature_c) if snap.apparent_temperature_c is not None else T
    sw = snap.shortwave_radiation_wm2
    wind = max(float(snap.wind_speed_ms), float(snap.wind_gusts_ms or 0.0))
    hum = float(snap.humidity_pct)
    clouds = float(snap.cloud_cover_pct)
    rain = float(snap.precipitation_mm)

    hot_score = 0.0
    if max(T, Ta) >= 23.5:
        hot_score += 2.0
    if sw is not None:
        swf = float(sw)
        if swf >= 520:
            hot_score += 1.8
        elif swf >= 320:
            hot_score += 1.0
    if clouds <= 38:
        hot_score += 1.0
    if rain <= 0.08:
        hot_score += 0.35
    if Ta > T + 1.2:
        hot_score += 0.6

    cold_score = 0.0
    if min(T, Ta) <= 12.5:
        cold_score += 2.0
    if wind >= 9.0:
        cold_score += 1.2
    if hum >= 78:
        cold_score += 0.85
    if clouds >= 72 or rain >= 0.45:
        cold_score += 1.0
    if Ta < T - 0.8:
        cold_score += 0.5

    if hot_score >= 3.2 and cold_score < 2.2:
        return "hot"
    if cold_score >= 2.6 and hot_score < 3.0:
        return "cold"
    return "neutral"


def compute_weather_multipliers(
    snap: WeatherSnapshot,
    *,
    policy: Optional[Dict[str, Any]] = None,
    response_scale: float = 1.0,
) -> WeatherMultipliers:
    """Вычислить множители по политике и снимку погоды."""
    pol = policy if policy is not None else load_weather_policy()
    if not pol:
        return WeatherMultipliers()

    rs = max(0.5, min(2.5, float(response_scale)))
    m = pol.get("multipliers") or {}
    tc = pol.get("temperature_c") or {}
    pr = pol.get("precipitation_mm_h") or {}
    wn = pol.get("wind_ms") or {}
    cc = pol.get("cloud_cover_pct") or {}
    hu = pol.get("humidity_pct") or {}

    T = snap.temperature_c
    Ta = snap.apparent_temperature_c if snap.apparent_temperature_c is not None else T
    sw = snap.shortwave_radiation_wm2
    rain = snap.precipitation_mm
    wind = snap.wind_speed_ms
    gust = snap.wind_gusts_ms
    wind_eff = max(float(wind), float(gust or 0.0))
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
    if wind_eff > strong:
        phys *= 1.0 + float(mp.get("wind_headwind_proxy", 0.04)) * min(
            2.0, (wind_eff - strong) / max(strong, 1e-6)
        )
    if hum > float(hu.get("humid_above", 80)):
        phys *= float(mp.get("humid_surface", 1.05))

    # heat (thermal discomfort routing component): T, облака, осадки + ощущаемая температура и солнце
    heat = float(mh.get("base", 1.0))
    t_for_heat = max(T, Ta)
    if t_for_heat > ref_heat:
        heat += float(mh.get("per_temp_above_22", 0.022)) * (t_for_heat - ref_heat) * (
            0.85 + 0.15 * rs
        )
    if Ta > T + 0.5:
        heat *= 1.0 + 0.012 * rs * min(8.0, Ta - T)
    heat *= max(0.65, 1.0 - float(mh.get("cloud_reduction_per_pct", 0.0022)) * clouds)
    if rain > dry_max:
        heat *= float(
            mh.get(
                "rain_reduction_heavy" if rain >= heavy else "rain_reduction_light",
                0.9,
            )
        )
    if sw is not None:
        sw_norm = max(0.0, min(1.0, (float(sw) - 120.0) / 780.0))
        heat *= 1.0 + 0.14 * rs * sw_norm

    # green comfort multiplier (applied with trees_pct on edge)
    green = float(mg.get("base", 1.0))
    warm_min = float(tc.get("warm_min", 25))
    clear_max = float(cc.get("clear_max", 30))
    overcast = float(cc.get("overcast_min", 70))
    if T >= warm_min and clouds <= clear_max and rain <= dry_max:
        green *= float(mg.get("hot_sunny_bonus", 1.18)) ** (0.85 + 0.15 * rs)
    elif T < float(tc.get("cool_max", 15)) or clouds >= overcast:
        green *= float(mg.get("cool_cloud_penalty", 0.92)) ** (0.9 + 0.1 * rs)
    if rain > light_max:
        green *= float(mg.get("rain_neutralize", 0.72))
    if sw is not None and float(sw) >= 400 and T >= 22:
        green *= 1.0 + 0.06 * rs * min(1.0, (float(sw) - 400.0) / 500.0)

    # stress
    stress = float(ms.get("base", 1.0))
    if rain > dry_max:
        stress *= float(ms.get("rain", 1.12))
    if wind_eff > strong:
        stress *= float(ms.get("wind_strong", 1.1)) ** (0.9 + 0.1 * min(1.5, rs))
    if clouds > float(cc.get("overcast_min", 70)) and rain > dry_max * 2:
        stress *= float(ms.get("low_visibility_cloud", 1.05))
    if gust is not None and float(gust) > wind + 4:
        stress *= 1.0 + 0.04 * rs * min(1.0, (float(gust) - wind) / 12.0)

    # surface: глобальный множитель не усиливает дождь/влажность — штраф по покрытию на уровне ребра.
    surface = float(msv.get("base", 1.0))

    # Диапазоны расширяются с response_scale, чтобы агрегаты батча различали контрастные дни.
    phys_lo, phys_hi = 0.72 - 0.04 * (rs - 1.0), 1.48 + 0.08 * (rs - 1.0)
    heat_lo = 0.42 - 0.12 * (rs - 1.0)
    heat_hi = 1.82 + 0.35 * (rs - 1.0)
    green_lo = 0.48 - 0.08 * (rs - 1.0)
    green_hi = 1.48 + 0.12 * (rs - 1.0)
    stress_lo, stress_hi = 0.82 - 0.05 * (rs - 1.0), 1.52 + 0.12 * (rs - 1.0)

    return WeatherMultipliers(
        physical=_clamp(phys, phys_lo, phys_hi),
        heat=_clamp(heat, heat_lo, heat_hi),
        green=_clamp(green, green_lo, green_hi),
        stress=_clamp(stress, stress_lo, stress_hi),
        surface=_clamp(surface, 0.97, 1.03),
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
    thermal_scales: Optional[Dict[str, float]] = None,
    settings: Optional[Any] = None,
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
    return snap, src, build_weather_weight_params(
        snap,
        enabled=enabled,
        thermal_scales=thermal_scales,
        settings=settings,
    )
