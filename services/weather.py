"""Погодный контекст: Open-Meteo, кэш, множители для весов рёбер."""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import requests

from .policy_data import load_weather_policy
from .seasonal import (
    normalized_snow_signals,
    resolve_season_routing_context,
    season_stairs_route_multiplier,
    season_stress_route_multiplier,
    season_tree_heat_route_multiplier,
    season_wind_orientation_route_multiplier,
    snow_depth_phys_multiplier,
    snow_fresh_phys_multiplier,
    snow_route_model_strength,
    parse_route_calendar_date,
)

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
    # Направление ветра по метеоконвенции Open-Meteo: градусы, **откуда** дует ветер (0° — север).
    wind_direction_deg: Optional[float] = None
    cloud_cover_pct: float = 50.0
    humidity_pct: float = 60.0
    shortwave_radiation_wm2: Optional[float] = None
    # Снег и код WMO (Open-Meteo). snow_depth — модельная глубина, не состояние тротуара.
    snowfall_cm_h: float = 0.0
    snow_depth_m: float = 0.0
    weather_code: Optional[int] = None


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
    heat_edge_k_open: float = 0.66
    heat_edge_k_tree: float = 0.54
    heat_edge_k_building: float = 0.50
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
    heat_edge_rain_open_mult: float = 0.48
    heat_edge_rain_building_mult: float = 0.56
    heat_edge_rain_wind_exp_mult: float = 0.22
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
    # Сезон и зима (заполняется в build_weather_weight_params при settings).
    routing_season: str = ""
    season_green_route_mult: float = 1.0
    season_tree_heat_route_mult: float = 1.0
    season_stress_route_mult: float = 1.0
    season_stairs_route_mult: float = 1.0
    season_wind_orientation_route_mult: float = 1.0
    stress_route_regime_factor: float = 1.0
    reference_temperature_c: float = 20.0
    weather_code: Optional[int] = None
    snow_model_strength: float = 0.0
    snow_depth_norm: float = 0.0
    snow_fresh_norm: float = 0.0
    snow_depth_phys_mult: float = 1.0
    snow_fresh_phys_mult: float = 1.0
    snow_stress_global_add: float = 0.0
    snow_export_surface_amp: float = 1.0
    snow_export_stress_amp: float = 1.0
    snow_export_phys_amp: float = 1.0
    winter_heat_tree_scale: float = 1.0
    winter_heat_open_scale: float = 1.0
    winter_heat_wind_scale: float = 1.0
    stress_edge_snow_open: float = 0.18
    stress_edge_snow_surface: float = 0.22
    stress_edge_snow_stairs: float = 0.16
    snow_surface_tier0_amp: float = 0.012
    snow_surface_tier1_amp: float = 0.055
    snow_surface_tier2_amp: float = 0.14
    snow_stairs_ped_base: float = 1.08
    snow_stairs_ped_frozen_boost: float = 1.12
    snow_stairs_cyclist_base: float = 1.18
    snow_stairs_cyclist_frozen_boost: float = 1.22
    snow_phys_cyclist_mult_boost: float = 0.12
    # Направление ветра (если False — используется прежняя безнаправленная модель).
    wind_direction_deg: Optional[float] = None
    wind_direction_available: bool = False
    heat_wind_along_open_amp: float = 0.28
    heat_wind_cross_build_amp: float = 0.35
    stress_wind_along_open_amp: float = 0.32
    stress_wind_cross_shelter_amp: float = 0.40
    # Ослабление бонуса зданий, когда ветер «вдоль» коридора (мало экранирования фасадом).
    heat_wind_along_build_damp: float = 0.24
    # Сезон (календарь vs адаптивный снимок) — дублируется для логов/Excel.
    routing_season_calendar: str = ""
    routing_season_effective: str = ""
    routing_season_source: str = "calendar"
    # Интерпретация WMO weather_code (маршрутизация + нормы в normalized_signals).
    wc_snow_model_strength_amp: float = 0.22
    wc_wet_slip_physical_amp: float = 0.08
    wc_mixed_precip_stress_amp: float = 0.12
    wc_wet_slip_surface_amp: float = 0.26
    wc_mixed_surface_amp: float = 0.18
    wc_wet_stairs_amp: float = 0.14
    wc_mixed_snow_stress_amp: float = 0.16
    wc_winter_open_sky_penalty_amp: float = 0.05
    wc_winter_building_shelter_bonus_amp: float = 0.06
    winter_clearance_low_amp: float = 0.32
    winter_clearance_low_amp_cyclist: float = 0.22
    winter_clearance_high_mitigate: float = 0.22


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
    snowfall_cm_h: Optional[float] = None,
    snow_depth_m: Optional[float] = None,
    weather_code: Optional[int] = None,
    wind_direction_deg: Optional[float] = None,
) -> WeatherSnapshot:
    wd_norm: Optional[float] = None
    if wind_direction_deg is not None:
        try:
            fv = float(wind_direction_deg)
            if math.isfinite(fv):
                wd_norm = fv % 360.0
        except (TypeError, ValueError):
            wd_norm = None
    return WeatherSnapshot(
        temperature_c=float(temperature_c if temperature_c is not None else 20.0),
        apparent_temperature_c=apparent_temperature_c,
        precipitation_mm=float(precipitation_mm if precipitation_mm is not None else 0.0),
        wind_speed_ms=float(wind_speed_ms if wind_speed_ms is not None else 3.0),
        wind_gusts_ms=wind_gusts_ms,
        cloud_cover_pct=float(cloud_cover_pct if cloud_cover_pct is not None else 50.0),
        humidity_pct=float(humidity_pct if humidity_pct is not None else 60.0),
        shortwave_radiation_wm2=shortwave_radiation_wm2,
        snowfall_cm_h=float(snowfall_cm_h if snowfall_cm_h is not None else 0.0),
        snow_depth_m=float(snow_depth_m if snow_depth_m is not None else 0.0),
        weather_code=weather_code,
        wind_direction_deg=wd_norm,
    )


def wmo_weather_code_profile(code: Optional[int]) -> Dict[str, float]:
    """Нормированные признаки по коду WMO (Open-Meteo).

    Не заменяют snowfall/snow_depth: уточняют тип осадков/ледяного тумана.
    Ключи: wc_snow_event, wc_wet_slip, wc_mixed, wc_fog — в диапазоне [0..1].
    """
    z = {"wc_snow_event": 0.0, "wc_wet_slip": 0.0, "wc_mixed": 0.0, "wc_fog": 0.0}
    if code is None:
        return z
    try:
        c = int(round(float(code)))
    except (TypeError, ValueError):
        return z

    if c in (71, 72, 73, 74, 75, 76, 77, 85, 86):
        z["wc_snow_event"] = 1.0
    if c in (45, 48):
        z["wc_fog"] = 1.0
    if c in (56, 57, 66, 67):
        z["wc_wet_slip"] = 1.0
    if c in (95, 96, 97, 98, 99):
        z["wc_mixed"] = 1.0
    elif c in (80, 81, 82):
        z["wc_mixed"] = 0.22 + 0.28 * float(c - 80)
    if c in (61, 62, 63, 64, 65) and float(z["wc_wet_slip"]) < 0.35:
        z["wc_wet_slip"] = max(z["wc_wet_slip"], 0.25)
    return z


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
    snd, snf = normalized_snow_signals(snap, s)
    wc = wmo_weather_code_profile(getattr(snap, "weather_code", None))
    out = {
        "temp_norm": temp_norm,
        "rain_norm": rain_norm,
        "wind_norm": wind_norm,
        "gust_norm": gust_norm,
        "cloud_norm": cloud_norm,
        "humidity_norm": humidity_norm,
        "cold_like_norm": cold_like_norm,
        "snow_depth_norm": float(snd),
        "snow_fresh_norm": float(snf),
    }
    out.update(wc)
    return out


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
        "heat_edge_k_open": float(getattr(s, "heat_edge_k_open", 0.66)),
        "heat_edge_k_tree": float(getattr(s, "heat_edge_k_tree", 0.54)),
        "heat_edge_k_building": float(getattr(s, "heat_edge_k_building", 0.50)),
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
            getattr(s, "heat_edge_rain_open_mult", 0.48)
        ),
        "heat_edge_rain_building_mult": float(
            getattr(s, "heat_edge_rain_building_mult", 0.56)
        ),
        "heat_edge_rain_wind_exp_mult": float(
            getattr(s, "heat_edge_rain_wind_exp_mult", 0.22)
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
        "stress_edge_snow_open": float(
            getattr(s, "weather_stress_edge_snow_open", 0.18)
        ),
        "stress_edge_snow_surface": float(
            getattr(s, "weather_stress_edge_snow_surface", 0.22)
        ),
        "stress_edge_snow_stairs": float(
            getattr(s, "weather_stress_edge_snow_stairs", 0.16)
        ),
    }


def _wc_routing_params_from_settings(s: Any) -> Dict[str, float]:
    return {
        "wc_snow_model_strength_amp": float(
            getattr(s, "wc_snow_model_strength_amp", 0.22)
        ),
        "wc_wet_slip_physical_amp": float(
            getattr(s, "wc_wet_slip_physical_amp", 0.08)
        ),
        "wc_mixed_precip_stress_amp": float(
            getattr(s, "wc_mixed_precip_stress_amp", 0.12)
        ),
        "wc_wet_slip_surface_amp": float(getattr(s, "wc_wet_slip_surface_amp", 0.26)),
        "wc_mixed_surface_amp": float(getattr(s, "wc_mixed_surface_amp", 0.18)),
        "wc_wet_stairs_amp": float(getattr(s, "wc_wet_stairs_amp", 0.14)),
        "wc_mixed_snow_stress_amp": float(getattr(s, "wc_mixed_snow_stress_amp", 0.16)),
        "winter_clearance_low_amp": float(getattr(s, "winter_clearance_low_amp", 0.32)),
        "winter_clearance_low_amp_cyclist": float(
            getattr(s, "winter_clearance_low_amp_cyclist", 0.22)
        ),
        "winter_clearance_high_mitigate": float(
            getattr(s, "winter_clearance_high_mitigate", 0.22)
        ),
        "wc_winter_open_sky_penalty_amp": float(
            getattr(s, "wc_winter_open_sky_penalty_amp", 0.05)
        ),
        "wc_winter_building_shelter_bonus_amp": float(
            getattr(s, "wc_winter_building_shelter_bonus_amp", 0.06)
        ),
    }


def _wind_direction_params_from_settings(s: Any) -> Dict[str, float]:
    return {
        "heat_wind_along_open_amp": float(
            getattr(s, "heat_wind_along_open_amp", 0.28)
        ),
        "heat_wind_cross_build_amp": float(
            getattr(s, "heat_wind_cross_build_amp", 0.35)
        ),
        "stress_wind_along_open_amp": float(
            getattr(s, "stress_wind_along_open_amp", 0.32)
        ),
        "stress_wind_cross_shelter_amp": float(
            getattr(s, "stress_wind_cross_shelter_amp", 0.40)
        ),
        "heat_wind_along_build_damp": float(
            getattr(s, "heat_wind_along_build_damp", 0.24)
        ),
    }


def build_weather_weight_params(
    snap: WeatherSnapshot,
    *,
    enabled: bool,
    policy: Optional[Dict[str, Any]] = None,
    thermal_scales: Optional[Dict[str, float]] = None,
    settings: Optional[Any] = None,
    reference_iso: Optional[str] = None,
) -> WeatherWeightParams:
    """Собрать параметры для RouteService из снимка погоды."""
    if not enabled:
        return WeatherWeightParams(enabled=False)
    wdir_raw = getattr(snap, "wind_direction_deg", None)
    wdir_ok = False
    wdir_val: Optional[float] = None
    if wdir_raw is not None:
        try:
            fv = float(wdir_raw)
            if math.isfinite(fv):
                wdir_ok = True
                wdir_val = fv % 360.0
        except (TypeError, ValueError):
            pass
    pol = policy if policy is not None else load_weather_policy()
    gcc = float((pol or {}).get("green_edge_coupling", 0.55))
    scales = thermal_scales or {}
    rs = float(scales.get("response", 1.0))
    mults = compute_weather_multipliers(snap, policy=pol, response_scale=rs)
    regime = classify_weather_regime(snap)

    sig: Dict[str, float] = {}
    spw: Dict[str, float] = {}
    routing_season = ""
    season_green_rm = 1.0
    tree_heat_rm = 1.0
    snow_strength = 0.0
    snd = 0.0
    snf = 0.0
    dpm = 1.0
    fpm = 1.0
    snow_stress_add = 0.0
    w_tree = 1.0
    w_open = 1.0
    w_wind = 1.0
    snow_surf_export = 1.0
    snow_stress_export = 1.0
    stress_before = float(mults.stress)
    phys_before = float(mults.physical)
    routing_season_calendar = ""
    routing_season_effective = ""
    routing_season_source = "calendar"

    if settings is not None:
        if not wdir_ok:
            logger.info(
                "weather: направление ветра недоступно — используется "
                "омнинаправленная ветровая модель (как до внедрения wind_direction)"
            )
        sig = _normalized_weather_signals(snap, settings)
        spw = _stress_and_phys_wet_params_from_settings(settings)
        ref_d = parse_route_calendar_date(reference_iso) or datetime.now(
            timezone.utc
        ).date()
        sctx = resolve_season_routing_context(ref_d, snap, settings)
        routing_season = sctx.effective_season
        routing_season_calendar = sctx.calendar_season
        routing_season_effective = sctx.effective_season
        routing_season_source = sctx.source
        season_green_rm = float(sctx.season_green_route_mult)
        tree_heat_rm = season_tree_heat_route_multiplier(routing_season, settings)
        snow_strength = snow_route_model_strength(routing_season, snap, settings)
        snd = float(sig.get("snow_depth_norm", 0.0))
        snf = float(sig.get("snow_fresh_norm", 0.0))
        wcs = float(sig.get("wc_snow_event", 0.0))
        wcw = float(sig.get("wc_wet_slip", 0.0))
        wcm = float(sig.get("wc_mixed", 0.0))
        w_amp = float(getattr(settings, "wc_snow_model_strength_amp", 0.22))
        snow_strength = min(1.0, snow_strength * (1.0 + w_amp * wcs))
        dpm = snow_depth_phys_multiplier(float(snap.snow_depth_m or 0.0), settings)
        fpm = snow_fresh_phys_multiplier(float(snap.snowfall_cm_h or 0.0), settings)
        comb = float(dpm * fpm)
        mults.physical *= 1.0 + snow_strength * (comb - 1.0)
        s_coupling = float(getattr(settings, "snow_stress_phys_coupling", 0.45))
        mults.stress *= 1.0 + snow_strength * min(0.28, (comb - 1.0) * s_coupling)
        mults.physical *= 1.0 + snow_strength * float(
            getattr(settings, "wc_wet_slip_physical_amp", 0.08)
        ) * wcw
        mults.stress *= 1.0 + snow_strength * float(
            getattr(settings, "wc_mixed_precip_stress_amp", 0.12)
        ) * max(wcm, 0.5 * wcw)
        snow_stress_add = float(
            getattr(settings, "snow_stress_global_add_winter", 0.035)
        ) * snow_strength + 0.05 * snow_strength * snd
        ddamp = float(getattr(settings, "winter_heat_tree_damp_snow_depth", 0.38))
        fdamp = float(getattr(settings, "winter_heat_tree_damp_snow_fresh", 0.22))
        w_tree = max(
            0.05,
            float(tree_heat_rm)
            * (1.0 - ddamp * snd * snow_strength)
            * (1.0 - fdamp * snf * snow_strength),
        )
        if routing_season == "winter":
            wo = float(getattr(settings, "winter_heat_open_scale_winter", 1.18))
            ww = float(getattr(settings, "winter_heat_wind_scale_winter", 1.22))
        elif routing_season == "green_season":
            wo = ww = 1.0
        else:
            wo = float(getattr(settings, "winter_heat_open_scale_transition", 1.08))
            ww = float(getattr(settings, "winter_heat_wind_scale_transition", 1.12))
        w_open = 1.0 + (wo - 1.0) * snow_strength + 0.06 * snd * snow_strength
        w_wind = 1.0 + (ww - 1.0) * snow_strength + 0.08 * snf * snow_strength
        tier0 = float(getattr(settings, "snow_surface_tier0_amp", 0.012))
        tier1 = float(getattr(settings, "snow_surface_tier1_amp", 0.055))
        tier2 = float(getattr(settings, "snow_surface_tier2_amp", 0.14))
        snow_surf_export = 1.0 + snow_strength * (
            tier0 * (1.0 - max(snd, snf))
            + tier1 * min(1.0, snd + snf)
            + tier2 * snd * snf
        )
        snow_stress_export = float(mults.stress / max(1e-9, stress_before))

    wind_dir_kw: Dict[str, float] = {}
    wc_kw: Dict[str, float] = {}
    if settings is not None:
        wind_dir_kw = _wind_direction_params_from_settings(settings)
        wc_kw = _wc_routing_params_from_settings(settings)

    season_stress_rm = 1.0
    season_stairs_rm = 1.0
    season_wind_orient_rm = 1.0
    if settings is not None:
        lab = (
            (routing_season_effective or routing_season or "green_season")
            .strip()
            .lower()
            or "green_season"
        )
        season_stress_rm = season_stress_route_multiplier(lab, settings)
        season_stairs_rm = season_stairs_route_multiplier(lab, settings)
        season_wind_orient_rm = season_wind_orientation_route_multiplier(
            lab, settings
        )
    blend_sg = float(spw.get("weather_stress_global_blend", 0.38)) if spw else 0.38
    stress_route_regime_factor = (
        1.0 + blend_sg * (float(mults.stress) - 1.0) + float(snow_stress_add)
    ) * float(season_stress_rm)

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
        routing_season=routing_season,
        routing_season_calendar=routing_season_calendar,
        routing_season_effective=routing_season_effective,
        routing_season_source=routing_season_source,
        season_green_route_mult=float(season_green_rm),
        season_tree_heat_route_mult=float(tree_heat_rm),
        season_stress_route_mult=float(season_stress_rm),
        season_stairs_route_mult=float(season_stairs_rm),
        season_wind_orientation_route_mult=float(season_wind_orient_rm),
        stress_route_regime_factor=float(stress_route_regime_factor),
        reference_temperature_c=float(snap.temperature_c),
        weather_code=snap.weather_code,
        snow_model_strength=float(snow_strength),
        snow_depth_norm=float(snd),
        snow_fresh_norm=float(snf),
        snow_depth_phys_mult=float(dpm),
        snow_fresh_phys_mult=float(fpm),
        snow_stress_global_add=float(snow_stress_add),
        snow_export_surface_amp=float(snow_surf_export),
        snow_export_stress_amp=float(snow_stress_export),
        snow_export_phys_amp=float(mults.physical / max(1e-9, phys_before)),
        wind_direction_deg=wdir_val,
        wind_direction_available=bool(wdir_ok),
        **wind_dir_kw,
        **wc_kw,
        winter_heat_tree_scale=float(w_tree),
        winter_heat_open_scale=float(w_open),
        winter_heat_wind_scale=float(w_wind),
        snow_surface_tier0_amp=float(
            getattr(settings, "snow_surface_tier0_amp", 0.012)
        )
        if settings is not None
        else 0.012,
        snow_surface_tier1_amp=float(
            getattr(settings, "snow_surface_tier1_amp", 0.055)
        )
        if settings is not None
        else 0.055,
        snow_surface_tier2_amp=float(
            getattr(settings, "snow_surface_tier2_amp", 0.14)
        )
        if settings is not None
        else 0.14,
        snow_stairs_ped_base=float(getattr(settings, "snow_stairs_ped_base", 1.08))
        if settings is not None
        else 1.08,
        snow_stairs_ped_frozen_boost=float(
            getattr(settings, "snow_stairs_ped_frozen_boost", 1.12)
        )
        if settings is not None
        else 1.12,
        snow_stairs_cyclist_base=float(
            getattr(settings, "snow_stairs_cyclist_base", 1.18)
        )
        if settings is not None
        else 1.18,
        snow_stairs_cyclist_frozen_boost=float(
            getattr(settings, "snow_stairs_cyclist_frozen_boost", 1.22)
        )
        if settings is not None
        else 1.22,
        snow_phys_cyclist_mult_boost=float(
            getattr(settings, "snow_phys_cyclist_mult_boost", 0.12)
        )
        if settings is not None
        else 0.12,
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
    snow_d = float(getattr(snap, "snow_depth_m", 0.0) or 0.0)
    snow_f = float(getattr(snap, "snowfall_cm_h", 0.0) or 0.0)

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
    if snow_d >= 0.02 or snow_f >= 0.25:
        cold_score += 0.85

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
                "wind_direction_10m",
                "shortwave_radiation",
                "snowfall",
                "snow_depth",
                "weather_code",
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

    snow_fall = max(0.0, _arr("snowfall", 0.0))
    snow_dep = max(0.0, _arr("snow_depth", 0.0))
    wc_raw = None
    wc_arr = hourly.get("weather_code")
    if wc_arr and best_i < len(wc_arr) and wc_arr[best_i] is not None:
        try:
            wc_raw = int(round(float(wc_arr[best_i])))
        except (TypeError, ValueError):
            wc_raw = None

    wd_snap: Optional[float] = None
    wd_arr = hourly.get("wind_direction_10m")
    if wd_arr and best_i < len(wd_arr) and wd_arr[best_i] is not None:
        try:
            vwd = float(wd_arr[best_i])
            if math.isfinite(vwd):
                wd_snap = vwd % 360.0
        except (TypeError, ValueError):
            wd_snap = None

    return WeatherSnapshot(
        temperature_c=_arr("temperature_2m", 20.0),
        apparent_temperature_c=app_t,
        precipitation_mm=max(0.0, _arr("precipitation", 0.0)),
        precipitation_probability=pr_prob,
        wind_speed_ms=wind_ms,
        wind_gusts_ms=gust,
        wind_direction_deg=wd_snap,
        cloud_cover_pct=_clamp(_arr("cloud_cover", 50.0), 0.0, 100.0),
        humidity_pct=_clamp(_arr("relative_humidity_2m", 60.0), 0.0, 100.0),
        shortwave_radiation_wm2=sw,
        snowfall_cm_h=snow_fall,
        snow_depth_m=snow_dep,
        weather_code=wc_raw,
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
        reference_iso=when,
    )
