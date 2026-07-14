"""Сервис маршрутизации и расчёта статистик маршрута."""

import logging
import math
import numbers
from collections import Counter
from heapq import heappop, heappush
from itertools import count
from typing import Any, Callable, Dict, List, Optional, Tuple

import networkx as nx
import osmnx as ox

from ..config import (
    MAX_ROUTE_GRADIENT_DISPLAY,
    ModeProfile,
    RoutingPreferenceProfile,
    TURN_ANGLE_THRESHOLD_DEG,
    TURN_PENALTY_BASE,
    TIME_SLOTS,
)
from .heat import angle_diff_deg, wet_surface_edge_slip_factor
from .routing_criteria import (
    CRITERION_KEYS,
    EdgeRouteCriteriaFactors,
    EffectiveEdgeComponents,
    compute_edge_route_criteria,
)
from .stress import stress_metrics_for_route
from .weather import WeatherWeightParams
from ..exceptions import RouteNotFoundError

logger = logging.getLogger(__name__)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _wind_flow_bearing_deg(wind_from_deg: float) -> float:
    """Направление потока воздуха (°): метео «откуда» → куда дует ветер."""
    return (float(wind_from_deg) + 180.0) % 360.0


def edge_wind_along_cross_01(
    edge_data: dict, wind_from_deg: float
) -> Tuple[float, float, float]:
    """Доля «вдоль ветра» / «поперёк» по оси улицы и мин. угол оси к потоку [0..90]."""
    brg = float(edge_data.get("edge_bearing_deg") or 0.0) % 360.0
    flow = _wind_flow_bearing_deg(wind_from_deg)
    delta = min(
        angle_diff_deg(brg, flow),
        angle_diff_deg((brg + 180.0) % 360.0, flow),
    )
    dr = math.radians(delta)
    c = math.cos(dr)
    s = math.sin(dr)
    return c * c, s * s, float(delta)


def _wind_dir_edge_pack(
    edge_data: dict, weather: WeatherWeightParams
) -> Optional[Tuple[float, float, float, float, float]]:
    """Если direction-aware активен: along01, cross01, delta_deg, wind_act_heat, wind_comb_stress."""
    if not weather.enabled:
        return None
    if not bool(getattr(weather, "wind_direction_available", False)):
        return None
    wd = getattr(weather, "wind_direction_deg", None)
    if wd is None:
        return None
    try:
        wdf = float(wd)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(wdf):
        return None
    along01, cross01, delta_deg = edge_wind_along_cross_01(edge_data, wdf)
    sig = getattr(weather, "normalized_signals", None) or {}
    wn = _clamp01(float(sig.get("wind_norm", 0.0)))
    gn = _clamp01(float(sig.get("gust_norm", 0.0)))
    wind_act_heat = max(wn, gn * 0.75)
    wind_comb_stress = max(wn, gn * 0.85)
    return (along01, cross01, delta_deg, wind_act_heat, wind_comb_stress)


def _surface_wet_route_tier(surface_effective: Any) -> int:
    """0 — хорошие покрытия; 1 — умеренные; 2 — плохие/скользкие при дожде."""
    s = str(surface_effective or "unknown").strip().lower()
    if s in ("asphalt", "paved", "concrete") or s.startswith("concrete:"):
        return 0
    if s in ("paving_stones", "bricks", "paving_stone"):
        return 0
    if s in ("compacted", "fine_gravel"):
        return 1
    return 2


def winter_clearance_priority_01(edge_data: dict) -> float:
    """0..1: эвристика «вероятнее расчищается» (без внешних данных о уборке).

    1 — пешеходные магистрали/жилые связи с хорошим покрытием; 0 — тропы, треки, лестницы.
    """
    hw = str(edge_data.get("highway") or "").strip().lower()
    if hw == "steps":
        return 0.06
    if hw in ("path", "track", "bridleway"):
        return 0.14
    if hw in ("pedestrian", "living_street"):
        return 0.88
    if hw in ("primary", "primary_link", "secondary", "secondary_link"):
        return 0.72
    if hw in ("tertiary", "tertiary_link", "unclassified"):
        return 0.58
    if hw in ("residential", "service"):
        return 0.62
    if hw == "cycleway":
        return 0.52
    if hw == "footway":
        tier = _surface_wet_route_tier(edge_data.get("surface_effective"))
        base = 0.48 + 0.22 * float(2 - min(tier, 2))
        ms = _parse_maxspeed_kmh(edge_data.get("maxspeed"))
        if ms is not None and ms >= 45.0:
            base = min(0.92, base + 0.18)
        return max(0.22, min(0.92, base))
    return 0.45


def _edge_winter_harsh_surface_indicator(edge_data: dict) -> float:
    """0..1: лестницы и «тяжёлые» покрытия для зимней модели (объясняемые метрики)."""
    hw = str(edge_data.get("highway") or "").strip().lower()
    if hw == "steps":
        return 1.0
    tier = _surface_wet_route_tier(edge_data.get("surface_effective"))
    if tier >= 2:
        return 1.0
    s = str(edge_data.get("surface_effective") or "unknown").strip().lower()
    if s in (
        "path",
        "track",
        "grass",
        "gravel",
        "dirt",
        "ground",
        "earth",
        "mud",
        "sand",
        "wood",
        "unpaved",
    ) or s.startswith("ground"):
        return 0.95
    if tier == 1:
        return 0.55
    return 0.12


def snow_surface_route_penalty(edge_data: dict, weather: WeatherWeightParams) -> float:
    """Снег × покрытие × weather_code × приоритет зимней уборки (эвристика OSM)."""
    if not weather.enabled:
        return 1.0
    ss = float(getattr(weather, "snow_model_strength", 0.0))
    if ss <= 1e-9:
        return 1.0
    surface_effective = edge_data.get("surface_effective")
    tier = _surface_wet_route_tier(surface_effective)
    t0 = float(getattr(weather, "snow_surface_tier0_amp", 0.012))
    t1 = float(getattr(weather, "snow_surface_tier1_amp", 0.055))
    t2 = float(getattr(weather, "snow_surface_tier2_amp", 0.14))
    amp = (t0, t1, t2)[min(max(tier, 0), 2)]
    snd = float(getattr(weather, "snow_depth_norm", 0.0))
    snf = float(getattr(weather, "snow_fresh_norm", 0.0))
    sn = max(snd, snf * 0.9)
    tier_w = 1.0 if tier == 0 else 1.65 if tier == 1 else 2.35
    extra = ss * sn * amp * 12.0 * tier_w
    clr = winter_clearance_priority_01(edge_data)
    low_n = max(0.0, min(1.0, 1.0 - clr))
    wcla = float(getattr(weather, "winter_clearance_low_amp", 0.32))
    extra *= 1.0 + wcla * low_n * min(1.0, sn + 0.25 * ss)
    mit = float(getattr(weather, "winter_clearance_high_mitigate", 0.22))
    extra *= max(0.68, 1.0 - mit * clr * ss * min(1.0, sn + 0.15))
    sig = getattr(weather, "normalized_signals", None) or {}
    wcw = float(sig.get("wc_wet_slip", 0.0))
    wcm = float(sig.get("wc_mixed", 0.0))
    w_amp_w = float(getattr(weather, "wc_wet_slip_surface_amp", 0.26))
    w_amp_m = float(getattr(weather, "wc_mixed_surface_amp", 0.18))
    tier_bad = 0.38 + 0.32 * float(min(max(tier, 0), 2))
    extra += (
        ss
        * sn
        * (w_amp_w * wcw + w_amp_m * wcm * 1.28)
        * tier_bad
        * 0.52
    )
    return max(1.0, min(1.55, 1.0 + extra))


def snow_stairs_phys_mult(weather: WeatherWeightParams, profile_key: str) -> float:
    """Дополнительный множитель физики для highway=steps при снеге."""
    ss = float(getattr(weather, "snow_model_strength", 0.0))
    if ss <= 1e-9:
        return 1.0
    T = float(getattr(weather, "reference_temperature_c", 20.0))
    near_freeze = max(0.0, min(1.0, (2.0 - T) / 2.5)) if T <= 2.0 else 0.0
    snd = float(getattr(weather, "snow_depth_norm", 0.0))
    snf = float(getattr(weather, "snow_fresh_norm", 0.0))
    sn = max(snd, snf * 0.85)
    if profile_key == "cyclist":
        base = float(getattr(weather, "snow_stairs_cyclist_base", 1.18))
        frz = float(getattr(weather, "snow_stairs_cyclist_frozen_boost", 1.22))
    else:
        base = float(getattr(weather, "snow_stairs_ped_base", 1.08))
        frz = float(getattr(weather, "snow_stairs_ped_frozen_boost", 1.12))
    m = 1.0 + ss * sn * (base - 1.0)
    if snf > 0.08 and T <= 0.5:
        m *= 1.0 + ss * near_freeze * (frz - 1.0)
    sig = getattr(weather, "normalized_signals", None) or {}
    wcw = float(sig.get("wc_wet_slip", 0.0))
    wcm = float(sig.get("wc_mixed", 0.0))
    w_st = float(getattr(weather, "wc_wet_stairs_amp", 0.14))
    w_mx = float(getattr(weather, "wc_mixed_surface_amp", 0.18))
    m *= 1.0 + ss * sn * (w_st * wcw + 0.62 * w_mx * wcm)
    return max(1.0, min(1.62, m))


def wet_surface_route_penalty(surface_effective: Any, weather: WeatherWeightParams) -> float:
    """Множитель к физической стоимости ребра: дождь бьёт по плохому покрытию, асфальт почти не трогаем."""
    if not weather.enabled:
        return 1.0
    sig = weather.normalized_signals or {}
    rn = float(sig.get("rain_norm", 0.0))
    hn = float(sig.get("humidity_norm", 0.0))
    wet_env = max(rn, hn * 0.82)
    if wet_env <= 1e-6:
        return 1.0
    tier = _surface_wet_route_tier(surface_effective)
    t0 = float(getattr(weather, "phys_wet_tier0_cap", 0.016))
    t1 = float(getattr(weather, "phys_wet_tier1_coef", 0.048))
    t2 = float(getattr(weather, "phys_wet_tier2_coef", 0.12))
    if tier == 0:
        extra = wet_env * t0
    elif tier == 1:
        extra = wet_env * t1
    else:
        slip = wet_surface_edge_slip_factor(surface_effective)
        extra = wet_env * (t2 * (0.55 + 0.45 * slip))
    return max(1.0, min(1.28, 1.0 + extra))


def _parse_maxspeed_kmh(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if not s or s in ("none", "walk", "signals"):
        return None
    try:
        v = float(s.replace(" mph", "").split()[0])
        if "mph" in s:
            return v * 1.60934
        return v
    except (TypeError, ValueError):
        return None


def weather_edge_stress_factor(
    edge_data: dict, weather: WeatherWeightParams
) -> float:
    """Погодная поправка stress на уровне ребра (дождь×покрытие, ветер×открытость, LTS×скорость)."""
    if not weather.enabled:
        return 1.0
    sig = weather.normalized_signals or {}
    rn = float(sig.get("rain_norm", 0.0))
    wn = float(sig.get("wind_norm", 0.0))
    gn = float(sig.get("gust_norm", 0.0))
    wind_comb = max(wn, gn * 0.85)

    O = _clamp01(float(edge_data.get("thermal_open_sky_share") or 0.5) or 0.5)
    B = _clamp01(float(edge_data.get("thermal_building_shade_share") or 0.0) or 0.0)
    C = _clamp01(float(edge_data.get("thermal_covered_share") or 0.0) or 0.0)
    w1 = float(getattr(weather, "heat_wind_exp_w1", 0.45))
    w2 = float(getattr(weather, "heat_wind_exp_w2", 0.35))
    w3 = float(getattr(weather, "heat_wind_exp_w3", 0.35))
    wind_exp_base = _clamp01(w1 * O + w2 * (1.0 - B) + w3 * (1.0 - C))
    pack = _wind_dir_edge_pack(edge_data, weather)
    shelter_mult = 1.0
    if pack is not None:
        along01, cross01, _delta, _wah, _wcs = pack
        sa = float(getattr(weather, "stress_wind_along_open_amp", 0.32))
        wind_exp = _clamp01(
            wind_exp_base * (1.0 + sa * along01 * wind_comb)
        )
        sc = float(getattr(weather, "stress_wind_cross_shelter_amp", 0.40))
        shelter_mult = 1.0 + sc * cross01 * wind_comb
    else:
        wind_exp = wind_exp_base

    slip = wet_surface_edge_slip_factor(edge_data.get("surface_effective"))
    lts = float(edge_data.get("stress_lts", 1.5) or 1.5)
    lts_excess = max(0.0, min(1.0, (lts - 1.0) / 3.0))
    ms_kmh = _parse_maxspeed_kmh(edge_data.get("maxspeed"))
    fast = 0.0
    if ms_kmh is not None:
        if ms_kmh >= 50.0:
            fast = 1.0
        elif ms_kmh >= 40.0:
            fast = 0.5

    kr = float(getattr(weather, "stress_edge_rain_slip", 0.22))
    kw = float(getattr(weather, "stress_edge_wind_open", 0.20))
    kl = float(getattr(weather, "stress_edge_lts_fast", 0.17))
    ks = float(getattr(weather, "stress_edge_building_shelter", 0.11))
    rs = float(getattr(weather, "weather_response_scale", 1.0) or 1.0)

    wet_bad = rn * slip * kr
    wind_bad = wind_comb * wind_exp * kw
    lts_bad = lts_excess * (1.0 + fast * 0.95) * (0.05 + rn * 0.2) * kl
    shelter = B * wind_comb * ks * shelter_mult
    snow_bad = 0.0
    ss = float(getattr(weather, "snow_model_strength", 0.0))
    if ss > 1e-9:
        ksurf = float(getattr(weather, "stress_edge_snow_surface", 0.22))
        kopen = float(getattr(weather, "stress_edge_snow_open", 0.18))
        kstairs = float(getattr(weather, "stress_edge_snow_stairs", 0.16))
        snd = float(sig.get("snow_depth_norm", 0.0))
        snf = float(sig.get("snow_fresh_norm", 0.0))
        sn = max(snd, snf * 0.88)
        tier = _surface_wet_route_tier(edge_data.get("surface_effective"))
        surf_w = 0.35 if tier == 0 else 0.78 if tier == 1 else 1.0
        hw = str(edge_data.get("highway") or "").strip().lower()
        stair_e = 1.0 if hw == "steps" else 0.0
        snow_bad = (
            ksurf * sn * ss * surf_w * wet_surface_edge_slip_factor(
                edge_data.get("surface_effective")
            )
            + kopen * sn * ss * O * wind_comb
            + kstairs * sn * ss * stair_e
        )
        wcm = float(sig.get("wc_mixed", 0.0))
        wc_mix_amp = float(getattr(weather, "wc_mixed_snow_stress_amp", 0.16))
        snow_bad += wc_mix_amp * wcm * sn * ss * (0.42 + 0.58 * surf_w)
        wcf = float(sig.get("wc_fog", 0.0))
        if wcf > 1e-9:
            snow_bad += 0.09 * wcf * sn * ss * O
    raw = 1.0 + rs * (wet_bad + wind_bad + lts_bad - shelter + snow_bad)
    lo = float(getattr(weather, "stress_edge_factor_min", 0.82))
    hi = float(getattr(weather, "stress_edge_factor_max", 1.48))
    return max(lo, min(hi, raw))


def continuous_heat_edge_weather_factor(
    edge_data: dict, weather: WeatherWeightParams
) -> float:
    """Непрерывный множитель микроклимата для тепловой компоненты (без hot/cold/neutral)."""
    O = float(edge_data.get("thermal_open_sky_share") or 0.5) or 0.5
    O = _clamp01(O)
    B = _clamp01(float(edge_data.get("thermal_building_shade_share") or 0.0) or 0.0)
    V = _clamp01(float(edge_data.get("thermal_vegetation_shade_share") or 0.0) or 0.0)
    C = _clamp01(float(edge_data.get("thermal_covered_share") or 0.0) or 0.0)

    se = edge_data.get("surface_effective")
    wet = wet_surface_edge_slip_factor(se)
    rs = float(getattr(weather, "weather_response_scale", 1.0) or 1.0)

    w1 = float(getattr(weather, "heat_wind_exp_w1", 0.45))
    w2 = float(getattr(weather, "heat_wind_exp_w2", 0.35))
    w3 = float(getattr(weather, "heat_wind_exp_w3", 0.35))
    wind_exp_base = _clamp01(w1 * O + w2 * (1.0 - B) + w3 * (1.0 - C))
    pack = _wind_dir_edge_pack(edge_data, weather)
    b_build = 1.0
    if pack is not None:
        along01, cross01, _delta, wind_act, _wcs = pack
        ha = float(getattr(weather, "heat_wind_along_open_amp", 0.28))
        wind_exp = _clamp01(
            wind_exp_base * (1.0 + ha * along01 * wind_act * O)
        )
        hc = float(getattr(weather, "heat_wind_cross_build_amp", 0.35))
        hd = float(getattr(weather, "heat_wind_along_build_damp", 0.24))
        b_build = 1.0 + hc * cross01 * wind_act - hd * along01 * wind_act
        b_build = max(0.45, min(1.55, b_build))
    else:
        wind_exp = wind_exp_base

    k1 = float(getattr(weather, "heat_edge_k_open", 0.66))
    k2 = float(getattr(weather, "heat_edge_k_tree", 0.54))
    k3 = float(getattr(weather, "heat_edge_k_building", 0.50))
    k4 = float(getattr(weather, "heat_edge_k_covered", 0.16))
    k5 = float(getattr(weather, "heat_edge_k_wet", 0.28))
    k6 = float(getattr(weather, "heat_edge_k_wind", 0.30))

    sig = getattr(weather, "normalized_signals", None) or {}
    rn = _clamp01(float(sig.get("rain_norm", 0.0)))
    exn = _clamp01(float(sig.get("extreme_heat_norm", 0.0)))
    rain_open_amp = 1.0 + float(
        getattr(weather, "heat_edge_rain_open_mult", 0.48)
    ) * rn
    rain_build_amp = 1.0 + float(
        getattr(weather, "heat_edge_rain_building_mult", 0.56)
    ) * rn
    rain_wind_exp_amp = 1.0 + float(
        getattr(weather, "heat_edge_rain_wind_exp_mult", 0.22)
    ) * rn
    k1_eff = k1 * rain_open_amp * (
        1.0 + float(getattr(weather, "heat_extreme_open_route_gain", 0.22)) * exn
    )
    k3_eff = k3 * rain_build_amp
    k6_eff = k6 * rain_wind_exp_amp

    osp = float(getattr(weather, "open_sky_penalty", 1.0))
    tsb = float(getattr(weather, "tree_shade_bonus", 1.0))
    bsb = float(getattr(weather, "building_shade_bonus", 1.0))
    cbn = float(getattr(weather, "covered_bonus", 1.0))
    wsp = float(getattr(weather, "wet_surface_penalty", 1.0))
    wop = float(getattr(weather, "wind_open_penalty", 1.0))

    syn = float(getattr(weather, "heat_open_wet_synergy", 0.14))
    wet_scale = float(getattr(weather, "heat_wet_surface_edge_bad_max", 0.85))
    wet_eff = _clamp01(wet * wet_scale)

    wt = float(getattr(weather, "winter_heat_tree_scale", 1.0))
    wo = float(getattr(weather, "winter_heat_open_scale", 1.0))
    wwind = float(getattr(weather, "winter_heat_wind_scale", 1.0))

    raw = (
        1.0
        + rs * k1_eff * osp * O * wo
        - rs
        * k2
        * tsb
        * V
        * wt
        * (1.0 + float(getattr(weather, "heat_extreme_tree_route_gain", 0.18)) * exn)
        - rs * k3_eff * bsb * B * b_build
        - rs * k4 * cbn * C
        + rs * k5 * wsp * wet_eff
        + rs * k6_eff * wop * wind_exp * wwind
        + rs * syn * O * wet_eff * wind_exp * max(1.0, (wo + wwind) * 0.5)
    )
    lo = float(getattr(weather, "heat_edge_factor_min", 0.65))
    hi = float(getattr(weather, "heat_edge_factor_max", 1.75))
    return max(lo, min(hi, raw))


def effective_edge_components(
    edge_data: dict,
    profile_key: str,
    time_slot_key: str,
    heat_context_mult: float,
    weather: Optional[WeatherWeightParams],
    *,
    physical_weight_key: Optional[str] = None,
) -> EffectiveEdgeComponents:
    """Физика без green_route, heat, stress и явные критерии (таблица критериев)."""
    phys_k = physical_weight_key or f"weight_{profile_key}_full"
    heat_k = f"heat_{time_slot_key}"
    phys = coerce_edge_weight_numeric(
        edge_data.get(phys_k),
        fallback=float("inf"),
    )
    h = coerce_edge_weight_numeric(edge_data.get(heat_k), fallback=0.0)
    hm = float(heat_context_mult) if math.isfinite(heat_context_mult) else 1.0
    st = coerce_edge_weight_numeric(edge_data.get("stress_cost"), fallback=0.0)
    if not weather or not weather.enabled:
        return EffectiveEdgeComponents(phys, h * hm, st, EdgeRouteCriteriaFactors())
    factors, phys_eff, heat_eff, st_eff = compute_edge_route_criteria(
        edge_data,
        profile_key,
        time_slot_key,
        heat_context_mult,
        weather,
        physical_weight_key=physical_weight_key,
    )
    return EffectiveEdgeComponents(phys_eff, heat_eff, st_eff, factors)


def _edge_gradient_abs_raw(edge_data: dict) -> float:
    """Максимальный уклон по реальному ``gradient_raw`` (без клипа 50% для весов)."""
    raw = edge_data.get("gradient_raw")
    if raw is None:
        raw = edge_data.get("gradient", 0.0)
    return abs(float(raw or 0.0))


def _edge_gradient_abs_capped(edge_data: dict) -> float:
    """Доля 0…1 для подписей по сегменту на карте (мягкий потолок от артефактов DEM)."""
    return min(_edge_gradient_abs_raw(edge_data), float(MAX_ROUTE_GRADIENT_DISPLAY))


def _edge_gradient_abs_for_display(edge_data: dict) -> float:
    """Уклон для подписей сегментов на карте (не для агрегата «макс. уклон» маршрута)."""
    return _edge_gradient_abs_capped(edge_data)


# Минимальная длина ребра (м), чтобы |Δh|/L не взрывался от шума DEM на коротких звеньях.
_MIN_GRADIENT_STAT_SEGMENT_M: float = 15.0
# Сегменты с |Δh|/L выше порога не участвуют в макс./среднем (ошибка высот или лестница-артефакт).
_ABSURD_SEGMENT_GRADIENT_RATIO: float = 0.85


def _segment_gradient_ratio_for_stats(edge_data: dict) -> Optional[float]:
    """|Δh|/L по ребру; None если сегмент не подходит для агрегатов."""
    ln = float(edge_data.get("length", 0) or 0)
    if ln < _MIN_GRADIENT_STAT_SEGMENT_M:
        return None
    ed = float(edge_data.get("elevation_diff", 0) or 0)
    if not math.isfinite(ed) or not math.isfinite(ln) or ln <= 0:
        return None
    r = abs(ed) / ln
    if not math.isfinite(r):
        return None
    if r >= _ABSURD_SEGMENT_GRADIENT_RATIO:
        return None
    return r


def _route_max_gradient_pct_from_ratios(ratios: List[float]) -> float:
    """Максимум по валидным сегментам; ослабление единичного выброса относительно остальных."""
    if not ratios:
        return 0.0
    s = sorted(ratios)
    n = len(s)
    peak = s[-1]
    if n >= 2 and peak > 0.32 and (peak - s[-2]) > 0.14:
        peak = s[-2]
    pct = peak * 100.0
    return round(min(pct, 55.0), 1)


def _route_avg_gradient_pct_from_ratios(ratios: List[float]) -> float:
    if not ratios:
        return 0.0
    return round((sum(ratios) / len(ratios)) * 100.0, 1)


def coerce_edge_weight_numeric(val: Any, *, fallback: float = float("inf")) -> float:
    """Привести вес ребра к float (GraphML/OSMnx иногда дают строки)."""
    if val is None:
        return fallback
    if isinstance(val, bool):
        return fallback
    if isinstance(val, numbers.Real):
        x = float(val)
        return x if math.isfinite(x) else fallback
    if isinstance(val, str):
        s = val.strip().replace(",", ".")
        if not s:
            return fallback
        try:
            x = float(s)
            return x if math.isfinite(x) else fallback
        except ValueError:
            return fallback
    try:
        x = float(val)
        return x if math.isfinite(x) else fallback
    except (TypeError, ValueError):
        return fallback


# После GraphML (кэш коридора) float часто оказываются строками — ломает сравнения и Dijkstra.
_GRAPHML_NUMERIC_EDGE_ATTRS = frozenset(
    {
        "gradient",
        "gradient_raw",
        "elevation_diff",
        "edge_climb",
        "edge_descent",
        "gradient_percent",
        "trees_percent",
        "grass_percent",
        "green_percent",
        "green_coeff",
        "edge_bearing_deg",
        "stress_lts",
        "stress_cost",
        "stress_segment_cost",
        "stress_intersection_cost",
        "stress_intersection_score",
        "thermal_open_sky_share",
        "thermal_vegetation_shade_share",
        "thermal_building_shade_share",
        "thermal_covered_share",
        "street_width_proxy_m",
        "urban_canyon_score",
        "solar_shade_potential",
        *[f"heat_{s.key}" for s in TIME_SLOTS],
        *[f"heat_exposure_{s.key}" for s in TIME_SLOTS],
    }
)


def sanitize_multidigraph_routing_weights(G: nx.MultiDiGraph) -> None:
    """На рёбрах привести числовые поля к float (GraphML / OSMnx дают строки)."""
    for _u, _v, _k, d in G.edges(data=True, keys=True):
        if "length" in d:
            ln = coerce_edge_weight_numeric(d["length"], fallback=float("nan"))
            if math.isnan(ln):
                ln = 1.0
            d["length"] = ln
        for attr in list(d.keys()):
            if attr != "length" and not str(attr).startswith("weight_"):
                continue
            if attr == "length":
                continue
            d[attr] = coerce_edge_weight_numeric(
                d.get(attr),
                fallback=coerce_edge_weight_numeric(
                    d.get("length", 1.0), fallback=1.0
                ),
            )
        for attr in list(d.keys()):
            sa = str(attr)
            if (
                sa.startswith("heat_")
                or sa.startswith("thermal_")
                or sa.startswith("stress_")
                or sa in ("edge_bearing_deg",)
            ):
                d[attr] = coerce_edge_weight_numeric(d.get(attr), fallback=0.0)
        for attr in _GRAPHML_NUMERIC_EDGE_ATTRS:
            if attr not in d:
                continue
            d[attr] = coerce_edge_weight_numeric(d.get(attr), fallback=0.0)


def _coords_from_shapely_linear(geom: Any) -> List[Tuple[float, float]]:
    """Вершины (lon, lat) для LineString / MultiLineString / LinearRing / GeometryCollection.

    У рёбер OSM иногда бывает ``MultiLineString`` — у него нет ``.coords``, из-за чего
    падал весь ``POST /alternatives`` с 500.
    """
    if geom is None:
        return []
    gt = getattr(geom, "geom_type", "") or ""
    try:
        if gt in ("LineString", "LinearRing"):
            return list(geom.coords)
        if gt == "MultiLineString":
            out: List[Tuple[float, float]] = []
            for g in geom.geoms:
                out.extend(list(g.coords))
            return out
        if gt == "GeometryCollection":
            acc: List[Tuple[float, float]] = []
            for g in geom.geoms:
                acc.extend(_coords_from_shapely_linear(g))
            return acc
        if hasattr(geom, "coords"):
            return list(geom.coords)
    except Exception as exc:
        logger.warning(
            "Не удалось разобрать geometry ребра (type=%s): %s",
            gt or type(geom).__name__,
            exc,
        )
    return []


def _first_value(val: Any) -> Any:
    if isinstance(val, list):
        return val[0] if val else None
    return val


def _raw_osm_surface_from_edge(d: Dict[str, Any]) -> str:
    """Тег surface из OSM (без эвристик): ``surface_osm`` или устаревшее ``surface``."""
    for key in ("surface_osm", "surface"):
        v = _first_value(d.get(key))
        if v is None:
            continue
        s = str(v).strip().lower()
        if s and s != "nan":
            return s
    return ""


def _effective_surface_from_edge(d: Dict[str, Any]) -> str:
    """Итоговая метка покрытия для статистики вдоль маршрута."""
    v = _first_value(d.get("surface_effective"))
    if v is not None:
        s = str(v).strip().lower()
        if s and s != "nan":
            return s
    r = _raw_osm_surface_from_edge(d)
    return r if r else "unknown"


class RouteResult:
    """Результат поиска маршрута (узлы + рёбра с ключами).

    Attributes:
        nodes: последовательность узлов маршрута.
        edges: список ``(u, v, key)`` — конкретные рёбра MultiDiGraph,
               по которым прошёл алгоритм (с учётом параллельных рёбер).
    """

    def __init__(
        self,
        nodes: List[int],
        edges: List[Tuple[int, int, int]],
        start_node: int,
        end_node: int,
    ) -> None:
        self.nodes = nodes
        self.edges = edges
        self.start_node = start_node
        self.end_node = end_node

    @property
    def edge_count(self) -> int:
        return len(self.edges)


class RouteService:
    """Поиск маршрутов и расчёт метрик."""

    @staticmethod
    def _collapse_multidigraph_to_digraph(
        G: nx.MultiDiGraph, weight_key: str
    ) -> nx.DiGraph:
        """``shortest_simple_paths`` в NetworkX не работает с MultiDiGraph.

        Между каждой парой (u, v) оставляем одно ребро с минимальным *weight_key*
        (как при выборе key в :meth:`_resolve_edge_keys`). Узлы без рёбер не копируются.
        """
        H = nx.DiGraph()
        for u, v, _k, d in G.edges(data=True, keys=True):
            w = coerce_edge_weight_numeric(d.get(weight_key), fallback=float("inf"))
            if not math.isfinite(w):
                continue
            if H.has_edge(u, v):
                if H.edges[u, v][weight_key] > w:
                    H.edges[u, v][weight_key] = w
            else:
                H.add_edge(u, v, **{weight_key: w})
        return H

    @staticmethod
    def _resolve_edge_keys(
        G: nx.MultiDiGraph,
        nodes: List[int],
        weight_key: str,
    ) -> List[Tuple[int, int, int]]:
        """Для каждой пары узлов определить key с минимальным весом."""
        edges = []
        for i in range(len(nodes) - 1):
            u, v = nodes[i], nodes[i + 1]
            best_key = min(
                G[u][v],
                key=lambda k: coerce_edge_weight_numeric(
                    G[u][v][k].get(weight_key), fallback=float("inf")
                ),
            )
            edges.append((u, v, best_key))
        return edges

    def find_route(
        self,
        G: nx.MultiDiGraph,
        start_coords: Tuple[float, float],
        end_coords: Tuple[float, float],
        weight_key: str,
    ) -> RouteResult:
        """Кратчайший маршрут. Бросает :exc:`RouteNotFoundError` если пути нет."""
        start = ox.distance.nearest_nodes(
            G, X=start_coords[1], Y=start_coords[0]
        )
        end = ox.distance.nearest_nodes(
            G, X=end_coords[1], Y=end_coords[0]
        )
        try:
            nodes = nx.shortest_path(
                G, source=start, target=end, weight=weight_key
            )
            edges = self._resolve_edge_keys(G, nodes, weight_key)
            return RouteResult(nodes, edges, start, end)
        except nx.NetworkXNoPath:
            raise RouteNotFoundError(weight_key)

    @staticmethod
    def find_next_simple_path_by_length(
        G: nx.MultiDiGraph,
        start_coords: Tuple[float, float],
        end_coords: Tuple[float, float],
        exclude_signatures: set,
        max_hops: int = 200,
        max_candidates: int = 12,
    ) -> Optional[RouteResult]:
        """Следующий простой путь по весу ``length``, не входящий в ``exclude_signatures``."""
        sn = ox.distance.nearest_nodes(
            G, X=start_coords[1], Y=start_coords[0]
        )
        en = ox.distance.nearest_nodes(
            G, X=end_coords[1], Y=end_coords[0]
        )
        try:
            H = RouteService._collapse_multidigraph_to_digraph(G, "length")
            gen = nx.shortest_simple_paths(H, sn, en, weight="length")
            next(gen)
            n = 0
            for nodes in gen:
                n += 1
                if n > max_candidates:
                    break
                if len(nodes) > max_hops:
                    continue
                edges = RouteService._resolve_edge_keys(G, nodes, "length")
                sig = tuple(edges)
                if sig in exclude_signatures:
                    continue
                return RouteResult(nodes, edges, sn, en)
        except (nx.NetworkXNoPath, StopIteration):
            pass
        return None

    def find_route_safe(
        self,
        G: nx.MultiDiGraph,
        start_coords: Tuple[float, float],
        end_coords: Tuple[float, float],
        weight_key: str,
    ) -> Optional[RouteResult]:
        """Поиск без исключения — возвращает ``None`` если пути нет."""
        try:
            return self.find_route(G, start_coords, end_coords, weight_key)
        except RouteNotFoundError:
            logger.warning("Маршрут не найден (weight='%s')", weight_key)
            return None

    @staticmethod
    def effective_heat_beta(
        pref: RoutingPreferenceProfile,
        weather: Optional[WeatherWeightParams] = None,
    ) -> float:
        """β с динамическим усилением для extreme-heat режимов heat/heat_stress."""
        beta = float(pref.beta)
        if (
            weather is not None
            and getattr(weather, "enabled", False)
            and pref.key in ("thermal_physical_base", "heat_stress_physical_base")
        ):
            beta *= float(getattr(weather, "heat_extreme_route_mult", 1.0) or 1.0)
        return beta

    @staticmethod
    def combined_edge_weight(
        edge_data: dict,
        profile_key: str,
        time_slot_key: str,
        pref: RoutingPreferenceProfile,
        *,
        heat_context_mult: float = 1.0,
        weather: Optional[WeatherWeightParams] = None,
        physical_weight_key: Optional[str] = None,
    ) -> float:
        """α·physical' + β·heat' + γ·stress' (без поворотов). Погода — в effective_edge_components."""
        ec = effective_edge_components(
            edge_data,
            profile_key,
            time_slot_key,
            heat_context_mult,
            weather,
            physical_weight_key=physical_weight_key,
        )
        phys_r = ec.phys_eff * ec.criteria.green_route_factor
        beta_eff = RouteService.effective_heat_beta(pref, weather)
        c = pref.alpha * phys_r + beta_eff * ec.heat_eff + pref.gamma * ec.st_eff
        if not math.isfinite(c) or c <= 0:
            return float("inf")
        return float(c)

    @staticmethod
    def _multigraph_combined_weight_fn(
        profile_key: str,
        time_slot_key: str,
        pref: RoutingPreferenceProfile,
        *,
        heat_context_mult: float = 1.0,
        weather: Optional[WeatherWeightParams] = None,
        physical_weight_key: Optional[str] = None,
    ) -> Callable[..., float]:
        def wf(_u: Any, _v: Any, dk: Dict[int, dict]) -> float:
            costs = []
            for _k, ed in dk.items():
                costs.append(
                    RouteService.combined_edge_weight(
                        ed,
                        profile_key,
                        time_slot_key,
                        pref,
                        heat_context_mult=heat_context_mult,
                        weather=weather,
                        physical_weight_key=physical_weight_key,
                    )
                )
            return min(costs) if costs else float("inf")

        return wf

    def find_route_combined(
        self,
        G: nx.MultiDiGraph,
        start_coords: Tuple[float, float],
        end_coords: Tuple[float, float],
        profile_key: str,
        time_slot_key: str,
        pref: RoutingPreferenceProfile,
        *,
        heat_context_mult: float = 1.0,
        weather: Optional[WeatherWeightParams] = None,
        physical_weight_key: Optional[str] = None,
    ) -> RouteResult:
        """Кратчайший путь по комбинированному весу (без штрафа за поворот)."""
        start = ox.distance.nearest_nodes(
            G, X=start_coords[1], Y=start_coords[0]
        )
        end = ox.distance.nearest_nodes(
            G, X=end_coords[1], Y=end_coords[0]
        )
        wf = self._multigraph_combined_weight_fn(
            profile_key,
            time_slot_key,
            pref,
            heat_context_mult=heat_context_mult,
            weather=weather,
            physical_weight_key=physical_weight_key,
        )
        try:
            nodes = nx.shortest_path(G, source=start, target=end, weight=wf)
        except nx.NetworkXNoPath:
            raise RouteNotFoundError(f"combined:{time_slot_key}")
        edges = self._resolve_edge_keys_combined(
            G,
            nodes,
            profile_key,
            time_slot_key,
            pref,
            heat_context_mult=heat_context_mult,
            weather=weather,
            physical_weight_key=physical_weight_key,
        )
        return RouteResult(nodes, edges, start, end)

    @staticmethod
    def _resolve_edge_keys_combined(
        G: nx.MultiDiGraph,
        nodes: List[int],
        profile_key: str,
        time_slot_key: str,
        pref: RoutingPreferenceProfile,
        *,
        heat_context_mult: float = 1.0,
        weather: Optional[WeatherWeightParams] = None,
        physical_weight_key: Optional[str] = None,
    ) -> List[Tuple[int, int, int]]:
        edges: List[Tuple[int, int, int]] = []
        for i in range(len(nodes) - 1):
            u, v = nodes[i], nodes[i + 1]
            best_key = min(
                G[u][v],
                key=lambda k: RouteService.combined_edge_weight(
                    G[u][v][k],
                    profile_key,
                    time_slot_key,
                    pref,
                    heat_context_mult=heat_context_mult,
                    weather=weather,
                    physical_weight_key=physical_weight_key,
                ),
            )
            edges.append((u, v, best_key))
        return edges

    @staticmethod
    def _turn_penalty_at_node(
        bd_in: float,
        bd_out: float,
        pref: RoutingPreferenceProfile,
    ) -> float:
        ta = angle_diff_deg(float(bd_in or 0.0), float(bd_out or 0.0))
        if ta < TURN_ANGLE_THRESHOLD_DEG:
            return 0.0
        return float(
            pref.delta
            * TURN_PENALTY_BASE
            * min(1.0, (ta - TURN_ANGLE_THRESHOLD_DEG) / 50.0)
        )

    def find_route_combined_with_turns(
        self,
        G: nx.MultiDiGraph,
        start_coords: Tuple[float, float],
        end_coords: Tuple[float, float],
        profile_key: str,
        time_slot_key: str,
        pref: RoutingPreferenceProfile,
        *,
        heat_context_mult: float = 1.0,
        weather: Optional[WeatherWeightParams] = None,
        physical_weight_key: Optional[str] = None,
    ) -> RouteResult:
        """Dijkstra в пространстве состояний (узел, входящее ребро) — штраф за поворот."""
        start = ox.distance.nearest_nodes(
            G, X=start_coords[1], Y=start_coords[0]
        )
        end = ox.distance.nearest_nodes(
            G, X=end_coords[1], Y=end_coords[0]
        )

        State = Tuple[int, Optional[Tuple[int, int, int]]]
        dist: Dict[State, float] = {}
        pred: Dict[State, Optional[State]] = {}
        c = count()
        fringe: List[Tuple[float, int, State]] = []

        s0: State = (start, None)
        dist[s0] = 0.0
        pred[s0] = None
        heappush(fringe, (0.0, next(c), s0))

        final_state: Optional[State] = None

        while fringe:
            d, _, st_v = heappop(fringe)
            if d > dist.get(st_v, float("inf")):
                continue
            v, inc_e = st_v
            if v == end:
                final_state = st_v
                break

            for _vv, w, k in G.out_edges(v, keys=True):
                ed = G[v][w][k]
                base = self.combined_edge_weight(
                    ed,
                    profile_key,
                    time_slot_key,
                    pref,
                    heat_context_mult=heat_context_mult,
                    weather=weather,
                    physical_weight_key=physical_weight_key,
                )
                if not math.isfinite(base):
                    continue
                turn_p = 0.0
                if inc_e is not None:
                    pu, pv, pk = inc_e
                    bd_in = float(G[pu][pv][pk].get("edge_bearing_deg", 0.0))
                    bd_out = float(ed.get("edge_bearing_deg", 0.0))
                    turn_p = self._turn_penalty_at_node(bd_in, bd_out, pref)
                nd = d + base + turn_p
                new_prev: Tuple[int, int, int] = (v, w, k)
                st_w: State = (w, new_prev)
                if nd < dist.get(st_w, float("inf")):
                    dist[st_w] = nd
                    pred[st_w] = st_v
                    heappush(fringe, (nd, next(c), st_w))

        if final_state is None:
            raise RouteNotFoundError(f"combined_turns:{time_slot_key}")

        edges_rev: List[Tuple[int, int, int]] = []
        cur: Optional[State] = final_state
        while cur is not None and cur[1] is not None:
            edges_rev.append(cur[1])
            cur = pred.get(cur)
        edges_rev.reverse()
        if not edges_rev:
            raise RouteNotFoundError(f"combined_turns_empty:{time_slot_key}")
        nodes = [edges_rev[0][0]]
        for u, v, _k in edges_rev:
            nodes.append(v)
        return RouteResult(nodes, edges_rev, start, end)

    @staticmethod
    def count_significant_turns(
        G: nx.MultiDiGraph,
        route: Optional[RouteResult],
        *,
        threshold_deg: float = TURN_ANGLE_THRESHOLD_DEG,
    ) -> int:
        """Число поворотов между рёбрами (по разнице азимутов)."""
        if route is None or route.edge_count < 2:
            return 0
        n = 0
        for i in range(route.edge_count - 1):
            d1 = G.edges[route.edges[i]]
            d2 = G.edges[route.edges[i + 1]]
            b1 = float(d1.get("edge_bearing_deg", 0.0) or 0.0)
            b2 = float(d2.get("edge_bearing_deg", 0.0) or 0.0)
            if angle_diff_deg(b1, b2) >= threshold_deg:
                n += 1
        return n

    @staticmethod
    def route_segment_costs(
        G: nx.MultiDiGraph,
        route: Optional[RouteResult],
        profile_key: str,
        time_slot_key: str,
        *,
        heat_context_mult: float = 1.0,
        weather: Optional[WeatherWeightParams] = None,
        physical_weight_key: Optional[str] = None,
    ) -> Dict[str, float]:
        """Суммы physical, heat (с κ), stress по выбранному слоту."""
        z: Dict[str, float] = {
            "physical_base": 0.0,
            "physical": 0.0,
            "green_cost": 0.0,
            "heat": 0.0,
            "heat_raw": 0.0,
            "stress": 0.0,
            "stress_segment": 0.0,
            "stress_intersection": 0.0,
            "length_m": 0.0,
            "stress_route_regime_factor": 0.0,
            "_criteria_weight_len": 0.0,
        }
        for _ck in CRITERION_KEYS:
            z[f"route_mean_{_ck}"] = 0.0
        if route is None:
            return z
        hm = float(heat_context_mult) if math.isfinite(heat_context_mult) else 1.0
        phys_k = physical_weight_key or f"weight_{profile_key}_full"
        heat_k = f"heat_{time_slot_key}"
        sw = 1.0
        if weather and weather.enabled:
            sw = float(weather.mults.stress)
            z["stress_route_regime_factor"] = float(
                getattr(weather, "stress_route_regime_factor", 1.0) or 1.0
            )
        for i in range(route.edge_count):
            d = G.edges[route.edges[i]]
            hr = coerce_edge_weight_numeric(d.get(heat_k), fallback=0.0)
            z["heat_raw"] += hr
            if weather and weather.enabled:
                ec = effective_edge_components(
                    d,
                    profile_key,
                    time_slot_key,
                    heat_context_mult,
                    weather,
                    physical_weight_key=physical_weight_key,
                )
                ln = float(d.get("length", 0.0) or 0.0)
                z["_criteria_weight_len"] += ln
                for _ck in CRITERION_KEYS:
                    z[f"route_mean_{_ck}"] += ln * float(getattr(ec.criteria, _ck))
                pe_base = ec.phys_eff
                pe = pe_base * ec.criteria.green_route_factor
                z["physical_base"] += pe_base
                z["green_cost"] += pe - pe_base
                z["physical"] += pe
                z["heat"] += ec.heat_eff
                z["stress"] += ec.st_eff
                z["stress_segment"] += coerce_edge_weight_numeric(
                    d.get("stress_segment_cost"), fallback=0.0
                ) * sw
                z["stress_intersection"] += coerce_edge_weight_numeric(
                    d.get("stress_intersection_cost"), fallback=0.0
                ) * sw
            else:
                pe_base = coerce_edge_weight_numeric(
                    d.get(phys_k), fallback=0.0
                )
                z["physical_base"] += pe_base
                z["physical"] += pe_base
                z["heat"] += hr * hm
                z["stress"] += coerce_edge_weight_numeric(
                    d.get("stress_cost"), fallback=0.0
                )
                z["stress_segment"] += coerce_edge_weight_numeric(
                    d.get("stress_segment_cost"), fallback=0.0
                )
                z["stress_intersection"] += coerce_edge_weight_numeric(
                    d.get("stress_intersection_cost"), fallback=0.0
                )
            z["length_m"] += float(d.get("length", 0.0) or 0.0)
        wl = float(z.pop("_criteria_weight_len", 0.0) or 0.0)
        if wl > 1e-9:
            for _ck in CRITERION_KEYS:
                k = f"route_mean_{_ck}"
                z[k] = float(z[k]) / wl
        return z

    @staticmethod
    def route_combined_total(
        G: nx.MultiDiGraph,
        route: Optional[RouteResult],
        profile_key: str,
        time_slot_key: str,
        pref: RoutingPreferenceProfile,
        *,
        turn_count: int,
        heat_context_mult: float = 1.0,
        weather: Optional[WeatherWeightParams] = None,
        physical_weight_key: Optional[str] = None,
    ) -> float:
        """Итоговая комбинированная стоимость (с δ·turn)."""
        seg = RouteService.route_segment_costs(
            G,
            route,
            profile_key,
            time_slot_key,
            heat_context_mult=heat_context_mult,
            weather=weather,
            physical_weight_key=physical_weight_key,
        )
        turn_part = float(pref.delta * TURN_PENALTY_BASE * max(0, turn_count))
        beta_eff = RouteService.effective_heat_beta(pref, weather)
        return float(
            pref.alpha * seg["physical"]
            + beta_eff * seg["heat"]
            + pref.gamma * seg["stress"]
            + turn_part
        )

    @staticmethod
    def route_exposure_metrics(
        G: nx.MultiDiGraph,
        route: Optional[RouteResult],
        time_slot_key: str,
        *,
        exposure_threshold: float = 0.45,
    ) -> Dict[str, float]:
        """Длина участков с высокой экспозицией для слота."""
        if route is None:
            return {"exposed_high_length_m": 0.0, "avg_exposure": 0.0}
        hk = f"heat_exposure_{time_slot_key}"
        tlen = 0.0
        hi = 0.0
        wsum = 0.0
        for i in range(route.edge_count):
            d = G.edges[route.edges[i]]
            ln = float(d.get("length", 0.0) or 0.0)
            eu = float(d.get(hk, 0.0) or 0.0)
            tlen += ln
            wsum += eu * ln
            if eu >= exposure_threshold:
                hi += ln
        return {
            "exposed_high_length_m": float(hi),
            "avg_exposure": float(wsum / tlen) if tlen > 0 else 0.0,
        }

    @staticmethod
    def route_stress_levels(
        G: nx.MultiDiGraph,
        route: Optional[RouteResult],
        *,
        high_threshold: float = 2.5,
    ) -> Dict[str, float]:
        """Средний/макс LTS и доля длины с LTS ≥ порога."""
        if route is None:
            return {
                "avg_lts": 1.0,
                "max_lts": 1.0,
                "high_stress_fraction": 0.0,
            }
        lengths: List[float] = []
        lts: List[float] = []
        for i in range(route.edge_count):
            d = G.edges[route.edges[i]]
            ln = float(d.get("length", 0.0) or 0.0)
            lengths.append(ln)
            lts.append(float(d.get("stress_lts", 1.5) or 1.5))
        m = stress_metrics_for_route(lengths, lts, high_threshold=high_threshold)
        return {
            "avg_lts": m["avg_lts"],
            "max_lts": m["max_lts"],
            "high_stress_fraction": float(m["high_stress_fraction"]),
        }

    @staticmethod
    def route_shade_shares(
        G: nx.MultiDiGraph,
        route: Optional[RouteResult],
    ) -> Dict[str, float]:
        """Средневзвешенные по длине доли тени (растительность / здания)."""
        if route is None:
            return {"vegetation_shade_share": 0.0, "building_shade_share": 0.0}
        tlen = 0.0
        sv = 0.0
        sb = 0.0
        for i in range(route.edge_count):
            d = G.edges[route.edges[i]]
            ln = float(d.get("length", 0.0) or 0.0)
            tlen += ln
            sv += ln * float(d.get("thermal_vegetation_shade_share", 0.0) or 0.0)
            sb += ln * float(d.get("thermal_building_shade_share", 0.0) or 0.0)
        if tlen <= 0:
            return {"vegetation_shade_share": 0.0, "building_shade_share": 0.0}
        return {
            "vegetation_shade_share": float(sv / tlen),
            "building_shade_share": float(sb / tlen),
        }

    @staticmethod
    def route_shelter_length_weighted_averages(
        G: nx.MultiDiGraph,
        route: Optional["RouteResult"],
    ) -> Dict[str, float]:
        """Средние по длине: открытость, тень зданий, укрытие, «плохое мокрое» покрытие."""
        if route is None or route.edge_count <= 0:
            return {
                "route_open_sky_share": 0.0,
                "route_building_shade_share": 0.0,
                "route_covered_share": 0.0,
                "route_bad_wet_surface_share": 0.0,
                "route_winter_harsh_surface_share": 0.0,
                "route_mean_street_width_proxy_m": 0.0,
                "route_mean_urban_canyon_score": 0.0,
                "route_mean_solar_shade_potential": 0.0,
            }
        tlen = 0.0
        so = 0.0
        sb = 0.0
        sc = 0.0
        sw = 0.0
        sw_h = 0.0
        swidth = 0.0
        scan = 0.0
        sshade = 0.0
        for i in range(route.edge_count):
            d = G.edges[route.edges[i]]
            ln = float(d.get("length", 0.0) or 0.0)
            if ln <= 0:
                continue
            tlen += ln
            so += ln * float(d.get("thermal_open_sky_share", 0.5) or 0.5)
            sb += ln * float(d.get("thermal_building_shade_share", 0.0) or 0.0)
            sc += ln * float(d.get("thermal_covered_share", 0.0) or 0.0)
            sw += ln * wet_surface_edge_slip_factor(d.get("surface_effective"))
            sw_h += ln * _edge_winter_harsh_surface_indicator(d)
            swidth += ln * float(d.get("street_width_proxy_m", 18.0) or 18.0)
            scan += ln * float(d.get("urban_canyon_score", 0.0) or 0.0)
            sshade += ln * float(d.get("solar_shade_potential", 0.0) or 0.0)
        if tlen <= 0:
            return {
                "route_open_sky_share": 0.0,
                "route_building_shade_share": 0.0,
                "route_covered_share": 0.0,
                "route_bad_wet_surface_share": 0.0,
                "route_winter_harsh_surface_share": 0.0,
                "route_mean_street_width_proxy_m": 0.0,
                "route_mean_urban_canyon_score": 0.0,
                "route_mean_solar_shade_potential": 0.0,
            }
        return {
            "route_open_sky_share": float(so / tlen),
            "route_building_shade_share": float(sb / tlen),
            "route_covered_share": float(sc / tlen),
            "route_bad_wet_surface_share": float(sw / tlen),
            "route_winter_harsh_surface_share": float(sw_h / tlen),
            "route_mean_street_width_proxy_m": float(swidth / tlen),
            "route_mean_urban_canyon_score": float(scan / tlen),
            "route_mean_solar_shade_potential": float(sshade / tlen),
        }

    @staticmethod
    def route_wind_direction_metrics(
        G: nx.MultiDiGraph,
        route: Optional["RouteResult"],
        weather: Optional[WeatherWeightParams],
    ) -> Dict[str, float]:
        """Средние по длине и доли «опасного» ветра, если активен direction-aware режим."""
        z = {
            "route_wind_direction_aware": 0.0,
            "route_mean_wind_to_street_angle_deg": 0.0,
            "route_mean_heat_directional_wind_exp": 0.0,
            "route_mean_heat_building_wind_factor": 0.0,
            "route_frac_wind_along_open_hostile": 0.0,
            "route_frac_wind_cross_building_screen": 0.0,
        }
        if (
            route is None
            or route.edge_count <= 0
            or weather is None
            or not weather.enabled
        ):
            return z
        if not bool(getattr(weather, "wind_direction_available", False)):
            return z
        w1 = float(getattr(weather, "heat_wind_exp_w1", 0.45))
        w2 = float(getattr(weather, "heat_wind_exp_w2", 0.35))
        w3 = float(getattr(weather, "heat_wind_exp_w3", 0.35))
        tlen = 0.0
        s_ang = 0.0
        s_wexp = 0.0
        s_bb = 0.0
        hostile = 0.0
        screen = 0.0
        for i in range(route.edge_count):
            d = G.edges[route.edges[i]]
            ln = float(d.get("length", 0.0) or 0.0)
            if ln <= 0:
                continue
            O = _clamp01(float(d.get("thermal_open_sky_share") or 0.5) or 0.5)
            B = _clamp01(float(d.get("thermal_building_shade_share") or 0.0) or 0.0)
            C = _clamp01(float(d.get("thermal_covered_share") or 0.0) or 0.0)
            w_base = _clamp01(w1 * O + w2 * (1.0 - B) + w3 * (1.0 - C))
            pack = _wind_dir_edge_pack(d, weather)
            if pack is None:
                continue
            along01, cross01, delta_deg, wind_act, _wcs = pack
            ha = float(getattr(weather, "heat_wind_along_open_amp", 0.28))
            w_exp = _clamp01(w_base * (1.0 + ha * along01 * wind_act * O))
            hc = float(getattr(weather, "heat_wind_cross_build_amp", 0.35))
            hd = float(getattr(weather, "heat_wind_along_build_damp", 0.24))
            b_b = 1.0 + hc * cross01 * wind_act - hd * along01 * wind_act
            b_b = max(0.45, min(1.55, b_b))
            tlen += ln
            s_ang += ln * float(delta_deg)
            s_wexp += ln * float(w_exp)
            s_bb += ln * float(b_b)
            if along01 >= 0.52 and O >= 0.42 and wind_act >= 0.22:
                hostile += ln
            if cross01 >= 0.52 and B >= 0.14 and wind_act >= 0.22:
                screen += ln
        if tlen <= 0:
            return z
        return {
            "route_wind_direction_aware": 1.0,
            "route_mean_wind_to_street_angle_deg": float(s_ang / tlen),
            "route_mean_heat_directional_wind_exp": float(s_wexp / tlen),
            "route_mean_heat_building_wind_factor": float(s_bb / tlen),
            "route_frac_wind_along_open_hostile": float(hostile / tlen),
            "route_frac_wind_cross_building_screen": float(screen / tlen),
        }

    @staticmethod
    def count_stressful_intersections(
        G: nx.MultiDiGraph,
        route: Optional[RouteResult],
        *,
        score_threshold: float = 0.35,
    ) -> int:
        """Число рёбер с заметным стрессом пересечения."""
        if route is None:
            return 0
        n = 0
        for i in range(route.edge_count):
            d = G.edges[route.edges[i]]
            sc = float(d.get("stress_intersection_score", 0.0) or 0.0)
            if sc >= score_threshold:
                n += 1
        return n

    @staticmethod
    def count_high_stress_segments(
        G: nx.MultiDiGraph,
        route: Optional[RouteResult],
        *,
        lts_threshold: float = 2.5,
    ) -> int:
        """Число рёбер с высоким сегментным LTS."""
        if route is None:
            return 0
        n = 0
        for i in range(route.edge_count):
            d = G.edges[route.edges[i]]
            if float(d.get("stress_lts", 0.0) or 0.0) >= lts_threshold:
                n += 1
        return n

    # ------------------------------------------------------------------
    # Геометрия маршрута
    # ------------------------------------------------------------------

    @staticmethod
    def route_geometry(
        G: nx.MultiDiGraph, route: Optional["RouteResult"]
    ) -> list[list[float]]:
        """Полилиния маршрута по геометриям рёбер, а не только по узлам.

        Если у ребра есть ``geometry`` (Shapely LineString), берутся все
        его промежуточные точки — линия на карте точно повторяет дорогу.
        Дубликаты на стыках рёбер удаляются.

        Returns:
            ``[[lat, lon], ...]``
        """
        if route is None:
            return []

        coords: list[list[float]] = []
        for u, v, key in route.edges:
            data = G.edges[u, v, key]
            geom = data.get("geometry")
            edge_pts = _coords_from_shapely_linear(geom)
            if len(edge_pts) < 2:
                edge_pts = [
                    (G.nodes[u]["x"], G.nodes[u]["y"]),
                    (G.nodes[v]["x"], G.nodes[v]["y"]),
                ]
            else:
                first_node_y = G.nodes[u]["y"]
                first_node_x = G.nodes[u]["x"]
                start_lon, start_lat = edge_pts[0]
                end_lon, end_lat = edge_pts[-1]
                dist_to_start = (
                    (start_lat - first_node_y) ** 2
                    + (start_lon - first_node_x) ** 2
                )
                dist_to_end = (
                    (end_lat - first_node_y) ** 2
                    + (end_lon - first_node_x) ** 2
                )
                if dist_to_end < dist_to_start:
                    edge_pts = edge_pts[::-1]

            for lon, lat in edge_pts:
                pt = [round(lat, 7), round(lon, 7)]
                if not coords or coords[-1] != pt:
                    coords.append(pt)

        return coords

    # ------------------------------------------------------------------
    # Метрики маршрута
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_cost(
        G: nx.MultiDiGraph,
        route: Optional[RouteResult],
        weight_key: str,
    ) -> float:
        """Суммарная стоимость маршрута по указанному весу."""
        if route is None:
            return 0.0
        total = 0.0
        for i in range(route.edge_count):
            d = G.edges[route.edges[i]]
            w = d.get(weight_key)
            if w is None:
                logger.warning(
                    "У ребра нет веса %s — пропуск в сумме cost", weight_key
                )
                continue
            total += float(w)
        if not math.isfinite(total):
            logger.warning("Суммарный cost не конечен (%s), подставляем 0", total)
            return 0.0
        return total

    @staticmethod
    def calculate_length(
        G: nx.MultiDiGraph, route: Optional[RouteResult]
    ) -> float:
        """Длина маршрута в метрах."""
        if route is None:
            return 0.0
        return sum(
            G.edges[route.edges[i]].get("length", 0)
            for i in range(route.edge_count)
        )

    @staticmethod
    def green_stats(
        G: nx.MultiDiGraph, route: Optional[RouteResult]
    ) -> Dict[str, Any]:
        """Статистика озеленения: категории, средний % деревьев и травы."""
        zero = {
            "categories": {},
            "percent": 0.0,
            "avg_trees": 0.0,
            "avg_grass": 0.0,
        }
        if route is None:
            return zero

        counter: Counter = Counter()
        tw = gw = glen = tlen = 0.0

        for i in range(route.edge_count):
            d = G.edges[route.edges[i]]
            gt = d.get("green_type", "none")
            ln = d.get("length", 0)
            counter[gt] += 1
            tlen += ln
            tw += d.get("trees_percent", 0.0) * ln
            gw += d.get("grass_percent", 0.0) * ln
            if gt != "none":
                glen += ln

        return {
            "categories": dict(counter),
            "percent": (glen / tlen * 100) if tlen else 0.0,
            "avg_trees": (tw / tlen) if tlen else 0.0,
            "avg_grass": (gw / tlen) if tlen else 0.0,
        }

    @staticmethod
    def elevation_stats(
        G: nx.MultiDiGraph, route: Optional[RouteResult]
    ) -> Dict[str, float]:
        """Статистика рельефа: набор, спуск (интерполированные), макс/средний уклон."""
        zero = {
            "climb": 0.0,
            "descent": 0.0,
            "max_gradient_pct": 0.0,
            "avg_gradient_pct": 0.0,
        }
        if route is None:
            return zero

        climb = descent = 0.0
        ratios: List[float] = []

        for i in range(route.edge_count):
            d = G.edges[route.edges[i]]
            climb += d.get("edge_climb", 0.0)
            descent += d.get("edge_descent", 0.0)
            gr = _segment_gradient_ratio_for_stats(d)
            if gr is not None:
                ratios.append(gr)

        return {
            "climb": climb,
            "descent": descent,
            "max_gradient_pct": _route_max_gradient_pct_from_ratios(ratios),
            "avg_gradient_pct": _route_avg_gradient_pct_from_ratios(ratios),
        }

    @staticmethod
    def surface_stats(
        G: nx.MultiDiGraph, route: Optional[RouteResult]
    ) -> Tuple[Counter, Counter]:
        """Подсчёт типов покрытий и дорог вдоль маршрута."""
        sc: Counter = Counter()
        hc: Counter = Counter()
        if route is None:
            return sc, hc

        for i in range(route.edge_count):
            d = G.edges[route.edges[i]]
            sc[_effective_surface_from_edge(d) or "N/A"] += 1
            hc[_first_value(d.get("highway", "N/A")) or "N/A"] += 1
        return sc, hc

    @staticmethod
    def route_weight_fallback_metrics(
        G: nx.MultiDiGraph, route: RouteResult, profile: ModeProfile
    ) -> Dict[str, float]:
        """Метрики fallback-коэффициентов (1.0) по рёбрам маршрута.

        Доли для UI и порогов предупреждений считаются по **сумме длин рёбер**
        (метры), согласованно с подписью «доля длины маршрута».

        ``na_surface_length_m`` — нет тега ``surface`` в OSM (пустой ``surface_osm``).
        ``unknown_surface_length_m`` — тег есть, но нет в справочнике профиля.
        Эвристика (``surface_effective``) не уменьшает ``na_*`` по отсутствию тега в OSM.
        """
        total_m = 0.0
        na_surf_m = 0.0
        unk_surf_m = 0.0
        na_hw_m = 0.0
        unk_hw_m = 0.0
        for i in range(route.edge_count):
            d = G.edges[route.edges[i]]
            ln = float(d.get("length", 0.0) or 0.0)
            total_m += ln
            raw_osm = _raw_osm_surface_from_edge(d)
            hw_raw = _first_value(d.get("highway"))
            h = (
                str(hw_raw).strip().lower()
                if hw_raw is not None and str(hw_raw).strip()
                else ""
            )
            if not raw_osm:
                na_surf_m += ln
            elif raw_osm not in profile.surface:
                unk_surf_m += ln
            if not h:
                na_hw_m += ln
            elif h not in profile.highway:
                unk_hw_m += ln
        return {
            "total_length_m": total_m,
            "na_surface_length_m": na_surf_m,
            "unknown_surface_length_m": unk_surf_m,
            "na_highway_length_m": na_hw_m,
            "unknown_highway_length_m": unk_hw_m,
        }

    @staticmethod
    def elevation_profile(
        G: nx.MultiDiGraph, route: Optional[RouteResult]
    ) -> Dict[str, float]:
        """Профиль рельефа относительно стартовой точки.

        Returns:
            ``max_above`` — максимальный подъём от старта (м),
            ``max_below`` — максимальный спуск от старта (м, ≤0),
            ``end_diff`` — перепад старт→финиш (м).
        """
        zero = {"max_above": 0.0, "max_below": 0.0, "end_diff": 0.0}
        if route is None:
            return zero

        cumulative = 0.0
        max_above = 0.0
        max_below = 0.0

        for i in range(route.edge_count):
            d = G.edges[route.edges[i]]
            cumulative += d.get("elevation_diff", 0.0)
            max_above = max(max_above, cumulative)
            max_below = min(max_below, cumulative)

        return {
            "max_above": max_above,
            "max_below": max_below,
            "end_diff": cumulative,
        }

    @staticmethod
    def stairs_count(
        G: nx.MultiDiGraph, route: Optional[RouteResult]
    ) -> Dict[str, Any]:
        """Количество лестниц (highway=steps) и их суммарная длина."""
        if route is None:
            return {"count": 0, "length": 0.0}

        count = 0
        length = 0.0
        for i in range(route.edge_count):
            d = G.edges[route.edges[i]]
            hw = _first_value(d.get("highway", ""))
            if hw == "steps":
                count += 1
                length += d.get("length", 0)

        return {"count": count, "length": length}

    @staticmethod
    def elevation_profile_data(
        G: nx.MultiDiGraph, route: Optional[RouteResult]
    ) -> List[Tuple[float, float]]:
        """Точки (кумулятивная дистанция м, относительная высота м).

        Первая точка — (0, 0); реальную абсолютную высоту добавляет
        вызывающий код (через ElevationService).
        """
        if route is None:
            return []
        points: List[Tuple[float, float]] = [(0.0, 0.0)]
        cum_dist = 0.0
        cum_elev = 0.0
        for i in range(route.edge_count):
            d = G.edges[route.edges[i]]
            cum_dist += d.get("length", 0)
            cum_elev += d.get("elevation_diff", 0.0)
            points.append((cum_dist, cum_elev))
        return points

    @staticmethod
    def estimate_time(
        G: nx.MultiDiGraph,
        route: Optional[RouteResult],
        profile: ModeProfile,
    ) -> float:
        """Оценка времени в пути (секунды).

        Учитывает базовую скорость профиля, уклон и лестницы.
        """
        if route is None:
            return 0.0

        base = profile.base_speed_ms
        total = 0.0

        for i in range(route.edge_count):
            d = G.edges[route.edges[i]]
            length = d.get("length", 0)
            gradient = d.get("gradient", 0.0)
            hw = _first_value(d.get("highway", ""))

            if hw == "steps":
                speed = profile.stairs_speed_ms
            elif gradient > 0:
                speed = base * max(0.15, 1.0 - gradient * profile.uphill_penalty)
            elif gradient < 0:
                speed = base * min(1.5, 1.0 + abs(gradient) * profile.downhill_bonus)
            else:
                speed = base

            if speed > 0:
                total += length / speed

        return total

    # ------------------------------------------------------------------
    # GeoJSON-слои для карты (сегменты маршрута)
    # ------------------------------------------------------------------

    #: Опасные / напряжённые типы дорог (автотрафик).
    PROBLEMATIC_HIGHWAYS = frozenset(
        {
            "motorway",
            "motorway_link",
            "trunk",
            "trunk_link",
            "primary",
            "primary_link",
            "secondary",
            "secondary_link",
        }
    )
    #: Порог крутизны (доля, 0.10 = 10 %) для слоя «проблемные».
    STEEP_GRADIENT_ABS = 0.10

    @staticmethod
    def _edge_linestring_coords(
        G: nx.MultiDiGraph, u: int, v: int, key: int
    ) -> List[List[float]]:
        """Координаты линии ребра ``[[lon, lat], ...]`` для GeoJSON."""
        data = G.edges[u, v, key]
        geom = data.get("geometry")
        if geom is not None:
            pts = _coords_from_shapely_linear(geom)
            if len(pts) >= 2:
                return [list(c) for c in pts]
        return [
            [G.nodes[u]["x"], G.nodes[u]["y"]],
            [G.nodes[v]["x"], G.nodes[v]["y"]],
        ]

    @staticmethod
    def _map_feature(coords: List[List[float]], props: Dict[str, Any]) -> dict:
        return {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": props,
        }

    @staticmethod
    def build_route_map_layers(
        G: nx.MultiDiGraph,
        route: Optional[RouteResult],
    ) -> Dict[str, Any]:
        """Собрать GeoJSON FeatureCollection по категориям сегментов маршрута.

        Returns:
            dict с ключами ``greenery``, ``stairs``, ``problematic``, ``na_surface`` —
            каждый — полноценный GeoJSON FeatureCollection (сериализуемый в JSON).
        """
        def _empty_fc() -> dict:
            return {"type": "FeatureCollection", "features": []}

        if route is None:
            return {
                "greenery": _empty_fc(),
                "stairs": _empty_fc(),
                "problematic": _empty_fc(),
                "na_surface": _empty_fc(),
            }

        greenery_f: List[dict] = []
        stairs_f: List[dict] = []
        prob_f: List[dict] = []
        na_f: List[dict] = []

        for u, v, key in route.edges:
            d = G.edges[u, v, key]
            coords = RouteService._edge_linestring_coords(G, u, v, key)
            hw = _first_value(d.get("highway", "")) or ""
            hw = str(hw).lower()
            surf_raw = _raw_osm_surface_from_edge(d)
            surf = surf_raw or "N/A"
            gt = str(d.get("green_type", "none") or "none").lower()
            trees = float(d.get("trees_percent", 0.0) or 0.0)
            grass = float(d.get("grass_percent", 0.0) or 0.0)
            grad_disp = _edge_gradient_abs_for_display(d)
            gr = d.get("gradient_raw")
            if gr is None:
                gr = d.get("gradient", 0.0)
            grad_phys = float(gr or 0.0)
            ln = float(d.get("length", 0.0) or 0.0)

            base = {
                "highway": hw,
                "length_m": round(ln, 1),
                "gradient_pct": round(grad_disp * 100, 1),
            }

            if gt != "none" or trees >= 2.0 or grass >= 2.0:
                greenery_f.append(
                    RouteService._map_feature(
                        coords,
                        {
                            **base,
                            "green_type": gt,
                            "trees_pct": round(trees, 1),
                            "grass_pct": round(grass, 1),
                        },
                    )
                )

            if hw == "steps":
                stairs_f.append(
                    RouteService._map_feature(coords, {**base, "kind": "steps"})
                )

            if hw != "steps":
                reasons: List[str] = []
                if abs(grad_phys) >= RouteService.STEEP_GRADIENT_ABS:
                    reasons.append("steep")
                if hw in RouteService.PROBLEMATIC_HIGHWAYS:
                    reasons.append("traffic_class")
                if reasons:
                    prob_f.append(
                        RouteService._map_feature(
                            coords,
                            {**base, "reasons": ",".join(reasons)},
                        )
                    )

            if surf == "N/A" or surf == "":
                na_f.append(
                    RouteService._map_feature(
                        coords,
                        {**base, "surface": surf or "N/A"},
                    )
                )

        def _fc(features: List[dict]) -> dict:
            return {"type": "FeatureCollection", "features": features}

        return {
            "greenery": _fc(greenery_f),
            "stairs": _fc(stairs_f),
            "problematic": _fc(prob_f),
            "na_surface": _fc(na_f),
        }
