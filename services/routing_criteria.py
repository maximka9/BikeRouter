"""Таблица критериев маршрутизации: явные факторы ребра (источник истины для погоды×рельеф).

Каждый погодный эффект проходит через именованный множитель; итоговые веса собираются
из произведений критериев (зелень — отдельно от «базовой» физики).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, NamedTuple, Optional, Tuple

# Имена критериев (соответствие столбцам таблицы / экспорту route_*).
CRITERION_KEYS: Tuple[str, ...] = (
    "base_route_factor",
    "slope_weather_factor",
    "surface_weather_factor",
    "green_route_factor",
    "open_sky_weather_factor",
    "building_shelter_factor",
    "covered_shelter_factor",
    "stress_weather_factor",
    "stairs_weather_factor",
    "wind_orientation_factor",
)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


@dataclass(frozen=True)
class EdgeRouteCriteriaFactors:
    """Явные множители ребра (1.0 = нейтрально)."""

    base_route_factor: float = 1.0
    slope_weather_factor: float = 1.0
    surface_weather_factor: float = 1.0
    green_route_factor: float = 1.0
    open_sky_weather_factor: float = 1.0
    building_shelter_factor: float = 1.0
    covered_shelter_factor: float = 1.0
    stress_weather_factor: float = 1.0
    stairs_weather_factor: float = 1.0
    wind_orientation_factor: float = 1.0

    def as_dict(self) -> Dict[str, float]:
        return {k: float(getattr(self, k)) for k in CRITERION_KEYS}


class EffectiveEdgeComponents(NamedTuple):
    """Физика (без зелёного множителя), heat, stress и явные критерии ребра."""

    phys_eff: float
    heat_eff: float
    st_eff: float
    criteria: EdgeRouteCriteriaFactors


def slope_route_weather_factor(edge_data: dict, weather: Any) -> float:
    """Слабый weather×slope: жара + подъём; холод+снег; дождь×плохое покрытие×уклон."""
    if not weather or not getattr(weather, "enabled", False):
        return 1.0
    sig = getattr(weather, "normalized_signals", None) or {}
    tn = float(sig.get("temp_norm", 0.0))
    snd = float(sig.get("snow_depth_norm", 0.0))
    snf = float(sig.get("snow_fresh_norm", 0.0))
    rn = float(sig.get("rain_norm", 0.0))
    cl = float(sig.get("cold_like_norm", 0.0))
    gr = float(edge_data.get("gradient_raw"))
    if gr is None:
        gr = edge_data.get("gradient", 0.0)
    try:
        g = float(gr or 0.0)
    except (TypeError, ValueError):
        g = 0.0
    uphill = max(0.0, g)
    if uphill <= 1e-6:
        return 1.0
    # Локальный импорт, чтобы избежать цикла с routing при загрузке модуля.
    from .routing import _surface_wet_route_tier

    tier = _surface_wet_route_tier(edge_data.get("surface_effective"))
    k_heat = float(getattr(weather, "slope_weather_heat_uphill_amp", 0.045))
    k_cold_snow = float(getattr(weather, "slope_weather_cold_snow_amp", 0.055))
    k_rain = float(getattr(weather, "slope_weather_rain_bad_surface_amp", 0.028))
    ss = float(getattr(weather, "snow_model_strength", 0.0))
    f = 1.0
    f += k_heat * tn * uphill * 12.0
    f += k_cold_snow * cl * max(snd, snf * 0.85) * ss * uphill * 10.0
    if tier >= 1:
        f += k_rain * rn * (0.35 + 0.35 * float(tier)) * uphill * 8.0
    return max(0.94, min(1.14, f))


def stairs_route_weather_factor(
    edge_data: dict, weather: Any, profile_key: str
) -> float:
    """Лестницы: снег + дождь + ветер + ориентация к ветру + сезон (отдельный критерий)."""
    if not weather or not getattr(weather, "enabled", False):
        return 1.0
    hw = str(edge_data.get("highway") or "").strip().lower()
    if hw != "steps":
        return 1.0
    from .routing import _wind_dir_edge_pack, snow_stairs_phys_mult

    m = float(snow_stairs_phys_mult(weather, profile_key))
    sig = getattr(weather, "normalized_signals", None) or {}
    rn = float(sig.get("rain_norm", 0.0))
    hn = float(sig.get("humidity_norm", 0.0))
    wet_env = max(rn, hn * 0.82)
    w_rain = float(getattr(weather, "stairs_rain_wet_amp", 0.06))
    m *= 1.0 + w_rain * wet_env
    wn = float(sig.get("wind_norm", 0.0))
    gn = float(sig.get("gust_norm", 0.0))
    wcomb = max(wn, gn * 0.88)
    w_wind = float(getattr(weather, "stairs_wind_gust_amp", 0.05))
    m *= 1.0 + w_wind * wcomb
    pack = _wind_dir_edge_pack(edge_data, weather)
    if pack is not None:
        along01, _c, _d, wind_act, _wcs = pack
        w_dir = float(getattr(weather, "stairs_wind_along_open_amp", 0.04))
        m *= 1.0 + w_dir * along01 * wind_act
    sm = float(getattr(weather, "season_stairs_route_mult", 1.0) or 1.0)
    m *= sm
    if profile_key == "cyclist":
        m *= float(getattr(weather, "stairs_cyclist_route_extra_mult", 1.08))
    return max(1.0, min(1.85, m))


def continuous_heat_edge_factor_and_split(
    edge_data: dict, weather: Any
) -> Tuple[float, Dict[str, float]]:
    """Непрерывный heat-множитель ef и прокси-субмножители open/building/covered (для экспорта)."""
    from .heat import wet_surface_edge_slip_factor
    from .routing import _wind_dir_edge_pack

    O = _clamp01(float(edge_data.get("thermal_open_sky_share") or 0.5) or 0.5)
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
    sig = getattr(weather, "normalized_signals", None) or {}
    wcomb_sig = max(float(sig.get("wind_norm", 0.0)), float(sig.get("gust_norm", 0.0)) * 0.85)
    pack = _wind_dir_edge_pack(edge_data, weather)
    b_build = 1.0
    wind_exp = wind_exp_base
    wind_orient_f = 1.0
    if pack is not None:
        along01, cross01, _delta, wind_act, _wcs = pack
        ha = float(getattr(weather, "heat_wind_along_open_amp", 0.28))
        wind_dir_amp = 1.0 + 1.08 * min(1.45, max(0.0, wcomb_sig - 0.06) * 2.05)
        ha *= wind_dir_amp
        wind_exp = _clamp01(wind_exp_base * (1.0 + ha * along01 * wind_act * O))
        hc = float(getattr(weather, "heat_wind_cross_build_amp", 0.35)) * wind_dir_amp
        hd = float(getattr(weather, "heat_wind_along_build_damp", 0.24))
        b_build = 1.0 + hc * cross01 * wind_act - hd * along01 * wind_act
        b_build = max(0.45, min(1.55, b_build))
        w_mul = float(getattr(weather, "season_wind_orientation_route_mult", 1.0) or 1.0)
        wboost = 1.0 + 0.48 * min(1.0, max(0.0, wcomb_sig - 0.06) * 1.9)
        wind_orient_f = max(
            0.86,
            min(
                1.22,
                1.0 + (wind_exp / max(wind_exp_base, 1e-6) - 1.0) * w_mul * wboost,
            ),
        )
    k1 = float(getattr(weather, "heat_edge_k_open", 0.66))
    k2 = float(getattr(weather, "heat_edge_k_tree", 0.54))
    k3 = float(getattr(weather, "heat_edge_k_building", 0.50))
    k4 = float(getattr(weather, "heat_edge_k_covered", 0.16))
    k5 = float(getattr(weather, "heat_edge_k_wet", 0.28))
    k6 = float(getattr(weather, "heat_edge_k_wind", 0.30))
    tn = float(sig.get("temp_norm", 0.0))
    rn = _clamp01(float(sig.get("rain_norm", 0.0)))
    cn = float(sig.get("cloud_norm", 0.0))
    hn = float(sig.get("humidity_norm", 0.0))
    exn = _clamp01(float(sig.get("extreme_heat_norm", 0.0)))
    # Жара + ясное небо + почти без дождя — усилить open/tree/building на выборе маршрута.
    hot_clear_dry = min(
        1.0,
        max(0.0, tn - 0.38)
        * max(0.0, 1.0 - cn - 0.08)
        * max(0.0, 0.48 - rn)
        * 2.85,
    )
    rain_route = min(
        1.0,
        max(0.0, rn * 1.18 + 0.28 * hn * rn),
    )
    rain_open_amp = 1.0 + float(getattr(weather, "heat_edge_rain_open_mult", 0.72)) * rn
    rain_build_amp = 1.0 + float(getattr(weather, "heat_edge_rain_building_mult", 0.78)) * rn
    rain_wind_exp_amp = 1.0 + float(getattr(weather, "heat_edge_rain_wind_exp_mult", 0.32)) * rn
    k1_eff = (
        k1
        * rain_open_amp
        * (1.0 + 0.58 * hot_clear_dry)
        * (1.0 + 0.42 * rain_route)
        * (1.0 + float(getattr(weather, "heat_extreme_open_route_gain", 0.22)) * exn)
    )
    k3_eff = k3 * rain_build_amp * (1.0 + 0.24 * hot_clear_dry) * (1.0 + 0.38 * rain_route)
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
    ss = float(getattr(weather, "snow_model_strength", 0.0))
    winter_cover_boost = 1.0 + ss * float(getattr(weather, "winter_covered_snow_bonus_amp", 0.12)) * C
    wcw = float(sig.get("wc_wet_slip", 0.0))
    wcm = float(sig.get("wc_mixed", 0.0))
    wc_open = float(getattr(weather, "wc_winter_open_sky_penalty_amp", 0.05))
    wc_shelter = float(getattr(weather, "wc_winter_building_shelter_bonus_amp", 0.06))
    pos_open = rs * k1_eff * osp * O * wo * (1.0 + wc_open * (wcw + 0.65 * wcm) * O)
    tree_hot = (
        1.0
        + 0.46 * hot_clear_dry
        + float(getattr(weather, "heat_extreme_tree_route_gain", 0.18)) * exn
    )
    neg_tree = -rs * k2 * tree_hot * tsb * V * wt
    neg_build = -rs * k3_eff * bsb * B * b_build * (1.0 - wc_shelter * wcm * B)
    neg_cover = -rs * k4 * cbn * C * winter_cover_boost
    pos_wet = rs * k5 * (1.0 + 0.36 * rain_route) * wsp * wet_eff
    pos_wind = rs * k6_eff * wop * wind_exp * wwind
    pos_syn = rs * syn * O * wet_eff * wind_exp * max(1.0, (wo + wwind) * 0.5)
    raw = 1.0 + pos_open + neg_tree + neg_build + neg_cover + pos_wet + pos_wind + pos_syn
    lo = float(getattr(weather, "heat_edge_factor_min", 0.65))
    hi = float(getattr(weather, "heat_edge_factor_max", 1.75))
    ef = max(lo, min(hi, raw))
    # Прокси-критерии для экспорта (clamp в диапазоне ef); тень деревьев входит в суммарный ef, не в green_route_factor.
    open_f = max(lo, min(hi, 1.0 + pos_open))
    build_f = max(lo, min(hi, 1.0 + neg_build))
    cover_f = max(lo, min(hi, 1.0 + neg_cover))
    sub = {
        "open_sky_weather_factor": open_f,
        "building_shelter_factor": build_f,
        "covered_shelter_factor": cover_f,
        "wind_orientation_factor": wind_orient_f,
    }
    return ef, sub


def compute_edge_route_criteria(
    edge_data: dict,
    profile_key: str,
    time_slot_key: str,
    heat_context_mult: float,
    weather: Any,
    *,
    physical_weight_key: Optional[str] = None,
) -> Tuple[EdgeRouteCriteriaFactors, float, float, float]:
    """Собрать критерии и базовые phys/heat/stress до α,β,γ (green — отдельный множитель к phys)."""
    from .routing import (
        coerce_edge_weight_numeric,
        snow_surface_route_penalty,
        wet_surface_route_penalty,
        winter_clearance_priority_01,
    )

    phys_k = physical_weight_key or f"weight_{profile_key}_full"
    heat_k = f"heat_{time_slot_key}"
    phys = coerce_edge_weight_numeric(
        edge_data.get(phys_k),
        fallback=float("inf"),
    )
    h = coerce_edge_weight_numeric(edge_data.get(heat_k), fallback=0.0)
    hm = float(heat_context_mult) if math.isfinite(heat_context_mult) else 1.0
    st = coerce_edge_weight_numeric(edge_data.get("stress_cost"), fallback=0.0)
    if not weather or not getattr(weather, "enabled", False):
        z = EdgeRouteCriteriaFactors()
        return z, phys, h * hm, st

    wm = weather.mults
    gcc = float(weather.green_coupling)
    g_edge = min(1.0, max(0.0, float(edge_data.get("trees_pct") or 0) / 100.0))
    g_coupling = gcc * float(getattr(weather, "season_green_route_mult", 1.0))
    if wm.heat < 1.0:
        g_coupling *= max(0.25, float(wm.heat))
    green_f = 1.0 + max(0.0, wm.green - 1.0) * g_edge * g_coupling

    ss = float(getattr(weather, "snow_model_strength", 0.0))
    snd = float(getattr(weather, "snow_depth_norm", 0.0))
    snf = float(getattr(weather, "snow_fresh_norm", 0.0))
    wet_f = float(wet_surface_route_penalty(edge_data.get("surface_effective"), weather))
    snow_f = float(snow_surface_route_penalty(edge_data, weather))
    surface_f = wet_f * snow_f
    if ss > 1e-9 and profile_key == "cyclist":
        clr = winter_clearance_priority_01(edge_data)
        cclr = float(getattr(weather, "winter_clearance_low_amp_cyclist", 0.22))
        surface_f *= 1.0 + cclr * ss * max(snd, snf) * max(0.0, 1.0 - clr)

    slope_f = slope_route_weather_factor(edge_data, weather)
    stairs_f = stairs_route_weather_factor(edge_data, weather, profile_key)
    if getattr(weather, "heat_continuous", False):
        sig_s = getattr(weather, "normalized_signals", None) or {}
        ss0 = float(getattr(weather, "snow_model_strength", 0.0))
        tnx = float(sig_s.get("temp_norm", 0.0))
        rnx = float(sig_s.get("rain_norm", 0.0))
        cnx = float(sig_s.get("cloud_norm", 0.0))
        hot_dry = (
            max(0.0, tnx - 0.42)
            * max(0.0, 1.0 - cnx - 0.12)
            * max(0.0, 0.38 - rnx)
        )
        if ss0 < 0.05 and hot_dry > 0.015:
            stairs_f *= 1.0 + min(0.2, 0.55 * hot_dry)
    base_f = float(wm.physical) * float(wm.surface)
    if ss > 1e-9 and profile_key == "cyclist":
        cboost = float(getattr(weather, "snow_phys_cyclist_mult_boost", 0.12))
        base_f *= 1.0 + cboost * ss * max(snd, snf)

    if getattr(weather, "heat_continuous", False):
        ef, sub = continuous_heat_edge_factor_and_split(edge_data, weather)
        open_f = float(sub.get("open_sky_weather_factor", 1.0))
        build_f = float(sub.get("building_shelter_factor", 1.0))
        cover_f = float(sub.get("covered_shelter_factor", 1.0))
        wind_of = float(sub.get("wind_orientation_factor", 1.0))
        heat_eff = h * hm * float(wm.heat) * ef
    else:
        open_f = build_f = cover_f = 1.0
        wind_of = 1.0
        regime = str(getattr(weather, "regime", "neutral") or "neutral")
        rs = float(getattr(weather, "weather_response_scale", 1.0) or 1.0)
        O = float(edge_data.get("thermal_open_sky_share") or 0.5) or 0.5
        O = max(0.0, min(1.0, O))
        B = float(edge_data.get("thermal_building_shade_share") or 0.0) or 0.0
        B = max(0.0, min(1.0, B))
        V = float(edge_data.get("thermal_vegetation_shade_share") or 0.0) or 0.0
        V = max(0.0, min(1.0, V))
        heat_excess = max(0.0, float(wm.heat) - 1.0)
        heat_deficit = max(0.0, 1.0 - float(wm.heat))
        hh = h
        if regime == "hot":
            op = float(getattr(weather, "hot_open_penalty_scale", 1.0) or 1.0)
            tb = float(getattr(weather, "hot_tree_bonus_scale", 1.0) or 1.0)
            open_f = 1.0 + 0.32 * rs * op * O * min(1.35, heat_excess + 0.12)
            hh = hh * open_f * (1.0 - 0.26 * rs * tb * V * min(1.2, heat_excess + 0.08))
            build_f = cover_f = 1.0
        elif regime == "cold":
            cc = float(getattr(weather, "cold_canyon_bonus_scale", 1.0) or 1.0)
            td = float(getattr(weather, "cold_tree_damping", 0.65) or 0.65)
            td = max(0.0, min(1.0, td))
            canyon = max(0.0, min(1.0, B * (0.65 + 0.35 * (1.0 - O))))
            build_f = 1.0 - 0.22 * rs * cc * canyon * min(1.0, heat_deficit + 0.18)
            hh = hh * build_f * (
                1.0 + 0.16 * rs * (1.0 - td) * V * (0.6 + 0.4 * heat_deficit)
            )
            open_f = cover_f = 1.0
        else:
            if wm.heat >= 1.03:
                open_f = 1.0 + 0.28 * O * min(1.2, wm.heat - 1.0)
                hh = hh * open_f
            elif wm.heat <= 0.97:
                build_f = 1.0 - 0.18 * B * (1.0 - float(wm.heat))
                hh = hh * build_f
            else:
                open_f = build_f = 1.0
            cover_f = 1.0
        heat_eff = hh * hm * float(wm.heat)
        sigw = getattr(weather, "normalized_signals", None) or {}
        wcomb = max(float(sigw.get("wind_norm", 0.0)), float(sigw.get("gust_norm", 0.0)) * 0.85)
        if wcomb > 0.06:
            from .routing import _wind_dir_edge_pack

            p2 = _wind_dir_edge_pack(edge_data, weather)
            w_mul = float(getattr(weather, "season_wind_orientation_route_mult", 1.0) or 1.0)
            if p2 is not None:
                along01, _c2, _d2, wind_act, _wcs2 = p2
                w_slope = 0.055 + 0.05 * min(1.0, max(0.0, wcomb - 0.06) * 2.4)
                wind_of = max(
                    0.86,
                    min(
                        1.18,
                        1.0 + w_slope * along01 * wind_act * w_mul * min(1.0, wcomb * 1.2),
                    ),
                )

    regime_st = float(getattr(weather, "stress_route_regime_factor", 1.0) or 1.0)
    if regime_st <= 0.0 or not math.isfinite(regime_st):
        blend = float(getattr(weather, "weather_stress_global_blend", 0.28))
        snow_add = float(getattr(weather, "snow_stress_global_add", 0.0))
        srm = float(getattr(weather, "season_stress_route_mult", 1.0) or 1.0)
        regime_st = (1.0 + blend * (float(wm.stress) - 1.0) + snow_add) * srm
    from .routing import weather_edge_stress_factor

    se_st = float(weather_edge_stress_factor(edge_data, weather))
    st_eff = st * regime_st * se_st

    factors = EdgeRouteCriteriaFactors(
        base_route_factor=base_f,
        slope_weather_factor=slope_f,
        surface_weather_factor=surface_f,
        green_route_factor=green_f,
        open_sky_weather_factor=open_f,
        building_shelter_factor=build_f,
        covered_shelter_factor=cover_f,
        stress_weather_factor=se_st,
        stairs_weather_factor=stairs_f,
        wind_orientation_factor=wind_of,
    )
    phys_eff = phys * base_f * slope_f * surface_f * stairs_f
    return factors, phys_eff, heat_eff, st_eff
