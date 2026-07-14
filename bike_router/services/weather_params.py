"""Коэффициенты тепло/погодной маршрутизации (не из .env)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HeatWeatherParams:
    """Параметры непрерывной тепло-модели, apparent temp, weather stress, wind по рёбрам."""

    heat_max_detour_ratio: float = 0.42
    stress_max_detour_ratio: float = 0.55
    heat_hot_tree_bonus_scale: float = 1.45
    heat_hot_open_sky_penalty_scale: float = 1.55
    heat_cold_building_canyon_bonus_scale: float = 1.4
    heat_cold_tree_bonus_damping: float = 0.65
    heat_weather_response_scale: float = 1.35
    heat_continuous_enable: bool = True
    heat_temp_ref_max: float = 30.0
    heat_temp_cool_ref: float = 14.0
    heat_temp_cool_range: float = 10.0
    heat_rain_ref_max: float = 3.0
    heat_wind_ref_max: float = 12.0
    heat_gust_delta_ref_max: float = 10.0
    heat_tree_shade_temp_gain: float = 0.42
    heat_tree_shade_rain_damp: float = 0.35
    heat_tree_shade_cold_damp: float = 0.58
    heat_open_sky_temp_gain: float = 0.52
    heat_open_sky_wind_gain: float = 0.28
    heat_open_sky_gust_gain: float = 0.22
    heat_open_sky_humid_gain: float = 0.18
    heat_apparent_temp_enable: bool = True
    heat_apparent_temp_ref: float = 25.0
    heat_apparent_temp_hot_ref: float = 32.0
    heat_apparent_temp_extreme_ref: float = 38.0
    heat_open_sky_apparent_gain: float = 0.30
    heat_open_sky_radiation_gain: float = 0.25
    heat_open_sky_hot_extra_gain: float = 0.15
    heat_extreme_apparent_threshold: float = 35.0
    heat_extreme_route_beta_gain: float = 0.30
    heat_extreme_open_route_gain: float = 0.22
    heat_extreme_tree_route_gain: float = 0.18
    heat_tree_shade_apparent_gain: float = 0.38
    heat_tree_shade_radiation_gain: float = 0.30
    heat_tree_shade_extreme_gain: float = 0.20
    heat_building_shade_radiation_gain: float = 0.14
    heat_building_shade_wind_gain: float = 0.32
    heat_building_shade_rain_gain: float = 0.30
    heat_building_shade_humid_gain: float = 0.14
    heat_building_shade_gust_gain: float = 0.22
    heat_covered_rain_gain: float = 0.28
    heat_covered_wind_gain: float = 0.14
    heat_covered_gust_gain: float = 0.12
    heat_wet_surface_rain_gain: float = 0.38
    heat_wet_surface_humid_gain: float = 0.22
    heat_edge_k_open: float = 0.66
    heat_edge_k_tree: float = 0.54
    heat_edge_k_building: float = 0.50
    heat_edge_k_covered: float = 0.16
    heat_edge_k_wet: float = 0.34
    heat_edge_k_wind: float = 0.36
    heat_edge_rain_open_mult: float = 0.48
    heat_edge_rain_building_mult: float = 0.56
    heat_edge_rain_wind_exp_mult: float = 0.22
    heat_wind_exp_w1: float = 0.45
    heat_wind_exp_w2: float = 0.35
    heat_wind_exp_w3: float = 0.35
    heat_wet_surface_edge_bad_max: float = 0.85
    heat_open_wet_synergy: float = 0.14
    heat_wind_open_penalty_gain: float = 0.34
    heat_wind_open_gust_gain: float = 0.24
    heat_wind_along_open_amp: float = 0.28
    heat_wind_cross_build_amp: float = 0.35
    heat_wind_along_build_damp: float = 0.24
    stress_wind_along_open_amp: float = 0.32
    stress_wind_cross_shelter_amp: float = 0.40
    heat_edge_factor_min: float = 0.65
    heat_edge_factor_max: float = 1.75
    heat_coeff_clamp_lo: float = 0.72
    heat_coeff_clamp_hi: float = 1.60
    heat_coeff_soft_overflow_gain: float = 0.18
    heat_coeff_hard_hi: float = 1.85
    weather_stress_global_blend: float = 0.28
    weather_stress_edge_rain_slip: float = 0.22
    weather_stress_edge_wind_open: float = 0.20
    weather_stress_edge_lts_fast: float = 0.17
    weather_stress_edge_building_shelter: float = 0.11
    weather_stress_edge_factor_min: float = 0.82
    weather_stress_edge_factor_max: float = 1.48
    weather_phys_wet_penalty_tier0_cap: float = 0.016
    weather_phys_wet_penalty_tier1_coef: float = 0.048
    weather_phys_wet_penalty_tier2_coef: float = 0.12


DEFAULT_HEAT_WEATHER_PARAMS = HeatWeatherParams()
