"""Сезон, снег, WMO weather_code — коэффициенты маршрутизации (не из .env)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SeasonalSnowParams:
    season_green_ramp_start_day: int = 10
    season_green_ramp_end_day: int = 20
    season_early_spring_end_day: int = 9
    season_green_ramp_start_mult: float = 0.18
    season_green_ramp_end_mult: float = 1.0
    season_green_mult_winter: float = 0.08
    season_green_mult_early_spring: float = 0.18
    season_green_mult_green: float = 1.0
    season_green_mult_late_autumn: float = 0.42
    season_tree_heat_mult_winter: float = 0.12
    season_tree_heat_mult_early_spring: float = 0.28
    season_tree_heat_mult_spring_ramp: float = 0.55
    season_tree_heat_mult_green: float = 1.0
    season_tree_heat_mult_late_autumn: float = 0.45
    snow_spring_ramp_strength: float = 0.88
    snow_default_strength: float = 0.9
    snow_green_season_strength_floor: float = 0.06
    snow_green_season_depth_on_m: float = 0.02
    snow_green_season_fresh_on_cm_h: float = 0.25
    snow_depth_tier0_max_m: float = 0.02
    snow_depth_tier1_max_m: float = 0.05
    snow_depth_tier2_max_m: float = 0.10
    snow_depth_mult_tier0: float = 1.0
    snow_depth_mult_tier1: float = 1.06
    snow_depth_mult_tier2: float = 1.14
    snow_depth_mult_tier3: float = 1.28
    snow_fresh_tier0_max_cm_h: float = 0.2
    snow_fresh_tier1_max_cm_h: float = 1.0
    snow_fresh_tier2_max_cm_h: float = 3.0
    snow_fresh_mult_tier0: float = 1.0
    snow_fresh_mult_tier1: float = 1.04
    snow_fresh_mult_tier2: float = 1.12
    snow_fresh_mult_tier3: float = 1.22
    snow_depth_norm_ref_m: float = 0.25
    snow_fresh_norm_ref_cm_h: float = 4.0
    snow_stress_global_add_winter: float = 0.035
    snow_stress_phys_coupling: float = 0.45
    winter_heat_open_scale_winter: float = 1.18
    winter_heat_open_scale_transition: float = 1.08
    winter_heat_wind_scale_winter: float = 1.22
    winter_heat_wind_scale_transition: float = 1.12
    winter_heat_tree_damp_snow_depth: float = 0.38
    winter_heat_tree_damp_snow_fresh: float = 0.22
    weather_stress_edge_snow_open: float = 0.18
    weather_stress_edge_snow_surface: float = 0.22
    weather_stress_edge_snow_stairs: float = 0.16
    snow_surface_tier0_amp: float = 0.012
    snow_surface_tier1_amp: float = 0.055
    snow_surface_tier2_amp: float = 0.14
    snow_stairs_ped_base: float = 1.08
    snow_stairs_ped_frozen_boost: float = 1.12
    snow_stairs_cyclist_base: float = 1.18
    snow_stairs_cyclist_frozen_boost: float = 1.22
    snow_phys_cyclist_mult_boost: float = 0.12
    winter_clearance_high_mitigate: float = 0.22
    season_adaptive_mode: str = "calendar_only"
    season_adaptive_warm_anomaly_temp_c: float = 9.0
    season_adaptive_winter_snow_depth_max_m: float = 0.018
    season_adaptive_winter_fresh_max_cm_h: float = 0.35
    season_adaptive_snow_depth_on_green_m: float = 0.04
    season_adaptive_fresh_on_green_cm_h: float = 0.45
    season_adaptive_cold_on_green_c: float = 1.5
    season_adaptive_early_april_warm_c: float = 11.0
    wc_snow_model_strength_amp: float = 0.22
    wc_wet_slip_physical_amp: float = 0.08
    wc_mixed_precip_stress_amp: float = 0.12
    wc_wet_slip_surface_amp: float = 0.26
    wc_mixed_surface_amp: float = 0.18
    wc_wet_stairs_amp: float = 0.14
    wc_mixed_snow_stress_amp: float = 0.16
    winter_clearance_low_amp: float = 0.32
    winter_clearance_low_amp_cyclist: float = 0.22
    wc_winter_open_sky_penalty_amp: float = 0.05
    wc_winter_building_shelter_bonus_amp: float = 0.06


DEFAULT_SEASONAL_SNOW_PARAMS = SeasonalSnowParams()
