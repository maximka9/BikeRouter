"""Регрессия: weather_code влияет на snow_surface_route_penalty при том же snow_depth."""

from __future__ import annotations

from bike_router.config import Settings
from bike_router.services.routing import snow_surface_route_penalty
from bike_router.services.weather import (
    WeatherMultipliers,
    WeatherWeightParams,
    snapshot_from_manual,
    wmo_weather_code_profile,
)


def test_wmo_profiles_distinct_groups() -> None:
    dry_snow = wmo_weather_code_profile(71)
    wet = wmo_weather_code_profile(56)
    mixed = wmo_weather_code_profile(95)
    assert dry_snow["wc_snow_event"] >= 0.99
    assert wet["wc_wet_slip"] >= 0.99
    assert mixed["wc_mixed"] >= 0.99


def test_snow_surface_penalty_differs_by_weather_code_same_snow_depth() -> None:
    s = Settings()
    base_sig = {
        "temp_norm": 0.2,
        "rain_norm": 0.0,
        "wind_norm": 0.3,
        "gust_norm": 0.2,
        "cloud_norm": 0.7,
        "humidity_norm": 0.6,
        "cold_like_norm": 0.8,
        "snow_depth_norm": 0.35,
        "snow_fresh_norm": 0.12,
    }
    snap_dry = snapshot_from_manual(
        temperature_c=-8.0,
        snowfall_cm_h=0.5,
        snow_depth_m=0.12,
        weather_code=71,
    )
    sig_dry = dict(base_sig)
    sig_dry.update(wmo_weather_code_profile(snap_dry.weather_code))
    wp_dry = WeatherWeightParams(
        enabled=True,
        mults=WeatherMultipliers(physical=1.1, stress=1.05),
        snow_model_strength=0.55,
        snow_depth_norm=0.35,
        snow_fresh_norm=0.12,
        normalized_signals=sig_dry,
        reference_temperature_c=-8.0,
    )
    snap_wet = snapshot_from_manual(
        temperature_c=-1.0,
        snowfall_cm_h=0.5,
        snow_depth_m=0.12,
        weather_code=56,
    )
    sig_wet = dict(base_sig)
    sig_wet.update(wmo_weather_code_profile(snap_wet.weather_code))
    wp_wet = WeatherWeightParams(
        enabled=True,
        mults=WeatherMultipliers(physical=1.1, stress=1.05),
        snow_model_strength=0.55,
        snow_depth_norm=0.35,
        snow_fresh_norm=0.12,
        normalized_signals=sig_wet,
        reference_temperature_c=-1.0,
    )
    edge_easy = {"highway": "pedestrian", "surface_effective": "asphalt"}
    p_dry = snow_surface_route_penalty(edge_easy, wp_dry)
    p_wet = snow_surface_route_penalty(edge_easy, wp_wet)
    assert 1.0 < p_dry < 1.52
    assert p_wet > p_dry + 0.004
