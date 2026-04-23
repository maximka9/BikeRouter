"""Сезонность и снег: календарь, множители, Open-Meteo снимок."""

from __future__ import annotations

from datetime import date

from bike_router.config import Settings
from bike_router.services.seasonal import (
    routing_season_label,
    season_green_route_multiplier,
    snow_depth_phys_multiplier,
    snow_fresh_phys_multiplier,
    snow_route_model_strength,
)
from bike_router.services.weather import WeatherSnapshot, snapshot_from_manual


def test_routing_season_april_ramp() -> None:
    s = Settings()
    assert routing_season_label(date(2000, 4, 5), s) == "early_spring"
    assert routing_season_label(date(2000, 4, 15), s) == "spring_ramp"
    assert routing_season_label(date(2000, 4, 25), s) == "green_season"
    assert routing_season_label(date(2000, 2, 10), s) == "winter"


def test_green_multiplier_ramp_april() -> None:
    s = Settings()
    g10 = season_green_route_multiplier(date(2000, 4, 10), s)
    g20 = season_green_route_multiplier(date(2000, 4, 20), s)
    assert g10 < g20 - 1e-6
    assert g20 >= 0.99


def test_snow_strength_green_vs_winter() -> None:
    s = Settings()
    snap_clear = WeatherSnapshot(snow_depth_m=0.0, snowfall_cm_h=0.0)
    snap_snow = WeatherSnapshot(snow_depth_m=0.12, snowfall_cm_h=0.5)
    assert snow_route_model_strength("green_season", snap_clear, s) < 0.2
    assert snow_route_model_strength("winter", snap_snow, s) >= 0.99


def test_snow_depth_tiers_increase() -> None:
    s = Settings()
    a = snow_depth_phys_multiplier(0.01, s)
    b = snow_depth_phys_multiplier(0.04, s)
    c = snow_depth_phys_multiplier(0.08, s)
    d = snow_depth_phys_multiplier(0.15, s)
    assert a <= b <= c <= d


def test_snow_fresh_tiers_increase() -> None:
    s = Settings()
    assert snow_fresh_phys_multiplier(0.0, s) <= snow_fresh_phys_multiplier(2.0, s)


def test_snapshot_from_manual_snow() -> None:
    snap = snapshot_from_manual(
        temperature_c=-5.0,
        snowfall_cm_h=1.2,
        snow_depth_m=0.08,
        weather_code=73,
    )
    assert snap.snowfall_cm_h == 1.2
    assert snap.snow_depth_m == 0.08
    assert snap.weather_code == 73
