"""Быстрые проверки тепла, погоды и сезона (без сети и Open-Meteo)."""

from __future__ import annotations

from datetime import date, datetime, timezone

from bike_router.config import Settings
from bike_router.engine import _resolve_season_for_heat_alternatives
from bike_router.services.routing import continuous_heat_edge_weather_factor, weather_edge_stress_factor
from bike_router.services.seasonal import (
    resolve_season_routing_context,
    routing_season_label,
    season_green_route_multiplier,
    snow_depth_phys_multiplier,
    snow_route_model_strength,
)
from bike_router.services.weather import (
    WeatherMultipliers,
    WeatherSnapshot,
    WeatherWeightParams,
    _normalized_weather_signals,
    snapshot_from_manual,
    wmo_weather_code_profile,
)


def test_wmo_profiles_distinct() -> None:
    assert wmo_weather_code_profile(71)["wc_snow_event"] >= 0.99
    assert wmo_weather_code_profile(56)["wc_wet_slip"] >= 0.99
    assert wmo_weather_code_profile(95)["wc_mixed"] >= 0.99


def test_apparent_temperature_increases_heat_temp_norm() -> None:
    s = Settings()
    snap_air = WeatherSnapshot(temperature_c=25.0, apparent_temperature_c=None)
    snap_at = WeatherSnapshot(temperature_c=25.0, apparent_temperature_c=38.0)
    sig_air = _normalized_weather_signals(snap_air, s)
    sig_at = _normalized_weather_signals(snap_at, s)
    assert sig_air["heat_temp_norm"] < sig_at["heat_temp_norm"]


def test_open_tree_building_give_different_continuous_factors() -> None:
    def _wp() -> WeatherWeightParams:
        return WeatherWeightParams(
            enabled=True,
            mults=WeatherMultipliers(),
            heat_continuous=True,
            weather_response_scale=1.0,
            open_sky_penalty=1.1,
            tree_shade_bonus=1.05,
            building_shade_bonus=1.08,
            covered_bonus=1.05,
            wet_surface_penalty=1.1,
            wind_open_penalty=1.1,
            normalized_signals={"rain_norm": 0.0},
            heat_edge_k_open=0.54,
            heat_edge_k_tree=0.44,
            heat_edge_k_building=0.40,
            heat_edge_k_covered=0.16,
            heat_edge_k_wet=0.34,
            heat_edge_k_wind=0.36,
            heat_edge_rain_open_mult=0.34,
            heat_edge_rain_building_mult=0.42,
            heat_edge_rain_wind_exp_mult=0.16,
            heat_open_wet_synergy=0.14,
            heat_wind_exp_w3=0.35,
        )

    w = _wp()
    open_edge = {
        "thermal_open_sky_share": 0.9,
        "thermal_building_shade_share": 0.05,
        "thermal_vegetation_shade_share": 0.05,
        "thermal_covered_share": 0.0,
        "surface_effective": "asphalt",
        "edge_bearing_deg": 0.0,
    }
    tree_edge = dict(open_edge, thermal_open_sky_share=0.2, thermal_vegetation_shade_share=0.7)
    build_edge = dict(open_edge, thermal_open_sky_share=0.2, thermal_building_shade_share=0.7)
    fo = continuous_heat_edge_weather_factor(open_edge, w)
    ft = continuous_heat_edge_weather_factor(tree_edge, w)
    fb = continuous_heat_edge_weather_factor(build_edge, w)
    assert max(fo, ft, fb) - min(fo, ft, fb) > 1e-6


def test_rain_wind_stress_factors_finite_non_negative() -> None:
    w = WeatherWeightParams(
        enabled=True,
        mults=WeatherMultipliers(physical=1.05, stress=1.1),
        normalized_signals={
            "rain_norm": 0.8,
            "wind_norm": 0.7,
            "gust_norm": 0.5,
            "snow_depth_norm": 0.3,
            "snow_fresh_norm": 0.2,
        },
        stress_edge_rain_slip=0.2,
        stress_edge_wind_open=0.2,
        stress_edge_factor_min=0.82,
        stress_edge_factor_max=1.48,
    )
    edge = {
        "thermal_open_sky_share": 0.5,
        "thermal_building_shade_share": 0.3,
        "thermal_vegetation_shade_share": 0.2,
        "surface_effective": "asphalt",
        "stress_lts": 1.2,
        "edge_bearing_deg": 90.0,
    }
    f = weather_edge_stress_factor(edge, w)
    assert f == f and f >= 0.0


def test_continuous_heat_clamped() -> None:
    ed = {
        "thermal_open_sky_share": 0.85,
        "thermal_building_shade_share": 0.15,
        "thermal_vegetation_shade_share": 0.1,
        "thermal_covered_share": 0.0,
        "surface_effective": "asphalt",
        "edge_bearing_deg": 0.0,
    }
    w = WeatherWeightParams(
        enabled=True,
        mults=WeatherMultipliers(),
        heat_continuous=True,
        weather_response_scale=1.0,
        open_sky_penalty=1.1,
        tree_shade_bonus=1.05,
        building_shade_bonus=1.08,
        covered_bonus=1.05,
        wet_surface_penalty=1.1,
        wind_open_penalty=1.1,
        normalized_signals={"rain_norm": 0.0},
        heat_edge_k_open=0.54,
        heat_edge_k_tree=0.44,
        heat_edge_k_building=0.40,
        heat_edge_k_covered=0.16,
        heat_edge_k_wet=0.34,
        heat_edge_k_wind=0.36,
        heat_edge_rain_open_mult=0.34,
        heat_edge_rain_building_mult=0.42,
        heat_edge_rain_wind_exp_mult=0.16,
        heat_open_wet_synergy=0.14,
        heat_wind_exp_w3=0.35,
        heat_edge_factor_min=0.65,
        heat_edge_factor_max=1.75,
    )
    v = continuous_heat_edge_weather_factor(ed, w)
    assert 0.65 <= v <= 1.75


def test_routing_season_simple_dates() -> None:
    s = Settings()
    assert routing_season_label(date(2000, 2, 10), s) == "winter"
    assert routing_season_label(date(2000, 7, 10), s) == "green_season"


def test_green_multiplier_april_ramp() -> None:
    s = Settings()
    g10 = season_green_route_multiplier(date(2000, 4, 10), s)
    g20 = season_green_route_multiplier(date(2000, 4, 20), s)
    assert g10 < g20 - 1e-6


def test_snow_strength_winter_vs_green() -> None:
    s = Settings()
    snap_clear = WeatherSnapshot(snow_depth_m=0.0, snowfall_cm_h=0.0)
    snap_snow = WeatherSnapshot(snow_depth_m=0.12, snowfall_cm_h=0.5)
    assert snow_route_model_strength("green_season", snap_clear, s) < 0.2
    assert snow_route_model_strength("winter", snap_snow, s) >= 0.99


def test_snow_depth_multiplier_monotone() -> None:
    s = Settings()
    a = snow_depth_phys_multiplier(0.01, s)
    b = snow_depth_phys_multiplier(0.08, s)
    assert a <= b


def test_heat_season_resolution_override() -> None:
    now = datetime(2026, 7, 15, 12, 0, 0, tzinfo=timezone.utc)
    assert (
        _resolve_season_for_heat_alternatives(
            season_override="spring_autumn",
            weather_time_iso="2000-06-15T12:00:00+00:00",
            departure_time_iso=None,
            now_utc=now,
        )
        == "spring_autumn"
    )


def test_adaptive_season_without_snow() -> None:
    s = Settings()
    s.season_adaptive_mode = "adaptive_if_possible"
    d = date(2000, 2, 10)
    snap = snapshot_from_manual(temperature_c=12.0, snow_depth_m=0.0, snowfall_cm_h=0.0)
    ctx = resolve_season_routing_context(d, snap, s)
    assert ctx.calendar_season == "winter"
    assert ctx.effective_season == "early_spring"
