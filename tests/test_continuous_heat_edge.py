"""Чувствительность continuous_heat_edge_weather_factor к k_open/k_tree и дождю (O/B)."""

from __future__ import annotations

from bike_router.services.routing import continuous_heat_edge_weather_factor
from bike_router.services.weather import WeatherMultipliers, WeatherWeightParams


def _edge(
    *,
    O: float = 0.85,
    B: float = 0.15,
    V: float = 0.1,
    C: float = 0.0,
    surface: str = "asphalt",
) -> dict:
    return {
        "thermal_open_sky_share": O,
        "thermal_building_shade_share": B,
        "thermal_vegetation_shade_share": V,
        "thermal_covered_share": C,
        "surface_effective": surface,
    }


def _wp(
    *,
    k_open: float = 0.54,
    k_tree: float = 0.44,
    rain_norm: float = 0.0,
    rain_open_mult: float = 0.34,
    rain_building_mult: float = 0.42,
    rain_wind_exp_mult: float = 0.16,
    wind_w3: float = 0.35,
) -> WeatherWeightParams:
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
        normalized_signals={"rain_norm": rain_norm},
        heat_edge_k_open=k_open,
        heat_edge_k_tree=k_tree,
        heat_edge_k_building=0.40,
        heat_edge_k_covered=0.16,
        heat_edge_k_wet=0.34,
        heat_edge_k_wind=0.36,
        heat_edge_rain_open_mult=rain_open_mult,
        heat_edge_rain_building_mult=rain_building_mult,
        heat_edge_rain_wind_exp_mult=rain_wind_exp_mult,
        heat_open_wet_synergy=0.14,
        heat_wind_exp_w3=wind_w3,
    )


def test_higher_k_open_increases_factor() -> None:
    ed = _edge()
    lo = continuous_heat_edge_weather_factor(ed, _wp(k_open=0.30))
    hi = continuous_heat_edge_weather_factor(ed, _wp(k_open=0.70))
    assert hi > lo


def test_higher_k_tree_decreases_factor_with_vegetation() -> None:
    ed = _edge(O=0.5, V=0.6, B=0.1)
    lo = continuous_heat_edge_weather_factor(ed, _wp(k_tree=0.20))
    hi = continuous_heat_edge_weather_factor(ed, _wp(k_tree=0.60))
    assert hi < lo


def test_rain_penalizes_open_more_than_dry() -> None:
    ed = _edge(O=0.55, B=0.2, V=0.15)
    dry = continuous_heat_edge_weather_factor(
        ed, _wp(rain_norm=0.0, k_open=0.35, rain_open_mult=0.5)
    )
    wet = continuous_heat_edge_weather_factor(
        ed, _wp(rain_norm=1.0, k_open=0.35, rain_open_mult=0.5)
    )
    assert wet > dry < 1.74


def test_rain_rewards_building_shade_more_than_dry() -> None:
    ed = _edge(O=0.25, B=0.85, V=0.05)
    dry = continuous_heat_edge_weather_factor(ed, _wp(rain_norm=0.0))
    wet = continuous_heat_edge_weather_factor(ed, _wp(rain_norm=1.0))
    assert wet < dry


def test_covered_share_near_zero_does_not_dominate() -> None:
    """При w3=0 вклад C только в −k4*cbn*C (ветровая экспозиция не зависит от C)."""
    ed0 = _edge(C=0.0)
    ed1 = _edge(C=1.0)
    wp = _wp(k_open=0.25, k_tree=0.25, wind_w3=0.0)
    f0 = continuous_heat_edge_weather_factor(ed0, wp)
    f1 = continuous_heat_edge_weather_factor(ed1, wp)
    assert abs(f1 - f0) < 0.22
