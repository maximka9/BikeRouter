"""Synthetic apparent temperature, snapshot и непрерывная heat-модель."""

from __future__ import annotations

from bike_router.services.weather import (
    WeatherSnapshot,
    _continuous_six_coefficients,
    _normalized_weather_signals,
    compute_weather_multipliers,
    snapshot_from_manual,
)
from bike_router.config import Settings


def test_synthetic_weather_case_apparent_temperature() -> None:
    from bike_router.tools._experiment_common import weather_summer_heat_grid

    g = weather_summer_heat_grid()
    ids = {c.case_id for c in g}
    for cid in (
        "app_T25_AT25_neutral_hot",
        "app_T25_AT32_humid_hot",
        "app_T25_AT38_extreme_hot",
        "app_T125_AT5_cold_wind",
        "app_T0_ATm8_frost_like",
    ):
        assert cid in ids
    ext = [c for c in g if c.case_id == "app_T25_AT38_extreme_hot"][0]
    assert ext.temperature_c == 25.0
    assert ext.apparent_temperature_c == 38.0


def test_manual_weather_accepts_apparent_temperature() -> None:
    s = snapshot_from_manual(
        temperature_c=25.0,
        apparent_temperature_c=38.0,
        shortwave_radiation_wm2=750.0,
    )
    assert s.temperature_c == 25.0
    assert s.apparent_temperature_c == 38.0


def test_effective_heat_temperature_prefers_apparent_in_signals() -> None:
    s = Settings()
    snap_air = WeatherSnapshot(temperature_c=25.0, apparent_temperature_c=None)
    snap_at = WeatherSnapshot(temperature_c=25.0, apparent_temperature_c=38.0)
    sig_air = _normalized_weather_signals(snap_air, s)
    sig_at = _normalized_weather_signals(snap_at, s)
    assert sig_air["heat_temp_norm"] < sig_at["heat_temp_norm"]
    assert sig_at["apparent_minus_air_norm"] > 0.5


def test_compute_weather_multipliers_heat_uses_apparent_not_max_air() -> None:
    """При AT < T жара в heat-mult не должна расти из-за max(T, Ta)."""
    snap = WeatherSnapshot(
        temperature_c=30.0,
        apparent_temperature_c=20.0,
        cloud_cover_pct=50.0,
    )
    m = compute_weather_multipliers(snap)
    snap_no_at = WeatherSnapshot(temperature_c=30.0, apparent_temperature_c=None)
    m2 = compute_weather_multipliers(snap_no_at)
    assert m.heat < m2.heat


def test_heat_open_sky_penalty_increases_with_apparent_temperature() -> None:
    s = Settings()
    base = WeatherSnapshot(
        temperature_c=25.0,
        apparent_temperature_c=25.0,
        shortwave_radiation_wm2=750.0,
        cloud_cover_pct=0.0,
        wind_speed_ms=2.0,
        precipitation_mm=0.0,
    )
    hot = WeatherSnapshot(
        temperature_c=25.0,
        apparent_temperature_c=38.0,
        shortwave_radiation_wm2=750.0,
        cloud_cover_pct=0.0,
        wind_speed_ms=2.0,
        precipitation_mm=0.0,
        humidity_pct=78.0,
    )
    sig_b = _normalized_weather_signals(base, s)
    sig_h = _normalized_weather_signals(hot, s)
    _, osp_b, _, _, _, _ = _continuous_six_coefficients(sig_b, s)
    _, osp_h, _, _, _, _ = _continuous_six_coefficients(sig_h, s)
    assert osp_h >= osp_b
