"""Импорт и сетка synthetic; модуль route_batch_experiment."""

from __future__ import annotations

import pytest


def test_weather_windows_test_grid_len_36() -> None:
    from bike_router.tools._experiment_common import weather_windows_test_grid

    g = weather_windows_test_grid()
    assert len(g) == 36


def test_weather_summer_wind_grid_len_144() -> None:
    from bike_router.tools._experiment_common import weather_summer_test_grid_with_wind_dirs

    g = weather_summer_test_grid_with_wind_dirs()
    assert len(g) == 144
    assert all(getattr(c, "wind_direction_deg", None) is not None for c in g)


def test_weather_winter_wind_grid_len_216() -> None:
    from bike_router.tools._experiment_common import weather_winter_synthetic_grid_with_wind_dirs

    g = weather_winter_synthetic_grid_with_wind_dirs()
    assert len(g) == 216
    assert all(getattr(c, "wind_direction_deg", None) is not None for c in g)


def test_combined_summer_winter_grid_len_360() -> None:
    from bike_router.tools._experiment_common import (
        weather_summer_test_grid_with_wind_dirs,
        weather_winter_synthetic_grid_with_wind_dirs,
    )

    g = weather_summer_test_grid_with_wind_dirs() + weather_winter_synthetic_grid_with_wind_dirs()
    assert len(g) == 360


def test_route_batch_experiment_module_has_main() -> None:
    from bike_router.tools import route_batch_experiment as m

    assert callable(getattr(m, "main", None))


def test_resolve_live_weather_once_single_call(monkeypatch: pytest.MonkeyPatch) -> None:
    from bike_router.config import Settings
    from bike_router.services.weather import WeatherSnapshot, WeatherWeightParams
    from bike_router.tools._experiment_common import (
        centroid_lat_lon_for_weather,
        resolve_live_weather_once_for_polygon,
    )

    calls: list = []

    def fake_resolve(*a, **k):
        calls.append((a, k))
        snap = WeatherSnapshot(
            temperature_c=21.0,
            precipitation_mm=0.0,
            wind_speed_ms=3.0,
            cloud_cover_pct=40.0,
            humidity_pct=55.0,
        )
        wp = WeatherWeightParams(enabled=True, heat_continuous=True)
        return snap, "unit_test", wp

    monkeypatch.setattr(
        "bike_router.engine._resolve_route_weather", fake_resolve, raising=True
    )
    s = Settings()
    class _Poly:
        bounds = (50.0, 53.0, 51.0, 54.0)

    lat, lon = centroid_lat_lon_for_weather(_Poly())
    assert abs(lat - 53.5) < 1e-9 and abs(lon - 50.5) < 1e-9

    snap, src, wp, dep, la, lo = resolve_live_weather_once_for_polygon(
        _Poly(), settings=s
    )
    assert len(calls) == 1
    assert src == "unit_test"
    assert snap.temperature_c == 21.0
    assert wp.enabled is True
    assert isinstance(dep, str) and len(dep) > 4
    assert la == lat and lo == lon
