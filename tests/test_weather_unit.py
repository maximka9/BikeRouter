"""Юнит-тесты погодного сервиса (ветер, выбор archive/forecast)."""

from __future__ import annotations

import pytest


def test_weather_target_in_past_distant_past() -> None:
    from bike_router.services.weather import _weather_target_in_past

    assert _weather_target_in_past("2000-06-15T14:00:00") is True


def test_weather_target_in_past_distant_future() -> None:
    from bike_router.services.weather import _weather_target_in_past

    assert _weather_target_in_past("2099-12-01T10:00:00") is False


def test_cache_key_distinguishes_archive() -> None:
    from bike_router.services.weather import _cache_key

    w = "2026-07-10T15:00:00"
    assert _cache_key(53.2, 50.1, w, False) != _cache_key(53.2, 50.1, w, True)
