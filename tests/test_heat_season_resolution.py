"""Сезон для heat: override, дата из ISO, fallback по месяцу now."""

from __future__ import annotations

from datetime import datetime, timezone

from bike_router.engine import _resolve_season_for_heat_alternatives


def test_season_override_wins() -> None:
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


def test_season_from_weather_time_when_no_override() -> None:
    now = datetime(2026, 7, 15, 12, 0, 0, tzinfo=timezone.utc)
    assert (
        _resolve_season_for_heat_alternatives(
            season_override=None,
            weather_time_iso="2000-02-10T12:00:00+00:00",
            departure_time_iso=None,
            now_utc=now,
        )
        == "spring_autumn"
    )


def test_season_from_departure_when_no_weather_time() -> None:
    now = datetime(2026, 7, 15, 12, 0, 0, tzinfo=timezone.utc)
    assert (
        _resolve_season_for_heat_alternatives(
            season_override=None,
            weather_time_iso=None,
            departure_time_iso="2000-07-05T10:00:00+00:00",
            now_utc=now,
        )
        == "summer"
    )


def test_season_fallback_now_month() -> None:
    now = datetime(2026, 7, 15, 12, 0, 0, tzinfo=timezone.utc)
    assert (
        _resolve_season_for_heat_alternatives(
            season_override=None,
            weather_time_iso=None,
            departure_time_iso=None,
            now_utc=now,
        )
        == "summer"
    )
