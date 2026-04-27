"""Smoke: тяжёлые эксперименты не запускаются из pytest, только импорт CLI."""

from __future__ import annotations


def test_surface_ai_experiment_import_and_main() -> None:
    from bike_router.tools import surface_ai_experiment as m

    assert callable(getattr(m, "main", None))


def test_route_batch_experiment_import_and_main() -> None:
    from bike_router.tools import route_batch_experiment as m

    assert callable(getattr(m, "main", None))


def test_surface_runtime_route_experiment_import_and_main() -> None:
    from bike_router.tools import surface_runtime_route_experiment as m

    assert callable(getattr(m, "main", None))
