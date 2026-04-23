"""Последовательный запуск двух пакетных экспериментов с параметрами по умолчанию.

1. ``route_variants_experiment`` — шесть вариантов маршрута, текущая погода Open-Meteo.
2. ``heat_weather_experiment`` — heat на synthetic-сетке ``--grid summer``.

Запуск (из корня репозитория NIR)::

    python -m bike_router.tools.route_batch_experiment
"""

from __future__ import annotations

import logging
import os
import sys


def _ensure_pkg_path() -> None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if root not in sys.path:
        sys.path.insert(0, root)


def main() -> None:
    _ensure_pkg_path()
    from bike_router.tools._experiment_common import DEFAULT_BATCH_LOG_EVERY
    from bike_router.tools.heat_weather_experiment import run_heat_weather_experiment
    from bike_router.tools.route_variants_experiment import run_variants_experiment

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("route_batch_experiment")

    log.info("=== route_variants_experiment (defaults) ===")
    p1 = run_variants_experiment(
        n_points=10,
        profiles_mode="both",
        verbose=False,
        log_every=DEFAULT_BATCH_LOG_EVERY,
    )
    print(p1, flush=True)

    log.info("=== heat_weather_experiment (defaults: grid=summer) ===")
    p2 = run_heat_weather_experiment(
        n_points=10,
        profiles_mode="both",
        verbose=False,
        log_every=DEFAULT_BATCH_LOG_EVERY,
        weather_grid="summer",
    )
    print(p2, flush=True)


if __name__ == "__main__":
    main()
