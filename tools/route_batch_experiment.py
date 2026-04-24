"""Последовательный запуск двух пакетных экспериментов с параметрами по умолчанию.

1. ``route_variants_experiment`` — шесть вариантов маршрута, текущая погода Open-Meteo.
2. ``heat_weather_experiment`` — heat на synthetic-сетке (по умолчанию ``--heat-grid all``).

Число маршрутов heat ≈ ``n*(n-1)/2 * n_profiles * n_weather`` (пары неориентированные;
``n_weather``: summer 90, winter 135, all 225). Параллель: в ``heat_weather_experiment``
``--max-workers 0``. См. ``--help``.

Запуск (из корня репозитория, где доступен пакет ``bike_router``)::

    python -m bike_router.tools.route_batch_experiment
    python -m bike_router.tools.route_batch_experiment --n-points 2
    python -m bike_router.tools.route_batch_experiment --n-points 10 --heat-grid summer
"""

from __future__ import annotations

import argparse
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

    parser = argparse.ArgumentParser(
        description="Подряд route_variants_experiment и heat_weather_experiment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Сетки heat: summer=90, winter=135, all=225. Шагов ≈ n*(n-1)/2 * profiles * сетка "
            "(неориентированные пары; directed: удвоить). Параллель в heat: --max-workers."
        ),
    )
    parser.add_argument("--n-points", type=int, default=10, metavar="N")
    parser.add_argument(
        "--profiles",
        choices=("both", "cyclist", "pedestrian"),
        default="both",
    )
    parser.add_argument(
        "--heat-grid",
        choices=("summer", "winter", "all"),
        default="all",
        metavar="NAME",
        help="Сетка synthetic для второго шага (summer быстрее all примерно в 2.5 раза).",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log-every", type=int, default=DEFAULT_BATCH_LOG_EVERY)
    args = parser.parse_args()
    n = max(2, int(args.n_points))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("route_batch_experiment")

    log.info("=== route_variants_experiment (n_points=%d, profiles=%s) ===", n, args.profiles)
    p1 = run_variants_experiment(
        n_points=n,
        profiles_mode=str(args.profiles),
        verbose=bool(args.verbose),
        log_every=max(0, int(args.log_every)),
    )
    print(p1, flush=True)

    log.info(
        "=== heat_weather_experiment (grid=%s, n_points=%d) ===",
        args.heat_grid,
        n,
    )
    p2 = run_heat_weather_experiment(
        n_points=n,
        profiles_mode=str(args.profiles),
        verbose=bool(args.verbose),
        log_every=max(0, int(args.log_every)),
        weather_grid=str(args.heat_grid),
    )
    print(p2, flush=True)


if __name__ == "__main__":
    main()
