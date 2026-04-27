"""Комбинированный батч: 6 вариантов маршрута и synthetic-сетка погоды (как heat_weather).

Один Excel: направленные пары A->B и B->A, ``compute_alternatives`` один раз на
(пара, профиль, погодный кейс). Параллель: ``min(6, CPU-1)`` процессов, чанк погоды 25,
``pool.imap`` chunksize 1.

Число строк маршрутов примерно ``n*(n-1) * n_profiles * n_weather * 6`` при направленных парах
(по умолчанию). Сетки: summer=95, winter=135, all=230.

Запуск::

    python -m bike_router.tools.route_batch_experiment
    python -m bike_router.tools.route_batch_experiment --n-points 10 --weather-grid summer
    python -m bike_router.tools.route_batch_experiment --weather-grid summer
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


def run_combined_route_batch_experiment(
    *,
    n_points: int,
    profiles_mode: str,
    weather_grid: str,
    verbose: bool,
    log_every: int,
    directed_pairs: bool,
) -> str:
    _ensure_pkg_path()
    from bike_router.tools._experiment_common import (
        mp_resolve_pool_workers,
        weather_summer_heat_grid,
        weather_winter_heat_grid,
    )
    from bike_router.tools._run_variants_weather_batch import run_variants_over_weather_cases

    wg = (weather_grid or "all").strip().lower()
    if wg in ("summer_wind", "summer-wind"):
        wg = "summer"
    summer_g = weather_summer_heat_grid()
    winter_g = weather_winter_heat_grid()
    if wg == "all":
        grid = summer_g + winter_g
    elif wg == "winter":
        grid = winter_g
    else:
        grid = summer_g
    assert len(grid) in (95, 135, 230)

    mw = mp_resolve_pool_workers(0)
    return run_variants_over_weather_cases(
        script_stem="route_batch_experiment",
        n_points=n_points,
        profiles_mode=profiles_mode,
        synthetic_weather_grid=grid,
        fixed_weather_kw=None,
        live_weather_bundle=None,
        weather_grid_label=wg,
        verbose=verbose,
        log_every=log_every,
        max_workers=mw,
        chunk_size=1,
        mp_weather_chunk_size=25,
        directed_pairs=directed_pairs,
        write_vertices=False,
    )


def main() -> None:
    _ensure_pkg_path()
    from bike_router.tools._experiment_common import DEFAULT_BATCH_LOG_EVERY

    parser = argparse.ArgumentParser(
        description=(
            "Один Excel: 6 вариантов, synthetic-сетка погоды, направленные O-D, профили."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Сетки: summer=95, winter=135, all=230. По умолчанию пары направленные n*(n-1). "
            "Вершины в csv.gz не пишутся. Параллель: min(6, CPU-1) процессов, чанк погоды 25."
        ),
    )
    parser.add_argument("--n-points", type=int, default=10, metavar="N")
    parser.add_argument(
        "--profiles",
        choices=("both", "cyclist", "pedestrian"),
        default="both",
    )
    parser.add_argument(
        "--weather-grid",
        choices=("summer", "winter", "all"),
        default="all",
        metavar="NAME",
        dest="weather_grid",
        help="Synthetic-сетка (как в heat_weather_experiment).",
    )
    parser.add_argument(
        "--undirected-pairs",
        action="store_true",
        help="Только пары i<j; по умолчанию направленные A->B и B->A.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--log-every",
        type=int,
        default=DEFAULT_BATCH_LOG_EVERY,
        metavar="N",
    )
    parser.add_argument(
        "--surface-ai-runtime",
        choices=("off", "on", "both"),
        default=None,
        help=(
            "Если задано: только сравнение одного маршрута (START/END из .env) "
            "с ML runtime off|on|both (both = off,on подряд); печать JSON в stdout."
        ),
    )
    args = parser.parse_args()
    if args.surface_ai_runtime is not None:
        _ensure_pkg_path()
        from bike_router.tools.surface_runtime_route_experiment import run_modes

        modes = "off,on" if args.surface_ai_runtime == "both" else str(args.surface_ai_runtime)
        import json

        print(json.dumps(run_modes(modes), ensure_ascii=False, indent=2), flush=True)
        return

    n = max(2, int(args.n_points))
    directed = not bool(args.undirected_pairs)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    path = run_combined_route_batch_experiment(
        n_points=n,
        profiles_mode=str(args.profiles),
        weather_grid=str(args.weather_grid),
        verbose=bool(args.verbose),
        log_every=max(0, int(args.log_every)),
        directed_pairs=directed,
    )
    print(path, flush=True)


if __name__ == "__main__":
    main()
