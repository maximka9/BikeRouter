"""Комбинированный батч: 6 вариантов маршрута и synthetic-сетка погоды (как heat_weather).

Один Excel: направленные пары A->B и B->A, ``compute_alternatives`` один раз на
(пара, профиль, погодный кейс). Параллель по умолчанию: авто (4–6 воркеров, не больше числа задач), чанк погоды 25,
``pool.imap`` chunksize 1. Главный процесс делает warmup графа до пула.

Число строк маршрутов примерно ``n*(n-1) * n_profiles * n_weather * 6`` при направленных парах
(по умолчанию). Сетки: ``summer``=95, ``winter``=135, ``all``=230 (лето+зима).

Запуск::

    python -m bike_router.tools.route_batch_experiment
    python -m bike_router.tools.route_batch_experiment --n-points 10 --weather-grid summer
    python -m bike_router.tools.route_batch_experiment --weather-grid winter
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

logger = logging.getLogger(__name__)


def _ensure_pkg_path() -> None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if root not in sys.path:
        sys.path.insert(0, root)


def _route_batch_output_xlsx_suffix(
    *,
    n_points: int,
    directed_pairs: bool,
    weather_grid: str,
    profiles_mode: str,
    surface_ai_runtime_marker: str = "",
) -> str:
    """Часть имени Excel без даты: точки, направления пар, сетка погоды, профили."""
    wg = (weather_grid or "all").strip().lower()
    if wg in ("summer_wind", "summer-wind"):
        wg = "summer"
    pair = "directed_AB_BA" if directed_pairs else "undirected_only"
    prof = (profiles_mode or "both").strip().lower().replace(" ", "_")
    parts = [f"n{n_points}", pair, wg, prof]
    marker = (surface_ai_runtime_marker or "").strip().lower()
    if marker:
        parts.append(marker)
    return "_".join(parts)


def _surface_ai_runtime_filename_marker() -> str:
    """Маркер для имени Excel: только если runtime ML включён в настройках."""
    _ensure_pkg_path()
    from bike_router.config import Settings

    settings = Settings()
    if not settings.surface_ai_runtime_enabled:
        return ""

    from bike_router.services.surface_prediction_store import SurfacePredictionStore

    store = SurfacePredictionStore(settings)
    store.load()
    if store.loaded:
        logger.info(
            "route_batch_experiment: Surface AI runtime ML active; Excel suffix includes ml_on"
        )
        return "ml_on"

    logger.warning(
        "route_batch_experiment: SURFACE_AI_RUNTIME_ENABLED=True, "
        "but SurfacePredictionStore.loaded=False; Excel suffix includes ml_unloaded "
        "(reason=%s)",
        store.failure_reason or "unknown",
    )
    return "ml_unloaded"


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
    surface_ai_runtime_marker = _surface_ai_runtime_filename_marker()
    suffix = _route_batch_output_xlsx_suffix(
        n_points=n_points,
        directed_pairs=directed_pairs,
        weather_grid=wg,
        profiles_mode=profiles_mode,
        surface_ai_runtime_marker=surface_ai_runtime_marker,
    )
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
        output_xlsx_suffix=suffix,
        include_surface_ml_report=(surface_ai_runtime_marker == "ml_on"),
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
            "Сетки: summer=95, winter=135, all=лето+зима (230). По умолчанию пары направленные A→B и B→A; "
            "флаг --undirected-pairs оставляет только i<j. Excel: имя файла с параметрами (без даты). "
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
        help="Synthetic-сетка: summer (только лето), winter (только зима), all (лето и зима).",
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
            "Если задано: сравнение маршрута START/END из .env с ML runtime "
            "off|on|both (both = off,on); JSON в stdout. Несколько пар O-D: "
            "python -m bike_router.tools.surface_runtime_route_experiment --pairs-csv ..."
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
