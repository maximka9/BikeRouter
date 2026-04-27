"""Сравнение шести вариантов маршрута при текущей погоде (один снимок Open-Meteo).

Один запрос погоды в центре полигона на момент запуска. Выход:
``bike_router/experiment_outputs/route_variants_experiment_YYYYMMDD_HHMMSS.xlsx`` (UTC в имени).

Параллель по умолчанию: авто (как в ``route_batch_experiment``; главный warmup до пула).

Запуск::

    python -m bike_router.tools.route_variants_experiment
    python -m bike_router.tools.route_variants_experiment --n-points 20 --profiles cyclist --verbose
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, Dict

_log = logging.getLogger(__name__)


def _ensure_pkg_path() -> None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if root not in sys.path:
        sys.path.insert(0, root)


def run_variants_experiment(
    *,
    n_points: int,
    profiles_mode: str,
    verbose: bool,
    log_every: int,
) -> str:
    _ensure_pkg_path()

    from bike_router.config import Settings
    from bike_router.services.area_graph_cache import parse_precache_polygon

    from bike_router.tools._experiment_common import (
        mp_resolve_pool_workers,
        resolve_live_weather_once_for_polygon,
        run_variants_over_weather_cases,
    )

    settings = Settings()
    poly = parse_precache_polygon(settings)

    w_snap, wsrc, wp_meta, dep_iso, lat_c, lon_c = resolve_live_weather_once_for_polygon(
        poly, settings=settings
    )
    _log.info(
        "Погода (текущий снимок): источник=%s dep=%s center_lat=%.6f center_lon=%.6f",
        wsrc,
        dep_iso,
        lat_c,
        lon_c,
    )

    fixed_kw: Dict[str, Any] = {
        "weather_mode": "fixed-snapshot",
        "use_live_weather": False,
        "weather_time": dep_iso,
        "temperature_c": float(w_snap.temperature_c),
        "precipitation_mm": float(w_snap.precipitation_mm),
        "wind_speed_ms": float(w_snap.wind_speed_ms),
        "cloud_cover_pct": float(w_snap.cloud_cover_pct),
        "humidity_pct": float(w_snap.humidity_pct),
        "wind_gusts_ms": w_snap.wind_gusts_ms,
        "shortwave_radiation_wm2": w_snap.shortwave_radiation_wm2,
    }
    wd_live = getattr(w_snap, "wind_direction_deg", None)
    if wd_live is not None:
        fixed_kw["wind_direction_deg"] = float(wd_live)

    mw = mp_resolve_pool_workers(0)
    return run_variants_over_weather_cases(
        script_stem="route_variants_experiment",
        n_points=n_points,
        profiles_mode=profiles_mode,
        synthetic_weather_grid=None,
        fixed_weather_kw=fixed_kw,
        live_weather_bundle=(w_snap, wsrc, wp_meta, dep_iso, lat_c, lon_c),
        weather_grid_label="live_snapshot",
        verbose=verbose,
        log_every=log_every,
        max_workers=mw,
        chunk_size=1,
        mp_weather_chunk_size=25,
        directed_pairs=True,
        write_vertices=True,
    )


def main() -> None:
    from bike_router.tools._experiment_common import DEFAULT_BATCH_LOG_EVERY

    parser = argparse.ArgumentParser(
        description=(
            "Шесть вариантов маршрута при одном снимке Open-Meteo "
            "(текущее время, центр полигона арены)."
        )
    )
    parser.add_argument("--n-points", type=int, default=10, help="Число случайных точек")
    parser.add_argument(
        "--profiles",
        choices=("both", "cyclist", "pedestrian"),
        default="both",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--log-every",
        type=int,
        default=DEFAULT_BATCH_LOG_EVERY,
        metavar="N",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    path = run_variants_experiment(
        n_points=args.n_points,
        profiles_mode=args.profiles,
        verbose=bool(args.verbose),
        log_every=max(0, int(args.log_every)),
    )
    print(path)


if __name__ == "__main__":
    main()
