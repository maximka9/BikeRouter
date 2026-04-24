"""Spawn-safe воркер для параллельного heat_weather_experiment (Windows).

Один процесс: один RouteEngine. Одна задача пула — одна O-D-пара × профиль × **чанк**
synthetic-кейсов (чтобы прогресс-бар обновлялся чаще, чем раз в полную сетку).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

_POINTS: List[Tuple[float, float]] = ()
_CORRIDOR: Tuple[float, ...] = (10.0, 100.0)
_ENGINE: Any = None


def init_worker(
    points: List[Tuple[float, float]],
    corridor: Tuple[float, ...],
) -> None:
    global _POINTS, _CORRIDOR, _ENGINE
    _POINTS = tuple(points)
    _CORRIDOR = tuple(corridor)
    for name in (
        "bike_router.engine",
        "bike_router.services.weather",
        "bike_router.services.routing",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)
    from bike_router.engine import RouteEngine

    _ENGINE = RouteEngine()
    _ENGINE.warmup()
    if _ENGINE.graph is None or not _ENGINE.is_loaded:
        raise RuntimeError("heat mp worker: warmup не загрузил граф")


def run_pair_profile_task(
    packed: Tuple[str, int, int, str, int, List[Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int, int, int]:
    """experiment_id, seed, i, j, prof, grid_chunk → rows (route_id=0), failures, ok, fail, skip."""
    experiment_id, seed, i, j, prof, grid_chunk = packed
    from bike_router.exceptions import BikeRouterError, RouteNotFoundError
    from bike_router.tools._experiment_common import (
        SYNTHETIC_TEST_WEATHER_ISO,
        kwargs_fixed_snapshot_from_case,
        route_to_raw_row,
        _point_id_fmt,
    )

    if _ENGINE is None:
        raise RuntimeError("heat mp worker: engine не инициализирован")

    o_lat, o_lon = _POINTS[i]
    d_lat, d_lon = _POINTS[j]
    start = (o_lat, o_lon)
    end = (d_lat, d_lon)
    raw_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    n_ok = n_fail = n_skip = 0
    corr = _CORRIDOR

    for syn_case in grid_chunk:
        wkw = kwargs_fixed_snapshot_from_case(syn_case)
        test_meta = {
            "weather_test_case_id": syn_case.case_id,
            "weather_test_temperature_c": syn_case.temperature_c,
            "weather_test_precipitation_mm": syn_case.precipitation_mm,
            "weather_test_wind_speed_ms": syn_case.wind_speed_ms,
            "weather_test_wind_gusts_ms": syn_case.wind_gusts_ms,
            "weather_test_cloud_cover_pct": syn_case.cloud_cover_pct,
            "weather_test_humidity_pct": syn_case.humidity_pct,
            "weather_test_shortwave_radiation_wm2": syn_case.shortwave_radiation_wm2,
            "weather_test_wind_direction_deg": getattr(
                syn_case, "wind_direction_deg", None
            ),
            "weather_test_snowfall_cm_h": float(
                getattr(syn_case, "snowfall_cm_h", 0.0) or 0.0
            ),
            "weather_test_snow_depth_m": float(
                getattr(syn_case, "snow_depth_m", 0.0) or 0.0
            ),
            "weather_test_weather_code": getattr(syn_case, "weather_code", None),
            "weather_test_time_iso": getattr(
                syn_case, "weather_time_iso", SYNTHETIC_TEST_WEATHER_ISO
            ),
        }
        try:
            alt = _ENGINE.compute_heat_alternative(
                start=start,
                end=end,
                profile_key=prof,
                green_enabled=True,
                corridor_expand_schedule_meters=corr,
                departure_time=str(wkw.get("weather_time") or ""),
                **wkw,
            )
        except RouteNotFoundError:
            n_skip += 1
            continue
        except (BikeRouterError, Exception) as e:
            code = getattr(e, "code", type(e).__name__)
            failures.append(
                {
                    "origin_point_id": _point_id_fmt(i),
                    "destination_point_id": _point_id_fmt(j),
                    "profile": prof,
                    "variant": "heat",
                    "error_code": str(code),
                    "error_message": str(e),
                }
            )
            n_fail += 1
            continue

        if not alt.routes:
            failures.append(
                {
                    "origin_point_id": _point_id_fmt(i),
                    "destination_point_id": _point_id_fmt(j),
                    "profile": prof,
                    "variant": "heat",
                    "error_code": "EMPTY",
                    "error_message": "Нет маршрутов в ответе",
                }
            )
            n_fail += 1
            continue

        r = alt.routes[0]
        row = route_to_raw_row(
            experiment_id=experiment_id,
            seed=seed,
            route_id=0,
            profile=prof,
            origin_id=i,
            dest_id=j,
            o_lat=o_lat,
            o_lon=o_lon,
            d_lat=d_lat,
            d_lon=d_lon,
            r=r,
            baseline_full=None,
            weather_date="",
            test_weather_meta=test_meta,
        )
        raw_rows.append(row)
        n_ok += 1

    return raw_rows, failures, n_ok, n_fail, n_skip
