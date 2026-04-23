"""Только маршрут ``heat`` на сетке из 36 синтетических погод (без Open-Meteo).

Запуск::

    python -m bike_router.tools.heat_weather_experiment --n-points 10

По умолчанию для чистоты анализа heat снижается ``WEATHER_STRESS_GLOBAL_BLEND``
до 0.12 на время прогона (не трогая .env). См. ``--stress-blend-from-settings``.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

_log = logging.getLogger(__name__)


def _ensure_pkg_path() -> None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if root not in sys.path:
        sys.path.insert(0, root)


def _apply_limit_offset(
    tasks: List[Tuple[int, int, str]],
    *,
    limit: Optional[int],
    offset: int,
) -> List[Tuple[int, int, str]]:
    off = max(0, int(offset))
    out = tasks[off:]
    if limit is not None and int(limit) > 0:
        out = out[: int(limit)]
    return out


def run_heat_weather_experiment(
    *,
    n_points: int,
    seed: int,
    min_spacing_m: float,
    max_sample_attempts: int,
    profiles_mode: str,
    verbose: bool,
    log_every: int,
    output_path: Optional[str],
    limit: Optional[int],
    offset: int,
    stress_blend_from_settings: bool,
    weather_stress_global_blend: Optional[float],
    stress_blend_analysis_default: bool,
) -> str:
    _ensure_pkg_path()

    from bike_router.config import ROUTING_ALGO_VERSION, Settings, routing_engine_cache_fingerprint
    from bike_router.engine import RouteEngine
    from bike_router.exceptions import BikeRouterError, RouteNotFoundError
    from bike_router.services.area_graph_cache import parse_precache_polygon

    from bike_router.tools._experiment_common import (
        DEFAULT_BATCH_LOG_EVERY,
        DEFAULT_MAX_SAMPLE_ATTEMPTS,
        DEFAULT_MIN_SPACING_M,
        DEFAULT_ROUTE_BATCH_SEED,
        EXPERIMENT_CORRIDOR_EXPAND_M,
        SYNTHETIC_TEST_WEATHER_ISO,
        _batch_profiles_from_arg,
        _direction_key,
        _iter_directed_pairs,
        _point_id_fmt,
        _set_quiet_mode_for_batch,
        build_heat_weather_kpi_rows,
        build_pair_comparison,
        build_summaries,
        heat_weather_output_xlsx_path,
        kwargs_fixed_snapshot_from_case,
        route_to_raw_row,
        sample_points_in_polygon,
        weather_windows_test_grid,
        write_route_vertices_csv_gzip,
        write_xlsx,
    )

    import numpy as np
    from tqdm import tqdm

    settings = Settings()
    if not settings.has_precache_area_polygon:
        raise SystemExit(
            "В .env должен быть задан PRECACHE_AREA_POLYGON_WKT (полигон арены)."
        )

    experiment_id = str(uuid.uuid4())
    wkt_stripped = settings.precache_area_polygon_wkt_stripped
    precache_wkt_sha256 = hashlib.sha256(wkt_stripped.encode("utf-8")).hexdigest()
    poly = parse_precache_polygon(settings)
    minx, miny, maxx, maxy = poly.bounds
    rng = np.random.default_rng(seed)

    t_start = time.perf_counter()
    eng = RouteEngine()
    eng.warmup()
    if eng.graph is None or not eng.is_loaded:
        raise SystemExit("Warmup не загрузил граф.")

    project_stress_blend = float(eng._app.settings.weather_stress_global_blend)
    effective_stress_blend: Optional[float] = None
    stress_blend_note = "project_settings"
    if stress_blend_from_settings:
        effective_stress_blend = project_stress_blend
    elif weather_stress_global_blend is not None:
        eng._app.settings.weather_stress_global_blend = float(
            weather_stress_global_blend
        )
        effective_stress_blend = float(weather_stress_global_blend)
        stress_blend_note = (
            "default_analysis" if stress_blend_analysis_default else "cli_override"
        )
        _log.info(
            "Heat experiment: WEATHER_STRESS_GLOBAL_BLEND %.4f -> %.4f (для прогона)",
            project_stress_blend,
            effective_stress_blend,
        )
    else:
        stress_blend_note = "default_analysis"
        effective_stress_blend = 0.12
        eng._app.settings.weather_stress_global_blend = effective_stress_blend
        _log.info(
            "Heat experiment: WEATHER_STRESS_GLOBAL_BLEND %.4f -> %.4f "
            "(дефолт анализа; отключить: --stress-blend-from-settings)",
            project_stress_blend,
            effective_stress_blend,
        )

    points = sample_points_in_polygon(
        poly=poly,
        n_target=n_points,
        rng=rng,
        engine=eng,
        min_spacing_m=min_spacing_m,
        max_attempts=max_sample_attempts,
    )
    n_pairs = n_points * (n_points - 1)
    profile_tuple = _batch_profiles_from_arg(profiles_mode)
    grid = weather_windows_test_grid()
    assert len(grid) == 36

    route_tasks: List[Tuple[int, int, str]] = [
        (i, j, prof)
        for i, j in _iter_directed_pairs(n_points)
        for prof in profile_tuple
    ]
    route_tasks = _apply_limit_offset(route_tasks, limit=limit, offset=offset)

    expected_routes = len(route_tasks) * len(grid)
    raw_rows: List[Dict[str, Any]] = []
    vertices: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    route_id_counter = 1
    n_ok = 0
    n_fail = 0
    n_skip = 0

    _set_quiet_mode_for_batch(verbose)
    total_steps = len(route_tasks) * len(grid)
    pbar = tqdm(
        total=total_steps,
        desc="Heat×36 погод",
        unit="запрос",
        leave=True,
        mininterval=2.0 if not verbose else 0.3,
    )

    for syn_case in grid:
        wkw = kwargs_fixed_snapshot_from_case(
            syn_case, weather_time_iso=SYNTHETIC_TEST_WEATHER_ISO
        )
        for i, j, prof in route_tasks:
            o_pt = points[i]
            d_pt = points[j]
            start = (o_pt.lat, o_pt.lon)
            end = (d_pt.lat, d_pt.lon)
            pbar.set_postfix(
                case=syn_case.case_id,
                od=_direction_key(i, j),
                p=prof,
                ok=n_ok,
                refresh=False,
            )
            test_meta = {
                "weather_test_case_id": syn_case.case_id,
                "weather_test_temperature_c": syn_case.temperature_c,
                "weather_test_precipitation_mm": syn_case.precipitation_mm,
                "weather_test_wind_speed_ms": syn_case.wind_speed_ms,
                "weather_test_wind_gusts_ms": syn_case.wind_gusts_ms,
                "weather_test_cloud_cover_pct": syn_case.cloud_cover_pct,
                "weather_test_humidity_pct": syn_case.humidity_pct,
                "weather_test_shortwave_radiation_wm2": syn_case.shortwave_radiation_wm2,
            }
            try:
                alt = eng.compute_heat_alternative(
                    start=start,
                    end=end,
                    profile_key=prof,
                    green_enabled=True,
                    corridor_expand_schedule_meters=EXPERIMENT_CORRIDOR_EXPAND_M,
                    departure_time=SYNTHETIC_TEST_WEATHER_ISO,
                    **wkw,
                )
            except RouteNotFoundError:
                n_skip += 1
                pbar.update(1)
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
                pbar.update(1)
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
                pbar.update(1)
                continue

            r = alt.routes[0]
            rid = route_id_counter
            route_id_counter += 1
            row = route_to_raw_row(
                experiment_id=experiment_id,
                seed=seed,
                route_id=rid,
                profile=prof,
                origin_id=i,
                dest_id=j,
                o_lat=o_pt.lat,
                o_lon=o_pt.lon,
                d_lat=d_pt.lat,
                d_lon=d_pt.lon,
                r=r,
                baseline_full=None,
                weather_date="",
                test_weather_meta=test_meta,
            )
            raw_rows.append(row)
            n_ok += 1
            for vi, pt in enumerate(r.geometry or []):
                if len(pt) >= 2:
                    vertices.append(
                        {
                            "route_id": rid,
                            "vertex_index": vi,
                            "lat": float(pt[0]),
                            "lon": float(pt[1]),
                        }
                    )
            pbar.update(1)
            if log_every > 0 and pbar.n > 0 and pbar.n % log_every == 0:
                elapsed = time.perf_counter() - t_start
                _log.info(
                    "Промежуточно %d/%d: ok=%d fail=%d skip=%d elapsed=%.0f с",
                    pbar.n,
                    total_steps,
                    n_ok,
                    n_fail,
                    n_skip,
                    elapsed,
                )
    pbar.close()

    s_var, s_prof, s_dir = build_summaries(raw_rows)
    pair_cmp = build_pair_comparison(raw_rows)
    heat_kpi = build_heat_weather_kpi_rows(raw_rows)
    out = output_path or heat_weather_output_xlsx_path()
    wkt_fp = routing_engine_cache_fingerprint()
    meta_rows: List[Tuple[str, Any]] = [
        ("experiment_id", experiment_id),
        ("started_at_utc", datetime.now(timezone.utc).isoformat()),
        ("seed", seed),
        ("n_points", n_points),
        ("n_directed_pairs", n_pairs),
        ("batch_profiles", ",".join(profile_tuple)),
        ("n_profiles", len(profile_tuple)),
        ("synthetic_test_cases", 36),
        ("expected_route_cells", expected_routes),
        ("successful_route_rows", len(raw_rows)),
        ("skipped_od_no_path_buffer_100m", n_skip),
        ("failure_rows", len(failures)),
        ("precache_polygon_wkt_sha256", precache_wkt_sha256),
        (
            "precache_polygon_bounds_lonlat",
            f"{minx:.8f},{miny:.8f},{maxx:.8f},{maxy:.8f}",
        ),
        ("precache_polygon_wkt_prefix", settings.precache_area_polygon_wkt_stripped[:120]),
        ("routing_algo_version", ROUTING_ALGO_VERSION),
        ("routing_weights_fingerprint", wkt_fp),
        ("elapsed_seconds", round(time.perf_counter() - t_start, 2)),
        ("pair_limit", limit if limit is not None else ""),
        ("pair_offset", offset),
        ("experiment_kind", "heat_weather_synthetic"),
        ("synthetic_weather_time_iso", SYNTHETIC_TEST_WEATHER_ISO),
        ("project_weather_stress_global_blend", project_stress_blend),
        ("experiment_weather_stress_global_blend_effective", effective_stress_blend),
        ("experiment_stress_blend_mode", stress_blend_note),
    ]

    vgz_path: Optional[str] = None
    if vertices:
        stem = out[:-5] if out.lower().endswith(".xlsx") else out
        vgz_path = f"{stem}_route_vertices.csv.gz"
        write_route_vertices_csv_gzip(vgz_path, vertices)
        meta_rows = meta_rows + [("vertices_file", vgz_path)]

    write_xlsx(
        out,
        meta_rows=meta_rows,
        points=points,
        raw_rows=raw_rows,
        vertices=vertices,
        failures=failures,
        summary_variant=s_var,
        summary_profile=s_prof,
        summary_direction=s_dir,
        pair_cmp=pair_cmp,
        legacy_multisheet=False,
        summary_by_weather_date=[],
        summary_by_weather_date_variant=[],
        summary_heat_kpi=heat_kpi,
    )
    _log.info("Готово: %s", out)
    return out


def main() -> None:
    from bike_router.tools._experiment_common import (
        DEFAULT_BATCH_LOG_EVERY,
        DEFAULT_MAX_SAMPLE_ATTEMPTS,
        DEFAULT_MIN_SPACING_M,
        DEFAULT_ROUTE_BATCH_SEED,
    )

    parser = argparse.ArgumentParser(
        description="Только heat на 36 синтетических погодах (fixed-snapshot)."
    )
    parser.add_argument("--n-points", type=int, default=10)
    parser.add_argument("--seed", type=int, default=DEFAULT_ROUTE_BATCH_SEED)
    parser.add_argument("--min-spacing-m", type=float, default=DEFAULT_MIN_SPACING_M)
    parser.add_argument(
        "--max-sample-attempts",
        type=int,
        default=DEFAULT_MAX_SAMPLE_ATTEMPTS,
    )
    parser.add_argument(
        "--profiles",
        choices=("both", "cyclist", "pedestrian"),
        default="both",
    )
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log-every", type=int, default=DEFAULT_BATCH_LOG_EVERY)
    parser.add_argument(
        "--stress-blend-from-settings",
        action="store_true",
        help="Не менять WEATHER_STRESS_GLOBAL_BLEND (брать из .env / Settings).",
    )
    parser.add_argument(
        "--weather-stress-global-blend",
        type=float,
        default=None,
        metavar="X",
        help=(
            "Явное значение blend на время прогона. "
            "Если не задано и без --stress-blend-from-settings — используется 0.12."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    blend_cli: Optional[float] = args.weather_stress_global_blend
    analysis_default = False
    if args.stress_blend_from_settings:
        blend_cli = None
    elif blend_cli is None:
        blend_cli = 0.12
        analysis_default = True

    path = run_heat_weather_experiment(
        n_points=args.n_points,
        seed=args.seed,
        min_spacing_m=float(args.min_spacing_m),
        max_sample_attempts=int(args.max_sample_attempts),
        profiles_mode=args.profiles,
        verbose=bool(args.verbose),
        log_every=max(0, int(args.log_every)),
        output_path=args.output,
        limit=args.limit,
        offset=int(args.offset),
        stress_blend_from_settings=bool(args.stress_blend_from_settings),
        weather_stress_global_blend=blend_cli
        if not args.stress_blend_from_settings
        else None,
        stress_blend_analysis_default=analysis_default,
    )
    print(path)


if __name__ == "__main__":
    main()
