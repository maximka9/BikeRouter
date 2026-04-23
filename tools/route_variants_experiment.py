"""Сравнение шести вариантов маршрута при одном погодном снимке (Open-Meteo, один запрос).

Запуск (из корня репозитория NIR)::

    python -m bike_router.tools.route_variants_experiment --n-points 10
    python -m bike_router.tools.route_variants_experiment --weather winter --n-points 6

По умолчанию погода: один снимок Open-Meteo в центре полигона; режим ``--weather winter`` —
96 зимних synthetic-кейсов (``weather_winter_synthetic_grid``), все 6 вариантов маршрута.
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


def run_variants_experiment(
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
    weather_mode: str = "live",
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
        EXPECTED_VARIANTS,
        EXPERIMENT_CORRIDOR_EXPAND_M,
        _batch_profiles_from_arg,
        _direction_key,
        _iter_directed_pairs,
        _point_id_fmt,
        _set_quiet_mode_for_batch,
        build_pair_comparison,
        build_summaries,
        build_winter_kpi_rows,
        centroid_lat_lon_for_weather,
        kwargs_fixed_snapshot_from_case,
        meta_append_batch_weather_snapshot,
        resolve_live_weather_once_for_polygon,
        route_to_raw_row,
        sample_points_in_polygon,
        variants_output_xlsx_path,
        weather_winter_synthetic_grid,
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
    wm = (weather_mode or "live").strip().lower()
    winter_mode = wm in ("winter_synthetic", "winter-synthetic", "winter")

    from bike_router.services.weather import (
        build_weather_weight_params,
        snapshot_from_manual,
    )

    grid: Optional[List[Any]] = None
    wp_meta: Any = None
    if winter_mode:
        grid = weather_winter_synthetic_grid()
        lat_c, lon_c = centroid_lat_lon_for_weather(poly)
        c0 = grid[0]
        w_snap = snapshot_from_manual(
            temperature_c=c0.temperature_c,
            precipitation_mm=c0.precipitation_mm,
            wind_speed_ms=c0.wind_speed_ms,
            wind_gusts_ms=c0.wind_gusts_ms,
            cloud_cover_pct=c0.cloud_cover_pct,
            humidity_pct=c0.humidity_pct,
            shortwave_radiation_wm2=c0.shortwave_radiation_wm2,
            snowfall_cm_h=float(c0.snowfall_cm_h),
            snow_depth_m=float(c0.snow_depth_m),
            weather_code=c0.weather_code,
        )
        dep_iso = str(c0.weather_time_iso)
        wsrc = "synthetic_winter_grid"
        wp_meta = build_weather_weight_params(
            w_snap,
            enabled=True,
            settings=settings,
            reference_iso=dep_iso,
        )
        _log.info(
            "Зимняя synthetic-сетка: %d кейсов, meta-снимок по первому (%s)",
            len(grid),
            c0.case_id,
        )
    else:
        w_snap, wsrc, wp_meta, dep_iso, lat_c, lon_c = resolve_live_weather_once_for_polygon(
            poly, settings=settings
        )
        _log.info(
            "Погода (один снимок): источник=%s dep=%s center_lat=%.6f center_lon=%.6f",
            wsrc,
            dep_iso,
            lat_c,
            lon_c,
        )

    route_tasks: List[Tuple[int, int, str]] = [
        (i, j, prof)
        for i, j in _iter_directed_pairs(n_points)
        for prof in profile_tuple
    ]
    route_tasks = _apply_limit_offset(route_tasks, limit=limit, offset=offset)

    if winter_mode and grid is not None:
        expected_routes = len(route_tasks) * len(EXPECTED_VARIANTS) * len(grid)
        pbar_total = len(route_tasks) * len(grid)
    else:
        expected_routes = len(route_tasks) * len(EXPECTED_VARIANTS)
        pbar_total = len(route_tasks)

    raw_rows: List[Dict[str, Any]] = []
    vertices: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    route_id_counter = 1
    n_ok = 0
    n_fail = 0
    n_skip = 0

    _set_quiet_mode_for_batch(verbose)
    pbar = tqdm(
        total=pbar_total,
        desc="Варианты×пары×зима" if winter_mode else "Варианты×пары",
        unit="задача" if winter_mode else "пара",
        leave=True,
        mininterval=2.0 if not verbose else 0.3,
    )

    def _process_od_variants(
        *,
        i: int,
        j: int,
        prof: str,
        departure_time: str,
        weather_kwargs: Dict[str, Any],
        test_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        nonlocal n_ok, n_fail, n_skip, route_id_counter
        o_pt = points[i]
        d_pt = points[j]
        start = (o_pt.lat, o_pt.lon)
        end = (d_pt.lat, d_pt.lon)
        try:
            alt = eng.compute_alternatives(
                start=start,
                end=end,
                profile_key=prof,
                green_enabled=True,
                corridor_expand_schedule_meters=EXPERIMENT_CORRIDOR_EXPAND_M,
                departure_time=departure_time,
                **weather_kwargs,
            )
            by_mode = {r.mode: r for r in alt.routes}
        except RouteNotFoundError:
            n_skip += 1
            pbar.update(1)
            return
        except (BikeRouterError, Exception) as e:
            code = getattr(e, "code", type(e).__name__)
            for mode in EXPECTED_VARIANTS:
                failures.append(
                    {
                        "origin_point_id": _point_id_fmt(i),
                        "destination_point_id": _point_id_fmt(j),
                        "profile": prof,
                        "variant": mode,
                        "error_code": str(code),
                        "error_message": str(e),
                    }
                )
            n_fail += 1
            pbar.update(1)
            return

        for mode in EXPECTED_VARIANTS:
            if mode not in by_mode:
                failures.append(
                    {
                        "origin_point_id": _point_id_fmt(i),
                        "destination_point_id": _point_id_fmt(j),
                        "profile": prof,
                        "variant": mode,
                        "error_code": "MISSING_VARIANT",
                        "error_message": "Маршрут не включён в ответ движка",
                    }
                )
                n_fail += 1
                continue
            r = by_mode[mode]
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
                baseline_full=by_mode.get("full"),
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
                pbar_total,
                n_ok,
                n_fail,
                n_skip,
                elapsed,
            )

    if winter_mode and grid is not None:
        for syn_case in grid:
            wkw = kwargs_fixed_snapshot_from_case(syn_case)
            dep_case = str(wkw.get("weather_time") or "")
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
                    syn_case, "weather_time_iso", dep_case
                ),
            }
            for i, j, prof in route_tasks:
                pbar.set_postfix(
                    case=syn_case.case_id,
                    od=_direction_key(i, j),
                    p=prof,
                    ok=n_ok,
                    refresh=False,
                )
                _process_od_variants(
                    i=i,
                    j=j,
                    prof=prof,
                    departure_time=dep_case,
                    weather_kwargs=wkw,
                    test_meta=test_meta,
                )
    else:
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
        for i, j, prof in route_tasks:
            pbar.set_postfix(od=_direction_key(i, j), p=prof, ok=n_ok, refresh=False)
            _process_od_variants(
                i=i,
                j=j,
                prof=prof,
                departure_time=dep_iso,
                weather_kwargs=fixed_kw,
                test_meta=None,
            )
    pbar.close()

    s_var, s_prof, s_dir = build_summaries(raw_rows)
    pair_cmp = build_pair_comparison(raw_rows)
    out = output_path or variants_output_xlsx_path()
    wkt_fp = routing_engine_cache_fingerprint()
    meta_rows: List[Tuple[str, Any]] = [
        ("experiment_id", experiment_id),
        ("started_at_utc", datetime.now(timezone.utc).isoformat()),
        ("seed", seed),
        ("n_points", n_points),
        ("n_directed_pairs", n_pairs),
        ("batch_profiles", ",".join(profile_tuple)),
        ("n_profiles", len(profile_tuple)),
        ("n_variants", len(EXPECTED_VARIANTS)),
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
    ]
    if winter_mode:
        meta_rows = [
            ("experiment_weather_grid", "winter_synthetic_96"),
            ("synthetic_test_cases", len(grid or [])),
        ] + meta_rows
    meta_rows = meta_append_batch_weather_snapshot(
        meta_rows,
        snap=w_snap,
        source=wsrc,
        departure_iso=dep_iso,
        center_lat=lat_c,
        center_lon=lon_c,
        experiment_kind="route_variants_winter"
        if winter_mode
        else "route_variants",
        wp=wp_meta,
    )

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
        summary_winter_kpi=build_winter_kpi_rows(raw_rows) if winter_mode else None,
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
        description="Шесть вариантов маршрута при одном снимке погоды (центр арены)."
    )
    parser.add_argument("--n-points", type=int, default=10, help="Число случайных точек")
    parser.add_argument("--seed", type=int, default=DEFAULT_ROUTE_BATCH_SEED)
    parser.add_argument(
        "--min-spacing-m",
        type=float,
        default=DEFAULT_MIN_SPACING_M,
        help="Мин. расстояние между точками, м",
    )
    parser.add_argument(
        "--max-sample-attempts",
        type=int,
        default=DEFAULT_MAX_SAMPLE_ATTEMPTS,
        help="Лимит попыток выборки точек",
    )
    parser.add_argument(
        "--profiles",
        choices=("both", "cyclist", "pedestrian"),
        default="both",
    )
    parser.add_argument(
        "--weather",
        choices=("live", "winter"),
        default="live",
        help="live — один снимок Open-Meteo; winter — synthetic 96×6 вариантов.",
    )
    parser.add_argument("--output", type=str, default=None, help="Путь к .xlsx")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Максимум пар O–D×профиль после offset",
    )
    parser.add_argument("--offset", type=int, default=0, help="Пропустить первые N задач")
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
        seed=args.seed,
        min_spacing_m=float(args.min_spacing_m),
        max_sample_attempts=int(args.max_sample_attempts),
        profiles_mode=args.profiles,
        verbose=bool(args.verbose),
        log_every=max(0, int(args.log_every)),
        output_path=args.output,
        limit=args.limit,
        offset=int(args.offset),
        weather_mode="winter" if args.weather == "winter" else "live",
    )
    print(path)


if __name__ == "__main__":
    main()
