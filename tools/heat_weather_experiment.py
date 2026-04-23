"""Только маршрут ``heat`` на synthetic-сетке погоды (fixed-snapshot, без Open-Meteo).

Выход: ``bike_router/experiment_outputs/heat_weather_experiment_YYYYMMDD_HHMMSS.xlsx`` (UTC в имени).

Сетка ``--grid``:

**summer** (144 кейса) — единственная «летняя» сетка: базовые **36** комбинаций
(температура × дождь × сила ветра × облачность/КВ, см. ``weather_windows_test_grid``),
для **каждой** четыре направления ветра «откуда дует» **0°, 90°, 180°, 270°** (как Open-Meteo).
Итого 36×4 = **144**; direction-aware ветер **всегда** включён.

**winter** (216 кейсов) — зимняя сетка **54** комбинации
(температура × снегопад × глубина × пара скорость/порыв), см. ``weather_winter_synthetic_grid``,
каждая × **4 направления** → 54×4 = **216**.

**all** (360 кейсов) — подряд **summer** затем **winter** в одном прогоне и одном Excel
(метка сетки в meta: ``all``).

Устаревшее имя ``summer_wind`` воспринимается как ``summer`` (то же самое 144).

``WEATHER_STRESS_GLOBAL_BLEND`` не подменяется (Settings / .env).

Запуск::

    python -m bike_router.tools.heat_weather_experiment
    python -m bike_router.tools.heat_weather_experiment --grid winter --verbose
    python -m bike_router.tools.heat_weather_experiment --grid all
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


def run_heat_weather_experiment(
    *,
    n_points: int,
    profiles_mode: str,
    verbose: bool,
    log_every: int,
    weather_grid: str = "summer",
) -> str:
    _ensure_pkg_path()

    from bike_router.config import ROUTING_ALGO_VERSION, Settings, routing_engine_cache_fingerprint
    from bike_router.engine import RouteEngine
    from bike_router.exceptions import BikeRouterError, RouteNotFoundError
    from bike_router.services.area_graph_cache import parse_precache_polygon

    from bike_router.tools._experiment_common import (
        DEFAULT_MAX_SAMPLE_ATTEMPTS,
        DEFAULT_MIN_SPACING_M,
        DEFAULT_ROUTE_BATCH_SEED,
        EXPERIMENT_CORRIDOR_EXPAND_M,
        SYNTHETIC_TEST_WEATHER_ISO,
        weather_winter_synthetic_grid_with_wind_dirs,
        _batch_profiles_from_arg,
        _direction_key,
        _iter_directed_pairs,
        _point_id_fmt,
        _set_quiet_mode_for_batch,
        build_heat_weather_kpi_rows,
        build_winter_kpi_rows,
        build_pair_comparison,
        build_summaries,
        experiment_output_xlsx_path,
        kwargs_fixed_snapshot_from_case,
        route_to_raw_row,
        sample_points_in_polygon,
        weather_summer_test_grid_with_wind_dirs,
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

    seed = int(DEFAULT_ROUTE_BATCH_SEED)
    min_spacing_m = float(DEFAULT_MIN_SPACING_M)
    max_sample_attempts = int(DEFAULT_MAX_SAMPLE_ATTEMPTS)

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

    stress_blend_used = float(eng._app.settings.weather_stress_global_blend)

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
    wg = (weather_grid or "summer").strip().lower()
    if wg in ("summer_wind", "summer-wind"):
        wg = "summer"
        _log.info("grid summer_wind совпадает с summer (лето всегда с направлением ветра)")
    summer_g = weather_summer_test_grid_with_wind_dirs()
    winter_g = weather_winter_synthetic_grid_with_wind_dirs()
    if wg == "all":
        grid = summer_g + winter_g
        grid_name = "лето+зима (144+216=360)"
    elif wg == "winter":
        grid = winter_g
        grid_name = "216 зимних (54×4 направления)"
    else:
        grid = summer_g
        grid_name = "144 летних (36×4 направления)"
    assert len(grid) in (144, 216, 360)

    route_tasks: List[Tuple[int, int, str]] = [
        (i, j, prof)
        for i, j in _iter_directed_pairs(n_points)
        for prof in profile_tuple
    ]

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
        desc=f"Heat×{grid_name}",
        unit="запрос",
        leave=True,
        mininterval=2.0 if not verbose else 0.3,
    )

    for syn_case in grid:
        wkw = kwargs_fixed_snapshot_from_case(syn_case)
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
                alt = eng.compute_heat_alternative(
                    start=start,
                    end=end,
                    profile_key=prof,
                    green_enabled=True,
                    corridor_expand_schedule_meters=EXPERIMENT_CORRIDOR_EXPAND_M,
                    departure_time=str(wkw.get("weather_time") or ""),
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
    if wg == "winter":
        winter_kpi = build_winter_kpi_rows(raw_rows)
    elif wg == "all":
        wrows = [
            r
            for r in raw_rows
            if str(r.get("weather_test_case_id") or "").startswith("winter_")
        ]
        winter_kpi = build_winter_kpi_rows(wrows) if wrows else None
    else:
        winter_kpi = None
    out = experiment_output_xlsx_path(script_stem="heat_weather_experiment")
    wkt_fp = routing_engine_cache_fingerprint()
    meta_rows: List[Tuple[str, Any]] = [
        ("experiment_id", experiment_id),
        ("started_at_utc", datetime.now(timezone.utc).isoformat()),
        ("seed", seed),
        ("n_points", n_points),
        ("n_directed_pairs", n_pairs),
        ("batch_profiles", ",".join(profile_tuple)),
        ("n_profiles", len(profile_tuple)),
        ("experiment_weather_grid", wg),
        ("synthetic_test_cases", len(grid)),
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
        ("experiment_kind", "heat_weather_synthetic"),
        ("synthetic_weather_time_iso", SYNTHETIC_TEST_WEATHER_ISO),
        ("weather_stress_global_blend", stress_blend_used),
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
        summary_winter_kpi=winter_kpi,
    )
    _log.info("Готово: %s", out)
    return out


def main() -> None:
    from bike_router.tools._experiment_common import DEFAULT_BATCH_LOG_EVERY

    grid_help = (
        "summer: 36 базовых летних × 4 направления ветра = 144 (ветер всегда с направлением). "
        "winter: 54 зимних × 4 направления = 216. "
        "all: summer затем winter в одном файле = 360. "
        "summer_wind — то же, что summer (устаревший алиас)."
    )
    parser = argparse.ArgumentParser(
        description="Heat на synthetic-сетке; см. модульный docstring.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=grid_help,
    )
    parser.add_argument(
        "--grid",
        choices=("summer", "summer_wind", "winter", "all"),
        default="summer",
        metavar="NAME",
    )
    parser.add_argument("--n-points", type=int, default=10)
    parser.add_argument(
        "--profiles",
        choices=("both", "cyclist", "pedestrian"),
        default="both",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log-every", type=int, default=DEFAULT_BATCH_LOG_EVERY)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    path = run_heat_weather_experiment(
        n_points=args.n_points,
        profiles_mode=args.profiles,
        verbose=bool(args.verbose),
        log_every=max(0, int(args.log_every)),
        weather_grid=str(args.grid),
    )
    print(path)


if __name__ == "__main__":
    main()
