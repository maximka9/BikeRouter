"""Только маршрут ``heat`` на synthetic-сетке погоды (fixed-snapshot, без Open-Meteo).

Выход: ``bike_router/experiment_outputs/heat_weather_experiment_YYYYMMDD_HHMMSS.xlsx`` (UTC в имени).

Сетка ``--grid`` (см. ``_experiment_common``):

**summer** (36 кейсов) — декартово произведение:

  - температура: 0, 12.5, 25 °C (3);
  - дождь: выкл / вкл с осадками 1.5 мм·ч⁻¹ (2);
  - ветер: слабый (2/3 м·с⁻¹) или сильный (9/14 м·с⁻¹) (2);
  - облачность + КВ: три уровня (0%+750, 50%+400, 100%+80 Вт·м⁻²) (3).

  Итого 3×2×2×3 = **36** сценариев. Направление ветра в кейсе не задано (direction-aware выключен).

**summer_wind** (144 кейса) — те же **36** базовых сценариев, для каждого четыре направления ветра
«откуда дует» в градусах: **0°, 90°, 180°, 270°** (метеоконвенция как у Open-Meteo). Итого 36×4 = **144**.

**winter** (96 кейсов) — зимняя сетка:

  - температура: −15, −5, 0, 2 °C (4);
  - свежий снег см·ч⁻¹: 0 / 0.45 / 3.2 с метками F0,F1,F2 (3);
  - глубина снега на земле: 0, 0.03, 0.10, 0.22 м (4);
  - ветер: слабый (2/3) или сильный (11/15) м·с⁻¹ (2).

  Итого 4×3×4×2 = **96**. Календарная дата снимка для сезона — зимняя (см. ``WINTER_SYNTH_WEATHER_ISO`` в общем модуле).

``WEATHER_STRESS_GLOBAL_BLEND`` не подменяется: используются значения из .env / Settings.

Запуск::

    python -m bike_router.tools.heat_weather_experiment
    python -m bike_router.tools.heat_weather_experiment --grid winter --verbose
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
        weather_winter_synthetic_grid,
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
        weather_windows_test_grid,
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
    if wg == "winter":
        grid = weather_winter_synthetic_grid()
        grid_name = "96 зимних погод"
    elif wg in ("summer_wind", "summer-wind"):
        grid = weather_summer_test_grid_with_wind_dirs()
        grid_name = "144 погод (36×4 направления)"
    else:
        grid = weather_windows_test_grid()
        grid_name = "36 погод"
    assert len(grid) in (36, 96, 144)

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
    winter_kpi = build_winter_kpi_rows(raw_rows) if wg == "winter" else None
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
        "summer: 3 темп × 2 дождя × 2 ветра × 3 облачности = 36 (без направления ветра). "
        "summer_wind: те же 36 × 4 направления (0/90/180/270°) = 144. "
        "winter: 4 темп × 3 снегопада × 4 глубины × 2 ветра = 96."
    )
    parser = argparse.ArgumentParser(
        description="Heat на synthetic-сетке; см. модульный docstring.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=grid_help,
    )
    parser.add_argument(
        "--grid",
        choices=("summer", "summer_wind", "winter"),
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
