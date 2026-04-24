"""Только маршрут ``heat`` на synthetic-сетке погоды (fixed-snapshot, без Open-Meteo).

Выход: ``bike_router/experiment_outputs/heat_weather_experiment_YYYYMMDD_HHMMSS.xlsx`` (UTC в имени).

Сетка ``--grid``:

**summer** (**90** кейсов): те же оси 3×2×2×3, но направления ветра только при **сильном** ветре
(4 угла); при слабом — одно направление **0°** → 18+72 = 90.

**winter** (**135** кейсов): 54 базовых по снегу/температуре; слабый ветер (W0) — одно направление,
сильный (W1) — четыре → 27+108 = 135.

**all** — **225** = 90 + 135 в одном Excel.

Пары O–D по умолчанию **неориентированные** (только ``i<j``) — вдвое меньше маршрутов, чем A→B и B→A;
флаг ``--directed-pairs`` включает полный directed-режим.

Вершины маршрутов в ``.csv.gz`` **не** пишутся (ускорение и объём).

При ``--max-workers`` > 1 прогресс идёт по **чанкам погоды** (см. ``--mp-weather-chunk``), иначе
долгое «молчание» на первой паре (сотни маршрутов подряд в одном воркере).

Устаревшее ``summer_wind`` = ``summer``.

``WEATHER_STRESS_GLOBAL_BLEND`` не подменяется (Settings / .env).

В листе «Сводка» дополнительно: расширенный блок KPI (зелёный критерий, ветер, surface,
валидационное покрытие), «QA — предупреждения», «Зима: объяснение reroute».

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
    directed_pairs: bool = False,
    max_workers: int = 1,
    chunk_size: int = 1,
    mp_weather_chunk_size: int = 25,
) -> str:
    _ensure_pkg_path()

    import gc
    from multiprocessing import Pool

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
        mp_resolve_pool_workers,
        mp_split_weather_grid_chunks,
        weather_summer_heat_grid,
        weather_winter_heat_grid,
        _batch_profiles_from_arg,
        _direction_key,
        _iter_directed_pairs,
        _iter_undirected_pairs,
        _point_id_fmt,
        _set_quiet_mode_for_batch,
        build_heat_experiment_extra_sheet_blocks,
        build_heat_weather_kpi_extras_dict,
        build_heat_weather_kpi_rows,
        build_winter_kpi_rows,
        build_pair_comparison,
        build_summaries,
        experiment_output_xlsx_path,
        kwargs_fixed_snapshot_from_case,
        route_to_raw_row,
        sample_points_in_polygon,
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
    profile_tuple = _batch_profiles_from_arg(profiles_mode)
    wg = (weather_grid or "summer").strip().lower()
    if wg in ("summer_wind", "summer-wind"):
        wg = "summer"
        _log.info("summer_wind → summer (сетка heat 90)")
    summer_g = weather_summer_heat_grid()
    winter_g = weather_winter_heat_grid()
    if wg == "all":
        grid = summer_g + winter_g
        grid_name = "лето+зима (90+135=225)"
    elif wg == "winter":
        grid = winter_g
        grid_name = "135 зимних (calm 1×WD + strong 4×WD)"
    else:
        grid = summer_g
        grid_name = "90 летних (calm 1×WD + strong 4×WD)"
    assert len(grid) in (90, 135, 225)

    pair_iter = _iter_directed_pairs(n_points) if directed_pairs else _iter_undirected_pairs(n_points)
    route_tasks: List[Tuple[int, int, str]] = [
        (i, j, prof) for i, j in pair_iter for prof in profile_tuple
    ]
    n_od_tracks = len(route_tasks)

    expected_routes = len(route_tasks) * len(grid)
    raw_rows: List[Dict[str, Any]] = []
    vertices: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    n_ok = 0
    n_fail = 0
    n_skip = 0

    _set_quiet_mode_for_batch(verbose)
    total_steps = len(route_tasks) * len(grid)

    mw = mp_resolve_pool_workers(int(max_workers))
    ch = max(1, int(chunk_size))
    wchunk = max(1, int(mp_weather_chunk_size))

    if mw > 1:
        from bike_router.tools._batch_experiment_mp import init_worker as _heat_mp_init
        from bike_router.tools._batch_experiment_mp import (
            run_heat_weather_chunk_task as _heat_mp_task,
        )

        pls = [(float(p.lat), float(p.lon)) for p in points]
        gchunks = mp_split_weather_grid_chunks(grid, wchunk)
        task_list = [
            (experiment_id, seed, i, j, prof, list(chunk))
            for i, j, prof in route_tasks
            for chunk in gchunks
        ]
        del eng
        gc.collect()
        _log.info(
            "heat parallel: workers=%d pool_chunksize=%d weather_chunk=%d "
            "→ %d задач (~%d маршрутов/задачу, прогресс по задачам)",
            mw,
            ch,
            wchunk,
            len(task_list),
            wchunk,
        )
        with Pool(
            processes=mw,
            initializer=_heat_mp_init,
            initargs=(pls, tuple(EXPERIMENT_CORRIDOR_EXPAND_M)),
        ) as pool:
            results = list(
                tqdm(
                    pool.imap(_heat_mp_task, task_list, chunksize=ch),
                    total=len(task_list),
                    desc=f"Heat×{grid_name} (чанки по погоде)",
                    unit="чанк",
                    leave=True,
                    mininterval=0.5 if not verbose else 0.15,
                )
            )
        for part in results:
            rs, fl, ok, fi, sk = part
            raw_rows.extend(rs)
            failures.extend(fl)
            n_ok += int(ok)
            n_fail += int(fi)
            n_skip += int(sk)
        raw_rows.sort(
            key=lambda r: (
                str(r.get("weather_test_case_id") or ""),
                str(r.get("origin_point_id") or ""),
                str(r.get("destination_point_id") or ""),
                str(r.get("profile") or ""),
            )
        )
        for idx, row in enumerate(raw_rows, start=1):
            row["route_id"] = idx
    else:
        pbar = tqdm(
            total=total_steps,
            desc=f"Heat×{grid_name}",
            unit="запрос",
            leave=True,
            mininterval=2.0 if not verbose else 0.3,
        )
        route_id_counter = 1
        for i, j, prof in route_tasks:
            o_pt = points[i]
            d_pt = points[j]
            start = (o_pt.lat, o_pt.lon)
            end = (d_pt.lat, d_pt.lon)
            for syn_case in grid:
                wkw = kwargs_fixed_snapshot_from_case(syn_case)
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
    if heat_kpi:
        heat_kpi[0].update(build_heat_weather_kpi_extras_dict(raw_rows))
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
        ("heat_route_tasks", n_od_tracks),
        ("heat_directed_pairs", bool(directed_pairs)),
        ("heat_max_workers", mw),
        ("heat_mp_weather_chunk", wchunk if mw > 1 else None),
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
        ("heat_continuous_enable", bool(settings.heat_continuous_enable)),
    ]

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
        summary_extra_blocks=build_heat_experiment_extra_sheet_blocks(raw_rows),
    )
    _log.info("Готово: %s", out)
    return out


def main() -> None:
    from bike_router.tools._experiment_common import DEFAULT_BATCH_LOG_EVERY

    grid_help = (
        "summer: 90 synthetic (слабый ветер — 1 направление, сильный — 4). "
        "winter: 135. all: 225. "
        "По умолчанию пары O–D неориентированные (i<j); --directed-pairs — A→B и B→A. "
        "--max-workers 0=auto; --mp-weather-chunk — размер чанка погоды на задачу пула. "
        "Вершины в csv.gz не пишутся."
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
    parser.add_argument(
        "--directed-pairs",
        action="store_true",
        help="Считать и A→B, и B→A (вдвое больше маршрутов).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        metavar="N",
        help="Параллель: 1=выкл, 0=авто (min(6, CPU−1)).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1,
        metavar="N",
        help="chunksize для pool.imap между процессами.",
    )
    parser.add_argument(
        "--mp-weather-chunk",
        type=int,
        default=25,
        metavar="N",
        help="Сколько synthetic-кейсов в одной задаче пула (меньше — чаще обновляется tqdm).",
    )
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
        directed_pairs=bool(args.directed_pairs),
        max_workers=int(args.max_workers),
        chunk_size=max(1, int(args.chunk_size)),
        mp_weather_chunk_size=max(1, int(args.mp_weather_chunk)),
    )
    print(path)


if __name__ == "__main__":
    main()
