"""Общий прогон: 6 вариантов маршрута × synthetic-сетка или один fixed-snapshot.

Используется ``route_batch_experiment`` и ``route_variants_experiment``.
"""

from __future__ import annotations

import gc
import hashlib
import logging
import time
import uuid
from datetime import datetime, timezone
from multiprocessing import Pool
from typing import Any, Dict, List, Optional, Tuple

_log = logging.getLogger(__name__)


def run_variants_over_weather_cases(
    *,
    script_stem: str,
    n_points: int,
    profiles_mode: str,
    synthetic_weather_grid: Optional[List[Any]],
    fixed_weather_kw: Optional[Dict[str, Any]],
    live_weather_bundle: Optional[Tuple[Any, str, Any, str, float, float]],
    weather_grid_label: str = "all",
    verbose: bool,
    log_every: int,
    max_workers: int,
    chunk_size: int,
    mp_weather_chunk_size: int,
    directed_pairs: bool,
    write_vertices: bool,
    output_xlsx_suffix: Optional[str] = None,
) -> str:
    """Один Excel: либо непустой ``synthetic_weather_grid``, либо ``fixed_weather_kw`` (одна погода)."""
    from bike_router.config import ROUTING_ALGO_VERSION, Settings, routing_engine_cache_fingerprint
    from bike_router.engine import RouteEngine
    from bike_router.exceptions import BikeRouterError, RouteNotFoundError
    from bike_router.services.area_graph_cache import parse_precache_polygon

    from bike_router.tools._experiment_common import (
        DEFAULT_MAX_SAMPLE_ATTEMPTS,
        DEFAULT_MIN_SPACING_M,
        DEFAULT_ROUTE_BATCH_SEED,
        EXPECTED_VARIANTS,
        EXPERIMENT_CORRIDOR_EXPAND_M,
        SYNTHETIC_TEST_WEATHER_ISO,
        build_heat_experiment_extra_sheet_blocks,
        build_heat_vs_green_by_weather_rows,
        build_heat_weather_influence_rows,
        build_heat_weather_kpi_extras_dict,
        build_heat_weather_kpi_rows,
        build_pair_comparison,
        build_summaries,
        build_summaries_by_weather_case_and_variant,
        build_summaries_by_weather_case_id,
        build_winter_kpi_rows,
        experiment_output_xlsx_path,
        kwargs_fixed_snapshot_from_case,
        meta_append_batch_weather_snapshot,
        mp_resolve_pool_workers,
        mp_split_weather_grid_chunks,
        route_to_raw_row,
        sample_points_in_polygon,
        write_route_vertices_csv_gzip,
        write_xlsx,
        _batch_profiles_from_arg,
        _direction_key,
        _iter_directed_pairs,
        _iter_undirected_pairs,
        _point_id_fmt,
        _set_quiet_mode_for_batch,
    )

    import numpy as np
    from tqdm import tqdm

    if synthetic_weather_grid is not None and fixed_weather_kw is not None:
        raise ValueError("Укажите только synthetic_weather_grid или только fixed_weather_kw")
    if synthetic_weather_grid is None and fixed_weather_kw is None:
        raise ValueError("Нужен synthetic_weather_grid или fixed_weather_kw")
    use_synthetic = bool(synthetic_weather_grid and len(synthetic_weather_grid) > 0)
    if not use_synthetic and not fixed_weather_kw:
        raise ValueError("Пустой synthetic_weather_grid и нет fixed_weather_kw")

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
    pair_iter = (
        _iter_directed_pairs(n_points) if directed_pairs else _iter_undirected_pairs(n_points)
    )
    route_tasks: List[Tuple[int, int, str]] = [
        (i, j, prof) for i, j in pair_iter for prof in profile_tuple
    ]
    n_pairs = n_points * (n_points - 1) if directed_pairs else n_points * (n_points - 1) // 2

    raw_rows: List[Dict[str, Any]] = []
    vertices: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    n_ok = n_fail = n_skip = 0
    route_id_counter = 1

    ch = max(1, int(chunk_size))
    wchunk = max(1, int(mp_weather_chunk_size))

    _set_quiet_mode_for_batch(verbose)

    if use_synthetic:
        grid: List[Any] = list(synthetic_weather_grid or [])
        wg_label = "synthetic_grid"
        expected_routes = len(route_tasks) * len(grid) * len(EXPECTED_VARIANTS)
        total_steps = len(route_tasks) * len(grid)
        _gchunks_for_mw = mp_split_weather_grid_chunks(grid, wchunk)
        mw = mp_resolve_pool_workers(
            int(max_workers),
            task_count=max(1, len(route_tasks) * len(_gchunks_for_mw)),
        )
        _log.info(
            "Пакет маршрутов: главный warmup завершён; workers=%d (synthetic-задач≈%d)",
            mw,
            len(route_tasks) * len(_gchunks_for_mw),
        )

        if mw > 1:
            from bike_router.tools._batch_experiment_mp import init_worker as _mp_init
            from bike_router.tools._batch_experiment_mp import (
                run_variants_weather_chunk_task as _mp_task,
            )

            pls = [(float(p.lat), float(p.lon)) for p in points]
            gchunks = _gchunks_for_mw
            task_list = [
                (experiment_id, seed, i, j, prof, list(chunk))
                for i, j, prof in route_tasks
                for chunk in gchunks
            ]
            del eng
            gc.collect()
            _log.info(
                "variants×weather parallel: workers=%d imap_chunksize=%d weather_chunk=%d "
                "→ %d задач",
                mw,
                ch,
                wchunk,
                len(task_list),
            )
            with Pool(
                processes=mw,
                initializer=_mp_init,
                initargs=(pls, tuple(EXPERIMENT_CORRIDOR_EXPAND_M)),
            ) as pool:
                results = list(
                    tqdm(
                        pool.imap(_mp_task, task_list, chunksize=ch),
                        total=len(task_list),
                        desc="Варианты×погода (чанки)",
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
                    str(r.get("variant_key") or ""),
                )
            )
            for idx, row in enumerate(raw_rows, start=1):
                row["route_id"] = idx
                if write_vertices:
                    _append_vertices_for_row(row, idx, vertices)
        else:
            pbar = tqdm(
                total=total_steps,
                desc="Варианты×погода",
                unit="запрос",
                leave=True,
                mininterval=2.0 if not verbose else 0.3,
            )
            for i, j, prof in route_tasks:
                o_pt = points[i]
                d_pt = points[j]
                start = (o_pt.lat, o_pt.lon)
                end = (d_pt.lat, d_pt.lon)
                for syn_case in grid:
                    wkw = kwargs_fixed_snapshot_from_case(syn_case)
                    dep_iso = str(wkw.get("weather_time") or "")
                    test_meta = {
                        "weather_test_case_id": syn_case.case_id,
                        "weather_test_temperature_c": syn_case.temperature_c,
                        "weather_test_apparent_temperature_c": getattr(
                            syn_case, "apparent_temperature_c", None
                        ),
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
                    pbar.set_postfix(
                        case=syn_case.case_id,
                        od=_direction_key(i, j),
                        p=prof,
                        ok=n_ok,
                        refresh=False,
                    )
                    try:
                        alt = eng.compute_alternatives(
                            start=start,
                            end=end,
                            profile_key=prof,
                            green_enabled=True,
                            corridor_expand_schedule_meters=EXPERIMENT_CORRIDOR_EXPAND_M,
                            departure_time=dep_iso,
                            **wkw,
                        )
                        by_mode = {r.mode: r for r in alt.routes}
                    except RouteNotFoundError:
                        n_skip += 1
                        pbar.update(1)
                        continue
                    except (BikeRouterError, Exception) as e:
                        code = getattr(e, "code", type(e).__name__)
                        for mode in EXPECTED_VARIANTS:
                            failures.append(
                                {
                                    "origin_point_id": _point_id_fmt(i),
                                    "destination_point_id": _point_id_fmt(j),
                                    "profile": prof,
                                    "weather_test_case_id": syn_case.case_id,
                                    "variant": mode,
                                    "error_code": str(code),
                                    "error_message": str(e),
                                }
                            )
                        n_fail += 1
                        pbar.update(1)
                        continue

                    baseline = by_mode.get("full")
                    for mode in EXPECTED_VARIANTS:
                        if mode not in by_mode:
                            failures.append(
                                {
                                    "origin_point_id": _point_id_fmt(i),
                                    "destination_point_id": _point_id_fmt(j),
                                    "profile": prof,
                                    "weather_test_case_id": syn_case.case_id,
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
                            baseline_full=baseline,
                            weather_date="",
                            test_weather_meta=test_meta,
                        )
                        raw_rows.append(row)
                        n_ok += 1
                        if write_vertices:
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
    else:
        wg_label = "fixed_snapshot"
        fk = dict(fixed_weather_kw or {})
        dep_iso = str(fk.get("weather_time") or "")
        expected_routes = len(route_tasks) * len(EXPECTED_VARIANTS)
        pbar_total = len(route_tasks)
        mw = mp_resolve_pool_workers(
            int(max_workers),
            task_count=max(1, len(route_tasks)),
        )
        _log.info(
            "Пакет маршрутов: главный warmup завершён; workers=%d (fixed O-D задач=%d)",
            mw,
            len(route_tasks),
        )

        if mw > 1:
            from bike_router.tools._batch_experiment_mp import init_worker as _mp_init
            from bike_router.tools._batch_experiment_mp import (
                run_variants_fixed_weather_od_task as _mp_od,
            )

            pls = [(float(p.lat), float(p.lon)) for p in points]
            task_list = [
                (experiment_id, seed, i, j, prof, fk) for i, j, prof in route_tasks
            ]
            del eng
            gc.collect()
            _log.info(
                "variants fixed-weather parallel: workers=%d imap_chunksize=%d → %d задач",
                mw,
                ch,
                len(task_list),
            )
            with Pool(
                processes=mw,
                initializer=_mp_init,
                initargs=(pls, tuple(EXPERIMENT_CORRIDOR_EXPAND_M)),
            ) as pool:
                results = list(
                    tqdm(
                        pool.imap(_mp_od, task_list, chunksize=ch),
                        total=len(task_list),
                        desc="Варианты×пары",
                        unit="пара",
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
                    str(r.get("origin_point_id") or ""),
                    str(r.get("destination_point_id") or ""),
                    str(r.get("profile") or ""),
                    str(r.get("variant_key") or ""),
                )
            )
            for idx, row in enumerate(raw_rows, start=1):
                row["route_id"] = idx
                if write_vertices:
                    _append_vertices_from_geometry_row(row, vertices)
        else:
            pbar = tqdm(
                total=pbar_total,
                desc="Варианты×пары",
                unit="пара",
                leave=True,
                mininterval=2.0 if not verbose else 0.3,
            )
            for i, j, prof in route_tasks:
                pbar.set_postfix(od=_direction_key(i, j), p=prof, ok=n_ok, refresh=False)
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
                        departure_time=dep_iso,
                        **fk,
                    )
                    by_mode = {r.mode: r for r in alt.routes}
                except RouteNotFoundError:
                    n_skip += 1
                    pbar.update(1)
                    continue
                except (BikeRouterError, Exception) as e:
                    code = getattr(e, "code", type(e).__name__)
                    for mode in EXPECTED_VARIANTS:
                        failures.append(
                            {
                                "origin_point_id": _point_id_fmt(i),
                                "destination_point_id": _point_id_fmt(j),
                                "profile": prof,
                                "weather_test_case_id": None,
                                "variant": mode,
                                "error_code": str(code),
                                "error_message": str(e),
                            }
                        )
                    n_fail += 1
                    pbar.update(1)
                    continue

                baseline = by_mode.get("full")
                for mode in EXPECTED_VARIANTS:
                    if mode not in by_mode:
                        failures.append(
                            {
                                "origin_point_id": _point_id_fmt(i),
                                "destination_point_id": _point_id_fmt(j),
                                "profile": prof,
                                "weather_test_case_id": None,
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
                        baseline_full=baseline,
                        weather_date="",
                        test_weather_meta=None,
                    )
                    raw_rows.append(row)
                    n_ok += 1
                    if write_vertices:
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
            pbar.close()

    s_var, s_prof, s_dir = build_summaries(raw_rows)
    pair_cmp = build_pair_comparison(raw_rows)
    out = experiment_output_xlsx_path(
        script_stem=script_stem,
        filename_suffix=output_xlsx_suffix if output_xlsx_suffix else None,
    )
    wkt_fp = routing_engine_cache_fingerprint()

    meta_rows: List[Tuple[str, Any]] = [
        ("experiment_id", experiment_id),
        ("started_at_utc", datetime.now(timezone.utc).isoformat()),
        ("seed", seed),
        ("n_points", n_points),
        ("n_od_pairs", n_pairs),
        ("batch_directed_pairs", bool(directed_pairs)),
        ("batch_profiles", ",".join(profile_tuple)),
        ("n_profiles", len(profile_tuple)),
        ("n_variants", len(EXPECTED_VARIANTS)),
        ("expected_variant_rows", expected_routes),
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
        ("batch_mp_mode", "parallel" if mw > 1 else "sequential"),
        ("batch_max_workers", mw),
        ("batch_chunk_size", ch),
        ("batch_mp_weather_chunk", wchunk if (mw > 1 and use_synthetic) else None),
        ("weather_stress_global_blend", stress_blend_used),
        ("heat_continuous_enable", bool(settings.heat_continuous_enable)),
        ("variants_weather_mode", wg_label),
    ]

    if use_synthetic:
        meta_rows.append(("n_weather_cases", len(grid)))
        meta_rows.append(("experiment_weather_grid", str(weather_grid_label)))
        meta_rows.append(("synthetic_test_cases", len(grid)))  # дублирует n_weather_cases для совместимости с heat
        meta_rows.append(("synthetic_weather_time_iso", SYNTHETIC_TEST_WEATHER_ISO))
        meta_rows.append(("experiment_kind", "route_batch_combined"))
        heat_only = [r for r in raw_rows if str(r.get("variant_key")) == "heat"]
        heat_kpi = build_heat_weather_kpi_rows(heat_only)
        if heat_kpi:
            heat_kpi[0].update(build_heat_weather_kpi_extras_dict(heat_only))
        wrows = [
            r
            for r in raw_rows
            if str(r.get("weather_test_case_id") or "").startswith("winter_")
        ]
        winter_kpi = build_winter_kpi_rows(wrows) if wrows else None
        extra_blocks: List[Tuple[str, List[Dict[str, Any]]]] = [
            ("Влияние погоды на heat (по кейсу)", build_heat_weather_influence_rows(raw_rows)),
            ("Heat vs Green (по погодному кейсу)", build_heat_vs_green_by_weather_rows(raw_rows)),
            ("Средние по погодному кейсу", build_summaries_by_weather_case_id(raw_rows)),
            (
                "Средние по погодному кейсу и варианту",
                build_summaries_by_weather_case_and_variant(raw_rows),
            ),
        ]
        extra_blocks.extend(build_heat_experiment_extra_sheet_blocks(heat_only))
    else:
        if live_weather_bundle:
            w_snap, wsrc, wp_meta, dep_iso2, lat_c, lon_c = live_weather_bundle
            meta_rows = meta_append_batch_weather_snapshot(
                meta_rows,
                snap=w_snap,
                source=wsrc,
                departure_iso=dep_iso2,
                center_lat=lat_c,
                center_lon=lon_c,
                experiment_kind="route_variants",
                wp=wp_meta,
            )
        heat_kpi = None
        winter_kpi = None
        extra_blocks = None

    if write_vertices and vertices:
        stem = out[:-5] if out.lower().endswith(".xlsx") else out
        vgz_path = f"{stem}_route_vertices.csv.gz"
        write_route_vertices_csv_gzip(vgz_path, vertices)
        meta_rows = list(meta_rows) + [("vertices_file", vgz_path)]

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
        summary_extra_blocks=extra_blocks,
    )

    _log.info("Готово: %s", out)
    return out


def _append_vertices_for_row(
    row: Dict[str, Any], rid: int, vertices: List[Dict[str, Any]]
) -> None:
    """Из geometry_json в raw_row (после MP строки уже содержат geometry)."""
    import json

    gj = row.get("geometry_json")
    if not gj:
        return
    try:
        geom = json.loads(str(gj))
    except (TypeError, ValueError, json.JSONDecodeError):
        return
    for vi, pt in enumerate(geom or []):
        if len(pt) >= 2:
            vertices.append(
                {
                    "route_id": rid,
                    "vertex_index": vi,
                    "lat": float(pt[0]),
                    "lon": float(pt[1]),
                }
            )


def _append_vertices_from_geometry_row(
    row: Dict[str, Any], vertices: List[Dict[str, Any]]
) -> None:
    rid = int(row.get("route_id") or 0)
    _append_vertices_for_row(row, rid, vertices)
