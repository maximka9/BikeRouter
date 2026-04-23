"""Сравнение шести вариантов маршрута при текущей погоде (один снимок Open-Meteo).

Один запрос погоды в центре полигона на момент запуска. Выход:
``bike_router/experiment_outputs/route_variants_experiment_YYYYMMDD_HHMMSS.xlsx`` (UTC в имени).

Запуск::

    python -m bike_router.tools.route_variants_experiment
    python -m bike_router.tools.route_variants_experiment --n-points 20 --profiles cyclist --verbose
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


def run_variants_experiment(
    *,
    n_points: int,
    profiles_mode: str,
    verbose: bool,
    log_every: int,
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
        experiment_output_xlsx_path,
        meta_append_batch_weather_snapshot,
        resolve_live_weather_once_for_polygon,
        route_to_raw_row,
        sample_points_in_polygon,
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

    route_tasks: List[Tuple[int, int, str]] = [
        (i, j, prof)
        for i, j in _iter_directed_pairs(n_points)
        for prof in profile_tuple
    ]

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
        desc="Варианты×пары",
        unit="пара",
        leave=True,
        mininterval=2.0 if not verbose else 0.3,
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
                **fixed_kw,
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
                        "variant": mode,
                        "error_code": str(code),
                        "error_message": str(e),
                    }
                )
            n_fail += 1
            pbar.update(1)
            continue

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
                test_weather_meta=None,
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
    pbar.close()

    s_var, s_prof, s_dir = build_summaries(raw_rows)
    pair_cmp = build_pair_comparison(raw_rows)
    out = experiment_output_xlsx_path(script_stem="route_variants_experiment")
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
    ]
    meta_rows = meta_append_batch_weather_snapshot(
        meta_rows,
        snap=w_snap,
        source=wsrc,
        departure_iso=dep_iso,
        center_lat=lat_c,
        center_lon=lon_c,
        experiment_kind="route_variants",
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
        summary_winter_kpi=None,
    )
    _log.info("Готово: %s", out)
    return out


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
