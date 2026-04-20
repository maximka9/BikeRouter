"""Пакетный эксперимент по случайным O-D парам внутри PRECACHE_AREA_POLYGON_WKT -> Excel.

Запуск::

    python -m bike_router.tools.route_batch_experiment --seed 42

Требования: в .env задан ``PRECACHE_AREA_POLYGON_WKT``, собран precache
(``python -m bike_router.tools.precache_area``), ``GRAPH_CORRIDOR_MODE=true``
без fixed ``AREA_POLYGON_WKT``, чтобы движок использовал граф арены без
пересборки на каждую пару.

Погода отключена (``weather_mode=none``), чтобы не нагружать API и лог.

Точки O-D отбираются **только** внутри полигона из ``PRECACHE_AREA_POLYGON_WKT``
(арена precache), через ``parse_precache_polygon`` / uniform sampling в bounds
с проверкой ``polygon.contains(point)`` — не из ``AREA_POLYGON_WKT`` и не из
произвольного bbox без привязки к этому WKT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from math import asin, cos, radians, sin, sqrt
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

_log = logging.getLogger(__name__)

# Порядок как в engine._UNIFIED_ROUTE_ORDER
EXPECTED_VARIANTS: Tuple[str, ...] = (
    "shortest",
    "full",
    "green",
    "heat",
    "stress",
    "heat_stress",
)

PROFILES: Tuple[str, ...] = ("cyclist", "pedestrian")


def _ensure_pkg_path() -> None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if root not in sys.path:
        sys.path.insert(0, root)


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Расстояние по поверхности сферы (м)."""
    r = 6371000.0
    p1, p2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dl = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(p1) * cos(p2) * sin(dl / 2) ** 2
    return 2 * r * asin(min(1.0, sqrt(a)))


def _point_id_fmt(i: int) -> str:
    return f"P{i + 1:02d}"


@dataclass
class SampledPoint:
    idx: int
    lat: float
    lon: float
    nearest_node_id: int


def _iter_directed_pairs(n: int) -> Iterator[Tuple[int, int]]:
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            yield i, j


def _direction_key(a: int, b: int) -> str:
    return f"{_point_id_fmt(a)}->{_point_id_fmt(b)}"


def sample_points_in_polygon(
    *,
    poly: Any,
    n_target: int,
    rng: Any,
    engine: Any,
    min_spacing_m: float,
    max_attempts: int,
) -> List[SampledPoint]:
    """Случайные точки в полигоне: уникальность, min_spacing_m, валидация на графе."""
    from shapely.geometry import Point

    import osmnx as ox

    from bike_router.exceptions import PointOutsideZoneError

    minx, miny, maxx, maxy = poly.bounds
    out: List[SampledPoint] = []
    attempts = 0
    while len(out) < n_target and attempts < max_attempts:
        attempts += 1
        x = float(rng.uniform(minx, maxx))
        y = float(rng.uniform(miny, maxy))
        p = Point(x, y)
        if not poly.contains(p):
            continue
        lat, lon = y, x
        if any(
            haversine_m(lat, lon, sp.lat, sp.lon) < min_spacing_m for sp in out
        ):
            continue
        try:
            engine._validate_point((lat, lon), "sample")  # noqa: SLF001
        except PointOutsideZoneError:
            continue
        G = engine.graph
        if G is None:
            raise RuntimeError("Граф не загружен после warmup")
        nid = int(ox.distance.nearest_nodes(G, X=lon, Y=lat))
        out.append(SampledPoint(idx=len(out), lat=lat, lon=lon, nearest_node_id=nid))

    if len(out) < n_target:
        raise RuntimeError(
            f"Не удалось набрать {n_target} точек за {max_attempts} попыток "
            f"(получено {len(out)}). Увеличьте max_attempts или ослабьте min_spacing_m."
        )
    _log.info(
        "Точки: готово %d шт. (попыток %d, min %.0f м)",
        len(out),
        attempts,
        min_spacing_m,
    )
    return out


def _warnings_text(r: Any) -> Tuple[int, str]:
    qh = r.quality_hints
    if not qh or not qh.warnings:
        return 0, ""
    w = qh.warnings
    return len(w), " | ".join(str(x) for x in w)


def _stress_fields(r: Any) -> Dict[str, Any]:
    hs = r.heat_stress
    out: Dict[str, Any] = {
        "stress_cost_total": float(r.stress_cost_total or 0.0),
        "avg_stress_lts": None,
        "max_stress_lts": None,
        "high_stress_segments_count": int(r.high_stress_segments_count or 0),
        "stressful_intersections_count": int(r.stressful_intersections_count or 0),
    }
    if hs:
        out["avg_stress_lts"] = float(hs.avg_stress_lts)
        out["max_stress_lts"] = float(hs.max_stress_lts)
        if out["stress_cost_total"] == 0.0 and hs.stress_cost_total:
            out["stress_cost_total"] = float(hs.stress_cost_total)
        if not out["high_stress_segments_count"] and hs.high_stress_segments_count:
            out["high_stress_segments_count"] = int(hs.high_stress_segments_count)
        if (
            not out["stressful_intersections_count"]
            and hs.stressful_intersections_count
        ):
            out["stressful_intersections_count"] = int(
                hs.stressful_intersections_count
            )
    return out


def route_to_raw_row(
    *,
    experiment_id: str,
    seed: int,
    route_id: int,
    profile: str,
    origin_id: int,
    dest_id: int,
    o_lat: float,
    o_lon: float,
    d_lat: float,
    d_lon: float,
    r: Any,
) -> Dict[str, Any]:
    st = _stress_fields(r)
    wn, wt = _warnings_text(r)
    elev = r.elevation
    green = r.green
    wc = r.weather.weather_time if r.weather else None
    geom = r.geometry or []
    geom_json = json.dumps(geom, ensure_ascii=False, separators=(",", ":"))

    return {
        "route_id": route_id,
        "experiment_id": experiment_id,
        "seed": seed,
        "profile": profile,
        "variant_key": r.mode,
        "variant_label": r.variant_label or "",
        "origin_point_id": _point_id_fmt(origin_id),
        "destination_point_id": _point_id_fmt(dest_id),
        "direction_key": _direction_key(origin_id, dest_id),
        "direction_order": "forward"
        if origin_id < dest_id
        else "reverse",
        "origin_lat": o_lat,
        "origin_lon": o_lon,
        "destination_lat": d_lat,
        "destination_lon": d_lon,
        "route_built_at_utc": r.route_built_at_utc or "",
        "weather_time": wc or "",
        "length_m": float(r.length_m),
        "length_km": round(float(r.length_m) / 1000.0, 6),
        "time_s": float(r.time_s),
        "time_min": round(float(r.time_s) / 60.0, 4),
        "time_display": r.time_display or "",
        "climb_m": float(elev.climb_m),
        "descent_m": float(elev.descent_m),
        "max_gradient_pct": float(elev.max_gradient_pct),
        "avg_gradient_pct": float(elev.avg_gradient_pct),
        "max_above_start_m": float(elev.max_above_start_m),
        "max_below_start_m": float(elev.max_below_start_m),
        "end_diff_m": float(elev.end_diff_m),
        "green_percent": float(green.percent),
        "avg_trees_pct": float(green.avg_trees_pct),
        "avg_grass_pct": float(green.avg_grass_pct),
        "stress_cost_total": st["stress_cost_total"],
        "avg_stress_lts": st["avg_stress_lts"],
        "max_stress_lts": st["max_stress_lts"],
        "high_stress_segments_count": st["high_stress_segments_count"],
        "stressful_intersections_count": st["stressful_intersections_count"],
        "cost": float(r.cost),
        "mode": r.mode,
        "warnings_count": wn,
        "warnings_text": wt,
        "geometry_json": geom_json,
    }


def _mean(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    return sum(xs) / len(xs)


def _write_sheet_table(
    ws: Any, headers: Sequence[str], rows: List[Dict[str, Any]]
) -> None:
    for c, h in enumerate(headers, start=1):
        ws.cell(row=1, column=c, value=h)
    for ri, row in enumerate(rows, start=2):
        for c, h in enumerate(headers, start=1):
            v = row.get(h)
            ws.cell(row=ri, column=c, value=v)


def build_summaries(
    raw_rows: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """SummaryByVariant, SummaryByProfile, SummaryDirection."""
    keys_num = (
        "length_m",
        "time_s",
        "climb_m",
        "descent_m",
        "max_gradient_pct",
        "green_percent",
        "avg_trees_pct",
        "stress_cost_total",
    )

    def make_row(prefix: Dict[str, Any], rs: List[Dict[str, Any]]) -> Dict[str, Any]:
        row = dict(prefix)
        for k in keys_num:
            vals: List[float] = []
            for r in rs:
                v = r.get(k)
                if v is None or v == "":
                    continue
                try:
                    vals.append(float(v))
                except (TypeError, ValueError):
                    continue
            m = _mean(vals)
            row[f"mean_{k}"] = round(m, 4) if m is not None else None
        row["routes_count"] = len(rs)
        return row

    by_pv: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    by_prof: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_dir: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for r in raw_rows:
        by_pv[(str(r["profile"]), str(r["variant_key"]))].append(r)
        by_prof[str(r["profile"])].append(r)
        by_dir[str(r["direction_order"])].append(r)

    s_var = [
        make_row({"profile": pk, "variant_key": vk}, rs)
        for (pk, vk), rs in sorted(by_pv.items())
    ]
    s_prof = [make_row({"profile": pk}, rs) for pk, rs in sorted(by_prof.items())]
    s_dir = [
        make_row({"direction_order": dk}, rs) for dk, rs in sorted(by_dir.items())
    ]
    return s_var, s_prof, s_dir


def build_pair_comparison(
    raw_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Сводка по (origin, dest, profile): min/max/avg length и time."""
    groups: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in raw_rows:
        key = (
            r["origin_point_id"],
            r["destination_point_id"],
            str(r["profile"]),
            str(r["direction_key"]),
        )
        groups[key].append(r)

    out: List[Dict[str, Any]] = []
    for key in sorted(groups.keys()):
        o, d, prof, dk = key
        rs = groups[key]
        lengths = [float(x["length_m"]) for x in rs]
        times = [float(x["time_s"]) for x in rs]
        out.append(
            {
                "origin_point_id": o,
                "destination_point_id": d,
                "profile": prof,
                "direction_key": dk,
                "variants_count": len(rs),
                "length_min_m": min(lengths) if lengths else None,
                "length_max_m": max(lengths) if lengths else None,
                "length_mean_m": _mean(lengths),
                "time_min_s": min(times) if times else None,
                "time_max_s": max(times) if times else None,
                "time_mean_s": _mean(times),
            }
        )
    return out


def write_xlsx(
    path: str,
    *,
    meta_rows: List[Tuple[str, Any]],
    points: List[SampledPoint],
    raw_rows: List[Dict[str, Any]],
    vertices: List[Dict[str, Any]],
    failures: List[Dict[str, Any]],
    summary_variant: List[Dict[str, Any]],
    summary_profile: List[Dict[str, Any]],
    summary_direction: List[Dict[str, Any]],
    pair_cmp: List[Dict[str, Any]],
) -> None:
    from openpyxl import Workbook

    wb = Workbook()
    wm = wb.active
    wm.title = "Meta"
    wm.cell(row=1, column=1, value="key")
    wm.cell(row=1, column=2, value="value")
    for i, (k, v) in enumerate(meta_rows, start=2):
        wm.cell(row=i, column=1, value=k)
        wm.cell(row=i, column=2, value=v)

    def add_sheet(name: str, headers: List[str], rows: List[Dict[str, Any]]) -> None:
        ws = wb.create_sheet(title=name)
        _write_sheet_table(ws, headers, rows)

    add_sheet(
        "SampledPoints",
        ["point_id", "lat", "lon", "nearest_node_id"],
        [
            {
                "point_id": _point_id_fmt(sp.idx),
                "lat": sp.lat,
                "lon": sp.lon,
                "nearest_node_id": sp.nearest_node_id,
            }
            for sp in points
        ],
    )

    if raw_rows:
        headers = list(raw_rows[0].keys())
    else:
        headers = []
    add_sheet("RoutesRaw", headers, raw_rows)
    add_sheet(
        "RouteVertices",
        ["route_id", "vertex_index", "lat", "lon"],
        vertices,
    )
    if summary_variant:
        svh = list(summary_variant[0].keys())
    else:
        svh = []
    add_sheet("SummaryByVariant", svh, summary_variant)
    if summary_profile:
        sph = list(summary_profile[0].keys())
    else:
        sph = []
    add_sheet("SummaryByProfile", sph, summary_profile)
    if summary_direction:
        sdh = list(summary_direction[0].keys())
    else:
        sdh = []
    add_sheet("SummaryDirection", sdh, summary_direction)
    if pair_cmp:
        pch = list(pair_cmp[0].keys())
    else:
        pch = []
    add_sheet("PairComparison", pch, pair_cmp)

    fh = [
        "origin_point_id",
        "destination_point_id",
        "profile",
        "variant",
        "error_code",
        "error_message",
    ]
    add_sheet("Failures", fh, failures)

    wb.save(path)
    _log.info("Excel записан: %s", path)


def run_experiment(
    *,
    seed: int,
    n_points: int,
    min_spacing_m: float,
    output_path: Optional[str],
    max_sample_attempts: int,
    show_progress: bool = True,
) -> str:
    _ensure_pkg_path()

    from bike_router.config import ROUTING_ALGO_VERSION, Settings, routing_engine_cache_fingerprint
    from bike_router.engine import RouteEngine
    from bike_router.exceptions import BikeRouterError
    from bike_router.services.area_graph_cache import parse_precache_polygon

    import numpy as np

    settings = Settings()
    if not settings.has_precache_area_polygon:
        raise SystemExit(
            "В .env должен быть задан PRECACHE_AREA_POLYGON_WKT (полигон арены)."
        )

    experiment_id = str(uuid.uuid4())
    wkt_stripped = settings.precache_area_polygon_wkt_stripped
    precache_wkt_sha256 = hashlib.sha256(
        wkt_stripped.encode("utf-8")
    ).hexdigest()
    poly = parse_precache_polygon(settings)
    minx, miny, maxx, maxy = poly.bounds
    rng = np.random.default_rng(seed)

    t_start = time.perf_counter()
    _log.info(
        "Арена выборки точек: PRECACHE_AREA_POLYGON_WKT (sha256=%s…, bounds lon[%.5f..%.5f] lat[%.5f..%.5f])",
        precache_wkt_sha256[:16],
        minx,
        maxx,
        miny,
        maxy,
    )
    _log.info("Warmup движка (area_precache / граф)…")
    eng = RouteEngine()
    eng.warmup()

    if eng.graph is None or not eng.is_loaded:
        raise SystemExit("Warmup не загрузил граф. Проверьте .env и precache_area.")

    _log.info("Генерация %d точек (seed=%s)…", n_points, seed)
    points = sample_points_in_polygon(
        poly=poly,
        n_target=n_points,
        rng=rng,
        engine=eng,
        min_spacing_m=min_spacing_m,
        max_attempts=max_sample_attempts,
    )
    n_pairs = n_points * (n_points - 1)
    expected_routes = n_pairs * len(PROFILES) * len(EXPECTED_VARIANTS)

    raw_rows: List[Dict[str, Any]] = []
    vertices: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    route_id_counter = 1
    n_ok_route_rows = 0
    n_fail_rows = 0

    route_tasks: List[Tuple[int, int, str]] = [
        (i, j, prof)
        for i, j in _iter_directed_pairs(n_points)
        for prof in PROFILES
    ]

    if show_progress:
        from tqdm import tqdm

        route_iter: Any = tqdm(
            route_tasks,
            desc="Маршруты",
            unit="запрос",
            leave=True,
            mininterval=0.3,
        )
    else:
        route_iter = route_tasks

    for i, j, prof in route_iter:
        o_pt = points[i]
        d_pt = points[j]
        start = (o_pt.lat, o_pt.lon)
        end = (d_pt.lat, d_pt.lon)
        if show_progress and hasattr(route_iter, "set_postfix"):
            route_iter.set_postfix(
                od=_direction_key(i, j),
                p=prof,
                ok=n_ok_route_rows,
                fail=n_fail_rows,
                refresh=False,
            )
        by_mode: Dict[str, Any] = {}
        try:
            alt = eng.compute_alternatives(
                start,
                end,
                prof,
                green_enabled=True,
                weather_mode="none",
            )
            by_mode = {r.mode: r for r in alt.routes}
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
                n_fail_rows += 1
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
                n_fail_rows += 1
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
            )
            raw_rows.append(row)
            n_ok_route_rows += 1
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

    s_var, s_prof, s_dir = build_summaries(raw_rows)
    pair_cmp = build_pair_comparison(raw_rows)

    if output_path is None:
        out = f"route_experiment_{n_points}pts_seed{seed}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.xlsx"
    else:
        out = output_path

    wkt_fp = routing_engine_cache_fingerprint()
    meta_rows: List[Tuple[str, Any]] = [
        ("experiment_id", experiment_id),
        ("started_at_utc", datetime.now(timezone.utc).isoformat()),
        ("seed", seed),
        ("n_points", n_points),
        ("n_directed_pairs", n_pairs),
        ("n_profiles", len(PROFILES)),
        ("n_variants", len(EXPECTED_VARIANTS)),
        ("expected_route_cells", expected_routes),
        ("successful_route_rows", len(raw_rows)),
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

    _log.info("Запись Excel (%d строк маршрутов, вершин %d)…", len(raw_rows), len(vertices))
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
    )
    _log.info("Готово за %.1f с.", time.perf_counter() - t_start)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Пакетный эксперимент O-D в PRECACHE_AREA_POLYGON_WKT -> Excel."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-points", type=int, default=20, help="Число случайных точек")
    parser.add_argument(
        "--min-spacing-m",
        type=float,
        default=100.0,
        help="Мин. расстояние между точками (м)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Путь к .xlsx (по умолчанию route_experiment_<n>pts_seed<seed>_timestamp.xlsx)",
    )
    parser.add_argument(
        "--max-sample-attempts",
        type=int,
        default=50_000,
        help="Макс. попыток генерации каждой точки (суммарный цикл)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Отключить прогресс-бар tqdm (удобно для логов в файл)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    path = run_experiment(
        seed=args.seed,
        n_points=args.n_points,
        min_spacing_m=args.min_spacing_m,
        output_path=args.output,
        max_sample_attempts=args.max_sample_attempts,
        show_progress=not args.no_progress,
    )
    print(path)


if __name__ == "__main__":
    main()
