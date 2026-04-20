"""Пакетный эксперимент по случайным O-D парам внутри PRECACHE_AREA_POLYGON_WKT -> Excel.

Запуск::

    python -m bike_router.tools.route_batch_experiment --seed 42

Требования: в .env задан ``PRECACHE_AREA_POLYGON_WKT``, собран precache
(``python -m bike_router.tools.precache_area``), ``GRAPH_CORRIDOR_MODE=true``
без fixed ``AREA_POLYGON_WKT``, чтобы движок использовал граф арены без
пересборки на каждую пару.

По умолчанию погода отключена (``weather_mode=none``), чтобы не бомбить Open-Meteo
и не раздувать лог. Включить текущий прогноз: ``--weather-mode auto``.

Расширение коридора в батче только **10 м → 100 м**. Если маршрута нет и на 100 м,
пара O-D **пропускается** (без попыток 1000/5000 м); в лог пишутся координаты
начала/конца и идентификаторы точек.

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

# Перебор буфера коридора в этом эксперименте; дальше — пропуск пары (см. RouteNotFoundError).
EXPERIMENT_CORRIDOR_EXPAND_M: Tuple[float, ...] = (10.0, 100.0)


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
    baseline_full: Any = None,
) -> Dict[str, Any]:
    st = _stress_fields(r)
    wn, wt = _warnings_text(r)
    elev = r.elevation
    green = r.green
    wc = r.weather.weather_time if r.weather else None
    geom = r.geometry or []
    geom_json = json.dumps(geom, ensure_ascii=False, separators=(",", ":"))

    out: Dict[str, Any] = {
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
    if baseline_full is not None:
        out["full_baseline_length_m"] = round(float(baseline_full.length_m), 2)
        be = baseline_full.elevation
        out["delta_length_vs_full_m"] = round(
            float(r.length_m) - float(baseline_full.length_m), 2
        )
        out["delta_climb_vs_full_m"] = round(
            float(elev.climb_m) - float(be.climb_m), 2
        )
    else:
        out["full_baseline_length_m"] = None
        out["delta_length_vs_full_m"] = None
        out["delta_climb_vs_full_m"] = None
    return out


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


def _meta_rows_ru(rows: List[Tuple[str, Any]]) -> List[Tuple[str, Any]]:
    ru = {
        "experiment_id": "ID эксперимента",
        "started_at_utc": "Время старта (UTC)",
        "seed": "Seed",
        "n_points": "Число точек",
        "n_directed_pairs": "Направленных пар O–D",
        "n_profiles": "Профилей",
        "n_variants": "Вариантов маршрута",
        "expected_route_cells": "Ожидалось строк маршрутов",
        "successful_route_rows": "Успешных строк",
        "skipped_od_no_path_buffer_100m": "Пропущено (нет пути до 100 м)",
        "failure_rows": "Строк ошибок",
        "precache_polygon_wkt_sha256": "SHA256 полигона precache",
        "precache_polygon_bounds_lonlat": "Границы полигона (lon/lat)",
        "precache_polygon_wkt_prefix": "Префикс WKT полигона",
        "routing_algo_version": "Версия алгоритма",
        "routing_weights_fingerprint": "Отпечаток весов",
        "elapsed_seconds": "Время работы, с",
        "weather_mode": "Режим погоды",
        "weather_time": "Время погоды",
        "vertices_file": "Файл вершин (отдельно)",
    }
    return [(ru.get(str(k), str(k)), v) for k, v in rows]


_ROUTE_COL_RU: Dict[str, str] = {
    "route_id": "ID маршрута",
    "experiment_id": "ID эксперимента",
    "seed": "Seed",
    "profile": "Профиль",
    "variant_key": "Вариант (ключ)",
    "variant_label": "Вариант",
    "origin_point_id": "Точка отправления",
    "destination_point_id": "Точка назначения",
    "direction_key": "Направление",
    "direction_order": "Порядок",
    "origin_lat": "Широта начала",
    "origin_lon": "Долгота начала",
    "destination_lat": "Широта конца",
    "destination_lon": "Долгота конца",
    "route_built_at_utc": "Построен в (UTC)",
    "weather_time": "Время погоды",
    "length_m": "Длина, м",
    "length_km": "Длина, км",
    "time_s": "Время, с",
    "time_min": "Время, мин",
    "time_display": "Время (текст)",
    "climb_m": "Набор, м",
    "descent_m": "Спуск, м",
    "max_gradient_pct": "Макс. уклон, %",
    "avg_gradient_pct": "Средн. уклон, %",
    "max_above_start_m": "Макс. выше старта, м",
    "max_below_start_m": "Макс. ниже старта, м",
    "end_diff_m": "Перепад финиша, м",
    "green_percent": "Озеленение, %",
    "avg_trees_pct": "Деревья, %",
    "avg_grass_pct": "Трава, %",
    "stress_cost_total": "Стресс, суммарно",
    "avg_stress_lts": "Средний LTS",
    "max_stress_lts": "Макс. LTS",
    "high_stress_segments_count": "Высокий стресс, сегм.",
    "stressful_intersections_count": "Стресс. пересечения",
    "cost": "Стоимость модели",
    "mode": "Режим (техн.)",
    "warnings_count": "Число предупреждений",
    "warnings_text": "Предупреждения",
    "geometry_json": "Геометрия (JSON)",
    "full_baseline_length_m": "Длина эталона full, м",
    "delta_length_vs_full_m": "Δ длины к full, м",
    "delta_climb_vs_full_m": "Δ набора к full, м",
}


def _rename_row_keys_ru(row: Dict[str, Any]) -> Dict[str, Any]:
    return {_ROUTE_COL_RU.get(k, k): v for k, v in row.items()}


def write_route_vertices_csv_gzip(path: str, vertices: List[Dict[str, Any]]) -> None:
    import csv
    import gzip

    with gzip.open(path, "wt", encoding="utf-8", newline="") as gz:
        w = csv.DictWriter(
            gz,
            fieldnames=["route_id", "vertex_index", "lat", "lon"],
            extrasaction="ignore",
        )
        w.writeheader()
        for row in vertices:
            w.writerow(row)
    _log.info("Вершины маршрутов: %s (%d строк)", path, len(vertices))


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
    legacy_multisheet: bool = False,
) -> None:
    from openpyxl import Workbook

    if legacy_multisheet:
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
        _log.info("Excel записан (англ. листы): %s", path)
        return

    wb = Workbook()
    ws0 = wb.active
    ws0.title = "Метаданные"
    ws0.cell(row=1, column=1, value="Параметр")
    ws0.cell(row=1, column=2, value="Значение")
    for i, (k, v) in enumerate(_meta_rows_ru(meta_rows), start=2):
        ws0.cell(row=i, column=1, value=k)
        ws0.cell(row=i, column=2, value=v)

    ws1 = wb.create_sheet(title="Точки")
    _write_sheet_table(
        ws1,
        ["ID точки", "Широта", "Долгота", "Узел графа"],
        [
            {
                "ID точки": _point_id_fmt(sp.idx),
                "Широта": sp.lat,
                "Долгота": sp.lon,
                "Узел графа": sp.nearest_node_id,
            }
            for sp in points
        ],
    )

    if raw_rows:
        rh = [_ROUTE_COL_RU.get(k, k) for k in raw_rows[0].keys()]
        ru_rows = [_rename_row_keys_ru(dict(rw)) for rw in raw_rows]
    else:
        rh, ru_rows = [], []
    ws2 = wb.create_sheet(title="Маршруты")
    _write_sheet_table(ws2, rh, ru_rows)

    ws3 = wb.create_sheet(title="Сводка")
    rr = 1
    for title, block in (
        ("Средние по варианту", summary_variant),
        ("Средние по профилю", summary_profile),
        ("Средние по направлению", summary_direction),
        ("Сводка по парам O–D", pair_cmp),
    ):
        ws3.cell(row=rr, column=1, value=title)
        rr += 1
        if block and block[0]:
            hdr = list(block[0].keys())
            for c, h in enumerate(hdr, start=1):
                ws3.cell(row=rr, column=c, value=h)
            rr += 1
            for i, row in enumerate(block):
                for c, h in enumerate(hdr, start=1):
                    ws3.cell(row=rr + i, column=c, value=row.get(h))
            rr += len(block) + 1
        else:
            rr += 1

    ws4 = wb.create_sheet(title="Ошибки")
    _write_sheet_table(
        ws4,
        ["Отправление", "Назначение", "Профиль", "Вариант", "Код", "Сообщение"],
        [
            {
                "Отправление": f.get("origin_point_id"),
                "Назначение": f.get("destination_point_id"),
                "Профиль": f.get("profile"),
                "Вариант": f.get("variant"),
                "Код": f.get("error_code"),
                "Сообщение": f.get("error_message"),
            }
            for f in failures
        ],
    )

    wb.save(path)
    _log.info("Excel записан (компактный RU): %s", path)


def run_experiment(
    *,
    seed: int,
    n_points: int,
    min_spacing_m: float,
    output_path: Optional[str],
    max_sample_attempts: int,
    show_progress: bool = True,
    weather_mode: str = "none",
    use_live_weather: bool = False,
    weather_time: Optional[str] = None,
    temperature_c: Optional[float] = None,
    precipitation_mm: Optional[float] = None,
    wind_speed_ms: Optional[float] = None,
    cloud_cover_pct: Optional[float] = None,
    humidity_pct: Optional[float] = None,
    export_legacy_xlsx: bool = False,
    vertices_gzip_path: Optional[str] = None,
) -> str:
    _ensure_pkg_path()

    from bike_router.config import ROUTING_ALGO_VERSION, Settings, routing_engine_cache_fingerprint
    from bike_router.engine import RouteEngine
    from bike_router.exceptions import BikeRouterError, RouteNotFoundError
    from bike_router.services.area_graph_cache import parse_precache_polygon

    import numpy as np

    settings = Settings()
    _log.info("Каталог данных: base_dir=%s (cache=%s)", settings.base_dir, settings.cache_dir)
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
    n_skipped_no_path_100m = 0

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
                skip=n_skipped_no_path_100m,
                refresh=False,
            )
        by_mode: Dict[str, Any] = {}
        try:
            alt = eng.compute_alternatives(
                start,
                end,
                prof,
                green_enabled=True,
                weather_mode=weather_mode,
                use_live_weather=use_live_weather,
                weather_time=weather_time,
                temperature_c=temperature_c,
                precipitation_mm=precipitation_mm,
                wind_speed_ms=wind_speed_ms,
                cloud_cover_pct=cloud_cover_pct,
                humidity_pct=humidity_pct,
                corridor_expand_schedule_meters=EXPERIMENT_CORRIDOR_EXPAND_M,
            )
            by_mode = {r.mode: r for r in alt.routes}
        except RouteNotFoundError as e:
            n_skipped_no_path_100m += 1
            _log.warning(
                "Пропуск O-D: нет маршрута при буфере коридора до 100 м. "
                "origin lat=%.6f lon=%.6f | dest lat=%.6f lon=%.6f | profile=%s | "
                "points %s→%s (%s). Детали: %s",
                start[0],
                start[1],
                end[0],
                end[1],
                prof,
                _point_id_fmt(i),
                _point_id_fmt(j),
                _direction_key(i, j),
                e,
            )
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
                baseline_full=by_mode.get("full"),
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
        ("skipped_od_no_path_buffer_100m", n_skipped_no_path_100m),
        ("failure_rows", len(failures)),
        ("precache_polygon_wkt_sha256", precache_wkt_sha256),
        (
            "precache_polygon_bounds_lonlat",
            f"{minx:.8f},{miny:.8f},{maxx:.8f},{maxy:.8f}",
        ),
        ("precache_polygon_wkt_prefix", settings.precache_area_polygon_wkt_stripped[:120]),
        ("routing_algo_version", ROUTING_ALGO_VERSION),
        ("routing_weights_fingerprint", wkt_fp),
        ("weather_mode", weather_mode),
        ("weather_time", weather_time or ""),
        ("elapsed_seconds", round(time.perf_counter() - t_start, 2)),
    ]

    vgz_path: Optional[str] = None
    if not export_legacy_xlsx and vertices:
        if vertices_gzip_path:
            vgz_path = vertices_gzip_path
        else:
            stem = out[:-5] if out.lower().endswith(".xlsx") else out
            vgz_path = f"{stem}_route_vertices.csv.gz"
        write_route_vertices_csv_gzip(vgz_path, vertices)
        meta_rows = meta_rows + [("vertices_file", vgz_path)]

    if n_skipped_no_path_100m:
        _log.info(
            "Пропущено направлений O-D (нет пути после буфера 100 м): %d",
            n_skipped_no_path_100m,
        )
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
        legacy_multisheet=export_legacy_xlsx,
    )
    _log.info("Готово за %.1f с.", time.perf_counter() - t_start)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Пакетный эксперимент O-D в PRECACHE_AREA_POLYGON_WKT -> Excel."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-points", type=int, default=10, help="Число случайных точек")
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
    parser.add_argument(
        "--weather-mode",
        choices=("none", "auto", "manual", "fixed-snapshot"),
        default="none",
        help=(
            "none — без погодных множителей; auto — Open-Meteo; "
            "manual / fixed-snapshot — поля temperature/осадки ниже (fixed = воспроизводимый снимок)."
        ),
    )
    parser.add_argument(
        "--weather-time",
        type=str,
        default=None,
        help="ISO 8601 — момент для почасовой погоды (auto) или подписи (manual)",
    )
    parser.add_argument("--temperature-c", type=float, default=None, help="Ручная t °C")
    parser.add_argument("--precipitation-mm", type=float, default=None, help="Осадки мм/ч")
    parser.add_argument("--wind-speed-ms", type=float, default=None, help="Ветер м/с")
    parser.add_argument("--cloud-cover-pct", type=float, default=None, help="Облачность %")
    parser.add_argument("--humidity-pct", type=float, default=None, help="Влажность %")
    parser.add_argument(
        "--use-live-weather",
        action="store_true",
        help="С API: как принудительный auto (прогноз по координатам и времени).",
    )
    parser.add_argument(
        "--export-legacy-xlsx",
        action="store_true",
        help="Полная англ. выгрузка со всеми листами и вершинами внутри .xlsx (как раньше).",
    )
    parser.add_argument(
        "--vertices-gzip",
        type=str,
        default=None,
        help="Путь к route_vertices.csv.gz (по умолчанию рядом с .xlsx, если не --export-legacy-xlsx).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    use_live = bool(args.use_live_weather) or (args.weather_mode == "auto")
    path = run_experiment(
        seed=args.seed,
        n_points=args.n_points,
        min_spacing_m=args.min_spacing_m,
        output_path=args.output,
        max_sample_attempts=args.max_sample_attempts,
        show_progress=not args.no_progress,
        weather_mode=args.weather_mode,
        use_live_weather=use_live,
        weather_time=args.weather_time,
        temperature_c=args.temperature_c,
        precipitation_mm=args.precipitation_mm,
        wind_speed_ms=args.wind_speed_ms,
        cloud_cover_pct=args.cloud_cover_pct,
        humidity_pct=args.humidity_pct,
        export_legacy_xlsx=args.export_legacy_xlsx,
        vertices_gzip_path=args.vertices_gzip,
    )
    print(path)


if __name__ == "__main__":
    main()
