"""Дисковый предкэш графа по полигону арены (отдельно от fixed-area и corridor cache).

Используется при ``GRAPH_CORRIDOR_MODE=true``: если bbox запроса целиком внутри
заранее подготовленного полигона, граф подгружается с диска без live Overpass.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import networkx as nx
import osmnx as ox
from shapely.geometry import box
from shapely.ops import unary_union

from ..config import Settings, routing_engine_cache_fingerprint
from .corridor_graph_cache import corridor_graph_cache_fingerprint

if TYPE_CHECKING:
    from ..app import Application

logger = logging.getLogger(__name__)

AREA_PRECACHE_SCHEMA = "area_precache_v1"

_SAT_PHASE_STUB = "stub"
_SAT_PHASE_FULL = "full"


def area_precache_content_fingerprint(settings: Settings) -> str:
    """Отпечаток логики весов и зелени для инвалидации каталога арены."""
    payload = {
        "algo_fp": routing_engine_cache_fingerprint(),
        "analyze_corridor": settings.analyze_corridor,
        "cache_tile_analysis": settings.cache_tile_analysis,
        "corridor_ctx": corridor_graph_cache_fingerprint(settings),
        "disable_satellite_green": settings.disable_satellite_green,
        "force_recalculate": settings.force_recalculate,
        "road_buffer_meters": settings.road_buffer_meters,
        "satellite_zoom": settings.satellite_zoom,
        "schema": AREA_PRECACHE_SCHEMA,
        "tms_server": settings.tms_server,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def area_precache_directory_id(
    wkt: str, area_name: str, content_fp: str
) -> str:
    """Стабильный идентификатор каталога ``cache/area_precache/<id>/``."""
    raw = json.dumps(
        {
            "content_fp": content_fp,
            "name": (area_name or "default").strip(),
            "wkt": wkt.strip(),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def parse_precache_polygon(settings: Settings) -> Any:
    """Разобрать ``PRECACHE_AREA_POLYGON_WKT`` в геометрию Shapely (lon, lat)."""
    from shapely import wkt as shapely_wkt

    raw = settings.precache_area_polygon_wkt_stripped
    if not raw:
        raise ValueError("PRECACHE_AREA_POLYGON_WKT пуст")
    geom = shapely_wkt.loads(raw)
    if geom.is_empty:
        raise ValueError("PRECACHE_AREA_POLYGON_WKT: пустая геометрия")
    if geom.geom_type == "MultiPolygon":
        geom = unary_union(geom)
    if geom.geom_type != "Polygon":
        raise ValueError(
            f"PRECACHE_AREA_POLYGON_WKT: нужен Polygon или MultiPolygon, "
            f"получено {geom.geom_type}"
        )
    if not geom.is_valid:
        geom = geom.buffer(0)
    return geom


def arena_wgs84_tuple(geom: Any) -> Tuple[float, float, float, float]:
    """Формат ``RouteEngine._corridor_wgs84``: (min_lat, max_lat, min_lon, max_lon)."""
    b = geom.bounds  # minx, miny, maxx, maxy = lon
    min_lon, min_lat, max_lon, max_lat = b[0], b[1], b[2], b[3]
    return (min_lat, max_lat, min_lon, max_lon)


def corridor_box_from_wgs84(
    required_wgs84: Tuple[float, float, float, float],
) -> Any:
    """Прямоугольник коридора запроса в lon/lat."""
    min_lat, max_lat, min_lon, max_lon = required_wgs84
    return box(min_lon, min_lat, max_lon, max_lat)


def precache_area_root(settings: Settings) -> Path:
    return Path(settings.cache_dir) / "area_precache"


def precache_area_dir(settings: Settings) -> Path:
    if not settings.has_precache_area_polygon:
        raise ValueError("нет PRECACHE_AREA_POLYGON_WKT")
    wkt = settings.precache_area_polygon_wkt_stripped
    fp = area_precache_content_fingerprint(settings)
    aid = area_precache_directory_id(wkt, settings.precache_area_name, fp)
    return precache_area_root(settings) / aid


def meta_path(settings: Settings) -> Path:
    return precache_area_dir(settings) / "meta.json"


def polygon_wkt_path(settings: Settings) -> Path:
    return precache_area_dir(settings) / "polygon.wkt"


def graph_base_path(settings: Settings) -> Path:
    return precache_area_dir(settings) / "graph_base.graphml"


def graph_green_path(settings: Settings) -> Path:
    return precache_area_dir(settings) / "graph_green.graphml"


def load_meta(settings: Settings) -> Optional[Dict[str, Any]]:
    p = meta_path(settings)
    if not p.is_file():
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("area_precache: не удалось прочитать meta.json: %s", exc)
        return None


def meta_matches_current(meta: Dict[str, Any], settings: Settings) -> bool:
    if meta.get("schema") != AREA_PRECACHE_SCHEMA:
        return False
    cur = area_precache_content_fingerprint(settings)
    return meta.get("fingerprint") == cur


def save_meta(
    settings: Settings,
    geom: Any,
    *,
    has_green_graph: bool,
    nodes: int,
    edges: int,
    green_phase_pending: bool = False,
) -> None:
    d = precache_area_dir(settings)
    d.mkdir(parents=True, exist_ok=True)
    wkt = settings.precache_area_polygon_wkt_stripped
    fp = area_precache_content_fingerprint(settings)
    payload = {
        "arena_wgs84": list(arena_wgs84_tuple(geom)),
        "area_name": settings.precache_area_name,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "edges": edges,
        "fingerprint": fp,
        "green_phase_pending": green_phase_pending,
        "has_green_graph": has_green_graph,
        "nodes": nodes,
        "routing_algo_fp": routing_engine_cache_fingerprint(),
        "schema": AREA_PRECACHE_SCHEMA,
        "wkt": wkt,
    }
    mp = meta_path(settings)
    tmp = mp.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp.replace(mp)
    with open(polygon_wkt_path(settings), "w", encoding="utf-8") as f:
        f.write(wkt)
    logger.info("area_precache: meta сохранён в %s", d)


def _save_graphml(G: nx.MultiDiGraph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".graphml.tmp")
    ox.save_graphml(G, filepath=tmp)
    tmp.replace(path)


def load_graphml_path(path: Path) -> Optional[nx.MultiDiGraph]:
    if not path.is_file() or path.stat().st_size < 64:
        return None
    try:
        G = ox.load_graphml(filepath=path)
        if not isinstance(G, nx.MultiDiGraph):
            G = nx.MultiDiGraph(G)
        if G.number_of_nodes() == 0:
            return None
        return G
    except Exception as exc:
        logger.warning("area_precache: не удалось загрузить %s: %s", path, exc)
        return None


def cleanup_stale_precache_tmp_files(settings: Settings) -> None:
    """Удалить *.tmp после прерванной записи (иначе сборка может зависнуть на битых файлах)."""
    if not settings.has_precache_area_polygon:
        return
    d = precache_area_dir(settings)
    if not d.is_dir():
        return
    for name in ("meta.json.tmp", "graph_base.graphml.tmp", "graph_green.graphml.tmp"):
        p = d / name
        if p.is_file():
            try:
                p.unlink()
                logger.info("area_precache: удалён незавершённый %s", name)
            except OSError as exc:
                logger.warning("area_precache: не удалось удалить %s: %s", p, exc)


def precache_area_is_complete(settings: Settings) -> bool:
    """True, если ``graph_base`` есть, ``meta.json`` актуален, а зелёная фаза либо на диске, либо осознанно пропущена."""
    if not settings.has_precache_area_polygon:
        return True
    meta = load_meta(settings)
    if not meta or not meta_matches_current(meta, settings):
        return False
    if not graph_base_path(settings).is_file():
        return False
    expect_green = (
        settings.precache_area_use_green_graph
        and not settings.disable_satellite_green
    )
    if not expect_green:
        return True
    if graph_green_path(settings).is_file():
        return True
    # Зелёная фаза ещё не догружена (прерван precache после graph_base)
    if meta.get("green_phase_pending"):
        return False
    # В ``meta`` зафиксировано, что зелень не собиралась (например, не было Pillow) — считаем готово
    return meta.get("has_green_graph") is False


def _complete_green_only_from_base(application: "Application", geom: Any) -> Path:
    """Доначитка ``graph_green`` из уже сохранённого ``graph_base`` (повтор после обрыва или только phase1)."""
    s = application.settings
    gb = application.graph_builder
    base_p = graph_base_path(s)
    G = load_graphml_path(base_p)
    if G is None:
        logger.warning("area_precache: graph_base битый — полная пересборка")
        return build_area_precache(application)

    if not (
        s.precache_area_use_green_graph
        and not s.disable_satellite_green
        and application.green._tiles.pil_available
    ):
        save_meta(
            s,
            geom,
            has_green_graph=False,
            nodes=G.number_of_nodes(),
            edges=G.number_of_edges(),
            green_phase_pending=False,
        )
        return precache_area_dir(s)

    logger.info("area_precache: доначитка спутниковой зелени с существующего graph_base…")
    G_work = G.copy()
    e2 = gb.to_geodataframe(G_work)
    e2 = gb.upgrade_edges_satellite_weights(e2)
    Gg = gb.apply_weights(G_work, e2)
    Gg.graph["bike_router_satellite_phase"] = _SAT_PHASE_FULL
    _save_graphml(Gg, graph_green_path(s))
    logger.info(
        "area_precache: сохранён graph_green.graphml (%d узлов)",
        Gg.number_of_nodes(),
    )
    save_meta(
        s,
        geom,
        has_green_graph=True,
        nodes=G.number_of_nodes(),
        edges=G.number_of_edges(),
        green_phase_pending=False,
    )
    return precache_area_dir(s)


def ensure_area_precache(application: "Application") -> Path:
    """Идемпотентная подготовка: если кэш полный — выход; иначе сборка или только зелёная фаза."""
    s = application.settings
    cleanup_stale_precache_tmp_files(s)
    if not s.has_precache_area_polygon:
        raise ValueError("Задайте PRECACHE_AREA_POLYGON_WKT")
    if precache_area_is_complete(s):
        out = precache_area_dir(s)
        logger.info(
            "area_precache: кэш арены уже готов (%s…), пропуск",
            out.name[:16],
        )
        return out
    geom = parse_precache_polygon(s)
    meta_ok = load_meta(s)
    meta_ok = meta_ok is not None and meta_matches_current(meta_ok, s)
    base_ok = graph_base_path(s).is_file()
    wants_green = (
        s.precache_area_use_green_graph
        and not s.disable_satellite_green
        and application.green._tiles.pil_available
    )
    if meta_ok and base_ok and wants_green and not graph_green_path(s).is_file():
        logger.info(
            "area_precache: graph_base есть, graph_green нет — догружаем только зелёную фазу"
        )
        return _complete_green_only_from_base(application, geom)
    return build_area_precache(application)


def build_area_precache(application: Application) -> Path:
    """Полная предсборка: OSM по полигону, base + опционально green graph, meta, polygon.wkt."""
    s = application.settings
    if not s.has_precache_area_polygon:
        raise ValueError("Задайте PRECACHE_AREA_POLYGON_WKT")
    cleanup_stale_precache_tmp_files(s)
    geom = parse_precache_polygon(s)
    b = geom.bounds
    mid_lat = (b[1] + b[3]) / 2.0
    mid_lon = (b[0] + b[2]) / 2.0
    application.elevation.init(test_lat=mid_lat, test_lon=mid_lon)

    gb = application.graph_builder
    logger.info("area_precache: загрузка OSM по полигону арены…")
    G = gb.load(geom)
    edges = gb.to_geodataframe(G)
    logger.info("area_precache: расчёт весов phase1 (без спутниковой зелени)…")
    edges = gb.calculate_weights(edges, skip_satellite_green=True)
    G = gb.apply_weights(G, edges)
    G.graph["bike_router_satellite_phase"] = _SAT_PHASE_STUB

    out_dir = precache_area_dir(s)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_p = graph_base_path(s)
    _save_graphml(G, base_p)
    logger.info(
        "area_precache: сохранён %s (%d узлов)",
        base_p.name,
        G.number_of_nodes(),
    )

    wants_satellite = (
        s.precache_area_use_green_graph
        and not s.disable_satellite_green
        and application.green._tiles.pil_available
    )
    has_green = False

    if wants_satellite:
        # Промежуточный meta: после обрыва на тайлах не перекачивать OSM — только зелёная фаза
        save_meta(
            s,
            geom,
            has_green_graph=False,
            nodes=G.number_of_nodes(),
            edges=G.number_of_edges(),
            green_phase_pending=True,
        )
        logger.info("area_precache: доначитка спутниковой зелени (как в runtime)…")
        G_work = G.copy()
        e2 = gb.to_geodataframe(G_work)
        e2 = gb.upgrade_edges_satellite_weights(e2)
        Gg = gb.apply_weights(G_work, e2)
        Gg.graph["bike_router_satellite_phase"] = _SAT_PHASE_FULL
        _save_graphml(Gg, graph_green_path(s))
        has_green = True
        logger.info(
            "area_precache: сохранён graph_green.graphml (%d узлов)",
            Gg.number_of_nodes(),
        )
        save_meta(
            s,
            geom,
            has_green_graph=True,
            nodes=G.number_of_nodes(),
            edges=G.number_of_edges(),
            green_phase_pending=False,
        )
    elif s.precache_area_use_green_graph and s.disable_satellite_green:
        logger.info(
            "area_precache: DISABLE_SATELLITE_GREEN=true — graph_green не строится"
        )
        save_meta(
            s,
            geom,
            has_green_graph=False,
            nodes=G.number_of_nodes(),
            edges=G.number_of_edges(),
            green_phase_pending=False,
        )
    else:
        if (
            s.precache_area_use_green_graph
            and not application.green._tiles.pil_available
        ):
            logger.info(
                "area_precache: Pillow недоступен — graph_green со спутником не строится"
            )
        save_meta(
            s,
            geom,
            has_green_graph=False,
            nodes=G.number_of_nodes(),
            edges=G.number_of_edges(),
            green_phase_pending=False,
        )
    return out_dir


def precache_corridor_fits_arena(
    settings: Settings, required_wgs84: Tuple[float, float, float, float]
) -> bool:
    """True, если ось-выровненный bbox коридора целиком внутри полигона арены."""
    try:
        geom = parse_precache_polygon(settings)
    except Exception:
        return False
    cb = corridor_box_from_wgs84(required_wgs84)
    return bool(geom.covers(cb))


def select_precache_graph_path(
    settings: Settings, need_satellite: bool
) -> Optional[Path]:
    """Какой graphml использовать; ``None`` — откат на live corridor."""
    g_green = graph_green_path(settings)
    g_base = graph_base_path(settings)
    if need_satellite:
        if g_green.is_file():
            return g_green
        if not settings.precache_area_use_green_graph and g_base.is_file():
            return g_base
        return None
    if g_base.is_file():
        return g_base
    if g_green.is_file():
        return g_green
    return None


def select_warmup_precache_graph_path(settings: Settings) -> Optional[Path]:
    """Лучший graphml для предзагрузки при старте: зелёный при наличии и настройках, иначе base."""
    if (
        settings.precache_area_use_green_graph
        and not settings.disable_satellite_green
    ):
        p = select_precache_graph_path(settings, need_satellite=True)
        if p is not None:
            return p
    return select_precache_graph_path(settings, need_satellite=False)
