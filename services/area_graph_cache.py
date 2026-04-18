"""Дисковый предкэш графа по полигону арены (отдельно от fixed-area и corridor cache).

Используется при ``GRAPH_CORRIDOR_MODE=true``: если bbox запроса целиком внутри
заранее подготовленного полигона, граф подгружается с диска без live Overpass.

**Слой A (статический):** каталог ``area_precache/<hash>`` и поле ``fingerprint`` в
``meta.json`` зависят только от :func:`area_static_content_fingerprint` — полигон
(через WKT в id), OSM-фильтра, спутника (zoom, TMS, буферы), пайплайна зелени.
Смена профилей balanced/safe, тепло/стресс, слотов, погоды и прочего scoring **не**
должна менять этот hash (см. schema ``area_precache_v3``).

**Слой B (модель):** в ``meta.json`` дополнительно пишется ``routing_algo_fp`` для
диагностики; маршрутизация и погода работают в runtime поверх уже загруженного графа.

**Слой C (статическое озеленение арены):** ``cache/area_green_edges/<id>/green_edges.pkl``
и sidecar ``meta.json`` — агрегация зелени по рёбрам один раз; fingerprint без
погоды/профилей (см. :func:`area_green_edges_content_fingerprint`). Повторный
precache подмешивает этот слой без повторной агрегации по коридору.

Если изменилась только формула весов в ``graph.calculate_weights``, которая меняет
сохранённые в GraphML поля, может понадобиться явная пересборка precache — это
отдельно от hash статики.
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

from ..config import OSM_HIGHWAY_FILTER, Settings, routing_engine_cache_fingerprint

if TYPE_CHECKING:
    from ..app import Application

logger = logging.getLogger(__name__)

AREA_PRECACHE_SCHEMA = "area_precache_v3"

# Статический кэш агрегации зелени по рёбрам арены (без погоды/профилей).
# Должен совпадать с префиксом в services/green.py (GREEN_EDGES_CACHE_SCHEMA не тот же объект).
AREA_GREEN_EDGES_SCHEMA = "area_green_edges_v1"

# Должен совпадать с services/green.TILE_VEG_MASK_VERSION (только для fingerprint).
_TILE_VEG_MASK_VERSION_FP = "ndisexg_v2_tileval"

_SAT_PHASE_STUB = "stub"
_SAT_PHASE_FULL = "full"


def area_static_content_fingerprint(settings: Settings) -> str:
    """Статический отпечаток арены: OSM, спутник, буферы, пайплайн зелени.

    Не включает ``routing_engine_cache_fingerprint`` (профили, тепло/стресс, слоты),
    погоду и прочий runtime scoring — чтобы не пересобирать тайлы и OSM при смене
    только модели маршрутизации.
    """
    payload = {
        "analyze_corridor": settings.analyze_corridor,
        "buffer_deg": settings.buffer,
        "cache_tile_analysis": settings.cache_tile_analysis,
        "corridor_buffer_m": settings.corridor_buffer_meters,
        "corridor_cache_grid_step_deg": settings.corridor_cache_grid_step_deg,
        "disable_satellite_green": settings.disable_satellite_green,
        "force_recalculate": settings.force_recalculate,
        "green_pixel_metric": "v1_sum_M_intersect",
        "green_tile_batch_size": settings.green_tile_batch_size,
        "osm_highway_filter": OSM_HIGHWAY_FILTER,
        "road_buffer_meters": settings.road_buffer_meters,
        "satellite_zoom": settings.satellite_zoom,
        "schema": AREA_PRECACHE_SCHEMA,
        "tms_server": settings.tms_server,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def area_precache_content_fingerprint(settings: Settings) -> str:
    """Историческое имя: то же, что :func:`area_static_content_fingerprint`."""
    return area_static_content_fingerprint(settings)


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


def area_green_edges_root(settings: Settings) -> Path:
    return Path(settings.cache_dir) / "area_green_edges"


def area_green_edges_arena_dir(settings: Settings) -> Path:
    """``cache/area_green_edges/<id>/`` — тот же ``id``, что у ``area_precache/<id>/``."""
    if not settings.has_precache_area_polygon:
        raise ValueError("нет PRECACHE_AREA_POLYGON_WKT")
    wkt = settings.precache_area_polygon_wkt_stripped
    fp = area_static_content_fingerprint(settings)
    aid = area_precache_directory_id(wkt, settings.precache_area_name, fp)
    return area_green_edges_root(settings) / aid


def area_green_edges_pkl_path(settings: Settings) -> Path:
    return area_green_edges_arena_dir(settings) / "green_edges.pkl"


def area_green_edges_bundle_meta_path(settings: Settings) -> Path:
    """meta.json слоя area_green_edges (не путать с meta.json в area_precache)."""
    return area_green_edges_arena_dir(settings) / "meta.json"


def area_green_edges_content_fingerprint(settings: Settings) -> str:
    """Отпечаток только пайплайна зелени (спутник, буфер, TMS, маски) — без scoring/погоды."""
    payload = {
        "analyze_corridor": settings.analyze_corridor,
        "cache_tile_analysis": settings.cache_tile_analysis,
        "disable_satellite_green": settings.disable_satellite_green,
        "green_pixel_metric": "v1_sum_M_intersect",
        "road_buffer_meters": settings.road_buffer_meters,
        "satellite_zoom": settings.satellite_zoom,
        "schema": AREA_GREEN_EDGES_SCHEMA,
        "tile_mask_ver": _TILE_VEG_MASK_VERSION_FP,
        "tms_server": settings.tms_server,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_area_green_edges_bundle_meta(
    settings: Settings,
) -> Optional[Dict[str, Any]]:
    p = area_green_edges_bundle_meta_path(settings)
    if not p.is_file():
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("area_green_edges: не удалось прочитать meta.json: %s", exc)
        return None


def area_green_edges_bundle_is_valid(
    settings: Settings, edges_count: int
) -> bool:
    """Файлы на месте, fingerprint и число рёбер совпадают с ожиданием."""
    meta = load_area_green_edges_bundle_meta(settings)
    if not meta or meta.get("schema") != AREA_GREEN_EDGES_SCHEMA:
        return False
    if meta.get("fingerprint") != area_green_edges_content_fingerprint(settings):
        return False
    if int(meta.get("edge_count", -1)) != int(edges_count):
        return False
    pkl = area_green_edges_pkl_path(settings)
    return pkl.is_file() and pkl.stat().st_size > 32


def save_area_green_edges_bundle(
    settings: Settings,
    cache_service: Any,
    records: Dict[Any, Any],
    *,
    edge_count: int,
    green_fp: str,
    green_edges_semantic_ok: int,
    green_edges_quality_ok: bool,
) -> None:
    """Сохранить статический edge-cache зелени арены (pickle + sidecar meta)."""
    d = area_green_edges_arena_dir(settings)
    d.mkdir(parents=True, exist_ok=True)
    pkl = area_green_edges_pkl_path(settings)
    ok = cache_service.save(pkl.as_posix(), records)
    if not ok:
        logger.warning("area_green_edges: не удалось записать %s", pkl)
        return
    mp = area_green_edges_bundle_meta_path(settings)
    tmp = mp.with_suffix(".json.tmp")
    sidecar = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "edge_count": int(edge_count),
        "fingerprint": green_fp,
        "green_edges_quality_ok": bool(green_edges_quality_ok),
        "green_edges_semantic_ok_count": int(green_edges_semantic_ok),
        "schema": AREA_GREEN_EDGES_SCHEMA,
    }
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, ensure_ascii=False, indent=2)
    tmp.replace(mp)
    logger.info(
        "area_green_edges: сохранён статический кэш (%d рёбер) в %s",
        edge_count,
        d,
    )


def persist_area_green_edges_if_snapshot(
    application: "Application",
) -> Dict[str, Any]:
    """После полного ``calculate_satellite_batch``: записать ``area_green_edges`` и поля для precache meta."""
    from ..services.green import normalize_edge_index_key

    g = application.green
    snap = getattr(g, "_last_edge_cache_snapshot", None)
    if not snap:
        return {}
    s = application.settings
    if not s.has_precache_area_polygon:
        return {}
    q = g.last_satellite_quality_report() or {}
    edge_count = len(snap)
    gfp = area_green_edges_content_fingerprint(s)
    n_semantic = int(q.get("edges_semantic_ok", 0))
    ok_qual = bool(
        q.get("persistable_for_cache") and q.get("all_edges_semantic_ok", False)
    )
    norm = {normalize_edge_index_key(k): dict(v) for k, v in snap.items()}
    save_area_green_edges_bundle(
        s,
        application.cache,
        norm,
        edge_count=edge_count,
        green_fp=gfp,
        green_edges_semantic_ok=n_semantic,
        green_edges_quality_ok=ok_qual,
    )
    root = Path(s.cache_dir)
    ar = area_green_edges_arena_dir(s)
    try:
        rel = str(ar.relative_to(root))
    except ValueError:
        rel = str(ar.name)
    return {
        "area_green_edges_fingerprint": gfp,
        "area_green_edges_rel_path": rel,
        "area_green_edges_schema": AREA_GREEN_EDGES_SCHEMA,
        "green_edges_count": int(edge_count),
        "green_edges_complete": True,
        "green_edges_quality_ok": ok_qual,
    }


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
    graph_base_ready: bool = True,
    green_quality_state: Optional[str] = None,
    invalid_tiles_rejected: int = 0,
    green_edges_semantic_ok_count: int = 0,
    green_from_valid_imagery: Optional[bool] = None,
    area_green_edges_fingerprint: Optional[str] = None,
    area_green_edges_rel_path: Optional[str] = None,
    area_green_edges_schema: Optional[str] = None,
    green_edges_count: Optional[int] = None,
    green_edges_complete: Optional[bool] = None,
    green_edges_quality_ok: Optional[bool] = None,
) -> None:
    d = precache_area_dir(settings)
    d.mkdir(parents=True, exist_ok=True)
    wkt = settings.precache_area_polygon_wkt_stripped
    fp = area_precache_content_fingerprint(settings)
    gqs = green_quality_state or (
        "disabled"
        if not (
            settings.precache_area_use_green_graph
            and not settings.disable_satellite_green
        )
        else "unknown"
    )
    expect_g = (
        settings.precache_area_use_green_graph
        and not settings.disable_satellite_green
    )
    stages = {
        "completed": graph_base_ready
        and ((not expect_g) or (gqs == "ok" and has_green_graph)),
        "graph_base_ready": graph_base_ready,
        "graph_green_ready": bool(has_green_graph and gqs == "ok"),
        "green_edges_ready": has_green_graph and gqs == "ok",
        "green_tiles_validated": (not expect_g) or (gqs == "ok" and invalid_tiles_rejected == 0),
    }
    payload = {
        "arena_wgs84": list(arena_wgs84_tuple(geom)),
        "area_name": settings.precache_area_name,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "edges": edges,
        "fingerprint": fp,
        "graph_base_ready": graph_base_ready,
        "green_edges_semantic_ok_count": int(green_edges_semantic_ok_count),
        "green_from_valid_imagery": green_from_valid_imagery,
        "green_phase_pending": green_phase_pending,
        "green_quality_state": gqs,
        "has_green_graph": has_green_graph,
        "invalid_tiles_rejected": int(invalid_tiles_rejected),
        "nodes": nodes,
        "routing_algo_fp": routing_engine_cache_fingerprint(),
        "schema": AREA_PRECACHE_SCHEMA,
        "stages": stages,
        "wkt": wkt,
    }
    if area_green_edges_fingerprint is not None:
        payload["area_green_edges_fingerprint"] = area_green_edges_fingerprint
    if area_green_edges_rel_path is not None:
        payload["area_green_edges_rel_path"] = area_green_edges_rel_path
    if area_green_edges_schema is not None:
        payload["area_green_edges_schema"] = area_green_edges_schema
    if green_edges_count is not None:
        payload["green_edges_count"] = int(green_edges_count)
    if green_edges_complete is not None:
        payload["green_edges_complete"] = bool(green_edges_complete)
    if green_edges_quality_ok is not None:
        payload["green_edges_quality_ok"] = bool(green_edges_quality_ok)
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
    """Готовность area precache: файлы + ``green_quality_state`` (v2 meta)."""
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
    green_p = graph_green_path(settings)
    has_green_file = green_p.is_file()
    gqs = (meta.get("green_quality_state") or "").strip()

    if not expect_green:
        return True

    if bool(meta.get("green_phase_pending")):
        return False

    if gqs in ("invalid_imagery", "incomplete", "unknown"):
        logger.info(
            "area_precache: не готов — green_quality_state=%s",
            gqs or "∅",
        )
        return False

    if gqs == "ok":
        if has_green_file:
            return True
        logger.warning(
            "area_precache: meta green_quality_state=ok, но нет %s",
            green_p.name,
        )
        return False

    if expect_green and gqs == "no_satellite":
        logger.info(
            "area_precache: не готов — спутниковая зелень недоступна (Pillow / no_satellite)"
        )
        return False

    # Устаревший meta без v2 или неизвестное состояние
    return False


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
            green_quality_state="no_satellite",
            green_from_valid_imagery=False,
        )
        return precache_area_dir(s)

    logger.info(
        "area_precache: этап 2 — статическое озеленение (graph_green из graph_base)…"
    )
    G_work = G.copy()
    e2 = gb.to_geodataframe(G_work)
    e2 = gb.upgrade_edges_satellite_weights(e2)
    q = application.green.last_satellite_quality_report() or {}
    if not application.green.green_analysis_is_acceptable():
        gp = graph_green_path(s)
        if gp.is_file():
            try:
                gp.unlink()
            except OSError:
                pass
        save_meta(
            s,
            geom,
            has_green_graph=False,
            nodes=G.number_of_nodes(),
            edges=G.number_of_edges(),
            green_phase_pending=False,
            green_quality_state="invalid_imagery",
            invalid_tiles_rejected=int(q.get("invalid_tiles_total", 0)),
            green_edges_semantic_ok_count=int(q.get("edges_semantic_ok", 0)),
            green_from_valid_imagery=False,
        )
        logger.error(
            "area_precache: graph_green не сохранён — спутниковые данные не прошли проверку качества"
        )
        return precache_area_dir(s)

    Gg = gb.apply_weights(G_work, e2)
    Gg.graph["bike_router_satellite_phase"] = _SAT_PHASE_FULL
    _save_graphml(Gg, graph_green_path(s))
    logger.info(
        "area_precache: этап 3 — сохранён graph_green.graphml (%d узлов)",
        Gg.number_of_nodes(),
    )
    ag_meta = persist_area_green_edges_if_snapshot(application)
    save_meta(
        s,
        geom,
        has_green_graph=True,
        nodes=G.number_of_nodes(),
        edges=G.number_of_edges(),
        green_phase_pending=False,
        green_quality_state="ok",
        invalid_tiles_rejected=int(q.get("invalid_tiles_total", 0)),
        green_edges_semantic_ok_count=int(q.get("edges_semantic_ok", 0)),
        green_from_valid_imagery=True,
        area_green_edges_fingerprint=ag_meta.get("area_green_edges_fingerprint"),
        area_green_edges_rel_path=ag_meta.get("area_green_edges_rel_path"),
        area_green_edges_schema=ag_meta.get("area_green_edges_schema"),
        green_edges_count=ag_meta.get("green_edges_count"),
        green_edges_complete=ag_meta.get("green_edges_complete"),
        green_edges_quality_ok=ag_meta.get("green_edges_quality_ok"),
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
        gg = graph_green_path(s)
        logger.info(
            "area_precache: кэш арены уже готов (%s…), пропуск · каталог=%s · graph_green=%s",
            out.name[:16],
            out,
            "есть" if gg.is_file() else "нет",
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
    logger.info("area_precache: этап 1 — загрузка OSM по полигону арены…")
    G = gb.load(geom)
    edges = gb.to_geodataframe(G)
    logger.info("area_precache: этап 1 — расчёт весов phase1 (без спутниковой зелени)…")
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
        save_meta(
            s,
            geom,
            has_green_graph=False,
            nodes=G.number_of_nodes(),
            edges=G.number_of_edges(),
            green_phase_pending=True,
            green_quality_state="incomplete",
        )
        logger.info(
            "area_precache: этап 2 — статическое озеленение (area_green_edges или полный спутник)…"
        )
        G_work = G.copy()
        e2 = gb.to_geodataframe(G_work)
        e2 = gb.upgrade_edges_satellite_weights(e2)
        q = application.green.last_satellite_quality_report() or {}
        if not application.green.green_analysis_is_acceptable():
            gp = graph_green_path(s)
            if gp.is_file():
                try:
                    gp.unlink()
                except OSError:
                    pass
            save_meta(
                s,
                geom,
                has_green_graph=False,
                nodes=G.number_of_nodes(),
                edges=G.number_of_edges(),
                green_phase_pending=False,
                green_quality_state="invalid_imagery",
                invalid_tiles_rejected=int(q.get("invalid_tiles_total", 0)),
                green_edges_semantic_ok_count=int(q.get("edges_semantic_ok", 0)),
                green_from_valid_imagery=False,
            )
            logger.error(
                "area_precache: graph_green не сохранён — невалидные или неполные спутниковые данные"
            )
            return out_dir

        Gg = gb.apply_weights(G_work, e2)
        Gg.graph["bike_router_satellite_phase"] = _SAT_PHASE_FULL
        _save_graphml(Gg, graph_green_path(s))
        has_green = True
        logger.info(
            "area_precache: этап 3 — сохранён graph_green.graphml (%d узлов)",
            Gg.number_of_nodes(),
        )
        ag_meta = persist_area_green_edges_if_snapshot(application)
        save_meta(
            s,
            geom,
            has_green_graph=True,
            nodes=G.number_of_nodes(),
            edges=G.number_of_edges(),
            green_phase_pending=False,
            green_quality_state="ok",
            invalid_tiles_rejected=int(q.get("invalid_tiles_total", 0)),
            green_edges_semantic_ok_count=int(q.get("edges_semantic_ok", 0)),
            green_from_valid_imagery=True,
            area_green_edges_fingerprint=ag_meta.get("area_green_edges_fingerprint"),
            area_green_edges_rel_path=ag_meta.get("area_green_edges_rel_path"),
            area_green_edges_schema=ag_meta.get("area_green_edges_schema"),
            green_edges_count=ag_meta.get("green_edges_count"),
            green_edges_complete=ag_meta.get("green_edges_complete"),
            green_edges_quality_ok=ag_meta.get("green_edges_quality_ok"),
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
            green_quality_state="disabled",
            green_from_valid_imagery=False,
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
            green_quality_state="no_satellite",
            green_from_valid_imagery=False,
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
