"""Движок маршрутизации: чистая функция compute_route.

При фиксированной зоне (AREA / полигон) граф загружается в ``warmup()`` один раз.
В режиме ``GRAPH_CORRIDOR_MODE`` граф и спутниковая зелень строятся по прямоугольнику
между точками запроса ± ``BUFFER`` (пересборка, если новая пара точек вне коридора).
"""

from __future__ import annotations

import logging
import math
import threading
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import osmnx as ox

from .app import Application
from .config import PROFILES, Settings, routing_engine_cache_fingerprint
from .exceptions import (
    BikeRouterError,
    PointOutsideZoneError,
    RouteNotFoundError,
    RouteTooLongError,
)
from .models import (
    AlternativesResponse,
    ElevationMetrics,
    ElevationPoint,
    GreenMetrics,
    MapLayersGeoJSON,
    RouteQualityHints,
    RouteResponse,
    StairsInfo,
    SurfaceBreakdown,
)
from .services.corridor_graph_cache import (
    CorridorGraphDiskCache,
    corridor_bbox_cache_key,
    quantize_corridor_bbox_expanding,
)
from .services.route_cache import RouteAlternativesDiskCache
from .metrics import (
    inc_corridor_disk_hit,
    inc_route_disk_hit,
    inc_route_disk_miss,
)
from .services.routing import (
    RouteResult,
    RouteService,
    sanitize_multidigraph_routing_weights,
)

logger = logging.getLogger(__name__)

# Метка в ``G.graph``: phase1 без тайлов vs готовность зелёных весов
_SAT_PHASE_STUB = "stub"
_SAT_PHASE_FULL = "full"

_PROFILE_MAP = {p.key: p for p in PROFILES}

_VARIANT_LABEL_DEFAULT = {
    "full": "Оптимальный по энергии",
    "green": "С учётом озеленения",
    "shortest": "Кратчайший по карте",
}


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    if m >= 60:
        h, m = divmod(m, 60)
        return f"{h} ч {m} мин"
    return f"{m} мин {s} сек"


def _raise_route_not_found_after_corridor_expand(
    last_exc: RouteNotFoundError,
    *,
    eff: float,
    base: float,
    max_extra: float,
    step: float,
    attempt: int,
    max_attempts: int,
) -> None:
    """После исчерпания AUTO_EXPAND — сообщение для API/UI (код NO_PATH)."""
    msg = (
        f"Маршрут не найден после {attempt} попыток (последний коридор ±{eff:.0f} м; "
        f"база {base:.0f} м, шаг {step:.0f} м, не более {max_attempts} попыток, "
        f"макс. доп. к буферу {max_extra:.0f} м). "
        "Попробуйте другие точки ближе к связной сети дорог."
    )
    raise RouteNotFoundError(last_exc.weight_key, msg) from last_exc


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Расстояние между двумя точками на сфере (метры)."""
    R = 6_371_000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class RouteEngine:
    """Движок с предзагруженным графом.

    Паттерн использования::

        engine = RouteEngine()
        engine.warmup()                       # один раз при старте
        result = engine.compute_route(...)     # многократно
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._app = Application(settings)
        self._graph: Optional[nx.MultiDiGraph] = None
        self._loaded = False
        self._bounds: Optional[Tuple[float, float, float, float]] = None
        s = self._app.settings
        self._route_disk_cache = RouteAlternativesDiskCache(
            Path(s.cache_dir) / "route_alternatives_cache",
            enabled=s.route_disk_cache,
        )
        self._corridor_graph_cache = CorridorGraphDiskCache(
            Path(s.cache_dir) / "corridor_graphs",
            enabled=s.corridor_graph_disk_cache,
        )
        self._graph_built_at_utc: Optional[str] = None
        self._graph_lock = threading.RLock()
        # Дедупликация одновременной сборки одного и того же ключа коридора (GraphML/OSM).
        self._corridor_gate_locks: Dict[str, threading.Lock] = {}
        self._corridor_gate_guard = threading.Lock()
        # (min_lat, max_lat, min_lon, max_lon) последнего коридора create_bbox(start,end,buffer)
        self._corridor_wgs84: Optional[Tuple[float, float, float, float]] = None
        # True — последний граф без спутниковой зелени (заглушки)
        self._graph_built_skip_satellite: Optional[bool] = None

    # ── Свойства ─────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def graph(self) -> Optional[nx.MultiDiGraph]:
        return self._graph

    @property
    def settings(self) -> Settings:
        return self._app.settings

    @property
    def bounds(self) -> Optional[Tuple[float, float, float, float]]:
        """(min_lat, max_lat, min_lon, max_lon) покрытия графа."""
        return self._bounds

    @property
    def graph_built_at_utc(self) -> Optional[str]:
        """ISO UTC времени последней успешной сборки графа."""
        return self._graph_built_at_utc

    @staticmethod
    def routing_weights_fingerprint() -> str:
        """Отпечаток текущих параметров весов (для health / кэша)."""
        return routing_engine_cache_fingerprint()

    # ── Валидация ────────────────────────────────────────────────

    def _validate_point(
        self, coords: Tuple[float, float], label: str
    ) -> None:
        """Проверить, что точка в зоне покрытия и рядом с дорогой."""
        if self._bounds is None or self._graph is None:
            return
        self._validate_point_on_graph(
            self._graph, self._bounds, coords, label
        )

    def _validate_point_on_graph(
        self,
        G: nx.MultiDiGraph,
        bounds: Tuple[float, float, float, float],
        coords: Tuple[float, float],
        label: str,
    ) -> None:
        """Та же проверка по уже зафиксированному графу (можно вызывать вне долгого lock)."""
        lat, lon = coords
        min_lat, max_lat, min_lon, max_lon = bounds
        margin = 0.002

        if not (
            min_lat - margin <= lat <= max_lat + margin
            and min_lon - margin <= lon <= max_lon + margin
        ):
            raise PointOutsideZoneError(lat, lon, label)

        nearest = ox.distance.nearest_nodes(G, X=lon, Y=lat)
        nd = G.nodes[nearest]
        snap_m = _haversine(lat, lon, nd["y"], nd["x"])
        max_snap = self._app.settings.max_snap_distance_m

        if snap_m > max_snap:
            raise PointOutsideZoneError(lat, lon, label, snap_m)

    # ── Инициализация ────────────────────────────────────────────

    def _set_active_graph(self, G: nx.MultiDiGraph) -> None:
        """Поставить уже готовый граф (после сборки или из GraphML-кэша)."""
        sanitize_multidigraph_routing_weights(G)
        self._graph = G
        self._loaded = True

        lats = [d["y"] for _, d in G.nodes(data=True)]
        lons = [d["x"] for _, d in G.nodes(data=True)]
        self._bounds = (min(lats), max(lats), min(lons), max(lons))

        logger.info(
            "Граф: %d узлов, %d рёбер, bounds=[%.4f..%.4f, %.4f..%.4f]",
            G.number_of_nodes(),
            G.number_of_edges(),
            *self._bounds,
        )
        self._graph_built_at_utc = datetime.now(timezone.utc).isoformat()
        fp = routing_engine_cache_fingerprint()
        self._route_disk_cache.set_cache_context(
            G.number_of_nodes(), G.number_of_edges(), fp
        )

    def _build_graph_from_bbox_shape(
        self, bbox: Any, *, skip_satellite_green: bool = False
    ) -> None:
        """Загрузить OSM, веса (спутник опционально), применить к графу."""
        G = self._app.graph_builder.load(bbox)
        edges = self._app.graph_builder.to_geodataframe(G)
        edges = self._app.graph_builder.calculate_weights(
            edges, skip_satellite_green=skip_satellite_green
        )
        G = self._app.graph_builder.apply_weights(G, edges)
        G.graph["bike_router_satellite_phase"] = (
            _SAT_PHASE_STUB if skip_satellite_green else _SAT_PHASE_FULL
        )
        self._set_active_graph(G)
        self._graph_built_skip_satellite = skip_satellite_green

    @staticmethod
    def _wgs84_corridor_contains(
        outer: Tuple[float, float, float, float],
        inner: Tuple[float, float, float, float],
        *,
        eps_deg: float = 1e-6,
    ) -> bool:
        """``outer`` = (min_lat, max_lat, min_lon, max_lon) как ``_corridor_wgs84``.

        True, если уже загруженный коридор полностью покрывает bbox запроса ``inner``
        (тот же формат), в т.ч. при увеличении буфера авторасширением.
        """
        o_lo_lat, o_hi_lat, o_lo_lon, o_hi_lon = outer
        i_lo_lat, i_hi_lat, i_lo_lon, i_hi_lon = inner
        return (
            o_lo_lat <= i_lo_lat + eps_deg
            and o_hi_lat >= i_hi_lat - eps_deg
            and o_lo_lon <= i_lo_lon + eps_deg
            and o_hi_lon >= i_hi_lon - eps_deg
        )

    def _quantized_corridor_bbox_for_request(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        eff_buf_m: Optional[float],
    ) -> Tuple[Any, Tuple[float, float, float, float], Tuple[float, float, float, float]]:
        """BBox для OSM/кэша; сырые bounds ключа; (min_lat, max_lat, min_lon, max_lon)."""
        s = self._app.settings
        bbox_kw: Dict[str, float] = {}
        if eff_buf_m is not None:
            bbox_kw["buffer_meters"] = float(eff_buf_m)
        raw_bbox = self._app.graph_builder.create_bbox_for_corridor(
            start, end, **bbox_kw
        )
        rmin_lon, rmin_lat, rmax_lon, rmax_lat = raw_bbox.bounds
        q = quantize_corridor_bbox_expanding(
            rmin_lon, rmin_lat, rmax_lon, rmax_lat, s.corridor_cache_grid_step_deg
        )
        if s.corridor_cache_grid_step_deg > 0.0:
            from shapely.geometry import box

            bbox = box(q[0], q[1], q[2], q[3])
        else:
            bbox = raw_bbox
        min_lon, min_lat, max_lon, max_lat = bbox.bounds
        required_wgs84 = (min_lat, max_lat, min_lon, max_lon)
        return bbox, (rmin_lon, rmin_lat, rmax_lon, rmax_lat), required_wgs84

    def _corridor_gate(self, key_hash: str) -> threading.Lock:
        with self._corridor_gate_guard:
            if key_hash not in self._corridor_gate_locks:
                self._corridor_gate_locks[key_hash] = threading.Lock()
            return self._corridor_gate_locks[key_hash]

    def _upgrade_corridor_graph_satellite(self) -> None:
        """Спутник + пересчёт только weight_*_green; OSM и weight_*_full без изменений."""
        if self._graph is None:
            raise BikeRouterError("Граф не загружен.")
        edges = self._app.graph_builder.to_geodataframe(self._graph)
        edges = self._app.graph_builder.upgrade_edges_satellite_weights(edges)
        G = self._app.graph_builder.apply_weights(self._graph, edges)
        G.graph["bike_router_satellite_phase"] = _SAT_PHASE_FULL
        self._set_active_graph(G)
        self._graph_built_skip_satellite = False

    def _ensure_graph_for_corridor(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        *,
        skip_satellite_green: bool = False,
        corridor_buffer_meters: Optional[float] = None,
    ) -> None:
        """Подгрузить/пересобрать граф по прямоугольнику start–end ± BUFFER."""
        s = self._app.settings
        if not s.use_dynamic_corridor_graph:
            return

        eff_buf_m: Optional[float] = None
        if corridor_buffer_meters is not None:
            eff_buf_m = float(corridor_buffer_meters)

        need_satellite = not skip_satellite_green

        bbox, raw_bounds, required_wgs84 = self._quantized_corridor_bbox_for_request(
            start, end, eff_buf_m
        )
        rmin_lon, rmin_lat, rmax_lon, rmax_lat = raw_bounds
        min_lat, max_lat, min_lon, max_lon = required_wgs84

        key_hash = corridor_bbox_cache_key(
            rmin_lon,
            rmin_lat,
            rmax_lon,
            rmax_lat,
            s,
            skip_satellite_green=skip_satellite_green,
        )
        gate = self._corridor_gate(key_hash)
        with gate:
            with self._graph_lock:
                covers = (
                    self._loaded
                    and self._graph is not None
                    and self._corridor_wgs84 is not None
                    and self._wgs84_corridor_contains(
                        self._corridor_wgs84, required_wgs84
                    )
                )
                if covers:
                    phase = self._graph.graph.get(
                        "bike_router_satellite_phase", _SAT_PHASE_STUB
                    )
                    if not need_satellite or phase == _SAT_PHASE_FULL:
                        return
                    logger.info(
                        "Коридор: доначитка спутниковой зелени без повторной загрузки OSM…"
                    )
                    self._app.elevation.init(
                        test_lat=start[0], test_lon=start[1]
                    )
                    self._upgrade_corridor_graph_satellite()
                    if self._graph is not None:
                        self._corridor_graph_cache.save(key_hash, self._graph)
                    return

                self._corridor_wgs84 = required_wgs84
                log_buf = (
                    eff_buf_m
                    if eff_buf_m is not None
                    else float(s.corridor_buffer_meters)
                )
                if s.corridor_buffer_meters > 0 or eff_buf_m is not None:
                    logger.info(
                        "Коридор маршрута (±%.0f м): lat [%.5f..%.5f] lon [%.5f..%.5f]",
                        log_buf,
                        min_lat,
                        max_lat,
                        min_lon,
                        max_lon,
                    )
                else:
                    logger.info(
                        "Коридор маршрута (±BUFFER=%.5f°): lat [%.5f..%.5f] lon [%.5f..%.5f]",
                        s.buffer,
                        min_lat,
                        max_lat,
                        min_lon,
                        max_lon,
                    )
                self._app.elevation.init(test_lat=start[0], test_lon=start[1])

                G_cached = self._corridor_graph_cache.load(key_hash)
                if G_cached is not None:
                    inc_corridor_disk_hit()
                    logger.info(
                        "Коридор: граф из дискового кэша (%s…)",
                        key_hash[:16],
                    )
                    self._set_active_graph(G_cached)
                    phase = G_cached.graph.get(
                        "bike_router_satellite_phase", _SAT_PHASE_STUB
                    )
                    self._graph_built_skip_satellite = phase == _SAT_PHASE_STUB
                    if not need_satellite or phase == _SAT_PHASE_FULL:
                        return
                    logger.info(
                        "Коридор: в кэше phase1 — доначитка зелени без повторной OSM…"
                    )
                    self._upgrade_corridor_graph_satellite()
                    if self._graph is not None:
                        self._corridor_graph_cache.save(key_hash, self._graph)
                    return

                self._build_graph_from_bbox_shape(
                    bbox, skip_satellite_green=skip_satellite_green
                )
                if self._graph is not None:
                    self._corridor_graph_cache.save(key_hash, self._graph)

    def warmup(self) -> None:
        """Предзагрузка графа и расчёт весов. Вызвать один раз.

        Приоритет области: ``AREA_POLYGON_WKT`` (произвольный полигон) →
        ``AREA_*`` (прямоугольник) → ``GRAPH_CORRIDOR_MODE`` (граф по первому запросу) →
        start/end + ``BUFFER`` из .env (CLI-демо).
        """
        s = self._app.settings
        logger.info("Warmup: загрузка графа...")

        if s.use_dynamic_corridor_graph:
            logger.info(
                "GRAPH_CORRIDOR_MODE: фиксированной зоны нет — граф и спутник "
                "строятся в прямоугольнике между точками POST ± BUFFER."
            )
            self._app.elevation.init(test_lat=s.start_lat, test_lon=s.start_lon)
            self._graph = None
            self._loaded = False
            self._bounds = None
            self._corridor_wgs84 = None
            self._graph_built_at_utc = None
            fp = routing_engine_cache_fingerprint()
            self._route_disk_cache.set_cache_context(0, 0, fp)
            if s.corridor_warmup_prebuild:
                try:
                    eff_buf: Optional[float] = (
                        float(s.corridor_buffer_meters)
                        if s.corridor_buffer_meters > 0
                        else None
                    )
                    self._ensure_graph_for_corridor(
                        s.start_coords,
                        s.end_coords,
                        skip_satellite_green=True,
                        corridor_buffer_meters=eff_buf,
                    )
                    logger.info(
                        "Warmup: коридор по START/END из .env предзагружен "
                        "(phase1, OSM/OSMnx cache; зелёная фаза — по запросу)."
                    )
                except Exception as exc:
                    logger.warning(
                        "Warmup: предзагрузка коридора не удалась (первый POST "
                        "всё равно построит граф): %s",
                        exc,
                    )
            return

        self._app.elevation.init(test_lat=s.start_lat, test_lon=s.start_lon)

        if s.has_area_polygon:
            from shapely import wkt as shapely_wkt
            from shapely.ops import unary_union

            raw = s.area_polygon_wkt_stripped
            try:
                geom = shapely_wkt.loads(raw)
            except Exception as e:
                raise ValueError(f"AREA_POLYGON_WKT: невалидный WKT: {e}") from e
            if geom.is_empty:
                raise ValueError("AREA_POLYGON_WKT: пустая геометрия")
            if geom.geom_type == "MultiPolygon":
                geom = unary_union(geom)
            if geom.geom_type != "Polygon":
                raise ValueError(
                    f"AREA_POLYGON_WKT: нужен Polygon или MultiPolygon, "
                    f"получено {geom.geom_type}"
                )
            if not geom.is_valid:
                geom = geom.buffer(0)
            bbox = geom
            b = geom.bounds
            logger.info(
                "Область: полигон WKT, bounds [%.4f..%.4f, %.4f..%.4f]",
                b[1], b[3], b[0], b[2],
            )
            if s.has_area_bbox:
                logger.warning(
                    "AREA_* заданы, но при активном AREA_POLYGON_WKT граф режется по полигону"
                )
        elif s.has_area_bbox:
            from shapely.geometry import box

            bbox = box(s.area_min_lon, s.area_min_lat,
                       s.area_max_lon, s.area_max_lat)
            logger.info(
                "Область покрытия AREA_*: [%.4f..%.4f, %.4f..%.4f]",
                s.area_min_lat, s.area_max_lat,
                s.area_min_lon, s.area_max_lon,
            )
        else:
            bbox = self._app.graph_builder.create_bbox(
                s.start_coords, s.end_coords, s.buffer
            )

        self._build_graph_from_bbox_shape(bbox)
        logger.info("Warmup завершён.")

    # ── Чистая функция маршрутизации ─────────────────────────────

    @staticmethod
    def _route_quality_hints(
        route_length_m: float,
        stairs: Dict[str, Any],
        fb: Dict[str, float],
    ) -> RouteQualityHints:
        """Доли по сумме длин рёбер (совпадает с подписью «длина маршрута» в UI)."""
        denom = route_length_m if route_length_m > 0 else 0.0
        na_frac = (fb["na_surface_length_m"] / denom) if denom else 0.0
        inf_surf_m = fb["na_surface_length_m"] + fb["unknown_surface_length_m"]
        inf_hw_m = fb["na_highway_length_m"] + fb["unknown_highway_length_m"]
        inf_surf = (inf_surf_m / denom) if denom else 0.0
        inf_hw = (inf_hw_m / denom) if denom else 0.0
        # Pydantic ge/le [0,1] — при погрешности float или пересечении категорий не даём 500
        na_frac = max(0.0, min(1.0, na_frac))
        inf_surf = max(0.0, min(1.0, inf_surf))
        inf_hw = max(0.0, min(1.0, inf_hw))
        warnings: List[str] = []
        if na_frac >= 0.22:
            warnings.append(
                "Значительная часть длины маршрута без тега surface в OSM — оценка "
                "покрытия и комфорта условная."
            )
        if inf_surf >= 0.30:
            warnings.append(
                "По длине маршрута много участков с коэффициентом покрытия по умолчанию "
                "(нет тега surface или неизвестное значение в справочнике)."
            )
        if inf_hw >= 0.12:
            warnings.append(
                "Значительная часть длины с коэффициентом highway по умолчанию — тип "
                "дороги в OSM не сопоставлен таблице штрафов."
            )
        scount = int(stairs["count"])
        slen = float(stairs["length"])
        if scount >= 4 or slen >= 80.0:
            warnings.append(
                "На маршруте много участков highway=steps — проверьте "
                "проходимость (для велосипеда часто нужно нести велосипед)."
            )
        return RouteQualityHints(
            warnings=warnings,
            na_surface_fraction=round(na_frac, 3),
            inferred_surface_fraction=round(inf_surf, 3),
            inferred_highway_fraction=round(inf_hw, 3),
        )

    def _build_route_response(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        profile_key: str,
        mode: str,
        route: RouteResult,
        cost_weight_key: str,
        variant_label: str = "",
        *,
        graph: Optional[nx.MultiDiGraph] = None,
    ) -> RouteResponse:
        """Собрать :class:`RouteResponse` из уже найденного :class:`RouteResult`."""
        profile = _PROFILE_MAP[profile_key]
        G = graph if graph is not None else self._graph
        if G is None:
            raise BikeRouterError("Граф не загружен.")
        router = self._app.router

        length = router.calculate_length(G, route)
        max_m = self._app.settings.max_route_km * 1000
        if length > max_m:
            raise RouteTooLongError(length, max_m)

        cost = router.calculate_cost(G, route, cost_weight_key)
        time_s = router.estimate_time(G, route, profile)
        gs = router.green_stats(G, route)
        es = router.elevation_stats(G, route)
        ep = router.elevation_profile(G, route)
        stairs = router.stairs_count(G, route)
        surf_counts, hw_counts = router.surface_stats(G, route)
        geometry = router.route_geometry(G, route)

        start_elev = self._app.elevation.get_elevation(start[0], start[1])
        raw_profile = router.elevation_profile_data(G, route)
        elev_points = [
            ElevationPoint(
                distance_m=round(d, 1),
                elevation_m=round(e + start_elev, 1),
            )
            for d, e in raw_profile
        ]

        fb = router.route_weight_fallback_metrics(G, route, profile)
        total_m = float(fb.get("total_length_m") or 0.0)
        na_len_frac = (
            float(fb.get("na_surface_length_m") or 0.0) / total_m
            if total_m > 0
            else 0.0
        )

        quality_hints = self._route_quality_hints(length, stairs, fb)
        if mode == "green":
            sw = self._app.green.consume_satellite_warning()
            if sw:
                quality_hints = RouteQualityHints(
                    warnings=list(quality_hints.warnings) + [sw],
                    na_surface_fraction=quality_hints.na_surface_fraction,
                    inferred_surface_fraction=quality_hints.inferred_surface_fraction,
                    inferred_highway_fraction=quality_hints.inferred_highway_fraction,
                )

        label = variant_label or _VARIANT_LABEL_DEFAULT.get(
            mode, mode
        )

        layers_dict = router.build_route_map_layers(G, route)
        map_layers = MapLayersGeoJSON(
            greenery=layers_dict["greenery"],
            stairs=layers_dict["stairs"],
            problematic=layers_dict["problematic"],
            na_surface=layers_dict["na_surface"],
        )

        return RouteResponse(
            profile=profile.key,
            mode=mode,
            variant_label=label,
            geometry=geometry,
            length_m=round(length, 1),
            time_s=round(time_s, 1),
            time_display=_fmt_time(time_s),
            cost=round(cost, 1),
            elevation=ElevationMetrics(
                climb_m=round(es["climb"], 1),
                descent_m=round(es["descent"], 1),
                max_gradient_pct=round(es["max_gradient_pct"], 1),
                avg_gradient_pct=round(es["avg_gradient_pct"], 1),
                max_above_start_m=round(ep["max_above"], 1),
                max_below_start_m=round(ep["max_below"], 1),
                end_diff_m=round(ep["end_diff"], 1),
            ),
            green=GreenMetrics(
                percent=round(gs["percent"], 1),
                avg_trees_pct=round(gs["avg_trees"], 1),
                avg_grass_pct=round(gs["avg_grass"], 1),
                categories={
                    str(k): int(v) for k, v in gs["categories"].items()
                },
            ),
            stairs=StairsInfo(
                count=stairs["count"],
                total_length_m=round(stairs["length"], 1),
            ),
            surfaces=SurfaceBreakdown(
                surfaces={str(k): int(v) for k, v in surf_counts.items()},
                highways={str(k): int(v) for k, v in hw_counts.items()},
                na_fraction=max(0.0, min(1.0, round(na_len_frac, 3))),
            ),
            elevation_profile=elev_points,
            map_layers=map_layers,
            quality_hints=quality_hints,
        )

    def _next_length_alternative(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        exclude: set,
        graph: Optional[nx.MultiDiGraph] = None,
    ) -> Optional[RouteResult]:
        """Следующий простой путь по ``length``, не из ``exclude``."""
        G = graph if graph is not None else self._graph
        if G is None:
            return None
        return RouteService.find_next_simple_path_by_length(
            G, start, end, exclude
        )

    def compute_route(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        profile_key: str,
        mode: str,
    ) -> RouteResponse:
        """Построить маршрут и вернуть JSON-совместимый результат.

        Args:
            start: ``(lat, lon)``
            end: ``(lat, lon)``
            profile_key: ``"cyclist"`` или ``"pedestrian"``
            mode: ``"full"``, ``"green"`` или ``"shortest"`` (минимум метров по OSM ``length``)

        Returns:
            :class:`RouteResponse` — единый объект со всеми метриками.

        Raises:
            BikeRouterError: граф не загружен.
            RouteNotFoundError: маршрут не найден.
            ValueError: неизвестный профиль или режим.
        """
        profile = _PROFILE_MAP.get(profile_key)
        if profile is None:
            raise ValueError(
                f"Неизвестный профиль: {profile_key!r}. "
                f"Допустимые: {list(_PROFILE_MAP)}"
            )
        if mode not in ("full", "green", "shortest"):
            raise ValueError(
                f"Неизвестный режим: {mode!r}. Допустимые: full, green, shortest"
            )

        with self._graph_lock:
            self._ensure_graph_for_corridor(start, end)
            if not self._loaded or self._graph is None:
                raise BikeRouterError("Граф не загружен. Вызовите warmup().")
            G = self._graph
            bounds = self._bounds
        assert bounds is not None
        self._validate_point_on_graph(G, bounds, start, "start")
        self._validate_point_on_graph(G, bounds, end, "end")

        router = self._app.router

        if mode == "shortest":
            path_w = "length"
            cost_w = f"weight_{profile.key}_full"
        else:
            path_w = cost_w = f"weight_{profile.key}_{mode}"

        route = router.find_route(G, start, end, path_w)
        return self._build_route_response(
            start,
            end,
            profile_key,
            mode,
            route,
            cost_w,
            "",
            graph=G,
        )

    @staticmethod
    def _reorder_routes_shortest_full_green(
        routes: List[RouteResponse],
    ) -> List[RouteResponse]:
        by = {r.mode: r for r in routes}
        out: List[RouteResponse] = []
        if "shortest" in by:
            out.append(by["shortest"])
        if "full" in by:
            out.append(by["full"])
        if "green" in by:
            out.append(by["green"])
        return out

    def _collect_alternative_routes(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        profile_key: str,
        *,
        include_green_route: bool,
        graph: Optional[nx.MultiDiGraph] = None,
    ) -> List[RouteResponse]:
        """Собрать варианты; порядок в ответе — кратчайший, полный, зелёный (если есть)."""
        profile = _PROFILE_MAP[profile_key]
        G = graph if graph is not None else self._graph
        if G is None:
            raise BikeRouterError("Граф не загружен.")
        router = self._app.router
        w_full = f"weight_{profile.key}_full"
        w_green = f"weight_{profile.key}_green"

        rf = router.find_route(G, start, end, w_full)
        rg = None
        if include_green_route:
            rg = router.find_route(G, start, end, w_green)

        exclude = {tuple(rf.edges)}
        if rg is not None:
            exclude.add(tuple(rg.edges))

        routes_out: List[RouteResponse] = []
        if include_green_route and rg is not None:
            routes_out.append(
                self._build_route_response(
                    start,
                    end,
                    profile_key,
                    "full",
                    rf,
                    w_full,
                    _VARIANT_LABEL_DEFAULT["full"],
                    graph=G,
                )
            )
            routes_out.append(
                self._build_route_response(
                    start,
                    end,
                    profile_key,
                    "green",
                    rg,
                    w_green,
                    _VARIANT_LABEL_DEFAULT["green"],
                    graph=G,
                )
            )
        else:
            routes_out.append(
                self._build_route_response(
                    start,
                    end,
                    profile_key,
                    "full",
                    rf,
                    w_full,
                    _VARIANT_LABEL_DEFAULT["full"],
                    graph=G,
                )
            )

        rs = router.find_route_safe(G, start, end, "length")
        if rs is not None and tuple(rs.edges) not in exclude:
            routes_out.append(
                self._build_route_response(
                    start,
                    end,
                    profile_key,
                    "shortest",
                    rs,
                    w_full,
                    _VARIANT_LABEL_DEFAULT["shortest"],
                    graph=G,
                )
            )
        else:
            alt = self._next_length_alternative(start, end, exclude, graph=G)
            if alt is not None:
                routes_out.append(
                    self._build_route_response(
                        start,
                        end,
                        profile_key,
                        "shortest",
                        alt,
                        w_full,
                        "Альтернативный путь",
                        graph=G,
                    )
                )

        return self._reorder_routes_shortest_full_green(routes_out)

    def _compute_alternatives_once(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        profile_key: str,
        *,
        include_green_route: bool,
        skip_satellite_green: bool,
        corridor_buffer_meters: Optional[float],
    ) -> List[RouteResponse]:
        with self._graph_lock:
            self._ensure_graph_for_corridor(
                start,
                end,
                skip_satellite_green=skip_satellite_green,
                corridor_buffer_meters=corridor_buffer_meters,
            )
            if not self._loaded or self._graph is None:
                raise BikeRouterError("Граф не загружен. Вызовите warmup().")
            G = self._graph
            bounds = self._bounds
        assert bounds is not None
        self._validate_point_on_graph(G, bounds, start, "start")
        self._validate_point_on_graph(G, bounds, end, "end")
        return self._collect_alternative_routes(
            start,
            end,
            profile_key,
            include_green_route=include_green_route,
            graph=G,
        )

    def compute_green_route_addon(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        profile_key: str,
    ) -> RouteResponse:
        """Третий вариант (green): при необходимости пересобирает граф со спутниковой зеленью."""
        profile = _PROFILE_MAP.get(profile_key)
        if profile is None:
            raise ValueError(f"Неизвестный профиль: {profile_key!r}")
        with self._graph_lock:
            self._ensure_graph_for_corridor(
                start, end, skip_satellite_green=False
            )
            if not self._loaded or self._graph is None:
                raise BikeRouterError("Граф не загружен.")
            G = self._graph
            bounds = self._bounds
        assert bounds is not None
        self._validate_point_on_graph(G, bounds, start, "start")
        self._validate_point_on_graph(G, bounds, end, "end")
        router = self._app.router
        w_green = f"weight_{profile.key}_green"
        rg = router.find_route(G, start, end, w_green)
        return self._build_route_response(
            start,
            end,
            profile_key,
            "green",
            rg,
            w_green,
            _VARIANT_LABEL_DEFAULT["green"],
            graph=G,
        )

    def compute_alternatives(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        profile_key: str,
        *,
        green_enabled: bool = True,
    ) -> AlternativesResponse:
        """Несколько вариантов; green_enabled=False — 2 маршрута без спутника."""
        profile = _PROFILE_MAP.get(profile_key)
        if profile is None:
            raise ValueError(
                f"Неизвестный профиль: {profile_key!r}. "
                f"Допустимые: {list(_PROFILE_MAP)}"
            )

        skip_sat = not green_enabled
        include_green = green_enabled

        cached = self._route_disk_cache.get(
            start, end, profile_key, green_enabled=green_enabled
        )
        if cached is not None:
            try:
                inc_route_disk_hit()
                logger.info(
                    "Alternatives: hit дискового кэша (профиль %s green=%s)",
                    profile_key,
                    green_enabled,
                )
                return AlternativesResponse.model_validate(cached)
            except Exception:
                logger.debug("Route disk cache invalid, recalculating")

        with self._graph_lock:
            cached = self._route_disk_cache.get(
                start, end, profile_key, green_enabled=green_enabled
            )
            if cached is not None:
                try:
                    inc_route_disk_hit()
                    return AlternativesResponse.model_validate(cached)
                except Exception:
                    pass

            inc_route_disk_miss()
            logger.info(
                "Alternatives: расчёт профиль=%s green_enabled=%s",
                profile_key,
                green_enabled,
            )

            s = self._app.settings
            routes_out: List[RouteResponse]

            if not s.use_dynamic_corridor_graph or s.corridor_buffer_meters <= 0:
                routes_out = self._compute_alternatives_once(
                    start,
                    end,
                    profile_key,
                    include_green_route=include_green,
                    skip_satellite_green=skip_sat,
                    corridor_buffer_meters=None,
                )
            else:
                base = float(s.corridor_buffer_meters)
                extra = 0.0
                attempt = 0
                step = max(1.0, float(s.auto_expand_step_meters))
                max_extra = max(0.0, float(s.auto_expand_max_meters))
                max_attempts = max(1, int(s.auto_expand_max_attempts))
                while True:
                    attempt += 1
                    eff = base + extra
                    try:
                        routes_out = self._compute_alternatives_once(
                            start,
                            end,
                            profile_key,
                            include_green_route=include_green,
                            skip_satellite_green=skip_sat,
                            corridor_buffer_meters=eff,
                        )
                        logger.info(
                            "auto_expand_corridor attempt=%d corridor_m=%.0f "
                            "result=ok routes=%d",
                            attempt,
                            eff,
                            len(routes_out),
                        )
                        break
                    except RouteNotFoundError as e:
                        logger.warning(
                            "auto_expand_corridor attempt=%d corridor_m=%.0f "
                            "result=no_path code=%s",
                            attempt,
                            eff,
                            e.code,
                        )
                        at_attempts = attempt >= max_attempts
                        at_extra_cap = extra + step > max_extra + 1e-6
                        if at_attempts or at_extra_cap:
                            logger.warning(
                                "auto_expand_corridor exhausted attempt=%d "
                                "corridor_m=%.0f max_extra_m=%.0f max_attempts=%d",
                                attempt,
                                eff,
                                max_extra,
                                max_attempts,
                            )
                            _raise_route_not_found_after_corridor_expand(
                                e,
                                eff=eff,
                                base=base,
                                max_extra=max_extra,
                                step=step,
                                attempt=attempt,
                                max_attempts=max_attempts,
                            )
                        extra += step

            out = AlternativesResponse(routes=routes_out)
            try:
                self._route_disk_cache.put(
                    start,
                    end,
                    profile_key,
                    [r.model_dump(mode="json") for r in out.routes],
                    green_enabled=green_enabled,
                )
            except Exception as exc:
                logger.debug("Route cache store skipped: %s", exc)
            return out

    def compute_alternatives_phase1_two_routes(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        profile_key: str,
    ) -> List[RouteResponse]:
        """Первые два маршрута (кратчайший + энергия) без спутниковой зелени; с авторасширением.

        Третий вариант («с учётом озеленения») считается в ``compute_green_route_addon`` на графе со спутником.
        """
        if profile_key not in _PROFILE_MAP:
            raise ValueError(f"Неизвестный профиль: {profile_key!r}")
        s = self._app.settings
        cached = self._route_disk_cache.get(
            start, end, profile_key, green_enabled=False
        )
        if cached is not None:
            try:
                alt = AlternativesResponse.model_validate(cached)
                by_mode = {r.mode: r for r in alt.routes}
                if "shortest" in by_mode and "full" in by_mode:
                    inc_route_disk_hit()
                    logger.info(
                        "alternatives_phase1: hit route disk cache (green_enabled=false)"
                    )
                    return [by_mode["shortest"], by_mode["full"]]
            except Exception:
                logger.debug("alternatives_phase1: route cache invalid, recalculating")

        inc_route_disk_miss()
        if not s.use_dynamic_corridor_graph or s.corridor_buffer_meters <= 0:
            return self._compute_alternatives_once(
                start,
                end,
                profile_key,
                include_green_route=False,
                skip_satellite_green=True,
                corridor_buffer_meters=None,
            )
        base = float(s.corridor_buffer_meters)
        extra = 0.0
        attempt = 0
        step = max(1.0, float(s.auto_expand_step_meters))
        max_extra = max(0.0, float(s.auto_expand_max_meters))
        max_attempts = max(1, int(s.auto_expand_max_attempts))
        while True:
            attempt += 1
            eff = base + extra
            try:
                routes = self._compute_alternatives_once(
                    start,
                    end,
                    profile_key,
                    include_green_route=False,
                    skip_satellite_green=True,
                    corridor_buffer_meters=eff,
                )
                logger.info(
                    "alternatives_phase1 attempt=%d corridor_m=%.0f routes=%d",
                    attempt,
                    eff,
                    len(routes),
                )
                return routes
            except RouteNotFoundError as e:
                logger.warning(
                    "alternatives_phase1 attempt=%d corridor_m=%.0f no_path",
                    attempt,
                    eff,
                )
                at_attempts = attempt >= max_attempts
                at_extra_cap = extra + step > max_extra + 1e-6
                if at_attempts or at_extra_cap:
                    logger.warning(
                        "alternatives_phase1 exhausted attempt=%d "
                        "corridor_m=%.0f max_extra_m=%.0f max_attempts=%d",
                        attempt,
                        eff,
                        max_extra,
                        max_attempts,
                    )
                    _raise_route_not_found_after_corridor_expand(
                        e,
                        eff=eff,
                        base=base,
                        max_extra=max_extra,
                        step=step,
                        attempt=attempt,
                        max_attempts=max_attempts,
                    )
                extra += step
