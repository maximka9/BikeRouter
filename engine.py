"""Движок маршрутизации: чистая функция compute_route.

При фиксированной зоне (AREA / полигон) граф загружается в ``warmup()`` один раз.
В режиме ``GRAPH_CORRIDOR_MODE`` граф и спутниковая зелень строятся по прямоугольнику
между точками запроса ± ``BUFFER`` (пересборка, если новая пара точек вне коридора).
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import asdict, replace
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import osmnx as ox

from .app import Application
from .config import (
    PROFILES,
    TIME_SLOTS,
    TURN_PENALTY_BASE,
    RoutingPreferenceProfile,
    Settings,
    routing_engine_cache_fingerprint,
    routing_preference_profile,
    time_slot_key_for_hour,
)
from .exceptions import (
    BikeRouterError,
    PointOutsideZoneError,
    RouteNotFoundError,
    RouteTooLongError,
)
from .models import (
    AlternativesResponse,
    CombinedCostBreakdown,
    ElevationMetrics,
    ElevationPoint,
    GreenMetrics,
    HeatStressMetrics,
    MapLayersGeoJSON,
    RouteQualityHints,
    RouteResponse,
    RoutingContextMeta,
    StairsInfo,
    SurfaceBreakdown,
    WeatherRouteContext,
    WeatherSnapshotValues,
)
from .services.heat import heat_context_multiplier
from .services.area_graph_cache import (
    graph_base_path,
    load_graphml_path,
    load_meta,
    meta_matches_current,
    meta_path,
    precache_area_is_complete,
    precache_corridor_fits_arena,
    select_precache_graph_path,
    select_warmup_precache_graph_path,
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
    _first_value,
    sanitize_multidigraph_routing_weights,
)
from .services.weather import (
    WeatherSnapshot,
    WeatherWeightParams,
    resolve_weather_for_route,
    snapshot_from_manual,
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

# Порядок вариантов в ответе API / единый пользовательский сценарий
_UNIFIED_ROUTE_ORDER = (
    "shortest",
    "full",
    "green",
    "heat",
    "stress",
    "heat_stress",
)

# Минимизация только физической части веса с погодными множителями (режим full/green).
_WEATHER_PHYSICAL_ROUTING_PREF = RoutingPreferenceProfile(
    key="weather_physical",
    label="",
    alpha=1.0,
    beta=0.0,
    gamma=0.0,
    delta=0.0,
)


def _engine_weather_mode(request_mode: Optional[str], use_live_weather: bool) -> str:
    if use_live_weather:
        return "auto"
    m = (request_mode or "none").strip().lower()
    if m in ("fixed-snapshot", "fixed_snapshot"):
        return "fixed-snapshot"
    return m if m else "none"


def _weather_snapshot_to_values(snap: WeatherSnapshot) -> WeatherSnapshotValues:
    return WeatherSnapshotValues(
        temperature_c=float(snap.temperature_c),
        apparent_temperature_c=snap.apparent_temperature_c,
        precipitation_mm=float(snap.precipitation_mm),
        precipitation_probability=snap.precipitation_probability,
        wind_speed_ms=float(snap.wind_speed_ms),
        wind_gusts_ms=snap.wind_gusts_ms,
        wind_direction_deg=getattr(snap, "wind_direction_deg", None),
        cloud_cover_pct=float(snap.cloud_cover_pct),
        humidity_pct=float(snap.humidity_pct),
        shortwave_radiation_wm2=snap.shortwave_radiation_wm2,
        snowfall_cm_h=float(getattr(snap, "snowfall_cm_h", 0.0) or 0.0),
        snow_depth_m=float(getattr(snap, "snow_depth_m", 0.0) or 0.0),
        weather_code=getattr(snap, "weather_code", None),
    )


def _weather_summary_ru(snap: WeatherSnapshot, wp: WeatherWeightParams) -> str:
    if not wp.enabled:
        return "Погода не учитывалась при выборе путей."
    m = wp.mults
    lines: List[str] = []
    if m.physical > 1.02:
        lines.append(f"рельеф и покрытие дороже на {_pct_deviation(m.physical)}%")
    elif m.physical < 0.98:
        lines.append(f"рельеф и покрытие дешевле на {_pct_deviation(m.physical)}%")
    if m.heat > 1.02:
        lines.append(f"тепловая нагрузка выше обычной для слота на {_pct_deviation(m.heat)}%")
    elif m.heat < 0.98:
        lines.append(f"тепловая нагрузка ниже обычной для слота на {_pct_deviation(m.heat)}%")
    if m.green > 1.02:
        lines.append(f"влияние озеленения на комфорт сильнее на {_pct_deviation(m.green)}%")
    elif m.green < 0.98:
        lines.append(f"влияние озеленения на комфорт слабее на {_pct_deviation(m.green)}%")
    if m.stress > 1.02:
        lines.append(f"транспортный стресс выше на {_pct_deviation(m.stress)}%")
    elif m.stress < 0.98:
        lines.append(f"транспортный стресс ниже на {_pct_deviation(m.stress)}%")
    if m.surface > 1.02:
        lines.append(f"учёт покрытия строже на {_pct_deviation(m.surface)}%")
    if float(getattr(snap, "snowfall_cm_h", 0.0) or 0.0) >= 0.3:
        lines.append("идёт снег — скользкие и слабые покрытия дороже")
    elif float(getattr(snap, "snow_depth_m", 0.0) or 0.0) >= 0.05:
        lines.append("лежит снег — маршрут по плохим покрытиям и лестницам дороже")
    if snap.precipitation_mm > 0.3 and not lines:
        lines.append("осадки усиливают штрафы за подъёмы и скользкое покрытие")
    if not lines:
        return "Погодные поправки близки к нейтральным."
    return "\n".join(lines)


def _build_weather_route_context(
    *,
    request_mode: str,
    use_live_weather: bool,
    weather_time: Optional[str],
    departure_time: Optional[str],
    snap: WeatherSnapshot,
    source: str,
    wp: WeatherWeightParams,
) -> WeatherRouteContext:
    mults: Dict[str, float] = {}
    if wp.enabled:
        mults = {k: round(float(v), 4) for k, v in asdict(wp.mults).items()}
    wt = weather_time or departure_time
    hm: Dict[str, float] = {}
    hc = False
    if wp.enabled and getattr(wp, "heat_continuous", False):
        hc = True
        hm = {
            "tree_shade_bonus": float(getattr(wp, "tree_shade_bonus", 1.0)),
            "open_sky_penalty": float(getattr(wp, "open_sky_penalty", 1.0)),
            "building_shade_bonus": float(getattr(wp, "building_shade_bonus", 1.0)),
            "covered_bonus": float(getattr(wp, "covered_bonus", 1.0)),
            "wind_open_penalty": float(getattr(wp, "wind_open_penalty", 1.0)),
            "wet_surface_penalty": float(getattr(wp, "wet_surface_penalty", 1.0)),
        }
        ns = getattr(wp, "normalized_signals", None) or {}
        for k, v in ns.items():
            try:
                hm[f"norm_{k}"] = float(v)
            except (TypeError, ValueError):
                pass
    return WeatherRouteContext(
        enabled=bool(wp.enabled),
        mode=request_mode,
        use_live_weather=use_live_weather,
        weather_time=wt,
        source=source,
        snapshot=_weather_snapshot_to_values(snap),
        multipliers=mults,
        summary_ru=_weather_summary_ru(snap, wp),
        heat_continuous=hc,
        heat_microclimate=hm,
        routing_season=str(getattr(wp, "routing_season", "") or ""),
        routing_season_calendar=str(
            getattr(wp, "routing_season_calendar", "") or ""
        ),
        routing_season_source=str(getattr(wp, "routing_season_source", "") or ""),
        season_green_route_mult=float(getattr(wp, "season_green_route_mult", 1.0)),
        season_tree_heat_route_mult=float(
            getattr(wp, "season_tree_heat_route_mult", 1.0)
        ),
        snow_model_strength=float(getattr(wp, "snow_model_strength", 0.0)),
        snow_export_phys_amp=float(getattr(wp, "snow_export_phys_amp", 1.0)),
        snow_export_stress_amp=float(getattr(wp, "snow_export_stress_amp", 1.0)),
        snow_export_surface_amp=float(getattr(wp, "snow_export_surface_amp", 1.0)),
        wind_direction_available=bool(
            getattr(wp, "wind_direction_available", False)
        ),
    )


def _iso_utc_z(dt: datetime) -> str:
    """UTC ISO-8601 с суффиксом Z (без смещения)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _log_route_weather_line(
    *,
    built_at_utc: datetime,
    start: Tuple[float, float],
    end: Tuple[float, float],
    snap: WeatherSnapshot,
    source: str,
    wp: WeatherWeightParams,
    weather_time_effective: Optional[str],
) -> None:
    """Один компактный блок в лог на запрос альтернатив (см. ТЗ route_weather)."""
    lat = (float(start[0]) + float(end[0])) * 0.5
    lon = (float(start[1]) + float(end[1])) * 0.5
    sw = snap.shortwave_radiation_wm2
    sw_txt = f"{sw:.2f}" if sw is not None else "—"
    m = wp.mults
    mult = (
        f"physical:{m.physical:.3f},heat:{m.heat:.3f},green:{m.green:.3f},"
        f"stress:{m.stress:.3f},surface:{m.surface:.3f}"
    )
    regime = getattr(wp, "regime", "—") if wp else "—"
    wd = getattr(snap, "wind_direction_deg", None)
    wd_txt = f"{float(wd):.1f}°" if wd is not None else "—"
    wdir_on = bool(getattr(wp, "wind_direction_available", False))
    logger.info(
        "route_weather: built_at=%s weather_time=%s source=%s lat=%.5f lon=%.5f "
        "temp_c=%.2f precip_mm=%.4f wind_ms=%.2f wind_dir=%s wind_dir_aware=%s "
        "cloud_pct=%.1f humidity_pct=%.1f sw_wm2=%s mult={%s} regime=%s weather_enabled=%s",
        _iso_utc_z(built_at_utc),
        weather_time_effective or "—",
        source or "none",
        lat,
        lon,
        snap.temperature_c,
        snap.precipitation_mm,
        snap.wind_speed_ms,
        wd_txt,
        wdir_on,
        snap.cloud_cover_pct,
        snap.humidity_pct,
        sw_txt,
        mult,
        regime,
        wp.enabled,
    )


def _resolve_route_weather(
    start: Tuple[float, float],
    end: Tuple[float, float],
    *,
    request_mode: str,
    use_live_weather: bool,
    weather_time: Optional[str],
    departure_time: Optional[str],
    temperature_c: Optional[float],
    precipitation_mm: Optional[float],
    wind_speed_ms: Optional[float],
    cloud_cover_pct: Optional[float] = None,
    humidity_pct: Optional[float] = None,
    wind_gusts_ms: Optional[float] = None,
    wind_direction_deg: Optional[float] = None,
    shortwave_radiation_wm2: Optional[float] = None,
    snowfall_cm_h: Optional[float] = None,
    snow_depth_m: Optional[float] = None,
    weather_code: Optional[int] = None,
    thermal_scales: Optional[Dict[str, float]] = None,
    settings: Any = None,
) -> Tuple[WeatherSnapshot, str, WeatherWeightParams]:
    lat = (float(start[0]) + float(end[0])) * 0.5
    lon = (float(start[1]) + float(end[1])) * 0.5
    wm = _engine_weather_mode(request_mode, use_live_weather)
    manual: Optional[WeatherSnapshot] = None
    if wm in ("manual", "fixed-snapshot"):
        manual = snapshot_from_manual(
            temperature_c=temperature_c,
            precipitation_mm=precipitation_mm,
            wind_speed_ms=wind_speed_ms,
            wind_gusts_ms=wind_gusts_ms,
            cloud_cover_pct=cloud_cover_pct,
            humidity_pct=humidity_pct,
            shortwave_radiation_wm2=shortwave_radiation_wm2,
            snowfall_cm_h=snowfall_cm_h,
            snow_depth_m=snow_depth_m,
            weather_code=weather_code,
            wind_direction_deg=wind_direction_deg,
        )
    return resolve_weather_for_route(
        lat=lat,
        lon=lon,
        weather_mode=wm,
        use_live_weather=use_live_weather,
        weather_time_iso=weather_time,
        departure_time=departure_time,
        manual=manual,
        thermal_scales=thermal_scales,
        settings=settings,
    )


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    if m >= 60:
        h, m = divmod(m, 60)
        return f"{h} ч {m} мин"
    return f"{m} мин {s} сек"


def _resolve_time_slot_key(
    departure_time: Optional[str],
    time_slot_override: Optional[str],
) -> str:
    """Слот тепловой модели: явный time_slot или час из departure_time (ISO), иначе полдень."""
    valid = {s.key for s in TIME_SLOTS}
    if time_slot_override:
        k = str(time_slot_override).strip().lower()
        if k in valid:
            return k
    if departure_time:
        raw = str(departure_time).strip()
        try:
            iso = raw.replace("Z", "+00:00")
            dt = datetime.fromisoformat(iso)
            return time_slot_key_for_hour(dt.hour)
        except ValueError:
            pass
    return "noon"


def _infer_season_from_month(month_1_12: int) -> str:
    """Лето — июнь–август; иначе весна/осень (как в SeasonEnum)."""
    if month_1_12 in (6, 7, 8):
        return "summer"
    return "spring_autumn"


def _pct_deviation(mult: float) -> int:
    """Отклонение множителя от 1.0 в процентах (округление)."""
    return int(round(abs(float(mult) - 1.0) * 100.0))


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


def _raise_route_not_found_after_schedule(
    last_exc: RouteNotFoundError,
    *,
    schedule: Sequence[float],
    last_eff: float,
    attempts: int,
) -> None:
    """После перебора CORRIDOR_BUFFER_EXPAND_SCHEDULE — NO_PATH."""
    sched_s = ", ".join(f"{x:.0f}" for x in schedule)
    msg = (
        f"Маршрут не найден после {attempts} попыток "
        f"(буферы коридора м: {sched_s}; последний ±{last_eff:.0f} м). "
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

    Сборка/обновление тяжёлого графа коридора выполняется вне ``_graph_lock``; под lock —
    только проверки и атомарная установка ``_graph`` (см. ``_ensure_graph_for_corridor``).
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
        # Короткие критические секции: снимок/замена графа; тяжёлый OSM/спутник — вне lock.
        self._graph_lock = threading.Lock()
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

    def _build_graph_bundle_from_bbox(
        self, bbox: Any, *, skip_satellite_green: bool = False
    ) -> Tuple[nx.MultiDiGraph, bool]:
        """Собрать граф по bbox без установки в движок (для смены под коротким lock)."""
        G = self._app.graph_builder.load(bbox)
        edges = self._app.graph_builder.to_geodataframe(G)
        edges = self._app.graph_builder.calculate_weights(
            edges, skip_satellite_green=skip_satellite_green
        )
        G = self._app.graph_builder.apply_weights(G, edges)
        G.graph["bike_router_satellite_phase"] = (
            _SAT_PHASE_STUB if skip_satellite_green else _SAT_PHASE_FULL
        )
        return G, skip_satellite_green

    def _build_graph_from_bbox_shape(
        self, bbox: Any, *, skip_satellite_green: bool = False
    ) -> None:
        """Загрузить OSM, веса (спутник опционально), применить к графу (фиксированная зона / warmup)."""
        G, sk = self._build_graph_bundle_from_bbox(
            bbox, skip_satellite_green=skip_satellite_green
        )
        self._set_active_graph(G)
        self._graph_built_skip_satellite = sk

    def _graph_satisfies_corridor_request(
        self,
        required_wgs84: Tuple[float, float, float, float],
        need_satellite: bool,
    ) -> bool:
        """Проверка под активным lock: коридор покрывает запрос и фаза спутника достаточна."""
        if not self._loaded or self._graph is None or self._corridor_wgs84 is None:
            return False
        if not self._wgs84_corridor_contains(
            self._corridor_wgs84, required_wgs84
        ):
            return False
        if not need_satellite:
            return True
        ph = self._graph.graph.get(
            "bike_router_satellite_phase", _SAT_PHASE_STUB
        )
        return ph == _SAT_PHASE_FULL

    def _upgrade_satellite_on_graph_copy(self, G: nx.MultiDiGraph) -> nx.MultiDiGraph:
        """Доначитка зелени на копии графа (исходный объект не мутируем — безопасно вне lock)."""
        G_work = G.copy()
        edges = self._app.graph_builder.to_geodataframe(G_work)
        edges = self._app.graph_builder.upgrade_edges_satellite_weights(edges)
        G2 = self._app.graph_builder.apply_weights(G_work, edges)
        G2.graph["bike_router_satellite_phase"] = _SAT_PHASE_FULL
        return G2

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

    def _try_activate_area_precache(
        self,
        required_wgs84: Tuple[float, float, float, float],
        need_satellite: bool,
    ) -> bool:
        """Подставить граф с диска ``area_precache`` без Overpass, если коридор ⊂ арена."""
        s = self._app.settings
        if not s.precache_area_enabled or not s.has_precache_area_polygon:
            return False
        if not s.use_dynamic_corridor_graph:
            return False
        meta = load_meta(s)
        if not meta:
            logger.info(
                "area_precache: нет meta.json под BIKE_ROUTER_BASE_DIR — live коридор "
                "(офлайн: python -m bike_router.tools.precache_area)"
            )
            return False
        if not meta_matches_current(meta, s):
            logger.info(
                "area_precache: статический fingerprint в meta.json не совпадает "
                "(полигон/OSM/zoom/TMS/буфер/зелень; schema area_precache_v3) — "
                "пересоберите precache_area или выровняйте .env"
            )
            return False
        if not precache_area_is_complete(s):
            logger.info(
                "area_precache: кэш полигона не готов семантически (meta) — live коридор"
            )
            return False
        if not precache_corridor_fits_arena(s, required_wgs84):
            logger.info(
                "area_precache: bbox коридора запроса (с буфером) не целиком внутри "
                "PRECACHE_AREA_POLYGON_WKT — live коридор"
            )
            return False
        path = select_precache_graph_path(s, need_satellite)
        if path is None:
            logger.info(
                "area_precache: нет подходящего graphml (need_satellite=%s) — "
                "проверьте graph_base/graph_green и PRECACHE_AREA_USE_GREEN_GRAPH",
                need_satellite,
            )
            return False
        G = load_graphml_path(path)
        if G is None:
            return False
        aw = tuple(meta.get("arena_wgs84", ()))
        if len(aw) != 4:
            return False
        ph = G.graph.get("bike_router_satellite_phase", _SAT_PHASE_STUB)
        skip_res = ph == _SAT_PHASE_STUB
        with self._graph_lock:
            if self._graph_satisfies_corridor_request(
                required_wgs84, need_satellite
            ):
                return True
            self._corridor_wgs84 = (
                float(aw[0]),
                float(aw[1]),
                float(aw[2]),
                float(aw[3]),
            )
            self._set_active_graph(G)
            self._graph_built_skip_satellite = skip_res
        logger.info(
            "Коридор: граф из area_precache (name=%s, file=%s)",
            s.precache_area_name,
            path.name,
        )
        return True

    def _warmup_preload_area_precache(self) -> bool:
        """Загрузить граф арены с диска в память при старте (без Overpass)."""
        s = self._app.settings
        if not s.precache_area_enabled or not s.has_precache_area_polygon:
            return False
        if not s.use_dynamic_corridor_graph:
            return False
        meta = load_meta(s)
        if not meta:
            d = graph_base_path(s).parent
            extra = ""
            if d.is_dir() and not meta_path(s).is_file():
                try:
                    n_files = sum(1 for _ in d.iterdir())
                    extra = (
                        f" Папка на диске есть ({n_files} объектов), но без meta.json "
                        "кэш не действителен — сборка не завершилась или файлы удалены. "
                        "Удалите эту папку и заново выполните "
                        "`python -m bike_router.tools.precache_area` "
                        "(в Docker: `docker compose run --rm bike-router "
                        "python -m bike_router.tools.precache_area`)."
                    )
                except OSError:
                    pass
            logger.info(
                "Warmup: area_precache ещё не собран (нет meta.json в %s).%s",
                d,
                extra,
            )
            return False
        if not meta_matches_current(meta, s):
            logger.info(
                "Warmup: area_precache на диске не подходит: статический fingerprint "
                "или schema в meta.json не совпадают. Удалите каталог %s или пересоберите кэш.",
                graph_base_path(s).parent,
            )
            return False
        if not precache_area_is_complete(s):
            logger.info(
                "Warmup: area_precache не готов по meta (green_quality_state) — пропуск предзагрузки"
            )
            return False
        path = select_warmup_precache_graph_path(s)
        if path is None or not path.is_file():
            logger.info(
                "Warmup: в %s нет graph_base.graphml / graph_green.graphml",
                graph_base_path(s).parent,
            )
            return False
        G = load_graphml_path(path)
        if G is None:
            return False
        aw = tuple(meta.get("arena_wgs84", ()))
        if len(aw) != 4:
            logger.warning("Warmup: area_precache meta.json без arena_wgs84")
            return False
        ph = G.graph.get("bike_router_satellite_phase", _SAT_PHASE_STUB)
        skip_res = ph == _SAT_PHASE_STUB
        with self._graph_lock:
            self._corridor_wgs84 = (
                float(aw[0]),
                float(aw[1]),
                float(aw[2]),
                float(aw[3]),
            )
            self._set_active_graph(G)
            self._graph_built_skip_satellite = skip_res
        logger.info(
            "Warmup: граф area_precache предзагружен в память (%s, арена lat [%.4f..%.4f])",
            path.name,
            aw[0],
            aw[1],
        )
        return True

    def _ensure_graph_for_corridor(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        *,
        skip_satellite_green: bool = False,
        corridor_buffer_meters: Optional[float] = None,
    ) -> None:
        """Подгрузить/пересобрать граф по прямоугольнику start–end ± BUFFER.

        Долгая работа (Overpass, веса, тайлы) выполняется **вне** ``_graph_lock``;
        под lock только проверки и атомарная установка ``_graph`` / ``_corridor_wgs84``.
        """
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

        with self._graph_lock:
            if self._graph_satisfies_corridor_request(
                required_wgs84, need_satellite
            ):
                return

        if self._try_activate_area_precache(required_wgs84, need_satellite):
            return

        gate = self._corridor_gate(key_hash)
        with gate:
            with self._graph_lock:
                if self._graph_satisfies_corridor_request(
                    required_wgs84, need_satellite
                ):
                    return

            covers = False
            phase = _SAT_PHASE_STUB
            g0: Optional[nx.MultiDiGraph] = None
            with self._graph_lock:
                g0 = self._graph
                cw = self._corridor_wgs84
                loaded = self._loaded
                covers = bool(
                    loaded
                    and g0 is not None
                    and cw is not None
                    and self._wgs84_corridor_contains(cw, required_wgs84)
                )
                if g0 is not None:
                    phase = g0.graph.get(
                        "bike_router_satellite_phase", _SAT_PHASE_STUB
                    )

            G_result: Optional[nx.MultiDiGraph] = None
            skip_res: Optional[bool] = None

            if covers and g0 is not None and need_satellite and phase == _SAT_PHASE_STUB:
                logger.info(
                    "Коридор: доначитка спутниковой зелени без повторной загрузки OSM…"
                )
                self._app.elevation.init(test_lat=start[0], test_lon=start[1])
                G_result = self._upgrade_satellite_on_graph_copy(g0)
                skip_res = False
            elif covers:
                with self._graph_lock:
                    if self._graph_satisfies_corridor_request(
                        required_wgs84, need_satellite
                    ):
                        return
                return
            else:
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
                    ph = G_cached.graph.get(
                        "bike_router_satellite_phase", _SAT_PHASE_STUB
                    )
                    if not need_satellite or ph == _SAT_PHASE_FULL:
                        G_result = G_cached
                        skip_res = ph == _SAT_PHASE_STUB
                    else:
                        logger.info(
                            "Коридор: в кэше phase1 — доначитка зелени без повторной OSM…"
                        )
                        G_result = self._upgrade_satellite_on_graph_copy(G_cached)
                        skip_res = False

                if G_result is None:
                    G_result, skip_res = self._build_graph_bundle_from_bbox(
                        bbox, skip_satellite_green=skip_satellite_green
                    )

            if G_result is None or skip_res is None:
                raise BikeRouterError(
                    "Внутренняя ошибка: не удалось подготовить граф коридора."
                )

            with self._graph_lock:
                if self._graph_satisfies_corridor_request(
                    required_wgs84, need_satellite
                ):
                    return
                if not covers:
                    self._corridor_wgs84 = required_wgs84
                self._set_active_graph(G_result)
                self._graph_built_skip_satellite = skip_res
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

            # Предзагрузка area precache в память — без второго запроса к Overpass на первом POST.
            if s.precache_area_enabled and s.has_precache_area_polygon:
                try:
                    self._warmup_preload_area_precache()
                except Exception as exc:
                    logger.warning(
                        "Warmup: предзагрузка area_precache в память не удалась: %s",
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
        cost_override: Optional[float] = None,
        heat_stress_metrics: Optional[HeatStressMetrics] = None,
        routing_context: Optional[RoutingContextMeta] = None,
        weather: Optional[WeatherRouteContext] = None,
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

        if cost_override is not None:
            cost = float(cost_override)
        else:
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

        hm = heat_stress_metrics
        rc = routing_context
        return RouteResponse(
            profile=profile.key,
            mode=mode,
            variant_label=label,
            geometry=geometry,
            length_m=round(length, 1),
            time_s=round(time_s, 1),
            time_display=_fmt_time(time_s),
            cost=round(cost, 1),
            routing_context=rc,
            weather=weather,
            route_built_at_utc=datetime.now(timezone.utc).isoformat(),
            heat_cost_total=float(hm.total_heat_cost) if hm else 0.0,
            stress_cost_total=float(hm.stress_cost_total) if hm else 0.0,
            exposed_length_m=float(hm.exposed_high_length_m) if hm else 0.0,
            building_shade_share=float(hm.building_shade_share) if hm else 0.0,
            route_open_sky_share=float(hm.route_open_sky_share) if hm else 0.0,
            route_building_shade_share=float(hm.route_building_shade_share)
            if hm
            else 0.0,
            route_covered_share=float(hm.route_covered_share) if hm else 0.0,
            route_bad_wet_surface_share=float(hm.route_bad_wet_surface_share)
            if hm
            else 0.0,
            route_winter_harsh_surface_share=float(hm.route_winter_harsh_surface_share)
            if hm
            else 0.0,
            vegetation_shade_share=float(hm.vegetation_shade_share) if hm else 0.0,
            stressful_intersections_count=int(hm.stressful_intersections_count)
            if hm
            else 0,
            high_stress_segments_count=int(hm.high_stress_segments_count)
            if hm
            else 0,
            turn_count_analytics=int(hm.turn_count) if hm else 0,
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
            heat_stress=heat_stress_metrics,
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

    def build_stress_overlay_geojson(self) -> Dict[str, Any]:
        """GeoJSON FeatureCollection: рёбра текущего графа со стресс-полями (отладочная маска)."""
        with self._graph_lock:
            if not self._loaded or self._graph is None:
                raise BikeRouterError("Граф не загружен. Вызовите warmup().")
            G = self._graph
        feats: List[dict] = []
        for u, v, key, d in G.edges(data=True, keys=True):
            coords = RouteService._edge_linestring_coords(G, u, v, key)
            if len(coords) < 2:
                continue
            ln = float(d.get("length", 0.0) or 0.0)
            lts = float(d.get("stress_lts", 1.5) or 1.5)
            isc = float(d.get("stress_intersection_score", 0.0) or 0.0)
            sc = float(d.get("stress_cost", 0.0) or 0.0)
            hw = _first_value(d.get("highway", ""))
            hw_s = str(hw or "").lower()
            feats.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": coords},
                    "properties": {
                        "stress_lts": round(lts, 3),
                        "intersection_stress": round(isc, 4),
                        "stress_cost": round(sc, 2),
                        "highway": hw_s,
                        "length_m": round(ln, 1),
                    },
                }
            )
        return {"type": "FeatureCollection", "features": feats}

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

        self._ensure_graph_for_corridor(start, end)
        with self._graph_lock:
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
        weather: Optional[WeatherWeightParams] = None,
        time_slot_key: str = "noon",
        heat_context_mult: float = 1.0,
        weather_ctx: Optional[WeatherRouteContext] = None,
        enrich_heat_stress_metrics: bool = False,
        routing_profile_key: str = "balanced",
        season: str = "summer",
        air_temperature_c: Optional[float] = None,
    ) -> List[RouteResponse]:
        """Собрать варианты; порядок в ответе — кратчайший, полный, зелёный (если есть)."""
        profile = _PROFILE_MAP[profile_key]
        G = graph if graph is not None else self._graph
        if G is None:
            raise BikeRouterError("Граф не загружен.")
        router = self._app.router
        w_full = f"weight_{profile.key}_full"
        w_green = f"weight_{profile.key}_green"

        if weather is not None and weather.enabled:
            rf = router.find_route_combined(
                G,
                start,
                end,
                profile_key,
                time_slot_key,
                _WEATHER_PHYSICAL_ROUTING_PREF,
                heat_context_mult=heat_context_mult,
                weather=weather,
                physical_weight_key=w_full,
            )
        else:
            rf = router.find_route(G, start, end, w_full)
        rg = None
        if include_green_route:
            if weather is not None and weather.enabled:
                rg = router.find_route_combined(
                    G,
                    start,
                    end,
                    profile_key,
                    time_slot_key,
                    _WEATHER_PHYSICAL_ROUTING_PREF,
                    heat_context_mult=heat_context_mult,
                    weather=weather,
                    physical_weight_key=w_green,
                )
            else:
                rg = router.find_route(G, start, end, w_green)

        exclude = {tuple(rf.edges)}
        if rg is not None:
            exclude.add(tuple(rg.edges))

        def _bundle_hm(
            route_rr: RouteResult, physical_weight_key: str
        ) -> Optional[HeatStressMetrics]:
            if not enrich_heat_stress_metrics:
                return None
            return self._make_heat_stress_metrics(
                G,
                route_rr,
                profile_key,
                time_slot_key,
                routing_profile_key,
                season=season,
                air_temperature_c=air_temperature_c,
                heat_mult=heat_context_mult,
                weather=weather,
                physical_weight_key=physical_weight_key,
            )

        routes_out: List[RouteResponse] = []
        if include_green_route and rg is not None:
            seg_full = router.route_segment_costs(
                G,
                rf,
                profile_key,
                time_slot_key,
                heat_context_mult=heat_context_mult,
                weather=weather,
                physical_weight_key=w_full,
            )
            seg_green = router.route_segment_costs(
                G,
                rg,
                profile_key,
                time_slot_key,
                heat_context_mult=heat_context_mult,
                weather=weather,
                physical_weight_key=w_green,
            )
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
                    cost_override=round(seg_full["physical"], 1),
                    heat_stress_metrics=_bundle_hm(rf, w_full),
                    weather=weather_ctx,
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
                    cost_override=round(seg_green["physical"], 1),
                    heat_stress_metrics=_bundle_hm(rg, w_green),
                    weather=weather_ctx,
                )
            )
        else:
            seg_full = (
                router.route_segment_costs(
                    G,
                    rf,
                    profile_key,
                    time_slot_key,
                    heat_context_mult=heat_context_mult,
                    weather=weather,
                    physical_weight_key=w_full,
                )
                if weather is not None and weather.enabled
                else None
            )
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
                    cost_override=(
                        round(seg_full["physical"], 1)
                        if seg_full is not None
                        else None
                    ),
                    heat_stress_metrics=_bundle_hm(rf, w_full),
                    weather=weather_ctx,
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
                    heat_stress_metrics=_bundle_hm(rs, w_full),
                    weather=weather_ctx,
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
                        heat_stress_metrics=_bundle_hm(alt, w_full),
                        weather=weather_ctx,
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
        weather: Optional[WeatherWeightParams] = None,
        time_slot_key: str = "noon",
        heat_context_mult: float = 1.0,
        weather_ctx: Optional[WeatherRouteContext] = None,
        enrich_heat_stress_metrics: bool = False,
        routing_profile_key: str = "balanced",
        season: str = "summer",
        air_temperature_c: Optional[float] = None,
    ) -> List[RouteResponse]:
        self._ensure_graph_for_corridor(
            start,
            end,
            skip_satellite_green=skip_satellite_green,
            corridor_buffer_meters=corridor_buffer_meters,
        )
        with self._graph_lock:
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
            weather=weather,
            time_slot_key=time_slot_key,
            heat_context_mult=heat_context_mult,
            weather_ctx=weather_ctx,
            enrich_heat_stress_metrics=enrich_heat_stress_metrics,
            routing_profile_key=routing_profile_key,
            season=season,
            air_temperature_c=air_temperature_c,
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
        self._ensure_graph_for_corridor(start, end, skip_satellite_green=False)
        with self._graph_lock:
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

    def _make_heat_stress_metrics(
        self,
        G: nx.MultiDiGraph,
        route: RouteResult,
        profile_key: str,
        time_slot_key: str,
        routing_profile_key: str,
        *,
        season: str = "summer",
        air_temperature_c: Optional[float] = None,
        heat_mult: Optional[float] = None,
        weather: Optional[WeatherWeightParams] = None,
        physical_weight_key: Optional[str] = None,
    ) -> HeatStressMetrics:
        pref = routing_preference_profile(routing_profile_key)
        router = self._app.router
        turns = router.count_significant_turns(G, route)
        hm = (
            float(heat_mult)
            if heat_mult is not None and math.isfinite(float(heat_mult))
            else heat_context_multiplier(
                season, time_slot_key, air_temperature_c
            )
        )
        seg = router.route_segment_costs(
            G,
            route,
            profile_key,
            time_slot_key,
            heat_context_mult=hm,
            weather=weather,
            physical_weight_key=physical_weight_key,
        )
        expm = router.route_exposure_metrics(G, route, time_slot_key)
        exp_open = router.route_exposure_metrics(
            G, route, time_slot_key, exposure_threshold=0.32
        )
        st = router.route_stress_levels(G, route)
        sh = router.route_shade_shares(G, route)
        sh_rt = router.route_shelter_length_weighted_averages(G, route)
        wdm = router.route_wind_direction_metrics(G, route, weather)
        n_int = router.count_stressful_intersections(G, route)
        n_hi_seg = router.count_high_stress_segments(G, route)
        comb = router.route_combined_total(
            G,
            route,
            profile_key,
            time_slot_key,
            pref,
            turn_count=turns,
            heat_context_mult=hm,
            weather=weather,
            physical_weight_key=physical_weight_key,
        )
        turn_part = float(pref.delta * TURN_PENALTY_BASE * max(0, turns))
        wm_dict: Optional[Dict[str, float]] = None
        if weather is not None and weather.enabled:
            wm_dict = {
                k: round(float(v), 4) for k, v in asdict(weather.mults).items()
            }
        br = CombinedCostBreakdown(
            physical=round(pref.alpha * seg["physical"], 1),
            heat_effective=round(pref.beta * seg["heat"], 1),
            heat_raw=round(pref.beta * seg["heat_raw"], 1),
            stress=round(pref.gamma * seg["stress"], 1),
            stress_segment=round(pref.gamma * seg["stress_segment"], 1),
            stress_intersection=round(pref.gamma * seg["stress_intersection"], 1),
            turn_penalty=round(turn_part, 1),
            heat_context_multiplier=round(hm, 4),
            weather_multipliers=wm_dict,
        )
        return HeatStressMetrics(
            time_slot=time_slot_key,
            season=(season or "summer").lower(),
            routing_profile=pref.key,
            total_heat_cost=round(seg["heat"], 1),
            heat_cost_raw=round(seg["heat_raw"], 1),
            stress_cost_total=round(seg["stress"], 1),
            exposed_high_length_m=round(expm["exposed_high_length_m"], 1),
            exposed_open_unfavorable_length_m=round(
                exp_open["exposed_high_length_m"], 1
            ),
            avg_exposure_unit=round(expm["avg_exposure"], 3),
            vegetation_shade_share=round(sh["vegetation_shade_share"], 3),
            building_shade_share=round(sh["building_shade_share"], 3),
            route_open_sky_share=round(sh_rt["route_open_sky_share"], 4),
            route_building_shade_share=round(sh_rt["route_building_shade_share"], 4),
            route_covered_share=round(sh_rt["route_covered_share"], 4),
            route_bad_wet_surface_share=round(sh_rt["route_bad_wet_surface_share"], 4),
            route_winter_harsh_surface_share=round(
                float(sh_rt.get("route_winter_harsh_surface_share", 0.0)), 4
            ),
            route_wind_direction_aware=float(wdm.get("route_wind_direction_aware", 0.0)),
            route_mean_wind_to_street_angle_deg=round(
                float(wdm.get("route_mean_wind_to_street_angle_deg", 0.0)), 2
            ),
            route_mean_heat_directional_wind_exp=round(
                float(wdm.get("route_mean_heat_directional_wind_exp", 0.0)), 4
            ),
            route_mean_heat_building_wind_factor=round(
                float(wdm.get("route_mean_heat_building_wind_factor", 0.0)), 4
            ),
            route_frac_wind_along_open_hostile=round(
                float(wdm.get("route_frac_wind_along_open_hostile", 0.0)), 4
            ),
            route_frac_wind_cross_building_screen=round(
                float(wdm.get("route_frac_wind_cross_building_screen", 0.0)), 4
            ),
            avg_stress_lts=round(st["avg_lts"], 2),
            max_stress_lts=round(st["max_lts"], 2),
            high_stress_length_fraction=round(st["high_stress_fraction"], 3),
            high_stress_segments_count=n_hi_seg,
            stressful_intersections_count=n_int,
            turn_count=turns,
            combined_cost=round(comb, 1),
            combined_breakdown=br,
        )

    @staticmethod
    def _route_uses_thermal_proxy(
        G: nx.MultiDiGraph, route: Optional[RouteResult]
    ) -> bool:
        if route is None or route.edge_count <= 0:
            return False
        for i in range(route.edge_count):
            d = G.edges[route.edges[i]]
            if int(d.get("thermal_use_proxy", 0) or 0):
                return True
        return False

    def _routing_context_meta(
        self,
        *,
        time_slot_key: str,
        season: str,
        routing_profile_key: str,
        air_temperature_c: Optional[float],
        heat_mult: float,
        criterion: str,
        G: nx.MultiDiGraph,
        route: Optional[RouteResult],
    ) -> RoutingContextMeta:
        return RoutingContextMeta(
            time_slot=time_slot_key,
            season=(season or "summer").lower(),
            routing_profile=routing_profile_key,
            criterion=criterion,
            air_temperature_c=air_temperature_c,
            heat_context_multiplier=round(float(heat_mult), 4),
            thermal_model_proxy=self._route_uses_thermal_proxy(G, route),
        )

    def _combined_route_with_detour_cap(
        self,
        G: nx.MultiDiGraph,
        start: Tuple[float, float],
        end: Tuple[float, float],
        profile_key: str,
        time_slot_key: str,
        preference_key: str,
        *,
        max_detour_ratio: float,
        heat_context_mult: float,
        weather: Optional[WeatherWeightParams],
    ) -> Tuple[RouteResult, str]:
        """Поиск по α·physical+β·heat+γ·stress с ограничением длины относительно full.

        Возвращает маршрут и короткую русскую подпись, если пришлось упасть на эталон full.
        """
        router = self._app.router
        w_full = f"weight_{profile_key}_full"
        base = routing_preference_profile(preference_key)
        rf = router.find_route(G, start, end, w_full)
        lf = router.calculate_length(G, rf)
        beta = float(base.beta)
        gamma = float(base.gamma)
        r_last: Optional[RouteResult] = None
        cap = max(0.0, float(max_detour_ratio))
        for _ in range(7):
            pref = replace(base, beta=beta, gamma=gamma)
            r = router.find_route_combined(
                G,
                start,
                end,
                profile_key,
                time_slot_key,
                pref,
                heat_context_mult=heat_context_mult,
                weather=weather,
                physical_weight_key=w_full,
            )
            r_last = r
            lh = router.calculate_length(G, r)
            if lf <= 1e-9 or lh <= lf * (1.0 + cap):
                return r, ""
            beta *= 0.52
            gamma *= 0.96
        assert r_last is not None
        return rf, "ограничение обхода: как «Оптимальный по энергии»"

    def _build_criterion_routes(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        profile_key: str,
        G: nx.MultiDiGraph,
        criterion: str,
        routing_profile_key: str,
        time_slot_key: str,
        *,
        season: str = "summer",
        air_temperature_c: Optional[float] = None,
        weather: Optional[WeatherWeightParams] = None,
        weather_ctx: Optional[WeatherRouteContext] = None,
    ) -> List[RouteResponse]:
        router = self._app.router
        pref = routing_preference_profile(routing_profile_key)
        routes_out: List[RouteResponse] = []
        crit = criterion.strip().lower()
        hm_ctx = heat_context_multiplier(
            season, time_slot_key, air_temperature_c
        )

        w_full_key = f"weight_{profile_key}_full"

        if crit == "heat":
            r, cap_note = self._combined_route_with_detour_cap(
                G,
                start,
                end,
                profile_key,
                time_slot_key,
                "thermal_physical_base",
                max_detour_ratio=self._app.settings.heat_max_detour_ratio,
                heat_context_mult=hm_ctx,
                weather=weather,
            )
            h_label = "С учётом теплового комфорта"
            if cap_note:
                h_label = f"{h_label} ({cap_note})"
            seg = router.route_segment_costs(
                G,
                r,
                profile_key,
                time_slot_key,
                heat_context_mult=hm_ctx,
                weather=weather,
                physical_weight_key=w_full_key,
            )
            metrics = self._make_heat_stress_metrics(
                G,
                r,
                profile_key,
                time_slot_key,
                routing_profile_key,
                season=season,
                air_temperature_c=air_temperature_c,
                heat_mult=hm_ctx,
                weather=weather,
                physical_weight_key=w_full_key,
            )
            rc = self._routing_context_meta(
                time_slot_key=time_slot_key,
                season=season,
                routing_profile_key=routing_profile_key,
                air_temperature_c=air_temperature_c,
                heat_mult=hm_ctx,
                criterion="heat",
                G=G,
                route=r,
            )
            routes_out.append(
                self._build_route_response(
                    start,
                    end,
                    profile_key,
                    "heat",
                    r,
                    w_full_key,
                    h_label,
                    graph=G,
                    cost_override=round(seg["heat"], 1),
                    heat_stress_metrics=metrics,
                    routing_context=rc,
                    weather=weather_ctx,
                )
            )
        elif crit == "stress":
            r, cap_note = self._combined_route_with_detour_cap(
                G,
                start,
                end,
                profile_key,
                time_slot_key,
                "stress_physical_base",
                max_detour_ratio=self._app.settings.stress_max_detour_ratio,
                heat_context_mult=hm_ctx,
                weather=weather,
            )
            s_label = "С учётом безопасности (минимальный стресс)"
            if cap_note:
                s_label = f"{s_label} ({cap_note})"
            seg = router.route_segment_costs(
                G,
                r,
                profile_key,
                time_slot_key,
                heat_context_mult=hm_ctx,
                weather=weather,
                physical_weight_key=w_full_key,
            )
            metrics = self._make_heat_stress_metrics(
                G,
                r,
                profile_key,
                time_slot_key,
                routing_profile_key,
                season=season,
                air_temperature_c=air_temperature_c,
                heat_mult=hm_ctx,
                weather=weather,
                physical_weight_key=w_full_key,
            )
            rc = self._routing_context_meta(
                time_slot_key=time_slot_key,
                season=season,
                routing_profile_key=routing_profile_key,
                air_temperature_c=air_temperature_c,
                heat_mult=hm_ctx,
                criterion="stress",
                G=G,
                route=r,
            )
            routes_out.append(
                self._build_route_response(
                    start,
                    end,
                    profile_key,
                    "stress",
                    r,
                    w_full_key,
                    s_label,
                    graph=G,
                    cost_override=round(seg["stress"], 1),
                    heat_stress_metrics=metrics,
                    routing_context=rc,
                    weather=weather_ctx,
                )
            )
        elif crit == "heat_stress":
            if pref.delta > 0:
                r = router.find_route_combined_with_turns(
                    G,
                    start,
                    end,
                    profile_key,
                    time_slot_key,
                    pref,
                    heat_context_mult=hm_ctx,
                    weather=weather,
                    physical_weight_key=w_full_key,
                )
            else:
                r = router.find_route_combined(
                    G,
                    start,
                    end,
                    profile_key,
                    time_slot_key,
                    pref,
                    heat_context_mult=hm_ctx,
                    weather=weather,
                    physical_weight_key=w_full_key,
                )
            metrics = self._make_heat_stress_metrics(
                G,
                r,
                profile_key,
                time_slot_key,
                routing_profile_key,
                season=season,
                air_temperature_c=air_temperature_c,
                heat_mult=hm_ctx,
                weather=weather,
                physical_weight_key=w_full_key,
            )
            rc = self._routing_context_meta(
                time_slot_key=time_slot_key,
                season=season,
                routing_profile_key=routing_profile_key,
                air_temperature_c=air_temperature_c,
                heat_mult=hm_ctx,
                criterion="heat_stress",
                G=G,
                route=r,
            )
            routes_out.append(
                self._build_route_response(
                    start,
                    end,
                    profile_key,
                    "heat_stress",
                    r,
                    f"weight_{profile_key}_full",
                    "С учётом тепла и безопасности",
                    graph=G,
                    cost_override=metrics.combined_cost,
                    heat_stress_metrics=metrics,
                    routing_context=rc,
                    weather=weather_ctx,
                )
            )
        else:
            raise ValueError(
                f"Неизвестный критерий: {criterion!r}. "
                f"Ожидалось heat, stress, heat_stress"
            )

        return routes_out

    def _compute_alternatives_criterion_once(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        profile_key: str,
        *,
        criterion: str,
        routing_profile_key: str,
        time_slot_key: str,
        green_enabled: bool,
        corridor_buffer_meters: Optional[float],
        season: str = "summer",
        air_temperature_c: Optional[float] = None,
        weather: Optional[WeatherWeightParams] = None,
        weather_ctx: Optional[WeatherRouteContext] = None,
    ) -> List[RouteResponse]:
        self._ensure_graph_for_corridor(
            start,
            end,
            skip_satellite_green=not green_enabled,
            corridor_buffer_meters=corridor_buffer_meters,
        )
        with self._graph_lock:
            if not self._loaded or self._graph is None:
                raise BikeRouterError("Граф не загружен. Вызовите warmup().")
            G = self._graph
            bounds = self._bounds
        assert bounds is not None
        self._validate_point_on_graph(G, bounds, start, "start")
        self._validate_point_on_graph(G, bounds, end, "end")
        return self._build_criterion_routes(
            start,
            end,
            profile_key,
            G,
            criterion,
            routing_profile_key,
            time_slot_key,
            season=season,
            air_temperature_c=air_temperature_c,
            weather=weather,
            weather_ctx=weather_ctx,
        )

    @staticmethod
    def _order_unified_routes(routes: List[RouteResponse]) -> List[RouteResponse]:
        """Один маршрут на mode; порядок — см. _UNIFIED_ROUTE_ORDER."""
        by_mode: Dict[str, RouteResponse] = {}
        for r in routes:
            if r.mode not in by_mode:
                by_mode[r.mode] = r
        return [by_mode[m] for m in _UNIFIED_ROUTE_ORDER if m in by_mode]

    def _annotate_unified_routes(self, routes: List[RouteResponse]) -> List[RouteResponse]:
        """Пояснение для теплового варианта и короткое сравнение длины с кратчайшим."""
        by_mode = {r.mode: r for r in routes}
        shortest = by_mode.get("shortest")
        base_len = float(shortest.length_m) if shortest else None
        out: List[RouteResponse] = []
        for r in routes:
            note: Optional[str] = None
            if r.mode == "heat":
                note = (
                    "Тепловой комфорт на базе энергетического маршрута (рельеф/покрытие), "
                    "с учётом погоды и геометрии среды; не эквивалентен «максимально зелёному»."
                )
            effect: Optional[str] = None
            if base_len is not None and r.mode != "shortest":
                delta = float(r.length_m) - base_len
                if delta >= 30.0:
                    effect = (
                        f"Длинее кратчайшего по сети примерно на "
                        f"{int(round(delta))} м."
                    )
                elif delta <= -30.0:
                    effect = (
                        f"Короче кратчайшего по сети примерно на "
                        f"{int(round(-delta))} м."
                    )
            out.append(
                r.model_copy(
                    update={"variant_note_ru": note, "effect_summary_ru": effect}
                )
            )
        return out

    def compute_alternatives(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        profile_key: str,
        *,
        green_enabled: bool = True,
        departure_time: Optional[str] = None,
        time_slot_override: Optional[str] = None,
        air_temperature_c: Optional[float] = None,
        weather_mode: str = "none",
        use_live_weather: bool = False,
        weather_time: Optional[str] = None,
        temperature_c: Optional[float] = None,
        precipitation_mm: Optional[float] = None,
        wind_speed_ms: Optional[float] = None,
        wind_direction_deg: Optional[float] = None,
        cloud_cover_pct: Optional[float] = None,
        humidity_pct: Optional[float] = None,
        wind_gusts_ms: Optional[float] = None,
        shortwave_radiation_wm2: Optional[float] = None,
        snowfall_cm_h: Optional[float] = None,
        snow_depth_m: Optional[float] = None,
        weather_code: Optional[int] = None,
        corridor_expand_schedule_meters: Optional[Sequence[float]] = None,
    ) -> AlternativesResponse:
        """Все доступные варианты сразу: кратчайший, энергия, зелёный, тепло, стресс, тепло+безопасность.

        Внутри для тепло/стресс комбинаций используется профиль предпочтений
        ``balanced`` (см. ``routing_preference_profile`` в config).

        ``corridor_expand_schedule_meters``: если задано, подставляется вместо
        ``Settings.corridor_expand_schedule_meters`` при переборе буферов коридора
        (режим dynamic corridor). Иначе — из конфигурации.
        """
        profile = _PROFILE_MAP.get(profile_key)
        if profile is None:
            raise ValueError(
                f"Неизвестный профиль: {profile_key!r}. "
                f"Допустимые: {list(_PROFILE_MAP)}"
            )

        rp_internal = "balanced"
        now_utc = datetime.now(timezone.utc)
        dep_for_weather = departure_time or now_utc.replace(microsecond=0).isoformat()

        _s = self._app.settings
        _thermal = {
            "hot_tree": float(_s.heat_hot_tree_bonus_scale),
            "hot_open": float(_s.heat_hot_open_sky_penalty_scale),
            "cold_canyon": float(_s.heat_cold_building_canyon_bonus_scale),
            "cold_tree_damp": float(_s.heat_cold_tree_bonus_damping),
            "response": float(_s.heat_weather_response_scale),
        }
        w_snap, wsrc, wp = _resolve_route_weather(
            start,
            end,
            request_mode=weather_mode,
            use_live_weather=use_live_weather,
            weather_time=weather_time,
            departure_time=dep_for_weather,
            temperature_c=temperature_c,
            precipitation_mm=precipitation_mm,
            wind_speed_ms=wind_speed_ms,
            cloud_cover_pct=cloud_cover_pct,
            humidity_pct=humidity_pct,
            wind_gusts_ms=wind_gusts_ms,
            wind_direction_deg=wind_direction_deg,
            shortwave_radiation_wm2=shortwave_radiation_wm2,
            snowfall_cm_h=snowfall_cm_h,
            snow_depth_m=snow_depth_m,
            weather_code=weather_code,
            thermal_scales=_thermal,
            settings=_s,
        )
        season_val = _infer_season_from_month(now_utc.month)
        air_eff: Optional[float] = air_temperature_c
        if air_eff is None:
            air_eff = float(w_snap.temperature_c)

        slot_key = _resolve_time_slot_key(dep_for_weather, time_slot_override)
        hm_default = heat_context_multiplier(season_val, slot_key, air_eff)
        disp_mode = _engine_weather_mode(weather_mode, use_live_weather)
        weather_ctx = _build_weather_route_context(
            request_mode=disp_mode,
            use_live_weather=use_live_weather,
            weather_time=weather_time,
            departure_time=dep_for_weather,
            snap=w_snap,
            source=wsrc,
            wp=wp,
        )
        _log_route_weather_line(
            built_at_utc=now_utc,
            start=start,
            end=end,
            snap=w_snap,
            source=wsrc,
            wp=wp,
            weather_time_effective=weather_ctx.weather_time,
        )

        inc_route_disk_miss()
        logger.info(
            "Alternatives (unified): профиль=%s green=%s slot=%s season=%s",
            profile_key,
            green_enabled,
            slot_key,
            season_val,
        )

        s = _s
        skip_sat = not green_enabled
        include_green = green_enabled
        routes_base: List[RouteResponse]

        buf_sched = (
            list(corridor_expand_schedule_meters)
            if corridor_expand_schedule_meters is not None
            else s.corridor_expand_schedule_meters
        )
        if not s.use_dynamic_corridor_graph or not buf_sched:
            routes_base = self._compute_alternatives_once(
                start,
                end,
                profile_key,
                include_green_route=include_green,
                skip_satellite_green=skip_sat,
                corridor_buffer_meters=None,
                weather=wp,
                time_slot_key=slot_key,
                heat_context_mult=hm_default,
                weather_ctx=weather_ctx,
                enrich_heat_stress_metrics=True,
                routing_profile_key=rp_internal,
                season=season_val,
                air_temperature_c=air_eff,
            )
        else:
            routes_base = None
            last_nf: Optional[RouteNotFoundError] = None
            for attempt, eff in enumerate(buf_sched, start=1):
                try:
                    routes_base = self._compute_alternatives_once(
                        start,
                        end,
                        profile_key,
                        include_green_route=include_green,
                        skip_satellite_green=skip_sat,
                        corridor_buffer_meters=float(eff),
                        weather=wp,
                        time_slot_key=slot_key,
                        heat_context_mult=hm_default,
                        weather_ctx=weather_ctx,
                        enrich_heat_stress_metrics=True,
                        routing_profile_key=rp_internal,
                        season=season_val,
                        air_temperature_c=air_eff,
                    )
                    logger.info(
                        "auto_expand_corridor attempt=%d corridor_m=%.0f "
                        "result=ok routes=%d",
                        attempt,
                        eff,
                        len(routes_base),
                    )
                    break
                except RouteNotFoundError as e:
                    last_nf = e
                    logger.warning(
                        "auto_expand_corridor attempt=%d corridor_m=%.0f "
                        "result=no_path code=%s",
                        attempt,
                        eff,
                        e.code,
                    )
                    if attempt >= len(buf_sched):
                        logger.warning(
                            "auto_expand_corridor exhausted schedule (%s) м",
                            ",".join(f"{x:.0f}" for x in buf_sched),
                        )
                        assert last_nf is not None
                        _raise_route_not_found_after_schedule(
                            last_nf,
                            schedule=buf_sched,
                            last_eff=float(eff),
                            attempts=attempt,
                        )
            assert routes_base is not None

        with self._graph_lock:
            if not self._loaded or self._graph is None:
                raise BikeRouterError("Граф не загружен. Вызовите warmup().")
            G = self._graph

        extra_routes: List[RouteResponse] = []
        for sub in ("heat", "stress", "heat_stress"):
            try:
                lst = self._build_criterion_routes(
                    start,
                    end,
                    profile_key,
                    G,
                    sub,
                    rp_internal,
                    slot_key,
                    season=season_val,
                    air_temperature_c=air_eff,
                    weather=wp,
                    weather_ctx=weather_ctx,
                )
                if lst:
                    extra_routes.append(lst[0])
            except RouteNotFoundError as e:
                logger.info(
                    "Alternatives unified: режим %s недоступен (нет пути): %s",
                    sub,
                    e,
                )
            except BikeRouterError as e:
                logger.warning(
                    "Alternatives unified: режим %s — ошибка: %s", sub, e
                )

        merged = self._order_unified_routes(list(routes_base) + extra_routes)
        merged = self._annotate_unified_routes(merged)
        return AlternativesResponse(routes=merged, criteria_bundle=None)

    def compute_heat_alternative(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        profile_key: str,
        *,
        green_enabled: bool = True,
        departure_time: Optional[str] = None,
        time_slot_override: Optional[str] = None,
        air_temperature_c: Optional[float] = None,
        weather_mode: str = "none",
        use_live_weather: bool = False,
        weather_time: Optional[str] = None,
        temperature_c: Optional[float] = None,
        precipitation_mm: Optional[float] = None,
        wind_speed_ms: Optional[float] = None,
        wind_direction_deg: Optional[float] = None,
        cloud_cover_pct: Optional[float] = None,
        humidity_pct: Optional[float] = None,
        wind_gusts_ms: Optional[float] = None,
        shortwave_radiation_wm2: Optional[float] = None,
        snowfall_cm_h: Optional[float] = None,
        snow_depth_m: Optional[float] = None,
        weather_code: Optional[int] = None,
        corridor_expand_schedule_meters: Optional[Sequence[float]] = None,
    ) -> AlternativesResponse:
        """Только вариант ``heat`` (один критерий), без shortest/full/green/stress.

        Используется пакетными synthetic-экспериментами, чтобы не считать лишние
        альтернативы. Погода и слот — как в ``compute_alternatives``.
        """
        if profile_key not in _PROFILE_MAP:
            raise ValueError(
                f"Неизвестный профиль: {profile_key!r}. "
                f"Допустимые: {list(_PROFILE_MAP)}"
            )

        rp_internal = "balanced"
        now_utc = datetime.now(timezone.utc)
        dep_for_weather = departure_time or now_utc.replace(microsecond=0).isoformat()

        _s = self._app.settings
        _thermal = {
            "hot_tree": float(_s.heat_hot_tree_bonus_scale),
            "hot_open": float(_s.heat_hot_open_sky_penalty_scale),
            "cold_canyon": float(_s.heat_cold_building_canyon_bonus_scale),
            "cold_tree_damp": float(_s.heat_cold_tree_bonus_damping),
            "response": float(_s.heat_weather_response_scale),
        }
        w_snap, wsrc, wp = _resolve_route_weather(
            start,
            end,
            request_mode=weather_mode,
            use_live_weather=use_live_weather,
            weather_time=weather_time,
            departure_time=dep_for_weather,
            temperature_c=temperature_c,
            precipitation_mm=precipitation_mm,
            wind_speed_ms=wind_speed_ms,
            cloud_cover_pct=cloud_cover_pct,
            humidity_pct=humidity_pct,
            wind_gusts_ms=wind_gusts_ms,
            wind_direction_deg=wind_direction_deg,
            shortwave_radiation_wm2=shortwave_radiation_wm2,
            snowfall_cm_h=snowfall_cm_h,
            snow_depth_m=snow_depth_m,
            weather_code=weather_code,
            thermal_scales=_thermal,
            settings=_s,
        )
        season_val = _infer_season_from_month(now_utc.month)
        air_eff: Optional[float] = air_temperature_c
        if air_eff is None:
            air_eff = float(w_snap.temperature_c)

        slot_key = _resolve_time_slot_key(dep_for_weather, time_slot_override)
        disp_mode = _engine_weather_mode(weather_mode, use_live_weather)
        weather_ctx = _build_weather_route_context(
            request_mode=disp_mode,
            use_live_weather=use_live_weather,
            weather_time=weather_time,
            departure_time=dep_for_weather,
            snap=w_snap,
            source=wsrc,
            wp=wp,
        )
        _log_route_weather_line(
            built_at_utc=now_utc,
            start=start,
            end=end,
            snap=w_snap,
            source=wsrc,
            wp=wp,
            weather_time_effective=weather_ctx.weather_time,
        )

        inc_route_disk_miss()
        logger.info(
            "Alternatives heat-only: профиль=%s green=%s slot=%s season=%s",
            profile_key,
            green_enabled,
            slot_key,
            season_val,
        )

        s = _s

        buf_sched = (
            list(corridor_expand_schedule_meters)
            if corridor_expand_schedule_meters is not None
            else s.corridor_expand_schedule_meters
        )

        def _heat_once(
            corridor_buffer_meters: Optional[float],
        ) -> List[RouteResponse]:
            return self._compute_alternatives_criterion_once(
                start,
                end,
                profile_key,
                criterion="heat",
                routing_profile_key=rp_internal,
                time_slot_key=slot_key,
                green_enabled=green_enabled,
                corridor_buffer_meters=corridor_buffer_meters,
                season=season_val,
                air_temperature_c=air_eff,
                weather=wp,
                weather_ctx=weather_ctx,
            )

        heat_routes: List[RouteResponse]
        if not s.use_dynamic_corridor_graph or not buf_sched:
            heat_routes = _heat_once(None)
        else:
            heat_routes = []
            last_nf: Optional[RouteNotFoundError] = None
            for attempt, eff in enumerate(buf_sched, start=1):
                try:
                    heat_routes = _heat_once(float(eff))
                    logger.info(
                        "heat_only auto_expand_corridor attempt=%d corridor_m=%.0f "
                        "result=ok routes=%d",
                        attempt,
                        eff,
                        len(heat_routes),
                    )
                    break
                except RouteNotFoundError as e:
                    last_nf = e
                    logger.warning(
                        "heat_only auto_expand_corridor attempt=%d corridor_m=%.0f "
                        "result=no_path code=%s",
                        attempt,
                        eff,
                        e.code,
                    )
                    if attempt >= len(buf_sched):
                        logger.warning(
                            "heat_only auto_expand_corridor exhausted schedule (%s) м",
                            ",".join(f"{x:.0f}" for x in buf_sched),
                        )
                        assert last_nf is not None
                        _raise_route_not_found_after_schedule(
                            last_nf,
                            schedule=buf_sched,
                            last_eff=float(eff),
                            attempts=attempt,
                        )

        merged = self._order_unified_routes(heat_routes)
        merged = self._annotate_unified_routes(merged)
        return AlternativesResponse(routes=merged, criteria_bundle=None)

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
        buf_sched = s.corridor_expand_schedule_meters
        if not s.use_dynamic_corridor_graph or not buf_sched:
            return self._compute_alternatives_once(
                start,
                end,
                profile_key,
                include_green_route=False,
                skip_satellite_green=True,
                corridor_buffer_meters=None,
            )
        last_nf: Optional[RouteNotFoundError] = None
        for attempt, eff in enumerate(buf_sched, start=1):
            try:
                routes = self._compute_alternatives_once(
                    start,
                    end,
                    profile_key,
                    include_green_route=False,
                    skip_satellite_green=True,
                    corridor_buffer_meters=float(eff),
                )
                logger.info(
                    "alternatives_phase1 attempt=%d corridor_m=%.0f routes=%d",
                    attempt,
                    eff,
                    len(routes),
                )
                return routes
            except RouteNotFoundError as e:
                last_nf = e
                logger.warning(
                    "alternatives_phase1 attempt=%d corridor_m=%.0f no_path",
                    attempt,
                    eff,
                )
                if attempt >= len(buf_sched):
                    logger.warning(
                        "alternatives_phase1 exhausted schedule (%s) м",
                        ",".join(f"{x:.0f}" for x in buf_sched),
                    )
                    assert last_nf is not None
                    _raise_route_not_found_after_schedule(
                        last_nf,
                        schedule=buf_sched,
                        last_eff=float(eff),
                        attempts=attempt,
                    )
        raise RuntimeError("alternatives_phase1: пустой corridor_expand_schedule_meters")
