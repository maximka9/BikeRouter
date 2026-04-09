"""Сервис маршрутизации и расчёта статистик маршрута."""

import logging
import math
import numbers
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import osmnx as ox

from ..config import ModeProfile
from ..exceptions import RouteNotFoundError

logger = logging.getLogger(__name__)


def coerce_edge_weight_numeric(val: Any, *, fallback: float = float("inf")) -> float:
    """Привести вес ребра к float (GraphML/OSMnx иногда дают строки)."""
    if val is None:
        return fallback
    if isinstance(val, bool):
        return fallback
    if isinstance(val, numbers.Real):
        x = float(val)
        return x if math.isfinite(x) else fallback
    if isinstance(val, str):
        s = val.strip().replace(",", ".")
        if not s:
            return fallback
        try:
            x = float(s)
            return x if math.isfinite(x) else fallback
        except ValueError:
            return fallback
    try:
        x = float(val)
        return x if math.isfinite(x) else fallback
    except (TypeError, ValueError):
        return fallback


# После GraphML (кэш коридора) float часто оказываются строками — ломает сравнения и Dijkstra.
_GRAPHML_NUMERIC_EDGE_ATTRS = frozenset(
    {
        "gradient",
        "elevation_diff",
        "edge_climb",
        "edge_descent",
        "gradient_percent",
        "trees_percent",
        "grass_percent",
        "green_percent",
        "green_coeff",
    }
)


def sanitize_multidigraph_routing_weights(G: nx.MultiDiGraph) -> None:
    """На рёбрах привести числовые поля к float (GraphML / OSMnx дают строки)."""
    for _u, _v, _k, d in G.edges(data=True, keys=True):
        if "length" in d:
            ln = coerce_edge_weight_numeric(d["length"], fallback=float("nan"))
            if math.isnan(ln):
                ln = 1.0
            d["length"] = ln
        for attr in list(d.keys()):
            if attr != "length" and not str(attr).startswith("weight_"):
                continue
            if attr == "length":
                continue
            d[attr] = coerce_edge_weight_numeric(
                d.get(attr),
                fallback=coerce_edge_weight_numeric(
                    d.get("length", 1.0), fallback=1.0
                ),
            )
        for attr in _GRAPHML_NUMERIC_EDGE_ATTRS:
            if attr not in d:
                continue
            d[attr] = coerce_edge_weight_numeric(d.get(attr), fallback=0.0)


def _coords_from_shapely_linear(geom: Any) -> List[Tuple[float, float]]:
    """Вершины (lon, lat) для LineString / MultiLineString / LinearRing / GeometryCollection.

    У рёбер OSM иногда бывает ``MultiLineString`` — у него нет ``.coords``, из-за чего
    падал весь ``POST /alternatives`` с 500.
    """
    if geom is None:
        return []
    gt = getattr(geom, "geom_type", "") or ""
    try:
        if gt in ("LineString", "LinearRing"):
            return list(geom.coords)
        if gt == "MultiLineString":
            out: List[Tuple[float, float]] = []
            for g in geom.geoms:
                out.extend(list(g.coords))
            return out
        if gt == "GeometryCollection":
            acc: List[Tuple[float, float]] = []
            for g in geom.geoms:
                acc.extend(_coords_from_shapely_linear(g))
            return acc
        if hasattr(geom, "coords"):
            return list(geom.coords)
    except Exception as exc:
        logger.warning(
            "Не удалось разобрать geometry ребра (type=%s): %s",
            gt or type(geom).__name__,
            exc,
        )
    return []


def _first_value(val: Any) -> Any:
    if isinstance(val, list):
        return val[0] if val else None
    return val


def _raw_osm_surface_from_edge(d: Dict[str, Any]) -> str:
    """Тег surface из OSM (без эвристик): ``surface_osm`` или устаревшее ``surface``."""
    for key in ("surface_osm", "surface"):
        v = _first_value(d.get(key))
        if v is None:
            continue
        s = str(v).strip().lower()
        if s and s != "nan":
            return s
    return ""


def _effective_surface_from_edge(d: Dict[str, Any]) -> str:
    """Итоговая метка покрытия для статистики вдоль маршрута."""
    v = _first_value(d.get("surface_effective"))
    if v is not None:
        s = str(v).strip().lower()
        if s and s != "nan":
            return s
    r = _raw_osm_surface_from_edge(d)
    return r if r else "unknown"


class RouteResult:
    """Результат поиска маршрута (узлы + рёбра с ключами).

    Attributes:
        nodes: последовательность узлов маршрута.
        edges: список ``(u, v, key)`` — конкретные рёбра MultiDiGraph,
               по которым прошёл алгоритм (с учётом параллельных рёбер).
    """

    def __init__(
        self,
        nodes: List[int],
        edges: List[Tuple[int, int, int]],
        start_node: int,
        end_node: int,
    ) -> None:
        self.nodes = nodes
        self.edges = edges
        self.start_node = start_node
        self.end_node = end_node

    @property
    def edge_count(self) -> int:
        return len(self.edges)


class RouteService:
    """Поиск маршрутов и расчёт метрик."""

    @staticmethod
    def _collapse_multidigraph_to_digraph(
        G: nx.MultiDiGraph, weight_key: str
    ) -> nx.DiGraph:
        """``shortest_simple_paths`` в NetworkX не работает с MultiDiGraph.

        Между каждой парой (u, v) оставляем одно ребро с минимальным *weight_key*
        (как при выборе key в :meth:`_resolve_edge_keys`). Узлы без рёбер не копируются.
        """
        H = nx.DiGraph()
        for u, v, _k, d in G.edges(data=True, keys=True):
            w = coerce_edge_weight_numeric(d.get(weight_key), fallback=float("inf"))
            if not math.isfinite(w):
                continue
            if H.has_edge(u, v):
                if H.edges[u, v][weight_key] > w:
                    H.edges[u, v][weight_key] = w
            else:
                H.add_edge(u, v, **{weight_key: w})
        return H

    @staticmethod
    def _resolve_edge_keys(
        G: nx.MultiDiGraph,
        nodes: List[int],
        weight_key: str,
    ) -> List[Tuple[int, int, int]]:
        """Для каждой пары узлов определить key с минимальным весом."""
        edges = []
        for i in range(len(nodes) - 1):
            u, v = nodes[i], nodes[i + 1]
            best_key = min(
                G[u][v],
                key=lambda k: coerce_edge_weight_numeric(
                    G[u][v][k].get(weight_key), fallback=float("inf")
                ),
            )
            edges.append((u, v, best_key))
        return edges

    def find_route(
        self,
        G: nx.MultiDiGraph,
        start_coords: Tuple[float, float],
        end_coords: Tuple[float, float],
        weight_key: str,
    ) -> RouteResult:
        """Кратчайший маршрут. Бросает :exc:`RouteNotFoundError` если пути нет."""
        start = ox.distance.nearest_nodes(
            G, X=start_coords[1], Y=start_coords[0]
        )
        end = ox.distance.nearest_nodes(
            G, X=end_coords[1], Y=end_coords[0]
        )
        try:
            nodes = nx.shortest_path(
                G, source=start, target=end, weight=weight_key
            )
            edges = self._resolve_edge_keys(G, nodes, weight_key)
            return RouteResult(nodes, edges, start, end)
        except nx.NetworkXNoPath:
            raise RouteNotFoundError(weight_key)

    @staticmethod
    def find_next_simple_path_by_length(
        G: nx.MultiDiGraph,
        start_coords: Tuple[float, float],
        end_coords: Tuple[float, float],
        exclude_signatures: set,
        max_hops: int = 200,
        max_candidates: int = 12,
    ) -> Optional[RouteResult]:
        """Следующий простой путь по весу ``length``, не входящий в ``exclude_signatures``."""
        sn = ox.distance.nearest_nodes(
            G, X=start_coords[1], Y=start_coords[0]
        )
        en = ox.distance.nearest_nodes(
            G, X=end_coords[1], Y=end_coords[0]
        )
        try:
            H = RouteService._collapse_multidigraph_to_digraph(G, "length")
            gen = nx.shortest_simple_paths(H, sn, en, weight="length")
            next(gen)
            n = 0
            for nodes in gen:
                n += 1
                if n > max_candidates:
                    break
                if len(nodes) > max_hops:
                    continue
                edges = RouteService._resolve_edge_keys(G, nodes, "length")
                sig = tuple(edges)
                if sig in exclude_signatures:
                    continue
                return RouteResult(nodes, edges, sn, en)
        except (nx.NetworkXNoPath, StopIteration):
            pass
        return None

    def find_route_safe(
        self,
        G: nx.MultiDiGraph,
        start_coords: Tuple[float, float],
        end_coords: Tuple[float, float],
        weight_key: str,
    ) -> Optional[RouteResult]:
        """Поиск без исключения — возвращает ``None`` если пути нет."""
        try:
            return self.find_route(G, start_coords, end_coords, weight_key)
        except RouteNotFoundError:
            logger.warning("Маршрут не найден (weight='%s')", weight_key)
            return None

    # ------------------------------------------------------------------
    # Геометрия маршрута
    # ------------------------------------------------------------------

    @staticmethod
    def route_geometry(
        G: nx.MultiDiGraph, route: Optional["RouteResult"]
    ) -> list[list[float]]:
        """Полилиния маршрута по геометриям рёбер, а не только по узлам.

        Если у ребра есть ``geometry`` (Shapely LineString), берутся все
        его промежуточные точки — линия на карте точно повторяет дорогу.
        Дубликаты на стыках рёбер удаляются.

        Returns:
            ``[[lat, lon], ...]``
        """
        if route is None:
            return []

        coords: list[list[float]] = []
        for u, v, key in route.edges:
            data = G.edges[u, v, key]
            geom = data.get("geometry")
            edge_pts = _coords_from_shapely_linear(geom)
            if len(edge_pts) < 2:
                edge_pts = [
                    (G.nodes[u]["x"], G.nodes[u]["y"]),
                    (G.nodes[v]["x"], G.nodes[v]["y"]),
                ]
            else:
                first_node_y = G.nodes[u]["y"]
                first_node_x = G.nodes[u]["x"]
                start_lon, start_lat = edge_pts[0]
                end_lon, end_lat = edge_pts[-1]
                dist_to_start = (
                    (start_lat - first_node_y) ** 2
                    + (start_lon - first_node_x) ** 2
                )
                dist_to_end = (
                    (end_lat - first_node_y) ** 2
                    + (end_lon - first_node_x) ** 2
                )
                if dist_to_end < dist_to_start:
                    edge_pts = edge_pts[::-1]

            for lon, lat in edge_pts:
                pt = [round(lat, 7), round(lon, 7)]
                if not coords or coords[-1] != pt:
                    coords.append(pt)

        return coords

    # ------------------------------------------------------------------
    # Метрики маршрута
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_cost(
        G: nx.MultiDiGraph,
        route: Optional[RouteResult],
        weight_key: str,
    ) -> float:
        """Суммарная стоимость маршрута по указанному весу."""
        if route is None:
            return 0.0
        total = 0.0
        for i in range(route.edge_count):
            d = G.edges[route.edges[i]]
            w = d.get(weight_key)
            if w is None:
                logger.warning(
                    "У ребра нет веса %s — пропуск в сумме cost", weight_key
                )
                continue
            total += float(w)
        if not math.isfinite(total):
            logger.warning("Суммарный cost не конечен (%s), подставляем 0", total)
            return 0.0
        return total

    @staticmethod
    def calculate_length(
        G: nx.MultiDiGraph, route: Optional[RouteResult]
    ) -> float:
        """Длина маршрута в метрах."""
        if route is None:
            return 0.0
        return sum(
            G.edges[route.edges[i]].get("length", 0)
            for i in range(route.edge_count)
        )

    @staticmethod
    def green_stats(
        G: nx.MultiDiGraph, route: Optional[RouteResult]
    ) -> Dict[str, Any]:
        """Статистика озеленения: категории, средний % деревьев и травы."""
        zero = {
            "categories": {},
            "percent": 0.0,
            "avg_trees": 0.0,
            "avg_grass": 0.0,
        }
        if route is None:
            return zero

        counter: Counter = Counter()
        tw = gw = glen = tlen = 0.0

        for i in range(route.edge_count):
            d = G.edges[route.edges[i]]
            gt = d.get("green_type", "none")
            ln = d.get("length", 0)
            counter[gt] += 1
            tlen += ln
            tw += d.get("trees_percent", 0.0) * ln
            gw += d.get("grass_percent", 0.0) * ln
            if gt != "none":
                glen += ln

        return {
            "categories": dict(counter),
            "percent": (glen / tlen * 100) if tlen else 0.0,
            "avg_trees": (tw / tlen) if tlen else 0.0,
            "avg_grass": (gw / tlen) if tlen else 0.0,
        }

    @staticmethod
    def elevation_stats(
        G: nx.MultiDiGraph, route: Optional[RouteResult]
    ) -> Dict[str, float]:
        """Статистика рельефа: набор, спуск (интерполированные), макс/средний уклон."""
        zero = {
            "climb": 0.0,
            "descent": 0.0,
            "max_gradient_pct": 0.0,
            "avg_gradient_pct": 0.0,
        }
        if route is None:
            return zero

        climb = descent = max_g = 0.0
        grads: List[float] = []

        for i in range(route.edge_count):
            d = G.edges[route.edges[i]]
            g = d.get("gradient", 0.0)
            grads.append(abs(g))
            climb += d.get("edge_climb", 0.0)
            descent += d.get("edge_descent", 0.0)
            max_g = max(max_g, abs(g))

        return {
            "climb": climb,
            "descent": descent,
            "max_gradient_pct": max_g * 100,
            "avg_gradient_pct": (
                (sum(grads) / len(grads) * 100) if grads else 0.0
            ),
        }

    @staticmethod
    def surface_stats(
        G: nx.MultiDiGraph, route: Optional[RouteResult]
    ) -> Tuple[Counter, Counter]:
        """Подсчёт типов покрытий и дорог вдоль маршрута."""
        sc: Counter = Counter()
        hc: Counter = Counter()
        if route is None:
            return sc, hc

        for i in range(route.edge_count):
            d = G.edges[route.edges[i]]
            sc[_effective_surface_from_edge(d) or "N/A"] += 1
            hc[_first_value(d.get("highway", "N/A")) or "N/A"] += 1
        return sc, hc

    @staticmethod
    def route_weight_fallback_metrics(
        G: nx.MultiDiGraph, route: RouteResult, profile: ModeProfile
    ) -> Dict[str, float]:
        """Метрики fallback-коэффициентов (1.0) по рёбрам маршрута.

        Доли для UI и порогов предупреждений считаются по **сумме длин рёбер**
        (метры), согласованно с подписью «доля длины маршрута».

        ``na_surface_length_m`` — нет тега ``surface`` в OSM (пустой ``surface_osm``).
        ``unknown_surface_length_m`` — тег есть, но нет в справочнике профиля.
        Эвристика (``surface_effective``) не уменьшает ``na_*`` по отсутствию тега в OSM.
        """
        total_m = 0.0
        na_surf_m = 0.0
        unk_surf_m = 0.0
        na_hw_m = 0.0
        unk_hw_m = 0.0
        for i in range(route.edge_count):
            d = G.edges[route.edges[i]]
            ln = float(d.get("length", 0.0) or 0.0)
            total_m += ln
            raw_osm = _raw_osm_surface_from_edge(d)
            hw_raw = _first_value(d.get("highway"))
            h = (
                str(hw_raw).strip().lower()
                if hw_raw is not None and str(hw_raw).strip()
                else ""
            )
            if not raw_osm:
                na_surf_m += ln
            elif raw_osm not in profile.surface:
                unk_surf_m += ln
            if not h:
                na_hw_m += ln
            elif h not in profile.highway:
                unk_hw_m += ln
        return {
            "total_length_m": total_m,
            "na_surface_length_m": na_surf_m,
            "unknown_surface_length_m": unk_surf_m,
            "na_highway_length_m": na_hw_m,
            "unknown_highway_length_m": unk_hw_m,
        }

    @staticmethod
    def elevation_profile(
        G: nx.MultiDiGraph, route: Optional[RouteResult]
    ) -> Dict[str, float]:
        """Профиль рельефа относительно стартовой точки.

        Returns:
            ``max_above`` — максимальный подъём от старта (м),
            ``max_below`` — максимальный спуск от старта (м, ≤0),
            ``end_diff`` — перепад старт→финиш (м).
        """
        zero = {"max_above": 0.0, "max_below": 0.0, "end_diff": 0.0}
        if route is None:
            return zero

        cumulative = 0.0
        max_above = 0.0
        max_below = 0.0

        for i in range(route.edge_count):
            d = G.edges[route.edges[i]]
            cumulative += d.get("elevation_diff", 0.0)
            max_above = max(max_above, cumulative)
            max_below = min(max_below, cumulative)

        return {
            "max_above": max_above,
            "max_below": max_below,
            "end_diff": cumulative,
        }

    @staticmethod
    def stairs_count(
        G: nx.MultiDiGraph, route: Optional[RouteResult]
    ) -> Dict[str, Any]:
        """Количество лестниц (highway=steps) и их суммарная длина."""
        if route is None:
            return {"count": 0, "length": 0.0}

        count = 0
        length = 0.0
        for i in range(route.edge_count):
            d = G.edges[route.edges[i]]
            hw = _first_value(d.get("highway", ""))
            if hw == "steps":
                count += 1
                length += d.get("length", 0)

        return {"count": count, "length": length}

    @staticmethod
    def elevation_profile_data(
        G: nx.MultiDiGraph, route: Optional[RouteResult]
    ) -> List[Tuple[float, float]]:
        """Точки (кумулятивная дистанция м, относительная высота м).

        Первая точка — (0, 0); реальную абсолютную высоту добавляет
        вызывающий код (через ElevationService).
        """
        if route is None:
            return []
        points: List[Tuple[float, float]] = [(0.0, 0.0)]
        cum_dist = 0.0
        cum_elev = 0.0
        for i in range(route.edge_count):
            d = G.edges[route.edges[i]]
            cum_dist += d.get("length", 0)
            cum_elev += d.get("elevation_diff", 0.0)
            points.append((cum_dist, cum_elev))
        return points

    @staticmethod
    def estimate_time(
        G: nx.MultiDiGraph,
        route: Optional[RouteResult],
        profile: ModeProfile,
    ) -> float:
        """Оценка времени в пути (секунды).

        Учитывает базовую скорость профиля, уклон и лестницы.
        """
        if route is None:
            return 0.0

        base = profile.base_speed_ms
        total = 0.0

        for i in range(route.edge_count):
            d = G.edges[route.edges[i]]
            length = d.get("length", 0)
            gradient = d.get("gradient", 0.0)
            hw = _first_value(d.get("highway", ""))

            if hw == "steps":
                speed = profile.stairs_speed_ms
            elif gradient > 0:
                speed = base * max(0.15, 1.0 - gradient * profile.uphill_penalty)
            elif gradient < 0:
                speed = base * min(1.5, 1.0 + abs(gradient) * profile.downhill_bonus)
            else:
                speed = base

            if speed > 0:
                total += length / speed

        return total

    # ------------------------------------------------------------------
    # GeoJSON-слои для карты (сегменты маршрута)
    # ------------------------------------------------------------------

    #: Опасные / напряжённые типы дорог (автотрафик).
    PROBLEMATIC_HIGHWAYS = frozenset(
        {
            "motorway",
            "motorway_link",
            "trunk",
            "trunk_link",
            "primary",
            "primary_link",
            "secondary",
            "secondary_link",
        }
    )
    #: Порог крутизны (доля, 0.10 = 10 %) для слоя «проблемные».
    STEEP_GRADIENT_ABS = 0.10

    @staticmethod
    def _edge_linestring_coords(
        G: nx.MultiDiGraph, u: int, v: int, key: int
    ) -> List[List[float]]:
        """Координаты линии ребра ``[[lon, lat], ...]`` для GeoJSON."""
        data = G.edges[u, v, key]
        geom = data.get("geometry")
        if geom is not None:
            pts = _coords_from_shapely_linear(geom)
            if len(pts) >= 2:
                return [list(c) for c in pts]
        return [
            [G.nodes[u]["x"], G.nodes[u]["y"]],
            [G.nodes[v]["x"], G.nodes[v]["y"]],
        ]

    @staticmethod
    def _map_feature(coords: List[List[float]], props: Dict[str, Any]) -> dict:
        return {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": props,
        }

    @staticmethod
    def build_route_map_layers(
        G: nx.MultiDiGraph,
        route: Optional[RouteResult],
    ) -> Dict[str, Any]:
        """Собрать GeoJSON FeatureCollection по категориям сегментов маршрута.

        Returns:
            dict с ключами ``greenery``, ``stairs``, ``problematic``, ``na_surface`` —
            каждый — полноценный GeoJSON FeatureCollection (сериализуемый в JSON).
        """
        def _empty_fc() -> dict:
            return {"type": "FeatureCollection", "features": []}

        if route is None:
            return {
                "greenery": _empty_fc(),
                "stairs": _empty_fc(),
                "problematic": _empty_fc(),
                "na_surface": _empty_fc(),
            }

        greenery_f: List[dict] = []
        stairs_f: List[dict] = []
        prob_f: List[dict] = []
        na_f: List[dict] = []

        for u, v, key in route.edges:
            d = G.edges[u, v, key]
            coords = RouteService._edge_linestring_coords(G, u, v, key)
            hw = _first_value(d.get("highway", "")) or ""
            hw = str(hw).lower()
            surf_raw = _raw_osm_surface_from_edge(d)
            surf = surf_raw or "N/A"
            gt = str(d.get("green_type", "none") or "none").lower()
            trees = float(d.get("trees_percent", 0.0) or 0.0)
            grass = float(d.get("grass_percent", 0.0) or 0.0)
            grad = float(d.get("gradient", 0.0) or 0.0)
            ln = float(d.get("length", 0.0) or 0.0)

            base = {
                "highway": hw,
                "length_m": round(ln, 1),
                "gradient_pct": round(abs(grad) * 100, 1),
            }

            if gt != "none" or trees >= 2.0 or grass >= 2.0:
                greenery_f.append(
                    RouteService._map_feature(
                        coords,
                        {
                            **base,
                            "green_type": gt,
                            "trees_pct": round(trees, 1),
                            "grass_pct": round(grass, 1),
                        },
                    )
                )

            if hw == "steps":
                stairs_f.append(
                    RouteService._map_feature(coords, {**base, "kind": "steps"})
                )

            if hw != "steps":
                reasons: List[str] = []
                if abs(grad) >= RouteService.STEEP_GRADIENT_ABS:
                    reasons.append("steep")
                if hw in RouteService.PROBLEMATIC_HIGHWAYS:
                    reasons.append("traffic_class")
                if reasons:
                    prob_f.append(
                        RouteService._map_feature(
                            coords,
                            {**base, "reasons": ",".join(reasons)},
                        )
                    )

            if surf == "N/A" or surf == "":
                na_f.append(
                    RouteService._map_feature(
                        coords,
                        {**base, "surface": surf or "N/A"},
                    )
                )

        def _fc(features: List[dict]) -> dict:
            return {"type": "FeatureCollection", "features": features}

        return {
            "greenery": _fc(greenery_f),
            "stairs": _fc(stairs_f),
            "problematic": _fc(prob_f),
            "na_surface": _fc(na_f),
        }
