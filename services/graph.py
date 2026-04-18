"""Сервис построения и взвешивания дорожного графа."""

import logging
import math
import os
from typing import Any, List, Optional, Sequence, Tuple

import geopandas as gpd
import pandas as pd
import requests
import networkx as nx
import numpy as np
import osmnx as ox
from shapely.geometry import box

from ..config import (
    DEFAULT_COEFFICIENT,
    OSM_HIGHWAY_FILTER,
    PROFILES,
    TIME_SLOTS,
    Settings,
)
from ..exceptions import OverpassUnavailableError
from ..metrics import inc_overpass_subretry
from .elevation import ElevationService
from .green import GreenAnalyzer
from .heat import (
    edge_bearing_deg_from_geom,
    exposure_units_detailed_all_slots,
    heat_cost_for_edge,
    thermal_edge_features,
)
from .policy_data import load_stress_policy
from .stress import (
    intersection_stress_score,
    lts_from_osm_tags,
    stress_costs_for_edge,
)
from .retry import sleep_backoff
from .surface_resolve import apply_surface_resolution

logger = logging.getLogger(__name__)

# Теги OSM для стресса/тепла (каньон по width)
_OSM_STRESS_TAG_KEYS = (
    "highway",
    "lanes",
    "maxspeed",
    "cycleway",
    "cycleway:left",
    "cycleway:right",
    "cycleway:both",
    "sidewalk",
    "junction",
    "crossing",
    "traffic_signals",
    "segregated",
    "bicycle",
    "width",
    "foot",
)


def _first_value(val: Any) -> Any:
    """Извлекает первый элемент, если *val* — список."""
    if isinstance(val, list):
        return val[0] if val else None
    return val


def _segmented_base_weight_long_edge(
    profile: Any,
    elevs: Sequence[float],
    edge_len: float,
) -> float:
    """Сумма базового веса по подсегментам длинного ребра (та же формула, что в цикле)."""
    n_pts = len(elevs)
    if n_pts < 2 or edge_len <= 0:
        return float("nan")
    sub_len = edge_len / (n_pts - 1)
    seg_base = 0.0
    for j in range(n_pts - 1):
        g = (elevs[j + 1] - elevs[j]) / sub_len
        g = max(-profile.max_gradient, min(profile.max_gradient, g))
        if g >= 0:
            seg_eff = profile.nu + g
        else:
            seg_eff = max(profile.nu - abs(g), profile.min_descent_coeff)
        seg_base += profile.mg * sub_len * seg_eff
    return seg_base


class GraphBuilder:
    """Загрузка дорожного графа OSM и расчёт весов рёбер.

    Веса рассчитываются отдельно для каждого профиля участника
    движения (велосипедист / пешеход) с учётом:
    * физической модели (масса, сопротивление, уклон);
    * типа покрытия (``surface``);
    * типа дороги (``highway``);
    * степени озеленения (спутниковые данные).
    """

    def __init__(
        self,
        elevation_service: ElevationService,
        green_analyzer: GreenAnalyzer,
        settings: Settings,
    ) -> None:
        self._elevation = elevation_service
        self._green = green_analyzer
        self._settings = settings
        self._init_osmnx()

    def _init_osmnx(self) -> None:
        s = self._settings
        cache_dir = s.osmnx_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        ox.settings.cache_folder = cache_dir
        # Overpass в [out:json][timeout:N] допускает только целое N; float → "180.0" и 400 Bad Request.
        _rt = int(max(1, round(float(s.osm_requests_timeout))))
        ox.settings.requests_timeout = _rt
        ox.settings.overpass_rate_limit = s.osm_overpass_rate_limit
        chain: List[str] = s.resolved_overpass_endpoints()
        ox.settings.overpass_url = chain[0]
        for tag in (
            "surface",
            "tracktype",
            "lanes",
            "maxspeed",
            "cycleway",
            "cycleway:left",
            "cycleway:right",
            "cycleway:both",
            "sidewalk",
            "junction",
            "crossing",
            "traffic_signals",
            "segregated",
            "bicycle",
            "width",
            "foot",
            "lit",
        ):
            if tag not in ox.settings.useful_tags_way:
                ox.settings.useful_tags_way += [tag]
        logger.info(
            "OSMnx: Overpass chain (последовательно) %s · max HTTP timeout=%ss · "
            "first_sub_attempt_short_timeout=%ss · rate_limit=%s",
            chain,
            _rt,
            float(s.osm_overpass_first_attempt_timeout),
            ox.settings.overpass_rate_limit,
        )

    # ------------------------------------------------------------------
    # Загрузка и преобразование
    # ------------------------------------------------------------------

    @staticmethod
    def create_bbox(
        start: Tuple[float, float],
        end: Tuple[float, float],
        buffer: float,
    ) -> Any:
        """Bounding box для загрузки данных (buffer — градусы со всех сторон)."""
        min_lat = min(start[0], end[0]) - buffer
        min_lon = min(start[1], end[1]) - buffer
        max_lat = max(start[0], end[0]) + buffer
        max_lon = max(start[1], end[1]) + buffer
        return box(min_lon, min_lat, max_lon, max_lat)

    def create_bbox_for_corridor(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        buffer_meters: Any = None,
    ) -> Any:
        """Прямоугольник коридора: метры (CORRIDOR_BUFFER_METERS) или градусы (BUFFER).

        buffer_meters — переопределение половины запаса по широте/долготе (м);
        для авторасширения коридора при NO_PATH.
        """
        s = self._settings
        if s.corridor_buffer_meters > 0 or buffer_meters is not None:
            m = float(buffer_meters if buffer_meters is not None else s.corridor_buffer_meters)
            mid_lat = (start[0] + end[0]) / 2.0
            dlat = m / 111_000.0
            cos_lat = math.cos(math.radians(mid_lat))
            dlon = m / (111_000.0 * max(0.2, abs(cos_lat)))
            min_lat = min(start[0], end[0]) - dlat
            min_lon = min(start[1], end[1]) - dlon
            max_lat = max(start[0], end[0]) + dlat
            max_lon = max(start[1], end[1]) + dlon
            return box(min_lon, min_lat, max_lon, max_lat)
        if buffer_meters is not None:
            raise ValueError(
                "buffer_meters override только при CORRIDOR_BUFFER_METERS > 0"
            )
        return self.create_bbox(start, end, s.buffer)

    def load(self, bbox: Any) -> nx.MultiDiGraph:
        """Загрузить дорожный граф из OpenStreetMap (цепочка Overpass без параллельных запросов)."""
        s = self._settings
        endpoints = s.resolved_overpass_endpoints()
        main_timeout = int(max(1, round(float(s.osm_requests_timeout))))
        first_t = float(s.osm_overpass_first_attempt_timeout)

        logger.info("Загрузка дорог из OSM… endpoints=%d", len(endpoints))
        last_exc: Optional[Exception] = None
        per_ep = max(1, int(getattr(s, "overpass_per_endpoint_retries", 3)))
        rb = float(s.http_retry_base_delay_sec)
        rmax = float(s.http_retry_max_delay_sec)
        rjit = float(s.http_retry_jitter)
        for i, url in enumerate(endpoints):
            ox.settings.overpass_url = url
            for attempt in range(1, per_ep + 1):
                # Короткий timeout на первой sub-попытке к каждому endpoint (и при одном зеркале),
                # дальше — полный OSM_REQUESTS_TIMEOUT (см. OSM_OVERPASS_FIRST_TIMEOUT).
                if first_t > 0 and attempt == 1:
                    tsec = int(max(1, round(min(first_t, float(main_timeout)))))
                else:
                    tsec = main_timeout
                ox.settings.requests_timeout = tsec
                logger.info(
                    "Overpass endpoint %d/%d sub_attempt %d/%d url=%s timeout=%ss",
                    i + 1,
                    len(endpoints),
                    attempt,
                    per_ep,
                    url,
                    tsec,
                )
                try:
                    if attempt > 1:
                        inc_overpass_subretry()
                        logger.info(
                            "overpass_retry endpoint=%s sub_attempt=%d/%d",
                            url,
                            attempt,
                            per_ep,
                        )
                        sleep_backoff(
                            attempt - 1,
                            base_sec=rb,
                            max_sec=rmax,
                            jitter=rjit,
                        )
                    G = ox.graph_from_polygon(
                        bbox,
                        custom_filter=OSM_HIGHWAY_FILTER,
                        simplify=False,
                        retain_all=True,
                    )
                except requests.exceptions.RequestException as exc:
                    last_exc = exc
                    logger.warning(
                        "Overpass сбой endpoint=%s sub_attempt=%d/%d: %s",
                        url,
                        attempt,
                        per_ep,
                        exc,
                    )
                    # Повторная попытка на том же хосте редко чинит TLS/SSL — сразу следующее зеркало.
                    if isinstance(exc, requests.exceptions.SSLError):
                        break
                    if attempt >= per_ep:
                        break
                    continue
                else:
                    logger.info(
                        "Overpass успех endpoint=%s · узлов %d, рёбер %d",
                        url,
                        G.number_of_nodes(),
                        G.number_of_edges(),
                    )
                    return G

        logger.error(
            "Overpass: все endpoint'ы исчерпаны после %s",
            last_exc,
        )
        if last_exc is not None:
            raise OverpassUnavailableError() from last_exc
        raise OverpassUnavailableError()

    @staticmethod
    def to_geodataframe(G: nx.MultiDiGraph) -> gpd.GeoDataFrame:
        """Конвертировать граф в ``GeoDataFrame`` рёбер."""
        edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
        return edges.explode(index_parts=False)

    @staticmethod
    def check_completeness(edges_gdf: gpd.GeoDataFrame) -> dict:
        """Анализ полноты данных OSM для рёбер графа."""
        total = len(edges_gdf)

        def _has_data(col: str) -> int:
            if col not in edges_gdf.columns:
                return 0
            vals = edges_gdf[col].apply(_first_value)
            return int((vals.notna() & (vals.astype(str) != "")).sum())

        has_surface = _has_data("surface")
        has_name = _has_data("name")
        has_maxspeed = _has_data("maxspeed")

        highway_dist = {}
        if "highway" in edges_gdf.columns:
            highway_dist = dict(
                edges_gdf["highway"]
                .apply(_first_value)
                .value_counts()
                .items()
            )

        surface_dist = {}
        if "surface" in edges_gdf.columns:
            vals = edges_gdf["surface"].apply(_first_value).dropna()
            surface_dist = dict(vals.value_counts().items())

        lengths = edges_gdf["length"].values
        has_length = int((np.isfinite(lengths) & (lengths > 0)).sum())
        zero_length = int((lengths == 0).sum())

        has_elev = 0
        zero_elev = 0
        if "elevation_diff" in edges_gdf.columns:
            ed = edges_gdf["elevation_diff"].values
            has_elev = int(np.isfinite(ed).sum())
            zero_elev = int((ed == 0).sum())

        return {
            "total_edges": total,
            "total_length_km": round(float(lengths.sum()) / 1000, 1),
            "avg_edge_length_m": round(float(lengths.mean()), 1),
            "has_length": has_length,
            "zero_length": zero_length,
            "has_elevation": has_elev,
            "zero_elevation": zero_elev,
            "has_surface": has_surface,
            "has_name": has_name,
            "has_maxspeed": has_maxspeed,
            "surface_pct": round(has_surface / total * 100, 1) if total else 0,
            "name_pct": round(has_name / total * 100, 1) if total else 0,
            "highway_dist": highway_dist,
            "surface_dist": surface_dist,
        }

    # ------------------------------------------------------------------
    # Расчёт весов
    # ------------------------------------------------------------------

    def enrich_edges_base_physical(
        self, edges_gdf: gpd.GeoDataFrame
    ) -> Tuple[gpd.GeoDataFrame, Any]:
        """Базовое обогащение: длина (OSM), высоты, surface_effective, градиенты.

        Без спутниковой зелени — она добавляется в :meth:`_apply_green_enrichment`.
        """
        n = len(edges_gdf)
        geom_arr = edges_gdf.geometry.values
        len_arr = edges_gdf["length"].values.astype(np.float64, copy=False)
        ediff = np.empty(n, dtype=np.float64)
        eclimb = np.empty(n, dtype=np.float64)
        edesc = np.empty(n, dtype=np.float64)
        elev_profiles = np.empty(n, dtype=object)
        for i in range(n):
            t = self._elevation.get_edge_elevation(geom_arr[i], float(len_arr[i]))
            ediff[i] = t[0]
            eclimb[i] = t[1]
            edesc[i] = t[2]
            elev_profiles[i] = t[3]

        edges_gdf["elevation_diff"] = ediff
        edges_gdf["edge_climb"] = eclimb
        edges_gdf["edge_descent"] = edesc

        long_mask = len_arr > 30 * 1.5
        n_long = int(long_mask.sum())
        logger.info(
            "Высоты: %d рёбер интерполировано (~30м шаг), "
            "%d коротких (только нач/кон)",
            n_long,
            n - n_long,
        )

        edges_gdf = apply_surface_resolution(edges_gdf)

        length = edges_gdf["length"].values
        elev = edges_gdf["elevation_diff"].values
        raw_gradient = np.where(length > 0, elev / length, 0)
        edges_gdf["gradient_raw"] = raw_gradient
        display_max = max(p.max_gradient for p in PROFILES)
        clipped_gradient = np.clip(raw_gradient, -display_max, display_max)
        edges_gdf["gradient"] = clipped_gradient
        edges_gdf["gradient_percent"] = clipped_gradient * 100

        return edges_gdf, elev_profiles

    def _apply_green_enrichment(
        self,
        edges_gdf: gpd.GeoDataFrame,
        *,
        skip_satellite_green: bool,
    ) -> gpd.GeoDataFrame:
        """Спутник или заглушки: ``trees_*``, ``grass_*``, ``green_coeff``, ``green_type``."""
        use_satellite = (
            self._green._tiles.pil_available
            and not self._settings.disable_satellite_green
            and not skip_satellite_green
        )
        if use_satellite:
            return self._green.calculate_satellite_batch(edges_gdf)
        if skip_satellite_green:
            logger.info(
                "Запрос без спутниковой зелени — тайлы не загружаются, заглушки."
            )
        elif self._settings.disable_satellite_green:
            logger.info(
                "DISABLE_SATELLITE_GREEN=true — озеленение по снимкам отключено, заглушки."
            )
        return self._green._fill_empty_green(edges_gdf)

    def _compute_profile_weights_full(
        self,
        edges_gdf: gpd.GeoDataFrame,
        elev_profiles: Any,
    ) -> None:
        """Только ``weight_*_full`` (энергетическая модель без озеленения)."""
        length = edges_gdf["length"].values
        raw_gradient = np.where(
            length > 0,
            edges_gdf["elevation_diff"].values / length,
            0.0,
        )
        clipped_gradient = edges_gdf["gradient"].values
        long_mask = edges_gdf["length"].values > 30 * 1.5
        long_indices = np.where(long_mask)[0]

        n_edges = len(edges_gdf)
        surf_base = None
        surf_first = None
        if "surface_effective" in edges_gdf.columns:
            surf_base = (
                edges_gdf["surface_effective"]
                .astype(str)
                .str.strip()
                .str.lower()
            )
        elif "surface" in edges_gdf.columns:
            surf_first = [_first_value(x) for x in edges_gdf["surface"].values]

        if "highway" in edges_gdf.columns:
            hw_first = [_first_value(x) for x in edges_gdf["highway"].values]
        else:
            hw_first = None

        for profile in PROFILES:
            gradient = np.clip(
                raw_gradient, -profile.max_gradient, profile.max_gradient
            )

            if surf_base is not None:
                surf = (
                    surf_base.map(profile.surface)
                    .fillna(DEFAULT_COEFFICIENT)
                    .values.astype(np.float64, copy=False)
                )
            elif surf_first is not None:
                surf = (
                    pd.Series(surf_first)
                    .map(profile.surface)
                    .fillna(DEFAULT_COEFFICIENT)
                    .values.astype(np.float64, copy=False)
                )
            else:
                surf = np.full(n_edges, DEFAULT_COEFFICIENT)

            if hw_first is not None:
                hway = (
                    pd.Series(hw_first)
                    .map(profile.highway)
                    .fillna(DEFAULT_COEFFICIENT)
                    .values.astype(np.float64, copy=False)
                )
            else:
                hway = np.full(n_edges, DEFAULT_COEFFICIENT)

            eff = np.where(
                gradient >= 0,
                profile.nu + gradient,
                np.maximum(
                    profile.nu - np.abs(gradient), profile.min_descent_coeff
                ),
            )
            base = profile.mg * length * eff

            for idx in long_indices:
                elevs = elev_profiles[idx]
                edge_len = float(length[idx])
                if elevs is None or edge_len <= 0:
                    continue
                seg = _segmented_base_weight_long_edge(profile, elevs, edge_len)
                if not np.isnan(seg):
                    base[idx] = seg

            w_full = np.maximum(base * surf * hway, 0)
            edges_gdf[f"weight_{profile.key}_full"] = w_full

            logger.info(
                "  %s %s: вес full=%.1f (средний по ребру)",
                profile.icon,
                profile.label,
                w_full.mean(),
            )

        logger.info(
            "Уклоны: средний %.1f%%, макс %.1f%%",
            np.abs(clipped_gradient).mean() * 100,
            np.abs(clipped_gradient).max() * 100,
        )

    @staticmethod
    def _log_green_edge_stats(edges_gdf: gpd.GeoDataFrame) -> None:
        if "green_type" not in edges_gdf.columns:
            return
        gc = (edges_gdf["green_type"] != "none").sum()
        logger.info(
            "Озеленённых рёбер: %d / %d (%.1f%%)",
            gc,
            len(edges_gdf),
            100 * gc / max(len(edges_gdf), 1),
        )

    def recompute_green_weights_only(self, edges_gdf: gpd.GeoDataFrame) -> None:
        """Пересчитать только ``weight_*_green`` при неизменных ``weight_*_full``."""
        # После load_graphml веса часто строки (XML) — без float ломается w_full * eff_green
        green = (
            pd.to_numeric(edges_gdf["green_coeff"], errors="coerce")
            .fillna(1.0)
            .to_numpy(dtype=np.float64)
        )
        for profile in PROFILES:
            w_full = pd.to_numeric(
                edges_gdf[f"weight_{profile.key}_full"],
                errors="coerce",
            ).to_numpy(dtype=np.float64)
            eff_green = np.clip(
                1.0 + (green - 1.0) * profile.green_sensitivity, 0.2, 1.5
            )
            edges_gdf[f"weight_{profile.key}_green"] = np.maximum(
                w_full * eff_green, 0
            )

    def enrich_heat_stress_features(self, edges_gdf: gpd.GeoDataFrame) -> None:
        """Ориентация ребра, LTS/стресс, тепловые штрафы и экспозиция по слотам."""
        n = len(edges_gdf)
        if n == 0:
            return
        geom_arr = edges_gdf.geometry.values
        len_arr = (
            pd.to_numeric(edges_gdf["length"], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )
        trees = (
            pd.to_numeric(edges_gdf.get("trees_percent"), errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
            if "trees_percent" in edges_gdf.columns
            else np.zeros(n, dtype=np.float64)
        )
        grass = (
            pd.to_numeric(edges_gdf.get("grass_percent"), errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
            if "grass_percent" in edges_gdf.columns
            else np.zeros(n, dtype=np.float64)
        )

        bearing = np.zeros(n, dtype=np.float64)
        lts_arr = np.ones(n, dtype=np.float64)
        stress_seg = np.zeros(n, dtype=np.float64)
        stress_int = np.zeros(n, dtype=np.float64)
        stress_c = np.zeros(n, dtype=np.float64)
        int_score_arr = np.zeros(n, dtype=np.float64)
        o_sky = np.zeros(n, dtype=np.float64)
        v_sh = np.zeros(n, dtype=np.float64)
        b_sh = np.zeros(n, dtype=np.float64)
        use_proxy = np.zeros(n, dtype=np.int8)
        heat_by_slot = {s.key: np.zeros(n, dtype=np.float64) for s in TIME_SLOTS}
        exp_by_slot = {s.key: np.zeros(n, dtype=np.float64) for s in TIME_SLOTS}

        pol_stress = load_stress_policy()
        use_fallback_green = bool(self._settings.disable_satellite_green)

        for i in range(n):
            geom = geom_arr[i]
            brg = float(edge_bearing_deg_from_geom(geom))
            bearing[i] = brg
            row = edges_gdf.iloc[i]
            tags = {
                k: row[k]
                for k in _OSM_STRESS_TAG_KEYS
                if k in edges_gdf.columns
            }
            if "highway" not in tags and "highway" in edges_gdf.columns:
                tags["highway"] = row.get("highway")

            O, V, B, _ = thermal_edge_features(
                brg, float(trees[i]), float(grass[i]), tags
            )
            o_sky[i] = O
            v_sh[i] = V
            b_sh[i] = B

            fb = use_fallback_green or (
                float(trees[i]) < 0.5 and float(grass[i]) < 0.5
            )
            use_proxy[i] = 1 if fb else 0

            lv = float(lts_from_osm_tags(tags, pol_stress))
            lts_arr[i] = lv
            ln = float(len_arr[i])
            isc = float(intersection_stress_score(tags, pol_stress))
            int_score_arr[i] = isc
            seg, inter, tot = stress_costs_for_edge(
                ln, lv, isc, pol_stress
            )
            stress_seg[i] = seg
            stress_int[i] = inter
            stress_c[i] = tot

            exps = exposure_units_detailed_all_slots(
                brg,
                float(trees[i]),
                float(grass[i]),
                tags,
                use_fallback=fb,
            )

            for slot in TIME_SLOTS:
                eu = float(exps.get(slot.key, 0.0))
                exp_by_slot[slot.key][i] = eu
                heat_by_slot[slot.key][i] = heat_cost_for_edge(ln, eu)

        edges_gdf["edge_bearing_deg"] = bearing
        edges_gdf["thermal_open_sky_share"] = o_sky
        edges_gdf["thermal_vegetation_shade_share"] = v_sh
        edges_gdf["thermal_building_shade_share"] = b_sh
        edges_gdf["thermal_use_proxy"] = use_proxy
        edges_gdf["stress_lts"] = lts_arr
        edges_gdf["stress_intersection_score"] = int_score_arr
        edges_gdf["stress_segment_cost"] = stress_seg
        edges_gdf["stress_intersection_cost"] = stress_int
        edges_gdf["stress_cost"] = stress_c
        for slot in TIME_SLOTS:
            edges_gdf[f"heat_{slot.key}"] = heat_by_slot[slot.key]
            edges_gdf[f"heat_exposure_{slot.key}"] = exp_by_slot[slot.key]

    def upgrade_edges_satellite_weights(
        self, edges_gdf: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """Green-addon: только спутник + ``weight_*_green``; база и ``weight_*_full`` без изменений."""
        import time as _time

        t0 = _time.perf_counter()
        logger.info(
            "Green-addon (без OSM/высот/surface/full): спутник и weight_*_green, рёбер=%d",
            len(edges_gdf),
        )
        use_satellite = (
            self._green._tiles.pil_available
            and not self._settings.disable_satellite_green
        )
        if use_satellite:
            edges_gdf = self._green.calculate_satellite_batch(edges_gdf)
        else:
            edges_gdf = self._green._fill_empty_green(edges_gdf)
        self.recompute_green_weights_only(edges_gdf)
        self.enrich_heat_stress_features(edges_gdf)
        logger.info(
            "Доначитка зелени: готово за %.2f с",
            _time.perf_counter() - t0,
        )
        return edges_gdf

    def calculate_weights(
        self,
        edges_gdf: gpd.GeoDataFrame,
        *,
        skip_satellite_green: bool = False,
    ) -> gpd.GeoDataFrame:
        """Рассчитать веса рёбер для всех профилей (велосипедист / пешеход).

        Pipeline: база (высоты, surface, ``weight_*_full``) → зелёное обогащение →
        ``weight_*_green``. Так green-addon при phase1→full не трогает базу.

        skip_satellite_green: не грузить тайлы — заглушки зелени (запрос без «зелёного» маршрута).
        """
        import time as _time

        t_all = _time.perf_counter()
        logger.info("Расчёт весов рёбер (базовая фаза + зелёная фаза)...")
        t0 = _time.perf_counter()
        edges_gdf, elev_profiles = self.enrich_edges_base_physical(edges_gdf)
        t1 = _time.perf_counter()
        self._compute_profile_weights_full(edges_gdf, elev_profiles)
        t2 = _time.perf_counter()
        edges_gdf = self._apply_green_enrichment(
            edges_gdf, skip_satellite_green=skip_satellite_green
        )
        t3 = _time.perf_counter()
        self.recompute_green_weights_only(edges_gdf)
        self.enrich_heat_stress_features(edges_gdf)
        t4 = _time.perf_counter()
        for profile in PROFILES:
            logger.info(
                "  %s %s: green=%.1f (среднее по ребру)",
                profile.icon,
                profile.label,
                edges_gdf[f"weight_{profile.key}_green"].values.mean(),
            )
        self._log_green_edge_stats(edges_gdf)
        logger.info(
            "Расчёт весов: elev+surface %.2fs | weight_*_full %.2fs | "
            "зелёные данные %.2fs | weight_*_green %.2fs | всего %.2fs (рёбер=%d)",
            t1 - t0,
            t2 - t1,
            t3 - t2,
            t4 - t3,
            t4 - t_all,
            len(edges_gdf),
        )
        return edges_gdf

    # ------------------------------------------------------------------
    # Применение весов к графу
    # ------------------------------------------------------------------

    @staticmethod
    def apply_weights(
        G: nx.MultiDiGraph, edges_gdf: gpd.GeoDataFrame
    ) -> nx.MultiDiGraph:
        """Перенести рассчитанные веса обратно в граф."""
        edges_gdf = edges_gdf.reset_index()

        base_cols = [
            "elevation_diff",
            "edge_climb",
            "edge_descent",
            "green_coeff",
            "green_type",
            "gradient",
            "gradient_raw",
            "trees_percent",
            "grass_percent",
            "green_percent",
            "surface_osm",
            "surface_effective",
            "edge_bearing_deg",
            "thermal_open_sky_share",
            "thermal_vegetation_shade_share",
            "thermal_building_shade_share",
            "thermal_use_proxy",
            "stress_lts",
            "stress_intersection_score",
            "stress_segment_cost",
            "stress_intersection_cost",
            "stress_cost",
        ]
        for s in TIME_SLOTS:
            base_cols.append(f"heat_{s.key}")
            base_cols.append(f"heat_exposure_{s.key}")
        for p in PROFILES:
            base_cols.extend(
                [f"weight_{p.key}_full", f"weight_{p.key}_green"]
            )

        avail = [c for c in base_cols if c in edges_gdf.columns]
        lookup = edges_gdf.set_index(["u", "v", "key"])[avail].to_dict(
            "index"
        )

        weight_cols = {c for c in avail if c.startswith("weight_")}

        heat_exp_cols = {c for c in avail if c.startswith("heat_exposure_")}
        heat_cost_cols = {c for c in avail if c.startswith("heat_") and not c.startswith("heat_exposure_")}

        for u, v, key, data in G.edges(keys=True, data=True):
            w = lookup.get((u, v, key), {})
            for col in avail:
                if col in w:
                    data[col] = w[col]
                elif col in weight_cols:
                    data[col] = data.get("length", 1)
                elif col == "green_coeff":
                    data[col] = 1.0
                elif col == "green_type":
                    data[col] = "none"
                elif col == "surface_osm":
                    data[col] = w.get(col, "") or ""
                elif col == "surface_effective":
                    data[col] = w.get(col, "unknown") or "unknown"
                elif col == "stress_lts":
                    data[col] = 1.5
                elif col == "thermal_use_proxy":
                    data[col] = 1
                elif col == "stress_intersection_score":
                    data[col] = 0.0
                elif col in (
                    "thermal_open_sky_share",
                    "thermal_vegetation_shade_share",
                    "thermal_building_shade_share",
                    "stress_segment_cost",
                    "stress_intersection_cost",
                ):
                    data[col] = 0.0
                elif col in heat_exp_cols:
                    data[col] = 0.0
                elif col in heat_cost_cols:
                    data[col] = 0.0
                elif col == "stress_cost":
                    data[col] = 0.0
                elif col == "edge_bearing_deg":
                    data[col] = 0.0
                else:
                    data[col] = 0.0

        return G
