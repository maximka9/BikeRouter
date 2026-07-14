"""Дисковый кэш взвешенных графов коридора (OSMnx GraphML).

Ключ: bbox (опционально расширенный до сетки ``CORRIDOR_CACHE_GRID_STEP_DEG``) +
отпечаток параметров построения весов и спутниковой зелени —
см. :func:`corridor_graph_cache_fingerprint`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Optional, Tuple

import networkx as nx
import osmnx as ox

from ..config import Settings, routing_engine_cache_fingerprint

logger = logging.getLogger(__name__)


def corridor_graph_cache_fingerprint(settings: Settings) -> str:
    """SHA-256 от настроек, влияющих на веса рёбер (кроме самого bbox)."""
    payload = {
        "algo_fp": routing_engine_cache_fingerprint(),
        "analyze_corridor": settings.analyze_corridor,
        "buffer_deg": settings.buffer,
        "cache_tile_analysis": settings.cache_tile_analysis,
        "corridor_buffer_m": settings.corridor_buffer_meters,
        "corridor_cache_grid_step_deg": settings.corridor_cache_grid_step_deg,
        "disable_satellite_green": settings.disable_satellite_green,
        "force_recalculate": settings.force_recalculate,
        "green_pixel_metric": "v1_sum_M_intersect",
        "road_buffer_meters": settings.road_buffer_meters,
        "satellite_zoom": settings.satellite_zoom,
        "tms_server": settings.tms_server,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def quantize_corridor_bbox_expanding(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    step_deg: float,
) -> Tuple[float, float, float, float]:
    """Расширить bbox до ячеек сетки (floor минимумов, ceil максимумов) — безопасно для повторного использования графа."""
    if step_deg <= 0.0 or not math.isfinite(step_deg):
        return min_lon, min_lat, max_lon, max_lat
    return (
        math.floor(min_lon / step_deg) * step_deg,
        math.floor(min_lat / step_deg) * step_deg,
        math.ceil(max_lon / step_deg) * step_deg,
        math.ceil(max_lat / step_deg) * step_deg,
    )


def corridor_bbox_cache_key(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    settings: Settings,
    *,
    skip_satellite_green: bool = False,
) -> str:
    """Стабильный ключ файла кэша для прямоугольника коридора.

    Фаза спутника (phase1 vs green) в ключ не входит: один файл на bbox+ctx;
    «зелёная» доначитка обновляет тот же graphml без второй загрузки OSM.
    ``skip_satellite_green`` оставлен в сигнатуре для обратной совместимости вызовов.
    """
    del skip_satellite_green
    step = settings.corridor_cache_grid_step_deg
    qmin_lon, qmin_lat, qmax_lon, qmax_lat = quantize_corridor_bbox_expanding(
        min_lon, min_lat, max_lon, max_lat, step
    )
    prec = 9 if step > 0.0 else 6
    payload = {
        "ctx": corridor_graph_cache_fingerprint(settings),
        "max_lat": round(qmax_lat, prec),
        "max_lon": round(qmax_lon, prec),
        "min_lat": round(qmin_lat, prec),
        "min_lon": round(qmin_lon, prec),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class CorridorGraphDiskCache:
    """``{cache_dir}/{sha256}.graphml`` — готовый MultiDiGraph после весов."""

    def __init__(self, cache_dir: Path, enabled: bool = True) -> None:
        self._dir = Path(cache_dir)
        self._enabled = enabled
        if self._enabled:
            self._dir.mkdir(parents=True, exist_ok=True)

    def path(self, key_hash: str) -> Path:
        return self._dir / f"{key_hash}.graphml"

    def load(self, key_hash: str) -> Optional[nx.MultiDiGraph]:
        if not self._enabled:
            return None
        path = self.path(key_hash)
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
            logger.warning("Кэш графа коридора: не удалось загрузить %s: %s", path, exc)
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            return None

    def save(self, key_hash: str, G: nx.MultiDiGraph) -> None:
        if not self._enabled or G.number_of_nodes() == 0:
            return
        path = self.path(key_hash)
        tmp = path.with_suffix(".graphml.tmp")
        try:
            ox.save_graphml(G, filepath=tmp)
            tmp.replace(path)
            logger.info(
                "Кэш графа коридора сохранён: %s… (%d узлов)",
                key_hash[:16],
                G.number_of_nodes(),
            )
        except Exception as exc:
            logger.warning("Кэш графа коридора: сохранение не удалось: %s", exc)
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
