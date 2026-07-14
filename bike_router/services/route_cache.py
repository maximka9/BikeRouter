"""Персистентный кэш ответов ``AlternativesResponse`` на диске.

Ключ — округлённые координаты + профиль. В файле: число узлов/рёбер графа,
отпечаток движка весов (профили + OSM-фильтр + ROUTING_ALGO_VERSION), чтобы
не подставлять ответы после смены графа или логики весов без смены размера графа.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class RouteAlternativesDiskCache:
    """JSON-файлы ``{cache_dir}/{sha256}.json``."""

    def __init__(self, cache_dir: Path, enabled: bool = True) -> None:
        self._dir = Path(cache_dir)
        self._enabled = enabled
        self._lock = threading.Lock()
        self._graph_nodes = 0
        self._graph_edges = 0
        self._engine_fingerprint = ""
        if self._enabled:
            self._dir.mkdir(parents=True, exist_ok=True)

    def set_cache_context(
        self, nodes: int, edges: int, engine_fingerprint: str
    ) -> None:
        """Вызывать после каждого успешного ``warmup()``."""
        self._graph_nodes = int(nodes)
        self._graph_edges = int(edges)
        self._engine_fingerprint = str(engine_fingerprint)

    @staticmethod
    def _payload_key(
        start: Tuple[float, float],
        end: Tuple[float, float],
        profile_key: str,
        *,
        green_enabled: bool = True,
    ) -> str:
        raw = json.dumps(
            {
                "e": [round(end[0], 5), round(end[1], 5)],
                "g": 1 if green_enabled else 0,
                "p": profile_key,
                "s": [round(start[0], 5), round(start[1], 5)],
            },
            sort_keys=True,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _path(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        profile: str,
        *,
        green_enabled: bool = True,
    ) -> Path:
        return self._dir / f"{self._payload_key(start, end, profile, green_enabled=green_enabled)}.json"

    def get(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        profile_key: str,
        *,
        green_enabled: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Вернуть ``{"routes": [...]}`` для :meth:`AlternativesResponse.model_validate` или ``None``."""
        if not self._enabled:
            return None
        path = self._path(start, end, profile_key, green_enabled=green_enabled)
        if not path.is_file():
            return None
        try:
            with self._lock, open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.debug("Route cache read error %s: %s", path, exc)
            return None
        if data.get("gn") != self._graph_nodes or data.get("ge") != self._graph_edges:
            return None
        if data.get("fp") != self._engine_fingerprint:
            return None
        routes = data.get("routes")
        if not isinstance(routes, list):
            return None
        return {"routes": routes}

    def put(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        profile_key: str,
        routes_dump: list,
        *,
        green_enabled: bool = True,
    ) -> None:
        if not self._enabled:
            return
        path = self._path(start, end, profile_key, green_enabled=green_enabled)
        body = {
            "fp": self._engine_fingerprint,
            "ge": self._graph_edges,
            "gn": self._graph_nodes,
            "routes": routes_dump,
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            with self._lock:
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(body, f, ensure_ascii=False, separators=(",", ":"))
                os.replace(tmp, path)
        except OSError as exc:
            logger.warning("Route cache write failed %s: %s", path, exc)
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
