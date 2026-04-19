"""Сервис геокодирования с кэшем и rate-limiter.

Абстрактный базовый класс ``GeocodingProvider`` позволяет подменить
Nominatim на Photon, self-hosted Nominatim, Google Maps и т.д.
без изменения остального кода.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from ..models import GeocodingResult
from .retry import is_transient_http, retry_call

logger = logging.getLogger(__name__)


# ── Абстракция провайдера ────────────────────────────────────────


class GeocodingProvider(ABC):
    """Интерфейс для подмены провайдера геокодирования."""

    @abstractmethod
    def forward(self, query: str, limit: int) -> List[GeocodingResult]:
        ...

    @abstractmethod
    def reverse(self, lat: float, lon: float) -> Optional[GeocodingResult]:
        ...


class NominatimProvider(GeocodingProvider):
    """Публичный Nominatim (OSM). Лимит: 1 req/s, без автодополнения.

    Для своего инстанса унаследуйте класс и переопределите ``BASE`` (или добавьте
    отдельный класс с тем же интерфейсом :class:`GeocodingProvider`).
    """

    BASE = "https://nominatim.openstreetmap.org"
    HEADERS = {"User-Agent": "bike-router/3.0 (student-thesis-project)"}
    # viewbox: west, north, east, south (Самара и окрестности) — приоритет выдачи
    _SAMARA_VIEWBOX = "49.70,53.45,50.75,52.85"
    _SAMARA_LAT = 53.195878
    _SAMARA_LON = 50.151612

    def forward(self, query: str, limit: int) -> List[GeocodingResult]:
        resp = requests.get(
            f"{self.BASE}/search",
            params={
                "q": query,
                "format": "jsonv2",
                "limit": limit,
                "countrycodes": "ru",
                "viewbox": self._SAMARA_VIEWBOX,
                "bounded": "0",
                "dedupe": "1",
                "addressdetails": "1",
            },
            headers=self.HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        raw = resp.json()
        out = [
            GeocodingResult(
                lat=float(r["lat"]),
                lon=float(r["lon"]),
                display_name=r.get("display_name", ""),
            )
            for r in raw
        ]
        return self._prioritize_samara(out)

    @classmethod
    def _prioritize_samara(cls, results: List[GeocodingResult]) -> List[GeocodingResult]:
        """Самара в названии и ближе к центру города — выше в списке."""

        def key(r: GeocodingResult) -> Tuple[int, float]:
            name = (r.display_name or "").lower()
            in_city = 0 if ("самара" in name or "samara" in name) else 1
            dlat = r.lat - cls._SAMARA_LAT
            dlon = r.lon - cls._SAMARA_LON
            dist2 = dlat * dlat + dlon * dlon
            return (in_city, dist2)

        return sorted(results, key=key)

    def reverse(self, lat: float, lon: float) -> Optional[GeocodingResult]:
        resp = requests.get(
            f"{self.BASE}/reverse",
            params={"lat": lat, "lon": lon, "format": "jsonv2"},
            headers=self.HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            return None
        return GeocodingResult(
            lat=float(data.get("lat", lat)),
            lon=float(data.get("lon", lon)),
            display_name=data.get("display_name", ""),
        )


# ── Сервис с кэшем и rate-limiter ───────────────────────────────


class GeocodingService:
    """Обёртка над провайдером: добавляет LRU-кэш и rate-limiting.

    Кэш реализован как LRU поверх ``dict`` (Python 3.7+ сохраняет
    порядок вставки): при cache-hit ключ перемещается в конец
    (``move_to_end``-семантика через delete + re-insert), при
    вытеснении удаляется самый старый (первый) ключ.

    Args:
        provider: конкретная реализация (по умолч. :class:`NominatimProvider`).
        min_interval: минимальный интервал между HTTP-запросами (сек).
        max_cache_size: максимум записей в кэше (forward + reverse).
    """

    def __init__(
        self,
        provider: Optional[GeocodingProvider] = None,
        min_interval: float = 1.1,
        max_cache_size: int = 2048,
        disk_cache_dir: Optional[str] = None,
        settings: Optional[Any] = None,
    ) -> None:
        self._provider = provider or NominatimProvider()
        self._settings = settings
        self._min_interval = min_interval
        self._max_cache = max_cache_size

        self._fwd_cache: Dict[Tuple[str, int], List[GeocodingResult]] = {}
        self._rev_cache: Dict[Tuple[float, float], GeocodingResult] = {}

        self._last_ts: float = 0.0
        self._lock = threading.Lock()

        self._disk_dir: Optional[Path] = (
            Path(disk_cache_dir) if disk_cache_dir else None
        )
        if self._disk_dir is not None:
            self._disk_dir.mkdir(parents=True, exist_ok=True)

    def _retry_kw(self) -> Dict[str, Any]:
        st = self._settings
        if st is None:
            return {
                "max_attempts": 4,
                "base_sec": 0.5,
                "max_sec": 30.0,
                "jitter": 0.25,
            }
        return {
            "max_attempts": max(1, int(st.http_retry_max_attempts)),
            "base_sec": float(st.http_retry_base_delay_sec),
            "max_sec": float(st.http_retry_max_delay_sec),
            "jitter": float(st.http_retry_jitter),
        }

    def _disk_path(self, prefix: str, key_material: str) -> Path:
        assert self._disk_dir is not None
        h = hashlib.sha256(key_material.encode("utf-8")).hexdigest()[:32]
        return self._disk_dir / f"{prefix}_{h}.json"

    def _disk_get_fwd(self, query: str, limit: int) -> Optional[List[GeocodingResult]]:
        if self._disk_dir is None:
            return None
        path = self._disk_path("fwd", f"{query.strip().lower()}\n{limit}")
        if not path.is_file():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            return [
                GeocodingResult(lat=r["lat"], lon=r["lon"], display_name=r["display_name"])
                for r in raw
            ]
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            return None

    def _disk_put_fwd(self, query: str, limit: int, results: List[GeocodingResult]) -> None:
        if self._disk_dir is None:
            return
        path = self._disk_path("fwd", f"{query.strip().lower()}\n{limit}")
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            data = [r.model_dump() for r in results]
            with self._lock:
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False)
                os.replace(tmp, path)
        except OSError as exc:
            logger.debug("Geocode disk write fwd: %s", exc)
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

    def _disk_get_rev(self, lat: float, lon: float) -> Optional[GeocodingResult]:
        if self._disk_dir is None:
            return None
        path = self._disk_path("rev", f"{round(lat, 5)},{round(lon, 5)}")
        if not path.is_file():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                r = json.load(f)
            return GeocodingResult(
                lat=float(r["lat"]),
                lon=float(r["lon"]),
                display_name=r.get("display_name", ""),
            )
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            return None

    def _disk_put_rev(self, lat: float, lon: float, result: GeocodingResult) -> None:
        if self._disk_dir is None:
            return
        path = self._disk_path("rev", f"{round(lat, 5)},{round(lon, 5)}")
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            with self._lock:
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(result.model_dump(), f, ensure_ascii=False)
                os.replace(tmp, path)
        except OSError as exc:
            logger.debug("Geocode disk write rev: %s", exc)
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

    # ── Rate-limiter ─────────────────────────────────────────────

    def _wait(self) -> None:
        with self._lock:
            elapsed = time.monotonic() - self._last_ts
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_ts = time.monotonic()

    # ── LRU-кэш ─────────────────────────────────────────────────

    @staticmethod
    def _lru_get(cache: dict, key):
        """Получить значение и переместить ключ в конец (LRU-touch)."""
        value = cache.get(key)
        if value is not None:
            del cache[key]
            cache[key] = value
        return value

    @staticmethod
    def _lru_put(cache: dict, key, value, max_size: int) -> None:
        """Добавить значение; если кэш переполнен — вытеснить oldest."""
        cache[key] = value
        while len(cache) > max_size:
            cache.pop(next(iter(cache)))

    @property
    def cache_stats(self) -> Dict[str, int]:
        return {
            "forward_entries": len(self._fwd_cache),
            "reverse_entries": len(self._rev_cache),
        }

    # ── Публичный API ────────────────────────────────────────────

    def geocode(self, query: str, limit: int = 5) -> List[GeocodingResult]:
        """Прямое геокодирование (адрес → координаты) с LRU и дисковым кэшем."""
        key = (query.strip().lower(), limit)
        cached = self._lru_get(self._fwd_cache, key)
        if cached is not None:
            logger.debug("Geocode memory hit: %s", query)
            return cached

        disk_hit = self._disk_get_fwd(query, limit)
        if disk_hit is not None:
            qshort = query if len(query) <= 72 else query[:69] + "…"
            logger.info("Geocode диск hit (forward): «%s»", qshort)
            half = self._max_cache // 2
            self._lru_put(self._fwd_cache, key, disk_hit, half)
            return disk_hit

        qshort = query if len(query) <= 72 else query[:69] + "…"
        logger.info("Geocode диск miss (forward) → HTTP: «%s»", qshort)
        self._wait()
        rk = self._retry_kw()

        def _fwd() -> List[GeocodingResult]:
            return self._provider.forward(query, limit)

        results = retry_call(
            _fwd,
            should_retry=is_transient_http,
            label="nominatim_forward",
            **rk,
        )
        half = self._max_cache // 2
        self._lru_put(self._fwd_cache, key, results, half)
        self._disk_put_fwd(query, limit, results)
        return results

    def reverse_geocode(
        self, lat: float, lon: float
    ) -> Optional[GeocodingResult]:
        """Обратное геокодирование (координаты → адрес) с LRU и дисковым кэшем."""
        key = (round(lat, 5), round(lon, 5))
        cached = self._lru_get(self._rev_cache, key)
        if cached is not None:
            logger.debug("Reverse geocode memory hit: %s", key)
            return cached

        disk_hit = self._disk_get_rev(lat, lon)
        if disk_hit is not None:
            logger.info("Geocode диск hit (reverse): %s", key)
            half = self._max_cache // 2
            self._lru_put(self._rev_cache, key, disk_hit, half)
            return disk_hit

        logger.info("Geocode диск miss (reverse) → HTTP: %s", key)
        self._wait()
        rk = self._retry_kw()

        def _rev() -> Optional[GeocodingResult]:
            return self._provider.reverse(lat, lon)

        result = retry_call(
            _rev,
            should_retry=is_transient_http,
            label="nominatim_reverse",
            **rk,
        )

        if result is not None:
            half = self._max_cache // 2
            self._lru_put(self._rev_cache, key, result, half)
            self._disk_put_rev(lat, lon, result)
        return result
