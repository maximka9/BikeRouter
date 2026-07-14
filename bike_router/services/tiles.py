"""Сервис работы с TMS-тайлами (спутниковые снимки)."""

import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import BytesIO
from typing import Dict, Optional, Set, Tuple

import numpy as np
import requests
from requests.adapters import HTTPAdapter

from ..config import TMS_SERVERS, Settings
from .cache import CacheService
from ..metrics import inc_tile_fail
from .retry import is_transient_http, retry_call

logger = logging.getLogger(__name__)

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning(
        "Pillow не установлен (pip install pillow). "
        "Анализ спутниковых снимков недоступен."
    )

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

_MIN_TILE_BYTES = 400


@dataclass(frozen=True)
class TileBatchStats:
    """Сводка по одному вызову ``download_batch``."""

    requested: int
    delivered: int
    from_cache: int
    from_network: int
    http_fail: int


class TileService:
    """Загрузка, кэширование и координатные преобразования TMS-тайлов."""

    def __init__(self, cache_service: CacheService, settings: Settings) -> None:
        self._cache = cache_service
        self._settings = settings
        self._session = requests.Session()
        # По умолчанию urllib3 держит pool_maxsize=10 на хост — при TILE_DOWNLOAD_THREADS>10
        # сыпятся WARNING «Connection pool is full» (см. mt*.google.com).
        _t = max(1, int(self._settings.tile_download_threads))
        _adapter = HTTPAdapter(
            pool_connections=max(_t, 10),
            pool_maxsize=_t,
        )
        self._session.mount("https://", _adapter)
        self._session.mount("http://", _adapter)
        self._session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36"
                )
            }
        )

    @property
    def pil_available(self) -> bool:
        return PIL_AVAILABLE

    # ------------------------------------------------------------------
    # Координатные преобразования
    # ------------------------------------------------------------------

    @staticmethod
    def lat_lon_to_tile(lat: float, lon: float, zoom: int) -> Tuple[int, int]:
        """Географические координаты → номер тайла."""
        lat_rad = math.radians(lat)
        n = 2**zoom
        x = int((lon + 180.0) / 360.0 * n)
        y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return x, y

    @staticmethod
    def tile_to_lat_lon(x: int, y: int, zoom: int) -> Tuple[float, float]:
        """Номер тайла → географические координаты (верхний левый угол)."""
        n = 2**zoom
        lon = x / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
        return math.degrees(lat_rad), lon

    @staticmethod
    def get_tile_bounds(
        x: int, y: int, zoom: int
    ) -> Tuple[float, float, float, float]:
        """Границы тайла: ``(min_lon, min_lat, max_lon, max_lat)``."""
        lat1, lon1 = TileService.tile_to_lat_lon(x, y, zoom)
        lat2, lon2 = TileService.tile_to_lat_lon(x + 1, y + 1, zoom)
        return (lon1, lat2, lon2, lat1)

    # ------------------------------------------------------------------
    # Загрузка
    # ------------------------------------------------------------------

    def download(self, x: int, y: int, zoom: int) -> Tuple[Optional["Image.Image"], str]:
        """Скачать один тайл.

        Второе значение: ``cache`` | ``network`` | ``fail``.
        """
        if not PIL_AVAILABLE:
            return None, "fail"

        server = self._settings.tms_server
        cache_file = self._cache.tile_path(server, zoom, x, y)

        if self._settings.cache_satellite:
            if os.path.exists(cache_file):
                try:
                    # Иначе PIL держит открытый файл на каждый тайл → при сотнях тысяч
                    # кэш-хитов «Too many open files» (OSError 24).
                    with Image.open(cache_file) as im:
                        pic = im.copy()
                    return pic, "cache"
                except Exception as exc:
                    logger.debug(
                        "Битый кэш тайла (%d,%d,z=%d): %s", x, y, zoom, exc
                    )

        url = TMS_SERVERS.get(server, TMS_SERVERS["google"]).format(
            x=x, y=y, z=zoom
        )
        try:
            st = self._settings
            rk = {
                "max_attempts": max(1, int(st.http_retry_max_attempts)),
                "base_sec": float(st.http_retry_base_delay_sec),
                "max_sec": float(st.http_retry_max_delay_sec),
                "jitter": float(st.http_retry_jitter),
            }

            def _get():
                r = self._session.get(url, timeout=10)
                if r.status_code >= 500:
                    r.raise_for_status()
                return r

            resp = retry_call(
                _get,
                should_retry=is_transient_http,
                label=f"tms_z{zoom}",
                **rk,
            )
            if resp.status_code == 200:
                if len(resp.content) < _MIN_TILE_BYTES:
                    logger.warning(
                        "tile: ответ слишком короткий (%d байт) (%d,%d,z=%d)",
                        len(resp.content),
                        x,
                        y,
                        zoom,
                    )
                    return None, "fail"
                ct = (resp.headers.get("Content-Type") or "").lower()
                if ct and any(
                    bad in ct
                    for bad in (
                        "text/html",
                        "text/plain",
                        "application/json",
                        "application/xml",
                    )
                ):
                    logger.warning(
                        "tile: неожиданный Content-Type %r (%d,%d,z=%d)",
                        ct,
                        x,
                        y,
                        zoom,
                    )
                    return None, "fail"
                img = Image.open(BytesIO(resp.content))
                img.load()
                if self._settings.cache_satellite:
                    try:
                        img.save(cache_file, "JPEG")
                    except Exception as exc:
                        logger.debug(
                            "Не удалось сохранить тайл в кэш (%d,%d,z=%d): %s",
                            x,
                            y,
                            zoom,
                            exc,
                        )
                return img, "network"
            logger.debug(
                "HTTP %s для тайла (%d,%d,z=%d)", resp.status_code, x, y, zoom
            )
        except Exception as exc:
            logger.debug("Ошибка загрузки тайла (%d,%d,z=%d): %s", x, y, zoom, exc)

        return None, "fail"

    def download_batch(
        self, tiles: Set[Tuple[int, int]], zoom: int
    ) -> Tuple[Dict[Tuple[int, int], "Image.Image"], TileBatchStats]:
        """Параллельная подготовка набора тайлов (чтение JPEG с диска и/или HTTP).

        Даже при полном попадании в ``cache/tiles`` каждый файл открывается PIL
        и копируется в память для зелени — это не «лишняя» сеть, а обязательный шаг.
        """
        result: Dict[Tuple[int, int], "Image.Image"] = {}
        if not PIL_AVAILABLE or not tiles:
            return result, TileBatchStats(
                requested=0,
                delivered=0,
                from_cache=0,
                from_network=0,
                http_fail=0,
            )

        tiles_list = list(tiles)
        threads = self._settings.tile_download_threads
        logger.info(
            "Параллельное чтение/загрузка %d тайлов (%d потоков; кэш = JPEG с диска)...",
            len(tiles_list),
            threads,
        )

        def _dl(xy: Tuple[int, int]):
            img, src = self.download(xy[0], xy[1], zoom)
            return xy, img, src

        n_cache = 0
        n_network = 0
        n_fail = 0
        worker_errors = 0

        with ThreadPoolExecutor(max_workers=threads) as pool:
            futures = {pool.submit(_dl, xy): xy for xy in tiles_list}
            iterator = (
                tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="   Тайлы (кэш/сеть)",
                )
                if TQDM_AVAILABLE
                else as_completed(futures)
            )
            for future in iterator:
                try:
                    (x, y), img, src = future.result()
                    if src == "cache":
                        n_cache += 1
                    elif src == "network":
                        n_network += 1
                    else:
                        n_fail += 1
                    if img is not None:
                        result[(x, y)] = img
                except Exception as exc:
                    worker_errors += 1
                    logger.debug("Исключение воркера тайла: %s", exc)

        extra = ""
        if worker_errors:
            extra += f", ошибок воркеров {worker_errors}"
        logger.info(
            "Тайлы: в памяти %d / %d (прочитано с диска %d, с HTTP %d, ошибок %d%s)",
            len(result),
            len(tiles_list),
            n_cache,
            n_network,
            n_fail,
            extra,
        )
        if n_fail:
            inc_tile_fail(n_fail)
        stats = TileBatchStats(
            requested=len(tiles_list),
            delivered=len(result),
            from_cache=n_cache,
            from_network=n_network,
            http_fail=n_fail,
        )
        return result, stats
