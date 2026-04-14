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

# Эвристики: серый placeholder / однотонная «заглушка» дают нулевую зелень в green.py
_MIN_TILE_BYTES = 400
_MIN_CHANNEL_SPREAD = 5.0  # mean(|R-G|+|G-B|+|R-B|) по пикселям, 0–255
_MIN_LUMINANCE_STD = 2.0  # std яркости по тайлу


def validate_satellite_tile_for_green(
    img: "Image.Image",
    *,
    enabled: bool = True,
) -> Tuple[bool, str]:
    """Проверка, что тайл пригоден для цветового анализа зелени (не серый/плоский)."""
    if not enabled or not PIL_AVAILABLE:
        return True, "ok"
    try:
        rgb = img.convert("RGB")
        arr = np.asarray(rgb, dtype=np.float32)
    except Exception as exc:
        return False, f"convert:{exc}"
    if arr.ndim != 3 or arr.shape[2] != 3:
        return False, "bad_shape"
    h, w = int(arr.shape[0]), int(arr.shape[1])
    if h < 4 or w < 4:
        return False, "too_small"
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    spread = float(np.mean(np.abs(r - g) + np.abs(g - b) + np.abs(r - b)))
    if spread < _MIN_CHANNEL_SPREAD:
        return False, f"grayscale_or_flat_spread={spread:.2f}"
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    lstd = float(np.std(lum))
    if lstd < _MIN_LUMINANCE_STD:
        return False, f"uniform_tile_lum_std={lstd:.2f}"
    return True, "ok"


@dataclass(frozen=True)
class TileBatchStats:
    """Сводка по одному вызову ``download_batch``."""

    requested: int
    delivered: int
    from_cache: int
    from_network: int
    http_fail: int
    invalid_rejected: int


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

        Второе значение: ``cache`` | ``network`` | ``fail`` | ``invalid``
        (``invalid`` — отклонён проверкой цвета/контраста, в кэш не пишем).
        """
        if not PIL_AVAILABLE:
            return None, "fail"

        server = self._settings.tms_server
        cache_file = self._cache.tile_path(server, zoom, x, y)
        validate = self._settings.tile_validate_for_green

        if self._settings.cache_satellite:
            if os.path.exists(cache_file):
                try:
                    # Иначе PIL держит открытый файл на каждый тайл → при сотнях тысяч
                    # кэш-хитов «Too many open files» (OSError 24).
                    with Image.open(cache_file) as im:
                        pic = im.copy()
                    ok, reason = validate_satellite_tile_for_green(
                        pic, enabled=validate
                    )
                    if ok:
                        return pic, "cache"
                    logger.debug(
                        "tile: кэш не прошёл проверку (%d,%d,z=%d) %s — удаляем файл",
                        x,
                        y,
                        zoom,
                        reason,
                    )
                    try:
                        os.remove(cache_file)
                    except OSError as exc:
                        logger.debug("не удалось удалить кэш тайла: %s", exc)
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
                ok, reason = validate_satellite_tile_for_green(
                    img, enabled=validate
                )
                if not ok:
                    logger.debug(
                        "tile: отклонён для зелени (%d,%d,z=%d): %s",
                        x,
                        y,
                        zoom,
                        reason,
                    )
                    return None, "invalid"
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
        """Параллельная загрузка набора тайлов. Второй элемент — счётчики для зелени/логов."""
        result: Dict[Tuple[int, int], "Image.Image"] = {}
        if not PIL_AVAILABLE or not tiles:
            return result, TileBatchStats(
                requested=0,
                delivered=0,
                from_cache=0,
                from_network=0,
                http_fail=0,
                invalid_rejected=0,
            )

        tiles_list = list(tiles)
        threads = self._settings.tile_download_threads
        logger.info(
            "Параллельная загрузка %d тайлов (%d потоков)...",
            len(tiles_list),
            threads,
        )

        def _dl(xy: Tuple[int, int]):
            img, src = self.download(xy[0], xy[1], zoom)
            return xy, img, src

        n_cache = 0
        n_network = 0
        n_fail = 0
        n_invalid = 0
        worker_errors = 0

        with ThreadPoolExecutor(max_workers=threads) as pool:
            futures = {pool.submit(_dl, xy): xy for xy in tiles_list}
            iterator = (
                tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="   Загрузка тайлов",
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
                    elif src == "invalid":
                        n_invalid += 1
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
        if n_invalid:
            extra += f", отклонено как невалидные {n_invalid}"
        logger.info(
            "Тайлы: готово %d / %d (из кэша %d, с сети %d, не удалось %d%s)",
            len(result),
            len(tiles_list),
            n_cache,
            n_network,
            n_fail,
            extra,
        )
        if n_invalid:
            logger.warning(
                "tiles: отклонено %d тайлов проверкой цвета (серые/placeholder — см. debug по координатам). "
                "Если много: смените TMS_SERVER / SATELLITE_ZOOM или TILE_VALIDATE_FOR_GREEN=false.",
                n_invalid,
            )
        if n_fail:
            inc_tile_fail(n_fail)
        stats = TileBatchStats(
            requested=len(tiles_list),
            delivered=len(result),
            from_cache=n_cache,
            from_network=n_network,
            http_fail=n_fail,
            invalid_rejected=n_invalid,
        )
        return result, stats
