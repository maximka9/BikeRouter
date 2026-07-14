"""Сервис файлового кэширования (pickle)."""

import logging
import os
import pickle
import tempfile
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Версия обёртки на диске; при смене формата увеличить и старые файлы перестанут матчиться
_CACHE_WRAPPER_VERSION = 1


class CacheService:
    """Файловый кэш на основе pickle.

    Запись атомарная (временный файл + replace). Полезная нагрузка в обёртке
    ``{_cache_fmt, data}`` для простого версионирования.
    """

    def __init__(self, base_dir: str) -> None:
        self._base_dir = base_dir
        self._cache_dir = os.path.join(base_dir, "cache")
        os.makedirs(self._cache_dir, exist_ok=True)

    @property
    def cache_dir(self) -> str:
        return self._cache_dir

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def get_path(self, bbox_bounds: tuple, cache_type: str = "green") -> str:
        """Путь к файлу кэша, уникальный для *bbox* и *cache_type*."""
        key = (
            f"{cache_type}_{bbox_bounds[0]:.4f}_{bbox_bounds[1]:.4f}"
            f"_{bbox_bounds[2]:.4f}_{bbox_bounds[3]:.4f}"
        )
        return os.path.join(self._cache_dir, f"{key}.pkl")

    def load(self, path: str) -> Optional[Any]:
        """Загрузить данные из кэша. ``None`` если файла нет или он повреждён."""
        if not os.path.exists(path):
            return None
        try:
            with open(path, "rb") as fh:
                raw = pickle.load(fh)
            if isinstance(raw, dict) and raw.get("_cache_fmt") == _CACHE_WRAPPER_VERSION:
                data = raw.get("data")
                logger.debug("Кэш прочитан: %s", path)
                return data
            # Legacy: весь объект — полезная нагрузка
            logger.debug("Кэш прочитан (legacy): %s", path)
            return raw
        except Exception as exc:
            logger.warning("Ошибка чтения кэша %s: %s", path, exc)
            return None

    def save(self, path: str, data: Any) -> bool:
        """Сохранить данные в кэш атомарно. Возвращает ``True`` при успехе."""
        os.makedirs(os.path.dirname(path) or self._cache_dir, exist_ok=True)
        wrapper = {"_cache_fmt": _CACHE_WRAPPER_VERSION, "data": data}
        try:
            d = os.path.dirname(path) or self._cache_dir
            fd, tmp = tempfile.mkstemp(
                suffix=".pkl.tmp", dir=d, prefix=".cache_"
            )
            try:
                with os.fdopen(fd, "wb") as fh:
                    pickle.dump(wrapper, fh, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(tmp, path)
            except Exception:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
            logger.debug("Кэш записан: %s", path)
            return True
        except Exception as exc:
            logger.warning("Ошибка записи кэша %s: %s", path, exc)
            return False

    # ------------------------------------------------------------------
    # Утилиты для тайлов
    # ------------------------------------------------------------------

    def tile_dir(self) -> str:
        """Директория для кэшированных тайлов спутниковых снимков."""
        path = os.path.join(self._cache_dir, "tiles")
        os.makedirs(path, exist_ok=True)
        return path

    def tile_path(self, server: str, zoom: int, x: int, y: int) -> str:
        """Путь к конкретному кэшированному тайлу."""
        return os.path.join(self.tile_dir(), f"{server}_{zoom}_{x}_{y}.jpg")
