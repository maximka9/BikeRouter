"""Сервис работы с данными высот (SRTM)."""

import logging
from typing import Any, Optional, Tuple

import srtm

from ..exceptions import ElevationDataError

logger = logging.getLogger(__name__)


class ElevationService:
    """Обёртка над библиотекой ``srtm`` с внутренним кэшем высот.

    Перед первым использованием необходимо вызвать :meth:`init`.
    """

    def __init__(self) -> None:
        self._data: Optional[srtm.data.GeoElevationData] = None
        self._cache: dict[tuple, float] = {}

    def init(
        self,
        test_lat: Optional[float] = None,
        test_lon: Optional[float] = None,
    ) -> None:
        """Загрузить данные SRTM и (опционально) проверить тестовую точку.

        Повторные вызовы не перезагружают SRTM с диска/сети — идемпотентно.
        """
        if self._data is not None:
            logger.debug("SRTM уже инициализирован, повторная загрузка пропущена")
        else:
            logger.info("Инициализация SRTM...")
            self._data = srtm.get_data()

        if test_lat is not None and test_lon is not None:
            elev = self.get_elevation(test_lat, test_lon)
            if elev == 0:
                logger.warning(
                    "SRTM не вернул данные для точки (%.4f, %.4f)",
                    test_lat,
                    test_lon,
                )
            else:
                logger.info("SRTM OK — высота тестовой точки: %d м", elev)

    def get_elevation(self, lat: float, lon: float) -> float:
        """Высота в точке (метры). Кэширует до 5 знаков координат."""
        if self._data is None:
            raise ElevationDataError(
                "SRTM не инициализирован. Вызовите init() перед использованием."
            )
        key = (round(lon, 5), round(lat, 5))
        if key not in self._cache:
            h = self._data.get_elevation(lat, lon)
            self._cache[key] = h if h is not None else 0
        return self._cache[key]

    def get_elevation_diff(
        self,
        lon_start: float,
        lat_start: float,
        lon_end: float,
        lat_end: float,
    ) -> float:
        """Разница высот (конец − начало)."""
        return self.get_elevation(lat_end, lon_end) - self.get_elevation(
            lat_start, lon_start
        )

    def get_edge_elevation(
        self,
        geometry: Any,
        length_m: float,
        step_m: float = 30.0,
    ) -> Tuple[float, float, float, list]:
        """Высотный анализ вдоль геометрии ребра с интерполяцией.

        Для рёбер длиннее ``step_m`` промежуточные точки ставятся
        каждые ~``step_m`` метров, что позволяет учесть холмы и впадины
        внутри длинных сегментов (разрешение SRTM ≈ 30 м).

        Returns:
            ``(net_diff, climb, descent, elevations)`` — чистый перепад,
            суммарный набор, суммарный спуск (метры) и список высот
            в равноотстоящих точках вдоль ребра.
        """
        coords = list(geometry.coords)

        if length_m <= step_m * 1.5 or len(coords) < 2:
            e0 = self.get_elevation(coords[0][1], coords[0][0])
            e1 = self.get_elevation(coords[-1][1], coords[-1][0])
            diff = e1 - e0
            return diff, max(0.0, diff), max(0.0, -diff), [e0, e1]

        n = max(3, int(length_m / step_m) + 1)
        elevs = []
        for i in range(n):
            pt = geometry.interpolate(i / (n - 1), normalized=True)
            elevs.append(self.get_elevation(pt.y, pt.x))

        climb = 0.0
        descent = 0.0
        for i in range(1, len(elevs)):
            d = elevs[i] - elevs[i - 1]
            if d > 0:
                climb += d
            elif d < 0:
                descent -= d

        return elevs[-1] - elevs[0], climb, descent, elevs
