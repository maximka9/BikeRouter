"""Пользовательские исключения проекта.

Каждое исключение несёт машинно-читаемый ``code`` для структурированных
ответов API (см. :class:`bike_router.models.ErrorDetail`).
"""


class BikeRouterError(Exception):
    """Базовое исключение проекта."""

    code: str = "INTERNAL_ERROR"


class RouteNotFoundError(BikeRouterError):
    """Маршрут между указанными точками не найден."""

    code = "NO_PATH"

    def __init__(self, weight_key: str, message: str = ""):
        self.weight_key = weight_key
        super().__init__(message or f"Маршрут не найден по весу '{weight_key}'")


class PointOutsideZoneError(BikeRouterError):
    """Запрошенная точка находится за пределами загруженного графа."""

    code = "POINT_OUTSIDE_ZONE"

    def __init__(
        self, lat: float, lon: float, label: str = "", snap_m: float = 0
    ):
        self.lat = lat
        self.lon = lon
        self.label = label
        self.snap_m = snap_m
        if snap_m > 0:
            msg = (
                f"Точка {label}({lat:.5f}, {lon:.5f}) слишком далеко от "
                f"дорожной сети ({snap_m:.0f} м)"
            )
        else:
            msg = (
                f"Точка {label}({lat:.5f}, {lon:.5f}) вне зоны покрытия графа"
            )
        super().__init__(msg)


class OverpassUnavailableError(BikeRouterError):
    """Overpass/OSMnx не получили ответ (сеть, таймаут, 5xx)."""

    code = "OVERPASS_UNAVAILABLE"

    def __init__(self, message: str = "") -> None:
        super().__init__(
            message
            or (
                "Не удалось загрузить дорожную сеть с сервера Overpass "
                "(таймаут, отказ в соединении или перегрузка). "
                "Повторите позже или задайте другое зеркало: OSM_OVERPASS_URL в .env."
            )
        )


class RouteTooLongError(BikeRouterError):
    """Маршрут превышает допустимую длину."""

    code = "ROUTE_TOO_LONG"

    def __init__(self, length_m: float, max_m: float):
        self.length_m = length_m
        self.max_m = max_m
        super().__init__(
            f"Маршрут слишком длинный: {length_m / 1000:.1f} км "
            f"(макс. {max_m / 1000:.0f} км)"
        )


class TileDownloadError(BikeRouterError):
    """Ошибка загрузки спутникового тайла."""

    code = "TILE_DOWNLOAD_ERROR"

    def __init__(self, x: int, y: int, zoom: int, cause: Exception | None = None):
        self.x, self.y, self.zoom = x, y, zoom
        msg = f"Не удалось загрузить тайл ({x}, {y}, z={zoom})"
        if cause:
            msg += f": {cause}"
        super().__init__(msg)


class ElevationDataError(BikeRouterError):
    """Ошибка получения данных высот SRTM."""

    code = "ELEVATION_ERROR"


class CacheError(BikeRouterError):
    """Ошибка чтения/записи файлового кэша."""

    code = "CACHE_ERROR"


class GreenAnalysisError(BikeRouterError):
    """Ошибка анализа озеленения."""

    code = "GREEN_ANALYSIS_ERROR"
